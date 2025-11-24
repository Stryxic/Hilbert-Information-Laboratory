"""
===============================================================================
hilbert_pipeline/fusion.py – Span-to-Element Fusion Layer (Pipeline 3.1)
===============================================================================

Responsibilities:
  1. Load span embeddings from lsa_field.json.
  2. Load element embeddings from hilbert_elements.csv.
  3. Compute span→element soft assignments (top-k, cosine similarity).
  4. Adaptive similarity thresholding based on span entropy.
  5. Emit span_element_fusion.csv with both element and element_id.
  6. Aggregate compound-level semantic contexts for molecules.

Backward-compatible with all 2024–2025 orchestrators, and upgraded to match
the 2025 graph-contract requirements.

This module **never mutates** hilbert_elements.csv. Instead it enriches output
tables in results_dir for downstream layers to consume.
===============================================================================
"""

from __future__ import annotations
import json
import os
import re
from typing import Any, Callable, Dict, List, Tuple
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Default no-op emitter. The orchestrator will inject the real one.
DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda *_: None


# =============================================================================
# Logging utilities
# =============================================================================

def _log(msg: str, emit=DEFAULT_EMIT):
    """
    Mirror logs to stdout and orchestrator if available.
    """
    print(msg)
    try:
        emit("log", {"message": msg})
    except Exception:
        pass


def _safe_float(x: Any, default=0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _safe_json_load(path: str, emit=DEFAULT_EMIT):
    if not os.path.exists(path):
        _log(f"[fusion][warn] Missing: {path}", emit)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        _log(f"[fusion][warn] Failed to load {path}: {e}", emit)
        return None


# =============================================================================
# Embedding loading helpers
# =============================================================================

def _resolve_element_id_column(df: pd.DataFrame) -> str | None:
    """
    Determine which identifier column to use as element_id.
    Priority:
      1. element_id
      2. index
      3. None (fallback to row order)
    """
    if "element_id" in df.columns:
        return "element_id"
    if "index" in df.columns:
        return "index"
    return None


def _parse_vec(raw: Any) -> np.ndarray | None:
    """
    Parse a vector stored in CSV (list-like or stringified list).
    """
    if isinstance(raw, (list, tuple)):
        arr = np.asarray(raw, dtype=float)
        return arr if arr.size else None

    if isinstance(raw, str):
        try:
            arr = np.asarray(json.loads(raw), dtype=float)
            return arr if arr.size else None
        except Exception:
            try:
                arr = np.asarray(eval(raw, {"__builtins__": {}}), dtype=float)
                return arr if arr.size else None
            except Exception:
                return None

    if isinstance(raw, np.ndarray):
        return raw if raw.size else None

    return None


def _load_span_embeddings(
    lsa_path: str,
    emit=DEFAULT_EMIT,
) -> tuple[np.ndarray, dict[int, dict]]:
    """
    Load span embeddings and span_map from lsa_field.json.
    Normalises span embeddings to unit vectors.
    """
    data = _safe_json_load(lsa_path, emit)
    if not isinstance(data, dict) or "embeddings" not in data:
        raise ValueError("lsa_field.json missing embeddings array.")

    emb = np.asarray(data["embeddings"], dtype=float)
    if emb.ndim != 2:
        raise ValueError("Invalid embeddings array in lsa_field.json.")

    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norms

    span_map_raw = data.get("span_map", []) or []
    span_map_index: dict[int, dict] = {}

    for i, rec in enumerate(span_map_raw):
        if not isinstance(rec, dict):
            rec = {}
        span_map_index[i] = {
            "doc": rec.get("doc"),
            "span_id": rec.get("span_id", i),
            "text": rec.get("text", ""),
            "elements": rec.get("elements", []),
        }

    return emb, span_map_index


def _load_element_embeddings_from_csv(
    elements_csv: str,
    emit=DEFAULT_EMIT,
) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Load element embeddings from hilbert_elements.csv when column 'embedding' exists.
    Returns L2-normalised vectors.
    """
    df = pd.read_csv(elements_csv)
    if "element" not in df.columns or "embedding" not in df.columns:
        raise ValueError("hilbert_elements.csv lacks 'embedding' column.")

    id_col = _resolve_element_id_column(df)

    elem_labels: list[str] = []
    elem_ids: list[int] = []
    vecs: list[np.ndarray] = []

    for row_idx, row in df.iterrows():
        el = str(row["element"])
        v = _parse_vec(row["embedding"])
        if v is None:
            continue

        if id_col:
            try:
                el_id = int(row[id_col])
            except Exception:
                el_id = row_idx
        else:
            el_id = row_idx

        elem_labels.append(el)
        elem_ids.append(el_id)
        vecs.append(v.astype(float))

    if not vecs:
        raise ValueError("No valid element embeddings in CSV.")

    X = np.vstack(vecs)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X, elem_labels, elem_ids


def _derive_element_centroids_from_span_map(
    lsa_path: str,
    elements_csv: str,
    emit=DEFAULT_EMIT,
) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Fallback centroid-based element embeddings from span_map.
    """
    data = _safe_json_load(lsa_path, emit)
    if not isinstance(data, dict):
        raise ValueError("lsa_field.json unreadable.")

    span_emb = np.asarray(data.get("embeddings", []), dtype=float)
    span_map = data.get("span_map") or []
    if span_emb.ndim != 2 or not span_map:
        raise ValueError("No span embeddings / span_map for centroid fallback.")

    span_emb = span_emb / (np.linalg.norm(span_emb, axis=1, keepdims=True) + 1e-12)

    buckets: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(span_map):
        els = rec.get("elements") or []
        for el in els:
            buckets[str(el)].append(i)

    df = pd.read_csv(elements_csv)
    if "element" not in df.columns:
        raise ValueError("hilbert_elements.csv missing 'element'.")

    id_col = _resolve_element_id_column(df)
    elem_labels: list[str] = []
    elem_ids: list[int] = []
    vecs: list[np.ndarray] = []

    for row_idx, row in df.iterrows():
        el = str(row["element"])
        idxs = buckets.get(el, [])
        if not idxs:
            continue

        v = span_emb[idxs].mean(axis=0)

        if id_col:
            try:
                el_id = int(row[id_col])
            except Exception:
                el_id = row_idx
        else:
            el_id = row_idx

        elem_labels.append(el)
        elem_ids.append(el_id)
        vecs.append(v)

    if not vecs:
        raise ValueError("Unable to derive any element centroids.")

    X = np.vstack(vecs)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X, elem_labels, elem_ids


def _load_element_embeddings(
    elements_csv: str,
    lsa_path: str,
    emit=DEFAULT_EMIT,
) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Hybrid loading strategy: prefer CSV embeddings, otherwise centroid fallback.
    """
    try:
        return _load_element_embeddings_from_csv(elements_csv, emit)
    except Exception as e:
        _log(f"[fusion] Falling back to centroid embeddings: {e}", emit)
        return _derive_element_centroids_from_span_map(lsa_path, elements_csv, emit)


# =============================================================================
# Adaptive Similarity Thresholding
# =============================================================================

def adaptive_threshold(entropy: float,
                       base=0.22,
                       min_t=0.15,
                       max_t=0.40) -> float:
    """
    Entropy-aware threshold:
      high entropy → lower threshold,
      low entropy → higher threshold.
    """
    if entropy <= 0.1:
        return max_t
    if entropy >= 4.0:
        return min_t

    scale = np.exp(-entropy / 2.8)
    return min_t + (max_t - min_t) * scale


# =============================================================================
# Core fusion
# =============================================================================

def fuse_spans_to_elements(
    results_dir: str,
    top_k: int = 3,
    emit=DEFAULT_EMIT,
) -> pd.DataFrame:
    """
    Main fusion function. Produces:
      span_element_fusion.csv
      columns = [
        span_index,
        span_id,
        doc,
        element,
        element_id,
        similarity,
        threshold
      ]
    """
    lsa_path = os.path.join(results_dir, "lsa_field.json")
    el_path = os.path.join(results_dir, "hilbert_elements.csv")

    if not (os.path.exists(lsa_path) and os.path.exists(el_path)):
        _log("[fusion] Missing required inputs; skipping.", emit)
        return pd.DataFrame()

    try:
        span_vecs, span_map = _load_span_embeddings(lsa_path, emit)
        elem_vecs, elem_labels, elem_ids = _load_element_embeddings(el_path, lsa_path, emit)
    except Exception as e:
        _log(f"[fusion][warn] Failed to load embeddings: {e}", emit)
        return pd.DataFrame()

    sims = cosine_similarity(span_vecs, elem_vecs)

    meta = _safe_json_load(lsa_path, emit)
    H_span = meta.get("H_span") if isinstance(meta, dict) else None
    if not isinstance(H_span, list):
        H_span = [1.0] * sims.shape[0]

    rows: list[dict[str, Any]] = []

    for i in range(sims.shape[0]):
        row_sims = sims[i]
        idxs = np.argsort(row_sims)[::-1][:top_k]

        thr = adaptive_threshold(_safe_float(H_span[i], 1.0))

        span_rec = span_map.get(i, {})
        span_id = span_rec.get("span_id", i)
        doc = span_rec.get("doc")

        for j in idxs:
            score = float(row_sims[j])
            if score < thr:
                continue
            rows.append({
                "span_index": i,
                "span_id": span_id,
                "doc": doc,
                "element": elem_labels[j],
                "element_id": elem_ids[j],
                "similarity": score,
                "threshold": thr,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        _log("[fusion] No span→element links above threshold.", emit)
        return df

    out_path = os.path.join(results_dir, "span_element_fusion.csv")
    df.to_csv(out_path, index=False)
    _log(f"[fusion] wrote {out_path} with {len(df)} rows", emit)

    try:
        emit("artifact", {"path": out_path, "kind": "span_fusion"})
    except Exception:
        pass

    return df


# =============================================================================
# Compound context enrichment
# =============================================================================

def aggregate_compound_context(
    results_dir: str,
    elements_csv: str = "hilbert_elements.csv",
    compounds_json: str = "informational_compounds.json",
    fuse_csv: str = "span_element_fusion.csv",
    emit=DEFAULT_EMIT,
) -> pd.DataFrame:
    """
    Build compound_contexts.json summarising:
      - representative spans
      - keyword distributions
      - cross-element span overlap
      - context examples
    """
    el_path = os.path.join(results_dir, elements_csv)
    comp_path = os.path.join(results_dir, compounds_json)
    fuse_path = os.path.join(results_dir, fuse_csv)

    elements_df = pd.read_csv(el_path) if os.path.exists(el_path) else None
    compounds_raw = _safe_json_load(comp_path, emit)
    fusion_df = pd.read_csv(fuse_path) if os.path.exists(fuse_path) else pd.DataFrame()

    if not isinstance(compounds_raw, (dict, list)):
        _log("[fusion] No compounds to enrich.", emit)
        return pd.DataFrame()

    if isinstance(compounds_raw, dict) and "compounds" in compounds_raw:
        compounds = compounds_raw["compounds"]
    elif isinstance(compounds_raw, dict):
        compounds = list(compounds_raw.values())
    else:
        compounds = compounds_raw

    lsa_data = _safe_json_load(os.path.join(results_dir, "lsa_field.json"), emit) or {}
    span_map = lsa_data.get("span_map") or []

    enriched: list[dict[str, Any]] = []

    for c in compounds:
        if not isinstance(c, dict):
            continue

        members = [str(e) for e in c.get("elements", [])]
        cid = c.get("compound_id")

        # gather candidate spans
        collected: list[str] = []

        if not fusion_df.empty:
            sub = fusion_df[fusion_df["element"].isin(members)]
            for _, row in sub.iterrows():
                idx = int(row["span_index"])
                if 0 <= idx < len(span_map):
                    txt = span_map[idx].get("text", "")
                    if txt:
                        collected.append(txt)

        toks_all: list[str] = []
        for s in collected:
            toks_all.extend(re.findall(r"[A-Za-z']+", s.lower()))
        counts = Counter(toks_all)

        member_lower = {m.lower() for m in members}
        scored: list[tuple[int, str]] = []
        for s in collected:
            toks = set(re.findall(r"[A-Za-z']+", s.lower()))
            overlap = len(toks & member_lower)
            scored.append((overlap, s))

        top_spans = [s for _, s in sorted(scored, reverse=True)[:5]]
        top_words = [w for w, _ in counts.most_common(15)]

        rec = dict(c)
        rec["compound_id"] = cid
        rec["context_examples"] = top_spans
        rec["aggregate_keywords"] = top_words
        rec["n_spans"] = len(collected)
        enriched.append(rec)

    out_path = os.path.join(results_dir, "compound_contexts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    _log(f"[fusion] wrote {out_path}", emit)
    try:
        emit("artifact", {"path": out_path, "kind": "compound_context"})
    except Exception:
        pass

    return pd.DataFrame(enriched)


# =============================================================================
# Orchestrator API
# =============================================================================

def run_fusion_pipeline(results_dir: str, emit=DEFAULT_EMIT) -> None:
    """
    Execute:
      (1) span→element fusion
      (2) compound context aggregation
    """
    fuse_spans_to_elements(results_dir, emit=emit)
    aggregate_compound_context(results_dir, emit=emit)
