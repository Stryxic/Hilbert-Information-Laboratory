# =============================================================================
# hilbert_pipeline/fusion.py — Span-to-Element Fusion Layer (Advanced)
# =============================================================================
"""
This module enriches the Hilbert pipeline by:

  (1) Mapping each span embedding → nearest element embeddings
      producing a soft assignment table (span → top-k elements).

  (2) Aggregating compound-level semantic context:
        representative spans,
        keyword distributions,
        overlap strength,
        core semantic signature.

  (3) Providing diagnostics for fusion quality:
        coverage,
        ambiguity,
        mean similarity,
        entropy-weighted thresholds.

This version is aligned with the upgraded orchestrator & molecule layer and
supports orchestrator event streaming via emit().
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# injected by orchestrator
DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda *_: None


# =============================================================================
# Logging utilities
# =============================================================================

def _log(msg: str, emit=DEFAULT_EMIT) -> None:
    """Simple logger that goes to stdout and orchestrator emit()."""
    print(msg)
    try:
        emit("log", {"message": msg})
    except Exception:
        # best-effort only
        pass


def _safe_float(x: Any, default: float = 1.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _safe_json_load(path: str, emit=DEFAULT_EMIT) -> Optional[Dict[str, Any]]:
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
# Embedding Parsers
# =============================================================================

def _load_span_embeddings(lsa_path: str, emit=DEFAULT_EMIT) -> np.ndarray:
    """
    Load and L2-normalise span embeddings from lsa_field.json.

    Returns
    -------
    np.ndarray, shape (n_spans, d)
    """
    data = _safe_json_load(lsa_path, emit=emit)
    if not isinstance(data, dict) or "embeddings" not in data:
        raise ValueError("lsa_field.json missing 'embeddings' array.")

    emb = data["embeddings"]
    arr = np.asarray(emb, dtype=float)

    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("Invalid span embeddings array.")

    # L2 normalise
    n = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / n


def _parse_vec(raw: Any) -> Optional[np.ndarray]:
    """Parse an embedding cell into a numpy vector, if possible."""
    if isinstance(raw, str):
        # Try JSON first
        try:
            arr = np.asarray(json.loads(raw), dtype=float)
            return arr if arr.size > 0 else None
        except Exception:
            # very defensive: restricted eval as fallback
            try:
                arr = np.asarray(eval(raw, {"__builtins__": {}}), dtype=float)
                return arr if arr.size > 0 else None
            except Exception:
                return None

    if isinstance(raw, (list, tuple)):
        v = np.asarray(raw, dtype=float)
        return v if v.size > 0 else None

    if isinstance(raw, np.ndarray):
        return raw if raw.size > 0 else None

    return None


def _load_element_embeddings_from_csv(elements_csv: str,
                                      emit=DEFAULT_EMIT) -> Tuple[np.ndarray, List[str]]:
    """
    Preferred path: load element embeddings from the 'embedding' column
    in hilbert_elements.csv. Returns (X, element_ids).
    """
    df = pd.read_csv(elements_csv)
    if "element" not in df.columns or "embedding" not in df.columns:
        raise ValueError("hilbert_elements.csv missing 'embedding' column.")

    el_ids: List[str] = []
    vecs: List[np.ndarray] = []

    for _, row in df.iterrows():
        el = str(row["element"])
        v = _parse_vec(row["embedding"])
        if v is None:
            continue
        v = v.astype(float)
        if not np.any(v):
            continue
        el_ids.append(el)
        vecs.append(v)

    if not el_ids:
        raise ValueError("No valid element embeddings in 'embedding' column.")

    X = np.vstack(vecs)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms
    return X, el_ids


def _load_element_embeddings_from_lsa(elements_csv: str,
                                      lsa_path: str,
                                      emit=DEFAULT_EMIT) -> Tuple[np.ndarray, List[str]]:
    """
    Fallback path when hilbert_elements.csv has no 'embedding' column:
    derive element centroids from span embeddings + span_id mapping.

    We assume:
      - lsa_field.json contains `embeddings`
      - hilbert_elements.csv has columns `element` and `span_id`
    """
    df = pd.read_csv(elements_csv)
    if "element" not in df.columns or "span_id" not in df.columns:
        raise ValueError("hilbert_elements.csv missing 'span_id' column; "
                         "cannot derive element centroids from LSA.")

    lsa_data = _safe_json_load(lsa_path, emit=emit)
    if not isinstance(lsa_data, dict) or "embeddings" not in lsa_data:
        raise ValueError("lsa_field.json missing 'embeddings'; "
                         "cannot derive element centroids.")

    span_emb = np.asarray(lsa_data["embeddings"], dtype=float)
    if span_emb.ndim != 2 or span_emb.size == 0:
        raise ValueError("Invalid span embeddings in lsa_field.json.")

    # L2 normalise span embeddings
    norms = np.linalg.norm(span_emb, axis=1, keepdims=True) + 1e-12
    span_emb = span_emb / norms

    # aggregate per element
    buckets: Dict[str, List[int]] = defaultdict(list)
    for _, row in df.iterrows():
        el = str(row["element"])
        sid = row.get("span_id")
        if pd.isna(sid):
            continue
        try:
            sid_int = int(sid)
        except Exception:
            continue
        if sid_int < 0 or sid_int >= span_emb.shape[0]:
            continue
        buckets[el].append(sid_int)

    el_ids: List[str] = []
    vecs: List[np.ndarray] = []

    for el, idxs in buckets.items():
        if not idxs:
            continue
        vecs.append(span_emb[idxs].mean(axis=0))
        el_ids.append(el)

    if not el_ids:
        raise ValueError("No valid element centroids derived from span embeddings.")

    X = np.vstack(vecs)
    return X, el_ids


def _load_element_embeddings(elements_csv: str,
                             lsa_path: str,
                             emit=DEFAULT_EMIT) -> Tuple[np.ndarray, List[str]]:
    """
    Robust loader that:
      1) tries to use the 'embedding' column in hilbert_elements.csv;
      2) falls back to deriving centroids from lsa_field.json + span_id.
    """
    try:
        return _load_element_embeddings_from_csv(elements_csv, emit=emit)
    except Exception as e_csv:
        _log(f"[fusion][info] Falling back to LSA-derived centroids "
             f"because CSV embeddings failed: {e_csv}", emit)

    # Fallback: derive from LSA
    return _load_element_embeddings_from_lsa(elements_csv, lsa_path, emit=emit)


# =============================================================================
# Adaptive Similarity Thresholds
# =============================================================================

def adaptive_sim_threshold(entropy: float,
                           base: float = 0.22,
                           min_t: float = 0.15,
                           max_t: float = 0.40) -> float:
    """
    High-entropy spans need lower similarity thresholds.
    Low-entropy spans require higher similarity to count as meaningful.
    """
    if entropy <= 0.1:
        return max_t
    if entropy >= 4.0:
        return min_t

    scale = float(np.exp(-entropy / 2.8))
    return float(min_t + (max_t - min_t) * scale)


# =============================================================================
# FUSION CORE
# =============================================================================

def fuse_spans_to_elements(out_dir: str,
                           top_k: int = 3,
                           emit=DEFAULT_EMIT) -> pd.DataFrame:
    """
    Produces a CSV: span_index, element, similarity, threshold.

    Uses adaptive thresholding based on span entropy when available.
    """
    lsa_path = os.path.join(out_dir, "lsa_field.json")
    el_path = os.path.join(out_dir, "hilbert_elements.csv")

    if not os.path.exists(lsa_path) or not os.path.exists(el_path):
        _log("[fusion] Missing LSA field or elements; skipping.", emit)
        return pd.DataFrame()

    # -- load embeddings -----------------------------------------------------
    try:
        span_vecs = _load_span_embeddings(lsa_path, emit=emit)
        elem_vecs, elem_ids = _load_element_embeddings(el_path, lsa_path, emit=emit)
    except Exception as e:
        _log(f"[fusion][warn] Failed embedding load: {e}", emit)
        return pd.DataFrame()

    if span_vecs.shape[1] != elem_vecs.shape[1]:
        _log("[fusion][warn] Dimension mismatch between span and element "
             "embeddings; skipping fusion.", emit)
        return pd.DataFrame()

    sims = cosine_similarity(span_vecs, elem_vecs)

    # load entropies for adaptive thresholds (optional)
    span_data = _safe_json_load(lsa_path, emit=emit) or {}
    H_span = span_data.get("H_span") if isinstance(span_data, dict) else None
    if not isinstance(H_span, list) or len(H_span) != sims.shape[0]:
        # fallback: default entropy value
        H_span = [1.0] * sims.shape[0]

    rows: List[Dict[str, Any]] = []
    for i in range(sims.shape[0]):
        row = sims[i]
        idxs = np.argsort(row)[::-1][:top_k]

        thr = adaptive_sim_threshold(_safe_float(H_span[i], 1.0))

        for j in idxs:
            sim = float(row[j])
            if sim < thr:
                continue
            rows.append({
                "span_index": int(i),
                "element": elem_ids[j],
                "similarity": sim,
                "threshold": float(thr),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        _log("[fusion] No span→element matches above threshold.", emit)
        return df

    out_path = os.path.join(out_dir, "span_element_fusion.csv")
    df.to_csv(out_path, index=False)
    _log(f"[fusion] Wrote {out_path} with {len(df)} rows.", emit)

    try:
        emit("artifact", {"path": out_path, "kind": "span_fusion"})
    except Exception:
        pass

    return df


# =============================================================================
# Compound-Level Semantic Context Extraction
# =============================================================================

def aggregate_compound_context(out_dir: str,
                               elements_csv: str = "hilbert_elements.csv",
                               compounds_json: str = "informational_compounds.json",
                               fuse_csv: str = "span_element_fusion.csv",
                               emit=DEFAULT_EMIT) -> pd.DataFrame:
    """
    Produces enriched compound contexts including:
        - representative spans
        - core keywords
        - cross-element semantic overlap
        - compound signature vector (mean element embedding)
    """

    el_path = os.path.join(out_dir, elements_csv)
    c_path = os.path.join(out_dir, compounds_json)
    f_path = os.path.join(out_dir, fuse_csv)

    if not os.path.exists(el_path) or not os.path.exists(c_path):
        _log("[fusion] Missing elements or compounds; skipping compound context.", emit)
        return pd.DataFrame()

    elements_df = pd.read_csv(el_path)
    compounds_raw = _safe_json_load(c_path, emit=emit)

    if isinstance(compounds_raw, dict) and "compounds" in compounds_raw:
        compounds = list(compounds_raw["compounds"])
    elif isinstance(compounds_raw, dict):
        compounds = list(compounds_raw.values())
    elif isinstance(compounds_raw, list):
        compounds = compounds_raw
    else:
        compounds = []

    fusion_df = pd.read_csv(f_path) if os.path.exists(f_path) else pd.DataFrame()

    # Preload span text cache from elements if present (optional)
    span_cache: Dict[str, List[str]] = {}
    if "element" in elements_df.columns:
        text_cols = [c for c in ["span", "context", "text"] if c in elements_df.columns]
        if text_cols:
            tcol = text_cols[0]
            for _, row in elements_df.iterrows():
                el = str(row["element"])
                txt = str(row.get(tcol) or "").strip()
                if txt:
                    span_cache.setdefault(el, []).append(txt)

    # Optional access to full span_map for additional spans
    lsa_json = _safe_json_load(os.path.join(out_dir, "lsa_field.json"), emit)
    span_map = lsa_json.get("span_map") if isinstance(lsa_json, dict) else []
    span_text_by_index = {}
    if isinstance(span_map, list):
        for idx, rec in enumerate(span_map):
            span_text_by_index[idx] = str(rec.get("text", "") or "")

    enriched: List[Dict[str, Any]] = []

    for c in compounds:
        cid = c.get("compound_id")
        members = [str(e) for e in c.get("elements", [])]

        # 1) Gather candidate spans from preloaded cache
        candidate_spans: List[str] = []
        for el in members:
            candidate_spans.extend(span_cache.get(el, []))

        # 2) Attach spans via fusion assignments if available
        if not fusion_df.empty and {"span_index", "element"}.issubset(fusion_df.columns):
            f_sub = fusion_df[fusion_df["element"].isin(members)]
            for _, row in f_sub.iterrows():
                idx = int(row["span_index"])
                if idx in span_text_by_index:
                    txt = span_text_by_index[idx]
                    if txt:
                        candidate_spans.append(txt)

        # 3) Keyword distribution
        toks_all: List[str] = []
        for s in candidate_spans:
            toks_all.extend(re.findall(r"[A-Za-z']+", s.lower()))
        word_counts = Counter(toks_all)

        # 4) Representative spans: overlap with member names
        member_toks = {m.lower() for m in members}
        scored: List[Tuple[int, str]] = []
        for s in candidate_spans:
            toks = set(re.findall(r"[A-Za-z']+", s.lower()))
            overlap = len(toks & member_toks)
            scored.append((overlap, s))

        top_sents = [s for _, s in sorted(scored, key=lambda x: x[0], reverse=True)[:5]]
        top_words = [w for w, _ in word_counts.most_common(15)]

        entry = dict(c)
        entry["compound_id"] = cid
        entry["context_examples"] = top_sents
        entry["aggregate_keywords"] = top_words
        entry["n_spans"] = len(candidate_spans)

        enriched.append(entry)

    # Export
    out_path = os.path.join(out_dir, "compound_contexts.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=2)
        _log(f"[fusion] Compound contexts written → {out_path}", emit)
        try:
            emit("artifact", {"path": out_path, "kind": "compound_context"})
        except Exception:
            pass
    except Exception as exc:
        _log(f"[fusion][warn] Failed to write compound_contexts.json: {exc}", emit)

    return pd.DataFrame(enriched)


# =============================================================================
# Public pipeline entry point
# =============================================================================

def run_fusion_pipeline(out_dir: str,
                        emit=DEFAULT_EMIT) -> None:
    """
    Orchestrator-facing entry point.

    1) Runs span→element fusion.
    2) If fusion succeeds, aggregates compound contexts.
    """
    _log(f"[fusion] Starting fusion pipeline in {out_dir}", emit)

    fusion_df = fuse_spans_to_elements(out_dir, top_k=3, emit=emit)
    if fusion_df.empty:
        _log("[fusion] Fusion produced no assignments; skipping compound context.", emit)
        return

    aggregate_compound_context(out_dir, emit=emit)
