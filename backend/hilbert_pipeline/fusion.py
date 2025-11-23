# =============================================================================
# hilbert_pipeline/fusion.py — Span-to-Element Fusion Layer (Option A)
# =============================================================================
"""
Option A: fusion loads hilbert_elements.csv to map element → element_id.

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

Key differences in Option A:
  - Element embeddings are always aligned with hilbert_elements.csv.
  - We propagate a stable element identifier (element_id) taken from
    hilbert_elements.csv (prefer 'element_id', fall back to 'index').
  - span_element_fusion.csv now includes both 'element' (string) and
    'element_id' (int) for downstream consistency.

This version is aligned with the upgraded orchestrator & molecule layer and
supports orchestrator event streaming via emit().
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# injected by orchestrator; overridden at runtime
DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda *_: None


# =============================================================================
# Logging utilities
# =============================================================================

def _log(msg: str, emit=DEFAULT_EMIT):
    """Best-effort logging that also streams to the orchestrator."""
    print(msg)
    try:
        emit("log", {"message": msg})
    except Exception:
        # emit is best-effort
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

def _load_span_embeddings(lsa_path: str, emit=DEFAULT_EMIT) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    """
    Load span embeddings and span_map from lsa_field.json.

    Returns
    -------
    span_vecs : np.ndarray (n_spans, d), L2 normalised
    span_map_indexed : dict[int, dict]
        Mapping from span_index -> span_record (with keys: doc, span_id, text, elements)
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
    span_vecs = arr / n

    # Build span_index -> record mapping
    span_map = data.get("span_map", []) or []
    span_map_indexed: Dict[int, Dict[str, Any]] = {}

    for i, rec in enumerate(span_map):
        # Normalise record structure a bit
        if not isinstance(rec, dict):
            rec = {}
        rec_out = {
            "doc": rec.get("doc"),
            "span_id": rec.get("span_id", i),
            "text": rec.get("text", ""),
            "elements": rec.get("elements", []),
        }
        span_map_indexed[i] = rec_out

    return span_vecs, span_map_indexed


def _parse_vec(raw: Any) -> Optional[np.ndarray]:
    if isinstance(raw, str):
        # JSON-style or repr-style list
        try:
            arr = np.asarray(json.loads(raw), dtype=float)
        except Exception:
            try:
                arr = np.asarray(eval(raw, {"__builtins__": {}}), dtype=float)
            except Exception:
                return None
        return arr if arr.size > 0 else None

    if isinstance(raw, (list, tuple)):
        v = np.asarray(raw, dtype=float)
        return v if v.size > 0 else None

    if isinstance(raw, np.ndarray):
        return raw if raw.size > 0 else None

    return None


def _resolve_element_id_column(df: pd.DataFrame) -> Optional[str]:
    """
    Decide which column in hilbert_elements.csv should serve as the stable
    element identifier.

    Preference:
      1) 'element_id' (if present)
      2) 'index'      (LSA index written by orchestrator)
      3) None         (caller will have to fall back to row order)
    """
    if "element_id" in df.columns:
        return "element_id"
    if "index" in df.columns:
        return "index"
    return None


def _load_element_embeddings_from_csv(
    elements_csv: str,
    emit=DEFAULT_EMIT,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Primary path: load element embeddings from hilbert_elements.csv when an
    'embedding' column is present, aligned with element → element_id.

    Returns
    -------
    elem_vecs : np.ndarray       (n_elements, d)
    elem_ids  : list[str]        element string labels
    elem_idxs : list[int]        element_id / index from CSV
    """
    df = pd.read_csv(elements_csv)
    if "element" not in df.columns or "embedding" not in df.columns:
        raise ValueError("hilbert_elements.csv missing 'embedding' column.")

    id_col = _resolve_element_id_column(df)

    el_ids: List[str] = []
    el_indices: List[int] = []
    vecs: List[np.ndarray] = []

    for row_idx, row in df.iterrows():
        el = str(row["element"])
        v = _parse_vec(row["embedding"])
        if v is None:
            continue
        v = v.astype(float)
        if not np.any(v):
            continue

        if id_col is not None:
            try:
                el_id = int(row[id_col])
            except Exception:
                el_id = int(row_idx)
        else:
            el_id = int(row_idx)

        el_ids.append(el)
        el_indices.append(el_id)
        vecs.append(v)

    if not el_ids:
        raise ValueError("No valid element embeddings in CSV.")

    X = np.vstack(vecs)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms
    return X, el_ids, el_indices


def _derive_element_centroids_from_span_map(
    lsa_path: str,
    elements_csv: str,
    emit=DEFAULT_EMIT,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Fallback: derive element centroids directly from the LSA span embeddings +
    span_map 'elements' lists, but align them with hilbert_elements.csv so that
    element → element_id is consistent across the pipeline.

    This does NOT depend on hilbert_elements.csv containing span_id.
    """
    data = _safe_json_load(lsa_path, emit=emit)
    if not isinstance(data, dict):
        raise ValueError("lsa_field.json invalid for centroid derivation.")

    span_emb = np.asarray(data.get("embeddings", []), dtype=float)
    span_map = data.get("span_map", []) or []

    if span_emb.ndim != 2 or span_emb.size == 0 or not span_map:
        raise ValueError("No span embeddings or span_map for centroid fallback.")

    # L2 normalise span embeddings
    norms = np.linalg.norm(span_emb, axis=1, keepdims=True) + 1e-12
    span_emb = span_emb / norms

    # Aggregate embeddings by element using span_map.elements
    buckets: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(span_map):
        if not isinstance(rec, dict):
            continue
        elems = rec.get("elements") or []
        if not isinstance(elems, (list, tuple)):
            continue
        for el in elems:
            el_str = str(el)
            buckets[el_str].append(i)

    # Load canonical element list + IDs from hilbert_elements.csv
    elements_df = pd.read_csv(elements_csv)
    if "element" not in elements_df.columns:
        raise ValueError("hilbert_elements.csv missing 'element' column.")

    id_col = _resolve_element_id_column(elements_df)

    el_ids: List[str] = []
    el_indices: List[int] = []
    vecs: List[np.ndarray] = []

    for row_idx, row in elements_df.iterrows():
        el = str(row["element"])
        idxs = buckets.get(el, [])
        if not idxs:
            # This element never actually appears in span_map; skip it for
            # centroid estimation. It will still exist in hilbert_elements.csv
            # but simply won't be used in fusion similarity comparisons.
            continue

        v = span_emb[idxs].mean(axis=0)

        if id_col is not None:
            try:
                el_id = int(row[id_col])
            except Exception:
                el_id = int(row_idx)
        else:
            el_id = int(row_idx)

        el_ids.append(el)
        el_indices.append(el_id)
        vecs.append(v)

    if not el_ids:
        raise ValueError("No valid element centroids derived from span_map.")

    X = np.vstack(vecs)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms
    return X, el_ids, el_indices


def _load_element_embeddings(
    elements_csv: str,
    lsa_path: str,
    emit=DEFAULT_EMIT,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Hybrid loader:
      - first try CSV 'embedding' column
      - if missing, fall back to LSA-derived centroids via span_map.elements,
        always keeping element → element_id aligned with hilbert_elements.csv.
    """
    try:
        return _load_element_embeddings_from_csv(elements_csv, emit=emit)
    except Exception as e_csv:
        _log(
            "[fusion][info] Falling back to LSA-derived centroids because CSV "
            f"embeddings failed: {e_csv}",
            emit,
        )
        return _derive_element_centroids_from_span_map(
            lsa_path=lsa_path,
            elements_csv=elements_csv,
            emit=emit,
        )


# =============================================================================
# Adaptive Similarity Thresholds
# =============================================================================

def adaptive_sim_threshold(entropy: float,
                           base=0.22,
                           min_t=0.15,
                           max_t=0.40) -> float:
    """
    High-entropy spans need lower similarity thresholds.
    Low-entropy spans require higher similarity to count as meaningful.
    """
    if entropy <= 0.1:
        return max_t
    if entropy >= 4.0:
        return min_t

    scale = np.exp(-entropy / 2.8)
    return min_t + (max_t - min_t) * scale


# =============================================================================
# FUSION CORE
# =============================================================================

def fuse_spans_to_elements(out_dir: str,
                           top_k: int = 3,
                           emit=DEFAULT_EMIT) -> pd.DataFrame:
    """
    Produces a CSV: span_index, span_id, doc, element, element_id, similarity, threshold.

    Uses adaptive thresholding based on span entropy when available.
    element_id is taken from hilbert_elements.csv (column 'element_id' if present,
    otherwise 'index', otherwise row order).
    """
    lsa_path = os.path.join(out_dir, "lsa_field.json")
    el_path = os.path.join(out_dir, "hilbert_elements.csv")

    if not os.path.exists(lsa_path) or not os.path.exists(el_path):
        _log("[fusion] Missing LSA field or elements; skipping.", emit)
        return pd.DataFrame()

    # -- load embeddings & span map ------------------------------------------
    try:
        span_vecs, span_map_indexed = _load_span_embeddings(lsa_path, emit)
        elem_vecs, elem_ids, elem_indices = _load_element_embeddings(el_path, lsa_path, emit)
    except Exception as e:
        _log(f"[fusion][warn] Failed embedding load: {e}", emit)
        return pd.DataFrame()

    if span_vecs.shape[0] == 0 or elem_vecs.shape[0] == 0:
        _log("[fusion][warn] Empty span or element embeddings; skipping fusion.", emit)
        return pd.DataFrame()

    sims = cosine_similarity(span_vecs, elem_vecs)

    # load entropies for adaptive thresholds, if present
    span_data = _safe_json_load(lsa_path, emit=emit)
    H_span = span_data.get("H_span") if isinstance(span_data, dict) else None
    if not isinstance(H_span, list):
        H_span = [1.0] * sims.shape[0]  # fallback flat entropy

    rows = []
    for i in range(sims.shape[0]):
        row = sims[i]
        idxs = np.argsort(row)[::-1][:top_k]

        thr = adaptive_sim_threshold(_safe_float(H_span[i], 1.0))

        span_rec = span_map_indexed.get(i, {})
        span_id = span_rec.get("span_id", i)
        doc = span_rec.get("doc")

        for j in idxs:
            sim = float(row[j])
            if sim < thr:
                continue
            rows.append({
                "span_index": i,
                "span_id": span_id,
                "doc": doc,
                "element": elem_ids[j],
                "element_id": elem_indices[j],
                "similarity": sim,
                "threshold": thr,
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
        - compound signature vector (mean element embedding, future work)
    """

    el_path = os.path.join(out_dir, elements_csv)
    c_path = os.path.join(out_dir, compounds_json)
    f_path = os.path.join(out_dir, fuse_csv)

    if not os.path.exists(el_path):
        _log(f"[fusion][warn] Missing elements CSV: {el_path}", emit)
        return pd.DataFrame()

    elements_df = pd.read_csv(el_path)
    compounds_raw = _safe_json_load(c_path, emit=emit)
    if isinstance(compounds_raw, dict) and "compounds" in compounds_raw:
        compounds = list(compounds_raw["compounds"])
    elif isinstance(compounds_raw, dict):
        compounds = list(compounds_raw.values())
    else:
        compounds = compounds_raw if isinstance(compounds_raw, list) else []

    fusion_df = pd.read_csv(f_path) if os.path.exists(f_path) else pd.DataFrame()

    # Preload example spans if present in elements_df (legacy support)
    span_cache: Dict[str, List[str]] = {}
    if "span" in elements_df.columns:
        for _, row in elements_df.iterrows():
            el = str(row["element"])
            txt = str(row.get("span") or row.get("context") or "").strip()
            if txt:
                span_cache.setdefault(el, []).append(txt)

    # Also load span_map so we can pull actual text contexts from LSA
    lsa_json = _safe_json_load(os.path.join(out_dir, "lsa_field.json"), emit)
    span_map = lsa_json.get("span_map") if isinstance(lsa_json, dict) else []
    span_map = span_map or []

    enriched = []
    for c in compounds:
        if not isinstance(c, dict):
            continue

        cid = c.get("compound_id")
        members = [str(e) for e in c.get("elements", [])]

        # 1) Gather spans from span-cache and fusion assignments
        candidate_spans: List[str] = []

        # from legacy span_cache
        for el in members:
            candidate_spans.extend(span_cache.get(el, []))

        # from fusion assignments + LSA span_map
        if not fusion_df.empty and "element" in fusion_df.columns:
            f_sub = fusion_df[fusion_df["element"].isin(members)]
            for _, row in f_sub.iterrows():
                idx = int(row.get("span_index", -1))
                if 0 <= idx < len(span_map):
                    txt = span_map[idx].get("text", "")
                    if txt:
                        candidate_spans.append(str(txt))

        # 2) Build keyword distribution
        toks_all: List[str] = []
        for s in candidate_spans:
            toks_all.extend(re.findall(r"[A-Za-z']+", s.lower()))
        word_counts = Counter(toks_all)

        # 3) Representative spans: high overlap with member names
        member_toks = {m.lower() for m in members}
        scored: List[Tuple[int, str]] = []
        for s in candidate_spans:
            toks = set(re.findall(r"[A-Za-z']+", s.lower()))
            overlap = len(toks & member_toks)
            scored.append((overlap, s))

        top_sents = [s for _, s in sorted(scored, reverse=True)[:5]]
        top_words = [w for w, _ in word_counts.most_common(15)]

        entry = dict(c)
        entry["compound_id"] = cid
        entry["context_examples"] = top_sents
        entry["aggregate_keywords"] = top_words
        entry["n_spans"] = len(candidate_spans)

        enriched.append(entry)

    # Export
    out_path = os.path.join(out_dir, "compound_contexts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    _log(f"[fusion] Compound contexts written → {out_path}", emit)
    try:
        emit("artifact", {"path": out_path, "kind": "compound_context"})
    except Exception:
        pass
    return pd.DataFrame(enriched)


# =============================================================================
# Orchestrator-facing convenience
# =============================================================================

def run_fusion_pipeline(results_dir: str, emit=DEFAULT_EMIT) -> None:
    """
    Thin wrapper combining:
      - span→element fusion
      - compound context aggregation
    """
    fuse_spans_to_elements(results_dir, emit=emit)
    aggregate_compound_context(results_dir, emit=emit)
