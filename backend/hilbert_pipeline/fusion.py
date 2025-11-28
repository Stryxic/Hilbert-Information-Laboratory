"""
fusion.py
=========

Hilbert Pipeline – Span-to-Element Fusion Layer
-----------------------------------------------

This module implements the *fusion* stage of the Hilbert Information Pipeline.
It links **span embeddings** (from :mod:`hilbert_pipeline.lsa_layer`) to
**element embeddings** (from ``hilbert_elements.csv``), using cosine similarity
and entropy-aware adaptive thresholds, and returns:

1. ``span_element_fusion.csv``  
   Soft span→element assignments.

2. ``compound_contexts.json``  
   Aggregate semantic context summaries for molecule-level compounds.

This stage is required by all 2024–2025 orchestrators and meets the 2025
graph-contract specifications.

The fusion layer has two distinct responsibilities:

- **Micro-level association:**  
  For each span, determine its most likely semantic elements.

- **Macro-level aggregation:**  
  Given informational compounds (clusters from the molecule layer), summarise
  their supporting contexts and keywords.

Design Principles
-----------------

- Never mutates ``hilbert_elements.csv``.  
- Always writes new artefacts to the working ``results_dir``.  
- Embeddings are normalised (L2) before similarity computation.  
- Gracefully handles missing embeddings by falling back to centroid estimation.
- Thresholds are entropy-adaptive:  
  high-entropy spans are allowed more associations.

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


# =============================================================================
# Default no-op emitter
# =============================================================================

DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda *_: None


# =============================================================================
# Logging utilities
# =============================================================================

def _log(msg: str, emit: Callable = DEFAULT_EMIT) -> None:
    """
    Unified logging helper passed through both stdout and the orchestrator.

    Parameters
    ----------
    msg : str
        Human-readable message.
    emit : callable
        Emits structured logs via ``emit("log", {...})`` if supported.
    """
    print(msg)
    try:
        emit("log", {"message": msg})
    except Exception:
        pass


def _safe_float(x: Any, default: float = 0.0) -> float:
    """
    Convert any input to float safely.

    Parameters
    ----------
    x : Any
        Value to convert.
    default : float
        Fallback value.

    Returns
    -------
    float
    """
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _safe_json_load(path: str, emit=DEFAULT_EMIT):
    """
    Load a JSON file safely with structured warnings.

    Parameters
    ----------
    path : str
    emit : callable

    Returns
    -------
    dict or list or None
    """
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
# Embedding loading utilities
# =============================================================================

def _resolve_element_id_column(df: pd.DataFrame) -> str | None:
    """
    Determine which column should act as an element identifier.

    Priorities
    ----------
    1. ``element_id``  
    2. ``index``  
    3. ``None`` → use row index

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    str or None
    """
    if "element_id" in df.columns:
        return "element_id"
    if "index" in df.columns:
        return "index"
    return None


def _parse_vec(raw: Any) -> np.ndarray | None:
    """
    Parse a serialized vector.

    Supports:
    - actual lists
    - stringified JSON lists
    - safe-evaluated Python literal lists

    Parameters
    ----------
    raw : Any

    Returns
    -------
    numpy.ndarray or None
    """
    if isinstance(raw, (list, tuple)):
        arr = np.asarray(raw, float)
        return arr if arr.size else None

    if isinstance(raw, str):
        try:
            return np.asarray(json.loads(raw), float)
        except Exception:
            try:
                return np.asarray(eval(raw, {"__builtins__": {}}), float)
            except Exception:
                return None

    if isinstance(raw, np.ndarray):
        return raw if raw.size else None

    return None


def _load_span_embeddings(
    lsa_path: str,
    emit=DEFAULT_EMIT,
) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    """
    Load and normalise span embeddings from ``lsa_field.json``.

    Parameters
    ----------
    lsa_path : str
        Path to LSA field JSON.
    emit : callable

    Returns
    -------
    embeddings : np.ndarray
    span_map : dict
        Maps span index → metadata dict.
    """
    data = _safe_json_load(lsa_path, emit)
    if not isinstance(data, dict) or "embeddings" not in data:
        raise ValueError("lsa_field.json missing embeddings array.")

    emb = np.asarray(data["embeddings"], float)
    if emb.ndim != 2:
        raise ValueError("Invalid embeddings array in lsa_field.json.")

    # Normalise
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    span_map_index = {}
    for i, rec in enumerate(data.get("span_map", []) or []):
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
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Load element vectors from ``hilbert_elements.csv``.

    Requires a column ``embedding``.

    Parameters
    ----------
    elements_csv : str
    emit : callable

    Returns
    -------
    elem_vecs : np.ndarray
    elem_labels : list[str]
    elem_ids : list[int]
    """
    df = pd.read_csv(elements_csv)
    if "element" not in df.columns or "embedding" not in df.columns:
        raise ValueError("hilbert_elements.csv lacks 'embedding' column.")

    id_col = _resolve_element_id_column(df)

    elem_labels, elem_ids, vecs = [], [], []

    for row_idx, row in df.iterrows():
        label = str(row["element"])
        vec = _parse_vec(row["embedding"])
        if vec is None:
            continue

        if id_col:
            try:
                eid = int(row[id_col])
            except Exception:
                eid = row_idx
        else:
            eid = row_idx

        elem_labels.append(label)
        elem_ids.append(eid)
        vecs.append(vec.astype(float))

    if not vecs:
        raise ValueError("No valid element embeddings in CSV.")

    X = np.vstack(vecs)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X, elem_labels, elem_ids


def _derive_element_centroids_from_span_map(
    lsa_path: str,
    elements_csv: str,
    emit=DEFAULT_EMIT,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Construct element embeddings using span centroids.

    Used when CSV lacks ``embedding`` column.

    Parameters
    ----------
    lsa_path : str
    elements_csv : str

    Returns
    -------
    elem_vecs : np.ndarray
    elem_labels : list[str]
    elem_ids : list[int]
    """
    data = _safe_json_load(lsa_path, emit)
    if not isinstance(data, dict):
        raise ValueError("lsa_field.json unreadable.")

    span_emb = np.asarray(data.get("embeddings", []), float)
    span_emb = span_emb / (np.linalg.norm(span_emb, axis=1, keepdims=True) + 1e-12)

    span_map = data.get("span_map") or []
    if span_emb.ndim != 2 or not span_map:
        raise ValueError("No span embeddings / span_map for centroid fallback.")

    # bucket element occurrences
    buckets = defaultdict(list)
    for i, rec in enumerate(span_map):
        for el in rec.get("elements") or []:
            buckets[str(el)].append(i)

    df = pd.read_csv(elements_csv)
    if "element" not in df.columns:
        raise ValueError("hilbert_elements.csv missing 'element'.")

    id_col = _resolve_element_id_column(df)

    elem_labels, elem_ids, vecs = [], [], []

    for row_idx, row in df.iterrows():
        label = str(row["element"])
        idxs = buckets.get(label, [])
        if not idxs:
            continue

        v = span_emb[idxs].mean(axis=0)

        if id_col:
            try:
                eid = int(row[id_col])
            except Exception:
                eid = row_idx
        else:
            eid = row_idx

        elem_labels.append(label)
        elem_ids.append(eid)
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
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Hybrid element loading strategy.

    Attempts:
    1. Load direct embeddings from CSV.
    2. On failure: derive centroids from span embeddings.

    Parameters
    ----------
    elements_csv : str
    lsa_path : str

    Returns
    -------
    elem_vecs : np.ndarray
    elem_labels : list[str]
    elem_ids : list[int]
    """
    try:
        return _load_element_embeddings_from_csv(elements_csv, emit)
    except Exception as e:
        _log(f"[fusion] Falling back to centroid embeddings: {e}", emit)
        return _derive_element_centroids_from_span_map(lsa_path, elements_csv, emit)


# =============================================================================
# Adaptive Similarity Threshold
# =============================================================================

def adaptive_threshold(
    entropy: float,
    base: float = 0.22,
    min_t: float = 0.15,
    max_t: float = 0.40,
) -> float:
    """
    Compute an entropy-aware similarity threshold.

    Parameters
    ----------
    entropy : float
        Span entropy (0 → low uncertainty, >3 → high uncertainty).
    base : float
        Soft reference value (unused but retained for API stability).
    min_t : float
        Minimum threshold (for high entropy).
    max_t : float
        Maximum threshold (for low entropy).

    Returns
    -------
    float
        Threshold for span→element similarity filtering.

    Notes
    -----
    High entropy → lower threshold (more permissive).  
    Low entropy → higher threshold (careful assignment).
    """
    if entropy <= 0.1:
        return max_t
    if entropy >= 4.0:
        return min_t

    scale = np.exp(-entropy / 2.8)
    return min_t + (max_t - min_t) * scale


# =============================================================================
# Core Fusion Stage
# =============================================================================

def fuse_spans_to_elements(
    results_dir: str,
    top_k: int = 3,
    emit=DEFAULT_EMIT,
) -> pd.DataFrame:
    """
    Fuse spans with elements using cosine similarity.

    Outputs
    -------
    span_element_fusion.csv

    Columns:
    - span_index
    - span_id
    - doc
    - element
    - element_id
    - similarity
    - threshold

    Parameters
    ----------
    results_dir : str
        Directory containing ``lsa_field.json`` and ``hilbert_elements.csv``.
    top_k : int
        Max number of candidate elements per span.
    emit : callable

    Returns
    -------
    pandas.DataFrame
        Fusion results.
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

    meta = _safe_json_load(lsa_path, emit) or {}
    H_span = meta.get("H_span")
    if not isinstance(H_span, list):
        H_span = [1.0] * sims.shape[0]

    rows = []

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
# Compound Context Aggregation
# =============================================================================

def aggregate_compound_context(
    results_dir: str,
    elements_csv: str = "hilbert_elements.csv",
    compounds_json: str = "informational_compounds.json",
    fuse_csv: str = "span_element_fusion.csv",
    emit=DEFAULT_EMIT,
) -> pd.DataFrame:
    """
    Summarise compound-level context into ``compound_contexts.json``.

    Parameters
    ----------
    results_dir : str
    elements_csv : str
    compounds_json : str
    fuse_csv : str
    emit : callable

    Returns
    -------
    pandas.DataFrame

    Schema
    ------
    - compound_id
    - context_examples (top spans)
    - aggregate_keywords
    - n_spans
    - + all original compound fields
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

    enriched = []

    for c in compounds:
        if not isinstance(c, dict):
            continue

        members = [str(e) for e in c.get("elements", [])]
        cid = c.get("compound_id")

        collected = []

        if not fusion_df.empty:
            sub = fusion_df[fusion_df["element"].isin(members)]
            for _, row in sub.iterrows():
                idx = int(row["span_index"])
                if 0 <= idx < len(span_map):
                    txt = span_map[idx].get("text", "")
                    if txt:
                        collected.append(txt)

        toks_all = []
        for s in collected:
            toks_all.extend(re.findall(r"[A-Za-z']+", s.lower()))
        counts = Counter(toks_all)

        member_lower = {m.lower() for m in members}
        scored = []
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
    Execute the full fusion stage:

    1. Run :func:`fuse_spans_to_elements`
    2. Run :func:`aggregate_compound_context`

    Parameters
    ----------
    results_dir : str
    emit : callable

    Returns
    -------
    None

    Notes
    -----
    This function is called by the Hilbert Orchestrator during stage ``fusion``.
    """
    fuse_spans_to_elements(results_dir, emit=emit)
    aggregate_compound_context(results_dir, emit=emit)
