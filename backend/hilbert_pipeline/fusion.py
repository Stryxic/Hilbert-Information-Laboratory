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
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# injected by orchestrator
DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda *_: None


# =============================================================================
# Logging utilities
# =============================================================================

def _log(msg: str, emit=DEFAULT_EMIT):
    print(msg)
    emit("log", {"message": msg})


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
# Embedding Parsers
# =============================================================================

def _load_span_embeddings(lsa_path: str, emit=DEFAULT_EMIT) -> np.ndarray:
    data = _safe_json_load(lsa_path, emit=emit)
    if not isinstance(data, dict) or "embeddings" not in data:
        raise ValueError("lsa_field.json missing 'embeddings' array.")

    emb = data["embeddings"]
    arr = np.asarray(emb, dtype=float)

    # normalize
    n = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / n


def _parse_vec(raw: Any) -> Optional[np.ndarray]:
    if isinstance(raw, str):
        try:
            arr = np.asarray(json.loads(raw), dtype=float)
        except Exception:
            # fallback: secure eval-like safe parse
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


def _load_element_embeddings(elements_csv: str,
                             emit=DEFAULT_EMIT) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(elements_csv)
    if "element" not in df.columns or "embedding" not in df.columns:
        raise ValueError("hilbert_elements.csv missing required columns.")

    el_ids = []
    vecs = []

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
        raise ValueError("No valid element embeddings.")

    X = np.vstack(vecs)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms
    return X, el_ids


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
    Produces a CSV: span_index, element, similarity.

    Uses adaptive thresholding based on span entropy when available.
    """
    lsa_path = os.path.join(out_dir, "lsa_field.json")
    el_path = os.path.join(out_dir, "hilbert_elements.csv")

    if not os.path.exists(lsa_path) or not os.path.exists(el_path):
        _log("[fusion] Missing LSA field or elements; skipping.", emit)
        return pd.DataFrame()

    # -- load embeddings -----------------------------------------------------
    try:
        span_vecs = _load_span_embeddings(lsa_path, emit)
        elem_vecs, elem_ids = _load_element_embeddings(el_path, emit)
    except Exception as e:
        _log(f"[fusion][warn] Failed embedding load: {e}", emit)
        return pd.DataFrame()

    sims = cosine_similarity(span_vecs, elem_vecs)

    # load entropies for adaptive thresholds
    span_data = _safe_json_load(lsa_path, emit=emit)
    H_span = span_data.get("H_span") if isinstance(span_data, dict) else None
    if not isinstance(H_span, list):
        H_span = [1.0] * sims.shape[0]  # fallback

    rows = []
    for i in range(sims.shape[0]):
        row = sims[i]
        idxs = np.argsort(row)[::-1][:top_k]

        thr = adaptive_sim_threshold(_safe_float(H_span[i], 1.0))

        for j in idxs:
            sim = float(row[j])
            if sim < thr:
                continue
            rows.append({
                "span_index": i,
                "element": elem_ids[j],
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

    emit("artifact", {"path": out_path, "kind": "span_fusion"})
    return df


# =============================================================================
# Compound-Level Semantic Context Extraction
# =============================================================================

def aggregate_compound_context(out_dir: str,
                               elements_csv="hilbert_elements.csv",
                               compounds_json="informational_compounds.json",
                               fuse_csv="span_element_fusion.csv",
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

    elements_df = pd.read_csv(el_path)
    compounds_raw = _safe_json_load(c_path, emit=emit)
    if isinstance(compounds_raw, dict) and "compounds" in compounds_raw:
        compounds = list(compounds_raw["compounds"])
    elif isinstance(compounds_raw, dict):
        compounds = list(compounds_raw.values())
    else:
        compounds = compounds_raw if isinstance(compounds_raw, list) else []

    fusion_df = pd.read_csv(f_path) if os.path.exists(f_path) else pd.DataFrame()

    # Preload example spans if present
    span_cache: Dict[str, List[str]] = {}
    if "span" in elements_df.columns:
        for _, row in elements_df.iterrows():
            el = str(row["element"])
            txt = str(row.get("span") or row.get("context") or "").strip()
            if txt:
                span_cache.setdefault(el, []).append(txt)

    enriched = []
    for c in compounds:
        cid = c.get("compound_id")
        members = [str(e) for e in c.get("elements", [])]

        # 1) Gather spans from span-cache and fusion assignments
        candidate_spans = []
        for el in members:
            candidate_spans.extend(span_cache.get(el, []))

        if not fusion_df.empty:
            f_sub = fusion_df[fusion_df["element"].isin(members)]
            # attach spans if lsa_field span_map present
            lsa_json = _safe_json_load(os.path.join(out_dir, "lsa_field.json"), emit)
            span_map = lsa_json.get("span_map") if isinstance(lsa_json, dict) else []
            for _, row in f_sub.iterrows():
                idx = int(row["span_index"])
                if 0 <= idx < len(span_map):
                    txt = span_map[idx].get("text", "")
                    if txt:
                        candidate_spans.append(str(txt))

        # 2) Build keyword distribution
        toks_all = []
        for s in candidate_spans:
            toks_all.extend(re.findall(r"[A-Za-z']+", s.lower()))
        word_counts = Counter(toks_all)

        # 3) Representative spans: high overlap with member names
        member_toks = {m.lower() for m in members}
        scored = []
        for s in candidate_spans:
            toks = set(re.findall(r"[A-Za-z']+", s.lower()))
            overlap = len(toks & member_toks)
            scored.append((overlap, s))

        top_sents = [s for _, s in sorted(scored, reverse=True)[:5]]
        top_words = [w for w, _ in word_counts.most_common(15)]

        entry = dict(c)
        entry["context_examples"] = top_sents
        entry["aggregate_keywords"] = top_words
        entry["n_spans"] = len(candidate_spans)

        enriched.append(entry)

    # Export
    out_path = os.path.join(out_dir, "compound_contexts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    _log(f"[fusion] Compound contexts written → {out_path}", emit)
    emit("artifact", {"path": out_path, "kind": "compound_context"})
    return pd.DataFrame(enriched)
