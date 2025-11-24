# =============================================================================
# hilbert_pipeline/edges_builder.py — Element co-occurrence edges (Pipeline 3.1)
# =============================================================================
"""
Builds a semantically meaningful element-element edge list (edges.csv)
from span_element_fusion.csv.

New in Pipeline 3.1:
  - Produces edges matching the Hilbert Graph Contract:
        source, target,
        weight,
        scaled_weight,
        polarity,
        confidence,
        is_backbone
  - Replicates legacy behaviour (co-occurrence graph) but upgrades it with:
        min_coocc filtering,
        per-node top-k pruning,
        backbone extraction,
        confidence scaling,
        reproducible ordering.

Downstream consumers:
  - Molecule Layer
  - Compound aggregation
  - Graph visualizer and snapshot engine
  - Analytics (centrality, hub detection, epistemic geometry)
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Any, Tuple, List
from collections import defaultdict

import numpy as np
import pandas as pd

DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda *_a, **_k: None


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def _log(emit, level, msg, **fields):
    payload = {"level": level, "msg": msg}
    payload.update(fields)
    try:
        emit("log", payload)
    except Exception:
        print(f"[{level}] {msg} {fields}")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Column resolution
# -----------------------------------------------------------------------------
def _resolve(df: pd.DataFrame, candidates: List[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"Missing required column among: {candidates}")


def _load_valid_elements(results_dir: str) -> List[str]:
    path = os.path.join(results_dir, "hilbert_elements.csv")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if "element" not in df.columns:
        return []
    return sorted(df["element"].astype(str).unique().tolist())


def _load_element_id_map(results_dir: str) -> Dict[str, int]:
    """
    Create mapping element(string) -> element_id (int)
    ensuring ID consistency across pipeline layers.
    """
    path = os.path.join(results_dir, "hilbert_elements.csv")
    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path)
    if "element" not in df.columns:
        return {}

    if "element_id" in df.columns:
        id_col = "element_id"
    elif "index" in df.columns:
        id_col = "index"
    else:
        id_col = None

    mapping = {}
    for i, row in df.iterrows():
        el = str(row["element"])
        if id_col:
            try:
                el_id = int(row[id_col])
            except Exception:
                el_id = i
        else:
            el_id = i
        mapping[el] = el_id

    return mapping


# -----------------------------------------------------------------------------
# Backbone extraction
# -----------------------------------------------------------------------------
def _extract_backbone(edges_df: pd.DataFrame, keep_fraction: float = 0.20) -> pd.Series:
    """
    Mark top edges as backbone. keep_fraction is the percentage of strongest
    edges to retain as backbone.

    Returns a boolean Series aligned with edges_df index.
    """
    if edges_df.empty:
        return pd.Series([False] * 0)

    n = max(1, int(len(edges_df) * keep_fraction))
    sorted_idx = edges_df["weight"].sort_values(ascending=False).index[:n]
    mask = pd.Series(False, index=edges_df.index)
    mask.loc[sorted_idx] = True
    return mask


# -----------------------------------------------------------------------------
# Main builder
# -----------------------------------------------------------------------------
def build_element_edges(
    results_dir: str,
    emit=DEFAULT_EMIT,
    min_coocc: int = 1,
    top_k: int = 30,
    max_edges: int = 250000,
    backbone_fraction: float = 0.20,
) -> str:
    """
    Build edges.csv under results_dir using span_element_fusion.csv.

    Output columns match Hilbert Graph Contract:
        source, target,
        weight,
        scaled_weight,
        polarity,
        confidence,
        is_backbone
    """
    fusion_path = os.path.join(results_dir, "span_element_fusion.csv")
    out_path = os.path.join(results_dir, "edges.csv")

    if not os.path.exists(fusion_path):
        _log(emit, "warn", "span_element_fusion.csv not found", path=fusion_path)
        return ""

    try:
        df = pd.read_csv(fusion_path)
    except Exception as exc:
        _log(emit, "warn", "Failed to read fusion CSV", error=str(exc))
        return ""

    if df.empty:
        _log(emit, "warn", "Empty fusion CSV; no edges.")
        return ""

    # Identify span and element columns
    try:
        span_col = _resolve(df, ["span_index", "span_id", "span"])
    except Exception as exc:
        _log(emit, "warn", f"{exc}; cannot build edges.")
        return ""

    try:
        elem_col = _resolve(df, ["element", "element_id"])
    except Exception as exc:
        _log(emit, "warn", f"{exc}; cannot build edges.")
        return ""

    valid_elements = _load_valid_elements(results_dir)
    if valid_elements:
        before = len(df)
        df = df[df[elem_col].astype(str).isin(valid_elements)]
        _log(emit, "info", "Filtered fusion rows to valid elements",
             before=before, after=len(df))
        if df.empty:
            _log(emit, "warn", "Filtering removed all rows; no edges.")
            return ""

    element_id_map = _load_element_id_map(results_dir)

    # -------------------------------------------------------------------------
    # 1. Co-occurrence accumulation
    # -------------------------------------------------------------------------
    coocc: dict[tuple[str, str], float] = defaultdict(float)

    for _, g in df.groupby(span_col):
        elems = sorted(set(str(e) for e in g[elem_col].dropna()))
        if len(elems) <= 1:
            continue

        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                a, b = elems[i], elems[j]
                if a > b:
                    a, b = b, a
                coocc[(a, b)] += 1.0

    if not coocc:
        _log(emit, "warn", "No co-occurring pairs found.")
        return ""

    # -------------------------------------------------------------------------
    # 2. Convert to DataFrame with graph-contract fields
    # -------------------------------------------------------------------------
    rows = []
    for (src, tgt), w in coocc.items():
        rows.append({
            "source": src,
            "target": tgt,
            "weight": float(w),
        })

    edges = pd.DataFrame(rows)
    _log(emit, "info", "Constructed raw edges", n_edges=len(edges))

    # Filter on minimum cooccurrence
    if min_coocc > 1:
        before = len(edges)
        edges = edges[edges["weight"] >= float(min_coocc)]
        _log(emit, "info", "Applied min_coocc", before=before, after=len(edges))

    if edges.empty:
        _log(emit, "warn", "All edges filtered by min_coocc.")
        return ""

    # -------------------------------------------------------------------------
    # 3. Per-node top-k pruning
    # -------------------------------------------------------------------------
    if top_k and top_k > 0:
        before = len(edges)
        top_src = (
            edges.sort_values("weight", ascending=False)
            .groupby("source")
            .head(top_k)
        )
        top_tgt = (
            edges.sort_values("weight", ascending=False)
            .groupby("target")
            .head(top_k)
        )
        edges = pd.concat([top_src, top_tgt], ignore_index=True)
        edges = edges.drop_duplicates(subset=["source", "target"])
        _log(emit, "info", "Applied per-node top_k", before=before, after=len(edges))

    # -------------------------------------------------------------------------
    # 4. Safety cap
    # -------------------------------------------------------------------------
    if max_edges and len(edges) > max_edges:
        before = len(edges)
        edges = edges.sort_values("weight", ascending=False).head(max_edges)
        _log(emit, "warn", "Truncated edges to max_edges",
             before=before, after=len(edges))

    # -------------------------------------------------------------------------
    # 5. Compute scaled_weight, polarity, confidence
    # -------------------------------------------------------------------------
    if not edges.empty:
        w = edges["weight"]
        w_min, w_max = float(w.min()), float(w.max())
        if w_max > w_min:
            edges["scaled_weight"] = (w - w_min) / (w_max - w_min)
        else:
            edges["scaled_weight"] = 1.0

        edges["polarity"] = 1

        # confidence proxy: log-normalised cooccurrence frequency
        edges["confidence"] = np.log1p(edges["weight"]) / np.log1p(w_max)

    else:
        edges["scaled_weight"] = []
        edges["polarity"] = []
        edges["confidence"] = []

    # -------------------------------------------------------------------------
    # 6. Backbone extraction
    # -------------------------------------------------------------------------
    backbone_mask = _extract_backbone(edges, keep_fraction=backbone_fraction)
    edges["is_backbone"] = backbone_mask.astype(bool)

    # -------------------------------------------------------------------------
    # 7. Attach element_ids for downstream alignment
    # -------------------------------------------------------------------------
    edges["source_id"] = edges["source"].map(element_id_map).fillna(-1).astype(int)
    edges["target_id"] = edges["target"].map(element_id_map).fillna(-1).astype(int)

    # Sort deterministically
    edges = edges.sort_values(
        ["weight", "source", "target"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 8. Write edges.csv
    # -------------------------------------------------------------------------
    try:
        edges.to_csv(out_path, index=False)
    except Exception as exc:
        _log(emit, "warn", "Failed to write edges.csv", error=str(exc))
        return ""

    _log(
        emit, "info", "Wrote edges.csv",
        path=out_path, n_edges=len(edges),
        min_coocc=min_coocc, top_k=top_k
    )

    try:
        emit("artifact", {"path": out_path, "kind": "edges"})
    except Exception:
        pass

    return out_path
