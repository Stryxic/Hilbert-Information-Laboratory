"""
edges_builder.py
================

Hilbert Pipeline – Element Co-occurrence Edge Builder (Pipeline 3.1)
--------------------------------------------------------------------

This module converts ``span_element_fusion.csv`` into a fully compliant
**Hilbert Graph Contract** edge list:

    source, target,
    weight,
    scaled_weight,
    polarity,
    confidence,
    is_backbone,
    source_id,
    target_id

It builds a semantic co-occurrence network of elements based on shared span
assignments. This graph is consumed downstream by:

- Molecule layer (compound inference)
- Graph snapshot engine (2D/3D layouts)
- Analytics (centralities, epistemic geometry, stability)
- Compound context aggregation
- Orchestrator run summaries

Includes:

- Per-node top-k pruning
- Backbone extraction (top 20 percent edges)
- Confidence scoring via log-normalised co-occurrence
- Deterministic sorting
- ID mapping consistency with ``hilbert_elements.csv``
- Graph-contract-aligned output schema

This file is intentionally deterministic and free of random seeds so that
graph export is reproducible across environments.

"""

from __future__ import annotations

import os
from typing import Callable, Dict, Any, Tuple, List
from collections import defaultdict

import numpy as np
import pandas as pd


# =============================================================================
# Default no-op emitter (orchestrator injects real emitter)
# =============================================================================

DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda *_a, **_k: None


# =============================================================================
# Logging utilities
# =============================================================================

def _log(emit: Callable, level: str, msg: str, **fields) -> None:
    """
    Structured logger for pipeline stages.

    Parameters
    ----------
    emit : callable
        Orchestrator emitter, typically ``emit("log", {...})``.
    level : str
        Log severity: "info", "warn", "error".
    msg : str
        Human readable message.
    fields : dict
        Additional structured metadata.
    """
    payload = {"level": level, "msg": msg}
    payload.update(fields)
    try:
        emit("log", payload)
    except Exception:
        print(f"[{level}] {msg} {fields}")


def _safe_float(x: Any, default: float = 0.0) -> float:
    """
    Safely cast to float.

    Parameters
    ----------
    x : Any
    default : float

    Returns
    -------
    float
    """
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


# =============================================================================
# CSV column resolution
# =============================================================================

def _resolve(df: pd.DataFrame, candidates: List[str]) -> str:
    """
    Resolve a column name from multiple acceptable candidates.

    Parameters
    ----------
    df : pandas.DataFrame
    candidates : list[str]
        Column names to search for in order.

    Returns
    -------
    str
        The first existing column in ``candidates``.

    Raises
    ------
    ValueError
        If none of the candidate column names exist.
    """
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"Missing required column among: {candidates}")


def _load_valid_elements(results_dir: str) -> List[str]:
    """
    Load valid element labels from ``hilbert_elements.csv``.

    Parameters
    ----------
    results_dir : str

    Returns
    -------
    list[str]
        Sorted list of unique elements or empty list if CSV missing.
    """
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
    Create mapping ``element -> element_id`` to ensure consistent IDs
    across the pipeline: LSA, fusion, edges, molecules, export.

    Priority order for ID selection:
    1. ``element_id`` column
    2. ``index`` column
    3. row index

    Parameters
    ----------
    results_dir : str

    Returns
    -------
    dict[str, int]
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

    mapping: Dict[str, int] = {}
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


# =============================================================================
# Backbone extraction
# =============================================================================

def _extract_backbone(
    edges_df: pd.DataFrame,
    keep_fraction: float = 0.20
) -> pd.Series:
    """
    Identify the strongest edges to mark as graph backbone.

    Parameters
    ----------
    edges_df : pandas.DataFrame
        Edge list with at least a ``weight`` column.
    keep_fraction : float
        Fraction of edges to mark as backbone (strongest first).

    Returns
    -------
    pandas.Series (bool)
        Boolean mask aligned with ``edges_df.index``.
    """
    if edges_df.empty:
        return pd.Series([False] * 0)

    n = max(1, int(len(edges_df) * keep_fraction))
    sorted_idx = edges_df["weight"].sort_values(ascending=False).index[:n]

    mask = pd.Series(False, index=edges_df.index)
    mask.loc[sorted_idx] = True
    return mask


# =============================================================================
# Main Edge Builder
# =============================================================================

def build_element_edges(
    results_dir: str,
    emit: Callable = DEFAULT_EMIT,
    min_coocc: int = 1,
    top_k: int = 30,
    max_edges: int = 250000,
    backbone_fraction: float = 0.20,
) -> str:
    """
    Build an element-element co-occurrence graph under ``results_dir``.

    Reads:
        ``span_element_fusion.csv``

    Writes:
        ``edges.csv`` (Hilbert Graph Contract schema)

    Parameters
    ----------
    results_dir : str
        Directory produced by the orchestrator run.
    emit : callable
        Structured logging emitter.
    min_coocc : int
        Minimum co-occurrence count to retain an edge.
    top_k : int
        For each node: keep only top-k strongest edges by weight.
    max_edges : int
        Safety cap to prevent runaway graph sizes.
    backbone_fraction : float
        Fraction of edges marked as backbone.

    Returns
    -------
    str
        Path to ``edges.csv`` or empty string on failure.

    Output Schema
    -------------
    The resulting CSV includes:

    - ``source`` (str)
    - ``target`` (str)
    - ``weight`` (float)
    - ``scaled_weight`` (float 0–1)
    - ``polarity`` (int, currently +1)
    - ``confidence`` (float)
    - ``is_backbone`` (bool)
    - ``source_id``, ``target_id`` (int)

    Notes
    -----
    Co-occurrence is computed within spans, not documents. This emphasises
    local semantic neighbourhoods rather than corpus-scale frequency.
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

    # Resolve span and element fields
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

    # Filter to valid elements
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

    # =========================================================================
    # 1. Co-occurrence computation
    # =========================================================================
    coocc: Dict[Tuple[str, str], float] = defaultdict(float)

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

    # =========================================================================
    # 2. Build DataFrame with graph-contract fields
    # =========================================================================
    rows = [{"source": a, "target": b, "weight": float(w)} for (a, b), w in coocc.items()]
    edges = pd.DataFrame(rows)
    _log(emit, "info", "Constructed raw edges", n_edges=len(edges))

    # Min co-occurrence filtering
    if min_coocc > 1:
        before = len(edges)
        edges = edges[edges["weight"] >= float(min_coocc)]
        _log(emit, "info", "Applied min_coocc", before=before, after=len(edges))
        if edges.empty:
            _log(emit, "warn", "All edges filtered by min_coocc.")
            return ""

    # =========================================================================
    # 3. Per-node top-k pruning
    # =========================================================================
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
        if edges.empty:
            _log(emit, "warn", "All edges removed during top_k pruning.")
            return ""

    # =========================================================================
    # 4. Safety cap
    # =========================================================================
    if max_edges and len(edges) > max_edges:
        before = len(edges)
        edges = edges.sort_values("weight", ascending=False).head(max_edges)
        _log(emit, "warn", "Truncated edges to max_edges",
             before=before, after=len(edges))

    # =========================================================================
    # 5. Compute scaled_weight, polarity, confidence
    # =========================================================================
    if not edges.empty:
        w = edges["weight"]
        w_min, w_max = float(w.min()), float(w.max())

        if w_max > w_min:
            edges["scaled_weight"] = (w - w_min) / (w_max - w_min)
        else:
            edges["scaled_weight"] = 1.0

        # Polarity is placeholder for future signed edges (e.g., antagonistic links)
        edges["polarity"] = 1

        # Confidence: log-normalised weight
        edges["confidence"] = np.log1p(edges["weight"]) / np.log1p(w_max)

    else:
        edges["scaled_weight"] = []
        edges["polarity"] = []
        edges["confidence"] = []

    # =========================================================================
    # 6. Backbone extraction
    # =========================================================================
    backbone_mask = _extract_backbone(edges, keep_fraction=backbone_fraction)
    edges["is_backbone"] = backbone_mask.astype(bool)

    # =========================================================================
    # 7. Attach element IDs
    # =========================================================================
    edges["source_id"] = edges["source"].map(element_id_map).fillna(-1).astype(int)
    edges["target_id"] = edges["target"].map(element_id_map).fillna(-1).astype(int)

    # Deterministic ordering
    edges = edges.sort_values(
        ["weight", "source", "target"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    # =========================================================================
    # 8. Write edges.csv
    # =========================================================================
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
