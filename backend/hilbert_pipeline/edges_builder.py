# =============================================================================
# hilbert_pipeline/edges_builder.py — Element co-occurrence edges
# =============================================================================
"""
Build an element-element edge list (edges.csv) from span_element_fusion.csv.

Basic idea:
  - Each span contains one or more elements.
  - Any pair of elements that co-occur in a span receives a co-occurrence count.
  - We then:
      * threshold on minimum co-occurrence (min_coocc)
      * keep the top_k strongest neighbors per node (optional)
  - Output is edges.csv with columns: source, target, weight

This gives a sparse, semantically meaningful graph suitable for:
  - Molecule construction (run_molecule_layer)
  - Graph snapshots and exports
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Any, Tuple, List

import numpy as np
import pandas as pd

DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda *_a, **_k: None


# ------------------------------------------------------------------------- #
# Logging helper
# ------------------------------------------------------------------------- #
def _log(msg: str, emit: Callable[[str, Dict[str, Any]], None]) -> None:
    print(msg)
    try:
        emit("log", {"message": msg})
    except Exception:
        pass


# ------------------------------------------------------------------------- #
# Column resolution helpers
# ------------------------------------------------------------------------- #
def _resolve_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """
    Pick the first column name in `candidates` that exists in df.
    Raises ValueError if none found.
    """
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"None of the candidate columns {candidates} found in span_element_fusion.csv")


# ------------------------------------------------------------------------- #
# Main builder
# ------------------------------------------------------------------------- #
def build_element_edges(
    results_dir: str,
    emit: Callable[[str, Dict[str, Any]], None] = DEFAULT_EMIT,
    min_coocc: int = 1,
    top_k: int = 30,
) -> str:
    """
    Build edges.csv under `results_dir` from span_element_fusion.csv.

    Parameters
    ----------
    results_dir:
        Hilbert results directory for this run.
    emit:
        Logging callback compatible with the orchestrator.
    min_coocc:
        Minimum number of span co-occurrences required for an edge to be kept.
    top_k:
        Maximum number of neighbors per node (approximate; we take top_k
        per-source and per-target and union them). If <= 0, no top-k pruning.

    Returns
    -------
    out_path:
        Path to edges.csv if built, otherwise an empty string.
    """
    fusion_path = os.path.join(results_dir, "span_element_fusion.csv")
    out_path = os.path.join(results_dir, "edges.csv")

    if not os.path.exists(fusion_path):
        _log(f"[edges] span_element_fusion.csv not found at {fusion_path}", emit)
        return ""

    try:
        df = pd.read_csv(fusion_path)
    except Exception as exc:
        _log(f"[edges] Failed to read span_element_fusion.csv: {exc}", emit)
        return ""

    if df.empty:
        _log("[edges] span_element_fusion.csv is empty; no edges to build.", emit)
        return ""

    # Try to infer span and element columns
    try:
        span_col = _resolve_column(df, ["span_id", "span", "span_index"])
    except ValueError as exc:
        _log(f"[edges] {exc}; cannot build edges.", emit)
        return ""

    try:
        elem_col = _resolve_column(df, ["element", "element_id"])
    except ValueError as exc:
        _log(f"[edges] {exc}; cannot build edges.", emit)
        return ""

    _log(f"[edges] Using '{span_col}' as span column and '{elem_col}' as element column.", emit)

    # ------------------------------------------------------------------ #
    # 1. Accumulate co-occurrence counts for element pairs
    # ------------------------------------------------------------------ #
    pair_weights: Dict[Tuple[str, str], float] = {}

    # Group by span, then look at unique elements within each span
    for span_id, g in df.groupby(span_col):
        elems = sorted(set(str(e) for e in g[elem_col].dropna()))
        n = len(elems)
        if n <= 1:
            continue

        # For each unordered pair (i, j)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = elems[i], elems[j]
                key = (a, b) if a <= b else (b, a)
                pair_weights[key] = pair_weights.get(key, 0.0) + 1.0

    if not pair_weights:
        _log("[edges] No co-occurring pairs found; edges would be empty.", emit)
        return ""

    # ------------------------------------------------------------------ #
    # 2. Convert to DataFrame and apply basic filtering
    # ------------------------------------------------------------------ #
    rows = [
        {"source": src, "target": tgt, "weight": w}
        for (src, tgt), w in pair_weights.items()
    ]
    edges_df = pd.DataFrame(rows)

    # Filter by minimum co-occurrence
    if min_coocc > 1:
        edges_df = edges_df[edges_df["weight"] >= float(min_coocc)]

    if edges_df.empty:
        _log("[edges] All edges filtered out by min_coocc; no edges written.", emit)
        return ""

    # ------------------------------------------------------------------ #
    # 3. Apply per-node top-k pruning (optional)
    # ------------------------------------------------------------------ #
    if top_k is not None and top_k > 0:
        # Top-k per source
        by_source = (
            edges_df.sort_values("weight", ascending=False)
            .groupby("source", as_index=False)
            .head(top_k)
        )

        # Top-k per target
        by_target = (
            edges_df.sort_values("weight", ascending=False)
            .groupby("target", as_index=False)
            .head(top_k)
        )

        edges_df = (
            pd.concat([by_source, by_target], ignore_index=True)
            .drop_duplicates(subset=["source", "target"])
            .reset_index(drop=True)
        )

    # Sort edges in a stable, human-readable way
    edges_df = edges_df.sort_values(
        ["weight", "source", "target"], ascending=[False, True, True]
    ).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 4. Write edges.csv
    # ------------------------------------------------------------------ #
    try:
        edges_df.to_csv(out_path, index=False)
    except Exception as exc:
        _log(f"[edges] Failed to write edges.csv: {exc}", emit)
        return ""

    _log(
        f"[edges] Wrote {len(edges_df)} edges to {out_path} "
        f"(min_coocc={min_coocc}, top_k={top_k}).",
        emit,
    )
    return out_path
