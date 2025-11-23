# =============================================================================
# hilbert_pipeline/edges_builder.py — Element co-occurrence edges (v2)
# =============================================================================
"""
Build an element-element edge list (edges.csv) from span_element_fusion.csv.

Basic idea:
  - Each span contains one or more elements.
  - Any pair of elements that co-occur in a span receives a co-occurrence count.
  - We then:
      * threshold on minimum co-occurrence (min_coocc)
      * keep the top_k strongest neighbors per node (optional)
      * optionally filter to elements present in hilbert_elements.csv
  - Output is edges.csv with columns: source, target, weight

This produces a sparse, semantically meaningful graph suitable for:
  - Molecule construction (run_molecule_stage)
  - Graph snapshots and exports
  - Stability and compound analysis
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Any, Tuple, List

import numpy as np
import pandas as pd

# Orchestrator-compatible default emitter
DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda *_a, **_k: None


# ------------------------------------------------------------------------- #
# Logging helpers
# ------------------------------------------------------------------------- #
def _log(
    emit: Callable[[str, Dict[str, Any]], None],
    level: str,
    msg: str,
    **fields: Any,
) -> None:
    payload = {"level": level, "msg": msg}
    payload.update(fields)
    try:
        emit("log", payload)
    except Exception:
        # Fallback to plain print if emit fails or is None-like
        print(f"[{level}] {msg} {fields}")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


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
    raise ValueError(
        f"None of the candidate columns {candidates} found in span_element_fusion.csv"
    )


def _load_valid_elements(results_dir: str) -> List[str]:
    """
    Load hilbert_elements.csv if available and return the list of valid element ids.

    This keeps edges aligned with the canonical element table.
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


# ------------------------------------------------------------------------- #
# Main builder
# ------------------------------------------------------------------------- #
def build_element_edges(
    results_dir: str,
    emit: Callable[[str, Dict[str, Any]], None] = DEFAULT_EMIT,
    min_coocc: int = 1,
    top_k: int = 30,
    max_edges: int = 250_000,
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
    max_edges:
        Safety cap on total number of edges written (after pruning). If the
        edge count exceeds this, we keep only the strongest edges globally.

    Returns
    -------
    out_path:
        Path to edges.csv if built, otherwise an empty string.
    """
    fusion_path = os.path.join(results_dir, "span_element_fusion.csv")
    out_path = os.path.join(results_dir, "edges.csv")

    if not os.path.exists(fusion_path):
        _log(
            emit,
            "warn",
            "span_element_fusion.csv not found",
            path=fusion_path,
        )
        return ""

    try:
        df = pd.read_csv(fusion_path)
    except Exception as exc:
        _log(
            emit,
            "warn",
            "Failed to read span_element_fusion.csv",
            error=str(exc),
            path=fusion_path,
        )
        return ""

    if df.empty:
        _log(emit, "warn", "span_element_fusion.csv is empty; no edges to build.")
        return ""

    # Try to infer span and element columns
    try:
        span_col = _resolve_column(df, ["span_id", "span", "span_index"])
    except ValueError as exc:
        _log(emit, "warn", str(exc) + " — cannot build edges.")
        return ""

    try:
        elem_col = _resolve_column(df, ["element", "element_id"])
    except ValueError as exc:
        _log(emit, "warn", str(exc) + " — cannot build edges.")
        return ""

    _log(
        emit,
        "info",
        "Using span and element columns",
        span_col=span_col,
        elem_col=elem_col,
        n_rows=int(len(df)),
    )

    # Optional: restrict to known elements from hilbert_elements.csv if present
    valid_elements = _load_valid_elements(results_dir)
    if valid_elements:
        valid_set = set(valid_elements)
        before = len(df)
        df = df[df[elem_col].astype(str).isin(valid_set)]
        _log(
            emit,
            "info",
            "Filtered fusion rows to elements present in hilbert_elements.csv",
            before=int(before),
            after=int(len(df)),
        )
        if df.empty:
            _log(
                emit,
                "warn",
                "No fusion rows left after filtering to valid elements; no edges.",
            )
            return ""

    # ------------------------------------------------------------------ #
    # 1. Accumulate co-occurrence counts for element pairs
    # ------------------------------------------------------------------ #
    pair_weights: Dict[Tuple[str, str], float] = {}

    for _span_id, g in df.groupby(span_col):
        elems = sorted(set(str(e) for e in g[elem_col].dropna()))
        if len(elems) <= 1:
            continue

        # simple uniform contribution per co-occurring pair
        n = len(elems)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = elems[i], elems[j]
                key = (a, b) if a <= b else (b, a)
                pair_weights[key] = pair_weights.get(key, 0.0) + 1.0

    if not pair_weights:
        _log(
            emit,
            "warn",
            "No co-occurring element pairs found; edges would be empty.",
        )
        return ""

    # ------------------------------------------------------------------ #
    # 2. Convert to DataFrame and apply basic filtering
    # ------------------------------------------------------------------ #
    rows = [
        {"source": src, "target": tgt, "weight": w}
        for (src, tgt), w in pair_weights.items()
    ]
    edges_df = pd.DataFrame(rows)

    _log(
        emit,
        "info",
        "Raw edge table constructed",
        n_raw_edges=int(len(edges_df)),
    )

    # Filter by minimum co-occurrence
    if min_coocc > 1:
        before = len(edges_df)
        edges_df = edges_df[edges_df["weight"] >= float(min_coocc)]
        _log(
            emit,
            "info",
            "Applied min_coocc filter",
            min_coocc=min_coocc,
            before=int(before),
            after=int(len(edges_df)),
        )

    if edges_df.empty:
        _log(
            emit,
            "warn",
            "All edges filtered out by min_coocc; no edges written.",
            min_coocc=min_coocc,
        )
        return ""

    # ------------------------------------------------------------------ #
    # 3. Per-node top-k pruning (optional)
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

        before = len(edges_df)
        edges_df = (
            pd.concat([by_source, by_target], ignore_index=True)
            .drop_duplicates(subset=["source", "target"])
            .reset_index(drop=True)
        )
        _log(
            emit,
            "info",
            "Applied per-node top_k pruning",
            top_k=top_k,
            before=int(before),
            after=int(len(edges_df)),
        )

    # ------------------------------------------------------------------ #
    # 4. Global safety cap on edges
    # ------------------------------------------------------------------ #
    if max_edges is not None and len(edges_df) > max_edges:
        before = len(edges_df)
        edges_df = (
            edges_df.sort_values("weight", ascending=False)
            .head(max_edges)
            .reset_index(drop=True)
        )
        _log(
            emit,
            "warn",
            "Edge count exceeded max_edges; truncated strongest edges only",
            max_edges=int(max_edges),
            before=int(before),
            after=int(len(edges_df)),
        )

    # Sort edges in a stable, human-readable way
    edges_df = edges_df.sort_values(
        ["weight", "source", "target"], ascending=[False, True, True]
    ).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 5. Write edges.csv
    # ------------------------------------------------------------------ #
    try:
        edges_df.to_csv(out_path, index=False)
    except Exception as exc:
        _log(
            emit,
            "warn",
            "Failed to write edges.csv",
            error=str(exc),
            path=out_path,
        )
        return ""

    _log(
        emit,
        "info",
        "Wrote edges.csv",
        path=out_path,
        n_edges=int(len(edges_df)),
        min_coocc=min_coocc,
        top_k=top_k,
    )

    # Register artifact for orchestrator / UI
    try:
        emit("artifact", {"path": out_path, "kind": "edges"})
    except Exception:
        pass

    return out_path
