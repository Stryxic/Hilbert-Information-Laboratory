# =============================================================================
# hilbert_pipeline/persistence_visuals.py — Stability & Persistence Visuals (v3.1)
# =============================================================================
"""
Generates persistence and stability visualizations from pipeline outputs.

Inputs in `out_dir`:

  - signal_stability.csv
      Old schema (span-level):
        index, entropy, coherence, stability, ...
      New schema (element-level or doc-level):
        doc, element, entropy, coherence, stability, ...

  - hilbert_elements.csv
      Element-level metrics:
        element, [doc], mean_entropy, mean_coherence, ...

  - graph_stability_nodes.csv (optional, from stability_layer)
      Graph-contract ready node metrics:
        element, element_id, entropy, coherence, stability, tf, df, idf, tfidf, lsa0, lsa1, lsa2

Outputs (PNG):

  - persistence_field.png
        1D profile of stability over spans or elements.

  - stability_scatter.png
        entropy vs coherence scatter for elements, optionally colored by stability.

  - stability_by_doc.png
        Mean stability per document (if doc info exists in signal_stability.csv).

  - stability_doc_entropy_heatmap.png (optional)
        Doc-level entropy vs coherence summary if doc columns exist in
        hilbert_elements.csv and or signal_stability.csv.

Graph visualizer helper:

  - persistence_visuals_index.json
        Index of produced plots for the frontend or graph visualizer.

Public API used by the orchestrator:

    from hilbert_pipeline.persistence_visuals import run_persistence_visuals

    run_persistence_visuals(results_dir, emit=ctx.emit)
"""

from __future__ import annotations

import json
import os
from typing import Callable, Optional, Dict, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------

DEFAULT_EMIT: Callable = lambda *_: None  # noqa: E731


def _log(
    msg: str,
    emit: Callable = DEFAULT_EMIT,
    level: str = "info",
    **fields: Any,
) -> None:
    payload = {"level": level, "msg": msg}
    payload.update(fields)
    print(f"[{level}] {msg} {fields}")
    try:
        emit("log", payload)
    except Exception:
        # emit is best-effort only
        pass


def _safe_numeric(series, default: float = 0.0) -> np.ndarray:
    if series is None:
        return np.array([default], dtype=float)
    arr = pd.to_numeric(series, errors="coerce").to_numpy()
    arr = np.where(np.isfinite(arr), arr, default)
    return arr


# ---------------------------------------------------------------------------
# Core plotting function
# ---------------------------------------------------------------------------

def plot_persistence_field(out_dir: str, emit: Callable = DEFAULT_EMIT) -> None:
    """
    Generate persistence and stability figures for the current run.

    Parameters
    ----------
    out_dir : str
        Output directory where signal_stability.csv, hilbert_elements.csv and
        optional graph_stability_nodes.csv live.
    emit : callable
        Logger (kind, payload) compatible with PipelineContext.emit.
    """
    os.makedirs(out_dir, exist_ok=True)

    stab_path = os.path.join(out_dir, "signal_stability.csv")
    elem_path = os.path.join(out_dir, "hilbert_elements.csv")
    graph_nodes_path = os.path.join(out_dir, "graph_stability_nodes.csv")

    plots_index: Dict[str, Dict[str, Any]] = {}
    stab_df: Optional[pd.DataFrame] = None

    # -----------------------------------------------------------------------
    # 1. Load signal_stability.csv if available
    # -----------------------------------------------------------------------
    if os.path.exists(stab_path):
        try:
            stab_df = pd.read_csv(stab_path)
            if stab_df.empty:
                _log("[persist] signal_stability.csv is empty; skipping line and doc plots.", emit, "warn")
                stab_df = None
        except Exception as e:
            _log("[persist] Failed to read signal_stability.csv", emit, "warn", error=str(e))
            stab_df = None
    else:
        _log("[persist] signal_stability.csv not found; skipping persistence line and doc-level stability.", emit, "warn")

    # -----------------------------------------------------------------------
    # 2. Persistence profile from signal_stability.csv
    # -----------------------------------------------------------------------
    if stab_df is not None and "stability" in stab_df.columns:
        df = stab_df.copy()

        # Determine index-like column
        index_col = None
        for cand in ("span_index", "index", "span_id", "row_id"):
            if cand in df.columns:
                index_col = cand
                break

        if index_col is not None:
            x = _safe_numeric(df[index_col], default=0.0)
            x_label = index_col
        else:
            if "element" in df.columns:
                x = np.arange(len(df))
                x_label = "Element row index"
            else:
                x = np.arange(len(df))
                x_label = "Row index"

        y = _safe_numeric(df["stability"], default=0.0)

        try:
            plt.figure(figsize=(8.5, 3.5))
            plt.plot(x, y, linewidth=1.2)
            plt.xlabel(x_label)
            plt.ylabel("Stability")
            plt.title("Signal Stability Profile")
            plt.tight_layout()
            out_img = os.path.join(out_dir, "persistence_field.png")
            plt.savefig(out_img, dpi=200)
            plt.close()
            _log("[persist] persistence_field.png written", emit, path=out_img)

            plots_index["persistence_field"] = {
                "file": "persistence_field.png",
                "kind": "line",
                "x": x_label,
                "y": "stability",
            }
            try:
                emit("artifact", {"path": out_img, "kind": "persistence_field"})
            except Exception:
                pass
        except Exception as e:
            _log("[persist] Failed to plot persistence field", emit, "warn", error=str(e))
    else:
        if stab_df is not None:
            _log(
                "[persist] signal_stability.csv has no usable 'stability' column; skipping persistence_field.png.",
                emit,
                "warn",
            )

    # -----------------------------------------------------------------------
    # 3. Element entropy vs coherence scatter (colored by stability if possible)
    # -----------------------------------------------------------------------
    if not os.path.exists(elem_path):
        _log("[persist] hilbert_elements.csv not found; skipping scatter.", emit, "warn")
        _write_index(out_dir, plots_index, emit)
        return

    try:
        edf = pd.read_csv(elem_path)
    except Exception as e:
        _log("[persist] Failed to read hilbert_elements.csv", emit, "warn", error=str(e))
        _write_index(out_dir, plots_index, emit)
        return

    # Normalize column names for entropy and coherence
    if "entropy" not in edf.columns and "mean_entropy" in edf.columns:
        edf = edf.rename(columns={"mean_entropy": "entropy"})
    if "coherence" not in edf.columns and "mean_coherence" in edf.columns:
        edf = edf.rename(columns={"coherence": "coherence"}) if "coherence" in edf.columns else edf.rename(columns={"mean_coherence": "coherence"})

    if not {"entropy", "coherence"}.issubset(edf.columns):
        _log("[persist] Missing entropy and or coherence columns; skipping scatter.", emit, "warn")
    else:
        x = _safe_numeric(edf["entropy"], default=0.0)
        y = _safe_numeric(edf["coherence"], default=0.0)

        stability_series: Optional[np.ndarray] = None

        # Prefer graph_stability_nodes.csv if present
        if os.path.exists(graph_nodes_path):
            try:
                gdf = pd.read_csv(graph_nodes_path)
                if (
                    not gdf.empty
                    and "element" in gdf.columns
                    and "stability" in gdf.columns
                    and "element" in edf.columns
                ):
                    merged = edf[["element"]].merge(
                        gdf[["element", "stability"]].dropna(),
                        on="element",
                        how="left",
                    )
                    stability_series = _safe_numeric(merged["stability"], default=np.nan)
                    if not np.isfinite(stability_series).any():
                        stability_series = None
            except Exception as e:
                _log("[persist] Failed to join graph_stability_nodes for coloring", emit, "warn", error=str(e))
                stability_series = None

        # If no graph node table, fall back to signal_stability.csv
        if stability_series is None and stab_df is not None:
            if (
                not stab_df.empty
                and "element" in stab_df.columns
                and "element" in edf.columns
                and "stability" in stab_df.columns
            ):
                try:
                    merged = edf[["element"]].merge(
                        stab_df[["element", "stability"]].dropna(),
                        on="element",
                        how="left",
                    )
                    stability_series = _safe_numeric(merged["stability"], default=np.nan)
                    if not np.isfinite(stability_series).any():
                        stability_series = None
                except Exception:
                    stability_series = None

        try:
            plt.figure(figsize=(4.8, 4.8))
            color_field = None

            if stability_series is not None:
                mask = np.isfinite(stability_series)
                if mask.any():
                    sc = plt.scatter(
                        x[mask],
                        y[mask],
                        c=stability_series[mask],
                        s=18,
                        alpha=0.8,
                        cmap="viridis",
                    )
                    plt.colorbar(sc, label="Stability")
                    color_field = "stability"
                else:
                    plt.scatter(x, y, s=18, alpha=0.7)
            else:
                plt.scatter(x, y, s=18, alpha=0.7)

            plt.xlabel("Element entropy")
            plt.ylabel("Element coherence")
            plt.title("Element Stability Field")
            plt.tight_layout()
            out_img = os.path.join(out_dir, "stability_scatter.png")
            plt.savefig(out_img, dpi=200)
            plt.close()
            _log("[persist] stability_scatter.png written", emit, path=out_img)

            plots_index["stability_scatter"] = {
                "file": "stability_scatter.png",
                "kind": "scatter",
                "x": "entropy",
                "y": "coherence",
                "color": color_field,
            }
            try:
                emit("artifact", {"path": out_img, "kind": "stability_scatter"})
            except Exception:
                pass
        except Exception as e:
            _log("[persist] Scatter plot failed", emit, "warn", error=str(e))

    # -----------------------------------------------------------------------
    # 4. Mean stability per document (if doc + stability info exist)
    # -----------------------------------------------------------------------
    if (
        stab_df is not None
        and not stab_df.empty
        and "doc" in stab_df.columns
        and "stability" in stab_df.columns
    ):
        sdf = stab_df.copy()
        sdf["stability"] = _safe_numeric(sdf["stability"], default=0.0)

        try:
            doc_stats = (
                sdf.groupby("doc")["stability"]
                .mean()
                .sort_values(ascending=False)
            )
            if not doc_stats.empty:
                fig_width = max(4.0, 0.35 * len(doc_stats) + 1.5)
                plt.figure(figsize=(fig_width, 3.5))
                idx = np.arange(len(doc_stats))
                plt.bar(idx, doc_stats.values, align="center")
                plt.xticks(
                    idx,
                    [str(d)[:18] for d in doc_stats.index],
                    rotation=45,
                    ha="right",
                )
                plt.ylabel("Mean stability")
                plt.title("Mean Stability by Document")
                plt.tight_layout()
                out_img = os.path.join(out_dir, "stability_by_doc.png")
                plt.savefig(out_img, dpi=200)
                plt.close()
                _log("[persist] stability_by_doc.png written", emit, path=out_img)

                plots_index["stability_by_doc"] = {
                    "file": "stability_by_doc.png",
                    "kind": "bar",
                    "x": "doc",
                    "y": "mean_stability",
                }
                try:
                    emit("artifact", {"path": out_img, "kind": "stability_by_doc"})
                except Exception:
                    pass
        except Exception as e:
            _log("[persist] Mean stability-by-doc plot failed", emit, "warn", error=str(e))

    # -----------------------------------------------------------------------
    # 5. Optional doc-level entropy and coherence summary
    # -----------------------------------------------------------------------
    try:
        if "doc" in edf.columns and {"entropy", "coherence"}.issubset(edf.columns):
            doc_agg = (
                edf.groupby("doc")[["entropy", "coherence"]]
                .mean()
                .dropna()
            )
            if not doc_agg.empty:
                e_norm = (doc_agg["entropy"] - doc_agg["entropy"].min()) / (
                    doc_agg["entropy"].max() - doc_agg["entropy"].min() + 1e-9
                )
                c_norm = (doc_agg["coherence"] - doc_agg["coherence"].min()) / (
                    doc_agg["coherence"].max() - doc_agg["coherence"].min() + 1e-9
                )

                plt.figure(figsize=(4.8, 4.0))
                idx = np.arange(len(doc_agg))
                plt.scatter(
                    idx,
                    e_norm,
                    s=30,
                    alpha=0.8,
                    label="Entropy (norm)",
                    marker="o",
                )
                plt.scatter(
                    idx,
                    c_norm,
                    s=30,
                    alpha=0.8,
                    label="Coherence (norm)",
                    marker="s",
                )
                plt.xticks(
                    idx,
                    [str(d)[:14] for d in doc_agg.index],
                    rotation=45,
                    ha="right",
                )
                plt.ylabel("Normalised value")
                plt.title("Doc-level Entropy and Coherence Summary")
                plt.legend(loc="best", fontsize=8)
                plt.tight_layout()
                out_img = os.path.join(out_dir, "stability_doc_entropy_heatmap.png")
                plt.savefig(out_img, dpi=200)
                plt.close()
                _log("[persist] stability_doc_entropy_heatmap.png written", emit, path=out_img)

                plots_index["stability_doc_entropy_heatmap"] = {
                    "file": "stability_doc_entropy_heatmap.png",
                    "kind": "scatter_multi",
                    "x": "doc",
                    "series": ["entropy_norm", "coherence_norm"],
                }
                try:
                    emit("artifact", {"path": out_img, "kind": "stability_doc_entropy_heatmap"})
                except Exception:
                    pass
    except Exception as e:
        _log("[persist] Doc-level summary plot failed", emit, "warn", error=str(e))

    # -----------------------------------------------------------------------
    # 6. Write persistence_visuals_index.json for the graph visualizer
    # -----------------------------------------------------------------------
    _write_index(out_dir, plots_index, emit)


def _write_index(out_dir: str, plots_index: Dict[str, Dict[str, Any]], emit: Callable) -> None:
    """Write persistence_visuals_index.json describing produced plots."""
    index_path = os.path.join(out_dir, "persistence_visuals_index.json")
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(plots_index, f, indent=2)
        _log("[persist] persistence_visuals_index.json written", emit, path=index_path)
        try:
            emit("artifact", {"path": index_path, "kind": "persistence_visuals_index"})
        except Exception:
            pass
    except Exception as e:
        _log("[persist] Failed to write persistence_visuals_index.json", emit, "warn", error=str(e))


# ---------------------------------------------------------------------------
# Orchestrator-facing wrapper
# ---------------------------------------------------------------------------

def run_persistence_visuals(out_dir: str, emit: Optional[Callable] = None) -> None:
    """
    Orchestrator wrapper that emits start and end events and delegates to
    plot_persistence_field.

    Used in Pipeline stage:

        run_persistence_visuals(ctx.results_dir, emit=ctx.emit)
    """
    emit = emit or DEFAULT_EMIT

    try:
        emit("pipeline", {"stage": "persistence_visuals", "event": "start"})
    except Exception:
        pass

    plot_persistence_field(out_dir, emit=emit)

    try:
        emit("pipeline", {"stage": "persistence_visuals", "event": "end"})
    except Exception:
        pass


__all__ = [
    "plot_persistence_field",
    "run_persistence_visuals",
]
