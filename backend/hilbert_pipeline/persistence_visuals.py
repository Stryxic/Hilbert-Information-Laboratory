# =============================================================================
# hilbert_pipeline/persistence_visuals.py — Stability & Persistence Visuals (v2)
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
      element-level metrics:
        element, [doc], mean_entropy, mean_coherence, ...

Outputs:

  - persistence_field.png
        1D profile of stability over spans/elements/rows.

  - stability_scatter.png
        entropy vs coherence scatter for elements, optionally colored by stability.

  - stability_by_doc.png
        mean stability per document (if doc info exists in signal_stability.csv).

  - stability_doc_entropy_heatmap.png (optional)
        doc-level entropy vs coherence summary if doc columns exist in
        hilbert_elements.csv and/or signal_stability.csv.

Public API used by the orchestrator:

    from hilbert_pipeline.persistence_visuals import run_persistence_visuals

    run_persistence_visuals(results_dir, emit=ctx.emit)
"""

from __future__ import annotations

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


def _log(msg: str, emit: Callable = DEFAULT_EMIT) -> None:
    print(msg)
    try:
        emit("log", {"message": msg})
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
    Generate persistence / stability figures for the current run.

    Parameters
    ----------
    out_dir : str
        Output directory where signal_stability.csv and hilbert_elements.csv live.
    emit : callable
        Optional logger (kind, payload) compatible with PipelineContext.emit.
    """
    os.makedirs(out_dir, exist_ok=True)

    stab_path = os.path.join(out_dir, "signal_stability.csv")
    elem_path = os.path.join(out_dir, "hilbert_elements.csv")

    # -----------------------------------------------------------------------
    # 1. Persistence profile from signal_stability.csv
    # -----------------------------------------------------------------------
    stab_df: Optional[pd.DataFrame] = None
    if os.path.exists(stab_path):
        try:
            stab_df = pd.read_csv(stab_path)
        except Exception as e:
            _log(f"[persist][warn] Failed to read signal_stability.csv: {e}", emit)
    else:
        _log("[persist] signal_stability.csv not found; skipping persistence line.", emit)

    if stab_df is not None and not stab_df.empty and "stability" in stab_df.columns:
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
            # If we have element-based stability only
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
            _log(f"[persist] persistence_field.png written to {out_img}", emit)
        except Exception as e:
            _log(f"[persist][warn] Failed to plot persistence field: {e}", emit)
    else:
        if stab_df is not None:
            _log(
                "[persist] signal_stability.csv has no usable 'stability' column; "
                "skipping persistence_field.png.",
                emit,
            )

    # -----------------------------------------------------------------------
    # 2. Element entropy vs coherence scatter (colored by stability if possible)
    # -----------------------------------------------------------------------
    if not os.path.exists(elem_path):
        _log("[persist] hilbert_elements.csv not found; skipping scatter.", emit)
        return

    try:
        edf = pd.read_csv(elem_path)
    except Exception as e:
        _log(f"[persist][warn] Failed to read hilbert_elements.csv: {e}", emit)
        return

    # Normalize column names for entropy / coherence
    if "entropy" not in edf.columns and "mean_entropy" in edf.columns:
        edf = edf.rename(columns={"mean_entropy": "entropy"})
    if "coherence" not in edf.columns and "mean_coherence" in edf.columns:
        edf = edf.rename(columns={"mean_coherence": "coherence"})

    if not {"entropy", "coherence"}.issubset(edf.columns):
        _log("[persist] Missing entropy/coherence columns; skipping scatter.", emit)
        return

    x = _safe_numeric(edf["entropy"], default=0.0)
    y = _safe_numeric(edf["coherence"], default=0.0)

    # Try to merge stability info if available
    stability_series: Optional[np.ndarray] = None
    if (
        stab_df is not None
        and not stab_df.empty
        and "element" in stab_df.columns
        and "element" in edf.columns
    ):
        try:
            merged = edf[["element"]].merge(
                stab_df[["element", "stability"]].dropna(),
                on="element",
                how="left",
            )
            stability_series = _safe_numeric(merged["stability"], default=np.nan)
            # If everything is NaN, treat as absent
            if not np.isfinite(stability_series).any():
                stability_series = None
        except Exception:
            stability_series = None

    try:
        plt.figure(figsize=(4.8, 4.8))
        if stability_series is not None:
            # Mask NaNs
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
        _log(f"[persist] stability_scatter.png written to {out_img}", emit)
    except Exception as e:
        _log(f"[persist][warn] Scatter plot failed: {e}", emit)

    # -----------------------------------------------------------------------
    # 3. Mean stability per document (if doc + stability info exist)
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
                plt.bar(
                    np.arange(len(doc_stats)),
                    doc_stats.values,
                    align="center",
                )
                plt.xticks(
                    np.arange(len(doc_stats)),
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
                _log(f"[persist] stability_by_doc.png written to {out_img}", emit)
        except Exception as e:
            _log(f"[persist][warn] Mean stability-by-doc plot failed: {e}", emit)

    # -----------------------------------------------------------------------
    # 4. Optional doc-level entropy/coherence summary heatmap
    # -----------------------------------------------------------------------
    try:
        if "doc" in edf.columns:
            doc_agg = (
                edf.groupby("doc")[["entropy", "coherence"]]
                .mean()
                .dropna()
            )
            if not doc_agg.empty:
                # Normalize to [0,1] for a simple pseudo-heatmap
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
                plt.title("Doc-level Entropy / Coherence Summary")
                plt.legend(loc="best", fontsize=8)
                plt.tight_layout()
                out_img = os.path.join(out_dir, "stability_doc_entropy_heatmap.png")
                plt.savefig(out_img, dpi=200)
                plt.close()
                _log(
                    f"[persist] stability_doc_entropy_heatmap.png written to {out_img}",
                    emit,
                )
    except Exception as e:
        _log(f"[persist][warn] Doc-level summary plot failed: {e}", emit)


# ---------------------------------------------------------------------------
# Orchestrator-facing wrapper
# ---------------------------------------------------------------------------

def run_persistence_visuals(out_dir: str, emit: Optional[Callable] = None) -> None:
    """
    Orchestrator wrapper that emits start/end events and delegates to
    plot_persistence_field.

    Used in the Pipeline stage:

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
