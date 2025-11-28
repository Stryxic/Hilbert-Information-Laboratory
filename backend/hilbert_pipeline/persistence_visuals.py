"""
persistence_visuals.py
======================

Hilbert Information Pipeline – Persistence and Stability Visuals (v3.1)
-----------------------------------------------------------------------

This module generates all *visual diagnostics* for the Hilbert Information
Pipeline, including:

1. **Signal stability profile** – plots stability across spans or elements.
2. **Element entropy–coherence field** – core epistemic-geometry diagram.
3. **Mean document stability** – shows how information quality varies across documents.
4. **Document-level entropy/coherence summary** – complementary profile showing
   document-scale informational structure.

Each visual corresponds directly to structures defined in the thesis, allowing
the diagrams produced here to be embedded in Chapter 6 (Hilbert Information
Detection Tool) and Chapter 7 (Hilbert Epistemic Geometry).

Inputs
------

The following files are expected in the run directory:

``signal_stability.csv``  
    Stability signals for spans or elements. Must contain:
    - ``stability``
    - ``doc`` (optional)
    - ``element`` (optional)
    - index or span identifiers (various legacy names)

``hilbert_elements.csv``  
    Element-level metrics (entropy, coherence, tf/df/idf/etc).

``graph_stability_nodes.csv`` *(optional)*  
    Provides stable alignment between LSA spectral field and stability.

Outputs
-------

``persistence_field.png``  
    1-D stability signal over spans or elements.

``stability_scatter.png``  
    Entropy–coherence diagram (optionally coloured by stability).

``stability_by_doc.png``  
    Mean document stability bar chart.

``stability_doc_entropy_heatmap.png``  
    Normalised document-level entropy/coherence overlay.

``persistence_visuals_index.json``  
    Machine-readable index consumed by frontend and graph-visualiser.

Public API
----------

.. autofunction:: plot_persistence_field  
.. autofunction:: run_persistence_visuals

"""

from __future__ import annotations

import os
import json
from typing import Callable, Optional, Dict, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# =============================================================================
# Logging
# =============================================================================

DEFAULT_EMIT: Callable = lambda *_: None


def _log(msg: str, emit: Callable = DEFAULT_EMIT, level: str = "info", **fields):
    """
    Orchestrator-compatible logger.

    Parameters
    ----------
    msg : str
        Message text.
    emit : callable
        Logging callback with signature ``emit(kind, payload)``.
    level : str
        Severity level: "info", "warn", etc.
    """
    payload = {"level": level, "msg": msg}
    payload.update(fields)
    print(f"[{level}] {msg} {fields}")
    try:
        emit("log", payload)
    except Exception:
        pass


# =============================================================================
# Helpers
# =============================================================================

def _safe_numeric(series, default: float = 0.0) -> np.ndarray:
    """
    Convert Series to numeric array with NaN replacement.

    Returns
    -------
    numpy.ndarray
    """
    if series is None:
        return np.array([default], dtype=float)
    arr = pd.to_numeric(series, errors="coerce").to_numpy()
    return np.where(np.isfinite(arr), arr, default)


# =============================================================================
# Core Visualisation Logic
# =============================================================================

def plot_persistence_field(out_dir: str, emit: Callable = DEFAULT_EMIT) -> None:
    """
    Generate all persistence and stability figures for a single Hilbert run.

    Parameters
    ----------
    out_dir : str
        Directory containing signal_stability.csv, hilbert_elements.csv,
        and optional graph_stability_nodes.csv.
    emit : callable
        Orchestrator-compatible logger.

    Notes
    -----
    This function produces all persistence/signal/stability diagrams described
    in the Hilbert Epistemic Geometry chapter:

    - Stability as an energy–like functional.
    - Entropy–coherence phase field.
    - Document-level information flow.

    The output JSON index enables the frontend and visualiser to dynamically
    load plots without guessing file names.
    """
    os.makedirs(out_dir, exist_ok=True)

    stab_path = os.path.join(out_dir, "signal_stability.csv")
    elem_path = os.path.join(out_dir, "hilbert_elements.csv")
    graph_nodes_path = os.path.join(out_dir, "graph_stability_nodes.csv")

    plots_index: Dict[str, Dict[str, Any]] = {}
    stab_df: Optional[pd.DataFrame] = None

    # -------------------------------------------------------------------------
    # 1. Load signal_stability.csv
    # -------------------------------------------------------------------------
    if os.path.exists(stab_path):
        try:
            stab_df = pd.read_csv(stab_path)
            if stab_df.empty:
                _log("[persist] signal_stability.csv is empty; skipping.", emit, "warn")
                stab_df = None
        except Exception as exc:
            _log("[persist] Failed to read signal_stability.csv", emit, "warn", error=str(exc))
            stab_df = None
    else:
        _log("[persist] signal_stability.csv missing; skipping line and doc plots.", emit, "warn")

    # -------------------------------------------------------------------------
    # 2. Persistence field: stability vs index
    # -------------------------------------------------------------------------
    if stab_df is not None and "stability" in stab_df.columns:
        df = stab_df.copy()

        # Determine best available x-axis
        index_col = next(
            (c for c in ("span_index", "index", "span_id", "row_id") if c in df.columns),
            None,
        )

        if index_col:
            x = _safe_numeric(df[index_col])
            x_label = index_col
        else:
            x = np.arange(len(df))
            x_label = "Row index"

        y = _safe_numeric(df["stability"])

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

        except Exception as exc:
            _log("[persist] Failed to render persistence field", emit, "warn", error=str(exc))

    # -------------------------------------------------------------------------
    # 3. Entropy–coherence scatter (Hilbert phase field)
    # -------------------------------------------------------------------------
    if not os.path.exists(elem_path):
        _log("[persist] hilbert_elements.csv missing; skipping scatter.", emit, "warn")
        _write_index(out_dir, plots_index, emit)
        return

    try:
        edf = pd.read_csv(elem_path)
    except Exception as exc:
        _log("[persist] Failed to read hilbert_elements.csv", emit, "warn", error=str(exc))
        _write_index(out_dir, plots_index, emit)
        return

    # Resolve entropy/coherence fields
    if "entropy" not in edf.columns and "mean_entropy" in edf.columns:
        edf = edf.rename(columns={"mean_entropy": "entropy"})
    if "coherence" not in edf.columns and "mean_coherence" in edf.columns:
        edf = edf.rename(columns={"mean_coherence": "coherence"})

    if not {"entropy", "coherence"}.issubset(edf.columns):
        _log("[persist] Missing entropy/coherence; skipping scatter.", emit, "warn")
    else:
        x = _safe_numeric(edf["entropy"])
        y = _safe_numeric(edf["coherence"])

        stability_series = None

        # Prefer graph_stability_nodes.csv
        if os.path.exists(graph_nodes_path):
            try:
                gdf = pd.read_csv(graph_nodes_path)
                if (
                    "element" in gdf.columns
                    and "stability" in gdf.columns
                    and "element" in edf.columns
                ):
                    merged = edf[["element"]].merge(
                        gdf[["element", "stability"]], on="element", how="left"
                    )
                    stability_series = _safe_numeric(merged["stability"], default=np.nan)
            except Exception:
                stability_series = None

        # Fallback to signal_stability.csv
        if stability_series is None and stab_df is not None:
            if (
                "element" in stab_df.columns
                and "stability" in stab_df.columns
                and "element" in edf.columns
            ):
                merged = edf[["element"]].merge(
                    stab_df[["element", "stability"]], on="element", how="left"
                )
                stability_series = _safe_numeric(merged["stability"], default=np.nan)

        try:
            plt.figure(figsize=(5.0, 5.0))
            color_field = None

            if stability_series is not None and np.isfinite(stability_series).any():
                mask = np.isfinite(stability_series)
                sc = plt.scatter(
                    x[mask],
                    y[mask],
                    c=stability_series[mask],
                    cmap="viridis",
                    s=22,
                    alpha=0.85,
                )
                plt.colorbar(sc, label="Stability")
                color_field = "stability"
            else:
                plt.scatter(x, y, s=22, alpha=0.75)

            plt.xlabel("Entropy")
            plt.ylabel("Coherence")
            plt.title("Entropy–Coherence Field")
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

        except Exception as exc:
            _log("[persist] Scatter plot failed", emit, "warn", error=str(exc))

    # -------------------------------------------------------------------------
    # 4. Mean stability per document
    # -------------------------------------------------------------------------
    if (
        stab_df is not None
        and "doc" in stab_df.columns
        and "stability" in stab_df.columns
    ):
        try:
            sdf = stab_df.copy()
            sdf["stability"] = _safe_numeric(sdf["stability"])
            doc_stats = (
                sdf.groupby("doc")["stability"]
                .mean()
                .sort_values(ascending=False)
            )

            if not doc_stats.empty:
                width = max(4.0, 0.32 * len(doc_stats) + 1.0)
                plt.figure(figsize=(width, 3.6))
                idx = np.arange(len(doc_stats))
                plt.bar(idx, doc_stats.values)
                plt.xticks(idx, [str(d)[:18] for d in doc_stats.index],
                           rotation=45, ha="right")
                plt.ylabel("Mean Stability")
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
                emit("artifact", {"path": out_img, "kind": "stability_by_doc"})
        except Exception as exc:
            _log("[persist] Document stability plot failed", emit, "warn", error=str(exc))

    # -------------------------------------------------------------------------
    # 5. Doc-level entropy and coherence summary
    # -------------------------------------------------------------------------
    try:
        if "doc" in edf.columns and {"entropy", "coherence"}.issubset(edf.columns):
            doc_agg = edf.groupby("doc")[["entropy", "coherence"]].mean().dropna()

            if not doc_agg.empty:
                ent = doc_agg["entropy"]
                coh = doc_agg["coherence"]

                e_norm = (ent - ent.min()) / (ent.max() - ent.min() + 1e-9)
                c_norm = (coh - coh.min()) / (coh.max() - coh.min() + 1e-9)

                plt.figure(figsize=(5.0, 4.2))
                idx = np.arange(len(doc_agg))
                plt.scatter(idx, e_norm, marker="o", s=28, alpha=0.8, label="Entropy (norm)")
                plt.scatter(idx, c_norm, marker="s", s=28, alpha=0.8, label="Coherence (norm)")
                plt.xticks(idx, [str(d)[:14] for d in doc_agg.index],
                           rotation=45, ha="right")
                plt.ylabel("Normalised Value")
                plt.title("Doc-Level Entropy & Coherence Summary")
                plt.legend(fontsize=8)
                plt.tight_layout()

                out_img = os.path.join(out_dir, "stability_doc_entropy_heatmap.png")
                plt.savefig(out_img, dpi=200)
                plt.close()

                _log("[persist] stability_doc_entropy_heatmap.png written", emit, path=out_img)
                plots_index["stability_doc_entropy_heatmap"] = {
                    "file": "stability_doc_entropy_heatmap.png",
                    "kind": "scatter_multi",
                    "series": ["entropy_norm", "coherence_norm"],
                }
                emit("artifact", {"path": out_img, "kind": "stability_doc_entropy_heatmap"})
    except Exception as exc:
        _log("[persist] Doc-level summary plot failed", emit, "warn", error=str(exc))

    # -------------------------------------------------------------------------
    # 6. Write visual index
    # -------------------------------------------------------------------------
    _write_index(out_dir, plots_index, emit)


# =============================================================================
# Index Writer
# =============================================================================

def _write_index(out_dir: str, plots_index: Dict[str, Dict[str, Any]], emit: Callable) -> None:
    """
    Write ``persistence_visuals_index.json`` describing all generated plots.

    Used by the frontend to decide what plots exist for a given run.
    """
    index_path = os.path.join(out_dir, "persistence_visuals_index.json")
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(plots_index, f, indent=2)
        _log("[persist] persistence_visuals_index.json written", emit, path=index_path)
        emit("artifact", {"path": index_path, "kind": "persistence_visuals_index"})
    except Exception as exc:
        _log("[persist] Failed to write persistence_visuals_index.json", emit, "warn", error=str(exc))


# =============================================================================
# Orchestrator Wrapper
# =============================================================================

def run_persistence_visuals(out_dir: str, emit: Optional[Callable] = None) -> None:
    """
    Orchestrator-facing wrapper for persistence visual generation.

    Emits:

    - ``pipeline: start`` event
    - calls :func:`plot_persistence_field`
    - ``pipeline: end`` event

    Parameters
    ----------
    out_dir : str
        Results directory.
    emit : callable, optional
        Orchestrator logging callback.
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
