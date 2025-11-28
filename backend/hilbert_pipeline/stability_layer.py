"""
stability_layer.py
==================

Hilbert Information Pipeline – Stability Layer (Pipeline v3.1)
--------------------------------------------------------------

This module computes **element-level stability signals** derived from each
element’s *entropy–coherence* profile. Stability is used throughout the Hilbert
Pipeline as an analogue of *local informational equilibrium*, following the
Hilbert Field Equations (Appendix X of the thesis):

.. math::

    S(e) = f(H_e, C_e)

where:
- :math:`H_e` is element entropy,
- :math:`C_e` is element coherence,
- and :math:`f` is a stability functional depending on the chosen mode.

Supported stability modes
-------------------------

1. **classic**

.. math::

    S = \\frac{C}{1 + H}

2. **entropy_weighted**

.. math::

    S = C \\cdot e^{-H}

3. **normalized**
    - Normalize :math:`H` and :math:`C` to :math:`[0,1]`
    - Compute:

.. math::

    S = C_{norm} (1 - H_{norm})

The goal is not physical accuracy, but to provide a **monotonic, robust,
interpretable stability signal** that aligns with Hilbert’s information
equilibrium model across corpora.

Outputs
-------

This layer writes up to **five artifacts** into the results directory:

1. ``signal_stability.csv``  
   Legacy stable schema used by older tools.

2. ``stability_meta.json``  
   Global statistics.

3. ``graph_stability_nodes.csv``  
   Graph-contract compliant node table (visualizer-ready).

4. ``graph_stability_meta.json``  
   Metadata for graph visualisation.

5. ``compound_stability.csv``  
   Produced by ``compute_compound_stability``.

Inputs
------

- ``hilbert_elements.csv``  
  Must contain at least:
    - ``element``
    - ``entropy`` or ``mean_entropy``
    - ``coherence`` or ``mean_coherence``

- ``molecules.csv`` (optional, for compound-level aggregation)

Orchestrator Integration
------------------------

Used internally by the orchestrator steps:

- ``_stage_stability``
- ``_stage_compounds``

and exposed publicly as:

- ``compute_signal_stability``  
- ``compute_compound_stability``

"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Callable

import numpy as np
import pandas as pd


# =============================================================================
# Logging
# =============================================================================

DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda *_: None


def _log(emit, level: str, msg: str, **fields):
    """
    Orchestrator-compatible logging wrapper.

    Parameters
    ----------
    emit : callable
        Logging callback from orchestrator.
    level : str
        Severity ("info", "warn", ...)
    msg : str
        Human-readable message.
    """
    payload = {"level": level, "msg": msg}
    payload.update(fields)
    try:
        emit("log", payload)
    except Exception:
        print(f"[{level}] {msg} {fields}")


# =============================================================================
# Numeric Helpers
# =============================================================================

def _safe(series: pd.Series, default=0.0) -> np.ndarray:
    """
    Convert Series to a numeric array with NaNs replaced by ``default``.

    Parameters
    ----------
    series : pandas.Series
    default : float

    Returns
    -------
    numpy.ndarray
    """
    arr = pd.to_numeric(series, errors="coerce").to_numpy()
    return np.where(np.isfinite(arr), arr, default)


def _stab_classic(entropy: np.ndarray, coherence: np.ndarray) -> np.ndarray:
    """Classic stability functional :math:`C / (1 + H)`."""
    entropy = np.maximum(entropy, 0.0)
    return coherence / (1.0 + entropy)


def _stab_entropy_weighted(entropy: np.ndarray, coherence: np.ndarray) -> np.ndarray:
    """Entropy-weighted stability :math:`C e^{-H}`."""
    entropy = np.maximum(entropy, 0.0)
    return coherence * np.exp(-entropy)


def _stab_normalized(entropy: np.ndarray, coherence: np.ndarray) -> np.ndarray:
    """
    Normalized stability functional:

    .. math::

        S = C_{norm} (1 - H_{norm})

    """
    c = coherence.copy()
    if c.max() > c.min():
        c = (c - c.min()) / (c.max() - c.min() + 1e-12)

    e = entropy.copy()
    if e.max() > e.min():
        e = (e - e.min()) / (e.max() - e.min() + 1e-12)

    return c * (1.0 - e)


def _compute_stability(ent: pd.Series, coh: pd.Series, mode="classic") -> np.ndarray:
    """
    Apply selected stability functional.

    Parameters
    ----------
    ent : pandas.Series
    coh : pandas.Series
    mode : {"classic", "entropy_weighted", "normalized"}

    Returns
    -------
    numpy.ndarray
    """
    entropy = _safe(ent, default=0.0)
    coherence = _safe(coh, default=0.0)

    if mode == "classic":
        return _stab_classic(entropy, coherence)
    if mode == "entropy_weighted":
        return _stab_entropy_weighted(entropy, coherence)
    if mode == "normalized":
        return _stab_normalized(entropy, coherence)

    return _stab_classic(entropy, coherence)


# =============================================================================
# Element-Level Stability
# =============================================================================

def compute_signal_stability(
    elements_csv: str,
    out_csv: str,
    mode: str = "classic",
    emit=DEFAULT_EMIT,
) -> None:
    """
    Compute element-level stability from entropy and coherence fields.

    Parameters
    ----------
    elements_csv : str
        Path to ``hilbert_elements.csv``.
    out_csv : str
        Output path for ``signal_stability.csv``.
    mode : str, optional
        Stability mode ("classic", "entropy_weighted", "normalized").
    emit : callable, optional
        Orchestrator logging callback.

    Produces
    --------
    signal_stability.csv :
        Legacy stable schema.

    graph_stability_nodes.csv :
        Graph-contract compatible table.

    stability_meta.json :
        Global statistical summary.

    graph_stability_meta.json :
        Graph visualizer metadata.
    """
    if not os.path.exists(elements_csv):
        _log(emit, "warn", "[stability] elements_csv missing", path=elements_csv)
        return

    try:
        df = pd.read_csv(elements_csv)
    except Exception as exc:
        _log(emit, "warn", "[stability] failed to read elements_csv", error=str(exc))
        return

    if df.empty:
        _log(emit, "warn", "[stability] hilbert_elements.csv empty")
        return

    _log(emit, "info", "[stability] loaded elements", n_rows=len(df), mode=mode)

    # ---- Resolve entropy and coherence fields --------------------------------
    entropy_col = next((c for c in ("entropy", "mean_entropy") if c in df.columns), None)
    coherence_col = next((c for c in ("coherence", "mean_coherence") if c in df.columns), None)

    if entropy_col is None or coherence_col is None:
        _log(
            emit,
            "warn",
            "[stability] missing entropy/coherence fields",
            columns=list(df.columns),
        )
        return

    # ---- Resolve identifiers ---------------------------------------------------
    if "element" not in df.columns:
        df["element"] = df.get("token", df.index.astype(str)).astype(str)

    if "element_id" not in df.columns:
        df["element_id"] = df.index.astype(int)

    if "doc" not in df.columns:
        df["doc"] = "corpus"

    # ---- Compute stability -----------------------------------------------------
    entropy = df[entropy_col]
    coherence = df[coherence_col]
    df["stability"] = _compute_stability(entropy, coherence, mode=mode)

    # ---- Legacy stability table ------------------------------------------------
    legacy = df[["doc", "element", entropy_col, coherence_col, "stability"]].copy()
    legacy = legacy.rename(
        columns={entropy_col: "entropy", coherence_col: "coherence"}
    )

    try:
        legacy.to_csv(out_csv, index=False)
        _log(emit, "info", "[stability] wrote signal_stability.csv", path=out_csv)
    except Exception as exc:
        _log(emit, "warn", "[stability] failed to write signal_stability.csv", error=str(exc))

    # ---- Graph node table (visualizer) ----------------------------------------
    graph_cols = [
        "element", "element_id",
        entropy_col, coherence_col, "stability",
        "tf", "df", "idf", "tfidf",
        "lsa0", "lsa1", "lsa2",
    ]

    for c in graph_cols:
        if c not in df.columns:
            df[c] = np.nan

    graph_nodes = df[graph_cols].rename(
        columns={entropy_col: "entropy", coherence_col: "coherence"}
    )

    vis_path = os.path.join(os.path.dirname(out_csv), "graph_stability_nodes.csv")
    try:
        graph_nodes.to_csv(vis_path, index=False)
        _log(
            emit,
            "info",
            "[stability] wrote graph_stability_nodes.csv",
            path=vis_path,
            n_rows=len(graph_nodes),
        )
        emit("artifact", {"path": vis_path, "kind": "graph_stability_nodes"})
    except Exception as exc:
        _log(emit, "warn", "[stability] failed to write graph_stability_nodes.csv", error=str(exc))

    # ---- Meta diagnostics ------------------------------------------------------
    meta = {
        "mode": mode,
        "num_elements": int(graph_nodes["element"].nunique()),
        "entropy_min": float(graph_nodes["entropy"].min()),
        "entropy_max": float(graph_nodes["entropy"].max()),
        "coherence_min": float(graph_nodes["coherence"].min()),
        "coherence_max": float(graph_nodes["coherence"].max()),
        "stability_min": float(graph_nodes["stability"].min()),
        "stability_max": float(graph_nodes["stability"].max()),
        "stability_mean": float(graph_nodes["stability"].mean()),
        "stability_median": float(graph_nodes["stability"].median()),
    }

    meta_path = os.path.join(os.path.dirname(out_csv), "stability_meta.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        _log(emit, "info", "[stability] wrote stability_meta.json", path=meta_path)
    except Exception as exc:
        _log(emit, "warn", "[stability] failed to write stability_meta.json", error=str(exc))

    vis_meta = {
        "schema_version": "1.0",
        "mode": mode,
        "fields": list(graph_nodes.columns),
        "count": int(len(graph_nodes)),
    }

    gmeta_path = os.path.join(os.path.dirname(out_csv), "graph_stability_meta.json")
    try:
        with open(gmeta_path, "w", encoding="utf-8") as f:
            json.dump(vis_meta, f, indent=2)
        _log(emit, "info", "[stability] wrote graph_stability_meta.json", path=gmeta_path)
    except Exception as exc:
        _log(emit, "warn", "[stability] failed to write graph_stability_meta.json", error=str(exc))

    _log(emit, "info", "[stability] stability computation complete.")


# =============================================================================
# Compound-Level Aggregation
# =============================================================================

def compute_compound_stability(
    *,
    compounds_json: str,
    elements_csv: str,
    stability_csv: str,
    out_csv: str,
    emit=DEFAULT_EMIT,
) -> str | None:
    """
    Aggregate element-level stability to the compound level.

    Parameters
    ----------
    compounds_json : str
        Path to ``informational_compounds.json`` or directory containing it.
    elements_csv : str
        Path to ``hilbert_elements.csv``.
    stability_csv : str
        Path to ``signal_stability.csv``.
    out_csv : str
        Output path for ``compound_stability.csv``.
    emit : callable

    Returns
    -------
    str or None
        Path to the output CSV, or None on failure.
    """
    base = Path(compounds_json).parent
    mol_path = base / "molecules.csv"

    if not mol_path.exists():
        _log(emit, "warn", "[compound-stability] molecules.csv not found", path=str(mol_path))
        return None

    if not Path(stability_csv).exists():
        _log(emit, "warn", "[compound-stability] stability_csv missing", path=stability_csv)
        return None

    try:
        mol_df = pd.read_csv(mol_path)
    except Exception as exc:
        _log(emit, "warn", "[compound-stability] failed to read molecules.csv", error=str(exc))
        return None

    try:
        stab_df = pd.read_csv(stability_csv)
    except Exception as exc:
        _log(emit, "warn", "[compound-stability] failed to read stability_csv", error=str(exc))
        return None

    if mol_df.empty or stab_df.empty:
        _log(emit, "warn", "[compound-stability] one or both tables empty")
        return None

    if "element" not in stab_df.columns:
        _log(emit, "warn", "[compound-stability] stability_csv missing element")
        return None

    needed = {"entropy", "coherence", "stability"}
    if not needed.issubset(stab_df.columns):
        _log(
            emit,
            "warn",
            "[compound-stability] stability_csv missing required fields",
            columns=list(stab_df.columns),
        )
        return None

    merged = mol_df.merge(
        stab_df[["element", "entropy", "coherence", "stability"]],
        on="element",
        how="left",
    )

    rows = []

    if "compound_id" not in merged.columns:
        _log(emit, "warn", "[compound-stability] molecules.csv missing compound_id")
        return None

    for cid, g in merged.groupby("compound_id"):
        coh = g["coherence"].dropna()
        stab = g["stability"].dropna()

        rows.append(
            {
                "compound_id": cid,
                "n_elements": len(g),
                "mean_element_coherence": float(coh.mean()) if len(coh) else float("nan"),
                "mean_element_stability": float(stab.mean()) if len(stab) else float("nan"),
                "stability_variance": float(stab.var(ddof=0)) if len(stab) else float("nan"),
            }
        )

    out_df = pd.DataFrame(rows).sort_values(
        "mean_element_stability", ascending=False
    )

    try:
        out_df.to_csv(out_csv, index=False)
        _log(emit, "info", "[compound-stability] wrote compound_stability.csv", path=out_csv)
        emit("artifact", {"path": out_csv, "kind": "compound_stability"})
    except Exception as exc:
        _log(emit, "warn", "[compound-stability] failed to write output", error=str(exc))
        return None

    return out_csv


__all__ = [
    "compute_signal_stability",
    "compute_compound_stability",
]
