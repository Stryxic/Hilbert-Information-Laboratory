# =============================================================================
# hilbert_pipeline/stability_layer.py — Advanced Stability Layer (v3.1)
# =============================================================================
"""
Modern signal-stability computation compatible with the Hilbert Pipeline 3.1.

Features:
  - Robust handling of entropy/coherence fields (classic, normalized,
    entropy-weighted modes).
  - Produces graph-contract compatible outputs for visualization:
        graph_stability_nodes.csv
        graph_stability_meta.json
  - Continues to support legacy outputs:
        signal_stability.csv
        stability_meta.json
        compound_stability.csv
  - Zero-division and NaN protection.

Inputs:
  hilbert_elements.csv        (canonical upstream element table)
  molecules.csv               (optional compound structures)
Outputs:
  signal_stability.csv
  stability_meta.json
  graph_stability_nodes.csv
  graph_stability_meta.json
  compound_stability.csv      (from compound-level aggregator)
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Callable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda *_: None

def _log(emit, level, msg, **fields):
    payload = {"level": level, "msg": msg}
    payload.update(fields)
    try:
        emit("log", payload)
    except Exception:
        print(f"[{level}] {msg} {fields}")


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #

def _safe(series, default=0.0):
    arr = pd.to_numeric(series, errors="coerce").to_numpy()
    return np.where(np.isfinite(arr), arr, default)


def _stab_classic(entropy, coherence):
    entropy = np.maximum(entropy, 0.0)
    return coherence / (1.0 + entropy)


def _stab_entropy_weighted(entropy, coherence):
    entropy = np.maximum(entropy, 0.0)
    return coherence * np.exp(-entropy)


def _stab_normalized(entropy, coherence):
    c = coherence.copy()
    if c.max() > c.min():
        c = (c - c.min()) / (c.max() - c.min() + 1e-12)

    e = entropy.copy()
    if e.max() > e.min():
        e = (e - e.min()) / (e.max() - e.min() + 1e-12)

    return c * (1.0 - e)


def _compute_stability(ent, coh, mode="classic"):
    entropy = _safe(ent, default=0.0)
    coherence = _safe(coh, default=0.0)

    if mode == "classic":
        return _stab_classic(entropy, coherence)
    if mode == "entropy_weighted":
        return _stab_entropy_weighted(entropy, coherence)
    if mode == "normalized":
        return _stab_normalized(entropy, coherence)

    return _stab_classic(entropy, coherence)


# --------------------------------------------------------------------------- #
# Core API: compute element-level stability
# --------------------------------------------------------------------------- #

def compute_signal_stability(
    elements_csv: str,
    out_csv: str,
    mode: str = "classic",
    emit=DEFAULT_EMIT,
) -> None:
    """
    Compute stability for each element in hilbert_elements.csv.

    Produces:
      - signal_stability.csv
      - graph_stability_nodes.csv  (visualizer-ready)
      - stability_meta.json
      - graph_stability_meta.json

    signal_stability.csv schema:
      doc, element, entropy, coherence, stability

    graph_stability_nodes.csv schema:
      element, element_id,
      entropy, coherence, stability,
      tf, df, idf, tfidf,
      lsa0, lsa1, lsa2
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

    # ----------------------------
    # Resolve fields
    # ----------------------------
    entropy_col = None
    coherence_col = None

    for col in ("entropy", "mean_entropy"):
        if col in df.columns:
            entropy_col = col
            break

    for col in ("coherence", "mean_coherence"):
        if col in df.columns:
            coherence_col = col
            break

    if entropy_col is None or coherence_col is None:
        _log(
            emit,
            "warn",
            "[stability] missing entropy or coherence fields",
            columns=list(df.columns),
        )
        return

    if "element" not in df.columns:
        df["element"] = df.get("token", df.index.astype(str)).astype(str)

    if "element_id" not in df.columns:
        df["element_id"] = df.index.astype(int)

    if "doc" not in df.columns:
        df["doc"] = "corpus"

    # ----------------------------
    # Compute stability
    # ----------------------------
    entropy = df[entropy_col]
    coherence = df[coherence_col]
    stability = _compute_stability(entropy, coherence, mode=mode)

    df["stability"] = stability

    # ----------------------------
    # Legacy CSV (stable schema)
    # ----------------------------
    legacy = df[["doc", "element", entropy_col, coherence_col, "stability"]].copy()
    legacy = legacy.rename(
        columns={entropy_col: "entropy", coherence_col: "coherence"}
    )

    try:
        legacy.to_csv(out_csv, index=False)
        _log(emit, "info", "[stability] wrote signal_stability.csv", path=out_csv)
    except Exception as exc:
        _log(emit, "warn", "[stability] failed to write signal_stability.csv", error=str(exc))

    # ----------------------------
    # Graph visualizer node table
    # ----------------------------
    graph_cols = [
        "element", "element_id",
        entropy_col, coherence_col, "stability",
        "tf", "df", "idf", "tfidf",
        "lsa0", "lsa1", "lsa2",
    ]

    for c in graph_cols:
        if c not in df.columns:
            df[c] = np.nan

    graph_nodes = df[graph_cols].copy()
    graph_nodes = graph_nodes.rename(
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

    # ----------------------------
    # Diagnostics meta
    # ----------------------------
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

    # Graph visualizer meta
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


# --------------------------------------------------------------------------- #
# Compound-level aggregation
# --------------------------------------------------------------------------- #

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

    Produces compound_stability.csv.
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

    merged = mol_df.merge(stab_df[["element", "entropy", "coherence", "stability"]],
                          on="element",
                          how="left")

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

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("mean_element_stability", ascending=False)

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
