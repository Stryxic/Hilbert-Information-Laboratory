# =============================================================================
# hilbert_pipeline/signatures.py — Epistemic Signatures Layer (v3.1)
# =============================================================================
"""
Computes epistemic signatures for informational elements:

    information
    misinformation
    disinformation
    ambiguous

Works with both:
  - span-level label files (span_id → label)
  - element-level label files (element → label)

Outputs:
  - signatures.csv                    (human-readable table)
  - signatures.json                   (structured)
  - graph_signatures_nodes.csv        (graph-contract ready)

This module is fully compatible with:
  - hilbert_elements.csv (v3)
  - orchestrator 3.1
  - molecule and stability layers
  - graph visualizer (contract-ready node tables)
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, Callable, List

import numpy as np
import pandas as pd

try:
    from . import DEFAULT_EMIT
except Exception:
    DEFAULT_EMIT = lambda *_: None


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def _log(emit, level, msg, **fields):
    payload = {"level": level, "msg": msg}
    payload.update(fields)
    print(f"[{level}] {msg} {fields}")
    try:
        emit("log", payload)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Canonical label mapping
# -----------------------------------------------------------------------------

_CANON = {
    "information": "information",
    "info": "information",
    "true": "information",
    "truth": "information",

    "misinformation": "misinformation",
    "misinfo": "misinformation",
    "error": "misinformation",

    "disinformation": "disinformation",
    "disinfo": "disinformation",
    "propaganda": "disinformation",

    "ambiguous": "ambiguous",
    "unknown": "ambiguous",
    "noise": "ambiguous",
    "unclassified": "ambiguous",
}

_LABEL_SPACE = ["information", "misinformation", "disinformation", "ambiguous"]


def _canon_label(val: Any) -> str:
    if not isinstance(val, str):
        return "ambiguous"
    key = val.strip().lower()
    return _CANON.get(key, "ambiguous")


def _shannon_entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    probs = probs[np.isfinite(probs) & (probs > 0)]
    if probs.size == 0:
        return 0.0
    total = probs.sum()
    if total <= 0:
        return 0.0
    p = probs / total
    return float(-np.sum(p * np.log2(p)))


# -----------------------------------------------------------------------------
# Label loading
# -----------------------------------------------------------------------------

def _load_label_table(results_dir: str, emit) -> pd.DataFrame:
    """
    Loads whichever labels exist:
        - span-level: span_id, label
        - element-level: element, label

    Accepts:
        epistemic_labels.csv
        misinfo_labels.csv
        truth_intent_labels.csv
        labels.csv
    """
    candidates = [
        "epistemic_labels.csv",
        "misinfo_labels.csv",
        "truth_intent_labels.csv",
        "labels.csv",
    ]

    path = None
    for c in candidates:
        p = os.path.join(results_dir, c)
        if os.path.exists(p):
            path = p
            break

    if not path:
        _log(emit, "warn", "[signatures] No label file found", looked_for=candidates)
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        _log(emit, "warn", "[signatures] Failed to load label file", path=path, error=str(exc))
        return pd.DataFrame()

    # Detect mode
    if "span_id" in df.columns:
        mode = "span"
    elif "element" in df.columns:
        mode = "element"
    else:
        _log(emit, "warn", "[signatures] Label file has neither span_id nor element column", path=path)
        return pd.DataFrame()

    # Identify label column
    label_col = None
    for c in ["label", "epistemic_label", "truth_label", "class"]:
        if c in df.columns:
            label_col = c
            break

    if not label_col:
        _log(emit, "warn", "[signatures] No usable label column detected", columns=list(df.columns))
        return pd.DataFrame()

    df = df[[col for col in df.columns if col in ("span_id", "element", label_col)]].copy()
    df.rename(columns={label_col: "label"}, inplace=True)
    df["label"] = df["label"].apply(_canon_label)

    return df


def _load_elements(results_dir: str, emit) -> pd.DataFrame:
    path = os.path.join(results_dir, "hilbert_elements.csv")
    if not os.path.exists(path):
        _log(emit, "warn", "[signatures] hilbert_elements.csv missing", path=path)
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "element" not in df.columns:
        if "token" in df.columns:
            df["element"] = df["token"].astype(str)
        else:
            _log(emit, "warn", "[signatures] hilbert_elements.csv has no element column", path=path)
            return pd.DataFrame()

    df["element"] = df["element"].astype(str)
    return df


# -----------------------------------------------------------------------------
# Main computation
# -----------------------------------------------------------------------------

def compute_signatures(results_dir: str, emit=DEFAULT_EMIT) -> str:
    _log(emit, "info", "[signatures] Computing epistemic signatures...")

    elements = _load_elements(results_dir, emit)
    if elements.empty:
        _log(emit, "warn", "[signatures] No elements available")
        return ""

    labels = _load_label_table(results_dir, emit)
    if labels.empty:
        _log(emit, "warn", "[signatures] No labels available")
        return ""

    # Determine label mode
    join_mode = "span" if "span_id" in labels.columns else "element"

    if join_mode == "span":
        if "span_id" not in elements.columns:
            _log(emit, "warn", "[signatures] span-level labels but no span_id in elements")
            return ""

        df = elements.merge(labels, on="span_id", how="inner")
    else:
        df = elements.merge(labels, on="element", how="inner")

    if df.empty:
        _log(emit, "warn", "[signatures] No matching labels after join")
        return ""

    # Count per (element, label)
    counts = (
        df.groupby(["element", "label"])
        .size()
        .reset_index(name="count")
    )

    pivot = counts.pivot_table(
        index="element",
        columns="label",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )

    for lab in _LABEL_SPACE:
        if lab not in pivot.columns:
            pivot[lab] = 0

    pivot = pivot[_LABEL_SPACE].reset_index()

    # Build signature rows
    rows: List[Dict[str, Any]] = []
    for _, row in pivot.iterrows():
        e = str(row["element"])
        vec = np.array([row["information"], row["misinformation"], row["disinformation"], row["ambiguous"]], dtype=float)

        support = int(vec.sum())
        smoothed = vec + 1
        probs = smoothed / smoothed.sum()
        ent = _shannon_entropy(probs)
        dom_idx = int(np.argmax(probs))
        dom = _LABEL_SPACE[dom_idx]

        rows.append(
            {
                "element": e,
                "n_information": int(vec[0]),
                "n_misinformation": int(vec[1]),
                "n_disinformation": int(vec[2]),
                "n_ambiguous": int(vec[3]),
                "support": support,
                "p_information": float(probs[0]),
                "p_misinformation": float(probs[1]),
                "p_disinformation": float(probs[2]),
                "p_ambiguous": float(probs[3]),
                "entropy_bits": ent,
                "dominant_label": dom,
            }
        )

    out_df = pd.DataFrame(rows).sort_values(
        ["support", "entropy_bits"],
        ascending=[False, True],
    )

    # Write CSV
    csv_path = os.path.join(results_dir, "signatures.csv")
    out_df.to_csv(csv_path, index=False)

    # JSON export
    json_path = os.path.join(results_dir, "signatures.json")
    out = {}
    for rec in rows:
        e = rec["element"]
        out[e] = {
            "support": rec["support"],
            "p": {
                "information": rec["p_information"],
                "misinformation": rec["p_misinformation"],
                "disinformation": rec["p_disinformation"],
                "ambiguous": rec["p_ambiguous"],
            },
            "entropy_bits": rec["entropy_bits"],
            "dominant_label": rec["dominant_label"],
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results_dir": results_dir,
                "label_space": _LABEL_SPACE,
                "elements": out,
            },
            f,
            indent=2,
        )

    # ----------------------------------------------------------------------
    # Graph-contract signature table
    # ----------------------------------------------------------------------
    g_path = os.path.join(results_dir, "graph_signatures_nodes.csv")

    gdf = out_df[[
        "element",
        "support",
        "entropy_bits",
        "p_information",
        "p_misinformation",
        "p_disinformation",
        "p_ambiguous",
        "dominant_label",
    ]].copy()

    # Add element_id if available
    if "element_id" in elements.columns:
        gdf = gdf.merge(
            elements[["element", "element_id"]],
            on="element",
            how="left",
        )
    else:
        gdf["element_id"] = np.arange(len(gdf))

    gdf.to_csv(g_path, index=False)

    _log(emit, "info", "[signatures] signatures.csv and graph_signatures_nodes.csv written", n_elements=len(out_df))

    return csv_path


__all__ = ["compute_signatures"]
