"""
signatures.py — seed layer for informational crystals (mis/disinformation)

This module computes *epistemic signatures* for each informational element,
based on span-level labels such as:

    - information
    - misinformation
    - disinformation
    - ambiguous / unknown

Inputs (from results_dir):
    - hilbert_elements.csv
    - epistemic_labels.csv  (optional, but needed for useful output)

Outputs (into results_dir):
    - signatures.csv
    - signatures.json

Each element gets:
    - counts per label
    - probabilities per label (Laplace-smoothed)
    - Shannon entropy in bits
    - dominant_label
    - total support

This is intentionally lightweight but structurally aligned with the future
"crystal" storage: it is layer 0 of a misinfo crystal.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Any

import numpy as np
import pandas as pd

from . import DEFAULT_EMIT  # reuse orchestrator logging lambda if you already have one


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _log(msg: str, emit=DEFAULT_EMIT) -> None:
    print(msg)
    emit("log", {"message": msg})


_CANON_LABELS = {
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


_LABEL_ORDER = ["information", "misinformation", "disinformation", "ambiguous"]


def _canon_label(raw: str) -> str:
    if not isinstance(raw, str):
        return "ambiguous"
    key = raw.strip().lower()
    return _CANON_LABELS.get(key, "ambiguous")


def _shannon_entropy(probs: np.ndarray) -> float:
    """
    Compute Shannon entropy (bits) for a probability vector.
    Robust to:
        - unnormalized inputs
        - zero entries
        - NaNs / infs
        - boolean accidental inputs
    """
    probs = np.asarray(probs, dtype=float)

    # Remove invalid values
    probs = probs[np.isfinite(probs) & (probs > 0.0)]
    if probs.size == 0:
        return 0.0

    # Normalise to a proper probability distribution
    total = probs.sum()
    if total <= 0:
        return 0.0
    probs = probs / total

    # Shannon entropy in bits
    return float(-np.sum(probs * np.log2(probs)))


# ---------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------


def _load_elements(results_dir: str, emit=DEFAULT_EMIT) -> pd.DataFrame:
    path = os.path.join(results_dir, "hilbert_elements.csv")
    if not os.path.exists(path):
        _log(f"[info] hilbert_elements.csv not found at {path}", emit)
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "element" not in df.columns:
        if "token" in df.columns:
            df["element"] = df["token"].astype(str)
        else:
            _log("[info] No 'element' or 'token' column in hilbert_elements.csv", emit)
            return pd.DataFrame()

    if "span_id" not in df.columns:
        _log("[info] hilbert_elements.csv has no 'span_id' column; "
             "cannot join to labels. Skipping.", emit)
        return pd.DataFrame()

    df["element"] = df["element"].astype(str)
    return df


def _load_labels(results_dir: str, emit=DEFAULT_EMIT) -> pd.DataFrame:
    """
    Attempt to load a span-level epistemic label file.

    We accept any CSV in results_dir called one of:
        - epistemic_labels.csv
        - misinfo_labels.csv
        - truth_intent_labels.csv
    and expect at least:
        span_id, label
    """
    candidate_names = [
        "epistemic_labels.csv",
        "misinfo_labels.csv",
        "truth_intent_labels.csv",
    ]
    path = None
    for name in candidate_names:
        candidate = os.path.join(results_dir, name)
        if os.path.exists(candidate):
            path = candidate
            break

    if path is None:
        _log(
            "[info] No epistemic label file found "
            "(looked for epistemic_labels.csv / misinfo_labels.csv / truth_intent_labels.csv); "
            "skipping misinfo signatures.",
            emit,
        )
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        _log(f"[info] Failed to read label file {path}: {exc}", emit)
        return pd.DataFrame()

    if "span_id" not in df.columns:
        _log(f"[info] Label file {path} missing 'span_id' column; skipping.", emit)
        return pd.DataFrame()

    # Choose label column
    label_col = None
    for cand in ["label", "epistemic_label", "truth_label", "class"]:
        if cand in df.columns:
            label_col = cand
            break

    if label_col is None:
        _log(
            f"[info] Label file {path} has no recognised label column "
            "(expected one of: label / epistemic_label / truth_label / class); "
            "skipping.",
            emit,
        )
        return pd.DataFrame()

    df = df[["span_id", label_col]].copy()
    df.rename(columns={label_col: "label"}, inplace=True)
    df["span_id"] = pd.to_numeric(df["span_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["span_id"])
    df["span_id"] = df["span_id"].astype(int)
    df["label"] = df["label"].apply(_canon_label)

    return df


def compute_signatures(results_dir: str, emit=DEFAULT_EMIT) -> None:
    """
    Public entry point used by the orchestrator.

    Parameters
    ----------
    results_dir : str
        Path to the hilbert_run results directory.
    emit : callable
        Logging/telemetry hook; same shape as in other pipeline steps.
    """
    _log("[info] Computing epistemic signatures for elements...", emit)

    elements = _load_elements(results_dir, emit)
    if elements.empty:
        _log("[info] Elements frame empty; aborting info signatures.", emit)
        return

    labels = _load_labels(results_dir, emit)
    if labels.empty:
        _log("[info] No labels available; aborting info signatures.", emit)
        return

    # Join elements to labels at span level
    df = elements.merge(labels, on="span_id", how="inner")
    if df.empty:
        _log("[info] Join of elements and labels is empty; nothing to do.", emit)
        return

    # Normalise label column
    df["label"] = df["label"].apply(_canon_label)

    # Count by (element, label)
    counts = (
        df.groupby(["element", "label"])
        .size()
        .reset_index(name="count")
    )

    # Pivot to columns per label
    pivot = counts.pivot_table(
        index="element",
        columns="label",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )

    # Ensure all label columns exist
    for lab in _LABEL_ORDER:
        if lab not in pivot.columns:
            pivot[lab] = 0

    pivot = pivot[_LABEL_ORDER].copy()
    pivot.reset_index(inplace=True)

    # Compute support, probabilities (Laplace), entropy, dominant label
    out_rows = []
    for _, row in pivot.iterrows():
        element = str(row["element"])
        n_info = int(row["information"])
        n_mis = int(row["misinformation"])
        n_dis = int(row["disinformation"])
        n_amb = int(row["ambiguous"])

        counts_vec = np.array([n_info, n_mis, n_dis, n_amb], dtype=float)
        support = int(counts_vec.sum())

        # Laplace smoothing: +1 to each class
        smoothed = counts_vec + 1.0
        probs = smoothed / smoothed.sum()

        entropy_bits = _shannon_entropy(probs)
        dominant_idx = int(np.argmax(probs))
        dominant_label = _LABEL_ORDER[dominant_idx]

        out_rows.append(
            {
                "element": element,
                "n_information": n_info,
                "n_misinformation": n_mis,
                "n_disinformation": n_dis,
                "n_ambiguous": n_amb,
                "support": support,
                "p_information": float(probs[0]),
                "p_misinformation": float(probs[1]),
                "p_disinformation": float(probs[2]),
                "p_ambiguous": float(probs[3]),
                "entropy_bits": entropy_bits,
                "dominant_label": dominant_label,
            }
        )

    out_df = pd.DataFrame(out_rows).sort_values(
        ["support", "entropy_bits"], ascending=[False, True]
    )

    csv_path = os.path.join(results_dir, "signatures.csv")
    json_path = os.path.join(results_dir, "signatures.json")

    out_df.to_csv(csv_path, index=False)

    # JSON is a simple list of records keyed by element
    records: Dict[str, Any] = {}
    for rec in out_rows:
        el = rec["element"]
        records[el] = {
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
                "label_space": _LABEL_ORDER,
                "elements": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    _log(
        f"[info] signatures.csv written with {len(out_df)} elements.",
        emit,
    )
    _log(
        f"[misinfo] signatures.json written (seed layer for misinfo crystals).",
        emit,
    )
