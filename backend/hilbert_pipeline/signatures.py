# =============================================================================
# hilbert_pipeline/signatures.py — Epistemic Signatures Layer (v3.1)
# =============================================================================
"""
Epistemic Signatures Layer
==========================

This module computes **epistemic signatures** for informational elements in the
Hilbert pipeline, quantifying how strongly each element is associated with:

- information
- misinformation
- disinformation
- ambiguous

It is designed to bridge between human annotation (span- or element-level
labels) and graph-level diagnostics. The resulting signatures are consumable by:

- the molecule and stability layers
- the graph visualizer (via graph-contract ready node tables)
- higher level analytics and reporting modules

Supported label sources
-----------------------

The layer is intentionally flexible about label provenance and file naming.
It will look for one of the following label files in ``results_dir``:

- ``epistemic_labels.csv``
- ``misinfo_labels.csv``
- ``truth_intent_labels.csv``
- ``labels.csv``

and interpret them in one of two ways:

- **Span-level labels**:
    A column ``span_id`` identifies spans, and a label column
    (``label``, ``epistemic_label``, ``truth_label``, or ``class``)
    provides the epistemic class.
- **Element-level labels**:
    A column ``element`` identifies elements directly.

Labels are mapped into a canonical label space using a conservative mapping
that normalises common synonyms and fallbacks:

- information / info / true / truth  -> ``"information"``
- misinformation / misinfo / error   -> ``"misinformation"``
- disinformation / disinfo / propaganda -> ``"disinformation"``
- unknown / noise / unclassified and any unknown string -> ``"ambiguous"``

Outputs
-------

Given a run directory containing:

- ``hilbert_elements.csv`` (v3 element table)
- at least one label CSV as described above

this layer writes three outputs into ``results_dir``:

1. ``signatures.csv``

   Human readable table with columns:

   - ``element``
   - ``n_information``, ``n_misinformation``, ``n_disinformation``, ``n_ambiguous``
   - ``support`` (total label count)
   - ``p_information``, ``p_misinformation``, ``p_disinformation``, ``p_ambiguous``
   - ``entropy_bits`` (Shannon entropy in bits, with Laplace smoothing)
   - ``dominant_label`` (argmax over smoothed probabilities)

2. ``signatures.json``

   Structured mapping suitable for programmatic access:

   .. code-block:: json

      {
        "results_dir": "...",
        "label_space": ["information", "misinformation", "disinformation", "ambiguous"],
        "elements": {
          "element_string": {
            "support": 42,
            "p": {
              "information": 0.7,
              "misinformation": 0.1,
              "disinformation": 0.1,
              "ambiguous": 0.1
            },
            "entropy_bits": 1.28,
            "dominant_label": "information"
          },
          ...
        }
      }

3. ``graph_signatures_nodes.csv``

   Graph-contract compatible node table:

   - ``element``
   - ``element_id`` (if available in ``hilbert_elements.csv``)
   - ``support``
   - ``entropy_bits``
   - ``p_information``, ``p_misinformation``, ``p_disinformation``, ``p_ambiguous``
   - ``dominant_label``

Public API
----------

.. autofunction:: compute_signatures

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
    # Fallback when imported outside a fully initialised package
    DEFAULT_EMIT = lambda *_: None  # type: ignore


# =============================================================================
# Logging utilities
# =============================================================================

def _log(emit: Callable[[str, Dict[str, Any]], None],
         level: str,
         msg: str,
         **fields: Any) -> None:
    """
    Emit a structured log message to both stdout and the orchestrator.

    Parameters
    ----------
    emit : callable
        Orchestrator-compatible logger with signature
        ``emit(kind: str, payload: dict)``.
    level : str
        Severity or channel label, for example ``"info"`` or ``"warn"``.
    msg : str
        Human readable log message.
    **fields :
        Additional context fields to attach to the log payload.
    """
    payload = {"level": level, "msg": msg}
    payload.update(fields)
    print(f"[{level}] {msg} {fields}")
    try:
        emit("log", payload)
    except Exception:
        # Logging is best-effort only
        pass


# =============================================================================
# Canonical label mapping
# =============================================================================

_CANON: Dict[str, str] = {
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

_LABEL_SPACE: List[str] = [
    "information",
    "misinformation",
    "disinformation",
    "ambiguous",
]


def _canon_label(val: Any) -> str:
    """
    Map a raw label into the canonical epistemic label space.

    Non string inputs and unknown strings are mapped to ``"ambiguous"``.

    Parameters
    ----------
    val : Any
        Raw label value (often a string).

    Returns
    -------
    str
        Canonical label in ``_LABEL_SPACE``.
    """
    if not isinstance(val, str):
        return "ambiguous"
    key = val.strip().lower()
    return _CANON.get(key, "ambiguous")


def _shannon_entropy(probs: np.ndarray) -> float:
    """
    Compute Shannon entropy in bits for a probability vector.

    Zero or non finite entries are removed before normalisation.

    Parameters
    ----------
    probs : array-like
        Non negative probabilities (do not need to be normalised).

    Returns
    -------
    float
        Entropy in bits.
    """
    probs = np.asarray(probs, dtype=float)
    probs = probs[np.isfinite(probs) & (probs > 0)]
    if probs.size == 0:
        return 0.0
    total = probs.sum()
    if total <= 0:
        return 0.0
    p = probs / total
    return float(-np.sum(p * np.log2(p)))


# =============================================================================
# Label and element loading
# =============================================================================

def _load_label_table(results_dir: str,
                      emit: Callable[[str, Dict[str, Any]], None]) -> pd.DataFrame:
    """
    Load epistemic labels from the first available label file.

    Search order inside ``results_dir``:

    1. ``epistemic_labels.csv``
    2. ``misinfo_labels.csv``
    3. ``truth_intent_labels.csv``
    4. ``labels.csv``

    Expected schemas
    ----------------

    Span-level labels
        File contains ``span_id`` and one label column:
        ``label``, ``epistemic_label``, ``truth_label``, or ``class``.

    Element-level labels
        File contains ``element`` and one label column as above.

    Returns
    -------
    pandas.DataFrame
        Table with columns:

        - ``span_id`` or ``element`` (depending on mode)
        - ``label`` (canonicalised into ``_LABEL_SPACE``)

        An empty DataFrame is returned if no usable label file is found.
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

    # Detect mode: span-level or element-level
    if "span_id" in df.columns:
        mode = "span"
    elif "element" in df.columns:
        mode = "element"
    else:
        _log(
            emit,
            "warn",
            "[signatures] Label file has neither span_id nor element column",
            path=path,
        )
        return pd.DataFrame()

    # Identify label column
    label_col = None
    for c in ["label", "epistemic_label", "truth_label", "class"]:
        if c in df.columns:
            label_col = c
            break

    if not label_col:
        _log(
            emit,
            "warn",
            "[signatures] No usable label column detected",
            columns=list(df.columns),
        )
        return pd.DataFrame()

    keep_cols = [col for col in df.columns if col in ("span_id", "element", label_col)]
    df = df[keep_cols].copy()
    df.rename(columns={label_col: "label"}, inplace=True)
    df["label"] = df["label"].apply(_canon_label)

    # Explicitly mark mode for downstream logic (via column presence)
    if mode == "span":
        # ensure span_id is present in the subset
        if "span_id" not in df.columns:
            _log(emit, "warn", "[signatures] span-level detected but span_id missing after filtering")
            return pd.DataFrame()
    else:
        if "element" not in df.columns:
            _log(emit, "warn", "[signatures] element-level detected but element column missing after filtering")
            return pd.DataFrame()

    return df


def _load_elements(results_dir: str,
                   emit: Callable[[str, Dict[str, Any]], None]) -> pd.DataFrame:
    """
    Load the canonical element table from ``hilbert_elements.csv``.

    If the file lacks an ``element`` column but has ``token``, that
    column is promoted to ``element``.

    Parameters
    ----------
    results_dir : str
        Hilbert run directory.
    emit : callable
        Orchestrator logger.

    Returns
    -------
    pandas.DataFrame
        Element table with at least an ``element`` column, or an empty
        DataFrame if loading fails.
    """
    path = os.path.join(results_dir, "hilbert_elements.csv")
    if not os.path.exists(path):
        _log(emit, "warn", "[signatures] hilbert_elements.csv missing", path=path)
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception:
        _log(emit, "warn", "[signatures] Failed to read hilbert_elements.csv", path=path)
        return pd.DataFrame()

    if "element" not in df.columns:
        if "token" in df.columns:
            df["element"] = df["token"].astype(str)
        else:
            _log(
                emit,
                "warn",
                "[signatures] hilbert_elements.csv has no element or token column",
                path=path,
            )
            return pd.DataFrame()

    df["element"] = df["element"].astype(str)
    return df


# =============================================================================
# Main computation
# =============================================================================

def compute_signatures(results_dir: str, emit: Callable[[str, Dict[str, Any]], None] = DEFAULT_EMIT) -> str:
    """
    Compute epistemic signatures for each element in ``hilbert_elements.csv``.

    The function joins the canonical element table with span-level or
    element-level labels, aggregates counts over the canonical label space,
    applies Laplace smoothing, and derives:

    - per element label counts  
    - smoothed label probabilities  
    - Shannon entropy in bits  
    - a dominant label

    It writes:

    - ``signatures.csv`` (human readable table)
    - ``signatures.json`` (structured mapping)
    - ``graph_signatures_nodes.csv`` (graph-contract node attributes)

    Parameters
    ----------
    results_dir : str
        Hilbert run results directory containing ``hilbert_elements.csv``
        and at least one label CSV.
    emit : callable, optional
        Orchestrator logger. Defaults to :data:`DEFAULT_EMIT`.

    Returns
    -------
    str
        Path to ``signatures.csv`` if successful, or an empty string if
        the computation is skipped due to missing inputs.
    """
    _log(emit, "info", "[signatures] Computing epistemic signatures...")

    elements = _load_elements(results_dir, emit)
    if elements.empty:
        _log(emit, "warn", "[signatures] No elements available")
        return ""

    labels = _load_label_table(results_dir, emit)
    if labels.empty:
        _log(emit, "warn", "[signatures] No labels available")
        return ""

    # Determine join mode from label columns
    join_mode = "span" if "span_id" in labels.columns else "element"

    if join_mode == "span":
        if "span_id" not in elements.columns:
            _log(
                emit,
                "warn",
                "[signatures] span-level labels but no span_id in elements",
            )
            return ""
        df = elements.merge(labels, on="span_id", how="inner")
    else:
        df = elements.merge(labels, on="element", how="inner")

    if df.empty:
        _log(emit, "warn", "[signatures] No matching labels after join")
        return ""

    # -------------------------------------------------------------------------
    # Count per (element, label) and pivot into a fixed label space
    # -------------------------------------------------------------------------
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

    # Ensure all canonical labels are present as columns
    for lab in _LABEL_SPACE:
        if lab not in pivot.columns:
            pivot[lab] = 0

    pivot = pivot[_LABEL_SPACE].reset_index()

    # -------------------------------------------------------------------------
    # Build signature rows
    # -------------------------------------------------------------------------
    rows: List[Dict[str, Any]] = []
    for _, row in pivot.iterrows():
        e = str(row["element"])
        vec = np.array(
            [
                row["information"],
                row["misinformation"],
                row["disinformation"],
                row["ambiguous"],
            ],
            dtype=float,
        )

        support = int(vec.sum())

        # Laplace smoothing for robust probability estimates
        smoothed = vec + 1.0
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

    # -------------------------------------------------------------------------
    # CSV and JSON exports
    # -------------------------------------------------------------------------
    csv_path = os.path.join(results_dir, "signatures.csv")
    out_df.to_csv(csv_path, index=False)

    json_path = os.path.join(results_dir, "signatures.json")
    elements_json: Dict[str, Any] = {}
    for rec in rows:
        e = rec["element"]
        elements_json[e] = {
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
                "elements": elements_json,
            },
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Graph-contract signature table (node attributes)
    # -------------------------------------------------------------------------
    g_path = os.path.join(results_dir, "graph_signatures_nodes.csv")

    gdf = out_df[
        [
            "element",
            "support",
            "entropy_bits",
            "p_information",
            "p_misinformation",
            "p_disinformation",
            "p_ambiguous",
            "dominant_label",
        ]
    ].copy()

    # Attach element_id for use in graph visualiser and analytics joins
    if "element_id" in elements.columns:
        gdf = gdf.merge(
            elements[["element", "element_id"]],
            on="element",
            how="left",
        )
    else:
        gdf["element_id"] = np.arange(len(gdf), dtype=int)

    gdf.to_csv(g_path, index=False)

    _log(
        emit,
        "info",
        "[signatures] signatures.csv and graph_signatures_nodes.csv written",
        n_elements=len(out_df),
    )

    return csv_path


__all__ = ["compute_signatures"]
