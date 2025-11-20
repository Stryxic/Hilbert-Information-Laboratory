# =============================================================================
# hilbert_pipeline/stability_layer.py — Advanced Stability Layer
# =============================================================================
"""
Modernized signal-stability computation compatible with the upgraded
Hilbert pipeline (condensation, molecules, export).

Features:
  - Robust handling of entropy/coherence fields.
  - Multiple stability modes (classic, normalized, entropy-weighted).
  - Degenerate-signal protection (zero entropy, infinite coherence).
  - Optional per-document stability aggregation.
  - Fully backward-compatible: always writes signal_stability.csv in the
    expected (doc, element, entropy, coherence, stability) schema.

Inputs:
  hilbert_elements.csv     (canonical elements)
Outputs:
  signal_stability.csv     (per-element stability)
  stability_meta.json      (diagnostic statistics)
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_numeric(series, default=0.0):
    """Return numeric array with NaNs replaced by default."""
    if series is None:
        return np.array([default], dtype=float)
    arr = pd.to_numeric(series, errors="coerce").to_numpy()
    arr = np.where(np.isfinite(arr), arr, default)
    return arr


def _stability_classic(entropy, coherence):
    """Classic stability formula used in earlier Hilbert versions."""
    entropy = np.maximum(entropy, 0.0)
    return coherence / (1.0 + entropy)


def _stability_entropy_weighted(entropy, coherence):
    """Entropy-weighted stability: coherence * exp(-entropy)."""
    entropy = np.maximum(entropy, 0.0)
    return coherence * np.exp(-entropy)


def _stability_normalized(entropy, coherence):
    """Normalize coherence and entropy into comparable ranges."""
    # normalize coherence into [0, 1]
    c = coherence.copy()
    if c.size and c.max() > c.min():
        c = (c - c.min()) / (c.max() - c.min() + 1e-12)

    # normalize entropy into [0, 1]
    e = entropy.copy()
    if e.size and e.max() > e.min():
        e = (e - e.min()) / (e.max() - e.min() + 1e-12)

    return c * (1.0 - e)


def _compute_stability(entropy, coherence, mode="classic"):
    entropy = _safe_numeric(entropy, default=0.0)
    coherence = _safe_numeric(coherence, default=0.0)

    if mode == "classic":
        return _stability_classic(entropy, coherence)
    if mode == "entropy_weighted":
        return _stability_entropy_weighted(entropy, coherence)
    if mode == "normalized":
        return _stability_normalized(entropy, coherence)

    # fallback
    return _stability_classic(entropy, coherence)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_signal_stability(
    elements_csv: str,
    out_csv: str,
    mode: str = "classic",
    emit=lambda *_: None,
):
    """
    Compute stability for all informational elements.

    Parameters:
      elements_csv : str      path to hilbert_elements.csv
      out_csv      : str      output path for signal_stability.csv
      mode         : str      'classic', 'normalized', 'entropy_weighted'
      emit         : callable optional logger callback (type, data)

    Notes:
      - Works with both element-level and span-level schemas.
      - Produces stability_meta.json with diagnostics.
      - Per-document aggregation included.

    Output CSV fields:
      doc, element, entropy, coherence, stability
    """
    if not os.path.exists(elements_csv):
        print(f"[stability] File not found: {elements_csv}")
        return

    try:
        df = pd.read_csv(elements_csv)
    except Exception as e:
        print(f"[stability] Failed to read {elements_csv}: {e}")
        return

    emit("log", {"message": f"[stability] Loaded {len(df)} element rows"})

    # Resolve entropy/coherence fields
    entropy_col = None
    coherence_col = None

    for field in ("entropy", "mean_entropy"):
        if field in df.columns:
            entropy_col = field
            break

    for field in ("coherence", "mean_coherence"):
        if field in df.columns:
            coherence_col = field
            break

    if entropy_col is None or coherence_col is None:
        print("[stability] Missing entropy/coherence fields; aborting.")
        return

    # Ensure 'doc' exists for grouping
    if "doc" not in df.columns:
        df["doc"] = "corpus"

    # Ensure 'element' is present
    if "element" not in df.columns:
        df["element"] = df.get("token", df.index.astype(str))

    entropy = df[entropy_col]
    coherence = df[coherence_col]

    # Compute stability
    stab = _compute_stability(entropy, coherence, mode=mode)
    df_out = df.copy()
    df_out["stability"] = stab

    # Prepare output shape
    out = df_out[["doc", "element", entropy_col, coherence_col, "stability"]]
    out = out.rename(columns={
        entropy_col: "entropy",
        coherence_col: "coherence",
    })

    # Export CSV
    out.to_csv(out_csv, index=False)
    emit("log", {"message": f"[stability] Wrote {out_csv}"})

    # -----------------------------------------------------------------------
    # Diagnostics: meta file
    # -----------------------------------------------------------------------
    meta = {
        "mode": mode,
        "num_elements": int(out["element"].astype(str).nunique()),
        "num_docs": int(out["doc"].astype(str).nunique()),
        "entropy_min": float(out["entropy"].min()),
        "entropy_max": float(out["entropy"].max()),
        "coherence_min": float(out["coherence"].min()),
        "coherence_max": float(out["coherence"].max()),
        "stability_min": float(out["stability"].min()),
        "stability_max": float(out["stability"].max()),
        "stability_median": float(out["stability"].median()),
        "stability_mean": float(out["stability"].mean()),
    }

    # Per-document averages
    doc_stats: Dict[str, Dict[str, Any]] = {}
    for doc, g in out.groupby("doc"):
        doc_stats[doc] = {
            "num_elements": int(g["element"].nunique()),
            "entropy_mean": float(g["entropy"].mean()),
            "coherence_mean": float(g["coherence"].mean()),
            "stability_mean": float(g["stability"].mean()),
            "stability_median": float(g["stability"].median()),
        }

    meta["doc_stats"] = doc_stats

    meta_path = os.path.join(os.path.dirname(out_csv), "stability_meta.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        emit("log", {"message": f"[stability] Wrote {meta_path}"})
    except Exception as e:
        print(f"[stability] Failed to write {meta_path}: {e}")

    print("[stability] Stability computation complete.")

# backend/hilbert_pipeline/stability.py

from pathlib import Path
import pandas as pd

def _log(emit, level: str, msg: str, **fields):
    """
    Lightweight logger compatible with the pipeline's emit signature.
    Falls back to print if emit is None or fails.
    """
    if emit is not None:
        try:
            emit("log", {"level": level, "msg": msg, **fields})
            return
        except Exception:
            pass
    print(f"[{level}] {msg} {fields}")


def compute_compound_stability(
    *,
    compounds_json,
    elements_csv,
    stability_csv,
    out_csv,
    emit=None,
):
    """
    Aggregate per-element stability to the compound level.

    Inputs (all paths):
      - compounds_json: informational_compounds.json (for consistency / future use)
      - elements_csv:   hilbert_elements.csv  (currently unused, but kept for API symmetry)
      - stability_csv:  signal_stability.csv
      - out_csv:        compound_stability.csv

    Produces:
      - compound_stability.csv with columns like:
          compound_id,
          n_elements,
          mean_element_coherence,
          mean_element_stability,
          stability_variance
    """
    base_dir = Path(compounds_json).parent
    molecules_path = base_dir / "molecules.csv"

    if not molecules_path.exists():
        _log(
            emit,
            "warn",
            "[compound-stability] molecules.csv not found; aborting",
            path=str(molecules_path),
        )
        return None

    if not Path(stability_csv).exists():
        _log(
            emit,
            "warn",
            "[compound-stability] signal_stability.csv not found; aborting",
            path=str(stability_csv),
        )
        return None

    try:
        mol_df = pd.read_csv(molecules_path)
    except Exception as exc:
        _log(
            emit,
            "warn",
            "[compound-stability] Failed to read molecules.csv",
            error=str(exc),
        )
        return None

    if mol_df.empty:
        _log(emit, "warn", "[compound-stability] molecules.csv empty; aborting")
        return None

    try:
        stab_df = pd.read_csv(stability_csv)
    except Exception as exc:
        _log(
            emit,
            "warn",
            "[compound-stability] Failed to read signal_stability.csv",
            error=str(exc),
        )
        return None

    if stab_df.empty:
        _log(emit, "warn", "[compound-stability] stability table empty; aborting")
        return None

    # Normalise and keep the core columns we actually need
    if "element" not in stab_df.columns:
        _log(
            emit,
            "warn",
            "[compound-stability] stability table missing 'element' column; aborting",
            columns=list(stab_df.columns),
        )
        return None

    stab_core = stab_df[["element", "entropy", "coherence", "stability"]].copy()

    # Join per-element stability to molecules
    merged = mol_df.merge(stab_core, on="element", how="left", suffixes=("", "_stab"))

    rows = []
    for cid, grp in merged.groupby("compound_id"):
        coh = grp["coherence"].dropna()
        stab = grp["stability"].dropna()

        mean_coh = coh.mean() if len(coh) else float("nan")
        mean_stab = stab.mean() if len(stab) else float("nan")
        var_stab = stab.var(ddof=0) if len(stab) else float("nan")

        rows.append(
            {
                "compound_id": cid,
                "n_elements": int(len(grp)),
                "mean_element_coherence": float(mean_coh),
                "mean_element_stability": float(mean_stab),
                "stability_variance": float(var_stab),
            }
        )

    if not rows:
        _log(emit, "warn", "[compound-stability] No compound rows produced; aborting")
        return None

    out_df = pd.DataFrame(rows).sort_values(
        "mean_element_stability", ascending=False
    )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    _log(
        emit,
        "info",
        "[compound-stability] Wrote compound_stability.csv",
        path=str(out_path),
        n_compounds=int(len(out_df)),
    )
    return str(out_path)
