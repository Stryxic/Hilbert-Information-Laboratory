# =============================================================================
# hilbert_pipeline/molecule_layer.py — Molecular Field & Compound Builder (v2)
# =============================================================================
"""
Constructs informational molecules and compounds from the word-level graph.

This layer is responsible for:
  - Turning hilbert_elements.csv + edges.csv into molecular structures
  - Computing per-element thermodynamic signals:
        local_stability, entropy, coherence, temperature
  - Aggregating elements → molecules → compounds
  - Exporting informational_compounds.json

This version integrates:
  - orchestrator event streaming (emit)
  - NaN-safe stats
  - Robust connected-component discovery
  - Regime-aware compound profiling
  - Temperature modelling (entropy - coherence)
  - Protection against degenerate / empty graphs
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

# orchestrator-injected callback
DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda *_: None


# =============================================================================
# Logging / utility
# =============================================================================

def _log(
    emit: Callable[[str, Dict[str, Any]], None],
    level: str,
    msg: str,
    **fields: Any,
) -> None:
    """Pipeline-compatible logger with fallback to print."""
    payload = {"level": level, "msg": msg}
    payload.update(fields)
    try:
        emit("log", payload)
    except Exception:
        print(f"[{level}] {msg} {fields}")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _load_df(path: str, cols: List[str], emit=DEFAULT_EMIT) -> pd.DataFrame:
    """Load CSV and ensure required columns exist (filling with NaNs if missing)."""
    if not os.path.exists(path):
        _log(emit, "warn", "[molecule] Missing file", path=path)
        return pd.DataFrame(columns=cols)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        _log(emit, "warn", "[molecule] Failed to read CSV", path=path, error=str(e))
        return pd.DataFrame(columns=cols)

    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    return df


# =============================================================================
# Element & Regime Loading
# =============================================================================

def _load_elements(elements_csv: str, emit=DEFAULT_EMIT) -> pd.DataFrame:
    """
    Load element table and normalise entropy / coherence field names.

    Ensures:
      - element (str)
      - token (str)
      - mean_entropy
      - mean_coherence
    """
    df = _load_df(
        elements_csv,
        ["element", "token", "mean_entropy", "mean_coherence", "entropy", "coherence"],
        emit,
    )
    if df.empty:
        _log(emit, "warn", "[molecule] hilbert_elements.csv empty", path=elements_csv)
        return df

    df = df.copy()

    # core ids
    if "element" not in df.columns:
        df["element"] = df.get("token", df.index.astype(str))
    df["element"] = df["element"].astype(str)

    if "token" in df.columns:
        df["token"] = df["token"].astype(str)
    else:
        df["token"] = df["element"]

    # normalise entropy/coherence naming
    if "mean_entropy" not in df.columns and "entropy" in df.columns:
        df["mean_entropy"] = df["entropy"]
    if "mean_coherence" not in df.columns and "coherence" in df.columns:
        df["mean_coherence"] = df["coherence"]

    df["mean_entropy"] = df["mean_entropy"].apply(_safe_float)
    df["mean_coherence"] = df["mean_coherence"].apply(_safe_float)

    _log(
        emit,
        "info",
        "[molecule] Loaded elements",
        n_rows=len(df),
        n_unique_elements=int(df["element"].nunique()),
    )
    return df


def _load_edges(edges_csv: str,
                elements_df: pd.DataFrame,
                emit=DEFAULT_EMIT) -> pd.DataFrame:
    """Load co-occurrence edges and restrict to known elements."""
    df = _load_df(edges_csv, ["source", "target", "weight"], emit)
    if df.empty:
        _log(emit, "warn", "[molecule] edges.csv empty", path=edges_csv)
        return df

    df = df.copy()
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    df["weight"] = df["weight"].apply(_safe_float)

    valid = set(elements_df["element"])
    df = df[df["source"].isin(valid) & df["target"].isin(valid)]
    df = df[df["source"] != df["target"]]

    if df.empty:
        _log(emit, "warn", "[molecule] No valid edges after filtering.", path=edges_csv)
    else:
        _log(
            emit,
            "info",
            "[molecule] Loaded edges",
            n_edges=len(df),
            n_nodes=int(len(valid)),
        )

    return df


def _load_regimes(elements_csv: str, emit=DEFAULT_EMIT) -> Dict[str, Dict[str, float]]:
    """
    Optional: load element regime scores from lsa_field.json.

    Expects (if present):
      lsa_field.json["element_regimes"][token] = {info, misinfo, disinfo}
    """
    base = os.path.dirname(elements_csv)
    lsa_path = os.path.join(base, "lsa_field.json")
    if not os.path.exists(lsa_path):
        return {}

    try:
        with open(lsa_path, "r", encoding="utf-8") as f:
            lsa = json.load(f)
        reg_by_tok = lsa.get("element_regimes") or {}
    except Exception as e:
        _log(emit, "warn", "[molecule] Failure loading regimes", error=str(e))
        return {}

    try:
        df = pd.read_csv(elements_csv)
    except Exception:
        return {}

    if "element" not in df.columns or "token" not in df.columns:
        return {}

    reg_map: Dict[str, Dict[str, float]] = {}
    for _, row in df[["element", "token"]].dropna().iterrows():
        el = str(row["element"])
        tok = str(row["token"])
        if tok in reg_by_tok:
            r = reg_by_tok[tok]
            reg_map[el] = {
                "info": _safe_float(r.get("info"), 0.0),
                "misinfo": _safe_float(r.get("misinfo"), 0.0),
                "disinfo": _safe_float(r.get("disinfo"), 0.0),
            }

    if reg_map:
        _log(
            emit,
            "info",
            "[molecule] Loaded element regimes",
            n_regimes=int(len(reg_map)),
        )
    return reg_map


# =============================================================================
# Molecular Mechanics
# =============================================================================

def _connected_components(edges: pd.DataFrame,
                          elements_df: pd.DataFrame,
                          emit=DEFAULT_EMIT) -> List[List[str]]:
    """Return a list of components, each a list of element_ids."""
    if edges.empty:
        _log(emit, "warn", "[molecule] No edges: treating each element as singleton.",)
        return [[el] for el in elements_df["element"].tolist()]

    neighbors: Dict[str, set] = {}
    for _, row in edges.iterrows():
        s, t = row["source"], row["target"]
        neighbors.setdefault(s, set()).add(t)
        neighbors.setdefault(t, set()).add(s)

    visited = set()
    comps: List[List[str]] = []

    for el in elements_df["element"]:
        if el in visited:
            continue

        stack = [el]
        comp: List[str] = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.append(node)
            for nb in neighbors.get(node, []):
                if nb not in visited:
                    stack.append(nb)

        comps.append(sorted(comp))

    _log(
        emit,
        "info",
        "[molecule] Discovered connected components",
        n_components=len(comps),
    )
    return comps


def _compute_local_stability(edges: pd.DataFrame) -> Dict[str, float]:
    """Return element→mean_edge_weight map."""
    if edges.empty:
        return {}

    weights: Dict[str, List[float]] = {}
    for _, row in edges.iterrows():
        s, t = row["source"], row["target"]
        w = _safe_float(row["weight"], 0.0)
        weights.setdefault(s, []).append(w)
        weights.setdefault(t, []).append(w)

    return {el: float(np.mean(ws)) for el, ws in weights.items()}


def build_molecule_df(
    elements_df: pd.DataFrame,
    edges: pd.DataFrame,
    regimes: Dict[str, Dict[str, float]],
    emit=DEFAULT_EMIT,
) -> pd.DataFrame:
    """
    Produce a row-per-element molecule_df with:

      compound_id, element, degree,
      local_stability, mean_entropy_elem, mean_coherence_elem,
      info_score, misinfo_score, disinfo_score
    """
    if elements_df.empty:
        _log(emit, "warn", "[molecule] Empty element table; aborting molecule build.")
        return pd.DataFrame()

    comps = _connected_components(edges, elements_df, emit)
    local = _compute_local_stability(edges)

    rows: List[Dict[str, Any]] = []
    for idx, comp in enumerate(comps, start=1):
        cid = f"C{idx:04d}"
        for el in comp:
            el = str(el)
            sub = elements_df.loc[elements_df["element"] == el]

            row: Dict[str, Any] = {
                "compound_id": cid,
                "element": el,
                "degree": int(
                    edges[
                        (edges["source"] == el) | (edges["target"] == el)
                    ].shape[0]
                ),
                "local_stability": float(local.get(el, 0.0)),
                "mean_entropy_elem": float(sub["mean_entropy"].mean())
                if not sub.empty
                else 0.0,
                "mean_coherence_elem": float(sub["mean_coherence"].mean())
                if not sub.empty
                else 0.0,
            }

            # append regime scores if available
            r = regimes.get(el, {})
            row["info_score"] = float(r.get("info", 0.0))
            row["misinfo_score"] = float(r.get("misinfo", 0.0))
            row["disinfo_score"] = float(r.get("disinfo", 0.0))

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        _log(emit, "warn", "[molecule] Molecule table empty after assembly.")
    else:
        _log(
            emit,
            "info",
            "[molecule] Constructed molecules",
            n_molecules=int(df["compound_id"].nunique()),
            n_elements=int(df["element"].nunique()),
        )
    return df


# =============================================================================
# Temperature Model
# =============================================================================

def add_temperature(molecule_df: pd.DataFrame,
                    emit=DEFAULT_EMIT) -> pd.DataFrame:
    """T = normalized(mean_entropy - mean_coherence)."""
    if molecule_df.empty:
        return molecule_df

    x = molecule_df["mean_entropy_elem"] - molecule_df["mean_coherence_elem"]
    xmin, xmax = float(x.min()), float(x.max())

    if not np.isfinite(xmin) or not np.isfinite(xmax) or (xmax - xmin) < 1e-9:
        T = np.zeros_like(x) + 0.5
    else:
        T = (x - xmin) / (xmax - xmin + 1e-9)

    df = molecule_df.copy()
    df["temperature"] = T.astype(float)
    _log(emit, "info", "[molecule] Temperature field added.")
    return df


# =============================================================================
# Compound Aggregation
# =============================================================================

def aggregate_compounds(
    molecule_df: pd.DataFrame,
    edges: pd.DataFrame,
    regimes: Dict[str, Dict[str, float]],
    emit=DEFAULT_EMIT,
) -> pd.DataFrame:
    """Aggregate per-compound metrics (stability, temperature, regime scores)."""
    if molecule_df.empty:
        _log(emit, "warn", "[molecule] Empty molecule table; cannot aggregate.")
        return pd.DataFrame()

    comp_map = molecule_df.set_index("element")["compound_id"].to_dict()

    # per-compound bond counts & weights
    bond_counts: Dict[str, int] = {}
    wcollect: Dict[str, List[float]] = {}

    if not edges.empty:
        for _, row in edges.iterrows():
            s, t = row["source"], row["target"]
            w = _safe_float(row["weight"], 0.0)
            cs = comp_map.get(s)
            ct = comp_map.get(t)
            if cs and cs == ct:
                bond_counts[cs] = bond_counts.get(cs, 0) + 1
                wcollect.setdefault(cs, []).append(w)

    rows: List[Dict[str, Any]] = []
    for cid, group in molecule_df.groupby("compound_id"):
        ne = group["element"].nunique()
        nb = int(bond_counts.get(cid, 0))

        mean_ls = float(group["local_stability"].mean())
        mean_temp = float(group["temperature"].mean())

        info = float(group["info_score"].mean())
        mis = float(group["misinfo_score"].mean())
        dis = float(group["disinfo_score"].mean())

        # compound stability: mean of bond weights if available
        cstab = float(np.mean(wcollect[cid])) if cid in wcollect else mean_ls

        rows.append(
            {
                "compound_id": cid,
                "num_elements": int(ne),
                "num_bonds": nb,
                "compound_stability": cstab,
                "mean_local_stability": mean_ls,
                "mean_temperature": mean_temp,
                "mean_info": info,
                "mean_misinfo": mis,
                "mean_disinfo": dis,
            }
        )

    out = (
        pd.DataFrame(rows)
        .sort_values(
            ["compound_stability", "num_elements"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    _log(
        emit,
        "info",
        "[molecule] Aggregated compounds",
        n_compounds=int(len(out)),
    )
    return out


# =============================================================================
# Export Layer
# =============================================================================

def export_compounds(
    out_dir: str,
    molecule_df: pd.DataFrame,
    compound_df: pd.DataFrame,
    emit=DEFAULT_EMIT,
) -> None:
    """Export informational_compounds.json from molecule/compound tables."""
    if compound_df.empty:
        _log(emit, "warn", "[molecule] No compounds to export.")
        return

    os.makedirs(out_dir, exist_ok=True)

    elements_by_cid: Dict[str, List[str]] = {}
    if not molecule_df.empty:
        for cid, g in molecule_df.groupby("compound_id"):
            elements_by_cid[cid] = sorted(g["element"].astype(str).tolist())

    out: Dict[str, Any] = {}
    for _, row in compound_df.iterrows():
        cid = str(row["compound_id"])
        out[cid] = {
            "compound_id": cid,
            "elements": elements_by_cid.get(cid, []),
            "num_elements": int(row["num_elements"]),
            "num_bonds": int(row["num_bonds"]),
            "stability": float(row["compound_stability"]),
            "temperature": float(row["mean_temperature"]),
            "info": float(row["mean_info"]),
            "mis": float(row["mean_misinfo"]),
            "dis": float(row["mean_disinfo"]),
        }

    path = os.path.join(out_dir, "informational_compounds.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        _log(emit, "info", "[molecule] Exported informational_compounds.json", path=path)
        try:
            emit("artifact", {"path": path, "kind": "informational_compounds"})
        except Exception:
            pass
    except Exception as e:
        _log(emit, "warn", "[molecule] Failed to write compounds JSON", path=path, error=str(e))


# =============================================================================
# Pipeline Entry
# =============================================================================

def run_molecule_stage(
    results_dir: str,
    emit=DEFAULT_EMIT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point for orchestrator (new name).

    Reads:
      - hilbert_elements.csv
      - edges.csv
      - lsa_field.json (optional regimes)

    Writes:
      - informational_compounds.json
      - returns (molecule_df, compound_df)
    """
    el_path = os.path.join(results_dir, "hilbert_elements.csv")
    edges_path = os.path.join(results_dir, "edges.csv")

    elements_df = _load_elements(el_path, emit)
    if elements_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    edges = _load_edges(edges_path, elements_df, emit)
    regimes = _load_regimes(el_path, emit)

    molecule_df = build_molecule_df(elements_df, edges, regimes, emit)
    molecule_df = add_temperature(molecule_df, emit)

    compound_df = aggregate_compounds(molecule_df, edges, regimes, emit)

    export_compounds(results_dir, molecule_df, compound_df, emit)

    return molecule_df, compound_df


# Backwards-compatible alias for orchestrator imports
def run_molecule_layer(results_dir: str, emit=DEFAULT_EMIT):
    """Alias so orchestrator can import run_molecule_layer(...)."""
    return run_molecule_stage(results_dir, emit=emit)


# -------------------------------------------------------------------------#
# Backwards-compat convenience helpers
# -------------------------------------------------------------------------#

def compute_molecule_stability(
    edges_csv: str,
    elements_csv: str,
    emit=DEFAULT_EMIT,
) -> pd.DataFrame:
    elements_df = _load_elements(elements_csv, emit)
    edges = _load_edges(edges_csv, elements_df, emit)
    regimes = _load_regimes(elements_csv, emit)
    return build_molecule_df(elements_df, edges, regimes, emit)


def compute_molecule_temperature(
    molecule_df: pd.DataFrame,
    elements_df: pd.DataFrame | None = None,
):
    # elements_df kept for API symmetry, but unused now
    return add_temperature(molecule_df)


def export_molecule_summary(
    out_dir: str,
    molecule_df: pd.DataFrame,
    compound_df: pd.DataFrame,
):
    export_compounds(out_dir, molecule_df, compound_df)


__all__ = [
    "run_molecule_stage",
    "run_molecule_layer",
    "aggregate_compounds",
    "compute_molecule_stability",
    "compute_molecule_temperature",
    "export_molecule_summary",
]
