# =============================================================================
# hilbert_pipeline/molecule_layer.py — Molecular Field & Compound Builder
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
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

# orchestrator-injected callback
DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda *_: None


# =============================================================================
# Utility
# =============================================================================

def _log(msg: str, emit=DEFAULT_EMIT):
    print(msg)
    emit("log", {"message": msg})


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _load_df(path: str, cols: List[str], emit=DEFAULT_EMIT) -> pd.DataFrame:
    """Load CSV and ensure required columns exist."""
    if not os.path.exists(path):
        _log(f"[molecule][warn] Missing file: {path}", emit)
        return pd.DataFrame(columns=cols)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        _log(f"[molecule][warn] Failed to read {path}: {e}", emit)
        return pd.DataFrame(columns=cols)

    for c in cols:
        if c not in df.columns:
            df[c] = None

    return df


# =============================================================================
# Element & Regime Loading
# =============================================================================

def _load_elements(elements_csv: str, emit=DEFAULT_EMIT) -> pd.DataFrame:
    df = _load_df(elements_csv,
                  ["element", "token", "mean_entropy", "mean_coherence"],
                  emit)
    if df.empty:
        return df

    df = df.copy()
    df["element"] = df["element"].astype(str)

    if "token" in df.columns:
        df["token"] = df["token"].astype(str)
    else:
        df["token"] = df["element"]

    # unify stats
    df["mean_entropy"] = df["mean_entropy"].apply(_safe_float)
    df["mean_coherence"] = df["mean_coherence"].apply(_safe_float)

    return df


def _load_edges(edges_csv: str,
                elements_df: pd.DataFrame,
                emit=DEFAULT_EMIT) -> pd.DataFrame:
    df = _load_df(edges_csv, ["source", "target", "weight"], emit)
    if df.empty:
        return df

    df = df.copy()
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    df["weight"] = df["weight"].apply(_safe_float)

    valid = set(elements_df["element"])
    df = df[df["source"].isin(valid) & df["target"].isin(valid)]

    if df.empty:
        _log("[molecule][warn] edges.csv has no valid edges after filtering.", emit)

    return df


def _load_regimes(elements_csv: str, emit=DEFAULT_EMIT) -> Dict[str, Dict[str, float]]:
    base = os.path.dirname(elements_csv)
    lsa_path = os.path.join(base, "lsa_field.json")
    if not os.path.exists(lsa_path):
        return {}

    try:
        with open(lsa_path, "r", encoding="utf-8") as f:
            lsa = json.load(f)
        reg_by_tok = lsa.get("element_regimes") or {}
    except Exception as e:
        _log(f"[molecule][warn] Failure loading regimes: {e}", emit)
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
    return reg_map


# =============================================================================
# Molecular Mechanics
# =============================================================================

def _connected_components(edges: pd.DataFrame,
                          elements_df: pd.DataFrame,
                          emit=DEFAULT_EMIT) -> List[List[str]]:
    """Return a list of components, each a list of element_ids."""
    if edges.empty:
        _log("[molecule][warn] No edges: graph is disconnected.", emit)
        return [[el] for el in elements_df["element"].tolist()]

    neighbors: Dict[str, set] = {}
    for _, row in edges.iterrows():
        s, t = row["source"], row["target"]
        neighbors.setdefault(s, set()).add(t)
        neighbors.setdefault(t, set()).add(s)

    visited = set()
    comps = []

    for el in elements_df["element"]:
        if el in visited:
            continue

        stack = [el]
        comp = []
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


def build_molecule_df(elements_df: pd.DataFrame,
                      edges: pd.DataFrame,
                      regimes: Dict[str, Dict[str, float]],
                      emit=DEFAULT_EMIT) -> pd.DataFrame:
    """Produce a row-per-element molecule_df."""
    comps = _connected_components(edges, elements_df, emit)
    local = _compute_local_stability(edges)

    rows = []
    for idx, comp in enumerate(comps, start=1):
        cid = f"C{idx:04d}"
        for el in comp:
            row: Dict[str, Any] = {
                "compound_id": cid,
                "element": el,
                "degree": edges[(edges["source"] == el) | (edges["target"] == el)].shape[0],
                "local_stability": local.get(el, 0.0),
                "mean_entropy_elem": float(
                    elements_df.loc[elements_df["element"] == el, "mean_entropy"]
                    .mean()
                ),
                "mean_coherence_elem": float(
                    elements_df.loc[elements_df["element"] == el, "mean_coherence"]
                    .mean()
                ),
            }

            # append regime scores if available
            r = regimes.get(el, {})
            row["info_score"] = r.get("info", 0.0)
            row["misinfo_score"] = r.get("misinfo", 0.0)
            row["disinfo_score"] = r.get("disinfo", 0.0)

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        _log("[molecule][warn] Molecule table empty after assembly.", emit)
    else:
        _log(
            f"[molecule] Constructed {df['compound_id'].nunique()} molecules "
            f"over {df['element'].nunique()} elements.",
            emit,
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

    if xmax - xmin < 1e-9:
        T = np.zeros_like(x) + 0.5
    else:
        T = (x - xmin) / (xmax - xmin + 1e-9)

    df = molecule_df.copy()
    df["temperature"] = T.astype(float)
    return df


# =============================================================================
# Compound Aggregation
# =============================================================================

def aggregate_compounds(molecule_df: pd.DataFrame,
                        edges: pd.DataFrame,
                        regimes: Dict[str, Dict[str, float]],
                        emit=DEFAULT_EMIT) -> pd.DataFrame:
    """Aggregate per-compound metrics."""
    if molecule_df.empty:
        return pd.DataFrame()

    comp_map = molecule_df.set_index("element")["compound_id"].to_dict()

    # per-compound bond counts & weights
    bond_counts = {}
    wcollect = {}

    if not edges.empty:
        for _, row in edges.iterrows():
            s, t = row["source"], row["target"]
            w = _safe_float(row["weight"], 0.0)
            cs = comp_map.get(s)
            ct = comp_map.get(t)
            if cs and cs == ct:
                bond_counts[cs] = bond_counts.get(cs, 0) + 1
                wcollect.setdefault(cs, []).append(w)

    rows = []
    for cid, group in molecule_df.groupby("compound_id"):
        ne = group["element"].nunique()
        nb = int(bond_counts.get(cid, 0))

        mean_ls = float(group["local_stability"].mean())
        mean_temp = float(group["temperature"].mean())

        # regime averages
        info = float(group["info_score"].mean())
        mis = float(group["misinfo_score"].mean())
        dis = float(group["disinfo_score"].mean())

        # compound stability: mean of bond weights if available
        cstab = float(np.mean(wcollect[cid])) if cid in wcollect else mean_ls

        rows.append(
            {
                "compound_id": cid,
                "num_elements": ne,
                "num_bonds": nb,
                "compound_stability": cstab,
                "mean_local_stability": mean_ls,
                "mean_temperature": mean_temp,
                "mean_info": info,
                "mean_misinfo": mis,
                "mean_disinfo": dis,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["compound_stability", "num_elements"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


# =============================================================================
# Export Layer
# =============================================================================

def export_compounds(out_dir: str,
                     molecule_df: pd.DataFrame,
                     compound_df: pd.DataFrame,
                     emit=DEFAULT_EMIT) -> None:
    if compound_df.empty:
        _log("[molecule] No compounds to export.", emit)
        return

    os.makedirs(out_dir, exist_ok=True)

    elements_by_cid = {}
    if not molecule_df.empty:
        for cid, g in molecule_df.groupby("compound_id"):
            elements_by_cid[cid] = sorted(g["element"].astype(str).tolist())

    out = {}
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
        _log(f"[molecule] Exported {path}", emit)
        emit("artifact", {"path": path, "kind": "informational_compounds"})
    except Exception as e:
        _log(f"[molecule][warn] Failed to write {path}: {e}", emit)


# =============================================================================
# Pipeline Entry
# =============================================================================

def run_molecule_stage(results_dir: str,
                       emit=DEFAULT_EMIT) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main entry point for orchestrator."""
    el_path = os.path.join(results_dir, "hilbert_elements.csv")
    edges_path = os.path.join(results_dir, "edges.csv")

    elements_df = _load_elements(el_path, emit)
    edges = _load_edges(edges_path, elements_df, emit)
    regimes = _load_regimes(el_path, emit)

    molecule_df = build_molecule_df(elements_df, edges, regimes, emit)
    molecule_df = add_temperature(molecule_df, emit)

    compound_df = aggregate_compounds(molecule_df, edges, regimes, emit)

    export_compounds(results_dir, molecule_df, compound_df, emit)

    return molecule_df, compound_df

# -------------------------------------------------------------------------
# Backwards compatibility for orchestrator
# -------------------------------------------------------------------------

def compute_molecule_stability(edges_csv, elements_csv, emit=DEFAULT_EMIT):
    elements_df = _load_elements(elements_csv, emit)
    edges = _load_edges(edges_csv, elements_df, emit)
    regimes = _load_regimes(elements_csv, emit)
    df = build_molecule_df(elements_df, edges, regimes, emit)
    return df

def compute_molecule_temperature(molecule_df, elements_df):
    return add_temperature(molecule_df)

def export_molecule_summary(out_dir, molecule_df, compound_df):
    export_compounds(out_dir, molecule_df, compound_df)
