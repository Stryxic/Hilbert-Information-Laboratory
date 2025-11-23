"""
Data loading and basic graph construction for Hilbert Graph Engine.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Callable, List

import pandas as pd
import numpy as np
import networkx as nx


# ============================================================================ #
# Data structures
# ============================================================================ #

@dataclass
class LoadedData:
    elements: pd.DataFrame
    edges: pd.DataFrame
    compounds: List[Dict[str, Any]]
    root_map: Dict[str, str]
    compound_temps: Dict[str, float]


# ============================================================================ #
# Logging helpers
# ============================================================================ #

def _log(msg: str, emit: Callable[[str, Dict[str, Any]], None] | None) -> None:
    print(msg)
    if emit is None:
        return
    try:
        emit("log", {"message": msg})
    except Exception:
        pass


def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


# ============================================================================ #
# Loading: elements + edges
# ============================================================================ #

def load_elements_and_edges(results_dir: str, emit):
    """
    Load hilbert_elements.csv and edges.csv safely.

    Normalises:
      • element id → str
      • entropy / coherence / tf / df → float
      • edges weight → float
      • drops bad rows and self-loops
    """
    el_path = os.path.join(results_dir, "hilbert_elements.csv")
    ed_path = os.path.join(results_dir, "edges.csv")

    if not os.path.exists(el_path):
        _log("[graphs] hilbert_elements.csv missing", emit)
        return pd.DataFrame(), pd.DataFrame()
    if not os.path.exists(ed_path):
        _log("[graphs] edges.csv missing", emit)
        return pd.DataFrame(), pd.DataFrame()

    try:
        elements = pd.read_csv(el_path)
        edges = pd.read_csv(ed_path)
    except Exception as exc:
        _log(f"[graphs] Failed reading CSVs: {exc}", emit)
        return pd.DataFrame(), pd.DataFrame()

    # ----------------------------- Elements ---------------------------------- #
    if "element" not in elements.columns:
        # Backward compatibility with "token"
        if "token" in elements.columns:
            elements["element"] = elements["token"].astype(str)
        else:
            _log("[graphs] Missing 'element' column in hilbert_elements.csv", emit)
            return pd.DataFrame(), pd.DataFrame()

    elements["element"] = elements["element"].astype(str)

    # known numeric fields
    numeric_cols = [
        "mean_entropy", "entropy",
        "mean_coherence", "coherence",
        "tf", "df", "doc_freq"
    ]
    for c in numeric_cols:
        if c in elements.columns:
            elements[c] = pd.to_numeric(elements[c], errors="coerce").fillna(0.0)

    # unify df/doc_freq
    if "doc_freq" not in elements.columns:
        if "df" in elements.columns:
            elements["doc_freq"] = elements["df"]
        else:
            elements["doc_freq"] = 1.0

    # Drop any row lacking an element id
    elements = elements.dropna(subset=["element"]).reset_index(drop=True)

    # ----------------------------- Edges ------------------------------------- #
    if "source" not in edges.columns or "target" not in edges.columns:
        _log("[graphs] edges.csv missing 'source' or 'target' columns", emit)
        return elements, pd.DataFrame()

    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)

    # Remove self-loops
    edges = edges[edges["source"] != edges["target"]]

    # weight
    if "weight" not in edges.columns:
        edges["weight"] = 1.0
    edges["weight"] = edges["weight"].apply(_safe_float)

    edges = edges.dropna(subset=["source", "target"]).reset_index(drop=True)

    return elements, edges


# ============================================================================ #
# Loading: compounds and root map
# ============================================================================ #

def load_compound_data(results_dir: str, emit):
    """
    Load:
      - informational_compounds.json   → list of compounds
      - element_roots.csv              → element → root
      - compound temperatures          → compound_id → temp
    """
    emit_fn = lambda msg: _log(msg, emit)

    # ---------------- informational_compounds.json --------------------------- #
    compounds: List[Dict[str, Any]] = []
    temps: Dict[str, float] = {}

    comp_path = os.path.join(results_dir, "informational_compounds.json")
    if os.path.exists(comp_path):
        try:
            with open(comp_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            if isinstance(raw, list):
                compounds = raw
            elif isinstance(raw, dict):
                # Might be {id: compound}
                compounds = list(raw.values())
            else:
                compounds = []

            emit_fn(f"[graphs] Loaded {len(compounds)} compounds")

        except Exception as exc:
            emit_fn(f"[graphs] Failed to read compounds JSON: {exc}")
            compounds = []
    else:
        emit_fn("[graphs] No informational_compounds.json found (optional)")

    # temperature extraction
    for comp in compounds:
        cid = str(
            comp.get("compound_id")
            or comp.get("id")
            or ""
        )
        t = comp.get("temperature")
        if cid and t is not None:
            temps[cid] = _safe_float(t, 0.5)

    # ---------------- element_roots.csv -------------------------------------- #
    roots_path = os.path.join(results_dir, "element_roots.csv")
    root_map: Dict[str, str] = {}

    if os.path.exists(roots_path):
        try:
            df = pd.read_csv(roots_path)
            if "element" in df.columns and "root_element" in df.columns:
                for _, row in df.iterrows():
                    e = str(row["element"])
                    r = str(row["root_element"])
                    if e and r:
                        root_map[e] = r
                emit_fn(f"[graphs] Loaded {len(root_map)} element→root mappings")
            else:
                emit_fn("[graphs] element_roots.csv missing required columns")
        except Exception as exc:
            emit_fn(f"[graphs] Failed to read element_roots.csv: {exc}")
    else:
        emit_fn("[graphs] No element_roots.csv found (optional)")

    return compounds, root_map, temps


# ============================================================================ #
# Graph construction
# ============================================================================ #

def build_graph(elements: pd.DataFrame, edges: pd.DataFrame, emit) -> nx.Graph:
    """
    Build the base element graph.

    Node attributes:
        entropy, coherence, tf, doc_freq

    Edge attributes:
        weight
    """
    emit_fn = lambda msg: _log(msg, emit)

    G = nx.Graph()

    if elements.empty:
        emit_fn("[graphs] Empty elements table – graph will be empty")
        return G

    # ------------------------- Add nodes ------------------------------------ #
    for _, row in elements.iterrows():
        el = str(row["element"])
        ent = _safe_float(
            row.get("mean_entropy", row.get("entropy", 0.0)), default=0.0
        )
        coh = _safe_float(
            row.get("mean_coherence", row.get("coherence", 0.0)), default=0.0
        )
        tf = _safe_float(row.get("tf", 1.0), default=1.0)
        df = _safe_float(row.get("doc_freq", 1.0), default=1.0)

        G.add_node(el,
                   entropy=ent,
                   coherence=coh,
                   tf=tf,
                   doc_freq=df)

    # ------------------------- Add edges ------------------------------------ #
    if edges.empty:
        emit_fn("[graphs] Warning: edges.csv empty – graph will have no edges")

    for _, row in edges.iterrows():
        s = str(row["source"])
        t = str(row["target"])
        if s not in G or t not in G:
            continue

        w = _safe_float(row.get("weight", 1.0), default=1.0)
        if w > 0:
            G.add_edge(s, t, weight=w)

    emit_fn(
        f"[graphs] Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    return G


# ============================================================================ #
# Unified loader
# ============================================================================ #

def load_all_data(results_dir: str, emit=None) -> LoadedData:
    """
    Unified helper returning all required data for visualisation:

      • elements (DataFrame)
      • edges (DataFrame)
      • compounds (list of dicts)
      • root_map (dict)
      • compound_temps (dict)
    """
    elements, edges = load_elements_and_edges(results_dir, emit)
    compounds, root_map, temps = load_compound_data(results_dir, emit)

    return LoadedData(
        elements=elements,
        edges=edges,
        compounds=compounds,
        root_map=root_map,
        compound_temps=temps,
    )
