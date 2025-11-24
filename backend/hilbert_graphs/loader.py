"""
Enhanced data loader for Hilbert graphs.

Responsibilities:
  - Read and normalise element and edge tables
  - Read compounds and element roots
  - Construct a NetworkX graph with rich node and edge attributes
  - Apply pre-normalisation for entropy, coherence, stability, tf, doc_freq
  - Attach stability_z, temperature, centralities
  - Emit a graph_loading_report.json for diagnostics
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
# Data structure returned by load_all_data()
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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _safe_norm(series: pd.Series) -> pd.Series:
    """Min-max normalisation with safe fallback."""
    if series.empty:
        return series
    lo = float(series.min())
    hi = float(series.max())
    if hi - lo < 1e-9:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - lo) / (hi - lo)


def _safe_z(series: pd.Series) -> pd.Series:
    """Z-score with robust fallback."""
    if series.empty:
        return series
    mean = float(series.mean())
    std = float(series.std())
    if std < 1e-9:
        return pd.Series([0.0] * len(series), index=series.index)
    z = (series - mean) / std
    return np.clip(z, -4.0, 4.0)


# ============================================================================ #
# 1. Load elements and edges
# ============================================================================ #

def load_elements_and_edges(results_dir: str, emit=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load hilbert_elements.csv and edges.csv from results_dir.

    Enforces:
      - string element IDs
      - numeric entropy/coherence/tf/doc_freq
      - safe removal of NaNs
      - edges with source, target, weight
    """
    el_path = os.path.join(results_dir, "hilbert_elements.csv")
    ed_path = os.path.join(results_dir, "edges.csv")

    if not os.path.exists(el_path):
        _log("[loader] hilbert_elements.csv missing", emit)
        return pd.DataFrame(), pd.DataFrame()

    if not os.path.exists(ed_path):
        _log("[loader] edges.csv missing", emit)
        return pd.DataFrame(), pd.DataFrame()

    try:
        elements = pd.read_csv(el_path)
        edges = pd.read_csv(ed_path)
    except Exception as exc:
        _log(f"[loader] Failed reading CSVs: {exc}", emit)
        return pd.DataFrame(), pd.DataFrame()

    # Ensure element IDs exist
    if "element" not in elements.columns:
        if "token" in elements.columns:
            elements["element"] = elements["token"].astype(str)
        else:
            _log("[loader] Missing 'element' column", emit)
            return pd.DataFrame(), pd.DataFrame()

    elements["element"] = elements["element"].astype(str)

    # Clean numeric fields
    num_cols = ["entropy", "mean_entropy", "coherence", "mean_coherence", "tf", "df", "doc_freq"]
    for col in num_cols:
        if col in elements.columns:
            elements[col] = pd.to_numeric(elements[col], errors="coerce")

    # Standardise doc_freq
    if "doc_freq" not in elements.columns:
        elements["doc_freq"] = elements.get("df", pd.Series([1.0] * len(elements)))

    elements = elements.fillna(0.0)

    # Edges
    if "source" not in edges.columns or "target" not in edges.columns:
        _log("[loader] edges.csv missing source/target columns", emit)
        return elements, pd.DataFrame()

    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)
    edges = edges[edges["source"] != edges["target"]]

    if "weight" not in edges.columns:
        edges["weight"] = 1.0
    edges["weight"] = edges["weight"].apply(_safe_float)

    edges = edges.dropna(subset=["source", "target"]).reset_index(drop=True)

    return elements, edges


# ============================================================================ #
# 2. Load compounds and root mappings
# ============================================================================ #

def load_compound_data(results_dir: str, emit=None):
    """
    Load:
      - informational_compounds.json
      - element_roots.csv
      - compound temperatures (per compound)
    """
    compounds: List[Dict[str, Any]] = []
    root_map: Dict[str, str] = {}
    temps: Dict[str, float] = {}

    comp_path = os.path.join(results_dir, "informational_compounds.json")
    if os.path.exists(comp_path):
        try:
            raw = json.load(open(comp_path, "r", encoding="utf-8"))
            if isinstance(raw, list):
                compounds = raw
            elif isinstance(raw, dict):
                compounds = list(raw.values())

            _log(f"[loader] Loaded {len(compounds)} compounds", emit)
        except Exception as exc:
            _log(f"[loader] Failed to read compounds JSON: {exc}", emit)
            compounds = []
    else:
        _log("[loader] No informational_compounds.json found (optional)", emit)

    # Compound temperature extraction
    for comp in compounds:
        cid = str(comp.get("compound_id") or comp.get("id") or "")
        t = comp.get("temperature")
        if cid and t is not None:
            temps[cid] = _safe_float(t, 0.5)

    # Element roots
    root_path = os.path.join(results_dir, "element_roots.csv")
    if os.path.exists(root_path):
        try:
            df = pd.read_csv(root_path)
            if "element" in df.columns and "root_element" in df.columns:
                for _, r in df.iterrows():
                    root_map[str(r["element"])] = str(r["root_element"])
                _log(f"[loader] Loaded {len(root_map)} element→root mappings", emit)
        except Exception as exc:
            _log(f"[loader] Failed to read element_roots.csv: {exc}", emit)
    else:
        _log("[loader] No element_roots.csv found (optional)", emit)

    return compounds, root_map, temps


# ============================================================================ #
# 3. Graph construction with pre-normalisation
# ============================================================================ #

def _attach_centralities(G: nx.Graph) -> None:
    """Add degree, eigenvector centrality, betweenness centrality."""
    if G.number_of_nodes() == 0:
        return

    deg = dict(G.degree())
    nx.set_node_attributes(G, deg, "degree")

    try:
        ev = nx.eigenvector_centrality_numpy(G)
        nx.set_node_attributes(G, ev, "eigencentrality")
    except Exception:
        nx.set_node_attributes(G, {n: 0.0 for n in G}, "eigencentrality")

    try:
        bt = nx.betweenness_centrality(G, k=min(200, G.number_of_nodes()))
        nx.set_node_attributes(G, bt, "betweenness")
    except Exception:
        nx.set_node_attributes(G, {n: 0.0 for n in G}, "betweenness")


def build_graph(elements: pd.DataFrame, edges: pd.DataFrame, emit=None) -> nx.Graph:
    """
    Build enriched NetworkX graph including:
        - normalized entropy, coherence
        - stability_z if available
        - tf, doc_freq, centralities
        - compound_id, root_id (attached later)
    """
    G = nx.Graph()

    if elements is None or elements.empty:
        _log("[loader] Empty elements table - graph empty", emit)
        return G

    # Pre-normalise numeric attributes
    ent_raw = elements.get("entropy", elements.get("mean_entropy", pd.Series([0]*len(elements))))
    coh_raw = elements.get("coherence", elements.get("mean_coherence", pd.Series([0]*len(elements))))
    tf_raw = elements.get("tf", pd.Series([1.0] * len(elements)))
    df_raw = elements.get("doc_freq", pd.Series([1.0] * len(elements)))

    elements["entropy_norm"] = _safe_norm(ent_raw)
    elements["coherence_norm"] = _safe_norm(coh_raw)
    elements["tf_norm"] = _safe_norm(tf_raw)
    elements["df_norm"] = _safe_norm(df_raw)

    # Stability
    if "stability" in elements.columns:
        elements["stability_z"] = _safe_z(elements["stability"])
    else:
        elements["stability_z"] = 0.0

    # Construct nodes
    for _, row in elements.iterrows():
        el = str(row["element"])
        G.add_node(
            el,
            entropy=float(row.get("entropy_norm", 0.0)),
            coherence=float(row.get("coherence_norm", 0.0)),
            tf=float(row.get("tf_norm", 1.0)),
            doc_freq=float(row.get("df_norm", 1.0)),
            stability_z=float(row.get("stability_z", 0.0)),
            raw_entropy=float(row.get("entropy", 0.0)),
            raw_coherence=float(row.get("coherence", 0.0)),
        )

    # Add edges
    if edges is not None and not edges.empty:
        for _, r in edges.iterrows():
            s = str(r["source"])
            t = str(r["target"])
            if s in G and t in G:
                w = _safe_float(r.get("weight", 1.0))
                if w > 0:
                    G.add_edge(s, t, weight=w)
    else:
        _log("[loader] Edge table empty", emit)

    # Centrality metrics
    _attach_centralities(G)

    _log(f"[loader] Built enriched graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", emit)
    return G


# ============================================================================ #
# 4. Diagnostics: graph_loading_report.json
# ============================================================================ #

def write_graph_loading_report(results_dir: str, elements: pd.DataFrame, edges: pd.DataFrame, emit=None):
    """Creates graph_loading_report.json with summary statistics and missing-data checks."""
    report = {
        "n_elements": len(elements),
        "n_edges": len(edges),
        "missing_entropy": int(elements["entropy"].isna().sum()) if "entropy" in elements else None,
        "missing_coherence": int(elements["coherence"].isna().sum()) if "coherence" in elements else None,
        "missing_tf": int(elements["tf"].isna().sum()) if "tf" in elements else None,
        "missing_doc_freq": int(elements["doc_freq"].isna().sum()) if "doc_freq" in elements else None,
        "edge_weight_min": float(edges["weight"].min()) if "weight" in edges.columns and not edges.empty else None,
        "edge_weight_max": float(edges["weight"].max()) if "weight" in edges.columns and not edges.empty else None,
        "report_version": "hilbert.loader.v2",
    }

    path = os.path.join(results_dir, "graph_loading_report.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        if emit:
            emit("artifact", {"kind": "graph-loading-report", "path": path})
    except Exception:
        pass

    return path


# ============================================================================ #
# 5. Unified loading convenience
# ============================================================================ #

def load_all_data(results_dir: str, emit=None) -> LoadedData:
    """
    Loads elements, edges, compounds, root_map, temperatures.
    Writes graph_loading_report.json.
    """
    elements, edges = load_elements_and_edges(results_dir, emit)
    compounds, root_map, temps = load_compound_data(results_dir, emit)

    # Save report
    write_graph_loading_report(results_dir, elements, edges, emit)

    return LoadedData(
        elements=elements,
        edges=edges,
        compounds=compounds,
        root_map=root_map,
        compound_temps=temps,
    )
