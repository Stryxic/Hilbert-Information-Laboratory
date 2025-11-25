"""
prep_graph.py - Edge scoring and intelligent pruning for Hilbert graphs.

This module is loaded BEFORE layouts and styling in the visualizer
pipeline. It enriches the graph with importance-weighted edges and applies
a configurable pruning and structural analysis policy to reduce visual
clutter.

Outputs:
    - edge importance scores added to G[u][v]["importance"]
    - optional edge pruning applied in-place
    - node-level structural tags for:
        * isolates
        * rare leaves
        * degree-2 chains
    - pruning_metadata.json written to results_dir
    - graph_structure_metadata.json written to results_dir
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Tuple, Any, Optional, Literal, List

import numpy as np
import networkx as nx


# --------------------------------------------------------------------------- #
# Utility normalisation
# --------------------------------------------------------------------------- #

def _safe_norm(arr: np.ndarray) -> np.ndarray:
    """Min–max normalisation with safe fallback."""
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _bool_int(x: bool) -> float:
    return 1.0 if x else 0.0


# --------------------------------------------------------------------------- #
# Edge scoring
# --------------------------------------------------------------------------- #

def score_edges(G: nx.Graph, *, emit=None) -> None:
    """
    Add composite semantic 'importance' scores to edges.

    The score includes:
        - normalised co-occurrence weight
        - shared compound membership
        - shared root membership
        - stability similarity
    """

    edges = list(G.edges())
    if not edges:
        return

    # Raw weights
    w_raw = np.array([float(G[u][v].get("weight", 1.0)) for u, v in edges])
    w_norm = _safe_norm(w_raw)

    # Shared compound
    shared_comp = np.array([
        1.0 if G.nodes[u].get("compound_id") == G.nodes[v].get("compound_id") else 0.0
        for u, v in edges
    ], dtype=float)

    # Shared root
    shared_root = np.array([
        1.0 if G.nodes[u].get("root_id") == G.nodes[v].get("root_id") else 0.0
        for u, v in edges
    ], dtype=float)

    # Stability similarity
    stab_u = np.array([float(G.nodes[u].get("stability_z", 0.0)) for u, v in edges])
    stab_v = np.array([float(G.nodes[v].get("stability_z", 0.0)) for u, v in edges])
    stab_sim = 1.0 - np.tanh(np.abs(stab_u - stab_v))

    # Composite score
    score = (
        0.50 * w_norm
        + 0.20 * shared_comp
        + 0.15 * shared_root
        + 0.15 * stab_sim
    )

    # Normalize final score to [0,1]
    score = _safe_norm(score)

    # Assign to graph
    for (u, v), s in zip(edges, score):
        G[u][v]["importance"] = float(s)

    if emit:
        emit("log", {"message": f"[prep_graph] Scored {len(edges)} edges"})


# --------------------------------------------------------------------------- #
# Structural simplification analysis (pre-layout)
# --------------------------------------------------------------------------- #

def _find_isolates_and_leaves(G: nx.Graph) -> Tuple[List[str], List[str]]:
    degrees = dict(G.degree())
    isolates = [n for n, d in degrees.items() if d == 0]
    leaves = [n for n, d in degrees.items() if d == 1]
    return isolates, leaves


def _estimate_rare_leaves(
    G: nx.Graph,
    leaves: List[str],
    rare_threshold: float = 0.05,
) -> List[str]:
    """
    Approximate "occur only once in corpus" by using the normalised doc_freq
    field written by loader. Nodes with doc_freq <= rare_threshold are treated
    as rare.
    """
    rare = []
    for n in leaves:
        df = float(G.nodes[n].get("doc_freq", 1.0))
        if df <= rare_threshold:
            rare.append(n)
    return rare


def _find_degree2_chains(G: nx.Graph) -> List[List[str]]:
    """
    Identify maximal simple chains of degree-2 nodes.

    These chains can be treated as polyline segments when considering
    pre-layout condensation or force tuning.

    Returned as a list of node lists, each representing a chain
    (internal nodes mostly degree-2).
    """
    degrees = dict(G.degree())
    deg2_nodes = {n for n, d in degrees.items() if d == 2}
    visited: set[str] = set()
    chains: List[List[str]] = []

    for start in list(deg2_nodes):
        if start in visited:
            continue

        if degrees.get(start, 0) != 2:
            continue

        # Grow in both directions
        neighbors = list(G.neighbors(start))
        if len(neighbors) != 2:
            continue

        chain = [start]
        visited.add(start)

        # extend in direction 0
        for dir_idx in (0, 1):
            current = neighbors[dir_idx]
            prev = start
            while current in deg2_nodes and current not in visited:
                chain.append(current)
                visited.add(current)
                neighs = list(G.neighbors(current))
                # move to the neighbor that is not prev
                next_nodes = [x for x in neighs if x != prev]
                if len(next_nodes) != 1:
                    break
                prev, current = current, next_nodes[0]

        # normalise order
        if len(chain) >= 2:
            chains.append(chain)

    return chains


def analyse_and_tag_structure(
    G: nx.Graph,
    *,
    remove_isolates: bool = True,
    rare_leaf_threshold: float = 0.05,
    emit=None,
) -> Dict[str, Any]:
    """
    Analyse graph structure for:
        - isolates (degree 0)
        - rare leaves (degree 1, low doc_freq)
        - degree-2 chains

    Tags nodes with attributes:
        - hilbert_isolate: bool
        - hilbert_leaf_contract: bool
        - hilbert_chain_id: int

    Optionally removes isolates from the working graph, since they
    do not affect layout or connectivity.

    Returns a metadata dict summarising the analysis.
    """
    isolates, leaves = _find_isolates_and_leaves(G)
    rare_leaves = _estimate_rare_leaves(G, leaves, rare_threshold=rare_leaf_threshold)
    chains = _find_degree2_chains(G)

    # Tag isolates
    for n in isolates:
        G.nodes[n]["hilbert_isolate"] = True

    # Tag rare leaves that are good candidates for contraction treatment
    for n in rare_leaves:
        G.nodes[n]["hilbert_leaf_contract"] = True

    # Tag chains
    for cid, chain_nodes in enumerate(chains, start=1):
        for n in chain_nodes:
            G.nodes[n]["hilbert_chain_id"] = int(cid)

    # Optionally remove isolates (they will not take part in layout)
    removed_isolates = []
    if remove_isolates and isolates:
        for n in isolates:
            if n in G:
                G.remove_node(n)
                removed_isolates.append(n)

    meta = {
        "n_isolates_detected": len(isolates),
        "n_isolates_removed": len(removed_isolates),
        "n_leaves_detected": len(leaves),
        "rare_leaf_threshold": float(rare_leaf_threshold),
        "n_rare_leaves_tagged": len(rare_leaves),
        "n_degree2_chains": len(chains),
        "chains_example": chains[:5],  # small sample for debugging
    }

    if emit:
        emit("log", {"message": f"[prep_graph] Structure: {meta}"})

    return meta


# --------------------------------------------------------------------------- #
# Edge pruning policies
# --------------------------------------------------------------------------- #

def _prune_by_median_strength(G: nx.Graph) -> Tuple[int, float]:
    """
    Remove edges with importance < median.
    Returns (n_removed, threshold).
    """
    edges = list(G.edges(data=True))
    if not edges:
        return (0, 0.0)

    scores = np.array([float(d.get("importance", 0.0)) for _, _, d in edges])
    thresh = float(np.median(scores))

    n_before = len(edges)
    for u, v, d in edges:
        if float(d.get("importance", 0.0)) < thresh:
            G.remove_edge(u, v)
    return (n_before - G.number_of_edges(), thresh)


def _prune_thermal(G: nx.Graph) -> Tuple[int, float]:
    """
    Prefer edges connecting stable nodes.
    Compute threshold = median importance of edges linking stable pairs.
    """
    edges = list(G.edges(data=True))
    if not edges:
        return (0, 0.0)

    stab_u = np.array([float(G.nodes[u].get("stability_z", 0.0)) for u, v, _ in edges])
    stab_v = np.array([float(G.nodes[v].get("stability_z", 0.0)) for u, v, _ in edges])
    mask_stable = (stab_u > 0.0) & (stab_v > 0.0)

    imp = np.array([float(d.get("importance", 0.0)) for _, _, d in edges])
    stable_scores = imp[mask_stable]
    if stable_scores.size == 0:
        thresh = float(np.median(imp))
    else:
        thresh = float(np.median(stable_scores))

    n_before = len(edges)
    for u, v, d in edges:
        if float(d.get("importance", 0.0)) < thresh:
            G.remove_edge(u, v)
    return (n_before - G.number_of_edges(), thresh)


def _prune_compound_rarity(G: nx.Graph) -> Tuple[int, float]:
    """
    Retain edges linking rare compounds.
    Threshold = weighted percentile where rare-compound edges receive priority.
    """
    edges = list(G.edges(data=True))
    if not edges:
        return (0, 0.0)

    # Count compound frequency
    comp_hist: Dict[str, int] = {}
    for n in G.nodes():
        cid = G.nodes[n].get("compound_id", "C?")
        comp_hist[cid] = comp_hist.get(cid, 0) + 1

    compound_sizes = {cid: comp_hist[cid] for cid in comp_hist}

    scores = []
    for u, v, d in edges:
        cid_u = G.nodes[u].get("compound_id", "C?")
        cid_v = G.nodes[v].get("compound_id", "C?")
        rarity = 1.0 / (math.sqrt(compound_sizes.get(cid_u, 10)) +
                        math.sqrt(compound_sizes.get(cid_v, 10)))
        s = float(d.get("importance", 0.0)) * rarity
        scores.append(s)

    scores = np.array(scores)
    thresh = float(np.percentile(scores, 40))  # keep top 60%

    n_before = len(edges)
    for (u, v, d), val in zip(edges, scores):
        if val < thresh:
            G.remove_edge(u, v)

    return (n_before - G.number_of_edges(), thresh)


def _prune_topk(G: nx.Graph, k: int) -> Tuple[int, float]:
    """
    Keep only top-k most important edges.
    """
    edges = list(G.edges(data=True))
    if not edges:
        return (0, 0.0)

    importance = np.array([float(d.get("importance", 0.0)) for _, _, d in edges])
    if importance.size <= k:
        return (0, 0.0)

    idx = np.argsort(importance)[::-1]  # descending
    keep_idx = set(idx[:k])
    thresh = float(importance[idx[k - 1]])

    for i, (u, v, _) in enumerate(edges):
        if i not in keep_idx:
            G.remove_edge(u, v)

    return (len(edges) - G.number_of_edges(), thresh)


# --------------------------------------------------------------------------- #
# Public pruning entry point
# --------------------------------------------------------------------------- #

def prune_edges(
    G: nx.Graph,
    *,
    mode: Literal["none", "median_strength", "compound_rarity", "thermal", "topk_importance"] = "median_strength",
    topk: int = 2000,
    emit=None,
) -> Dict[str, Any]:
    """
    Apply pruning mode to G.

    Returns a small dict with:
        {
            "mode": ...,
            "removed": ...,
            "threshold": ...
        }
    """

    if mode == "none":
        return {"mode": "none", "removed": 0, "threshold": None}

    if emit:
        emit("log", {"message": f"[prep_graph] Pruning mode: {mode}"})

    if mode == "median_strength":
        removed, thresh = _prune_by_median_strength(G)
    elif mode == "thermal":
        removed, thresh = _prune_thermal(G)
    elif mode == "compound_rarity":
        removed, thresh = _prune_compound_rarity(G)
    elif mode == "topk_importance":
        removed, thresh = _prune_topk(G, k=topk)
    else:
        removed, thresh = 0, None

    return {"mode": mode, "removed": int(removed), "threshold": thresh}


# --------------------------------------------------------------------------- #
# Metadata export
# --------------------------------------------------------------------------- #

def write_pruning_metadata(results_dir: str, meta: Dict[str, Any]) -> str:
    """Write pruning metadata to results_dir/pruning_metadata.json"""
    path = os.path.join(results_dir, "pruning_metadata.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass
    return path


def write_structure_metadata(results_dir: str, meta: Dict[str, Any]) -> str:
    """Write structural analysis metadata to results_dir/graph_structure_metadata.json"""
    path = os.path.join(results_dir, "graph_structure_metadata.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass
    return path


# --------------------------------------------------------------------------- #
# High-level orchestration used by visualizer
# --------------------------------------------------------------------------- #

def prepare_graph_for_visualization(
    G: nx.Graph,
    results_dir: str,
    *,
    pruning_mode: str = "median_strength",
    topk: int = 2000,
    remove_isolates: bool = True,
    rare_leaf_threshold: float = 0.05,
    emit=None,
) -> nx.Graph:
    """
    Complete preprocessing step:
        1. Analyse and tag structural simplifications
           (isolates, rare leaves, degree-2 chains)
        2. Score edges
        3. Apply pruning
        4. Write structure and pruning metadata

    Returns the modified graph (in-place).
    """

    # 1. Structural analysis and soft condensation tags
    struct_meta = analyse_and_tag_structure(
        G,
        remove_isolates=remove_isolates,
        rare_leaf_threshold=rare_leaf_threshold,
        emit=emit,
    )
    struct_path = write_structure_metadata(results_dir, struct_meta)
    if emit:
        emit("artifact", {"kind": "graph-structure-metadata", "path": struct_path})

    # 2. Edge scoring
    score_edges(G, emit=emit)

    # 3. Edge pruning
    prune_meta = prune_edges(
        G,
        mode=pruning_mode,
        topk=topk,
        emit=emit,
    )
    prune_path = write_pruning_metadata(results_dir, prune_meta)
    if emit:
        emit("artifact", {"kind": "pruning-metadata", "path": prune_path})

    return G
