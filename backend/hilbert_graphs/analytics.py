"""
Analytic layers: communities, clusters, stats, diagnostics, and node features.

This module expands the Hilbert analytics layer from simple clustering
to a richer scientific diagnostic engine that produces high-quality
information for layout, pruning, rendering, and reporting.

Enhancements include:
  - multi-scale structure detection
  - compound + root + community + component clustering
  - spectral curvature estimate
  - stability / entropy / coherence banding
  - compound diameter metrics
  - community diameter metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Callable, Iterable, List
from collections import Counter, defaultdict

import networkx as nx
import numpy as np

# Optional: Louvain community detection (python-louvain)
try:
    import community as community_louvain  # type: ignore
    _HAS_LOUVAIN = True
except Exception:
    _HAS_LOUVAIN = False


# =========================================================================== #
# Data classes
# =========================================================================== #

@dataclass
class ClusterInfo:
    """
    Hierarchical cluster descriptors for each node.

    These clusters inform:
      - layout (sector placement, shells)
      - styling (hue channels)
      - metadata
      - pruning modes (compound-centric or community-centric)

    Additional fields are included for scientific reporting:
      - compound diameters
      - community diameters
    """

    cluster_ids: Dict[str, str]
    compound_ids: Dict[str, str]
    root_ids: Dict[str, str]
    community_ids: Dict[str, str]
    component_ids: Dict[str, str]

    community_sizes: Dict[str, int]
    component_sizes: Dict[str, int]

    # New scientific diagnostics
    compound_diameters: Dict[str, float]
    community_diameters: Dict[str, float]
    spectral_gap: float
    stability_bands: Dict[str, List[str]]
    entropy_bands: Dict[str, List[str]]
    coherence_bands: Dict[str, List[str]]


@dataclass
class GraphStats:
    n_nodes: int
    n_edges: int
    density: float
    avg_degree: float
    transitivity: float


# =========================================================================== #
# Logging helpers
# =========================================================================== #

def _get_emit(emit: Callable[[str, Dict[str, Any]], None] | None):
    if emit:
        return emit
    return lambda *_args, **_kwargs: None


def _log(msg: str, emit):
    print(msg)
    try:
        _get_emit(emit)("log", {"message": msg})
    except Exception:
        pass


# =========================================================================== #
# Community detection
# =========================================================================== #

def _detect_communities(G: nx.Graph, emit) -> Dict[str, str]:
    """
    Detect structural communities via Louvain or greedy modularity.
    Returns node -> "Qxxx".
    """
    emitf = _get_emit(emit)

    if G.number_of_nodes() == 0:
        return {}

    # Louvain (best)
    if _HAS_LOUVAIN:
        try:
            part = community_louvain.best_partition(G)
            comm_map = {str(n): f"Q{cid:03d}" for n, cid in part.items()}
            emitf(
                "log",
                {"message": f"[analytics] Louvain communities: {len(set(part.values()))} clusters."},
            )
            return comm_map
        except Exception as exc:
            _log(f"[analytics] Louvain failed, fallback to greedy. ({exc})", emit)

    # Greedy modularity fallback
    try:
        from networkx.algorithms import community as nx_comm
        comms = list(nx_comm.greedy_modularity_communities(G))
        comm_map = {}
        for idx, com in enumerate(comms, start=1):
            cid = f"Q{idx:03d}"
            for n in com:
                comm_map[str(n)] = cid
        emitf("log", {"message": f"[analytics] Greedy communities: {len(comms)} clusters."})
        return comm_map
    except Exception as exc:
        _log(f"[analytics] Community detection failed: {exc}", emit)
        return {}


# =========================================================================== #
# Scientific diagnostics
# =========================================================================== #

def _compute_spectral_gap(G: nx.Graph) -> float:
    """
    Estimate spectral gap λ2 - λ1 from the Laplacian.
    Gives a curvature-like measure for graph rigidity / fragmentation.
    """
    if G.number_of_nodes() < 3:
        return 0.0

    try:
        L = nx.normalized_laplacian_matrix(G).astype(float).todense()
        ev = np.linalg.eigvalsh(L)
        evs = np.sort(np.real(ev))
        if len(evs) >= 2:
            return float(evs[1] - evs[0])
        return 0.0
    except Exception:
        return 0.0


def _band_by_quantiles(values: Dict[str, float], n_bands: int = 4) -> Dict[str, List[str]]:
    """
    Assign each node to a band based on quantiles of a scalar attribute.
    Good for stability, entropy, or coherence layering.
    """
    if not values:
        return {}

    nodes = list(values.keys())
    arr = np.array([values[n] for n in nodes], float)

    qs = np.quantile(arr, np.linspace(0, 1, n_bands + 1))
    bands = defaultdict(list)

    for n, v in zip(nodes, arr):
        # Find which interval v fits in
        for i in range(n_bands):
            if qs[i] <= v <= qs[i + 1]:
                bands[f"B{i+1}"].append(n)
                break

    return dict(bands)


def _compute_group_diameters(G: nx.Graph, groups: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Compute approximate eccentricity-based diameter per group.

    For efficiency, we compute:
      diameter ≈ max shortest-path distance among 20 sample nodes.
    """
    diam = {}
    for gid, members in groups.items():
        if len(members) < 2:
            diam[gid] = 0.0
            continue

        sample = members if len(members) <= 20 else np.random.choice(members, 20, replace=False)
        max_dist = 0.0

        for u in sample:
            try:
                dist_map = nx.single_source_shortest_path_length(G, u)
            except Exception:
                continue
            for v in sample:
                if v in dist_map:
                    max_dist = max(max_dist, dist_map[v])

        diam[gid] = float(max_dist)
    return diam


# =========================================================================== #
# Main analytic function
# =========================================================================== #

def compute_cluster_info(
    G: nx.Graph,
    root_map: Dict[str, str],
    compound_map: Dict[str, str],
    emit,
) -> ClusterInfo:
    """
    Compute the full hierarchical clustering & diagnostics package.
    Priority of cluster_ids:
      1) compound
      2) root
      3) community
      4) connected component
    """

    emitf = _get_emit(emit)

    if G.number_of_nodes() == 0:
        return ClusterInfo(
            cluster_ids={},
            compound_ids={},
            root_ids={},
            community_ids={},
            component_ids={},
            community_sizes={},
            component_sizes={},
            compound_diameters={},
            community_diameters={},
            spectral_gap=0.0,
            stability_bands={},
            entropy_bands={},
            coherence_bands={},
        )

    nodes = [str(n) for n in G.nodes()]

    # ------------------------------------------------------------------ #
    # Compound and root assignment
    # ------------------------------------------------------------------ #

    compound_ids = {}
    for n in nodes:
        if n in compound_map:
            compound_ids[n] = str(compound_map[n])

    root_ids = {}
    for n in nodes:
        if n in root_map:
            root_ids[n] = str(root_map[n])

    # ------------------------------------------------------------------ #
    # Community detection
    # ------------------------------------------------------------------ #

    community_ids = _detect_communities(G, emit)

    # ------------------------------------------------------------------ #
    # Connected components
    # ------------------------------------------------------------------ #

    component_ids: Dict[str, str] = {}
    for idx, comp in enumerate(nx.connected_components(G), start=1):
        cid = f"K{idx:03d}"
        for n in comp:
            component_ids[str(n)] = cid

    _log(f"[analytics] Connected components: {len(set(component_ids.values()))}", emit)

    # ------------------------------------------------------------------ #
    # Final composed cluster ID
    # ------------------------------------------------------------------ #

    cluster_ids = {}
    for n in nodes:
        if n in compound_ids:
            cluster_ids[n] = compound_ids[n]
        elif n in root_ids:
            cluster_ids[n] = root_ids[n]
        elif n in community_ids:
            cluster_ids[n] = community_ids[n]
        else:
            cluster_ids[n] = component_ids.get(n, "K000")

    # ------------------------------------------------------------------ #
    # Size tables
    # ------------------------------------------------------------------ #

    community_sizes = {cid: sz for cid, sz in Counter(community_ids.values()).items()}
    component_sizes = {cid: sz for cid, sz in Counter(component_ids.values()).items()}

    # Log top 5 communities
    if community_sizes:
        top = sorted(community_sizes.items(), key=lambda kv: kv[1], reverse=True)[:5]
        emitf(
            "log",
            {"message": "[analytics] Top communities",
             "top_communities": [{"community": c, "size": s} for c, s in top]},
        )

    # ------------------------------------------------------------------ #
    # Scientific metrics
    # ------------------------------------------------------------------ #

    # spectral curvature estimate
    spectral_gap = _compute_spectral_gap(G)

    # Bands for stability, entropy, coherence
    stability_vals = {n: float(G.nodes[n].get("stability_z", 0.0)) for n in nodes}
    entropy_vals = {n: float(G.nodes[n].get("entropy", 0.0)) for n in nodes}
    coherence_vals = {n: float(G.nodes[n].get("coherence", 0.0)) for n in nodes}

    stability_bands = _band_by_quantiles(stability_vals, n_bands=4)
    entropy_bands = _band_by_quantiles(entropy_vals, n_bands=4)
    coherence_bands = _band_by_quantiles(coherence_vals, n_bands=4)

    # Diameters
    comp_groups = defaultdict(list)
    for n, cid in compound_ids.items():
        comp_groups[cid].append(n)
    compound_diameters = _compute_group_diameters(G, comp_groups)

    comm_groups = defaultdict(list)
    for n, cid in community_ids.items():
        comm_groups[cid].append(n)
    community_diameters = _compute_group_diameters(G, comm_groups)

    emitf(
        "log",
        {"message": "[analytics] Cluster assignment complete",
         "n_nodes": len(nodes),
         "n_compounds": len(comp_groups),
         "n_roots": len(root_map),
         "n_communities": len(comm_groups),
         "spectral_gap": spectral_gap},
    )

    return ClusterInfo(
        cluster_ids=cluster_ids,
        compound_ids=compound_ids,
        root_ids=root_ids,
        community_ids=community_ids,
        component_ids=component_ids,
        community_sizes=community_sizes,
        component_sizes=component_sizes,
        compound_diameters=compound_diameters,
        community_diameters=community_diameters,
        spectral_gap=spectral_gap,
        stability_bands=stability_bands,
        entropy_bands=entropy_bands,
        coherence_bands=coherence_bands,
    )


# =========================================================================== #
# Graph statistics
# =========================================================================== #

def compute_graph_stats(G: nx.Graph) -> GraphStats:
    if G.number_of_nodes() == 0:
        return GraphStats(0, 0, 0.0, 0.0, 0.0)

    n = G.number_of_nodes()
    e = G.number_of_edges()

    density = float(nx.density(G)) if n > 1 else 0.0
    avg_degree = float(np.mean([d for _, d in G.degree()]))
    try:
        transitivity = float(nx.transitivity(G)) if n > 2 else 0.0
    except Exception:
        transitivity = 0.0

    return GraphStats(
        n_nodes=n,
        n_edges=e,
        density=density,
        avg_degree=avg_degree,
        transitivity=transitivity,
    )


# =========================================================================== #
# Component filtering (unchanged)
# =========================================================================== #

def filter_large_components(G: nx.Graph, min_size: int, emit) -> nx.Graph:
    """
    Keep only components >= min_size.
    """
    emitf = _get_emit(emit)

    if G.number_of_nodes() == 0 or min_size <= 1:
        return G.copy()

    kept = []
    for comp in nx.connected_components(G):
        if len(comp) >= min_size:
            kept.extend(list(comp))

    if not kept:
        _log(f"[analytics] No components >= {min_size}, keeping full graph.", emit)
        return G.copy()

    H = G.subgraph(kept).copy()

    emitf(
        "log",
        {"message": "[analytics] Filtered to large components",
         "n_nodes": H.number_of_nodes(),
         "n_edges": H.number_of_edges(),
         "min_size": min_size},
    )

    return H
