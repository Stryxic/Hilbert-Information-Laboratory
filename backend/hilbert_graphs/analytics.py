"""
Analytic layers: communities, clusters, stats, and node features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Callable
from collections import Counter

import networkx as nx
import numpy as np

# Optional: Louvain community detection (python-louvain)
try:  # pragma: no cover - optional dependency
    import community as community_louvain  # type: ignore

    _HAS_LOUVAIN = True
except Exception:  # pragma: no cover
    _HAS_LOUVAIN = False


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass
class ClusterInfo:
    """
    Hierarchical cluster descriptors for each node.

    - cluster_ids:
        Final "priority composed" cluster id per node.
    - compound_ids:
        Node -> compound_id (from informational_compounds.json).
    - root_ids:
        Node -> root_element (from element_roots.csv).
    - community_ids:
        Node -> structural community id.
    - component_ids:
        Node -> connected component id.
    - community_sizes:
        Community id -> node count.
    - component_sizes:
        Component id -> node count.
    """

    cluster_ids: Dict[str, str]
    compound_ids: Dict[str, str]
    root_ids: Dict[str, str]
    community_ids: Dict[str, str]
    component_ids: Dict[str, str]
    community_sizes: Dict[str, int]
    component_sizes: Dict[str, int]


@dataclass
class GraphStats:
    n_nodes: int
    n_edges: int
    density: float
    avg_degree: float
    transitivity: float


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _get_emit(emit: Callable[[str, Dict[str, Any]], None] | None):
    if emit is None:
        return lambda *_args, **_kwargs: None
    return emit


def _log(msg: str, emit: Callable[[str, Dict[str, Any]], None] | None) -> None:
    print(msg)
    try:
        _get_emit(emit)("log", {"message": msg})
    except Exception:
        pass


def _detect_communities(
    G: nx.Graph,
    emit: Callable[[str, Dict[str, Any]], None] | None,
) -> Dict[str, str]:
    """
    Detect structural communities via Louvain or greedy modularity.
    Returns node -> "QXXX" community label.
    """
    emit_fn = _get_emit(emit)

    if G.number_of_nodes() == 0:
        return {}

    # 1) Louvain if available
    if _HAS_LOUVAIN:
        try:
            part = community_louvain.best_partition(G)
            comm_map = {str(n): f"Q{cid:03d}" for n, cid in part.items()}
            emit_fn(
                "log",
                {"message": f"[graphs][analytics] Louvain communities: "
                            f"{len(set(part.values()))} clusters."},
            )
            return comm_map
        except Exception as exc:
            _log(
                f"[graphs][analytics] Louvain failed → fallback. ({exc})",
                emit,
            )

    # 2) Greedy modularity
    try:
        from networkx.algorithms import community as nx_comm
        comms = list(nx_comm.greedy_modularity_communities(G))
        comm_map = {}
        for idx, com in enumerate(comms, start=1):
            cid = f"Q{idx:03d}"
            for n in com:
                comm_map[str(n)] = cid
        emit_fn(
            "log",
            {"message": f"[graphs][analytics] Greedy communities: {len(comms)} clusters."},
        )
        return comm_map
    except Exception as exc:
        _log(f"[graphs][analytics] Community detection failed: {exc}", emit)
        return {}


# --------------------------------------------------------------------------- #
# Public analytics
# --------------------------------------------------------------------------- #

def compute_cluster_info(
    G: nx.Graph,
    root_map: Dict[str, str],
    compound_map: Dict[str, str],
    emit,
) -> ClusterInfo:
    """
    Compute hierarchical clustering metadata for the visualisation engine.
    Priority order:
      1) compound id
      2) root element
      3) community (Louvain / greedy)
      4) connected component
    """

    emit_fn = _get_emit(emit)

    if G.number_of_nodes() == 0:
        return ClusterInfo(
            cluster_ids={},
            compound_ids={},
            root_ids={},
            community_ids={},
            component_ids={},
            community_sizes={},
            component_sizes={},
        )

    nodes = [str(n) for n in G.nodes()]

    # --------------------------
    # Compound → node mapping
    # --------------------------
    compound_ids = {}
    for n in nodes:
        cid = compound_map.get(n)
        if cid:
            compound_ids[n] = str(cid)

    # --------------------------
    # Root element mapping
    # --------------------------
    root_ids = {}
    for n in nodes:
        rid = root_map.get(n)
        if rid:
            root_ids[n] = str(rid)

    # --------------------------
    # Communities
    # --------------------------
    community_ids = _detect_communities(G, emit)

    # --------------------------
    # Connected components
    # --------------------------
    component_ids = {}
    for idx, comp in enumerate(nx.connected_components(G), start=1):
        cid = f"K{idx:03d}"
        for n in comp:
            component_ids[str(n)] = cid

    _log(f"[graphs][analytics] Connected components: "
         f"{len(set(component_ids.values()))}", emit)

    # --------------------------
    # Compose final cluster_id
    # --------------------------
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

    # --------------------------
    # Compute size tables
    # --------------------------
    community_sizes = {cid: sz for cid, sz in Counter(community_ids.values()).items()}
    component_sizes = {cid: sz for cid, sz in Counter(component_ids.values()).items()}

    # Log top communities
    if community_sizes:
        top = sorted(community_sizes.items(), key=lambda kv: kv[1], reverse=True)[:5]
        emit_fn(
            "log",
            {"message": "[graphs][analytics] Top communities",
             "top_communities": [{"community": c, "size": s} for c, s in top]},
        )

    emit_fn(
        "log",
        {"message": "[graphs][analytics] Cluster assignment complete: "
                    f"{len(cluster_ids)} nodes; "
                    f"{len(set(compound_ids.values()))} compounds; "
                    f"{len(set(root_ids.values()))} roots; "
                    f"{len(set(community_ids.values()))} communities."},
    )

    return ClusterInfo(
        cluster_ids=cluster_ids,
        compound_ids=compound_ids,
        root_ids=root_ids,
        community_ids=community_ids,
        component_ids=component_ids,
        community_sizes=community_sizes,
        component_sizes=component_sizes,
    )


# --------------------------------------------------------------------------- #
# Graph statistics
# --------------------------------------------------------------------------- #

def compute_graph_stats(G: nx.Graph) -> GraphStats:
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = float(nx.density(G)) if n_nodes > 1 else 0.0
    avg_degree = float(np.mean([d for _, d in G.degree()])) if n_nodes else 0.0
    try:
        trans = float(nx.transitivity(G)) if n_nodes > 2 else 0.0
    except Exception:
        trans = 0.0

    return GraphStats(
        n_nodes=n_nodes,
        n_edges=n_edges,
        density=density,
        avg_degree=avg_degree,
        transitivity=trans,
    )


# --------------------------------------------------------------------------- #
# Graph pruning
# --------------------------------------------------------------------------- #

def filter_large_components(
    G: nx.Graph,
    min_size: int,
    emit,
) -> nx.Graph:
    """
    Keep only connected components ≥ min_size.
    """
    emit_fn = _get_emit(emit)

    if G.number_of_nodes() == 0 or min_size <= 1:
        return G.copy()

    kept_nodes = []
    for comp in nx.connected_components(G):
        if len(comp) >= min_size:
            kept_nodes.extend(list(comp))

    if not kept_nodes:
        _log(
            f"[graphs][analytics] No components >= {min_size}; "
            f"keeping full graph.", emit
        )
        return G.copy()

    H = G.subgraph(kept_nodes).copy()

    emit_fn(
        "log",
        {"message": "[graphs][analytics] Filtered to large components: "
                    f"{H.number_of_nodes()} nodes, "
                    f"{H.number_of_edges()} edges, "
                    f"min_size={min_size}."},
    )

    return H
