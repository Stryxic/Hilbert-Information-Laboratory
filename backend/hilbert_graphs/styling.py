"""
Node and edge styling: semantic colour, size, alpha, halo, labels (v5).
Electron-cloud compatible, alpha-safe, cluster-aware, entropy-aware.
Manifold-ready: node positions may be spherical, hyperbolic, or Euclidean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Any
import numpy as np
import networkx as nx


# ========================================================================== #
# Dataclasses
# ========================================================================== #

@dataclass
class NodeStyleMaps:
    sizes: Dict[str, float]                 # pre-layout magnitudes (semantic, not pixels)
    colors: Dict[str, Tuple[float,float,float,float]]
    halo_sizes: Dict[str, float]
    alphas: Dict[str, float]
    primary_labels: List[str]
    secondary_labels: List[str]


@dataclass
class EdgeStyleMaps:
    widths: Dict[Tuple[str,str], float]
    alphas: Dict[Tuple[str,str], float]
    alpha: float | None = None              # global alpha override


# ========================================================================== #
# Helpers
# ========================================================================== #

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def _norm(a: np.ndarray) -> np.ndarray:
    """Safe [0,1] normalization; returns zeros if degenerate."""
    if a.size == 0:
        return np.zeros_like(a)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo < 1e-9:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo + 1e-9)


# ========================================================================== #
# Cluster-aware categorical palette
# ========================================================================== #

_CAT20 = np.array([
    [0.121, 0.466, 0.705], [1.000, 0.498, 0.054], [0.172, 0.627, 0.172],
    [0.839, 0.152, 0.156], [0.580, 0.404, 0.741], [0.549, 0.337, 0.294],
    [0.890, 0.467, 0.761], [0.498, 0.498, 0.498], [0.737, 0.741, 0.133],
    [0.090, 0.745, 0.811],
    [0.650, 0.807, 0.890], [0.984, 0.603, 0.600], [0.737, 0.741, 0.133],
    [0.792, 0.698, 0.839], [1.000, 0.733, 0.470], [0.650, 0.337, 0.156],
    [0.600, 0.600, 0.600], [0.369, 0.788, 0.382], [0.925, 0.439, 0.384],
    [0.400, 0.522, 0.600],
])

def _categorical(idx: int) -> Tuple[float,float,float]:
    return tuple(_CAT20[idx % len(_CAT20)])


# ========================================================================== #
# Node Style Pipeline (v5)
# - electron-cloud renderer uses *tiny* node sizes, but we keep semantic sizes here
#   and compress them in the renderer
# - Includes spherical/hyperbolic manifold compatibility
# - All alphas strictly bounded
# ========================================================================== #

def compute_node_styles(G: nx.Graph) -> NodeStyleMaps:
    if G.number_of_nodes() == 0:
        return NodeStyleMaps({}, {}, {}, {}, [], [])

    nodes = list(G.nodes())

    # Semantic core features (all optional)
    deg = np.array([float(G.degree(n)) for n in nodes])
    coh = np.array([
        _safe_float(G.nodes[n].get("mean_coherence", G.nodes[n].get("coherence", 0.0)))
        for n in nodes
    ])
    tf = np.array([_safe_float(G.nodes[n].get("tf", 1.0)) for n in nodes])
    ent = np.array([
        _safe_float(G.nodes[n].get("mean_entropy", G.nodes[n].get("entropy", 0.0)))
        for n in nodes
    ])
    stab = np.array([
        _safe_float(G.nodes[n].get("stability", G.nodes[n].get("temperature", 1.0)))
        for n in nodes
    ])

    # Normalisation
    nd = _norm(deg)
    nc = _norm(coh)
    nt = _norm(tf)
    ne = _norm(ent)
    ns = _norm(stab)

    # Cluster metadata added earlier by analytics module
    compound_ids  = nx.get_node_attributes(G, "compound_id")
    root_ids      = nx.get_node_attributes(G, "root_id")
    community_ids = nx.get_node_attributes(G, "community_id")

    def _index(values: Dict[str,str]) -> Dict[str,int]:
        uniq = sorted(set(values.values()))
        return {cid: i for i, cid in enumerate(uniq)}

    map_comp = _index(compound_ids)   if compound_ids   else {}
    map_root = _index(root_ids)       if root_ids       else {}
    map_comm = _index(community_ids)  if community_ids  else {}

    sizes: Dict[str,float] = {}
    colors: Dict[str,Tuple[float,float,float,float]] = {}
    alphas: Dict[str,float] = {}
    halos: Dict[str,float] = {}

    # Compute per-node styles
    for i, n in enumerate(nodes):
        key = str(n)

        # Semantic node magnitude (renderer compresses this later)
        score = 0.40*nc[i] + 0.35*nt[i] + 0.25*nd[i]
        size = 40.0 + 280.0*(score**1.20)
        sizes[key] = size
        halos[key] = size * 1.55

        # Stability-based alpha (strictly clamped)
        a = 0.35 + 0.65*ns[i]
        a = float(np.clip(a, 0.25, 1.0))
        alphas[key] = a

        # Cluster → colour priority
        if key in compound_ids:
            idx = map_comp[compound_ids[key]]
            r,g,b = _categorical(idx)
            colors[key] = (r,g,b,a)
            continue

        if key in root_ids:
            idx = map_root[root_ids[key]]
            r,g,b = _categorical(idx)
            colors[key] = (r,g,b,a)
            continue

        if key in community_ids:
            idx = map_comm[community_ids[key]]
            r,g,b = _categorical(idx)
            colors[key] = (r,g,b,a)
            continue

        # Entropy-driven fallback colour (cool-to-warm)
        r = 0.25 + 0.70*ne[i]
        g = 0.15 + 0.45*nc[i]
        b = 0.35 + 0.55*(1.0 - ne[i])
        colors[key] = (
            float(np.clip(r, 0, 1)),
            float(np.clip(g, 0, 1)),
            float(np.clip(b, 0, 1)),
            a,
        )

    # Label ranking (global top-k by semantic size)
    ordered = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)
    ranked = [n for n,_ in ordered]

    primary = ranked[:20]
    secondary = ranked[20:80]

    return NodeStyleMaps(
        sizes=sizes,
        colors=colors,
        halo_sizes=halos,
        alphas=alphas,
        primary_labels=primary,
        secondary_labels=secondary,
    )


# ========================================================================== #
# Edge styling (v5)
# Safe alpha, depth-friendly widths, consistent with electron cloud aesthetic
# ========================================================================== #

def compute_edge_styles(G: nx.Graph) -> EdgeStyleMaps:
    if G.number_of_edges() == 0:
        return EdgeStyleMaps({}, {}, alpha=None)

    edges = list(G.edges(data=True))
    weights = np.array([
        _safe_float(d.get("weight", 1.0), 1.0) for _,_,d in edges
    ])
    w_norm = _norm(weights)

    widths: Dict[Tuple[str,str],float] = {}
    alphas: Dict[Tuple[str,str],float] = {}

    # Slightly lower base alpha for electron-cloud mode
    global_alpha = 0.07 if G.number_of_nodes()>800 else 0.14
    global_alpha = float(np.clip(global_alpha, 0.0, 1.0))

    for (u,v,_), w in zip(edges, w_norm):
        width = 0.20 + 1.90*float(w)
        alpha = global_alpha*(0.65 + 0.60*float(w))
        alpha = float(np.clip(alpha, 0.05, 0.9))

        uv = (str(u),str(v))
        vu = (str(v),str(u))

        widths[uv] = widths[vu] = width
        alphas[uv] = alphas[vu] = alpha

    return EdgeStyleMaps(widths=widths, alphas=alphas, alpha=global_alpha)
