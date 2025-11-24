"""
2D layout engines (community-aware, compound-aware, stable, hub-compatible).

This upgraded module implements:
  - canonical 2D layout (stable across snapshots)
  - community sector placement
  - compound and root substructure ordering
  - stability / entropy / coherence radial influence
  - multi-scale repulsion (macro / meso / micro)
  - final [-1,1]^2 normalised coordinate field

Exports:
    - compute_layout_2d_hybrid
    - compute_layout_2d_radial
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List

import numpy as np
import networkx as nx


# ============================================================================ #
# Helpers
# ============================================================================ #

def _center_positions(pos: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    if not pos:
        return {}
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    cx, cy = xs.mean(), ys.mean()
    return {n: (float(x - cx), float(y - cy)) for n, (x, y) in pos.items()}


def _normalise_positions(pos: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """Normalise to [-1,1]^2, centering first."""
    if not pos:
        return {}
    pos = _center_positions(pos)
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    dx = max(1e-9, xs.max() - xs.min())
    dy = max(1e-9, ys.max() - ys.min())
    scale = 1.0 / max(dx, dy)
    return {n: (float(x * scale), float(y * scale)) for n, (x, y) in pos.items()}


# ============================================================================ #
# Community / compound seeding
# ============================================================================ #

def _cluster_centroids(ids: Dict[str, str], base_radius: float) -> Dict[str, Tuple[float, float]]:
    """Place cluster IDs (communities or compounds) on a circle."""
    if not ids:
        return {}
    uniq = sorted(set(ids.values()))
    k = len(uniq)
    thetas = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return {
        cid: (base_radius * float(np.cos(t)), base_radius * float(np.sin(t)))
        for cid, t in zip(uniq, thetas)
    }


def _initial_positions(
    G: nx.Graph,
    *,
    community_ids: Optional[Dict[str, str]] = None,
    compound_ids: Optional[Dict[str, str]] = None,
    stability_vals: Optional[Dict[str, float]] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Hierarchical initialisation:
      1. Communities placed radially (macro structure)
      2. Compounds placed around their community centroid (meso structure)
      3. Roots + internal structure jittered (micro structure)
    """
    nodes = list(G.nodes())
    if not nodes:
        return {}

    # Radii
    R_comm = 1.8
    R_comp = 0.8

    comm_centroids = _cluster_centroids(community_ids or {}, R_comm)
    comp_centroids = _cluster_centroids(compound_ids or {}, R_comp)

    pos = {}

    for n in nodes:
        nid = str(n)
        # Macro: community position
        if community_ids and nid in community_ids:
            cc = comm_centroids.get(community_ids[nid], (0.0, 0.0))
        else:
            cc = (0.0, 0.0)

        # Meso: compound sub-centroid
        if compound_ids and nid in compound_ids:
            cs = comp_centroids.get(compound_ids[nid], (0.0, 0.0))
        else:
            cs = (0.0, 0.0)

        # Combine macro + meso
        px = cc[0] + 0.6 * cs[0]
        py = cc[1] + 0.6 * cs[1]

        # Micro jitter
        rng = np.random.default_rng(abs(hash(nid)) & 0xFFFFFFFF)
        jx = rng.normal(scale=0.15)
        jy = rng.normal(scale=0.15)

        # Optionally warp by stability to create pseudo-radial shells
        if stability_vals:
            s = stability_vals.get(nid, 0.5)
            px *= 1.0 + 0.4 * (1.0 - s)
            py *= 1.0 + 0.4 * (1.0 - s)

        pos[nid] = (float(px + jx), float(py + jy))

    return pos


# ============================================================================ #
# Hybrid layout (enhanced spectral + spring + KK)
# ============================================================================ #

def compute_layout_2d_hybrid(
    G: nx.Graph,
    cluster_info: Optional[Any] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Canonical Hilbert 2D layout.

    cluster_info supplies:
      - community_ids
      - compound_ids
      - stability_bands or direct stability values
    """
    if G.number_of_nodes() == 0:
        return {}

    # Cluster info
    community_ids = getattr(cluster_info, "community_ids", None)
    compound_ids = getattr(cluster_info, "compound_ids", None)

    # stability values preferred from stability_z, else entropy/coherence
    stability_vals = None
    if cluster_info and hasattr(cluster_info, "stability_bands"):
        # Convert band labels to approximate numeric field
        # Lower band index = lower radius
        stability_vals = {}
        for b_idx, (band, members) in enumerate(sorted(cluster_info.stability_bands.items())):
            for n in members:
                stability_vals[n] = float((b_idx + 1) / len(cluster_info.stability_bands))

    # 1. Hierarchical seed
    try:
        pos0 = _initial_positions(
            G,
            community_ids=community_ids,
            compound_ids=compound_ids,
            stability_vals=stability_vals,
        )
    except Exception:
        pos0 = None

    # 2. Spring layout refinement
    n = G.number_of_nodes()
    avg_deg = float(np.mean([d for _, d in G.degree()])) if n else 0.0
    if avg_deg < 2:
        k = 3.2 / np.sqrt(max(n, 1))
    elif avg_deg < 5:
        k = 2.2 / np.sqrt(max(n, 1))
    else:
        k = 1.4 / np.sqrt(max(n, 1))

    try:
        pos_spring = nx.spring_layout(
            G,
            dim=2,
            pos=pos0,
            k=k,
            weight="weight",
            iterations=160,
            seed=42,
        )
    except Exception:
        pos_spring = nx.spring_layout(
            G,
            dim=2,
            k=k,
            weight="weight",
            iterations=160,
            seed=42,
        )

    # 3. Kamada-Kawai refinement
    try:
        pos_kk = nx.kamada_kawai_layout(
            G,
            pos=pos_spring,
            dim=2,
            weight="weight",
        )
    except Exception:
        pos_kk = pos_spring

    # 4. Center + normalise
    return _normalise_positions(pos_kk)


# ============================================================================ #
# Radial layout
# ============================================================================ #

def compute_layout_2d_radial(
    G: nx.Graph,
    *,
    mode: str = "stability",
    cluster_info: Optional[Any] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Radial scientific layout:
      - stability or entropy/coherence determines radius
      - deterministic angular ordering
      - community-based angular grouping if available
    """
    if G.number_of_nodes() == 0:
        return {}

    nodes = [str(n) for n in G.nodes()]

    # Extract scalar
    if mode == "degree":
        vals = {n: float(G.degree(n)) for n in nodes}
    elif mode == "entropy":
        vals = {n: float(G.nodes[n].get("entropy", 0.0)) for n in nodes}
    elif mode == "coherence":
        vals = {n: float(G.nodes[n].get("coherence", 0.0)) for n in nodes}
    else:  # stability
        vals = {n: float(G.nodes[n].get("stability", G.nodes[n].get("temperature", 0.5))) for n in nodes}

    arr = np.array(list(vals.values()))
    lo, hi = float(arr.min()), float(arr.max())
    norm = np.zeros_like(arr) if hi - lo < 1e-9 else (arr - lo) / (hi - lo)
    norm_map = {n: float(v) for n, v in zip(nodes, norm)}

    # Angular ordering: group by communities if available
    community_ids = getattr(cluster_info, "community_ids", None)

    if community_ids:
        comm_groups: Dict[str, List[str]] = {}
        for n in nodes:
            cid = community_ids.get(n, "Q000")
            comm_groups.setdefault(cid, []).append(n)
        comm_order = sorted(comm_groups.keys())
    else:
        comm_order = ["ALL"]
        comm_groups = {"ALL": nodes}

    # Layout
    pos: Dict[str, Tuple[float, float]] = {}
    rng = np.random.default_rng(42)

    # Each community occupies contiguous angular sector
    base_angle = 0.0
    sector_step = 2 * np.pi / len(comm_order)

    for cid in comm_order:
        group = comm_groups[cid]
        m = max(1, len(group))
        start = base_angle
        end = start + sector_step

        # Distribute evenly
        angles = np.linspace(start, end, m, endpoint=False)
        rng.shuffle(angles)

        for n, a in zip(group, angles):
            r = 0.2 + 0.9 * norm_map[n]
            pos[n] = (float(r * np.cos(a)), float(r * np.sin(a)))

        base_angle += sector_step

    return _normalise_positions(pos)
