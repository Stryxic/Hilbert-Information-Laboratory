"""
2D layout engines (community-aware enhanced version).
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Optional
import numpy as np
import networkx as nx


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#

def _safe_normalise_positions(
    pos: Dict[str, Tuple[float, float]]
) -> Dict[str, Tuple[float, float]]:
    """Rescale layout to a rough unit-square bounding box."""
    if not pos:
        return {}

    xs = np.array([p[0] for p in pos.values()], dtype=float)
    ys = np.array([p[1] for p in pos.values()], dtype=float)

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    dx = max(1e-9, x_max - x_min)
    dy = max(1e-9, y_max - y_min)

    scale = 2.0 / max(dx, dy)

    out = {}
    for n, (x, y) in pos.items():
        xn = (x - x_min) * scale - (dx * scale) / 2.0
        yn = (y - y_min) * scale - (dy * scale) / 2.0
        out[n] = (float(xn), float(yn))

    return out


def _initial_cluster_positions(
    G: nx.Graph,
    community_map: Dict[str, int],
    radius_scale: float = 1.0,
) -> Dict[str, Tuple[float, float]]:
    """
    Place communities on a coarse circle before spring refinement.

    Returns an initial pos dict mapping node -> (x,y).
    """
    if not community_map:
        return {}

    nodes = list(G.nodes())
    comm_ids = sorted({community_map.get(n, 0) for n in nodes})
    k = len(comm_ids)

    # Angle for each community
    thetas = np.linspace(0, 2*np.pi, k, endpoint=False)
    cluster_centroids = {
        cid: (radius_scale * np.cos(t), radius_scale * np.sin(t))
        for cid, t in zip(comm_ids, thetas)
    }

    pos0 = {}
    for n in nodes:
        cid = community_map.get(n, 0)
        cx, cy = cluster_centroids[cid]
        # jitter inside cluster centre to avoid degenerate placements
        jx = (np.random.default_rng(hash(n) % (2**32)).normal() * 0.05)
        jy = (np.random.default_rng(~hash(n) % (2**32)).normal() * 0.05)
        pos0[n] = (cx + jx, cy + jy)

    return pos0


# -----------------------------------------------------------------------------#
# HYBRID LAYOUT (Improved: community-aware)
# -----------------------------------------------------------------------------#

def compute_layout_2d_hybrid(
    G: nx.Graph,
    community_map: Optional[Dict[str, int]] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Community-aware hybrid 2D layout:
      - Optional community-based initial placement
      - spectral_layout for global structure (fallback)
      - spring_layout with tuned k
      - optional Kamada-Kawai refinement for small graphs

    Normalises to a unit-square bounding box.
    """
    if G.number_of_nodes() == 0:
        return {}

    # --- Step 0: community seeded initial layout --------------------------------#
    if community_map:
        pos0 = _initial_cluster_positions(G, community_map, radius_scale=1.8)
    else:
        try:
            pos0 = nx.spectral_layout(G, dim=2, seed=42)
        except Exception:
            pos0 = None

    # --- Step 1: tuned spring refinement ----------------------------------------#
    n = max(G.number_of_nodes(), 1)
    k = 1.2 / np.sqrt(n)

    try:
        pos1 = nx.spring_layout(
            G,
            dim=2,
            pos=pos0,
            k=k,
            iterations=90,
            weight="weight",
            seed=42,
        )
    except Exception:
        pos1 = nx.spring_layout(
            G,
            dim=2,
            k=k,
            iterations=90,
            weight="weight",
            seed=42,
        )

    # --- Step 2: Kamada-Kawai refinement ----------------------------------------#
    if G.number_of_nodes() <= 800:
        try:
            pos2 = nx.kamada_kawai_layout(
                G,
                pos=pos1,
                weight="weight",
                dim=2,
            )
        except Exception:
            pos2 = pos1
    else:
        pos2 = pos1

    # --- Step 3: normalise ------------------------------------------------------#
    return _safe_normalise_positions(pos2)


# -----------------------------------------------------------------------------#
# RADIAL LAYOUT (Unchanged except cleanup)
# -----------------------------------------------------------------------------#

def compute_layout_2d_radial(
    G: nx.Graph,
    mode: str = "stability"
) -> Dict[str, Tuple[float, float]]:
    """
    Radial layout - nodes placed on concentric rings based on a scalar attribute.
    """
    if G.number_of_nodes() == 0:
        return {}

    nodes = list(G.nodes())

    # Scoring
    if mode == "degree":
        scores = np.array([float(G.degree(n)) for n in nodes], dtype=float)
    elif mode == "epistemic":
        scores = np.array([
            float(G.nodes[n].get("mean_entropy", G.nodes[n].get("entropy", 0.0)))
            for n in nodes
        ], dtype=float)
    else:
        scores = np.array([
            float(G.nodes[n].get("stability", G.nodes[n].get("temperature", 1.0)))
            for n in nodes
        ], dtype=float)

    # Normalise
    if scores.size == 0:
        return {}

    mn, mx = float(np.nanmin(scores)), float(np.nanmax(scores))
    if mx - mn < 1e-9:
        norm_scores = np.zeros_like(scores)
    else:
        norm_scores = (scores - mn) / (mx - mn + 1e-9)

    # Quantile buckets
    q = np.quantile(norm_scores, [0, 0.25, 0.5, 0.75, 1])
    radii = np.linspace(0.2, 1.0, 5)

    ring_nodes = [[] for _ in range(5)]
    for n, s in zip(nodes, norm_scores):
        if s <= q[1]:
            ring_nodes[0].append(n)
        elif s <= q[2]:
            ring_nodes[1].append(n)
        elif s <= q[3]:
            ring_nodes[2].append(n)
        elif s <= q[4]:
            ring_nodes[3].append(n)
        else:
            ring_nodes[4].append(n)

    pos = {}
    rng = np.random.default_rng(seed=42)

    for i, ring in enumerate(ring_nodes):
        r = radii[i]
        m = max(len(ring), 1)
        angles = np.linspace(0, 2*np.pi, m, endpoint=False)
        rng.shuffle(angles)
        for n, a in zip(ring, angles):
            pos[n] = (float(r*np.cos(a)), float(r*np.sin(a)))

    return pos
