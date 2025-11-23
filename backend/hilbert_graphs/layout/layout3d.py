"""
3D layout engines for Hilbert graphs (spherical manifold version).

Purpose
-------
This layout solves the persistent problems of spring_layout collapse by
placing every node on a semantically meaningful spherical shell using:

    - LSA vectors (if available)
    - spectral fallback (if not)
    - semantic importance radius modulation
    - jitter + whitening for shape stability
    - global unit-sphere normalisation

Output coordinates are ideal for the holographic shell renderer.
"""

from __future__ import annotations
from typing import Dict, Tuple, List

import numpy as np
import networkx as nx


# ============================================================================ #
# Utilities
# ============================================================================ #

def _norm_or_zero(v: np.ndarray) -> np.ndarray:
    """Normalise vectors row-wise; if any vector has near-zero norm, leave it."""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return v / n


def _extract_lsa_vectors(G: nx.Graph, nodes: List[str]):
    """
    Try to extract LSA vectors for each node, in any of these formats:
        - lsa_x, lsa_y, lsa_z
        - lsa0, lsa1, lsa2
    If missing, return None.
    """
    L = []
    found_any = False

    for n in nodes:
        d = G.nodes[n]

        if all(k in d for k in ("lsa_x", "lsa_y", "lsa_z")):
            found_any = True
            L.append([float(d["lsa_x"]), float(d["lsa_y"]), float(d["lsa_z"])])

        elif all(k in d for k in ("lsa0", "lsa1", "lsa2")):
            found_any = True
            L.append([float(d["lsa0"]), float(d["lsa1"]), float(d["lsa2"])])

        else:
            L.append([0.0, 0.0, 0.0])

    if not found_any:
        return None

    L = np.array(L, dtype=float)
    L -= L.mean(axis=0, keepdims=True)

    span = L.ptp(axis=0)
    span[span < 1e-9] = 1.0
    L /= span

    return L


def _fallback_spectral_embedding(G: nx.Graph, nodes: List[str]):
    """
    If LSA unavailable, obtain 3D embedding from spectral_layout.
    Then normalise & use as direction vectors.
    """
    try:
        pos_dict = nx.spectral_layout(G, dim=3, weight="weight")
    except Exception:
        pos2 = nx.spectral_layout(G, dim=2, weight="weight")
        rng = np.random.default_rng(42)
        pos_dict = {
            n: np.array([p[0], p[1], rng.normal(scale=0.1)], float)
            for n, p in pos2.items()
        }

    V = np.array([pos_dict[n] for n in nodes], dtype=float)
    V -= V.mean(axis=0, keepdims=True)
    return V


def _compute_importance(G: nx.Graph, nodes: List[str]):
    """
    Normalised importance score in [0,1] from:
        - degree
        - tf
        - coherence
    """
    deg = np.array([float(G.degree(n)) for n in nodes], float)
    tf = np.array([float(G.nodes[n].get("tf", 1.0)) for n in nodes], float)
    coh = np.array([
        float(G.nodes[n].get("mean_coherence", G.nodes[n].get("coherence", 0.0)))
        for n in nodes
    ], float)

    def norm(a: np.ndarray):
        lo, hi = float(a.min()), float(a.max())
        if hi - lo < 1e-9:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)

    nd = norm(deg)
    nt = norm(tf)
    nc = norm(coh)

    score = 0.35 * nd + 0.30 * nt + 0.35 * nc
    return norm(score)


def _final_normalisation_sphere(P: np.ndarray) -> np.ndarray:
    """
    Final step: normalise coordinates so max radius = 1.0.
    Ensures stable rendering in render3d.
    """
    radii = np.linalg.norm(P, axis=1)
    rmax = float(radii.max())
    if rmax < 1e-9:
        return P
    return P / rmax


# ============================================================================ #
# Public API: spherical manifold layout
# ============================================================================ #

def compute_layout_3d_spherical(G: nx.Graph) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute a 3D spherical manifold layout for the Hilbert graph.

    Steps
    -----
    1. Try to extract LSA vectors per node.
       - If found → use them as direction vectors.
       - If absent → fallback to spectral embedding.

    2. Normalise vectors to the unit sphere.

    3. Compute importance and assign radius:
         importance=1 → small radius
         importance=0 → large radius

    4. Add jitter tangent to sphere to prevent symmetry collapse.

    5. Normalise entire cloud to max radius 1.0.

    6. Return mapping {node: (x, y, z)}.
    """
    if G.number_of_nodes() == 0:
        return {}

    nodes = [str(n) for n in G.nodes()]
    N = len(nodes)

    # Step 1: extract semantic vectors
    L = _extract_lsa_vectors(G, nodes)
    if L is None:
        L = _fallback_spectral_embedding(G, nodes)

    # Step 2: normalise to get direction vectors
    V = _norm_or_zero(L)

    # Step 3: importance → radius (inner = important)
    imp = _compute_importance(G, nodes)

    r_inner = 0.55
    r_outer = 1.00
    radii = r_inner + (1.0 - imp) * (r_outer - r_inner)

    # Step 4: assign coordinates
    P = V * radii[:, None]

    # Step 5: tangent jitter
    eps = 0.03
    rng = np.random.default_rng(42)
    jitter = rng.normal(scale=eps, size=P.shape)

    dot = np.sum(jitter * V, axis=1, keepdims=True)
    jitter -= dot * V  # remove radial component
    P += jitter

    # Step 6: final normalisation
    P = _final_normalisation_sphere(P)

    # Return mapping
    out = {}
    for i, n in enumerate(nodes):
        x, y, z = P[i]
        out[n] = (float(x), float(y), float(z))

    return out
