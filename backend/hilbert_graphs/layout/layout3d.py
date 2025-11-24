"""
3D spherical-manifold layout (stable, semantic, community-aware, scientifically interpretable).

This module replaces the older 3D layout with a stable manifold layout whose
axes correspond to meaningful semantic structure:

  - Longitude  ~ semantic direction (LSA / spectral embedding)
  - Latitude   ~ community structure (information continents)
  - Radius     ~ epistemic depth (stability / entropy / coherence / tf/degree)
  - Tangential jitter to prevent collapse
  - Final normalisation to unit sphere for renderer stability

Public API:
    - compute_layout_3d_spherical
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List

import numpy as np
import networkx as nx


# ============================================================================ #
# Internal helpers
# ============================================================================ #

def _safe_norm(v: np.ndarray) -> np.ndarray:
    """Row-normalise v with zero-safety."""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return v / n


def _extract_lsa_vectors(G: nx.Graph, nodes: List[str]) -> Optional[np.ndarray]:
    """
    Extract LSA vectors in any recognised format:
      - (lsa_x, lsa_y, lsa_z)
      - (lsa0, lsa1, lsa2)
    If none present, return None.
    """
    L = []
    found = False

    for n in nodes:
        d = G.nodes[n]
        if all(k in d for k in ("lsa_x", "lsa_y", "lsa_z")):
            found = True
            L.append([float(d["lsa_x"]), float(d["lsa_y"]), float(d["lsa_z"])])
        elif all(k in d for k in ("lsa0", "lsa1", "lsa2")):
            found = True
            L.append([float(d["lsa0"]), float(d["lsa1"]), float(d["lsa2"])])
        else:
            L.append([0.0, 0.0, 0.0])

    if not found:
        return None

    A = np.array(L, float)
    A -= A.mean(axis=0, keepdims=True)
    span = A.ptp(axis=0)
    span[span < 1e-9] = 1.0
    return A / span


def _fallback_spectral_embedding(G: nx.Graph, nodes: List[str]) -> np.ndarray:
    """Use spectral_layout(dim=3) if no LSA vectors exist."""
    try:
        pos = nx.spectral_layout(G, dim=3, weight="weight")
    except Exception:
        pos2 = nx.spectral_layout(G, dim=2, weight="weight")
        rng = np.random.default_rng(42)
        pos = {n: np.array([p[0], p[1], rng.normal(scale=0.2)]) for n, p in pos2.items()}

    A = np.array([pos[n] for n in nodes], float)
    A -= A.mean(axis=0, keepdims=True)
    return A


def _compute_importance(G: nx.Graph, nodes: List[str]) -> np.ndarray:
    """
    Multi-metric epistemic importance score in [0,1], combining:
      - degree
      - tf
      - coherence
      - (optionally) stability
    """
    deg = np.array([float(G.degree(n)) for n in nodes], float)
    tf = np.array([float(G.nodes[n].get("tf", 1.0)) for n in nodes], float)
    coh = np.array([float(G.nodes[n].get("coherence", 0.0)) for n in nodes], float)
    stab = np.array([float(G.nodes[n].get("stability", G.nodes[n].get("temperature", 0.5)))
                     for n in nodes], float)

    def norm(a):
        lo, hi = float(a.min()), float(a.max())
        return np.zeros_like(a) if hi - lo < 1e-9 else (a - lo) / (hi - lo)

    score = 0.25 * norm(deg) + 0.30 * norm(tf) + 0.25 * norm(coh) + 0.20 * norm(stab)
    return norm(score)


def _community_latitudes(community_map: Dict[str, str]) -> Dict[str, float]:
    """
    Assign each community a latitude in [-0.90, +0.90].
    Communities sorted alphabetically for determinism.
    """
    if not community_map:
        return {}

    comm_ids = sorted(set(community_map.values()))
    k = len(comm_ids)

    # Avoid poles: keep communities away from +-1
    lats = np.linspace(0.90, -0.90, k)
    return {cid: float(lat) for cid, lat in zip(comm_ids, lats)}


def _compound_longitude_jitter(compound_map: Dict[str, str], nodes: List[str]) -> Dict[str, float]:
    """
    Produce deterministic small longitude offsets per compound so compounds
    form visible sub-continents inside each community continent.
    """
    if not compound_map:
        return {}

    comp_ids = sorted(set(compound_map.values()))
    # Map compound -> small angle jitter in [-0.15, 0.15] radians
    jitter_angles = np.linspace(-0.15, 0.15, len(comp_ids))

    return {cid: float(theta) for cid, theta in zip(comp_ids, jitter_angles)}


# ============================================================================ #
# Public API
# ============================================================================ #

def compute_layout_3d_spherical(
    G: nx.Graph,
    cluster_info: Optional[object] = None,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute a stable 3D spherical manifold layout.

    Inputs:
        - G: element graph
        - cluster_info: ClusterInfo from analytics.py (optional)
            Uses:
              community_ids
              compound_ids

    Output:
        node -> (x, y, z) in unit sphere
    """
    if G.number_of_nodes() == 0:
        return {}

    nodes = [str(n) for n in G.nodes()]
    N = len(nodes)

    # ---------------------------- #
    # 1. Semantic base direction
    # ---------------------------- #
    L = _extract_lsa_vectors(G, nodes)
    if L is None:
        L = _fallback_spectral_embedding(G, nodes)

    V = _safe_norm(L)  # semantic vector for each node

    # ---------------------------- #
    # 2. Community → latitude
    # ---------------------------- #
    if cluster_info and getattr(cluster_info, "community_ids", None):
        community_map = cluster_info.community_ids
        lat_map = _community_latitudes(community_map)
        latitudes = np.array([lat_map.get(community_map.get(n, ""), 0.0) for n in nodes], float)
    else:
        latitudes = np.zeros(N, float)

    # ---------------------------- #
    # 3. Compound → small longitude jitter
    # ---------------------------- #
    if cluster_info and getattr(cluster_info, "compound_ids", None):
        compound_map = cluster_info.compound_ids
        lon_jitter_map = _compound_longitude_jitter(compound_map, nodes)
        jitter = np.array([lon_jitter_map.get(compound_map.get(n, ""), 0.0) for n in nodes], float)
    else:
        jitter = np.zeros(N, float)

    # ---------------------------- #
    # 4. Convert semantic vectors to spherical coords
    #    Replacing theta/phi with semantic + cluster constraints
    # ---------------------------- #
    # Semantic direction gives (vx, vy) → crude longitude seed
    raw_theta = np.arctan2(V[:, 1], V[:, 0])  # [-π, π]
    theta = raw_theta + jitter               # compound offset in radians

    # Latitude override replaces V[:,2]
    phi = np.arcsin(latitudes)               # convert z in [-1,1] to phi

    # ---------------------------- #
    # 5. Reconstruct 3D coordinates
    # ---------------------------- #
    # radius will be added in next step
    xs = np.cos(phi) * np.cos(theta)
    ys = np.cos(phi) * np.sin(theta)
    zs = np.sin(phi)

    B = np.column_stack([xs, ys, zs])
    B = _safe_norm(B)

    # ---------------------------- #
    # 6. Importance → radius (shell depth)
    # ---------------------------- #
    imp = _compute_importance(G, nodes)
    # High importance → inner shell, low importance → outer shell
    r_inner, r_outer = 0.55, 1.00
    radii = r_inner + (1.0 - imp) * (r_outer - r_inner)

    P = B * radii[:, None]

    # ---------------------------- #
    # 7. Tangential jitter
    # ---------------------------- #
    rng = np.random.default_rng(42)
    J = rng.normal(scale=0.03, size=P.shape)

    # Remove radial component
    dot = np.sum(J * B, axis=1, keepdims=True)
    J -= dot * B

    P += J

    # ---------------------------- #
    # 8. Final safety norm
    # ---------------------------- #
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    P /= norms.max()

    return {n: (float(P[i, 0]), float(P[i, 1]), float(P[i, 2])) for i, n in enumerate(nodes)}
