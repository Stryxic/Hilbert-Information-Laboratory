"""
Scientific styling maps for Hilbert graphs (v4).

This module defines the semantic visual encodings used by render2d and
render3d, based on:

  - cluster-aware hues (community or compound)
  - stability/coherence saturation
  - centrality/value brightness
  - hybrid size function (degree * tf + epistemic depth)
  - edge widths and alphas scaled by co-occurrence significance
  - label priority based on multi-metric importance

Everything in this module is deterministic and scale-invariant so that
different graphs, datasets, and runs remain visually comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterable, Any, Optional, List

import numpy as np
import networkx as nx


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class NodeStyleMaps:
    sizes: Dict[str, float] = field(default_factory=dict)
    colors: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)
    halos: Dict[str, float] = field(default_factory=dict)
    alphas: Dict[str, float] = field(default_factory=dict)
    primary_labels: Dict[str, str] = field(default_factory=dict)
    secondary_labels: Dict[str, str] = field(default_factory=dict)

    @property
    def halo_sizes(self) -> Dict[str, float]:
        return self.halos

    def subset(self, nodes: Iterable[str]) -> "NodeStyleMaps":
        keep = set(nodes)
        return NodeStyleMaps(
            sizes={n: v for n, v in self.sizes.items() if n in keep},
            colors={n: v for n, v in self.colors.items() if n in keep},
            halos={n: v for n, v in self.halos.items() if n in keep},
            alphas={n: v for n, v in self.alphas.items() if n in keep},
            primary_labels={n: v for n, v in self.primary_labels.items() if n in keep},
            secondary_labels={n: v for n, v in self.secondary_labels.items() if n in keep},
        )


@dataclass
class EdgeStyleMaps:
    widths: Dict[Tuple[str, str], float] = field(default_factory=dict)
    colors: Dict[Tuple[str, str], Tuple[float, float, float, float]] = field(default_factory=dict)
    alphas: Dict[Tuple[str, str], float] = field(default_factory=dict)
    alpha: Optional[float] = 0.25  # global alpha for 3D

    def subset(self, edges: Iterable[Tuple[str, str]]) -> "EdgeStyleMaps":
        es = {tuple(sorted(e)) for e in edges}
        return EdgeStyleMaps(
            widths={e: v for e, v in self.widths.items() if e in es},
            colors={e: v for e, v in self.colors.items() if e in es},
            alphas={e: v for e, v in self.alphas.items() if e in es},
            alpha=self.alpha,
        )


# =============================================================================
# Helpers
# =============================================================================

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _hsv_to_rgba(h: float, s: float, v: float, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    """Convert HSV to RGBA, with all channels in [0,1]."""
    h = h % 1.0
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return float(r), float(g), float(b), float(alpha)


# =============================================================================
# Node styling (scientific v4)
# =============================================================================

def compute_node_styles(
    G: nx.Graph,
    *,
    cluster_info: Optional[Any] = None,
) -> NodeStyleMaps:
    """
    Semantic node styling:
      - Hue:      community / compound
      - Saturation: stability / coherence
      - Value:   importance (degree + tf + coherence)
      - Size:    hybrid size: f(degree, tf, stability)
      - Halo:    stability band (for 3D glow/depth)
      - Alpha:   rarity (doc_freq) or low importance
    """
    nodes = [str(n) for n in G.nodes()]
    if not nodes:
        return NodeStyleMaps()

    # -------------------------------------------------------------------------
    # Raw metrics
    # -------------------------------------------------------------------------
    deg = np.array([float(G.degree(n)) for n in nodes])
    tf = np.array([float(G.nodes[n].get("tf", 1.0)) for n in nodes])
    coh = np.array([float(G.nodes[n].get("coherence", 0.0)) for n in nodes])
    ent = np.array([float(G.nodes[n].get("entropy", 0.0)) for n in nodes])
    df = np.array([float(G.nodes[n].get("doc_freq", 1.0)) for n in nodes])
    stab = np.array([float(G.nodes[n].get("stability", G.nodes[n].get("temperature", 0.5)))
                     for n in nodes])

    # -------------------------------------------------------------------------
    # Importance metric for size & brightness
    # -------------------------------------------------------------------------
    imp = 0.40 * _norm(deg) + 0.30 * _norm(tf) + 0.30 * _norm(coh)
    imp_s = _norm(imp)

    # Node size (scientific scale)
    sizes = {n: float(30.0 + 120.0 * imp_s[i]) for i, n in enumerate(nodes)}

    # -------------------------------------------------------------------------
    # Hue assignment: based on cluster IDs (community > compound > root)
    # -------------------------------------------------------------------------
    if cluster_info and getattr(cluster_info, "cluster_ids", None):
        cids = cluster_info.cluster_ids
        unique = sorted(set(cids.values()))
        # Map each cluster -> hue in [0,1)
        hues = {cid: i / max(1, len(unique)) for i, cid in enumerate(unique)}
        hue_vec = np.array([hues.get(cids.get(n, ""), 0.0) for n in nodes])
    else:
        # fallback: entropy-based hue
        hue_vec = _norm(ent)

    # -------------------------------------------------------------------------
    # Saturation from stability / coherence
    # -------------------------------------------------------------------------
    sat = 0.50 + 0.45 * _norm(stab)  # [0.5, 1.0]
    # More coherent nodes slightly less saturated
    sat *= (0.9 - 0.4 * _norm(coh))

    # -------------------------------------------------------------------------
    # Value from importance / coherence
    # -------------------------------------------------------------------------
    val = 0.55 + 0.40 * imp_s  # core brightness

    # -------------------------------------------------------------------------
    # Convert HSV → RGBA
    # -------------------------------------------------------------------------
    colors = {
        n: _hsv_to_rgba(hue_vec[i], sat[i], val[i], alpha=0.92)
        for i, n in enumerate(nodes)
    }

    # -------------------------------------------------------------------------
    # Halo intensity (for 3D depth shading)
    # -------------------------------------------------------------------------
    halo_base = _norm(stab)
    halos = {n: float(1.2 + 2.3 * halo_base[i]) for i, n in enumerate(nodes)}

    # -------------------------------------------------------------------------
    # Alpha: rarity / importance fade
    # -------------------------------------------------------------------------
    df_n = _norm(df)
    alphas = {n: float(0.25 + 0.60 * df_n[i]) for i, n in enumerate(nodes)}

    # -------------------------------------------------------------------------
    # Label selection: top multi-metric importance
    # -------------------------------------------------------------------------
    combined = (
        0.40 * imp_s
        + 0.30 * _norm(stab)
        + 0.30 * _norm(deg)
    )
    order = sorted(nodes, key=lambda n: combined[nodes.index(n)], reverse=True)
    top_k = max(10, int(0.05 * len(nodes)))
    primary = set(order[:top_k])

    primary_labels = {n: n for n in primary}
    secondary_labels = {}  # placeholder for future multi-scale labels

    return NodeStyleMaps(
        sizes=sizes,
        colors=colors,
        halos=halos,
        alphas=alphas,
        primary_labels=primary_labels,
        secondary_labels=secondary_labels,
    )


# =============================================================================
# Edge styling (scientific v4)
# =============================================================================

def compute_edge_styles(
    G: nx.Graph,
    *,
    cluster_info: Optional[Any] = None,
) -> EdgeStyleMaps:
    """
    Scientific edge styling:
      - width ~ nonlinear(weight_norm)
      - alpha ~ significance (strong edges = more visible)
      - color ~ cluster-aware hue blending (optional)
    """
    edges = [(str(u), str(v)) for u, v in G.edges()]
    if not edges:
        return EdgeStyleMaps()

    weights = np.array([
        float(G[u][v].get("weight", 1.0)) for u, v in edges
    ])
    w_n = _norm(weights)

    # Width scaling: emphasise strongest quartile
    widths = {
        tuple(sorted((u, v))): float(0.15 + 2.3 * (w_n[i] ** 0.8))
        for i, (u, v) in enumerate(edges)
    }

    # Per-edge alpha: fade weak edges
    alphas = {
        tuple(sorted((u, v))): float(0.25 + 0.75 * (w_n[i] ** 0.7))
        for i, (u, v) in enumerate(edges)
    }

    # Cluster-aware hue blending for edges
    if cluster_info and getattr(cluster_info, "cluster_ids", None):
        cids = cluster_info.cluster_ids
        unique = sorted(set(cids.values()))
        hues = {cid: i / max(1, len(unique)) for i, cid in enumerate(unique)}
        # Blend hues of endpoint nodes
        colors = {}
        for i, (u, v) in enumerate(edges):
            hu = hues.get(cids.get(u, ""), 0.0)
            hv = hues.get(cids.get(v, ""), 0.0)
            h = 0.5 * (hu + hv)
            colors[tuple(sorted((u, v)))] = _hsv_to_rgba(h, 0.3, 0.85, alpha=alphas[tuple(sorted((u, v)))])
    else:
        # fallback: muted violet
        colors = {
            tuple(sorted((u, v))): (0.35, 0.28, 0.55, float(alphas[tuple(sorted((u, v)))]))
            for (u, v) in edges
        }

    return EdgeStyleMaps(
        widths=widths,
        colors=colors,
        alphas=alphas,
        alpha=0.25,
    )
