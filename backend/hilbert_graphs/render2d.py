"""
2D rendering of Hilbert graphs - Electron Field Mode (v4).

This renderer treats nodes like particles in an electron cloud:
  - extremely small node core
  - bright Gaussian-like halos
  - faint edges (almost invisible)
  - density encoded by cumulative brightness
  - smooth volumetric visual impression
"""

from __future__ import annotations

from typing import Dict, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import networkx as nx
import numpy as np

from .styling import NodeStyleMaps, EdgeStyleMaps
from .presets import VisualStyle


# ============================================================================ #
# Camera framing
# ============================================================================ #

def _camera_frame_2d(pos: Dict[str, Tuple[float, float]], margin_pct: float = 0.08):
    if not pos:
        return -1, 1, -1, 1

    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])

    x_min, x_max = np.percentile(xs, [2, 98])
    y_min, y_max = np.percentile(ys, [2, 98])

    dx = max(1e-9, x_max - x_min)
    dy = max(1e-9, y_max - y_min)

    return (
        float(x_min - dx * margin_pct),
        float(x_max + dx * margin_pct),
        float(y_min - dy * margin_pct),
        float(y_max + dy * margin_pct),
    )


# ============================================================================ #
# Labels
# ============================================================================ #

def _draw_labels(G, ax, pos, node_styles, style, label_budget):
    primary = list(node_styles.primary_labels or [])
    secondary = list(node_styles.secondary_labels or [])
    sizes = node_styles.sizes or {}

    # fallback: size-driven labels
    if not primary:
        ordered = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)
        nodes = [n for n, _ in ordered if n in G.nodes()]
        primary_n = max(1, int(label_budget * 0.7))
        primary = nodes[:primary_n]

    primary = [n for n in primary if n in G.nodes()]

    lbls = nx.draw_networkx_labels(
        G,
        pos,
        labels={n: n for n in primary},
        font_size=style.label_primary_size,
        font_weight="bold",
        font_color=style.label_color,
        alpha=0.95,
        ax=ax,
    )

    if patheffects:
        for txt in lbls.values():
            txt.set_path_effects([
                patheffects.Stroke(
                    linewidth=style.label_outline_width,
                    foreground=style.label_outline_color,
                ),
                patheffects.Normal(),
            ])


# ============================================================================ #
# Main Electron Field Renderer
# ============================================================================ #

def draw_2d_snapshot(
    G: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    node_styles: NodeStyleMaps,
    edge_styles: EdgeStyleMaps,
    style: VisualStyle,
    outfile: str,
    title: str,
    subtitle: str,
    label_budget: int,
):
    if G.number_of_nodes() == 0:
        return

    nodes = list(G.nodes())
    sub_pos = {n: pos[n] for n in nodes if n in pos}
    if not sub_pos:
        return

    # Camera
    x_min, x_max, y_min, y_max = _camera_frame_2d(sub_pos)

    # Style maps
    size_map = node_styles.sizes or {}
    color_map = node_styles.colors or {}

    # Electron field = smallest possible nodes
    base_core = 6.0         # core dot size
    halo_scale = 3.8        # multiplication factor for halo brightness

    # Faint edges
    widths_map = edge_styles.widths or {}
    edges = list(G.edges())
    edge_alpha = 0.04       # nearly invisible edges

    # Node style conversion
    def norm_color(c):
        if c is None:
            return (0.6, 0.75, 1.0, 0.35)
        if len(c) == 3:
            return (c[0], c[1], c[2], 0.35)
        return (c[0], c[1], c[2], c[3] * 0.35)

    # Node halo: brightness by size (coherence/tf/degree proxy)
    halos = []
    cores = []
    colors = []

    for n in nodes:
        metric = float(size_map.get(n, 40.0))
        # compress range, brighten only if large
        glow = np.log1p(metric) * halo_scale
        halos.append(glow)
        cores.append(base_core)
        colors.append(norm_color(color_map.get(n)))

    # --- Figure -------------------------------------------------------
    plt.figure(figsize=(18, 14), facecolor=style.background_color)
    ax = plt.gca()
    ax.set_facecolor(style.background_color)
    ax.set_axis_off()

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # --- Edges (very faint) ------------------------------------------
    if edges:
        faint = (0.55, 0.6, 0.8, edge_alpha)
        nx.draw_networkx_edges(
            G,
            sub_pos,
            edgelist=edges,
            width=0.25,
            edge_color=faint,
            alpha=edge_alpha,
            ax=ax,
        )

    # --- Halos (big fuzzy field) -------------------------------------
    nx.draw_networkx_nodes(
        G,
        sub_pos,
        nodelist=nodes,
        node_size=halos,
        node_color=colors,
        linewidths=0.0,
        alpha=0.45,
        ax=ax,
    )

    # --- Core electrons (small dots) ---------------------------------
    core_color = (0.85, 0.95, 1.0, 0.95)
    nx.draw_networkx_nodes(
        G,
        sub_pos,
        nodelist=nodes,
        node_size=cores,
        node_color=[core_color] * len(nodes),
        linewidths=0.0,
        alpha=1.0,
        ax=ax,
    )

    # --- Text metadata bar -------------------------------------------
    if title:
        ax.set_title(
            title,
            fontsize=14,
            weight="bold",
            color=style.label_color,
            loc="left",
            pad=12,
        )

    if subtitle:
        ax.text(
            0.01,
            0.965,
            subtitle,
            transform=ax.transAxes,
            color="#b9bec7",
            fontsize=9,
            ha="left",
            va="center",
        )

    # --- Labels -------------------------------------------------------
    if label_budget > 0:
        _draw_labels(G, ax, sub_pos, node_styles, style, label_budget)

    plt.tight_layout(pad=0.3)
    plt.savefig(outfile, dpi=style.dpi)
    plt.close()
