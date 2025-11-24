"""
Scientific 2D snapshot renderer for the Hilbert graph visualiser (v4).

New features:
    - cluster-aware translucent convex hulls
    - density shading (KDE-based) behind nodes
    - improved edge rendering with semantic colours
    - multi-pass halo + core node drawing
    - decluttered label placement (budget-limited)
    - stable axis scaling and deterministic ordering

This renderer produces scientific-quality PNGs suitable for reports
and publication, and is fully compatible with the upgraded styling
and layout engines.
"""

from __future__ import annotations

from typing import Dict, Tuple, Iterable, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.collections import PolyCollection
from scipy.stats import gaussian_kde

import networkx as nx

from .presets import VisualStyle
from .styling import NodeStyleMaps, EdgeStyleMaps


# =============================================================================
# Utilities
# =============================================================================

def _frame_from_positions(pos: Dict[str, Tuple[float, float]], margin: float = 0.08):
    """Compute padded axis limits from node positions."""
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    span = max(x_max - x_min, y_max - y_min, 1e-9)
    pad = span * margin

    return (
        x_min - pad,
        x_max + pad,
        y_min - pad,
        y_max + pad,
    )


def _depth_sorted_edges(G: nx.Graph, pos):
    """Sort edges back-to-front by average radial distance."""
    def r2(n):
        x, y = pos[n]
        return x * x + y * y

    buf = []
    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        rm = 0.5 * (r2(u) + r2(v))
        buf.append((rm, u, v))

    buf.sort(key=lambda t: t[0], reverse=True)
    for _, u, v in buf:
        yield u, v


def _rgba_tuple(c):
    """Normalize a colour mapping entry to RGBA."""
    if isinstance(c, tuple) and len(c) >= 4:
        return tuple(map(float, c))
    if isinstance(c, tuple) and len(c) == 3:
        return float(c[0]), float(c[1]), float(c[2]), 1.0
    return (0.7, 0.8, 0.9, 1.0)


# =============================================================================
# Convex Hull Overlay (cluster regions)
# =============================================================================

def _draw_cluster_hulls(
    ax,
    pos: Dict[str, Tuple[float, float]],
    cluster_ids: Optional[Dict[str, str]],
    style: VisualStyle,
):
    """
    Draw faint convex hulls around clusters (communities or compounds).
    This makes macro-scale structure immediately visible.
    """
    if not cluster_ids:
        return

    clusters = {}
    for n, cid in cluster_ids.items():
        if n in pos:
            clusters.setdefault(cid, []).append(pos[n])

    polys = []
    colors = []

    for cid, pts in clusters.items():
        pts = np.array(pts)
        if pts.shape[0] < 4:
            continue  # cannot form hull

        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts)
            poly = pts[hull.vertices]
            polys.append(poly)
            colors.append(style.hull_face_color)
        except Exception:
            continue

    if polys:
        coll = PolyCollection(
            polys,
            facecolors=[style.hull_face_color] * len(polys),
            edgecolors=[style.hull_edge_color] * len(polys),
            linewidths=1.2,
            alpha=style.hull_face_alpha,
            zorder=0,
        )
        ax.add_collection(coll)


# =============================================================================
# Density background layer
# =============================================================================

def _draw_density_field(ax, pos, style: VisualStyle):
    """
    Render a KDE-based density field beneath nodes for visual texture and depth.
    """
    if len(pos) < 40:
        return

    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])

    try:
        kde = gaussian_kde(np.vstack([xs, ys]))
    except Exception:
        return

    # grid resolution
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    gx, gy = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
    grid = np.vstack([gx.ravel(), gy.ravel()])
    vals = kde(grid).reshape(gx.shape)

    vals = np.log1p(vals)
    vals /= vals.max()

    ax.imshow(
        vals.T,
        extent=(xmin, xmax, ymin, ymax),
        cmap="inferno",
        alpha=0.18,
        origin="lower",
        interpolation="bicubic",
        zorder=0,
    )


# =============================================================================
# Main 2D renderer
# =============================================================================

def draw_2d_snapshot(
    G: nx.Graph,
    pos2d: Dict[str, Tuple[float, float]],
    node_styles: NodeStyleMaps,
    edge_styles: EdgeStyleMaps,
    style: VisualStyle,
    *,
    outfile: str,
    title: str = "",
    subtitle: str = "",
    label_budget: int = 20,
    cluster_info: Optional[Any] = None,
):
    """
    Render a scientific-quality 2D snapshot of the Hilbert information graph.
    """

    if G.number_of_nodes() == 0:
        return

    # Nodes that actually have positions
    nodes = [n for n in G.nodes() if n in pos2d]
    if not nodes:
        return

    pos = {n: pos2d[n] for n in nodes}

    # ----------------------------------------------------------
    # Canvas setup
    # ----------------------------------------------------------
    x_min, x_max, y_min, y_max = _frame_from_positions(pos)

    fig, ax = plt.subplots(
        figsize=(18, 14),
        facecolor=style.background_color,
    )
    ax.set_facecolor(style.background_color)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", "box")

    # ----------------------------------------------------------
    # Density / hull overlays (background)
    # ----------------------------------------------------------
    _draw_density_field(ax, pos, style)

    if cluster_info and getattr(cluster_info, "cluster_ids", None):
        _draw_cluster_hulls(ax, pos, cluster_info.cluster_ids, style)

    # ----------------------------------------------------------
    # Edges
    # ----------------------------------------------------------
    if G.number_of_edges() > 0:
        widths = edge_styles.widths or {}
        ecolors = edge_styles.colors or {}
        ealphas = edge_styles.alphas or {}

        for u, v in _depth_sorted_edges(G, pos):
            key = (u, v) if (u, v) in widths else (v, u)
            width = float(widths.get(key, (style.edge_width_min + style.edge_width_max) * 0.5))

            rgba = _rgba_tuple(ecolors.get(key, style.edge_color))
            alpha_edge = ealphas.get(key, 1.0)
            r, g, b, a = rgba
            a = float(np.clip(a * alpha_edge, 0.0, 1.0))

            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=(r, g, b, a),
                linewidth=width,
                solid_capstyle="round",
                zorder=1,
            )

    # ----------------------------------------------------------
    # Nodes — halos then cores
    # ----------------------------------------------------------
    sizes = np.array([node_styles.sizes.get(n, 20.0) for n in nodes], float)
    colors = np.array([_rgba_tuple(node_styles.colors.get(n)) for n in nodes], float)

    halos = np.array([node_styles.halos.get(n, 1.8) for n in nodes], float)
    halo_colors = colors.copy()
    halo_colors[:, 3] = halo_colors[:, 3] * style.node_halo_alpha

    core_sizes = sizes ** 0.92
    halo_sizes = core_sizes * halos * style.node_halo_scale

    # Halos
    ax.scatter(
        [pos[n][0] for n in nodes],
        [pos[n][1] for n in nodes],
        s=halo_sizes,
        c=halo_colors,
        marker="o",
        linewidths=0.0,
        zorder=2,
    )

    # Cores
    core_colors = colors.copy()
    core_colors[:, 3] = core_colors[:, 3] * style.node_alpha

    ax.scatter(
        [pos[n][0] for n in nodes],
        [pos[n][1] for n in nodes],
        s=core_sizes,
        c=core_colors,
        marker="o",
        edgecolors=style.node_edge_color,
        linewidths=0.3,
        zorder=3,
    )

    # ----------------------------------------------------------
    # Labels (decluttered)
    # ----------------------------------------------------------
    label_map = node_styles.primary_labels or {}
    if label_budget > 0 and label_map:
        importance = {n: node_styles.sizes.get(n, 0.0) for n in nodes}
        candidates = sorted(
            [n for n in nodes if n in label_map],
            key=lambda n: importance.get(n, 0.0),
            reverse=True,
        )[:label_budget]

        for n in candidates:
            x, y = pos[n]
            txt = ax.text(
                x,
                y,
                label_map[n],
                fontsize=style.label_primary_size,
                color=style.label_color,
                ha="center",
                va="center",
                zorder=5,
            )
            txt.set_path_effects([
                patheffects.Stroke(
                    linewidth=style.label_outline_width,
                    foreground=style.label_outline_color,
                ),
                patheffects.Normal(),
            ])

    # ----------------------------------------------------------
    # Titles
    # ----------------------------------------------------------
    if title:
        ax.set_title(
            title,
            fontsize=16,
            color=style.label_color,
            loc="left",
            pad=14,
        )
    if subtitle:
        fig.text(0.01, 0.965, subtitle, color="#c0c4d0", fontsize=9, ha="left")

    plt.tight_layout(pad=0.5)
    plt.savefig(outfile, dpi=style.dpi, facecolor=style.background_color)
    plt.close()
