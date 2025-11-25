# render2d.py

"""
Scientific 2D snapshot renderer for the Hilbert graph visualiser (v4).

New features:
    - cluster-aware translucent convex hulls
    - density shading (KDE-based) behind nodes
    - improved edge rendering with semantic colours
    - multi-pass halo + core node drawing
    - decluttered label placement (budget-limited, approximate collision)
    - stable axis scaling and deterministic ordering

This renderer produces scientific-quality PNGs suitable for reports
and publication, and is fully compatible with the upgraded styling
and layout engines.
"""

from __future__ import annotations

from typing import Dict, Tuple, Iterable, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.collections import PolyCollection

import networkx as nx

from .presets import VisualStyle
from .styling import NodeStyleMaps, EdgeStyleMaps


# =============================================================================
# Utilities
# =============================================================================

def _frame_from_positions(
    pos: Dict[str, Tuple[float, float]],
    margin: float = 0.08,
) -> Tuple[float, float, float, float]:
    """Compute padded axis limits from node positions."""
    xs = np.array([p[0] for p in pos.values()], float)
    ys = np.array([p[1] for p in pos.values()], float)

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    span = max(x_max - x_min, y_max - y_min, 1e-9)
    pad = span * margin

    return (
        x_min - pad,
        x_max + pad,
        y_min - pad,
        y_max + pad,
    )


def _depth_sorted_edges(
    G: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
) -> Iterable[Tuple[str, str]]:
    """Sort edges back-to-front by average radial distance."""
    def r2(n: str) -> float:
        x, y = pos[n]
        return float(x * x + y * y)

    buf: List[Tuple[float, str, str]] = []
    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        rm = 0.5 * (r2(u) + r2(v))
        buf.append((rm, u, v))

    buf.sort(key=lambda t: t[0], reverse=True)
    for _, u, v in buf:
        yield u, v


def _rgba_tuple(c: Any) -> Tuple[float, float, float, float]:
    """Normalize a colour mapping entry to RGBA."""
    if isinstance(c, tuple) or isinstance(c, list):
        if len(c) >= 4:
            return float(c[0]), float(c[1]), float(c[2]), float(c[3])
        if len(c) == 3:
            return float(c[0]), float(c[1]), float(c[2]), 1.0
    return 0.7, 0.8, 0.9, 1.0


# =============================================================================
# Convex Hull Overlay (cluster regions)
# =============================================================================

def _draw_cluster_hulls(
    ax,
    pos: Dict[str, Tuple[float, float]],
    cluster_ids: Optional[Dict[str, str]],
    style: VisualStyle,
) -> None:
    """
    Draw faint convex hulls around clusters (communities or compounds).
    This makes macro-scale structure immediately visible.
    """
    if not cluster_ids:
        return

    clusters: Dict[str, List[Tuple[float, float]]] = {}
    for n, cid in cluster_ids.items():
        if n in pos:
            clusters.setdefault(str(cid), []).append(pos[n])

    polys: List[np.ndarray] = []

    for cid, pts in clusters.items():
        pts_arr = np.array(pts, float)
        if pts_arr.shape[0] < 4:
            continue  # cannot form hull

        try:
            from scipy.spatial import ConvexHull  # type: ignore
        except Exception:
            # SciPy not available - skip hulls gracefully
            return

        try:
            hull = ConvexHull(pts_arr)
            poly = pts_arr[hull.vertices]
            polys.append(poly)
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

def _draw_density_field(
    ax,
    pos: Dict[str, Tuple[float, float]],
    style: VisualStyle,
) -> None:
    """
    Render a KDE-based density field beneath nodes for visual texture and depth.
    This is optional and will be skipped if SciPy is not available.
    """
    if len(pos) < 40:
        return

    xs = np.array([p[0] for p in pos.values()], float)
    ys = np.array([p[1] for p in pos.values()], float)

    try:
        from scipy.stats import gaussian_kde  # type: ignore
    except Exception:
        # SciPy not available - skip density field gracefully
        return

    try:
        kde = gaussian_kde(np.vstack([xs, ys]))
    except Exception:
        return

    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())

    gx, gy = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
    grid = np.vstack([gx.ravel(), gy.ravel()])
    vals = kde(grid).reshape(gx.shape)

    vals = np.log1p(vals)
    vmax = float(vals.max())
    if vmax <= 0:
        return
    vals /= vmax

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
# Label declutter
# =============================================================================

def _place_labels_decluttered(
    ax,
    nodes: List[str],
    pos: Dict[str, Tuple[float, float]],
    label_map: Dict[str, str],
    importance: Dict[str, float],
    style: VisualStyle,
    label_budget: int,
) -> None:
    """
    Draw labels for up to label_budget nodes, avoiding heavy overlap.

    This is an approximate screen-space declutter: it ranks nodes by importance,
    then accepts a new label only if its center is not too close (in data
    coordinates) to previously placed labels.
    """
    if label_budget <= 0 or not label_map:
        return

    # Sort candidate nodes by importance
    candidates = sorted(
        [n for n in nodes if n in label_map],
        key=lambda n: importance.get(n, 0.0),
        reverse=True,
    )[: max(label_budget * 3, label_budget)]

    placed: List[Tuple[float, float, float]] = []  # (x, y, r_data)
    n_labels = 0

    # Rough data-radius for label footprint (scaled by figure span)
    xs = np.array([pos[n][0] for n in nodes], float)
    ys = np.array([pos[n][1] for n in nodes], float)
    span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1e-9)

    base_radius = span * 0.012 * (style.label_primary_size / 10.0)

    for n in candidates:
        if n_labels >= label_budget:
            break
        x, y = pos[n]
        r_here = base_radius

        # Check collision against previously placed labels
        too_close = False
        for px, py, pr in placed:
            dx = x - px
            dy = y - py
            dist2 = dx * dx + dy * dy
            thresh = 0.7 * (r_here + pr)  # > 50% overlap threshold approx
            if dist2 < thresh * thresh:
                too_close = True
                break

        if too_close:
            continue

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
        txt.set_path_effects(
            [
                patheffects.Stroke(
                    linewidth=style.label_outline_width,
                    foreground=style.label_outline_color,
                ),
                patheffects.Normal(),
            ]
        )

        placed.append((x, y, r_here))
        n_labels += 1


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
) -> None:
    """
    Render a scientific-quality 2D snapshot of the Hilbert information graph.
    """

    if G.number_of_nodes() == 0:
        return

    # Nodes that actually have positions
    nodes = [n for n in G.nodes() if n in pos2d]
    if not nodes:
        return

    pos: Dict[str, Tuple[float, float]] = {n: pos2d[n] for n in nodes}

    # ----------------------------------------------------------
    # Canvas setup
    # ----------------------------------------------------------
    x_min, x_max, y_min, y_max = _frame_from_positions(pos)
    span = max(x_max - x_min, y_max - y_min, 1e-9)

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
    ax.grid(False)

    # ----------------------------------------------------------
    # Density / hull overlays (background)
    # ----------------------------------------------------------
    _draw_density_field(ax, pos, style)

    if cluster_info is not None and getattr(cluster_info, "cluster_ids", None):
        _draw_cluster_hulls(ax, pos, cluster_info.cluster_ids, style)

    # ----------------------------------------------------------
    # Edges (with length-aware alpha)
    # ----------------------------------------------------------
    if G.number_of_edges() > 0:
        widths = edge_styles.widths or {}
        ecolors = edge_styles.colors or {}
        ealphas = edge_styles.alphas or {}
        base_alpha = edge_styles.alpha
        if base_alpha is None:
            base_alpha = (
                style.edge_alpha_dense if G.number_of_nodes() > 700 else style.edge_alpha_sparse
            )

        for u, v in _depth_sorted_edges(G, pos):
            if u not in pos or v not in pos:
                continue

            key = (u, v)
            if key not in widths and (v, u) in widths:
                key = (v, u)

            width = float(
                widths.get(
                    key,
                    (style.edge_width_min + style.edge_width_max) * 0.5,
                )
            )

            c_key = (u, v)
            if c_key not in ecolors and (v, u) in ecolors:
                c_key = (v, u)
            r, g, b, a0 = _rgba_tuple(ecolors.get(c_key, style.edge_color))

            alpha_edge = float(ealphas.get(key, 1.0))

            x0, y0 = pos[u]
            x1, y1 = pos[v]
            dx = x1 - x0
            dy = y1 - y0
            length = float((dx * dx + dy * dy) ** 0.5)
            length_n = np.clip(length / span, 0.0, 1.0)

            # Long, weak edges get extra fade
            length_scale = 0.45 + 0.55 * (1.0 - length_n)

            alpha = float(
                np.clip(a0 * alpha_edge * base_alpha * length_scale, 0.0, 1.0)
            )

            ax.plot(
                [x0, x1],
                [y0, y1],
                color=(r, g, b, alpha),
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
    halo_colors[:, 3] = np.clip(
        halo_colors[:, 3] * style.node_halo_alpha, 0.0, 1.0
    )

    # Slightly compress size dynamic range for visual stability
    core_sizes = np.power(sizes, 0.92)
    halo_sizes = core_sizes * halos * style.node_halo_scale

    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]

    # Halos
    ax.scatter(
        xs,
        ys,
        s=halo_sizes,
        c=halo_colors,
        marker="o",
        linewidths=0.0,
        zorder=2,
    )

    # Cores
    core_colors = colors.copy()
    core_colors[:, 3] = np.clip(
        core_colors[:, 3] * style.node_alpha, 0.0, 1.0
    )

    ax.scatter(
        xs,
        ys,
        s=core_sizes,
        c=core_colors,
        marker="o",
        edgecolors=style.node_edge_color,
        linewidths=0.3,
        zorder=3,
    )

    # ----------------------------------------------------------
    # Labels (decluttered, halo text)
    # ----------------------------------------------------------
    label_map = node_styles.primary_labels or {}
    if label_budget > 0 and label_map:
        importance = {n: node_styles.sizes.get(n, 0.0) for n in nodes}
        _place_labels_decluttered(
            ax,
            nodes,
            pos,
            label_map,
            importance,
            style,
            label_budget,
        )

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
    plt.close(fig)
