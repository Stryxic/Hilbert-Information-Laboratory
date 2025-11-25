# render3d.py

"""
3D holographic snapshot renderer for the Hilbert graph visualiser.

Responsibilities:
    - take a prepared NetworkX graph + 3D positions
    - use NodeStyleMaps / EdgeStyleMaps and VisualStyle
    - render a depth-aware "holographic shell" PNG

Layout geometry is assumed to be handled upstream by layout3d; here we:
    - apply mild importance-based radial warping
    - apply stability/community-based z-jitter to avoid flat rings
    - use depth to attenuate colours and edge opacity (depth fog)
    - use degree-based "ambient occlusion" to darken dense cores
    - apply simple directional lighting to enhance shape perception
    - standardise camera and framing
"""

from __future__ import annotations

from typing import Dict, Tuple, Iterable

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection
import matplotlib.patheffects as patheffects
import networkx as nx
import numpy as np

from .presets import VisualStyle
from .styling import NodeStyleMaps, EdgeStyleMaps


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def _frame_from_positions_3d(
    pos: Dict[str, Tuple[float, float, float]],
    margin: float = 0.08,
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute a padded 3D bounding box from node coordinates.
    """
    xs = np.array([p[0] for p in pos.values()], float)
    ys = np.array([p[1] for p in pos.values()], float)
    zs = np.array([p[2] for p in pos.values()], float)

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    z_min, z_max = float(zs.min()), float(zs.max())

    dx = max(x_max - x_min, 1e-9)
    dy = max(y_max - y_min, 1e-9)
    dz = max(z_max - z_min, 1e-9)
    span = max(dx, dy, dz)

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    cz = 0.5 * (z_min + z_max)

    pad = span * margin

    return (
        cx - span / 2.0 - pad,
        cx + span / 2.0 + pad,
        cy - span / 2.0 - pad,
        cy + span / 2.0 + pad,
        cz - span / 2.0 - pad,
        cz + span / 2.0 + pad,
    )


def _depth_sorted_edges_3d(
    G: nx.Graph,
    pos: Dict[str, Tuple[float, float, float]],
    azim: float,
    elev: float,
) -> Iterable[Tuple[str, str, float]]:
    """
    Sort edges back-to-front along the current camera direction.

    We approximate depth as the projection of the midpoint onto the
    camera vector derived from (elev, azim).
    """
    # Camera direction in data space (unit vector)
    ea = np.deg2rad(elev)
    aa = np.deg2rad(azim)
    cam_dir = np.array(
        [
            np.cos(ea) * np.cos(aa),
            np.cos(ea) * np.sin(aa),
            np.sin(ea),
        ],
        float,
    )

    edges = []
    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        mid = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5], float)
        depth = float(np.dot(mid, cam_dir))
        edges.append((depth, u, v))

    edges.sort(key=lambda t: t[0], reverse=True)  # furthest first
    for depth, u, v in edges:
        yield u, v, depth


def _rgba_tuple(c):
    """Normalise a color mapping entry into an (r,g,b,a) tuple."""
    if c is None:
        return 0.7, 0.8, 0.9, 1.0
    if isinstance(c, (list, tuple)):
        if len(c) == 3:
            return float(c[0]), float(c[1]), float(c[2]), 1.0
        if len(c) >= 4:
            return (
                float(c[0]),
                float(c[1]),
                float(c[2]),
                float(c[3]),
            )
    return 0.7, 0.8, 0.9, 1.0


def _norm01(a: np.ndarray) -> np.ndarray:
    """Safe normalisation to [0,1]."""
    if a.size == 0:
        return a
    lo = float(a.min())
    hi = float(a.max())
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


# --------------------------------------------------------------------------- #
# Main 3D renderer
# --------------------------------------------------------------------------- #


def draw_3d_snapshot(
    G: nx.Graph,
    pos3d: Dict[str, Tuple[float, float, float]],
    node_styles: NodeStyleMaps,
    edge_styles: EdgeStyleMaps,
    style: VisualStyle,
    *,
    outfile: str,
    title: str = "",
    subtitle: str = "",
    label_budget: int = 12,
) -> None:
    """
    Render a 3D holographic snapshot of the Hilbert information graph.

    Parameters mirror draw_2d_snapshot, but with 3D positions.
    """
    if G.number_of_nodes() == 0:
        return

    nodes = [n for n in G.nodes() if n in pos3d]
    if not nodes:
        return

    pos = {n: pos3d[n] for n in nodes}

    # Base coordinates as array, for simple radial warping
    P = np.array([[pos[n][0], pos[n][1], pos[n][2]] for n in nodes], float)

    # Importance from style.sizes → used for inner/outer shell
    size_map = node_styles.sizes or {}
    size_arr = np.array([float(size_map.get(n, 40.0)) for n in nodes], float)
    s_min, s_max = float(size_arr.min()), float(size_arr.max())
    s_span = max(s_max - s_min, 1e-9)
    importance = (size_arr - s_min) / s_span  # 0..1

    # Mild radial warp: important nodes slightly drawn inward
    centroid = P.mean(axis=0)
    P_shift = P - centroid
    radii = np.linalg.norm(P_shift, axis=1)
    directions = np.zeros_like(P_shift)
    mask = radii > 1e-9
    directions[mask] = P_shift[mask] / radii[mask, None]

    inner_scale = 0.8
    outer_scale = 1.4
    scale = outer_scale - (outer_scale - inner_scale) * importance
    scale = np.clip(scale, 0.7, 1.7)
    radii_new = radii * scale
    P_new = centroid + directions * radii_new[:, None]

    # ------------------------------------------------------------------ #
    # Stability / community-based z-jitter to avoid flat rings
    # ------------------------------------------------------------------ #
    # 1) Try stability / temperature
    stab_vals = []
    for n in nodes:
        d = G.nodes[n]
        if "stability" in d:
            stab_vals.append(float(d.get("stability", 0.0)))
        elif "temperature" in d:
            stab_vals.append(float(d.get("temperature", 0.0)))
        else:
            stab_vals.append(0.5)
    stab_arr = np.array(stab_vals, float)
    stab_norm = _norm01(stab_arr)

    # 2) Fallback / blend with community id to get structured bands
    comm_ids = [G.nodes[n].get("community_id") for n in nodes]
    if any(c is not None for c in comm_ids):
        # map community ids deterministically to [-0.5, 0.5]
        uniq = sorted({str(c) for c in comm_ids if c is not None})
        if uniq:
            comm_map = {cid: i / max(1, len(uniq) - 1 or 1) for i, cid in enumerate(uniq)}
            comm_vals = np.array(
                [comm_map.get(str(c), 0.5) if c is not None else 0.5 for c in comm_ids],
                float,
            )
            comm_centered = comm_vals - 0.5
        else:
            comm_centered = np.zeros_like(stab_norm)
    else:
        comm_centered = np.zeros_like(stab_norm)

    # Blend stability and community for jitter sign & magnitude
    jitter_driver = 0.6 * (stab_norm - 0.5) + 0.4 * comm_centered

    # Scale jitter by vertical span so it is layout-scale aware but gentle
    z_span = max(float(P_new[:, 2].max() - P_new[:, 2].min()), 1e-9)
    jitter_scale = 0.15 * z_span  # ~15% of vertical span
    z_jitter = jitter_scale * jitter_driver

    P_new[:, 2] = P_new[:, 2] + z_jitter

    # Write back warped + jittered positions
    for i, n in enumerate(nodes):
        x, y, z = P_new[i]
        pos[n] = (float(x), float(y), float(z))

    # Camera framing from warped positions
    x_min, x_max, y_min, y_max, z_min, z_max = _frame_from_positions_3d(pos)

    # ------------------------------------------------------------------ #
    # Figure + axes
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(18, 14), facecolor=style.background_color)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(style.background_color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_axis_off()

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # keep box roughly cubic
    try:
        ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))
    except Exception:
        pass

    elev = getattr(style, "camera_elev", 22.0)
    azim = getattr(style, "camera_azim", 38.0)
    try:
        ax.set_proj_type(getattr(style, "camera_projection", "persp"))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)

    # ------------------------------------------------------------------ #
    # Depth helpers and lighting
    # ------------------------------------------------------------------ #
    # Recompute projected depth for each node
    ea = np.deg2rad(elev)
    aa = np.deg2rad(azim)
    cam_dir = np.array(
        [
            np.cos(ea) * np.cos(aa),
            np.cos(ea) * np.sin(aa),
            np.sin(ea),
        ],
        float,
    )

    node_depth: Dict[str, float] = {}
    for n in nodes:
        x, y, z = pos[n]
        node_depth[n] = float(np.dot(np.array([x, y, z], float), cam_dir))

    depths = np.array([node_depth[n] for n in nodes], float)
    d_min, d_max = float(depths.min()), float(depths.max())
    d_span = max(d_max - d_min, 1e-9)
    depth_norm = (depths - d_min) / d_span  # 0 = near, 1 = far

    # Degree-based "ambient occlusion" factor (high-degree = more occluded)
    deg_arr = np.array([float(G.degree(n)) for n in nodes], float)
    deg_norm = _norm01(deg_arr)
    occlusion = deg_norm  # 0 = isolated, 1 = very dense core

    # Simple directional lighting: light from top-front-right
    light_dir = np.array([0.6, 0.4, 0.7], float)
    light_dir /= np.linalg.norm(light_dir) + 1e-12
    node_pos_vecs = np.array([[pos[n][0], pos[n][1], pos[n][2]] for n in nodes], float)
    node_pos_norm = node_pos_vecs / (
        np.linalg.norm(node_pos_vecs, axis=1, keepdims=True) + 1e-12
    )
    lambert = np.clip(
        np.sum(node_pos_norm * light_dir[None, :], axis=1),
        -1.0,
        1.0,
    )
    # convert to a [0.7, 1.1] brightness factor
    light_factor = 0.7 + 0.4 * (lambert * 0.5 + 0.5)

    # ------------------------------------------------------------------ #
    # Edges with depth-aware alpha and mild occlusion
    # ------------------------------------------------------------------ #
    width_map = edge_styles.widths or {}
    edge_color_map = edge_styles.colors or {}
    base_alpha = edge_styles.alpha
    if base_alpha is None:
        base_alpha = (
            style.edge_alpha_dense if len(nodes) > 700 else style.edge_alpha_sparse
        )

    # Precompute index lookups to avoid repeated list.index calls
    idx_map = {n: i for i, n in enumerate(nodes)}

    for u, v, depth in _depth_sorted_edges_3d(G, pos, azim=azim, elev=elev):
        if u not in pos or v not in pos:
            continue

        # width
        key = (u, v)
        if key not in width_map and (v, u) in width_map:
            key = (v, u)
        width = float(
            width_map.get(
                key,
                (style.edge_width_min + style.edge_width_max) * 0.5,
            )
        )

        # color
        c_key = (u, v)
        if c_key not in edge_color_map and (v, u) in edge_color_map:
            c_key = (v, u)
        r, g, b, a0 = _rgba_tuple(edge_color_map.get(c_key, style.edge_color))

        # depth fog: farther edges are fainter
        depth_norm_edge = (depth - d_min) / d_span
        depth_fog = 0.35 + 0.65 * (1.0 - depth_norm_edge)

        # occlusion based on endpoints' degree
        iu = idx_map.get(u, None)
        iv = idx_map.get(v, None)
        oc_u = occlusion[iu] if iu is not None else 0.0
        oc_v = occlusion[iv] if iv is not None else 0.0
        oc_edge = 0.5 * (oc_u + oc_v)
        ao_factor = 0.4 + 0.6 * (1.0 - oc_edge)  # high-degree edges darker/fainter

        alpha = float(np.clip(a0 * base_alpha * depth_fog * ao_factor, 0.0, 1.0))

        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        ax.plot(
            [x0, x1],
            [y0, y1],
            [z0, z1],
            linewidth=width,
            color=(r, g, b, alpha),
            solid_capstyle="round",
            zorder=1,
        )

    # ------------------------------------------------------------------ #
    # Nodes: halo + core with depth-aware alpha, occlusion, and lighting
    # ------------------------------------------------------------------ #
    halo_map = node_styles.halo_sizes or {}
    color_map = node_styles.colors or {}

    raw_sizes = size_arr
    rs_min, rs_max = float(raw_sizes.min()), float(raw_sizes.max())
    rs_span = max(rs_max - rs_min, 1e-9)
    size_norm = (raw_sizes - rs_min) / rs_span

    core_sizes = 14.0 + 26.0 * (size_norm ** 0.9)
    halo_sizes = []
    for i, n in enumerate(nodes):
        if halo_map:
            halo_sizes.append(
                float(halo_map.get(n, core_sizes[i] * style.node_halo_scale))
            )
        else:
            halo_sizes.append(core_sizes[i] * style.node_halo_scale)
    halo_sizes = np.asarray(halo_sizes, float)

    cols = np.array([_rgba_tuple(color_map.get(n)) for n in nodes], float)

    # depth-aware fade + ambient occlusion for nodes
    depth_fog_nodes = 0.3 + 0.7 * (1.0 - depth_norm)
    ao_nodes = 0.4 + 0.6 * (1.0 - occlusion)  # dense core slightly darkened
    fog = depth_fog_nodes * ao_nodes

    # Apply lighting as a mild brightness modulation on RGB only
    rgb = cols[:, :3]
    alpha_base = cols[:, 3]

    rgb_lit = rgb * light_factor[:, None]
    rgb_lit = np.clip(rgb_lit, 0.0, 1.0)

    halo_colors = np.concatenate(
        [
            rgb_lit,
            (alpha_base * fog * style.node_halo_alpha)[:, None],
        ],
        axis=1,
    )
    halo_colors[:, 3] = np.clip(halo_colors[:, 3], 0.0, 1.0)

    core_colors = np.concatenate(
        [
            rgb_lit,
            (alpha_base * fog * style.node_alpha)[:, None],
        ],
        axis=1,
    )
    core_colors[:, 3] = np.clip(core_colors[:, 3], 0.0, 1.0)

    xs = np.array([pos[n][0] for n in nodes], float)
    ys = np.array([pos[n][1] for n in nodes], float)
    zs = np.array([pos[n][2] for n in nodes], float)

    # Draw halos slightly behind cores
    ax.scatter(
        xs,
        ys,
        zs,
        s=halo_sizes,
        c=halo_colors,
        marker="o",
        linewidths=0.0,
        depthshade=False,
        zorder=3,
    )

    ax.scatter(
        xs,
        ys,
        zs,
        s=core_sizes,
        c=core_colors,
        marker="o",
        linewidths=0.0,
        edgecolors=style.node_edge_color,
        depthshade=False,
        zorder=4,
    )

    # ------------------------------------------------------------------ #
    # Labels (budget-limited, drawn for nearer / larger nodes)
    # ------------------------------------------------------------------ #
    label_map = node_styles.primary_labels or {}

    if label_budget > 0 and label_map:
        # combine importance (size) and nearness for label ranking
        rank_score: Dict[str, float] = {}
        for i, n in enumerate(nodes):
            score = 0.7 * size_norm[i] + 0.3 * (1.0 - depth_norm[i])
            rank_score[n] = float(score)

        candidates = sorted(
            [n for n in nodes if n in label_map],
            key=lambda n: rank_score.get(n, 0.0),
            reverse=True,
        )[:label_budget]

        for n in candidates:
            x, y, z = pos[n]
            text = str(label_map.get(n, n))
            txt = ax.text(
                x,
                y,
                z,
                text,
                fontsize=style.label_primary_size,
                color=style.label_color,
                ha="center",
                va="center",
                zorder=6,
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

    # ------------------------------------------------------------------ #
    # Titles / subtitles
    # ------------------------------------------------------------------ #
    if title:
        ax.set_title(
            title,
            fontsize=16,
            color=style.label_color,
            loc="left",
            pad=14,
        )
    if subtitle:
        fig.text(0.01, 0.965, subtitle, color="#bfc5ce", fontsize=9, ha="left")

    plt.tight_layout(pad=0.5)
    plt.savefig(outfile, dpi=style.dpi, facecolor=style.background_color)
    plt.close()
