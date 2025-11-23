"""
3D rendering of Hilbert graphs (electron-cloud v6 - alpha-safe, holographic shell).

Features
--------
- Electron-cloud tiny-point 3D rendering
- Volume stabilisation:
    * detect collapsed / cigar-shaped layouts
    * inject small jitter and whiten axes to use full 3D volume
- Optional LSA integration:
    * if per-node LSA coordinates exist, blend them into the layout
- Importance-centred radial warp:
    * high-importance nodes pulled toward the weighted centroid
    * low-importance nodes gently pushed outward to the periphery
- Depth-sorted nodes
- Depth fog
- Depth-sorted faint edges (filaments)
- Volumetric halos
- Stable camera framing
- Safe alpha bounds (never > 1.0)
"""

from __future__ import annotations
from typing import Dict, Tuple, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import networkx as nx
import numpy as np

from .styling import NodeStyleMaps, EdgeStyleMaps
from .presets import VisualStyle


# --------------------------------------------------------------------------- #
# Camera utilities
# --------------------------------------------------------------------------- #


def _camera_frame_3d(
    pos: Dict[str, Tuple[float, float, float]],
    margin_pct: float = 0.08,
) -> Tuple[float, float, float, float, float, float]:
    """Percentile-based 3D bounding cube with margin padding."""
    if not pos:
        return -1.0, 1.0, -1.0, 1.0, -1.0, 1.0

    xs = np.array([p[0] for p in pos.values()], float)
    ys = np.array([p[1] for p in pos.values()], float)
    zs = np.array([p[2] for p in pos.values()], float)

    def rng(a: np.ndarray) -> Tuple[float, float]:
        # Slightly tighter percentiles for static figures to avoid huge voids
        lo, hi = np.percentile(a, [5, 95])
        span = max(hi - lo, 1e-9)
        pad = span * margin_pct
        return float(lo - pad), float(hi + pad)

    x_min, x_max = rng(xs)
    y_min, y_max = rng(ys)
    z_min, z_max = rng(zs)
    return x_min, x_max, y_min, y_max, z_min, z_max


def _view_vector(elev: float, azim: float) -> np.ndarray:
    """Compute camera view vector from elev/azim (degrees)."""
    er, ar = np.deg2rad(elev), np.deg2rad(azim)
    return np.array(
        [
            np.cos(er) * np.sin(ar),
            np.sin(er),
            np.cos(er) * np.cos(ar),
        ],
        float,
    )


def _depth_sort_points(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    elev: float,
    azim: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Sort nodes by depth; return order, normalised depths, and raw min/max."""
    view = _view_vector(elev, azim)
    pts = np.stack([xs, ys, zs], axis=1)
    depths = pts @ view

    dmin = float(depths.min())
    dmax = float(depths.max())
    span = max(dmax - dmin, 1e-9)
    depths_norm = (depths - dmin) / span

    order = np.argsort(depths)  # furthest first
    return order, depths_norm, dmin, dmax


def _sort_edges_by_depth(
    G: nx.Graph,
    pos: Dict[str, Tuple[float, float, float]],
    elev: float,
    azim: float,
) -> Tuple[List[Tuple[str, str]], List[float]]:
    """Return edges sorted by depth (furthest first) plus their depths."""
    view = _view_vector(elev, azim)
    data: List[Tuple[Tuple[str, str], float]] = []

    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        mid = (np.array(pos[u], float) + np.array(pos[v], float)) / 2.0
        data.append(((u, v), float(mid @ view)))

    data.sort(key=lambda x: x[1])
    edges = [e for e, _ in data]
    depths = [d for _, d in data]
    return edges, depths


# --------------------------------------------------------------------------- #
# Volume stabilisation / holographic shell support
# --------------------------------------------------------------------------- #


def _maybe_blend_lsa_offsets(
    G: nx.Graph,
    nodes: List[str],
    base_coords: np.ndarray,
    strength: float = 0.25,
) -> np.ndarray:
    """
    Optionally blend LSA coordinates into the layout.

    If nodes carry attributes like `lsa_x/lsa_y/lsa_z` or `lsa0/lsa1/lsa2`,
    we use them as an additional semantic offset. If not present, this is a
    no-op.
    """
    lsa_vecs: List[np.ndarray] = []

    for n in nodes:
        d = G.nodes[n]
        if "lsa_x" in d and "lsa_y" in d and "lsa_z" in d:
            lsa_vecs.append(
                np.array(
                    [float(d["lsa_x"]), float(d["lsa_y"]), float(d["lsa_z"])],
                    float,
                )
            )
        elif "lsa0" in d and "lsa1" in d and "lsa2" in d:
            lsa_vecs.append(
                np.array(
                    [float(d["lsa0"]), float(d["lsa1"]), float(d["lsa2"])],
                    float,
                )
            )
        else:
            # No LSA coordinates for this node.
            lsa_vecs.append(np.zeros(3, float))

    L = np.stack(lsa_vecs, axis=0)
    if not np.any(L):
        return base_coords

    # Centre and normalise LSA vectors
    L -= L.mean(axis=0, keepdims=True)
    span = L.ptp(axis=0)
    span[span < 1e-9] = 1.0
    L /= span

    return base_coords + strength * L


def _stabilise_volume(
    coords: np.ndarray,
    jitter_scale: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """
    Ensure the 3D cloud uses the volumetric space.

    - Recentre to origin
    - Detect collapsed / narrow axes; add small jitter if necessary
    - Whiten axes so spans are comparable
    - Normalise radius so most points lie within unit sphere
    """
    if coords.size == 0:
        return coords

    rng = np.random.default_rng(seed)

    X = coords.copy().astype(float)
    centre = X.mean(axis=0)
    X -= centre

    spans = X.max(axis=0) - X.min(axis=0)
    max_span = float(spans.max())
    min_span = float(spans.min())

    if max_span < 1e-9:
        # Degenerate case: everything at a point
        X += rng.normal(scale=1.0, size=X.shape)
        spans = X.max(axis=0) - X.min(axis=0)
        max_span = float(spans.max())
        min_span = float(spans.min())

    # If highly anisotropic (needle or flat sheet), inject jitter
    if min_span < 0.12 * max_span:
        jitter = rng.normal(scale=jitter_scale * max_span, size=X.shape)
        X += jitter
        spans = X.max(axis=0) - X.min(axis=0)
        spans[spans < 1e-9] = max_span

    # Whiten axes so they occupy similar ranges
    spans = X.max(axis=0) - X.min(axis=0)
    spans[spans < 1e-9] = spans[spans >= 1e-9].max()
    X /= spans

    # Normalise radius to lie roughly in unit ball
    radii = np.linalg.norm(X, axis=1)
    r_max = float(np.max(radii))
    if r_max > 1e-9:
        X /= r_max

    return X


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def _draw_labels_3d(
    ax,
    G: nx.Graph,
    pos: Dict[str, Tuple[float, float, float]],
    node_styles: NodeStyleMaps,
    style: VisualStyle,
    label_budget: int,
) -> None:
    """Draw a small set of primary labels in 3D."""
    if label_budget <= 0:
        return

    primary = list(getattr(node_styles, "primary_labels", []))
    if not primary:
        sizes = getattr(node_styles, "sizes", {}) or {}
        if not sizes:
            return
        ordered = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)
        primary = [n for n, _ in ordered]

    primary = [n for n in primary if n in G.nodes() and n in pos][:label_budget]

    for n in primary:
        x, y, z = pos[n]
        ax.text(
            x,
            y,
            z,
            str(n),
            fontsize=style.label_primary_size,
            color=style.label_color,
            ha="center",
            va="center",
            zorder=30,
        )


# --------------------------------------------------------------------------- #
# Main renderer
# --------------------------------------------------------------------------- #


def draw_3d_snapshot(
    G: nx.Graph,
    pos3d: Dict[str, Tuple[float, float, float]],
    node_styles: NodeStyleMaps,
    edge_styles: EdgeStyleMaps,
    style: VisualStyle,
    outfile: str,
    title: str,
    subtitle: str,
    label_budget: int,
) -> None:
    """
    Render a single 3D PNG snapshot as a holographic, importance-centred
    electron cloud.

    - High-importance nodes (by node_styles.sizes) pulled toward centroid
    - Low-importance nodes pushed outward
    - Layout volume stabilised and optionally LSA-enhanced
    - Tiny points with halos and depth fog
    - Faint, depth-sorted edges
    """
    if G.number_of_nodes() == 0:
        return

    nodes = list(G.nodes())
    if not nodes:
        return

    # Filter positions to subgraph
    pos = {n: pos3d[n] for n in nodes if n in pos3d}
    if not pos:
        return

    # ------------------------------------------------------------------ #
    # Base coordinates as array
    # ------------------------------------------------------------------ #
    P = np.array([[pos[n][0], pos[n][1], pos[n][2]] for n in nodes], float)

    # Optional: blend in LSA offsets if present
    P = _maybe_blend_lsa_offsets(G, nodes, P, strength=0.25)

    # Stabilise volume (prevent 1D collapse)
    P = _stabilise_volume(P, jitter_scale=0.06, seed=42)

    # ------------------------------------------------------------------ #
    # Importance & simple density (for alpha boost)
    # ------------------------------------------------------------------ #
    size_map = node_styles.sizes or {}
    size_arr = np.array([float(size_map.get(n, 40.0)) for n in nodes], float)

    s_min = float(size_arr.min())
    s_max = float(size_arr.max())
    s_span = max(s_max - s_min, 1e-9)
    importance = (size_arr - s_min) / s_span  # 0..1

    # Degree as a proxy for local density
    deg_arr = np.array([float(G.degree(n)) for n in nodes], float)
    if deg_arr.size > 0:
        d_min, d_max = float(deg_arr.min()), float(deg_arr.max())
        d_span = max(d_max - d_min, 1e-9)
        density = (deg_arr - d_min) / d_span
    else:
        density = np.zeros_like(importance)

    # ------------------------------------------------------------------ #
    # Importance-based radial warp (semantically centred shell)
    # ------------------------------------------------------------------ #
    weights = importance + 1e-3
    weights = weights / weights.sum()
    centroid = (P * weights[:, None]).sum(axis=0)

    P_shift = P - centroid
    radii = np.linalg.norm(P_shift, axis=1)
    directions = np.zeros_like(P_shift)
    mask = radii > 1e-9
    directions[mask] = P_shift[mask] / radii[mask, None]

    # Shell scaling:
    #   importance=1   → radius compressed (inner shell)
    #   importance=0   → radius expanded (outer shell)
    inner_scale = 0.75
    outer_scale = 1.6
    scale = outer_scale - (outer_scale - inner_scale) * importance
    scale = np.clip(scale, 0.6, 2.0)

    radii_new = radii * scale
    P_new = centroid + directions * radii_new[:, None]

    # Write back warped positions
    for i, n in enumerate(nodes):
        pos[n] = (float(P_new[i, 0]), float(P_new[i, 1]), float(P_new[i, 2]))

    # Camera bounds use warped positions
    x_min, x_max, y_min, y_max, z_min, z_max = _camera_frame_3d(pos)

    # ------------------------------------------------------------------ #
    # Style maps and edge alpha
    # ------------------------------------------------------------------ #
    color_map = node_styles.colors
    widths_map = edge_styles.widths

    if edge_styles.alpha is not None:
        edge_alpha = float(edge_styles.alpha)
    else:
        edge_alpha = (
            style.edge_alpha_dense if len(nodes) > 600 else style.edge_alpha_sparse
        )
    edge_alpha = float(np.clip(edge_alpha, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # Node sizes (electron-cloud, but more generous for visibility)
    # ------------------------------------------------------------------ #
    raw = size_arr.copy()
    rmin, rmax = float(raw.min()), float(raw.max())
    span = max(rmax - rmin, 1e-9)
    norm = (raw - rmin) / span

    size_min, size_max = 4.0, 22.0
    sizes = size_min + (size_max - size_min) * (norm ** 0.9)
    halos = sizes * 1.9

    # ------------------------------------------------------------------ #
    # Node colours
    # ------------------------------------------------------------------ #
    def NC(c):
        if c is None:
            return (0.5, 0.6, 0.9, 1.0)
        if len(c) == 3:
            return (float(c[0]), float(c[1]), float(c[2]), 1.0)
        return (float(c[0]), float(c[1]), float(c[2]), float(c[3]))

    cols = np.array([NC(color_map.get(n)) for n in nodes], float)

    # Coordinates from warped positions
    xs = np.array([pos[n][0] for n in nodes], float)
    ys = np.array([pos[n][1] for n in nodes], float)
    zs = np.array([pos[n][2] for n in nodes], float)

    # ------------------------------------------------------------------ #
    # Depth sort nodes + global depth range
    # ------------------------------------------------------------------ #
    elev = float(getattr(style, "camera_elev", 25.0))
    azim = float(getattr(style, "camera_azim", 35.0))

    order, dnorm, dmin, dmax = _depth_sort_points(xs, ys, zs, elev, azim)

    xs, ys, zs = xs[order], ys[order], zs[order]
    sizes, halos = sizes[order], halos[order]
    cols = cols[order]
    dnorm = dnorm[order]
    density = density[order]

    # ------------------------------------------------------------------ #
    # Depth fog plus simple density-based alpha boost
    # ------------------------------------------------------------------ #
    fog_near, fog_far = 1.0, 0.22
    fog = fog_far + (fog_near - fog_far) * dnorm
    fog = np.clip(fog, 0.0, 1.0)

    # Dense regions glow a bit more
    dens_boost = 0.6 + 0.7 * density  # in [0.6, 1.3]
    dens_boost = np.clip(dens_boost, 0.4, 1.4)

    halo_colors = cols.copy()
    core_colors = cols.copy()

    halo_colors[:, 3] = np.clip(
        halo_colors[:, 3] * fog * dens_boost * style.node_halo_alpha,
        0.0,
        1.0,
    )
    core_colors[:, 3] = np.clip(
        core_colors[:, 3] * fog * dens_boost * style.node_alpha,
        0.0,
        1.0,
    )

    # ------------------------------------------------------------------ #
    # Figure
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

    try:
        ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))
    except Exception:
        pass

    try:
        ax.set_proj_type(getattr(style, "camera_projection", "persp"))
    except Exception:
        pass

    ax.view_init(elev=elev, azim=azim)

    # ------------------------------------------------------------------ #
    # Depth-sorted edges (α-safe)
    # ------------------------------------------------------------------ #
    sorted_edges, edge_depths = _sort_edges_by_depth(G, pos, elev, azim)

    edge_depths = np.array(edge_depths, float)
    if edge_depths.size > 0:
        e_min = float(edge_depths.min())
        e_max = float(edge_depths.max())
    else:
        e_min, e_max = 0.0, 1.0
    e_span = max(e_max - e_min, 1e-9)

    fog_near_e, fog_far_e = 1.0, 0.22  # reuse fog range for edges

    for (u, v), depth in zip(sorted_edges, edge_depths):
        if u not in pos or v not in pos:
            continue

        w = widths_map.get((u, v)) or widths_map.get((v, u))
        if w is None:
            w = (style.edge_width_min + style.edge_width_max) / 2.0

        depth_n = (depth - e_min) / e_span
        depth_n = float(np.clip(depth_n, 0.0, 1.0))

        efog = fog_far_e + (fog_near_e - fog_far_e) * depth_n
        efog = float(np.clip(efog, 0.0, 1.0))

        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            [pos[u][2], pos[v][2]],
            linewidth=float(w),
            color=style.edge_color,
            alpha=float(np.clip(edge_alpha * efog, 0.0, 1.0)),
            zorder=2,
        )

    # ------------------------------------------------------------------ #
    # Halos + core electrons
    # ------------------------------------------------------------------ #
    ax.scatter(
        xs,
        ys,
        zs,
        s=halos,
        c=halo_colors,
        edgecolors="none",
        depthshade=False,
        zorder=5,
    )
    ax.scatter(
        xs,
        ys,
        zs,
        s=sizes,
        c=core_colors,
        edgecolors=style.node_edge_color,
        linewidths=0.2,
        depthshade=False,
        zorder=10,
    )

    # ------------------------------------------------------------------ #
    # Labels & metadata
    # ------------------------------------------------------------------ #
    _draw_labels_3d(ax, G, pos, node_styles, style, label_budget)

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
    plt.savefig(outfile, dpi=style.dpi)
    plt.close()
