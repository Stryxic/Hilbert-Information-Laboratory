# =============================================================================
# graph_export.py - Hilbert Graph Snapshot Exporter (Refined + Labeled)
# =============================================================================
"""
Generates static PNG exports of the informational graph at multiple stages.

Outputs to results/hilbert_run/figures/:
    graph_full.png
    graph_100.png
    graph_200.png
    graph_500.png
    graph_<N>.png (adaptive depending on total elements)

Design choices:
  - Hybrid layout for more stable, readable structure.
  - A single global layout is reused across all snapshots so shapes are
    comparable as the graph "grows".
  - Node size scales with a mix of coherence, tf, and degree.
  - Node color encodes entropy and coherence:
        - red/magenta component increases with entropy
        - green component increases with coherence
        - blue component decreases with entropy
  - Each figure has:
        - informative title
        - legend and key explaining colors, sizes, and edges
        - labels for the most important nodes (top by size)
"""

import os
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List, Tuple

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import networkx as nx

DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda *_: None


# -------------------------------------------------------------------------#
# Helpers
# -------------------------------------------------------------------------#
def _log(msg: str, emit: Callable[[str, Dict[str, Any]], None]) -> None:
    print(msg)
    try:
        emit("log", {"message": msg})
    except Exception:
        pass


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


# -------------------------------------------------------------------------#
# Graph construction
# -------------------------------------------------------------------------#
def _load_graph(results_dir: str, emit=DEFAULT_EMIT):
    """
    Load hilbert_elements.csv + edges.csv and assemble a NetworkX graph.

    Node attributes:
        - entropy (mean_entropy or entropy)
        - coherence (mean_coherence or coherence)
        - tf (optional term frequency proxy)
    Edge attributes:
        - weight (strength of relation)
    """
    el_path = os.path.join(results_dir, "hilbert_elements.csv")
    edges_path = os.path.join(results_dir, "edges.csv")

    if not os.path.exists(el_path) or not os.path.exists(edges_path):
        _log("[graph] Missing hilbert_elements.csv or edges.csv", emit)
        return None, None, None

    elements = pd.read_csv(el_path)
    edges = pd.read_csv(edges_path)

    if elements.empty or edges.empty:
        _log("[graph] Empty elements or edges. Skipping graph export.", emit)
        return None, None, None

    # Normalisation
    elements["element"] = elements["element"].astype(str)
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)

    # Build graph and attach basic attributes
    G = nx.Graph()
    for _, row in elements.iterrows():
        el = str(row["element"])
        G.add_node(
            el,
            entropy=_safe_float(row.get("mean_entropy", row.get("entropy", 0.0))),
            coherence=_safe_float(row.get("mean_coherence", row.get("coherence", 0.0))),
            tf=_safe_float(row.get("tf", 1.0)),
        )

    for _, row in edges.iterrows():
        w = _safe_float(row.get("weight", 0.0))
        if w <= 0.0:
            continue
        s = str(row["source"])
        t = str(row["target"])
        if s == t:
            continue
        if s not in G.nodes or t not in G.nodes:
            continue
        G.add_edge(s, t, weight=w)

    if G.number_of_nodes() == 0:
        _log("[graph] Graph has no nodes after filtering; skipping.", emit)
        return None, None, None

    return G, elements, edges


# -------------------------------------------------------------------------#
# Layout and visual helpers
# -------------------------------------------------------------------------#
def _compute_layout(G: nx.Graph) -> Dict[str, Tuple[float, float]]:
    """
    Hybrid layout:

      1) spectral_layout as a coarse embedding
      2) spring_layout to refine using the spectral positions

    This usually gives a compact, less noisy structure than a raw layout.
    """
    n = max(G.number_of_nodes(), 1)
    try:
        base_pos = nx.spectral_layout(G, dim=2)
    except Exception:
        base_pos = None

    k = 0.8 / np.sqrt(n)
    if base_pos is not None:
        pos = nx.spring_layout(G, k=k, iterations=80, weight="weight", pos=base_pos)
    else:
        pos = nx.spring_layout(G, k=k, iterations=80, weight="weight")

    return pos


def _camera_frame(
    pos: Dict[str, Tuple[float, float]]
) -> Tuple[float, float, float, float]:
    """
    Compute a tight camera frame using percentile cropping to avoid extreme
    outliers dominating the canvas.
    """
    if not pos:
        return -1.0, 1.0, -1.0, 1.0

    xs = np.array([p[0] for p in pos.values()], dtype=float)
    ys = np.array([p[1] for p in pos.values()], dtype=float)
    if xs.size == 0:
        return -1.0, 1.0, -1.0, 1.0

    x_min, x_max = np.percentile(xs, [2, 98])
    y_min, y_max = np.percentile(ys, [2, 98])

    dx = (x_max - x_min) * 0.1 + 1e-9
    dy = (y_max - y_min) * 0.1 + 1e-9

    return float(x_min - dx), float(x_max + dx), float(y_min - dy), float(y_max + dy)


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Simple 0-1 normalisation, robust to constant arrays."""
    if arr.size == 0:
        return arr
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn + 1e-9)


def _compute_node_sizes_and_colors(
    G: nx.Graph,
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float, float]]]:
    """
    Node size is a mix of coherence, tf and degree.
    Node color is a two-channel mapping of coherence and entropy.

    Returns:
        size_map:  node -> marker size
        color_map: node -> (r, g, b) tuple
    """
    nodes = list(G.nodes())
    if not nodes:
        return {}, {}

    deg = np.array([float(G.degree(n)) for n in nodes], dtype=float)
    coh = np.array(
        [_safe_float(G.nodes[n].get("coherence", 0.0)) for n in nodes], dtype=float
    )
    ent = np.array(
        [_safe_float(G.nodes[n].get("entropy", 0.0)) for n in nodes], dtype=float
    )
    tf = np.array(
        [_safe_float(G.nodes[n].get("tf", 1.0)) for n in nodes], dtype=float
    )

    nd = _normalise(deg)
    nc = _normalise(coh)
    nt = _normalise(tf)
    ne = _normalise(ent)

    size_map: Dict[str, float] = {}
    color_map: Dict[str, Tuple[float, float, float]] = {}

    for i, n in enumerate(nodes):
        # Mix emphasises coherence and tf, with some degree
        score = 0.4 * nc[i] + 0.35 * nt[i] + 0.25 * nd[i]
        size = 26.0 + 260.0 * (score**1.2)
        size_map[n] = float(size)

        # Colors: coherence on green, entropy on magenta, low entropy closer to blue.
        g = nc[i]
        r = 0.30 + 0.60 * ne[i]
        b = 0.50 + 0.40 * (1.0 - ne[i])
        color_map[n] = (r, max(0.1, g), b)

    return size_map, color_map


def _annotate_top_nodes(
    ax: plt.Axes,
    pos: Dict[str, Tuple[float, float]],
    importance: Dict[str, float],
    top_k: int = 12,
) -> None:
    """
    Annotate the top-k nodes by importance (size proxy) with small labels.
    """
    if not importance:
        return

    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    for node, _score in items:
        if node not in pos:
            continue
        x, y = pos[node]
        ax.text(
            x,
            y,
            node,
            fontsize=6,
            color="#c9d1d9",
            alpha=0.75,
            ha="center",
            va="center",
        )


def _add_legend(ax: plt.Axes) -> None:
    """
    Add a compact legend / key describing node colors, sizes, and edges.
    """
    # representative colors used in the legend only
    high_entropy_color = (0.9, 0.3, 0.6)
    stable_color = (0.3, 0.9, 0.7)

    handles: List[Any] = [
        Patch(
            facecolor=stable_color,
            edgecolor="none",
            label="Stable cluster (high coherence, lower entropy)",
        ),
        Patch(
            facecolor=high_entropy_color,
            edgecolor="none",
            label="Unstable cluster (high entropy)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=6,
            linestyle="None",
            markerfacecolor="#d0d7de",
            markeredgecolor="#0d1117",
            label="Node size ~ importance (coherence, tf, degree)",
        ),
        Line2D(
            [0],
            [0],
            color="#5b4c8a",
            linewidth=1.4,
            label="Edge weight (stronger = thicker)",
        ),
    ]

    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=7,
        frameon=False,
        borderpad=0.4,
        handletextpad=0.4,
    )


# -------------------------------------------------------------------------#
# Graph rendering
# -------------------------------------------------------------------------#
def _render_graph(
    G: nx.Graph,
    outfile: str,
    emit=DEFAULT_EMIT,
    pos: Optional[Dict[str, Tuple[float, float]]] = None,
    node_sizes: Optional[Dict[str, float]] = None,
    node_colors: Optional[Dict[str, Tuple[float, float, float]]] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    annotate_top_k: int = 12,
) -> None:
    """Render the graph as a static PNG using a refined layout and legend."""

    if G is None or G.number_of_nodes() == 0:
        return

    _log(f"[graph] Rendering {outfile}", emit)

    nodes = list(G.nodes())
    if not nodes:
        return

    if pos is None:
        pos = _compute_layout(G)

    # restrict pos to nodes present in this subgraph
    pos = {n: pos[n] for n in nodes if n in pos}
    if not pos:
        pos = _compute_layout(G)

    # compute styling if missing, but treat them as node -> value maps
    if node_sizes is None or node_colors is None:
        node_sizes, node_colors = _compute_node_sizes_and_colors(G)

    # node-specific lists matching the node order in this subgraph
    sizes = [node_sizes.get(n, 40.0) for n in nodes]
    colors = [node_colors.get(n, (0.6, 0.6, 0.9)) for n in nodes]

    # camera framing
    x_min, x_max, y_min, y_max = _camera_frame(pos)

    # edge widths from weights
    weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    if weights:
        w_arr = np.array([_safe_float(w, 1.0) for w in weights], dtype=float)
        w_arr = _normalise(w_arr)
        widths = 0.2 + 1.8 * w_arr
    else:
        widths = []

    # edge alpha tuned to density
    n_nodes = len(nodes)
    edge_alpha = 0.08 if n_nodes > 600 else 0.16

    # figure
    plt.figure(figsize=(11.0, 8.0), facecolor="#050713")
    ax = plt.gca()
    ax.set_facecolor("#050713")
    ax.set_axis_off()

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # edges
    nx.draw_networkx_edges(
        G,
        pos,
        width=widths if len(widths) > 0 else 0.4,
        edge_color="#5b4c8a",
        alpha=edge_alpha,
    )

    # halo layer for nodes
    halo_sizes = [s * 1.6 for s in sizes]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes,
        node_size=halo_sizes,
        node_color=colors,
        linewidths=0.0,
        alpha=0.18,
    )

    # main node layer
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes,
        node_size=sizes,
        node_color=colors,
        linewidths=0.4,
        edgecolors="#050713",
        alpha=0.96,
    )

    # titles
    if title:
        ax.set_title(
            title,
            fontsize=13,
            color="#e6edf3",
            pad=12,
            loc="left",
        )
    if subtitle:
        ax.text(
            0.01,
            0.96,
            subtitle,
            transform=ax.transAxes,
            fontsize=8,
            color="#8b949e",
            ha="left",
            va="top",
        )

    # legend / key
    _add_legend(ax)

    # annotate top nodes
    _annotate_top_nodes(ax, pos, {n: node_sizes.get(n, 0.0) for n in nodes}, top_k=annotate_top_k)

    plt.tight_layout(pad=0.4)
    plt.savefig(outfile, dpi=260)
    plt.close()


# -------------------------------------------------------------------------#
# Stage snapshots
# -------------------------------------------------------------------------#
def _stage_cut_sequence(n_total: int) -> List[int]:
    """Generate stage cutoffs dynamically based on element count."""
    cuts: List[int] = []

    if n_total > 50:
        cuts.append(100)

    for c in [200, 300, 500]:
        if n_total > c:
            cuts.append(c)

    if n_total > 1000:
        cuts.extend([1000, 1500, 2000])

    return cuts


def export_graph_snapshots(results_dir: str, emit=DEFAULT_EMIT) -> None:
    """
    Entry point for orchestrator.

    - Loads the full informational graph from results_dir.
    - Computes a stable, global layout and styling.
    - Renders:
        - graph_full.png
        - graph_<cut>.png for each stage cutoff.
    """
    out_fig = Path(results_dir) / "figures"
    out_fig.mkdir(exist_ok=True)

    G, elements, edges = _load_graph(results_dir, emit)
    if G is None:
        return

    n_total = G.number_of_nodes()
    _log(f"[graph] Total nodes: {n_total}", emit)

    # global layout and styling computed once
    pos = _compute_layout(G)
    node_sizes, node_colors = _compute_node_sizes_and_colors(G)

    # full graph
    full_png = out_fig / "graph_full.png"
    full_title = "Hilbert information graph - full"
    full_subtitle = f"{n_total} elements, {G.number_of_edges()} bonds"
    _render_graph(
        G,
        str(full_png),
        emit=emit,
        pos=pos,
        node_sizes=node_sizes,
        node_colors=node_colors,
        title=full_title,
        subtitle=full_subtitle,
        annotate_top_k=14,
    )

    # stage snapshots in order of node importance (degree + coherence + tf, via node_sizes)
    ordered_nodes = list(G.nodes())
    ordered_nodes.sort(key=lambda n: node_sizes.get(n, 0.0), reverse=True)

    cuts = _stage_cut_sequence(n_total)

    for cutoff in cuts:
        sub_nodes = ordered_nodes[:cutoff]
        H = G.subgraph(sub_nodes).copy()
        out = out_fig / f"graph_{cutoff}.png"
        title = f"Hilbert information graph - top {cutoff} elements"
        subtitle = f"Top {cutoff} elements by importance; total {n_total}"
        _render_graph(
            H,
            str(out),
            emit=emit,
            pos=pos,
            node_sizes=node_sizes,
            node_colors=node_colors,
            title=title,
            subtitle=subtitle,
            annotate_top_k=min(12, cutoff),
        )

    # register artifacts
    try:
        emit("artifact", {"path": str(full_png), "kind": "graph_full"})
        for cutoff in cuts:
            emit(
                "artifact",
                {
                    "path": str(out_fig / f"graph_{cutoff}.png"),
                    "kind": "graph_stage",
                },
            )
    except Exception:
        pass

    _log("[graph] Graph snapshots complete.", emit)
