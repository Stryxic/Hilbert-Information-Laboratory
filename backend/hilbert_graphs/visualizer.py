"""
Unified Hilbert graph visualiser - v4.2 (cluster-aware, modular metadata).

This version integrates the modular metadata system (Option 2):
    - graph_metadata_core.json
    - graph_layout.json
    - graph_analytics.json
    - graph_diagnostics.json
    - graph_snapshots_index.json (lightweight pointer file)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Callable, List, Tuple
import os

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from .presets import VisualizerConfig, DEFAULT_CONFIG
from .loader import load_elements_and_edges, load_compound_data, build_graph
from .analytics import (
    compute_cluster_info,
    compute_graph_stats,
    filter_large_components,
)
from .layout.layout2d import compute_layout_2d_hybrid, compute_layout_2d_radial
from .layout.layout3d import compute_layout_3d_spherical
from .styling import compute_node_styles, compute_edge_styles
from .render2d import draw_2d_snapshot
from .render3d import draw_3d_snapshot

# Modular metadata writers
from .metadata import (
    write_snapshot_metadata,
    write_global_index,
    write_metadata_core,
    write_metadata_layout,
    write_metadata_analytics,
    write_metadata_diagnostics,
)

from .graph_state import GraphState


DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda *_: None


# ====================================================================== #
# Logging helpers
# ====================================================================== #

def _log_global(msg: str, emit: Callable[[str, Dict[str, Any]], None] | None) -> None:
    print(msg)
    if emit:
        try:
            emit("log", {"message": msg})
        except Exception:
            pass


def _safe_mkdir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _nodes_for_pct(pct: float | None, total: int) -> int:
    if pct is None:
        return total
    if pct >= 100.0:
        return total
    return max(1, int(round(total * pct / 100.0)))


# ====================================================================== #
# Diagnostics - unchanged from previous working version
# ====================================================================== #

def _plot_histogram_1d(
    data: np.ndarray,
    *,
    bins: int,
    title: str,
    xlabel: str,
    ylabel: str,
    outfile: str,
    style,
):
    if data.size == 0:
        return

    _safe_mkdir(os.path.dirname(outfile) or ".")
    bg = style.background_color
    fg = style.label_color
    dpi = style.dpi

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg)
    ax.set_facecolor(bg)

    ax.hist(
        data, bins=bins, color="#6fa8ff",
        alpha=0.9, edgecolor="#111318", linewidth=0.6,
    )

    ax.set_title(title, fontsize=13, color=fg)
    ax.set_xlabel(xlabel, fontsize=11, color=fg)
    ax.set_ylabel(ylabel, fontsize=11, color=fg)
    ax.tick_params(colors=fg)

    for spine in ax.spines.values():
        spine.set_color("#2a2e3a")

    fig.tight_layout(pad=0.8)
    plt.savefig(outfile, dpi=dpi, facecolor=bg)
    plt.close(fig)


def _render_cluster_diagnostics(G, cluster_info, style, results_dir) -> Dict[str, str]:
    """
    Generates:
      - graph_community_sizes.png
      - graph_community_degree_panels.png
    """
    out = {}
    if cluster_info is None or not cluster_info.community_ids:
        return out

    comm_to_nodes = {}
    for node, cid in cluster_info.community_ids.items():
        comm_to_nodes.setdefault(cid, []).append(node)

    # Histogram of community sizes
    sizes = np.array([len(v) for v in comm_to_nodes.values()], float)
    if sizes.size > 0:
        out_sizes = os.path.join(results_dir, "graph_community_sizes.png")
        _plot_histogram_1d(
            sizes, bins=min(40, max(8, int(np.sqrt(sizes.size)) + 3)),
            title="Community size distribution",
            xlabel="size", ylabel="count",
            outfile=out_sizes, style=style,
        )
        out["community_size_hist"] = out_sizes

    # Panel plots for top communities
    top = sorted(comm_to_nodes.items(), key=lambda kv: len(kv[1]), reverse=True)
    top_k = min(4, len(top))
    if top_k > 0:
        bg = style.background_color
        fg = style.label_color
        dpi = style.dpi

        fig, axes = plt.subplots(
            1, top_k, figsize=(4.2 * top_k, 4.2),
            sharey=True, facecolor=bg
        )
        if top_k == 1:
            axes = [axes]

        for ax, (cid, nodes) in zip(axes, top[:top_k]):
            degs = np.array([G.degree(n) for n in nodes], float)
            if degs.size == 0:
                continue
            ax.set_facecolor(bg)
            ax.hist(
                degs, bins=min(25, max(5, int(np.sqrt(degs.size)) + 2)),
                color="#6fa8ff", alpha=0.9,
                edgecolor="#111318", linewidth=0.6,
            )
            ax.set_title(f"Community {cid} (n={len(nodes)})",
                         fontsize=9, color=fg)
            ax.tick_params(colors=fg)
            for spine in ax.spines.values():
                spine.set_color("#2a2e3a")

        fig.suptitle("Top communities - degree distributions",
                     fontsize=12, color=fg)
        fig.tight_layout(rect=[0, 0.03, 1, 0.92])

        out_panels = os.path.join(results_dir, "graph_community_degree_panels.png")
        plt.savefig(out_panels, dpi=dpi, facecolor=bg)
        plt.close(fig)

        out["community_degree_panels"] = out_panels

    return out


def _render_diagnostics_from_state(state: GraphState) -> Dict[str, str]:
    """
    Produces:
      - degree hist
      - component sizes
      - stability / entropy hist
      - community diagnostics
    """
    G = state.G
    if G is None or G.number_of_nodes() == 0:
        return {}

    cfg = state.config
    style = cfg.style
    results_dir = state.results_dir

    out = {}

    # Degree hist
    degrees = np.array([d for _, d in G.degree()], float)
    if degrees.size > 0:
        out_deg = os.path.join(results_dir, "graph_degree_hist.png")
        _plot_histogram_1d(
            degrees,
            bins=min(60, max(15, int(np.sqrt(degrees.size)) + 5)),
            title="Degree distribution",
            xlabel="degree", ylabel="count",
            outfile=out_deg, style=style,
        )
        out["degree_hist"] = out_deg

    # Component sizes
    comp_sizes = (
        np.array(list(state.cluster_info.component_sizes.values()), float)
        if state.cluster_info
        else np.array([len(c) for c in nx.connected_components(G)], float)
    )
    if comp_sizes.size > 0:
        out_comp = os.path.join(results_dir, "graph_component_sizes.png")
        _plot_histogram_1d(
            comp_sizes,
            bins=min(40, max(10, int(np.sqrt(comp_sizes.size)) + 3)),
            title="Connected component sizes",
            xlabel="size", ylabel="count",
            outfile=out_comp, style=style,
        )
        out["component_hist"] = out_comp

    # Stability/entropy histogram
    stab_vals = []
    for n in G.nodes():
        d = G.nodes[n]
        if "stability" in d:
            stab_vals.append(float(d["stability"]))
        elif "temperature" in d:
            stab_vals.append(float(d["temperature"]))
        elif "entropy" in d:
            stab_vals.append(float(d["entropy"]))

    if stab_vals:
        arr = np.array(stab_vals, float)
        out_stab = os.path.join(results_dir, "graph_stability_hist.png")
        _plot_histogram_1d(
            arr,
            bins=min(50, max(12, int(np.sqrt(arr.size)) + 4)),
            title="Stability / Entropy distribution",
            xlabel="score", ylabel="count",
            outfile=out_stab, style=style,
        )
        out["stability_hist"] = out_stab

    # Community diagnostics
    out.update(
        _render_cluster_diagnostics(G, state.cluster_info, style, results_dir)
    )

    state.diagnostic_plots = out
    return out


# ====================================================================== #
# GraphState builder
# ====================================================================== #

def build_graph_state(
    results_dir: str,
    *,
    config: VisualizerConfig | None = None,
    emit: Callable[[str, Dict[str, Any]], None] = DEFAULT_EMIT,
) -> GraphState:

    cfg = config or DEFAULT_CONFIG
    cfg.ensure_defaults()

    state = GraphState(results_dir=results_dir, config=cfg, emit=emit)

    # Load data
    elements, edges = load_elements_and_edges(results_dir, emit)
    compounds, root_map, temps = load_compound_data(results_dir, emit)

    state.elements = elements
    state.edges = edges
    state.compounds = compounds
    state.root_map = root_map
    state.compound_temps = temps

    G = build_graph(elements, edges, emit)
    state.G = G

    if G.number_of_nodes() == 0:
        _log_global("[visualizer] Empty graph.", emit)
        return state

    # Filter components + stats
    G = filter_large_components(G, cfg.min_component_size, emit)
    state.G = G
    state.global_stats = compute_graph_stats(G)

    # Build compound map
    comp_map = {}
    for comp in (compounds or []):
        cid = str(comp.get("compound_id") or comp.get("id") or "C?")
        elems = comp.get("elements") or comp.get("element_ids") or comp.get("nodes") or []
        if isinstance(elems, str):
            elems = [s.strip() for s in elems.split(",") if s.strip()]
        for e in elems:
            comp_map[str(e)] = cid

    # Compute cluster hierarchy
    cluster_info = compute_cluster_info(G, root_map, comp_map, emit)
    state.cluster_info = cluster_info

    # Inject cluster attributes
    for n in G.nodes():
        if n in cluster_info.compound_ids:
            G.nodes[n]["compound_id"] = cluster_info.compound_ids[n]
        if n in cluster_info.root_ids:
            G.nodes[n]["root_id"] = cluster_info.root_ids[n]
        if n in cluster_info.community_ids:
            G.nodes[n]["community_id"] = cluster_info.community_ids[n]

    # Layouts
    pos2d = (
        compute_layout_2d_radial(G)
        if cfg.layout_mode_2d == "radial"
        else compute_layout_2d_hybrid(G)
    )
    pos3d = compute_layout_3d_spherical(G, community_map=cluster_info.community_ids)

    state.pos2d = pos2d
    state.pos3d = pos3d

    # Styles
    state.node_styles = compute_node_styles(G)
    state.edge_styles = compute_edge_styles(G)

    return state



# ====================================================================== #
# Snapshot rendering
# ====================================================================== #

def render_all_snapshots_from_state(state: GraphState) -> List[Dict[str, Any]]:
    """
    Renders all snapshots and produces their metadata dicts.
    """
    G = state.G
    cfg = state.config

    if G is None or state.node_styles is None:
        return []

    nodes_sorted = sorted(
        G.nodes(),
        key=lambda n: state.node_styles.sizes.get(n, 0.0),
        reverse=True,
    )
    total = len(nodes_sorted)

    metas = []

    for spec in cfg.snapshots:
        k = _nodes_for_pct(spec.pct, total)
        sub_nodes = nodes_sorted[:k]
        SG = G.subgraph(sub_nodes).copy()

        subtitle = (
            f"{SG.number_of_nodes()} of {total} nodes ({spec.pct}%), "
            f"{SG.number_of_edges()} edges"
        )

        # 2D
        out2 = f"{spec.name}.png"
        draw_2d_snapshot(
            SG, state.pos2d, state.node_styles, state.edge_styles,
            cfg.style, outfile=os.path.join(state.results_dir, out2),
            title="Hilbert Graph - 2D", subtitle=subtitle,
            label_budget=14,
        )

        # 3D
        out3 = f"{spec.name}_3d.png"
        draw_3d_snapshot(
            SG, state.pos3d, state.node_styles, state.edge_styles,
            cfg.style, outfile=os.path.join(state.results_dir, out3),
            title="Hilbert Graph - 3D", subtitle=subtitle,
            label_budget=12,
        )

        stats = compute_graph_stats(SG)
        top_nodes = nodes_sorted[:10]

        meta = write_snapshot_metadata(
            state.results_dir,
            snapshot_name=out2,
            snapshot_name_3d=out3,
            pct=spec.pct,
            stats=stats,
            top_nodes=[str(n) for n in top_nodes],
            layout_info={
                "layout_mode_2d": cfg.layout_mode_2d,
                "layout_mode_3d": cfg.layout_mode_3d,
            },
            diagnostics=None,
        )
        metas.append(meta)

    state.snapshot_meta = metas
    return metas


# ====================================================================== #
# Visualizer Driver
# ====================================================================== #

class HilbertGraphVisualizer:
    """
    High-level orchestrator for all graph visualisation tasks.
    """

    def __init__(
        self,
        results_dir: str,
        *,
        config: VisualizerConfig | None = None,
        emit: Callable[[str, Dict[str, Any]], None] = DEFAULT_EMIT,
    ):
        self.results_dir = results_dir
        self.config = config or DEFAULT_CONFIG
        self.config.ensure_defaults()
        self.emit = emit

        self.state: GraphState | None = None

        # Backwards compat mirror fields
        self.G = None
        self.cluster_info = None
        self.global_stats = None
        self.pos2d = {}
        self.pos3d = {}
        self.node_styles = None
        self.edge_styles = None
        self.snapshot_meta = []
        self.diagnostic_plots = {}

    def run(self):
        """
        Main pipeline:
            - Build GraphState
            - Render snapshots
            - Create diagnostics
            - Write modular metadata
            - Write global index
        """
        try:
            self.emit("pipeline", {"stage": "graph_visualizer", "event": "start"})
        except Exception:
            pass

        state = self.state = build_graph_state(
            self.results_dir, config=self.config, emit=self.emit
        )

        G = state.G
        if G is None or G.number_of_nodes() == 0:
            return

        # Mirror state into legacy fields
        self.G = G
        self.cluster_info = state.cluster_info
        self.global_stats = state.global_stats
        self.pos2d = state.pos2d
        self.pos3d = state.pos3d
        self.node_styles = state.node_styles
        self.edge_styles = state.edge_styles

        # Render snapshots
        self.snapshot_meta = render_all_snapshots_from_state(state)

        # Diagnostics
        self.diagnostic_plots = _render_diagnostics_from_state(state)

        # ========== NEW: Modular metadata outputs ========== #

        # 1. Core metadata
        core_file = write_metadata_core(
            self.results_dir,
            version="hilbert.graphmeta.core.v2",
            run_seed=42,
            orchestrator_version="4.2",
            lsa_model_version=None,
            embedding_parameters=None,
            cluster_hierarchy_info={
                "num_compounds": len(state.cluster_info.compound_ids),
                "num_roots": len(state.cluster_info.root_ids),
                "num_communities": len(state.cluster_info.community_ids),
            },
            pruning_info={
                "min_component_size": self.config.min_component_size,
            },
        )

        # 2. Layout metadata
        layout_file = write_metadata_layout(
            self.results_dir,
            layout2d={"mode": self.config.layout_mode_2d},
            layout3d={"mode": self.config.layout_mode_3d},
        )

        # 3. Analytics metadata
        comp_summary = {}
        root_summary = {}

        # Count by compound / root
        for n in G.nodes():
            cid = G.nodes[n].get("compound_id")
            rid = G.nodes[n].get("root_id")
            if cid:
                comp_summary[cid] = comp_summary.get(cid, 0) + 1
            if rid:
                root_summary[rid] = root_summary.get(rid, 0) + 1

        analytics_file = write_metadata_analytics(
            self.results_dir,
            global_stats=asdict(self.global_stats),
            community_sizes=state.cluster_info.community_sizes,
            component_sizes=state.cluster_info.component_sizes,
            compound_summary=comp_summary,
            root_summary=root_summary,
        )

        # 4. Diagnostics metadata
        diag_file = write_metadata_diagnostics(
            self.results_dir,
            diagnostics=self.diagnostic_plots,
        )

        # 5. Global index (pointer file)
        metadata_files = {
            "core": core_file,
            "layout": layout_file,
            "analytics": analytics_file,
            "diagnostics": diag_file,
        }

        cfg_dict = asdict(self.config)
        write_global_index(
            self.results_dir,
            self.global_stats,
            self.snapshot_meta,
            config=cfg_dict,
            metadata_files=metadata_files,
        )

        try:
            self.emit("pipeline", {"stage": "graph_visualizer", "event": "end"})
        except Exception:
            pass


def generate_all_graph_views(
    results_dir: str,
    out_dir: str | None = None,
    config: VisualizerConfig | None = None,
    emit: Callable[[str, Dict[str, Any]], None] = DEFAULT_EMIT,
):
    """
    Main convenience wrapper.
    """
    viz = HilbertGraphVisualizer(results_dir, config=config, emit=emit)
    viz.run()
    return viz.snapshot_meta
