"""
Unified Hilbert graph visualiser - v4.2 (cluster-aware, modular metadata).

This version integrates the modular metadata system (Option 2):
    - graph_metadata_core.json
    - graph_layout.json
    - graph_analytics.json
    - graph_diagnostics.json
    - graph_snapshots_index.json (lightweight pointer file)

It also:
    - passes cluster_info into 2D/3D layouts and styling
    - passes cluster_info into the 2D renderer (for hull overlays etc.)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Callable, List, Tuple, Optional
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

# Modular metadata writers (Option 2)
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
# Logging / small helpers
# ====================================================================== #

def _log_global(msg: str, emit: Callable[[str, Dict[str, Any]], None] | None) -> None:
    print(msg)
    if emit:
        try:
            emit("log", {"message": msg})
        except Exception:
            pass


def _safe_mkdir(path: str) -> None:
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
# Diagnostics (degree, components, stability, communities)
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
) -> None:
    if data.size == 0:
        return

    _safe_mkdir(os.path.dirname(outfile) or ".")
    bg = style.background_color
    fg = style.label_color
    dpi = style.dpi

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg)
    ax.set_facecolor(bg)

    ax.hist(
        data,
        bins=bins,
        color="#6fa8ff",
        alpha=0.9,
        edgecolor="#111318",
        linewidth=0.6,
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


def _render_cluster_diagnostics(
    G: nx.Graph,
    cluster_info,
    style,
    results_dir: str,
) -> Dict[str, str]:
    """
    Generates:
      - graph_community_sizes.png
      - graph_community_degree_panels.png
    """
    out: Dict[str, str] = {}

    if cluster_info is None or not getattr(cluster_info, "community_ids", None):
        return out

    comm_to_nodes: Dict[str, List[str]] = {}
    for node, cid in cluster_info.community_ids.items():
        comm_to_nodes.setdefault(str(cid), []).append(str(node))

    # Community size distribution
    sizes = np.array([len(v) for v in comm_to_nodes.values()], float)
    if sizes.size > 0:
        out_sizes = os.path.join(results_dir, "graph_community_sizes.png")
        _plot_histogram_1d(
            sizes,
            bins=min(40, max(8, int(np.sqrt(sizes.size)) + 3)),
            title="Community size distribution",
            xlabel="community size (nodes)",
            ylabel="count",
            outfile=out_sizes,
            style=style,
        )
        out["community_size_hist"] = out_sizes

    # Degree distributions in top-k communities
    top = sorted(comm_to_nodes.items(), key=lambda kv: len(kv[1]), reverse=True)
    top_k = min(4, len(top))
    if top_k <= 0:
        return out

    bg = style.background_color
    fg = style.label_color
    dpi = style.dpi

    fig, axes = plt.subplots(
        1,
        top_k,
        figsize=(4.2 * top_k, 4.2),
        sharey=True,
        facecolor=bg,
    )
    if top_k == 1:
        axes = [axes]

    for ax, (cid, nodes) in zip(axes, top[:top_k]):
        degs = np.array([G.degree(n) for n in nodes], float)
        if degs.size == 0:
            continue
        ax.set_facecolor(bg)
        ax.hist(
            degs,
            bins=min(25, max(5, int(np.sqrt(degs.size)) + 2)),
            color="#6fa8ff",
            alpha=0.9,
            edgecolor="#111318",
            linewidth=0.6,
        )
        ax.set_title(
            f"Community {cid} (n={len(nodes)})",
            fontsize=9,
            color=fg,
        )
        ax.tick_params(colors=fg)
        for spine in ax.spines.values():
            spine.set_color("#2a2e3a")

    for ax in axes:
        ax.set_ylabel("count", fontsize=8, color=fg)

    fig.suptitle(
        "Top communities - internal degree distributions",
        fontsize=12,
        color=fg,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])

    out_panels = os.path.join(results_dir, "graph_community_degree_panels.png")
    plt.savefig(out_panels, dpi=dpi, facecolor=bg)
    plt.close(fig)

    out["community_degree_panels"] = out_panels
    return out


def _render_diagnostics_from_state(state: GraphState) -> Dict[str, str]:
    """
    Produces:
      - graph_degree_hist.png
      - graph_component_sizes.png
      - graph_stability_hist.png
      - graph_community_sizes.png
      - graph_community_degree_panels.png
    """
    G = state.G
    if G is None or G.number_of_nodes() == 0:
        return {}

    cfg = state.config
    style = cfg.style
    results_dir = state.results_dir

    out: Dict[str, str] = {}

    # Degree distribution
    degrees = np.array([d for _, d in G.degree()], float)
    if degrees.size > 0:
        out_deg = os.path.join(results_dir, "graph_degree_hist.png")
        _plot_histogram_1d(
            degrees,
            bins=min(60, max(15, int(np.sqrt(degrees.size)) + 5)),
            title="Degree distribution",
            xlabel="node degree",
            ylabel="count",
            outfile=out_deg,
            style=style,
        )
        out["degree_hist"] = out_deg

    # Component sizes
    if getattr(state, "cluster_info", None) and getattr(
        state.cluster_info, "component_sizes", None
    ):
        comp_sizes = np.array(
            list(state.cluster_info.component_sizes.values()),
            float,
        )
    else:
        comp_sizes = np.array(
            [len(c) for c in nx.connected_components(G)],
            float,
        )

    if comp_sizes.size > 0:
        out_comp = os.path.join(results_dir, "graph_component_sizes.png")
        _plot_histogram_1d(
            comp_sizes,
            bins=min(40, max(10, int(np.sqrt(comp_sizes.size)) + 3)),
            title="Connected component sizes",
            xlabel="component size (nodes)",
            ylabel="count",
            outfile=out_comp,
            style=style,
        )
        out["component_hist"] = out_comp

    # Stability / temperature / entropy distribution
    stab_vals: List[float] = []
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
            title="Stability / entropy distribution",
            xlabel="score",
            ylabel="count",
            outfile=out_stab,
            style=style,
        )
        out["stability_hist"] = out_stab

    # Community diagnostics
    out.update(
        _render_cluster_diagnostics(G, state.cluster_info, style, results_dir)
    )

    # Attach to state for introspection
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
    """
    Build a fully analysed, laid-out, and styled GraphState.

    Steps:
      1) Load CSVs / JSONs and build the base graph.
      2) Filter small components; compute global stats.
      3) Compute compound / root / community cluster hierarchy.
      4) Inject cluster attributes into the NetworkX graph.
      5) Compute 2D and 3D layouts (cluster-aware).
      6) Compute node and edge styling maps (cluster-aware).
    """
    cfg = config or DEFAULT_CONFIG
    cfg.ensure_defaults()

    state = GraphState(results_dir=results_dir, config=cfg, emit=emit)

    # Load tabular data and construct base graph
    elements, edges = load_elements_and_edges(results_dir, emit)
    compounds, root_map, temps = load_compound_data(results_dir, emit)

    state.elements = elements
    state.edges = edges
    state.compounds = compounds
    state.root_map = root_map
    state.compound_temps = temps

    G = build_graph(elements, edges, emit)
    state.G = G

    if G is None or G.number_of_nodes() == 0:
        _log_global("[visualizer] Empty graph after loading.", emit)
        return state

    # Filter small components + global stats
    G = filter_large_components(G, cfg.min_component_size, emit)
    state.G = G
    state.global_stats = compute_graph_stats(G)

    _log_global(
        f"[visualizer] Graph loaded: {state.global_stats.n_nodes} nodes, "
        f"{state.global_stats.n_edges} edges.",
        emit,
    )

    # Build element → compound mapping
    compound_map: Dict[str, str] = {}
    if isinstance(compounds, list):
        comp_iter = compounds
    elif isinstance(compounds, dict):
        comp_iter = list(compounds.values())
    else:
        comp_iter = []

    for comp in comp_iter:
        if not isinstance(comp, dict):
            continue

        cid = str(
            comp.get("compound_id")
            or comp.get("id")
            or comp.get("compound")
            or "C?",
        )

        elems = (
            comp.get("elements")
            or comp.get("element_ids")
            or comp.get("nodes")
            or []
        )

        if isinstance(elems, str):
            elems = [s.strip() for s in elems.split(",") if s.strip()]

        for e in elems:
            compound_map[str(e)] = cid

    # Cluster hierarchy + attribute injection
    cluster_info = compute_cluster_info(G, root_map, compound_map, emit)
    state.cluster_info = cluster_info

    for n in G.nodes():
        key = str(n)
        if key in cluster_info.compound_ids:
            G.nodes[n]["compound_id"] = cluster_info.compound_ids[key]
        if key in cluster_info.root_ids:
            G.nodes[n]["root_id"] = cluster_info.root_ids[key]
        if key in cluster_info.community_ids:
            G.nodes[n]["community_id"] = cluster_info.community_ids[key]

    # Layouts (cluster-aware)
    if cfg.layout_mode_2d == "radial":
        pos2d = compute_layout_2d_radial(G, cluster_info=cluster_info)
    else:
        pos2d = compute_layout_2d_hybrid(G, cluster_info=cluster_info)

    pos3d = compute_layout_3d_spherical(G, cluster_info=cluster_info)

    state.pos2d = pos2d
    state.pos3d = pos3d

    # Styles (cluster-aware)
    state.node_styles = compute_node_styles(G, cluster_info=cluster_info)
    state.edge_styles = compute_edge_styles(G, cluster_info=cluster_info)

    return state


# ====================================================================== #
# Snapshot rendering
# ====================================================================== #

def render_all_snapshots_from_state(state: GraphState) -> List[Dict[str, Any]]:
    """
    Render all configured snapshots (2D and 3D) for a prepared GraphState.

    Returns a list of per-snapshot metadata dicts.
    """
    G = state.G
    cfg = state.config
    emit = state.emit or DEFAULT_EMIT

    if G is None or state.node_styles is None:
        return []

    # Sort nodes by semantic size from node_styles
    nodes_sorted = sorted(
        G.nodes(),
        key=lambda n: state.node_styles.sizes.get(str(n), 0.0),
        reverse=True,
    )
    total = len(nodes_sorted)

    metas: List[Dict[str, Any]] = []

    for spec in cfg.snapshots:
        k = _nodes_for_pct(spec.pct, total)
        sub_nodes = nodes_sorted[:k]
        SG = G.subgraph(sub_nodes).copy()

        subtitle = (
            f"{SG.number_of_nodes()} of {total} nodes ({spec.pct}%), "
            f"{SG.number_of_edges()} edges"
        )

        # 2D snapshot
        out2 = f"{spec.name}.png"
        path2d = os.path.join(state.results_dir, out2)
        draw_2d_snapshot(
            SG,
            state.pos2d,
            state.node_styles,
            state.edge_styles,
            cfg.style,
            outfile=path2d,
            title="Hilbert Information Graph - 2D",
            subtitle=subtitle,
            label_budget=14,
            cluster_info=state.cluster_info,
        )
        try:
            emit("artifact", {"kind": "graph_2d", "path": path2d})
        except Exception:
            pass

        # 3D snapshot
        out3 = f"{spec.name}_3d.png"
        path3d = os.path.join(state.results_dir, out3)
        draw_3d_snapshot(
            SG,
            state.pos3d,
            state.node_styles,
            state.edge_styles,
            cfg.style,
            outfile=path3d,
            title="Hilbert Information Graph - 3D",
            subtitle=subtitle,
            label_budget=12,
        )
        try:
            emit("artifact", {"kind": "graph_3d", "path": path3d})
        except Exception:
            pass

        # Per-snapshot metadata
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
            diagnostics=None,  # global diagnostics added separately
        )
        metas.append(meta)

    state.snapshot_meta = metas
    return metas


# ====================================================================== #
# Visualizer driver
# ====================================================================== #

class HilbertGraphVisualizer:
    """
    High-level orchestrator for:
      - loading
      - analytics
      - layouts
      - styling
      - 2D and 3D rendering
      - diagnostics
      - modular metadata writing
    """

    def __init__(
        self,
        results_dir: str,
        *,
        config: VisualizerConfig | None = None,
        emit: Callable[[str, Dict[str, Any]], None] = DEFAULT_EMIT,
    ) -> None:
        self.results_dir = results_dir
        self.config = config or DEFAULT_CONFIG
        self.config.ensure_defaults()
        self.emit = emit

        self.state: GraphState | None = None

        # Legacy mirror fields for callers that still poke directly
        self.G: Optional[nx.Graph] = None
        self.cluster_info = None
        self.global_stats = None
        self.pos2d: Dict[str, Tuple[float, float]] = {}
        self.pos3d: Dict[str, Tuple[float, float, float]] = {}
        self.node_styles = None
        self.edge_styles = None
        self.snapshot_meta: List[Dict[str, Any]] = []
        self.diagnostic_plots: Dict[str, str] = {}

    def run(self) -> None:
        """
        Main pipeline:
          - build GraphState
          - render snapshots
          - compute diagnostics
          - write modular metadata files
          - write global index pointer
        """
        try:
            self.emit("pipeline", {"stage": "graph_visualizer", "event": "start"})
        except Exception:
            pass

        state = self.state = build_graph_state(
            self.results_dir,
            config=self.config,
            emit=self.emit,
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

        # ------------------------------------------------------------------
        # Modular metadata outputs (Option 2)
        # ------------------------------------------------------------------

        # 1. Core metadata
        core_file = write_metadata_core(
            self.results_dir,
            version="hilbert.graphmeta.core.v2",
            run_seed=42,
            orchestrator_version="4.2",
            lsa_model_version=None,
            embedding_parameters=None,
            cluster_hierarchy_info={
                "num_compounds": len(getattr(state.cluster_info, "compound_ids", {})),
                "num_roots": len(getattr(state.cluster_info, "root_ids", {})),
                "num_communities": len(getattr(state.cluster_info, "community_ids", {})),
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
        comp_summary: Dict[str, int] = {}
        root_summary: Dict[str, int] = {}

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
            community_sizes=getattr(state.cluster_info, "community_sizes", {}),
            component_sizes=getattr(state.cluster_info, "component_sizes", {}),
            compound_summary=comp_summary,
            root_summary=root_summary,
        )

        # 4. Diagnostics metadata
        diag_file = write_metadata_diagnostics(
            self.results_dir,
            diagnostics=self.diagnostic_plots,
        )

        # 5. Global index pointer
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


# ====================================================================== #
# Orchestrator convenience wrapper
# ====================================================================== #

def generate_all_graph_views(
    results_dir: str,
    out_dir: str | None = None,
    config: VisualizerConfig | None = None,
    emit: Callable[[str, Dict[str, Any]], None] = DEFAULT_EMIT,
):
    """
    Main convenience wrapper used by the pipeline.
    out_dir is currently ignored; all artifacts go into results_dir.
    """
    viz = HilbertGraphVisualizer(results_dir, config=config, emit=emit)
    viz.run()
    return viz.snapshot_meta
