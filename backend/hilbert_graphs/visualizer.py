"""
Unified Hilbert graph visualiser — v4 (cluster-aware, styling-safe).

This class replaces the older graph_export and graph_snapshots entry points.
It orchestrates:

  - loading
  - analytics
  - layouts (2D + 3D, including spherical manifold mode)
  - styling
  - snapshot rendering
  - metadata writing
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Callable, List, Tuple

import networkx as nx
import numpy as np

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
from .metadata import write_snapshot_metadata, write_global_index


DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda *_: None


# =========================================================================== #
# Visualiser Driver
# =========================================================================== #

class HilbertGraphVisualizer:
    """
    High level visualisation orchestrator for:
      - loading
      - analytics
      - layouts
      - styling
      - 2D & 3D rendering
      - metadata writing
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

        # raw data
        self.elements = None
        self.edges = None
        self.compounds: Dict[str, Any] = {}
        self.root_map: Dict[str, str] = {}
        self.compound_temps: Dict[str, float] = {}

        # graph
        self.G: nx.Graph | None = None
        self.cluster_info = None
        self.global_stats = None

        # layouts
        self.pos2d: Dict[str, Tuple[float, float]] = {}
        self.pos3d: Dict[str, Tuple[float, float, float]] = {}

        # styling
        self.node_styles = None
        self.edge_styles = None

        # snapshot metadata
        self.snapshot_meta: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def _log(self, msg: str) -> None:
        print(msg)
        try:
            self.emit("log", {"message": msg})
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Load CSVs and construct the initial base graph."""
        self.elements, self.edges = load_elements_and_edges(self.results_dir, self.emit)
        self.compounds, self.root_map, self.compound_temps = load_compound_data(
            self.results_dir, self.emit
        )

        self.G = build_graph(self.elements, self.edges, self.emit)

        if self.G is None or self.G.number_of_nodes() == 0:
            self._log("[graphs] No usable graph after loading.")
            return

        # Filter small components
        self.G = filter_large_components(
            self.G, self.config.min_component_size, self.emit
        )
        self.global_stats = compute_graph_stats(self.G)

        self._log(
            f"[graphs] Graph loaded: {self.global_stats.n_nodes} nodes, "
            f"{self.global_stats.n_edges} edges."
        )

    # ------------------------------------------------------------------ #
    def compute_analytics(self) -> None:
        """Compute compounds, communities, root clusters, assign attributes."""
        if self.G is None:
            return

        # Build element→compound mapping
        compound_map: Dict[str, str] = {}

        if isinstance(self.compounds, list):
            comp_iter = self.compounds
        elif isinstance(self.compounds, dict):
            comp_iter = list(self.compounds.values())
        else:
            comp_iter = []

        for comp in comp_iter:
            if not isinstance(comp, dict):
                continue

            cid = str(
                comp.get("compound_id")
                or comp.get("id")
                or comp.get("compound")
                or "C?"
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

        # Compute cluster hierarchy
        self.cluster_info = compute_cluster_info(
            self.G,
            self.root_map,
            compound_map,
            self.emit,
        )

        # Inject attributes into NetworkX graph
        for n in self.G.nodes():
            key = str(n)
            if key in self.cluster_info.compound_ids:
                self.G.nodes[n]["compound_id"] = self.cluster_info.compound_ids[key]
            if key in self.cluster_info.root_ids:
                self.G.nodes[n]["root_id"] = self.cluster_info.root_ids[key]
            if key in self.cluster_info.community_ids:
                self.G.nodes[n]["community_id"] = self.cluster_info.community_ids[key]

    # ------------------------------------------------------------------ #
    def compute_layouts(self) -> None:
        """Compute full 2D and 3D layouts + style maps."""
        if self.G is None:
            return

        # --- 2D layout ---
        if self.config.layout_mode_2d == "radial":
            self.pos2d = compute_layout_2d_radial(self.G)
        else:
            self.pos2d = compute_layout_2d_hybrid(self.G)

        # --- 3D layout (spherical manifold) ---
        self._log("[graphs] Using spherical 3D layout.")
        self.pos3d = compute_layout_3d_spherical(self.G)

        # --- Style maps ---
        self.node_styles = compute_node_styles(self.G)
        self.edge_styles = compute_edge_styles(self.G)

    # ------------------------------------------------------------------ #
    def _nodes_for_pct(self, pct: float, total: int) -> int:
        if pct is None:
            return total
        if pct >= 100.0:
            return total
        return max(1, int(round(total * (pct / 100.0))))

    # ------------------------------------------------------------------ #
    def render_snapshots(self) -> None:
        """Render configured snapshots (percentile cuts)."""
        if self.G is None or self.node_styles is None:
            return

        nodes_sorted = sorted(
            self.G.nodes(),
            key=lambda n: self.node_styles.sizes.get(str(n), 0.0),
            reverse=True,
        )
        total = len(nodes_sorted)

        for spec in self.config.snapshots:
            k = self._nodes_for_pct(spec.pct, total)
            sub_nodes = nodes_sorted[:k]

            SG = self.G.subgraph(sub_nodes).copy()

            suffix = (
                "100% (full field)"
                if spec.pct and spec.pct >= 100.0
                else f"{spec.pct:.0f}%"
            )

            subtitle = (
                f"{SG.number_of_nodes()} of {total} nodes ({suffix}), "
                f"{SG.number_of_edges()} edges"
            )

            # ----- 2D -----
            out2 = f"{spec.name}.png"
            draw_2d_snapshot(
                SG,
                self.pos2d,
                self.node_styles,
                self.edge_styles,
                self.config.style,
                outfile=self._output_path(out2),
                title="Hilbert Information Graph - 2D",
                subtitle=subtitle,
                label_budget=14,
            )

            try:
                self.emit(
                    "artifact",
                    {"kind": "graph_2d", "path": self._output_path(out2)},
                )
            except Exception:
                pass

            # ----- 3D -----
            out3 = f"{spec.name}_3d.png"
            draw_3d_snapshot(
                SG,
                self.pos3d,
                self.node_styles,
                self.edge_styles,
                self.config.style,
                outfile=self._output_path(out3),
                title="Hilbert Information Graph - 3D",
                subtitle=subtitle,
                label_budget=10,
            )

            try:
                self.emit(
                    "artifact",
                    {"kind": "graph_3d", "path": self._output_path(out3)},
                )
            except Exception:
                pass

            # Metadata
            stats = compute_graph_stats(SG)
            top_nodes = nodes_sorted[:10]

            meta = write_snapshot_metadata(
                self.results_dir,
                snapshot_name=out2,
                snapshot_name_3d=out3,
                stats=stats,
                pct=spec.pct,
                top_nodes=[str(n) for n in top_nodes],
            )
            self.snapshot_meta.append(meta)

    # ------------------------------------------------------------------ #
    def _output_path(self, filename: str) -> str:
        import os
        return os.path.join(self.results_dir, filename)

    # ------------------------------------------------------------------ #
    def write_indexes(self) -> None:
        if self.global_stats is None:
            return

        cfg_dict = asdict(self.config)

        write_global_index(
            self.results_dir,
            self.global_stats,
            self.snapshot_meta,
            cfg_dict,
        )

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        try:
            self.emit("pipeline", {"stage": "graph_visualizer", "event": "start"})
        except Exception:
            pass

        self.load()
        if self.G is None:
            return

        self.compute_analytics()
        self.compute_layouts()
        self.render_snapshots()
        self.write_indexes()

        try:
            self.emit("pipeline", {"stage": "graph_visualizer", "event": "end"})
        except Exception:
            pass


# =========================================================================== #
# Orchestrator wrapper
# =========================================================================== #

def generate_all_graph_views(
    results_dir: str,
    out_dir: str | None = None,
    config: VisualizerConfig | None = None,
    emit: Callable[[str, Dict[str, Any]], None] = DEFAULT_EMIT,
):
    """
    Convenience wrapper used by the main Hilbert pipeline.

    `out_dir` is currently ignored - snapshots are written into `results_dir`.
    Adjust `_output_path` if you want to separate them.
    """
    viz = HilbertGraphVisualizer(results_dir, config=config, emit=emit)
    viz.run()
    return viz.snapshot_meta
