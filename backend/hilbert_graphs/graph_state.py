"""
Unified GraphState for the Hilbert visualisation pipeline.

This merges the ORIGINAL minimal GraphState with the EXTENDED fields required
by the new functional hub and modern visualiser.

It is intentionally permissive: any field not used by the renderer or
metadata writer simply passes through.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Optional

import networkx as nx

from .analytics import ClusterInfo, GraphStats
from .styling import NodeStyleMaps, EdgeStyleMaps


@dataclass
class GraphState:
    """
    Full graph state used by the visualiser.

    Contains everything needed to:
      - build layouts
      - compute styling
      - render snapshots
      - write metadata
    """

    # Core context
    results_dir: str
    config: Any
    emit: Any

    # Base graph + tables
    G: Optional[nx.Graph] = None
    elements: Any = None
    edges: Any = None
    compounds: Any = None
    root_map: Dict[str, str] = field(default_factory=dict)
    compound_temps: Dict[str, float] = field(default_factory=dict)

    # Analytics
    cluster_info: Optional[ClusterInfo] = None
    global_stats: Optional[GraphStats] = None

    # Layouts (2D / 3D)
    pos2d: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    pos3d: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)

    # Styling
    node_styles: Optional[NodeStyleMaps] = None
    edge_styles: Optional[EdgeStyleMaps] = None

    # Per-snapshot metadata
    snapshot_meta: list = field(default_factory=list)

    # Optional metadata bag
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def subset(self, nodes_subset) -> "GraphState":
        """
        Return a shallow subgraph view (used for some experimental renderers).
        """
        if self.G is None:
            return self

        nodes = [n for n in self.G.nodes() if n in nodes_subset]
        H = self.G.subgraph(nodes).copy()

        # Filter positions and styles
        pos2d = {n: self.pos2d[n] for n in nodes if n in self.pos2d}
        pos3d = {n: self.pos3d[n] for n in nodes if n in self.pos3d}

        node_styles = None
        if self.node_styles:
            node_styles = NodeStyleMaps(
                sizes={n: self.node_styles.sizes.get(n, 0.0) for n in nodes},
                colors={n: self.node_styles.colors.get(n) for n in nodes},
                halo_sizes={n: self.node_styles.halo_sizes.get(n, 0.0) for n in nodes},
                alphas={n: self.node_styles.alphas.get(n, 1.0) for n in nodes},
                primary_labels=[
                    n for n in self.node_styles.primary_labels if n in nodes
                ],
                secondary_labels=[
                    n for n in self.node_styles.secondary_labels if n in nodes
                ],
            )

        edge_styles = None
        if self.edge_styles:
            edge_styles = EdgeStyleMaps(
                widths={
                    e: self.edge_styles.widths[e]
                    for e in self.edge_styles.widths
                    if e[0] in nodes and e[1] in nodes
                },
                alphas={
                    e: self.edge_styles.alphas[e]
                    for e in self.edge_styles.alphas
                    if e[0] in nodes and e[1] in nodes
                },
                alpha=self.edge_styles.alpha,
            )

        return GraphState(
            results_dir=self.results_dir,
            config=self.config,
            emit=self.emit,
            G=H,
            elements=self.elements,
            edges=self.edges,
            compounds=self.compounds,
            root_map=self.root_map,
            compound_temps=self.compound_temps,
            cluster_info=self.cluster_info,
            global_stats=None,
            pos2d=pos2d,
            pos3d=pos3d,
            node_styles=node_styles,
            edge_styles=edge_styles,
            snapshot_meta=[],
            meta=dict(self.meta),
        )
