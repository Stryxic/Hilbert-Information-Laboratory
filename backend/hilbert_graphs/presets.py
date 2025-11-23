"""
Presets and configuration defaults for the Hilbert graph visualisation system.
This version is guaranteed compatible with render2d, render3d, styling, layout2d,
layout3d, and visualizer, with no missing attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Dict, Any, List, Optional


# --------------------------------------------------------------------------- #
# Literal types defining available modes
# --------------------------------------------------------------------------- #
LayoutMode2D = Literal["hybrid", "radial", "spring_only"]
LayoutMode3D = Literal["hybrid", "spring_only"]

SnapshotMode = Literal[
    "percentile_basic",
    "percentile_extended",
    "structural",
]


# --------------------------------------------------------------------------- #
# Snapshot specification
# --------------------------------------------------------------------------- #
@dataclass
class SnapshotSpec:
    """
    Definition of a single snapshot cut.
    name:   base output name
    pct:    numeric percentage of nodes (None if structural)
    structural_filter: optional structural selection
    """
    name: str
    pct: Optional[float] = None
    structural_filter: Optional[str] = None


# --------------------------------------------------------------------------- #
# Visual styling (clean, non-crashing)
# --------------------------------------------------------------------------- #
@dataclass
class VisualStyle:
    """
    All currently-used styling parameters. This version contains only attributes
    referenced anywhere in render2d, render3d, and styling.py.
    """

    # Background
    background_color: str = "#050713"
    background_color_3d: str = "#050713"

    # Nodes
    node_edge_color: str = "#050713"
    node_alpha: float = 0.95
    node_halo_alpha: float = 0.13
    node_halo_scale: float = 1.55   # used in 2D renderer
    # 3D uses the same halo-scale. No separate node_halo_scale_3d.

    # Edges
    edge_color: str = "#5b4c8a"
    edge_alpha_dense: float = 0.12
    edge_alpha_sparse: float = 0.18
    edge_width_min: float = 0.2
    edge_width_max: float = 2.0

    # Labels
    label_color: str = "#ffffff"
    label_outline_color: str = "#000000"
    label_outline_width: float = 2.5
    label_primary_size: int = 10
    label_secondary_size: int = 8

    # Hulls (future, referenced but optional)
    hull_face_alpha: float = 0.08
    hull_edge_alpha: float = 0.30
    hull_face_color: str = "#7cc7ff"
    hull_edge_color: str = "#b0e7ff"

    # 3D camera
    camera_elev: float = 25.0   # used in render3d
    camera_azim: float = 35.0   # used in render3d

    # Output resolution
    dpi: int = 260


# --------------------------------------------------------------------------- #
# Top-level visualiser configuration
# --------------------------------------------------------------------------- #
@dataclass
class VisualizerConfig:
    """
    Full configuration passed into HilbertGraphVisualizer.
    """

    layout_mode_2d: LayoutMode2D = "hybrid"
    layout_mode_3d: LayoutMode3D = "hybrid"

    snapshot_mode: SnapshotMode = "percentile_basic"

    # Filtering
    min_component_size: int = 4
    max_edges: Optional[int] = 25_000
    edge_weight_min: float = 0.0

    snapshots: List[SnapshotSpec] = field(default_factory=list)

    # Style bundle
    style: VisualStyle = field(default_factory=VisualStyle)

    # Version tag for reproducibility
    version: str = "hilbert.visualizer.v3"

    # ------------------------------------------------------------------ #
    def ensure_defaults(self) -> None:
        """Populate default snapshots if missing."""
        if self.snapshots:
            return

        if self.snapshot_mode == "percentile_basic":
            self.snapshots = [
                SnapshotSpec("graph_1pct", pct=1.0),
                SnapshotSpec("graph_5pct", pct=5.0),
                SnapshotSpec("graph_10pct", pct=10.0),
                SnapshotSpec("graph_25pct", pct=25.0),
                SnapshotSpec("graph_50pct", pct=50.0),
                SnapshotSpec("graph_full", pct=100.0),
            ]

        elif self.snapshot_mode == "percentile_extended":
            self.snapshots = [
                SnapshotSpec("graph_1pct", pct=1.0),
                SnapshotSpec("graph_2pct", pct=2.0),
                SnapshotSpec("graph_3pct", pct=3.0),
                SnapshotSpec("graph_5pct", pct=5.0),
                SnapshotSpec("graph_10pct", pct=10.0),
                SnapshotSpec("graph_15pct", pct=15.0),
                SnapshotSpec("graph_20pct", pct=20.0),
                SnapshotSpec("graph_25pct", pct=25.0),
                SnapshotSpec("graph_33pct", pct=33.0),
                SnapshotSpec("graph_50pct", pct=50.0),
                SnapshotSpec("graph_66pct", pct=66.0),
                SnapshotSpec("graph_80pct", pct=80.0),
                SnapshotSpec("graph_full", pct=100.0),
            ]

        elif self.snapshot_mode == "structural":
            self.snapshots = [
                SnapshotSpec("graph_giant", structural_filter="giant_component"),
                SnapshotSpec("graph_compounds", structural_filter="compound_map"),
                SnapshotSpec("graph_roots", structural_filter="root_element"),
                SnapshotSpec("graph_full", pct=100.0),
            ]

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Ensure values are sensible."""
        if self.min_component_size < 1:
            self.min_component_size = 1

        if self.max_edges is not None and self.max_edges < 100:
            self.max_edges = 100

        if not isinstance(self.edge_weight_min, (float, int)):
            self.edge_weight_min = 0.0

        self.ensure_defaults()

    # ------------------------------------------------------------------ #
    def merged(self, overrides: Dict[str, Any]) -> "VisualizerConfig":
        """Return a new config with overrides applied."""
        cfg = replace(self)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        cfg.validate()
        return cfg


# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = VisualizerConfig()
DEFAULT_CONFIG.ensure_defaults()
DEFAULT_CONFIG.validate()
