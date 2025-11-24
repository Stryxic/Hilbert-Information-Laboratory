"""
Preset configuration for the Hilbert graph visualiser.

These are deliberately conservative, with a bias toward:
  - stable layouts
  - low-contrast but readable styling
  - a small number of snapshots
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any


# --------------------------------------------------------------------------- #
# Snapshot specification
# --------------------------------------------------------------------------- #

@dataclass
class SnapshotSpec:
    name: str
    pct: float
    structural_filter: Optional[str] = None  # future: giant components, etc.


# --------------------------------------------------------------------------- #
# Visual style
# --------------------------------------------------------------------------- #

@dataclass
class VisualStyle:
    background_color: str = "#050713"
    background_color_3d: str = "#050713"
    node_edge_color: str = "#050713"

    node_alpha: float = 0.95
    node_halo_alpha: float = 0.13
    node_halo_scale: float = 1.55  # used mainly for 2D halo brightness

    edge_color: str = "#5b4c8a"
    edge_alpha_dense: float = 0.12
    edge_alpha_sparse: float = 0.18
    edge_width_min: float = 0.2
    edge_width_max: float = 2.0

    label_color: str = "#ffffff"
    label_outline_color: str = "#000000"
    label_outline_width: float = 2.5
    label_primary_size: int = 10
    label_secondary_size: int = 8

    hull_face_alpha: float = 0.08
    hull_edge_alpha: float = 0.3
    hull_face_color: str = "#7cc7ff"
    hull_edge_color: str = "#b0e7ff"

    camera_elev: float = 25.0
    camera_azim: float = 35.0

    dpi: int = 260

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Visualiser configuration
# --------------------------------------------------------------------------- #

@dataclass
class VisualizerConfig:
    """
    High-level configuration used by the visualiser and written to the
    snapshot index for reproducibility.
    """

    layout_mode_2d: str = "hybrid"   # "hybrid" or "radial"
    layout_mode_3d: str = "hybrid"   # currently "hybrid" == spherical

    snapshot_mode: str = "percentile_basic"
    min_component_size: int = 4
    max_edges: int = 25000
    edge_weight_min: float = 0.0

    # IMPORTANT: mutable defaults must use default_factory
    snapshots: List[SnapshotSpec] = field(default_factory=list)
    style: VisualStyle = field(default_factory=VisualStyle)

    version: str = "hilbert.visualizer.v3"

    def __post_init__(self):
        # Ensure we always have a sensible default snapshot schedule
        if not self.snapshots:
            self.snapshots = [
                SnapshotSpec(name="graph_1pct", pct=1.0),
                SnapshotSpec(name="graph_5pct", pct=5.0),
                SnapshotSpec(name="graph_10pct", pct=10.0),
                SnapshotSpec(name="graph_25pct", pct=25.0),
                SnapshotSpec(name="graph_50pct", pct=50.0),
                SnapshotSpec(name="graph_full", pct=100.0),
            ]

        # Style might be explicitly set to None by callers - guard against that
        if self.style is None:
            self.style = VisualStyle()

    # Called by the visualiser to normalise config before use
    def ensure_defaults(self) -> None:
        """
        Idempotent normalisation hook.

        - Ensures snapshots list is populated.
        - Ensures style is a VisualStyle instance.
        """
        if not self.snapshots:
            self.snapshots = [
                SnapshotSpec(name="graph_1pct", pct=1.0),
                SnapshotSpec(name="graph_5pct", pct=5.0),
                SnapshotSpec(name="graph_10pct", pct=10.0),
                SnapshotSpec(name="graph_25pct", pct=25.0),
                SnapshotSpec(name="graph_50pct", pct=50.0),
                SnapshotSpec(name="graph_full", pct=100.0),
            ]
        if self.style is None:
            self.style = VisualStyle()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layout_mode_2d": self.layout_mode_2d,
            "layout_mode_3d": self.layout_mode_3d,
            "snapshot_mode": self.snapshot_mode,
            "min_component_size": self.min_component_size,
            "max_edges": self.max_edges,
            "edge_weight_min": self.edge_weight_min,
            "snapshots": [asdict(s) for s in self.snapshots],
            "style": self.style.to_dict(),
            "version": self.version,
        }


# Singleton default config used by the visualiser
DEFAULT_CONFIG = VisualizerConfig()
