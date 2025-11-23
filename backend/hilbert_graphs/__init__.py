"""
Hilbert Graph Visualisation package.

Unified, compound-aware graph loader, analytics, layout, styling,
rendering, and metadata utilities.
"""

# ---------------------------------------------------------------------------
# High-level Visualiser
# ---------------------------------------------------------------------------
from .visualizer import (
    HilbertGraphVisualizer,
    generate_all_graph_views,
)

# ---------------------------------------------------------------------------
# Configuration & presets
# ---------------------------------------------------------------------------
from .presets import (
    VisualizerConfig,
    SnapshotSpec,
    VisualStyle,
    DEFAULT_CONFIG,
)

# ---------------------------------------------------------------------------
# Loader & graph construction
# ---------------------------------------------------------------------------
from .loader import (
    load_elements_and_edges,
    load_compound_data,
    build_graph,
)

# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
from .analytics import (
    compute_cluster_info,
    compute_graph_stats,
    filter_large_components,
)

# ---------------------------------------------------------------------------
# Layout engines
# ---------------------------------------------------------------------------
from .layout.layout2d import (
    compute_layout_2d_hybrid,
    compute_layout_2d_radial,
)
from .layout.layout3d import (
    compute_layout_3d_spherical,
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
from .styling import (
    compute_node_styles,
    compute_edge_styles,
    NodeStyleMaps,
    EdgeStyleMaps,
)

# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
from .render2d import draw_2d_snapshot
from .render3d import draw_3d_snapshot

# ---------------------------------------------------------------------------
# Metadata writers
# ---------------------------------------------------------------------------
from .metadata import (
    write_snapshot_metadata,
    write_global_index,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    # Main driver
    "HilbertGraphVisualizer",
    "generate_all_graph_views",

    # Config
    "VisualizerConfig",
    "SnapshotSpec",
    "VisualStyle",
    "DEFAULT_CONFIG",

    # Loader
    "load_elements_and_edges",
    "load_compound_data",
    "build_graph",

    # Analytics
    "compute_cluster_info",
    "compute_graph_stats",
    "filter_large_components",

    # Layouts
    "compute_layout_2d_hybrid",
    "compute_layout_2d_radial",
    "compute_layout_3d_hybrid",

    # Styling
    "compute_node_styles",
    "compute_edge_styles",
    "NodeStyleMaps",
    "EdgeStyleMaps",

    # Rendering
    "draw_2d_snapshot",
    "draw_3d_snapshot",

    # Metadata
    "write_snapshot_metadata",
    "write_global_index",
]
