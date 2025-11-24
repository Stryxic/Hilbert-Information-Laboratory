# hilbert_graphs/layout/__init__.py

"""
Layout subpackage for Hilbert graphs.

Provides:
  - 2D hybrid / radial layouts
  - 3D spherical-manifold layout
"""

from __future__ import annotations

from .layout2d import (
    compute_layout_2d_hybrid,
    compute_layout_2d_radial,
)
from .layout3d import (
    compute_layout_3d_spherical,
)

__all__ = [
    "compute_layout_2d_hybrid",
    "compute_layout_2d_radial",
    "compute_layout_3d_spherical",
]
