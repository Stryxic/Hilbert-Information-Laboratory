"""
edges_stage.py
================

Implements the Element Edges stage of the Hilbert Information Pipeline.

This stage consumes:
    • span_element_fusion.csv

and produces:
    • element_edges.csv (or backend-defined filename)

Edges represent co-occurrence and semantic proximity relationships
between informational elements, and are required for:

    - Molecule layer construction
    - Cluster / root extraction
    - Stability and compound analyses
"""

from __future__ import annotations

from hilbert_pipeline import build_element_edges
from hilbert_orchestrator.core.context import PipelineContext
from hilbert_orchestrator.core.registry import GLOBAL_STAGE_REGISTRY


@GLOBAL_STAGE_REGISTRY.decorator(
    key="edges",
    order=3.0,
    label="Element edges",
    required=True,
    depends_on=["fusion"],
)
def stage_edges(ctx: PipelineContext) -> None:
    ctx.log("info", "Running element edge builder")

    build_element_edges(ctx.results_dir, emit=ctx.emit)

    ctx.add_artifact("hilbert_edges.csv", kind="edges")
