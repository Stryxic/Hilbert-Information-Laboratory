"""
elements_stage.py
=================

Contains two related Hilbert pipeline stages:

    1. Element Roots Stage
       - Computes structural root clusters of informational elements
       - Produces:
            element_roots.csv
            element_cluster_metrics.json

    2. Element Labels Stage
       - Generates semantic descriptions for each informational element
       - Produces:
            element_descriptions.json
            element_intensity.csv

These stages are kept together because they operate directly
on the hilbert_elements.csv table.
"""

from __future__ import annotations

from hilbert_pipeline import run_element_roots, build_element_descriptions

from hilbert_orchestrator.core.context import PipelineContext
from hilbert_orchestrator.core.registry import GLOBAL_STAGE_REGISTRY


@GLOBAL_STAGE_REGISTRY.decorator(
    key="element_roots",
    order=5.0,
    label="Element roots",
    required=True,
    depends_on=["molecules"],
)
def stage_element_roots(ctx: PipelineContext) -> None:
    ctx.log("info", "Running element root discovery")

    # Correct API function
    run_element_roots(ctx.results_dir, emit=ctx.emit)

    # Real output filenames
    ctx.add_artifact("element_roots.csv", kind="element-roots")
    ctx.add_artifact("element_clusters.json", kind="element-roots")


# ---------------------------------------------------------------------------
# Element Labels
# ---------------------------------------------------------------------------

from hilbert_pipeline import build_element_descriptions
import os
import pandas as pd

@GLOBAL_STAGE_REGISTRY.decorator(
    key="element_labels",
    order=6.0,
    label="Element labels",
    required=False,
    depends_on=["element_roots"],
)
def stage_element_labels(ctx):
    ctx.log("info", "Building element labels")

    elements_csv = os.path.join(ctx.results_dir, "hilbert_elements.csv")

    # Spans are optional in v4 – do not require hilbert_spans.csv
    spans = None

    try:
        build_element_descriptions(
            elements_csv=elements_csv,
            spans=spans,
            out_dir=ctx.results_dir,
        )
    except Exception as exc:
        ctx.log("error", "Element label generation failed", error=str(exc))
        raise

    ctx.log("info", "Element labels complete")

