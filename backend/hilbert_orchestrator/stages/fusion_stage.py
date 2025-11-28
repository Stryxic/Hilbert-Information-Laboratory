"""
fusion_stage.py
================

Implements the Span–Element Fusion stage of the Hilbert Information Pipeline.

This stage consumes:
    • LSA output (hilbert_elements.csv and span_map inside lsa_field.json)

and produces:
    • span_element_fusion.csv

These fused span-to-element alignments are required by:
    - edges
    - molecule layer
    - element language model
"""

from __future__ import annotations

import os

from hilbert_pipeline import run_fusion_pipeline

from hilbert_orchestrator.core.registry import GLOBAL_STAGE_REGISTRY
from hilbert_orchestrator.core.context import PipelineContext


# ---------------------------------------------------------------------------
# Stage Registration
# ---------------------------------------------------------------------------

@GLOBAL_STAGE_REGISTRY.decorator(
    key="fusion",
    order=2.0,
    label="Span–element fusion",
    required=True,
    depends_on=["lsa_field"],
)
def fusion_stage(ctx: PipelineContext) -> None:
    """
    Execute the fusion layer.

    This stage performs the following:

        • Reads LSA outputs from ``ctx.results_dir``
        • Invokes the backend fusion pipeline
        • Writes ``span_element_fusion.csv``
        • Registers the artifact in the pipeline context

    Parameters
    ----------
    ctx : PipelineContext
        Shared orchestrator execution context.
    """

    ctx.log("info", "Running Span–Element Fusion stage")

    # ----------------------------------------------------------------------
    # 1. Execute native fusion
    # ----------------------------------------------------------------------
    run_fusion_pipeline(ctx.results_dir, emit=ctx.emit)

    # ----------------------------------------------------------------------
    # 2. Register output artifact
    # ----------------------------------------------------------------------
    fusion_csv = os.path.join(ctx.results_dir, "hilbert_spans.csv")

    if os.path.exists(fusion_csv):
        ctx.add_artifact("hilbert_spans.csv", kind="fusion")
        ctx.log("info", "Fusion stage completed", path=fusion_csv)
    else:
        ctx.log("warn", "Fusion stage finished but no fusion CSV found")
    