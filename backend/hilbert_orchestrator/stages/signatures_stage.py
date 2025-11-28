"""
hilbert_orchestrator/stages/signatures_stage.py
===============================================

Epistemic signatures stage for the Hilbert Orchestrator.

This stage wraps :func:`hilbert_pipeline.compute_signatures`, which computes
per-element epistemic signatures (information / misinformation / disinformation
/ ambiguous) based on prior annotations and field structure.
"""

from __future__ import annotations

from hilbert_orchestrator.core.context import PipelineContext
from hilbert_orchestrator.core.registry import GLOBAL_STAGE_REGISTRY

from hilbert_pipeline import compute_signatures


@GLOBAL_STAGE_REGISTRY.decorator(
    key="signatures",
    order=8.0,
    label="Epistemic signatures",
    required=False,
    depends_on=["element_labels"],
    dialectic_role="evidence",
    supports=["stability_metrics"],
)
def stage_signatures(ctx: PipelineContext) -> None:
    """
    Compute epistemic signatures for elements.

    This stage is optional - if inputs are missing, we log and return without
    halting the pipeline.
    """
    ctx.log("info", "Running epistemic signatures stage")

    try:
        compute_signatures(ctx.results_dir, emit=ctx.emit)
        ctx.log("info", "Epistemic signatures stage complete")
    except FileNotFoundError as exc:
        # Soft skip if required CSVs are not present (e.g. no annotations)
        ctx.log(
            "warn",
            "Epistemic signatures inputs missing - skipping signatures stage",
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - optional analytical layer
        ctx.log(
            "warn",
            "Epistemic signatures stage failed - continuing without signatures",
            error=str(exc),
        )
