"""
hilbert_orchestrator/stages/stage_export.py
===========================================

Final export stage for the Hilbert Orchestrator.

This wraps :func:`hilbert_pipeline.run_full_export`, which:

- Builds a deterministic manifest of the run.
- Writes a ZIP archive containing all key artefacts.
- Emits an ``artifact`` event for the ZIP, which the PipelineContext
  records for later DB import (export_key).

The stage depends on the main structural layers being complete so that the
export contains a coherent graph and metrics.
"""

from __future__ import annotations

from hilbert_orchestrator.core.context import PipelineContext
from hilbert_orchestrator.core.registry import GLOBAL_STAGE_REGISTRY

from hilbert_pipeline import run_full_export


@GLOBAL_STAGE_REGISTRY.decorator(
    key="export",
    order=10.0,
    label="Deterministic export",
    required=False,
    depends_on=[
        "element_labels",
        "molecules",
        "compound_stability",
    ],
    dialectic_role="export",
)
def stage_export(ctx: PipelineContext) -> None:
    """
    Final export stage: build manifest + ZIP and register artefacts.

    The underlying :func:`run_full_export` function is responsible for:
      - calling :func:`build_manifest`
      - writing ``hilbert_manifest.json``
      - writing ``hilbert_export.zip``
      - emitting an ``artifact`` event for the ZIP with kind
        ``hilbert_export_zip``

    We simply delegate and log around it.
    """
    ctx.log("info", "Running deterministic export stage")

    try:
        run_full_export(ctx.results_dir, emit=ctx.emit)
        ctx.log("info", "Deterministic export stage complete")
    except Exception as exc:  # noqa: BLE001 - export is helpful but not core
        ctx.log(
            "warn",
            "Deterministic export stage failed - run results remain on disk",
            error=str(exc),
        )
