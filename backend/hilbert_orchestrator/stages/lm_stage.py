"""
hilbert_orchestrator/stages/lm_stage.py
=======================================

Optional language-model stages for the Hilbert Orchestrator.

This module declares two *non-required* stages:

1. ``element_lm``:
   - Runs the element-level language model over the element graph.
   - Uses :func:`hilbert_pipeline.run_element_lm_stage`.

2. ``lm_perplexity``:
   - Computes global corpus perplexity via the Ollama LM layer.
   - Uses :func:`hilbert_pipeline.compute_corpus_perplexity`.

Both stages are designed to fail *softly*: if the underlying LM modules are
unavailable or raise, we log a warning and continue the pipeline.
"""

from __future__ import annotations

from typing import Any

from hilbert_orchestrator.core.context import PipelineContext
from hilbert_orchestrator.core.registry import GLOBAL_STAGE_REGISTRY

from hilbert_pipeline import (
    run_element_lm_stage,
    compute_corpus_perplexity,
)


# ---------------------------------------------------------------------------
# Element-level LM
# ---------------------------------------------------------------------------


@GLOBAL_STAGE_REGISTRY.decorator(
    key="element_lm",
    order=7.0,
    label="Element language model",
    required=False,
    depends_on=["element_labels"],
    dialectic_role="evidence",
    supports=["lm_perplexity"],
)
def stage_element_lm(ctx: PipelineContext) -> None:
    """
    Optional stage: train / run the element-level LM.

    Errors are logged but do *not* halt the pipeline. This stage is primarily
    for research / diagnostics and may be disabled in production.
    """
    ctx.log("info", "Running element LM stage")

    try:
        run_element_lm_stage(ctx.results_dir, emit=ctx.emit)
        ctx.log("info", "Element LM stage complete")
    except Exception as exc:  # noqa: BLE001 - we deliberately soft-fail here
        ctx.log(
            "warn",
            "Element LM stage failed - continuing without LM outputs",
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Corpus-level LM perplexity
# ---------------------------------------------------------------------------


@GLOBAL_STAGE_REGISTRY.decorator(
    key="lm_perplexity",
    order=7.5,
    label="Corpus LM perplexity",
    required=False,
    depends_on=["element_labels"],
    dialectic_role="evidence",
)
def stage_perplexity(ctx: PipelineContext) -> None:
    """
    Optional stage: compute corpus-level LM perplexity.

    This calls the Ollama LM layer. If the LM backend is not available, the
    underlying :mod:`hilbert_pipeline.ollama_lm` module provides safe stubs.
    """
    ctx.log("info", "Running corpus LM perplexity stage")

    try:
        compute_corpus_perplexity()
        ctx.log("info", "Corpus LM perplexity stage complete")
    except Exception as exc:  # noqa: BLE001 - optional diagnostics
        ctx.log(
            "warn",
            "Corpus LM perplexity stage failed - continuing without perplexity",
            error=str(exc),
        )
