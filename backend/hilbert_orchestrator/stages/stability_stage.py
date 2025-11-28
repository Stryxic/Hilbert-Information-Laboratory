"""
hilbert_orchestrator/stages/stability_stage.py
==============================================

Stability and persistence stages for the Hilbert Orchestrator.

This module exposes three related stages:

1. ``stability_metrics``:
   - Per-element stability and related metrics from the signatures layer.
   - Uses :func:`hilbert_pipeline.compute_signal_stability`.

2. ``compound_stability``:
   - Aggregate stability metrics to informational compounds / molecules.
   - Uses :func:`hilbert_pipeline.compute_compound_stability`.

3. ``persistence``:
   - Generate persistence-field visualisations.
   - Uses :func:`hilbert_pipeline.run_persistence_visuals`.

All three stages are treated as *optional analytics*: missing inputs are
handled gracefully with warnings rather than hard failures.
"""

from __future__ import annotations

from hilbert_orchestrator.core.context import PipelineContext
from hilbert_orchestrator.core.registry import GLOBAL_STAGE_REGISTRY

from hilbert_pipeline import (
    compute_signal_stability,
    compute_compound_stability,
    run_persistence_visuals,
)

import os

# ---------------------------------------------------------------------------
# Element-level stability metrics
# ---------------------------------------------------------------------------


@GLOBAL_STAGE_REGISTRY.decorator(
    key="stability_metrics",
    order=8.5,
    label="Element stability metrics",
    required=False,
    depends_on=["signatures"],
    dialectic_role="evidence",
    supports=["compound_stability", "persistence"],
)
def stage_stability(ctx: PipelineContext) -> None:
    """
    Compute per-element stability metrics from epistemic signatures.
    """
    ctx.log("info", "Running stability metrics stage")

    try:
        compute_signal_stability(
    ctx.results_dir,
    out_csv=os.path.join(ctx.results_dir, "signal_stability.csv")
)
        ctx.log("info", "Stability metrics stage complete")
    except FileNotFoundError as exc:
        ctx.log(
            "warn",
            "Stability metrics inputs missing - skipping stability stage",
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - analytics, not core pipeline
        ctx.log(
            "warn",
            "Stability metrics stage failed - continuing without stability outputs",
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Compound-level stability
# ---------------------------------------------------------------------------


@GLOBAL_STAGE_REGISTRY.decorator(
    key="compound_stability",
    order=8.7,
    label="Compound stability",
    required=False,
    depends_on=["stability_metrics", "molecules"],
    dialectic_role="structure",
)
# compound_stability_stage.py


def stage_compound_stability(ctx):
    ctx.log("Running compound stability stage")

    # ------------------------------------------------------------------
    # Required file paths
    # ------------------------------------------------------------------
    compounds_json = os.path.join(ctx.results_dir, "informational_compounds.json")
    elements_csv   = os.path.join(ctx.results_dir, "hilbert_elements.csv")
    stability_csv  = os.path.join(ctx.results_dir, "signal_stability.csv")
    out_csv        = os.path.join(ctx.results_dir, "compound_stability.csv")

    # ------------------------------------------------------------------
    # Dependency checks
    # ------------------------------------------------------------------
    missing = []
    for p in (compounds_json, elements_csv, stability_csv):
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        ctx.warn(
            "Compound stability skipped - missing required inputs",
            {"missing_files": missing}
        )
        return True   # do NOT fail the pipeline

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------
    try:
        compute_compound_stability(
            compounds_json=compounds_json,
            elements_csv=elements_csv,
            stability_csv=stability_csv,
            out_csv=out_csv,
        )

        ctx.log("Compound stability complete", {"out_csv": out_csv})
        return True

    except Exception as exc:
        ctx.warn(
            "Compound stability stage failed - continuing without compound stability",
            {"error": str(exc)}
        )
        return True


# ---------------------------------------------------------------------------
# Persistence field visualisations
# ---------------------------------------------------------------------------


@GLOBAL_STAGE_REGISTRY.decorator(
    key="persistence",
    order=9.0,
    label="Persistence field",
    required=False,
    depends_on=["stability_metrics"],
    dialectic_role="visualisation",
)
def stage_persistence(ctx: PipelineContext) -> None:
    """
    Generate persistence-field visualisations based on the stability layer.
    """
    ctx.log("info", "Running persistence field visualisation stage")

    try:
        run_persistence_visuals(ctx.results_dir, emit=ctx.emit)
        ctx.log("info", "Persistence field visualisation stage complete")
    except FileNotFoundError as exc:
        ctx.log(
            "warn",
            "Persistence inputs missing - skipping persistence visuals",
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        ctx.log(
            "warn",
            "Persistence visualisation stage failed - continuing without visuals",
            error=str(exc),
        )
