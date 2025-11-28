"""
molecule_stage.py
=================

Implements the Molecule Layer stage of the Hilbert Information Pipeline.

This stage consumes:
    • element_edges.csv (produced by the Edges stage)

and produces:
    • molecules.csv
    • informational_compounds.json

The molecule layer identifies higher-order informational structures:
coherent clusters, compound groupings, and proto-hierarchical dependencies.
"""

from __future__ import annotations

from hilbert_pipeline import run_molecule_stage
from hilbert_orchestrator.core.context import PipelineContext
from hilbert_orchestrator.core.registry import GLOBAL_STAGE_REGISTRY


@GLOBAL_STAGE_REGISTRY.decorator(
    key="molecules",
    order=4.0,
    label="Molecule layer",
    required=True,
    depends_on=["edges"],
)
def stage_molecule_layer(ctx: PipelineContext) -> None:
    ctx.log("info", "Running molecule layer")

    # Correct API call
    run_molecule_stage(ctx.results_dir, emit=ctx.emit)

    # These filenames match what the real molecule layer writes
    ctx.add_artifact("molecules.csv", kind="molecules")
    ctx.add_artifact("informational_compounds.json", kind="molecules")
