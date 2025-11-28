"""
lsa_stage.py
============

Implements the LSA spectral field stage for the modular
Hilbert Information Pipeline orchestrator.

This stage performs:

    • Corpus normalization (delegated to corpus_loader upstream)
    • LSA spectral embedding
    • Element extraction
    • Element metric merging
    • Writing:
          - lsa_field.json
          - hilbert_elements.csv

The results are saved into the PipelineContext and registered
as artifacts so that downstream stages (fusion, edges, stability)
can consume them.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from hilbert_pipeline import build_lsa_field

from hilbert_orchestrator.core.registry import GLOBAL_STAGE_REGISTRY
from hilbert_orchestrator.core.context import PipelineContext


# ---------------------------------------------------------------------------
# Stage Registration
# ---------------------------------------------------------------------------

@GLOBAL_STAGE_REGISTRY.decorator(
    key="lsa_field",
    order=1.0,
    label="LSA spectral field",
    required=True,
)
def lsa_stage(ctx: PipelineContext) -> None:
    """
    Execute the LSA spectral field stage.

    Parameters
    ----------
    ctx : PipelineContext
        The pipeline execution context containing corpus paths,
        DB handles, settings, and artifact registries.

    Side Effects
    ------------
    • Writes hilbert_elements.csv
    • Writes lsa_field.json
    • Stores a full LSA result in ctx.extras["lsa_result"]
    • Registers artifacts with ctx.add_artifact(...)
    """

    ctx.log("info", "Running LSA spectral field stage")

    # ----------------------------------------------------------------------
    # 1. Invoke backend LSA
    # ----------------------------------------------------------------------

    corpus_path = ctx.extras.get("normalized_corpus", ctx.corpus_dir)
    lsa_result = build_lsa_field(corpus_path, emit=ctx.emit) or {}
    ctx.extras["lsa_result"] = lsa_result

    field = lsa_result.get("field", {}) or {}
    embeddings = field.get("embeddings")
    span_map = field.get("span_map")

    if isinstance(embeddings, np.ndarray):
        embeddings = embeddings.tolist()

    # ----------------------------------------------------------------------
    # 2. Write lsa_field.json
    # ----------------------------------------------------------------------

    lsa_field_json = {
        "embeddings": embeddings,
        "span_map": span_map,
        "vocab": field.get("vocab"),
    }

    lsa_json_path = os.path.join(ctx.results_dir, "lsa_field.json")
    with open(lsa_json_path, "w", encoding="utf-8") as f:
        json.dump(lsa_field_json, f, indent=2)

    ctx.add_artifact("lsa_field.json", kind="lsa-field")

    # ----------------------------------------------------------------------
    # 3. Generate hilbert_elements.csv
    # ----------------------------------------------------------------------

    elements = lsa_result.get("elements", [])
    metrics = lsa_result.get("element_metrics", [])

    el_df = pd.DataFrame(elements)
    met_df = pd.DataFrame(metrics)

    # Patch inconsistent names from older pipeline versions
    if "entropy" not in met_df and "mean_entropy" in met_df:
        met_df["entropy"] = met_df["mean_entropy"]
    if "coherence" not in met_df and "mean_coherence" in met_df:
        met_df["coherence"] = met_df["mean_coherence"]

    merged = el_df.merge(met_df, on=["element", "index"], how="left")

    el_csv_path = os.path.join(ctx.results_dir, "hilbert_elements.csv")
    merged.to_csv(el_csv_path, index=False)

    ctx.add_artifact("hilbert_elements.csv", kind="elements")

    ctx.log("info", "LSA spectral field stage complete")
