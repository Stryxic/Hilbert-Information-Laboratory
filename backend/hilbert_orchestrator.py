"""
Hilbert Orchestrator 3.0

Declarative, dialectical orchestration of the Hilbert information pipeline.

- Stages are defined in a single declarative table (STAGE_TABLE).
- Each stage is a node in a dialectical graph, with explicit dependencies,
  support edges, and challenge edges.
- The orchestrator executes stages in order, respecting dependencies and
  surfacing a rich run summary that the frontend can render.

This file is intentionally self-contained: the only assumptions about other
modules are the public functions re-exported by `hilbert_pipeline.__init__`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Pipeline imports (all via hilbert_pipeline public API)
# -----------------------------------------------------------------------------

try:
    from hilbert_pipeline import (
        DEFAULT_EMIT,

        # LSA
        build_lsa_field,

        # Element descriptions
        build_element_descriptions,

        # Fusion (span -> element)
        run_fusion_pipeline,

        # Molecules
        run_molecule_layer,

        # Stability
        compute_signal_stability,

        # Persistence
        plot_persistence_field,
        run_persistence_visuals,

        # Graphs
        generate_graph_snapshots,
        export_graph_snapshots,

        # Epistemic signatures
        compute_signatures,

        # Full export (PDF + ZIP)
        run_full_export,

        # Element LM
        run_element_lm_stage,
    )
except Exception as exc:  # very defensive import
    raise RuntimeError(f"[orchestrator] Failed to import hilbert_pipeline: {exc}") from exc


EmitFn = Callable[[str, Dict[str, Any]], None]


# -----------------------------------------------------------------------------
# Settings and context objects
# -----------------------------------------------------------------------------

@dataclass
class PipelineSettings:
    """
    User-tunable knobs.

    Keep this conservative - anything heavy should be configured per stage.
    """
    use_native: bool = True
    max_docs: Optional[int] = None
    random_seed: int = 13
    # Future: thermodynamic controls, misinfo sensitivity, etc.


@dataclass
class StageState:
    """Runtime state for a single stage execution."""
    key: str
    label: str
    status: Literal["pending", "running", "ok", "skipped", "failed"] = "pending"
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        if self.start_ts is None or self.end_ts is None:
            return None
        return self.end_ts - self.start_ts


@dataclass
class StageSpec:
    """
    Declarative description of a pipeline stage.

    dialectic_role:
        - "evidence"  - generates or sharpens observations
        - "structure" - reshapes observations into higher-order structure
        - "thermo"    - computes thermodynamic / stability metrics
        - "epistemic" - evaluates truth, intent, misinfo signatures
        - "visual"    - builds views on the structure
        - "export"    - packages results
    """
    key: str
    order: float
    label: str
    func: Optional[Callable[["PipelineContext"], None]]
    required: bool = True
    dialectic_role: Literal[
        "evidence", "structure", "thermo", "epistemic", "visual", "export"
    ] = "structure"
    depends_on: List[str] = field(default_factory=list)
    supports: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    consumes: List[str] = field(default_factory=list)


@dataclass
class PipelineContext:
    """
    Shared context for a single run.

    The orchestrator deals with directories, paths and coarse messages.
    Internal math lives inside hilbert_pipeline modules.
    """
    corpus_dir: str
    results_dir: str
    settings: PipelineSettings = field(default_factory=PipelineSettings)
    emit: EmitFn = DEFAULT_EMIT

    # Computed during the run
    run_id: str = field(default_factory=lambda: str(int(time.time())))
    stages: Dict[str, StageState] = field(default_factory=dict)
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    def log(self, level: str, msg: str, **fields: Any) -> None:
        payload = {"level": level, "msg": msg, "ts": time.time()}
        payload.update(fields)
        try:
            self.emit("log", payload)
        except Exception:
            # Fallback to stdout if user supplied an emit that fails
            print(f"[{level}] {msg} {fields}")

    # --- stage lifecycle -----------------------------------------------------

    def begin_stage(self, spec: StageSpec) -> StageState:
        state = self.stages.get(spec.key) or StageState(key=spec.key, label=spec.label)
        state.status = "running"
        state.start_ts = time.time()
        self.stages[spec.key] = state
        self.log("info", f"{spec.label} - starting", stage=spec.key)
        return state

    def end_stage_ok(self, spec: StageSpec, meta: Optional[Dict[str, Any]] = None) -> None:
        state = self.stages[spec.key]
        state.status = "ok"
        state.end_ts = time.time()
        if meta:
            state.meta.update(meta)
        self.log("info", f"{spec.label} - ok", stage=spec.key, duration=state.duration)

    def end_stage_skipped(self, spec: StageSpec, reason: str) -> None:
        state = self.stages.get(spec.key) or StageState(key=spec.key, label=spec.label)
        state.status = "skipped"
        state.error = reason
        state.start_ts = state.start_ts or time.time()
        state.end_ts = time.time()
        self.stages[spec.key] = state
        self.log("warn", f"{spec.label} - skipped: {reason}", stage=spec.key)

    def end_stage_failed(self, spec: StageSpec, error: str) -> None:
        state = self.stages[spec.key]
        state.status = "failed"
        state.error = error
        state.end_ts = time.time()
        self.log("error", f"{spec.label} - failed: {error}", stage=spec.key)

    # --- artifact tracking ---------------------------------------------------

    def add_artifact(self, name: str, kind: str, **meta: Any) -> None:
        path = os.path.join(self.results_dir, name)
        self.artifacts[name] = {"kind": kind, "path": path, **meta}

    # --- dialectical graph export -------------------------------------------

    def dialectic_graph(self, specs: List[StageSpec]) -> Dict[str, Any]:
        """Return a JSON-serialisable dialectic graph describing this run."""
        nodes = []
        edges = []

        for spec in specs:
            st = self.stages.get(spec.key)
            nodes.append(
                {
                    "id": spec.key,
                    "label": spec.label,
                    "role": spec.dialectic_role,
                    "status": getattr(st, "status", "pending"),
                    "duration": getattr(st, "duration", None),
                    "required": spec.required,
                    "produces": spec.produces,
                    "consumes": spec.consumes,
                }
            )

            for dep in spec.depends_on:
                edges.append({"source": dep, "target": spec.key, "type": "depends_on"})
            for s in spec.supports:
                edges.append({"source": spec.key, "target": s, "type": "supports"})
            for c in spec.challenges:
                edges.append({"source": spec.key, "target": c, "type": "challenges"})

        return {"nodes": nodes, "edges": edges}

    # --- run summary --------------------------------------------------------

    def run_summary(self, specs: List[StageSpec]) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "corpus_dir": self.corpus_dir,
            "results_dir": self.results_dir,
            "settings": self.settings.__dict__,
            "stages": {
                key: {
                    "label": st.label,
                    "status": st.status,
                    "error": st.error,
                    "duration": st.duration,
                    "meta": st.meta,
                }
                for key, st in self.stages.items()
            },
            "artifacts": self.artifacts,
            "dialectic_graph": self.dialectic_graph(specs),
        }


# -----------------------------------------------------------------------------
# Stage implementations (wrappers around hilbert_pipeline functions)
# -----------------------------------------------------------------------------

def _stage_lsa(ctx: PipelineContext) -> None:
    """
    Run LSA, then write hilbert_elements.csv in a stability-compatible format.

    We:
      - call build_lsa_field() to compute the spectral field
      - store the raw result in ctx.extras["lsa_result"]
      - export:
          * hilbert_elements.csv   (element-level table)
          * lsa_field.json         (flattened LSA field: embeddings + span_map)
    """
    res = build_lsa_field(ctx.corpus_dir, emit=ctx.emit) or {}
    ctx.extras["lsa_result"] = res

    elements = res.get("elements", []) or []
    metrics = res.get("element_metrics", []) or []

    field = res.get("field", {}) or {}
    span_map = field.get("span_map", []) or []
    embeddings = field.get("embeddings")

    # Convert ndarray embeddings to list-of-lists for JSON, but keep ndarray for internal use
    if isinstance(embeddings, np.ndarray):
        emb_array = embeddings
        emb_for_json = embeddings.tolist()
    else:
        emb_array = np.asarray(embeddings or [], dtype=float)
        emb_for_json = embeddings

    # ---------------------------------------------
    # Build element-level rows
    # ---------------------------------------------
    element_rows = []
    for rec in elements:
        el = rec.get("element")
        idx = rec.get("index")
        element_rows.append(
            {
                "element": el,
                "index": idx,
                # we keep span linkage separate via span_map / fusion;
                # these columns remain for backward compatibility
                "span_id": -1,
                "doc": None,
                "text": None,
            }
        )

    elements_df = pd.DataFrame(element_rows)

    # ---------------------------------------------
    # Build metrics dataframe
    # ---------------------------------------------
    metrics_df = pd.DataFrame(metrics)

    # Standardise names if mean_* present
    if "mean_entropy" in metrics_df.columns and "entropy" not in metrics_df.columns:
        metrics_df["entropy"] = metrics_df["mean_entropy"]
    if "mean_coherence" in metrics_df.columns and "coherence" not in metrics_df.columns:
        metrics_df["coherence"] = metrics_df["mean_coherence"]

    # ---------------------------------------------
    # Merge metrics into element rows
    # ---------------------------------------------
    if not metrics_df.empty:
        merged = elements_df.merge(metrics_df, on=["element", "index"], how="left")
    else:
        merged = elements_df

    # ---------------------------------------------
    # Write hilbert_elements.csv
    # ---------------------------------------------
    elements_path = os.path.join(ctx.results_dir, "hilbert_elements.csv")
    merged.to_csv(elements_path, index=False)
    ctx.add_artifact("hilbert_elements.csv", "elements")

    # ---------------------------------------------
    # Write LSA field (flat for compatibility)
    # ---------------------------------------------
    lsa_flat = {
        "embeddings": emb_array.tolist() if isinstance(emb_array, np.ndarray) else emb_for_json,
        "span_map": span_map,
        "vocab": field.get("vocab"),
    }
    with open(os.path.join(ctx.results_dir, "lsa_field.json"), "w", encoding="utf-8") as f:
        json.dump(lsa_flat, f, indent=2)

    ctx.add_artifact("lsa_field.json", "lsa-field")


def _stage_molecules(ctx: PipelineContext) -> None:
    """
    Molecule construction & compounds.

    Uses run_molecule_layer(results_dir, emit) from hilbert_pipeline.

    Produces:
      - molecules.csv
      - informational_compounds.json (handled inside run_molecule_layer)
    """
    mol_df, comp_df = run_molecule_layer(ctx.results_dir, emit=ctx.emit)

    # Molecule table
    if isinstance(mol_df, pd.DataFrame) and not mol_df.empty:
        path = os.path.join(ctx.results_dir, "molecules.csv")
        try:
            mol_df.to_csv(path, index=False)
            ctx.add_artifact("molecules.csv", "molecule-table")
        except Exception as exc:
            ctx.log("warn", f"Failed to write molecules.csv: {exc}")

    # Compound JSON – produced by the molecule layer
    comp_path = os.path.join(ctx.results_dir, "informational_compounds.json")
    if os.path.exists(comp_path):
        ctx.add_artifact("informational_compounds.json", "informational-compounds")


def _stage_fusion(ctx: PipelineContext) -> None:
    """
    Span → element fusion and compound context enrichment.

    Delegates to hilbert_pipeline.run_fusion_pipeline(results_dir, emit).
    """
    run_fusion_pipeline(ctx.results_dir, emit=ctx.emit)


def _stage_stability(ctx: PipelineContext) -> None:
    """
    Compute signal stability metrics over elements.

    Uses compute_signal_stability(elements_csv, out_csv, mode, emit).
    If entropy/coherence fields are missing, the stability module will log
    a warning and abort gracefully.
    """
    elements_csv = os.path.join(ctx.results_dir, "hilbert_elements.csv")
    out_csv = os.path.join(ctx.results_dir, "signal_stability.csv")
    compute_signal_stability(
        elements_csv=elements_csv,
        out_csv=out_csv,
        mode="classic",
        emit=ctx.emit,
    )
    # The stability module will decide whether it actually wrote the file.
    if os.path.exists(out_csv):
        ctx.add_artifact("signal_stability.csv", "signal-stability")


def _stage_persistence(ctx: PipelineContext) -> None:
    """
    Persistence and stability visuals.

    Delegates to run_persistence_visuals(results_dir, emit) which handles:
      - persistence_field.png
      - stability_scatter.png
      - stability_by_doc.png
    """
    run_persistence_visuals(ctx.results_dir, emit=ctx.emit)

    for fname, kind in [
        ("persistence_field.png", "persistence-field"),
        ("stability_scatter.png", "stability-scatter"),
        ("stability_by_doc.png", "stability-by-doc"),
    ]:
        path = os.path.join(ctx.results_dir, fname)
        if os.path.exists(path):
            ctx.add_artifact(fname, kind)


def _stage_element_labels(ctx: PipelineContext) -> None:
    """
    Build element labels and human-readable descriptions.

    We pass the span_map from the LSA layer (if available) for nicer contexts.
    """
    spans = (ctx.extras.get("lsa_result") or {}).get("field", {}).get("span_map", [])
    build_element_descriptions(
        elements_csv=os.path.join(ctx.results_dir, "hilbert_elements.csv"),
        spans=spans,
        out_dir=ctx.results_dir,
    )
    ctx.add_artifact("element_descriptions.json", "element-descriptions")
    ctx.add_artifact("element_intensity.csv", "element-intensity")


def _stage_signatures(ctx: PipelineContext) -> None:
    """
    Compute epistemic / misinfo signatures.

    The underlying module is tolerant of missing labels and will no-op gracefully.
    """
    out_path = compute_signatures(ctx.results_dir, emit=ctx.emit)
    if out_path:
        ctx.add_artifact(os.path.basename(out_path), "epistemic-signatures")


def _stage_graph_snapshots(ctx: PipelineContext) -> None:
    """
    Generate graph snapshots.

    Relies on the hilbert_pipeline.graph_snapshots.generate_graph_snapshots
    implementation, which reads hilbert_elements.csv / edges.csv / molecules.csv
    from results_dir and writes PNGs (graph_*.png) into results_dir.
    """
    try:
        generate_graph_snapshots(ctx.results_dir, emit=ctx.emit)
        # The underlying module writes graph_*.png; we register the directory as a logical artifact.
        ctx.add_artifact("graph_snapshots", "graph-snapshots", artifact_kind="directory")
    except Exception as exc:
        ctx.log("error", f"[graph] Snapshot generation failed: {exc}")


def _stage_graph_export(ctx: PipelineContext) -> None:
    """
    Legacy graph export (DOT / PNG variants).

    Delegates to export_graph_snapshots(results_dir, emit).
    """
    export_graph_snapshots(ctx.results_dir, emit=ctx.emit)
    ctx.add_artifact("graph_export", "graph-export", artifact_kind="directory")


def _stage_element_lm(ctx: PipelineContext) -> None:
    """
    Train the element-level language model on span-element sequences.

    The training code will no-op gracefully if span_element_fusion.csv is missing
    or underspecified.
    """
    run_element_lm_stage(ctx.results_dir, emit=ctx.emit)

    # Register typical LM artifacts if they exist
    for fname, kind in [
        ("element_lm.pt", "element-lm"),
        ("element_vocab.json", "element-vocab"),
    ]:
        path = os.path.join(ctx.results_dir, fname)
        if os.path.exists(path):
            ctx.add_artifact(fname, kind)


def _stage_full_export(ctx: PipelineContext) -> None:
    """
    Full export: summary PDF + ZIP bundle.

    Delegates to run_full_export(results_dir, emit).
    """
    run_full_export(ctx.results_dir, emit=ctx.emit)

    for fname, kind in [
        ("hilbert_summary.pdf", "summary-pdf"),
        ("hilbert_run.zip", "archive-zip"),
    ]:
        path = os.path.join(ctx.results_dir, fname)
        if os.path.exists(path):
            ctx.add_artifact(fname, kind)


# -----------------------------------------------------------------------------
# Declarative stage table - dialectical graph definition
# -----------------------------------------------------------------------------

STAGE_TABLE: List[StageSpec] = [
    StageSpec(
        key="lsa_field",
        order=1.0,
        label="[1] LSA spectral field",
        func=_stage_lsa,
        required=True,
        dialectic_role="evidence",
        produces=["lsa_field.json", "hilbert_elements.csv"],
    ),
    StageSpec(
        key="molecules",
        order=2.0,
        label="[2] Molecule layer",
        func=_stage_molecules,
        required=False,
        dialectic_role="structure",
        depends_on=["lsa_field"],
        consumes=["hilbert_elements.csv"],
        produces=["molecules.csv", "informational_compounds.json"],
        supports=["graph_snapshots"],
    ),
    StageSpec(
        key="fusion",
        order=3.0,
        label="[3] Span - element fusion",
        func=_stage_fusion,
        required=False,
        dialectic_role="structure",
        depends_on=["lsa_field", "molecules"],
        consumes=["hilbert_elements.csv"],
        produces=["span_element_fusion.csv", "compound_contexts.json"],
    ),
    StageSpec(
        key="stability_metrics",
        order=4.0,
        label="[4] Signal stability metrics",
        func=_stage_stability,
        required=False,
        dialectic_role="thermo",
        depends_on=["lsa_field"],
        consumes=["hilbert_elements.csv"],
        produces=["signal_stability.csv"],
        supports=["persistence_visuals", "epistemic_signatures"],
    ),
    StageSpec(
        key="persistence_visuals",
        order=5.0,
        label="[5] Persistence and stability visuals",
        func=_stage_persistence,
        required=False,
        dialectic_role="visual",
        depends_on=["stability_metrics"],
        consumes=["signal_stability.csv"],
        produces=[
            "persistence_field.png",
            "stability_scatter.png",
            "stability_by_doc.png",
        ],
    ),
    StageSpec(
        key="element_labels",
        order=6.0,
        label="[6] Element labels and descriptions",
        func=_stage_element_labels,
        required=True,
        dialectic_role="structure",
        depends_on=["lsa_field"],
        consumes=["hilbert_elements.csv"],
        produces=["element_descriptions.json", "element_intensity.csv"],
        supports=["epistemic_signatures"],
    ),
    StageSpec(
        key="epistemic_signatures",
        order=6.5,
        label="[6.5] Epistemic signatures (misinfo layer)",
        func=_stage_signatures,
        required=False,
        dialectic_role="epistemic",
        depends_on=["element_labels", "stability_metrics"],
        consumes=["hilbert_elements.csv", "signal_stability.csv"],
        produces=["epistemic_signatures.json"],
        challenges=["raw_corpus"],  # conceptual node only
    ),
    StageSpec(
        key="element_lm",
        order=6.8,
        label="[6.8] Element Language Model",
        func=_stage_element_lm,
        required=False,
        dialectic_role="epistemic",
        depends_on=["fusion"],
        consumes=["hilbert_elements.csv", "span_element_fusion.csv"],
        produces=["element_lm.pt", "element_vocab.json"],
    ),
    StageSpec(
        key="graph_snapshots",
        order=7.0,
        label="[7] Graph snapshots",
        func=_stage_graph_snapshots,
        required=False,
        dialectic_role="visual",
        depends_on=["molecules"],
        produces=["graph_snapshots"],
    ),
    StageSpec(
        key="graph_export",
        order=7.5,
        label="[7.5] Graph export",
        func=_stage_graph_export,
        required=False,
        dialectic_role="visual",
        depends_on=["graph_snapshots"],
        produces=["graph_export"],
    ),
    StageSpec(
        key="export_all",
        order=8.0,
        label="[8] Full export (PDF + archive)",
        func=_stage_full_export,
        required=False,
        dialectic_role="export",
        depends_on=["element_labels"],
        supports=["epistemic_signatures"],
        produces=["hilbert_summary.pdf", "hilbert_run.zip"],
    ),
]


# -----------------------------------------------------------------------------
# Execution engine
# -----------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_pipeline(
    corpus_dir: str,
    results_dir: str,
    settings: Optional[PipelineSettings] = None,
    emit: Optional[EmitFn] = None,
) -> Dict[str, Any]:
    """
    Run the full Hilbert pipeline on `corpus_dir`, writing outputs to
    `results_dir`.

    Returns a JSON-serialisable run summary for UI and API use.
    """
    settings = settings or PipelineSettings()
    emit = emit or DEFAULT_EMIT

    _ensure_dir(results_dir)
    ctx = PipelineContext(
        corpus_dir=os.path.abspath(corpus_dir),
        results_dir=os.path.abspath(results_dir),
        settings=settings,
        emit=emit,
    )

    ctx.log("info", "Starting Hilbert pipeline run", corpus=ctx.corpus_dir)

    for spec in sorted(STAGE_TABLE, key=lambda s: s.order):
        # Check hard dependencies
        unmet = [
            d for d in spec.depends_on
            if ctx.stages.get(d, StageState(d, d)).status != "ok"
        ]

        if unmet:
            reason = f"dependencies not satisfied: {', '.join(unmet)}"
            if spec.required:
                ctx.begin_stage(spec)
                ctx.end_stage_failed(spec, error=reason)
                break
            else:
                ctx.end_stage_skipped(spec, reason=reason)
                continue

        # Conceptual node only
        if spec.func is None:
            ctx.end_stage_skipped(spec, reason="no-op conceptual stage")
            continue

        ctx.begin_stage(spec)
        try:
            spec.func(ctx)
            ctx.end_stage_ok(spec)
        except Exception as exc:
            ctx.end_stage_failed(spec, error=str(exc))
            if spec.required:
                ctx.log(
                    "error",
                    "Aborting pipeline due to failure in required stage.",
                    stage=spec.key,
                )
                break

    summary = ctx.run_summary(STAGE_TABLE)

    # Persist run summary
    try:
        run_json_path = os.path.join(ctx.results_dir, "hilbert_run.json")
        with open(run_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        ctx.add_artifact("hilbert_run.json", "run-summary")
    except Exception as exc:
        ctx.log("warn", f"Failed to write hilbert_run.json: {exc}")

    ctx.log("info", "Hilbert pipeline run complete", run_id=ctx.run_id)
    return summary


# Backwards-compatible alias
run_hilbert_pipeline = run_pipeline
