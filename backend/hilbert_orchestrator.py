from __future__ import annotations

"""
Hilbert Orchestrator 4.1
Database-integrated deterministic pipeline execution with stage events.

This orchestrator:

- Registers corpora, runs, artifacts in HilbertDB
- Executes the multi-stage Hilbert pipeline
- Produces deterministic exports consumed by hilbert_import
- Stores all artifacts in the DB + object store
- Persists run status and metadata
- Emits structured stage / run events for UI progress streaming
"""

import json
import os
import time
import shutil
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd

from hilbert_db.core import HilbertDB
from hilbert_pipeline import (
    DEFAULT_EMIT,
    build_lsa_field,
    run_fusion_pipeline,
    build_element_edges,
    run_molecule_layer,
    run_element_roots,
    compute_signal_stability,
    run_persistence_visuals,
    compute_signatures,
    run_element_lm_stage,
    compute_corpus_perplexity,
    compute_compound_stability,
    build_element_descriptions,
)
from hilbert_pipeline.hilbert_export import run_full_export

ORCHESTRATOR_VERSION = "4.1"


# ----------------------------------------------------------------------
# PDF helper
# ----------------------------------------------------------------------

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


def _pdf_to_text(path: str) -> str:
    if PdfReader is None:
        return ""
    try:
        texts = []
        r = PdfReader(path)
        for p in r.pages:
            try:
                texts.append(p.extract_text() or "")
            except Exception:
                pass
        return "\n\n".join(texts).strip()
    except Exception:
        return ""


# ----------------------------------------------------------------------
# Pipeline settings and state
# ----------------------------------------------------------------------

EmitFn = Callable[[str, Dict[str, Any]], None]


@dataclass
class PipelineSettings:
    use_native: bool = True
    max_docs: Optional[int] = None
    random_seed: int = 13


@dataclass
class StageState:
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
    key: str
    order: float
    label: str
    func: Optional[Callable[["PipelineContext"], None]]
    required: bool = True
    dialectic_role: str = "structure"
    depends_on: List[str] = field(default_factory=list)
    supports: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    consumes: List[str] = field(default_factory=list)


@dataclass
class PipelineContext:
    corpus_dir: str
    results_dir: str
    settings: PipelineSettings
    emit: EmitFn
    run_id: str
    db: HilbertDB

    stages: Dict[str, StageState] = field(default_factory=dict)
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    def log(self, level: str, msg: str, **fields: Any) -> None:
        entry = {"level": level, "msg": msg, "ts": time.time()}
        entry.update(fields)
        try:
            self.emit("log", entry)
        except Exception:
            # Last-resort fallback to stdout
            print(f"[{level}] {msg} {fields}")

    # ------------------------------------------------------------------
    # Stage lifecycle helpers
    # ------------------------------------------------------------------

    def begin_stage(self, spec: StageSpec) -> StageState:
        st = self.stages.get(spec.key) or StageState(spec.key, spec.label)
        st.status = "running"
        st.start_ts = time.time()
        self.stages[spec.key] = st
        self.log("info", f"{spec.label} - starting", stage=spec.key)
        return st

    def end_stage_ok(self, spec: StageSpec, meta: Optional[Dict[str, Any]] = None) -> None:
        st = self.stages[spec.key]
        st.status = "ok"
        st.end_ts = time.time()
        if meta:
            st.meta.update(meta)
        self.log("info", f"{spec.label} - ok", stage=spec.key, duration=st.duration)

    def end_stage_failed(self, spec: StageSpec, error: str) -> None:
        st = self.stages[spec.key]
        st.status = "failed"
        st.end_ts = time.time()
        st.error = error
        self.log("error", f"{spec.label} - failed: {error}", stage=spec.key)

    def end_stage_skipped(self, spec: StageSpec, reason: str) -> None:
        st = self.stages.get(spec.key) or StageState(spec.key, spec.label)
        st.status = "skipped"
        st.error = reason
        st.start_ts = st.start_ts or time.time()
        st.end_ts = time.time()
        self.stages[spec.key] = st
        self.log("warn", f"{spec.label} - skipped ({reason})", stage=spec.key)

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def add_artifact(self, name: str, kind: str, **meta: Any) -> None:
        path = os.path.join(self.results_dir, name)
        self.artifacts[name] = {
            "kind": kind,
            "path": path,
            **meta,
        }


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _prepare_lsa_corpus(ctx: PipelineContext) -> str:
    import pathlib

    src_root = pathlib.Path(ctx.corpus_dir).resolve()
    dst_root = pathlib.Path(ctx.results_dir).resolve() / "_normalized_corpus"

    shutil.rmtree(dst_root, ignore_errors=True)
    dst_root.mkdir(parents=True, exist_ok=True)

    paths = [p for p in src_root.rglob("*") if p.is_file()]
    paths.sort()

    if ctx.settings.max_docs:
        paths = paths[: ctx.settings.max_docs]

    for p in paths:
        rel = p.relative_to(src_root)
        out_dir = dst_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        root, ext = os.path.splitext(rel.name)
        ext = ext.lower()

        if ext == ".pdf":
            txt = _pdf_to_text(str(p))
            if not txt.strip():
                continue
            with open(out_dir / f"{root}.txt", "w", encoding="utf-8") as f:
                f.write(txt)
        else:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as fs, \
                        open(out_dir / rel.name, "w", encoding="utf-8") as fd:
                    fd.write(fs.read())
            except Exception:
                # Non-fatal; skip unreadable files
                pass

    return str(dst_root)


# ----------------------------------------------------------------------
# Stage implementations
# ----------------------------------------------------------------------

def _stage_lsa(ctx: PipelineContext) -> None:
    corpus = _prepare_lsa_corpus(ctx)
    res = build_lsa_field(corpus, emit=ctx.emit) or {}
    ctx.extras["lsa_result"] = res

    field = res.get("field", {}) or {}
    emb = field.get("embeddings")
    span_map = field.get("span_map")

    if isinstance(emb, np.ndarray):
        emb = emb.tolist()

    out = {
        "embeddings": emb,
        "span_map": span_map,
        "vocab": field.get("vocab"),
    }

    p = os.path.join(ctx.results_dir, "lsa_field.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    ctx.add_artifact("lsa_field.json", "lsa-field")

    elements = res.get("elements", [])
    metrics = res.get("element_metrics", [])

    el_df = pd.DataFrame(elements)
    met_df = pd.DataFrame(metrics)

    if "entropy" not in met_df and "mean_entropy" in met_df:
        met_df["entropy"] = met_df["mean_entropy"]
    if "coherence" not in met_df and "mean_coherence" in met_df:
        met_df["coherence"] = met_df["mean_coherence"]

    merged = el_df.merge(met_df, on=["element", "index"], how="left")
    merged.to_csv(os.path.join(ctx.results_dir, "hilbert_elements.csv"), index=False)
    ctx.add_artifact("hilbert_elements.csv", "elements")


def _stage_fusion(ctx: PipelineContext) -> None:
    run_fusion_pipeline(ctx.results_dir, emit=ctx.emit)
    f = os.path.join(ctx.results_dir, "span_element_fusion.csv")
    if os.path.exists(f):
        ctx.add_artifact("span_element_fusion.csv", "fusion")


def _stage_edges(ctx: PipelineContext) -> None:
    out = build_element_edges(ctx.results_dir, emit=ctx.emit)
    if out:
        ctx.add_artifact(os.path.basename(out), "edges")


def _stage_molecules(ctx: PipelineContext) -> None:
    run_molecule_layer(ctx.results_dir, emit=ctx.emit)
    for x in ["molecules.csv", "informational_compounds.json"]:
        p = os.path.join(ctx.results_dir, x)
        if os.path.exists(p):
            ctx.add_artifact(x, "molecules")


def _stage_element_roots(ctx: PipelineContext) -> None:
    run_element_roots(ctx.results_dir, emit=ctx.emit)
    for x in ["element_roots.csv", "element_cluster_metrics.json"]:
        p = os.path.join(ctx.results_dir, x)
        if os.path.exists(p):
            ctx.add_artifact(x, "element-roots")


def _stage_stability(ctx: PipelineContext) -> None:
    compute_signal_stability(
        os.path.join(ctx.results_dir, "hilbert_elements.csv"),
        os.path.join(ctx.results_dir, "signal_stability.csv"),
        mode="classic",
        emit=ctx.emit,
    )
    if os.path.exists(os.path.join(ctx.results_dir, "signal_stability.csv")):
        ctx.add_artifact("signal_stability.csv", "stability")


def _stage_compound_stability(ctx: PipelineContext) -> None:
    compute_compound_stability(
        compounds_json=os.path.join(ctx.results_dir, "informational_compounds.json"),
        elements_csv=os.path.join(ctx.results_dir, "hilbert_elements.csv"),
        stability_csv=os.path.join(ctx.results_dir, "signal_stability.csv"),
        out_csv=os.path.join(ctx.results_dir, "compound_stability.csv"),
        emit=ctx.emit,
    )
    if os.path.exists(os.path.join(ctx.results_dir, "compound_stability.csv")):
        ctx.add_artifact("compound_stability.csv", "compound-stability")


def _stage_persistence(ctx: PipelineContext) -> None:
    run_persistence_visuals(ctx.results_dir, emit=ctx.emit)
    for x in [
        "persistence_field.png",
        "stability_scatter.png",
        "stability_by_doc.png",
    ]:
        if os.path.exists(os.path.join(ctx.results_dir, x)):
            ctx.add_artifact(x, "persistence")


def _stage_element_labels(ctx: PipelineContext) -> None:
    spans = ctx.extras.get("lsa_result", {}).get("field", {}).get("span_map", [])
    build_element_descriptions(
        os.path.join(ctx.results_dir, "hilbert_elements.csv"),
        spans,
        ctx.results_dir,
    )
    for x in ["element_descriptions.json", "element_intensity.csv"]:
        if os.path.exists(os.path.join(ctx.results_dir, x)):
            ctx.add_artifact(x, "labels")


def _stage_signatures(ctx: PipelineContext) -> None:
    out = compute_signatures(ctx.results_dir, emit=ctx.emit)
    if out:
        ctx.add_artifact(os.path.basename(out), "signatures")


def _stage_element_lm(ctx: PipelineContext) -> None:
    run_element_lm_stage(ctx.results_dir, emit=ctx.emit)
    for x in ["element_lm.pt", "element_vocab.json"]:
        if os.path.exists(os.path.join(ctx.results_dir, x)):
            ctx.add_artifact(x, "element-lm")


def _stage_perplexity(ctx: PipelineContext) -> None:
    text = ctx.extras.get("corpus_text")

    if not text:
        spans = os.path.join(ctx.results_dir, "spans.jsonl")
        if os.path.exists(spans):
            parts: List[str] = []
            with open(spans, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        parts.append(json.loads(line).get("text", ""))
                    except Exception:
                        pass
            text = "\n".join(parts)[:15000]

    compute_corpus_perplexity(
        corpus_text=text or "Hilbert perplexity fallback.",
        out_path=os.path.join(ctx.results_dir, "lm_metrics.json"),
    )
    ctx.add_artifact("lm_metrics.json", "lm-metrics")


def _stage_export(ctx: PipelineContext) -> None:
    """
    Deterministic export stage.

    - Builds hilbert_manifest.json
    - Creates run_<ts>.zip
    - Registers both as artifacts so they are pushed to the object store
      and the ZIP key can be recorded as export_key on the run.
    """
    run_full_export(ctx.results_dir, emit=ctx.emit)

    # Manifest
    manifest_name = "hilbert_manifest.json"
    manifest_path = os.path.join(ctx.results_dir, manifest_name)
    if os.path.exists(manifest_path):
        ctx.add_artifact(manifest_name, "hilbert_manifest_json")

    # ZIP archive name is derived from the results directory basename
    run_name = os.path.basename(os.path.abspath(ctx.results_dir).rstrip(os.sep))
    zip_name = f"{run_name}.zip"
    zip_path = os.path.join(ctx.results_dir, zip_name)
    if os.path.exists(zip_path):
        ctx.add_artifact(zip_name, "hilbert_export_zip")


# ----------------------------------------------------------------------
# Stage Table
# ----------------------------------------------------------------------

STAGE_TABLE: List[StageSpec] = [
    StageSpec(
        "lsa_field",
        1.0,
        "LSA spectral field",
        _stage_lsa,
        dialectic_role="evidence",
        supports=["fusion", "edges"],
    ),
    StageSpec(
        "fusion",
        2.0,
        "Span–element fusion",
        _stage_fusion,
        depends_on=["lsa_field"],
    ),
    StageSpec(
        "edges",
        2.5,
        "Element edges",
        _stage_edges,
        depends_on=["fusion"],
    ),
    StageSpec(
        "molecules",
        3.0,
        "Molecule layer",
        _stage_molecules,
        depends_on=["edges"],
    ),
    StageSpec(
        "element_roots",
        3.5,
        "Element roots",
        _stage_element_roots,
        depends_on=["molecules"],
    ),
    StageSpec(
        "stability_metrics",
        4.0,
        "Stability metrics",
        _stage_stability,
        depends_on=["lsa_field"],
    ),
    StageSpec(
        "compound_stability",
        4.5,
        "Compound stability",
        _stage_compound_stability,
        depends_on=["stability_metrics", "molecules"],
    ),
    StageSpec(
        "persistence",
        5.0,
        "Persistence visuals",
        _stage_persistence,
        depends_on=["stability_metrics"],
    ),
    StageSpec(
        "element_labels",
        6.0,
        "Element labels",
        _stage_element_labels,
        depends_on=["lsa_field"],
    ),
    StageSpec(
        "epistemic_signatures",
        6.5,
        "Epistemic signatures",
        _stage_signatures,
        depends_on=["element_labels", "stability_metrics"],
    ),
    StageSpec(
        "element_lm",
        6.8,
        "Element LM",
        _stage_element_lm,
        depends_on=["fusion"],
    ),
    StageSpec(
        "lm_perplexity",
        6.85,
        "LM Perplexity",
        _stage_perplexity,
        depends_on=["lsa_field"],
    ),
    StageSpec(
        "export_all",
        8.0,
        "Deterministic Export",
        _stage_export,
        depends_on=["element_labels", "compound_stability", "lm_perplexity"],
        dialectic_role="export",
    ),
]


# ----------------------------------------------------------------------
# Orchestration Engine
# ----------------------------------------------------------------------

def run_hilbert_orchestration(
    db: HilbertDB,
    *,
    corpus_dir: str,
    corpus_name: str,
    results_dir: str,
    settings: Optional[PipelineSettings] = None,
    emit: Optional[EmitFn] = None,
) -> Dict[str, Any]:
    """
    Run the full Hilbert pipeline for a corpus.

    Emits structured events:

        emit("run_start", {...})
        emit("stage_start", {...})
        emit("stage_end", {...})
        emit("run_end", {...})

    where payloads always include at least: run_id, stage (for stage_*) and
    orchestrator_version.
    """
    settings = settings or PipelineSettings()
    emit = emit or DEFAULT_EMIT

    random.seed(settings.random_seed)
    np.random.seed(settings.random_seed)

    _ensure_dir(results_dir)

    # ------------------------------------------------------------------
    # Register corpus and run
    # ------------------------------------------------------------------

    # TODO: replace with real content-hash fingerprint
    fingerprint = str(int(time.time()))
    corpus = db.get_or_create_corpus(
        fingerprint=fingerprint,
        name=corpus_name,
        source_uri=corpus_dir,
    )

    run_id = str(int(time.time() * 1000))
    run = db.create_run(
        run_id=run_id,
        corpus_id=corpus.corpus_id,
        orchestrator_version=ORCHESTRATOR_VERSION,
        settings_json=settings.__dict__,
    )

    if hasattr(db, "mark_run_running"):
        db.mark_run_running(run_id)

    ctx = PipelineContext(
        corpus_dir=os.path.abspath(corpus_dir),
        results_dir=os.path.abspath(results_dir),
        settings=settings,
        emit=emit,
        run_id=run_id,
        db=db,
    )

    # Announce run start
    emit(
        "run_start",
        {
            "run_id": run_id,
            "corpus_id": corpus.corpus_id,
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "settings": settings.__dict__,
            "ts": time.time(),
        },
    )

    ordered_specs = sorted(STAGE_TABLE, key=lambda s: s.order)
    total_stages = len(ordered_specs)

    # ------------------------------------------------------------------
    # Execute stages
    # ------------------------------------------------------------------

    for idx, spec in enumerate(ordered_specs):
        unmet = [
            d
            for d in spec.depends_on
            if ctx.stages.get(d, StageState(d, d)).status != "ok"
        ]

        if unmet:
            if spec.required:
                ctx.begin_stage(spec)
                ctx.end_stage_failed(spec, f"Missing dependencies: {unmet}")
                st = ctx.stages[spec.key]
                emit(
                    "stage_end",
                    {
                        "run_id": run_id,
                        "stage": spec.key,
                        "label": spec.label,
                        "status": st.status,
                        "error": st.error,
                        "duration": st.duration,
                        "index": idx,
                        "total_stages": total_stages,
                    },
                )
                # Hard failure: stop pipeline
                break
            else:
                ctx.end_stage_skipped(spec, f"Missing deps: {unmet}")
                st = ctx.stages[spec.key]
                emit(
                    "stage_end",
                    {
                        "run_id": run_id,
                        "stage": spec.key,
                        "label": spec.label,
                        "status": st.status,
                        "error": st.error,
                        "duration": st.duration,
                        "index": idx,
                        "total_stages": total_stages,
                    },
                )
                continue

        # Normal stage execution
        ctx.begin_stage(spec)
        emit(
            "stage_start",
            {
                "run_id": run_id,
                "stage": spec.key,
                "label": spec.label,
                "index": idx,
                "total_stages": total_stages,
                "ts": time.time(),
            },
        )

        try:
            if spec.func is not None:
                spec.func(ctx)
            ctx.end_stage_ok(spec)
        except Exception as exc:
            ctx.end_stage_failed(spec, str(exc))
            # Emit and bail if required
            st = ctx.stages[spec.key]
            emit(
                "stage_end",
                {
                    "run_id": run_id,
                    "stage": spec.key,
                    "label": spec.label,
                    "status": st.status,
                    "error": st.error,
                    "duration": st.duration,
                    "index": idx,
                    "total_stages": total_stages,
                },
            )
            if spec.required:
                break
        else:
            # Successful completion
            st = ctx.stages[spec.key]
            emit(
                "stage_end",
                {
                    "run_id": run_id,
                    "stage": spec.key,
                    "label": spec.label,
                    "status": st.status,
                    "error": st.error,
                    "duration": st.duration,
                    "index": idx,
                    "total_stages": total_stages,
                },
            )

    # ------------------------------------------------------------------
    # Write hilbert_run.json summary
    # ------------------------------------------------------------------

    summary_path = os.path.join(ctx.results_dir, "hilbert_run.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "corpus_id": corpus.corpus_id,
                "orchestrator_version": ORCHESTRATOR_VERSION,
                "settings": settings.__dict__,
                "stages": {
                    k: {
                        "label": st.label,
                        "status": st.status,
                        "error": st.error,
                        "duration": st.duration,
                        "meta": st.meta,
                    }
                    for k, st in ctx.stages.items()
                },
                "artifacts": ctx.artifacts,
            },
            f,
            indent=2,
        )
    ctx.add_artifact("hilbert_run.json", "run-summary")

    # ------------------------------------------------------------------
    # Store run artifacts in DB + object store
    # ------------------------------------------------------------------

    export_key: Optional[str] = None

    for name, info in ctx.artifacts.items():
        local_path = info["path"]
        kind = info["kind"]
        meta = {k: v for k, v in info.items() if k not in ("path", "kind")}

        key = f"corpora/{corpus.corpus_id}/runs/{run_id}/{name}"
        if os.path.exists(local_path):
            with open(local_path, "rb") as f:
                db.object_store.save_bytes(key, f.read())

        # If this is the deterministic export ZIP, remember its key so
        # the DB can later reconstruct an import root via load_imported_run.
        if kind == "hilbert_export_zip":
            export_key = key

        db.register_artifact(
            run_id=run_id,
            name=name,
            kind=kind,
            key=key,
            meta=meta,
        )

    # Record export_key on the run so hilbert_db.apis.* can call
    # db.load_imported_run(run_id) and find the ZIP in the object store.
    if export_key:
        # Try a few plausible DB APIs, but never crash if they are absent.
        if hasattr(db, "set_run_export_key"):
            db.set_run_export_key(run_id, export_key)  # type: ignore[attr-defined]
        elif hasattr(db, "update_run_export_key"):
            db.update_run_export_key(run_id, export_key)  # type: ignore[attr-defined]
        elif hasattr(db, "mark_run_export"):
            db.mark_run_export(run_id, export_key)  # type: ignore[attr-defined]
        elif hasattr(db, "update_run"):
            try:
                db.update_run(run_id=run_id, export_key=export_key)  # type: ignore[arg-type]
            except Exception:
                # Best-effort only
                pass

    # Mark run status in DB (best-effort)
    if hasattr(db, "mark_run_ok"):
        db.mark_run_ok(run_id)

    # Emit run completion event
    emit(
        "run_end",
        {
            "run_id": run_id,
            "corpus_id": corpus.corpus_id,
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "stages": {
                k: {
                    "status": st.status,
                    "error": st.error,
                    "duration": st.duration,
                }
                for k, st in ctx.stages.items()
            },
            "ts": time.time(),
        },
    )

    return {
        "run_id": run_id,
        "corpus_id": corpus.corpus_id,
        "results_dir": results_dir,
    }


__all__ = [
    "run_hilbert_orchestration",
    "PipelineSettings",
    "StageSpec",
    "StageState",
    "PipelineContext",
]
