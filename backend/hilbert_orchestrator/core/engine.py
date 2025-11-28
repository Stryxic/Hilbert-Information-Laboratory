# hilbert_orchestrator/core/engine.py

"""
Hilbert Orchestrator Engine (v4.x)
==================================

Execution engine for the modular Hilbert Information Pipeline.

Responsibilities
----------------
- Initialise a PipelineContext
- Import and validate registered stages
- Execute stages in dependency order
- Emit structured run / stage events
- Record artifacts into HilbertDB
- Write hilbert_run.json
- Record export_key for downstream import tools

Stage implementations live in the ``hilbert_orchestrator.stages`` package.
They register themselves with ``GLOBAL_STAGE_REGISTRY`` via decorators.
"""

from __future__ import annotations

import json
import os
import time
import random
from typing import Any, Dict, Optional, Callable

import numpy as np

from hilbert_db.core import HilbertDB

from .events import (
    RunStartEvent,
    RunEndEvent,
    StageStartEvent,
    StageEndEvent,
)
from .context import PipelineContext
from .stages import PipelineSettings, StageState, StageSpec
from .registry import GLOBAL_STAGE_REGISTRY


__all__ = ["run_hilbert_orchestration", "run_orchestrator"]

ORCHESTRATOR_VERSION = "4.1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _finalize_run_json(ctx: PipelineContext) -> None:
    """Write hilbert_run.json summarizing all stages + artifacts."""
    summary = {
        "run_id": ctx.run_id,
        "corpus_id": ctx.corpus_id,
        "orchestrator_version": ctx.version,
        "settings": ctx.settings.as_dict(),
        "stages": {
            key: {
                "label": st.label,
                "status": st.status,
                "error": st.error,
                "duration": st.duration,
                "meta": dict(st.meta),
            }
            for key, st in ctx.stages.items()
        },
        "artifacts": ctx.artifacts,
    }

    out_path = os.path.join(ctx.results_dir, "hilbert_run.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    ctx.add_artifact("hilbert_run.json", kind="run-summary")


def _safe_emit(emit: Callable[[str, Dict[str, Any]], None],
               kind: str,
               payload: Dict[str, Any]) -> None:
    """Guarded emit so a bad emitter cannot kill the run."""
    try:
        emit(kind, payload)
    except Exception:
        # Last-resort: do not propagate emitter failures into the engine
        pass


# ---------------------------------------------------------------------------
# Core orchestration logic
# ---------------------------------------------------------------------------

def run_hilbert_orchestration(
    *,
    db: HilbertDB,
    corpus_dir: str,
    corpus_name: str,
    results_dir: str,
    settings: Optional[PipelineSettings] = None,
    emit: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Execute the Hilbert pipeline using the global stage registry.

    Parameters
    ----------
    db : HilbertDB
        Database handle for corpus / run registration and object storage.

    corpus_dir : str
        Path to the raw input corpus.

    corpus_name : str
        Human-readable corpus label.

    results_dir : str
        Directory where all pipeline outputs will be written.

    settings : PipelineSettings, optional
        Execution settings such as max_docs, random_seed, and native/py preference.

    emit : callable, optional
        Structured event emitter with signature:
            emit(kind: str, payload: dict)
        If None, a no-op emitter is used.

    Returns
    -------
    dict
        {
            "run_id": ...,
            "corpus_id": ...,
            "results_dir": ...
        }
    """

    settings = settings or PipelineSettings()
    emit = emit or (lambda *_: None)

    # Seed all RNGs
    random.seed(settings.random_seed)
    np.random.seed(settings.random_seed)

    _ensure_dir(results_dir)

    # ------------------------------------------------------------------
    # Register in HilbertDB
    # ------------------------------------------------------------------

    fingerprint = str(int(time.time()))
    corpus = db.get_or_create_corpus(
        fingerprint=fingerprint,
        name=corpus_name,
        source_uri=corpus_dir,
    )

    run_id = str(int(time.time() * 1000))
    db.create_run(
        run_id=run_id,
        corpus_id=corpus.corpus_id,
        orchestrator_version=ORCHESTRATOR_VERSION,
        settings_json=settings.as_dict(),
    )

    if hasattr(db, "mark_run_running"):
        try:
            db.mark_run_running(run_id)
        except Exception:
            # Best-effort only
            pass

    # ------------------------------------------------------------------
    # Initialise context
    # ------------------------------------------------------------------

    ctx = PipelineContext(
        run_id=run_id,
        corpus_id=corpus.corpus_id,
        corpus_dir=os.path.abspath(corpus_dir),
        results_dir=os.path.abspath(results_dir),
        settings=settings,
        emit=emit,
        db=db,
    )
    ctx.version = ORCHESTRATOR_VERSION

    # ------------------------------------------------------------------
    # Import stages package (trigger registration side-effects)
    # ------------------------------------------------------------------

    try:
        # Importing the package causes all stage modules to be imported,
        # which in turn registers their StageSpec objects with the registry.
        from hilbert_orchestrator import stages as _stages  # noqa: F401
    except Exception as exc:
        ctx.log("error", "Failed to import hilbert_orchestrator.stages", error=str(exc))
        _safe_emit(
            emit,
            "log",
            {
                "event": "log",
                "level": "error",
                "msg": "Failed to import hilbert_orchestrator.stages",
                "error": str(exc),
            },
        )

    # ------------------------------------------------------------------
    # Emit run start
    # ------------------------------------------------------------------

    _safe_emit(emit, "run_start", RunStartEvent(ctx).to_dict())

    # ------------------------------------------------------------------
    # Collect and order stages
    # ------------------------------------------------------------------

    stage_specs = GLOBAL_STAGE_REGISTRY.get_ordered()
    total = len(stage_specs)

    if total == 0:
        ctx.log("error", "No stages registered in GLOBAL_STAGE_REGISTRY")
        _safe_emit(
            emit,
            "log",
            {
                "event": "log",
                "level": "error",
                "msg": "No stages registered in GLOBAL_STAGE_REGISTRY",
            },
        )
        # We still finalise the run summary and mark the run as ok,
        # so the UI has something to show even if the configuration is broken.
        _finalize_run_json(ctx)
        if hasattr(db, "mark_run_ok"):
            try:
                db.mark_run_ok(ctx.run_id)
            except Exception:
                pass
        _safe_emit(emit, "run_end", RunEndEvent(ctx).to_dict())
        return {
            "run_id": ctx.run_id,
            "corpus_id": ctx.corpus_id,
            "results_dir": ctx.results_dir,
        }

    # ------------------------------------------------------------------
    # Execute each stage in sequence
    # ------------------------------------------------------------------

    for index, spec in enumerate(stage_specs):
        unmet = spec.unmet_dependencies(ctx)

        if unmet:
            if spec.required:
                # Hard failure: record and stop
                state = ctx.begin_stage(spec)
                ctx.end_stage_failed(spec, f"Missing dependencies: {unmet}")
                _safe_emit(
                    emit,
                    "stage_end",
                    StageEndEvent(ctx, spec, state).to_dict(index, total),
                )
                break
            else:
                # Soft skip
                state = ctx.end_stage_skipped(spec, f"Missing dependencies: {unmet}")
                _safe_emit(
                    emit,
                    "stage_end",
                    StageEndEvent(ctx, spec, state).to_dict(index, total),
                )
                continue

        # Normal execution
        state = ctx.begin_stage(spec)
        _safe_emit(
            emit,
            "stage_start",
            StageStartEvent(ctx, spec).to_dict(index, total),
        )

        try:
            spec.run(ctx)
            ctx.end_stage_ok(spec)
        except Exception as exc:
            ctx.end_stage_failed(spec, str(exc))
            _safe_emit(
                emit,
                "stage_end",
                StageEndEvent(ctx, spec, ctx.stages[spec.key]).to_dict(index, total),
            )
            if spec.required:
                break
        else:
            _safe_emit(
                emit,
                "stage_end",
                StageEndEvent(ctx, spec, ctx.stages[spec.key]).to_dict(index, total),
            )

    # ------------------------------------------------------------------
    # Write run summary and register artifacts
    # ------------------------------------------------------------------

    _finalize_run_json(ctx)

    export_key: Optional[str] = None

    for name, info in ctx.artifacts.items():
        local_path = info["path"]
        kind = info["kind"]
        meta = {k: v for k, v in info.items() if k not in ("path", "kind")}

        key = f"corpora/{ctx.corpus_id}/runs/{ctx.run_id}/{name}"

        if os.path.exists(local_path):
            try:
                with open(local_path, "rb") as f:
                    data = f.read()
                db.object_store.save_bytes(key, data)
            except Exception:
                # Artifact upload is best-effort
                pass

        if kind == "hilbert_export_zip":
            export_key = key

        try:
            db.register_artifact(
                run_id=ctx.run_id,
                name=name,
                kind=kind,
                key=key,
                meta=meta,
            )
        except Exception:
            # Artifact registration must not break the run summary
            pass

    # Record export key if available (best-effort)
    if export_key:
        try:
            if hasattr(db, "set_run_export_key"):
                db.set_run_export_key(ctx.run_id, export_key)
            elif hasattr(db, "update_run_export_key"):
                db.update_run_export_key(ctx.run_id, export_key)
            elif hasattr(db, "mark_run_export"):
                db.mark_run_export(ctx.run_id, export_key)
            elif hasattr(db, "update_run"):
                db.update_run(run_id=ctx.run_id, export_key=export_key)
        except Exception:
            pass

    # Mark the run as complete
    if hasattr(db, "mark_run_ok"):
        try:
            db.mark_run_ok(ctx.run_id)
        except Exception:
            pass

    # Emit run end
    _safe_emit(emit, "run_end", RunEndEvent(ctx).to_dict())

    return {
        "run_id": ctx.run_id,
        "corpus_id": ctx.corpus_id,
        "results_dir": ctx.results_dir,
    }


# Backwards-compatible alias
def run_orchestrator(**kwargs: Any) -> Dict[str, Any]:
    """Alias retained for older callers."""
    return run_hilbert_orchestration(**kwargs)
