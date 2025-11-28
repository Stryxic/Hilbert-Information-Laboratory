"""
PipelineContext
===============

Shared state object passed to every stage in the orchestrator pipeline.

Responsibilities:
    - Hold run metadata (corpus, run ID, settings, version)
    - Provide structured logging via the emit() callback
    - Track stage state (start, end, errors, metadata)
    - Register produced artifacts
    - Store additional per-run data (ctx.extras)
    - Provide safe access to the database interface and object store

This module contains *no stage logic*.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .stages import StageState, StageSpec, PipelineSettings


# Event emitter signature
EmitFn = Callable[[str, Dict[str, Any]], None]


@dataclass
class PipelineContext:
    """
    Encapsulates all state shared during a Hilbert pipeline run.

    Parameters
    ----------
    run_id : str
        Unique run identifier assigned by the orchestrator.

    corpus_id : int
        Database identifier of the corpus.

    corpus_dir : str
        Filesystem path to the raw corpus.

    results_dir : str
        Output directory for all pipeline artifacts.

    settings : PipelineSettings
        Execution settings specific to this run.

    emit : callable
        Structured event emitter:
            emit(kind: str, payload: dict)

    db : Any
        Database interface (typically DBInterface or HilbertDB).

    Notes
    -----
    - Stages, artifacts, and extras are populated dynamically.
    - The context is intentionally simple and self-contained so that
      orchestration logic remains readable and strongly typed.
    """

    run_id: str
    corpus_id: int
    corpus_dir: str
    results_dir: str
    settings: PipelineSettings
    emit: EmitFn
    db: Any  # forward compatible with HilbertDB or DBInterface

    # Runtime bookkeeping
    stages: Dict[str, StageState] = field(default_factory=dict)
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    # Orchestrator version (injected by high-level engine)
    version: str = "4.1"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, level: str, msg: str, **fields: Any) -> None:
        """
        Emit a structured log event.

        Parameters
        ----------
        level : str
            One of: "info", "warn", "error".

        msg : str
            Human-readable log message.

        fields : dict
            Additional context fields merged into event payload.
        """
        payload = {"level": level, "msg": msg, "ts": time.time()}
        payload.update(fields)
        try:
            self.emit("log", payload)
        except Exception:
            # Last-resort fallback to stdout
            print(f"[{level}] {msg} {fields}")

    # ------------------------------------------------------------------
    # Stage lifecycle
    # ------------------------------------------------------------------

    def begin_stage(self, spec: StageSpec) -> StageState:
        """
        Mark a stage as started and emit a log event.
        """
        st = self.stages.get(spec.key) or StageState(key=spec.key, label=spec.label)
        st.status = "running"
        st.start_ts = time.time()
        self.stages[spec.key] = st

        self.log("info", f"Stage {spec.key} starting", stage=spec.key)
        return st

    def end_stage_ok(self, spec: StageSpec, meta: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark a stage as successfully completed.
        """
        st = self.stages[spec.key]
        st.status = "ok"
        st.end_ts = time.time()
        if meta:
            st.meta.update(meta)

        self.log("info", f"Stage {spec.key} completed", stage=spec.key, duration=st.duration)

    def end_stage_failed(self, spec: StageSpec, error: str) -> None:
        """
        Mark a stage as failed.
        """
        st = self.stages[spec.key]
        st.status = "failed"
        st.error = error
        st.end_ts = time.time()

        self.log("error", f"Stage {spec.key} failed: {error}", stage=spec.key, duration=st.duration)

    def end_stage_skipped(self, spec: StageSpec, reason: str) -> StageState:
        """
        Mark a stage as skipped and record the reason.
        """
        st = self.stages.get(spec.key) or StageState(key=spec.key, label=spec.label)
        st.status = "skipped"
        st.error = reason
        st.start_ts = st.start_ts or time.time()
        st.end_ts = time.time()
        self.stages[spec.key] = st

        self.log("warn", f"Stage {spec.key} skipped ({reason})", stage=spec.key, duration=st.duration)
        return st

    # ------------------------------------------------------------------
    # Artifact registration
    # ------------------------------------------------------------------

    def add_artifact(self, name: str, kind: str, **meta: Any) -> None:
        """
        Register an artifact produced by a stage.

        Parameters
        ----------
        name : str
            Filename relative to results_dir.

        kind : str
            Artifact classification (e.g. "lsa-field", "stability").

        meta : dict
            Additional metadata.
        """
        path = os.path.join(self.results_dir, name)
        self.artifacts[name] = {
            "kind": kind,
            "path": path,
            **meta,
        }

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def object_store(self) -> Any:
        """Convenience passthrough for ctx.db.object_store."""
        return getattr(self.db, "object_store", None)

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Return settings as a dict (used by event models)."""
        return self.settings.as_dict()
