"""
Structured Event Containers
===========================

Defines strongly typed, structured event payloads emitted during pipeline
execution. These classes produce JSON-compatible payloads consumed by:

    - UI event streams
    - Logging subsystems
    - Orchestrator engine hooks
    - DB persistence layers
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Any

from .context import PipelineContext
from .stages import StageSpec, StageState


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _ts() -> float:
    """Return a UNIX timestamp for event emission."""
    return time.time()


# ---------------------------------------------------------------------------
# Run-level events
# ---------------------------------------------------------------------------

@dataclass
class RunStartEvent:
    """
    Emitted at the beginning of a Hilbert pipeline run.
    """

    ctx: PipelineContext

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": "run_start",
            "run_id": self.ctx.run_id,
            "corpus_id": self.ctx.corpus_id,
            "corpus_dir": self.ctx.corpus_dir,
            "results_dir": self.ctx.results_dir,
            "orchestrator_version": self.ctx.version,
            "settings": self.ctx.settings_dict,
            "ts": _ts(),
        }


@dataclass
class RunEndEvent:
    """
    Emitted once when a pipeline run completes.
    """

    ctx: PipelineContext

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": "run_end",
            "run_id": self.ctx.run_id,
            "corpus_id": self.ctx.corpus_id,
            "results_dir": self.ctx.results_dir,
            "orchestrator_version": self.ctx.version,
            "stages": {
                key: {
                    "label": st.label,
                    "status": st.status,
                    "error": st.error,
                    "duration": st.duration,
                }
                for key, st in self.ctx.stages.items()
            },
            "ts": _ts(),
        }


# ---------------------------------------------------------------------------
# Stage-level events
# ---------------------------------------------------------------------------

@dataclass
class StageStartEvent:
    """
    Emitted when an individual pipeline stage begins execution.
    """

    ctx: PipelineContext
    spec: StageSpec

    def to_dict(self, index: int, total: int) -> Dict[str, Any]:
        return {
            "event": "stage_start",
            "run_id": self.ctx.run_id,
            "corpus_id": self.ctx.corpus_id,
            "stage": self.spec.key,
            "label": self.spec.label,
            "index": index,
            "total_stages": total,
            "ts": _ts(),
        }


@dataclass
class StageEndEvent:
    """
    Emitted whenever a stage ends (success, failure, or skip).
    """

    ctx: PipelineContext
    spec: StageSpec
    state: StageState

    def to_dict(self, index: int, total: int) -> Dict[str, Any]:
        return {
            "event": "stage_end",
            "run_id": self.ctx.run_id,
            "corpus_id": self.ctx.corpus_id,
            "stage": self.spec.key,
            "label": self.spec.label,
            "status": self.state.status,
            "error": self.state.error,
            "duration": self.state.duration,
            "index": index,
            "total_stages": total,
            "ts": _ts(),
        }


# ---------------------------------------------------------------------------
# Artifact and Log events
# ---------------------------------------------------------------------------

@dataclass
class ArtifactEvent:
    """
    Emitted whenever a pipeline stage produces an artifact that the engine
    must register and upload.
    """

    ctx: PipelineContext
    name: str
    kind: str
    path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": "artifact",
            "run_id": self.ctx.run_id,
            "corpus_id": self.ctx.corpus_id,
            "name": self.name,
            "kind": self.kind,
            "path": self.path,
            "ts": _ts(),
        }


@dataclass
class LogEvent:
    """
    Structured logging event for UI and debugging.
    """

    ctx: PipelineContext
    level: str
    message: str
    fields: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "event": "log",
            "run_id": self.ctx.run_id,
            "corpus_id": self.ctx.corpus_id,
            "level": self.level,
            "msg": self.message,
            "ts": _ts(),
        }
        payload.update(self.fields or {})
        return payload


__all__ = [
    "RunStartEvent",
    "RunEndEvent",
    "StageStartEvent",
    "StageEndEvent",
    "ArtifactEvent",
    "LogEvent",
]
