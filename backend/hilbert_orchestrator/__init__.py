"""
Hilbert Orchestrator (v4)
====================================

Modular orchestration subsystem for the Hilbert Information Pipeline.

Provides:

  • run_orchestrator / run_hilbert_orchestration – pipeline entry points
  • PipelineSettings – configuration object
  • StageSpec, StageState – stage metadata and runtime status
  • StageRegistry, GLOBAL_STAGE_REGISTRY – stage registration
  • Structured event classes for UI/DB logging

The orchestrator keeps component boundaries strict:

    - core.engine: execution engine
    - core.stages: dataclasses (PipelineSettings, StageSpec, StageState)
    - core.context: internal runtime state (not exported)
    - core.registry: global stage registry
    - concrete stage modules: register themselves via decorators

External users should import:

    from hilbert_orchestrator import run_orchestrator, PipelineSettings
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Public Surface Imports
# ---------------------------------------------------------------------------

# Stage metadata types (these are safe to export)
from .core.stages import (
    PipelineSettings,
    StageSpec,
    StageState,
)

# The global registry for stage definitions
from .core.registry import StageRegistry, GLOBAL_STAGE_REGISTRY

# Main execution engine
from .core.engine import run_orchestrator, run_hilbert_orchestration

# Structured events for UI and DB logging
from .core.events import (
    RunStartEvent,
    RunEndEvent,
    StageStartEvent,
    StageEndEvent,
    ArtifactEvent,
    LogEvent,
)

# Backwards-compatible alias
orchestrate = run_orchestrator

# ---------------------------------------------------------------------------
# Public API symbol list
# ---------------------------------------------------------------------------

__all__ = [
    # Entry points
    "run_orchestrator",
    "run_hilbert_orchestration",
    "orchestrate",

    # Stage metadata types
    "PipelineSettings",
    "StageSpec",
    "StageState",

    # Registry
    "StageRegistry",
    "GLOBAL_STAGE_REGISTRY",

    # Event types
    "RunStartEvent",
    "RunEndEvent",
    "StageStartEvent",
    "StageEndEvent",
    "ArtifactEvent",
    "LogEvent",
]
