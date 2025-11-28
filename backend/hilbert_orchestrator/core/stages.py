"""
hilbert_orchestrator/core/stages.py
===================================

Static stage types used across the orchestrator:

    • PipelineSettings — configuration for a run
    • StageState       — runtime status for a stage
    • StageSpec        — declarative stage definition

IMPORTANT:
    This module MUST NOT import:
        - PipelineContext
        - GLOBAL_STAGE_REGISTRY
        - any stage implementation modules

The purpose of this isolation is to avoid circular-import cascades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Literal


# ============================================================================
# Pipeline Settings
# ============================================================================

@dataclass
class PipelineSettings:
    """
    Execution settings for a pipeline run.
    These settings remain stable across orchestrator versions.
    """

    use_native: bool = True
    max_docs: Optional[int] = None
    random_seed: int = 13

    def as_dict(self) -> Dict[str, Any]:
        """Return settings in JSON-serializable form."""
        return {
            "use_native": self.use_native,
            "max_docs": self.max_docs,
            "random_seed": self.random_seed,
        }


# ============================================================================
# Stage Runtime State
# ============================================================================

@dataclass
class StageState:
    """
    Runtime state for a pipeline stage, tracking:

        - execution status
        - timestamps
        - encountered errors
        - optional metadata
    """

    key: str
    label: str
    status: Literal["pending", "running", "ok", "skipped", "failed"] = "pending"
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Elapsed time for the stage, or None if incomplete."""
        if self.start_ts is None or self.end_ts is None:
            return None
        return self.end_ts - self.start_ts


# ============================================================================
# Stage Specification
# ============================================================================

@dataclass
class StageSpec:
    """
    Declarative definition of a pipeline stage.

    This object is immutable after registration and contains metadata
    used by the orchestrator to order, validate, and execute stages.
    """

    key: str
    order: float
    label: str
    func: Optional[Callable[[Any], None]]

    required: bool = True
    depends_on: List[str] = field(default_factory=list)

    # Semantic (optional) metadata for UI / dialectic structure
    dialectic_role: str = "structure"
    supports: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    consumes: List[str] = field(default_factory=list)

    # ----------------------------------------------------------------------
    # Execution Wrapper
    # ----------------------------------------------------------------------

    def run(self, ctx: Any) -> None:
        """
        Execute the stage function with robust handling for:
            - missing function pointer
            - unexpected exceptions
            - non-callable functions

        Any raised exception propagates upward — the orchestrator decides
        whether to halt or continue.
        """
        if self.func is None:
            raise RuntimeError(
                f"Stage '{self.key}' has no callable function attached."
            )

        if not callable(self.func):
            raise TypeError(
                f"Stage '{self.key}' func is not callable: {self.func!r}"
            )

        # Execute the stage
        return self.func(ctx)

    # ----------------------------------------------------------------------
    # Dependency Checking
    # ----------------------------------------------------------------------

    def unmet_dependencies(self, ctx: Any) -> List[str]:
        """
        Return a list of dependency stage keys that have *not* completed OK.

        Handles:
            - Missing stage keys
            - Typos in dependency lists
            - Stages never executed
        """
        unmet: List[str] = []

        for dep in self.depends_on:
            st = ctx.stages.get(dep)  # StageState or None

            # If missing entirely
            if st is None:
                unmet.append(dep)
                continue

            # If not successful
            if st.status != "ok":
                unmet.append(dep)

        return unmet

    # ----------------------------------------------------------------------
    # Cycle Safety (optional)
    # ----------------------------------------------------------------------

    def check_for_cycles(self, visited: Optional[set] = None) -> None:
        """
        Optional helper: detect self-dependency cycles.

        Not invoked by default (engine ensures ordering),
        but available to debugging scripts.
        """
        visited = visited or set()
        if self.key in visited:
            raise RuntimeError(f"Cyclic dependency detected at stage '{self.key}'")
        visited.add(self.key)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "PipelineSettings",
    "StageState",
    "StageSpec",
]
