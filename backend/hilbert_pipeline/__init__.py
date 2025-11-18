"""
Hilbert Information Pipeline - public API aggregator.

This module re-exports the main computational primitives of the pipeline so that
the orchestrator and any higher level tools can import everything from the
`hilbert_pipeline` package root.

It also defines DEFAULT_EMIT, a no-op logger used as a safe fallback by
optional layers such as the epistemic signatures module.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

# ======================================================================
# Default emitter
# ======================================================================

def DEFAULT_EMIT(kind: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """
    Fallback "do nothing" emitter.

    Real callers (the orchestrator, web API, CLI) usually provide their own
    `emit(kind, payload)` function. Code inside pipeline modules should always
    accept an `emit` argument but never rely on it doing anything.
    """
    return None


# ======================================================================
# LSA layer
# ======================================================================

from .lsa_layer import (  # noqa: E402,F401
    build_lsa_field,
)


# ======================================================================
# Molecule layer
# ======================================================================

from .molecule_layer import (  # noqa: E402,F401
    run_molecule_stage,
    aggregate_compounds,
    compute_molecule_stability,
    compute_molecule_temperature,
    export_molecule_summary,
)

def build_molecules(*args, **kwargs):
    """Backwards compatible alias expected by older orchestrator code."""
    return run_molecule_stage(*args, **kwargs)

def run_molecule_layer(*args, **kwargs):
    """Compatibility alias - identical to `run_molecule_stage`."""
    return run_molecule_stage(*args, **kwargs)


# ======================================================================
# Span - element fusion and compound context
# ======================================================================

from .fusion import (  # noqa: E402,F401
    fuse_spans_to_elements,
    aggregate_compound_context,
)

def run_fusion_pipeline(results_dir: str, emit: Callable = DEFAULT_EMIT) -> None:
    """
    Thin wrapper for span -> element fusion + compound context aggregation.
    """
    fuse_spans_to_elements(results_dir, emit=emit)
    aggregate_compound_context(results_dir, emit=emit)


# ======================================================================
# Element labels and descriptions
# ======================================================================

from .element_labels import (  # noqa: E402,F401
    build_element_descriptions,
)


# ======================================================================
# Stability / persistence visuals
# ======================================================================

from .stability_layer import (  # noqa: E402,F401
    compute_signal_stability,
)

from .persistence_visuals import (  # noqa: E402,F401
    plot_persistence_field,
)

def run_persistence_visuals(results_dir: str, emit: Callable = DEFAULT_EMIT) -> None:
    """Compatibility wrapper for persistence visuals."""
    emit("log", {"stage": "persistence_visuals", "event": "start"})
    plot_persistence_field(results_dir)
    emit("log", {"stage": "persistence_visuals", "event": "end"})


# ======================================================================
# Graph export and snapshots
# ======================================================================

from .graph_snapshots import (  # noqa: E402,F401
    generate_graph_snapshots,
)

from .graph_export import (  # noqa: E402,F401
    export_graph_snapshots,
)


# ======================================================================
# Full export (PDF, ZIP)
# ======================================================================

from .hilbert_export import (  # noqa: E402,F401
    export_summary_pdf,
    export_zip,
)

def run_full_export(results_dir: str, emit: Callable = DEFAULT_EMIT) -> None:
    """Produce both PDF and ZIP summaries."""
    emit("log", {"stage": "export", "event": "start"})
    export_summary_pdf(results_dir)
    export_zip(results_dir)
    emit("log", {"stage": "export", "event": "end"})


# ======================================================================
# Misinfo / epistemic signatures layer
# ======================================================================

from .signatures import (  # noqa: E402,F401
    compute_signatures,
)


# ======================================================================
# Public API
# ======================================================================

__all__ = [
    # Emitter
    "DEFAULT_EMIT",

    # LSA
    "build_lsa_field",

    # Molecules
    "run_molecule_stage",
    "run_molecule_layer",
    "build_molecules",
    "aggregate_compounds",
    "compute_molecule_stability",
    "compute_molecule_temperature",
    "export_molecule_summary",

    # Fusion
    "fuse_spans_to_elements",
    "aggregate_compound_context",
    "run_fusion_pipeline",

    # Labels
    "build_element_descriptions",

    # Stability and visuals
    "compute_signal_stability",
    "plot_persistence_field",
    "run_persistence_visuals",

    # Graph views
    "generate_graph_snapshots",
    "export_graph_snapshots",

    # Exports
    "export_summary_pdf",
    "export_zip",
    "run_full_export",

    # Misinfo layer
    "compute_signatures",
]
