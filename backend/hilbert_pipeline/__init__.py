"""
Hilbert Information Pipeline - public API aggregator.

This module re-exports the main computational primitives of the pipeline so that
the orchestrator and higher-level tools can import everything from the
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
    """Fallback 'do nothing' emitter."""
    return None


# ======================================================================
# LSA layer
# ======================================================================

from .lsa_layer import (
    build_lsa_field,
)


# ======================================================================
# Molecule layer
# ======================================================================

from .molecule_layer import (
    run_molecule_stage,
    aggregate_compounds,
    compute_molecule_stability,
    compute_molecule_temperature,
    export_molecule_summary,
)

def build_molecules(*args, **kwargs):
    """Backwards-compatible alias for older orchestrator code."""
    return run_molecule_stage(*args, **kwargs)

def run_molecule_layer(*args, **kwargs):
    """Alias for run_molecule_stage."""
    return run_molecule_stage(*args, **kwargs)


# ======================================================================
# Span–element fusion + compound context
# ======================================================================

from .fusion import (
    fuse_spans_to_elements,
    aggregate_compound_context,
)

def run_fusion_pipeline(results_dir: str, emit: Callable = DEFAULT_EMIT) -> None:
    """Thin wrapper combining fusion + compound aggregation."""
    fuse_spans_to_elements(results_dir, emit=emit)
    aggregate_compound_context(results_dir, emit=emit)


# ======================================================================
# Element labels
# ======================================================================

from .element_labels import (
    build_element_descriptions,
)


# ======================================================================
# Stability + persistence visuals
# ======================================================================

from .stability_layer import compute_signal_stability, compute_compound_stability
from .persistence_visuals import plot_persistence_field

def run_persistence_visuals(results_dir: str, emit: Callable = DEFAULT_EMIT) -> None:
    emit("log", {"stage": "persistence_visuals", "event": "start"})
    plot_persistence_field(results_dir)
    emit("log", {"stage": "persistence_visuals", "event": "end"})


# ======================================================================
# Graphs
# ======================================================================

from .graph_snapshots import generate_graph_snapshots
from .graph_export import export_graph_snapshots


# ======================================================================
# Export (PDF + ZIP)
# ======================================================================

from .hilbert_export import (
    export_summary_pdf,
    export_zip,
)

def run_full_export(results_dir: str, emit: Callable = DEFAULT_EMIT) -> None:
    emit("log", {"stage": "export", "event": "start"})
    export_summary_pdf(results_dir)
    export_zip(results_dir)
    emit("log", {"stage": "export", "event": "end"})


# ======================================================================
# Epistemic Signatures
# ======================================================================

from .signatures import compute_signatures


# ======================================================================
# Optional: Element Language Model (safe import)
# ======================================================================

try:
    from .element_language_model import (
        run_element_lm_stage,
        suggest_next_elements,
        score_element_sequence,
    )
except Exception as exc:
    print(f"[hilbert_pipeline] Element LM unavailable: {exc}")

    # Safe fallback stubs so orchestrator import NEVER fails
    def run_element_lm_stage(*args, **kwargs):
        print("[element_lm] Skipped: module unavailable")

    def suggest_next_elements(*args, **kwargs):
        return []

    def score_element_sequence(*args, **kwargs):
        return 0.0


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

    # Stability + persistence
    "compute_signal_stability",
    "plot_persistence_field",
    "run_persistence_visuals",

    # Graphs
    "generate_graph_snapshots",
    "export_graph_snapshots",

    # Export
    "export_summary_pdf",
    "export_zip",
    "run_full_export",

    # Epistemic signatures
    "compute_signatures",

    # Element LM (optional but guaranteed to exist)
    "run_element_lm_stage",
    "suggest_next_elements",
    "score_element_sequence",
]

from .edges_builder import build_element_edges
__all__ += ["build_element_edges"]

# backend/hilbert_pipeline/__init__.py
from .ollama_lm import compute_corpus_perplexity  # new export
__all__ += ["compute_corpus_perplexity"]