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
    """Fallback no-op event emitter."""
    return None


# ======================================================================
# LSA layer
# ======================================================================

from .lsa_layer import build_lsa_field


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
    """Backwards compatible alias."""
    return run_molecule_stage(*args, **kwargs)

def run_molecule_layer(*args, **kwargs):
    return run_molecule_stage(*args, **kwargs)


# ======================================================================
# Fusion layer (spans -> elements + compound context)
# ======================================================================

from .fusion import fuse_spans_to_elements, aggregate_compound_context

def run_fusion_pipeline(results_dir: str, emit: Callable = DEFAULT_EMIT) -> None:
    fuse_spans_to_elements(results_dir, emit=emit)
    aggregate_compound_context(results_dir, emit=emit)


# ======================================================================
# Element labels
# ======================================================================

from .element_labels import build_element_descriptions


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
# Export (deterministic manifest + ZIP)
# ======================================================================

from .hilbert_export import (
    build_manifest,
    export_zip,
    run_full_export,
)


# ======================================================================
# Epistemic Signatures
# ======================================================================

from .signatures import compute_signatures


# ======================================================================
# Element LM (optional)
# ======================================================================

try:
    from .element_language_model import (
        run_element_lm_stage,
        suggest_next_elements,
        score_element_sequence,
    )
except Exception as exc:
    print(f"[hilbert_pipeline] Element LM unavailable: {exc}")

    def run_element_lm_stage(*args, **kwargs):
        print("[element_lm] Skipped: module unavailable")

    def suggest_next_elements(*args, **kwargs):
        return []

    def score_element_sequence(*args, **kwargs):
        return 0.0


# ======================================================================
# Edges builder
# ======================================================================

from .edges_builder import build_element_edges


# ======================================================================
# Corpus probing (diagnostics)
# ======================================================================

from .corpus_probe import probe_corpus, run_lsa_seed_profile


# ======================================================================
# Element roots
# ======================================================================

from .element_roots import run_element_roots


# ======================================================================
# Perplexity (Ollama LM)
# ======================================================================

from .ollama_lm import compute_corpus_perplexity


# ======================================================================
# Public API surface
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

    # Stability + visuals
    "compute_signal_stability",
    "compute_compound_stability",
    "plot_persistence_field",
    "run_persistence_visuals",

    # Export
    "build_manifest",
    "export_zip",
    "run_full_export",

    # Signatures
    "compute_signatures",

    # Edges
    "build_element_edges",

    # Element LM
    "run_element_lm_stage",
    "suggest_next_elements",
    "score_element_sequence",

    # Corpus diagnostics
    "probe_corpus",
    "run_lsa_seed_profile",

    # Roots
    "run_element_roots",

    # LM perplexity
    "compute_corpus_perplexity",
]
