# hilbert_orchestrator/hilbert_run.py
"""
High-level public entry point for running a Hilbert pipeline.

External callers (FastAPI handlers, CLI tools, notebooks, tests) should use:

    from hilbert_orchestrator.hilbert_run import hilbert_run

This wrapper:
    - Normalises arguments
    - Constructs a DBInterface for defensive DB access
    - Invokes the orchestrator engine
    - Returns a minimal run summary

Detailed run metadata is written into ``hilbert_run.json`` inside ``results_dir``
and persisted via the HilbertDB object store + artifact registry.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

from hilbert_db.core import HilbertDB

from hilbert_orchestrator.db.db_interface import DBInterface
from hilbert_orchestrator.core.engine import run_orchestrator
from hilbert_orchestrator.core.stages import PipelineSettings


__all__ = ["hilbert_run"]


def hilbert_run(
    *,
    db: HilbertDB,
    corpus_dir: str,
    corpus_name: Optional[str] = None,
    results_dir: Optional[str] = None,
    max_docs: Optional[int] = None,
    use_native: bool = True,
    emit: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    settings: Optional[PipelineSettings] = None,
) -> Dict[str, Any]:
    """
    Execute the full Hilbert Information Pipeline.

    Parameters
    ----------
    db : HilbertDB
        Database instance for run + corpus registration and artifact storage.

    corpus_dir : str
        Filesystem directory containing raw corpus documents (pdf, txt, jsonl...).

    corpus_name : str, optional
        Human-readable name; defaults to basename(corpus_dir).

    results_dir : str, optional
        Output directory for pipeline artifacts; defaults to <corpus_dir>/_hilbert_results.

    max_docs : int, optional
        Limit number of documents included in the LSA preprocessing step.

    use_native : bool
        Passed to PipelineSettings.use_native.

    emit : callable, optional
        Event emitter with signature: emit(kind: str, payload: dict)

    settings : PipelineSettings, optional
        Full override for pipeline settings.

    Returns
    -------
    dict
        {
            "run_id": <str>,
            "corpus_id": <int>,
            "results_dir": <str>
        }
    """

    # ------------------------------------------------------------------
    # Validate corpus path
    # ------------------------------------------------------------------
    corpus_dir = os.path.abspath(corpus_dir)
    if not os.path.isdir(corpus_dir):
        raise ValueError(f"Corpus directory does not exist: {corpus_dir}")

    # Derive corpus_name
    corpus_name = corpus_name or os.path.basename(corpus_dir.rstrip(os.sep))

    # Determine destination results directory
    if results_dir is None:
        results_dir = os.path.join(corpus_dir, "_hilbert_results")
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build settings
    # ------------------------------------------------------------------
    if settings is None:
        settings = PipelineSettings(
            use_native=use_native,
            max_docs=max_docs,
        )

    # ------------------------------------------------------------------
    # Wrap DB for defensive compatibility
    # ------------------------------------------------------------------
    dbi = DBInterface(db)

    # ------------------------------------------------------------------
    # Execute engine
    # ------------------------------------------------------------------
    result = run_orchestrator(
        db=dbi.db,
        corpus_dir=corpus_dir,
        corpus_name=corpus_name,
        results_dir=results_dir,
        settings=settings,
        emit=emit,
    )

    return result
