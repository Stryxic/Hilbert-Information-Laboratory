"""
Database Interface for the Hilbert Orchestrator
===============================================

This module provides a defensive, forward-compatible abstraction layer on
top of `HilbertDB`.  It ensures the orchestrator can:

    - register corpora and runs
    - manage run lifecycle state
    - register artifacts
    - record export keys
    - use the DB's object store

without binding itself to any one particular DB schema.

Design Principles
-----------------
• Best-effort: missing DB methods never crash the orchestrator.  
• Normalised handles: all corpus and run objects are wrapped in thin
  dataclasses that expose stable fields (`corpus_id`, `run_id`).  
• Forward-compatibility: the orchestrator relies only on a *soft* API
  contract; DB implementations may evolve independently.  

Typical Usage
-------------

    from hilbert_db.core import HilbertDB
    from hilbert_orchestrator.db_interface import DBInterface

    dbi = DBInterface(HilbertDB(...))
    corpus = dbi.get_or_create_corpus(...)
    run = dbi.create_run(...)

    dbi.mark_run_running(run.run_id)
    dbi.register_artifact(...)
    dbi.set_export_key(run.run_id, "corpora/.../export.zip")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from hilbert_db.core import HilbertDB


# ---------------------------------------------------------------------------
# Handle Types
# ---------------------------------------------------------------------------

@dataclass
class CorpusHandle:
    """
    Lightweight wrapper around a corpus DB row.

    Attributes
    ----------
    corpus_id : int
        Primary key of the corpus.
    fingerprint : str
        Corpus fingerprint (content hash, path hash, etc.).
    name : str
        Human-readable corpus name.
    source_uri : str
        Path / URI to the raw corpus.
    raw : Any
        The original DB object returned by HilbertDB.
    """

    corpus_id: int
    fingerprint: str
    name: str
    source_uri: str
    raw: Any


@dataclass
class RunHandle:
    """
    Lightweight wrapper around a run DB row.

    Attributes
    ----------
    run_id : str
        Primary key / run identifier.
    corpus_id : int
        Foreign key to corpus.
    orchestrator_version : str
        The version of the orchestrator that created this run.
    settings : dict
        Run settings passed to the orchestrator.
    raw : Any
        The original DB object returned by HilbertDB.
    """

    run_id: str
    corpus_id: int
    orchestrator_version: str
    settings: Dict[str, Any]
    raw: Any


# ---------------------------------------------------------------------------
# DB Interface Wrapper
# ---------------------------------------------------------------------------

class DBInterface:
    """
    A minimal, defensive wrapper around `HilbertDB`.

    This class standardises the DB contract used by the orchestrator.
    Missing or failing methods never crash the orchestrator: they simply
    no-op or fall back to softer update calls.

    Parameters
    ----------
    db : HilbertDB
        A concrete instance from `hilbert_db.core`.
    """

    def __init__(self, db: HilbertDB):
        self._db = db

    # ------------------------------------------------------------------
    # Low-level access
    # ------------------------------------------------------------------

    @property
    def db(self) -> HilbertDB:
        """Return the underlying DB instance."""
        return self._db

    @property
    def object_store(self) -> Any:
        """
        Return the DB's object store.

        Expected interface:
            save_bytes(key: str, data: bytes) -> None
        """
        return self._db.object_store  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Corpus Registration
    # ------------------------------------------------------------------

    def get_or_create_corpus(
        self,
        *,
        fingerprint: str,
        name: str,
        source_uri: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> CorpusHandle:
        """
        Retrieve or create a corpus record.

        Parameters
        ----------
        fingerprint : str
            Corpus fingerprint.
        name : str
            Human-readable name.
        source_uri : str
            Source path / URI.
        extra : dict, optional
            Extra keyword arguments passed to the DB.

        Returns
        -------
        CorpusHandle
        """
        extra = extra or {}

        corpus_obj = self._db.get_or_create_corpus(
            fingerprint=fingerprint,
            name=name,
            source_uri=source_uri,
            **extra,
        )

        # Defensive extraction of corpus_id
        corpus_id = getattr(corpus_obj, "corpus_id", getattr(corpus_obj, "id", None))
        if corpus_id is None:
            raise RuntimeError(
                "HilbertDB.get_or_create_corpus() returned an object with no corpus_id."
            )

        return CorpusHandle(
            corpus_id=int(corpus_id),
            fingerprint=getattr(corpus_obj, "fingerprint", fingerprint),
            name=getattr(corpus_obj, "name", name),
            source_uri=getattr(corpus_obj, "source_uri", source_uri),
            raw=corpus_obj,
        )

    # ------------------------------------------------------------------
    # Run Registration
    # ------------------------------------------------------------------

    def create_run(
        self,
        *,
        run_id: str,
        corpus_id: int,
        orchestrator_version: str,
        settings: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None
    ) -> RunHandle:
        """
        Create a run record.

        Returns
        -------
        RunHandle
        """
        extra = extra or {}

        run_obj = self._db.create_run(
            run_id=run_id,
            corpus_id=corpus_id,
            orchestrator_version=orchestrator_version,
            settings_json=settings,
            **extra,
        )

        # Defensive extraction of run_id
        rid = getattr(run_obj, "run_id", getattr(run_obj, "id", run_id))

        return RunHandle(
            run_id=str(rid),
            corpus_id=corpus_id,
            orchestrator_version=orchestrator_version,
            settings=settings,
            raw=run_obj,
        )

    # ------------------------------------------------------------------
    # Run Lifecycle State
    # ------------------------------------------------------------------

    def mark_run_running(self, run_id: str) -> None:
        """Best-effort mark: run is now running."""
        if hasattr(self._db, "mark_run_running"):
            try:
                self._db.mark_run_running(run_id)  # type: ignore[attr-defined]
            except Exception:
                pass

    def mark_run_ok(self, run_id: str) -> None:
        """Best-effort mark: run completed successfully."""
        if hasattr(self._db, "mark_run_ok"):
            try:
                self._db.mark_run_ok(run_id)  # type: ignore[attr-defined]
            except Exception:
                pass

    def mark_run_failed(self, run_id: str, error: str) -> None:
        """Best-effort mark: run failed."""
        if hasattr(self._db, "mark_run_failed"):
            try:
                self._db.mark_run_failed(run_id, error)  # type: ignore[attr-defined]
                return
            except Exception:
                pass

        # Fallback generic update
        if hasattr(self._db, "update_run"):
            try:
                self._db.update_run(run_id=run_id, status="failed", error=error)  # type: ignore[arg-type]
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Export Key Management
    # ------------------------------------------------------------------

    def set_export_key(self, run_id: str, export_key: str) -> None:
        """
        Persist the export ZIP key to the run record.

        Try several DB APIs, fail silently if none exist.
        """
        for method in (
            "set_run_export_key",
            "update_run_export_key",
            "mark_run_export",
        ):
            if hasattr(self._db, method):
                try:
                    getattr(self._db, method)(run_id, export_key)  # type: ignore
                    return
                except Exception:
                    pass

        # Fallback generic update
        if hasattr(self._db, "update_run"):
            try:
                self._db.update_run(run_id=run_id, export_key=export_key)  # type: ignore[arg-type]
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Artifact Registration
    # ------------------------------------------------------------------

    def register_artifact(
        self,
        *,
        run_id: str,
        name: str,
        kind: str,
        key: str,
        meta: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register an artifact record for a run.

        Parameters
        ----------
        run_id : str
        name : str
        kind : str
        key : str
            Object-store path.
        meta : dict, optional
            Additional metadata.
        """
        meta = meta or {}
        try:
            self._db.register_artifact(
                run_id=run_id,
                name=name,
                kind=kind,
                key=key,
                meta=meta,
            )
        except Exception:
            # Never break the orchestrator
            pass


__all__ = [
    "DBInterface",
    "CorpusHandle",
    "RunHandle",
]
