from __future__ import annotations

"""
Core façade for the Hilbert DB subsystem.

HilbertDB is the single, high-level entrypoint used by:

    - the orchestrator (to register corpora, runs, and artifacts),
    - the backend API (to query corpora / runs / artifacts, and
      to rehydrate deterministic exports),
    - the frontend (indirectly, via the backend APIs).

It wraps:

    - DB backend + pool
    - Object store
    - Registry layer (corpora, runs, artifacts)
    - Import cache + run importer

Design goals:
    - Deterministic behaviour
    - Minimal, explicit API
    - Easy to test (can inject in-memory registries etc.)
    - Future-proof for Postgres / cloud object stores
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import HilbertDBConfig, load_config
from .db import DBPool, SQLiteBackend, PostgresBackend
from .object_store.base import ObjectStoreConfig
from .object_store.local_fs import LocalFSObjectStore
from .registry.corpus_registry import DBCorpusRegistry
from .registry.run_registry import DBRunRegistry
from .registry.artifact_registry import DBArtifactRegistry
from .registry.models import (
    CorpusRecord,
    RunRecord,
    ArtifactRecord,
    RunStatus,
)
from .importer.cache import CacheManager
from .importer.run_importer import RunImporter, ImportedRun

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HilbertDB façade
# ---------------------------------------------------------------------------

@dataclass
class HilbertDB:
    """
    High-level façade over the Hilbert DB ecosystem.

    This object is intended to be long-lived and shared:
        - one instance per process (or per service)
        - safe to hand to API handlers

    Attributes
    ----------
    config:
        HilbertDBConfig used to construct this instance.

    db_pool:
        DBPool that provides DBConnection objects on-demand.

    object_store:
        Object store backend used for large immutable artifacts
        (exports, CSVs, graphs, etc.).

    corpus_registry, run_registry, artifact_registry:
        Registry implementations (typically DB-backed) used to
        track corpora, runs, and artifacts.

    cache:
        CacheManager used for local rehydration of exports.

    importer:
        RunImporter that knows how to unpack deterministic exports
        into the local cache.
    """

    config: HilbertDBConfig
    db_pool: DBPool
    object_store: LocalFSObjectStore
    corpus_registry: DBCorpusRegistry
    run_registry: DBRunRegistry
    artifact_registry: DBArtifactRegistry
    cache: CacheManager
    importer: RunImporter

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Optional[HilbertDBConfig] = None,
        *,
        init_schema: bool = True,
    ) -> "HilbertDB":
        """
        Construct a HilbertDB instance from a HilbertDBConfig.

        This:
            - selects the DB backend (sqlite/postgres),
            - optionally bootstraps the schema,
            - wires up object store, registries, cache, and importer.
        """
        cfg = config or load_config()

        if cfg.enable_logging:
            logging.basicConfig(level=logging.INFO)
            logger.info("Initializing HilbertDB with config: %s", cfg)

        # ----------------------------
        # DB backend
        # ----------------------------
        backend = _create_backend_from_config(cfg)
        db_pool = DBPool(backend)

        if init_schema and hasattr(backend, "init_schema"):
            conn = backend.connect()
            try:
                backend.init_schema(conn)
            finally:
                try:
                    conn.close()
                except Exception:
                    logger.exception("Error closing DB connection during schema init")

        # ----------------------------
        # Object store
        # ----------------------------
        os_cfg = ObjectStoreConfig(
            base_path=os.path.abspath(cfg.object_store_root),
            bucket=None,
            read_only=False,
        )
        object_store = LocalFSObjectStore(os_cfg)

        # ----------------------------
        # Cache + importer
        # ----------------------------
        cache = CacheManager(root_dir=cfg.cache_root)
        importer = RunImporter(cache=cache)

        # ----------------------------
        # DB-backed registries
        # ----------------------------
        corpus_registry = DBCorpusRegistry(db_pool)
        run_registry = DBRunRegistry(db_pool)
        artifact_registry = DBArtifactRegistry(db_pool)

        return cls(
            config=cfg,
            db_pool=db_pool,
            object_store=object_store,
            corpus_registry=corpus_registry,
            run_registry=run_registry,
            artifact_registry=artifact_registry,
            cache=cache,
            importer=importer,
        )

    @classmethod
    def from_env(cls, *, init_schema: bool = True) -> "HilbertDB":
        """Construct HilbertDB using environment variables."""
        return cls.from_config(load_config(), init_schema=init_schema)

    # ------------------------------------------------------------------
    # Corpus operations
    # ------------------------------------------------------------------

    def get_or_create_corpus(
        self,
        *,
        fingerprint: str,
        name: str,
        source_uri: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> CorpusRecord:
        """
        Register a corpus or return the existing one by fingerprint.

        This is the main entrypoint the orchestrator should use after
        computing a content hash for the corpus directory.
        """
        existing = self.corpus_registry.get_by_fingerprint(fingerprint)
        if existing is not None:
            return existing

        # By default, we use the fingerprint as the stable corpus_id.
        corpus_id = fingerprint
        return self.corpus_registry.register_corpus(
            corpus_id=corpus_id,
            name=name,
            fingerprint=fingerprint,
            source_uri=source_uri,
            notes=notes,
        )

    def get_corpus(self, corpus_id: str) -> Optional[CorpusRecord]:
        """Fetch a corpus by its id."""
        return self.corpus_registry.get_by_id(corpus_id)

    def list_corpora(self) -> List[CorpusRecord]:
        """List all corpora, newest first."""
        return self.corpus_registry.list_corpora()

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def create_run(
        self,
        *,
        run_id: str,
        corpus_id: str,
        orchestrator_version: Optional[str],
        settings_json: Optional[Dict[str, Any]] = None,
        settings_hash: Optional[str] = None,
    ) -> RunRecord:
        """
        Create (or retrieve) a run record for a corpus.

        settings_hash:
            Optional deterministic hash of (settings + orchestrator_version).
            If provided, stored under "__signature__" in settings_json.
        """
        settings: Dict[str, Any] = dict(settings_json or {})
        if settings_hash is not None:
            settings.setdefault("__signature__", settings_hash)

        return self.run_registry.create_run(
            run_id=run_id,
            corpus_id=corpus_id,
            orchestrator_version=orchestrator_version,
            settings_json=settings,
        )

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """Fetch a run by its id."""
        return self.run_registry.get_by_id(run_id)

    def list_runs_for_corpus(self, corpus_id: str) -> List[RunRecord]:
        """List all runs associated with a corpus, newest first."""
        return self.run_registry.list_runs_for_corpus(corpus_id)

    def get_run_by_signature(
        self,
        *,
        corpus_id: str,
        settings_hash: str,
        orchestrator_version: Optional[str] = None,
    ) -> Optional[RunRecord]:
        """
        Look up an existing run by (corpus_id, settings_hash).

        Implementation detail:
            - The hash is stored under "__signature__" in settings_json.
        """
        runs = self.run_registry.list_runs_for_corpus(corpus_id)
        for run in runs:
            sig = run.settings_json.get("__signature__")
            if sig != settings_hash:
                continue
            if orchestrator_version and run.orchestrator_version != orchestrator_version:
                continue
            return run
        return None

    # ----------------------------
    # Status helpers
    # ----------------------------

    def _set_run_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        finished: bool = False,
    ) -> Optional[RunRecord]:
        """
        Internal helper to update a run's status.

        If finished=True, sets finished_at to the current UTC time.
        """
        finished_at = datetime.utcnow() if finished else None
        return self.run_registry.update_status(
            run_id=run_id,
            status=status,
            finished_at=finished_at,
        )

    def mark_run_pending(self, run_id: str) -> Optional[RunRecord]:
        return self._set_run_status(run_id, "pending")

    def mark_run_running(self, run_id: str) -> Optional[RunRecord]:
        return self._set_run_status(run_id, "running")

    def mark_run_ok(self, run_id: str) -> Optional[RunRecord]:
        return self._set_run_status(run_id, "ok", finished=True)

    def mark_run_failed(self, run_id: str) -> Optional[RunRecord]:
        return self._set_run_status(run_id, "failed", finished=True)

    # ------------------------------------------------------------------
    # Artifact + export handling
    # ------------------------------------------------------------------

    def register_artifact(
        self,
        *,
        run_id: str,
        name: str,
        kind: str,
        key: str,
        artifact_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRecord:
        """
        Register an artifact produced by a run.

        If artifact_id is not provided, a deterministic id is derived from
        (run_id, kind, name).
        """
        if artifact_id is None:
            artifact_id = f"{run_id}:{kind}:{name}"

        return self.artifact_registry.register_artifact(
            artifact_id=artifact_id,
            run_id=run_id,
            name=name,
            kind=kind,
            key=key,
            meta=meta or {},
        )

    def list_artifacts_for_run(
        self,
        run_id: str,
        *,
        kind: Optional[str] = None,
    ) -> List[ArtifactRecord]:
        """
        List artifacts for a given run, optionally filtering by kind.
        """
        return self.artifact_registry.list_for_run(run_id, kind)

    def store_run_export(
        self,
        *,
        run_id: str,
        export_zip_path: str,
        artifact_id: Optional[str] = None,
    ) -> ArtifactRecord:
        """
        Persist a deterministic run export ZIP into the object store and
        register it as an artifact.

        Object store key convention:
            "corpora/{corpus_id}/runs/{run_id}/hilbert_export.zip"

        Also sets RunRecord.export_key for this run.
        """
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id!r} does not exist")

        corpus_id = run.corpus_id
        key = f"corpora/{corpus_id}/runs/{run_id}/hilbert_export.zip"

        # Persist ZIP bytes to object store
        with open(export_zip_path, "rb") as f:
            data = f.read()
        self.object_store.save_bytes(key, data)

        # Record export key on the run
        self.run_registry.set_export_key(run_id, key)

        # Register as an artifact
        return self.register_artifact(
            run_id=run_id,
            name="hilbert_export.zip",
            kind="export",
            key=key,
            artifact_id=artifact_id,
            meta={"corpus_id": corpus_id, "export_path": export_zip_path},
        )

    # ------------------------------------------------------------------
    # Import / rehydration
    # ------------------------------------------------------------------

    def load_imported_run(self, run_id: str) -> ImportedRun:
        """
        Ensure a run's deterministic export is unpacked into the local cache,
        and return an ImportedRun descriptor.

        Fast-path:
            - If a cached run directory exists and contains hilbert_manifest.json,
              reuse it.

        Slow-path:
            - Fetch the export ZIP from object store
            - Feed it into RunImporter.import_from_fileobj(...)
            - Return ImportedRun pointing at the new cache directory.
        """
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id!r} does not exist")
        if not run.export_key:
            raise ValueError(f"Run {run_id!r} has no export_key recorded")

        # Fast-path: reuse cache if available
        try:
            if self.cache.has_cached_run(run_id):
                cache_dir = self.cache.get_run_cache_dir(run_id)
                manifest_path = os.path.join(cache_dir, "hilbert_manifest.json")
                if os.path.isfile(manifest_path):
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    return ImportedRun(
                        run_id=run_id,
                        cache_dir=cache_dir,
                        manifest=manifest,
                    )
        except Exception:
            logger.exception("Cache error for run %s; falling back to full import", run_id)

        # Slow-path: fetch from object store and import
        with self.object_store.open(run.export_key) as fh:
            imported = self.importer.import_from_fileobj(fh, run_id=run_id)
        return imported


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_backend_from_config(config: HilbertDBConfig):
    """
    Instantiate the appropriate DB backend for a given configuration.
    """
    name = (config.db_backend or "").lower()

    if name == "sqlite":
        return SQLiteBackend(config.db_uri)

    if name in ("postgres", "postgresql", "psql"):
        return PostgresBackend(config.db_uri)

    raise ValueError(f"Unsupported Hilbert DB backend: {config.db_backend!r}")


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def create_hilbert_db(
    config: Optional[HilbertDBConfig] = None,
    *,
    init_schema: bool = True,
) -> HilbertDB:
    """
    Convenience constructor used by services / scripts.
    """
    return HilbertDB.from_config(config, init_schema=init_schema)


__all__ = [
    "HilbertDB",
    "create_hilbert_db",
]
