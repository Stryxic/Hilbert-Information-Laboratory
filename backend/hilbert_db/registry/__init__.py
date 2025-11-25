"""
Hilbert DB - Registry package.

This package provides:
    - Data model records (CorpusRecord, RunRecord, ArtifactRecord)
    - Database-backed registry implementations:
          * DBCorpusRegistry
          * DBRunRegistry
          * DBArtifactRegistry

The registries expose a stable logical interface for:
    - registering corpora, runs, and artifacts
    - retrieving existing records
    - listing items for navigation
    - updating metadata or status

They sit above the database backend (SQLite, Postgres, etc.) and are used
by the HilbertDB façade, orchestrator, backend API, and importer layer.
"""

from .models import CorpusRecord, RunRecord, ArtifactRecord
from .corpus_registry import DBCorpusRegistry
from .run_registry import DBRunRegistry
from .artifact_registry import DBArtifactRegistry

__all__ = [
    # Data model records
    "CorpusRecord",
    "RunRecord",
    "ArtifactRecord",

    # DB-backed registries
    "DBCorpusRegistry",
    "DBRunRegistry",
    "DBArtifactRegistry",
]
