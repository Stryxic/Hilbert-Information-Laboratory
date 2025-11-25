"""
SQLite backend for Hilbert DB.

Used for:
    - local development
    - tests
    - small to medium corpora
    - CLI tools

Implements:
    - connect()
    - helpers      (required by DBBackend abstract interface)
    - init_schema()

SQLite is fully capable for deterministic local runs.
"""

from __future__ import annotations
from typing import Any
import sqlite3
from pathlib import Path

from . import helpers
from .backend_base import DBBackend


# ----------------------------------------------------------------------
# Canonical Schema (matches schema.sql v1 exactly)
# ----------------------------------------------------------------------

SQL_SCHEMA = """
-- ============================================================
-- Hilbert DB Canonical Schema (v1)
-- ============================================================

-- ------------------------------------------------------------
-- Schema version table (for future migrations)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_version (
    version      INTEGER NOT NULL,
    applied_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- Corpus Table
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS corpus (
    corpus_id      TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    fingerprint    TEXT UNIQUE NOT NULL,
    source_uri     TEXT,
    notes          TEXT,
    status         TEXT DEFAULT 'active',
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_corpus_fingerprint
    ON corpus(fingerprint);

-- ------------------------------------------------------------
-- Runs Table
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS runs (
    run_id                TEXT PRIMARY KEY,
    corpus_id             TEXT NOT NULL,
    orchestrator_version  TEXT,
    settings_json         TEXT,
    status                TEXT DEFAULT 'pending',
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at           TIMESTAMP,
    export_key            TEXT,
    FOREIGN KEY (corpus_id) REFERENCES corpus(corpus_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_corpus
    ON runs(corpus_id);

-- ------------------------------------------------------------
-- Artifacts Table
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id   TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    name          TEXT NOT NULL,
    kind          TEXT,
    key           TEXT NOT NULL,
    meta_json     TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_run
    ON artifacts(run_id);

-- ------------------------------------------------------------
-- Initial schema version
-- ------------------------------------------------------------
INSERT INTO schema_version (version)
SELECT 1
WHERE NOT EXISTS (SELECT 1 FROM schema_version);
"""


# ----------------------------------------------------------------------
# Backend implementation
# ----------------------------------------------------------------------

class SQLiteBackend(DBBackend):
    """
    Minimal SQLite backend.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    """

    def __init__(self, db_path: str):
        self.path = Path(db_path)
        # Store helpers internally - DBBackend requires a `helpers` property
        self._helpers = helpers

    # ------------------------------------------------------------------
    # Required abstract property implementation
    # ------------------------------------------------------------------
    @property
    def helpers(self):
        """
        Required by DBBackend.

        Returns the module containing query helpers, serializers,
        and other DB utilities.
        """
        return self._helpers

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """
        Open a SQLite3 connection with row_factory=dict-like access.

        Also ensures foreign keys are enforced.
        """
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row

        try:
            conn.execute("PRAGMA foreign_keys = ON;")
        except Exception:
            pass

        return conn

    # ------------------------------------------------------------------
    # Schema initializer
    # ------------------------------------------------------------------

    def init_schema(self, conn) -> None:
        """
        Create tables and indices if they do not exist.

        Idempotent – safe to call multiple times.
        """
        cur = conn.cursor()
        cur.executescript(SQL_SCHEMA)
        conn.commit()
