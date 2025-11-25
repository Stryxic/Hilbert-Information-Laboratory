"""
DB-backed Run Registry.
"""

from __future__ import annotations

import json
from typing import List, Optional, Dict, Any
from dataclasses import replace
from datetime import datetime

from .models import RunRecord
from ..db.connection import DBPool


class DBRunRegistry:
    """
    Database-backed registry for pipeline runs.

    Schema (canonical):

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
    """

    def __init__(self, pool: DBPool):
        self.pool = pool

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        corpus_id: str,
        *,
        orchestrator_version: Optional[str],
        settings_json: Dict[str, Any],
    ) -> RunRecord:
        """
        Idempotent: if run exists, return it without modifying.
        """
        existing = self.get_by_id(run_id)
        if existing:
            return existing

        conn = self.pool.get()
        try:
            conn.execute(
                """
                INSERT INTO runs(run_id, corpus_id, orchestrator_version, settings_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    corpus_id,
                    orchestrator_version,
                    json.dumps(settings_json),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return RunRecord(
            run_id=run_id,
            corpus_id=corpus_id,
            orchestrator_version=orchestrator_version,
            settings_json=settings_json,
        )

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_by_id(self, run_id: str) -> Optional[RunRecord]:
        conn = self.pool.get()
        try:
            row = conn.fetch_one(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            )
        finally:
            conn.close()

        return self._row_to_rec(row) if row else None

    def list_runs_for_corpus(self, corpus_id: str) -> List[RunRecord]:
        conn = self.pool.get()
        try:
            rows = conn.fetch_all(
                """
                SELECT *
                FROM runs
                WHERE corpus_id = ?
                ORDER BY created_at DESC
                """,
                (corpus_id,),
            )
        finally:
            conn.close()

        return [self._row_to_rec(r) for r in rows]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update_status(
        self,
        run_id: str,
        status: str,
        *,
        finished_at: Optional[datetime] = None,
    ) -> Optional[RunRecord]:
        """
        Update the status and optional finished_at timestamp for a run.

        finished_at:
            If provided, stored as ISO-8601 string.
        """
        rec = self.get_by_id(run_id)
        if not rec:
            return None

        # Persist as an ISO string, but keep the Python object on RunRecord if desired.
        finished_value = finished_at.isoformat() if finished_at else rec.finished_at

        conn = self.pool.get()
        try:
            conn.execute(
                "UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?",
                (status, finished_value, run_id),
            )
            conn.commit()
        finally:
            conn.close()

        return replace(rec, status=status, finished_at=finished_value)

    def set_export_key(self, run_id: str, key: str) -> Optional[RunRecord]:
        """
        Record the object-store key of the deterministic export ZIP for this run.
        """
        rec = self.get_by_id(run_id)
        if not rec:
            return None

        conn = self.pool.get()
        try:
            conn.execute(
                "UPDATE runs SET export_key = ? WHERE run_id = ?",
                (key, run_id),
            )
            conn.commit()
        finally:
            conn.close()

        return replace(rec, export_key=key)

    def update_settings(
        self,
        run_id: str,
        settings_json: Dict[str, Any],
    ) -> Optional[RunRecord]:
        """
        Overwrite the settings_json blob for a run.

        This is primarily useful when attaching additional metadata
        or normalized configuration after the run has started.
        """
        rec = self.get_by_id(run_id)
        if not rec:
            return None

        conn = self.pool.get()
        try:
            conn.execute(
                "UPDATE runs SET settings_json = ? WHERE run_id = ?",
                (json.dumps(settings_json), run_id),
            )
            conn.commit()
        finally:
            conn.close()

        return replace(rec, settings_json=settings_json)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _row_to_rec(self, row: dict) -> RunRecord:
        """
        Map a raw row dict -> RunRecord.
        """
        settings = json.loads(row.get("settings_json") or "{}")

        return RunRecord(
            run_id=row["run_id"],
            corpus_id=row["corpus_id"],
            orchestrator_version=row.get("orchestrator_version"),
            settings_json=settings,
            status=row.get("status"),
            created_at=row.get("created_at"),
            finished_at=row.get("finished_at"),
            export_key=row.get("export_key"),
        )
