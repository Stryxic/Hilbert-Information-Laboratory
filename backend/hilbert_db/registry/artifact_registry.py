"""
DB-backed Artifact Registry.

Maps artifacts produced by a run into the database.
"""

from __future__ import annotations

import json
from typing import List, Optional

from .models import ArtifactRecord
from ..db.connection import DBPool


class DBArtifactRegistry:
    """
    Database-backed artifact registry.

    Responsible for storing and retrieving artifacts produced by a run.
    Artifacts include CSVs, JSON files, graph snapshots, export ZIPs, etc.
    """

    def __init__(self, pool: DBPool):
        self.pool = pool

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def register_artifact(
        self,
        artifact_id: str,
        run_id: str,
        name: str,
        kind: str,
        key: str,
        meta: dict,
    ) -> ArtifactRecord:
        """
        Insert a new artifact row.

        If an artifact with this artifact_id already exists, return the
        existing ArtifactRecord instead of overwriting.

        Parameters
        ----------
        artifact_id : str
            Deterministic identifier, usually f"{run_id}:{kind}:{name}".
        run_id : str
        name : str
        kind : str
        key : str
            Logical object-store key.
        meta : dict
            Arbitrary JSON metadata.

        Returns
        -------
        ArtifactRecord
        """
        existing = self.get_by_id(artifact_id)
        if existing:
            return existing

        conn = self.pool.get()
        try:
            conn.execute(
                """
                INSERT INTO artifacts(artifact_id, run_id, name, kind, key, meta_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (artifact_id, run_id, name, kind, key, json.dumps(meta)),
            )
            conn.commit()
        finally:
            conn.close()

        return ArtifactRecord(
            artifact_id=artifact_id,
            run_id=run_id,
            name=name,
            kind=kind,
            key=key,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_by_id(self, artifact_id: str) -> Optional[ArtifactRecord]:
        conn = self.pool.get()
        try:
            row = conn.fetch_one(
                """
                SELECT *
                FROM artifacts
                WHERE artifact_id = ?
                """,
                (artifact_id,),
            )
        finally:
            conn.close()

        if not row:
            return None

        return ArtifactRecord(
            artifact_id=row["artifact_id"],
            run_id=row["run_id"],
            name=row["name"],
            kind=row["kind"],
            key=row["key"],
            meta=json.loads(row.get("meta_json") or "{}"),
            created_at=row.get("created_at"),
        )

    def list_for_run(
        self,
        run_id: str,
        kind: Optional[str] = None,
    ) -> List[ArtifactRecord]:
        conn = self.pool.get()
        try:
            if kind is None:
                rows = conn.fetch_all(
                    """
                    SELECT *
                    FROM artifacts
                    WHERE run_id = ?
                    ORDER BY created_at ASC
                    """,
                    (run_id,),
                )
            else:
                rows = conn.fetch_all(
                    """
                    SELECT *
                    FROM artifacts
                    WHERE run_id = ? AND kind = ?
                    ORDER BY created_at ASC
                    """,
                    (run_id, kind),
                )
        finally:
            conn.close()

        return [
            ArtifactRecord(
                artifact_id=row["artifact_id"],
                run_id=row["run_id"],
                name=row["name"],
                kind=row["kind"],
                key=row["key"],
                meta=json.loads(row.get("meta_json") or "{}"),
                created_at=row.get("created_at"),
            )
            for row in rows
        ]
