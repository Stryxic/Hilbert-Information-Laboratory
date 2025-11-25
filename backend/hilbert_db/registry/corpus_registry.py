"""
DB-backed Corpus Registry.

Responsible for insertions and lookups of corpus metadata.
"""

from __future__ import annotations

from typing import List, Optional

from .models import CorpusRecord
from ..db.connection import DBPool


class DBCorpusRegistry:
    """
    Database-backed registry for corpus metadata.

    A corpus is uniquely identified by its fingerprint (content hash),
    which also typically serves as its corpus_id.

    Schema (canonical):
        corpus(
            corpus_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            fingerprint TEXT UNIQUE NOT NULL,
            source_uri TEXT,
            notes TEXT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """

    def __init__(self, pool: DBPool):
        self.pool = pool

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def register_corpus(
        self,
        corpus_id: str,
        name: str,
        fingerprint: str,
        source_uri: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> CorpusRecord:
        """
        Register a new corpus unless one with the same fingerprint
        already exists.

        Returns an existing CorpusRecord if fingerprint already known.
        """
        existing = self.get_by_fingerprint(fingerprint)
        if existing:
            return existing

        conn = self.pool.get()
        try:
            conn.execute(
                """
                INSERT INTO corpus (corpus_id, name, fingerprint, source_uri, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (corpus_id, name, fingerprint, source_uri, notes),
            )
            conn.commit()
        finally:
            conn.close()

        return CorpusRecord(
            corpus_id=corpus_id,
            name=name,
            fingerprint=fingerprint,
            source_uri=source_uri,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_by_id(self, corpus_id: str) -> Optional[CorpusRecord]:
        conn = self.pool.get()
        try:
            row = conn.fetch_one(
                """
                SELECT *
                FROM corpus
                WHERE corpus_id = ?
                """,
                (corpus_id,),
            )
        finally:
            conn.close()

        return self._row_to_rec(row) if row else None

    def get_by_fingerprint(self, fingerprint: str) -> Optional[CorpusRecord]:
        conn = self.pool.get()
        try:
            row = conn.fetch_one(
                """
                SELECT *
                FROM corpus
                WHERE fingerprint = ?
                """,
                (fingerprint,),
            )
        finally:
            conn.close()

        return self._row_to_rec(row) if row else None

    def list_corpora(self) -> List[CorpusRecord]:
        conn = self.pool.get()
        try:
            rows = conn.fetch_all(
                """
                SELECT *
                FROM corpus
                ORDER BY created_at DESC
                """
            )
        finally:
            conn.close()

        return [self._row_to_rec(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _row_to_rec(self, row) -> CorpusRecord:
        """
        Map a raw SQL row dict → CorpusRecord dataclass.
        """
        return CorpusRecord(
            corpus_id=row["corpus_id"],
            name=row["name"],
            fingerprint=row["fingerprint"],
            source_uri=row["source_uri"],
            notes=row["notes"],
            status=row.get("status"),
            created_at=row.get("created_at"),
        )
