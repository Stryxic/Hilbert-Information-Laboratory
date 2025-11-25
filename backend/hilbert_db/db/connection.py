"""
Unified database connection abstraction for Hilbert DB.

This file defines:
- DBConnection: a wrapper around a live database handle
- DBPool: simple pool/manager to allocate backend connections

Backends must expose:
    backend.connect() -> raw connection
    backend.helpers   -> module with:
        safe_execute
        safe_executemany
        safe_fetch_all
        safe_fetch_one
        row_to_dict
"""

from __future__ import annotations
from typing import Any, Iterable, Optional


class DBConnection:
    """
    Thin wrapper around a raw DB-API 2.0 connection.

    Responsibilities:
        - Provide a stable API for SQL execution (execute, fetch, etc.)
        - Normalize rows across backends (return Python dicts)
        - Leave transaction handling to the caller

    Notes:
        - No implicit autocommit unless the backend enforces it
        - Caller must commit() after mutating operations
        - Safe to close() multiple times
    """

    def __init__(self, raw_conn: Any, helpers: Any):
        self.raw = raw_conn
        self.helpers = helpers

    # ------------------------------------------------------------------
    # SQL execution wrappers
    # ------------------------------------------------------------------

    def execute(self, query: str, params: Optional[tuple] = None):
        """
        Execute a single SQL statement.
        Returns the underlying cursor.
        """
        return self.helpers.safe_execute(self.raw, query, params)

    def executemany(self, query: str, seq: Iterable[tuple]):
        """
        Bulk-execute the same SQL statement with a sequence of parameters.
        Returns the underlying cursor.
        """
        return self.helpers.safe_executemany(self.raw, query, seq)

    def fetch_all(self, query: str, params: Optional[tuple] = None):
        """
        Execute a SELECT statement and return a list of dict rows.
        """
        rows = self.helpers.safe_fetch_all(self.raw, query, params)
        return [self.helpers.row_to_dict(r) for r in rows]

    def fetch_one(self, query: str, params: Optional[tuple] = None):
        """
        Execute a SELECT statement and return a single dict row or None.
        """
        row = self.helpers.safe_fetch_one(self.raw, query, params)
        return self.helpers.row_to_dict(row) if row else None

    # ------------------------------------------------------------------
    # Transaction and connection lifecycle
    # ------------------------------------------------------------------

    def commit(self) -> None:
        """
        Commit the current transaction.
        """
        try:
            self.raw.commit()
        except Exception as e:
            raise RuntimeError(f"Commit failed: {e}") from e

    def rollback(self) -> None:
        """
        Roll back the current transaction.
        """
        try:
            self.raw.rollback()
        except Exception:
            # Some backends auto-handle rollback; this is best-effort only.
            pass

    def close(self) -> None:
        """
        Close the underlying connection safely.
        """
        try:
            self.raw.close()
        except Exception:
            # Allow double-close or backend errors w/out propagating
            pass


# ----------------------------------------------------------------------
# DB Pool
# ----------------------------------------------------------------------

class DBPool:
    """
    Simple database connection factory.

    The backend must provide:
        - connect()  -> raw DB-API connection
        - helpers    -> module with DB helper functions
    """

    def __init__(self, backend: Any):
        self.backend = backend

    def get(self) -> DBConnection:
        """
        Acquire a new DBConnection wrapper.
        """
        raw = self.backend.connect()
        return DBConnection(raw, self.backend.helpers)

    # ------------------------------------------------------------------
    # Context manager syntax:
    #     with db_pool.connection() as conn:
    #         ...
    # ------------------------------------------------------------------

    def connection(self):
        return _ConnectionContext(self)


class _ConnectionContext:
    """
    Internal context manager for DBConnection.
    """

    def __init__(self, pool: DBPool):
        self.pool = pool
        self.conn: Optional[DBConnection] = None

    def __enter__(self) -> DBConnection:
        self.conn = self.pool.get()
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        if self.conn is None:
            return False

        # Rollback on error
        if exc_type is not None:
            try:
                self.conn.rollback()
            except Exception:
                pass

        # Always close
        self.conn.close()

        # Propagate exceptions
        return False
