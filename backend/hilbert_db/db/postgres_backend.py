"""
Postgres backend for Hilbert DB.

This backend mirrors the interface expected by:
    - DBPool
    - Registry layer
    - HilbertDB core façade

It provides:
    - connect()
    - helpers (safe_execute, safe_fetch_all, etc.)
    - init_schema(conn)  (stub, since production Postgres uses migrations)

This file intentionally keeps the connection semantics simple.
"""

from __future__ import annotations
from typing import Any
try:
    import psycopg2  # type: ignore
    import psycopg2.extras  # type: ignore
except ImportError:
    psycopg2 = None


from . import helpers
from .backend_base import DBBackend


class PostgresBackend(DBBackend):
    """
    Minimal Postgres backend implementation.

    Parameters
    ----------
    dsn : str
        Full Postgres DSN, e.g.:
        "postgresql://user:pass@host:5432/dbname"
    """

    def __init__(self, dsn: str):
        self.dsn = dsn
        self.helpers = helpers

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------

    def connect(self) -> Any:
        """
        Create a psycopg2 connection with dict-like row access.

        DBPool will wrap this in DBConnection and manage commits.
        """
        conn = psycopg2.connect(self.dsn)

        # psycopg2 uses connection.cursor(factory=...) rather than setting
        # cursor_factory globally. Storing it here ensures DBConnection.get_cursor()
        # will produce dict-like rows.
        conn.autocommit = False
        conn.cursor_factory = psycopg2.extras.RealDictCursor  # type: ignore

        return conn

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def init_schema(self, conn) -> None:
        """
        Postgres deployments should use explicit migrations.

        This is intentionally a no-op:
            - production environments rely on Alembic or another migration tool
            - tables must NOT be implicitly created
        """
        return None
