"""
hilbert_db.db

Database backend abstraction layer for Hilbert DB.

This package provides:

- A backend-agnostic connection abstraction:
      * DBConnection
      * DBPool

- Helper functions for safe SQL execution and row mapping:
      * safe_execute
      * safe_fetch_all
      * safe_fetch_one
      * safe_executemany
      * row_to_dict

- Concrete database backend implementations:
      * SQLiteBackend   (default: local development + tests)
      * PostgresBackend (future production backend)

- Migration utilities for schema upgrades:
      * MigrationManager

- Backend contracts:
      * DBBackend
      * BackendLike
      * ensure_backend
"""

from .connection import DBConnection, DBPool
from .sqlite_backend import SQLiteBackend
from .postgres_backend import PostgresBackend
from .backend_base import DBBackend, BackendLike, ensure_backend
from .helpers import (
    safe_execute,
    safe_executemany,
    safe_fetch_all,
    safe_fetch_one,
    row_to_dict,
)
from .migrations import MigrationManager

__all__ = [
    # Connection / Pool
    "DBConnection",
    "DBPool",

    # Backends
    "SQLiteBackend",
    "PostgresBackend",
    "DBBackend",
    "BackendLike",
    "ensure_backend",

    # Helpers
    "safe_execute",
    "safe_executemany",
    "safe_fetch_all",
    "safe_fetch_one",
    "row_to_dict",

    # Migrations
    "MigrationManager",
]
