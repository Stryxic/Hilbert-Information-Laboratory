"""
Backend base interfaces for Hilbert DB.

This module defines the minimal contracts that all database backends
(SQLite, Postgres, etc.) must satisfy.

It is intentionally light-weight:

- It does NOT depend on any specific DB driver.
- It only encodes the structural requirements assumed by:
      * hilbert_db.db.connection.DBPool
      * hilbert_db.db.helpers
      * hilbert_db.db.migrations

Backends must expose:

    backend.connect() -> raw_connection
    backend.helpers   -> module with:
                           - safe_execute(conn, query, params)
                           - safe_fetch_all(conn, query, params)
                           - safe_fetch_one(conn, query, params)
                           - safe_executemany(conn, seq)
                           - row_to_dict(row)

    backend.init_schema(conn)  # optional

This file provides:
- DBBackend: abstract base class
- BackendLike: structural protocol
- ensure_backend: runtime validator
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any, runtime_checkable
try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


# ---------------------------------------------------------------------------
# Abstract Base Backend
# ---------------------------------------------------------------------------

class DBBackend(ABC):
    """
    Abstract base class for a Hilbert DB backend.

    Concrete subclasses may define any constructor signature they want
    (e.g. SQLiteBackend(db_path), PostgresBackend(dsn)).

    Required interface:

        @property
        helpers  – module providing safe_execute, row_to_dict, etc.
        connect() -> raw DB-API connection
        init_schema(conn) -> None (optional; default is a no-op)
    """

    @property
    @abstractmethod
    def helpers(self) -> Any:
        """
        Return the helper module associated with this backend.

        Normally this is hilbert_db.db.helpers, but alternative backends
        (mocks, test backends) may provide compatible modules.
        """
        raise NotImplementedError

    @abstractmethod
    def connect(self) -> Any:
        """
        Acquire and return a new raw DB-API 2.0 connection.
        """
        raise NotImplementedError

    def init_schema(self, conn: Any) -> None:
        """
        Optional schema bootstrap.

        SQLite backends typically populate the canonical schema here.
        Postgres backends normally leave this empty and rely on migrations.

        Default: no-op.
        """
        return None


# ---------------------------------------------------------------------------
# Structural Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class BackendLike(Protocol):
    """
    Structural protocol for objects usable as a Hilbert DB backend.

    Any backend must satisfy:

        backend.helpers
        backend.connect()
        backend.init_schema(conn)

    This enables DBPool and the registry layer to operate on it without
    knowing the concrete backend implementation.
    """

    helpers: Any

    def connect(self) -> Any:
        ...

    def init_schema(self, conn: Any) -> None:
        ...


# ---------------------------------------------------------------------------
# Runtime Guard
# ---------------------------------------------------------------------------

def ensure_backend(backend: Any) -> BackendLike:
    """
    Validate that an object behaves like a Hilbert DB backend.

    First, try an isinstance check against BackendLike.
    If that fails (runtime_checkable sometimes struggles with C extensions),
    fall back to manual attribute inspection.

    Raises:
        TypeError if required attributes are missing.
    """
    # First attempt: protocol check
    if not isinstance(backend, BackendLike):
        missing = []

        if not hasattr(backend, "connect"):
            missing.append("connect")
        if not hasattr(backend, "helpers"):
            missing.append("helpers")
        if not hasattr(backend, "init_schema"):
            missing.append("init_schema")

        if missing:
            raise TypeError(
                f"Invalid Hilbert DB backend {backend!r}: missing attributes {missing}"
            )

    return backend  # type: ignore[return-value]


__all__ = [
    "DBBackend",
    "BackendLike",
    "ensure_backend",
]
