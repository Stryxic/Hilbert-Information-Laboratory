"""
Shared DB helper utilities.

These wrappers ensure:
    - safety
    - consistent interfaces across backends
    - predictable row→dict mapping
    - structured error handling

Backends import this module as `.helpers`
"""

from __future__ import annotations
from typing import Any, Optional, Iterable


# ----------------------------------------------------------------------
# Execution helpers
# ----------------------------------------------------------------------

def safe_execute(conn: Any, query: str, params: Optional[tuple] = None):
    """
    Execute a single SQL statement safely.
    Returns the raw cursor.

    Parameters
    ----------
    conn:
        DB-API compatible connection object (sqlite3, psycopg2, etc.).
    query:
        SQL string with placeholders.
    params:
        Optional parameter tuple.

    Raises
    ------
    RuntimeError
        Wrapped execution error with context.
    """
    cur = conn.cursor()
    try:
        cur.execute(query, params or ())
    except Exception as e:
        raise RuntimeError(
            f"DB execute failed: {e} | Query: {query!r} | Params: {params!r}"
        ) from e
    return cur


def safe_executemany(conn: Any, query: str, seq: Iterable[tuple]):
    """
    Execute the same SQL statement for multiple parameter sets.

    Parameters
    ----------
    conn:
        DB-API compatible connection object.
    query:
        SQL string with placeholders.
    seq:
        Iterable of parameter tuples.

    Raises
    ------
    RuntimeError
        Wrapped execution error with context.
    """
    cur = conn.cursor()
    try:
        cur.executemany(query, seq)
    except Exception as e:
        raise RuntimeError(
            f"DB executemany failed: {e} | Query: {query!r}"
        ) from e
    return cur


def safe_fetch_all(conn: Any, query: str, params: Optional[tuple] = None):
    """
    Execute a SELECT query and fetch all rows.

    Returns
    -------
    list
        List of backend-specific row records (e.g., sqlite3.Row).
    """
    cur = safe_execute(conn, query, params)
    return cur.fetchall()


def safe_fetch_one(conn: Any, query: str, params: Optional[tuple] = None):
    """
    Execute a SELECT query and fetch one row.

    Returns
    -------
    Any
        Backend-specific row object or None.
    """
    cur = safe_execute(conn, query, params)
    return cur.fetchone()


# ----------------------------------------------------------------------
# Row mapping
# ----------------------------------------------------------------------

def row_to_dict(row: Any) -> dict:
    """
    Convert sqlite3.Row or psycopg2 RealDictRow to a plain Python dict.

    This normalizes row outputs across backends.

    Parameters
    ----------
    row:
        Backend-specific row object.

    Returns
    -------
    dict
        Plain Python dictionary representation of the row.
    """
    if row is None:
        return {}

    # sqlite3.Row, psycopg2.extras.RealDictRow, etc.
    if hasattr(row, "keys"):
        return {k: row[k] for k in row.keys()}

    # Fallback: treat as a tuple-like sequence
    return dict(enumerate(row))


__all__ = [
    "safe_execute",
    "safe_executemany",
    "safe_fetch_all",
    "safe_fetch_one",
    "row_to_dict",
]
