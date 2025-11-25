"""
Global configuration settings for Hilbert DB.

This module centralizes configuration for:

    - database backend selection
    - database URI
    - object store root
    - importer cache root
    - feature flags (logging, etc.)

It provides:
    HilbertDBConfig  – structured config object
    load_config()    – load from environment variables or defaults
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional


@dataclass
class HilbertDBConfig:
    """
    Canonical configuration for the hilbert_db subsystem.

    Attributes
    ----------
    db_backend:
        Name of the backend: "sqlite" or "postgres".

    db_uri:
        - For SQLite: path to the .db file (e.g. "./hilbert.db").
        - For Postgres: a full DSN string
              (e.g. "postgresql://user:pass@host/dbname").

    object_store_root:
        Directory where the object store backend persists artifacts.

    cache_root:
        Directory for importer / rehydration cache.

    enable_logging:
        Whether to enable internal debug logging.
    """

    db_backend: str = "sqlite"
    db_uri: str = "hilbert.db"

    object_store_root: str = "./object_store_data"
    cache_root: str = "./hilbert_cache"

    enable_logging: bool = False


def load_config() -> HilbertDBConfig:
    """
    Load HilbertDBConfig from environment variables, falling back to defaults.

    Recognized variables:
        HILBERT_DB_BACKEND         (sqlite|postgres)
        HILBERT_DB_URI             (path or DSN)
        HILBERT_OBJECT_STORE_ROOT  (directory path)
        HILBERT_CACHE_ROOT         (directory path)
        HILBERT_ENABLE_LOGGING     ("true" / "false" / "1" / "0")

    Returns
    -------
    HilbertDBConfig
    """

    def _env_flag(name: str, default: bool) -> bool:
        val = os.getenv(name)
        if val is None:
            return default
        return val.strip().lower() in ("1", "true", "yes", "on")

    return HilbertDBConfig(
        db_backend=os.getenv("HILBERT_DB_BACKEND", "sqlite"),
        db_uri=os.getenv("HILBERT_DB_URI", "hilbert.db"),

        object_store_root=os.getenv(
            "HILBERT_OBJECT_STORE_ROOT",
            "./object_store_data"
        ),

        cache_root=os.getenv(
            "HILBERT_CACHE_ROOT",
            "./hilbert_cache"
        ),

        enable_logging=_env_flag(
            "HILBERT_ENABLE_LOGGING",
            default=False
        ),
    )
