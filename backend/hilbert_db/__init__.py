"""
hilbert_db

Top-level package initializer for the Hilbert database system.

This module does not contain any logic.
It exposes configuration utilities and ensures the package loads cleanly.

Submodules include:
    - object_store/
    - registry/
    - importer/
    - apis/
    - utils/
    - hashing/
    - db/

This root package exports only the global config loader for convenience.
"""

from .config import HilbertDBConfig, load_config

__all__ = [
    "HilbertDBConfig",
    "load_config",
]
