"""
Hilbert DB - Importer package.

Utilities for rehydrating deterministic Hilbert exports into a local
cache for fast, repeated access from the frontend or analysis tools.

Primary entrypoints:

    - RunImporter  : orchestrates unzip + cache + manifest loading
    - CacheManager  : filesystem-backed cache for imported runs
    - ZipUnpacker  : safe ZIP unpacking with directory traversal protection
    - ImportedRun  : lightweight descriptor for a rehydrated run
"""

from .cache import CacheManager
from .unzipper import ZipUnpacker
from .run_importer import RunImporter, ImportedRun

__all__ = [
    "CacheManager",
    "ZipUnpacker",
    "RunImporter",
    "ImportedRun",
]
