"""
hilbert_db.hashing

Unified content hashing utilities for the Hilbert DB.

This package provides:
- compute_content_hash: stable, deterministic hashing for files,
  directories, raw bytes, or strings.

Hashing is used throughout Hilbert DB to deduplicate corpora,
detect already-processed inputs, and form registry keys.
"""

from .content_hash import compute_content_hash

__all__ = [
    "compute_content_hash",
]
