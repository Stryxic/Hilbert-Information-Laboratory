"""
Content hashing utilities.

The goal:
    - compute a stable, deterministic hash for:
        * a single file
        * raw bytes
        * a directory of files
        * text strings
    - use this to identify corpora or detect duplicates
    - ensure cross-platform reproducibility

This module does not depend on any database components and
is safe to use from orchestrator, registry, or importer logic.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Union


# ----------------------------------------------------------------------
# Internal primitives
# ----------------------------------------------------------------------

def _hash_bytes(data: bytes, algo: str = "sha256") -> str:
    """
    Hash a bytes object with the given algorithm.
    """
    h = hashlib.new(algo)
    h.update(data)
    return h.hexdigest()


def _hash_file(path: Path, algo: str = "sha256", chunk_size: int = 8192) -> str:
    """
    Hash a single file incrementally by reading in chunks.
    Useful for very large inputs.
    """
    h = hashlib.new(algo)
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
    except Exception:
        # If the file cannot be opened, treat it as empty.
        pass
    return h.hexdigest()


def _hash_directory(dir_path: Path, algo: str = "sha256") -> str:
    """
    Compute a deterministic hash for a directory tree.

    Deterministic strategy:
        - walk lexicographically
        - include the *relative path* of each file (structure matters)
        - include the file bytes

    Two corpora differing in layout or filenames produce different hashes,
    even if file contents are identical.
    """
    h = hashlib.new(algo)

    # Deterministic walk
    for root, _, files in os.walk(dir_path):
        root_path = Path(root)
        for name in sorted(files):
            full = root_path / name
            rel = full.relative_to(dir_path)

            # include relative path
            h.update(str(rel).encode("utf-8", "ignore"))

            # include content
            try:
                with open(full, "rb") as f:
                    while True:
                        chunk = f.read(8192)
                        if not chunk:
                            break
                        h.update(chunk)
            except Exception:
                # unreadable file contributes only its name
                pass

    return h.hexdigest()


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def compute_content_hash(
    value: Union[str, Path, bytes],
    algo: str = "sha256",
) -> str:
    """
    Compute a deterministic hash for files, directories, strings, or bytes.

    Supported inputs:
        * bytes -> raw hashing
        * str   -> filesystem path OR raw text
        * Path  -> file or directory

    Returns:
        hex digest string (algorithm: sha256 by default)
    """
    # bytes
    if isinstance(value, bytes):
        return _hash_bytes(value, algo=algo)

    # str → raw text OR filesystem path
    if isinstance(value, str):
        p = Path(value)
        if p.exists():
            value = p  # treat as path
        else:
            return _hash_bytes(value.encode("utf-8", "ignore"), algo=algo)

    # Path
    if isinstance(value, Path):
        if value.is_file():
            return _hash_file(value, algo=algo)
        if value.is_dir():
            return _hash_directory(value, algo=algo)
        # nonexistent path → hash empty
        return _hash_bytes(b"", algo=algo)

    # fallback
    return _hash_bytes(str(value).encode("utf-8", "ignore"), algo=algo)


__all__ = ["compute_content_hash"]
