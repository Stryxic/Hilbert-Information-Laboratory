"""
Temporary directory utilities.

Centralized helpers for creating and cleaning up temporary directories.
These are used across HilbertDB for:

    - safe ZIP extraction
    - object-store staging
    - importer sandboxes
    - ephemeral pipeline artifacts

All cleanup operations are best-effort and error-tolerant.
"""

from __future__ import annotations

import tempfile
import shutil
from pathlib import Path
from typing import Optional


def ensure_temp_dir(prefix: str = "hilbertdb_") -> Path:
    """
    Create and return a temporary directory under the system temp root.

    Parameters
    ----------
    prefix : str
        Prefix for the directory name. Defaults to "hilbertdb_".

    Returns
    -------
    Path
        Newly created temporary directory path.

    Caller is responsible for invoking cleanup_temp_dir() when finished.
    """
    path = Path(tempfile.mkdtemp(prefix=prefix))
    return path


def cleanup_temp_dir(path: Optional[Path]) -> None:
    """
    Safely remove a temporary directory and all its contents.

    Parameters
    ----------
    path : Optional[Path]
        Path to the directory to remove. If None, does nothing.

    Notes
    -----
    This function intentionally suppresses all errors. Temp directory
    cleanup is not a critical operation, and callers should not need to
    worry about race conditions, permissions, or partial failures.
    """
    if not path:
        return

    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        # Best-effort cleanup – intentionally ignore all exceptions
        pass


__all__ = [
    "ensure_temp_dir",
    "cleanup_temp_dir",
]
