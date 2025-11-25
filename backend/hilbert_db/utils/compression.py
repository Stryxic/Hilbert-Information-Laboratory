"""
Compression helpers for Hilbert DB.

These functions intentionally mirror the importer ZIP operations
but remain separate for easier unit testing and reuse.

All ZIP operations are "safe mode":
    - no path traversal
    - explicit destination directories
    - defensive error handling
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import List


def zip_directory(src_dir: Path, zip_path: Path) -> bool:
    """
    Create a ZIP archive from a directory tree.

    Parameters
    ----------
    src_dir : Path
        Directory to archive (recursive).
    zip_path : Path
        Output ZIP file path (parent directories will be created).

    Returns
    -------
    bool
        True on success, False otherwise.
    """
    src_dir = Path(src_dir)
    zip_path = Path(zip_path)

    try:
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(src_dir):
                root_path = Path(root)
                for name in files:
                    full = root_path / name
                    # Store paths relative to src_dir to keep archives portable
                    rel = full.relative_to(src_dir)
                    zf.write(full, rel.as_posix())

        return True
    except Exception:
        # Best-effort; callers can inspect filesystem if needed.
        return False


def unzip_safe(zip_path: Path, dst_dir: Path) -> List[Path]:
    """
    Safely extract a ZIP archive into a destination directory.

    Path traversal prevention:
        - Normalize each member path.
        - Reject absolute paths.
        - Reject paths that would escape dst_dir after normalization.

    Parameters
    ----------
    zip_path : Path
        Path to the ZIP archive.
    dst_dir : Path
        Destination directory (created if missing).

    Returns
    -------
    List[Path]
        List of extracted file paths (absolute).
    """
    zip_path = Path(zip_path)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    extracted: List[Path] = []
    base = dst_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            name = member.filename

            # Skip directories (they will be created as needed)
            if not name or name.endswith("/"):
                continue

            # Normalize the path
            norm = Path(name)

            # Reject absolute paths
            if norm.is_absolute():
                continue

            # Compute the final target path and ensure it is within base
            target = (dst_dir / norm).resolve()
            if not str(target).startswith(str(base) + os.sep) and target != base:
                # Potential zip-slip, skip
                continue

            target.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(member, "r") as src, open(target, "wb") as dst:
                dst.write(src.read())

            extracted.append(target)

    return extracted


__all__ = [
    "zip_directory",
    "unzip_safe",
]
