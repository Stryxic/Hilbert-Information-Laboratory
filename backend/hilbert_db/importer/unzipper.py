"""
Safe ZIP unpacking utilities for Hilbert run imports.

ZipUnpacker:
    - prevents directory traversal ("zip slip")
    - optionally filters by allowed file extensions
    - writes files into a destination directory safely
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class ZipUnpacker:
    """
    Safe ZIP extraction helper.

    Parameters
    ----------
    allowed_extensions: Optional[Iterable[str]]
        Only extract files with these extensions (including the dot).
        If None, extract all entries.
    """

    allowed_extensions: Optional[Iterable[str]] = None

    def extract(self, zip_path: str, dest_dir: str) -> List[str]:
        """
        Extract a ZIP file into dest_dir with safety protections.

        Returns
        -------
        List[str]:
            List of *relative* extracted file paths.

        Raises
        ------
        zipfile.BadZipFile
        OSError / IOError
        """
        dest_dir = os.path.abspath(dest_dir)
        os.makedirs(dest_dir, exist_ok=True)

        allowed = (
            {ext.lower() for ext in self.allowed_extensions}
            if self.allowed_extensions is not None
            else None
        )

        extracted: List[str] = []

        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                name = member.filename

                # Skip empty and directory entries
                if not name or name.endswith("/"):
                    continue

                # Normalize
                normalized = os.path.normpath(name).replace("\\", "/")

                # Prevent zip-slip attacks
                if normalized.startswith("../") or normalized.startswith("..\\"):
                    continue

                if allowed is not None:
                    _, ext = os.path.splitext(normalized)
                    if ext.lower() not in allowed:
                        continue

                dest_path = os.path.abspath(os.path.join(dest_dir, normalized))

                # Ensure final path is still within dest_dir
                if not dest_path.startswith(dest_dir + os.sep):
                    continue

                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                with zf.open(member, "r") as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())

                rel = os.path.relpath(dest_path, dest_dir).replace("\\", "/")
                extracted.append(rel)

        return extracted
