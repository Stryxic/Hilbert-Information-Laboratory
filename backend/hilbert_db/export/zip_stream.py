"""
Deterministic ZIP writer for Hilbert exports.

ZipStreamWriter walks a directory tree, collects files, and writes
them into a ZIP archive using a stable, lexicographic ordering of
paths. This ensures reproducible archives when the underlying files
and directory layout are unchanged.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Union, BinaryIO


class ZipStreamWriter:
    """
    Deterministic ZIP builder.

    Parameters
    ----------
    base_dir :
        Root directory whose contents will be archived.
    ignore_suffixes :
        Optional set of file suffixes to skip (for example,
        {".zip"} to avoid zipping existing archives).
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        ignore_suffixes: Optional[Iterable[str]] = None,
    ) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.ignore_suffixes = {s.lower() for s in (ignore_suffixes or [])}

    def _collect_files(self) -> List[Path]:
        """
        Collect all files under base_dir, sorted lexicographically
        by their relative path.
        """
        files: List[Path] = []
        for root, _, filenames in os.walk(self.base_dir):
            root_path = Path(root)
            for name in filenames:
                full = root_path / name
                if self.ignore_suffixes and full.suffix.lower() in self.ignore_suffixes:
                    continue
                files.append(full)

        files.sort(key=lambda p: str(p.relative_to(self.base_dir)).replace("\\", "/"))
        return files

    def write_to_path(self, zip_path: Union[str, Path]) -> Path:
        """
        Write the ZIP archive to a file path.

        Returns the path to the written archive.
        """
        zip_path = Path(zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        files = self._collect_files()

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for full in files:
                rel = full.relative_to(self.base_dir)
                zf.write(full, rel.as_posix())

        return zip_path

    def write_to_fileobj(self, fp: BinaryIO) -> None:
        """
        Write the ZIP archive into an open file-like object.

        The caller is responsible for opening and closing the file.
        """
        files = self._collect_files()

        with zipfile.ZipFile(fp, "w", zipfile.ZIP_DEFLATED) as zf:
            for full in files:
                rel = full.relative_to(self.base_dir)
                zf.write(full, rel.as_posix())


__all__ = [
    "ZipStreamWriter",
]
