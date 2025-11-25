"""
ExportWriter - helper for assembling export directories.

This layer is intentionally light weight. It provides small utilities
to:

    - copy existing artifacts into a structured export directory
    - write JSON files with deterministic formatting
    - keep the export layout explicit and testable
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union


@dataclass
class ExportWriter:
    """
    Helper around an export directory on disk.

    Parameters
    ----------
    root_dir :
        Root directory for the export contents. It will be created
        if it does not exist.
    """

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def add_file(self, src: Union[str, Path], dest_rel: str) -> Path:
        """
        Copy a file into the export directory at the given relative path.

        Returns the destination path.
        """
        src_path = Path(src)
        dest_path = self.root_dir / dest_rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)
        return dest_path

    def write_json(self, dest_rel: str, data: Any) -> Path:
        """
        Write a JSON file at the given relative path with deterministic
        formatting.
        """
        dest_path = self.root_dir / dest_rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with dest_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return dest_path

    def exists(self, dest_rel: str) -> bool:
        """
        Check if a file exists at the given relative path inside the
        export directory.
        """
        return (self.root_dir / dest_rel).exists()


__all__ = [
    "ExportWriter",
]
