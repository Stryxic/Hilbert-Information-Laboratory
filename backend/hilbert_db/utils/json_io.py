"""
Safe JSON read/write helpers.

These functions guarantee:
- UTF-8 encoding
- deterministic indentation
- graceful failure on malformed files
- directory creation for writes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def json_read(path: Path) -> Optional[Any]:
    """
    Safely read and parse a JSON file.

    Parameters
    ----------
    path : Path
        Path to a JSON file.

    Returns
    -------
    Optional[Any]
        Parsed object if valid JSON, else None.
    """
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Return None on any failure (permissions, malformed JSON, etc.)
        return None


def json_write(path: Path, data: Any) -> bool:
    """
    Safely write JSON with deterministic formatting.

    Parameters
    ----------
    path : Path
        Output JSON file path.
    data : Any
        JSON-serializable Python structure.

    Returns
    -------
    bool
        True if successfully written, False otherwise.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


__all__ = [
    "json_read",
    "json_write",
]
