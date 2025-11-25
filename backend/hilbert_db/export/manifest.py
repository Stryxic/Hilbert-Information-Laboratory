"""
Export manifest helpers.

This module wraps the hilbert_manifest.json file produced by the
pipeline's deterministic export stage.

Design constraints:

    - Never discard unknown keys from the manifest.
    - Expose a minimal typed interface for common fields
      (run_id, orchestrator_version, created_at).
    - Keep the manifest as a plain JSON object on disk with
      stable formatting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union


JsonDict = Dict[str, Any]


@dataclass
class ExportManifest:
    """
    Thin wrapper around the raw manifest JSON.

    All fields are stored in `data`. The properties below provide
    convenient access to commonly used keys.
    """

    data: JsonDict = field(default_factory=dict)

    # - - - Typed properties - - -

    @property
    def run_id(self) -> Optional[str]:
        v = self.data.get("run_id")
        return str(v) if v is not None else None

    @run_id.setter
    def run_id(self, value: str) -> None:
        self.data["run_id"] = value

    @property
    def orchestrator_version(self) -> Optional[str]:
        v = self.data.get("orchestrator_version")
        return str(v) if v is not None else None

    @orchestrator_version.setter
    def orchestrator_version(self, value: str) -> None:
        self.data["orchestrator_version"] = value

    @property
    def created_at(self) -> Optional[str]:
        v = self.data.get("created_at")
        return str(v) if v is not None else None

    @created_at.setter
    def created_at(self, value: str) -> None:
        self.data["created_at"] = value

    # - - - Convenience methods - - -

    def ensure_created_at(self) -> None:
        """
        Ensure created_at is set to an ISO8601 timestamp if missing.
        """
        if not self.created_at:
            now = datetime.now(timezone.utc).isoformat()
            self.created_at = now

    def to_dict(self) -> JsonDict:
        """
        Return the raw manifest dictionary. Mutations are allowed.
        """
        return self.data

    @classmethod
    def from_dict(cls, d: JsonDict) -> "ExportManifest":
        return cls(data=dict(d))

    # - - - I/O helpers - - -


def load_manifest(path: Union[str, Path]) -> Optional[ExportManifest]:
    """
    Load an ExportManifest from a JSON file.

    Returns None if the file does not exist or is invalid JSON.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return ExportManifest.from_dict(data)
    except Exception:
        return None


def save_manifest(manifest: ExportManifest, path: Union[str, Path]) -> bool:
    """
    Write an ExportManifest to disk with deterministic formatting.

    Returns True on success, False otherwise.
    """
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


__all__ = [
    "ExportManifest",
    "load_manifest",
    "save_manifest",
]
