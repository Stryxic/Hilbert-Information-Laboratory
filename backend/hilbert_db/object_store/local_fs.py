"""
Local filesystem object store backend.

Keys map to subdirectories under `config.base_path`. This backend is fully
deterministic and ideal for local development, small to medium corpora,
and CI environments.

Example mapping:
    key = "corpora/abc123/runs/run001/hilbert_export.zip"
    real_path = "<base_path>/corpora/abc123/runs/run001/hilbert_export.zip"
"""

from __future__ import annotations

import os
from typing import BinaryIO, List

from .base import ObjectStore, ObjectStoreConfig


class LocalFSObjectStore(ObjectStore):
    """
    Local filesystem implementation of ObjectStore.

    The key namespace is entirely under config.base_path.
    """

    def __init__(self, config: ObjectStoreConfig):
        self.config = config
        os.makedirs(self.config.base_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Key resolution
    # ------------------------------------------------------------------

    def _resolve(self, key: str) -> str:
        """
        Translate logical key into a physical filesystem path under base_path.

        Ensures:
          - no leading slash
          - no Windows backslashes
          - no path traversal
        """
        key = key.strip("/").replace("\\", "/")
        path = os.path.abspath(os.path.join(self.config.base_path, key))

        base = os.path.abspath(self.config.base_path)
        if not path.startswith(base + os.sep):
            raise ValueError(f"Suspicious key outside root: {key}")

        return path

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def exists(self, key: str) -> bool:
        return os.path.exists(self._resolve(key))

    def save_bytes(self, key: str, data: bytes) -> None:
        if self.config.read_only:
            raise PermissionError("ObjectStore is in read-only mode")

        path = self._resolve(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def open(self, key: str) -> BinaryIO:
        path = self._resolve(key)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Object not found: {key}")
        return open(path, "rb")

    def list(self, prefix: str) -> List[str]:
        prefix = prefix.strip("/")
        resolved_prefix = self._resolve(prefix)

        if not os.path.exists(resolved_prefix):
            return []

        base = os.path.abspath(self.config.base_path)
        result: List[str] = []

        for dirpath, _, files in os.walk(resolved_prefix):
            for file in files:
                full_path = os.path.join(dirpath, file)
                rel = os.path.relpath(full_path, base).replace("\\", "/")
                result.append(rel)

        return sorted(result)

    def delete(self, key: str) -> None:
        if self.config.read_only:
            raise PermissionError("ObjectStore is in read-only mode")

        path = self._resolve(key)
        if os.path.exists(path):
            os.remove(path)
