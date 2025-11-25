"""
Base object store interface for HilbertDB.

Object stores persist large immutable artifacts produced by the pipeline:
    - exports
    - LSA fields
    - CSVs
    - graphs
    - PNGs
    - compound descriptors

Concrete implementations:
    - LocalFSObjectStore (local filesystem)
    - S3 / MinIO (future)
    - GCS, Azure Blob (future)
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Optional, BinaryIO
try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol
# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

@dataclass
class ObjectStoreConfig:
    """
    Configuration for an object store backend.

    Parameters
    ----------
    base_path : str
        Root directory or cloud bucket.
    bucket : Optional[str]
        For S3-like stores (unused in local FS).
    read_only : bool
        If True, write operations should raise PermissionError.
    """
    base_path: str
    bucket: Optional[str] = None
    read_only: bool = False


# ----------------------------------------------------------------------
# Protocol (interface)
# ----------------------------------------------------------------------

class ObjectStore(Protocol):
    """
    Abstract interface used by HilbertDB for storing artifacts.

    Logical keys such as:
        "corpora/<hash>/runs/<run_id>/hilbert_export.zip"
    map to real storage paths or cloud keys.
    """

    config: ObjectStoreConfig

    def exists(self, key: str) -> bool:
        """Return True if object exists."""
        raise NotImplementedError

    def save_bytes(self, key: str, data: bytes) -> None:
        """Persist a binary blob to the object store."""
        raise NotImplementedError

    def open(self, key: str) -> BinaryIO:
        """Open an artifact as a readable binary stream."""
        raise NotImplementedError

    def list(self, prefix: str) -> list[str]:
        """List all objects under a key prefix."""
        raise NotImplementedError

    def delete(self, key: str) -> None:
        """Delete object if not read-only."""
        raise NotImplementedError
