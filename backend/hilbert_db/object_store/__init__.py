"""
Hilbert DB - Object Store package.

Provides:

    - ObjectStoreConfig: backend configuration
    - ObjectStore: protocol describing required interface
    - LocalFSObjectStore: default local filesystem implementation

Future implementations:
    - S3ObjectStore
    - MinIOObjectStore
    - GCSObjectStore
    - AzureBlobObjectStore
"""

from .base import ObjectStoreConfig, ObjectStore
from .local_fs import LocalFSObjectStore

__all__ = [
    "ObjectStore",
    "ObjectStoreConfig",
    "LocalFSObjectStore",
]
