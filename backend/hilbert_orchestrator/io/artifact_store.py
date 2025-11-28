"""
Artifact Store Integration
==========================

This module defines a thin abstraction layer between:

    - Local filesystem artifacts produced by the Hilbert pipeline
    - The HilbertDB object store (S3, filesystem backend, SQLite blob store, etc.)

The orchestrator records every output artifact in ``ctx.artifacts`` and uses
``store_artifact`` to push them into persistent storage.

Goals
-----
• Provide a stable interface for artifact persistence  
• Avoid any backend-specific code inside the orchestrator  
• Ensure artifacts are always namespaced as::

      corpora/<corpus_id>/runs/<run_id>/<artifact_name>

• Cleanly separate *what* an artifact is from *where* it is stored
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

from hilbert_db.core import HilbertDB


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class StoredArtifact:
    """
    Metadata returned after storing a pipeline artifact.

    Attributes
    ----------
    key : str
        Object-store key (unique path inside the backend).
    kind : str
        Artifact category ("lsa", "stability", "export", etc.).
    path : str
        Local filesystem path where the artifact originated.
    size_bytes : int
        Size of the stored artifact.
    """
    key: str
    kind: str
    path: str
    size_bytes: int


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def store_artifact(
    *,
    db: HilbertDB,
    run_id: str,
    corpus_id: int,
    name: str,
    kind: str,
    local_path: str,
    meta: Dict[str, Any] | None = None,
) -> StoredArtifact:
    """
    Store a single artifact in the HilbertDB object store.

    Parameters
    ----------
    db : HilbertDB
        Database instance with an attached object_store.

    run_id : str
        ID of the run the artifact belongs to.

    corpus_id : int
        Corpus ID, used to namespace the artifact path.

    name : str
        The artifact filename (e.g., ``"signal_stability.csv"``).

    kind : str
        Artifact category (``"edges"``, ``"molecules"``, ``"export"``, etc.).

    local_path : str
        Filesystem path to the artifact produced during a pipeline run.

    meta : dict, optional
        Any additional metadata to attach in the DB record.

    Returns
    -------
    StoredArtifact
        A structured record describing the stored artifact.

    Notes
    -----
    - Missing local files do *not* raise exceptions; they produce empty uploads.
    - This is intentional to avoid breaking entire runs when a non-critical
      stage fails to generate a particular artifact.
    """

    meta = meta or {}

    # Namespace: corpora/<corpus_id>/runs/<run_id>/<artifact_name>
    key = f"corpora/{corpus_id}/runs/{run_id}/{name}"

    size = 0
    if os.path.exists(local_path):
        # Read entire file (simple, robust)
        with open(local_path, "rb") as f:
            data = f.read()
        size = len(data)
        db.object_store.save_bytes(key, data)

    # Register metadata in DB
    db.register_artifact(
        run_id=run_id,
        name=name,
        kind=kind,
        key=key,
        meta=meta,
    )

    return StoredArtifact(
        key=key,
        kind=kind,
        path=local_path,
        size_bytes=size,
    )
