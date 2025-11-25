"""
Helpers for integrating exports with HilbertDB registries.

These functions are small convenience wrappers that allow callers to
work with exports through the hilbert_db.export namespace instead of
reaching directly into hilbert_db.core or the registry internals.
"""

from __future__ import annotations

from typing import Optional

from ..core import HilbertDB
from ..registry.models import ArtifactRecord


def record_run_export(
    db: HilbertDB,
    run_id: str,
    export_zip_path: str,
    *,
    artifact_id: Optional[str] = None,
) -> ArtifactRecord:
    """
    Persist a run export ZIP into the object store and register it
    as an artifact.

    This is a thin wrapper around HilbertDB.store_run_export.
    """
    return db.store_run_export(
        run_id=run_id,
        export_zip_path=export_zip_path,
        artifact_id=artifact_id,
    )


def get_run_export_artifact(
    db: HilbertDB,
    run_id: str,
) -> Optional[ArtifactRecord]:
    """
    Return the first export artifact for a run, if any.

    Convention:
        - exports are registered with kind="export"
        - there is usually at most one per run
    """
    artifacts = db.list_artifacts_for_run(run_id, kind="export")
    return artifacts[0] if artifacts else None


__all__ = [
    "record_run_export",
    "get_run_export_artifact",
]
