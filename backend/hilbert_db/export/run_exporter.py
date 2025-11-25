"""
RunExporter - connects export building with HilbertDB.

This helper is meant to be used by orchestrator integration or
backend services that want to:

    1) ensure a deterministic export ZIP exists for a run
    2) store the ZIP in the configured object store
    3) record the export as an artifact in the DB
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from ..core import HilbertDB
from ..registry.models import ArtifactRecord
from .exporter import build_export_for_results_dir


@dataclass
class RunExporter:
    """
    High-level export orchestrator bound to a HilbertDB instance.
    """

    db: HilbertDB

    def export_and_store_run(
        self,
        run_id: str,
        results_dir: Union[str, Path],
        *,
        artifact_id: Optional[str] = None,
    ) -> ArtifactRecord:
        """
        Build or discover an export ZIP for the given run's results_dir
        and persist it to the object store via HilbertDB.

        Returns the registered ArtifactRecord for the export.
        """
        results_dir = Path(results_dir)
        _, zip_path = build_export_for_results_dir(results_dir, run_id=run_id)

        artifact = self.db.store_run_export(
            run_id=run_id,
            export_zip_path=str(zip_path),
            artifact_id=artifact_id,
        )
        return artifact


__all__ = [
    "RunExporter",
]
