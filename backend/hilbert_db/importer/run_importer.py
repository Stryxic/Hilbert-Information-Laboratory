"""
Run importer - rehydrate deterministic Hilbert exports.

Responsible for:
    - extracting export ZIPs into the CacheManager
    - validating manifests
    - returning ImportedRun descriptors the backend uses to read artifacts
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, IO, Optional

from .cache import CacheManager
from .unzipper import ZipUnpacker


MANIFEST_FILENAME = "hilbert_manifest.json"


@dataclass
class ImportedRun:
    """
    Lightweight descriptor for an imported run.
    """

    run_id: str
    cache_dir: str
    manifest: Dict[str, Any]


class RunImporter:
    """
    High-level importer for deterministic Hilbert exports.

    Used by:
        - HilbertDB.load_imported_run
        - Developer tools
        - Backend API handlers
    """

    def __init__(
        self,
        cache: Optional[CacheManager] = None,
        unzipper: Optional[ZipUnpacker] = None,
    ) -> None:
        self.cache = cache or CacheManager()
        self.unzipper = unzipper or ZipUnpacker()

    # ------------------------------------------------------------------
    # Public import API
    # ------------------------------------------------------------------

    def import_from_zip_path(
        self,
        zip_path: str,
        run_id: Optional[str] = None,
    ) -> ImportedRun:
        """
        Import a run from a ZIP already on disk.

        Extracts twice:
            1. into a temp dir to read manifest + derive run_id
            2. into the final cache dir for that run
        """
        # Step 1: Inspect manifest
        with tempfile.TemporaryDirectory(prefix="hilbert_import_tmp_") as tmpdir:
            self.unzipper.extract(zip_path, tmpdir)
            manifest_path = self._find_manifest(tmpdir)
            manifest = self._load_manifest(manifest_path)

            manifest_run_id = manifest.get("run_id")
            effective_run_id = run_id or str(manifest_run_id)
            if not effective_run_id:
                raise ValueError("Export manifest missing 'run_id'.")

        # Step 2: Extract into cache
        cache_dir = self.cache.get_run_cache_dir(effective_run_id)
        self.unzipper.extract(zip_path, cache_dir)

        return ImportedRun(
            run_id=effective_run_id,
            cache_dir=cache_dir,
            manifest=manifest,
        )

    def import_from_fileobj(
        self,
        fileobj: IO[bytes],
        run_id: Optional[str] = None,
    ) -> ImportedRun:
        """
        Import a ZIP provided as a file-like object.
        """
        with tempfile.NamedTemporaryFile(prefix="hilbert_import_", suffix=".zip") as tmp:
            # Stream to the temp file
            while True:
                chunk = fileobj.read(8192)
                if not chunk:
                    break
                tmp.write(chunk)
            tmp.flush()

            # Delegate to import_from_zip_path
            return self.import_from_zip_path(tmp.name, run_id=run_id)

    def import_if_not_cached(
        self,
        zip_path: str,
        run_id: Optional[str] = None,
    ) -> ImportedRun:
        """
        Import unless a cached run already exists.
        """
        # Step 1: Derive run_id if needed
        if run_id is None:
            with tempfile.TemporaryDirectory(prefix="hilbert_import_tmp_") as tmpdir:
                self.unzipper.extract(zip_path, tmpdir)
                manifest_path = self._find_manifest(tmpdir)
                manifest = self._load_manifest(manifest_path)
                run_id = str(manifest.get("run_id") or "")
                if not run_id:
                    raise ValueError("Export manifest missing 'run_id'.")
        else:
            manifest = None

        # Step 2: Cache hit
        if self.cache.has_cached_run(run_id):
            cache_dir = self.cache.get_run_cache_dir(run_id)
            manifest_path = os.path.join(cache_dir, MANIFEST_FILENAME)
            manifest = manifest or self._load_manifest(manifest_path)
            return ImportedRun(
                run_id=run_id,
                cache_dir=cache_dir,
                manifest=manifest,
            )

        # Step 3: Cache miss → full import
        return self.import_from_zip_path(zip_path, run_id=run_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_manifest(self, root_dir: str) -> str:
        for dirpath, _, filenames in os.walk(root_dir):
            if MANIFEST_FILENAME in filenames:
                return os.path.join(dirpath, MANIFEST_FILENAME)
        raise FileNotFoundError(f"{MANIFEST_FILENAME} not found in export.")

    def _load_manifest(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Export manifest must be a JSON object.")
        return data
