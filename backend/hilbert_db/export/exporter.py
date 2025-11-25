"""
Core export logic for HilbertDB.

The pipeline (hilbert_pipeline.hilbert_export.run_full_export) already
produces:

    - hilbert_manifest.json
    - a deterministic ZIP archive with the run artifacts

This module provides a layer that:

    - loads or constructs an ExportManifest
    - discovers the existing ZIP archive if present
    - if needed, creates a new ZIP deterministically from results_dir

It does not change the pipeline's export behavior. It simply provides
a DB-friendly interface for working with those exports.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Union

from .manifest import ExportManifest, load_manifest, save_manifest
from .zip_stream import ZipStreamWriter


def _discover_zip(results_dir: Path, manifest: Optional[ExportManifest]) -> Optional[Path]:
    """
    Try to discover an existing ZIP archive for the run.

    Strategy:
        1) Look for file name hints in the manifest (export_zip, export_path).
        2) If not found, pick the first *.zip file in results_dir.
    """
    # 1) Hints from manifest
    if manifest is not None:
        for key in ("export_zip", "export_path", "export_file"):
            name = manifest.data.get(key)
            if isinstance(name, str):
                candidate = results_dir / name
                if candidate.exists():
                    return candidate

    # 2) Fallback: first zip in directory
    zips = sorted(results_dir.glob("*.zip"))
    if zips:
        return zips[0]

    return None


def _build_manifest_from_run(results_dir: Path) -> ExportManifest:
    """
    Construct a minimal manifest from hilbert_run.json if available.
    """
    run_json = results_dir / "hilbert_run.json"
    data = {}

    if run_json.exists():
        try:
            with run_json.open("r", encoding="utf-8") as f:
                run_data = json.load(f)
            if isinstance(run_data, dict):
                data["run_id"] = run_data.get("run_id")
                data["orchestrator_version"] = run_data.get("orchestrator_version")
                data["settings"] = run_data.get("settings")
        except Exception:
            # best effort
            pass

    manifest = ExportManifest(data=data)
    manifest.ensure_created_at()
    return manifest


def build_export_for_results_dir(
    results_dir: Union[str, Path],
    run_id: Optional[str] = None,
) -> Tuple[ExportManifest, Path]:
    """
    Ensure a hilbert-compatible export manifest and ZIP exist for
    a given results directory.

    Returns:
        (manifest, zip_path)

    Behavior:
        - If hilbert_manifest.json exists, it is loaded and used.
        - Otherwise, a minimal manifest is derived from hilbert_run.json.
        - If a ZIP archive can be discovered, it is reused.
        - If not, a new ZIP is created from the contents of results_dir,
          ignoring existing ZIP files.
    """
    results_dir = Path(results_dir).resolve()
    manifest_path = results_dir / "hilbert_manifest.json"

    manifest = load_manifest(manifest_path)
    if manifest is None:
        manifest = _build_manifest_from_run(results_dir)

    # Ensure run_id is set consistently
    if run_id is not None:
        manifest.data["run_id"] = run_id
    if not manifest.run_id:
        # derive naive default if nothing available
        manifest.data["run_id"] = run_id or results_dir.name

    # Ensure created_at
    manifest.ensure_created_at()

    # Discover or create ZIP
    zip_path = _discover_zip(results_dir, manifest)
    if zip_path is None:
        default_name = f"{manifest.run_id or 'hilbert_run'}.zip"
        zip_path = results_dir / default_name
        writer = ZipStreamWriter(results_dir, ignore_suffixes={".zip"})
        writer.write_to_path(zip_path)
        # record file name hint in manifest
        manifest.data.setdefault("export_zip", default_name)

    # Save manifest back to disk with deterministic formatting
    save_manifest(manifest, manifest_path)

    return manifest, zip_path


__all__ = [
    "build_export_for_results_dir",
]
