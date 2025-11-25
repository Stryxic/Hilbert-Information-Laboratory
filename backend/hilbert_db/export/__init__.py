"""
hilbert_db.export

Deterministic export helpers for HilbertDB.

This package provides:

    - ExportManifest    : thin wrapper around hilbert_manifest.json
    - load_manifest     : read an export manifest from disk
    - save_manifest     : write an export manifest to disk

    - ZipStreamWriter   : deterministic ZIP builder from a directory

    - ExportWriter      : helper to assemble an export directory

    - build_export_for_results_dir:
          high-level function that ensures a hilbert-compatible
          manifest and ZIP exist for a given results_dir

    - RunExporter       : integrates the export process with
          HilbertDB, object store, and run registry

    - record_run_export / get_run_export_artifact:
          small helpers that connect exports to the DB registries
"""

from .manifest import ExportManifest, load_manifest, save_manifest
from .zip_stream import ZipStreamWriter
from .writer import ExportWriter
from .exporter import build_export_for_results_dir
from .run_exporter import RunExporter
from .registry_integration import record_run_export, get_run_export_artifact

__all__ = [
    # Manifest
    "ExportManifest",
    "load_manifest",
    "save_manifest",

    # ZIP utilities
    "ZipStreamWriter",

    # Export directory helper
    "ExportWriter",

    # High level export builder
    "build_export_for_results_dir",

    # HilbertDB integration
    "RunExporter",
    "record_run_export",
    "get_run_export_artifact",
]
