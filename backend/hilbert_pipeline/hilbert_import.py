# =============================================================================
# hilbert_pipeline/hilbert_import.py - Deterministic Run Import / Restore
# =============================================================================
"""
Import / restore a Hilbert pipeline run from a deterministic export.

This module is the counterpart to `hilbert_export` and focuses on:

  - Loading a `hilbert_manifest.json` that describes a run.
  - Restoring a run directory from an exported ZIP archive created by
    `hilbert_export.export_zip`.
  - Verifying that the restored files match the manifest (size + SHA256).
  - Providing small helper functions to repopulate core tables and graphs
    in memory (DataFrames, NetworkX graphs) for use by the API layer.

It does *not* re-run any pipeline stages or recompute metrics. It simply
reconstructs the on-disk state of a run and gives you easy access to the
tables and graph needed to drive the frontend.

Typical usage:

    from hilbert_pipeline.hilbert_import import (
        restore_run_from_zip,
        verify_run,
        load_elements_table,
        load_graph,
    )

    result = restore_run_from_zip("my_run.zip", "./results/restored_run")
    ok, report = verify_run("./results/restored_run")

    elements_df = load_elements_table("./results/restored_run")
    G = load_graph("./results/restored_run")
"""

from __future__ import annotations

import json
import os
import zipfile
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -------------------------------------------------------------------------#
# Orchestrator compatibility emit
# -------------------------------------------------------------------------#
try:
    # Reuse pipeline default emitter if available
    from . import DEFAULT_EMIT  # type: ignore
except Exception:  # pragma: no cover - fallback for standalone usage
    DEFAULT_EMIT = lambda *_a, **_k: None  # type: ignore


# -------------------------------------------------------------------------#
# Shared helpers
# -------------------------------------------------------------------------#
def _read_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """
    Compute a SHA256 hash of a file. Used to verify artifacts against
    entries in the manifest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# -------------------------------------------------------------------------#
# Manifest loading and basic accessors
# -------------------------------------------------------------------------#
def load_manifest(path_or_dir: str) -> Dict[str, Any]:
    """
    Load a `hilbert_manifest.json` from either:

      - a directory containing `hilbert_manifest.json`, or
      - a direct path to `hilbert_manifest.json`.

    Returns an empty dict if nothing could be loaded.
    """
    if os.path.isdir(path_or_dir):
        path = os.path.join(path_or_dir, "hilbert_manifest.json")
    else:
        path = path_or_dir

    data = _read_json(path)
    if isinstance(data, dict):
        return data
    return {}


def get_artifact_list(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract the list of artifact records from a loaded manifest.

    Each record typically has keys:
      - path (relative to run_dir)
      - ext
      - size_bytes
      - sha256
    """
    return manifest.get("artifacts", {}).get("files", []) or []


def get_stable_paths(manifest: Dict[str, Any]) -> List[str]:
    """
    Extract the list of "stable" artifact paths (e.g. stable_* files)
    from a loaded manifest.
    """
    paths = manifest.get("artifacts", {}).get("stable_paths", []) or []
    return [str(p) for p in paths]


# -------------------------------------------------------------------------#
# Restore from ZIP
# -------------------------------------------------------------------------#
def restore_run_from_zip(
    zip_path: str,
    out_dir: str,
    emit=DEFAULT_EMIT,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Restore a Hilbert run from a ZIP archive created by `hilbert_export`.

    Parameters
    ----------
    zip_path:
        Path to the ZIP file (e.g. 'hilbert_run.zip').
    out_dir:
        Directory into which the run should be restored. All files inside
        the top-level directory in the ZIP will be extracted into this path.
        If the directory does not exist, it is created.
    overwrite:
        If False (default), an error is raised if the target directory is
        non-empty. If True, existing files may be overwritten.

    Returns
    -------
    manifest : dict
        The loaded manifest for the restored run.

    Notes
    -----
    The ZIP layout produced by `export_zip` is:

        <run_name>/
            hilbert_manifest.json
            ... other CSV/JSON/TXT files ...

    This importer strips the top-level `<run_name>/` prefix on extraction
    so that the resulting on-disk structure matches what the orchestrator
    and frontend expect.
    """
    emit("import", {"event": "start", "zip_path": zip_path})

    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    os.makedirs(out_dir, exist_ok=True)
    # Basic safety: if not overwriting, refuse to import into a non-empty dir.
    if not overwrite:
        existing = [
            f
            for f in os.listdir(out_dir)
            if f not in (".", "..") and not f.startswith(".DS_Store")
        ]
        if existing:
            raise RuntimeError(
                f"Target directory '{out_dir}' is not empty and overwrite=False."
            )

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if not names:
            raise RuntimeError(f"ZIP archive appears to be empty: {zip_path}")

        # Infer common top-level prefix (run_name/)
        # e.g. 'hilbert_run/hilbert_manifest.json' -> 'hilbert_run'
        top_level_prefix = None
        for name in names:
            parts = name.split("/")
            if parts and parts[0]:
                top_level_prefix = parts[0]
                break

        for member in names:
            # Skip directories
            if member.endswith("/"):
                continue
            # Strip the top-level prefix if present
            rel_parts = member.split("/")
            if top_level_prefix and rel_parts[0] == top_level_prefix:
                rel_parts = rel_parts[1:]
            rel_path = "/".join(rel_parts).strip("/")
            if not rel_path:
                # Nothing left after stripping, skip
                continue

            dest_path = os.path.join(out_dir, rel_path)
            dest_dir = os.path.dirname(dest_path)
            os.makedirs(dest_dir, exist_ok=True)
            with zf.open(member, "r") as src, open(dest_path, "wb") as dst:
                dst.write(src.read())

    manifest = load_manifest(out_dir)
    if not manifest:
        emit("import", {"event": "warn", "msg": "No hilbert_manifest.json found"})
    else:
        emit("import", {"event": "manifest_loaded", "run": manifest.get("run")})

    emit("import", {"event": "end", "out_dir": out_dir})
    return manifest


# -------------------------------------------------------------------------#
# Verification against manifest
# -------------------------------------------------------------------------#
def verify_run(out_dir: str, manifest: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify that files in `out_dir` match the checksums and sizes recorded
    in the given manifest (or the manifest found in `out_dir`).

    Parameters
    ----------
    out_dir:
        Run directory on disk (restored or original).
    manifest:
        Optional manifest dict. If None, `hilbert_manifest.json` is loaded
        from `out_dir`.

    Returns
    -------
    ok : bool
        True if all manifest-listed files exist and match SHA256 and size.
    report : dict
        Detailed report with keys:
            - missing: list of paths
            - size_mismatch: list of {path, expected, actual}
            - hash_mismatch: list of {path, expected, actual}
            - checked: number of files checked
    """
    if manifest is None:
        manifest = load_manifest(out_dir)

    artifacts = get_artifact_list(manifest)
    missing: List[str] = []
    size_mismatch: List[Dict[str, Any]] = []
    hash_mismatch: List[Dict[str, Any]] = []

    for row in artifacts:
        rel = row.get("path")
        if not rel:
            continue
        expected_size = int(row.get("size_bytes", -1))
        expected_hash = row.get("sha256", "")

        fpath = os.path.join(out_dir, rel)
        if not os.path.isfile(fpath):
            missing.append(rel)
            continue

        actual_size = os.path.getsize(fpath)
        if expected_size >= 0 and actual_size != expected_size:
            size_mismatch.append(
                {
                    "path": rel,
                    "expected": expected_size,
                    "actual": actual_size,
                }
            )

        if expected_hash:
            actual_hash = _sha256(fpath)
            if actual_hash != expected_hash:
                hash_mismatch.append(
                    {
                        "path": rel,
                        "expected": expected_hash,
                        "actual": actual_hash,
                    }
                )

    ok = not missing and not size_mismatch and not hash_mismatch
    report: Dict[str, Any] = {
        "checked": len(artifacts),
        "missing": missing,
        "size_mismatch": size_mismatch,
        "hash_mismatch": hash_mismatch,
    }
    return ok, report


# -------------------------------------------------------------------------#
# Convenience loaders to repopulate tables and graphs
# -------------------------------------------------------------------------#
def load_elements_table(out_dir: str) -> pd.DataFrame:
    """
    Load the main elements table (`hilbert_elements.csv`).

    Returns
    -------
    df : pandas.DataFrame
        Empty DataFrame if file is missing or unreadable.
    """
    path = os.path.join(out_dir, "hilbert_elements.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_element_metrics(out_dir: str) -> pd.DataFrame:
    """
    Load `element_metrics.csv` if available.
    """
    path = os.path.join(out_dir, "element_metrics.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_signal_stability(out_dir: str) -> pd.DataFrame:
    """
    Load `signal_stability.csv` if available.
    """
    path = os.path.join(out_dir, "signal_stability.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_compounds(out_dir: str) -> Dict[str, Any]:
    """
    Load `informational_compounds.json` if available.

    Returns a dict keyed by compound_id, or {} if missing.
    """
    path = os.path.join(out_dir, "informational_compounds.json")
    data = _read_json(path)
    if isinstance(data, dict):
        return data
    return {}


def load_span_element_fusion(out_dir: str) -> pd.DataFrame:
    """
    Load `span_element_fusion.csv` if available.
    """
    path = os.path.join(out_dir, "span_element_fusion.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_graph(out_dir: str):
    """
    Load the element co-occurrence graph from `edges.csv` as a NetworkX Graph.

    Returns
    -------
    G : networkx.Graph or None
        The constructed graph, or None if edges cannot be read.

    Notes
    -----
    The graph construction is identical to what `hilbert_export` uses
    for summarising graph metrics: an undirected simple graph with
    nodes representing elements and edges representing co-occurrences.
    """
    edges_path = os.path.join(out_dir, "edges.csv")
    if not os.path.exists(edges_path):
        return None

    try:
        import networkx as nx  # type: ignore
    except Exception:
        return None

    try:
        df = pd.read_csv(edges_path)
    except Exception:
        return None

    if "source" not in df.columns or "target" not in df.columns:
        return None

    G = nx.Graph()
    for s, t in zip(df["source"].astype(str), df["target"].astype(str)):
        if s and t and s != t:
            G.add_edge(s, t)

    return G


def load_stable_core(out_dir: str) -> pd.DataFrame:
    """
    Load a "stable epistemic core" file if present.

    This is a convenience wrapper that looks for a variety of names that
    might be produced by a downstream condenser, such as:

        - final_knowledge.json
        - stable_information.json
        - stable_elements.json

    and returns the first one it finds as a DataFrame.
    """
    candidates = [
        "final_knowledge.json",
        "stable_information.json",
        "stable_elements.json",
    ]
    for name in candidates:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            try:
                data = _read_json(path)
                if isinstance(data, list):
                    return pd.DataFrame(data)
            except Exception:
                continue
    return pd.DataFrame()


# -------------------------------------------------------------------------#
# Orchestrator-facing helper for imports (optional)
# -------------------------------------------------------------------------#
def import_run(
    zip_path: str,
    out_dir: str,
    emit=DEFAULT_EMIT,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    High-level helper to:

      1. Restore a run from a ZIP archive into `out_dir`.
      2. Verify the restored files against the manifest.
      3. Return a small structured report.

    Parameters
    ----------
    zip_path:
        Path to the ZIP archive created by `hilbert_export.export_zip`.
    out_dir:
        Directory to restore into.
    overwrite:
        Passed through to `restore_run_from_zip`.

    Returns
    -------
    report : dict
        Keys:
            - out_dir
            - manifest
            - verification_ok
            - verification_report
    """
    manifest = restore_run_from_zip(zip_path, out_dir, emit=emit, overwrite=overwrite)
    ok, verification = verify_run(out_dir, manifest=manifest)

    report = {
        "out_dir": out_dir,
        "manifest": manifest,
        "verification_ok": ok,
        "verification_report": verification,
    }
    return report


__all__ = [
    "load_manifest",
    "get_artifact_list",
    "get_stable_paths",
    "restore_run_from_zip",
    "verify_run",
    "load_elements_table",
    "load_element_metrics",
    "load_signal_stability",
    "load_compounds",
    "load_span_element_fusion",
    "load_graph",
    "load_stable_core",
    "import_run",
]
