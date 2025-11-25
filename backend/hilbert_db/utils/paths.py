"""
Key-building utilities for object-store and registry paths.

These helpers produce deterministic, normalized keys so that every corpus,
run, artifact, and graph snapshot is stored under a predictable namespace,
independent of backend storage (local filesystem, S3, MinIO, etc.).

All keys are POSIX-style ("a/b/c") with no leading slashes.
"""

from __future__ import annotations

from pathlib import Path


# ----------------------------------------------------------------------
# Corpus
# ----------------------------------------------------------------------

def build_corpus_key(corpus_hash: str) -> str:
    """
    Deterministic namespace for a corpus.

    Example:
        corpus_hash = "abc123"
        -> "corpora/abc123/"
    """
    corpus_hash = corpus_hash.strip("/")
    return f"corpora/{corpus_hash}/"


# ----------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------

def build_run_key(run_id: str) -> str:
    """
    Deterministic namespace for a Hilbert run.

    Example:
        run_id = "run001"
        -> "runs/run001/"
    """
    run_id = run_id.strip("/")
    return f"runs/{run_id}/"


# ----------------------------------------------------------------------
# Artifacts
# ----------------------------------------------------------------------

def build_artifact_key(run_id: str, name: str) -> str:
    """
    Deterministic key for a single artifact file inside a run.

    Example:
        run_id = "run001"
        name = "hilbert_elements.csv"
        -> "runs/run001/artifacts/hilbert_elements.csv"
    """
    run_id = run_id.strip("/")
    name = name.strip("/")
    return f"runs/{run_id}/artifacts/{name}"


# ----------------------------------------------------------------------
# Graph snapshots
# ----------------------------------------------------------------------

def build_graph_key(run_id: str, depth: str) -> str:
    """
    Deterministic key for graph snapshot layers (2D, 3D, depth slices).

    `depth` may contain characters like "%" which are normalized.

    Example:
        depth = "2d"
        -> "runs/<id>/graphs/2d.json"
    """
    run_id = run_id.strip("/")
    depth = depth.strip().replace("%", "pct")
    return f"runs/{run_id}/graphs/{depth}.json"


# ----------------------------------------------------------------------
# Local FS mapping
# ----------------------------------------------------------------------

def local_path(root: Path, key: str) -> Path:
    """
    Convert an object-store key into a local filesystem path.

    Useful for:
        - LocalFSObjectStore
        - importer cache
        - rehydrated run directories

    Example:
        root = Path("/tmp/hilbert/storage")
        key = "runs/run001/artifacts/lsa.json"
        -> /tmp/hilbert/storage/runs/run001/artifacts/lsa.json
    """
    key = key.strip("/")
    return root / key


__all__ = [
    "build_corpus_key",
    "build_run_key",
    "build_artifact_key",
    "build_graph_key",
    "local_path",
]
