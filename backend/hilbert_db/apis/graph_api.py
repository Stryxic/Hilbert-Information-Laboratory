"""
Graph API - provides access to graph snapshots, metadata,
node/edge listings, and graph-level metrics.

This is the interface used by the frontend graph viewer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hilbert_db.core import HilbertDB


# ----------------------------------------------------------------------
# Request / Response Models
# ----------------------------------------------------------------------

@dataclass
class GraphRequest:
    run_id: str
    depth: Optional[str] = None    # e.g. "1pct", "10pct", "full"


@dataclass
class GraphResponse:
    run_id: str
    depth: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _load_import_root(db: HilbertDB, run_id: str) -> Path:
    """
    Load (or rehydrate) the imported run into the local cache.
    Returns the path to the run cache directory.
    """
    imported = db.load_imported_run(run_id)
    return Path(imported.cache_dir)


def _load_graph_index(root: Path) -> Any:
    """
    Load optional graph_snapshots_index.json if present.
    Returns parsed JSON or None.
    """
    idx_path = root / "graph_snapshots_index.json"
    if not idx_path.exists():
        return None
    with idx_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_depth_and_path(root: Path, requested_depth: Optional[str]) -> Tuple[str, Path]:
    """
    Resolve the depth label and actual file path for the graph snapshot.

    Handles multiple legacy index formats gracefully.
    """
    graphs_root = root / "graphs"
    idx_data = _load_graph_index(root)

    # If user explicitly requested a depth, try direct file first
    if requested_depth:
        depth = requested_depth.strip()
        direct = graphs_root / f"{depth}.json"
        if direct.exists():
            return depth, direct

    # No direct match — inspect index file if present
    depth: Optional[str] = None
    path: Optional[Path] = None

    # Index format: list of dicts
    if isinstance(idx_data, list):
        for entry in idx_data:
            if not isinstance(entry, dict):
                continue
            candidate_depth = str(entry.get("depth") or entry.get("id") or "")
            if requested_depth and candidate_depth != requested_depth:
                continue
            depth = candidate_depth or requested_depth
            candidate_path = entry.get("path") or entry.get("file")
            if candidate_path:
                path = (root / candidate_path).resolve()
            break

    # Index format: dict with lists of depths + snapshots
    if isinstance(idx_data, dict) and depth is None:
        depth_list = (
            idx_data.get("available_depths")
            or idx_data.get("depths")
            or []
        )
        if depth_list:
            if requested_depth in depth_list:
                depth = requested_depth
            else:
                depth = str(depth_list[0])

        # Resolve file path for depth
        snaps = idx_data.get("snapshots") or idx_data.get("maps") or []
        for s in snaps:
            if not isinstance(s, dict):
                continue
            if str(s.get("depth") or s.get("id")) == depth:
                p = s.get("path") or s.get("file")
                if p:
                    path = (root / p).resolve()
                break

    # Fall back to plain graphs/<depth>.json
    if depth is None:
        depth = requested_depth or "full"
    if path is None:
        path = graphs_root / f"{depth}.json"

    return depth, path


def _load_graph(root: Path, depth: Optional[str]) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load a graph snapshot JSON file and return structured components.
    """
    depth_resolved, path = _resolve_depth_and_path(root, depth)

    if not path.exists():
        raise KeyError(f"Graph snapshot not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Graph snapshot at {path} is not a JSON object")

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    metadata = data.get("metadata", {})

    return depth_resolved, nodes, edges, metadata


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def get_graph_snapshot(db: HilbertDB, req: GraphRequest) -> GraphResponse:
    """
    Load a graph snapshot (2D or 3D) for a given run.

    Raises:
        KeyError if run missing or snapshot missing.
    """
    root = _load_import_root(db, req.run_id)
    depth, nodes, edges, metadata = _load_graph(root, req.depth)

    return GraphResponse(
        run_id=req.run_id,
        depth=depth,
        nodes=nodes,
        edges=edges,
        metadata=metadata,
    )


def list_available_graphs(db: HilbertDB, run_id: str) -> List[str]:
    """
    Return list of available graph snapshot depth labels.

    Example:
        ["1pct", "5pct", "10pct", "full"]
    """
    root = _load_import_root(db, run_id)
    idx_data = _load_graph_index(root)
    depths: List[str] = []

    # Index format: list of dicts
    if isinstance(idx_data, list):
        for entry in idx_data:
            if not isinstance(entry, dict):
                continue
            d = entry.get("depth") or entry.get("id")
            if d:
                depths.append(str(d))

    # Index format: dict
    if isinstance(idx_data, dict):
        listed = idx_data.get("available_depths") or idx_data.get("depths")
        if isinstance(listed, list):
            depths.extend(str(d) for d in listed)

    # Fallback: inspect the filesystem
    if not depths:
        graphs_root = root / "graphs"
        if graphs_root.exists():
            for p in graphs_root.glob("*.json"):
                depths.append(p.stem)

    return sorted(set(depths))


__all__ = [
    "GraphRequest",
    "GraphResponse",
    "get_graph_snapshot",
    "list_available_graphs",
]
