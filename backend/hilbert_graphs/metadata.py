"""
Metadata and index writing for graph snapshots.

This module produces:
    - Per-snapshot metadata JSON files (graph_XXX.meta.json)
    - A global snapshot index (graph_snapshots_index.json)

It is intentionally simple, deterministic, and versioned so downstream
tools (dashboard, API clients, notebooks) can rely on consistent fields.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Any, List, Optional

from .analytics import GraphStats


# ============================================================================ #
# Utility: robust JSON writing
# ============================================================================ #

def _safe_write_json(path: str, payload: Dict[str, Any], emit=None) -> None:
    """
    Safely write JSON with directory creation and logging.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        if emit:
            emit("log", {"message": f"[metadata] wrote {path}"})
    except Exception as e:
        if emit:
            emit("log", {
                "message": f"[metadata] failed to write {path}: {e}",
                "error": str(e)
            })


# ============================================================================ #
# Snapshot metadata writer
# ============================================================================ #

def write_snapshot_metadata(
    results_dir: str,
    snapshot_name: str,
    snapshot_name_3d: Optional[str],
    stats: GraphStats,
    pct: Optional[float],
    top_nodes: List[str],
    layout_info: Optional[Dict[str, Any]] = None,
    emit=None,
) -> Dict[str, Any]:
    """
    Write metadata describing a single graph snapshot.

    Parameters
    ----------
    results_dir : str
        Path to Hilbert run directory.
    snapshot_name : str
        2D PNG file name, e.g. "graph_10pct.png".
    snapshot_name_3d : str | None
        3D PNG name if produced.
    stats : GraphStats
        Basic graph statistics.
    pct : float | None
        Percentage used by the snapshot; None for structural snapshots.
    top_nodes : list[str]
        The most important nodes in the snapshot.
    layout_info : dict | None
        Optional layout metadata (layout mode, seed, parameters, etc).
    emit : callable
        Optional Hilbert emitter.

    Returns
    -------
    dict
        The metadata dictionary (also persisted to disk).
    """

    meta = {
        "version": "hilbert.graphmeta.v1",
        "timestamp": time.time(),

        "snapshot": snapshot_name,
        "snapshot_3d": snapshot_name_3d,

        "pct": pct,

        "stats": {
            "n_nodes": stats.n_nodes,
            "n_edges": stats.n_edges,
            "density": stats.density,
            "avg_degree": stats.avg_degree,
            "transitivity": stats.transitivity,
        },

        "top_nodes": top_nodes,
        "layout": layout_info or {},
    }

    out_path = os.path.join(
        results_dir,
        snapshot_name.replace(".png", ".meta.json")
    )
    _safe_write_json(out_path, meta, emit=emit)

    return meta


# ============================================================================ #
# Global index writer
# ============================================================================ #

def write_global_index(
    results_dir: str,
    global_stats: GraphStats,
    snapshot_entries: List[Dict[str, Any]],
    config_info: Dict[str, Any],
    emit=None,
) -> None:
    """
    Write the global graph snapshot index.

    This provides a single entry point for dashboard integrations.
    """

    payload = {
        "version": "hilbert.snapshot.index.v1",
        "timestamp": time.time(),

        "global_stats": {
            "n_nodes": global_stats.n_nodes,
            "n_edges": global_stats.n_edges,
            "density": global_stats.density,
            "avg_degree": global_stats.avg_degree,
            "transitivity": global_stats.transitivity,
        },

        "config": config_info or {},
        "snapshots": snapshot_entries,
    }

    index_path = os.path.join(results_dir, "graph_snapshots_index.json")
    _safe_write_json(index_path, payload, emit=emit)


# ============================================================================ #
# Metadata merge helper
# ============================================================================ #

def merge_snapshot_metadata(
    base: Dict[str, Any],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge two metadata dictionaries in a stable, predictable way.

    Rules:
        - Keys in `extra` override keys in `base`
        - Lists concatenate uniquely (order preserved)
        - Dicts merge shallowly
        - All other values overwrite directly

    This allows 2D and 3D renderers to write metadata independently,
    then merge their results into a unified record.
    """

    out = dict(base)

    for key, val in extra.items():
        if key not in out:
            out[key] = val
            continue

        # merge lists (unique)
        if isinstance(out[key], list) and isinstance(val, list):
            merged = list(dict.fromkeys(out[key] + val))
            out[key] = merged
            continue

        # merge nested dictionaries
        if isinstance(out[key], dict) and isinstance(val, dict):
            merged = dict(out[key])
            merged.update(val)
            out[key] = merged
            continue

        # default: overwrite
        out[key] = val

    return out
