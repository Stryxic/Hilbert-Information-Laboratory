"""
Modular scientific metadata writers for Hilbert graph snapshots.

New in v2 (Option 2 architecture):

We now produce FOUR metadata outputs:

1. graph_metadata_core.json
      - stable top-level run metadata (version, seeds, upstream LSA config, pruning info)

2. graph_layout.json
      - 2D/3D layout parameters, community latitudes, radius bounds, timings

3. graph_analytics.json
      - cluster sizes, component sizes, compound statistics, root summaries

4. graph_diagnostics.json
      - paths to diagnostic plots (degree hist, component hist, community panels, etc.)

Plus the existing:
5. graph_snapshots_index.json
      - lightweight index referencing snapshot metadata & pointers to the above files.

Snapshot-level .meta.json files are unchanged.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union

from .analytics import GraphStats

META_VERSION = "hilbert.graphmeta.v2"
INDEX_VERSION = "hilbert.snapshot.index.v2"


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def _stats_to_dict(stats: Union[GraphStats, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert either a GraphStats dataclass or a dict into a stable,
    serialisable dictionary.
    """
    if isinstance(stats, GraphStats):
        return {
            "n_nodes": stats.n_nodes,
            "n_edges": stats.n_edges,
            "density": stats.density,
            "avg_degree": stats.avg_degree,
            "transitivity": stats.transitivity,
        }

    # assume dict-like
    return {
        "n_nodes": stats.get("n_nodes", 0),
        "n_edges": stats.get("n_edges", 0),
        "density": stats.get("density", 0.0),
        "avg_degree": stats.get("avg_degree", 0.0),
        "transitivity": stats.get("transitivity", 0.0),
    }


@dataclass
class SnapshotMeta:
    """
    Per-snapshot metadata written to <snapshot>.meta.json.
    """
    version: str
    timestamp: float
    snapshot: str
    snapshot_3d: Optional[str]
    pct: float
    stats: Dict[str, Any]
    top_nodes: List[str]
    layout: Dict[str, Any]
    diagnostics: Optional[Dict[str, str]] = None


# --------------------------------------------------------------------------- #
# Snapshot-level metadata writers
# --------------------------------------------------------------------------- #

def write_snapshot_metadata(
    out_dir: str,
    snapshot_name: str,
    snapshot_name_3d: Optional[str],
    pct: float,
    stats: Union[GraphStats, Dict[str, Any]],
    top_nodes: List[str],
    layout_info: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Write a single snapshot metadata JSON file.

    Returned as a dict so visualizer.py can embed it into the global index
    or other aggregation documents.
    """
    stats_dict = _stats_to_dict(stats)

    meta_obj = SnapshotMeta(
        version=META_VERSION,
        timestamp=time.time(),
        snapshot=snapshot_name,
        snapshot_3d=snapshot_name_3d,
        pct=float(pct),
        stats=stats_dict,
        top_nodes=list(top_nodes),
        layout=dict(layout_info or {}),
        diagnostics=diagnostics or {},
    )

    meta_dict = asdict(meta_obj)
    path = os.path.join(out_dir, f"{snapshot_name}.meta.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    return meta_dict


def merge_snapshot_metadata(metas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Pass-through normaliser; ensures all items are plain dicts.
    """
    out: List[Dict[str, Any]] = []
    for m in metas:
        if isinstance(m, dict):
            out.append(m)
        else:
            out.append(asdict(m))
    return out


# --------------------------------------------------------------------------- #
# Modular metadata writers (Option 2 architecture)
# --------------------------------------------------------------------------- #

def write_metadata_core(
    out_dir: str,
    *,
    version: str,
    run_seed: int,
    orchestrator_version: str,
    lsa_model_version: Optional[str],
    embedding_parameters: Optional[Dict[str, Any]],
    cluster_hierarchy_info: Optional[Dict[str, Any]],
    pruning_info: Optional[Dict[str, Any]],
) -> str:
    """
    Write graph_metadata_core.json containing stable upstream provenance.
    """
    core = {
        "version": version,
        "timestamp": time.time(),
        "run_seed": run_seed,
        "orchestrator_version": orchestrator_version,
        "lsa_model_version": lsa_model_version,
        "embedding_parameters": embedding_parameters,
        "cluster_hierarchy_info": cluster_hierarchy_info,
        "pruning": pruning_info,
    }

    path = os.path.join(out_dir, "graph_metadata_core.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(core, f, indent=2)

    return path


def write_metadata_layout(
    out_dir: str,
    layout2d: Dict[str, Any],
    layout3d: Dict[str, Any],
) -> str:
    """
    Layout parameters (seeds, latitudes, radius bounds, timings).
    """
    data = {
        "version": META_VERSION,
        "timestamp": time.time(),
        "layout2d": layout2d,
        "layout3d": layout3d,
    }

    path = os.path.join(out_dir, "graph_layout.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return path


def write_metadata_analytics(
    out_dir: str,
    *,
    global_stats: Dict[str, Any],
    community_sizes: Dict[str, int],
    component_sizes: Dict[str, int],
    compound_summary: Dict[str, Any],
    root_summary: Dict[str, Any],
) -> str:
    """
    Analytics-level metadata (clusters, components, compounds).
    """
    data = {
        "version": META_VERSION,
        "timestamp": time.time(),
        "global_stats": global_stats,
        "community_sizes": community_sizes,
        "component_sizes": component_sizes,
        "compound_summary": compound_summary,
        "root_summary": root_summary,
    }

    path = os.path.join(out_dir, "graph_analytics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return path


def write_metadata_diagnostics(
    out_dir: str,
    diagnostics: Dict[str, str],
) -> str:
    """
    Scientific diagnostic plot index.
    """
    data = {
        "version": META_VERSION,
        "timestamp": time.time(),
        "diagnostics": diagnostics or {},
    }

    path = os.path.join(out_dir, "graph_diagnostics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return path


# --------------------------------------------------------------------------- #
# Global index (entrypoint) - now lightweight
# --------------------------------------------------------------------------- #

def write_global_index(
    out_dir: str,
    global_stats: Union[GraphStats, Dict[str, Any]],
    snapshot_meta_dicts: List[Dict[str, Any]],
    config: Dict[str, Any],
    *,
    metadata_files: Optional[Dict[str, str]] = None,
) -> str:
    """
    Write graph_snapshots_index.json:

    This is now a *thin pointer document* referencing:
      - snapshot meta dicts
      - separate modular metadata files (core/layout/analytics/diagnostics)
    """
    gs = _stats_to_dict(global_stats)

    index = {
        "version": INDEX_VERSION,
        "timestamp": time.time(),
        "global_stats": gs,
        "config": config,
        "snapshots": merge_snapshot_metadata(snapshot_meta_dicts),
        "metadata_files": metadata_files or {},  # pointers to the four new JSON files
    }

    path = os.path.join(out_dir, "graph_snapshots_index.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    return path
