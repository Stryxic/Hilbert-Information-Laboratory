# =============================================================================
# hilbert_pipeline/hilbert_export.py - Deterministic Run Serialization
# =============================================================================
"""
Serialise a Hilbert pipeline run into a deterministic, easy to archive format.

This version includes FIXES:
  * ZIP artifact now emits an explicit "key" field (zip filename)
    so orchestrator can register export_key into the DB.
"""

from __future__ import annotations

import json
import os
import hashlib
import zipfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# -------------------------------------------------------------------------#
# Orchestrator compatibility emit
# -------------------------------------------------------------------------#
try:
    from . import DEFAULT_EMIT  # type: ignore
except Exception:
    DEFAULT_EMIT = lambda *_a, **_k: None  # type: ignore


# -------------------------------------------------------------------------#
# Helper functions for safe IO and simple stats
# -------------------------------------------------------------------------#
def _safe_mean(series: pd.Series, default: float = 0.0) -> float:
    if series is None or len(series) == 0:
        return default
    arr = pd.to_numeric(series, errors="coerce").to_numpy()
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return default
    val = float(np.nanmean(arr))
    return val if np.isfinite(val) else default


def _read_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _sha256(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# -------------------------------------------------------------------------#
# Run level readers and light metrics
# -------------------------------------------------------------------------#
def _read_run_summary(out_dir: str) -> Dict[str, Any]:
    path = os.path.join(out_dir, "hilbert_run.json")
    data = _read_json(path)
    return data if isinstance(data, dict) else {}


def _read_elements_info(out_dir: str) -> Dict[str, Any]:
    path = os.path.join(out_dir, "hilbert_elements.csv")
    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    info: Dict[str, Any] = {}

    if "element" in df.columns:
        info["num_elements"] = int(df["element"].astype(str).nunique())
    if "token" in df.columns:
        info["num_tokens"] = int(df["token"].astype(str).nunique())

    info["mean_entropy"] = (
        _safe_mean(df.get("mean_entropy", df.get("entropy", pd.Series([], dtype=float))))
    )
    info["mean_coherence"] = (
        _safe_mean(df.get("mean_coherence", df.get("coherence", pd.Series([], dtype=float))))
    )

    return info


def _read_num_spans(out_dir: str) -> Optional[int]:
    path = os.path.join(out_dir, "lsa_field.json")
    data = _read_json(path)
    if not isinstance(data, dict):
        return None
    emb = data.get("embeddings")
    return len(emb) if isinstance(emb, list) else None


def _read_regime_profile(out_dir: str) -> Dict[str, float]:
    path = os.path.join(out_dir, "hilbert_elements.csv")
    if not os.path.exists(path):
        return {"info": 0.0, "misinfo": 0.0, "disinfo": 0.0}

    try:
        df = pd.read_csv(path)
    except Exception:
        return {"info": 0.0, "misinfo": 0.0, "disinfo": 0.0}

    info = _safe_mean(df.get("info_score", pd.Series([], dtype=float)))
    mis = _safe_mean(df.get("misinfo_score", pd.Series([], dtype=float)))
    dis = _safe_mean(df.get("disinfo_score", pd.Series([], dtype=float)))

    return {"info": float(info), "misinfo": float(mis), "disinfo": float(dis)}


def _read_root_stats(out_dir: str) -> Dict[str, Any]:
    path = os.path.join(out_dir, "element_roots.csv")
    stats = {"num_roots": None, "median_cluster_size": None, "max_cluster_size": None}

    if not os.path.exists(path):
        return stats

    try:
        df = pd.read_csv(path)
    except Exception:
        return stats

    if "element" in df.columns:
        stats["num_roots"] = int(df["element"].astype(str).nunique())

    if "cluster_size" in df.columns:
        sizes = pd.to_numeric(df["cluster_size"], errors="coerce")
        sizes = sizes[np.isfinite(sizes)]
        if sizes.size:
            stats["median_cluster_size"] = float(np.median(sizes))
            stats["max_cluster_size"] = float(np.max(sizes))

    return stats


def _read_graph_metrics(out_dir: str) -> Dict[str, Any]:
    metrics = {
        "num_nodes": None,
        "num_edges": None,
        "avg_degree": None,
        "num_components": None,
        "num_communities": None,
        "modularity": None,
    }

    edges_path = os.path.join(out_dir, "edges.csv")
    if not os.path.exists(edges_path):
        return metrics

    try:
        df = pd.read_csv(edges_path)
    except Exception:
        return metrics

    if "source" not in df.columns or "target" not in df.columns:
        return metrics

    sources = df["source"].astype(str)
    targets = df["target"].astype(str)
    nodes = set(sources) | set(targets)
    m = len(df)
    n = len(nodes)

    if n == 0:
        return metrics

    metrics["num_nodes"] = n
    metrics["num_edges"] = m
    metrics["avg_degree"] = 2.0 * m / float(n)

    try:
        import networkx as nx
    except Exception:
        return metrics

    G = nx.Graph()
    for s, t in zip(sources, targets):
        if s and t and s != t:
            G.add_edge(s, t)

    if G.number_of_nodes() == 0:
        return metrics

    metrics["num_components"] = nx.number_connected_components(G)

    try:
        from networkx.algorithms.community import greedy_modularity_communities
        from networkx.algorithms.community.quality import modularity

        comms = list(greedy_modularity_communities(G))
        metrics["num_communities"] = len(comms)
        if len(comms) > 1:
            metrics["modularity"] = float(modularity(G, comms))
    except Exception:
        pass

    return metrics


# -------------------------------------------------------------------------#
# Artifact discovery
# -------------------------------------------------------------------------#
_STRUCTURED_EXT = {".csv", ".json", ".txt"}
_STABLE_PREFIXES = ("stable_", "condensed_", "hilbert_manifest")


def _discover_artifact_files(out_dir: str) -> List[Dict[str, Any]]:
    base = os.path.abspath(out_dir)
    rows: List[Dict[str, Any]] = []

    for root, _, files in os.walk(base):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _STRUCTURED_EXT:
                continue
            fpath = os.path.join(root, fname)
            if not os.path.isfile(fpath):
                continue
            rel = os.path.relpath(fpath, base).replace("\\", "/")
            rows.append(
                {"path": rel, "ext": ext,
                 "size_bytes": os.path.getsize(fpath),
                 "sha256": _sha256(fpath)}
            )

    rows.sort(key=lambda r: r["path"])
    return rows


def _identify_stable_artifacts(artifacts: List[Dict[str, Any]]) -> List[str]:
    out = []
    for row in artifacts:
        name = os.path.basename(row["path"])
        if any(name.startswith(p) for p in _STABLE_PREFIXES):
            out.append(row["path"])
    return sorted(out)


# -------------------------------------------------------------------------#
# Manifest
# -------------------------------------------------------------------------#
def build_manifest(out_dir: str, emit=DEFAULT_EMIT) -> str:
    os.makedirs(out_dir, exist_ok=True)

    run_summary = _read_run_summary(out_dir)

    manifest = {
        "schema_version": 1,
        "run": {
            "run_id": run_summary.get("run_id"),
            "corpus_dir": run_summary.get("corpus_dir"),
            "results_dir": run_summary.get("results_dir", out_dir),
            "settings": dict(sorted((run_summary.get("settings") or {}).items())),
        },
        "metrics": {
            "spans": _read_num_spans(out_dir),
            "elements": _read_elements_info(out_dir),
            "regimes": _read_regime_profile(out_dir),
            "roots": _read_root_stats(out_dir),
            "graph": _read_graph_metrics(out_dir),
        },
        "environment": run_summary.get("env") or {},
        "versions": run_summary.get("versions") or {},
        "artifacts": {
            "files": _discover_artifact_files(out_dir),
            "stable_paths": None,  # compatibility, rarely used now
        },
    }

    path = os.path.join(out_dir, "hilbert_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"[export] Manifest written to {path}")
    try:
        emit("artifact", {"path": path, "kind": "hilbert_manifest_json"})
    except Exception:
        pass

    return path


# -------------------------------------------------------------------------#
# ZIP Export
# -------------------------------------------------------------------------#
def export_zip(out_dir: str, manifest_path: Optional[str] = None, emit=DEFAULT_EMIT):
    base = os.path.abspath(out_dir)
    run_name = os.path.basename(base.rstrip(os.sep))
    zip_name = "hilbert_export.zip"
    zip_path = os.path.join(base, zip_name)

    if manifest_path is None:
        manifest_path = build_manifest(out_dir, emit=emit)

    manifest = _read_json(manifest_path) or {}
    files_meta = manifest.get("artifacts", {}).get("files", [])

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for row in files_meta:
            rel = row.get("path")
            if not rel:
                continue
            fpath = os.path.join(base, rel)
            if not os.path.isfile(fpath):
                continue
            arcname = os.path.join(run_name, rel)
            zf.write(fpath, arcname=arcname)

        # ensure manifest included
        rel_man = os.path.relpath(manifest_path, base).replace("\\", "/")
        arcname = os.path.join(run_name, rel_man)
        if not any(r.get("path") == rel_man for r in files_meta):
            zf.write(manifest_path, arcname=arcname)

    print(f"[export] Created archive: {zip_path}")

    # FIX: Emit explicit 'key' so orchestrator can register export_key
    try:
        emit("artifact",
             {"path": zip_path,
              "key": zip_name,     # <---- critical fix
              "kind": "hilbert_export_zip"})
    except Exception:
        pass


# -------------------------------------------------------------------------#
# Public orchestrator entrypoint
# -------------------------------------------------------------------------#
def run_full_export(out_dir: str, emit=DEFAULT_EMIT) -> None:
    try:
        emit("pipeline", {"stage": "export", "event": "start"})
    except Exception:
        pass

    manifest = build_manifest(out_dir, emit=emit)
    export_zip(out_dir, manifest_path=manifest, emit=emit)

    try:
        emit("pipeline", {"stage": "export", "event": "end"})
    except Exception:
        pass


__all__ = ["build_manifest", "export_zip", "run_full_export"]
