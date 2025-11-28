"""
element_roots.py
================

Hilbert Information Pipeline – Element Root Consolidation Layer
-----------------------------------------------------------------------------

This module consolidates *surface-level elements* into coarse, interpretable
root clusters. The purpose is *not* to perform full linguistic lemmatisation or
semantic synonymy resolution. Instead, it applies transparent, reproducible
heuristics to merge:

- trivial plural/singular alternations
- cosmetic inflectional variants
- obvious token-level morphological siblings

The output is a set of *root clusters* that help downstream visualisations,
compound summaries, and analytics by reducing superficial element noise.

Inputs
------
- ``hilbert_elements.csv``  
  Produced by the LSA layer; required.

- ``edges.csv`` *(optional)*  
  Provided by the fusion/edges layer; used only to help choose a representative
  member for each cluster (via degree ranking).

Outputs
-------
This module produces three artifacts in ``results_dir``:

1. **element_roots.csv**  
   One row per element.

   Columns:
       - ``element`` : str  
       - ``root_id`` : str (``R0001`` pattern)  
       - ``root_token`` : stemmed token phrase  
       - ``representative`` : canonical element for the cluster  
       - ``cluster_size`` : int  
       - ``collection_freq`` : optional  
       - ``document_freq`` : optional  

2. **element_clusters.json**  
   One entry per cluster.

   Keys:
       - ``root_id``  
       - ``root_token``  
       - ``label`` (representative)  
       - ``cluster_size``  
       - ``members``  
       - ``total_collection_freq``  
       - ``total_document_freq``  
       - ``mean_entropy``  
       - ``mean_coherence``  

3. **element_cluster_metrics.json**  
   Global statistics:
       - ``n_elements``  
       - ``n_roots``  
       - ``n_singletons``  
       - ``mean_cluster_size``  
       - ``median_cluster_size``  
       - ``max_cluster_size``  
       - ``top_roots_by_size``  

Design Principles
-----------------
- **Conservative merging** – avoids over-aggressive conflation.  
- **Deterministic ordering** – ensures reproducibility across runs.  
- **Explainable heuristics** – only token-level transformations, no opaque NLP.  
- **Orchestrator-safe logging** – uses injected ``emit`` callback.

Pipeline Position
-----------------
This layer is run *after* the molecule/edge layers and *before* graph-level
visualisation or export steps.

"""
# ============================================================================
# element_roots.py – Robust Edition
# Hilbert Information Pipeline – Element Root Consolidation Layer
# ============================================================================

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Logging
# =============================================================================

DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda *_a, **_k: None


def _safe_emit(emit: Callable, kind: str, payload: Dict[str, Any]) -> None:
    """Emit without crashing orchestrator."""
    try:
        emit(kind, payload)
    except Exception:
        pass


# =============================================================================
# Tokenisation / Stemming
# =============================================================================

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenise(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    try:
        return _TOKEN_RE.findall(text.lower())
    except Exception:
        return []


def _stem_token(tok: str) -> str:
    t = tok.lower()
    if len(t) <= 3:
        return t

    try:
        if len(t) > 4 and t.endswith("ies") and not t.endswith("eies"):
            return t[:-3] + "y"

        for suf in ("ing", "ers", "er", "ed"):
            if len(t) > len(suf) + 2 and t.endswith(suf):
                return t[: -len(suf)]

        if len(t) > 3 and t.endswith("es") and not t.endswith("ses"):
            return t[:-2]
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            return t[:-1]
    except Exception:
        return t

    return t


def _root_key(element: str) -> str:
    toks = _tokenise(element)
    if not toks:
        return (element or "").strip().lower()
    stems = []
    for t in toks:
        try:
            stems.append(_stem_token(t))
        except Exception:
            stems.append(t)
    return " ".join(stems).strip()


# =============================================================================
# Clustering Data Structure
# =============================================================================

@dataclass
class RootCluster:
    root_id: str
    root_token: str
    members: List[str]


# =============================================================================
# Loading Helpers
# =============================================================================

def _load_elements(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"hilbert_elements.csv missing: {path}")
    df = pd.read_csv(path)
    if "element" not in df.columns:
        raise ValueError("Missing required column 'element'.")
    return df


def _load_degrees(path: str) -> Dict[str, int]:
    if not os.path.exists(path):
        return {}

    try:
        edf = pd.read_csv(path)
    except Exception:
        return {}

    if "source" not in edf.columns or "target" not in edf.columns:
        return {}

    deg: Dict[str, int] = {}
    for col in ("source", "target"):
        try:
            ser = edf[col].astype(str)
            for v in ser:
                deg[v] = deg.get(v, 0) + 1
        except Exception:
            continue
    return deg


# =============================================================================
# Clustering
# =============================================================================

def _build_clusters(df: pd.DataFrame, degrees: Dict[str, int]) -> Tuple[List[RootCluster], Dict[str, RootCluster]]:
    bucket: Dict[str, List[str]] = {}

    for raw_el in df["element"].astype(str):
        try:
            rk = _root_key(raw_el)
        except Exception:
            rk = raw_el.strip().lower()

        bucket.setdefault(rk, []).append(str(raw_el))

    # Deterministic ordering
    sorted_items = sorted(bucket.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    clusters: List[RootCluster] = []
    e2c: Dict[str, RootCluster] = {}

    for idx, (rk, members) in enumerate(sorted_items, start=1):
        uniq = sorted(set(members))
        cid = f"R{idx:04d}"
        cluster = RootCluster(root_id=cid, root_token=rk, members=uniq)
        clusters.append(cluster)
        for el in uniq:
            e2c[el] = cluster

    return clusters, e2c


def _select_representative(
    cluster: RootCluster,
    df: pd.DataFrame,
    degrees: Dict[str, int]
) -> str:
    subset = df[df["element"].astype(str).isin(cluster.members)].copy()

    # 1. Highest collection frequency
    if "collection_freq" in subset.columns:
        try:
            numeric = pd.to_numeric(subset["collection_freq"], errors="coerce")
            if numeric.notna().any():
                idx = numeric.idxmax()
                return str(subset.loc[idx, "element"])
        except Exception:
            pass

    # 2. Highest graph degree
    if degrees:
        try:
            return sorted(cluster.members, key=lambda el: (-degrees.get(el, 0), el))[0]
        except Exception:
            pass

    # 3. Fallback
    return sorted(cluster.members)[0]


def _safe_mean(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    try:
        arr = pd.to_numeric(series, errors="coerce").to_numpy()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        val = float(np.mean(arr))
        return val if np.isfinite(val) else None
    except Exception:
        return None


# =============================================================================
# Artifact Writing
# =============================================================================

def _write_cluster_artifacts(out: str, df: pd.DataFrame, clusters: List[RootCluster],
                             e2c: Dict[str, RootCluster], degrees: Dict[str, int]) -> None:
    os.makedirs(out, exist_ok=True)

    rows = []
    clusters_json: Dict[str, Any] = {}

    has_cf = "collection_freq" in df.columns
    has_df = "document_freq" in df.columns
    has_entropy = "entropy" in df.columns
    has_coherence = "coherence" in df.columns

    for c in clusters:
        members = c.members
        sub = df[df["element"].astype(str).isin(members)].copy()

        rep = _select_representative(c, df, degrees)

        # Compute cluster-level metrics
        total_cf = None
        total_df_ = None
        mean_ent = None
        mean_coh = None

        if has_cf:
            try:
                nums = pd.to_numeric(sub["collection_freq"], errors="coerce").fillna(0)
                total_cf = float(nums.sum())
            except Exception:
                total_cf = None

        if has_df:
            try:
                nums = pd.to_numeric(sub["document_freq"], errors="coerce").fillna(0)
                total_df_ = float(nums.sum())
            except Exception:
                total_df_ = None

        if has_entropy:
            mean_ent = _safe_mean(sub["entropy"])
        if has_coherence:
            mean_coh = _safe_mean(sub["coherence"])

        clusters_json[c.root_id] = {
            "root_id": c.root_id,
            "root_token": c.root_token,
            "label": rep,
            "cluster_size": len(members),
            "members": members,
            "total_collection_freq": total_cf,
            "total_document_freq": total_df_,
            "mean_entropy": mean_ent,
            "mean_coherence": mean_coh,
        }

        for _, row in sub.iterrows():
            rows.append({
                "element": str(row["element"]),
                "root_id": c.root_id,
                "root_token": c.root_token,
                "representative": rep,
                "cluster_size": len(members),
                "collection_freq": row.get("collection_freq"),
                "document_freq": row.get("document_freq"),
            })

    pd.DataFrame(rows).to_csv(os.path.join(out, "element_roots.csv"), index=False)

    with open(os.path.join(out, "element_clusters.json"), "w", encoding="utf-8") as f:
        json.dump(clusters_json, f, indent=2)

    # Global metrics
    sizes = [len(c.members) for c in clusters]
    arr = np.asarray(sizes, dtype=float)
    metrics = {
        "n_elements": len(df),
        "n_roots": len(clusters),
        "n_singletons": sum(1 for s in sizes if s == 1),
        "mean_cluster_size": float(arr.mean()) if arr.size else 0.0,
        "median_cluster_size": float(np.median(arr)) if arr.size else 0.0,
        "max_cluster_size": int(arr.max()) if arr.size else 0,
        "top_roots_by_size": [
            {
                "root_id": c.root_id,
                "root_token": c.root_token,
                "cluster_size": len(c.members),
            }
            for c in sorted(clusters, key=lambda c: (-len(c.members), c.root_id))[:20]
        ],
    }

    with open(os.path.join(out, "element_cluster_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# =============================================================================
# Public API
# =============================================================================

def run_element_roots(out_dir: str,
                      emit: Callable[[str, Dict[str, Any]], None] = DEFAULT_EMIT) -> None:
    elements_csv = os.path.join(out_dir, "hilbert_elements.csv")
    edges_csv = os.path.join(out_dir, "edges.csv")

    _safe_emit(emit, "log", {"stage": "element_roots", "event": "start"})

    # Load elements
    try:
        df = _load_elements(elements_csv)
    except Exception as exc:
        _safe_emit(emit, "log", {
            "stage": "element_roots",
            "event": "error",
            "error": str(exc),
        })
        return

    if df.empty:
        _safe_emit(emit, "log", {
            "stage": "element_roots",
            "event": "skip",
            "reason": "hilbert_elements.csv empty"
        })
        return

    degrees = _load_degrees(edges_csv)

    # Build clusters
    try:
        clusters, e2c = _build_clusters(df, degrees)
    except Exception as exc:
        _safe_emit(emit, "log", {
            "stage": "element_roots",
            "event": "error",
            "error": f"Cluster build failed: {exc}"
        })
        return

    # Write artifacts
    try:
        _write_cluster_artifacts(out_dir, df, clusters, e2c, degrees)
    except Exception as exc:
        _safe_emit(emit, "log", {
            "stage": "element_roots",
            "event": "error",
            "error": f"Artifact writing failed: {exc}"
        })
        return

    _safe_emit(emit, "log", {
        "stage": "element_roots",
        "event": "end",
        "n_elements": len(df),
        "n_roots": len(clusters),
    })


__all__ = ["run_element_roots"]
