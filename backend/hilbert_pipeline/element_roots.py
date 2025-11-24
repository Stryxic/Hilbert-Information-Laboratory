# =============================================================================
# hilbert_pipeline/element_roots.py - Element Root Consolidator
# =============================================================================
"""
Consolidate surface-level elements into approximate "roots".

This module operates on the outputs of earlier Hilbert pipeline stages,
in particular:

  - hilbert_elements.csv
  - edges.csv (optional, for degree-based ranking)

It produces three artifacts:

  1. element_roots.csv
       Per-element mapping to a root cluster.

       Columns:
         - element: original element string
         - root_id: cluster identifier (e.g. R0001)
         - root_token: canonical root token/phrase
         - representative: representative element label for the cluster
         - cluster_size: number of elements in the cluster
         - collection_freq: original collection_freq (if present)
         - document_freq: original document_freq (if present)

  2. element_clusters.json
       Cluster-level summary keyed by root_id.

       Each entry contains:
         - root_id
         - root_token
         - label (representative element)
         - cluster_size
         - members: list of member element strings
         - total_collection_freq
         - total_document_freq
         - mean_entropy
         - mean_coherence

  3. element_cluster_metrics.json
       Global statistics over the clustering:
         - n_elements
         - n_roots
         - n_singletons
         - mean_cluster_size
         - median_cluster_size
         - max_cluster_size
         - top_roots_by_size: small table of the largest clusters

The goal is to provide a *conservative* consolidation step that groups
obvious inflectional and cosmetic variants of the same underlying
informational form, without pretending to solve full lemmatisation or
synonymy. It is intentionally transparent and easy to extend.

Typical orchestrator usage:

    from hilbert_pipeline.element_roots import run_element_roots
    run_element_roots(results_dir, emit=ctx.emit)

If hilbert_elements.csv is missing or empty, the module logs a warning
and no artifacts are written.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable, Optional

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Orchestrator-compatible emitter
# -------------------------------------------------------------------------

DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda *_a, **_k: None  # type: ignore


# -------------------------------------------------------------------------
# Basic tokenisation + stemming heuristics
# -------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenise(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.lower())


def _stem_token(tok: str) -> str:
    """
    Very lightweight, explainable stemming heuristic.

    - Lowercases input.
    - Handles a few common English inflectional patterns.
    - Leaves short tokens and acronyms alone.

    This is not intended to be linguistically perfect; its aim is to
    cluster obvious variants that would clearly be considered the same
    "root" in downstream analysis.
    """
    t = tok.lower()
    if len(t) <= 3:
        return t

    # -ies -> y (stories -> story)
    if len(t) > 4 and t.endswith("ies") and not t.endswith("eies"):
        return t[:-3] + "y"

    # -ing, -ers, -er, -ed
    for suf in ("ing", "ers", "er", "ed"):
        if len(t) > len(suf) + 2 and t.endswith(suf):
            return t[: -len(suf)]

    # plural -es / -s (avoid chopping off in cases like 'class')
    if len(t) > 3 and t.endswith("es") and not t.endswith("ses"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]

    return t


def _root_key(element: str) -> str:
    """
    Compute a canonical "root token" for an element.

    - tokenises the element
    - applies _stem_token to each token
    - joins stems with single spaces

    Falls back to the raw lowercased element if tokenisation yields
    nothing (for example, non-alphanumeric strings).
    """
    toks = _tokenise(element)
    if not toks:
        return (element or "").strip().lower()
    stems = [_stem_token(t) for t in toks]
    return " ".join(stems).strip()


# -------------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------------

@dataclass
class RootCluster:
    root_id: str
    root_token: str
    members: List[str]


# -------------------------------------------------------------------------
# Core consolidation logic
# -------------------------------------------------------------------------

def _load_elements(elements_csv: str) -> pd.DataFrame:
    if not os.path.exists(elements_csv):
        raise FileNotFoundError(f"hilbert_elements.csv not found at {elements_csv}")
    df = pd.read_csv(elements_csv)
    if "element" not in df.columns:
        raise ValueError("hilbert_elements.csv must contain an 'element' column.")
    return df


def _load_degrees(edges_csv: str) -> Dict[str, int]:
    """
    Optional degree map from edges.csv, used for ranking cluster representatives.

    Returns a dict: element -> degree, or empty dict if edges.csv is missing.
    """
    if not os.path.exists(edges_csv):
        return {}
    try:
        edf = pd.read_csv(edges_csv)
    except Exception:
        return {}
    if "source" not in edf.columns or "target" not in edf.columns:
        return {}

    degrees: Dict[str, int] = {}
    for col in ("source", "target"):
        vals = edf[col].astype(str)
        for v in vals:
            degrees[v] = degrees.get(v, 0) + 1
    return degrees


def _build_clusters(
    elements_df: pd.DataFrame,
    degrees: Dict[str, int],
) -> Tuple[List[RootCluster], Dict[str, RootCluster]]:
    """
    Group elements into root clusters.

    Strategy:
      - compute root_token for each element
      - group by root_token
      - assign root_id = R0001, R0002, ... in descending cluster-size order
      - return both the ordered list of RootCluster and an index
        mapping element -> RootCluster
    """
    # Map root_token -> list[element]
    bucket: Dict[str, List[str]] = {}
    for el in elements_df["element"].astype(str):
        rk = _root_key(el)
        bucket.setdefault(rk, []).append(el)

    # Order roots by cluster size (largest first), deterministic tiebreaker
    sorted_roots = sorted(
        bucket.items(),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )

    clusters: List[RootCluster] = []
    element_to_cluster: Dict[str, RootCluster] = {}

    for i, (rk, members) in enumerate(sorted_roots, start=1):
        root_id = f"R{i:04d}"
        uniq_members = sorted(set(members))
        cluster = RootCluster(root_id=root_id, root_token=rk, members=uniq_members)
        clusters.append(cluster)
        for el in uniq_members:
            element_to_cluster[el] = cluster

    return clusters, element_to_cluster


def _select_representative(
    cluster: RootCluster,
    elements_df: pd.DataFrame,
    degrees: Dict[str, int],
) -> str:
    """
    Choose a representative element for a cluster.

    Preference order:
      1) Highest collection_freq (if present)
      2) Highest graph degree (if degrees available)
      3) Lexicographically smallest element
    """
    mset = set(cluster.members)
    sub = elements_df[elements_df["element"].astype(str).isin(mset)].copy()

    # 1) collection frequency
    if "collection_freq" in sub.columns:
        try:
            sub["collection_freq"] = pd.to_numeric(
                sub["collection_freq"], errors="coerce"
            )
        except Exception:
            pass
        if sub["collection_freq"].notna().any():
            best = sub.sort_values("collection_freq", ascending=False).iloc[0]
            return str(best["element"])

    # 2) graph degree
    if degrees:
        by_deg = sorted(
            cluster.members,
            key=lambda el: (-degrees.get(el, 0), el),
        )
        return by_deg[0]

    # 3) lexical fallback
    return sorted(cluster.members)[0]


def _safe_mean(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    arr = pd.to_numeric(series, errors="coerce").to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    val = float(np.nanmean(arr))
    return val if np.isfinite(val) else None


def _write_cluster_artifacts(
    out_dir: str,
    elements_df: pd.DataFrame,
    clusters: List[RootCluster],
    element_to_cluster: Dict[str, RootCluster],
    degrees: Dict[str, int],
) -> None:
    """
    Write:
      - element_roots.csv
      - element_clusters.json
      - element_cluster_metrics.json
    """
    os.makedirs(out_dir, exist_ok=True)

    # Map stats per cluster
    clusters_json: Dict[str, Dict[str, Any]] = {}

    element_rows: List[Dict[str, Any]] = []

    # Pre-attach numeric columns if present
    has_collection = "collection_freq" in elements_df.columns
    has_document = "document_freq" in elements_df.columns
    has_entropy = "entropy" in elements_df.columns
    has_coherence = "coherence" in elements_df.columns

    for cluster in clusters:
        members = cluster.members
        csize = len(members)

        sub = elements_df[elements_df["element"].astype(str).isin(members)].copy()

        representative = _select_representative(cluster, elements_df, degrees)

        total_collection = None
        total_document = None
        mean_entropy = None
        mean_coherence = None

        if has_collection:
            try:
                total_collection = float(
                    pd.to_numeric(sub["collection_freq"], errors="coerce")
                    .fillna(0.0)
                    .sum()
                )
            except Exception:
                total_collection = None

        if has_document:
            try:
                total_document = float(
                    pd.to_numeric(sub["document_freq"], errors="coerce")
                    .fillna(0.0)
                    .sum()
                )
            except Exception:
                total_document = None

        if has_entropy:
            mean_entropy = _safe_mean(sub["entropy"])
        if has_coherence:
            mean_coherence = _safe_mean(sub["coherence"])

        clusters_json[cluster.root_id] = {
            "root_id": cluster.root_id,
            "root_token": cluster.root_token,
            "label": representative,
            "cluster_size": csize,
            "members": members,
            "total_collection_freq": total_collection,
            "total_document_freq": total_document,
            "mean_entropy": mean_entropy,
            "mean_coherence": mean_coherence,
        }

        # element_roots rows (one row per element)
        for _, row in sub.iterrows():
            el = str(row["element"])
            element_rows.append(
                {
                    "element": el,
                    "root_id": cluster.root_id,
                    "root_token": cluster.root_token,
                    "representative": representative,
                    "cluster_size": csize,
                    "collection_freq": row.get("collection_freq"),
                    "document_freq": row.get("document_freq"),
                }
            )

    # Write element_roots.csv
    roots_path = os.path.join(out_dir, "element_roots.csv")
    pd.DataFrame(element_rows).to_csv(roots_path, index=False)

    # Write element_clusters.json
    clusters_path = os.path.join(out_dir, "element_clusters.json")
    with open(clusters_path, "w", encoding="utf-8") as f:
        json.dump(clusters_json, f, indent=2)

    # Global metrics
    cluster_sizes = [len(c.members) for c in clusters]
    n_elements = int(len(elements_df))
    n_roots = int(len(clusters))
    n_singletons = int(sum(1 for s in cluster_sizes if s == 1))

    arr = np.asarray(cluster_sizes, dtype=float)
    mean_size = float(np.mean(arr)) if arr.size else 0.0
    median_size = float(np.median(arr)) if arr.size else 0.0
    max_size = int(np.max(arr)) if arr.size else 0

    # Top roots by size (for quick inspection)
    top_by_size = sorted(
        clusters,
        key=lambda c: (-len(c.members), c.root_id),
    )[:20]
    top_table = [
        {
            "root_id": c.root_id,
            "root_token": c.root_token,
            "cluster_size": len(c.members),
        }
        for c in top_by_size
    ]

    metrics = {
        "n_elements": n_elements,
        "n_roots": n_roots,
        "n_singletons": n_singletons,
        "mean_cluster_size": mean_size,
        "median_cluster_size": median_size,
        "max_cluster_size": max_size,
        "top_roots_by_size": top_table,
    }

    metrics_path = os.path.join(out_dir, "element_cluster_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

def run_element_roots(out_dir: str, emit: Callable[[str, Dict[str, Any]], None] = DEFAULT_EMIT) -> None:
    """
    High-level entry point used by the orchestrator.

    Parameters
    ----------
    out_dir:
        Hilbert run results directory (contains hilbert_elements.csv, edges.csv, etc.).

    emit:
        Optional logging callback compatible with the orchestrator:
            emit(kind: str, payload: Dict[str, Any])

    Side effects
    ------------
    Writes three files into out_dir:

      - element_roots.csv
      - element_clusters.json
      - element_cluster_metrics.json
    """
    elements_csv = os.path.join(out_dir, "hilbert_elements.csv")
    edges_csv = os.path.join(out_dir, "edges.csv")

    try:
        emit("log", {"stage": "element_roots", "event": "start"})
    except Exception:
        pass

    if not os.path.exists(elements_csv):
        msg = "[element-roots] hilbert_elements.csv not found; skipping root consolidation."
        try:
            emit("log", {"stage": "element_roots", "event": "skip", "reason": msg})
        except Exception:
            print(msg)
        return

    try:
        elements_df = _load_elements(elements_csv)
    except Exception as exc:
        msg = f"[element-roots] Failed to load hilbert_elements.csv: {exc}"
        try:
            emit("log", {"stage": "element_roots", "event": "error", "error": str(exc)})
        except Exception:
            print(msg)
        return

    if elements_df.empty:
        msg = "[element-roots] hilbert_elements.csv is empty; nothing to consolidate."
        try:
            emit("log", {"stage": "element_roots", "event": "skip", "reason": msg})
        except Exception:
            print(msg)
        return

    degrees = _load_degrees(edges_csv)

    clusters, element_to_cluster = _build_clusters(elements_df, degrees)

    _write_cluster_artifacts(out_dir, elements_df, clusters, element_to_cluster, degrees)

    try:
        emit(
            "log",
            {
                "stage": "element_roots",
                "event": "end",
                "n_elements": int(len(elements_df)),
                "n_roots": int(len(clusters)),
            },
        )
    except Exception:
        print(
            f"[element-roots] Completed root consolidation: "
            f"{len(elements_df)} elements -> {len(clusters)} roots."
        )


__all__ = ["run_element_roots"]
