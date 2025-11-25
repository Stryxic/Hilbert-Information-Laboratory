"""
Elements API - used by the periodic table and element inspectors.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from hilbert_db.core import HilbertDB


@dataclass
class ElementRequest:
    run_id: str
    element_id: str


@dataclass
class ElementSummary:
    element_id: str
    frequency: float
    entropy: float
    coherence: float
    description: Optional[str]


@dataclass
class ElementDetailResponse:
    run_id: str
    summary: ElementSummary
    spans: List[Dict[str, Any]]
    neighbours: List[Dict[str, Any]]  # graph neighbours
    metrics: Dict[str, Any]


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _load_import_root(db: HilbertDB, run_id: str) -> Path:
    imported = db.load_imported_run(run_id)
    return Path(imported.cache_dir)


def _load_element_descriptions(root: Path) -> Dict[str, str]:
    """
    Load element_descriptions.json if present.

    Expected formats:
        - {"element_id": "description", ...}
        - [{"element": "...", "description": "..."}, ...]
    """
    desc_path = root / "element_descriptions.json"
    if not desc_path.exists():
        return {}

    with desc_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # direct mapping
        return {str(k): str(v) for k, v in data.items()}

    if isinstance(data, list):
        out: Dict[str, str] = {}
        for row in data:
            if not isinstance(row, dict):
                continue
            el = row.get("element") or row.get("element_id")
            desc = row.get("description")
            if el and desc:
                out[str(el)] = str(desc)
        return out

    return {}


def _pick_float(row: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            try:
                return float(row[k])
            except Exception:
                continue
    return default


def _load_elements_csv(root: Path) -> List[Dict[str, Any]]:
    path = root / "hilbert_elements.csv"
    if not path.exists():
        raise FileNotFoundError("hilbert_elements.csv not found in run export")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_signal_stability(root: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load signal_stability.csv into a mapping element_id -> metrics dict.
    """
    path = root / "signal_stability.csv"
    if not path.exists():
        return {}

    by_element: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            el = row.get("element") or row.get("element_id")
            if not el:
                continue
            by_element[str(el)] = dict(row)
    return by_element


def _load_span_fusion(root: Path) -> List[Dict[str, Any]]:
    path = root / "span_element_fusion.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_default_graph(root: Path) -> Dict[str, Any]:
    """
    Load a default graph snapshot.

    Strategy:
        - read graph_snapshots_index.json
        - if it lists depths, take the first
        - if snapshots include explicit paths, use them
        - fall back to graphs/full.json if present
    """
    idx_path = root / "graph_snapshots_index.json"
    graphs_root = root / "graphs"

    depth: Optional[str] = None
    graph_path: Optional[Path] = None

    if idx_path.exists():
        with idx_path.open("r", encoding="utf-8") as f:
            idx_data = json.load(f)

        # If index is a list of {depth, path}
        if isinstance(idx_data, list) and idx_data:
            entry = idx_data[0]
            if isinstance(entry, dict):
                depth = str(entry.get("depth") or entry.get("id") or "full")
                p = entry.get("path") or entry.get("file")
                if p:
                    graph_path = (root / p).resolve()

        # If index is a dict with "available_depths"
        if depth is None and isinstance(idx_data, dict):
            depths = idx_data.get("available_depths") or idx_data.get("depths")
            if isinstance(depths, list) and depths:
                depth = str(depths[0])

    if graph_path is None:
        if depth is not None:
            graph_path = graphs_root / f"{depth}.json"
        else:
            # final fallback
            graph_path = graphs_root / "full.json"
            depth = "full"

    if not graph_path.exists():
        # no graph available
        return {"depth": depth or "unknown", "nodes": [], "edges": [], "metadata": {}}

    with graph_path.open("r", encoding="utf-8") as f:
        graph_data = json.load(f)

    if not isinstance(graph_data, dict):
        return {"depth": depth or "unknown", "nodes": [], "edges": [], "metadata": {}}

    return {
        "depth": depth or graph_data.get("depth") or "unknown",
        "nodes": graph_data.get("nodes", []),
        "edges": graph_data.get("edges", []),
        "metadata": graph_data.get("metadata", {}),
    }


def _find_element_id(row: Dict[str, Any]) -> Optional[str]:
    return (
        row.get("element")
        or row.get("element_id")
        or row.get("id")
        or row.get("token")
    )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def list_elements(db: HilbertDB, run_id: str) -> List[ElementSummary]:
    """
    List all elements for a run, with core metrics.

    Data sources:
        - hilbert_elements.csv
        - element_descriptions.json (optional)
    """
    root = _load_import_root(db, run_id)
    rows = _load_elements_csv(root)
    descriptions = _load_element_descriptions(root)

    summaries: List[ElementSummary] = []

    for row in rows:
        el_id = _find_element_id(row)
        if not el_id:
            continue

        freq = _pick_float(row, ["frequency", "freq", "count", "span_count"], 0.0)
        entropy = _pick_float(row, ["entropy", "mean_entropy"], 0.0)
        coherence = _pick_float(row, ["coherence", "mean_coherence"], 0.0)
        desc = descriptions.get(str(el_id))

        summaries.append(
            ElementSummary(
                element_id=str(el_id),
                frequency=freq,
                entropy=entropy,
                coherence=coherence,
                description=desc,
            )
        )

    return summaries


def get_element_detail(db: HilbertDB, req: ElementRequest) -> ElementDetailResponse:
    """
    Return detailed information about one element.

    Includes:
        - ElementSummary
        - LSA spans / fusion rows related to the element
        - Graph neighbours
        - Stability metrics (signal_stability.csv)
    """
    root = _load_import_root(db, req.run_id)

    # Summaries
    summaries = {s.element_id: s for s in list_elements(db, req.run_id)}
    summary = summaries.get(req.element_id)
    if summary is None:
        raise KeyError(f"Element {req.element_id!r} not found for run {req.run_id!r}")

    # Stability metrics
    stability_map = _load_signal_stability(root)
    metrics = stability_map.get(req.element_id, {})

    # Spans - from span_element_fusion.csv, filtered by element
    span_rows = _load_span_fusion(root)
    spans: List[Dict[str, Any]] = []
    for row in span_rows:
        el = row.get("element") or row.get("element_id")
        if str(el) == req.element_id:
            spans.append(dict(row))

    # Neighbours - from default graph snapshot
    graph = _load_default_graph(root)
    neighbours: List[Dict[str, Any]] = []
    for edge in graph.get("edges", []):
        src = edge.get("source") or edge.get("from")
        tgt = edge.get("target") or edge.get("to")
        if src == req.element_id or tgt == req.element_id:
            neighbours.append(dict(edge))

    return ElementDetailResponse(
        run_id=req.run_id,
        summary=summary,
        spans=spans,
        neighbours=neighbours,
        metrics=metrics,
    )


__all__ = [
    "ElementRequest",
    "ElementSummary",
    "ElementDetailResponse",
    "list_elements",
    "get_element_detail",
]
