"""
Molecules API - provides access to molecule (connected component)
and compound-level representations.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hilbert_db.core import HilbertDB


@dataclass
class MoleculeSummary:
    molecule_id: str
    num_elements: int
    stability: float
    compound_id: str


@dataclass
class MoleculeDetailResponse:
    run_id: str
    molecule_id: str
    elements: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    graph_substructure: Dict[str, Any]


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _load_import_root(db: HilbertDB, run_id: str) -> Path:
    imported = db.load_imported_run(run_id)
    return Path(imported.cache_dir)


def _load_molecules_csv(root: Path) -> List[Dict[str, Any]]:
    path = root / "molecules.csv"
    if not path.exists():
        raise FileNotFoundError("molecules.csv not found in run export")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_compound_stability_csv(root: Path) -> List[Dict[str, Any]]:
    path = root / "compound_stability.csv"
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_compounds_json(root: Path) -> Any:
    path = root / "informational_compounds.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _group_molecule_elements(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_mol: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        mol_id = row.get("molecule_id") or row.get("molecule") or row.get("id")
        if not mol_id:
            continue
        by_mol.setdefault(str(mol_id), []).append(row)
    return by_mol


def _pick_float(row: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            try:
                return float(row[k])
            except Exception:
                continue
    return default


def _build_summary_from_rows(mol_id: str, rows: List[Dict[str, Any]]) -> MoleculeSummary:
    # Try direct summary fields if they exist
    first = rows[0] if rows else {}

    if "num_elements" in first and "stability" in first:
        num_elements = int(float(first.get("num_elements") or 0))
        stability = _pick_float(first, ["stability", "compound_stability"], 0.0)
        compound_id = str(first.get("compound_id", "") or "")
    else:
        # Derive summary from raw rows
        elements = set()
        for r in rows:
            el = r.get("element") or r.get("element_id") or r.get("node_id")
            if el:
                elements.add(str(el))
        num_elements = len(elements)
        stability = 0.0
        compound_id = str(first.get("compound_id", "") or "")

    return MoleculeSummary(
        molecule_id=mol_id,
        num_elements=num_elements,
        stability=stability,
        compound_id=compound_id,
    )


def _load_graph_for_molecule(root: Path, element_ids: List[str]) -> Dict[str, Any]:
    """
    Build a small subgraph for a molecule by filtering the default graph
    to only edges where both endpoints are in element_ids.
    """
    from .graph_api import _load_graph as _load_graph_internal  # type: ignore

    depth = None  # default selection logic
    depth_resolved, nodes, edges, metadata = _load_graph_internal(root, depth)

    element_set = set(element_ids)

    sub_nodes = [n for n in nodes if str(n.get("id")) in element_set]
    sub_edges = [
        e
        for e in edges
        if (str(e.get("source") or e.get("from")) in element_set)
        and (str(e.get("target") or e.get("to")) in element_set)
    ]

    return {
        "depth": depth_resolved,
        "nodes": sub_nodes,
        "edges": sub_edges,
        "metadata": metadata,
    }


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def list_molecules(db: HilbertDB, run_id: str) -> List[MoleculeSummary]:
    """
    Return molecule-level summaries (components).
    """
    root = _load_import_root(db, run_id)
    rows = _load_molecules_csv(root)
    grouped = _group_molecule_elements(rows)

    summaries: List[MoleculeSummary] = []
    for mol_id, mol_rows in grouped.items():
        summaries.append(_build_summary_from_rows(mol_id, mol_rows))

    summaries.sort(key=lambda m: m.molecule_id)
    return summaries


def get_molecule_detail(db: HilbertDB, run_id: str, molecule_id: str) -> MoleculeDetailResponse:
    """
    Return detailed information for a single molecule.
    """
    root = _load_import_root(db, run_id)
    rows = _load_molecules_csv(root)
    grouped = _group_molecule_elements(rows)

    if molecule_id not in grouped:
        raise KeyError(f"Molecule {molecule_id!r} not found for run {run_id!r}")

    mol_rows = grouped[molecule_id]
    # Basic element list
    elements = [dict(r) for r in mol_rows]

    # Metrics - combine any numeric summary fields in first row
    first = mol_rows[0]
    metrics: Dict[str, Any] = {}
    for k, v in first.items():
        if v is None:
            continue
        # try to store floats where sensible
        try:
            metrics[k] = float(v)
        except Exception:
            metrics[k] = v

    # Graph substructure
    element_ids: List[str] = []
    for r in mol_rows:
        el = r.get("element") or r.get("element_id") or r.get("node_id")
        if el:
            element_ids.append(str(el))

    graph_sub = _load_graph_for_molecule(root, element_ids) if element_ids else {
        "depth": "unknown",
        "nodes": [],
        "edges": [],
        "metadata": {},
    }

    return MoleculeDetailResponse(
        run_id=run_id,
        molecule_id=molecule_id,
        elements=elements,
        metrics=metrics,
        graph_substructure=graph_sub,
    )


__all__ = [
    "MoleculeSummary",
    "MoleculeDetailResponse",
    "list_molecules",
    "get_molecule_detail",
]
