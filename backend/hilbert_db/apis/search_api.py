"""
Search API - optional semantic/substring search over elements and molecules.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from hilbert_db.core import HilbertDB


@dataclass
class SearchRequest:
    run_id: str
    query: str
    limit: int = 20


@dataclass
class SearchResult:
    kind: str                 # "element", "molecule"
    id: str
    score: float
    snippet: str


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _load_import_root(db: HilbertDB, run_id: str) -> Path:
    imported = db.load_imported_run(run_id)
    return Path(imported.cache_dir)


def _load_elements(root: Path) -> List[Dict[str, Any]]:
    path = root / "hilbert_elements.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_element_descriptions(root: Path) -> Dict[str, str]:
    path = root / "element_descriptions.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
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


def _load_molecules(root: Path) -> List[Dict[str, Any]]:
    path = root / "molecules.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _element_id(row: Dict[str, Any]) -> str:
    return str(
        row.get("element")
        or row.get("element_id")
        or row.get("id")
        or ""
    )


def _molecule_id(row: Dict[str, Any]) -> str:
    return str(row.get("molecule_id") or row.get("molecule") or row.get("id") or "")


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def search(db: HilbertDB, req: SearchRequest) -> List[SearchResult]:
    """
    Perform simple substring search over elements and molecules.

    This is deliberately minimal and deterministic. It can later be
    swapped for semantic search without changing the API surface.
    """
    root = _load_import_root(db, req.run_id)
    q = req.query.strip().lower()
    if not q:
        return []

    elements = _load_elements(root)
    descriptions = _load_element_descriptions(root)
    molecules = _load_molecules(root)

    results: List[SearchResult] = []

    # Element search
    for row in elements:
        el_id = _element_id(row)
        if not el_id:
            continue
        desc = descriptions.get(el_id) or ""
        haystack = f"{el_id} {desc}".lower()
        if q in haystack:
            # crude score: shorter match distance gets higher score
            pos = haystack.find(q)
            score = 1.0 / (1.0 + float(pos if pos >= 0 else 0))
            snippet = desc or el_id
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
            results.append(
                SearchResult(
                    kind="element",
                    id=el_id,
                    score=score,
                    snippet=snippet,
                )
            )

    # Molecule search - search in IDs and any compound_id field
    for row in molecules:
        mid = _molecule_id(row)
        if not mid:
            continue
        compound_id = str(row.get("compound_id") or "")
        haystack = f"{mid} {compound_id}".lower()
        if q in haystack:
            pos = haystack.find(q)
            score = 1.0 / (1.0 + float(pos if pos >= 0 else 0))
            snippet = f"Molecule {mid}"
            if compound_id:
                snippet += f" (compound {compound_id})"
            results.append(
                SearchResult(
                    kind="molecule",
                    id=mid,
                    score=score,
                    snippet=snippet,
                )
            )

    # Sort by score descending, then kind, then id
    results.sort(key=lambda r: (-r.score, r.kind, r.id))
    if req.limit and req.limit > 0:
        results = results[: req.limit]

    return results


__all__ = [
    "SearchRequest",
    "SearchResult",
    "search",
]
