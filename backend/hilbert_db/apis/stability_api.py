"""
Stability API - exposes element stability, compound stability,
persistence fields, and stability distributions.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from hilbert_db.core import HilbertDB


@dataclass
class StabilityPoint:
    element_id: str
    entropy: float
    coherence: float
    stability: float


@dataclass
class PersistenceResponse:
    thresholds: List[float]
    active_counts: List[int]     # active molecules or regions
    plot_url: str                # path in object store or local cache


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _load_import_root(db: HilbertDB, run_id: str) -> Path:
    imported = db.load_imported_run(run_id)
    return Path(imported.cache_dir)


def _pick_float(row: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            try:
                return float(row[k])
            except Exception:
                continue
    return default


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def get_stability_table(db: HilbertDB, run_id: str) -> List[StabilityPoint]:
    """
    Return all per-element stability scores.

    Data source:
        - signal_stability.csv
    """
    root = _load_import_root(db, run_id)
    path = root / "signal_stability.csv"
    if not path.exists():
        raise FileNotFoundError("signal_stability.csv not found in run export")

    points: List[StabilityPoint] = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            el = row.get("element") or row.get("element_id")
            if not el:
                continue

            entropy = _pick_float(row, ["entropy", "mean_entropy"], 0.0)
            coherence = _pick_float(row, ["coherence", "mean_coherence"], 0.0)
            stability = _pick_float(row, ["stability", "signal_stability"], 0.0)

            points.append(
                StabilityPoint(
                    element_id=str(el),
                    entropy=entropy,
                    coherence=coherence,
                    stability=stability,
                )
            )

    return points


def get_compound_stability(db: HilbertDB, run_id: str) -> List[Dict[str, Any]]:
    """
    Return per-compound stability metrics.

    Data source:
        - compound_stability.csv
    """
    root = _load_import_root(db, run_id)
    path = root / "compound_stability.csv"
    if not path.exists():
        raise FileNotFoundError("compound_stability.csv not found in run export")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def get_persistence_field(db: HilbertDB, run_id: str) -> PersistenceResponse:
    """
    Return persistence sweep (threshold vs active structure count).

    Data source:
        - persistence_field.json

    Expected JSON structure:

        {
          "thresholds": [...],
          "active_counts": [...],
          "plot_url": "..."   # optional
        }

    If any of these keys are missing, they default to sensible values.
    """
    root = _load_import_root(db, run_id)
    path = root / "persistence_field.json"
    if not path.exists():
        raise FileNotFoundError("persistence_field.json not found in run export")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    thresholds = data.get("thresholds") or []
    active_counts = data.get("active_counts") or []
    plot_url = data.get("plot_url") or data.get("image", "")

    thresholds = [float(t) for t in thresholds]
    active_counts = [int(c) for c in active_counts]

    return PersistenceResponse(
        thresholds=thresholds,
        active_counts=active_counts,
        plot_url=str(plot_url),
    )


__all__ = [
    "StabilityPoint",
    "PersistenceResponse",
    "get_stability_table",
    "get_compound_stability",
    "get_persistence_field",
]
