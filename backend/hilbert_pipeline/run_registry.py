# =============================================================================
# run_registry.py - Hilbert Information Laboratory
# Persistent registry for pipeline runs
# =============================================================================

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class RunRegistry:
    """
    Stores metadata for each pipeline run as JSONL in results/run_registry.jsonl.
    Also exposes a “latest state” snapshot for the front end and Hilbert Assistant.
    """

    def __init__(self, results_base: Path):
        self.results_base = Path(results_base)
        self.registry_path = self.results_base / "run_registry.jsonl"
        self.latest_path = self.results_base / "control_plane_state.json"

    # ----------------------------------------------------------------------
    def record_run(self, run_id: str, config: Dict[str, Any],
                   field_stats: Dict[str, float],
                   tuning: Dict[str, Any]):
        """
        Append a new run to the registry.
        """
        entry = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "config": config,
            "field_stats": field_stats,
            "self_tuning": tuning,
        }

        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        # Also produce a latest snapshot for assistant + frontend
        with open(self.latest_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)

    # ----------------------------------------------------------------------
    def load_latest(self) -> Optional[Dict[str, Any]]:
        if not self.latest_path.exists():
            return None
        try:
            return json.loads(self.latest_path.read_text())
        except Exception:
            return None

    # ----------------------------------------------------------------------
    def load_all(self):
        if not self.registry_path.exists():
            return []
        lines = self.registry_path.read_text().splitlines()
        return [json.loads(l) for l in lines if l.strip()]
