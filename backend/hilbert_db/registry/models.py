from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


# ----------------------------------------------------------------------
# Corpus
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class CorpusRecord:
    corpus_id: str
    name: str
    fingerprint: str
    source_uri: Optional[str] = None
    notes: Optional[str] = None
    status: str = "active"
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class RunRecord:
    run_id: str
    corpus_id: str

    orchestrator_version: Optional[str]
    settings_json: Dict[str, object]

    status: str = "pending"
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    finished_at: Optional[str] = None
    export_key: Optional[str] = None


RunStatus = str   # for type clarity ("pending", "running", "ok", "failed", "canceled")


# ----------------------------------------------------------------------
# Artifact
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    run_id: str
    name: str
    kind: str
    key: str
    meta: Dict[str, object]
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
