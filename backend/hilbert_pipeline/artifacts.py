# hilbert_pipeline/artifacts.py
# =============================================================================
# Canonical artifact schemas and helpers for the Hilbert pipeline.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------#
# Artifact kinds
# -----------------------------------------------------------------------------#

class ArtifactKind(str, Enum):
    HILBERT_ELEMENTS = "hilbert_elements"
    LSA_FIELD = "lsa_field"
    SIGNAL_STABILITY = "signal_stability"
    MOLECULES = "molecules"
    COMPOUNDS = "informational_compounds"
    SPAN_ELEMENT_FUSION = "span_element_fusion"
    ELEMENT_DESCRIPTIONS = "element_descriptions"
    ELEMENT_INTENSITY = "element_intensity"
    ELEMENT_SPAN_ENRICHED = "element_span_enriched"
    ELEMENT_SPAN_MAP = "element_span_map"
    GRAPH_EDGES = "edges"
    GRAPH_SNAPSHOTS = "graph_snapshots"
    RUN_SUMMARY = "hilbert_run"
    RUN_CONFIG = "run_config"


# -----------------------------------------------------------------------------#
# Core schema types
# -----------------------------------------------------------------------------#

class SpanRecord(TypedDict):
    doc: str
    span_id: int
    text: str


@dataclass
class LSAField:
    embeddings: np.ndarray
    span_map: List[SpanRecord]
    vocab: List[str]

    def to_jsonable(self) -> Dict[str, Any]:
        emb = self.embeddings.tolist() if isinstance(self.embeddings, np.ndarray) else self.embeddings
        return {
            "embeddings": emb,
            "span_map": list(self.span_map),
            "vocab": list(self.vocab),
        }


# -----------------------------------------------------------------------------#
# Validation helpers for common artifacts
# -----------------------------------------------------------------------------#

REQUIRED_HILBERT_ELEMENT_COLS = ["element", "span_id", "doc", "text"]
OPTIONAL_HILBERT_ELEMENT_COLS = [
    "embedding",
    "entropy",
    "coherence",
    "collection_freq",
    "document_freq",
]


def ensure_hilbert_elements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that a dataframe looks like hilbert_elements.csv.

    Raises ValueError if required columns are missing.
    """
    missing = [c for c in REQUIRED_HILBERT_ELEMENT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"hilbert_elements.csv missing required columns: {missing}")
    return df


def ensure_signal_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic validation for signal_stability.csv.
    """
    required = ["element", "stability"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"signal_stability.csv missing required columns: {missing}")
    return df


def load_hilbert_elements(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return ensure_hilbert_elements(df)


def load_signal_stability(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return ensure_signal_stability(df)
