# hilbert_pipeline/artifacts.py
# =============================================================================
# Canonical artifact schemas and helpers for the Hilbert pipeline.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Union

import numpy as np
import pandas as pd


# =============================================================================
# Artifact kind registry
# =============================================================================

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
    SIGNATURES = "signatures"
    COMPOUND_STABILITY = "compound_stability"
    PERSISTENCE_VISUALS = "persistence_visuals"


# =============================================================================
# Shared schema: span map and embeddings
# =============================================================================

class SpanRecord(TypedDict):
    doc: str
    doc_id: int
    span_id: int
    position: int
    text: str
    elements: List[str]


@dataclass
class LSAField:
    embeddings: np.ndarray
    span_map: List[SpanRecord]
    vocab: List[str]
    H_span: Optional[List[float]] = None

    def to_jsonable(self) -> Dict[str, Any]:
        # Convert numpy matrix to python lists
        if isinstance(self.embeddings, np.ndarray):
            emb = self.embeddings.tolist()
        else:
            emb = self.embeddings

        out = {
            "embeddings": emb,
            "span_map": list(self.span_map),
            "vocab": list(self.vocab),
        }
        if self.H_span is not None:
            out["H_span"] = list(self.H_span)
        return out


# =============================================================================
# Validation helpers for common artifacts
# =============================================================================

# Elements CSV required by fusion, edges, molecules, stability
REQUIRED_ELEMENTS_COLS = ["element"]
OPTIONAL_ELEMENTS_COLS = [
    "collection_freq",
    "document_freq",
    "entropy",
    "coherence",
    "embedding",
    "span_id",
    "doc",
    "text",
    "mean_entropy",
    "mean_coherence",
]

def ensure_hilbert_elements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate hilbert_elements.csv.
    Only requires the core element column but warns when critical columns
    for downstream stages are missing.
    """
    missing_required = [c for c in REQUIRED_ELEMENTS_COLS if c not in df.columns]
    if missing_required:
        raise ValueError(f"hilbert_elements.csv missing required columns: {missing_required}")

    # warn on optional missing columns but do not raise
    for col in OPTIONAL_ELEMENTS_COLS:
        if col not in df.columns:
            # silently tolerated for backward compatibility
            pass

    return df


def ensure_signal_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate signal_stability.csv used by persistence layer and graph visualizer.
    """
    required = ["element", "stability", "entropy", "coherence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"signal_stability.csv missing required columns: {missing}")
    return df


def ensure_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate edges.csv for molecule layer and graph visualization.
    """
    required = ["source", "target", "weight"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"edges.csv missing required columns: {missing}")
    return df


def ensure_molecules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate molecules.csv.
    """
    required = ["compound_id", "element"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"molecules.csv missing required columns: {missing}")
    return df


def ensure_compounds(data: Union[Dict[str, Any], List[Any]]) -> Dict[str, Any]:
    """
    Validate informational_compounds.json.
    Always returns a dict mapping compound_id -> record.
    """
    if isinstance(data, list):
        # convert list of {compound_id: "..."} to dict
        mapped = {str(d.get("compound_id")): d for d in data if isinstance(d, dict)}
        return mapped

    if isinstance(data, dict):
        return data

    raise ValueError("informational_compounds.json must be dict or list of dicts.")


def ensure_signatures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate signatures.csv (misinfo crystal seed layer).
    """
    required = [
        "element",
        "support",
        "p_information",
        "p_misinformation",
        "p_disinformation",
        "p_ambiguous",
        "entropy_bits",
        "dominant_label",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"signatures.csv missing required columns: {missing}")
    return df


def ensure_compound_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate compound_stability.csv.
    """
    required = [
        "compound_id",
        "n_elements",
        "mean_element_coherence",
        "mean_element_stability",
        "stability_variance",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"compound_stability.csv missing required columns: {missing}")
    return df


# =============================================================================
# Public loading helpers
# =============================================================================

def load_hilbert_elements(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return ensure_hilbert_elements(df)


def load_signal_stability(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return ensure_signal_stability(df)


def load_edges(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return ensure_edges(df)


def load_molecules(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return ensure_molecules(df)


def load_compounds(path: str) -> Dict[str, Any]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ensure_compounds(data)


def load_signatures(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return ensure_signatures(df)


def load_compound_stability(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return ensure_compound_stability(df)


# =============================================================================
# JSON safe conversion helpers (for orchestrator)
# =============================================================================

def numpy_to_list(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, dict)):
        return x
    return x


def dataclass_to_json(dc: Any) -> Dict[str, Any]:
    d = asdict(dc)
    return {k: numpy_to_list(v) for k, v in d.items()}
