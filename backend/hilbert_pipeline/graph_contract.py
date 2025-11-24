# hilbert_pipeline/graph_contract.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


GRAPH_CONTRACT_VERSION = "1.0"


# ============================================================
# Node schema
# ============================================================

@dataclass
class HilbertGraphNode:
    node_id: str
    label: str

    element_id: Optional[str] = None
    molecule_id: Optional[str] = None
    compound_id: Optional[str] = None
    parent_compound: Optional[str] = None

    lsa0: float = 0.0
    lsa1: float = 0.0
    lsa2: float = 0.0

    tf: float = 0.0
    df: float = 0.0
    idf: float = 0.0
    tfidf: float = 0.0

    temperature: float = 0.0
    stability: float = 0.0
    entropy: float = 0.0
    mean_entropy: float = 0.0
    coherence: float = 0.0
    mean_coherence: float = 0.0

    community_id: int = -1
    root_cluster: int = -1
    component_id: int = -1

    degree: int = 0
    betweenness: float = 0.0
    importance_score: float = 0.0
    is_hub: bool = False
    is_outlier: bool = False

    # Optional cached semantic coordinates for reproducibility
    x_semantic: Optional[float] = None
    y_semantic: Optional[float] = None
    z_semantic: Optional[float] = None

    pruned: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


# ============================================================
# Edge schema
# ============================================================

@dataclass
class HilbertGraphEdge:
    source: str
    target: str

    weight: float
    scaled_weight: float
    polarity: int
    confidence: float

    is_backbone: bool = False

    direction: Optional[int] = None
    causal_strength: Optional[float] = None
    relation_type: Optional[str] = None

    pruned: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


# ============================================================
# Metadata schema
# ============================================================

@dataclass
class HilbertGraphMetadata:
    version: str = GRAPH_CONTRACT_VERSION

    run_seed: Optional[int] = None
    orchestrator_version: Optional[str] = None

    lsa_model_version: Optional[str] = None
    embedding_parameters: Optional[Dict[str, Any]] = None

    cluster_hierarchy_info: Optional[Dict[str, Any]] = None

    pruning: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()
