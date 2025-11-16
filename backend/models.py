from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


# ============================================================
# 1. Spectral Span Field
# ============================================================

class Span(BaseModel):
    span_id: int
    doc: str
    text: str
    entropy: Optional[float] = None
    coherence: Optional[float] = None
    stability: Optional[float] = None


class GlobalFieldStats(BaseModel):
    H_bar: float
    C_global: float
    n_spans: int


class SpectralField(BaseModel):
    spans: List[Span]
    global_: GlobalFieldStats = Field(..., alias="global")

    class Config:
        allow_population_by_field_name = True


# ============================================================
# 2. Element Layer (Agents)
# ============================================================

class Element(BaseModel):
    element: str
    token: str
    doc: Optional[str] = None

    tf: Optional[int] = None
    df: Optional[int] = None

    entropy: Optional[float] = None
    coherence: Optional[float] = None

    embedding: Optional[List[float]] = None

    root_element: Optional[str] = None

    info_score: Optional[float] = None
    misinfo_score: Optional[float] = None
    disinfo_score: Optional[float] = None


class ElementStats(BaseModel):
    n_elements: Optional[int] = None
    n_roots: Optional[int] = None


class ElementLayer(BaseModel):
    elements: List[Element]
    stats: Optional[ElementStats] = None


# ============================================================
# 3. Relationship Layer (Edges / Bonds)
# ============================================================

class Edge(BaseModel):
    source: str
    target: str
    weight: float


class EdgeStats(BaseModel):
    n_edges: Optional[int] = None
    similarity_threshold: Optional[float] = None
    top_k: Optional[int] = None


class EdgeLayer(BaseModel):
    edges: List[Edge]
    stats: Optional[EdgeStats] = None


# ============================================================
# 4. Molecular / Compound Layer
# ============================================================

class CompoundNode(BaseModel):
    id: str
    entropy: Optional[float] = None
    coherence: Optional[float] = None


class CompoundEdge(BaseModel):
    source: str
    target: str
    weight: float


class CompoundGraph(BaseModel):
    nodes: List[CompoundNode]
    edges: List[CompoundEdge]


class RegimeProfile(BaseModel):
    info: Optional[float] = None
    misinfo: Optional[float] = None
    disinfo: Optional[float] = None


class Compound(BaseModel):
    compound_id: str
    elements: List[str]

    num_elements: Optional[int] = None
    num_bonds: Optional[int] = None

    compound_stability: Optional[float] = None
    mean_temperature: Optional[float] = None

    regime_profile: Optional[RegimeProfile] = None
    graph: Optional[CompoundGraph] = None


class CompoundStats(BaseModel):
    n_compounds: Optional[int] = None


class CompoundLayer(BaseModel):
    compounds: List[Compound]
    stats: Optional[CompoundStats] = None


# ============================================================
# 5. Document Signatures
# ============================================================

class DocumentTopElement(BaseModel):
    element: str
    tf: Optional[int] = None
    entropy: Optional[float] = None


class EntropyDistribution(BaseModel):
    mean: Optional[float] = None
    stdev: Optional[float] = None


class DocumentSignature(BaseModel):
    top_elements: Optional[List[DocumentTopElement]] = None
    compound_membership: Optional[List[str]] = None
    entropy_distribution: Optional[EntropyDistribution] = None


class DocumentRecord(BaseModel):
    doc: str
    signature: DocumentSignature


class DocumentLayer(BaseModel):
    documents: List[DocumentRecord]


# ============================================================
# 6. Timeline Annotations
# ============================================================

class TimelineEntry(BaseModel):
    span_id: int
    timestamp: str
    stability: Optional[float] = None
    entropy: Optional[float] = None
    coherence: Optional[float] = None
    doc: str


class TimelineLayer(BaseModel):
    timeline: List[TimelineEntry]


# ============================================================
# 7. Figures
# ============================================================

class Figures(BaseModel):
    __root__: Dict[str, str]

    def __getitem__(self, item):
        return self.__root__.get(item)


# ============================================================
# 8. Meta Data
# ============================================================

class Meta(BaseModel):
    run: str
    dir: str
    generated_at: str


# ============================================================
# 9. Top-Level /get_results Response
# ============================================================

class HilbertResults(BaseModel):
    status: str

    meta: Meta

    field: SpectralField
    elements: ElementLayer
    edges: EdgeLayer
    compounds: CompoundLayer

    documents: Optional[DocumentLayer] = None
    timeline: Optional[TimelineLayer] = None
    figures: Optional[Figures] = None
