# hilbert_orchestrator/stages/__init__.py

from __future__ import annotations

# Re-export core stage types
from hilbert_orchestrator.core.stages import (
    PipelineSettings,
    StageState,
    StageSpec,
)
from hilbert_orchestrator.core.registry import GLOBAL_STAGE_REGISTRY

# Import concrete pipeline stage implementations so they register
# themselves with GLOBAL_STAGE_REGISTRY via decorators.

# LSA
from .lsa_stage import lsa_stage  # noqa: F401

# Fusion
from .fusion_stage import fusion_stage  # noqa: F401

# Edges
from .edge_stage import stage_edges  # noqa: F401

# Molecules
from .molecule_stage import stage_molecule_layer  # noqa: F401

# Elements (roots + labels)
from .element_stage import (  # noqa: F401
    stage_element_roots,
    stage_element_labels,
)

# Stability + persistence
from .stability_stage import (  # noqa: F401
    stage_stability,
    stage_compound_stability,
    stage_persistence,
)

# Signatures
from .signatures_stage import stage_signatures  # noqa: F401

# LM
from .lm_stage import (  # noqa: F401
    stage_element_lm,
    stage_perplexity,
)

# Export
from .stage_export import stage_export  # noqa: F401


__all__ = [
    "PipelineSettings",
    "StageState",
    "StageSpec",
    "GLOBAL_STAGE_REGISTRY",
]
