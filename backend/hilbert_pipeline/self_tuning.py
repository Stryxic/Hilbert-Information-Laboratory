# =============================================================================
# self_tuning.py - Hilbert Information Laboratory
# Epistemic-field-based configuration tuning
# =============================================================================

from __future__ import annotations
from typing import Dict, Any
import numpy as np


def compute_field_stats(elements_df, stability_df) -> Dict[str, float]:
    """
    Compute global entropy/coherence/stability aggregates.
    These values are used both for documentation and for self-tuning heuristics.
    """

    stats = {}

    if "entropy" in elements_df.columns:
        ent = elements_df["entropy"].astype(float)
        stats["entropy_mean"] = float(ent.mean())
        stats["entropy_median"] = float(ent.median())
        stats["entropy_p90"] = float(ent.quantile(0.90))
        stats["entropy_max"] = float(ent.max())

        # fraction of high entropy elements
        stats["entropy_high_fraction"] = float((ent > ent.mean() + ent.std()).mean())

    if "coherence" in elements_df.columns:
        coh = elements_df["coherence"].astype(float)
        stats["coherence_mean"] = float(coh.mean())
        stats["coherence_median"] = float(coh.median())
        stats["coherence_low_fraction"] = float((coh < 0.6).mean())
        stats["coherence_min"] = float(coh.min())

    if "stability" in elements_df.columns:
        stab = elements_df["stability"].astype(float)
        stats["stability_mean"] = float(stab.mean())
        stats["stability_median"] = float(stab.median())
        stats["stability_low_fraction"] = float((stab < 0.25).mean())
        stats["stability_p10"] = float(stab.quantile(0.10))

    # Document-level stability spread
    if stability_df is not None and "doc_stability" in stability_df.columns:
        ds = stability_df["doc_stability"].astype(float)
        stats["doc_stability_spread"] = float(ds.std())

    return stats


def generate_tuning_suggestions(stats: Dict[str, float], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate human-readable suggestions for pipeline tuning.
    No parameters are changed automatically; this is advisory.
    """

    suggestions = []
    severity = 0  # 0 = OK, increasing levels indicate concern

    # High entropy - maybe element tokenizer too permissive
    if stats.get("entropy_high_fraction", 0) > 0.25:
        suggestions.append(
            "High proportion of high-entropy elements (>{:.2f}). Consider increasing min_df or reducing max_vocab in LSA."
            .format(stats["entropy_high_fraction"])
        )
        severity += 1

    # Low coherence - spans too fragmented or tokenizer too noisy
    if stats.get("coherence_low_fraction", 0) > 0.20:
        suggestions.append(
            "Large fraction of low-coherence elements. Consider tightening element filters or increasing n_components."
        )
        severity += 1

    # Very low stability tail
    if stats.get("stability_low_fraction", 0) > 0.20:
        suggestions.append(
            "A significant number of elements show low stability. Consider adjusting stability weightings or span segmentation."
        )
        severity += 1

    # Document stability imbalance
    if stats.get("doc_stability_spread", 0) > 0.25:
        suggestions.append(
            "Document stability variance is high. Corpus may contain highly inconsistent files; consider pre-cleaning or weighting by document."
        )
        severity += 1

    return {
        "severity": severity,
        "suggestions": suggestions,
        "config_used": config,
        "stats": stats,
    }
