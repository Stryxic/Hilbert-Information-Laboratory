# Package for Hilbert pipeline modules.
# =============================================================================
# hilbert_pipeline/element_labels.py — Element Naming & Descriptors
# =============================================================================
"""
Derives human-readable labels and descriptors for Hilbert elements based on
their statistical behavior across the corpus.

Inputs:
    - hilbert_elements.csv (doc,sid,element,probability,entropy,coherence,stability)
    - spans: list of span dicts from orchestrator (optional, for examples)

Outputs:
    - element_descriptions.json
    - element_intensity.csv
"""

import os
import json
from typing import List, Dict
import numpy as np
import pandas as pd


def _categorize_role(mean_entropy: float, mean_coherence: float) -> str:
    """
    Rough role categorization:
      - low entropy, high coherence    -> backbone / anchor
      - high entropy, high coherence   -> diverse-but-focused
      - low coherence                  -> fringe / volatile
    """
    if mean_coherence >= 0.65 and mean_entropy <= 0.45:
        return "backbone"
    if mean_coherence >= 0.65 and mean_entropy > 0.45:
        return "focused-diverse"
    if mean_coherence < 0.35 and mean_entropy >= 0.55:
        return "volatile"
    if mean_coherence < 0.35:
        return "fringe"
    return "neutral"


def _generate_name(element_id: str, role: str) -> str:
    """
    Generate a compact human-readable name based on role.
    """
    base = element_id
    if role == "backbone":
        return f"{base} — Core Factual Backbone"
    if role == "focused-diverse":
        return f"{base} — Thematic Hub"
    if role == "volatile":
        return f"{base} — Volatile Discourse"
    if role == "fringe":
        return f"{base} — Peripheral Context"
    return f"{base} — Neutral Carrier"


def _generate_description(element_id: str,
                          role: str,
                          mean_entropy: float,
                          mean_coherence: float,
                          doc_coverage: float) -> str:
    """
    Short descriptor; keep it informative but compact.
    """
    ent = f"{mean_entropy:.2f}"
    coh = f"{mean_coherence:.2f}"
    cov = f"{100.0 * doc_coverage:.1f}%"

    if role == "backbone":
        return (
            f"{element_id} appears consistently across documents with low lexical noise "
            f"(entropy {ent}) and strong alignment (coherence {coh}). It acts as a "
            f"stable factual backbone (~{cov} of docs)."
        )
    if role == "focused-diverse":
        return (
            f"{element_id} clusters many varied expressions (entropy {ent}) around a "
            f"shared theme (coherence {coh}), serving as a central thematic hub "
            f"spanning ~{cov} of documents."
        )
    if role == "volatile":
        return (
            f"{element_id} is highly variable (entropy {ent}) with weak alignment "
            f"(coherence {coh}), indicating volatile or contested framings. Present in "
            f"~{cov} of documents."
        )
    if role == "fringe":
        return (
            f"{element_id} surfaces sporadically with low coherence (coherence {coh}), "
            f"acting as a fringe or contextual modifier (~{cov} of docs)."
        )
    return (
        f"{element_id} exhibits moderate entropy ({ent}) and coherence ({coh}), "
        f"providing neutral connective tissue across ~{cov} of documents."
    )


def build_element_descriptions(elements_csv: str,
                               spans: List[Dict],
                               out_dir: str) -> str:
    """
    Construct element_descriptions.json and element_intensity.csv.

    Parameters
    ----------
    elements_csv : str
        Path to hilbert_elements.csv produced by orchestrator.
    spans : list of dict
        Span metadata; used only for optional context (not required).
    out_dir : str
        Output directory for JSON/CSV.

    Returns
    -------
    str : path to element_descriptions.json
    """
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(elements_csv):
        print(f"[elements] Missing {elements_csv}; cannot build descriptions.")
        path = os.path.join(out_dir, "element_descriptions.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)
        return path

    df = pd.read_csv(elements_csv)
    if df.empty or "element" not in df.columns:
        print("[elements] Empty or invalid hilbert_elements.csv.")
        path = os.path.join(out_dir, "element_descriptions.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)
        return path

    # Ensure expected numeric fields
    for col in ("entropy", "coherence", "stability", "probability"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["doc"] = df.get("doc", "").astype(str)

    n_docs = max(1, df["doc"].nunique())

    grouped = df.groupby("element")
    records = []
    desc_map = {}

    for el, sub in grouped:
        count = len(sub)
        doc_coverage = sub["doc"].nunique() / n_docs if n_docs > 0 else 0.0
        mean_entropy = float(sub["entropy"].mean()) if "entropy" in sub else 0.0
        mean_coherence = float(sub["coherence"].mean()) if "coherence" in sub else 0.0
        mean_stability = float(sub["stability"].mean()) if "stability" in sub else 0.0

        role = _categorize_role(mean_entropy, mean_coherence)
        name = _generate_name(el, role)
        desc = _generate_description(el, role, mean_entropy, mean_coherence, doc_coverage)

        records.append({
            "element": el,
            "count": int(count),
            "doc_coverage": float(doc_coverage),
            "mean_entropy": round(mean_entropy, 6),
            "mean_coherence": round(mean_coherence, 6),
            "mean_stability": round(mean_stability, 6),
            "role": role,
        })

        desc_map[el] = {
            "id": el,
            "name": name,
            "role": role,
            "summary": desc,
            "metrics": {
                "count": int(count),
                "doc_coverage": float(doc_coverage),
                "mean_entropy": float(mean_entropy),
                "mean_coherence": float(mean_coherence),
                "mean_stability": float(mean_stability),
            },
        }

    # Save intensity table (used by PDF + dashboards)
    intensity_path = os.path.join(out_dir, "element_intensity.csv")
    pd.DataFrame(records).sort_values("count", ascending=False).to_csv(
        intensity_path, index=False
    )

    # Save descriptions JSON
    desc_path = os.path.join(out_dir, "element_descriptions.json")
    with open(desc_path, "w", encoding="utf-8") as f:
        json.dump(desc_map, f, indent=2)

    print(f"[elements] Wrote {desc_path} and {intensity_path}")
    return desc_path
