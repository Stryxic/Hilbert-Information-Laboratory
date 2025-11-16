# =============================================================================
# hilbert_pipeline/element_labels.py — Element Description & Label Builder
# =============================================================================
"""
Build human-facing labels and summaries for Hilbert elements.

Inputs
------
  - hilbert_elements.csv
      Canonical element table with columns such as:
        element, token, tf, doc_freq, mean_entropy, mean_coherence,
        info_score, misinfo_score, disinfo_score, ...
  - lsa_field.json  (optional, same folder as hilbert_elements.csv)
      May contain token-level element_regimes:
        {
          "element_regimes": {
            "<token>": { "info": float, "misinfo": float, "disinfo": float },
            ...
          }
        }
  - spans (optional, recommended)
      List of dicts:
        { "sid": str, "doc": str, "text": str, "type": "text" | "code" | ... }

Outputs
-------
  - element_descriptions.json

    {
      "<element_id>": {
        "element": "<id>",
        "token": "<token>",
        "label": "<short label>",
        "summary": "<1-2 line description>",
        "examples": [ "<example sent 1>", "<example sent 2>", ... ],
        "regime": {
          "info": float,
          "misinfo": float,
          "disinfo": float,
          "polarity": "neutral" | "informationally stable"
                      | "noisy / misinfo-prone"
                      | "highly polarized / disinfo-prone"
        },
        "metrics": {
          "tf": float,
          "doc_freq": int,
          "mean_entropy": float,
          "mean_coherence": float
        }
      },
      ...
    }

The output is intentionally simple JSON so the frontend can render and
filter it easily, while still being rich enough for thesis-aligned analysis.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Orchestrator integration
# -----------------------------------------------------------------------------

# Default no-op emit so this module can be used standalone
DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda _t, _d: None  # noqa: E731


def _log(msg: str, emit: Callable[[str, Dict[str, Any]], None] = DEFAULT_EMIT) -> None:
    """Unified logger that also emits structured events when available."""
    print(msg)
    emit("log", {"message": msg})


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        v = int(x)
        return v
    except Exception:
        return default


def _infer_token_map(df: pd.DataFrame) -> Dict[str, str]:
    """
    Build a mapping element_id -> token string.

    If 'token' is missing, fallback to the element id itself.
    """
    if "element" not in df.columns:
        raise ValueError("hilbert_elements.csv must contain an 'element' column")

    if "token" in df.columns:
        grouped = df.groupby("element")["token"].first().astype(str)
        return grouped.to_dict()
    else:
        return {e: str(e) for e in df["element"].astype(str)}


def _aggregate_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate metrics per element.

    Returns:
      {
        "element_id": {
          "tf": ...,
          "doc_freq": ...,
          "mean_entropy": ...,
          "mean_coherence": ...
        },
        ...
      }
    """
    # Normalize column names we care about
    df = df.copy()

    # tf aliases
    if "tf" not in df.columns and "count" in df.columns:
        df["tf"] = df["count"]
    if "tf" not in df.columns and "frequency" in df.columns:
        df["tf"] = df["frequency"]
    if "tf" not in df.columns:
        df["tf"] = 1.0

    # doc_freq aliases
    if "doc_freq" not in df.columns and "df" in df.columns:
        df["doc_freq"] = df["df"]
    if "doc_freq" not in df.columns:
        df["doc_freq"] = 1

    # entropy / coherence aliases
    if "mean_entropy" not in df.columns and "entropy" in df.columns:
        df["mean_entropy"] = df["entropy"]
    if "mean_coherence" not in df.columns and "coherence" in df.columns:
        df["mean_coherence"] = df["coherence"]

    def _agg(group: pd.DataFrame) -> Dict[str, Any]:
        return {
            "tf": float(group["tf"].astype(float).sum()),
            "doc_freq": int(group["doc_freq"].astype(int).max()),
            "mean_entropy": float(
                group["mean_entropy"].astype(float).mean()
                if "mean_entropy" in group.columns
                else 0.0
            ),
            "mean_coherence": float(
                group["mean_coherence"].astype(float).mean()
                if "mean_coherence" in group.columns
                else 0.0
            ),
        }

    metrics: Dict[str, Dict[str, Any]] = {}
    for el, group in df.groupby("element"):
        metrics[str(el)] = _agg(group)

    return metrics


def _load_regimes(elements_csv: str) -> Dict[str, Dict[str, float]]:
    """
    Load token-level regime scores from lsa_field.json if available.

    Returns:
      { token: { "info": float, "misinfo": float, "disinfo": float }, ... }
    """
    base_dir = os.path.dirname(os.path.abspath(elements_csv))
    lsa_path = os.path.join(base_dir, "lsa_field.json")

    if not os.path.exists(lsa_path):
        return {}

    try:
        with open(lsa_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(
            f"[element_labels][warn] Failed to load {lsa_path}: {e}"
        )
        return {}

    regimes = data.get("element_regimes") or {}
    # Ensure numeric, finite floats
    cleaned: Dict[str, Dict[str, float]] = {}
    for tok, reg in regimes.items():
        if not isinstance(reg, dict):
            continue
        cleaned[tok] = {
            "info": _safe_float(reg.get("info", 0.0), 0.0),
            "misinfo": _safe_float(reg.get("misinfo", 0.0), 0.0),
            "disinfo": _safe_float(reg.get("disinfo", 0.0), 0.0),
        }
    return cleaned


def _classify_polarity(info: float, mis: float, dis: float) -> str:
    """
    Turn regime scores into a coarse polarity label.

    The thresholds are deliberately soft; this is a narrative label for
    human interpretation, not a hard classifier.
    """
    scores = [info, mis, dis]
    max_score = max(scores)
    total = info + mis + dis

    if total < 1e-6 or max_score < 0.25:
        return "neutral"

    if max_score == info and info > 0.5:
        return "informationally stable"

    if max_score == dis and dis > 0.5:
        return "highly polarized / disinfo-prone"

    if max_score == mis and mis > 0.5:
        return "noisy / misinfo-prone"

    return "neutral"


def _short_label(token: str, max_len: int = 24) -> str:
    """Generate a concise, UI-friendly label for a token."""
    tok = token.strip()
    if len(tok) <= max_len:
        return tok
    # Try not to cut in the middle of a word if possible
    cut = tok[: max_len - 3]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "..."


def _choose_example_targets(
    metrics: Dict[str, Dict[str, Any]],
    max_targets: int = 500,
    min_tf: float = 1.0,
) -> List[str]:
    """
    Decide which elements are worth searching for examples.

    We cap by max_targets so that large corpora do not explode
    into O(|spans| * |elements|) work.
    """
    # Sort by tf desc (informative elements first)
    items: List[Tuple[str, Dict[str, Any]]] = list(metrics.items())
    items.sort(key=lambda kv: _safe_float(kv[1].get("tf", 0.0)), reverse=True)

    targets: List[str] = []
    for el, m in items:
        tf = _safe_float(m.get("tf", 0.0))
        if tf < min_tf:
            continue
        targets.append(el)
        if len(targets) >= max_targets:
            break
    return targets


def _build_examples(
    spans: List[dict],
    token_map: Dict[str, str],
    metrics: Dict[str, Dict[str, Any]],
    max_examples_per_element: int = 3,
) -> Dict[str, List[str]]:
    """
    Extract short example spans per element, using a lightweight
    regex-based containment heuristic.

    To keep runtime sane, we only search for examples for elements
    that pass _choose_example_targets.
    """
    examples: Dict[str, List[str]] = {el: [] for el in token_map.keys()}
    if not spans:
        return examples

    targets = set(_choose_example_targets(metrics))
    # Precompile regex per targeted element
    compiled: Dict[str, re.Pattern] = {}
    for el in targets:
        tok = token_map.get(el, "")
        t = tok.strip()
        if len(t) < 2:
            continue
        try:
            compiled[el] = re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE)
        except re.error:
            continue

    if not compiled:
        return examples

    for s in spans:
        txt = str(s.get("text", "") or "").strip()
        if not txt:
            continue
        low = txt.lower()
        for el, pattern in compiled.items():
            if len(examples[el]) >= max_examples_per_element:
                continue
            if pattern.search(low):
                examples[el].append(txt)

    return examples


# -----------------------------------------------------------------------------
# Main builder
# -----------------------------------------------------------------------------

def build_element_descriptions(
    elements_csv: str,
    spans: List[dict] | None,
    out_dir: str,
    emit: Callable[[str, Dict[str, Any]], None] = DEFAULT_EMIT,
) -> None:
    """
    Build element_descriptions.json from hilbert_elements.csv and optional spans.

    This function is robust to partial / missing metrics and to missing
    lsa_field.json regimes; it will still produce usable descriptions.
    """
    if not os.path.exists(elements_csv):
        _log("[element_labels][warn] elements_csv not found; nothing to do.", emit)
        return

    try:
        elements_df = pd.read_csv(elements_csv)
    except Exception as e:
        _log(f"[element_labels][warn] Failed to read {elements_csv}: {e}", emit)
        return

    if elements_df.empty:
        _log("[element_labels][warn] elements table is empty; skipping.", emit)
        return

    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1) Token map and metrics
    # -------------------------------------------------------------------------
    try:
        token_map = _infer_token_map(elements_df)
    except Exception as e:
        _log(f"[element_labels][warn] Cannot infer token map: {e}", emit)
        return

    metrics = _aggregate_metrics(elements_df)
    _log(
        f"[element_labels] Aggregated metrics for {len(metrics)} elements.",
        emit,
    )

    # -------------------------------------------------------------------------
    # 2) Regimes (optional)
    # -------------------------------------------------------------------------
    regimes_by_token = _load_regimes(elements_csv)
    if regimes_by_token:
        _log(
            f"[element_labels] Loaded regime profiles for "
            f"{len(regimes_by_token)} tokens.",
            emit,
        )

    # -------------------------------------------------------------------------
    # 3) Example spans (optional)
    # -------------------------------------------------------------------------
    spans = spans or []
    examples_map = _build_examples(spans, token_map, metrics)

    # -------------------------------------------------------------------------
    # 4) Build description records
    # -------------------------------------------------------------------------
    descriptions: Dict[str, Dict[str, Any]] = {}

    for el in sorted(token_map.keys(), key=str):
        token = token_map[el]

        m = metrics.get(el, {})
        tf = _safe_float(m.get("tf", 0.0))
        df = _safe_int(m.get("doc_freq", 0))
        me = _safe_float(m.get("mean_entropy", 0.0))
        mc = _safe_float(m.get("mean_coherence", 0.0))

        reg = regimes_by_token.get(token, {})
        info = _safe_float(reg.get("info", 0.0))
        mis = _safe_float(reg.get("misinfo", 0.0))
        dis = _safe_float(reg.get("disinfo", 0.0))
        polarity = _classify_polarity(info, mis, dis)

        label = _short_label(token)

        # A succinct, thesis-aligned description
        summary_parts = [
            f"Element {el} ('{token}') models a recurring semantic unit in the corpus.",
            f"It appears with tf={tf:.1f} across {df:d} document(s),",
            f"showing mean entropy={me:.3f} and coherence={mc:.3f}."
        ]

        if polarity == "informationally stable":
            summary_parts.append(
                "Its regime profile suggests it is predominantly informationally stable."
            )
        elif polarity == "noisy / misinfo-prone":
            summary_parts.append(
                "Its regime profile indicates noise or misinfo-prone usage."
            )
        elif polarity == "highly polarized / disinfo-prone":
            summary_parts.append(
                "Its regime profile indicates strong polarization or disinfo-prone behavior."
            )
        else:
            summary_parts.append(
                "Its regime profile is approximately neutral in the information/misinfo/disinfo space."
            )

        summary = " ".join(summary_parts)

        descriptions[el] = {
            "element": el,
            "token": token,
            "label": label,
            "summary": summary,
            "examples": examples_map.get(el, []),
            "regime": {
                "info": info,
                "misinfo": mis,
                "disinfo": dis,
                "polarity": polarity,
            },
            "metrics": {
                "tf": tf,
                "doc_freq": df,
                "mean_entropy": me,
                "mean_coherence": mc,
            },
        }

    out_path = os.path.join(out_dir, "element_descriptions.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(descriptions, f, indent=2)
        _log(
            f"[element_labels] Wrote {out_path} with {len(descriptions)} elements",
            emit,
        )
        # Notify orchestrator that an artifact was created, if it cares
        emit("artifact", {"path": out_path, "kind": "element_descriptions"})
    except Exception as e:
        _log(f"[element_labels][warn] Failed to write {out_path}: {e}", emit)


# -----------------------------------------------------------------------------
# CLI helper (manual debugging)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build element_descriptions.json from hilbert_elements.csv"
    )
    parser.add_argument(
        "--elements",
        type=str,
        default="results/hilbert_run/hilbert_elements.csv",
        help="Path to hilbert_elements.csv",
    )
    parser.add_argument(
        "--spans",
        type=str,
        default="",
        help="Optional path to spans JSON (list of {sid, doc, text, type})",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/hilbert_run",
        help="Output directory for element_descriptions.json",
    )

    args = parser.parse_args()

    spans_data: List[dict] | None = None
    if args.spans and os.path.exists(args.spans):
        try:
            with open(args.spans, "r", encoding="utf-8") as f:
                spans_data = json.load(f)
            if not isinstance(spans_data, list):
                spans_data = None
        except Exception as exc:
            print(f"[element_labels][warn] Failed to load spans: {exc}")
            spans_data = None

    build_element_descriptions(args.elements, spans_data or [], args.out)
