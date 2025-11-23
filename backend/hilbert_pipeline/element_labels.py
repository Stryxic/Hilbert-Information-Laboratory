# =============================================================================
# hilbert_pipeline/element_labels.py — Element Description & Label Builder (v2)
# =============================================================================
"""
Build human-facing labels and summaries for Hilbert elements.

Inputs
------
  - hilbert_elements.csv
  - lsa_field.json (optional)
  - spans (optional)

Outputs
-------
  - element_descriptions.json
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------#
# Orchestrator integration
# -----------------------------------------------------------------------------#

DEFAULT_EMIT: Callable[[str, Dict[str, Any]], None] = lambda *_: None


def _log(
    emit: Callable[[str, Dict[str, Any]], None],
    level: str,
    msg: str,
    **fields: Any,
):
    payload = {"level": level, "msg": msg}
    payload.update(fields)
    try:
        emit("log", payload)
    except Exception:
        print(f"[{level}] {msg} {fields}")


# -----------------------------------------------------------------------------#
# Utilities
# -----------------------------------------------------------------------------#

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _infer_token_map(df: pd.DataFrame) -> Dict[str, str]:
    """Map element_id -> token (fallback to element)."""
    if "element" not in df.columns:
        raise ValueError("hilbert_elements.csv must contain an 'element' column")

    if "token" in df.columns:
        grouped = df.groupby("element")["token"].first().astype(str)
        return grouped.to_dict()
    return {str(e): str(e) for e in df["element"].astype(str)}


def _aggregate_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute TF, doc_freq, entropy, coherence per element."""
    df = df.copy()

    # tf fallbacks
    if "tf" not in df.columns:
        if "count" in df.columns:
            df["tf"] = df["count"]
        elif "frequency" in df.columns:
            df["tf"] = df["frequency"]
        else:
            df["tf"] = 1.0

    # doc_freq fallbacks
    if "doc_freq" not in df.columns:
        if "df" in df.columns:
            df["doc_freq"] = df["df"]
        else:
            df["doc_freq"] = 1

    # entropy / coherence fallbacks
    if "mean_entropy" not in df.columns and "entropy" in df.columns:
        df["mean_entropy"] = df["entropy"]
    if "mean_coherence" not in df.columns and "coherence" in df.columns:
        df["mean_coherence"] = df["coherence"]

    metrics: Dict[str, Dict[str, Any]] = {}

    for el, g in df.groupby("element"):
        metrics[str(el)] = {
            "tf": float(g["tf"].astype(float).sum()),
            "doc_freq": int(g["doc_freq"].astype(int).max()),
            "mean_entropy": float(g["mean_entropy"].astype(float).mean())
            if "mean_entropy" in g.columns else 0.0,
            "mean_coherence": float(g["mean_coherence"].astype(float).mean())
            if "mean_coherence" in g.columns else 0.0,
        }

    return metrics


def _load_regimes(elements_csv: str) -> Dict[str, Dict[str, float]]:
    """Load token-level regimes from lsa_field.json (optional)."""
    base = os.path.dirname(os.path.abspath(elements_csv))
    lsa_path = os.path.join(base, "lsa_field.json")

    if not os.path.exists(lsa_path):
        return {}

    try:
        with open(lsa_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    raw = data.get("element_regimes") or {}
    cleaned: Dict[str, Dict[str, float]] = {}

    for tok, reg in raw.items():
        if not isinstance(reg, dict):
            continue
        cleaned[tok] = {
            "info": _safe_float(reg.get("info", 0.0)),
            "misinfo": _safe_float(reg.get("misinfo", 0.0)),
            "disinfo": _safe_float(reg.get("disinfo", 0.0)),
        }

    return cleaned


def _classify_polarity(info: float, mis: float, dis: float) -> str:
    """Convert regime scores → coarse semantic polarity."""
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
    """UI-friendly truncated label."""
    t = token.strip()
    if len(t) <= max_len:
        return t
    cut = t[: max_len - 3]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "..."


def _choose_example_targets(
    metrics: Dict[str, Dict[str, Any]],
    max_targets: int = 500,
    min_tf: float = 1.0,
) -> List[str]:
    """Pick elements to search for example spans."""
    items = list(metrics.items())
    items.sort(key=lambda kv: _safe_float(kv[1].get("tf", 0.0)), reverse=True)

    out: List[str] = []
    for el, m in items:
        tf = _safe_float(m.get("tf", 0.0))
        if tf < min_tf:
            continue
        out.append(el)
        if len(out) >= max_targets:
            break
    return out


def _build_examples(
    spans: List[dict],
    token_map: Dict[str, str],
    metrics: Dict[str, Dict[str, Any]],
    max_examples: int = 3,
) -> Dict[str, List[str]]:
    """Extract short example spans via regex matching."""
    examples = {el: [] for el in token_map.keys()}
    if not spans:
        return examples

    targets = set(_choose_example_targets(metrics))
    compiled: Dict[str, re.Pattern] = {}

    for el in targets:
        tok = token_map.get(el, "").strip()
        if len(tok) < 2:
            continue
        try:
            compiled[el] = re.compile(r"\b" + re.escape(tok) + r"\b", re.IGNORECASE)
        except re.error:
            continue

    for s in spans:
        txt = str(s.get("text", "") or "").strip()
        if not txt:
            continue
        low = txt.lower()
        for el, pat in compiled.items():
            if len(examples[el]) >= max_examples:
                continue
            if pat.search(low):
                examples[el].append(txt)

    return examples


# -----------------------------------------------------------------------------#
# Main builder
# -----------------------------------------------------------------------------#

def build_element_descriptions(
    elements_csv: str,
    spans: List[dict] | None,
    out_dir: str,
    emit: Callable[[str, Dict[str, Any]], None] = DEFAULT_EMIT,
) -> None:
    """Main entry point. Produces element_descriptions.json."""

    if not os.path.exists(elements_csv):
        _log(emit, "warn", "hilbert_elements.csv not found", path=elements_csv)
        return

    try:
        df = pd.read_csv(elements_csv)
    except Exception as e:
        _log(emit, "warn", "Failed to load hilbert_elements.csv", error=str(e))
        return

    if df.empty:
        _log(emit, "warn", "hilbert_elements.csv empty")
        return

    os.makedirs(out_dir, exist_ok=True)

    # 1) token map + metrics
    try:
        token_map = _infer_token_map(df)
    except Exception as e:
        _log(emit, "warn", "Cannot infer token map", error=str(e))
        return

    metrics = _aggregate_metrics(df)
    _log(emit, "info", "Aggregated element metrics", n_elements=len(metrics))

    # 2) regimes
    regimes = _load_regimes(elements_csv)
    if regimes:
        _log(emit, "info", "Loaded regime profiles", n=len(regimes))

    # 3) examples
    spans = spans or []
    examples_map = _build_examples(spans, token_map, metrics)

    # 4) build JSON records
    out: Dict[str, Dict[str, Any]] = {}

    for el in sorted(token_map.keys(), key=str):
        token = token_map[el]
        m = metrics.get(el, {})

        tf = _safe_float(m.get("tf", 0.0))
        dfreq = _safe_int(m.get("doc_freq", 0))
        me = _safe_float(m.get("mean_entropy", 0.0))
        mc = _safe_float(m.get("mean_coherence", 0.0))

        reg = regimes.get(token, {})
        info = _safe_float(reg.get("info", 0.0))
        mis = _safe_float(reg.get("misinfo", 0.0))
        dis = _safe_float(reg.get("disinfo", 0.0))
        polarity = _classify_polarity(info, mis, dis)

        summary = (
            f"Element {el} ('{token}') models a recurrent semantic unit. "
            f"It appears with term frequency {tf:.1f} across {dfreq} documents, "
            f"with mean entropy {me:.3f} and coherence {mc:.3f}. "
        )

        if polarity == "informationally stable":
            summary += "Its regime profile indicates predominantly stable informational usage."
        elif polarity == "noisy / misinfo-prone":
            summary += "Its regime profile suggests noisy or misinfo-prone usage."
        elif polarity == "highly polarized / disinfo-prone":
            summary += "Its regime profile indicates polarized or disinfo-prone behavior."
        else:
            summary += "Its regime profile appears approximately neutral."

        out[el] = {
            "element": el,
            "token": token,
            "label": _short_label(token),
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
                "doc_freq": dfreq,
                "mean_entropy": me,
                "mean_coherence": mc,
            },
        }

    # write file
    out_path = os.path.join(out_dir, "element_descriptions.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        _log(emit, "info", "Wrote element_descriptions.json", path=out_path)
        emit("artifact", {"path": out_path, "kind": "element_descriptions"})
    except Exception as e:
        _log(emit, "warn", "Failed to write element_descriptions.json", error=str(e))
