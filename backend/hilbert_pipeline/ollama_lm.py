# backend/hilbert_pipeline/ollama_lm.py
from __future__ import annotations
import os
import math
import json
from pathlib import Path
from typing import Dict, Any, List

import requests

# ---------------------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# Recommended default for strong perplexity + consistency:
#   - mistral
#   - llama3
#   - llama3.1:8b
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

def _ollama_openai_url() -> str:
    """
    Returns the OpenAI-compatible completions endpoint for Ollama.
    """
    base = OLLAMA_URL.rstrip("/")
    return f"{base}/v1/completions"

# ---------------------------------------------------------------------
# Low-level scorer: uses the OpenAI-compatible API for logprobs
# ---------------------------------------------------------------------
def score_text_tokens(
    text: str,
    model: str | None = None,
    timeout: int = 180,
) -> Dict[str, Any]:
    """
    Score a full text string using Ollama's OpenAI-compatible API.

    Requirements for logprob scoring:
      - echo=True (so prompt tokens are included)
      - max_tokens=0 (we aren't generating anything)
      - logprobs=N (we ask for per-token logprobs)

    NOTE: The OpenAI-compatible schema expects:
        "prompt": "the text"   (string)
    or:
        "messages": [{"role":"user", "content": "..."}]

    But for logprobs you're required to use "prompt" mode.
    """
    if model is None:
        model = OLLAMA_MODEL

    url = _ollama_openai_url()

    payload = {
        "model": model,
        "prompt": text,
        "max_tokens": 0,
        "temperature": 0.0,
        "echo": True,
        "logprobs": 1,
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as exc:
        return {
            "error": str(exc),
            "model": model,
            "choices": [{"logprobs": {"tokens": [], "token_logprobs": []}}],
        }

# ---------------------------------------------------------------------
# Perplexity computation
# ---------------------------------------------------------------------
def compute_perplexity_from_logprobs(logprobs: List[float]) -> float:
    """
    Perplexity = exp( -mean(log p_i) ).

    Ollama returns natural-log logprobs, so this formula is correct.
    """
    if not logprobs:
        return float("nan")
    # Filter out None values just in case.
    clean = [lp for lp in logprobs if lp is not None]
    if not clean:
        return float("nan")
    avg_logprob = sum(clean) / len(clean)
    return math.exp(-avg_logprob)

# ---------------------------------------------------------------------
# High-level scorer: returns perplexity + basic metadata
# ---------------------------------------------------------------------
def score_text_perplexity(
    text: str,
    model: str | None = None,
    timeout: int = 180,
) -> Dict[str, Any]:

    data = score_text_tokens(text, model=model, timeout=timeout)
    if "error" in data:
        return {
            "model": model,
            "n_tokens": 0,
            "perplexity": float("nan"),
            "avg_logprob": None,
            "error": data["error"],
        }

    # Ollama returns "choices": [{"logprobs": {...}}]
    choice = data.get("choices", [{}])[0]
    logprob_block = choice.get("logprobs", {}) or {}

    token_logprobs = logprob_block.get("token_logprobs") or []
    tokens = logprob_block.get("tokens") or []

    perplexity = compute_perplexity_from_logprobs(token_logprobs)

    return {
        "model": data.get("model", model),
        "n_tokens": len(tokens),
        "perplexity": perplexity,
        "avg_logprob": (
            sum(token_logprobs) / len(token_logprobs)
            if token_logprobs else None
        ),
        "tokens": tokens,
    }

# ---------------------------------------------------------------------
# Hilbert pipeline entry-point
# ---------------------------------------------------------------------
def compute_corpus_perplexity(
    corpus_text: str,
    out_path,
    *,
    model: str | None = None,
) -> dict:
    """
    Compute perplexity for a block of corpus text and write lm_metrics.json.
    """
    out_path = Path(out_path)
    result = score_text_perplexity(corpus_text, model=model)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
