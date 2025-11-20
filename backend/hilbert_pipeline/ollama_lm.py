# backend/hilbert_pipeline/ollama_lm.py
from __future__ import annotations
import os
import math
import json
from pathlib import Path
from typing import Dict, Any, List

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")  # or llama3.2, etc

def _ollama_openai_url() -> str:
    # OpenAI compatible completions endpoint
    # Docs: Ollama OpenAI compatibility, including logprobs.:contentReference[oaicite:1]{index=1}
    base = OLLAMA_URL.rstrip("/")
    return f"{base}/v1/completions"

def score_text_tokens(text: str, model: str | None = None) -> Dict[str, Any]:
    """
    Ask Ollama to score a piece of text and return per token logprobs.

    We use the OpenAI compatible completion API with:
    - echo=True so prompt tokens are returned
    - max_tokens=0 so we do not actually generate extra text
    - logprobs=1 to get logprobs per token
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

    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()

def compute_perplexity_from_logprobs(logprobs: List[float]) -> float:
    """
    Standard definition:
        ppl = exp( - (1/N) * sum_i log p_i )
    Assuming logprobs are natural log (Ollama follows this in its logprobs support).
    """
    if not logprobs:
        return float("nan")
    avg_logprob = sum(lp for lp in logprobs if lp is not None) / len(logprobs)
    return math.exp(-avg_logprob)

def score_text_perplexity(text: str, model: str | None = None) -> Dict[str, Any]:
    data = score_text_tokens(text, model=model)
    choice = data["choices"][0]
    lp = choice.get("logprobs", {})
    token_logprobs = lp.get("token_logprobs") or []
    tokens = lp.get("tokens") or []

    perplexity = compute_perplexity_from_logprobs(token_logprobs)

    return {
        "model": data.get("model", model),
        "n_tokens": len(tokens),
        "perplexity": perplexity,
        "avg_logprob": (
            sum(token_logprobs) / len(token_logprobs) if token_logprobs else None
        ),
    }

# backend/hilbert_pipeline/ollama_lm.py

from pathlib import Path
import json

def compute_corpus_perplexity(
    corpus_text: str,
    out_path,
    *,
    model: str | None = None,
) -> dict:
    """
    High level helper for the Hilbert pipeline:
    - Takes a chunk of corpus text (already concatenated / sampled)
    - Scores it with Ollama
    - Writes lm_metrics.json
    """
    # <-- This line is the critical fix:
    out_path = Path(out_path)

    result = score_text_perplexity(corpus_text, model=model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
