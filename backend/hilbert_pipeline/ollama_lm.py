# =============================================================================
# hilbert_pipeline/ollama_lm.py — Local LM Perplexity Layer (v3.1)
# =============================================================================
"""
Local LM Perplexity Layer (Ollama)
==================================

This module provides a minimal but robust wrapper around **Ollama's
OpenAI-compatible API** for computing:

- token-level log-probabilities  
- average log-probability over a passage  
- perplexity for a text span or full corpus  
- LM-based diagnostics for Hilbert pipeline runs

It is intentionally lightweight and does *not* introduce any dependency on
OpenAI APIs or cloud infrastructure. When the Ollama server is unavailable or
misconfigured, the functions fail gracefully with explicit error fields.

Motivation
----------

The **Hilbert Information Detection Tool (HIDT)** uses perplexity as:

- a measure of linguistic “regularity” of the corpus,
- an additional signal for epistemic irregularity,
- a cross-check on span entropy fields from the LSA layer.

This module computes perplexity directly via:

.. math::

    \mathrm{PPL} = \exp\bigl(- \frac{1}{N} \sum_i \log p_i \bigr)

where Ollama’s OpenAI-compatible completions API provides token-level ``logprobs``.

Environment and Defaults
------------------------

The module respects two environment variables:

``OLLAMA_URL``:
    Base URL of the Ollama server.  
    Defaults to ``http://localhost:11434``.

``OLLAMA_MODEL``:
    Name of the LM model to request.  
    Defaults to ``"llama3"``.  
    Recommended: ``"mistral"``, ``"llama3"``, ``"llama3.1:8b"``.

Outputs
-------

- **Perplexity JSON** written to a path you provide (``lm_metrics.json``).
- In-memory dictionaries for orchestrator and dashboard integration.

Public API
----------

.. autofunction:: score_text_tokens  
.. autofunction:: compute_perplexity_from_logprobs  
.. autofunction:: score_text_perplexity  
.. autofunction:: compute_corpus_perplexity

"""

from __future__ import annotations

import os
import math
import json
from pathlib import Path
from typing import Dict, Any, List

import requests


# =============================================================================
# Ollama configuration
# =============================================================================

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
"""
Base URL of the Ollama server. If unset, defaults to a local instance.

Example:
    export OLLAMA_URL=http://localhost:11434
"""

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
"""
Default model used for perplexity scoring if none is provided explicitly.
"""


def _ollama_openai_url() -> str:
    """
    Construct the **OpenAI-compatible completions endpoint** for an Ollama server.

    Returns
    -------
    str
        Full URL for ``POST /v1/completions``.
    """
    base = OLLAMA_URL.rstrip("/")
    return f"{base}/v1/completions"


# =============================================================================
# Low-level scoring (OpenAI-compatible API call)
# =============================================================================

def score_text_tokens(
    text: str,
    model: str | None = None,
    timeout: int = 180,
) -> Dict[str, Any]:
    """
    Request per-token log-probabilities for a text string using Ollama.

    This is the lowest-level function and directly mirrors the OpenAI-style
    completions endpoint. It **does not generate text**, because:

    - ``max_tokens = 0`` forces evaluation-only mode  
    - ``echo = True`` ensures returned tokens include the prompt tokens  
    - ``logprobs = 1`` requests log probabilities for each token

    Parameters
    ----------
    text : str
        The full text to score.
    model : str, optional
        Name of the Ollama model to use. Defaults to :data:`OLLAMA_MODEL`.
    timeout : int
        Network timeout for the HTTP request.

    Returns
    -------
    dict
        The raw JSON response from Ollama, or an error container with fields:

        ::

            {
              "error": "...",
              "model": "...",
              "choices": [{"logprobs": {"tokens": [], "token_logprobs": []}}]
            }

    Notes
    -----
    - Ollama's OpenAI-compatible mode requires ``prompt`` rather than ``messages``.
    - Log-probabilities are returned in **natural log** units.
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
        return resp.json()
    except Exception as exc:
        # Robust fallback – structured error block
        return {
            "error": str(exc),
            "model": model,
            "choices": [{"logprobs": {"tokens": [], "token_logprobs": []}}],
        }


# =============================================================================
# Perplexity computation
# =============================================================================

def compute_perplexity_from_logprobs(logprobs: List[float]) -> float:
    """
    Compute perplexity from raw token-level log-probabilities.

    Formula:
        ``PPL = exp( -mean(log p_i) )``

    Parameters
    ----------
    logprobs : list of float
        Natural-log probabilities; ``None`` values are ignored.

    Returns
    -------
    float
        Perplexity value, or NaN if no usable probabilities are found.
    """
    if not logprobs:
        return float("nan")

    clean = [lp for lp in logprobs if lp is not None]
    if not clean:
        return float("nan")

    avg_logprob = sum(clean) / len(clean)
    return math.exp(-avg_logprob)


# =============================================================================
# High-level scoring (user-friendly)
# =============================================================================

def score_text_perplexity(
    text: str,
    model: str | None = None,
    timeout: int = 180,
) -> Dict[str, Any]:
    """
    Convenience function returning perplexity plus token metadata.

    Parameters
    ----------
    text : str
        Full text to evaluate.
    model : str, optional
        Override default model name.
    timeout : int
        Network timeout (seconds).

    Returns
    -------
    dict
        Dictionary containing fields:

        - ``model``  
        - ``n_tokens``  
        - ``perplexity``  
        - ``avg_logprob``  
        - ``tokens``  
        - ``error`` (optional)
    """
    data = score_text_tokens(text, model=model, timeout=timeout)

    # If the low-level request failed, propagate a structured error
    if "error" in data:
        return {
            "model": model,
            "n_tokens": 0,
            "perplexity": float("nan"),
            "avg_logprob": None,
            "error": data["error"],
        }

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


# =============================================================================
# Hilbert Pipeline Entry Point
# =============================================================================

def compute_corpus_perplexity(
    corpus_text: str,
    out_path,
    *,
    model: str | None = None,
) -> dict:
    """
    Compute perplexity for an entire corpus string and write ``lm_metrics.json``.

    This is the pipeline-facing entry point used by the orchestrator.

    Parameters
    ----------
    corpus_text : str
        Full corpus text concatenated or preprocessed upstream.
    out_path : str or Path
        Destination path for the resulting JSON.
    model : str, optional
        Override LM model (defaults to environment or :data:`OLLAMA_MODEL`).

    Returns
    -------
    dict
        The computed perplexity result dictionary.

    Side Effects
    ------------
    Writes the JSON structure to ``out_path``.

    Example JSON
    ------------

    .. code-block:: json

        {
            "model": "llama3",
            "n_tokens": 4821,
            "perplexity": 12.7,
            "avg_logprob": -2.54,
            "tokens": [...]
        }
    """
    out_path = Path(out_path)

    result = score_text_perplexity(corpus_text, model=model)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result
