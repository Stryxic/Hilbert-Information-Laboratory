import os
import json
from typing import Any, Dict, List, Optional

import requests

"""
Lightweight client for a local Ollama server.

Tier 1 goals:
- Robust model selection with sensible fallbacks.
- Clear, debuggable error messages.
- A single call_ollama() entry point used by the rest of the backend.

The client prefers a chat-style API (/api/chat) but will fall back
to the older generate API (/api/generate) if needed.
"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

# Ordered list of model names we are happy to use.
# The first one that exists on the local Ollama instance will be used.
DEFAULT_MODEL_CANDIDATES: List[str] = [
    os.getenv("OLLAMA_MODEL", "").strip() or "",
    "llama3.1",
    "llama3",
    "llama2",
    "mistral",
]

# Cache of discovered models so we only hit /api/tags once per process.
_AVAILABLE_MODELS: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

class OllamaError(RuntimeError):
    """Domain-specific error for Ollama failures."""


def _request_json(
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    url = f"{OLLAMA_BASE}{path}"
    try:
        if method.upper() == "POST":
            resp = requests.post(url, json=payload or {}, timeout=timeout)
        else:
            resp = requests.get(url, timeout=timeout)
    except Exception as exc:  # network-level error
        raise OllamaError(f"Ollama request failed ({url}): {exc}") from exc

    body_preview = ""
    try:
        body_preview = resp.text[:400]
    except Exception:
        body_preview = ""

    try:
        resp.raise_for_status()
    except Exception as exc:
        raise OllamaError(f"Ollama request failed ({url}): {exc} :: {body_preview}") from exc

    try:
        return resp.json()
    except Exception as exc:
        raise OllamaError(
            f"Ollama returned non-JSON response from {url}: {body_preview}"
        ) from exc


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def _fetch_available_models() -> List[str]:
    """
    Ask the local Ollama daemon which models are installed.

    Returns a list of model names like ["llama3.1", "mistral", ...].
    """
    global _AVAILABLE_MODELS
    if _AVAILABLE_MODELS is not None:
        return _AVAILABLE_MODELS

    try:
        data = _request_json("GET", "/api/tags", None, timeout=10)
    except OllamaError:
        # If we cannot talk to Ollama at all, report an empty list - the caller
        # will surface a clean error.
        _AVAILABLE_MODELS = []
        return _AVAILABLE_MODELS

    models: List[str] = []
    if isinstance(data, dict):
        for m in data.get("models", []) or []:
            # Each entry is typically {"name": "llama3.1", "model": "...", ...}
            if isinstance(m, dict):
                name = m.get("name") or m.get("model")
                if isinstance(name, str):
                    models.append(name)

    _AVAILABLE_MODELS = models
    return models


def _select_model(explicit: Optional[str] = None) -> str:
    """
    Choose a concrete model name to use.

    Preference order:
    1. Explicitly requested model, if installed.
    2. First candidate in DEFAULT_MODEL_CANDIDATES that is installed.
    3. First installed model reported by Ollama.
    4. Raise OllamaError if no models are available.
    """
    available = _fetch_available_models()

    # 1. Explicit request
    if explicit:
        # Some tags may include ":latest" etc - we accept substring matches.
        if explicit in available:
            return explicit
        # Also allow "llama3" to match "llama3:latest" etc.
        for tag in available:
            if tag.startswith(explicit):
                return tag

    # 2. Preferred candidates
    for candidate in DEFAULT_MODEL_CANDIDATES:
        c = candidate.strip()
        if not c:
            continue
        if c in available:
            return c
        for tag in available:
            if tag.startswith(c):
                return tag

    # 3. Any installed model
    if available:
        return available[0]

    # 4. No models installed or daemon unreachable
    raise OllamaError(
        "No Ollama models available - make sure the Ollama daemon is running "
        "and at least one model (for example 'ollama pull llama3') is installed."
    )


# ---------------------------------------------------------------------------
# Public LLM entry point
# ---------------------------------------------------------------------------

def call_ollama(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 768,
) -> str:
    """
    Send a prompt to the local Ollama server and return the assistant text.

    We try the chat endpoint first:
        POST /api/chat
          { "model": "...", "messages": [...], "stream": false }

    If that is unavailable (404), we fall back to:
        POST /api/generate
          { "model": "...", "prompt": "...", "stream": false }

    Parameters
    ----------
    prompt:
        User-facing prompt text, including any context.
    model:
        Optional explicit model name (for example "llama3.1"). If None, we will
        auto-select a suitable installed model.
    system:
        Optional system prompt to prepend.
    temperature:
        Sampling temperature.
    max_tokens:
        Soft limit on number of tokens to generate (Ollama interprets this
        as "num_predict").
    """
    model_name = _select_model(model)

    # First attempt: chat endpoint
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    chat_payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        data = _request_json("POST", "/api/chat", chat_payload)
    except OllamaError as exc:
        msg = str(exc)
        # If the chat endpoint is missing, fall back to generate API.
        if "404" not in msg and "/api/chat" not in msg:
            # Some other error - re-raise.
            raise

        gen_payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": (system + "\n\n" + prompt) if system else prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        data = _request_json("POST", "/api/generate", gen_payload)

    # Normalise reply:
    # - chat: {"message": {"role": "...", "content": "..."}}
    # - generate: {"response": "..."}
    if isinstance(data, dict):
        msg = data.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"]
        if isinstance(data.get("response"), str):
            return data["response"]

    # Fallback: stringify whatever we got
    return json.dumps(data, indent=2, ensure_ascii=False)
