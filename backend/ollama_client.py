import os
import json
from typing import Any, Dict
import requests

# Base Ollama host; override via env if needed
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Default model – we know you pulled `llama3`
DEFAULT_MODEL = os.getenv("HILBERT_OLLAMA_MODEL", "llama3")


def _post_json(path: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    """
    Helper for POSTing JSON to the local Ollama server.

    Raises RuntimeError with a detailed message if the request fails.
    """
    url = f"{OLLAMA_BASE}{path}"
    resp = requests.post(url, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except Exception as exc:
        body = ""
        try:
            body = resp.text[:400]
        except Exception:
            pass
        raise RuntimeError(f"Ollama request failed ({url}): {exc} :: {body}")
    return resp.json()


def _is_model_not_found_error(err: Exception) -> bool:
    msg = str(err)
    return "model" in msg.lower() and "not found" in msg.lower()


def call_ollama(prompt: str, model: str | None = None) -> str:
    """
    Send a prompt to the local Ollama server.

    Strategy:
    - First try chat API: POST /api/chat {model, messages, stream:false}
    - On a 404 for /api/chat, fall back to /api/generate.
    - If the chosen model does not exist, fall back to `mistral`.
    """
    chosen_model = model or DEFAULT_MODEL

    def _chat(model_name: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        return _post_json("/api/chat", payload)

    def _generate(model_name: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
        }
        return _post_json("/api/generate", payload)

    # 1) Try chat endpoint with chosen model
    try:
        data = _chat(chosen_model)
    except RuntimeError as exc:
        msg = str(exc)
        # If /api/chat itself is missing, fall back to /api/generate
        if "404" in msg and "/api/chat" in msg:
            try:
                data = _generate(chosen_model)
            except RuntimeError as exc2:
                # If the model is missing, try mistral
                if _is_model_not_found_error(exc2) and chosen_model != "mistral":
                    data = _generate("mistral")
                else:
                    raise
        # If the model is missing on /api/chat, retry with mistral
        elif _is_model_not_found_error(exc) and chosen_model != "mistral":
            try:
                data = _chat("mistral")
            except RuntimeError:
                data = _generate("mistral")
        else:
            raise

    # Normalise reply
    if isinstance(data, dict):
        if "message" in data and isinstance(data["message"], dict):
            content = data["message"].get("content")
            if isinstance(content, str):
                return content
        if "response" in data and isinstance(data["response"], str):
            return data["response"]

    # Fallback: stringify whole object
    return json.dumps(data, indent=2, ensure_ascii=False)
