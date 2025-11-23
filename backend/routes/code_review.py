"""
backend/routes/code_review.py

Tier 3 - Project-aware code review endpoint for the Hilbert Assistant.

This endpoint uses:

  - find_repo_root() to locate the project root.
  - make_snapshot() to build a focused view of the codebase.
  - build_prompt() to construct a rich LLM prompt.
  - call_ollama() to delegate reasoning to the local model.

It is designed to be called from the frontend when the user wants a
"review this project / file / feature" style interaction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException, Query

from .ollama_client import call_ollama
from .code_searcher import build_prompt, find_repo_root, make_snapshot

router = APIRouter(prefix="/api/v1", tags=["code-review"])


@router.post("/code_review")
async def code_review_endpoint(
    focus: Optional[str] = Query(
        default=None,
        description="Optional free-text hint like 'backend app.py' or 'graph snapshot generation'",
    ),
    question: Optional[str] = Query(
        default=None,
        description="What you want the assistant to do with the code snapshot.",
    ),
    payload: Dict[str, Any] = Body(
        default={},
        description="Optional JSON body; if it contains 'question' or 'focus', those override the query params.",
    ),
) -> Dict[str, Any]:
    """
    Run a repository-aware review using the local LLM.

    The endpoint is intentionally simple:
      1. Discover repo root from this file.
      2. Build a small file snapshot guided by `focus` / `question`.
      3. Build a prompt with the snapshot embedded.
      4. Ask the local model and return its reply plus some debug info.
    """
    # Allow JSON body to override query parameters for future flexibility
    body_focus = payload.get("focus")
    body_question = payload.get("question")

    if isinstance(body_focus, str) and body_focus.strip():
        focus = body_focus.strip()
    if isinstance(body_question, str) and body_question.strip():
        question = body_question.strip()

    # Reasonable default if neither focus nor question is supplied
    if not focus and not question:
        focus = "overall structure of the Hilbert backend and orchestrator"
        question = (
            "Give a high-level code review of this snapshot and suggest the top 3 "
            "refactors that would improve clarity and reliability."
        )

    # 1. Repo root
    this_file = Path(__file__).resolve()
    repo_root = find_repo_root(this_file)

    # 2. Build snapshot
    try:
        snapshot = make_snapshot(repo_root, focus_query=focus or question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build repository snapshot: {exc}",
        ) from exc

    # 3. Prompt assembly
    prompt = build_prompt(snapshot, focus_query=focus, question=question)

    # 4. LLM call
    try:
        reply = call_ollama(prompt)
    except Exception as exc:
        # Surface a clean error to the caller but keep the backend log noisy.
        raise HTTPException(
            status_code=500,
            detail=f"Local LLM call failed: {exc}",
        ) from exc

    return {
        "root": str(repo_root),
        "focus": focus,
        "question": question,
        "n_files": len(snapshot),
        "reply": reply,
        "snapshot": snapshot,
    }
