# backend/hilbert_agent.py
#
# Hilbert Agent - lightweight tool-using backend for the Hilbert Assistant.
#
# This module sits *behind* the /api/v1/agent_chat endpoint. It:
#   1. Receives a chat history (list of {role, content}).
#   2. Asks a local Ollama model to choose a TOOL or respond directly.
#   3. If a tool is chosen, executes it against the real repo.
#   4. Sends the tool result back to the model to generate a Markdown answer.
#
# Initial tool set:
#   - list_top_level: top-level directories of the repo.
#   - list_dir: list files/dirs in a given subdirectory.
#   - read_file: return a snippet of a file.
#   - search_text: simple grep-style search across the repo.
#
# The design is intentionally simple and stateless; the "memory" is the
# conversation the frontend sends us each time.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import textwrap
import traceback

from ollama_client import call_ollama


# ---------------------------------------------------------------------------
# Repo and filesystem helpers
# ---------------------------------------------------------------------------

BACKEND_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_ROOT.parent

IGNORE_DIRS = {
    ".git",
    ".github",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
    "results",
}

BINARY_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".whl",
    ".log",
}

TEXT_EXT_WHITELIST = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".md",
    ".txt",
    ".toml",
    ".yml",
    ".yaml",
    ".html",
    ".css",
}


MAX_FILE_BYTES = 64 * 1024
MAX_SEARCH_HITS = 40


def _is_ignored_dir(path: Path) -> bool:
    parts = set(p.name for p in path.resolve().parts)
    return any(name in IGNORE_DIRS for name in parts)


def _safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def _safe_read_text(path: Path, max_bytes: int = MAX_FILE_BYTES) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            data = f.read(max_bytes + 1)
    except Exception as exc:
        return f"<<error reading {_safe_rel(path)}: {exc}>>"

    if len(data) > max_bytes:
        data = data[:max_bytes] + "\n\n<<truncated after " + str(max_bytes) + " bytes>>"
    return data


# ---------------------------------------------------------------------------
# Tools - pure Python, no model calls
# ---------------------------------------------------------------------------

def tool_list_top_level() -> Dict[str, Any]:
    dirs = []
    files = []
    for child in REPO_ROOT.iterdir():
        if child.is_dir():
            if child.name in IGNORE_DIRS:
                continue
            dirs.append(child.name)
        else:
            files.append(child.name)
    return {"root": str(REPO_ROOT), "dirs": sorted(dirs), "files": sorted(files)}


def tool_list_dir(path: str) -> Dict[str, Any]:
    target = (REPO_ROOT / path).resolve()
    if not target.exists():
        return {"error": f"path does not exist: {path}"}
    if not target.is_dir():
        return {"error": f"path is not a directory: {path}"}
    if _is_ignored_dir(target):
        return {"error": f"path is ignored by agent rules: {path}"}

    dirs = []
    files = []
    for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
        if child.is_dir():
            if child.name in IGNORE_DIRS:
                continue
            dirs.append(child.name)
        else:
            files.append(child.name)

    return {"root": str(REPO_ROOT), "directory": _safe_rel(target), "dirs": dirs, "files": files}


def tool_read_file(path: str, max_bytes: int = MAX_FILE_BYTES) -> Dict[str, Any]:
    target = (REPO_ROOT / path).resolve()
    if not target.exists():
        return {"error": f"file does not exist: {path}"}
    if not target.is_file():
        return {"error": f"path is not a file: {path}"}
    if _is_ignored_dir(target.parent):
        return {"error": f"file is under an ignored directory: {path}"}

    if target.suffix.lower() in BINARY_EXTS:
        return {"error": f"file appears to be binary (extension {target.suffix}); refusing to dump."}

    text = _safe_read_text(target, max_bytes=max_bytes)
    return {
        "root": str(REPO_ROOT),
        "path": _safe_rel(target),
        "bytes_limit": max_bytes,
        "content": text,
    }


def tool_search_text(query: str, max_hits: int = MAX_SEARCH_HITS) -> Dict[str, Any]:
    """Very small grep-style search across the repo."""
    query = query.strip()
    if not query:
        return {"error": "empty query"}

    hits: List[Dict[str, Any]] = []

    for path in REPO_ROOT.rglob("*"):
        if len(hits) >= max_hits:
            break
        if not path.is_file():
            continue
        if _is_ignored_dir(path.parent):
            continue
        if path.suffix and path.suffix.lower() not in TEXT_EXT_WHITELIST:
            continue

        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for lineno, line in enumerate(f, start=1):
                    if query in line:
                        hits.append(
                            {
                                "path": _safe_rel(path),
                                "line": lineno,
                                "snippet": line.rstrip()[:200],
                            }
                        )
                        if len(hits) >= max_hits:
                            break
        except Exception:
            continue

    return {
        "root": str(REPO_ROOT),
        "query": query,
        "max_hits": max_hits,
        "hits": hits,
        "total_hits": len(hits),
    }


# Mapping from action name -> Python function and arg spec
TOOL_REGISTRY = {
    "list_top_level": tool_list_top_level,
    "list_dir": tool_list_dir,
    "read_file": tool_read_file,
    "search_text": tool_search_text,
}


# ---------------------------------------------------------------------------
# Planning & answer generation with Ollama
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are Hilbert Agent, a precise software assistant grounded in a real repository.

    You CANNOT invent files, directories or code. When you need repository facts,
    you must choose one of the following tools:

    1. list_top_level
       args: {}
       - List top-level directories and files of the repo.

    2. list_dir
       args: { "path": "string, relative to repo root (e.g. 'backend', 'frontend/src')" }
       - List contents of a directory.

    3. read_file
       args: { "path": "string, relative to repo root", "max_bytes": "optional int" }
       - Read a single file (text only). Use this for targeted inspection.

    4. search_text
       args: { "query": "string", "max_hits": "optional int" }
       - Search for a literal string across the repo.

    Your job is to decide *one* of:
      - an action = "final" if you can answer directly from previous context,
      - or an action = the name of a tool to call.

    You MUST respond with a single valid JSON object, **no extra text**, of the form:

    {
      "action": "final" | "list_top_level" | "list_dir" | "read_file" | "search_text",
      "args": { ... },        // omit or {} if no arguments needed
      "final_answer": "..."   // required only when action == "final"
    }

    Rules:
    - Prefer tools over guessing.
    - Use relative paths only, never absolute ones.
    - Use small, focused read_file calls (one file at a time).
    - For broad questions like "what are the top-level directories", use list_top_level.
    """
)


def _format_conversation(messages: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for m in messages:
        role = m.get("role") or "user"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role.upper()}: {content}")
    return "\n\n".join(lines)


def _safe_json_parse(text: str) -> Dict[str, Any]:
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try trimming to outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass

    raise ValueError(f"Planner returned non-JSON text: {text[:200]!r}")


def plan_action(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    conv = _format_conversation(messages)
    prompt = PLANNER_SYSTEM_PROMPT + "\n\nConversation so far:\n\n" + conv + "\n\nYour JSON decision:"

    raw = call_ollama(prompt, model="llama3")
    return _safe_json_parse(raw)


def _call_tool(action: str, args: Dict[str, Any]) -> Dict[str, Any]:
    func = TOOL_REGISTRY.get(action)
    if not func:
        return {"error": f"Unknown tool: {action}"}

    try:
        # Call with only kwargs that the function supports, but we keep it simple:
        if action == "list_top_level":
            result = func()
        elif action == "list_dir":
            result = func(str(args.get("path", "")))
        elif action == "read_file":
            path = str(args.get("path", ""))
            max_bytes = int(args.get("max_bytes", MAX_FILE_BYTES))
            result = func(path, max_bytes=max_bytes)
        elif action == "search_text":
            query = str(args.get("query", ""))
            max_hits = int(args.get("max_hits", MAX_SEARCH_HITS))
            result = func(query, max_hits=max_hits)
        else:
            result = {"error": f"Tool not wired: {action}"}
    except Exception as exc:
        result = {
            "error": f"Exception while running tool {action}: {exc}",
            "traceback": traceback.format_exc(),
        }

    return {
        "tool": action,
        "args": args,
        "result": result,
    }


def build_answer_from_tool(
    messages: List[Dict[str, str]],
    tool_call: Dict[str, Any],
) -> str:
    """
    Ask the model to produce a Markdown answer grounded in the tool output.
    """
    user_last = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_last = m.get("content", "")
            break

    prompt = textwrap.dedent(
        f"""
        You are Hilbert Agent. You have just called a repository-inspection tool
        to help answer the latest user question.

        Latest user message:
        ---
        {user_last}
        ---

        Tool call (JSON):
        {json.dumps(tool_call, indent=2, ensure_ascii=False)}

        Using ONLY the information in the tool result and the conversation
        context, write a helpful Markdown answer. You may include bullet lists,
        code blocks, and short suggestions for next steps. Do NOT invent files
        or directories that are not present in the tool result.
        """
    ).strip()

    return call_ollama(prompt, model="llama3")


# ---------------------------------------------------------------------------
# Public entry point used by app.py
# ---------------------------------------------------------------------------

def run_hilbert_agent(messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
    """
    Main orchestration entry.

    Returns:
        reply_markdown: str - what the frontend should display.
        debug_meta: dict - small JSON blob about what the agent did.
    """
    if not messages:
        return (
            "Hi - I am the Hilbert Agent. Ask me about the repository layout, "
            "specific files, or how the Hilbert pipeline works.",
            {"status": "empty_conversation"},
        )

    try:
        plan = plan_action(messages)
    except Exception as exc:
        # Planner failed - fall back to a simple, direct answer.
        fallback = (
            "The planning layer failed while deciding which tool to call.\n\n"
            f"Technical details:\n```\n{exc}\n```"
        )
        return fallback, {"status": "planner_error", "error": str(exc)}

    action = plan.get("action", "final")
    args = plan.get("args") or {}

    if action == "final":
        text = plan.get("final_answer") or ""
        if not text.strip():
            text = "I could not infer a useful answer from the current context."
        return text, {"status": "final_from_planner"}

    # Tool branch
    tool_call = _call_tool(action, args)
    answer = build_answer_from_tool(messages, tool_call)
    meta = {
        "status": "tool_used",
        "action": action,
        "args": args,
        "tool_result_summary": {
            "has_error": bool(tool_call["result"].get("error")),
        },
    }
    return answer, meta
