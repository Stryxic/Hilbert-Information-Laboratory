"""
backend/code_searcher.py

Tier 2 - Lightweight repository search and snapshot builder.

This module gives the Hilbert Assistant a project-aware view of the
codebase without doing anything too fancy:

- Find the repository root given an arbitrary starting file.
- Build a small in-memory index of source files under that root.
- Search for files relevant to a natural-language query.
- Package the top matches into a compact "snapshot" that can be fed
  to an LLM as context.

The goal is not to be perfect, but to be predictable and cheap.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Repo discovery
# ---------------------------------------------------------------------------

def find_repo_root(start: Path) -> Path:
    """
    Walk upwards from *start* until we hit what looks like the project root.

    We consider a directory a candidate root if it contains ANY of:
      - a .git directory
      - a pyproject.toml or requirements.txt
      - a 'backend' directory with app.py inside it

    Falls back to the original start.parent if nothing interesting is found.
    """
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent

    for _ in range(8):  # do not walk above 8 levels to avoid surprises
        markers = {
            (cur / ".git").is_dir(),
            (cur / "pyproject.toml").is_file(),
            (cur / "requirements.txt").is_file(),
        }
        backend_app = (cur / "backend" / "app.py").is_file()
        if any(markers) or backend_app:
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent

    return start.parent.resolve()


# ---------------------------------------------------------------------------
# File indexing and search
# ---------------------------------------------------------------------------

DEFAULT_INCLUDE_EXTS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".md",
    ".txt",
    ".tex",
}


@dataclass
class IndexedFile:
    path: Path
    rel_path: str
    size: int
    text: str
    tokens: Sequence[str]


class CodeSearcher:
    """
    Very small, in-memory search index for the repo.

    We deliberately use a simple bag-of-words scoring - this is "good enough"
    to surface obviously relevant files without adding heavy dependencies.
    """

    def __init__(
        self,
        root: Path,
        *,
        max_bytes_per_file: int = 16_000,
        include_exts: Optional[Iterable[str]] = None,
    ) -> None:
        self.root = root.resolve()
        self.max_bytes_per_file = max_bytes_per_file
        self.include_exts = set(include_exts or DEFAULT_INCLUDE_EXTS)
        self._files: List[IndexedFile] = []
        self._indexed = False

    # --- indexing -----------------------------------------------------------

    def _iter_candidate_paths(self) -> Iterable[Path]:
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            if any(part.startswith(".") for part in path.relative_to(self.root).parts):
                continue
            if path.suffix.lower() not in self.include_exts:
                continue
            yield path

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        # very small normaliser - split on non-alnum and lowercase
        tokens: List[str] = []
        cur = []
        for ch in text.lower():
            if ch.isalnum() or ch in {"_", "-"}:
                cur.append(ch)
            else:
                if cur:
                    tokens.append("".join(cur))
                    cur = []
        if cur:
            tokens.append("".join(cur))
        return tokens

    def _index_one(self, path: Path) -> Optional[IndexedFile]:
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

        snippet = raw[: self.max_bytes_per_file]
        rel = str(path.relative_to(self.root))
        tokens = self._tokenise(rel + "\n" + snippet)
        return IndexedFile(
            path=path,
            rel_path=rel,
            size=len(snippet),
            text=snippet,
            tokens=tokens,
        )

    def ensure_indexed(self) -> None:
        if self._indexed:
            return
        files: List[IndexedFile] = []
        for p in self._iter_candidate_paths():
            item = self._index_one(p)
            if item is not None:
                files.append(item)
        self._files = files
        self._indexed = True

    # --- search -------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> List[IndexedFile]:
        """
        Return top_k files ranked by a simple token overlap score.

        This is intentionally crude - the LLM will do the real reasoning.
        """
        self.ensure_indexed()
        if not query.strip():
            # Fallback: just return the first N files in a deterministic order
            return sorted(self._files, key=lambda f: f.rel_path)[:top_k]

        q_tokens = self._tokenise(query)
        if not q_tokens:
            return sorted(self._files, key=lambda f: f.rel_path)[:top_k]

        q_set = set(q_tokens)
        scored: List[Tuple[float, IndexedFile]] = []

        for f in self._files:
            overlap = sum(1.0 for t in f.tokens if t in q_set)
            if overlap <= 0:
                continue

            # Small heuristics:
            # - path matches get a boost
            # - shorter files are preferred (denser information)
            path_bonus = 0.0
            for t in q_tokens:
                if t in f.rel_path.lower():
                    path_bonus += 1.5

            score = overlap + path_bonus + (2_000.0 / (1_000.0 + f.size))
            scored.append((score, f))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:top_k]]


# ---------------------------------------------------------------------------
# Snapshot + prompt helpers
# ---------------------------------------------------------------------------

def make_snapshot(
    root: Path,
    focus_query: Optional[str] = None,
    *,
    max_files: int = 18,
    max_bytes_per_file: int = 8_000,
) -> List[Dict[str, str]]:
    """
    Build a lightweight snapshot of the repo for LLM consumption.

    Returns a list of dicts:
      { "path": "backend/app.py", "content": "<trimmed text>" }

    The snapshot is deliberately small to avoid blowing the context window.
    """
    searcher = CodeSearcher(root, max_bytes_per_file=max_bytes_per_file)
    if focus_query:
        files = searcher.search(focus_query, top_k=max_files)
    else:
        files = searcher.search("backend app hilbert orchestrator", top_k=max_files)

    snapshot: List[Dict[str, str]] = []
    seen: set[str] = set()

    for f in files:
        if f.rel_path in seen:
            continue
        seen.add(f.rel_path)
        snapshot.append(
            {
                "path": f.rel_path,
                "content": f.text,
            }
        )

    return snapshot


def build_prompt(
    snapshot: List[Dict[str, str]],
    focus_query: Optional[str] = None,
    question: Optional[str] = None,
) -> str:
    """
    Turn a snapshot into a single prompt string for the LLM.

    We use a simple fenced-block format that is easy for models to parse.
    """
    if not snapshot:
        core = "The repository snapshot is empty. Reason at a higher level."
    else:
        parts: List[str] = []
        for item in snapshot:
            parts.append(f"# File: {item['path']}\n```text\n{item['content']}\n```")
        core = "\n\n".join(parts)

    task_lines: List[str] = [
        "You are a senior engineer assisting with the Hilbert Information Laboratory codebase.",
        "You will be given a small snapshot of repository files.",
        "Read the snapshot carefully and then answer the user request.",
        "",
    ]

    if focus_query:
        task_lines.append(f"User focus / search query: {focus_query}")
    if question:
        task_lines.append(f"User question: {question}")

    task_lines.extend(
        [
            "",
            "Requirements:",
            "- Be concise but concrete.",
            "- When suggesting code changes, show minimal diffs or function-level rewrites.",
            "- If something is unclear or missing from the snapshot, say so explicitly.",
        ]
    )

    header = "\n".join(task_lines)

    return header + "\n\n--- Repository snapshot ---\n\n" + core
