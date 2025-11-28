"""
Corpus Loader
=============

Normalizes a raw corpus directory into a clean text-only directory suitable
for the LSA stage of the Hilbert Information Pipeline.

Responsibilities
----------------
- Recursively traverse the raw corpus directory
- Convert PDFs to text (via :func:`pdf_to_text`)
- Copy text-like files as-is
- Preserve directory layout
- Enforce optional max_docs limit
- Emit structured log events for UI + DB insight

This module is deliberately lightweight and stateless. All heavy lifting
(LSA, molecules, stability, etc.) happens downstream.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any

from .pdf_utils import pdf_to_text


# Event function type
EmitFn = Callable[[str, Dict[str, Any]], None]


def load_and_normalize_corpus(
    *,
    corpus_dir: str,
    out_dir: str,
    max_docs: Optional[int] = None,
    emit: Optional[EmitFn] = None,
) -> str:
    """
    Load and normalize the raw corpus.

    Parameters
    ----------
    corpus_dir : str
        Source directory containing PDFs, TXT, JSONL, log files, and arbitrary data.

    out_dir : str
        Destination folder where all *normalized* `.txt` files will be written.

    max_docs : int, optional
        Hard limit on number of documents processed.
        If None, all files are included.

    emit : callable, optional
        Structured logging function:
            emit(kind: str, payload: dict)
        If None, logging is skipped.

    Returns
    -------
    str
        Absolute path to the normalized corpus directory.

    Notes
    -----
    - Directory structure is preserved (subdirectories mirrored into `out_dir`).
    - PDFs are converted to `.txt`.
    - Unknown or unreadable files are skipped with a warning.
    """

    # Resolve paths
    src_root = Path(corpus_dir).resolve()
    dst_root = Path(out_dir).resolve()

    if emit:
        emit("log", {
            "level": "info",
            "msg": "Normalizing corpus...",
            "corpus_dir": str(src_root),
            "out_dir": str(dst_root),
            "max_docs": max_docs,
        })

    # Reset output directory
    shutil.rmtree(dst_root, ignore_errors=True)
    dst_root.mkdir(parents=True, exist_ok=True)

    # Collect all files (sorted for determinism)
    paths: List[Path] = [p for p in src_root.rglob("*") if p.is_file()]
    paths.sort()

    if max_docs is not None:
        paths = paths[:max_docs]

    # Process documents
    for p in paths:
        rel = p.relative_to(src_root)
        target_dir = dst_root / rel.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        ext = p.suffix.lower()

        # PDF → text
        if ext == ".pdf":
            try:
                text = pdf_to_text(str(p))
                if not text.strip():
                    if emit:
                        emit("log", {
                            "level": "warn",
                            "msg": "Empty PDF extracted text",
                            "path": str(p),
                        })
                    continue

                out_path = target_dir / f"{p.stem}.txt"
                out_path.write_text(text, encoding="utf-8")

            except Exception as exc:
                if emit:
                    emit("log", {
                        "level": "warn",
                        "msg": "Failed to convert PDF",
                        "path": str(p),
                        "error": str(exc),
                    })
            continue

        # Text-like files → direct copy
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as fs:
                data = fs.read()

            dest = target_dir / p.name
            dest.write_text(data, encoding="utf-8")

        except Exception as exc:
            if emit:
                emit("log", {
                    "level": "warn",
                    "msg": "Failed to read or copy corpus file",
                    "path": str(p),
                    "error": str(exc),
                })
            continue

    if emit:
        emit("log", {
            "level": "info",
            "msg": "Corpus normalization complete",
            "out_dir": str(dst_root),
            "count": len(paths),
        })

    return str(dst_root)
