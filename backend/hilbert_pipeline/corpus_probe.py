# =============================================================================
# hilbert_pipeline/corpus_probe.py - Corpus profiling and LSA seed benchmarking
# =============================================================================
"""
Lightweight corpus profiling and LSA seed benchmarking.

This module provides two main entry points:

- probe_corpus(corpus_dir, results_dir, emit):
    Fast pass over the corpus that:
      * counts files
      * groups by filename "slug"
      * estimates token counts per file/slug
      * normalises LaTeX into a text corpus
      * classifies files into theory / case / other
      * produces corpus_profile.json

- run_lsa_seed_profile(corpus_dir, results_dir, profile, max_seed_docs, emit):
    Runs a small LSA on a sampled subset of files to estimate:
      * docs per second
      * tokens per second
      * rough LSA runtime for the full corpus

The orchestrator uses these outputs as a "Stage 0" meta layer to reason about:
  - corpus complexity
  - batch sizes
  - estimated runtime
before running the heavy stages.
"""

from __future__ import annotations

import os
import re
import json
import random
import shutil
import time
from typing import Any, Dict, List, Optional

from . import DEFAULT_EMIT  # re-exported in hilbert_pipeline.__init__

# We import build_lsa_field via the public API to avoid tight coupling.
try:
    from hilbert_pipeline import build_lsa_field  # type: ignore
except Exception:
    # During local imports this may fail; run_lsa_seed_profile will guard.
    build_lsa_field = None  # type: ignore


# ---------------------------------------------------------------------------
# Utility helpers - generic
# ---------------------------------------------------------------------------

def _slug_from_filename(path: str) -> str:
    """
    Derive a "slug" from the filename.

    Example:
        "trump-proposes-us-takeover-of-17.txt" -> "trump-proposes-us-takeover-of.txt"

    This mirrors a common pattern in corpora where many files share the
    same stem with numeric suffixes.
    """
    base = os.path.basename(path)
    # remove trailing "-NN" before extension
    return re.sub(r"-\d+(\.[^.]+)$", r"\1", base)


def _estimate_tokens_in_file(path: str, max_bytes: int = 8192) -> int:
    """
    Very cheap token estimate by reading up to max_bytes and splitting on whitespace.
    This does not use spaCy - it is meant to be fast and approximate.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            chunk = f.read(max_bytes)
    except Exception:
        return 0

    # simple whitespace tokenisation
    tokens = chunk.split()
    if not tokens:
        return 0

    # extrapolate to full file size
    try:
        file_size = os.path.getsize(path)
    except OSError:
        file_size = len(chunk.encode("utf-8", errors="ignore"))

    # avoid zero division
    bytes_read = max(len(chunk.encode("utf-8", errors="ignore")), 1)
    density = len(tokens) / float(bytes_read)
    est_tokens = int(density * file_size)
    return max(est_tokens, len(tokens))


# ---------------------------------------------------------------------------
# Utility helpers - LaTeX normalisation and classification
# ---------------------------------------------------------------------------

LATEX_CMD_PATTERN = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?")


def _read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _latex_to_text(raw: str) -> str:
    """
    Very safe LaTeX normaliser.

    - Removes commands like \section{Text} but keeps the argument text.
    - Strips remaining control sequences.
    - Never calls regex.sub() incorrectly.
    """
    try:
        # remove commands but preserve argument text
        cleaned = re.sub(
            r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?\{([^}]*)\}",
            r"\1",
            raw,
        )
        # remove remaining control sequences (e.g. \cite, \label)
        cleaned = LATEX_CMD_PATTERN.sub("", cleaned)
        return cleaned
    except Exception:
        # last-resort fallback: return raw to avoid losing content
        return raw


def classify_file(path: str) -> str:
    """
    Simple classifier for splitting a thesis into theory / case-study / other.

    This operates purely on filename patterns and is intentionally heuristic.
    """
    name = os.path.basename(path).lower()

    # THEORY GROUP
    if any(k in name for k in [
        "paper1", "paper2", "k-it", "epistemic", "methodology",
        "theory", "introduction"
    ]):
        return "theory"

    # CASE-STUDY GROUP
    if any(k in name for k in [
        "connor", "reed", "case", "misinfo", "social", "covid"
    ]):
        return "case"

    return "other"

def _pdf_to_text(src_path: str) -> str:
    """
    Robust PDF-to-text extractor.

    - Uses PyPDF2 if available.
    - Falls back to empty string when extraction fails.
    - Normalises whitespace.
    """

    try:
        from PyPDF2 import PdfReader
    except Exception:
        # PyPDF2 not installed or import failed.
        return ""

    try:
        reader = PdfReader(src_path)
    except Exception:
        # PDF is corrupted or unreadable
        return ""

    chunks = []

    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""

        if txt:
            chunks.append(txt)

    if not chunks:
        return ""

    # Normalise whitespace lightly
    out = "\n\n".join(chunks)
    out = out.replace("\r", "")
    out = " ".join(out.split())

    return out.strip()

# ---------------------------------------------------------------------------
# Public API - corpus profiling
# ---------------------------------------------------------------------------

def probe_corpus(
    corpus_dir: str,
    results_dir: str,
    emit=DEFAULT_EMIT,
) -> tuple[str, Dict[str, Any]]:

    """
    Scan corpus_dir, build a normalised corpus, and write corpus_profile.json.

    Returns:
        profile_dict with at least:
            - normalised_root: path to the _normalised_corpus directory
            - total_files
            - total_size_bytes
            - file_stats: [{path, rel_path, slug, est_tokens}, ...]
            - slug_stats: [{slug, n_files, total_tokens, mean_tokens}, ...]
            - by_slug: {slug: {...}}
            - by_class: {"theory": n, "case": n, "other": n}
    """
    corpus_dir = os.path.abspath(corpus_dir)
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    norm_root = os.path.join(results_dir, "_normalised_corpus")
    os.makedirs(norm_root, exist_ok=True)

    file_stats: List[Dict[str, Any]] = []
    slug_map: Dict[str, Dict[str, Any]] = {}
    classes: Dict[str, List[str]] = {"theory": [], "case": [], "other": []}

    total_files = 0
    total_size_bytes = 0

    for root, _, files in os.walk(corpus_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            # Treat common text-like formats plus LaTeX.
            if ext not in (".txt", ".md", ".csv", ".json", ".tex"):
                continue

            src_path = os.path.join(root, fname)
            rel = os.path.relpath(src_path, corpus_dir)
            dst_path = os.path.join(norm_root, rel)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            # Normalise
            if ext == ".tex":
                raw = _read_file(src_path)
                cleaned = _latex_to_text(raw)
                with open(dst_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)
            if ext == ".pdf":
                text = _pdf_to_text(src_path)
                if text.strip():
                    dst_path = os.path.join(norm_root, rel.replace(".pdf", ".txt"))
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    with open(dst_path, "w", encoding="utf-8") as f:
                        f.write(text)
                continue
            else:
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception:
                    raw = _read_file(src_path)
                    with open(dst_path, "w", encoding="utf-8") as f:
                        f.write(raw)

            total_files += 1
            try:
                size_bytes = os.path.getsize(src_path)
            except OSError:
                size_bytes = 0
            total_size_bytes += size_bytes

            slug = _slug_from_filename(src_path)
            est_tokens = _estimate_tokens_in_file(src_path)

            cls = classify_file(src_path)
            if cls not in classes:
                cls = "other"
            classes[cls].append(rel)

            file_stats.append(
                {
                    "path": src_path,
                    "rel_path": rel,
                    "slug": slug,
                    "class": cls,
                    "size_bytes": size_bytes,
                    "est_tokens": est_tokens,
                }
            )

            bucket = slug_map.setdefault(
                slug,
                {"slug": slug, "n_files": 0, "total_tokens": 0, "files": []},
            )
            bucket["n_files"] += 1
            bucket["total_tokens"] += est_tokens
            bucket["files"].append(rel)

    slug_stats: List[Dict[str, Any]] = []
    for slug, rec in slug_map.items():
        n = rec["n_files"]
        total_toks = rec["total_tokens"]
        mean_toks = float(total_toks) / float(n) if n > 0 else 0.0
        slug_stats.append(
            {
                "slug": slug,
                "n_files": n,
                "total_tokens": total_toks,
                "mean_tokens": mean_toks,
            }
        )

    slug_stats.sort(key=lambda r: r.get("total_tokens", 0), reverse=True)

    profile: Dict[str, Any] = {
        "normalised_root": norm_root,
        "total_files": total_files,
        "total_size_bytes": total_size_bytes,
        "by_class": {k: len(v) for k, v in classes.items()},
        "files": classes,
        "file_stats": file_stats,
        "slug_stats": slug_stats,
    }

    out_path = os.path.join(results_dir, "corpus_profile.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    emit(
        "log",
        {
            "stage": "corpus_probe",
            "event": "profile_written",
            "path": out_path,
        },
    )
    emit(
        "log",
        {
            "stage": "corpus_probe",
            "event": "end",
            "n_files": total_files,
            "by_class": profile["by_class"],
        },
    )

    # Return dict only
    return norm_root, profile





# ---------------------------------------------------------------------------
# Public API - LSA seed benchmarking
# ---------------------------------------------------------------------------

def run_lsa_seed_profile(
    corpus_dir: str,
    results_dir: str,
    profile: Optional[Dict[str, Any]] = None,
    max_seed_docs: int = 50,
    emit=DEFAULT_EMIT,
) -> Optional[Dict[str, Any]]:
    """
    Run a small LSA "seed" experiment on a subset of the corpus to estimate
    throughput (docs/s and tokens/s).

    Implementation strategy:
      - Use slug_stats in the corpus_profile to choose the largest slugs.
      - From those, pick up to max_seed_docs files.
      - Copy them into a temporary seed directory under results_dir.
      - Call build_lsa_field(seed_dir, emit=emit).
      - Measure wall-clock time.
      - Estimate docs/s and tokens/s from the seed subset.

    Returns a dict summarising the seed run and writes lsa_seed_profile.json.

    If build_lsa_field is not available, returns None gracefully.
    """
    if build_lsa_field is None:
        try:
            emit(
                "log",
                {
                    "stage": "lsa_seed",
                    "event": "lsa_seed_skipped",
                    "reason": "build_lsa_field not available",
                },
            )
        except Exception:
            pass
        return None

    corpus_dir = os.path.abspath(corpus_dir)
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # If no profile provided, run probe_corpus first.
    if profile is None:
        _, profile = probe_corpus(corpus_dir, results_dir, emit=emit)

    slug_stats = profile.get("slug_stats", []) or []
    if not slug_stats:
        return None

    # choose files from largest slugs first; take up to max_seed_docs
    seed_files: List[str] = []
    for rec in slug_stats:
        slug = rec.get("slug")
        if not slug:
            continue

        # scan corpus_dir for files with this slug
        for root, _, fnames in os.walk(corpus_dir):
            for fname in fnames:
                if _slug_from_filename(fname) == slug:
                    seed_files.append(os.path.join(root, fname))
                    if len(seed_files) >= max_seed_docs:
                        break
            if len(seed_files) >= max_seed_docs:
                break

        if len(seed_files) >= max_seed_docs:
            break

    if not seed_files:
        return None

    seed_dir = os.path.join(results_dir, "_lsa_seed_corpus")
    if os.path.exists(seed_dir):
        shutil.rmtree(seed_dir, ignore_errors=True)
    os.makedirs(seed_dir, exist_ok=True)

    for idx, src in enumerate(seed_files):
        dst = os.path.join(seed_dir, f"seed_{idx:04d}_" + os.path.basename(src))
        try:
            shutil.copy2(src, dst)
        except Exception:
            # ignore missing / unreadable files
            continue

    try:
        emit(
            "log",
            {
                "stage": "lsa_seed",
                "event": "start",
                "seed_dir": seed_dir,
                "n_docs": len(seed_files),
            },
        )
    except Exception:
        pass

    t0 = time.time()
    try:
        _ = build_lsa_field(seed_dir, emit=emit)
    except Exception as exc:
        try:
            emit(
                "log",
                {
                    "stage": "lsa_seed",
                    "event": "failed",
                    "error": str(exc),
                },
            )
        except Exception:
            pass
        return None
    t1 = time.time()

    runtime = max(t1 - t0, 1e-6)
    n_docs_seed = len(seed_files)

    # estimate total tokens in the seed subset from the per-file estimates
    total_tokens_seed = 0
    for p in seed_files:
        total_tokens_seed += _estimate_tokens_in_file(p)

    docs_per_sec = n_docs_seed / runtime
    tokens_per_sec = (
        total_tokens_seed / runtime if total_tokens_seed > 0 else None
    )

    seed_profile: Dict[str, Any] = {
        "seed_dir": seed_dir,
        "n_docs_seed": n_docs_seed,
        "est_tokens_seed": int(total_tokens_seed),
        "runtime_sec": float(runtime),
        "docs_per_sec": float(docs_per_sec),
        "tokens_per_sec": float(tokens_per_sec)
        if tokens_per_sec is not None
        else None,
    }

    out_path = os.path.join(results_dir, "lsa_seed_profile.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(seed_profile, f, indent=2)
        emit(
            "log",
            {
                "stage": "lsa_seed",
                "event": "profile_written",
                "path": out_path,
            },
        )
    except Exception:
        pass

    try:
        emit("log", {"stage": "lsa_seed", "event": "end"})
    except Exception:
        pass

    return seed_profile
