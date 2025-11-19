# =============================================================================
# lsa_layer.py — Hilbert Information Chemistry Lab
# Latent Semantic Analysis + Element Extraction Layer
# =============================================================================
# Version: 2025 Thesis-Aligned Pipeline (Multi-format, upgraded, hybrid elements)
#
# Responsibilities:
#   - Ingest heterogeneous corpus files (PDF, LaTeX, code, plain text, HTML)
#   - Apply format-aware preprocessing to recover natural language spans
#   - Extract noun-like "elements" from spans via spaCy (with safe fallbacks)
#   - Build a TF–IDF term–span matrix over elements
#   - Compute an LSA (truncated SVD) spectral field
#   - Emit a compact JSON-serialisable representation for downstream stages
#
# Public entrypoint:
#   build_lsa_field(corpus_dir, emit=...) -> Dict[str, Any]
#
# Returned dictionary (schema used by the orchestrator):
#
#   {
#       "elements": [
#           { "element": str, "span_id": int },
#           ...
#       ],
#       "element_metrics": [
#           {
#               "element": str,
#               "index": int,
#               "collection_freq": int,
#               "document_freq": int,
#               "mean_entropy": float,
#               "mean_coherence": float,
#           },
#           ...
#       ],
#       "field": {
#           "embeddings": np.ndarray (n_spans, k),
#           "span_map": [ { "doc": str, "span_id": int, "text": str }, ... ],
#           "vocab": [ str, ... ],
#       },
#   }
#
# The orchestrator then:
#   - Expands per-span rows into hilbert_elements.csv
#   - Uses element_metrics for stability computation
#
# =============================================================================

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
except Exception as e:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required for the LSA layer "
        "(TfidfVectorizer, TruncatedSVD)."
    ) from e

# spaCy is optional – we degrade gracefully if unavailable
try:
    import spacy  # type: ignore

    _NLP = None  # lazy-loaded model
except ImportError:  # pragma: no cover
    spacy = None
    _NLP = None


# =============================================================================
# spaCy helper
# =============================================================================


def _get_nlp():
    """Return a shared spaCy pipeline or None if spaCy is unavailable."""
    global _NLP
    if spacy is None:
        return None
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except Exception:
            # Fallback to a blank English model if the small model is missing.
            _NLP = spacy.blank("en")
        # ensure sentence boundaries if using blank model
        if "sentencizer" not in _NLP.pipe_names:
            _NLP.add_pipe("sentencizer")
    return _NLP


# =============================================================================
# Format-aware text extraction
# =============================================================================

LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+(\*?)\s*(\[[^\]]*\])?(\{[^}]*\})?")
LATEX_ENV_RE = re.compile(r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}", re.DOTALL)
MATH_INLINE_RE = re.compile(r"\$[^$]*\$")
MATH_DISPLAY_RE = re.compile(r"\\\[.*?\\\]", re.DOTALL)
HLINE_RE = re.compile(r"^[-=]{3,}\s*$", re.MULTILINE)

HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTI_WS_RE = re.compile(r"[ \t]+")
MULTI_NL_RE = re.compile(r"\n{2,}")


def strip_latex(text: str) -> str:
    """
    Strip common LaTeX markup and structural noise.

    This is intentionally conservative: the goal is to keep ordinary prose
    while discarding environments, math, citation commands, and rule lines.
    """
    if not text:
        return ""

    # remove environments (\begin{...}...\end{...})
    text = LATEX_ENV_RE.sub(" ", text)
    # remove inline and display math
    text = MATH_INLINE_RE.sub(" ", text)
    text = MATH_DISPLAY_RE.sub(" ", text)
    # remove commands like \label{...}, \cite{...}, \ref{...}, \chapter{...}
    text = LATEX_COMMAND_RE.sub(" ", text)
    # remove purely decorative rule lines (====, ----)
    text = HLINE_RE.sub(" ", text)
    # normalise whitespace
    text = MULTI_WS_RE.sub(" ", text)
    text = MULTI_NL_RE.sub("\n", text)
    return text.strip()


def strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    if not text:
        return ""
    text = HTML_TAG_RE.sub(" ", text)
    text = MULTI_WS_RE.sub(" ", text)
    text = MULTI_NL_RE.sub("\n", text)
    return text.strip()


def extract_text_from_pdf(path: Path) -> str:
    """
    Extract text from a PDF.

    Uses PyPDF2 if available; otherwise returns an empty string and lets
    the caller decide how to handle it.
    """
    try:
        import PyPDF2  # type: ignore
    except ImportError:
        print(f"[lsa] PyPDF2 not available; skipping PDF: {path}")
        return ""

    try:
        text_chunks: List[str] = []
        with path.open("rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                if t.strip():
                    text_chunks.append(t)
        return "\n".join(text_chunks)
    except Exception as e:  # pragma: no cover
        print(f"[lsa] Failed to extract PDF text from {path}: {e}")
        return ""


def extract_text_from_python(path: Path) -> str:
    """
    Extract analyzable text from a Python source file.

    We keep:
      - comments starting with '#'
      - docstrings (triple-quoted strings)
    and discard executable code as far as practical.
    """
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        src = path.read_text(errors="ignore")

    # Collect comments
    comment_lines: List[str] = []
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            comment_lines.append(stripped.lstrip("#").strip())

    # Docstrings via a simple regex for triple-quoted strings
    docstring_re = re.compile(r'("""|\'\'\')(.*?)(\1)', re.DOTALL)
    docstrings: List[str] = []
    for m in docstring_re.finditer(src):
        body = m.group(2)
        if body and body.strip():
            docstrings.append(body.strip())

    chunks = [*comment_lines, *docstrings]
    text = "\n".join(chunks)
    return text.strip()


def extract_text_from_c_like(path: Path) -> str:
    """
    Extract comments from C / C++ / Java style files.

    Keeps:
      - // line comments
      - /* block comments */
    """
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        src = path.read_text(errors="ignore")

    # // comments
    line_comments: List[str] = []
    for line in src.splitlines():
        if "//" in line:
            _, cmt = line.split("//", 1)
            line_comments.append(cmt.strip())

    # /* ... */ block comments
    block_re = re.compile(r"/\*(.*?)\*/", re.DOTALL)
    block_comments: List[str] = []
    for m in block_re.finditer(src):
        body = m.group(1)
        if body and body.strip():
            block_comments.append(body.strip())

    chunks = [*line_comments, *block_comments]
    text = "\n".join(chunks)
    return text.strip()


def detect_file_kind(path: Path) -> str:
    """
    Classify a file path into a coarse type for preprocessing.

    Returns one of:
        "pdf", "latex", "python", "c_like", "html", "text"
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".tex", ".latex"}:
        return "latex"
    if suffix in {".py"}:
        return "python"
    if suffix in {".c", ".h", ".cpp", ".hpp", ".cc", ".java"}:
        return "c_like"
    if suffix in {".html", ".htm"}:
        return "html"
    # everything else: treat as generic text
    return "text"


def read_and_clean(path: Path) -> str:
    """
    Read a file and apply format-specific preprocessing.

    The goal is to return a single plain-text string suitable for
    sentence segmentation and LSA.
    """
    kind = detect_file_kind(path)
    if kind == "pdf":
        raw = extract_text_from_pdf(path)
        return strip_html(raw)  # PDFs sometimes carry HTML-ish text

    if kind == "latex":
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            raw = path.read_text(errors="ignore")
        return strip_latex(raw)

    if kind == "python":
        return extract_text_from_python(path)

    if kind == "c_like":
        return extract_text_from_c_like(path)

    if kind == "html":
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            raw = path.read_text(errors="ignore")
        return strip_html(raw)

    # generic text
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        raw = path.read_text(errors="ignore")

    if not raw:
        return ""
    raw = MULTI_WS_RE.sub(" ", raw)
    raw = MULTI_NL_RE.sub("\n", raw)
    return raw.strip()


# =============================================================================
# Span extraction and element tokenisation
# =============================================================================

STRUCTURAL_PREFIXES = (
    "label",
    "fig",
    "tab",
    "table",
    "eq",
    "chap",
    "sec",
    "subsec",
    "ref",
    "cite",
)


def is_informative_span(text: str) -> bool:
    """
    Heuristic filter for dropping boilerplate spans.

    - Reject spans with too few alphabetic characters
    - Reject spans that are mostly punctuation or symbols
    - Reject very short spans that are unlikely to carry semantic content
    """
    if not text:
        return False

    t = text.strip()
    if not t:
        return False

    # Very short lines: keep only if strongly alphabetical
    if len(t) < 20:
        alpha = sum(c.isalpha() for c in t)
        if alpha < 10:
            return False

    alpha_chars = sum(c.isalpha() for c in t)
    ratio = alpha_chars / max(1, len(t))
    if ratio < 0.3:
        return False

    return True


def normalise_token(t: str) -> str:
    """
    Normalise raw token text into a canonical "element" form.

    Rules:
      - lowercase
      - alphabetic characters only
      - length > 2
      - drop obvious structural / LaTeX / citation tokens
    """
    if not t:
        return ""
    t = t.strip().lower()
    t = re.sub(r"[^a-z]+", "", t)
    if len(t) <= 2:
        return ""

    for prefix in STRUCTURAL_PREFIXES:
        if t.startswith(prefix):
            return ""

    # optional: require at least one vowel to avoid most non-word junk
    if not re.search(r"[aeiou]", t):
        return ""

    return t


def extract_elements_from_text(text: str) -> List[str]:
    """
    Extract noun-like "elements" from a span of text (hybrid strategy).

    Strategy:
      - If spaCy is available, use:
          - noun chunks
          - NOUN / PROPN tokens
      - Otherwise, fall back to a simple regex-based word extractor.

    Returns a list of canonicalised element strings.
    """
    nlp = _get_nlp()
    elements: List[str] = []

    if nlp is not None:
        doc = nlp(text)

        # 1) Noun chunks
        for chunk in getattr(doc, "noun_chunks", []):
            norm = normalise_token(chunk.text)
            if norm:
                elements.append(norm)

        # 2) Standalone NOUN / PROPN tokens
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"}:
                norm = normalise_token(token.lemma_ or token.text)
                if norm:
                    elements.append(norm)

    else:
        # Fallback: simple word extraction
        for raw in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text):
            norm = normalise_token(raw)
            if norm:
                elements.append(norm)

    return elements


# =============================================================================
# LSA computation
# =============================================================================


def compute_lsa_embeddings(
    spans: List[str],
    n_components: int = 128,
    max_vocab: int = 5000,
):
    """
    Compute LSA embeddings for a list of span texts.

    Returns
    -------
    embeddings : np.ndarray
        Matrix of shape (n_spans, n_components).
    vocab : list of str
        Vocabulary of extracted elements.
    vectorizer : fitted TfidfVectorizer
        Returned so that caller can inspect IDF / feature names if needed.
    """
    if not spans:
        return np.zeros((0, 0), dtype=float), [], None

    vectorizer = TfidfVectorizer(
        tokenizer=extract_elements_from_text,
        lowercase=True,
        token_pattern=None,  # we use a custom tokenizer
        max_features=max_vocab,
        min_df=2,  # drop elements that occur in only one span
    )
    X = vectorizer.fit_transform(spans)  # (n_spans, n_terms)

    vocab = list(vectorizer.get_feature_names_out())

    if X.shape[1] == 0:
        return np.zeros((0, 0), dtype=float), [], vectorizer

    n_components = min(n_components, max(2, min(X.shape) - 1))

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(X)

    return embeddings.astype(float), vocab, vectorizer


# =============================================================================
# Core LSA field construction
# =============================================================================


def run_lsa_layer(
    corpus_dir: str | Path,
    n_components: int = 128,
    max_vocab: int = 5000,
) -> Dict[str, Any]:
    """
    Core LSA field construction.

    Steps:
      - Read files from corpus_dir (multi-format)
      - Apply format-specific cleaning (PDF, LaTeX, code, HTML, plain text)
      - Segment into spans using spaCy (or newline-based fallback)
      - Compute LSA spectral field over spans using hybrid element tokenizer
      - Build:
          * span_map (doc, span_id, text)
          * per-span element occurrences (element, span_id)
          * per-element metrics: frequency, and simple entropy/coherence

    Returns
    -------
    dict with keys:
      - "elements" (per-span occurrences)
      - "element_metrics" (per-element statistics)
      - "vocab"
      - "embeddings"
      - "span_map"
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    # Collect candidate files – we accept a variety of file types
    files = sorted(p for p in corpus_dir.glob("*") if p.is_file())

    spans: List[str] = []
    span_map: List[Dict[str, Any]] = []
    span_id = 0

    nlp = _get_nlp()

    for f in files:
        text = read_and_clean(f)
        if not text:
            continue

        if nlp is not None:
            doc = nlp(text)
            for sent in doc.sents:
                s = sent.text.strip()
                if not s:
                    continue
                if not is_informative_span(s):
                    continue
                spans.append(s)
                span_map.append({"doc": f.name, "span_id": span_id, "text": s})
                span_id += 1
        else:
            # Fallback: split on newlines
            for line in text.splitlines():
                s = line.strip()
                if not s:
                    continue
                if not is_informative_span(s):
                    continue
                spans.append(s)
                span_map.append({"doc": f.name, "span_id": span_id, "text": s})
                span_id += 1

    if not spans:
        print("[lsa] No spans extracted; empty field.")
        return {
            "elements": [],
            "element_metrics": [],
            "vocab": [],
            "embeddings": np.zeros((0, 0), dtype=float),
            "span_map": [],
        }

    embeddings, vocab, vectorizer = compute_lsa_embeddings(
        spans, n_components=n_components, max_vocab=max_vocab
    )

    # If vocab is empty, we cannot build meaningful elements.
    if len(vocab) == 0 or embeddings.shape[0] == 0:
        print("[lsa] Empty vocab or embeddings; returning minimal field.")
        return {
            "elements": [],
            "element_metrics": [],
            "vocab": vocab,
            "embeddings": embeddings,
            "span_map": span_map,
        }

    # -------------------------------------------------------------------------
    # Build per-span occurrences and per-element statistics
    # -------------------------------------------------------------------------

    element_to_index: Dict[str, int] = {e: i for i, e in enumerate(vocab)}

    cf_counter: Counter[str] = Counter()          # collection frequency
    df_counter: Counter[str] = Counter()          # document (span) frequency
    elem_to_spans: Dict[str, List[int]] = defaultdict(list)
    occurrences: List[Dict[str, Any]] = []        # per-span rows for orchestrator

    for sid, s in enumerate(spans):
        elems = extract_elements_from_text(s)
        if not elems:
            continue

        span_elems = set()
        for e in elems:
            if e not in element_to_index:
                # filtered out by vectorizer (e.g. low df) – ignore
                continue
            cf_counter[e] += 1
            span_elems.add(e)
            elem_to_spans[e].append(sid)
            occurrences.append({"element": e, "span_id": sid})

        for e in span_elems:
            df_counter[e] += 1

    # -------------------------------------------------------------------------
    # Per-element metrics, including simple entropy / coherence
    # -------------------------------------------------------------------------

    element_metrics: List[Dict[str, Any]] = []

    for e, idx in element_to_index.items():
        spans_idx = elem_to_spans.get(e, [])
        cf = int(cf_counter.get(e, 0))
        df = int(df_counter.get(e, 0))

        if spans_idx:
            embs = embeddings[spans_idx]
            # distance-based dispersion
            mu = embs.mean(axis=0, keepdims=True)
            dists = np.linalg.norm(embs - mu, axis=1)

            # Entropy over normalised distances (higher = more dispersed)
            if np.any(dists > 0):
                p = dists / (dists.sum() + 1e-12)
                H = float(-np.sum(p * np.log(p + 1e-12)))
            else:
                H = 0.0

            # Coherence as exp(-variance) of embeddings (higher = more coherent)
            var = float(np.var(embs, axis=0).mean())
            C = float(np.exp(-var))
        else:
            H = 0.0
            C = 0.0

        element_metrics.append(
            {
                "element": e,
                "index": idx,
                "collection_freq": cf,
                "document_freq": df,
                "mean_entropy": H,
                "mean_coherence": C,
            }
        )

    return {
        # per-span occurrences (orchestrator will join with span_map and embeddings)
        "elements": occurrences,
        # per-element aggregated metrics (used by stability layer)
        "element_metrics": element_metrics,
        # field-level structures
        "vocab": vocab,
        "embeddings": embeddings,
        "span_map": span_map,
    }


# =============================================================================
# Backwards-compatible LSA field builder (required by orchestrator + API)
# =============================================================================


def build_lsa_field(
    corpus_dir: str | Path,
    emit=print,
    n_components: int = 128,
    max_vocab: int = 5000,
) -> Dict[str, Any]:
    """
    Compatibility wrapper expected by:
      - hilbert_pipeline.__init__
      - hilbert_orchestrator _stage_lsa

    Wraps run_lsa_layer() and converts its outputs into the schema:
        {
            "elements": [...],
            "element_metrics": [...],
            "field": {
                "embeddings": ndarray,
                "span_map": [...],
                "vocab": [...]
            }
        }
    """
    emit("log", {"stage": "lsa", "event": "start"})

    res = run_lsa_layer(
        corpus_dir,
        n_components=n_components,
        max_vocab=max_vocab,
    )

    elements = res.get("elements", []) or []
    metrics = res.get("element_metrics", []) or []
    span_map = res.get("span_map", []) or []
    embeddings = res.get("embeddings")
    vocab = res.get("vocab", []) or []

    field = {
        "embeddings": embeddings,
        "span_map": span_map,
        "vocab": vocab,
    }

    out = {
        "elements": elements,
        "element_metrics": metrics,
        "field": field,
    }

    emit("log", {"stage": "lsa", "event": "end"})
    return out


# =============================================================================
# CLI hook (optional)
# =============================================================================


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Hilbert LSA Layer (multi-format)")
    ap.add_argument("--corpus", required=True, help="Folder of corpus files.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    args = ap.parse_args()

    res = run_lsa_layer(args.corpus)

    out: Dict[str, Any] = dict(res)
    emb = out.get("embeddings")
    if isinstance(emb, np.ndarray):
        out["embeddings"] = emb.tolist()

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[lsa] Saved LSA layer to {args.out}")
