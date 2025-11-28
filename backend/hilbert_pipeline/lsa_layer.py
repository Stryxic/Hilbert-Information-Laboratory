"""
lsa_layer.py
============

Hilbert Information Pipeline – Latent Semantic Analysis (LSA) Layer
-------------------------------------------------------------------

This module implements the **first major stage** of the Hilbert Information
Detection Tool (HIDT). The LSA layer ingests heterogeneous corpora,
preprocesses text with format-specific logic, extracts noun-like
"elements", builds a TF–IDF over spans, computes a truncated SVD
spectral field, and returns a structured representation that seeds
the downstream fusion, molecule, and graph stages.

It is intentionally format-aware:

- PDF extraction
- LaTeX cleaning and de-noising
- HTML stripping
- Python and C-like code extraction (docstrings, comments, identifiers)
- Generic plain-text handling

This module provides **two public entrypoints**:

1. :func:`run_lsa_layer`
   - Complete LSA computation stage
   - Returns the raw LSA field + metrics + element registry

2. :func:`build_lsa_field`
   - Compatibility wrapper expected by the orchestrator and backend API
   - Normalises output into the canonical pipeline schema

Notes
-----
This module is deliberately verbose

"""

from __future__ import annotations
from pathlib import Path
import json
import re
from typing import Any, Dict, List, Tuple
from collections import Counter

import numpy as np

# =============================================================================
# Third-party dependencies
# =============================================================================

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
except Exception as e:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required for the LSA layer "
        "(TfidfVectorizer, TruncatedSVD)."
    ) from e

# spaCy optional (graceful degradation)
try:
    import spacy
    _NLP = None
except ImportError:  # pragma: no cover
    spacy = None
    _NLP = None

# Optional code tokenizer (for language-aware code embeddings)
try:
    from hilbert_pipeline.code_tokenizer import tokenize_code_string
    _HAVE_CODE_TOKENIZER = True
except Exception:  # pragma: no cover
    tokenize_code_string = None
    _HAVE_CODE_TOKENIZER = False


# =============================================================================
# spaCy access helper
# =============================================================================

def _get_nlp():
    """
    Return a shared spaCy model, loading it lazily on first use.

    Returns
    -------
    nlp : spacy.language.Language or None
        If spaCy is unavailable, returns ``None`` and callers degrade
        to a regex-only fallback.

    Notes
    -----
    If ``en_core_web_sm`` is missing, we load a blank English model and
    attach a sentencizer to preserve sentence segmentation.
    """
    global _NLP
    if spacy is None:
        return None
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except Exception:
            _NLP = spacy.blank("en")
        if "sentencizer" not in _NLP.pipe_names:
            _NLP.add_pipe("sentencizer")
    return _NLP


# =============================================================================
# LaTeX / HTML noise removal
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
    Remove common LaTeX markup, math, and structural noise.

    Parameters
    ----------
    text : str
        Raw contents of a LaTeX file.

    Returns
    -------
    str
        Cleaned plain text.

    Notes
    -----
    This routine intentionally maintains textual content while removing
    environments, math, citation commands, and decorative rule lines.
    """
    if not text:
        return ""

    text = LATEX_ENV_RE.sub(" ", text)
    text = MATH_INLINE_RE.sub(" ", text)
    text = MATH_DISPLAY_RE.sub(" ", text)
    text = LATEX_COMMAND_RE.sub(" ", text)
    text = HLINE_RE.sub(" ", text)
    text = MULTI_WS_RE.sub(" ", text)
    text = MULTI_NL_RE.sub("\n", text)
    return text.strip()


def strip_html(text: str) -> str:
    """
    Remove HTML tags and collapse whitespace.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Simplified text with HTML tags removed.
    """
    if not text:
        return ""
    text = HTML_TAG_RE.sub(" ", text)
    text = MULTI_WS_RE.sub(" ", text)
    text = MULTI_NL_RE.sub("\n", text)
    return text.strip()


# =============================================================================
# PDF extraction
# =============================================================================

def extract_text_from_pdf(path: Path) -> str:
    """
    Extract text from a PDF file using PyPDF2.

    Parameters
    ----------
    path : pathlib.Path

    Returns
    -------
    str
        Extracted plain text. Returns an empty string if parsing fails.

    Notes
    -----
    A minimal cleaning pass joins hyphenated line breaks and merges
    continuation lines to improve downstream sentence segmentation.
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

        txt = "\n".join(text_chunks)
        txt = re.sub(r"-\s*\n", "", txt)
        txt = re.sub(r"\n(?=[a-z])", " ", txt)
        return txt
    except Exception as e:
        print(f"[lsa] Failed to extract PDF text from {path}: {e}")
        return ""


# =============================================================================
# Code-aware extractors
# =============================================================================

def _code_tokens_from_source(src: str, language: str) -> str:
    """
    Produce identifier-like tokens from code using the optional code tokenizer.

    Parameters
    ----------
    src : str
        Source code string.
    language : str
        Programming language label for the tokenizer.

    Returns
    -------
    str
        Bag-of-tokens representation of code for LSA.

    Notes
    -----
    If no tokenizer is available, returns an empty string.
    """
    if not (_HAVE_CODE_TOKENIZER and tokenize_code_string):
        return ""

    try:
        try:
            tokens = tokenize_code_string(src, language=language)
        except TypeError:
            tokens = tokenize_code_string(src)
    except Exception:
        return ""

    cleaned = [t.strip() for t in tokens if isinstance(t, str) and t.strip()]
    return " ".join(cleaned) if cleaned else ""


def extract_text_from_python(path: Path) -> str:
    """
    Extract meaningful text from a Python file.

    Captures:
      - comments
      - docstrings
      - identifiers via the optional code tokenizer

    Parameters
    ----------
    path : pathlib.Path

    Returns
    -------
    str
    """
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        src = path.read_text(errors="ignore")

    # --- comments ---
    comment_lines = [
        line.strip().lstrip("#").strip()
        for line in src.splitlines()
        if line.strip().startswith("#")
    ]

    # --- docstrings ---
    docstring_re = re.compile(r'("""|\'\'\')(.*?)(\1)', re.DOTALL)
    docstrings = [
        m.group(2).strip()
        for m in docstring_re.finditer(src)
        if m.group(2).strip()
    ]

    pieces = []
    comment_block = "\n".join(comment_lines + docstrings).strip()
    if comment_block:
        pieces.append(comment_block)

    # --- identifiers / code tokens ---
    code_terms = _code_tokens_from_source(src, language="python")
    if code_terms:
        pieces.append(code_terms)

    return "\n\n".join(pieces).strip()


def extract_text_from_c_like(path: Path) -> str:
    """
    Extract comments and code tokens from C/C++/Java files.

    Parameters
    ----------
    path : pathlib.Path

    Returns
    -------
    str
    """
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        src = path.read_text(errors="ignore")

    # // line comments
    line_comments = []
    for line in src.splitlines():
        if "//" in line:
            _, cmt = line.split("//", 1)
            line_comments.append(cmt.strip())

    # /* block comments */
    block_re = re.compile(r"/\*(.*?)\*/", re.DOTALL)
    block_comments = [
        m.group(1).strip()
        for m in block_re.finditer(src)
        if m.group(1).strip()
    ]

    pieces = []
    comment_block = "\n".join(line_comments + block_comments).strip()
    if comment_block:
        pieces.append(comment_block)

    # Tokenise identifiers
    suffix = path.suffix.lower()
    lang = "java" if suffix == ".java" else "c"
    code_terms = _code_tokens_from_source(src, language=lang)
    if code_terms:
        pieces.append(code_terms)

    return "\n\n".join(pieces).strip()


# =============================================================================
# File kind detection and unified input cleaning
# =============================================================================

def detect_file_kind(path: Path) -> str:
    """
    Heuristically classify files into preprocessing categories.

    Returns
    -------
    str
        One of:
        ``"pdf"``, ``"latex"``, ``"python"``, ``"c_like"``,
        ``"html"``, ``"text"``.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".tex", ".latex"}:
        return "latex"
    if suffix == ".py":
        return "python"
    if suffix in {".c", ".h", ".cpp", ".hpp", ".cc", ".java"}:
        return "c_like"
    if suffix in {".html", ".htm"}:
        return "html"
    return "text"


def read_and_clean(path: Path) -> str:
    """
    Read a file and apply format-specific preprocessing.

    Parameters
    ----------
    path : pathlib.Path

    Returns
    -------
    str
        Plain text ready for span segmentation and LSA.

    Notes
    -----
    - PDFs pass through PDF text extraction + HTML stripping.
    - LaTeX is cleaned of markup, math, and environments.
    - Code files extract comments, docstrings, and identifier tokens.
    - HTML is stripped to bare text.
    - All text is lightly normalised.
    """
    kind = detect_file_kind(path)

    if kind == "pdf":
        raw = extract_text_from_pdf(path)
        return strip_html(raw)

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

    # Fallback: generic text
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        raw = path.read_text(errors="ignore")

    raw = MULTI_WS_RE.sub(" ", raw or "")
    raw = MULTI_NL_RE.sub("\n", raw)
    return raw.strip()


# =============================================================================
# Span filtering and tokenisation
# =============================================================================

STRUCTURAL_PREFIXES = (
    "label", "fig", "tab", "table", "eq", "chap", "sec",
    "subsec", "ref", "cite",
)

ALPHA_SPACE_RE = re.compile(r"[^a-z\s]+")


def is_informative_span(text: str) -> bool:
    """
    Determine whether a candidate span likely carries semantic content.

    Parameters
    ----------
    text : str

    Returns
    -------
    bool
        ``True`` if the span contains enough alphabetic content.

    Criteria
    --------
    - Reject spans dominated by punctuation
    - Reject extremely short or low-alpha spans
    """
    if not text:
        return False

    t = text.strip()
    if not t:
        return False

    # Short lines must be sufficiently alphabetic
    if len(t) < 20:
        alpha = sum(c.isalpha() for c in t)
        return alpha >= 10

    # Proportion of alphabetic content
    alpha_chars = sum(c.isalpha() for c in t)
    if alpha_chars / max(1, len(t)) < 0.3:
        return False

    return True


def normalise_token(t: str) -> str:
    """
    Canonicalise a single token (lowercasing, alphabetic only).

    Parameters
    ----------
    t : str

    Returns
    -------
    str
        Normalised token or empty string if rejected.

    Notes
    -----
    - Drops tokens shorter than three letters.
    - Filters out LaTeX/structural prefixes (``fig``, ``sec`` etc.).
    - Requires a vowel to filter out non-words.
    """
    if not t:
        return ""
    t = re.sub(r"[^a-z]+", "", t.strip().lower())
    if len(t) <= 2:
        return ""

    for prefix in STRUCTURAL_PREFIXES:
        if t.startswith(prefix):
            return ""

    return t if re.search(r"[aeiou]", t) else ""


def normalise_phrase(text: str) -> str:
    """
    Canonicalise a multiword noun phrase.

    Parameters
    ----------
    text : str

    Returns
    -------
    str or ""
        Cleaned phrase with spaces preserved.

    Notes
    -----
    - Keeps letters and spaces only.
    - Requires at least two nontrivial words.
    - Requires a vowel in at least one word.
    - Filters structural tokens.
    """
    if not text:
        return ""
    t = ALPHA_SPACE_RE.sub(" ", text.lower())
    t = MULTI_WS_RE.sub(" ", t).strip()
    if not t:
        return ""

    words = [w for w in t.split(" ") if len(w) > 2]
    if len(words) < 2:
        return ""

    for w in words:
        for prefix in STRUCTURAL_PREFIXES:
            if w.startswith(prefix):
                return ""

    if not any(re.search(r"[aeiou]", w) for w in words):
        return ""

    return " ".join(words)


def extract_elements_from_text(text: str) -> List[str]:
    """
    Extract noun-like semantic units ("elements") from a span.

    Parameters
    ----------
    text : str

    Returns
    -------
    list of str
        Canonicalised element strings.

    Strategy
    --------
    - If spaCy is available:
        - Use noun chunks for multiword expressions
        - Extract NOUN and PROPN tokens
    - Otherwise:
        - Regex-based word extraction
        - Light bigram reconstruction
    """
    nlp = _get_nlp()
    elements = []

    if nlp is not None:
        doc = nlp(text)

        for chunk in getattr(doc, "noun_chunks", []):
            pn = normalise_phrase(chunk.text)
            if pn:
                elements.append(pn)

        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"}:
                norm = normalise_token(token.lemma_ or token.text)
                if norm:
                    elements.append(norm)
    else:
        raw_words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)
        norm_words = []
        for raw in raw_words:
            nt = normalise_token(raw)
            if nt:
                elements.append(nt)
                norm_words.append(nt)

        # simple bigram phrases
        for i in range(len(norm_words) - 1):
            phrase_norm = normalise_phrase(f"{norm_words[i]} {norm_words[i+1]}")
            if phrase_norm:
                elements.append(phrase_norm)

    return elements


# =============================================================================
# LSA computation (TF–IDF + SVD)
# =============================================================================

def compute_lsa_embeddings(
    spans: List[str],
    n_components: int = 128,
    max_vocab: int = 5000,
) -> Tuple[np.ndarray, List[str], Any]:
    """
    Compute a TF–IDF matrix over spans and apply truncated SVD.

    Parameters
    ----------
    spans : list of str
        List of span texts.
    n_components : int, default 128
        Truncation rank for SVD.
    max_vocab : int, default 5000
        Maximum number of unique terms.

    Returns
    -------
    embeddings : np.ndarray
        Array of shape (n_spans, k).
    vocab : list of str
        Vectorizer vocabulary (semantic element strings).
    vectorizer : TfidfVectorizer
        The fitted vectorizer (for IDF lookup / term stats).

    Notes
    -----
    - Randomness is fixed via ``random_state=42``.
    - Each span acts as a "document".
    """
    if not spans:
        return np.zeros((0, 0), float), [], None

    n_docs = len(spans)

    if n_docs <= 1:
        min_df = max_df = 1
    else:
        min_df = 2
        max_df = n_docs

    vectorizer = TfidfVectorizer(
        tokenizer=extract_elements_from_text,
        lowercase=True,
        token_pattern=None,
        max_features=max_vocab,
        min_df=min_df,
        max_df=max_df,
    )

    X = vectorizer.fit_transform(spans)
    vocab = list(vectorizer.get_feature_names_out())

    if X.shape[1] == 0:
        return np.zeros((0, 0), float), [], vectorizer

    n_components_effective = min(
        n_components, max(2, min(X.shape) - 1)
    )

    svd = TruncatedSVD(n_components=n_components_effective, random_state=42)
    embeddings = svd.fit_transform(X)

    return embeddings.astype(float), vocab, vectorizer


# =============================================================================
# Main pipeline computation
# =============================================================================

def run_lsa_layer(
    corpus_dir: str | Path,
    n_components: int = 128,
    max_vocab: int = 5000,
) -> Dict[str, Any]:
    """
    Perform full LSA processing on a corpus directory.

    Parameters
    ----------
    corpus_dir : str or pathlib.Path
        Root directory of the uploaded corpus.
    n_components : int, default 128
        Number of SVD dimensions.
    max_vocab : int, default 5000
        Maximum feature count for the TF–IDF vectorizer.

    Returns
    -------
    dict
        Dictionary containing:
        - ``elements``: list of element registry entries
        - ``element_metrics``: frequency, entropy, coherence, TF–IDF
        - ``vocab``: vocabulary list
        - ``embeddings``: np.ndarray (n_spans × k)
        - ``span_map``: list of span metadata
        - ``span_entropy``: entropy per span
        - ``config``: LSA configuration metadata

    Notes
    -----
    This function is called directly by early orchestrator versions.
    Later orchestrator versions call :func:`build_lsa_field`, which wraps
    this output into the canonical API schema.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    files = sorted(p for p in corpus_dir.rglob("*") if p.is_file())

    spans = []
    span_map = []
    span_id = 0

    doc_id_map = {}
    local_positions = {}

    nlp = _get_nlp()

    # -----------------------
    # Span extraction
    # -----------------------
    for f in files:
        if f.name not in doc_id_map:
            doc_id_map[f.name] = len(doc_id_map)
        if f.name not in local_positions:
            local_positions[f.name] = 0

        text = read_and_clean(f)
        if not text:
            continue

        if nlp:
            doc = nlp(text)
            sentences = (s.text.strip() for s in doc.sents)
        else:
            sentences = (l.strip() for l in text.splitlines())

        for s in sentences:
            if not s or not is_informative_span(s):
                continue

            elems = extract_elements_from_text(s)
            spans.append(s)
            span_map.append(
                {
                    "doc": f.name,
                    "doc_id": doc_id_map[f.name],
                    "span_id": span_id,
                    "position": local_positions[f.name],
                    "text": s,
                    "elements": elems,
                }
            )
            span_id += 1
            local_positions[f.name] += 1

    # Empty corpus behaviour
    if not spans:
        return {
            "elements": [],
            "element_metrics": [],
            "vocab": [],
            "embeddings": np.zeros((0, 0), float),
            "span_map": [],
            "span_entropy": [],
            "config": {
                "version": "hilbert-lsa-2025",
                "model": "svd",
                "model_version": "svd-0-l2",
                "n_components": 0,
                "max_vocab": max_vocab,
                "normalisation": "l2",
                "embedding_parameters": {
                    "model": "svd",
                    "n_components": 0,
                    "max_vocab": max_vocab,
                    "random_state": 42,
                },
                "n_spans": 0,
                "n_terms": 0,
            },
        }

    # -----------------------
    # LSA computation
    # -----------------------
    embeddings, vocab, vectorizer = compute_lsa_embeddings(
        spans, n_components=n_components, max_vocab=max_vocab
    )

    if vectorizer:
        X = vectorizer.transform(spans)
        idf_array = getattr(vectorizer, "idf_", None)
    else:
        X = None
        idf_array = None

    n_spans = len(spans)
    n_terms = len(vocab)

    # -----------------------
    # Span-level entropy
    # -----------------------
    H_span = []
    if X is not None:
        for i in range(n_spans):
            row = X.getrow(i).toarray().ravel()
            total = row.sum()
            if total <= 0:
                H_span.append(0.0)
                continue
            p = row / total
            p_masked = p[p > 0]
            H_span.append(float(-(p_masked * np.log(p_masked)).sum()))
    else:
        H_span = [0.0] * n_spans

    # -----------------------
    # Element-level statistics
    # -----------------------
    element_to_index = {e: i for i, e in enumerate(vocab)}

    cf_counter = Counter()
    df_counter = Counter()

    for rec in span_map:
        elems = rec["elements"]
        cf_counter.update(elems)
        for e in set(elems):
            df_counter[e] += 1

    element_entropy = {}
    element_coherence = {}
    element_idf = {}
    element_tfidf = {}

    if X is not None and embeddings is not None and embeddings.size > 0:
        emb_arr = embeddings.astype(float)

        for t_idx, term in enumerate(vocab):
            col = X[:, t_idx].toarray().ravel()
            total = col.sum()

            if total <= 0:
                element_entropy[term] = 0.0
                element_coherence[term] = 0.0
                element_tfidf[term] = 0.0
            else:
                p = col / total
                p_masked = p[p > 0]
                element_entropy[term] = float(-(p_masked * np.log(p_masked)).sum())
                element_tfidf[term] = float(total)

                span_idxs = np.where(col > 0)[0]
                if span_idxs.size <= 1:
                    element_coherence[term] = 0.0
                else:
                    vecs = emb_arr[span_idxs, :]
                    centroid = vecs.mean(axis=0)
                    vecs_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
                    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
                    sims = vecs_norm @ centroid_norm
                    element_coherence[term] = float(sims.mean())

            if idf_array is not None and len(idf_array) == n_terms:
                element_idf[term] = float(idf_array[t_idx])
            else:
                df = df_counter.get(term, 0)
                element_idf[term] = float(np.log((n_spans + 1) / (df + 1)) + 1.0)
    else:
        for term in vocab:
            element_entropy[term] = 0.0
            element_coherence[term] = 0.0
            element_idf[term] = 0.0
            element_tfidf[term] = 0.0

    # -----------------------
    # Assemble registry
    # -----------------------
    elements = []
    element_metrics = []

    for e, idx in element_to_index.items():
        cf = int(cf_counter.get(e, 0))
        df = int(df_counter.get(e, 0))

        elements.append({"element": e, "index": idx})
        element_metrics.append(
            {
                "element": e,
                "index": idx,
                "collection_freq": cf,
                "document_freq": df,
                "tf": float(cf),
                "df": float(df),
                "idf": float(element_idf[e]),
                "tfidf": float(element_tfidf[e]),
                "entropy": float(element_entropy[e]),
                "coherence": float(element_coherence[e]),
            }
        )

    # -----------------------
    # Configuration block
    # -----------------------
    n_effective = embeddings.shape[1] if isinstance(embeddings, np.ndarray) else 0
    model_version = f"svd-{n_effective or n_components}-l2"

    config = {
        "version": "hilbert-lsa-2025",
        "model": "svd",
        "model_version": model_version,
        "n_components": n_effective or n_components,
        "max_vocab": max_vocab,
        "normalisation": "l2",
        "embedding_parameters": {
            "model": "svd",
            "n_components": n_effective or n_components,
            "max_vocab": max_vocab,
            "random_state": 42,
        },
        "n_spans": n_spans,
        "n_terms": n_terms,
    }

    return {
        "elements": elements,
        "element_metrics": element_metrics,
        "vocab": vocab,
        "embeddings": embeddings,
        "span_map": span_map,
        "span_entropy": H_span,
        "config": config,
    }


# =============================================================================
# build_lsa_field – Canonical API wrapper
# =============================================================================

def build_lsa_field(
    corpus_dir: str | Path,
    emit=print,
    n_components: int = 128,
    max_vocab: int = 5000,
) -> Dict[str, Any]:
    """
    Build the canonical LSA field expected by the orchestrator and backend API.

    Parameters
    ----------
    corpus_dir : str or pathlib.Path
        Corpus directory.
    emit : callable, default print
        Event emitter. Receives ``emit(kind, payload)``.
    n_components : int
        SVD dimensionality.
    max_vocab : int

    Returns
    -------
    dict
        {
            "elements": [...],
            "element_metrics": [...],
            "field": {
                "embeddings": ndarray,
                "span_map": [...],
                "vocab": [...],
                "H_span": [...],
            },
            "config": {...}
        }

    Notes
    -----
    This wrapper preserves backward compatibility with orchestrator
    versions 2.x–4.x, which expect output under the ``"field"`` key.
    """
    emit("log", {"stage": "lsa", "event": "start"})

    res = run_lsa_layer(
        corpus_dir,
        n_components=n_components,
        max_vocab=max_vocab,
    )

    out = {
        "elements": res.get("elements", []),
        "element_metrics": res.get("element_metrics", []),
        "field": {
            "embeddings": res.get("embeddings"),
            "span_map": res.get("span_map", []),
            "vocab": res.get("vocab", []),
            "H_span": res.get("span_entropy", []),
        },
        "config": res.get("config", {}),
    }

    emit("log", {"stage": "lsa", "event": "end"})
    return out


# =============================================================================
# Optional CLI
# =============================================================================

if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Hilbert LSA Layer")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--max-vocab", type=int, default=5000)
    args = ap.parse_args()

    res = run_lsa_layer(
        args.corpus,
        n_components=args.k,
        max_vocab=args.max_vocab,
    )

    out = dict(res)
    emb = out.get("embeddings")
    if isinstance(emb, np.ndarray):
        out["embeddings"] = emb.tolist()

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[lsa] Saved LSA layer to {args.out}")
