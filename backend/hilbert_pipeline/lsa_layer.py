# =============================================================================
# lsa_layer.py — Hilbert Information Chemistry Lab
# Latent Semantic Analysis + Element Extraction Layer
# =============================================================================
# Version: 2025 Thesis-Aligned Pipeline (Multi-format, upgraded)
#
# Responsibilities:
#   - Ingest heterogeneous corpus files (PDF, LaTeX, code, plain text, HTML)
#   - Apply format-aware preprocessing to recover natural language spans
#   - Extract noun-like "elements" from spans via spaCy (with safe fallbacks)
#   - Build a TF–IDF term–span matrix over elements
#   - Compute an LSA (truncated SVD) spectral field
#   - Emit a compact JSON-serialisable representation for downstream stages
#
# Public entrypoints:
#   run_lsa_layer(corpus_dir: str | pathlib.Path, ...) -> Dict[str, Any]
#   build_lsa_field(corpus_dir: str | pathlib.Path, emit, ...) -> Dict[str, Any]
#
# Returned dictionary for build_lsa_field():
#   {
#       "elements":        [ { "element": str, "index": int } ],
#       "element_metrics": [ {
#                              "element": str,
#                              "index": int,
#                              "collection_freq": int,
#                              "document_freq": int,
#                              "entropy": float,
#                              "coherence": float,
#                            } ],
#       "field": {
#           "embeddings":  np.ndarray shape (n_spans, k),
#           "span_map":    [ {
#                              "doc": str,
#                              "doc_id": int,
#                              "span_id": int,
#                              "position": int,
#                              "text": str,
#                              "elements": [str, ...],
#                           } ],
#           "vocab":       [ str, ... ],
#           "H_span":      [ float, ... ]   # optional per-span entropy
#       }
#   }
# =============================================================================

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import Counter

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

        txt = "\n".join(text_chunks)
        # De-hyphenate and merge broken lines to improve sentence segmentation
        txt = re.sub(r"-\s*\n", "", txt)       # join hyphenated line breaks
        txt = re.sub(r"\n(?=[a-z])", " ", txt) # join lines that continue a sentence
        return txt
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
            # drop leading '#', keep the comment text
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
        # PDFs sometimes carry HTML-ish or layout noise
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

    # generic text
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        raw = path.read_text(errors="ignore")

    if not raw:
        return ""
    # very light normalisation for generic text
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

# keep letters and spaces for multiword phrases
ALPHA_SPACE_RE = re.compile(r"[^a-z\s]+")


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
    Normalise a single-word token into canonical element form.

    Rules:
      - lowercase
      - alphabetic characters only
      - length > 2
      - drop obvious structural / LaTeX / citation tokens
    """
    if not t:
        return ""
    t = t.strip().lower()
    # For single tokens we remove all non letters
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


def normalise_phrase(text: str) -> str:
    """
    Normalise a multiword noun phrase, preserving word boundaries.

    This avoids the earlier behaviour where "information theory" became
    "informationtheory". Spaces are retained and then regularised.

    Rules:
      - lowercase
      - keep letters and spaces only
      - collapse multiple spaces
      - require at least two non-trivial words
      - drop phrases that look like structural labels
    """
    if not text:
        return ""
    t = text.lower()
    # strip everything except letters and spaces
    t = ALPHA_SPACE_RE.sub(" ", t)
    t = MULTI_WS_RE.sub(" ", t).strip()
    if not t:
        return ""

    words = [w for w in t.split(" ") if len(w) > 2]
    if len(words) < 2:
        # if it is effectively a single word, let normalise_token handle it
        return ""

    # discard phrases that are obviously structural
    for w in words:
        for prefix in STRUCTURAL_PREFIXES:
            if w.startswith(prefix):
                return ""

    # require at least one vowel across all words
    if not any(re.search(r"[aeiou]", w) for w in words):
        return ""

    return " ".join(words)


def extract_elements_from_text(text: str) -> List[str]:
    """
    Extract noun-like "elements" from a span of text.

    Strategy:
      - If spaCy is available, use:
          - noun chunks as multiword elements (spaces preserved)
          - NOUN / PROPN tokens as single-word elements
      - Otherwise, fall back to a simple regex-based word extractor.

    Returns a list of canonicalised element strings.
    """
    nlp = _get_nlp()
    elements: List[str] = []

    if nlp is not None:
        doc = nlp(text)

        # 1) Multiword noun chunks, preserving spaces
        for chunk in getattr(doc, "noun_chunks", []):
            phrase_norm = normalise_phrase(chunk.text)
            if phrase_norm:
                elements.append(phrase_norm)

        # 2) Standalone NOUN / PROPN tokens
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"}:
                norm = normalise_token(token.lemma_ or token.text)
                if norm:
                    elements.append(norm)

    else:
        # Fallback: simple word extraction
        # Use explicit word regex on the raw text, then normalise each token
        raw_words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)
        norm_words: List[str] = []
        for raw in raw_words:
            norm = normalise_token(raw)
            if norm:
                norm_words.append(norm)
                elements.append(norm)

        # Optional: light bigram phrases from adjacent words
        # This gives a crude way of capturing "information theory" style
        # elements even without spaCy, while avoiding concatenation.
        for i in range(len(norm_words) - 1):
            w1, w2 = norm_words[i], norm_words[i + 1]
            if not w1 or not w2:
                continue
            phrase = f"{w1} {w2}"
            # respect phrase filters
            phrase_norm = normalise_phrase(phrase)
            if phrase_norm:
                elements.append(phrase_norm)

    return elements


# =============================================================================
# LSA computation
# =============================================================================


def compute_lsa_embeddings(
    spans: List[str],
    n_components: int = 128,
    max_vocab: int = 5000,
) -> Tuple[np.ndarray, List[str], Any]:
    """
    Compute LSA embeddings for a list of span texts.

    Returns
    -------
    embeddings : np.ndarray
        Matrix of shape (n_spans, n_components).
    vocab : list of str
        Vocabulary of extracted elements (can include spaces).
    vectorizer : fitted TfidfVectorizer
        Returned so that caller can inspect IDF / feature names if needed.
    """
    if not spans:
        return np.zeros((0, 0), dtype=float), [], None

    # Number of "documents" as seen by TfidfVectorizer:
    # here each span is treated as a document.
    n_docs = len(spans)

    # Choose min_df / max_df as *absolute counts* and ensure consistency.
    # - For a single-span corpus we must allow terms that occur in that span.
    # - For larger corpora we keep a light filter but always enforce
    #   max_df >= min_df so scikit-learn never raises.
    if n_docs <= 1:
        min_df = 1
        max_df = 1
    else:
        # keep terms that appear in at least 2 spans,
        # and do not drop anything by upper frequency
        min_df = 2
        max_df = n_docs  # absolute doc count, always >= min_df

    vectorizer = TfidfVectorizer(
        tokenizer=extract_elements_from_text,
        lowercase=True,
        token_pattern=None,  # we use a custom tokenizer
        max_features=max_vocab,
        min_df=min_df,
        max_df=max_df,
    )

    X = vectorizer.fit_transform(spans)  # (n_spans, n_terms)

    vocab = list(vectorizer.get_feature_names_out())

    if X.shape[1] == 0:
        return np.zeros((0, 0), dtype=float), [], vectorizer

    # Do not ask for more components than min(n_docs, vocab_size) - 1
    n_components = min(n_components, max(2, min(X.shape) - 1))

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(X)

    return embeddings.astype(float), vocab, vectorizer


# =============================================================================
# Main entrypoint
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
      - Extract elements per span
      - Build TF-IDF over spans using element tokenizer
      - Compute LSA spectral field
      - Construct lightweight element registry and metrics

    Returns a dict:
      {
        "elements":        [ { "element": e, "index": idx } ],
        "element_metrics": [ {
                               "element": e,
                               "index": idx,
                               "collection_freq": cf,
                               "document_freq": df,
                               "entropy": float,
                               "coherence": float,
                             } ],
        "vocab":           [ ... ],
        "embeddings":      np.ndarray (n_spans, k),
        "span_map":        [
                              { "doc": ..., "doc_id": int,
                                "span_id": int, "position": int,
                                "text": str, "elements": [str, ...] },
                              ...
                           ],
        "span_entropy":    [ float, ... ]  # per-span entropy (optional)
      }
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    # Collect candidate files – we accept a variety of file types
    files = sorted(p for p in corpus_dir.glob("*") if p.is_file())

    spans: List[str] = []
    span_map: List[Dict[str, Any]] = []
    span_id = 0

    # track per-document numeric IDs and positions
    doc_id_map: Dict[str, int] = {}
    local_positions: Dict[str, int] = {}

    nlp = _get_nlp()

    for f in files:
        # assign a stable doc_id per filename
        if f.name not in doc_id_map:
            doc_id_map[f.name] = len(doc_id_map)
        if f.name not in local_positions:
            local_positions[f.name] = 0

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
                elems = extract_elements_from_text(s) or []
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
        else:
            # Fallback: split on newlines
            for line in text.splitlines():
                s = line.strip()
                if not s:
                    continue
                if not is_informative_span(s):
                    continue
                elems = extract_elements_from_text(s) or []
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

    if not spans:
        print("[lsa] No spans extracted; empty field.")
        return {
            "elements": [],
            "element_metrics": [],
            "vocab": [],
            "embeddings": np.zeros((0, 0), dtype=float),
            "span_map": [],
            "span_entropy": [],
        }

    # LSA: embeddings + vocab + vectorizer
    embeddings, vocab, vectorizer = compute_lsa_embeddings(
        spans, n_components=n_components, max_vocab=max_vocab
    )

    # -------------------------------------------------------------------------
    # Derive TF-IDF matrix again from the fitted vectorizer to compute:
    #   - collection_freq (cf)
    #   - document_freq (df)
    #   - element entropy (over spans)
    #   - element coherence (via centroid similarity in embedding space)
    #   - span entropy (over vocabulary)
    # -------------------------------------------------------------------------
    if vectorizer is not None:
        X = vectorizer.transform(spans)  # (n_spans, n_terms)
    else:
        # Very defensive fallback
        X = None

    n_spans = len(spans)
    n_terms = len(vocab)

    # Per-span entropy H_span[i]
    H_span: List[float] = []
    if X is not None:
        for i in range(n_spans):
            row = X.getrow(i).toarray().ravel()
            total = float(row.sum())
            if total <= 0.0:
                H_span.append(0.0)
                continue
            p = row / total
            # numerical guard: ignore zero entries
            mask = p > 0
            p_masked = p[mask]
            H = -float(np.sum(p_masked * np.log(p_masked)))
            H_span.append(H)
    else:
        H_span = [0.0] * n_spans

    # Build element registry from vocab
    element_to_index: Dict[str, int] = {e: i for i, e in enumerate(vocab)}

    # Collection frequency and document frequency
    cf_counter: Counter[str] = Counter()
    df_counter: Counter[str] = Counter()

    # We use span_map "elements" if present; otherwise fallback to tokenisation.
    for span_rec in span_map:
        elems = span_rec.get("elements") or []
        if not elems:
            elems = extract_elements_from_text(span_rec.get("text", ""))
        if not elems:
            continue
        cf_counter.update(elems)
        for e in set(elems):
            df_counter[e] += 1

    # -------------------------------------------------------------------------
    # Element-level entropy & coherence
    # -------------------------------------------------------------------------
    # entropy(e): distribution of e across spans using TF–IDF magnitudes
    # coherence(e): how tightly clustered the spans of e are in LSA space
    element_entropy: Dict[str, float] = {}
    element_coherence: Dict[str, float] = {}

    if X is not None and embeddings is not None and embeddings.size > 0:
        emb_arr = np.asarray(embeddings, dtype=float)
        for term_idx, term in enumerate(vocab):
            col = X[:, term_idx].toarray().ravel()
            total = float(col.sum())
            if total <= 0.0:
                element_entropy[term] = 0.0
                element_coherence[term] = 0.0
                continue

            # entropy over spans
            p = col / total
            mask = p > 0
            p_masked = p[mask]
            H_e = -float(np.sum(p_masked * np.log(p_masked)))
            element_entropy[term] = H_e

            # coherence via centroid similarity in embedding space
            span_idxs = np.where(col > 0)[0]
            if span_idxs.size <= 1:
                element_coherence[term] = 0.0
                continue

            vecs = emb_arr[span_idxs, :]  # (m, k)
            centroid = vecs.mean(axis=0)
            # normalise for cosine similarity
            vecs_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
            sims = np.dot(vecs_norm, centroid_norm)
            element_coherence[term] = float(np.mean(sims))
    else:
        # conservative fallback if we somehow lack X or embeddings
        for term in vocab:
            element_entropy[term] = 0.0
            element_coherence[term] = 0.0

    # -------------------------------------------------------------------------
    # Assemble element registry + metrics
    # -------------------------------------------------------------------------
    elements: List[Dict[str, Any]] = []
    element_metrics: List[Dict[str, Any]] = []

    for e, idx in element_to_index.items():
        cf = int(cf_counter.get(e, 0))
        df = int(df_counter.get(e, 0))
        H_e = float(element_entropy.get(e, 0.0))
        C_e = float(element_coherence.get(e, 0.0))
        elements.append({"element": e, "index": idx})
        element_metrics.append(
            {
                "element": e,
                "index": idx,
                "collection_freq": cf,
                "document_freq": df,
                "entropy": H_e,
                "coherence": C_e,
            }
        )

    return {
        "elements": elements,
        "element_metrics": element_metrics,
        "vocab": vocab,
        "embeddings": embeddings,
        "span_map": span_map,
        "span_entropy": H_span,
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
                "vocab": [...],
                "H_span": [...],  # optional per-span entropy
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
    H_span = res.get("span_entropy")

    field: Dict[str, Any] = {
        "embeddings": embeddings,
        "span_map": span_map,
        "vocab": vocab,
    }
    if H_span is not None:
        field["H_span"] = H_span

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
    ap.add_argument("--k", type=int, default=128, help="LSA dimensionality.")
    ap.add_argument(
        "--max-vocab", type=int, default=5000, help="Max vocab size."
    )
    args = ap.parse_args()

    res = run_lsa_layer(
        args.corpus,
        n_components=args.k,
        max_vocab=args.max_vocab,
    )

    out: Dict[str, Any] = dict(res)
    emb = out.get("embeddings")
    if isinstance(emb, np.ndarray):
        out["embeddings"] = emb.tolist()

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[lsa] Saved LSA layer to {args.out}")
