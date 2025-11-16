# =============================================================================
# lsa_layer.py — Hilbert Information Chemistry Lab
# Latent Semantic Analysis + Element Extraction Layer
# =============================================================================
# Version: 2025 Thesis-Aligned Pipeline (Option B, upgraded)
#
# Implements:
#   - Robust text span extraction
#   - spaCy-based noun-phrase → element mapping (with safe fallbacks)
#   - Token → Element canonicalisation
#   - Tf-idf weighted term-document matrix
#   - Truncated SVD latent spectral field
#   - Per-element entropy and coherence (local + global)
#   - Document-level mapping enabling periodic table + molecules
#
# Output (internal Python objects):
#   {
#     "elements": [
#         { "element", "token", "doc", "span_id", "file", ... },
#         ...
#     ],
#     "element_metrics": [
#         {
#           "element",
#           "token",
#           "mean_entropy",
#           "mean_coherence",
#           "count",
#           "df",
#         },
#         ...
#     ],
#     "vocab": [ canonical element strings ... ],
#     "embeddings": np.ndarray of shape [num_spans, dim],
#     "span_map": [ { "doc", "span_id", "text" }, ... ],
#   }
#
# build_lsa_field(corpus_dir) wraps this and adds:
#   - H_span (per span entropy)
#   - H_bar (global entropy)
#   - C_global (global coherence)
#
# =============================================================================

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# spaCy loading with robust fallbacks
# ---------------------------------------------------------------------------

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    spacy = None

_NLP = None


def _get_nlp():
    """
    Lazy-load spaCy pipeline.

    Preference:
      - en_core_web_sm (full pipeline)
      - blank('en') + sentencizer
    """
    global _NLP
    if _NLP is not None:
        return _NLP

    if spacy is None:
        raise RuntimeError(
            "[lsa] spaCy is not available. Install 'spacy' and 'en_core_web_sm'."
        )

    try:
        nlp = spacy.load("en_core_web_sm")
        # ensure sentence boundaries
        if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        _NLP = nlp
        print("[lsa] Loaded spaCy model 'en_core_web_sm'.")
    except Exception:
        # Minimal pipeline
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        _NLP = nlp
        print(
            "[lsa][warn] Falling back to blank('en') with sentencizer only. "
            "No POS or noun-chunk information."
        )

    return _NLP


# ---------------------------------------------------------------------------
# 1. TEXT NORMALISATION & ELEMENT EXTRACTION
# ---------------------------------------------------------------------------

def normalise_token(t: str) -> str:
    """
    Collapse token to canonical 'element' form.

    Rules (aligned with thesis):
      - lowercase
      - keep alphabetic characters only
      - discard ultra-short tokens
    """
    if not t:
        return ""
    t = t.strip().lower()
    t = re.sub(r"[^a-z]+", "", t)
    if len(t) <= 2:
        return ""
    return t


def extract_elements_from_text(text: str) -> List[str]:
    """
    Extract noun-like elements from a text span.

    Strategy:
      - Use spaCy noun chunks where available
      - Use standalone NOUN / PROPN tokens
      - Fallback: content-word tokens when no linguistic detail exists
    """
    if not text or not text.strip():
        return []

    nlp = _get_nlp()
    doc = nlp(text)

    elements: List[str] = []

    # 1) Noun chunks (multiword concepts) - guard against missing parser
    try:
        for chunk in getattr(doc, "noun_chunks", []):
            root = chunk.root.lemma_.lower().strip()
            tok = normalise_token(root)
            if tok:
                elements.append(tok)
    except Exception:
        # noun_chunks not available in minimal pipeline
        pass

    # 2) Standalone nouns and proper nouns if POS is available
    has_pos = bool(doc) and any(tok.pos_ for tok in doc)  # heuristic
    if has_pos:
        for tok in doc:
            if tok.pos_ in ("NOUN", "PROPN"):
                s = normalise_token(tok.lemma_)
                if s:
                    elements.append(s)

    # 3) Fallback for minimal pipeline: content-like tokens
    if not elements:
        for tok in doc:
            if tok.is_alpha and len(tok.text) > 3:
                s = normalise_token(tok.text)
                if s:
                    elements.append(s)

    return elements


# ---------------------------------------------------------------------------
# 2. TF-IDF + LATENT SEMANTIC FIELD
# ---------------------------------------------------------------------------

def compute_lsa_embeddings(
    span_texts: List[str],
    dim: int = 128,
    max_vocab: int = 5000,
) -> Tuple[np.ndarray, List[str], TfidfVectorizer, TruncatedSVD]:
    """
    Compute TF-IDF matrix and reduce using Truncated SVD.

    Parameters
    ----------
    span_texts : list[str]
        Sentence-level spans.
    dim : int
        Target latent dimensionality.
    max_vocab : int
        Maximum vocabulary size.

    Returns
    -------
    emb : np.ndarray [n_spans, dim']
    vocab : list[str]
    vectorizer : TfidfVectorizer
    svd : TruncatedSVD
    """
    if not span_texts:
        return np.zeros((0, 0), dtype=float), [], None, None  # type: ignore

    vectorizer = TfidfVectorizer(
        tokenizer=extract_elements_from_text,
        max_features=max_vocab,
        lowercase=True,
        token_pattern=None,  # we use a custom tokenizer
    )

    tfidf = vectorizer.fit_transform(span_texts)
    vocab = vectorizer.get_feature_names_out().tolist()

    # Guard SVD components against tiny matrices
    max_components = min(tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    if max_components <= 0:
        # Degenerate: return dense TF-IDF rows as embeddings
        emb = tfidf.toarray().astype(float)
        return emb, vocab, vectorizer, None  # type: ignore

    n_components = min(dim, max_components)
    svd = TruncatedSVD(n_components=n_components)
    emb = svd.fit_transform(tfidf).astype(float)

    # Clean up any numerical issues
    emb[~np.isfinite(emb)] = 0.0

    return emb, vocab, vectorizer, svd


# ---------------------------------------------------------------------------
# 3. ENTROPY / COHERENCE METRICS
# ---------------------------------------------------------------------------

def compute_entropy(vec: np.ndarray) -> float:
    """
    Shannon entropy of an embedding distribution.

    Treats absolute component values as a discrete distribution
    and computes H = - sum_i p_i log p_i.
    """
    v = np.asarray(vec, dtype=float)
    if v.size == 0:
        return 0.0

    p = np.abs(v)
    s = float(p.sum())
    if s <= 0:
        return 0.0
    p = p / (s + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))


def compute_pairwise_coherence(vecs: np.ndarray) -> float:
    """
    Computes average cosine similarity across a set of vectors.

    - If there is only one vector, coherence is defined as 0.0.
    - If all off-diagonal similarities are NaN, coherence is 0.0.
    """
    arr = np.asarray(vecs, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    n = arr.shape[0]
    if n < 2:
        return 0.0

    sims = cosine_similarity(arr)
    # remove self-similarity
    np.fill_diagonal(sims, np.nan)

    mask = ~np.isnan(sims)
    if not mask.any():
        return 0.0

    return float(sims[mask].mean())


# ---------------------------------------------------------------------------
# 4. MAIN LSA PIPELINE
# ---------------------------------------------------------------------------

def run_lsa_layer(corpus_dir: str | Path) -> Dict[str, Any]:
    """
    Core LSA field construction.

    Steps:
      - Read raw text files from corpus_dir
      - Split into spans (sentences)
      - Extract elements per span
      - Build TF-IDF over spans using element tokenizer
      - Compute LSA spectral field
      - Build per-element registry with entropy and coherence metrics

    Parameters
    ----------
    corpus_dir : str or Path
        Directory of plain text files.

    Returns
    -------
    dict
        {
          "elements": [... span-level element records ...],
          "element_metrics": [... per-element summary rows ...],
          "vocab": [...],
          "embeddings": np.ndarray [n_spans, dim],
          "span_map": [...],
        }
    """
    corpus_dir = Path(corpus_dir)
    files = sorted(p for p in corpus_dir.glob("*") if p.is_file())

    spans: List[str] = []
    span_map: List[Dict[str, Any]] = []
    span_id = 0

    # ------------------------------------------------------------------
    # Extract spans
    # ------------------------------------------------------------------
    if not files:
        print(f"[lsa] No files found in {corpus_dir}. Empty field.")
        return {
            "elements": [],
            "element_metrics": [],
            "vocab": [],
            "embeddings": np.zeros((0, 0), dtype=float),
            "span_map": [],
        }

    nlp = _get_nlp()

    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            # last-resort: platform default encoding
            text = f.read_text(errors="ignore")

        if not text or not text.strip():
            continue

        doc = nlp(text)
        for sent in doc.sents:
            s = sent.text.strip()
            if not s:
                continue
            spans.append(s)
            span_map.append(
                {
                    "doc": f.name,
                    "span_id": span_id,
                    "text": s,
                }
            )
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

    # ------------------------------------------------------------------
    # Compute LSA field
    # ------------------------------------------------------------------
    emb, vocab, vectorizer, svd = compute_lsa_embeddings(spans)

    # ------------------------------------------------------------------
    # Build element table (span-level occurrences)
    # ------------------------------------------------------------------
    all_elements: List[Dict[str, Any]] = []
    doc_tf: Counter[str] = Counter()
    df_map: Dict[str, set] = defaultdict(set)

    for row_idx, entry in enumerate(span_map):
        doc_name = entry["doc"]
        text = entry["text"]
        els = extract_elements_from_text(text)

        for el in els:
            doc_tf[el] += 1
            df_map[el].add(doc_name)
            all_elements.append(
                {
                    "element": el,
                    "token": el,
                    "doc": doc_name,
                    "span_id": entry["span_id"],
                    "file": doc_name,
                }
            )

    # If for some reason the vectorizer produced a vocab but the separate
    # extractor finds nothing, we still allow empty element metrics.
    element_rows: List[Dict[str, Any]] = []
    emb_matrix = np.asarray(emb, dtype=float)

    if doc_tf:
        for el in sorted(doc_tf):
            # representation vector = mean of embeddings of spans containing it
            idxs = [r["span_id"] for r in all_elements if r["element"] == el]
            if not idxs:
                continue

            E = emb_matrix[idxs] if emb_matrix.size else np.zeros((len(idxs), 1))
            centroid = E.mean(axis=0)

            entropy = compute_entropy(centroid)
            coherence = compute_pairwise_coherence(E)

            element_rows.append(
                {
                    "element": el,
                    "token": el,
                    "mean_entropy": float(entropy),
                    "mean_coherence": float(coherence),
                    "count": int(doc_tf[el]),
                    "df": int(len(df_map[el])),
                }
            )

    return {
        "elements": all_elements,
        "element_metrics": element_rows,
        "vocab": vocab,
        "embeddings": emb,
        "span_map": span_map,
    }


# ---------------------------------------------------------------------------
# 5. ORCHESTRATOR INTEGRATION
# ---------------------------------------------------------------------------

def build_lsa_field(corpus_dir: str | Path, emit=None) -> Dict[str, Any]:
    """
    Orchestrator entry point.

    Wraps run_lsa_layer and augments the result with corpus-level
    entropy (H_bar) and coherence (C_global) metrics.

    Parameters
    ----------
    corpus_dir : str or Path
        Directory containing plain text files for the corpus.
    emit : callable or None
        Optional event sink: emit(type, payload). If None, no-op.
    """
    if emit is None:
        emit = lambda *_, **__: None  # noqa: E731

    emit("stage", {"name": "lsa", "status": "start"})
    results = run_lsa_layer(corpus_dir)

    emb = np.asarray(results.get("embeddings", []), dtype=float)
    if emb.size == 0:
        emit(
            "log",
            {"message": "[lsa] No embeddings generated; returning empty field."},
        )
        emit("stage", {"name": "lsa", "status": "end"})
        return {
            "field": {
                "embeddings": emb,
                "span_map": [],
                "vocab": [],
            },
            "elements": [],
            "element_metrics": [],
            "H_span": [],
            "H_bar": 0.0,
            "C_global": 0.0,
        }

    # per-span entropy and global entropy H_bar
    H_span = [float(compute_entropy(v)) for v in emb]
    H_bar = float(np.mean(H_span)) if H_span else 0.0

    # global coherence across the span field
    C_global = float(compute_pairwise_coherence(emb))

    emit(
        "log",
        {
            "message": (
                f"[lsa] Field built: n_spans={emb.shape[0]}, dim={emb.shape[1]}, "
                f"H_bar={H_bar:.4f}, C_global={C_global:.4f}"
            )
        },
    )

    field = {
        "embeddings": emb,
        "span_map": results["span_map"],
        "vocab": results["vocab"],
    }

    emit("stage", {"name": "lsa", "status": "end"})

    return {
        "field": field,
        "elements": results["elements"],
        "element_metrics": results["element_metrics"],
        "H_span": H_span,
        "H_bar": H_bar,
        "C_global": C_global,
    }


# ---------------------------------------------------------------------------
# 6. CLI mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Run Hilbert LSA layer over a corpus directory."
    )
    ap.add_argument("--corpus", required=True, help="Folder of plain text files.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    args = ap.parse_args()

    res = run_lsa_layer(args.corpus)

    # make JSON friendly if writing to disk
    out: Dict[str, Any] = dict(res)
    emb = out.get("embeddings")
    if isinstance(emb, np.ndarray):
        out["embeddings"] = emb.tolist()

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[lsa] Saved LSA layer to {args.out}")
