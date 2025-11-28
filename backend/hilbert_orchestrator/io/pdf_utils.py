"""
PDF Utilities
=============

Utility functions for extracting text from PDF documents.

This module is intentionally lightweight and dependency-tolerant:
it uses PyPDF2 when available and degrades gracefully when it is not.

The orchestrator uses this module during corpus normalization (e.g.,
to convert uploaded PDFs into usable text form for the LSA stage).
"""

from __future__ import annotations

from typing import List


# ---------------------------------------------------------------------------
# Optional dependency: PyPDF2
# ---------------------------------------------------------------------------

try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:
    PdfReader = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pdf_to_text(path: str) -> str:
    """
    Extract all textual content from a PDF file.

    Parameters
    ----------
    path : str
        Filesystem path to a PDF file.

    Returns
    -------
    str
        The concatenated textual content extracted from all pages.

        Returns an empty string if:
          - The file cannot be read,
          - PyPDF2 is unavailable,
          - Extraction fails on any or all pages.

    Notes
    -----
    - Extraction quality depends on the structure of the PDF. Scanned PDFs
      without embedded text will yield empty output.
    - Exceptions are suppressed by design so that the orchestrator never
      fails a run solely because of a malformed PDF.
    """

    if PdfReader is None:
        # PyPDF2 not installed or import failed
        return ""

    try:
        reader = PdfReader(path)
    except Exception:
        # File unreadable or corrupted
        return ""

    text_chunks: List[str] = []

    for page in reader.pages:
        try:
            content = page.extract_text() or ""
        except Exception:
            content = ""
        text_chunks.append(content)

    return "\n\n".join(text_chunks)
