# =============================================================================
# hilbert_pipeline/hilbert_report.py - Scientific Hilbert Run Report
# =============================================================================
"""
Build a structured, scientific style report for a single Hilbert pipeline run.

This module is intended to sit at the very end of the pipeline. It assumes
that all earlier stages have written their artifacts into a single results
directory (out_dir), including:

  - hilbert_run.json
  - lsa_field.json
  - hilbert_elements.csv
  - span_element_fusion.csv
  - edges.csv
  - molecules.csv
  - informational_compounds.json
  - compound_contexts.json (optional)
  - signal_stability.csv, stability_meta.json
  - compound_stability.csv (optional)
  - element_roots.csv, element_cluster_metrics.json (optional)
  - signatures.csv / signatures.json (optional)
  - lm_metrics.json (optional)
  - graph_*.png, persistence_field.png, stability_scatter.png, etc.

The goal is to synthesise these artifacts into a single PDF that reads like
a compact scientific report in the style of a short-format article:

  0. Executive summary
  1. Abstract
  2. Introduction
  3. Methods (pipeline description)
  4. Results (graphs, stability, compounds, regimes, LM metrics)
  5. Discussion
  6. Conclusion
  7. Reproducibility appendix
  8. Visual appendix (figures)

Optionally, if a local LLM endpoint is available (for example an Ollama
deployment that implements the OpenAI chat completions API) the report
can include short, data grounded explanatory paragraphs that are generated
automatically. This behaviour is controlled via environment variables:

  HILBERT_LLM_REPORT=1          enable LLM assisted narration (default off)
  HILBERT_LLM_URL               base URL for the OpenAI compatible endpoint
                                (defaults to OLLAMA_URL or http://localhost:11434)
  HILBERT_LLM_MODEL             model name (defaults to OLLAMA_MODEL or "mistral")

The orchestrator can invoke:

    from hilbert_pipeline.hilbert_report import run_hilbert_report
    run_hilbert_report(results_dir, emit=ctx.emit)

If reportlab is not installed, the module falls back to a plain text report
(hilbert_report.txt) with the same basic structure.
"""

from __future__ import annotations

import json
import os
import hashlib
import inspect
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------#
# Orchestrator compatible emitter
# -----------------------------------------------------------------------------#

DEFAULT_EMIT = lambda *_a, **_k: None  # type: ignore


# -----------------------------------------------------------------------------#
# Compatibility patch: ReportLab md5 on some Python builds
# -----------------------------------------------------------------------------#

try:
    sig = inspect.signature(hashlib.md5)
    _has_usedforsecurity = "usedforsecurity" in sig.parameters
except Exception:
    _has_usedforsecurity = False

if not _has_usedforsecurity:
    _old_md5 = hashlib.md5

    def _safe_md5(*args, **kwargs):
        kwargs.pop("usedforsecurity", None)
        return _old_md5(*args, **kwargs)

    hashlib.md5 = _safe_md5  # type: ignore


# -----------------------------------------------------------------------------#
# ReportLab imports (optional)
# -----------------------------------------------------------------------------#

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib.utils import ImageReader

    HAS_REPORTLAB = True
except ImportError:  # pragma: no cover - defensive
    HAS_REPORTLAB = False


# -----------------------------------------------------------------------------#
# Optional LLM support for narrative sections
# -----------------------------------------------------------------------------#

try:
    import requests  # type: ignore

    HAS_REQUESTS = True
except Exception:  # pragma: no cover - defensive
    HAS_REQUESTS = False

LLM_REPORT_ENABLED = os.environ.get("HILBERT_LLM_REPORT", "0") == "1"

# Use dedicated Hilbert variables first, fall back to ollama defaults.
_LLM_BASE_URL = (
    os.environ.get("HILBERT_LLM_URL")
    or os.environ.get("OLLAMA_URL")
    or "http://localhost:11434"
)
_LLM_MODEL = (
    os.environ.get("HILBERT_LLM_MODEL")
    or os.environ.get("OLLAMA_MODEL")
    or "mistral"
)
_LLM_CHAT_URL = _LLM_BASE_URL.rstrip("/") + "/v1/chat/completions"


def _llm_generate_paragraph(
    section: str,
    data_payload: Dict[str, Any],
    *,
    max_tokens: int = 260,
    temperature: float = 0.2,
) -> Optional[str]:
    """
    Optionally generate a short, Nature style paragraph for a given section.

    This helper is designed to be safe and non critical:
    - If requests is unavailable or the HTTP call fails, it returns None.
    - The caller always has a deterministic fall back text.

    Parameters
    ----------
    section:
        Semantic name of the section, for example "executive_summary" or
        "graph_structure".
    data_payload:
        Dictionary of numeric and categorical descriptors that the model
        should ground its explanation in. This is serialised to JSON and
        placed verbatim into the prompt.
    """
    if not (LLM_REPORT_ENABLED and HAS_REQUESTS):
        return None

    # Keep the payload compact and explicit for the model.
    try:
        context_json = json.dumps(data_payload, indent=2)[:6000]
    except Exception:
        context_json = str(data_payload)[:6000]

    system_msg = (
        "You are assisting with the writing of a short scientific report. "
        "Write in the style of a Results or Discussion paragraph in a Nature "
        "family journal: concise, precise, and grounded entirely in the data "
        "provided. Do not speculate beyond the numbers and descriptors you see. "
        "Avoid rhetorical flourish. Use past tense and neutral language."
    )

    user_msg = (
        f"Section type: {section}.\n\n"
        "Data summary (JSON):\n"
        f"{context_json}\n\n"
        "Task: Write a single concise paragraph of 3 to 5 sentences that "
        "explains the main patterns in this data. Refer to quantities in "
        "relative terms where possible (for example 'most elements', 'a small "
        "set of compounds') but you may cite specific counts if helpful. "
        "Do not add bullet points or headings."
    )

    payload = {
        "model": _LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    try:
        resp = requests.post(_LLM_CHAT_URL, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    try:
        content = data["choices"][0]["message"]["content"]
        text = str(content).strip()
        # Normalise whitespace and trim overly long responses.
        text = " ".join(text.split())
        if not text:
            return None
        return text
    except Exception:
        return None


# -----------------------------------------------------------------------------#
# Reuse helpers from hilbert_export where possible
# -----------------------------------------------------------------------------#

try:
    from .hilbert_export import (
        _read_run_summary,
        _read_compound_metrics,
        _read_compound_table,
        _read_elements_info,
        _read_num_spans,
        _read_regime_profile,
        _read_root_stats,
        _read_lm_metrics,
        _read_compound_stability_table,
        _read_graph_metrics,
        _collect_figure_paths,
        _split_graph_and_other_figs,
    )
except Exception:  # pragma: no cover - very defensive fallback

    def _read_json(path: str) -> Any:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _read_run_summary(out_dir: str) -> Dict[str, Any]:
        path = os.path.join(out_dir, "hilbert_run.json")
        data = _read_json(path)
        return data if isinstance(data, dict) else {}

    def _read_compound_metrics(out_dir: str) -> Dict[str, Any]:
        path = os.path.join(out_dir, "compound_metrics.json")
        data = _read_json(path)
        return data if isinstance(data, dict) else {}

    def _read_compound_table(out_dir: str) -> pd.DataFrame:
        path = os.path.join(out_dir, "informational_compounds.json")
        data = _read_json(path)
        if not isinstance(data, dict):
            return pd.DataFrame()
        rows: List[Dict[str, Any]] = []
        for cid, cdata in data.items():
            if not isinstance(cdata, dict):
                continue
            rows.append(
                {
                    "compound_id": str(cdata.get("compound_id", cid)),
                    "num_elements": cdata.get("num_elements"),
                    "num_bonds": cdata.get("num_bonds"),
                    "compound_stability": cdata.get("compound_stability"),
                    "mean_temperature": cdata.get("mean_temperature"),
                }
            )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        for col in (
            "num_elements",
            "num_bonds",
            "compound_stability",
            "mean_temperature",
        ):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _read_elements_info(out_dir: str) -> Dict[str, Any]:
        path = os.path.join(out_dir, "hilbert_elements.csv")
        if not os.path.exists(path):
            return {}
        try:
            df = pd.read_csv(path)
        except Exception:
            return {}
        info: Dict[str, Any] = {}
        if "element" in df.columns:
            info["num_elements"] = int(df["element"].astype(str).nunique())
        if "token" in df.columns:
            info["num_tokens"] = int(df["token"].astype(str).nunique())
        for ent_col in ("mean_entropy", "entropy"):
            if ent_col in df.columns:
                vals = pd.to_numeric(df[ent_col], errors="coerce")
                info["mean_entropy"] = float(np.nanmean(vals))
                break
        for coh_col in ("mean_coherence", "coherence"):
            if coh_col in df.columns:
                vals = pd.to_numeric(df[coh_col], errors="coerce")
                info["mean_coherence"] = float(np.nanmean(vals))
                break
        return info

    def _read_num_spans(out_dir: str) -> Optional[int]:
        path = os.path.join(out_dir, "lsa_field.json")
        data = _read_json(path)
        if not isinstance(data, dict):
            return None
        emb = data.get("embeddings")
        if isinstance(emb, list):
            return len(emb)
        return None

    def _read_regime_profile(out_dir: str) -> Tuple[float, float, float]:
        path = os.path.join(out_dir, "hilbert_elements.csv")
        if not os.path.exists(path):
            return 0.0, 0.0, 0.0
        try:
            df = pd.read_csv(path)
        except Exception:
            return 0.0, 0.0, 0.0

        def _mean(col: str) -> float:
            if col not in df.columns:
                return 0.0
            arr = pd.to_numeric(df[col], errors="coerce")
            arr = arr[np.isfinite(arr)]
            return float(np.nanmean(arr)) if arr.size else 0.0

        return _mean("info_score"), _mean("misinfo_score"), _mean("disinfo_score")

    def _read_root_stats(out_dir: str) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "num_roots": None,
            "median_cluster_size": None,
            "max_cluster_size": None,
            "has_cluster_metrics": False,
        }
        roots_path = os.path.join(out_dir, "element_roots.csv")
        metrics_path = os.path.join(out_dir, "element_cluster_metrics.json")
        if os.path.exists(roots_path):
            try:
                df = pd.read_csv(roots_path)
                if "element" in df.columns:
                    stats["num_roots"] = int(df["element"].astype(str).nunique())
                if "cluster_size" in df.columns:
                    vals = pd.to_numeric(df["cluster_size"], errors="coerce")
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        stats["median_cluster_size"] = float(np.median(vals))
                        stats["max_cluster_size"] = float(np.max(vals))
            except Exception:
                pass
        stats["has_cluster_metrics"] = os.path.exists(metrics_path)
        return stats

    def _read_lm_metrics(out_dir: str) -> Dict[str, Any]:
        path = os.path.join(out_dir, "lm_metrics.json")
        data = _read_json(path)
        if not isinstance(data, dict):
            return {}
        out: Dict[str, Any] = {}
        if "model" in data:
            out["model"] = data["model"]
        if "perplexity" in data:
            try:
                out["perplexity"] = float(data["perplexity"])
            except Exception:
                pass
        if "n_tokens" in data:
            try:
                out["n_tokens"] = int(data["n_tokens"])
            except Exception:
                pass
        return out

    def _read_compound_stability_table(out_dir: str) -> pd.DataFrame:
        path = os.path.join(out_dir, "compound_stability.csv")
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
        for col in (
            "n_elements",
            "mean_element_coherence",
            "mean_element_stability",
            "stability_variance",
        ):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "compound_id" in df.columns:
            df["compound_id"] = df["compound_id"].astype(str)
        return df

    def _read_graph_metrics(out_dir: str) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "num_nodes": None,
            "num_edges": None,
            "avg_degree": None,
            "num_components": None,
            "num_communities": None,
            "modularity": None,
        }
        edges_path = os.path.join(out_dir, "edges.csv")
        if not os.path.exists(edges_path):
            return metrics
        try:
            edges_df = pd.read_csv(edges_path)
        except Exception:
            return metrics
        if "source" not in edges_df.columns or "target" not in edges_df.columns:
            return metrics
        sources = edges_df["source"].astype(str)
        targets = edges_df["target"].astype(str)
        nodes = set(sources) | set(targets)
        m = len(edges_df)
        n = len(nodes)
        if n == 0:
            return metrics
        metrics["num_nodes"] = int(n)
        metrics["num_edges"] = int(m)
        metrics["avg_degree"] = 2.0 * m / float(n)
        try:
            import networkx as nx  # type: ignore
        except Exception:
            return metrics
        G = nx.Graph()
        for s, t in zip(sources, targets):
            if s and t and s != t:
                G.add_edge(s, t)
        if G.number_of_nodes() == 0:
            return metrics
        metrics["num_components"] = nx.number_connected_components(G)
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            from networkx.algorithms.community.quality import modularity

            comms = list(greedy_modularity_communities(G))
            metrics["num_communities"] = len(comms)
            if len(comms) > 1:
                metrics["modularity"] = float(modularity(G, comms))
        except Exception:
            pass
        return metrics

    def _collect_figure_paths(out_dir: str) -> List[str]:
        all_pngs: List[str] = []
        for root, _, files in os.walk(out_dir):
            for fname in files:
                if fname.lower().endswith(".png"):
                    all_pngs.append(os.path.join(root, fname))
        return sorted(all_pngs)

    def _split_graph_and_other_figs(
        out_dir: str, fig_paths: List[str]
    ) -> Tuple[List[str], List[str]]:
        # Basic fallback: treat all as "other"
        return [], fig_paths


# -----------------------------------------------------------------------------#
# Utility functions
# -----------------------------------------------------------------------------#

def _safe_mean(series: pd.Series, default: float = 0.0) -> float:
    if series is None or series.empty:
        return default
    arr = pd.to_numeric(series, errors="coerce").to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    val = float(np.nanmean(arr))
    return val if np.isfinite(val) else default


def _read_signatures(out_dir: str) -> pd.DataFrame:
    """
    Load signatures.csv if available. Returns empty frame otherwise.
    """
    path = os.path.join(out_dir, "signatures.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return df


def _has_compound_contexts(out_dir: str) -> bool:
    return os.path.exists(os.path.join(out_dir, "compound_contexts.json"))


# -----------------------------------------------------------------------------#
# Report writer helper for ReportLab canvas
# -----------------------------------------------------------------------------#

if HAS_REPORTLAB:

    class ReportWriter:
        """
        Small helper to manage page breaks and basic text layout on a canvas.
        """

        def __init__(self, c: "canvas.Canvas"):
            self.c = c
            self.width, self.height = A4
            self.margin_left = 72
            self.margin_right = 72
            self.margin_top = 72
            self.margin_bottom = 72
            self.y = self.height - self.margin_top

        # -------------- basic layout primitives --------------------------------

        def new_page(self):
            self.c.showPage()
            self.y = self.height - self.margin_top

        def ensure_space(self, needed: float):
            if self.y - needed < self.margin_bottom:
                self.new_page()

        # -------------- text helpers -------------------------------------------

        def add_heading(self, text: str, level: int = 1):
            if level == 0:
                font = "Helvetica-Bold"
                size = 18
            elif level == 1:
                font = "Helvetica-Bold"
                size = 14
            elif level == 2:
                font = "Helvetica-Bold"
                size = 12
            else:
                font = "Helvetica-Bold"
                size = 10

            self.ensure_space(size + 8)
            self.c.setFont(font, size)
            self.c.drawString(self.margin_left, self.y, text)
            self.y -= (size + 6)

        def _wrapped_lines(
            self, text: str, font: str, size: int, max_width: float
        ) -> List[str]:
            lines: List[str] = []
            for para in text.split("\n"):
                words = para.split()
                if not words:
                    lines.append("")
                    continue
                line = words[0]
                for w in words[1:]:
                    candidate = line + " " + w
                    if (
                        pdfmetrics.stringWidth(candidate, font, size)
                        <= max_width
                    ):
                        line = candidate
                    else:
                        lines.append(line)
                        line = w
                lines.append(line)
            return lines

        def add_text(
            self,
            text: str,
            size: int = 9,
            leading: int = 12,
            font: str = "Helvetica",
        ):
            max_width = self.width - self.margin_left - self.margin_right
            self.c.setFont(font, size)
            lines = self._wrapped_lines(text, font, size, max_width)
            for line in lines:
                if not line.strip():
                    self.ensure_space(leading)
                    self.y -= leading * 0.3
                    continue
                self.ensure_space(leading)
                self.c.drawString(self.margin_left, self.y, line)
                self.y -= leading
            self.y -= leading * 0.3

        def add_bullets(
            self,
            bullet_lines: List[str],
            size: int = 9,
            leading: int = 12,
            indent: int = 14,
        ):
            max_width = (
                self.width - self.margin_left - self.margin_right - indent
            )
            font = "Helvetica"
            self.c.setFont(font, size)
            for line in bullet_lines:
                text = "- " + line
                lines = self._wrapped_lines(
                    text, font, size, max_width
                )
                for i, seg in enumerate(lines):
                    self.ensure_space(leading)
                    x = self.margin_left + (0 if i == 0 else indent)
                    self.c.drawString(x, self.y, seg)
                    self.y -= leading
            self.y -= leading * 0.3

        def add_table(self, data: List[List[str]], col_widths: List[float]):
            table = Table(data, colWidths=col_widths)
            style = TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
            table.setStyle(style)
            table_height = 12 * len(data) + 8
            self.ensure_space(table_height + 10)
            table.wrapOn(self.c, self.width, table_height)
            table.drawOn(self.c, self.margin_left, self.y - table_height)
            self.y -= (table_height + 12)


# -----------------------------------------------------------------------------#
# PDF builder
# -----------------------------------------------------------------------------#

def export_hilbert_report(out_dir: str) -> None:
    """
    Main entry point for building a scientific style report PDF (or a plain
    text fallback) for a completed Hilbert run.
    """
    os.makedirs(out_dir, exist_ok=True)

    run_summary = _read_run_summary(out_dir)
    compound_stats = _read_compound_metrics(out_dir)
    compounds_df = _read_compound_table(out_dir)
    elem_info = _read_elements_info(out_dir)
    num_spans = _read_num_spans(out_dir)
    root_stats = _read_root_stats(out_dir)
    info_mean, mis_mean, dis_mean = _read_regime_profile(out_dir)
    lm_metrics = _read_lm_metrics(out_dir)
    graph_metrics = _read_graph_metrics(out_dir)
    compound_stab_df = _read_compound_stability_table(out_dir)
    signatures_df = _read_signatures(out_dir)
    compound_contexts_available = _has_compound_contexts(out_dir)

    run_id = run_summary.get("run_id", "n/a")
    corpus_dir = run_summary.get("corpus_dir", "n/a")
    results_dir = run_summary.get("results_dir", out_dir)
    settings = run_summary.get("settings", {}) or {}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = "Hilbert Information Chemistry Report"

    # ----------------------------------------------------------------------#
    # Fallback: plain text report if ReportLab is not installed
    # ----------------------------------------------------------------------#
    if not HAS_REPORTLAB:
        txt_path = os.path.join(out_dir, "hilbert_report.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"{title}\nGenerated at: {timestamp}\n\n")
            f.write(f"Run id: {run_id}\n")
            f.write(f"Corpus: {corpus_dir}\n")
            f.write(f"Results: {results_dir}\n\n")

            f.write("Executive summary (compact):\n")
            f.write(
                f"  Spans: {num_spans or 'n/a'}, "
                f"Elements: {elem_info.get('num_elements', 'n/a')}, "
                f"Graph nodes: {graph_metrics.get('num_nodes', 'n/a')}, "
                f"Compounds: "
            )
            if compound_stats and "num_compounds" in compound_stats:
                f.write(str(compound_stats["num_compounds"]))
            else:
                f.write(
                    str(
                        compounds_df["compound_id"].nunique()
                        if not compounds_df.empty
                        and "compound_id" in compounds_df.columns
                        else "n/a"
                    )
                )
            f.write("\n\nSettings:\n")
            for k, v in sorted(settings.items()):
                f.write(f"  - {k}: {v}\n")
            f.write("\n")
        print(f"[report] Plain text report written to {txt_path}")
        return

    # ----------------------------------------------------------------------#
    # Full PDF report
    # ----------------------------------------------------------------------#
    pdf_path = os.path.join(out_dir, "hilbert_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    writer = ReportWriter(c)

    spans_str = str(num_spans) if num_spans is not None else "n/a"
    nodes = graph_metrics.get("num_nodes", "n/a")
    edges = graph_metrics.get("num_edges", "n/a")
    n_compounds = (
        compound_stats.get("num_compounds")
        if compound_stats
        else compounds_df["compound_id"].nunique()
        if not compounds_df.empty and "compound_id" in compounds_df.columns
        else "n/a"
    )

    # ===========================#
    # 0. Executive summary
    # ===========================#
    writer.add_heading(title, level=0)
    writer.add_text(
        f"Generated at {timestamp} for run {run_id}. "
        "This report summarises the Hilbert Information Chemistry analysis "
        "for a single corpus."
    )

    writer.add_heading("Executive summary", level=1)

    executive_points = [
        f"Corpus root: {corpus_dir}",
        f"Spans analysed: {spans_str}",
        f"Unique informational elements: {elem_info.get('num_elements', 'n/a')}",
        f"Element tokens: {elem_info.get('num_tokens', 'n/a')}",
        f"Graph size: {nodes} nodes, {edges} edges",
        f"Compounds formed: {n_compounds}",
        f"Mean element entropy: {elem_info.get('mean_entropy', 0.0):.3f}",
        f"Mean element coherence: {elem_info.get('mean_coherence', 0.0):.3f}",
    ]
    if lm_metrics:
        if "perplexity" in lm_metrics:
            executive_points.append(
                f"Element language model: {lm_metrics.get('model', 'n/a')} "
                f"(perplexity {lm_metrics['perplexity']:.3f})"
            )
        else:
            executive_points.append(
                f"Element language model: {lm_metrics.get('model', 'n/a')}"
            )

    writer.add_bullets(executive_points)

    # Optional LLM paragraph for the executive summary
    es_llm = _llm_generate_paragraph(
        "executive_summary",
        {
            "run_id": run_id,
            "corpus_dir": corpus_dir,
            "n_spans": num_spans,
            "elements": elem_info,
            "graph_metrics": graph_metrics,
            "n_compounds": n_compounds,
            "lm_metrics": lm_metrics,
        },
    )
    if es_llm:
        writer.add_text(es_llm)

    writer.add_text(
        "Readers who are primarily interested in the global behaviour of the "
        "corpus can focus on the executive summary and the Results section, "
        "while those interested in implementation details can consult the "
        "Methods and Reproducibility appendix."
    )

    # ===========================#
    # 1. Abstract
    # ===========================#
    writer.add_heading("1. Abstract", level=1)
    abstract_parts: List[str] = []

    abstract_parts.append(
        "The Hilbert pipeline models text corpora as informational fields. "
        "Spans of text are embedded into a latent space, fused into recurring "
        "elements, and organised into a co occurrence graph from which molecular "
        "and compound level structures are derived."
    )

    abstract_parts.append(
        "Here we report a single run of the pipeline, quantifying the field with "
        "entropy, coherence, and stability metrics, and describing the resulting "
        "graph topology, compound spectrum, and language model perplexity. "
        "These quantities provide complementary views on how the corpus distributes "
        "information over spans and elements."
    )

    if n_compounds not in (None, "n/a"):
        abstract_parts.append(
            f"In this run, the system produced {n_compounds} compounds from "
            f"{elem_info.get('num_elements', 'n/a')} elements, forming a graph "
            f"with {nodes} nodes and {edges} edges."
        )

    writer.add_text("\n\n".join(abstract_parts))

    # ===========================#
    # 2. Introduction
    # ===========================#
    writer.add_heading("2. Introduction", level=1)

    writer.add_text(
        "Conventional corpus analyses often treat documents as bags of words or "
        "sparse topic mixtures. The Hilbert framework instead emphasises the "
        "geometry of an informational field: spans act as carriers of local "
        "semantics, elements aggregate these signals across contexts, and the "
        "co occurrence graph encodes how such elements interact."
    )

    writer.add_text(
        "The resulting representation supports multiple levels of description, "
        "from individual elements through molecules and compounds up to global "
        "regime composition. This report documents one instantiation of that "
        "pipeline, with an emphasis on reproducibility and explicit links to the "
        "derived artifacts."
    )

    # ===========================#
    # 3. Methods
    # ===========================#
    writer.add_heading("3. Methods", level=1)

    writer.add_heading("3.1 Corpus and preprocessing", level=2)
    writer.add_text(
        f"The corpus root for this run was {corpus_dir}. Input documents were "
        "normalised into a LSA compatible corpus and segmented into short spans. "
        f"The resulting spectral field comprises {spans_str} spans."
    )

    writer.add_heading("3.2 LSA spectral field", level=2)
    writer.add_text(
        "The LSA layer embeds each span into a fixed dimensional latent space, "
        "using a truncated singular value decomposition of the span term matrix. "
        "The file lsa_field.json records the embeddings, span map, and vocabulary "
        "required to reconstruct this field."
    )

    writer.add_heading("3.3 Span element fusion", level=2)
    writer.add_text(
        "Span element fusion links spans to informational elements. Elements "
        "behave like enriched tokens that accumulate statistics across their "
        "occurrences, including entropy, coherence, and document frequency. "
        "The mapping is stored in span_element_fusion.csv, while per element "
        "statistics reside in hilbert_elements.csv."
    )

    writer.add_heading("3.4 Graph construction", level=2)
    writer.add_text(
        "An undirected co occurrence graph is built over elements. Two elements "
        "are connected if they appear in the same span, with edge weights "
        "proportional to co occurrence frequency. Per node top k pruning and "
        "optional global caps are applied to keep the graph sparse. The final "
        "edge list is written to edges.csv."
    )

    writer.add_heading("3.5 Molecules and compounds", level=2)
    writer.add_text(
        "Connected components of the element graph define molecular fields. "
        "Within each component, local statistics are aggregated to obtain "
        "compound scale descriptors such as size, internal edge density, and "
        "temperature like measures. These are exported in "
        "informational_compounds.json and, when available, summarised in "
        "compound_stability.csv and compound_metrics.json."
    )

    writer.add_heading("3.6 Stability and persistence", level=2)
    writer.add_text(
        "Signal stability combines entropy and coherence into a scalar field "
        "over elements. Stability metrics are written to signal_stability.csv, "
        "with summary statistics in stability_meta.json. Persistence style "
        "visualisations of this field are captured in persistence_field.png, "
        "stability_scatter.png, and related figures."
    )

    writer.add_heading("3.7 Epistemic signatures", level=2)
    writer.add_text(
        "When span level labels are available, the signatures layer aggregates "
        "them over elements to estimate epistemic regimes such as information, "
        "misinformation, and disinformation. These quantities are recorded in "
        "signatures.csv and, where present, signatures.json."
    )

    writer.add_heading("3.8 Element language model", level=2)
    writer.add_text(
        "To probe generative regularities in the element field, the pipeline "
        "trains a small language model over element sequences and uses an "
        "external LLM to estimate corpus perplexity. The resulting metrics are "
        "stored in lm_metrics.json."
    )

    writer.add_heading("3.9 Graph visualisation", level=2)
    writer.add_text(
        "The unified graph visualiser computes both 2D and 3D layouts for the "
        "largest connected components, assigns node sizes and colours based on "
        "compound membership and stability, and exports progressive snapshots "
        "at several density levels (for example 1 percent, 5 percent, 25 percent, "
        "50 percent, and the full field)."
    )

    # ===========================#
    # 4. Results
    # ===========================#
    writer.add_heading("4. Results", level=1)

    # 4.1 Corpus composition
    writer.add_heading("4.1 Corpus composition", level=2)
    writer.add_text(
        f"The corpus contributed {spans_str} spans to the LSA field. The element "
        f"table contains {elem_info.get('num_elements', 'n/a')} unique elements "
        f"and {elem_info.get('num_tokens', 'n/a')} distinct tokens."
    )
    writer.add_text(
        f"The mean element entropy is "
        f"{elem_info.get('mean_entropy', 0.0):.3f}, while the mean coherence is "
        f"{elem_info.get('mean_coherence', 0.0):.3f}. Higher entropy indicates a "
        "more diffuse usage pattern across contexts, whereas higher coherence "
        "suggests that an element sits within a tight semantic neighbourhood."
    )

    if root_stats.get("num_roots") is not None:
        writer.add_text(
            "The condensed root field groups surface forms into shared roots. "
            f"In this run, the root index contains {root_stats['num_roots']} "
            f"roots. Median cluster size is "
            f"{root_stats.get('median_cluster_size', 'n/a')}, with a maximum "
            f"cluster size of {root_stats.get('max_cluster_size', 'n/a')}."
        )

    # 4.2 Graph structure
    writer.add_heading("4.2 Graph structure", level=2)
    if graph_metrics.get("num_nodes") is not None:
        gm = graph_metrics
        txt = (
            f"The element graph has {gm['num_nodes']} nodes and "
            f"{gm['num_edges']} edges, giving an average degree of "
            f"{gm.get('avg_degree', 0.0):.2f}."
        )
        writer.add_text(txt)

        comp_line = ""
        if gm.get("num_components") is not None:
            comp_line += (
                f"It splits into {gm['num_components']} connected components. "
            )
        if gm.get("num_communities") is not None:
            comp_line += (
                f"Community detection identifies {gm['num_communities']} clusters"
            )
            if gm.get("modularity") is not None:
                comp_line += f" with modularity Q = {gm['modularity']:.3f}."
            else:
                comp_line += "."
        if comp_line:
            writer.add_text(comp_line)

        writer.add_text(
            "Progressive graph snapshots reveal how the largest component emerges "
            "as edges are added. At low density, only the strongest co occurrence "
            "relations are visible; at full density, the core of the graph appears "
            "as a dense, approximately spherical region with peripheral tendrils "
            "corresponding to rarer combinations."
        )

        # Optional LLM paragraph on graph structure
        graph_llm = _llm_generate_paragraph(
            "graph_structure",
            {
                "graph_metrics": gm,
                "n_compounds": n_compounds,
                "root_stats": root_stats,
            },
        )
        if graph_llm:
            writer.add_text(graph_llm)

    else:
        writer.add_text(
            "The graph stage did not produce a valid edges.csv file, so no "
            "structural graph metrics are available for this run."
        )

    # 4.3 Stability
    writer.add_heading("4.3 Stability field", level=2)
    if not compound_stab_df.empty:
        mean_elem_stab = _safe_mean(
            compound_stab_df.get(
                "mean_element_stability",
                pd.Series([], dtype=float),
            )
        )
        writer.add_text(
            "Compound level aggregation reveals a typical mean per element "
            f"stability of {mean_elem_stab:.3f}. Compounds with higher stability "
            "represent combinations of elements that recur reliably across spans."
        )
    else:
        writer.add_text(
            "Compound level stability metrics were not available for this run."
        )

    stab_meta_path = os.path.join(out_dir, "stability_meta.json")
    stab_meta: Dict[str, Any] = {}
    if os.path.exists(stab_meta_path):
        try:
            with open(stab_meta_path, "r", encoding="utf-8") as f:
                stab_meta = json.load(f)
        except Exception:
            stab_meta = {}

    if isinstance(stab_meta, dict) and "stability_mean" in stab_meta:
        writer.add_text(
            "Element level stability statistics show a mean stability of "
            f"{stab_meta.get('stability_mean', float('nan')):.3f} "
            f"(median {stab_meta.get('stability_median', float('nan')):.3f})."
        )

    writer.add_text(
        "The persistence and stability figures visualise how stability values "
        "are distributed across elements and, when available, across documents. "
        "Smooth regions in the persistence curve correspond to extended bands of "
        "similar stability, whereas sharp transitions indicate rapid changes in "
        "element behaviour."
    )

    # Optional LLM paragraph on stability
    stability_llm = _llm_generate_paragraph(
        "stability_field",
        {
            "compound_stability_summary": stab_meta,
            "compound_stability_table_head": compound_stab_df.head(20).to_dict(
                orient="list"
            )
            if not compound_stab_df.empty
            else {},
        },
    )
    if stability_llm:
        writer.add_text(stability_llm)

    # 4.4 Epistemic regimes
    writer.add_heading("4.4 Epistemic regimes", level=2)
    total_reg = info_mean + mis_mean + dis_mean
    if total_reg > 0:
        info_p = info_mean / total_reg
        mis_p = mis_mean / total_reg
        dis_p = dis_mean / total_reg

        writer.add_text(
            "Mean regime scores across elements suggest the following aggregate "
            "composition (normalised for display):"
        )
        writer.add_bullets(
            [
                f"Informational: {info_p * 100:.1f} percent",
                f"Misinformational: {mis_p * 100:.1f} percent",
                f"Disinformational: {dis_p * 100:.1f} percent",
            ]
        )
        writer.add_text(
            "These scores are derived from element level regime fields and should "
            "be interpreted as tendencies rather than hard classifications."
        )
    else:
        if not signatures_df.empty:
            writer.add_text(
                "Epistemic signatures were computed for elements, but mean regime "
                "scores were not attached to the element table. The file "
                "signatures.csv contains per element probabilities over "
                "informational, misinformational, and disinformational classes."
            )
        else:
            writer.add_text(
                "No epistemic label file was provided for this run, so regime "
                "composition signals are not available."
            )

    # 4.5 Language model metrics
    writer.add_heading("4.5 Language model perplexity", level=2)
    if lm_metrics:
        txt = (
            f"The external language model used in this run was "
            f"{lm_metrics.get('model', 'n/a')}."
        )
        writer.add_text(txt)
        if "perplexity" in lm_metrics:
            writer.add_text(
                f"It reports a perplexity of {lm_metrics['perplexity']:.3f} over "
                f"{lm_metrics.get('n_tokens', 'n/a')} tokens. Lower perplexity "
                "values indicate that the corpus conforms more closely to the "
                "model's expectations."
            )
    else:
        writer.add_text(
            "Language model metrics were not available for this run. Either the "
            "element LM did not run, or lm_metrics.json was not written."
        )

    # 4.6 Compound contexts
    writer.add_heading("4.6 Compound contexts", level=2)
    if compound_contexts_available:
        writer.add_text(
            "Compound contexts were generated for this run and stored in "
            "compound_contexts.json. Each entry provides example spans that "
            "characterise a compound's behaviour in the corpus."
        )
    else:
        writer.add_text(
            "Compound contexts were not generated in this run."
        )

    # ===========================#
    # 5. Discussion
    # ===========================#
    writer.add_heading("5. Discussion", level=1)
    writer.add_text(
        "Viewed as an informational field, the corpus exhibits a structured mix "
        "of central and peripheral elements. Dense graph cores correspond to "
        "highly reused concepts that anchor the discourse, whereas peripheral "
        "branches capture rarer or more specialised configurations."
    )
    writer.add_text(
        "The compound spectrum provides an intermediate scale between single "
        "elements and full documents. Stable compounds mark robust patterns of "
        "co occurrence that may reflect domain specific idioms or argument "
        "templates, while unstable compounds highlight more transient or context "
        "dependent structures."
    )
    writer.add_text(
        "Interpretation of these patterns should take into account corpus size, "
        "sampling strategy, and any fallbacks used for embeddings or stability. "
        "The absence of epistemic labels, where applicable, limits the ability "
        "to draw conclusions about misinformation or disinformation regimes."
    )

    # Optional LLM paragraph for the discussion
    discussion_llm = _llm_generate_paragraph(
        "discussion",
        {
            "elements": elem_info,
            "graph_metrics": graph_metrics,
            "n_compounds": n_compounds,
            "lm_metrics": lm_metrics,
            "regime_means": {
                "info_mean": info_mean,
                "mis_mean": mis_mean,
                "dis_mean": dis_mean,
            },
        },
    )
    if discussion_llm:
        writer.add_text(discussion_llm)

    # ===========================#
    # 6. Conclusion
    # ===========================#
    writer.add_heading("6. Conclusion", level=1)
    writer.add_text(
        "This report illustrates how the Hilbert pipeline converts a corpus into "
        "an informational graph with associated stability and compound structure. "
        "By combining span level embeddings, element metrics, graph topology, "
        "epistemic signatures, and language model scores, the system offers a "
        "multi scale view of how information is organised and propagated in text."
    )

    # ===========================#
    # 7. Reproducibility appendix
    # ===========================#
    writer.add_heading("7. Reproducibility appendix", level=1)
    writer.add_heading("7.1 Run configuration", level=2)

    writer.add_text(f"Results directory for this run: {results_dir}")
    writer.add_text("Settings recorded by the orchestrator:")
    if settings:
        rows = [["Key", "Value"]]
        for k, v in sorted(settings.items()):
            rows.append([str(k), str(v)])
        col_widths = [2.2 * 72, 3.0 * 72]
        writer.add_table(rows, col_widths)
    else:
        writer.add_text("(No run settings were recorded.)")

    writer.add_heading("7.2 Artifacts", level=2)
    writer.add_text(
        "Core artifacts for this run include hilbert_run.json, lsa_field.json, "
        "hilbert_elements.csv, span_element_fusion.csv, edges.csv, "
        "informational_compounds.json, compound_contexts.json (optional), "
        "signal_stability.csv, stability_meta.json, compound_stability.csv, "
        "element_roots.csv, element_cluster_metrics.json, signatures.csv/json, "
        "lm_metrics.json, and the various PNG figures. Where available, the "
        "hilbert_run.zip archive collects these into a single bundle."
    )

    # ===========================#
    # 8. Visual appendix
    # ===========================#
    writer.add_heading("8. Visual appendix", level=1)
    writer.add_text(
        "The following pages collate the graph snapshots and stability figures "
        "produced by the pipeline. They are included for rapid inspection; the "
        "original PNG files in the results directory can be reused for external "
        "figures or further analysis."
    )

    fig_paths = _collect_figure_paths(out_dir)
    if fig_paths:
        graph_paths, other_paths = _split_graph_and_other_figs(out_dir, fig_paths)

        def draw_image_full(path: str, caption: str):
            writer.new_page()
            c.setFont("Helvetica-Bold", 11)
            c.drawString(writer.margin_left, writer.y, caption)
            writer.y -= 18

            img = ImageReader(path)
            iw, ih = img.getSize()
            if iw <= 0 or ih <= 0:
                return
            aspect = ih / float(iw)

            img_w_avail = writer.width - writer.margin_left - writer.margin_right
            img_h_avail = writer.y - writer.margin_bottom - 24

            draw_w = img_w_avail
            draw_h = draw_w * aspect
            if draw_h > img_h_avail:
                draw_h = img_h_avail
                draw_w = draw_h / aspect

            x = writer.margin_left + (img_w_avail - draw_w) / 2.0
            y = writer.margin_bottom + (img_h_avail - draw_h) / 2.0

            c.drawImage(
                img,
                x,
                y,
                width=draw_w,
                height=draw_h,
                preserveAspectRatio=True,
            )

        for p in graph_paths:
            caption = f"Graph snapshot: {os.path.basename(p)}"
            draw_image_full(p, caption)

        for p in other_paths:
            caption = f"Figure: {os.path.basename(p)}"
            draw_image_full(p, caption)

    # Footer on last page
    writer.ensure_space(40)
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.darkgray)
    c.drawString(
        writer.margin_left,
        writer.margin_bottom - 10,
        "Hilbert Information Chemistry Lab - report generated by hilbert_report.py",
    )
    c.setFillColor(colors.black)

    c.save()
    print(f"[report] PDF report written to {pdf_path}")


# -----------------------------------------------------------------------------#
# Orchestrator facing wrapper
# -----------------------------------------------------------------------------#

def run_hilbert_report(out_dir: str, emit=DEFAULT_EMIT) -> None:
    """
    Orchestrator friendly entry point.

    - Emits lightweight start and end events.
    - Builds hilbert_report.pdf (or hilbert_report.txt if ReportLab is missing).
    """
    try:
        emit("log", {"stage": "hilbert_report", "event": "start"})
    except Exception:
        pass

    export_hilbert_report(out_dir)

    try:
        emit("log", {"stage": "hilbert_report", "event": "end"})
    except Exception:
        pass
