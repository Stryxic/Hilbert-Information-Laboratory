# =============================================================================
# hilbert_pipeline/hilbert_export.py — Summary PDF and ZIP Export (Enhanced)
# =============================================================================
"""
Generate a comprehensive summary report (PDF) and an export ZIP bundle
for each Hilbert pipeline run.

The PDF integrates:
  - Corpus metrics (spans, elements, entropy, coherence)
  - Pipeline run overview (run id, settings, corpus, results folder)
  - Stage-by-stage timeline (status, duration, dialectic role when available)
  - Dialectic structure summary (support / challenge relations)
  - Element root field (from element_roots.csv / element_cluster_metrics.json)
  - Compound field summary (from compound_metrics.json)
  - Compound table (from informational_compounds.json, top 10 by stability)
  - Regime composition chart (info/misinfo/disinfo, if available)
  - Compound context availability
  - Embedded figures from the run, with:
        * Graph progression snapshots (graph_*.png) laid out as smaller
          panels showing evolution as a sequence.
        * Other figures as full-width pages.
  - Metadata footer

The ZIP bundle includes core CSV/JSON/PNG/PDF artifacts for archiving.

Used by hilbert_orchestrator via run_full_export().
"""

from __future__ import annotations

import os
import re
import json
import zipfile
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd
import numpy as np

# -------------------------------------------------------------------------#
# Orchestrator compatibility emit
# -------------------------------------------------------------------------#
DEFAULT_EMIT = lambda *_a, **_k: None  # type: ignore

# -------------------------------------------------------------------------#
# Compatibility patch: Fix ReportLab md5 issue on some Python builds
# -------------------------------------------------------------------------#
import hashlib
import inspect

try:
    sig = inspect.signature(hashlib.md5)
    has_usedforsecurity = "usedforsecurity" in sig.parameters
except (TypeError, ValueError):
    # If we cannot inspect, assume we may need to handle the kwarg
    has_usedforsecurity = False

if not has_usedforsecurity:
    _old_md5 = hashlib.md5

    def _safe_md5(*args, **kwargs):
        kwargs.pop("usedforsecurity", None)
        return _old_md5(*args, **kwargs)

    hashlib.md5 = _safe_md5


# -------------------------------------------------------------------------#
# Try to load ReportLab
# -------------------------------------------------------------------------#
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib.utils import ImageReader

    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False
    print("[export][warn] reportlab not installed - PDF will be plain text.")


# -------------------------------------------------------------------------#
# Helpers for safe stats / IO
# -------------------------------------------------------------------------#
def _safe_mean(series: pd.Series, default: float = 0.0) -> float:
    """Numerically safe mean for possibly-empty or NaN-heavy series."""
    if series is None or len(series) == 0:
        return default
    arr = pd.to_numeric(series, errors="coerce").to_numpy()
    if arr.size == 0:
        return default
    if np.all(~np.isfinite(arr)):
        return default
    val = float(np.nanmean(arr))
    return val if np.isfinite(val) else default


def _read_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# -------------------------------------------------------------------------#
# Run summary and metrics readers
# -------------------------------------------------------------------------#
def _read_run_summary(out_dir: str) -> Dict[str, Any]:
    """Load hilbert_run.json if present."""
    path = os.path.join(out_dir, "hilbert_run.json")
    data = _read_json(path)
    return data if isinstance(data, dict) else {}


def _read_compound_metrics(out_dir: str) -> Dict[str, Any]:
    """
    Read aggregate compound metrics from compound_metrics.json, if present.
    """
    path = os.path.join(out_dir, "compound_metrics.json")
    data = _read_json(path)
    return data if isinstance(data, dict) else {}


def _read_compound_table(out_dir: str) -> pd.DataFrame:
    """
    Read per-compound data from informational_compounds.json, if present.
    """
    path = os.path.join(out_dir, "informational_compounds.json")
    data = _read_json(path)
    if not isinstance(data, dict):
        return pd.DataFrame()

    rows = []
    for cid, cdata in data.items():
        if not isinstance(cdata, dict):
            continue
        cid_val = cdata.get("compound_id", cid)
        reg = cdata.get("regime_profile", {}) or {}
        rows.append(
            {
                "compound_id": str(cid_val),
                "num_elements": cdata.get("num_elements"),
                "num_bonds": cdata.get("num_bonds"),
                "compound_stability": cdata.get("compound_stability"),
                "mean_temperature": cdata.get("mean_temperature"),
                "mean_info": reg.get("info", 0.0),
                "mean_misinfo": reg.get("misinfo", 0.0),
                "mean_disinfo": reg.get("disinfo", 0.0),
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
        "mean_info",
        "mean_misinfo",
        "mean_disinfo",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _read_elements_info(out_dir: str) -> dict:
    """
    Read basic statistics from hilbert_elements.csv.
    """
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

    if "mean_entropy" in df.columns:
        info["mean_entropy"] = _safe_mean(df["mean_entropy"], default=0.0)
    elif "entropy" in df.columns:
        info["mean_entropy"] = _safe_mean(df["entropy"], default=0.0)
    else:
        info["mean_entropy"] = 0.0

    if "mean_coherence" in df.columns:
        info["mean_coherence"] = _safe_mean(df["mean_coherence"], default=0.0)
    elif "coherence" in df.columns:
        info["mean_coherence"] = _safe_mean(df["coherence"], default=0.0)
    else:
        info["mean_coherence"] = 0.0

    return info


def _read_num_spans(out_dir: str) -> Optional[int]:
    """
    Use lsa_field.json embeddings length as span count, if available.
    """
    path = os.path.join(out_dir, "lsa_field.json")
    data = _read_json(path)
    if not isinstance(data, dict):
        return None
    emb = data.get("embeddings")
    if isinstance(emb, list):
        return len(emb)
    return None


def _read_regime_profile(out_dir: str) -> Tuple[float, float, float]:
    """
    Estimate global regime composition from hilbert_elements.csv scores.
    """
    path = os.path.join(out_dir, "hilbert_elements.csv")
    if not os.path.exists(path):
        return 0.0, 0.0, 0.0

    try:
        df = pd.read_csv(path)
    except Exception:
        return 0.0, 0.0, 0.0

    info = _safe_mean(df.get("info_score", pd.Series([], dtype=float)), 0.0)
    mis = _safe_mean(df.get("misinfo_score", pd.Series([], dtype=float)), 0.0)
    dis = _safe_mean(df.get("disinfo_score", pd.Series([], dtype=float)), 0.0)
    return info, mis, dis


def _read_root_stats(out_dir: str) -> Dict[str, Any]:
    """
    Summarize condensed root field (if available).
    """
    roots_path = os.path.join(out_dir, "element_roots.csv")
    metrics_path = os.path.join(out_dir, "element_cluster_metrics.json")

    stats: Dict[str, Any] = {
        "num_roots": None,
        "median_cluster_size": None,
        "max_cluster_size": None,
    }

    if os.path.exists(roots_path):
        try:
            rdf = pd.read_csv(roots_path)
            if "element" in rdf.columns:
                stats["num_roots"] = int(rdf["element"].astype(str).nunique())
            if "cluster_size" in rdf.columns:
                sizes = pd.to_numeric(rdf["cluster_size"], errors="coerce")
                sizes = sizes[np.isfinite(sizes)]
                if sizes.size:
                    stats["median_cluster_size"] = float(np.median(sizes))
                    stats["max_cluster_size"] = float(np.max(sizes))
        except Exception:
            pass

    stats["has_cluster_metrics"] = os.path.exists(metrics_path)
    return stats


def _has_compound_contexts(out_dir: str) -> bool:
    path = os.path.join(out_dir, "compound_contexts.json")
    return os.path.exists(path)


# -------------------------------------------------------------------------#
# Stage and dialectic helpers
# -------------------------------------------------------------------------#
def _extract_stage_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a merged view of stage rows using dialectic_graph if available,
    falling back to the plain stages dict otherwise.
    """
    stages = summary.get("stages", {}) or {}
    graph = summary.get("dialectic_graph", {}) or {}
    nodes = {n.get("id"): n for n in graph.get("nodes", []) or []}

    rows: List[Dict[str, Any]] = []

    if nodes:
        for sid, node in nodes.items():
            st = stages.get(sid, {})
            rows.append(
                {
                    "id": sid,
                    "label": node.get("label") or st.get("label") or sid,
                    "role": node.get("role", "structure"),
                    "status": st.get("status", node.get("status", "pending")),
                    "duration": st.get("duration", node.get("duration")),
                }
            )
    else:
        for sid, st in stages.items():
            rows.append(
                {
                    "id": sid,
                    "label": st.get("label", sid),
                    "role": "structure",
                    "status": st.get("status", "pending"),
                    "duration": st.get("duration"),
                }
            )

    rows.sort(key=lambda r: (r.get("label") or "", r.get("id") or ""))
    return rows


def _extract_dialectic_edges(summary: Dict[str, Any]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Map edge types to (source, target) tuples for a compact text summary.
    """
    graph = summary.get("dialectic_graph", {}) or {}
    edges = graph.get("edges", []) or []
    buckets: Dict[str, List[Tuple[str, str]]] = {}
    for e in edges:
        etype = str(e.get("type", "depends_on"))
        src = str(e.get("source", ""))
        tgt = str(e.get("target", ""))
        if not src or not tgt:
            continue
        buckets.setdefault(etype, []).append((src, tgt))
    return buckets


def _extract_artifact_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten artifact metadata into rows suitable for listing in the PDF.
    """
    arts = summary.get("artifacts", {}) or {}
    rows: List[Dict[str, Any]] = []
    for name, meta in arts.items():
        if not isinstance(meta, dict):
            continue
        rows.append(
            {
                "name": name,
                "kind": meta.get("kind", ""),
                "path": meta.get("path", ""),
            }
        )
    rows.sort(key=lambda r: (r.get("kind") or "", r.get("name") or ""))
    return rows


# -------------------------------------------------------------------------#
# Figure collection and rendering
# -------------------------------------------------------------------------#
def _collect_figure_paths(out_dir: str) -> List[str]:
    """
    Collect PNG figures from the run directory (including subfolders).
    Deduplicates and tries to present key figures first.
    """
    all_pngs: List[str] = []
    base = os.path.abspath(out_dir)

    for root, _, files in os.walk(out_dir):
        for fname in files:
            if fname.lower().endswith(".png"):
                all_pngs.append(os.path.join(root, fname))

    if not all_pngs:
        return []

    seen = set()
    unique: List[str] = []
    for p in sorted(all_pngs):
        rel = os.path.relpath(p, base).replace("\\", "/")
        if rel in seen:
            continue
        seen.add(rel)
        unique.append(p)

    priority_order = [
        "persistence_field.png",
        "stability_scatter.png",
        "stability_by_doc.png",
        "graph_full.png",
    ]

    def score(path: str) -> Tuple[int, str]:
        name = os.path.basename(path)
        try:
            idx = priority_order.index(name)
            return (idx, name)
        except ValueError:
            return (len(priority_order), name)

    unique.sort(key=score)
    return unique


def _split_graph_and_other_figs(out_dir: str, fig_paths: List[str]) -> Tuple[List[str], List[str]]:
    """
    Split figures into:
      - graph progression snapshots (graph_*.png) from the run root
      - other figures (persistence, stability, export graphs, etc)

    We only treat graph_*.png in the run root as progression snapshots
    to avoid duplicates from results_dir/figures.
    """
    base = os.path.abspath(out_dir)
    graph_paths: List[str] = []
    other_paths: List[str] = []

    # temporary mapping by basename for root only
    root_graphs: Dict[str, str] = {}

    graph_re = re.compile(r"^graph_(\d+|full)\.png$", re.IGNORECASE)

    for p in fig_paths:
        name = os.path.basename(p)
        rel_dir = os.path.relpath(os.path.dirname(p), base).replace("\\", "/")
        if graph_re.match(name):
            # treat only root level as the canonical progression snapshots
            if rel_dir in (".", ""):
                # keep one path per basename
                root_graphs.setdefault(name, p)
            else:
                # falls back into other figures
                other_paths.append(p)
        else:
            other_paths.append(p)

    # sort graph snapshots by numeric size, with "full" last
    def gkey(name_path: Tuple[str, str]) -> Tuple[int, int]:
        name, _p = name_path
        m = graph_re.match(name)
        if not m:
            return (0, 0)
        val = m.group(1)
        if val.lower() == "full":
            return (1, 10**9)
        try:
            return (0, int(val))
        except Exception:
            return (0, 0)

    graph_items = sorted(root_graphs.items(), key=gkey)
    graph_paths = [p for _name, p in graph_items]

    # keep other_paths stable and de-duplicated
    seen = set()
    dedup_other: List[str] = []
    for p in other_paths:
        if p in seen:
            continue
        seen.add(p)
        dedup_other.append(p)

    return graph_paths, dedup_other


def _draw_figure_page(c: "canvas.Canvas", width: float, height: float,
                      img_path: str, fig_index: int):
    """
    Render a single figure as a full-width page.
    """
    c.showPage()
    margin_x = 72
    margin_top = 64
    margin_bottom = 64

    c.setFont("Helvetica-Bold", 12)
    title = f"Figure {fig_index}: {os.path.basename(img_path)}"
    c.drawString(margin_x, height - margin_top, title)

    img_width_avail = width - 2 * margin_x
    img_height_avail = height - margin_top - margin_bottom - 24

    try:
        img = ImageReader(img_path)
        iw, ih = img.getSize()
        if iw <= 0 or ih <= 0:
            raise ValueError("Invalid image size")
        aspect = ih / float(iw)

        draw_w = img_width_avail
        draw_h = draw_w * aspect
        if draw_h > img_height_avail:
            draw_h = img_height_avail
            draw_w = draw_h / aspect

        x = margin_x + (img_width_avail - draw_w) / 2.0
        y = margin_bottom + (img_height_avail - draw_h) / 2.0

        c.drawImage(img, x, y, width=draw_w, height=draw_h, preserveAspectRatio=True)

        c.setFont("Helvetica-Oblique", 7)
        c.setFillColor(colors.darkgray)
        rel_dir = os.path.basename(os.path.dirname(img_path))
        rel_name = (
            f"{rel_dir}/{os.path.basename(img_path)}"
            if rel_dir and rel_dir != "."
            else os.path.basename(img_path)
        )
        c.drawString(margin_x, margin_bottom - 12, rel_name)
        c.setFillColor(colors.black)
    except Exception as e:
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.red)
        c.drawString(margin_x, height - margin_top - 16, f"[export] Failed to render figure: {e}")
        c.setFillColor(colors.black)


def _render_graph_progression(c: "canvas.Canvas", width: float, height: float,
                              graph_paths: List[str]):
    """
    Render graph_*.png snapshots as a grid of smaller panels that show
    the evolution of the graph as a progression.

    Layout: 3 columns x 4 rows per page (up to 12 graphs per page).
    """
    if not graph_paths:
        return

    cols = 3
    rows = 4
    per_page = cols * rows

    margin_x = 48
    margin_top = 60
    margin_bottom = 48

    total_pages = (len(graph_paths) + per_page - 1) // per_page
    idx = 0
    fig_counter = 1

    for page in range(total_pages):
        c.showPage()
        c.setFont("Helvetica-Bold", 13)
        c.drawString(72, height - margin_top + 8, "Graph progression snapshots")

        available_height = height - margin_top - margin_bottom - 20
        available_width = width - 2 * margin_x

        cell_w = available_width / cols
        cell_h = available_height / rows

        for r in range(rows):
            for col in range(cols):
                if idx >= len(graph_paths):
                    break

                img_path = graph_paths[idx]
                idx += 1

                cell_x0 = margin_x + col * cell_w
                cell_y_top = height - margin_top - 20 - r * cell_h

                # Panel title: graph size or label
                base = os.path.basename(img_path)
                label = base.replace("graph_", "").replace(".png", "")
                c.setFont("Helvetica", 7)
                c.setFillColor(colors.darkgray)
                c.drawString(cell_x0 + 2, cell_y_top - 10, f"{fig_counter}. {label}")
                c.setFillColor(colors.black)

                img_width_avail = cell_w - 8
                img_height_avail = cell_h - 26

                try:
                    img = ImageReader(img_path)
                    iw, ih = img.getSize()
                    if iw <= 0 or ih <= 0:
                        raise ValueError("Invalid image size")
                    aspect = ih / float(iw)

                    draw_w = img_width_avail
                    draw_h = draw_w * aspect
                    if draw_h > img_height_avail:
                        draw_h = img_height_avail
                        draw_w = draw_h / aspect

                    x = cell_x0 + (img_width_avail - draw_w) / 2.0 + 2
                    y = cell_y_top - 10 - draw_h

                    c.drawImage(
                        img,
                        x,
                        y,
                        width=draw_w,
                        height=draw_h,
                        preserveAspectRatio=True,
                    )
                except Exception:
                    c.setFont("Helvetica", 7)
                    c.setFillColor(colors.red)
                    c.drawString(
                        cell_x0 + 2,
                        cell_y_top - 24,
                        "[export] Failed to render snapshot.",
                    )
                    c.setFillColor(colors.black)

                fig_counter += 1

            if idx >= len(graph_paths):
                break


def _render_all_figures(c: "canvas.Canvas", width: float, height: float,
                        out_dir: str):
    """
    Add figure pages to the PDF:

      - Graph progression snapshots (graph_*.png in run root) as smaller
        panels in a grid, ordered by graph size.
      - All other PNG figures (persistence, stability, export graphs, etc)
        as full-width one-per-page pages.
    """
    fig_paths = _collect_figure_paths(out_dir)
    if not fig_paths:
        return

    graph_paths, other_paths = _split_graph_and_other_figs(out_dir, fig_paths)

    # Graph progression as panel grids
    if graph_paths:
        _render_graph_progression(c, width, height, graph_paths)

    # Remaining figures as full-width pages
    fig_idx = 1
    for p in other_paths:
        _draw_figure_page(c, width, height, p, fig_idx)
        fig_idx += 1


# -------------------------------------------------------------------------#
# Public API - PDF
# -------------------------------------------------------------------------#
def export_summary_pdf(out_dir: str):
    """
    Build a multi-section, stage-annotated PDF summary of the run.
    Falls back to a text summary if ReportLab is unavailable.
    """
    os.makedirs(out_dir, exist_ok=True)

    run_summary = _read_run_summary(out_dir)
    compound_stats = _read_compound_metrics(out_dir)
    compounds_df = _read_compound_table(out_dir)
    elem_info = _read_elements_info(out_dir)
    root_stats = _read_root_stats(out_dir)
    num_spans = _read_num_spans(out_dir)
    info_mean, mis_mean, dis_mean = _read_regime_profile(out_dir)
    compound_contexts_available = _has_compound_contexts(out_dir)

    stage_rows = _extract_stage_rows(run_summary)
    dialectic_edges = _extract_dialectic_edges(run_summary)
    artifact_rows = _extract_artifact_rows(run_summary)

    title = "Hilbert Information Chemistry Lab - Run Summary"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    run_id = run_summary.get("run_id", "n/a")
    corpus_dir = run_summary.get("corpus_dir", "n/a")
    results_dir = run_summary.get("results_dir", out_dir)
    settings = run_summary.get("settings", {}) or {}

    # ------------------------------------------------------------------#
    # Fallback: plain text summary
    # ------------------------------------------------------------------#
    if not _HAS_REPORTLAB:
        txt_path = os.path.join(out_dir, "hilbert_summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"{title}\nGenerated at: {timestamp}\n\n")
            f.write(f"Run id: {run_id}\n")
            f.write(f"Corpus: {corpus_dir}\n")
            f.write(f"Results: {results_dir}\n\n")

            f.write("Settings:\n")
            for k, v in settings.items():
                f.write(f"  - {k}: {v}\n")

            f.write("\nCorpus statistics:\n")
            f.write(f"  Total spans: {num_spans or 'n/a'}\n")
            f.write(f"  Elements: {elem_info.get('num_elements', 'n/a')}\n")
            f.write(f"  Tokens: {elem_info.get('num_tokens', 'n/a')}\n")
            f.write(f"  Mean entropy: {elem_info.get('mean_entropy', 0.0):.3f}\n")
            f.write(f"  Mean coherence: {elem_info.get('mean_coherence', 0.0):.3f}\n")

            if root_stats.get("num_roots") is not None:
                f.write("\nCondensed root field:\n")
                f.write(
                    f"  Root elements: {root_stats['num_roots']} "
                    f"(median cluster size={root_stats.get('median_cluster_size', 'n/a')}, "
                    f"max cluster size={root_stats.get('max_cluster_size', 'n/a')})\n"
                )

            if compound_stats:
                f.write("\nCompound field:\n")
                if "num_compounds" in compound_stats:
                    f.write(f"  Compounds formed: {compound_stats['num_compounds']}\n")
                if "mean_stability" in compound_stats:
                    f.write(
                        f"  Mean compound stability: "
                        f"{compound_stats['mean_stability']:.3f}\n"
                    )
                if "stability_range" in compound_stats:
                    lo, hi = compound_stats["stability_range"]
                    f.write(
                        f"  Stability range: {float(lo):.3f} - {float(hi):.3f}\n"
                    )

            if compound_contexts_available:
                f.write("\nCompound contexts: available (compound_contexts.json)\n")
            else:
                f.write("\nCompound contexts: not generated.\n")

            if stage_rows:
                f.write("\nStages:\n")
                for r in stage_rows:
                    f.write(
                        f"  - {r['label']} [{r['id']}]: status={r['status']}, "
                        f"role={r['role']}, duration={r.get('duration')}\n"
                    )

            if artifact_rows:
                f.write("\nArtifacts:\n")
                for a in artifact_rows:
                    f.write(
                        f"  - {a['name']} (kind={a['kind']}): {a['path']}\n"
                    )

        print("[export][warn] reportlab not available; wrote text summary.")
        return

    # ------------------------------------------------------------------#
    # PDF rendering
    # ------------------------------------------------------------------#
    pdf_path = os.path.join(out_dir, "hilbert_summary.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    y = height - 72

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, title)
    y -= 20
    c.setFont("Helvetica", 9)
    c.drawString(72, y, f"Generated at: {timestamp}")
    y -= 12
    c.drawString(72, y, f"Run id: {run_id}")
    y -= 12
    c.drawString(72, y, f"Corpus: {corpus_dir}")
    y -= 12
    c.drawString(72, y, f"Results folder: {results_dir}")
    y -= 24

    # Settings
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y, "Pipeline Settings")
    y -= 16
    c.setFont("Helvetica", 9)
    if settings:
        for k, v in sorted(settings.items()):
            c.drawString(84, y, f"{k}: {v}")
            y -= 12
            if y < 120:
                c.showPage()
                y = height - 72
                c.setFont("Helvetica", 9)
    else:
        c.drawString(84, y, "(no settings recorded)")
        y -= 14

    y -= 4

    # Corpus summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y, "Corpus Summary")
    y -= 16
    c.setFont("Helvetica", 10)
    lines = [
        f"Total spans analyzed: {num_spans or 'n/a'}",
        f"Unique informational elements: {elem_info.get('num_elements', 'n/a')}",
        f"Distinct element tokens: {elem_info.get('num_tokens', 'n/a')}",
        f"Mean element entropy: {elem_info.get('mean_entropy', 0.0):.3f}",
        f"Mean element coherence: {elem_info.get('mean_coherence', 0.0):.3f}",
    ]
    for line in lines:
        c.drawString(84, y, line)
        y -= 14
        if y < 120:
            c.showPage()
            y = height - 72
            c.setFont("Helvetica", 10)

    # Condensed root field
    if root_stats.get("num_roots") is not None:
        y -= 4
        c.setFont("Helvetica-Bold", 11)
        c.drawString(72, y, "Condensed Element Roots")
        y -= 14
        c.setFont("Helvetica", 9)
        c.drawString(
            84,
            y,
            f"Root elements: {root_stats['num_roots']} "
            f"(median cluster size={root_stats.get('median_cluster_size', 'n/a')}, "
            f"max cluster size={root_stats.get('max_cluster_size', 'n/a')})",
        )
        y -= 14
        if root_stats.get("has_cluster_metrics"):
            c.drawString(
                84,
                y,
                "Detailed cluster metrics: element_cluster_metrics.json",
            )
            y -= 14

    # Compound field summary
    if compound_stats:
        if y < 120:
            c.showPage()
            y = height - 72
        y -= 4
        c.setFont("Helvetica-Bold", 11)
        c.drawString(72, y, "Compound Field Summary")
        y -= 14
        c.setFont("Helvetica", 9)
        if "num_compounds" in compound_stats:
            c.drawString(
                84,
                y,
                f"Compounds formed: {compound_stats['num_compounds']}",
            )
            y -= 12
        if "mean_stability" in compound_stats:
            c.drawString(
                84,
                y,
                f"Mean compound stability: "
                f"{compound_stats['mean_stability']:.3f}",
            )
            y -= 12
        if "stability_range" in compound_stats:
            lo, hi = compound_stats["stability_range"]
            c.drawString(
                84,
                y,
                f"Stability range: {float(lo):.3f} - {float(hi):.3f}",
            )
            y -= 12

    # Regime composition chart
    y -= 6
    if y < 160:
        c.showPage()
        y = height - 72
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y, "Mean Regime Composition")
    y -= 26

    total_reg = info_mean + mis_mean + dis_mean
    if total_reg > 0:
        info_p = info_mean / total_reg
        mis_p = mis_mean / total_reg
        dis_p = dis_mean / total_reg
        bars = [
            ("Informational", info_p, colors.green),
            ("Misinformational", mis_p, colors.orange),
            ("Disinformational", dis_p, colors.red),
        ]
        x0 = 84
        bar_height = 10
        bar_width = 260
        for label, val, col in bars:
            c.setFillColor(col)
            c.rect(x0, y, bar_width * val, bar_height, stroke=0, fill=1)
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 7)
            c.drawString(x0 + 3, y + 2, f"{label} {val*100:.1f}%")
            c.setFillColor(colors.black)
            y -= (bar_height + 6)
        y -= 10
    else:
        c.setFont("Helvetica", 9)
        c.drawString(84, y, "Regime signals not available.")
        y -= 18

    # Compound contexts
    c.setFont("Helvetica-Bold", 11)
    c.drawString(72, y, "Compound Contexts")
    y -= 14
    c.setFont("Helvetica", 9)
    if compound_contexts_available:
        c.drawString(
            84,
            y,
            "Compound contexts generated (see compound_contexts.json).",
        )
    else:
        c.drawString(
            84,
            y,
            "Compound contexts were not generated for this run.",
        )
    y -= 20

    # Stages and timeline
    if stage_rows:
        if y < 180:
            c.showPage()
            y = height - 72
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, y, "Pipeline Stages and Timeline")
        y -= 18

        headers = ["Stage", "Role", "Status", "Duration (s)"]
        data: List[List[str]] = [headers]
        for r in stage_rows:
            dur = r.get("duration")
            if dur is None:
                dur_str = "-"
            else:
                try:
                    dur_str = f"{float(dur):.2f}"
                except Exception:
                    dur_str = str(dur)
            data.append(
                [
                    str(r.get("label") or r.get("id") or ""),
                    str(r.get("role") or ""),
                    str(r.get("status") or ""),
                    dur_str,
                ]
            )

        table = Table(data, colWidths=[2.1 * inch, 1.0 * inch, 1.1 * inch, 1.0 * inch])
        style = TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
        table.setStyle(style)

        table_height = (len(data) + 1) * 11
        table.wrapOn(c, width, table_height)
        table.drawOn(c, 72, max(80, y - table_height))
        y = max(80, y - table_height - 16)

    # Dialectic structure summary
    if dialectic_edges:
        if y < 160:
            c.showPage()
            y = height - 72
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, y, "Dialectic Structure Summary")
        y -= 16
        c.setFont("Helvetica", 9)
        for etype, edges in sorted(dialectic_edges.items()):
            c.drawString(84, y, f"Edge type: {etype} (n = {len(edges)})")
            y -= 12
            max_edges_to_show = 6
            for i, (src, tgt) in enumerate(edges[:max_edges_to_show]):
                c.drawString(96, y, f"{src} -> {tgt}")
                y -= 10
                if y < 80:
                    c.showPage()
                    y = height - 72
                    c.setFont("Helvetica", 9)
            if len(edges) > max_edges_to_show:
                c.drawString(
                    96,
                    y,
                    f"... {len(edges) - max_edges_to_show} more edges not shown",
                )
                y -= 12
            y -= 4

    # Artifacts
    if artifact_rows:
        if y < 160:
            c.showPage()
            y = height - 72
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, y, "Pipeline Artifacts")
        y -= 18
        c.setFont("Helvetica", 8)
        for a in artifact_rows:
            line = f"{a['name']} (kind={a['kind']}): {a['path']}"
            c.drawString(84, y, line[:110])
            y -= 10
            if y < 72:
                c.showPage()
                y = height - 72
                c.setFont("Helvetica", 8)

    # Top compounds
    if not compounds_df.empty and "compound_stability" in compounds_df.columns:
        if y < 200:
            c.showPage()
            y = height - 72

        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, y, "Top Compounds by Stability")
        y -= 18

        cols = [
            "compound_id",
            "num_elements",
            "num_bonds",
            "compound_stability",
            "mean_temperature",
        ]
        available_cols = [col for col in cols if col in compounds_df.columns]

        top_df = (
            compounds_df.dropna(subset=["compound_stability"])
            .sort_values("compound_stability", ascending=False)
            .head(10)[available_cols]
        )

        if not top_df.empty:
            data = [available_cols] + [
                [str(x)[:12] for x in row] for row in top_df.to_numpy()
            ]

            table = Table(data, colWidths=[1.1 * inch] * len(available_cols))
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
            table_height = (len(data) + 1) * 12
            table.wrapOn(c, width, table_height)
            table.drawOn(c, 72, max(80, y - table_height))
            y = max(80, y - table_height - 20)
        else:
            c.setFont("Helvetica", 9)
            c.drawString(
                84,
                y,
                "No per-compound stability records available for table view.",
            )
            y -= 14

    # Footer on last text page
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.darkgray)
    c.drawString(
        72,
        40,
        "Hilbert Information Chemistry Lab - Generated by hilbert_orchestrator",
    )
    c.setFillColor(colors.black)

    # Figure pages: graph progression panels + other figures
    _render_all_figures(c, width, height, out_dir)

    c.save()
    print(f"[export] PDF summary written to {pdf_path}")


# -------------------------------------------------------------------------#
# Public API - ZIP
# -------------------------------------------------------------------------#
def export_zip(out_dir: str):
    """
    Create a compact ZIP of core outputs for this run.
    """
    base = os.path.abspath(out_dir)
    run_name = os.path.basename(base.rstrip(os.sep))
    zip_name = f"{run_name}.zip"
    zip_path = os.path.join(base, zip_name)

    include_ext = {".csv", ".json", ".png", ".pdf", ".txt"}
    include_exact = {
        "lsa_field.json",
        "hilbert_elements.csv",
        "edges.csv",
        "element_descriptions.json",
        "informational_compounds.json",
        "compound_metrics.json",
        "element_roots.csv",
        "element_clusters.json",
        "element_cluster_metrics.json",
        "signal_stability.csv",
        "compound_contexts.json",
        "hilbert_summary.pdf",
        "hilbert_summary.txt",
        "hilbert_run.json",
    }

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(base):
            for fname in files:
                fpath = os.path.join(root, fname)
                if not os.path.isfile(fpath):
                    continue
                rel_dir = os.path.relpath(root, base)
                ext = os.path.splitext(fname)[1].lower()
                rel_name = fname
                if rel_dir not in (".", ""):
                    rel_name = os.path.join(rel_dir, fname)
                if fname in include_exact or ext in include_ext:
                    arcname = os.path.join(run_name, rel_name)
                    zf.write(fpath, arcname=arcname)

    print(f"[export] Created archive: {zip_path}")


# -------------------------------------------------------------------------#
# Orchestrator-facing convenience
# -------------------------------------------------------------------------#
def run_full_export(out_dir: str, emit=DEFAULT_EMIT) -> None:
    """
    High level export routine used by the orchestrator stage.

    - Builds an integrated PDF summary of the run.
    - Creates a ZIP archive with key artifacts.
    - Emits lightweight stage events for the API.
    """
    try:
        emit("log", {"stage": "export", "event": "start"})
    except Exception:
        pass

    export_summary_pdf(out_dir)
    export_zip(out_dir)

    try:
        emit("log", {"stage": "export", "event": "end"})
    except Exception:
        pass
