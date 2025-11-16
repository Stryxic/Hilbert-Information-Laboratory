# =============================================================================
# hilbert_pipeline/hilbert_export.py — Summary PDF and ZIP Export (Enhanced)
# =============================================================================
"""
Generate a comprehensive summary report (PDF) and an export ZIP bundle
for each Hilbert pipeline run.

Includes:
  - Corpus metrics (spans, elements, entropy, coherence)
  - Element root field (from element_roots.csv / element_cluster_metrics.json)
  - Compound field summary (from compound_metrics.json)
  - Compound table (from informational_compounds.json, top 10 by stability)
  - Regime composition chart (info/misinfo/disinfo)
  - Optional compound context presence
  - Metadata footer

Used by hilbert_orchestrator.
"""

from __future__ import annotations

import os
import json
import zipfile
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np

# -------------------------------------------------------------------------
# Compatibility patch: Fix ReportLab md5 issue on some Python builds
# -------------------------------------------------------------------------
import hashlib
import inspect

try:
    sig = inspect.signature(hashlib.md5)
    has_usedforsecurity = "usedforsecurity" in sig.parameters
except (TypeError, ValueError):
    # If we can't inspect, assume we may need to handle the kwarg
    has_usedforsecurity = False

if not has_usedforsecurity:
    _old_md5 = hashlib.md5

    def _safe_md5(*args, **kwargs):
        kwargs.pop("usedforsecurity", None)
        return _old_md5(*args, **kwargs)

    hashlib.md5 = _safe_md5


# -------------------------------------------------------------------------
# Try to load ReportLab
# -------------------------------------------------------------------------
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import Table, TableStyle
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False
    print("[export][warn] reportlab not installed - PDF will be plain text.")


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
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


def _read_compound_metrics(out_dir: str) -> Dict[str, Any]:
    """
    Read aggregate compound metrics from compound_metrics.json, if present.

    Expected example:
      {
        "num_compounds": 11,
        "timestamp": "...",
        "mean_stability": 0.38,
        "stability_range": [0.12, 0.79],
        "mean_info": ...,
        "mean_misinfo": ...,
        "mean_disinfo": ...
      }
    """
    path = os.path.join(out_dir, "compound_metrics.json")
    data = _read_json(path)
    return data if isinstance(data, dict) else {}


def _read_compound_table(out_dir: str) -> pd.DataFrame:
    """
    Read per-compound data from informational_compounds.json, if present.

    Expected shape:
      {
        "C0001": {
          "compound_id": "C0001",
          "elements": [...],
          "num_elements": int,
          "num_bonds": int,
          "compound_stability": float,
          "mean_temperature": float,
          "regime_profile": {
            "info": float,
            "misinfo": float,
            "disinfo": float
          }
        },
        ...
      }
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
    # ensure numeric where appropriate
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
    Summarize condensed root field (if available):
      - number of roots
      - median cluster size
      - maximum cluster size
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

    # cluster_metrics currently not used numerically here, but we may later
    if os.path.exists(metrics_path):
        stats["has_cluster_metrics"] = True
    else:
        stats["has_cluster_metrics"] = False

    return stats


def _has_compound_contexts(out_dir: str) -> bool:
    path = os.path.join(out_dir, "compound_contexts.json")
    return os.path.exists(path)


# -------------------------------------------------------------------------
# Public API - PDF
# -------------------------------------------------------------------------
def export_summary_pdf(out_dir: str):
    """
    Build a multi-section PDF summary of the run.
    Falls back to a text summary if ReportLab is unavailable.
    """
    os.makedirs(out_dir, exist_ok=True)

    compound_stats = _read_compound_metrics(out_dir)
    compounds_df = _read_compound_table(out_dir)
    elem_info = _read_elements_info(out_dir)
    root_stats = _read_root_stats(out_dir)
    num_spans = _read_num_spans(out_dir)
    info_mean, mis_mean, dis_mean = _read_regime_profile(out_dir)
    compound_contexts_available = _has_compound_contexts(out_dir)

    title = "Hilbert Information Chemistry Lab — Run Summary"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------
    # Fallback: plain text summary
    # ------------------------------------------------------------------
    if not _HAS_REPORTLAB:
        txt_path = os.path.join(out_dir, "hilbert_summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"{title}\nGenerated at: {timestamp}\n\n")
            f.write(f"Total spans: {num_spans or 'n/a'}\n")
            f.write(f"Elements: {elem_info.get('num_elements', 'n/a')}\n")
            f.write(f"Tokens: {elem_info.get('num_tokens', 'n/a')}\n")
            f.write(
                f"Mean entropy: {elem_info.get('mean_entropy', 0.0):.3f}\n"
            )
            f.write(
                f"Mean coherence: {elem_info.get('mean_coherence', 0.0):.3f}\n"
            )

            if root_stats.get("num_roots") is not None:
                f.write("\nCondensed root field:\n")
                f.write(
                    f"  Root elements: {root_stats['num_roots']} "
                    f"(median cluster size="
                    f"{root_stats.get('median_cluster_size', 'n/a')}, "
                    f"max cluster size="
                    f"{root_stats.get('max_cluster_size', 'n/a')})\n"
                )

            if compound_stats:
                f.write("\nCompound field:\n")
                if "num_compounds" in compound_stats:
                    f.write(
                        f"  Compounds formed: "
                        f"{compound_stats['num_compounds']}\n"
                    )
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

        print("[export][warn] reportlab not available; wrote text summary.")
        return

    # ------------------------------------------------------------------
    # PDF rendering
    # ------------------------------------------------------------------
    pdf_path = os.path.join(out_dir, "hilbert_summary.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    y = height - 72

    # --- Title ---------------------------------------------------------
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, title)
    y -= 20
    c.setFont("Helvetica", 9)
    c.drawString(72, y, f"Generated at: {timestamp}")
    y -= 10
    c.drawString(72, y, f"Output folder: {out_dir}")
    y -= 25

    # --- Section 1: Corpus summary ------------------------------------
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y, "Corpus Summary")
    y -= 15
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

    # --- Section 1b: Condensed root field ------------------------------
    if root_stats.get("num_roots") is not None:
        y -= 4
        c.setFont("Helvetica-Bold", 11)
        c.drawString(72, y, "Condensed Element Roots")
        y -= 13
        c.setFont("Helvetica", 9)
        c.drawString(
            84,
            y,
            f"Root elements: {root_stats['num_roots']} "
            f"(median cluster size="
            f"{root_stats.get('median_cluster_size', 'n/a')}, "
            f"max cluster size="
            f"{root_stats.get('max_cluster_size', 'n/a')})",
        )
        y -= 14
        if root_stats.get("has_cluster_metrics"):
            c.drawString(
                84,
                y,
                "Detailed cluster metrics: element_cluster_metrics.json",
            )
            y -= 14

    # --- Section 1c: Compound field summary ----------------------------
    if compound_stats:
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
                f"Stability range: {float(lo):.3f} – {float(hi):.3f}",
            )
            y -= 12

    # --- Section 2: Regime Composition Chart ---------------------------
    y -= 6
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

    # --- Section 3: Compound contexts presence -------------------------
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

    # --- Section 4: Compound Field Table --------------------------------
    if not compounds_df.empty and "compound_stability" in compounds_df.columns:
        if y < 160:
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

    # --- Footer ---------------------------------------------------------
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.darkgray)
    c.drawString(
        72,
        40,
        "Hilbert Information Chemistry Lab – Generated by hilbert_orchestrator",
    )

    c.showPage()
    c.save()
    print(f"[export] PDF summary written to {pdf_path}")


# -------------------------------------------------------------------------
# Public API - ZIP
# -------------------------------------------------------------------------
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
    }

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(base):
            fpath = os.path.join(base, fname)
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if fname in include_exact or ext in include_ext:
                arcname = os.path.join(run_name, fname)
                zf.write(fpath, arcname=arcname)

    print(f"[export] Created archive: {zip_path}")
