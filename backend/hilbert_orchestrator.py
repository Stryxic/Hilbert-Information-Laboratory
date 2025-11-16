# =============================================================================
# hilbert_orchestrator.py — Unified Hilbert Information Chemistry Pipeline
# =============================================================================
#
# Orchestrates the full Hilbert pipeline for a given corpus directory:
#
#   1. LSA spectral field + span map (lsa_layer.build_lsa_field)
#   2. Element table construction with embeddings (hilbert_elements.csv)
#   3. Element condensation (condense_elements.run_condensation)
#   4. Molecule construction & compound aggregation (molecule_layer)
#   5. Optional span→element fusion & compound context aggregation (fusion)
#   6. Signal stability metrics (stability_layer)
#   7. Persistence / stability visualisations (persistence_visuals)
#   8. Element description labels (element_labels)
#   9. Frontend normalisation post-pass on hilbert_elements.csv
#  10. PDF summary & ZIP export (hilbert_export)
#  11. Sanity checks & diagnostics summary (hilbert_sanity, optional)
#
# This file is designed to be imported from FastAPI (app.py) via:
#     from hilbert_orchestrator import run_pipeline, PIPELINE_STEPS, get_pipeline_plan
#
# And also callable from the CLI:
#     python hilbert_orchestrator.py --corpus uploaded_corpus --out results/hilbert_run
# =============================================================================

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import threading

__all__ = [
    "PipelineContext",
    "run_pipeline",
    "PIPELINE_STEPS",
    "get_pipeline_plan",
]

# -------------------------------------------------------------------------
# Optional imports for pipeline layers
# -------------------------------------------------------------------------

# - LSA spectral field
try:
    # Package-style (backend/hilbert_pipeline/lsa_layer.py)
    from hilbert_pipeline.lsa_layer import build_lsa_field
except ImportError:
    # Flat module fallback (backend/lsa_layer.py)
    from lsa_layer import build_lsa_field  # type: ignore

# - Element condensation
try:
    # Preferred: flat module in backend root
    from condense_elements import run_condensation as run_condensation_layer
except ImportError:
    # Package-style fallback if you ever move it
    from hilbert_pipeline.condense_elements import run_condensation as run_condensation_layer  # type: ignore

# - Element labels / descriptions
try:
    from hilbert_pipeline.element_labels import build_element_descriptions
except ImportError:
    from element_labels import build_element_descriptions  # type: ignore

# - Molecules & compounds
try:
    from hilbert_pipeline.molecule_layer import (
        compute_molecule_stability,
        compute_molecule_temperature,
        aggregate_compounds,
        export_molecule_summary,
    )
except ImportError:
    # Correct flat fallback
    from hilbert_pipeline.molecule_layer import (  # type: ignore
        compute_molecule_stability,
        compute_molecule_temperature,
        aggregate_compounds,
        export_molecule_summary,
    )

# - Span→element fusion and compound context
try:
    from hilbert_pipeline.fusion import (
        fuse_spans_to_elements,
        aggregate_compound_context,
    )
except ImportError:
    from fusion import fuse_spans_to_elements, aggregate_compound_context  # type: ignore

# - Stability layer
try:
    from hilbert_pipeline.stability_layer import compute_signal_stability
except ImportError:
    from stability_layer import compute_signal_stability  # type: ignore

# - Persistence visuals
try:
    from hilbert_pipeline.persistence_visuals import plot_persistence_field
except ImportError:
    from persistence_visuals import plot_persistence_field  # type: ignore

# - Export (PDF + ZIP)
try:
    from hilbert_pipeline.hilbert_export import export_summary_pdf, export_zip
except ImportError:
    from hilbert_export import export_summary_pdf, export_zip  # type: ignore

# - Sanity checks (optional, separate module as requested)
try:
    from hilbert_pipeline.hilbert_sanity import check_run_sanity  # type: ignore
except ImportError:
    try:
        from hilbert_sanity import check_run_sanity  # type: ignore
    except Exception:
        check_run_sanity = None  # type: ignore


# -------------------------------------------------------------------------
# Optional native backend (diagnostics only)
# -------------------------------------------------------------------------
try:
    import hilbert_native as _hn  # type: ignore

    HN = _hn
    HILBERT_NATIVE_AVAILABLE = True
except Exception as _e:
    HN = None
    HILBERT_NATIVE_AVAILABLE = False
    print(f"[native][warn] hilbert_native backend not available: {_e}")


# -------------------------------------------------------------------------
# Pipeline Context + Thread-safe Emitter
# -------------------------------------------------------------------------

EmitFn = Callable[[str, Dict[str, Any]], None]


@dataclass
class StageRecord:
    """Structured record of each major pipeline stage."""

    name: str
    started_at: float
    finished_at: Optional[float] = None
    status: str = "pending"  # "pending", "ok", "error", "skipped"
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        if self.finished_at is None:
            return None
        return self.finished_at - self.started_at


@dataclass
class PipelineContext:
    corpus_dir: str
    out_dir: str
    start_time: float = field(default_factory=time.time)
    log_messages: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    aborted: bool = False

    # Optional external emitter: e.g. FastAPI websocket / SSE hook
    external_emit: Optional[EmitFn] = None

    # Internal stage registry
    stages: List[StageRecord] = field(default_factory=list)

    # Thread-safety for emit in multi-threaded environments
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def emit(self, event_type: str, payload: Dict[str, Any]):
        """
        Unified event emitter for pipeline stages.

        - Always prints to stdout.
        - Optionally forwards to external_emit in a thread-safe manner.
        - Stores log and error messages locally for later inspection.
        """
        with self._lock:
            payload = dict(payload or {})
            message = str(payload.get("message", "")).strip()
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            pretty_prefix = (
                f"[{event_type}] " if event_type not in ("log", "stage") else ""
            )

            if message:
                line = f"{ts} {pretty_prefix}{message}"
            else:
                line = f"{ts} [{event_type}] {payload}"

            # Console logging
            print(line)

            # Store logs / errors
            if event_type in ("log", "stage"):
                if message:
                    self.log_messages.append(message)
            elif event_type == "error":
                if message:
                    self.errors.append(message)

            # Forward to external emitter (if provided) with error shielding
            if self.external_emit is not None:
                try:
                    self.external_emit(event_type, {"timestamp": ts, **payload})
                except Exception as ext_exc:
                    # Do not let a failing external emitter break the pipeline
                    print(f"[emit][warn] external_emit raised: {ext_exc}")

    # Stage helpers ------------------------------------------------------
    def begin_stage(
        self, name: str, meta: Optional[Dict[str, Any]] = None
    ) -> StageRecord:
        rec = StageRecord(name=name, started_at=time.time(), meta=meta or {})
        self.stages.append(rec)
        self.emit("stage", {"message": f"▶ {name} — starting"})
        return rec

    def end_stage(
        self, rec: StageRecord, status: str = "ok", error: Optional[str] = None
    ):
        rec.finished_at = time.time()
        rec.status = status
        rec.error = error

        if status == "ok":
            self.emit(
                "stage",
                {
                    "message": f"✓ {rec.name} — completed in {rec.duration:.2f}s"
                    if rec.duration is not None
                    else f"✓ {rec.name} — completed",
                },
            )
        elif status == "skipped":
            self.emit(
                "stage",
                {"message": f"⟲ {rec.name} — skipped: {error or 'no-op'}"},
            )
        else:
            self.emit(
                "error",
                {
                    "message": f"✗ {rec.name} — failed: {error or 'unknown error'}",
                },
            )


def _stage(ctx: PipelineContext, msg: str):
    """Compat helper for shorter stage-style log lines."""
    ctx.emit("stage", {"message": msg})


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Helper: build hilbert_elements.csv with embeddings
# -------------------------------------------------------------------------

def _build_hilbert_elements_csv(
    ctx: PipelineContext,
    lsa_result: Dict[str, Any],
) -> Path:
    """
    Construct hilbert_elements.csv in ctx.out_dir based on the LSA result.

    Strategy:
      - Use span_map + per-span embeddings.
      - Use "elements" occurrence records to get (element, doc, span_id).
      - Aggregate per (element, doc) for document-aware TF.
      - Add corpus-level metrics from element_metrics.
      - Compute a corpus-level centroid embedding per element and attach it.

    The resulting CSV has columns (typical):
      element, token, doc, tf_doc, tf, doc_freq,
      mean_entropy, mean_coherence,
      embedding (JSON list)
    """
    out_dir = Path(ctx.out_dir)
    _ensure_dir(out_dir)

    span_map = lsa_result.get("field", {}).get("span_map", []) or []
    elements_occ = lsa_result.get("elements", []) or []
    metrics_rows = lsa_result.get("element_metrics", []) or []

    if not elements_occ or not metrics_rows:
        msg = "[elements] No element data returned from LSA layer."
        ctx.emit("error", {"message": msg})
        raise RuntimeError(msg)

    # Build occurrence and metrics tables
    occ_df = pd.DataFrame(elements_occ)
    metrics_df = pd.DataFrame(metrics_rows)

    if "element" not in occ_df.columns:
        raise RuntimeError("[elements] occurrence table missing 'element' column.")
    if "element" not in metrics_df.columns:
        raise RuntimeError("[elements] metrics table missing 'element' column.")

    # Normalise doc column
    if "doc" not in occ_df.columns and "file" in occ_df.columns:
        occ_df["doc"] = occ_df["file"]
    if "doc" not in occ_df.columns:
        occ_df["doc"] = "corpus"

    # Attach metrics (mean_entropy, mean_coherence, count, df) per element
    metrics_df = metrics_df.set_index("element")
    metrics_df = metrics_df.rename(
        columns={
            "count": "tf",
            "df": "doc_freq",
        }
    )

    # Compose span embedding matrix
    emb = np.asarray(lsa_result.get("field", {}).get("embeddings", []), dtype=float)
    if emb.ndim != 2:
        raise RuntimeError("[elements] LSA embeddings are not 2D.")

    # Build mapping span_id -> index in embedding matrix
    span_index_map: Dict[int, int] = {}
    for i, s in enumerate(span_map):
        sid = s.get("span_id", i)
        try:
            sid_int = int(sid)
        except Exception:
            sid_int = i
        span_index_map[sid_int] = i

    # Compute a centroid embedding per element (corpus-wide)
    element_embeddings: Dict[str, List[float]] = {}
    for el, group in occ_df.groupby("element"):
        el_str = str(el)
        sids = group.get("span_id")
        if sids is None:
            continue
        idxs = [span_index_map.get(int(s), None) for s in sids]
        idxs = [i for i in idxs if i is not None]
        if not idxs:
            continue
        E = emb[idxs]
        centroid = E.mean(axis=0)
        element_embeddings[el_str] = centroid.astype(float).tolist()

    # Aggregate per (element, doc) for hilbert_elements.csv
    rows: List[Dict[str, Any]] = []
    for (el, doc), g in occ_df.groupby(["element", "doc"]):
        el_str = str(el)
        doc_str = str(doc)
        tf_doc = int(len(g))

        if el_str in metrics_df.index:
            m = metrics_df.loc[el_str]
            tf_corpus = float(m.get("tf", tf_doc))
            df = int(m.get("doc_freq", 1))
            me = float(m.get("mean_entropy", 0.0))
            mc = float(m.get("mean_coherence", 0.0))
        else:
            tf_corpus = float(tf_doc)
            df = 1
            me = 0.0
            mc = 0.0

        emb_vec = element_embeddings.get(el_str, [])
        rows.append(
            {
                "element": el_str,
                "token": el_str,
                "doc": doc_str,
                "tf_doc": tf_doc,
                "tf": tf_corpus,
                "doc_freq": df,
                "mean_entropy": me,
                "mean_coherence": mc,
                "embedding": json.dumps(emb_vec),
            }
        )

    elements_df = pd.DataFrame(rows)
    out_path = out_dir / "hilbert_elements.csv"
    elements_df.to_csv(out_path, index=False)

    _stage(
        ctx,
        f"[elements] hilbert_elements.csv written with "
        f"{len(elements_df)} rows for {elements_df['element'].nunique()} elements.",
    )

    return out_path


# -------------------------------------------------------------------------
# Helper: compound metrics summary
# -------------------------------------------------------------------------

def _write_compound_metrics_json(
    ctx: PipelineContext,
    compound_df: pd.DataFrame,
) -> Optional[Path]:
    """
    Compute and write high-level compound metrics to compound_metrics.json.

    This is used by hilbert_export.export_summary_pdf.
    """
    if compound_df is None or compound_df.empty:
        return None

    path = Path(ctx.out_dir) / "compound_metrics.json"

    # numeric helpers with NaN safety
    def safemean(series: pd.Series) -> float:
        if series is None or series.empty:
            return 0.0
        arr = series.to_numpy(dtype=float)
        if arr.size == 0:
            return 0.0
        return float(np.nanmean(arr))

    def saferange(series: pd.Series) -> List[float]:
        if series is None or series.empty:
            return [0.0, 0.0]
        arr = series.to_numpy(dtype=float)
        if arr.size == 0:
            return [0.0, 0.0]
        return [float(np.nanmin(arr)), float(np.nanmax(arr))]

    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_compounds": int(compound_df["compound_id"].nunique()),
        "mean_stability": safemean(
            compound_df.get("compound_stability", pd.Series([], dtype=float))
        ),
        "stability_range": saferange(
            compound_df.get("compound_stability", pd.Series([], dtype=float))
        ),
        "mean_info": safemean(
            compound_df.get("mean_info", pd.Series([], dtype=float))
        ),
        "mean_misinfo": safemean(
            compound_df.get("mean_misinfo", pd.Series([], dtype=float))
        ),
        "mean_disinfo": safemean(
            compound_df.get("mean_disinfo", pd.Series([], dtype=float))
        ),
    }

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _stage(ctx, f"[compounds] compound_metrics.json written → {path}")
    return path


# -------------------------------------------------------------------------
# Public: pipeline step metadata for frontend plan
# -------------------------------------------------------------------------

PIPELINE_STEPS: List[Dict[str, Any]] = [
    {"id": 1, "key": "lsa", "title": "LSA spectral field", "description": "Build span-level latent semantic field and span map."},
    {"id": 2, "key": "elements", "title": "Element table", "description": "Construct hilbert_elements.csv with embeddings and metrics."},
    {"id": 3, "key": "condense", "title": "Element condensation", "description": "Merge elements into root-elements based on embeddings."},
    {"id": 4, "key": "molecules", "title": "Molecules & compounds", "description": "Construct informational molecules and aggregate compounds."},
    {"id": 5, "key": "fusion", "title": "Span→element fusion", "description": "Soft-assign spans to elements and aggregate compound contexts."},
    {"id": 6, "key": "stability", "title": "Signal stability", "description": "Compute stability metrics over elements/spans."},
    {"id": 7, "key": "persistence", "title": "Persistence visuals", "description": "Generate persistence and stability figures."},
    {"id": 8, "key": "labels", "title": "Element labels", "description": "Build human-readable labels and descriptions for elements."},
    {"id": 9, "key": "post", "title": "Post-processing", "description": "Normalise hilbert_elements.csv for frontend consumption."},
    {"id": 10, "key": "export", "title": "Export summary", "description": "Write PDF summary and ZIP archive of core artifacts."},
    {"id": 11, "key": "sanity", "title": "Sanity checks", "description": "Optional sanity checks and diagnostic report."},
]


def get_pipeline_plan() -> List[Dict[str, Any]]:
    """Return the ordered list of pipeline steps for the frontend."""
    return PIPELINE_STEPS


# -------------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------------

def run_pipeline(
    corpus_dir: str,
    out_dir: str,
    ctx: Optional[PipelineContext] = None,
) -> Dict[str, Any]:
    """
    Run the full Hilbert pipeline for a given corpus directory.

    Parameters
    ----------
    corpus_dir : str
        Directory of input text files.
    out_dir : str
        Output directory for all Hilbert artifacts.
    ctx : PipelineContext, optional
        If None, a new context is created.

    Returns
    -------
    dict
        Summary of key outputs and statuses.
    """
    corpus_dir = str(corpus_dir)
    out_dir = str(out_dir)

    if ctx is None:
        ctx = PipelineContext(corpus_dir=corpus_dir, out_dir=out_dir)

    out_path = Path(out_dir)
    _ensure_dir(out_path)

    _stage(
        ctx,
        f"[init] Hilbert pipeline starting. corpus_dir={corpus_dir}, out_dir={out_dir}",
    )
    _stage(ctx, f"[init] hilbert_native available: {HILBERT_NATIVE_AVAILABLE}")

    # ---------------------------------------------------------------------
    # Step 1. LSA spectral field
    # ---------------------------------------------------------------------
    s1 = ctx.begin_stage("[1] LSA spectral embeddings and field")
    try:
        lsa_result = build_lsa_field(corpus_dir)
        embeddings = np.asarray(
            lsa_result.get("field", {}).get("embeddings", []), dtype=float
        )
        n_spans = embeddings.shape[0]
        H_bar = float(lsa_result.get("H_bar", 0.0))
        C_global = float(lsa_result.get("C_global", 0.0))

        _stage(
            ctx,
            f"[lsa] Field built with {n_spans} spans, "
            f"H_bar={H_bar:.4f}, C_global={C_global:.4f}",
        )

        if n_spans == 0:
            msg = "[error] No span embeddings produced; aborting pipeline."
            ctx.emit("error", {"message": msg})
            ctx.aborted = True
            ctx.end_stage(s1, status="error", error=msg)
            return {"status": "error", "message": msg}

        # Persist lsa_field.json
        lsa_field_path = out_path / "lsa_field.json"
        lsa_out = dict(lsa_result)
        if isinstance(lsa_out.get("field", {}).get("embeddings"), np.ndarray):
            lsa_out["field"] = dict(lsa_out["field"])
            lsa_out["field"]["embeddings"] = embeddings.tolist()
        lsa_field_path.write_text(json.dumps(lsa_out, indent=2), encoding="utf-8")
        _stage(ctx, f"[lsa] lsa_field.json written → {lsa_field_path}")

        ctx.end_stage(s1, status="ok")
    except Exception as exc:
        msg = f"[error] LSA layer failed: {exc}"
        ctx.emit("error", {"message": msg})
        ctx.aborted = True
        ctx.end_stage(s1, status="error", error=str(exc))
        raise

    # ---------------------------------------------------------------------
    # Step 2. Build hilbert_elements.csv with embeddings
    # ---------------------------------------------------------------------
    s2 = ctx.begin_stage("[2] Build hilbert_elements.csv")
    try:
        elements_csv_path = _build_hilbert_elements_csv(ctx, lsa_result)
        ctx.end_stage(s2, status="ok")
    except Exception as exc:
        ctx.end_stage(s2, status="error", error=str(exc))
        ctx.aborted = True
        raise

    # ---------------------------------------------------------------------
    # Step 3. Element condensation
    # ---------------------------------------------------------------------
    s3 = ctx.begin_stage("[3] Element condensation")
    try:
        try:
            condense_result = run_condensation_layer(str(out_path), emit=ctx.emit)
        except TypeError:
            # backward-compat: old signature without emit
            condense_result = run_condensation_layer(str(out_path))

        if condense_result:
            _stage(
                ctx,
                f"[condense] Condensed {condense_result.get('n_total')} → "
                f"{condense_result.get('n_roots')} root elements.",
            )
        else:
            _stage(
                ctx,
                "[condense] Condensation step skipped or failed; continuing with raw elements.",
            )
        ctx.end_stage(s3, status="ok")
    except Exception as exc:
        _stage(ctx, f"[condense][warn] Condensation layer failed: {exc}")
        ctx.end_stage(s3, status="error", error=str(exc))

    # After condensation, before molecule stage
    sX = ctx.begin_stage("[3.5] Graph visualisation snapshots")

    try:
        from hilbert_pipeline.graph_snapshots import generate_graph_snapshots
        generate_graph_snapshots(str(out_path), emit=ctx.emit)
        ctx.end_stage(sX, status="ok")
    except Exception as exc:
        ctx.end_stage(sX, status="error", error=str(exc))


    # ---------------------------------------------------------------------
    # Step 4. Molecule construction & compounds
    # ---------------------------------------------------------------------
    s4 = ctx.begin_stage("[4] Molecule construction & compounds")
    edges_path = out_path / "edges.csv"
    molecule_df = pd.DataFrame()
    compound_df = pd.DataFrame()
    if not edges_path.exists():
        _stage(ctx, "[4] edges.csv not found; molecule layer will be skipped.")
        ctx.end_stage(s4, status="skipped", error="edges.csv not found")
    else:
        try:
            _stage(ctx, "[4] Constructing informational molecules from element graph...")
            molecule_df = compute_molecule_stability(
                str(edges_path), str(elements_csv_path)
            )
            molecule_df = compute_molecule_temperature(
                molecule_df, pd.read_csv(elements_csv_path)
            )
            compound_df = aggregate_compounds(
                molecule_df, str(elements_csv_path), str(edges_path)
            )
            export_molecule_summary(str(out_path), molecule_df, compound_df)
            if not molecule_df.empty and not compound_df.empty:
                _stage(
                    ctx,
                    f"[molecule] Built {compound_df['compound_id'].nunique()} compounds "
                    f"from {molecule_df['element'].nunique()} elements.",
                )
            else:
                _stage(
                    ctx,
                    "[molecule] Molecule or compound table is empty after construction.",
                )
            ctx.end_stage(s4, status="ok")
        except Exception as exc:
            _stage(ctx, f"[molecule][warn] Molecule/compound layer failed: {exc}")
            molecule_df = pd.DataFrame()
            compound_df = pd.DataFrame()
            ctx.end_stage(s4, status="error", error=str(exc))

    # Compound metrics JSON for export layer
    if compound_df is not None and not compound_df.empty:
        _write_compound_metrics_json(ctx, compound_df)


    # # ---------------------------------------------------------------------
    # # Step 4b. Graph snapshots
    # # ---------------------------------------------------------------------
    # s4b = ctx.begin_stage("[4b] Graph snapshots")
    # try:
    #     from hilbert_pipeline.graph_export import export_graph_snapshots
    #     export_graph_snapshots(str(out_path), emit=ctx.emit)
    #     ctx.end_stage(s4b, status="ok")
    # except Exception as exc:
    #     _stage(ctx, f"[graph][warn] Failed to generate graph snapshots: {exc}")
    #     ctx.end_stage(s4b, status="error", error=str(exc))


    # ---------------------------------------------------------------------
    # Step 5. Optional span→element fusion and compound context
    # ---------------------------------------------------------------------
    s5 = ctx.begin_stage("[5] Span→element fusion & compound context")
    try:
        _stage(ctx, "[5] Span→element fusion (optional)...")
        try:
            fuse_spans_to_elements(str(out_path))
        except Exception as exc:
            _stage(ctx, f"[fusion][warn] span-element fusion failed: {exc}")

        _stage(ctx, "[5b] Aggregating compound contexts (optional)...")
        try:
            aggregate_compound_context(str(out_path))
        except Exception as exc:
            _stage(ctx, f"[fusion][warn] compound context aggregation failed: {exc}")

        ctx.end_stage(s5, status="ok")
    except Exception as exc:
        ctx.end_stage(s5, status="error", error=str(exc))

    # ---------------------------------------------------------------------
    # Step 6. Signal stability metrics
    # ---------------------------------------------------------------------
    s6 = ctx.begin_stage("[6] Signal stability metrics")
    try:
        _stage(ctx, "[6] Computing signal stability metrics...")
        stability_csv = out_path / "signal_stability.csv"
        compute_signal_stability(str(elements_csv_path), str(stability_csv))
        ctx.end_stage(s6, status="ok")
    except Exception as exc:
        _stage(ctx, f"[stability][warn] stability computation failed: {exc}")
        ctx.end_stage(s6, status="error", error=str(exc))

    # ---------------------------------------------------------------------
    # Step 7. Persistence / stability visualisations
    # ---------------------------------------------------------------------
    s7 = ctx.begin_stage("[7] Persistence & stability visuals")
    try:
        _stage(ctx, "[7] Generating persistence & stability visuals...")
        plot_persistence_field(str(out_path))
        ctx.end_stage(s7, status="ok")
    except Exception as exc:
        _stage(ctx, f"[persist][warn] persistence visuals failed: {exc}")
        ctx.end_stage(s7, status="error", error=str(exc))

    # ---------------------------------------------------------------------
    # Step 8. Element description / label layer
    # ---------------------------------------------------------------------
    s8 = ctx.begin_stage("[8] Element labels & descriptions")
    try:
        _stage(ctx, "[8] Building element labels & descriptions...")
        spans = lsa_result.get("field", {}).get("span_map", []) or []
        build_element_descriptions(
            elements_csv=str(elements_csv_path),
            spans=spans,
            out_dir=str(out_path),
        )
        ctx.end_stage(s8, status="ok")
    except Exception as exc:
        _stage(ctx, f"[labels][warn] element label builder failed: {exc}")
        ctx.end_stage(s8, status="error", error=str(exc))

    # ---------------------------------------------------------------------
    # Step 9. Post-processing / normalisation for frontend
    # ---------------------------------------------------------------------
    s9 = ctx.begin_stage("[9] Post-processing hilbert_elements.csv")
    try:
        _stage(ctx, "[9] Normalising hilbert_elements.csv for frontend...")
        df = pd.read_csv(elements_csv_path)

        # Ensure doc column exists
        if "doc" not in df.columns:
            for col in ("source", "file", "filename"):
                if col in df.columns:
                    df["doc"] = df[col]
                    _stage(ctx, f"[post] Added 'doc' column from '{col}'.")
                    break
            else:
                df["doc"] = "corpus"
                _stage(ctx, "[post] Added synthetic 'doc' column = 'corpus'.")

        # Ensure token column is present
        if "token" not in df.columns and "element" in df.columns:
            df["token"] = df["element"]
            _stage(ctx, "[post] Added 'token' column from 'element'.")

        df.to_csv(elements_csv_path, index=False)
        _stage(
            ctx,
            "[post] Normalised hilbert_elements.csv -> "
            f"{elements_csv_path}",
        )
        ctx.end_stage(s9, status="ok")
    except Exception as exc:
        _stage(ctx, f"[post][warn] hilbert_elements normalisation failed: {exc}")
        ctx.end_stage(s9, status="error", error=str(exc))

    # ---------------------------------------------------------------------
    # Step 10. Export summary PDF and archive
    # ---------------------------------------------------------------------
    s10 = ctx.begin_stage("[10] Export summary & archive")
    try:
        _stage(ctx, "[10] Exporting final reports and archive...")
        try:
            export_summary_pdf(str(out_path))
        except Exception as exc:
            _stage(ctx, f"[export][warn] PDF export failed: {exc}")

        try:
            export_zip(str(out_path))
        except Exception as exc:
            _stage(ctx, f"[export][warn] ZIP export failed: {exc}")

        ctx.end_stage(s10, status="ok")
    except Exception as exc:
        ctx.end_stage(s10, status="error", error=str(exc))

    # ---------------------------------------------------------------------
    # Step 11. Sanity checks & diagnostics summary (optional)
    # ---------------------------------------------------------------------
    s11 = ctx.begin_stage("[11] Sanity checks & diagnostics")
    try:
        if check_run_sanity is None:
            _stage(ctx, "[11] No sanity check module available; skipping.")
            ctx.end_stage(s11, status="skipped", error="hilbert_sanity not found")
        else:
            _stage(ctx, "[11] Running sanity checks over Hilbert run artefacts...")
            try:
                # Preferred newer signature: (out_dir, emit=...)
                check_run_sanity(str(out_path), emit=ctx.emit)  # type: ignore
            except TypeError:
                # Backward compatibility: (out_dir)
                check_run_sanity(str(out_path))  # type: ignore
            ctx.end_stage(s11, status="ok")
    except Exception as exc:
        _stage(ctx, f"[sanity][warn] sanity checks failed: {exc}")
        ctx.end_stage(s11, status="error", error=str(exc))

    # ---------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------
    elapsed = time.time() - ctx.start_time
    _stage(ctx, f"[pipeline] Hilbert pipeline finished in {elapsed:.2f} seconds.")

    return {
        "status": "ok" if not ctx.aborted else "error",
        "corpus_dir": corpus_dir,
        "out_dir": out_dir,
        "elapsed_seconds": elapsed,
        "errors": ctx.errors,
        "stages": [
            {
                "name": s.name,
                "status": s.status,
                "duration": s.duration,
                "error": s.error,
                "meta": s.meta,
            }
            for s in ctx.stages
        ],
    }


# -------------------------------------------------------------------------
# CLI entrypoint
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Hilbert Information Chemistry pipeline."
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Input corpus directory of text files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/hilbert_run",
        help="Output directory for Hilbert artifacts.",
    )
    args = parser.parse_args()

    ctx = PipelineContext(corpus_dir=args.corpus, out_dir=args.out)

    try:
        run_pipeline(args.corpus, args.out, ctx=ctx)
    except Exception as exc:
        _stage(ctx, f"[fatal] Pipeline crashed: {exc}")
        traceback.print_exc()
