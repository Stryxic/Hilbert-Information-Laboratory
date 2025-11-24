"""
Hilbert Orchestrator 3.2

Declarative, dialectical orchestration of the Hilbert information pipeline.

- Stages are defined in a single declarative table (STAGE_TABLE).
- Each stage is a node in a dialectical graph, with explicit dependencies,
  support edges, and challenge edges.
- The orchestrator executes stages in order, respecting dependencies and
  surfacing a rich run summary that the frontend can render.

This file is intentionally self-contained: the only assumptions about other
modules are the public functions re-exported by `hilbert_pipeline.__init__`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional
import shutil
import random

import numpy as np
import pandas as pd

ORCHESTRATOR_VERSION = "3.2"

# -----------------------------------------------------------------------------
# Optional PDF support (for corpus normalisation)
# -----------------------------------------------------------------------------

try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:
    PdfReader = None


def _pdf_to_text(path: str) -> str:
    """
    Very simple PDF-to-text converter.

    Returns plain UTF-8 text. If conversion fails, returns empty string so the
    caller can decide to skip the file.
    """
    if PdfReader is None:
        return ""

    try:
        reader = PdfReader(path)
        chunks: List[str] = []
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt:
                chunks.append(txt)
        return "\n\n".join(chunks).strip()
    except Exception:
        return ""


# -----------------------------------------------------------------------------
# Pipeline imports (all via hilbert_pipeline public API)
# -----------------------------------------------------------------------------

try:
    from hilbert_pipeline import (
        DEFAULT_EMIT,
        # LSA
        build_lsa_field,
        # Element descriptions
        build_element_descriptions,
        # Fusion (span -> element)
        run_fusion_pipeline,
        # Molecules
        run_molecule_layer,
        # Element roots (consolidator)
        run_element_roots,
        # Stability
        compute_signal_stability,
        # Persistence
        plot_persistence_field,
        run_persistence_visuals,
        # Epistemic signatures
        compute_signatures,
        # Element LM
        run_element_lm_stage,
        # Graph edges
        build_element_edges,
        # LM perplexity (Ollama / LLM pipeline)
        compute_corpus_perplexity,
        # Compound stability
        compute_compound_stability,
    )
    from hilbert_pipeline.hilbert_report import run_hilbert_report
except Exception as exc:  # very defensive import
    raise RuntimeError(f"[orchestrator] Failed to import hilbert_pipeline: {exc}") from exc


EmitFn = Callable[[str, Dict[str, Any]], None]


# -----------------------------------------------------------------------------
# Settings and context objects
# -----------------------------------------------------------------------------

@dataclass
class PipelineSettings:
    """
    User-tunable knobs.

    Keep this conservative - anything heavy should be configured per stage.
    """
    use_native: bool = True
    max_docs: Optional[int] = None
    random_seed: int = 13
    # Future: thermodynamic controls, misinfo sensitivity, etc.


@dataclass
class StageState:
    """Runtime state for a single stage execution."""
    key: str
    label: str
    status: Literal["pending", "running", "ok", "skipped", "failed"] = "pending"
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        if self.start_ts is None or self.end_ts is None:
            return None
        return self.end_ts - self.start_ts


@dataclass
class StageSpec:
    """
    Declarative description of a pipeline stage.

    dialectic_role:
        - "evidence"  - generates or sharpens observations
        - "structure" - reshapes observations into higher-order structure
        - "thermo"    - computes thermodynamic / stability metrics
        - "epistemic" - evaluates truth, intent, misinfo signatures
        - "visual"    - builds views on the structure
        - "export"    - packages results
    """
    key: str
    order: float
    label: str
    func: Optional[Callable[["PipelineContext"], None]]
    required: bool = True
    dialectic_role: Literal[
        "evidence", "structure", "thermo", "epistemic", "visual", "export"
    ] = "structure"
    depends_on: List[str] = field(default_factory=list)
    supports: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    consumes: List[str] = field(default_factory=list)


@dataclass
class PipelineContext:
    """
    Shared context for a single run.

    The orchestrator deals with directories, paths and coarse messages.
    Internal math lives inside hilbert_pipeline modules.
    """
    corpus_dir: str
    results_dir: str
    settings: PipelineSettings = field(default_factory=PipelineSettings)
    emit: EmitFn = DEFAULT_EMIT

    # Computed during the run
    run_id: str = field(default_factory=lambda: str(int(time.time())))
    stages: Dict[str, StageState] = field(default_factory=dict)
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    def log(self, level: str, msg: str, **fields: Any) -> None:
        payload = {"level": level, "msg": msg, "ts": time.time()}
        payload.update(fields)
        try:
            self.emit("log", payload)
        except Exception:
            # Fallback to stdout if user supplied an emit that fails
            print(f"[{level}] {msg} {fields}")

    # --- stage lifecycle -----------------------------------------------------

    def begin_stage(self, spec: StageSpec) -> StageState:
        state = self.stages.get(spec.key) or StageState(key=spec.key, label=spec.label)
        state.status = "running"
        state.start_ts = time.time()
        self.stages[spec.key] = state
        self.log("info", f"{spec.label} - starting", stage=spec.key)
        return state

    def end_stage_ok(self, spec: StageSpec, meta: Optional[Dict[str, Any]] = None) -> None:
        state = self.stages[spec.key]
        state.status = "ok"
        state.end_ts = time.time()
        if meta:
            state.meta.update(meta)
        self.log("info", f"{spec.label} - ok", stage=spec.key, duration=state.duration)

    def end_stage_skipped(self, spec: StageSpec, reason: str) -> None:
        state = self.stages.get(spec.key) or StageState(key=spec.key, label=spec.label)
        state.status = "skipped"
        state.error = reason
        state.start_ts = state.start_ts or time.time()
        state.end_ts = time.time()
        self.stages[spec.key] = state
        self.log("warn", f"{spec.label} - skipped: {reason}", stage=spec.key)

    def end_stage_failed(self, spec: StageSpec, error: str) -> None:
        state = self.stages[spec.key]
        state.status = "failed"
        state.error = error
        state.end_ts = time.time()
        self.log("error", f"{spec.label} - failed: {error}", stage=spec.key)

    # --- artifact tracking ---------------------------------------------------

    def add_artifact(self, name: str, kind: str, **meta: Any) -> None:
        path = os.path.join(self.results_dir, name)
        self.artifacts[name] = {"kind": kind, "path": path, **meta}

    # --- dialectical graph export -------------------------------------------

    def dialectic_graph(self, specs: List[StageSpec]) -> Dict[str, Any]:
        """Return a JSON-serialisable dialectic graph describing this run."""
        nodes = []
        edges = []

        for spec in specs:
            st = self.stages.get(spec.key)
            nodes.append(
                {
                    "id": spec.key,
                    "label": spec.label,
                    "role": spec.dialectic_role,
                    "status": getattr(st, "status", "pending"),
                    "duration": getattr(st, "duration", None),
                    "required": spec.required,
                    "produces": spec.produces,
                    "consumes": spec.consumes,
                }
            )

            for dep in spec.depends_on:
                edges.append({"source": dep, "target": spec.key, "type": "depends_on"})
            for s in spec.supports:
                edges.append({"source": spec.key, "target": s, "type": "supports"})
            for c in spec.challenges:
                edges.append({"source": spec.key, "target": c, "type": "challenges"})

        return {"nodes": nodes, "edges": edges}

    # --- run summary --------------------------------------------------------

    def run_summary(self, specs: List[StageSpec]) -> Dict[str, Any]:
        # Important: we do not dump ctx.extras here to keep things JSON-safe.
        return {
            "run_id": self.run_id,
            "corpus_dir": self.corpus_dir,
            "results_dir": self.results_dir,
            "settings": self.settings.__dict__,
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "stages": {
                key: {
                    "label": st.label,
                    "status": st.status,
                    "error": st.error,
                    "duration": st.duration,
                    "meta": st.meta,
                }
                for key, st in self.stages.items()
            },
            "artifacts": self.artifacts,
            "dialectic_graph": self.dialectic_graph(specs),
        }


# -----------------------------------------------------------------------------
# Helpers for corpus normalisation (PDF + text) before LSA
# -----------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _prepare_lsa_corpus(ctx: PipelineContext) -> str:
    """
    Build a normalised corpus for LSA under results_dir/_normalized_corpus.

    - Converts PDFs to .txt with a simple PyPDF2-based extractor.
    - Copies .txt/.md/.tex as UTF-8 text (ignoring binary junk).
    - Respects settings.max_docs if set (limit on number of source files).
    """
    from pathlib import Path

    src_root = Path(ctx.corpus_dir).resolve()
    dst_root = Path(ctx.results_dir).resolve() / "_normalized_corpus"

    # Start from a clean scratch directory once
    shutil.rmtree(dst_root, ignore_errors=True)
    dst_root.mkdir(parents=True, exist_ok=True)

    # Collect all candidate files
    all_paths: List[Path] = [p for p in src_root.rglob("*") if p.is_file()]
    all_paths.sort()

    # Apply max_docs if specified and > 0
    max_docs = ctx.settings.max_docs
    if max_docs is not None and max_docs > 0:
        selected_paths = all_paths[:max_docs]
    else:
        selected_paths = all_paths

    n_converted = 0
    n_skipped = 0

    for src_path in selected_paths:
        rel = src_path.relative_to(src_root)
        out_dir = dst_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        root, ext = os.path.splitext(rel.name)
        ext = ext.lower()

        if ext == ".pdf":
            text = _pdf_to_text(str(src_path))
            if not text.strip():
                ctx.log("warn", "[lsa] PDF produced no text; skipping", path=str(src_path))
                n_skipped += 1
                continue

            out_name = f"{root}.txt"
            out_path = out_dir / out_name
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)
                n_converted += 1
            except Exception as exc:
                ctx.log(
                    "warn",
                    "[lsa] Failed to write PDF text",
                    path=str(src_path),
                    error=str(exc),
                )
                n_skipped += 1

        elif ext in (
            ".txt",
            ".md",
            ".tex",
            ".py",
            ".c",
            ".h",
            ".cpp",
            ".hpp",
            ".cc",
            ".java",
        ):
            # For text and source files, just copy them; the LSA layer
            # will do format-aware cleaning based on the extension.
            out_path = out_dir / rel.name
            try:
                with open(src_path, "r", encoding="utf-8", errors="ignore") as f_src, \
                        open(out_path, "w", encoding="utf-8") as f_dst:
                    f_dst.write(f_src.read())
                n_converted += 1
            except Exception as exc:
                ctx.log(
                    "warn",
                    "[lsa] Failed to copy text/source file",
                    path=str(src_path),
                    error=str(exc),
                )
                n_skipped += 1
        else:
            # Ignore other file types for LSA (likely binary)
            n_skipped += 1
            continue

    ctx.log(
        "info",
        "[lsa] Prepared normalised LSA corpus",
        src_root=str(src_root),
        dst_root=str(dst_root),
        total_files=len(all_paths),
        selected=len(selected_paths),
        converted=n_converted,
        skipped=n_skipped,
    )
    return str(dst_root)


# -----------------------------------------------------------------------------
# Stage implementations (wrappers around hilbert_pipeline functions)
# -----------------------------------------------------------------------------

def _stage_lsa(ctx: PipelineContext) -> None:
    """
    Run LSA, then write hilbert_elements.csv in a stability-compatible format.

    We:
      - normalise the corpus (PDF -> text, copy text) into _normalized_corpus
      - call build_lsa_field() to compute the spectral field on that corpus
      - store the raw result in ctx.extras["lsa_result"]
      - export:
          * hilbert_elements.csv   (element-level table)
          * lsa_field.json         (flattened LSA field: embeddings + span_map)
    """
    lsa_root = _prepare_lsa_corpus(ctx)

    # Important: use the same call shape as your working version
    res = build_lsa_field(lsa_root, emit=ctx.emit) or {}
    ctx.extras["lsa_result"] = res

    # Attempt to cache embedding config for graph metadata
    lsa_cfg = res.get("config") or {}
    ctx.extras["embedding_parameters"] = (
        lsa_cfg.get("embedding_parameters")
        or {
            "model": lsa_cfg.get("model", "svd"),
            "n_components": lsa_cfg.get("n_components"),
            "normalisation": lsa_cfg.get("normalisation"),
        }
    )
    ctx.extras["lsa_model_version"] = (
        lsa_cfg.get("model_version")
        or lsa_cfg.get("version")
        or "unknown"
    )

    elements = res.get("elements", []) or []
    metrics = res.get("element_metrics", []) or []

    field = res.get("field", {}) or {}
    span_map = field.get("span_map", [])
    embeddings = field.get("embeddings")

    # Convert ndarray embeddings to list-of-lists for JSON, but keep ndarray for internal use
    if isinstance(embeddings, np.ndarray):
        emb_array = embeddings
        emb_for_json = embeddings.tolist()
    else:
        emb_array = np.asarray(embeddings or [], dtype=float)
        emb_for_json = embeddings

    # ---------------------------------------------
    # Build element-level rows
    # ---------------------------------------------
    element_rows = []
    for rec in elements:
        el = rec.get("element")
        idx = rec.get("index")
        element_rows.append(
            {
                "element": el,
                "index": idx,
                # span linkage handled by fusion later; these columns remain
                # for backward compatibility
                "span_id": -1,
                "doc": None,
                "text": None,
            }
        )

    elements_df = pd.DataFrame(element_rows)

    # ---------------------------------------------
    # Build metrics dataframe
    # ---------------------------------------------
    metrics_df = pd.DataFrame(metrics)

    # Standardise names if mean_* present
    if "mean_entropy" in metrics_df.columns and "entropy" not in metrics_df.columns:
        metrics_df["entropy"] = metrics_df["mean_entropy"]
    if "mean_coherence" in metrics_df.columns and "coherence" not in metrics_df.columns:
        metrics_df["coherence"] = metrics_df["mean_coherence"]

    # ---------------------------------------------
    # Merge metrics into element rows
    # ---------------------------------------------
    if not metrics_df.empty:
        merged = elements_df.merge(metrics_df, on=["element", "index"], how="left")
    else:
        merged = elements_df

    # ---------------------------------------------
    # Write hilbert_elements.csv
    # ---------------------------------------------
    elements_path = os.path.join(ctx.results_dir, "hilbert_elements.csv")
    merged.to_csv(elements_path, index=False)
    ctx.add_artifact("hilbert_elements.csv", "elements")

    # ------------------------------------------------------------------
    # LSA layer no longer computes edges internally
    # ------------------------------------------------------------------
    ctx.log("info", "[lsa] Edge generation disabled - handled by stage_edges")

    # ---------------------------------------------
    # Write LSA field (flat for compatibility)
    # ---------------------------------------------
    lsa_flat = {
        "embeddings": emb_array.tolist() if isinstance(emb_array, np.ndarray) else emb_for_json,
        "span_map": span_map,
        "vocab": field.get("vocab"),
    }
    with open(os.path.join(ctx.results_dir, "lsa_field.json"), "w", encoding="utf-8") as f:
        json.dump(lsa_flat, f, indent=2)

    ctx.add_artifact("lsa_field.json", "lsa-field")
    ctx.stages["lsa_field"].meta["n_spans"] = len(span_map)
    ctx.stages["lsa_field"].meta["n_elements"] = len(elements)


def _stage_edges(ctx: PipelineContext) -> None:
    """
    Build element-element edges from span-element fusion.

    Produces:
      - edges.csv
    """
    out_path = build_element_edges(ctx.results_dir, emit=ctx.emit)
    if out_path:
        ctx.add_artifact(os.path.basename(out_path), "edges")


def _stage_molecules(ctx: PipelineContext) -> None:
    """
    Molecule construction and compounds.

    Uses run_molecule_layer(results_dir, emit) from hilbert_pipeline.

    Produces:
      - molecules.csv
      - informational_compounds.json (handled inside run_molecule_layer)
    """
    mol_df, comp_df = run_molecule_layer(
        ctx.results_dir,
        emit=ctx.emit,
    )

    # Molecule table
    if isinstance(mol_df, pd.DataFrame) and not mol_df.empty:
        path = os.path.join(ctx.results_dir, "molecules.csv")
        try:
            mol_df.to_csv(path, index=False)
            ctx.add_artifact("molecules.csv", "molecule-table")
        except Exception as exc:
            ctx.log("warn", f"Failed to write molecules.csv: {exc}")

    # Compound JSON - produced by the molecule layer
    comp_path = os.path.join(ctx.results_dir, "informational_compounds.json")
    if os.path.exists(comp_path):
        ctx.add_artifact("informational_compounds.json", "informational-compounds")


def _stage_element_roots(ctx: PipelineContext) -> None:
    """
    Element root consolidation.

    Groups surface-form elements into root clusters and exports:
      - element_roots.csv
      - element_cluster_metrics.json
    """
    try:
        run_element_roots(ctx.results_dir, emit=ctx.emit)
    except Exception as exc:
        ctx.log("warn", f"[element_roots] Failed: {exc}")
        return

    for fname, kind in [
        ("element_roots.csv", "element-roots"),
        ("element_cluster_metrics.json", "element-cluster-metrics"),
    ]:
        path = os.path.join(ctx.results_dir, fname)
        if os.path.exists(path):
            ctx.add_artifact(fname, kind)


def _stage_fusion(ctx: PipelineContext) -> None:
    """
    Span -> element fusion and compound context enrichment.

    Delegates to hilbert_pipeline.run_fusion_pipeline(results_dir, emit).
    """
    ctx.log("info", "[fusion] Running span->element fusion")
    run_fusion_pipeline(ctx.results_dir, emit=ctx.emit)


def _stage_stability(ctx: PipelineContext) -> None:
    """
    Compute signal stability metrics over elements.

    Uses compute_signal_stability(elements_csv, out_csv, mode, emit).
    If entropy/coherence fields are missing, the stability module will log
    a warning and abort gracefully.
    """
    elements_csv = os.path.join(ctx.results_dir, "hilbert_elements.csv")
    out_csv = os.path.join(ctx.results_dir, "signal_stability.csv")
    compute_signal_stability(
        elements_csv,
        out_csv,
        mode="classic",
        emit=ctx.emit,
    )
    # The stability module will decide whether it actually wrote the file.
    if os.path.exists(out_csv):
        ctx.add_artifact("signal_stability.csv", "signal-stability")


def _stage_compound_stability(ctx: PipelineContext) -> None:
    """
    Compute compound-level stability from element-level stability + molecule table.

    Produces:
      - compound_stability.csv
    """
    try:
        elements_csv = os.path.join(ctx.results_dir, "hilbert_elements.csv")
        stability_csv = os.path.join(ctx.results_dir, "signal_stability.csv")
        compounds_json = os.path.join(ctx.results_dir, "informational_compounds.json")
        out_csv = os.path.join(ctx.results_dir, "compound_stability.csv")

        if not os.path.exists(compounds_json):
            ctx.log("warn", "[compound-stability] Missing compounds - skipping")
            return

        compute_compound_stability(
            compounds_json=compounds_json,
            elements_csv=elements_csv,
            stability_csv=stability_csv,
            out_csv=out_csv,
            emit=ctx.emit,
        )

        if os.path.exists(out_csv):
            ctx.add_artifact("compound_stability.csv", "compound-stability")

    except Exception as exc:
        ctx.log("warn", f"[compound-stability] Failed: {exc}")


def _stage_persistence(ctx: PipelineContext) -> None:
    """
    Persistence and stability visuals.

    Delegates to run_persistence_visuals(results_dir, emit) which handles:
      - persistence_field.png
      - stability_scatter.png
      - stability_by_doc.png
    """
    run_persistence_visuals(ctx.results_dir, emit=ctx.emit)

    for fname, kind in [
        ("persistence_field.png", "persistence-field"),
        ("stability_scatter.png", "stability-scatter"),
        ("stability_by_doc.png", "stability-by-doc"),
    ]:
        path = os.path.join(ctx.results_dir, fname)
        if os.path.exists(path):
            ctx.add_artifact(fname, kind)


def _stage_element_labels(ctx: PipelineContext) -> None:
    """
    Build element labels and human-readable descriptions.

    We pass the span_map from the LSA layer (if available) for nicer contexts.
    """
    spans = ctx.extras.get("lsa_result", {}).get("field", {}).get("span_map", [])
    build_element_descriptions(
        elements_csv=os.path.join(ctx.results_dir, "hilbert_elements.csv"),
        spans=spans,
        out_dir=ctx.results_dir,
    )
    ctx.add_artifact("element_descriptions.json", "element-descriptions")
    ctx.add_artifact("element_intensity.csv", "element-intensity")


def _stage_signatures(ctx: PipelineContext) -> None:
    """
    Compute epistemic and misinfo signatures.

    The underlying module is tolerant of missing labels and will no-op gracefully.
    """
    out_path = compute_signatures(ctx.results_dir, emit=ctx.emit)
    if out_path:
        ctx.add_artifact(os.path.basename(out_path), "epistemic-signatures")


def _stage_element_lm(ctx: PipelineContext) -> None:
    """
    Train the element-level language model on span-element sequences.

    The training code will no-op gracefully if span_element_fusion.csv is missing
    or underspecified.
    """
    run_element_lm_stage(ctx.results_dir, emit=ctx.emit)

    # Register typical LM artifacts if they exist
    for fname, kind in [
        ("element_lm.pt", "element-lm"),
        ("element_vocab.json", "element-vocab"),
    ]:
        path = os.path.join(ctx.results_dir, fname)
        if os.path.exists(path):
            ctx.add_artifact(fname, kind)


def _stage_lm_perplexity(ctx: PipelineContext) -> None:
    """
    LM perplexity over the corpus.

    Produces:
      - lm_metrics.json

    A quick LLM health check is performed at the pipeline level; if the
    LLM is unavailable, this stage is skipped before it runs.
    """
    try:
        from pathlib import Path

        out_path = Path(ctx.results_dir) / "lm_metrics.json"

        # Prefer preloaded text if available
        text = ctx.extras.get("corpus_text")

        # Otherwise load spans.jsonl (default)
        if not text:
            spans_path = os.path.join(ctx.results_dir, "spans.jsonl")
            pieces: List[str] = []
            if os.path.exists(spans_path):
                with open(spans_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            pieces.append(obj.get("text", ""))
                        except Exception:
                            pass
            text = "\n".join(pieces)[:12000]  # Safe cap

        compute_corpus_perplexity(
            corpus_text=text,
            out_path=out_path,
        )
        ctx.add_artifact("lm_metrics.json", "lm-metrics")

    except Exception as exc:
        ctx.log("warn", f"[lm] Perplexity computation failed: {exc}")


def _stage_final_report(ctx: PipelineContext) -> None:
    """
    Final scientific report generator.

    Produces:
      - hilbert_report.pdf (preferred)
      - hilbert_report.txt (fallback when ReportLab is unavailable)
    """
    run_hilbert_report(ctx.results_dir, emit=ctx.emit)

    out_pdf = os.path.join(ctx.results_dir, "hilbert_report.pdf")
    out_txt = os.path.join(ctx.results_dir, "hilbert_report.txt")

    if os.path.exists(out_pdf):
        ctx.add_artifact("hilbert_report.pdf", "final-report")
    elif os.path.exists(out_txt):
        ctx.add_artifact("hilbert_report.txt", "final-report")


def _stage_graph_visualizer(ctx: PipelineContext) -> None:
    """
    Unified graph engine and snapshot generator.

    Delegates to hilbert_graphs.visualizer.HilbertGraphVisualizer, which
    is responsible for:
      - constructing node and edge tables that match the graph contract
      - writing graph_metadata.json if additional cluster or pruning info
        becomes available
      - exporting 2D and 3D snapshots at multiple depths
    """
    from hilbert_graphs.visualizer import HilbertGraphVisualizer

    viz = HilbertGraphVisualizer(
        ctx.results_dir,
        emit=ctx.emit,
    )
    viz.run()

    # register only the core artifacts for the frontend
    for fname in [
        "graph_1pct.png",
        "graph_5pct.png",
        "graph_10pct.png",
        "graph_25pct.png",
        "graph_50pct.png",
        "graph_full.png",
        "graph_1pct_3d.png",
        "graph_5pct_3d.png",
        "graph_10pct_3d.png",
        "graph_25pct_3d.png",
        "graph_50pct_3d.png",
        "graph_full_3d.png",
        "graph_snapshots_index.json",
    ]:
        path = os.path.join(ctx.results_dir, fname)
        if os.path.exists(path):
            ctx.add_artifact(fname, "graph")


# -----------------------------------------------------------------------------
# Declarative stage table - dialectical graph definition
# -----------------------------------------------------------------------------

STAGE_TABLE: List[StageSpec] = [
    StageSpec(
        key="lsa_field",
        order=1.0,
        label="[1] LSA spectral field",
        func=_stage_lsa,
        required=True,
        dialectic_role="evidence",
        supports=["fusion", "edges"],
    ),
    StageSpec(
        key="fusion",
        order=2.0,
        label="[2] Span - element fusion",
        func=_stage_fusion,
        required=False,
        dialectic_role="structure",
        depends_on=["lsa_field"],
        consumes=["hilbert_elements.csv"],
        produces=["span_element_fusion.csv", "compound_contexts.json"],
        supports=["edges"],
    ),
    StageSpec(
        key="edges",
        order=2.5,
        label="[2.5] Element co-occurrence edges",
        func=_stage_edges,
        required=False,
        dialectic_role="structure",
        depends_on=["fusion"],
        consumes=["span_element_fusion.csv"],
        produces=["edges.csv"],
        supports=["molecules", "graph_visualizer"],
    ),
    StageSpec(
        key="molecules",
        order=3.0,
        label="[3] Molecule layer",
        func=_stage_molecules,
        required=False,
        dialectic_role="structure",
        depends_on=["edges"],
        consumes=["hilbert_elements.csv", "edges.csv"],
        produces=["molecules.csv", "informational_compounds.json"],
        supports=["graph_visualizer", "element_roots"],
    ),
    StageSpec(
        key="element_roots",
        order=3.5,
        label="[3.5] Element root consolidation",
        func=_stage_element_roots,
        required=False,
        dialectic_role="structure",
        depends_on=["molecules"],
        consumes=["hilbert_elements.csv", "molecules.csv"],
        produces=["element_roots.csv", "element_cluster_metrics.json"],
        supports=["stability_metrics", "graph_visualizer"],
    ),
    StageSpec(
        key="stability_metrics",
        order=4.0,
        label="[4] Signal stability metrics",
        func=_stage_stability,
        required=False,
        dialectic_role="thermo",
        depends_on=["lsa_field"],
        consumes=["hilbert_elements.csv"],
        produces=["signal_stability.csv"],
        supports=["persistence_visuals", "compound_stability", "epistemic_signatures"],
    ),
    StageSpec(
        key="compound_stability",
        order=4.5,
        label="[4.5] Compound stability metrics",
        func=_stage_compound_stability,
        required=False,
        dialectic_role="thermo",
        depends_on=["stability_metrics", "molecules"],
        consumes=["signal_stability.csv", "molecules.csv"],
        produces=["compound_stability.csv"],
        supports=["graph_visualizer", "export_all"],
    ),
    StageSpec(
        key="persistence_visuals",
        order=5.0,
        label="[5] Persistence and stability visuals",
        func=_stage_persistence,
        required=False,
        dialectic_role="visual",
        depends_on=["stability_metrics"],
        consumes=["signal_stability.csv"],
        produces=[
            "persistence_field.png",
            "stability_scatter.png",
            "stability_by_doc.png",
        ],
    ),
    StageSpec(
        key="element_labels",
        order=6.0,
        label="[6] Element labels and descriptions",
        func=_stage_element_labels,
        required=True,
        dialectic_role="structure",
        depends_on=["lsa_field"],
        consumes=["hilbert_elements.csv"],
        produces=["element_descriptions.json", "element_intensity.csv"],
        supports=["epistemic_signatures"],
    ),
    StageSpec(
        key="epistemic_signatures",
        order=6.5,
        label="[6.5] Epistemic signatures (misinfo layer)",
        func=_stage_signatures,
        required=False,
        dialectic_role="epistemic",
        depends_on=["element_labels", "stability_metrics"],
        consumes=["hilbert_elements.csv", "signal_stability.csv"],
        produces=["epistemic_signatures.json"],
        challenges=["raw_corpus"],
    ),
    StageSpec(
        key="element_lm",
        order=6.8,
        label="[6.8] Element Language Model",
        func=_stage_element_lm,
        required=False,
        dialectic_role="epistemic",
        depends_on=["fusion"],
        consumes=["hilbert_elements.csv", "span_element_fusion.csv"],
        produces=["element_lm.pt", "element_vocab.json"],
    ),
    StageSpec(
        key="lm_perplexity",
        order=6.85,
        label="[6.85] LM Perplexity",
        func=_stage_lm_perplexity,
        required=False,
        dialectic_role="epistemic",
        depends_on=["lsa_field"],
        produces=["lm_metrics.json"],
    ),
    # ------------------------------------------------------------------
    # Unified graph engine
    # ------------------------------------------------------------------
    StageSpec(
        key="graph_visualizer",
        order=7.0,
        label="[7] Unified graph visualizer",
        func=_stage_graph_visualizer,
        required=False,
        dialectic_role="visual",
        depends_on=["molecules"],
        consumes=[
            "hilbert_elements.csv",
            "edges.csv",
            "informational_compounds.json",
        ],
        produces=[
            "graph_1pct.png",
            "graph_5pct.png",
            "graph_10pct.png",
            "graph_25pct.png",
            "graph_50pct.png",
            "graph_full.png",
            "graph_1pct_3d.png",
            "graph_5pct_3d.png",
            "graph_10pct_3d.png",
            "graph_25pct_3d.png",
            "graph_50pct_3d.png",
            "graph_full_3d.png",
            "graph_snapshots_index.json",
        ],
    ),
    StageSpec(
        key="export_all",
        order=8.0,
        label="[8] Final Hilbert Report",
        func=_stage_final_report,
        required=False,
        dialectic_role="export",
        depends_on=["element_labels", "compound_stability", "lm_perplexity"],
        supports=["epistemic_signatures"],
        produces=["hilbert_report.pdf"],
    ),
]


# -----------------------------------------------------------------------------
# Graph metadata writer for the visualisation contract
# -----------------------------------------------------------------------------

def _write_graph_metadata(ctx: PipelineContext) -> None:
    """
    Emit a graph_metadata.json file matching the global metadata part of the
    graph contract.

    This file is intentionally lightweight and focuses on stable, upstream
    information. Downstream graph modules may extend or overwrite it with
    additional fields such as cluster_hierarchy_info or pruning statistics.
    """
    meta: Dict[str, Any] = {
        "version": "1.0",
        "run_seed": ctx.settings.random_seed,
        "orchestrator_version": ORCHESTRATOR_VERSION,
        "lsa_model_version": ctx.extras.get("lsa_model_version", "unknown"),
        "embedding_parameters": ctx.extras.get("embedding_parameters", None),
        "cluster_hierarchy_info": ctx.extras.get("cluster_hierarchy_info", None),
        "pruning": ctx.extras.get("graph_pruning", None),
    }

    path = os.path.join(ctx.results_dir, "graph_metadata.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        ctx.add_artifact("graph_metadata.json", "graph-metadata")
    except Exception as exc:
        ctx.log("warn", f"Failed to write graph_metadata.json: {exc}")


# -----------------------------------------------------------------------------
# LLM health probe (for LM perplexity stage)
# -----------------------------------------------------------------------------

def _probe_llm(ctx: PipelineContext) -> bool:
    """
    Quick sanity check that the external LLM (Ollama / OpenAI-compatible)
    is reachable and returns a well-formed response.

    This is intentionally lightweight and runs once per pipeline.
    """
    # If we already probed in this run, reuse the result
    if "llm_available" in ctx.extras:
        return bool(ctx.extras["llm_available"])

    try:
        # Import the low-level scorer directly
        from hilbert_pipeline.ollama_lm import score_text_perplexity  # type: ignore
    except Exception as exc:
        ctx.log(
            "warn",
            "[lm] ollama_lm module not available; skipping LM-based metrics",
            error=str(exc),
        )
        ctx.extras["llm_available"] = False
        return False

    try:
        res = score_text_perplexity("Hilbert LM health check.", model=None)
    except Exception as exc:
        ctx.log(
            "warn",
            "[lm] LLM health check failed; skipping LM-based metrics",
            error=str(exc),
        )
        ctx.extras["llm_available"] = False
        return False

    if not isinstance(res, dict) or "perplexity" not in res:
        ctx.log(
            "warn",
            "[lm] LLM health check returned unexpected payload; skipping LM-based metrics",
        )
        ctx.extras["llm_available"] = False
        return False

    ctx.log("info", "[lm] LLM health check ok", model=res.get("model"))
    ctx.extras["llm_available"] = True
    return True


# -----------------------------------------------------------------------------
# Execution engine
# -----------------------------------------------------------------------------

def run_pipeline(
    corpus_dir: str,
    results_dir: str,
    settings: Optional[PipelineSettings] = None,
    emit: Optional[EmitFn] = None,
) -> Dict[str, Any]:
    """
    Run the full Hilbert pipeline on `corpus_dir`, writing outputs to
    `results_dir`.

    Returns a JSON-serialisable run summary for UI and API use.

    Notes:
    - settings.max_docs is enforced at the corpus normalisation step
      (PDF/text selection) via _prepare_lsa_corpus.
    """
    settings = settings or PipelineSettings()
    emit = emit or DEFAULT_EMIT

    # Ensure deterministic random behaviour for upstream modules
    try:
        random.seed(settings.random_seed)
        np.random.seed(settings.random_seed)
    except Exception:
        pass

    _ensure_dir(results_dir)
    ctx = PipelineContext(
        corpus_dir=os.path.abspath(corpus_dir),
        results_dir=os.path.abspath(results_dir),
        settings=settings,
        emit=emit,
    )

    ctx.log("info", "Starting Hilbert pipeline run", corpus=ctx.corpus_dir)

    # Probe LLM availability once for this run; used to decide whether to
    # execute the LM perplexity stage.
    try:
        ctx.extras["llm_available"] = _probe_llm(ctx)
    except Exception as exc:
        ctx.log("warn", f"[lm] LLM probe failed: {exc}")
        ctx.extras["llm_available"] = False

    for spec in sorted(STAGE_TABLE, key=lambda s: s.order):
        # Check hard dependencies
        unmet = [
            d for d in spec.depends_on
            if ctx.stages.get(d, StageState(d, d)).status != "ok"
        ]

        if unmet:
            reason = f"dependencies not satisfied: {', '.join(unmet)}"
            if spec.required:
                ctx.begin_stage(spec)
                ctx.end_stage_failed(spec, error=reason)
                break
            else:
                ctx.end_stage_skipped(spec, reason=reason)
                continue

        # Conditional skip for LM-based perplexity if LLM is down
        if spec.key == "lm_perplexity" and not ctx.extras.get("llm_available", False):
            ctx.end_stage_skipped(
                spec,
                reason="LLM unavailable - skipping LM perplexity stage",
            )
            continue

        # Conceptual node only
        if spec.func is None:
            ctx.end_stage_skipped(spec, reason="no-op conceptual stage")
            continue

        ctx.begin_stage(spec)
        try:
            spec.func(ctx)
            ctx.end_stage_ok(spec)
        except Exception as exc:
            ctx.end_stage_failed(spec, error=str(exc))
            if spec.required:
                ctx.log(
                    "error",
                    "Aborting pipeline due to failure in required stage.",
                    stage=spec.key,
                )
                break

    # Write graph-level metadata snapshot based on whatever upstream info
    # we managed to collect during this run.
    _write_graph_metadata(ctx)

    summary = ctx.run_summary(STAGE_TABLE)

    # Persist run summary
    try:
        run_json_path = os.path.join(ctx.results_dir, "hilbert_run.json")
        with open(run_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        ctx.add_artifact("hilbert_run.json", "run-summary")
    except Exception as exc:
        ctx.log("warn", f"Failed to write hilbert_run.json: {exc}")

    ctx.log("info", "Hilbert pipeline run complete", run_id=ctx.run_id)
    return summary


# Backwards-compatible alias
run_hilbert_pipeline = run_pipeline

__all__ = [
    "PipelineSettings",
    "StageState",
    "StageSpec",
    "PipelineContext",
    "STAGE_TABLE",
    "run_pipeline",
    "run_hilbert_pipeline",
]
