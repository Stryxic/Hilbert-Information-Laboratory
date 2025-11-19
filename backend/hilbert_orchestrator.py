# =====================================================================
# Hilbert Orchestrator 2.2 - Fully Patched Version
# =====================================================================

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Pipeline imports (all via hilbert_pipeline public API)
# ---------------------------------------------------------------------

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

        # Stability
        compute_signal_stability,

        # Persistence
        plot_persistence_field,
        run_persistence_visuals,

        # Graphs
        generate_graph_snapshots,
        export_graph_snapshots,

        # Epistemic signatures
        compute_signatures,

        # Full export (PDF + ZIP)
        run_full_export,

        # Element LM
        run_element_lm_stage,
    )
except Exception as exc:
    raise RuntimeError(f"[orchestrator] Failed to import hilbert_pipeline: {exc}") from exc


EmitFn = Callable[[str, Dict[str, Any]], None]


# =====================================================================
# Settings and state classes
# =====================================================================

@dataclass
class PipelineSettings:
    use_native: bool = True
    max_docs: Optional[int] = None
    random_seed: int = 13


@dataclass
class StageState:
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
    key: str
    order: float
    label: str
    func: Optional[Callable[["PipelineContext"], None]]
    required: bool = True
    dialectic_role: Literal["evidence", "structure", "thermo", "epistemic", "visual", "export"] = "structure"
    depends_on: List[str] = field(default_factory=list)
    supports: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    consumes: List[str] = field(default_factory=list)


@dataclass
class PipelineContext:
    corpus_dir: str
    results_dir: str
    settings: PipelineSettings = field(default_factory=PipelineSettings)
    emit: EmitFn = DEFAULT_EMIT

    run_id: str = field(default_factory=lambda: str(int(time.time())))
    stages: Dict[str, StageState] = field(default_factory=dict)
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------
    def log(self, level: str, msg: str, **fields: Any) -> None:
        payload = {"level": level, "msg": msg, "ts": time.time()}
        payload.update(fields)
        try:
            self.emit("log", payload)
        except Exception:
            print(f"[{level}] {msg} {fields}")

    # -----------------------------------------------------
    # Stage lifecycle
    # -----------------------------------------------------
    def begin_stage(self, spec: StageSpec) -> StageState:
        state = self.stages.get(spec.key) or StageState(spec.key, spec.label)
        state.status = "running"
        state.start_ts = time.time()
        self.stages[spec.key] = state
        self.log("info", f"{spec.label} - starting", stage=spec.key)
        return state

    def end_stage_ok(self, spec: StageSpec, meta: Optional[Dict[str, Any]] = None) -> None:
        st = self.stages[spec.key]
        st.status = "ok"
        st.end_ts = time.time()
        if meta:
            st.meta.update(meta)
        self.log("info", f"{spec.label} - ok", stage=spec.key, duration=st.duration)

    def end_stage_skipped(self, spec: StageSpec, reason: str) -> None:
        st = self.stages.get(spec.key) or StageState(spec.key, spec.label)
        st.status = "skipped"
        st.error = reason
        st.start_ts = st.start_ts or time.time()
        st.end_ts = time.time()
        self.stages[spec.key] = st
        self.log("warn", f"{spec.label} - skipped: {reason}", stage=spec.key)

    def end_stage_failed(self, spec: StageSpec, error: str) -> None:
        st = self.stages[spec.key]
        st.status = "failed"
        st.error = error
        st.end_ts = time.time()
        self.log("error", f"{spec.label} - failed: {error}", stage=spec.key)

    # -----------------------------------------------------
    # Artifact tracking
    # -----------------------------------------------------
    def add_artifact(self, name: str, kind: str, **meta: Any) -> None:
        self.artifacts[name] = {
            "kind": kind,
            "path": os.path.join(self.results_dir, name),
            **meta,
        }

    # -----------------------------------------------------
    # Dialectic graph
    # -----------------------------------------------------
    def dialectic_graph(self, specs: List[StageSpec]) -> Dict[str, Any]:
        nodes = []
        edges = []
        for spec in specs:
            st = self.stages.get(spec.key)
            nodes.append({
                "id": spec.key,
                "label": spec.label,
                "role": spec.dialectic_role,
                "status": getattr(st, "status", "pending"),
                "duration": getattr(st, "duration", None),
                "required": spec.required,
                "produces": spec.produces,
                "consumes": spec.consumes,
            })

            for d in spec.depends_on:
                edges.append({"source": d, "target": spec.key, "type": "depends_on"})
            for s in spec.supports:
                edges.append({"source": spec.key, "target": s, "type": "supports"})
            for c in spec.challenges:
                edges.append({"source": spec.key, "target": c, "type": "challenges"})

        return {"nodes": nodes, "edges": edges}

    # -----------------------------------------------------
    # Run summary
    # -----------------------------------------------------
    def run_summary(self, specs: List[StageSpec]) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "corpus_dir": self.corpus_dir,
            "results_dir": self.results_dir,
            "settings": self.settings.__dict__,
            "stages": {
                k: {
                    "label": st.label,
                    "status": st.status,
                    "error": st.error,
                    "duration": st.duration,
                    "meta": st.meta,
                }
                for k, st in self.stages.items()
            },
            "artifacts": self.artifacts,
            "dialectic_graph": self.dialectic_graph(specs),
        }


# =====================================================================
# Stage implementations
# =====================================================================

# ---------------------------------------------------------------------
# 1. LSA
# ---------------------------------------------------------------------
def _stage_lsa(ctx: PipelineContext) -> None:
    res = build_lsa_field(ctx.corpus_dir, emit=ctx.emit) or {}
    ctx.extras["lsa_result"] = res

    elements = res.get("elements", []) or []
    metrics = res.get("element_metrics", []) or []
    field = res.get("field", {}) or {}

    span_map = field.get("span_map", [])
    embeddings = field.get("embeddings")
    vocab = field.get("vocab")

    # Ensure embeddings are JSON-safe
    if isinstance(embeddings, np.ndarray):
        embeddings = embeddings.tolist()

    # Build hilbert_elements.csv
    rows = []
    for rec in elements:
        el = rec.get("element")
        sid = rec.get("span_id")
        doc, text = None, None

        if sid is not None and sid < len(span_map):
            sp = span_map[sid]
            doc = sp.get("doc")
            text = sp.get("text")

        row = {
            "element": el,
            "span_id": sid,
            "doc": doc,
            "text": text,
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    mdf = pd.DataFrame(metrics)

    # Standard metric names
    if "mean_entropy" in mdf.columns:
        mdf["entropy"] = mdf["mean_entropy"]
    if "mean_coherence" in mdf.columns:
        mdf["coherence"] = mdf["mean_coherence"]

    merged = df.merge(mdf, on="element", how="left")

    out_csv = os.path.join(ctx.results_dir, "hilbert_elements.csv")
    merged.to_csv(out_csv, index=False)
    ctx.add_artifact("hilbert_elements.csv", "elements")

    # Write LSA field
    lsa_flat = {"embeddings": embeddings, "span_map": span_map, "vocab": vocab}
    with open(os.path.join(ctx.results_dir, "lsa_field.json"), "w", encoding="utf-8") as f:
        json.dump(lsa_flat, f, indent=2)
    ctx.add_artifact("lsa_field.json", "lsa")


# ---------------------------------------------------------------------
# 2. Graph edges (spectral centroid similarity)
# ---------------------------------------------------------------------
def _stage_graph_edges(ctx: PipelineContext) -> None:
    lsa = ctx.extras.get("lsa_result") or {}
    field = lsa.get("field", {})
    elements = lsa.get("elements", [])

    span_map = field.get("span_map", [])
    embeddings = field.get("embeddings", [])
    embeddings = np.asarray(embeddings, dtype=float)

    if embeddings.size == 0:
        ctx.log("warn", "[graph] no embeddings; skipping")
        return

    # map: span_id → embedding index
    span_index = {}
    for i, sp in enumerate(span_map):
        sid = sp.get("span_id", i)
        span_index[int(sid)] = i

    # build centroid per element
    from collections import defaultdict
    bucket = defaultdict(list)

    for rec in elements:
        el = str(rec.get("element"))
        sid = rec.get("span_id")
        if el is None or sid is None:
            continue
        if sid in span_index:
            bucket[el].append(span_index[sid])

    centroids = {}
    for el, idxs in bucket.items():
        centroids[el] = embeddings[idxs].mean(axis=0)

    if not centroids:
        ctx.log("warn", "[graph] no centroids; skipping")
        return

    from sklearn.metrics.pairwise import cosine_similarity

    keys = list(centroids.keys())
    vecs = np.stack([centroids[k] for k in keys])
    S = cosine_similarity(vecs)

    rows = []
    top_k = 8
    min_sim = 0.35

    for i, el in enumerate(keys):
        sims = S[i]
        idx_sorted = np.argsort(sims)[::-1]
        for j in idx_sorted[:top_k]:
            if sims[j] < min_sim:
                continue
            rows.append({
                "source": el,
                "target": keys[j],
                "weight": float(sims[j]),
            })

    df = pd.DataFrame(rows)
    out = os.path.join(ctx.results_dir, "edges.csv")
    df.to_csv(out, index=False)
    ctx.add_artifact("edges.csv", "edges")


# ---------------------------------------------------------------------
# 3. Molecules
# ---------------------------------------------------------------------
def _stage_molecules(ctx: PipelineContext) -> None:
    mol_df, comp_df = run_molecule_layer(ctx.results_dir, emit=ctx.emit)

    if isinstance(mol_df, pd.DataFrame) and not mol_df.empty:
        path = os.path.join(ctx.results_dir, "molecules.csv")
        mol_df.to_csv(path, index=False)
        ctx.add_artifact("molecules.csv", "molecule-table")


# ---------------------------------------------------------------------
# 4. Fusion layer
# ---------------------------------------------------------------------
def _stage_fusion(ctx: PipelineContext) -> None:
    run_fusion_pipeline(ctx.results_dir, emit=ctx.emit)


# ---------------------------------------------------------------------
# 5. Stability
# ---------------------------------------------------------------------
def _stage_stability(ctx: PipelineContext) -> None:
    elements_csv = os.path.join(ctx.results_dir, "hilbert_elements.csv")
    out_csv = os.path.join(ctx.results_dir, "signal_stability.csv")

    compute_signal_stability(
        elements_csv=elements_csv,
        out_csv=out_csv,
        mode="classic",
        emit=ctx.emit,
    )
    ctx.add_artifact("signal_stability.csv", "signal-stability")


# ---------------------------------------------------------------------
# 6. Persistence
# ---------------------------------------------------------------------
def _stage_persistence(ctx: PipelineContext) -> None:
    run_persistence_visuals(ctx.results_dir, emit=ctx.emit)
    ctx.add_artifact("persistence_field.png", "persistence")
    ctx.add_artifact("stability_scatter.png", "stability-scatter")
    ctx.add_artifact("stability_by_doc.png", "stability-by-doc")


# ---------------------------------------------------------------------
# 7. Element Labels
# ---------------------------------------------------------------------
def _stage_element_labels(ctx: PipelineContext) -> None:
    spans = (ctx.extras.get("lsa_result") or {}).get("field", {}).get("span_map", [])
    build_element_descriptions(
        elements_csv=os.path.join(ctx.results_dir, "hilbert_elements.csv"),
        spans=spans,
        out_dir=ctx.results_dir,
    )
    ctx.add_artifact("element_descriptions.json", "element-descriptions")
    ctx.add_artifact("element_intensity.csv", "element-intensity")


# ---------------------------------------------------------------------
# 8. Epistemic signatures
# ---------------------------------------------------------------------
def _stage_signatures(ctx: PipelineContext) -> None:
    out_path = compute_signatures(ctx.results_dir, emit=ctx.emit)
    if out_path:
        ctx.add_artifact(os.path.basename(out_path), "epistemic-signatures")


# ---------------------------------------------------------------------
# 9. Graph snapshots (FULL PATCHED VERSION)
# ---------------------------------------------------------------------
def _stage_graph_snapshots(ctx: PipelineContext) -> None:
    from collections import defaultdict

    results = ctx.results_dir
    elements_csv = os.path.join(results, "hilbert_elements.csv")
    edges_csv = os.path.join(results, "edges.csv")
    molecules_csv = os.path.join(results, "molecules.csv")
    fusion_csv = os.path.join(results, "span_element_fusion.csv")
    labels_csv = os.path.join(results, "epistemic_labels.csv")

    # Required files
    if not os.path.exists(elements_csv):
        ctx.log("warn", "[graph] missing hilbert_elements.csv")
        return
    if not os.path.exists(edges_csv):
        ctx.log("warn", "[graph] missing edges.csv")
        return

    try:
        elements_df = pd.read_csv(elements_csv)
    except Exception as exc:
        ctx.log("warn", f"[graph] failed reading hilbert_elements: {exc}")
        return

    try:
        edges_df = pd.read_csv(edges_csv)
    except Exception as exc:
        ctx.log("warn", f"[graph] failed reading edges.csv: {exc}")
        return

    # Optional molecules
    molecule_df = pd.DataFrame()
    if os.path.exists(molecules_csv):
        try:
            molecule_df = pd.read_csv(molecules_csv)
        except Exception as exc:
            ctx.log("warn", f"[graph] failed reading molecules.csv: {exc}")

    # Optional fusion
    fusion_df = None
    if os.path.exists(fusion_csv):
        try:
            fusion_df = pd.read_csv(fusion_csv)
        except Exception as exc:
            ctx.log("warn", f"[graph] failed reading fusion CSV: {exc}")

    # Optional label map
    label_map = {}
    if os.path.exists(labels_csv):
        try:
            ldf = pd.read_csv(labels_csv)
            if {"element", "label"}.issubset(ldf.columns):
                for _, r in ldf.iterrows():
                    label_map[str(r["element"])] = str(r["label"])
        except Exception as exc:
            ctx.log("warn", f"[graph] failed reading labels: {exc}")

    # -----------------------------------------------------
    # Enriched span mapping
    # -----------------------------------------------------
    element_spans = defaultdict(lambda: {
        "span_ids": [],
        "span_texts": [],
        "documents": set(),
        "doc_freq": defaultdict(int),
        "entropy_values": [],
        "coherence_values": [],
        "labels": set(),
    })

    # Extract from LSA span_map
    span_map_raw = (ctx.extras.get("lsa_result") or {}).get("field", {}).get("span_map", [])
    span_text_map = {}
    for rec in span_map_raw:
        doc = rec.get("doc")
        sid_raw = rec.get("span_id", 0)
        sid = f"{doc}_{sid_raw}"
        span_text_map[sid] = rec.get("text", "")

    # Fusion correlations
    if fusion_df is not None and {"doc", "span_id", "element"}.issubset(fusion_df.columns):
        for _, row in fusion_df.iterrows():
            el = str(row["element"])
            doc = str(row["doc"])
            sid = f"{doc}_{row['span_id']}"

            entry = element_spans[el]
            entry["span_ids"].append(sid)
            entry["documents"].add(doc)
            entry["doc_freq"][doc] += 1
            entry["span_texts"].append(span_text_map.get(sid, ""))

    # Merge entropy/coherence/labels from elements_df
    for _, row in elements_df.iterrows():
        el = str(row.get("element"))
        if el not in element_spans:
            continue

        if "entropy" in row and not pd.isna(row["entropy"]):
            element_spans[el]["entropy_values"].append(row["entropy"])

        if "coherence" in row and not pd.isna(row["coherence"]):
            element_spans[el]["coherence_values"].append(row["coherence"])

        if el in label_map:
            element_spans[el]["labels"].add(label_map[el])

    # -----------------------------------------------------
    # Final JSON-safe enriched structure
    # -----------------------------------------------------
    enriched = {}
    for el, vals in element_spans.items():
        enriched[el] = {
            "span_ids": vals["span_ids"],
            "span_texts": vals["span_texts"],
            "documents": list(vals["documents"]),
            "doc_freq": dict(vals["doc_freq"]),
            "avg_entropy": float(np.mean(vals["entropy_values"])) if vals["entropy_values"] else None,
            "avg_coherence": float(np.mean(vals["coherence_values"])) if vals["coherence_values"] else None,
            "labels": list(vals["labels"]),
        }

    # Write enriched maps
    try:
        path_full = os.path.join(results, "element_span_enriched.json")
        with open(path_full, "w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=2)
        ctx.add_artifact("element_span_enriched.json", "span-enriched")

        path_simple = os.path.join(results, "element_span_map.json")
        with open(path_simple, "w", encoding="utf-8") as f:
            json.dump({el: v["span_ids"] for el, v in enriched.items()}, f, indent=2)
        ctx.add_artifact("element_span_map.json", "span-map")
    except Exception as exc:
        ctx.log("warn", f"[graph] failed writing span maps: {exc}")

    # -----------------------------------------------------
    # Generate graph snapshots
    # -----------------------------------------------------
    out_dir = os.path.join(results, "graph_snapshots")
    os.makedirs(out_dir, exist_ok=True)

    try:
        # Correct patched call signature
        generate_graph_snapshots(
            results_dir=out_dir,
            emit=ctx.emit,
        )
        ctx.add_artifact("graph_snapshots", "graph-snapshots")
    except Exception as exc:
        ctx.log("error", f"[graph] Snapshot generation failed: {exc}")


# ---------------------------------------------------------------------
# 10. Graph export
# ---------------------------------------------------------------------
def _stage_graph_export(ctx: PipelineContext) -> None:
    export_graph_snapshots(ctx.results_dir, emit=ctx.emit)
    ctx.add_artifact("graph_export", "graph-export")


# ---------------------------------------------------------------------
# 11. Full export
# ---------------------------------------------------------------------
def _stage_full_export(ctx: PipelineContext) -> None:
    run_full_export(ctx.results_dir, emit=ctx.emit)
    ctx.add_artifact("hilbert_summary.pdf", "summary")
    ctx.add_artifact("hilbert_run.zip", "archive")


# =====================================================================
# Declarative Stage Table
# =====================================================================

STAGE_TABLE = [
    StageSpec(
        key="lsa_field",
        order=1.0,
        label="[1] LSA spectral field",
        func=_stage_lsa,
        required=True,
        dialectic_role="evidence",
        produces=["lsa_field.json"],
    ),

    StageSpec(
        key="graph_edges",
        order=1.5,
        label="[1.5] Element-element graph",
        func=_stage_graph_edges,
        required=False,
        dialectic_role="structure",
        depends_on=["lsa_field"],
        produces=["edges.csv"],
    ),

    StageSpec(
        key="molecules",
        order=2.0,
        label="[2] Molecule layer",
        func=_stage_molecules,
        required=False,
        dialectic_role="structure",
        depends_on=["lsa_field", "graph_edges"],
        consumes=["hilbert_elements.csv"],
        produces=["molecules.csv", "informational_compounds.json"],
        supports=["graph_snapshots"],
    ),

    StageSpec(
        key="fusion",
        order=3.0,
        label="[3] Span - element fusion",
        func=_stage_fusion,
        required=False,
        dialectic_role="structure",
        depends_on=["lsa_field"],
        consumes=["hilbert_elements.csv"],
        produces=["compound_contexts.json"],
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
        supports=["persistence_visuals", "epistemic_signatures"],
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
        produces=["persistence_field.png", "stability_scatter.png", "stability_by_doc.png"],
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
    ),

    StageSpec(
        key="element_lm",
        order=6.8,
        label="[6.8] Element Language Model",
        func=lambda ctx: run_element_lm_stage(ctx.results_dir, emit=ctx.emit),
        required=False,
        dialectic_role="epistemic",
        depends_on=["fusion"],
        consumes=["hilbert_elements.csv", "span_element_fusion.csv"],
        produces=["element_lm.pt", "element_vocab.json"],
    ),

    StageSpec(
        key="graph_snapshots",
        order=7.0,
        label="[7] Graph snapshots",
        func=_stage_graph_snapshots,
        required=False,
        dialectic_role="visual",
        depends_on=["molecules"],
        produces=["graph_snapshots"],
    ),

    StageSpec(
        key="graph_export",
        order=7.5,
        label="[7.5] Graph export",
        func=_stage_graph_export,
        required=False,
        dialectic_role="visual",
        depends_on=["graph_snapshots"],
        produces=["graph_export"],
    ),

    StageSpec(
        key="export_all",
        order=8.0,
        label="[8] Full export (PDF + archive)",
        func=_stage_full_export,
        required=False,
        dialectic_role="export",
        depends_on=["element_labels"],
        produces=["hilbert_summary.pdf", "hilbert_run.zip"],
    ),
]

# =====================================================================
# Execution Engine
# =====================================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_pipeline(
    corpus_dir: str,
    results_dir: str,
    settings: Optional[PipelineSettings] = None,
    emit: Optional[EmitFn] = None,
) -> Dict[str, Any]:

    settings = settings or PipelineSettings()
    emit = emit or DEFAULT_EMIT

    _ensure_dir(results_dir)

    ctx = PipelineContext(
        corpus_dir=os.path.abspath(corpus_dir),
        results_dir=os.path.abspath(results_dir),
        settings=settings,
        emit=emit,
    )

    ctx.log("info", "Starting Hilbert pipeline", corpus=ctx.corpus_dir)

    # Execute stages in declared order
    for spec in sorted(STAGE_TABLE, key=lambda s: s.order):
        unmet = [
            d for d in spec.depends_on
            if ctx.stages.get(d, StageState(d, d)).status != "ok"
        ]

        if unmet:
            reason = f"dependencies not satisfied: {', '.join(unmet)}"
            if spec.required:
                ctx.begin_stage(spec)
                ctx.end_stage_failed(spec, reason)
                break
            ctx.end_stage_skipped(spec, reason)
            continue

        if spec.func is None:
            ctx.end_stage_skipped(spec, "conceptual stage")
            continue

        ctx.begin_stage(spec)
        try:
            spec.func(ctx)
            ctx.end_stage_ok(spec)
        except Exception as exc:
            ctx.end_stage_failed(spec, str(exc))
            if spec.required:
                ctx.log("error", "Aborting pipeline", stage=spec.key)
                break

    # Save run summary
    summary = ctx.run_summary(STAGE_TABLE)
    try:
        run_json = os.path.join(ctx.results_dir, "hilbert_run.json")
        with open(run_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        ctx.add_artifact("hilbert_run.json", "run-summary")
    except Exception as exc:
        ctx.log("warn", f"Failed to write summary: {exc}")

    ctx.log("info", "Hilbert pipeline complete", run_id=ctx.run_id)
    return summary


run_hilbert_pipeline = run_pipeline
