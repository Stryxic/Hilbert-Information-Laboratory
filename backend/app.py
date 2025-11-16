# =============================================================================
# Hilbert Information Chemistry Lab — Backend API
# app.py — FastAPI server for corpus upload, orchestration, data access,
#          pipeline events & artifact streaming
# =============================================================================

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import List, Dict, Any, Tuple
import shutil
import os
import sys
import json
import time
import traceback
import importlib.util
import asyncio

# =============================================================================
# Load orchestrator
# =============================================================================

try:
    from hilbert_orchestrator import (
        run_pipeline,
        PipelineContext,
    )
    ORCHESTRATOR_AVAILABLE = True
    print("[init] hilbert_orchestrator loaded successfully.")
except Exception as e:
    print(f"[warn] hilbert_orchestrator import failed: {e}")
    ORCHESTRATOR_AVAILABLE = False

# A static pipeline plan for the frontend to render the step list.
# This mirrors the orchestrator's stages semantically.
PIPELINE_STEPS: List[Dict[str, Any]] = [
    {
        "id": "lsa",
        "title": "Spectral Field (LSA)",
        "description": "Compute TF-IDF, truncated SVD, and the latent spectral field over sentence-level spans.",
    },
    {
        "id": "elements",
        "title": "Element Table",
        "description": "Build hilbert_elements.csv and attach per-element embeddings, entropy, and coherence.",
    },
    {
        "id": "condense",
        "title": "Element Condensation",
        "description": "Condense near-duplicate elements into root elements using entropy-aware cosine clustering.",
    },
    {
        "id": "molecules",
        "title": "Molecules & Compounds",
        "description": "Construct informational molecules from the element graph and aggregate into compounds.",
    },
    {
        "id": "fusion",
        "title": "Span–Element Fusion",
        "description": "Optionally assign spans to elements and build compound-level context summaries.",
    },
    {
        "id": "stability",
        "title": "Signal Stability",
        "description": "Compute entropy/coherence-based stability metrics across spans and elements.",
    },
    {
        "id": "persistence",
        "title": "Persistence Visuals",
        "description": "Render stability and persistence visuals (line plots and scatter fields).",
    },
    {
        "id": "labels",
        "title": "Element Labels",
        "description": "Generate human-readable labels, summaries, and examples for each informational element.",
    },
    {
        "id": "post",
        "title": "Post-processing",
        "description": "Normalise hilbert_elements.csv for frontend consumption (doc, token, etc.).",
    },
    {
        "id": "export",
        "title": "Export",
        "description": "Produce the summary PDF report and ZIP bundle of core Hilbert artifacts.",
    },
    {
        "id": "sanity",
        "title": "Sanity Checks",
        "description": "Optionally run sanity checks over the run artifacts and summarise diagnostics.",
    },
]

# =============================================================================
# Native optional
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
NATIVE_DIR = BASE_DIR / "native"
NATIVE_FILE = NATIVE_DIR / "hilbert_native.pyd"

hn = None
HILBERT_NATIVE_AVAILABLE = False


def _try_load_native():
    """
    Attempt to load hilbert_native but DO NOT warn loudly on failure.
    Native mode is optional and Python fallback is fully supported.
    """
    global hn, HILBERT_NATIVE_AVAILABLE

    print("[native][debug] ==== hilbert_native environment ====")
    print(f"[native][debug] BASE_DIR      = {BASE_DIR}")
    print(f"[native][debug] NATIVE_DIR    = {NATIVE_DIR}")
    print(f"[native][debug] NATIVE_FILE   = {NATIVE_FILE}")
    print(f"[native][debug] exists?        = {NATIVE_FILE.exists()}")
    print(f"[native][debug] executable     = {sys.executable}")
    print(f"[native][debug] Python         = {sys.version}")
    print(f"[native][debug] =====================================")

    # Try standard import
    try:
        import hilbert_native as mod
        hn = mod
        HILBERT_NATIVE_AVAILABLE = True
        print("[native] Loaded hilbert_native from sys.path")
        return
    except Exception:
        pass

    # Try explicit load
    if not NATIVE_FILE.exists():
        print("[native] hilbert_native.pyd not present - using Python mode.")
        return

    try:
        spec = importlib.util.spec_from_file_location("hilbert_native", str(NATIVE_FILE))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["hilbert_native"] = mod
            spec.loader.exec_module(mod)
            hn = mod
            HILBERT_NATIVE_AVAILABLE = True
            print("[native] hilbert_native loaded explicitly from file")
        else:
            print("[native] Could not build import spec")
    except Exception as e:
        print(f"[native] Failed to load hilbert_native: {e}")
        print("[native] Falling back to Python LSA/graph implementations.")


_try_load_native()

# =============================================================================
# Lazy np/pd
# =============================================================================

pd = None
np = None


def _ensure_np_pd():
    global pd, np
    if pd is None:
        import pandas as _pd
        pd = _pd
    if np is None:
        import numpy as _np
        np = _np


# =============================================================================
# Global Paths
# =============================================================================

RESULTS_ROOT = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "uploaded_corpus"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_ROOT.mkdir(exist_ok=True)

TIMELINE_FILE = BASE_DIR / "connor_reed_timeline.json"


def pick_latest_results_dir(root: Path) -> Path:
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No results present under {root}")
    latest = max(dirs, key=lambda d: d.stat().st_mtime)
    print(f"[results] Auto-selected latest folder: {latest}")
    return latest


# =============================================================================
# FastAPI Setup
# =============================================================================

app = FastAPI(title="Hilbert Information Chemistry API")

# serve generated PNG/PDF/ZIP
app.mount("/results", StaticFiles(directory=str(RESULTS_ROOT)), name="results")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# WebSocket: Pipeline event stream
# =============================================================================

pipeline_ws_connections = set()


@app.websocket("/ws/pipeline")
async def pipeline_ws(ws: WebSocket):
    await ws.accept()
    pipeline_ws_connections.add(ws)
    print("[ws] client connected")

    try:
        while True:
            # nothing to receive, we just keep the connection open
            await asyncio.sleep(0.2)
    except Exception:
        pass
    finally:
        pipeline_ws_connections.discard(ws)
        print("[ws] client disconnected")


async def broadcast_event(evt: dict):
    """
    Push pipeline events to all websocket subscribers.
    """
    dead = []
    for ws in pipeline_ws_connections:
        try:
            await ws.send_text(json.dumps(evt))
        except Exception:
            dead.append(ws)
    for ws in dead:
        pipeline_ws_connections.discard(ws)


# =============================================================================
# Helpers
# =============================================================================

def normalize_doc_id(raw):
    if raw is None:
        return ""
    s = str(raw).replace("\\", "/")
    return s.split("/")[-1]


def json_load_safely(fh):
    try:
        return json.load(fh)
    except Exception:
        try:
            fh.seek(0)
            return fh.read()
        except Exception:
            return None


def _safe_float(x, default=None):
    try:
        v = float(x)
        return v if v == v else default
    except Exception:
        return default


def _load_timeline_events():
    if not TIMELINE_FILE.exists():
        return [], f"No timeline file: {TIMELINE_FILE}"

    try:
        with open(TIMELINE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        return [], str(e)

    if isinstance(raw, list):
        events = raw
    else:
        events = raw.get("timeline") or raw.get("events") or []

    def dt(ev):
        return ev.get("date") or ev.get("timestamp") or ""

    return sorted(events, key=dt), None


# =============================================================================
# Upload
# =============================================================================

@app.post("/api/v1/upload_corpus")
async def upload_corpus(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files uploaded")

    uploaded = []
    for f in files:
        dest = DATA_DIR / f.filename
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        uploaded.append(f.filename)
        print(f"[upload] Saved {f.filename}")

    return {"status": "uploaded", "files": uploaded}


# =============================================================================
# Pipeline Execution
# =============================================================================

# =============================================================================
# Pipeline Execution
# =============================================================================

@app.post("/api/v1/analyze_corpus")
async def analyze_corpus():
    """
    Run the full Hilbert pipeline asynchronously in a thread executor.
    After completion, return a summary and (optionally) broadcast the
    collected step events to any connected WebSocket clients.
    """
    if not ORCHESTRATOR_AVAILABLE:
        raise HTTPException(500, "Orchestrator not available")

    corpus_dir = str(DATA_DIR)
    out_dir = RESULTS_ROOT / "hilbert_run"
    out_dir.mkdir(exist_ok=True)

    print(f"[pipeline] analyze_corpus called. corpus={corpus_dir}, out={out_dir}")

    loop = asyncio.get_event_loop()

    # -------------------------------------------------------------
    # Worker: run pipeline inside executor thread
    # -------------------------------------------------------------
    def _runner():
        """
        This runs inside the thread pool and calls the orchestrator.
        run_pipeline() returns a PipelineContext instance.
        """
        print("[pipeline] Starting Hilbert pipeline...")
        ctx = run_pipeline(corpus_dir, str(out_dir))
        print("[pipeline] Hilbert pipeline completed.")
        return ctx

    try:
        # Run the pipeline in a background thread
        ctx = await loop.run_in_executor(None, _runner)

        # ctx is a PipelineContext from hilbert_orchestrator
        # It has: ctx.step_events, ctx.errors, etc.
        # Optionally push all recorded events to WebSocket listeners.
        try:
            for evt in getattr(ctx, "step_events", []):
                # evt is already a structured dict {type, timestamp, step, data}
                await broadcast_event(evt)
        except Exception:
            # WS broadcasting should never crash the API
            traceback.print_exc()

        status = "ok" if not getattr(ctx, "errors", None) else "error"

        return {
            "status": status,
            "message": "Pipeline completed" if status == "ok" else "Pipeline completed with errors",
            "output_dir": str(out_dir),
            "steps": getattr(ctx, "step_events", []),
            "errors": getattr(ctx, "errors", []),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Pipeline failed: {e}")



# =============================================================================
# Artifact API for UI Pipeline Orchestrator
# =============================================================================

@app.get("/api/v1/list_artifacts")
def list_artifacts(dir: str):
    """
    List all files in a given results directory.
    """
    p = Path(dir)
    if not p.exists():
        raise HTTPException(404, "Directory not found")

    files = []
    for f in p.glob("*"):
        if f.is_file():
            files.append(f.name)

    return {"files": files}


@app.get("/api/v1/get_artifact")
def get_artifact(path: str):
    """
    Return JSON, CSV, text or metadata description of a file.
    Raw bytes for images/PDF are handled by /get_artifact_raw.
    """
    import pandas as pd

    p = Path(path)
    if not p.exists():
        raise HTTPException(404, "Artifact not found")

    ext = p.suffix.lower()

    if ext in (".json",):
        with open(p, "r", encoding="utf-8") as f:
            return {"type": "json", "data": json_load_safely(f)}

    if ext == ".csv":
        df = pd.read_csv(p)
        return {
            "type": "csv",
            "columns": df.columns.tolist(),
            "rows": df.astype(str).values.tolist(),
        }

    if ext in (".txt", ".md"):
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return {"type": "text", "data": f.read()}

    if ext in (".png", ".jpg", ".jpeg", ".pdf", ".zip"):
        # UI should switch to raw endpoint
        return {
            "type": "binary",
            "message": "Use /api/v1/get_artifact_raw to retrieve binary content.",
        }

    return {"type": "unknown", "info": f"Cannot preview extension {ext}"}


@app.get("/api/v1/get_artifact_raw")
def get_artifact_raw(path: str):
    """
    Return raw binary (image/pdf/zip).
    """
    from fastapi.responses import FileResponse

    p = Path(path)
    if not p.exists():
        raise HTTPException(404, "Artifact not found")

    return FileResponse(str(p))


# =============================================================================
# Document(s)
# =============================================================================

@app.get("/api/v1/get_document_list")
def get_document_list():
    """
    Return uploaded corpus documents + element counts.
    """
    _ensure_np_pd()
    docs = []
    elements_by_doc = {}

    try:
        base = pick_latest_results_dir(RESULTS_ROOT)
        el_path = base / "hilbert_elements.csv"

        if el_path.exists():
            df = pd.read_csv(el_path)
            doc_col = next(
                (c for c in df.columns if c.lower() in ("doc", "document", "file", "filename")),
                None,
            )
            el_col = next(
                (c for c in df.columns if c.lower() in ("element", "token")),
                None,
            )
            if doc_col and el_col:
                df["__doc"] = df[doc_col].map(normalize_doc_id)
                group = df.groupby("__doc")[el_col].count()
                elements_by_doc = group.to_dict()

    except Exception:
        pass

    for f in DATA_DIR.glob("*"):
        if f.is_file():
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    preview = fh.read(400)
            except Exception:
                preview = ""
            docs.append(
                {
                    "name": f.name,
                    "size_kb": round(f.stat().st_size / 1024, 2),
                    "modified": time.ctime(f.stat().st_mtime),
                    "elements": elements_by_doc.get(f.name, 0),
                    "preview": preview.replace("\n", " ")
                    + ("..." if len(preview) == 400 else ""),
                }
            )

    return {"documents": docs}


@app.get("/api/v1/get_document_text")
def get_document_text(name: str):
    path = DATA_DIR / name
    if not path.exists():
        raise HTTPException(404, "Document not found")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return {"name": name, "text": f.read()}


# =============================================================================
# Compound Context
# =============================================================================

@app.get("/api/v1/get_compound_context")
def get_compound_context(element: str):
    base = pick_latest_results_dir(RESULTS_ROOT)
    path = base / "informational_compounds.json"

    if not path.exists():
        raise HTTPException(404, "informational_compounds.json missing")

    raw = json.loads(open(path).read())
    if isinstance(raw, dict):
        comps = list(raw.values())
    elif isinstance(raw, list):
        comps = raw
    else:
        comps = []

    hits = []
    for c in comps:
        elems = c.get("elements") or c.get("element_ids") or []
        if isinstance(elems, str):
            elems = [x.strip() for x in elems.split(",")]
        if element in elems:
            hits.append(c)

    return {"element": element, "compounds": hits}


# =============================================================================
# Element Map
# =============================================================================

@app.get("/api/v1/get_element_map")
def get_element_map():
    base = pick_latest_results_dir(RESULTS_ROOT)
    path = base / "element_descriptions.json"

    if not path.exists():
        raise HTTPException(404, "element_descriptions.json not found")

    raw = json_load_safely(open(path, "r", encoding="utf-8"))

    if isinstance(raw, dict) and "elements" in raw:
        return {"elements": raw["elements"]}

    if isinstance(raw, dict):
        return {"elements": list(raw.values())}

    if isinstance(raw, list):
        return {"elements": raw}

    return {"elements": []}


# =============================================================================
# Unified /get_results endpoint (Dashboard)
# =============================================================================

@app.get("/api/v1/get_results")
def get_results(latest: bool = Query(False)):
    _ensure_np_pd()

    try:
        base = pick_latest_results_dir(RESULTS_ROOT)
    except Exception as e:
        raise HTTPException(404, str(e))

    meta = {
        "dir": str(base),
        "run": base.name,
        "generated_at": time.ctime(base.stat().st_mtime),
    }

    # 1. LSA Field
    lsa_path = base / "lsa_field.json"
    stability_path = base / "signal_stability.csv"

    span_map = []
    H_bar = C_global = 0.0

    if lsa_path.exists():
        lsa = json_load_safely(open(lsa_path, "r"))
        if isinstance(lsa, str):
            try:
                lsa = json.loads(lsa)
            except Exception:
                lsa = {}
        span_map = lsa.get("field", {}).get("span_map") or lsa.get("span_map") or []
        H_bar = _safe_float(lsa.get("H_bar"), 0.0)
        C_global = _safe_float(lsa.get("C_global"), 0.0)

    spans_list = []
    stab_df = pd.read_csv(stability_path) if stability_path.exists() else None

    for i, s in enumerate(span_map):
        doc = normalize_doc_id(s.get("doc") or "corpus")
        text = s.get("text") or ""
        entropy = coherence = stability = None
        if stab_df is not None and i < len(stab_df):
            row = stab_df.iloc[i]
            entropy = _safe_float(row.get("entropy"))
            coherence = _safe_float(row.get("coherence"))
            stability = _safe_float(row.get("stability"))
        spans_list.append(
            {
                "span_id": i,
                "doc": doc,
                "text": text,
                "entropy": entropy,
                "coherence": coherence,
                "stability": stability,
            }
        )

    field = {
        "spans": spans_list,
        "global": {
            "H_bar": H_bar,
            "C_global": C_global,
            "n_spans": len(spans_list),
        },
    }

    # 2. Element layer
    elements_layer = {"elements": [], "stats": {}}
    el_path = base / "hilbert_elements.csv"

    if el_path.exists():
        df = pd.read_csv(el_path)
        doc_col = next(
            (c for c in df.columns if c.lower() in ("doc", "document", "file", "filename")),
            None,
        )
        if doc_col:
            df["doc"] = df[doc_col].map(normalize_doc_id)

        if "element" not in df.columns and "token" in df.columns:
            df["element"] = df["token"]
        if "token" not in df.columns and "element" in df.columns:
            df["token"] = df["element"]

        elements_layer["elements"] = df.to_dict(orient="records")
        elements_layer["stats"]["n_elements"] = len(df)

        # Optional root element info (from condensation)
        roots_path = base / "element_roots.csv"
        if roots_path.exists():
            try:
                rdf = pd.read_csv(roots_path)
                # Our current roots file is a table of root elements, not a mapping.
                # If a mapping is ever added with 'element'/'root_element', we detect it.
                if {"element", "root_element"}.issubset(rdf.columns):
                    roots_map = dict(zip(rdf["element"], rdf["root_element"]))
                    elements_layer["stats"]["n_roots"] = len(set(roots_map.values()))
                else:
                    elements_layer["stats"]["n_roots"] = len(rdf.get("element", rdf).unique())
            except Exception:
                pass

    # 3. Edge layer
    edges_layer = {"edges": [], "stats": {}}
    edges_path = base / "edges.csv"
    if edges_path.exists():
        df_edges = pd.read_csv(edges_path)
        edges_layer["edges"] = df_edges.to_dict(orient="records")
        edges_layer["stats"]["n_edges"] = len(df_edges)

    # 4. Compound layer
    comp_layer = {"compounds": [], "stats": {}}
    comp_path = base / "informational_compounds.json"
    if comp_path.exists():
        raw = json_load_safely(open(comp_path))
        if isinstance(raw, dict):
            comps = list(raw.values())
        elif isinstance(raw, list):
            comps = raw
        else:
            comps = []
        comp_layer["compounds"] = comps
        comp_layer["stats"]["n_compounds"] = len(comps)

    # 5. Documents
    doc_records = [
        {"doc": f.name, "signature": {}} for f in DATA_DIR.glob("*") if f.is_file()
    ]
    documents_layer = {"documents": doc_records}

    # 6. Timeline
    events, err = _load_timeline_events()
    timeline_layer = {"timeline": events, "error": err}

    # 7. Figures
    figures = {}
    for p in base.glob("*"):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf", ".zip"):
            v = int(p.stat().st_mtime)
            figures[p.name] = f"/results/{base.name}/{p.name}?v={v}"

    return {
        "status": "ok",
        "meta": meta,
        "field": field,
        "elements": elements_layer,
        "edges": edges_layer,
        "compounds": comp_layer,
        "documents": documents_layer,
        "timeline": timeline_layer,
        "figures": figures,
    }


# =============================================================================
# Timeline endpoint
# =============================================================================

@app.get("/api/v1/get_timeline_annotations")
def get_timeline_annotations():
    events, err = _load_timeline_events()
    return {"timeline": events, "error": err}


# =============================================================================
# Status
# =============================================================================

@app.get("/api/v1/status")
def status():
    info = {
        "status": "ok",
        "orchestrator": ORCHESTRATOR_AVAILABLE,
        "native": HILBERT_NATIVE_AVAILABLE,
    }
    if HILBERT_NATIVE_AVAILABLE and hn:
        try:
            v = getattr(hn, "version", None)
            info["hilbert_version"] = float(v()) if callable(v) else None
        except Exception:
            info["hilbert_version"] = None
    return info


# =============================================================================
# Pipeline Plan (frontend uses this to render the step list)
# =============================================================================

@app.get("/api/v1/get_pipeline_plan")
def get_pipeline_plan_api():
    """
    Return the ordered list of pipeline steps with id, title, description.
    This is kept in sync with hilbert_orchestrator's stage design.
    """
    if not ORCHESTRATOR_AVAILABLE:
        raise HTTPException(500, "Orchestrator not available")

    return {"steps": PIPELINE_STEPS}


@app.get("/")
def root():
    return {"message": "Hilbert Information Chemistry API online"}
