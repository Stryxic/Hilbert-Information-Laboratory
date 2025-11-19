# =============================================================================
# Hilbert Information Chemistry Lab — Backend API (Upgraded 2025-11-18)
# app.py — FastAPI server with structured error logging & staged pipeline
# =============================================================================

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from typing import List, Dict, Any
import shutil
import os
import sys
import json
import time
import traceback
import asyncio
import importlib.util
import math


# =============================================================================
# Logging Utilities
# =============================================================================

def log_stage(stage: str, msg: str):
    print(f"[{stage}] {msg}")

def log_error(stage: str, exc: Exception):
    print(f"[ERROR][{stage}] {exc}")
    traceback.print_exc()

def safe_load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_error("json", e)
        return None

def run_with_error_boundary(fn, stage: str):
    try:
        return fn()
    except Exception as e:
        log_error(stage, e)
        raise HTTPException(500, f"{stage} failed: {e}")

def _clean_for_json(obj):
    """
    Recursively replace NaN / +/-inf float values with None so that the
    response is JSON-compliant for FastAPI.
    """
    if isinstance(obj, float):
        # NaN, +inf, -inf are not JSON-compliant
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]

    return obj


# =============================================================================
# Load orchestrator
# =============================================================================

try:
    from hilbert_orchestrator import run_pipeline, PipelineContext
    ORCHESTRATOR_AVAILABLE = True
    log_stage("init", "hilbert_orchestrator loaded successfully.")
except Exception as e:
    log_error("orchestrator", e)
    ORCHESTRATOR_AVAILABLE = False

# =============================================================================
# Pipeline Steps for UI
# =============================================================================

PIPELINE_STEPS = [
    {"id": "lsa", "title": "Spectral Field (LSA)", "description": "Compute SVD field."},
    {"id": "elements", "title": "Element Table", "description": "Build hilbert_elements.csv."},
    {"id": "condense", "title": "Element Condensation", "description": "Cluster redundant elements."},
    {"id": "molecules", "title": "Molecules", "description": "Construct informational molecules."},
    {"id": "fusion", "title": "Span–Element Fusion", "description": "Assign spans to elements."},
    {"id": "stability", "title": "Signal Stability", "description": "Compute entropy/coherence."},
    {"id": "persistence", "title": "Persistence Visuals", "description": "Scatter, persistence plots."},
    {"id": "labels", "title": "Element Labels", "description": "Generate human-readable labels."},
    {"id": "post", "title": "Post-processing", "description": "Normalise elements for UI."},
    {"id": "export", "title": "Export", "description": "PDF + ZIP export."},
    {"id": "sanity", "title": "Sanity Checks", "description": "Optional backend sanity checks."},
]

# =============================================================================
# Native module loading
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
NATIVE_DIR = BASE_DIR / "native"
NATIVE_FILE = NATIVE_DIR / "hilbert_native.pyd"

hn = None
HILBERT_NATIVE_AVAILABLE = False

def _try_load_native():
    global hn, HILBERT_NATIVE_AVAILABLE

    log_stage("native", "==== hilbert_native environment ====")
    print(f"[native] Path: {NATIVE_FILE}")
    print(f"[native] Exists: {NATIVE_FILE.exists()}")
    print(f"[native] Executable: {sys.executable}")
    log_stage("native", "=====================================")

    # Attempt 1: import from sys.path
    try:
        import hilbert_native as mod
        hn = mod
        HILBERT_NATIVE_AVAILABLE = True
        log_stage("native", "Loaded hilbert_native from sys.path")
        return
    except Exception:
        pass

    # Attempt 2: explicit .pyd load
    if NATIVE_FILE.exists():
        try:
            spec = importlib.util.spec_from_file_location("hilbert_native", str(NATIVE_FILE))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["hilbert_native"] = mod
            spec.loader.exec_module(mod)
            hn = mod
            HILBERT_NATIVE_AVAILABLE = True
            log_stage("native", "Loaded hilbert_native explicitly from .pyd")
        except Exception as e:
            log_error("native", e)
    else:
        log_stage("native", "No native module found, running Python-only mode.")

_try_load_native()

# =============================================================================
# Lazy numpy/pandas import
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
TIMELINE_FILE = BASE_DIR / "connor_reed_timeline.json"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_ROOT.mkdir(exist_ok=True)

def pick_latest_results_dir(root: Path) -> Path:
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No results present under {root}")
    latest = max(dirs, key=lambda d: d.stat().st_mtime)
    log_stage("results", f"Auto-selected latest folder: {latest}")
    return latest

# =============================================================================
# FastAPI Setup
# =============================================================================

app = FastAPI(title="Hilbert Information Chemistry API")
app.mount("/results", StaticFiles(directory=str(RESULTS_ROOT)), name="results")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# WebSocket bus
# =============================================================================

pipeline_ws_connections = set()

@app.websocket("/ws/pipeline")
async def pipeline_ws(ws: WebSocket):
    await ws.accept()
    pipeline_ws_connections.add(ws)
    log_stage("ws", "client connected")
    try:
        while True:
            await asyncio.sleep(0.2)
    except Exception:
        pass
    finally:
        pipeline_ws_connections.discard(ws)
        log_stage("ws", "client disconnected")

async def broadcast_event(evt: dict):
    dead = []
    for ws in pipeline_ws_connections:
        try:
            await ws.send_text(json.dumps(evt))
        except Exception:
            dead.append(ws)
    for ws in dead:
        pipeline_ws_connections.discard(ws)

# =============================================================================
# Upload Corpus
# =============================================================================

@app.post("/api/v1/upload_corpus")
async def upload_corpus(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files uploaded")

    out = []
    for f in files:
        dest = DATA_DIR / f.filename
        with open(dest, "wb") as o:
            shutil.copyfileobj(f.file, o)
        out.append(f.filename)
        log_stage("upload", f"Saved {f.filename}")

    return {"status": "uploaded", "files": out}

# =============================================================================
# Pipeline Execution
# =============================================================================

@app.post("/api/v1/analyze_corpus")
async def analyze_corpus():
    if not ORCHESTRATOR_AVAILABLE:
        raise HTTPException(500, "Orchestrator unavailable")

    corpus_dir = str(DATA_DIR)
    out_dir = RESULTS_ROOT / "hilbert_run"
    out_dir.mkdir(exist_ok=True)
    log_stage("pipeline", f"Analyze called. corpus={corpus_dir}, out={out_dir}")

    loop = asyncio.get_event_loop()

    def _runner():
        log_stage("pipeline", "Starting Hilbert pipeline...")
        try:
            ctx = run_pipeline(corpus_dir, str(out_dir))
            log_stage("pipeline", "Hilbert pipeline completed.")
            return ctx
        except Exception as e:
            log_error("pipeline", e)
            raise

    try:
        ctx = await loop.run_in_executor(None, _runner)

        # WebSocket event broadcasting
        for evt in getattr(ctx, "step_events", []):
            try:
                await broadcast_event(evt)
            except Exception as e:
                log_error("ws", e)

        return {
            "status": "ok",
            "message": "Pipeline completed",
            "output_dir": str(out_dir),
            "steps": getattr(ctx, "step_events", []),
            "errors": getattr(ctx, "errors", []),
        }

    except Exception as e:
        log_error("pipeline", e)
        raise HTTPException(500, f"Pipeline failed: {e}")

# =============================================================================
# Artifact APIs
# =============================================================================

@app.get("/api/v1/list_artifacts")
def list_artifacts(dir: str):
    p = Path(dir)
    if not p.exists():
        raise HTTPException(404, "Directory not found")
    return {"files": [f.name for f in p.glob("*") if f.is_file()]}

@app.get("/api/v1/get_artifact")
def get_artifact(path: str):
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, "Artifact not found")

    ext = p.suffix.lower()
    if ext == ".json":
        return {"type": "json", "data": safe_load_json(p)}

    if ext == ".csv":
        import pandas as pd
        df = pd.read_csv(p)
        return {"type": "csv", "columns": list(df.columns), "rows": df.astype(str).values.tolist()}

    if ext in (".txt", ".md"):
        return {"type": "text", "data": open(p, "r", encoding="utf-8", errors="ignore").read()}

    if ext in (".png", ".jpg", ".jpeg", ".pdf", ".zip"):
        return {"type": "binary", "message": "Use /api/v1/get_artifact_raw"}

    return {"type": "unknown", "info": f"Cannot preview {ext}"}

@app.get("/api/v1/get_artifact_raw")
def get_artifact_raw(path: str):
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(str(p))

# =============================================================================
# Documents
# =============================================================================

@app.get("/api/v1/get_document_list")
def get_document_list():
    _ensure_np_pd()
    docs = []
    elements_by_doc = {}

    try:
        base = pick_latest_results_dir(RESULTS_ROOT)
        el_path = base / "hilbert_elements.csv"
        if el_path.exists():
            df = pd.read_csv(el_path)
            doc_col = next((c for c in df.columns if c.lower() in ("doc", "document", "file")), None)
            el_col = "element"
            if doc_col:
                df["__doc"] = df[doc_col].map(lambda s: str(s).split("/")[-1])
                group = df.groupby("__doc")[el_col].count()
                elements_by_doc = group.to_dict()
    except Exception as e:
        log_error("documents", e)

    for f in DATA_DIR.glob("*"):
        if f.is_file():
            preview = ""
            try:
                preview = open(f, "r", encoding="utf-8", errors="ignore").read(400)
            except Exception:
                pass

            docs.append({
                "name": f.name,
                "size_kb": round(f.stat().st_size / 1024, 2),
                "modified": time.ctime(f.stat().st_mtime),
                "elements": elements_by_doc.get(f.name, 0),
                "preview": preview.replace("\n", " ") + ("..." if len(preview) == 400 else "")
            })

    return {"documents": docs}

@app.get("/api/v1/get_document_text")
def get_document_text(name: str):
    p = DATA_DIR / name
    if not p.exists():
        raise HTTPException(404, "Document not found")
    return {"name": name, "text": open(p, "r", encoding="utf-8", errors="ignore").read()}

# =============================================================================
# Compound Context
# =============================================================================

@app.get("/api/v1/get_compound_context")
def get_compound_context(element: str):
    base = pick_latest_results_dir(RESULTS_ROOT)
    path = base / "informational_compounds.json"
    data = safe_load_json(path) or []
    if isinstance(data, dict):
        data = list(data.values())

    hits = []
    for c in data:
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
    path = pick_latest_results_dir(RESULTS_ROOT) / "element_descriptions.json"
    raw = safe_load_json(path)
    if isinstance(raw, dict):
        if "elements" in raw:
            return {"elements": raw["elements"]}
        return {"elements": list(raw.values())}
    if isinstance(raw, list):
        return {"elements": raw}
    return {"elements": []}

# =============================================================================
# Unified results endpoint
# =============================================================================

@app.get("/api/v1/get_results")
def get_results(latest: bool = Query(False)):
    _ensure_np_pd()

    base = pick_latest_results_dir(RESULTS_ROOT)

    # meta
    meta = {"dir": str(base), "run": base.name, "generated_at": time.ctime(base.stat().st_mtime)}

    # Load LSA field
    lsa = safe_load_json(base / "lsa_field.json") or {}
    span_map = lsa.get("span_map", [])
    H_bar = lsa.get("H_bar", 0.0)
    C_global = lsa.get("C_global", 0.0)

    # Build spans
    spans = []
    for i, s in enumerate(span_map):
        spans.append({
            "span_id": i,
            "doc": str(s.get("doc")),
            "text": s.get("text"),
            "entropy": s.get("entropy"),
            "coherence": s.get("coherence"),
            "stability": s.get("stability"),
        })

    # Elements layer
    el_data = []
    el_path = base / "hilbert_elements.csv"
    if el_path.exists():
        df = pd.read_csv(el_path)
        el_data = df.to_dict(orient="records")

    # Edges layer
    edges = []
    edges_path = base / "edges.csv"
    if edges_path.exists():
        df_edges = pd.read_csv(edges_path)
        edges = df_edges.to_dict(orient="records")

    # Compounds
    comp_raw = safe_load_json(base / "informational_compounds.json") or []
    if isinstance(comp_raw, dict):
        comp_raw = list(comp_raw.values())

    # Figures
    figures = {}
    for p in base.glob("*"):
        if p.suffix.lower() in (".png", ".jpeg", ".jpg", ".pdf", ".zip"):
            figures[p.name] = f"/results/{base.name}/{p.name}"

    payload = {
        "status": "ok",
        "meta": meta,
        "field": {"spans": spans, "global": {"H_bar": H_bar, "C_global": C_global}},
        "elements": {"elements": el_data},
        "edges": {"edges": edges},
        "compounds": {"compounds": comp_raw},
        "documents": {"documents": []},
        "timeline": {"timeline": []},
        "figures": figures,
    }

    # Ensure NaN / inf do not break JSON encoding
    return _clean_for_json(payload)


# =============================================================================
# Timeline
# =============================================================================

@app.get("/api/v1/get_timeline_annotations")
def get_timeline_annotations():
    data = safe_load_json(TIMELINE_FILE) or []
    if isinstance(data, dict):
        data = data.get("timeline", [])
    return {"timeline": data}

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
    if hn:
        try:
            v = getattr(hn, "version", None)
            info["hilbert_version"] = float(v()) if callable(v) else None
        except:
            info["hilbert_version"] = None
    return info

# =============================================================================
# Pipeline Plan for UI
# =============================================================================

@app.get("/api/v1/get_pipeline_plan")
def get_pipeline_plan_api():
    return {"steps": PIPELINE_STEPS}

@app.get("/")
def root():
    return {"message": "Hilbert Information Chemistry API online"}
