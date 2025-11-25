from __future__ import annotations

"""
Hilbert Information Laboratory Backend API (DB-integrated version)
Uses HilbertDB + run_hilbert_orchestration for full persistent pipeline tracking.
"""

import json
import os
import shutil
import time
import traceback
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    Body,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    Query,
    APIRouter,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

# ---------------------------------------------------------------------------
# Locate backend root
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

UPLOAD_BASE = BASE_DIR / "uploaded_corpus"
WORK_BASE = BASE_DIR / "workdirs"           # temporary working dirs for runs
FRONTEND_BUILD = ROOT_DIR / "webapp" / "build"

UPLOAD_BASE.mkdir(parents=True, exist_ok=True)
WORK_BASE.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load HilbertDB
# ---------------------------------------------------------------------------

from hilbert_db.core import create_hilbert_db
db = create_hilbert_db(init_schema=True)

# Import orchestrator 4.0
try:
    from hilbert_orchestrator import (
        run_hilbert_orchestration,
        PipelineSettings,
    )
except Exception as exc:
    raise RuntimeError(f"[app] Failed to import hilbert_orchestrator: {exc}") from exc

# ---------------------------------------------------------------------------
# PDF helper (optional)
# ---------------------------------------------------------------------------

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Hilbert Information Laboratory API",
    description="DB-integrated backend API for the Hilbert Information Laboratory",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve static if present
if FRONTEND_BUILD.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_BUILD)), name="static")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str, **extra: Any) -> None:
    payload = {"msg": msg}
    payload.update(extra)
    print(json.dumps(payload, ensure_ascii=False))


def _json_safe(value: Any) -> Any:
    """Recursively replace NaN/inf with None."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    _log("[http] request", method=request.method, path=request.url.path)
    try:
        response = await call_next(request)
    except Exception as exc:
        tb = traceback.format_exc()
        _log("[http] error", error=str(exc), traceback=tb)
        raise
    duration_ms = int((time.time() - start) * 1000)
    _log("[http] response", path=request.url.path, status=response.status_code, ms=duration_ms)
    return response


# ---------------------------------------------------------------------------
# Root / health
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse("<h1>Hilbert DB API Running</h1>")


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "time": time.time()}


# ============================================================================
# API router
# ============================================================================

router = APIRouter(prefix="/api/v1")


# ============================================================================
# Upload Corpus
# ============================================================================

@router.post("/upload_corpus")
async def upload_corpus(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """Upload corpus files (PDF, text, directories, or ZIP)."""
    if not files:
        raise HTTPException(400, "No files uploaded.")

    shutil.rmtree(UPLOAD_BASE, ignore_errors=True)
    UPLOAD_BASE.mkdir(parents=True, exist_ok=True)

    saved = []
    for up in files:
        try:
            dest = UPLOAD_BASE / up.filename
            dest.write_bytes(await up.read())
            saved.append(up.filename)
        except Exception as exc:
            raise HTTPException(500, f"Failed to save {up.filename}: {exc}")

    return {"status": "ok", "files": saved}


def _detect_corpus_arg() -> str:
    """ZIP or directory."""
    items = list(UPLOAD_BASE.iterdir())
    zips = [p for p in items if p.suffix.lower() == ".zip"]
    if len(zips) == 1:
        return str(zips[0])
    return str(UPLOAD_BASE)


# ============================================================================
# Run Pipeline (DB-integrated)
# ============================================================================

def _emit_bridge(kind: str, payload: Dict[str, Any]) -> None:
    """Bridge orchestrator events to the backend log."""
    if kind == "log":
        _log("[pipeline]", **payload)
    else:
        _log(f"[pipeline:{kind}]", **payload)


@router.post("/analyze_corpus")
async def analyze_corpus(
    max_docs: Optional[int] = Query(None),
    use_native: bool = Query(True),
    corpus_name: Optional[str] = Query("Uploaded Corpus")
) -> Dict[str, Any]:

    corpus_arg = _detect_corpus_arg()

    # create unique working directory
    run_ts = int(time.time() * 1000)
    workdir = WORK_BASE / f"run_{run_ts}"
    workdir.mkdir(parents=True, exist_ok=True)

    settings = PipelineSettings(use_native=use_native, max_docs=max_docs)

    try:
        result = run_hilbert_orchestration(
            db,
            corpus_dir=corpus_arg,
            corpus_name=str(corpus_name or "Uploaded Corpus"),
            results_dir=str(workdir),
            settings=settings,
            emit=_emit_bridge,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        _log("[api] orchestration failed", error=str(exc), traceback=tb)
        raise HTTPException(500, f"Pipeline failed: {exc}")

    return {"status": "ok", "result": result}


@router.post("/run_full")
async def run_full(use_native: bool = Query(True)):
    return await analyze_corpus(max_docs=None, use_native=use_native)


@router.post("/run_quick")
async def run_quick(max_docs: int = Query(5), use_native: bool = Query(True)):
    return await analyze_corpus(max_docs=max_docs, use_native=use_native)


# ============================================================================
# Query from DB
# ============================================================================

@router.get("/runs")
async def list_runs(corpus_id: Optional[str] = Query(None)):
    if corpus_id:
        runs = db.list_runs_for_corpus(corpus_id)
    else:
        corpora = db.list_corpora()
        runs = []
        for c in corpora:
            runs.extend(db.list_runs_for_corpus(c.corpus_id))
    return {"runs": [r.__dict__ for r in runs]}


@router.get("/run/{run_id}")
async def get_run(run_id: str):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    return run.__dict__


@router.get("/artifacts/{run_id}")
async def get_artifacts(run_id: str):
    arts = db.list_artifacts_for_run(run_id)
    return {"run_id": run_id, "artifacts": [a.__dict__ for a in arts]}


# ============================================================================
# Graph API
# ============================================================================

from hilbert_db.apis.graph_api import (
    GraphRequest,
    get_graph_snapshot,
    list_available_graphs,
)

@router.get("/graphs/{run_id}/available")
async def api_list_graphs(run_id: str):
    depths = list_available_graphs(db, run_id)
    return {"run_id": run_id, "available": depths}


@router.get("/graphs/{run_id}/snapshot")
async def api_graph_snapshot(run_id: str, depth: Optional[str] = Query(None)):
    req = GraphRequest(run_id=run_id, depth=depth)
    resp = get_graph_snapshot(db, req)
    return {
        "run_id": resp.run_id,
        "depth": resp.depth,
        "nodes": resp.nodes,
        "edges": resp.edges,
        "metadata": resp.metadata,
    }


# ============================================================================
# Mount router
# ============================================================================

app.include_router(router)
app.include_router(router, prefix="/Hilbert-Information-Laboratory")
