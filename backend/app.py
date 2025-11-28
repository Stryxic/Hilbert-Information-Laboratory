from __future__ import annotations

"""
Hilbert Information Laboratory Backend API (DB-integrated version)

Updated to use the new modular orchestrator subsystem:

    - hilbert_orchestrator.hilbert_run.hilbert_run
    - Modular stage registry
    - Structured event model
"""

import json
import shutil
import time
import traceback
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import asdict

from fastapi import (
    FastAPI, File, HTTPException, UploadFile, Query, APIRouter
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

# ============================================================================
# Backend paths
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

UPLOAD_BASE = BASE_DIR / "uploaded_corpus"
WORK_BASE = BASE_DIR / "workdirs"
FRONTEND_BUILD = ROOT_DIR / "webapp" / "build"

UPLOAD_BASE.mkdir(parents=True, exist_ok=True)
WORK_BASE.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Database
# ============================================================================

from hilbert_db.core import create_hilbert_db

try:
    db = create_hilbert_db(init_schema=True)
except Exception as exc:
    # Very early failure - make this as loud as possible
    print(json.dumps({
        "msg": "[app] Failed to initialise HilbertDB",
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }, ensure_ascii=False))
    raise

# ============================================================================
# New Orchestrator Import
# ============================================================================

try:
    from hilbert_orchestrator.cli.hilbert_run import hilbert_run, PipelineSettings
except Exception as exc:
    # Explicit debug info if orchestrator import fails (e.g. circular import)
    print(json.dumps({
        "msg": "[app] Failed to import hilbert_orchestrator module",
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }, ensure_ascii=False))
    raise RuntimeError(f"[app] Failed to import hilbert_orchestrator module: {exc}")

# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="Hilbert Information Laboratory API",
    version="4.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_BUILD.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_BUILD)), name="static")

# ============================================================================
# Helpers
# ============================================================================

def _log(msg: str, **extra: Any) -> None:
    """
    Centralised structured logging.

    All logs go through here so we can easily tweak format or sink later.
    """
    try:
        print(json.dumps({"msg": msg, **extra}, ensure_ascii=False))
    except Exception:
        # Last-ditch fallback - never let logging crash the app
        print(f"{msg} {extra}")


def _json_safe(v: Any) -> Any:
    """
    Make values JSON safe by:

        - converting NaN/inf floats to None
        - recursing into dicts/lists
    """
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_json_safe(x) for x in v]
    return v


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    _log("[http] request", method=request.method, path=request.url.path)
    try:
        resp = await call_next(request)
    except Exception as exc:
        _log("[http] error", error=str(exc), traceback=traceback.format_exc())
        raise
    _log(
        "[http] response",
        path=request.url.path,
        duration_ms=int((time.time() - start) * 1000),
        status_code=getattr(resp, "status_code", None),
    )
    return resp

# ============================================================================
# Root / health
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    return "<h1>Hilbert DB API Running</h1>"


@app.get("/health")
async def health():
    return {"status": "ok", "time": time.time()}

# ============================================================================
# Router
# ============================================================================

router = APIRouter(prefix="/api/v1")

# ============================================================================
# Upload corpus
# ============================================================================

@router.post("/upload_corpus")
async def upload_corpus(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files uploaded.")

    _log("[api] upload_corpus", n_files=len(files))

    shutil.rmtree(UPLOAD_BASE, ignore_errors=True)
    UPLOAD_BASE.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []
    for up in files:
        dest = UPLOAD_BASE / up.filename
        try:
            dest.write_bytes(await up.read())
            saved.append(up.filename)
        except Exception as exc:
            _log("[api] upload_corpus write failed",
                 filename=up.filename, error=str(exc),
                 traceback=traceback.format_exc())
            raise HTTPException(500, f"Failed to save {up.filename}: {exc}")

    _log("[api] upload_corpus complete", saved=saved)
    return {"status": "ok", "files": saved}


def _detect_corpus_arg() -> str:
    """
    If exactly one ZIP exists in UPLOAD_BASE, treat it as the corpus;
    otherwise use the upload directory directly.
    """
    zips = [p for p in UPLOAD_BASE.iterdir() if p.suffix.lower() == ".zip"]
    if len(zips) == 1:
        arg = str(zips[0])
        _log("[api] corpus_arg selected zip", path=arg)
        return arg
    arg = str(UPLOAD_BASE)
    _log("[api] corpus_arg selected directory", path=arg)
    return arg

# ============================================================================
# Pipeline entry
# ============================================================================

def _emit_bridge(kind: str, payload: Dict[str, Any]) -> None:
    """
    Bridge orchestrator events to backend structured logs.

    This gives you a clean stream of:

        [pipeline:run_start]
        [pipeline:stage_start]
        [pipeline:stage_end]
        [pipeline:run_end]
        [pipeline:log]
        [pipeline:artifact]
    """
    _log(f"[pipeline:{kind}]", **_json_safe(payload))


@router.post("/analyze_corpus")
async def analyze_corpus(
    max_docs: Optional[int] = None,
    use_native: bool = True,
    corpus_name: str = "Uploaded Corpus",
):
    corpus_arg = _detect_corpus_arg()
    workdir = WORK_BASE / f"run_{int(time.time()*1000)}"
    workdir.mkdir(parents=True, exist_ok=True)

    settings = PipelineSettings(use_native=use_native, max_docs=max_docs)

    _log(
        "[api] analyze_corpus starting",
        corpus_arg=corpus_arg,
        workdir=str(workdir),
        settings=_json_safe(settings.as_dict()),
    )

    try:
        result = hilbert_run(
            db=db,
            corpus_dir=corpus_arg,
            corpus_name=str(corpus_name),
            results_dir=str(workdir),
            settings=settings,
            emit=_emit_bridge,
        )
    except Exception as exc:
        _log(
            "[api] pipeline failed",
            error=str(exc),
            traceback=traceback.format_exc(),
            corpus_arg=corpus_arg,
            workdir=str(workdir),
        )
        raise HTTPException(500, f"Pipeline failed: {exc}")

    _log(
        "[api] analyze_corpus complete",
        run_id=result.get("run_id"),
        corpus_id=result.get("corpus_id"),
        results_dir=result.get("results_dir"),
    )

    return {"status": "ok", "result": _json_safe(result)}


@router.post("/run_full")
async def run_full(use_native: bool = True):
    return await analyze_corpus(max_docs=None, use_native=use_native)

# ============================================================================
# Corpora & Runs
# ============================================================================

@router.get("/corpora")
async def api_list_corpora():
    corp = db.list_corpora()
    return {"corpora": [_json_safe(c.__dict__) for c in corp]}


@router.get("/runs")
async def api_list_runs(corpus_id: Optional[str] = Query(None)):
    if corpus_id:
        r = db.list_runs_for_corpus(corpus_id)
    else:
        r: List[Any] = []
        for c in db.list_corpora():
            r.extend(db.list_runs_for_corpus(c.corpus_id))
    return {"runs": [_json_safe(x.__dict__) for x in r]}


@router.get("/runs/{run_id}")
async def api_get_run(run_id: str):
    r = db.get_run(run_id)
    if not r:
        raise HTTPException(404, f"Run {run_id} not found")
    return _json_safe(r.__dict__)


@router.get("/runs/{run_id}/artifacts")
async def api_run_artifacts(run_id: str):
    arts = db.list_artifacts_for_run(run_id)
    return {"run_id": run_id, "artifacts": [_json_safe(a.__dict__) for a in arts]}

# ============================================================================
# Graph API
# ============================================================================

from hilbert_db.apis.graph_api import (
    GraphRequest, get_graph_snapshot, list_available_graphs
)

@router.get("/graphs/{run_id}/available")
async def api_graph_available(run_id: str):
    try:
        return {
            "run_id": run_id,
            "available": list_available_graphs(db, run_id),
        }
    except Exception as exc:
        _log("[api] graph_available failed", run_id=run_id,
             error=str(exc), traceback=traceback.format_exc())
        raise HTTPException(500, f"Graph not available: {exc}")


@router.get("/graphs/{run_id}/snapshot")
async def api_graph_snapshot(run_id: str, depth: Optional[str] = None):
    try:
        req = GraphRequest(run_id=run_id, depth=depth)
        snap = get_graph_snapshot(db, req)
        return {
            "run_id": run_id,
            "depth": snap.depth,
            "nodes": _json_safe(snap.nodes),
            "edges": _json_safe(snap.edges),
            "metadata": _json_safe(snap.metadata),
        }
    except Exception as exc:
        _log("[api] graph_snapshot failed", run_id=run_id,
             error=str(exc), traceback=traceback.format_exc())
        raise HTTPException(500, f"Graph snapshot failed: {exc}")

# ============================================================================
# Elements API
# ============================================================================

from hilbert_db.apis.elements_api import (
    ElementRequest, list_elements, get_element_detail
)

@router.get("/runs/{run_id}/elements")
async def api_elements(run_id: str, page: int = 1, page_size: int = 200):
    elems = list_elements(db, run_id)
    total = len(elems)
    start = (page - 1) * page_size
    return {
        "run_id": run_id,
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": _json_safe([asdict(x) for x in elems[start:start + page_size]]),
    }


@router.get("/runs/{run_id}/elements/{element_id}")
async def api_element(run_id: str, element_id: str):
    try:
        req = ElementRequest(run_id=run_id, element_id=element_id)
        return _json_safe(asdict(get_element_detail(db, req)))
    except KeyError:
        raise HTTPException(404, f"Element {element_id} not found")
    except Exception as exc:
        _log("[api] element detail failed", run_id=run_id,
             element_id=element_id, error=str(exc),
             traceback=traceback.format_exc())
        raise HTTPException(500, str(exc))

# ============================================================================
# Molecules API
# ============================================================================

try:
    from hilbert_db.apis.molecules_api import (
        MoleculeRequest, list_molecules, get_molecule_detail
    )

    @router.get("/runs/{run_id}/molecules")
    async def api_molecules(run_id: str, page: int = 1, page_size: int = 200):
        mols = list_molecules(db, run_id)
        total = len(mols)
        start = (page - 1) * page_size
        return {
            "run_id": run_id,
            "page": page,
            "page_size": page_size,
            "total": total,
            "items": _json_safe([asdict(x) for x in mols[start:start + page_size]]),
        }

    @router.get("/runs/{run_id}/molecules/{molecule_id}")
    async def api_molecule(run_id: str, molecule_id: str):
        try:
            req = MoleculeRequest(run_id=run_id, molecule_id=molecule_id)
            return _json_safe(asdict(get_molecule_detail(db, req)))
        except KeyError:
            raise HTTPException(404, f"Molecule {molecule_id} not found")
        except Exception as exc:
            _log("[api] molecule detail failed", run_id=run_id,
                 molecule_id=molecule_id, error=str(exc),
                 traceback=traceback.format_exc())
            raise HTTPException(500, str(exc))

except ImportError:
    _log("[api] Molecules API missing — skipping")

# ============================================================================
# Stability API
# ============================================================================

from hilbert_db.apis.stability_api import (
    get_stability_table,
    get_compound_stability,
    get_persistence_field,
)

@router.get("/runs/{run_id}/stability/table")
async def api_stability_table(run_id: str):
    return {
        "run_id": run_id,
        "items": _json_safe([asdict(p) for p in get_stability_table(db, run_id)]),
    }


@router.get("/runs/{run_id}/stability/compounds")
async def api_stability_compounds(run_id: str):
    return {
        "run_id": run_id,
        "items": _json_safe(get_compound_stability(db, run_id)),
    }


@router.get("/runs/{run_id}/persistence_field")
async def api_persistence_field(run_id: str):
    pf = get_persistence_field(db, run_id)
    return _json_safe(asdict(pf))

# ============================================================================
# Admin - Reset DB
# ============================================================================

def _find_candidate_db_paths() -> list[Path]:
    paths: list[Path] = []

    env = os.getenv("HILBERT_DB_URL") or os.getenv("HILBERT_DB_PATH")
    if env:
        if env.startswith("sqlite:///"):
            raw = env.split("sqlite:///", 1)[1]
            paths.append(Path(raw))
        elif env.endswith(".db"):
            paths.append(Path(env))

    cfg = getattr(db, "db_config", None)
    if cfg:
        for attr in ("db_url", "url", "database_url"):
            url = getattr(cfg, attr, None)
            if isinstance(url, str):
                if url.startswith("sqlite:///"):
                    paths.append(Path(url.split("sqlite:///", 1)[1]))
                elif url.endswith(".db"):
                    paths.append(Path(url))

    paths += [
        BASE_DIR / "hilbert.db",
        ROOT_DIR / "hilbert.db",
        Path("hilbert.db"),
    ]

    uniq: list[Path] = []
    seen = set()
    for p in paths:
        s = str(p)
        if s not in seen:
            uniq.append(p)
            seen.add(s)
    return uniq


@router.post("/admin/reset_db")
async def admin_reset_db():
    _log("[admin] reset_db requested")

    paths = _find_candidate_db_paths()
    _log("[admin] reset_db candidates", candidates=[str(p) for p in paths])

    db_file = next((p for p in paths if p.is_file()), None)

    if db_file is None:
        raise HTTPException(500, "SQLite DB file not found; cannot reset.")

    try:
        _log("[admin] deleting DB file", path=str(db_file))
        db_file.unlink()
    except Exception as exc:
        _log("[admin] unlink failed", error=str(exc),
             traceback=traceback.format_exc())
        raise HTTPException(500, f"Failed to delete {db_file}: {exc}")

    global db
    try:
        db = create_hilbert_db(init_schema=True)
    except Exception as exc:
        _log("[admin] recreate schema failed", error=str(exc),
             traceback=traceback.format_exc())
        raise HTTPException(500, f"DB deleted but schema failed to recreate: {exc}")

    return {
        "status": "ok",
        "message": "Database reset and schema recreated",
        "db_path": str(db_file),
    }

# ============================================================================
# Register routes
# ============================================================================

app.include_router(router)
app.include_router(router, prefix="/Hilbert-Information-Laboratory")

# ============================================================================
# Entrypoint
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
