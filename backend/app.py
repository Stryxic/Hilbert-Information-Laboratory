# =============================================================================
# app.py - Hilbert Information Laboratory Backend API
# =============================================================================

from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
import math

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
RESULTS_BASE = BASE_DIR / "results"
FRONTEND_BUILD = ROOT_DIR / "webapp" / "build"

UPLOAD_BASE.mkdir(parents=True, exist_ok=True)
RESULTS_BASE.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Import orchestrator
# ---------------------------------------------------------------------------

try:
    from hilbert_orchestrator import run_pipeline, PipelineSettings
except Exception as exc:  # pragma: no cover - startup failure
    raise RuntimeError(f"[app] Failed to import hilbert_orchestrator: {exc}") from exc

# Optional: batched runner
try:
    from hilbert_pipeline.hilbert_batch_runner import run_batched_pipeline
except Exception:
    run_batched_pipeline = None

# Optional: PDF text helper (currently unused, but kept for future)
try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:
    PdfReader = None


def _pdf_to_text(src_path: str) -> str:
    """
    Very simple PDF-to-text converter.

    Returns plain UTF-8 text. If conversion fails, returns empty string
    so the caller can decide to skip the file.
    """
    if PdfReader is None:
        return ""

    try:
        reader = PdfReader(src_path)
        chunks = []
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


# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Hilbert Information Laboratory API",
    description="Backend API for the Hilbert Information Laboratory / HIDT",
    version="3.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend if present
if FRONTEND_BUILD.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_BUILD)), name="static")

if RESULTS_BASE.exists():
    app.mount("/results", StaticFiles(directory=str(RESULTS_BASE)), name="results")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str, **extra: Any) -> None:
    """Safe backend logger - always JSON on stdout."""
    payload = {"msg": msg}
    payload.update(extra)
    print(json.dumps(payload, ensure_ascii=False))


def _safe_load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_results_dir() -> Optional[Path]:
    """Get most recent run directory."""
    if not RESULTS_BASE.exists():
        return None
    dirs = [p for p in RESULTS_BASE.iterdir() if p.is_dir()]
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def _get_run_dir(latest: bool = True, run_id: Optional[str] = None) -> Path:
    """Resolve a run directory and log what we picked."""
    if latest:
        run_dir = _latest_results_dir()
        if not run_dir:
            _log("[results] No runs available")
            raise HTTPException(404, "No results found.")
        _log("[results] Using latest run directory", run_dir=str(run_dir))
        return run_dir

    if not run_id:
        raise HTTPException(400, "run_id required when latest=false.")
    run_dir = RESULTS_BASE / run_id
    if not run_dir.exists():
        _log("[results] Run directory missing", run_dir=str(run_dir))
        raise HTTPException(404, f"Run directory missing: {run_dir}")
    _log("[results] Using explicit run directory", run_dir=str(run_dir))
    return run_dir


def _emit_bridge(kind: str, payload: Dict[str, Any]) -> None:
    """Bridge orchestrator events to logs."""
    if kind == "log":
        _log("[pipeline]", **payload)
    else:
        _log(f"[pipeline:{kind}]", **payload)


def _detect_corpus_argument() -> str:
    """
    Decide what to pass as `corpus_dir` into run_pipeline:

    - If there is a single ZIP in uploaded_corpus -> pass that ZIP.
    - Otherwise -> pass the directory itself.
    """
    uploaded_dir = UPLOAD_BASE
    if not uploaded_dir.exists():
        raise HTTPException(400, f"Corpus directory missing: {uploaded_dir}")

    items = list(uploaded_dir.iterdir())
    zip_files = [p for p in items if p.suffix.lower() == ".zip"]

    if len(zip_files) == 1:
        corpus_arg = str(zip_files[0])
        _log("[api] ZIP corpus detected", zip=str(zip_files[0]))
    else:
        corpus_arg = str(uploaded_dir)
        _log("[api] Directory corpus detected", dir=str(uploaded_dir))

    return corpus_arg


def _json_safe(value: Any) -> Any:
    """
    Recursively sanitise floats so that NaN / +/-inf become JSON-safe nulls.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# HTTP request logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    _log(
        "[http] request",
        method=request.method,
        path=request.url.path,
        query=str(request.url.query),
    )
    try:
        response = await call_next(request)
    except Exception as exc:
        tb = traceback.format_exc()
        _log(
            "[http] error",
            method=request.method,
            path=request.url.path,
            error=str(exc),
            traceback=tb,
        )
        raise

    duration_ms = int((time.time() - start) * 1000)
    _log(
        "[http] response",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Root / health
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    _log("[api] root")
    return HTMLResponse(
        "<h1>Hilbert Information Laboratory API</h1><p>Backend running.</p>"
    )


@app.get("/health")
async def health() -> Dict[str, Any]:
    now = time.time()
    _log("[api] health", time=now)
    return {"status": "ok", "time": now}


# =====================================================================
# API router - mounted at both /api/v1 and /Hilbert-Information-Laboratory/api/v1
# =====================================================================

router = APIRouter(prefix="/api/v1")


# ---------------------------------------------------------------------------
# Upload corpus
# ---------------------------------------------------------------------------

@router.post("/upload_corpus")
async def upload_corpus(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """
    Upload corpus files (including ZIPs and PDFs).

    Existing contents of uploaded_corpus are cleared before saving.
    """
    _log("[api] upload_corpus called", n_files=len(files) if files else 0)

    if not files:
        raise HTTPException(400, "No files uploaded.")

    # Clear previous upload
    if UPLOAD_BASE.exists():
        shutil.rmtree(UPLOAD_BASE)
    UPLOAD_BASE.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []

    for up in files:
        dest = UPLOAD_BASE / up.filename
        try:
            data = await up.read()
            dest.write_bytes(data)
            saved.append(up.filename)
        except Exception as exc:
            _log("[upload] Failed", filename=up.filename, error=str(exc))
            raise HTTPException(500, f"Failed to save {up.filename}") from exc

    _log("[upload] Corpus uploaded", files=saved)
    return {"status": "ok", "files": saved}


# ---------------------------------------------------------------------------
# High-level run endpoints
# ---------------------------------------------------------------------------

@router.post("/analyze_corpus")
async def analyze_corpus(
    max_docs: Optional[int] = Query(
        None, description="Limit number of docs via batching."
    ),
    use_native: bool = Query(True, description="Use native backend if available."),
) -> Dict[str, Any]:
    """
    Run the full Hilbert pipeline on the uploaded corpus.

    - If `max_docs` is provided, batching is applied via PipelineSettings.max_docs.
    - If a single ZIP is present in uploaded_corpus, it is passed as corpus_dir.
    - Otherwise, the uploaded_corpus directory is used as corpus_dir.
    """
    corpus_arg = _detect_corpus_argument()
    results_dir = RESULTS_BASE / "hilbert_run"

    settings = PipelineSettings(
        use_native=bool(use_native),
        max_docs=max_docs,
    )

    _log(
        "[api] Starting pipeline run",
        corpus=corpus_arg,
        results=str(results_dir),
        max_docs=max_docs,
        use_native=use_native,
    )

    try:
        summary = run_pipeline(
            corpus_dir=corpus_arg,
            results_dir=str(results_dir),
            settings=settings,
            emit=_emit_bridge,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        _log("[api] Pipeline failed", error=str(exc), traceback=tb)
        raise HTTPException(500, f"Pipeline failed: {exc}") from exc

    _log("[api] Pipeline complete", run_id=summary.get("run_id"))
    return _json_safe({"status": "ok", "summary": summary})


@router.post("/run_full")
async def run_full(
    use_native: bool = Query(True, description="Use native backend if available."),
) -> Dict[str, Any]:
    """
    Convenience endpoint: full-corpus run with no batching.
    """
    _log("[api] run_full", use_native=use_native)
    return await analyze_corpus(max_docs=None, use_native=use_native)


@router.post("/run_quick")
async def run_quick(
    max_docs: int = Query(5, description="Number of docs for a quick, batched run."),
    use_native: bool = Query(True, description="Use native backend if available."),
) -> Dict[str, Any]:
    """
    Convenience endpoint: quick - batched run limited to `max_docs` documents.
    """
    _log("[api] run_quick", max_docs=max_docs, use_native=use_native)
    return await analyze_corpus(max_docs=max_docs, use_native=use_native)


# ---------------------------------------------------------------------------
# Batched pipeline wrapper (multi-run fusion)
# ---------------------------------------------------------------------------

@router.post("/analyze_corpus_batched")
async def analyze_corpus_batched(
    batch_size: int = Query(5),
    use_native: bool = Query(True),
) -> Dict[str, Any]:
    """
    Run the batched pipeline using hilbert_batch_runner, if available.
    """
    _log(
        "[api] analyze_corpus_batched called",
        batch_size=batch_size,
        use_native=use_native,
    )

    if run_batched_pipeline is None:
        _log("[api] analyze_corpus_batched unavailable - no run_batched_pipeline")
        raise HTTPException(
            500,
            "Batched pipeline not available - hilbert_pipeline.hilbert_batch_runner.run_batched_pipeline could not be imported.",
        )

    corpus_root = UPLOAD_BASE
    results_root = RESULTS_BASE / "hilbert_batched"

    try:
        run_batched_pipeline(
            corpus_root=str(corpus_root),
            results_root=str(results_root),
            batch_size=batch_size,
            use_native=use_native,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        _log("[api] Batched pipeline failed", error=str(exc), traceback=tb)
        raise HTTPException(500, f"Batched pipeline failed: {exc}") from exc

    _log(
        "[api] Batched pipeline complete",
        results_root=str(results_root),
    )

    return {"status": "ok", "fused_results": str(results_root)}


# ---------------------------------------------------------------------------
# Fusion artefact endpoints (elements, molecules)
# ---------------------------------------------------------------------------

@router.get("/get_elements")
async def get_elements(
    latest: bool = Query(True),
    run_id: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=5000),
) -> Dict[str, Any]:
    """
    Return fusion-level element records from hilbert_elements.csv.

    All float values are sanitised so that NaN / +/-inf become JSON-safe nulls.
    """
    run_dir = _get_run_dir(latest=latest, run_id=run_id)
    csv_path = run_dir / "hilbert_elements.csv"

    if not csv_path.exists():
        _log("[elements] hilbert_elements.csv missing", run_dir=str(run_dir))
        raise HTTPException(404, f"hilbert_elements.csv not found in {run_dir}")

    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(csv_path)
    except Exception as exc:
        _log("[elements] Failed to read hilbert_elements.csv", error=str(exc))
        raise HTTPException(500, f"Failed to read elements CSV: {exc}") from exc

    if limit is not None and limit > 0:
        df = df.head(limit)

    raw_records = df.to_dict(orient="records")
    records = _json_safe(raw_records)

    return {"run_id": run_dir.name, "n_elements": len(records), "elements": records}


@router.get("/get_molecules")
async def get_molecules(
    latest: bool = Query(True),
    run_id: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=5000),
):
    """
    Combine compound_stability.csv (metrics)
    + informational_compounds.json (elements list)
    into a unified molecule record.
    """

    run_dir = _get_run_dir(latest=latest, run_id=run_id)

    import pandas as pd

    # --- Load metrics ---
    stab_csv = run_dir / "compound_stability.csv"
    if not stab_csv.exists():
        raise HTTPException(404, "compound_stability.csv missing")

    df = pd.read_csv(stab_csv)
    if limit:
        df = df.head(limit)

    metrics = {
        str(row["compound_id"]): row
        for _, row in df.iterrows()
        if "compound_id" in row
    }

    # --- Load elements list ---
    info_json = run_dir / "informational_compounds.json"
    if info_json.exists():
        try:
            comp_json = json.loads(info_json.read_text())
        except Exception:
            comp_json = {}
    else:
        comp_json = {}

    # JSON structure: { compound_id: { "elements": [...] } }
    elements_map = {
        str(k): v.get("elements", [])
        for k, v in comp_json.items()
    }

    # --- Merge ---
    merged = []
    for cid, row in metrics.items():
        rec = dict(row)
        rec["elements"] = elements_map.get(cid, [])
        merged.append(rec)

    return {
        "run_id": run_dir.name,
        "n_molecules": len(merged),
        "molecules": merged,
    }




# ---------------------------------------------------------------------------
# Results and metadata endpoints
# ---------------------------------------------------------------------------

@router.get("/get_results")
async def get_results(
    latest: bool = Query(True),
    run_id: Optional[str] = Query(None),
) -> Dict[str, Any]:
    run_dir = _get_run_dir(latest=latest, run_id=run_id)

    run_json = _safe_load_json(run_dir / "hilbert_run.json") or {}
    elem_desc = _safe_load_json(run_dir / "element_descriptions.json") or []

    sample_elements = elem_desc[:50] if isinstance(elem_desc, list) else []

    # stability extraction
    stability_by_doc: List[Dict[str, Any]] = []
    stab_csv = run_dir / "signal_stability.csv"
    if stab_csv.exists():
        try:
            import pandas as pd  # type: ignore

            df = pd.read_csv(stab_csv)
            if {"doc", "doc_stability"}.issubset(df.columns):
                stability_by_doc = df[["doc", "doc_stability"]].to_dict(
                    orient="records"
                )
        except Exception as exc:
            _log("[results] Failed to read stability CSV", error=str(exc))

    payload = {
        "run_id": run_json.get("run_id") or run_dir.name,
        "run_summary": run_json,
        "sample_elements": sample_elements,
        "stability_by_doc": stability_by_doc,
    }
    return _json_safe(payload)


@router.get("/get_document_list")
async def get_document_list(
    latest: bool = Query(True),
    run_id: Optional[str] = Query(None),
) -> Dict[str, Any]:
    run_dir = _get_run_dir(latest=latest, run_id=run_id)

    elem_desc = _safe_load_json(run_dir / "element_descriptions.json") or []
    docs: List[str] = []

    if isinstance(elem_desc, list):
        for rec in elem_desc:
            for d in rec.get("documents", []):
                if d not in docs:
                    docs.append(d)

    _log("[results] get_document_list", n_docs=len(docs), run_dir=str(run_dir))
    return {"run_id": run_dir.name, "documents": docs}


@router.get("/get_timeline_annotations")
async def get_timeline_annotations(
    latest: bool = Query(True),
    run_id: Optional[str] = Query(None),
) -> Dict[str, Any]:
    run_dir = _get_run_dir(latest=latest, run_id=run_id)

    tl = run_dir / "timeline_annotations.json"
    if tl.exists():
        data = _safe_load_json(tl) or {}
        data.setdefault("run_id", run_dir.name)
        _log(
            "[results] get_timeline_annotations (file)",
            n_events=len(data.get("timeline", [])),
            run_dir=str(run_dir),
        )
        return _json_safe(data)

    # fallback timeline from docs
    docs = (await get_document_list(latest=latest, run_id=run_id)).get(
        "documents", []
    )
    annotations = [
        {"id": f"doc_{i}", "label": d, "time_index": i, "meta": {}}
        for i, d in enumerate(docs)
    ]

    payload = {"run_id": run_dir.name, "annotations": annotations}
    _log(
        "[results] get_timeline_annotations (fallback)",
        n_events=len(annotations),
        run_dir=str(run_dir),
    )
    return _json_safe(payload)


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

@router.get("/download_summary")
async def download_summary(
    latest: bool = Query(True),
    run_id: Optional[str] = Query(None),
) -> FileResponse:
    run_dir = _get_run_dir(latest=latest, run_id=run_id)

    pdf_path = run_dir / "hilbert_summary.pdf"
    if not pdf_path.exists():
        _log("[download] Summary PDF missing", run_dir=str(run_dir))
        raise HTTPException(404, "Summary PDF not found.")

    _log("[download] Summary PDF", path=str(pdf_path))
    return FileResponse(str(pdf_path), filename="hilbert_summary.pdf")


@router.get("/download_archive")
async def download_archive(
    latest: bool = Query(True),
    run_id: Optional[str] = Query(None),
) -> FileResponse:
    run_dir = _get_run_dir(latest=latest, run_id=run_id)

    zip_path = run_dir / "hilbert_run.zip"
    if not zip_path.exists():
        _log("[download] Archive missing", run_dir=str(run_dir))
        raise HTTPException(404, "Archive not found.")

    _log("[download] Archive", path=str(zip_path))
    return FileResponse(str(zip_path), filename="hilbert_run.zip")

# =============================================================================
# Vault API Endpoints
# =============================================================================

from hilbert_pipeline.vault_manager import VaultManager

VAULT_BASE = BASE_DIR / "vaults"
VAULT_BASE.mkdir(parents=True, exist_ok=True)


def _resolve_vault_path(vault_name: str) -> Path:
    """Return the absolute path to a vault directory."""
    if not vault_name:
        raise HTTPException(400, "vault_name is required.")
    path = VAULT_BASE / vault_name
    return path


@router.post("/vault/create")
async def vault_create(vault_name: str = Body(..., embed=True)) -> Dict[str, Any]:
    """
    Create a new vault:
    {
        "vault_name": "my_vault"
    }
    """
    path = _resolve_vault_path(vault_name)

    if path.exists():
        raise HTTPException(400, f"Vault '{vault_name}' already exists.")

    try:
        VaultManager.create_vault(str(path))
    except Exception as exc:
        _log("[vault] create failed", error=str(exc))
        raise HTTPException(500, f"Failed to create vault: {exc}")

    _log("[vault] created", name=vault_name)
    return {"status": "ok", "vault": vault_name}


@router.get("/vault/open")
async def vault_open(vault_name: str = Query(...)) -> Dict[str, Any]:
    """
    Open a vault and return a list of notes.
    """
    path = _resolve_vault_path(vault_name)

    if not path.exists():
        raise HTTPException(404, f"Vault '{vault_name}' not found.")

    try:
        vm = VaultManager(str(path))
        notes = vm.list_notes()
    except Exception as exc:
        _log("[vault] open failed", error=str(exc))
        raise HTTPException(500, f"Failed to open vault: {exc}")

    return {"vault": vault_name, "notes": notes}


@router.get("/vault/get_note")
async def vault_get_note(
    vault_name: str = Query(...),
    note_id: str = Query(...)
) -> Dict[str, Any]:
    path = _resolve_vault_path(vault_name)
    vm = VaultManager(str(path))

    try:
        data = vm.load_note(note_id)
    except Exception as exc:
        raise HTTPException(404, f"Note not found: {exc}")

    return data


@router.post("/vault/create_note")
async def vault_create_note(
    vault_name: str = Body(...),
    title: str = Body("Untitled")
) -> Dict[str, Any]:
    path = _resolve_vault_path(vault_name)
    vm = VaultManager(str(path))

    try:
        note_id = vm.create_note(title=title)
    except Exception as exc:
        raise HTTPException(500, f"Failed to create note: {exc}")

    return {"status": "ok", "vault": vault_name, "note_id": note_id}


@router.post("/vault/save_note")
async def vault_save_note(
    vault_name: str = Body(...),
    note_id: str = Body(...),
    meta: Dict[str, Any] = Body(...),
    content: str = Body(...)
) -> Dict[str, Any]:
    path = _resolve_vault_path(vault_name)
    vm = VaultManager(str(path))

    try:
        vm.save_note(note_id, meta, content)
    except Exception as exc:
        raise HTTPException(500, f"Failed to save note: {exc}")

    return {"status": "ok", "note_id": note_id}


@router.delete("/vault/delete_note")
async def vault_delete_note(
    vault_name: str = Query(...),
    note_id: str = Query(...)
) -> Dict[str, Any]:
    path = _resolve_vault_path(vault_name)
    vm = VaultManager(str(path))

    try:
        vm.delete_note(note_id)
    except Exception as exc:
        raise HTTPException(500, f"Failed to delete note: {exc}")

    return {"status": "ok", "note_id": note_id}


@router.get("/vault/search")
async def vault_search(
    vault_name: str = Query(...),
    query: str = Query(...)
) -> Dict[str, Any]:
    path = _resolve_vault_path(vault_name)
    vm = VaultManager(str(path))

    try:
        matches = vm.search(query)
    except Exception as exc:
        raise HTTPException(500, f"Search failed: {exc}")

    return {"vault": vault_name, "query": query, "results": matches}


@router.get("/vault/graph")
async def vault_graph(vault_name: str = Query(...)) -> Dict[str, Any]:
    """
    Return a unified node-edge graph for the vault viewer.
    """
    path = _resolve_vault_path(vault_name)
    vm = VaultManager(str(path))

    try:
        graph = vm.build_graph()
    except Exception as exc:
        raise HTTPException(500, f"Graph generation failed: {exc}")

    return {"vault": vault_name, "graph": graph}

# =====================================================================
# Hilbert Assistant - Ollama chat endpoint
# =====================================================================
# Optional: Ollama client for chat / code review
try:
    from routes.ollama_client import call_ollama  # in backend/ollama_client.py
except Exception:
    call_ollama = None

# ---------------------------------------------------------------------------
# Ollama Chat Endpoint (for NotebookCanvas)
# ---------------------------------------------------------------------------
from pathlib import Path

ROOT_DIR = BASE_DIR.parent  # you already have this

def get_repo_top_dirs() -> list[str]:
    """Return a sorted list of top-level directories in the repo root."""
    dirs = []
    for entry in ROOT_DIR.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            # skip results / uploaded_corpus so it does not loop on its own output
            if entry.name in {"results", "uploaded_corpus", "__pycache__"}:
                continue
            dirs.append(entry.name)
    return sorted(dirs)

@router.post("/ollama_chat")
async def ollama_chat_api(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Chat endpoint behind /api/v1/ollama_chat.

    Expects:
      {
        "messages": [
          { "role": "user" | "assistant" | "system", "content": "<text>" },
          ...
        ],
        "focus": "<optional hint about what to look at>",
        "model": "<optional ollama model name>"
      }

    Returns:
      { "reply": "<assistant text>" }
    """
    print("Calling Ollama /api/v1/ollama_chat")

    if call_ollama is None:
        _log("[api] ollama_chat unavailable - ollama_client not importable")
        raise HTTPException(500, "Ollama client not available on this backend.")

    messages = payload.get("messages") or []
    focus = payload.get("focus")
    model = payload.get("model")

    if not isinstance(messages, list) or not messages:
        raise HTTPException(
            400, "Request must include a non-empty 'messages' list."
        )

    try:
        reply_text = chat_with_hilbert(messages, focus=focus, model=model)
    except Exception as exc:
        _log("[api] ollama_chat failed", error=str(exc))
        raise HTTPException(500, f"Ollama request failed: {exc}") from exc

    return {"reply": reply_text}



@router.get("/get_control_state")
async def get_control_state():
    """
    Returns the latest control-plane snapshot:
    - config used
    - global field stats
    - tuning suggestions
    """
    from hilbert_pipeline.run_registry import RunRegistry

    registry = RunRegistry(RESULTS_BASE)
    state = registry.load_latest()

    if not state:
        raise HTTPException(404, "No control-plane state available.")

    return state



# ---------------------------------------------------------------------------
# Mount router under both base paths
# ---------------------------------------------------------------------------

# Standard base: /api/v1/...
app.include_router(router)

# GitHub Pages / Vite public base: /Hilbert-Information-Laboratory/api/v1/...
app.include_router(router, prefix="/Hilbert-Information-Laboratory")

from routes.code_review import router as code_review_router

app.include_router(code_review_router)
