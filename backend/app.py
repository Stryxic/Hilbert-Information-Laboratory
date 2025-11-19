# =============================================================================
# app.py - Hilbert Information Laboratory Backend API
# =============================================================================
# - Serves the React frontend
# - Exposes endpoints for:
#     * corpus upload
#     * running the Hilbert pipeline
#     * fetching run summaries and results
#     * listing documents and timeline annotations
#
# This version is aligned with:
#   - hilbert_orchestrator 2.x (run_pipeline, PipelineSettings)
#   - upgraded LSA / fusion / molecule layers
#
# Critically:
#   /api/v1/analyze_corpus does NOT require a JSON body, so the existing
#   frontend can continue to POST without sending any payload.
# =============================================================================

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    Body,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# -----------------------------------------------------------------------------
# Locate backend root
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

UPLOAD_BASE = BASE_DIR / "uploaded_corpus"
RESULTS_BASE = BASE_DIR / "results"
FRONTEND_BUILD = ROOT_DIR / "webapp" / "build"

UPLOAD_BASE.mkdir(parents=True, exist_ok=True)
RESULTS_BASE.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Import orchestrator
# -----------------------------------------------------------------------------

try:
    from hilbert_orchestrator import run_pipeline, PipelineSettings
except Exception as exc:
    raise RuntimeError(f"[app] Failed to import hilbert_orchestrator: {exc}") from exc

# -----------------------------------------------------------------------------
# FastAPI app setup
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Hilbert Information Laboratory API",
    description="Backend API for the Hilbert Information Laboratory / HIDT",
    version="3.0.0",
)

# Allow local dev and typical front-end origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve results and frontend assets (if present)
if FRONTEND_BUILD.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_BUILD), html=False), name="static")

if (RESULTS_BASE).exists():
    app.mount(
        "/results",
        StaticFiles(directory=str(RESULTS_BASE), html=False),
        name="results",
    )

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _log(msg: str, **extra: Any) -> None:
    """
    Simple backend log wrapper.
    """
    payload = {"msg": msg, **extra}
    print(json.dumps(payload, ensure_ascii=False))


def _safe_load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_results_dir() -> Optional[Path]:
    """
    Return the most recent run directory under RESULTS_BASE, if any.
    """
    if not RESULTS_BASE.exists():
        return None

    candidates: List[Path] = [
        p for p in RESULTS_BASE.iterdir() if p.is_dir()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# -----------------------------------------------------------------------------
# Simple root / health
# -----------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(
        "<html><body><h1>Hilbert Information Laboratory API</h1>"
        "<p>Backend is running.</p></body></html>"
    )


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "time": time.time()}


# -----------------------------------------------------------------------------
# Corpus upload
# -----------------------------------------------------------------------------

@app.post("/api/v1/upload_corpus")
async def upload_corpus(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """
    Save uploaded documents into UPLOAD_BASE.

    The frontend typically calls this before /api/v1/analyze_corpus.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Clear previous corpus
    if UPLOAD_BASE.exists():
        shutil.rmtree(UPLOAD_BASE)
    UPLOAD_BASE.mkdir(parents=True, exist_ok=True)

    saved_files: List[str] = []

    for up in files:
        dest = UPLOAD_BASE / up.filename
        try:
            contents = await up.read()
            dest.write_bytes(contents)
            saved_files.append(up.filename)
        except Exception as exc:
            _log("[upload] Failed to save file", filename=up.filename, error=str(exc))
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save {up.filename}: {exc}",
            ) from exc

    _log("[upload] Corpus uploaded", files=saved_files)
    return {"status": "ok", "files": saved_files}


# -----------------------------------------------------------------------------
# Run the Hilbert pipeline (no body required)
# -----------------------------------------------------------------------------

@app.post("/api/v1/analyze_corpus")
async def analyze_corpus(
    # Optional query parameters; frontend can ignore them
    max_docs: Optional[int] = Query(None),
    use_native: bool = Query(True),
) -> Dict[str, Any]:
    """
    Run the full Hilbert pipeline on the current uploaded corpus.

    IMPORTANT:
        - No JSON body is required.
        - The React frontend can continue to POST with an empty payload.
    """
    corpus_dir = UPLOAD_BASE
    results_dir = RESULTS_BASE / "hilbert_run"

    if not corpus_dir.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Corpus directory does not exist: {corpus_dir}",
        )

    settings = PipelineSettings(
        use_native=bool(use_native),
        max_docs=max_docs,
    )

    # Emit hook that bridges orchestrator logs to stdout (and could be extended
    # to server-sent events / websockets if needed).
    def emit(kind: str, payload: Dict[str, Any]) -> None:
        if kind == "log":
            _log("[pipeline]", **payload)
        else:
            _log(f"[pipeline:{kind}]", **payload)

    _log(
        "[api] Starting pipeline run",
        corpus=str(corpus_dir),
        results=str(results_dir),
        max_docs=max_docs,
        use_native=use_native,
    )

    try:
        summary = run_pipeline(
            corpus_dir=str(corpus_dir),
            results_dir=str(results_dir),
            settings=settings,
            emit=emit,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        _log("[api] Pipeline failed", error=str(exc), traceback=tb)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {exc}",
        ) from exc

    _log("[api] Pipeline complete", run_id=summary.get("run_id"))
    return {"status": "ok", "summary": summary}


# -----------------------------------------------------------------------------
# Results and document listings
# -----------------------------------------------------------------------------

@app.get("/api/v1/get_results")
async def get_results(
    latest: bool = Query(True),
    run_id: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """
    Return a summary of the most recent (or selected) run.

    Shape is compatible with the existing frontend:
      {
        "run_id": ...,
        "run_summary": { ... },  # from hilbert_run.json
        "sample_elements": [...],
        "stability_by_doc": [...]
      }
    """
    if latest:
        run_dir = _latest_results_dir()
        if run_dir is None:
            raise HTTPException(status_code=404, detail="No results found.")
    else:
        if not run_id:
            raise HTTPException(
                status_code=400,
                detail="run_id must be provided when latest=false",
            )
        run_dir = RESULTS_BASE / run_id
        if not run_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Run directory not found: {run_dir}",
            )

    run_json = _safe_load_json(run_dir / "hilbert_run.json") or {}
    elem_desc = _safe_load_json(run_dir / "element_descriptions.json") or []

    # sample of elements for dashboard preview
    sample_elements = elem_desc[:50] if isinstance(elem_desc, list) else []

    # stability by document, if present
    stability_by_doc: List[Dict[str, Any]] = []
    stab_csv = run_dir / "signal_stability.csv"
    if stab_csv.exists():
        try:
            import pandas as pd

            df = pd.read_csv(stab_csv)
            if {"doc", "doc_stability"}.issubset(df.columns):
                stability_by_doc = df[["doc", "doc_stability"]].to_dict(
                    orient="records"
                )
        except Exception as exc:
            _log("[results] Failed to read signal_stability.csv", error=str(exc))

    payload: Dict[str, Any] = {
        "run_id": run_json.get("run_id"),
        "run_summary": run_json,
        "sample_elements": sample_elements,
        "stability_by_doc": stability_by_doc,
    }
    return payload


@app.get("/api/v1/get_document_list")
async def get_document_list(
    latest: bool = Query(True),
) -> Dict[str, Any]:
    """
    Return a list of document IDs / names discovered in the latest run,
    based on element_descriptions.json (or other metadata if needed).
    """
    if latest:
        run_dir = _latest_results_dir()
        if run_dir is None:
            raise HTTPException(status_code=404, detail="No results found.")
    else:
        raise HTTPException(
            status_code=400,
            detail="Explicit run selection not implemented for document list.",
        )

    elem_desc = _safe_load_json(run_dir / "element_descriptions.json") or []
    docs: List[str] = []

    if isinstance(elem_desc, list):
        for rec in elem_desc:
            if not isinstance(rec, dict):
                continue
            for d in rec.get("documents", []):
                if d not in docs:
                    docs.append(d)

    return {"documents": docs}


@app.get("/api/v1/get_timeline_annotations")
async def get_timeline_annotations(
    latest: bool = Query(True),
) -> Dict[str, Any]:
    """
    Return simple timeline annotations for the latest run.

    Currently:
      - If a dedicated timeline JSON exists, return it.
      - Otherwise, synthesise a trivial timeline from document list.
    """
    if latest:
        run_dir = _latest_results_dir()
        if run_dir is None:
            raise HTTPException(status_code=404, detail="No results found.")
    else:
        raise HTTPException(
            status_code=400,
            detail="Explicit run selection not implemented for timeline.",
        )

    tl_file = run_dir / "timeline_annotations.json"
    if tl_file.exists():
        data = _safe_load_json(tl_file) or {}
        return data

    # Fallback: build a trivial timeline from document names
    doc_resp = await get_document_list(latest=True)
    annotations: List[Dict[str, Any]] = []
    for i, d in enumerate(doc_resp.get("documents", [])):
        annotations.append(
            {
                "id": f"doc_{i}",
                "label": d,
                "time_index": i,
                "meta": {},
            }
        )

    return {"annotations": annotations}


# -----------------------------------------------------------------------------
# Optional: get pipeline configuration / stage table (for UI)
# -----------------------------------------------------------------------------

@app.get("/api/v1/get_pipeline_plan")
async def get_pipeline_plan() -> Dict[str, Any]:
    """
    Return a minimal description of the pipeline plan.

    This avoids importing heavy internals and just reports high-level info.
    """
    # We avoid importing hilbert_orchestrator.STAGE_TABLE here to keep
    # the API lightweight, but you could expose it directly if desired.
    plan = [
        {"key": "lsa_field", "order": 1.0, "label": "[1] LSA spectral field"},
        {"key": "graph_edges", "order": 1.5, "label": "[1.5] Element-element graph"},
        {"key": "molecules", "order": 2.0, "label": "[2] Molecule layer"},
        {"key": "fusion", "order": 3.0, "label": "[3] Span-element fusion"},
        {"key": "stability_metrics", "order": 4.0, "label": "[4] Signal stability"},
        {"key": "element_labels", "order": 6.0, "label": "[6] Element labels"},
        {"key": "graph_snapshots", "order": 7.0, "label": "[7] Graph snapshots"},
        {"key": "export_all", "order": 8.0, "label": "[8] Full export"},
    ]
    return {"plan": plan}


# -----------------------------------------------------------------------------
# Download endpoints (summary PDF, archive)
# -----------------------------------------------------------------------------

@app.get("/api/v1/download_summary")
async def download_summary(latest: bool = Query(True)) -> FileResponse:
    """
    Return hilbert_summary.pdf for the latest run.
    """
    if latest:
        run_dir = _latest_results_dir()
        if run_dir is None:
            raise HTTPException(status_code=404, detail="No results found.")
    else:
        raise HTTPException(
            status_code=400,
            detail="Explicit run selection not implemented for downloads.",
        )

    pdf_path = run_dir / "hilbert_summary.pdf"
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Summary PDF not found: {pdf_path}",
        )

    return FileResponse(str(pdf_path), filename="hilbert_summary.pdf")


@app.get("/api/v1/download_archive")
async def download_archive(latest: bool = Query(True)) -> FileResponse:
    """
    Return hilbert_run.zip for the latest run.
    """
    if latest:
        run_dir = _latest_results_dir()
        if run_dir is None:
            raise HTTPException(status_code=404, detail="No results found.")
    else:
        raise HTTPException(
            status_code=400,
            detail="Explicit run selection not implemented for downloads.",
        )

    zip_path = run_dir / "hilbert_run.zip"
    if not zip_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Archive not found: {zip_path}",
        )

    return FileResponse(str(zip_path), filename="hilbert_run.zip")
