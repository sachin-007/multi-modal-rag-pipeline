"""FastAPI entrypoint for ModalRAG."""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config, pipeline
from app.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestResponse,
    SourceItem,
    StatusResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = config.ROOT_DIR / "static"

_status_lock = threading.Lock()
_ingest_status: Dict[str, Any] = {
    "state": "idle",
    "message": "No document indexed yet",
    "progress": 0.0,
    "document_name": None,
    "error": None,
    "ready": False,
}
_ingest_thread: Optional[threading.Thread] = None


def _set_status(
    state: str,
    message: str,
    progress: float = 0.0,
    document_name: Optional[str] = None,
    error: Optional[str] = None,
    ready: Optional[bool] = None,
) -> None:
    with _status_lock:
        _ingest_status["state"] = state
        _ingest_status["message"] = message
        _ingest_status["progress"] = progress
        if document_name is not None:
            _ingest_status["document_name"] = document_name
        _ingest_status["error"] = error
        if ready is None:
            _ingest_status["ready"] = state == "ready"
        else:
            _ingest_status["ready"] = ready


def _progress_callback(state: str, message: str, progress: float) -> None:
    _set_status(state, message, progress=progress, ready=state == "ready")


def _refresh_ready_from_disk() -> None:
    ready = pipeline.chroma_is_ready()
    if ready:
        name = pipeline.get_document_name() or _ingest_status.get("document_name")
        _set_status(
            "ready",
            f"Ready{f' — {name}' if name else ''}",
            progress=1.0,
            document_name=name,
            error=None,
            ready=True,
        )
        if config.OLLAMA_API_KEY:
            try:
                pipeline.load_vectorstore()
            except Exception as e:
                logger.warning("Could not preload vector store: %s", e)
    else:
        _set_status(
            "idle",
            "No document indexed yet",
            progress=0.0,
            ready=False,
        )


def _run_ingest_job(pdf_path: str, document_name: str) -> None:
    try:
        _set_status(
            "parsing",
            f"Starting ingest for {document_name}",
            progress=0.05,
            document_name=document_name,
            error=None,
            ready=False,
        )
        pipeline.reset_vectorstore_cache()
        pipeline.run_ingestion(
            pdf_path,
            document_name=document_name,
            on_progress=_progress_callback,
        )
        _set_status(
            "ready",
            f"Indexed {document_name}",
            progress=1.0,
            document_name=document_name,
            error=None,
            ready=True,
        )
    except Exception as e:
        logger.exception("Ingest failed")
        _set_status(
            "error",
            "Ingest failed",
            progress=0.0,
            document_name=document_name,
            error=str(e),
            ready=False,
        )


def _start_ingest(pdf_path: str, document_name: str) -> None:
    global _ingest_thread
    with _status_lock:
        if _ingest_thread is not None and _ingest_thread.is_alive():
            raise HTTPException(
                status_code=409,
                detail="An ingest job is already running. Wait for it to finish.",
            )
        thread = threading.Thread(
            target=_run_ingest_job,
            args=(pdf_path, document_name),
            daemon=True,
        )
        _ingest_thread = thread
        thread.start()


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.ensure_dirs()
    _refresh_ready_from_disk()
    yield


app = FastAPI(title="ModalRAG", version="1.0.0", lifespan=lifespan)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    with _status_lock:
        ready = bool(_ingest_status["ready"]) or pipeline.chroma_is_ready()
        doc = _ingest_status.get("document_name") or pipeline.get_document_name()
    return HealthResponse(
        status="ok",
        ready=ready,
        document_name=doc,
        llm_model=config.LLM_MODEL,
        embed_model=config.EMBED_MODEL,
        has_api_key=bool(config.OLLAMA_API_KEY),
    )


@app.get("/api/status", response_model=StatusResponse)
def status() -> StatusResponse:
    with _status_lock:
        data = dict(_ingest_status)
    if not data.get("ready") and pipeline.chroma_is_ready() and data["state"] == "idle":
        _refresh_ready_from_disk()
        with _status_lock:
            data = dict(_ingest_status)
    return StatusResponse(**data)


@app.post("/api/ingest/demo", response_model=IngestResponse)
def ingest_demo() -> IngestResponse:
    if not config.OLLAMA_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="OLLAMA_API_KEY is not configured on the server.",
        )
    try:
        pdf_path = config.ensure_demo_pdf()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not prepare demo PDF: {e}",
        ) from e
    _start_ingest(str(pdf_path), pdf_path.name)
    return IngestResponse(
        started=True,
        message="Demo PDF ingest started. Poll /api/status for progress.",
    )


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_upload(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if not config.OLLAMA_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="OLLAMA_API_KEY is not configured on the server.",
        )

    config.ensure_dirs()
    safe_name = Path(file.filename).name
    dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    dest.write_bytes(content)

    _start_ingest(str(dest), safe_name)
    return IngestResponse(
        started=True,
        message=f"Upload received ({safe_name}). Poll /api/status for progress.",
    )


@app.post("/api/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    if not config.OLLAMA_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="OLLAMA_API_KEY is not configured on the server.",
        )
    with _status_lock:
        busy = _ingest_status["state"] in {
            "parsing",
            "summarising",
            "embedding",
        }
    if busy:
        raise HTTPException(
            status_code=409,
            detail="Document ingest is still running. Wait until status is ready.",
        )
    try:
        result = pipeline.ask_question(body.question.strip())
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Ask failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return AskResponse(
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
        trace=result.get("trace"),
    )


@app.get("/")
def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_path)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
