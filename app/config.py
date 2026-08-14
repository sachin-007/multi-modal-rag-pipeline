"""Application configuration from environment variables."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(DATA_DIR / "chroma_db")))
DEMO_PDF_PATH = Path(
    os.getenv("DEMO_PDF_PATH", str(DOCS_DIR / "attention-is-all-you-need.pdf"))
)
DEMO_PDF_URL = os.getenv(
    "DEMO_PDF_URL", "https://arxiv.org/pdf/1706.03762.pdf"
)
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(DATA_DIR / "uploads")))

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
OLLAMA_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://ollama.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss:120b")
# Ollama Cloud has chat models only — no /v1/embeddings. Default to local ST.
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "local").strip().lower()
_DEFAULT_LOCAL_EMBED = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_OLLAMA_EMBED = "nomic-embed-text"
_embed_model_env = (os.getenv("EMBED_MODEL") or "").strip()
if EMBED_BACKEND == "ollama":
    EMBED_MODEL = _embed_model_env or _DEFAULT_OLLAMA_EMBED
elif _embed_model_env in {"", _DEFAULT_OLLAMA_EMBED, f"{_DEFAULT_OLLAMA_EMBED}:latest"}:
    # Ignore leftover Cloud-era EMBED_MODEL secrets that are not HF model ids.
    EMBED_MODEL = _DEFAULT_LOCAL_EMBED
else:
    EMBED_MODEL = _embed_model_env

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
PARTITION_STRATEGY = os.getenv("PARTITION_STRATEGY", "hi_res")


def require_api_key() -> str:
    if not OLLAMA_API_KEY:
        raise RuntimeError(
            "OLLAMA_API_KEY is not set. Add it to .env locally or as a "
            "Hugging Face Space secret."
        )
    return OLLAMA_API_KEY


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_demo_pdf() -> Path:
    """Return local demo PDF path, downloading from arXiv if missing."""
    ensure_dirs()
    if DEMO_PDF_PATH.exists() and DEMO_PDF_PATH.stat().st_size > 1000:
        return DEMO_PDF_PATH

    logger.info("Downloading demo PDF from %s", DEMO_PDF_URL)
    request = Request(
        DEMO_PDF_URL,
        headers={"User-Agent": "ModalRAG/1.0 (Hugging Face Space)"},
    )
    with urlopen(request, timeout=120) as response:
        data = response.read()
    if len(data) < 1000 or not data.startswith(b"%PDF"):
        raise RuntimeError("Downloaded demo file does not look like a valid PDF.")
    DEMO_PDF_PATH.write_bytes(data)
    logger.info("Saved demo PDF to %s (%s bytes)", DEMO_PDF_PATH, len(data))
    return DEMO_PDF_PATH
