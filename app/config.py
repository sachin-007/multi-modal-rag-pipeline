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
    os.getenv("DEMO_PDF_PATH", str(DOCS_DIR / "demo_pdf.pdf"))
)
# Optional: only used when DEMO_PDF_PATH is missing (e.g. fresh HF Space).
DEMO_PDF_URL = (os.getenv("DEMO_PDF_URL") or "").strip()
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

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
PARTITION_STRATEGY = os.getenv("PARTITION_STRATEGY", "hi_res")

# Hybrid retrieval: multi-query + BM25 + vector + RRF + rerank
MULTI_QUERY_N = int(os.getenv("MULTI_QUERY_N", "3"))
BM25_K = int(os.getenv("BM25_K", "10"))
VECTOR_K = int(os.getenv("VECTOR_K", "10"))
RRF_K = int(os.getenv("RRF_K", "60"))
RRF_TOP_N = int(os.getenv("RRF_TOP_N", "30"))
BM25_PATH = Path(os.getenv("BM25_PATH", str(DATA_DIR / "bm25_docs.pkl")))
DOC_INDEX_PATH = Path(
    os.getenv("DOC_INDEX_PATH", str(DATA_DIR / "indexed_documents.json"))
)

RERANK_BACKEND = os.getenv("RERANK_BACKEND", "cohere").strip().lower()
RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    "rerank-english-v3.0"
    if RERANK_BACKEND == "cohere"
    else "BAAI/bge-reranker-base",
).strip()
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", str(RETRIEVAL_K or 5)))
RERANK_API_KEY = (
    os.getenv("RERANK_API_KEY") or os.getenv("COHERE_API_KEY") or ""
).strip()
RERANK_FALLBACK_BACKEND = os.getenv("RERANK_FALLBACK_BACKEND", "local").strip().lower()
RERANK_FALLBACK_MODEL = os.getenv(
    "RERANK_FALLBACK_MODEL", "BAAI/bge-reranker-base"
).strip()
TABLE_PROMPT_MAX_CHARS = int(os.getenv("TABLE_PROMPT_MAX_CHARS", "4000"))


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
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)


def ensure_demo_pdf() -> Path:
    """Return local demo PDF path (`docs/demo_pdf.pdf` by default).

    Downloads only when the file is missing and ``DEMO_PDF_URL`` is set.
    """
    ensure_dirs()
    if DEMO_PDF_PATH.exists() and DEMO_PDF_PATH.stat().st_size > 1000:
        return DEMO_PDF_PATH

    if not DEMO_PDF_URL:
        raise RuntimeError(
            f"Demo PDF not found at {DEMO_PDF_PATH}. "
            "Place docs/demo_pdf.pdf there, or set DEMO_PDF_URL to download it."
        )

    logger.info("Downloading demo PDF from %s", DEMO_PDF_URL)
    request = Request(
        DEMO_PDF_URL,
        headers={"User-Agent": "ModalRAG/1.0 (Hugging Face Space)"},
    )
    with urlopen(request, timeout=120) as response:
        data = response.read()
    if len(data) < 1000 or not data.startswith(b"%PDF"):
        raise RuntimeError("Downloaded demo file does not look like a valid PDF.")
    DEMO_PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEMO_PDF_PATH.write_bytes(data)
    logger.info("Saved demo PDF to %s (%s bytes)", DEMO_PDF_PATH, len(data))
    return DEMO_PDF_PATH
