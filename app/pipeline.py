"""Multimodal RAG pipeline with hybrid retrieval (BM25 + vector + RRF + rerank)."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

from app import config
from app.retrieval import format_trace_markdown, retrieve_with_trace

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str, str, float], None]]

_vectorstore: Optional[Chroma] = None
_document_names: Set[str] = set()
_embeddings: Optional[Embeddings] = None
_bm25_docs: List[Document] = []
_bm25_retriever = None


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=0,
        base_url=config.OLLAMA_BASE_URL,
        api_key=config.require_api_key(),
    )


def get_embeddings() -> Embeddings:
    """Local sentence-transformers by default; Ollama only if EMBED_BACKEND=ollama."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    if config.EMBED_BACKEND == "ollama":
        _embeddings = OpenAIEmbeddings(
            model=config.EMBED_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            api_key=config.require_api_key(),
        )
        return _embeddings

    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("Loading local embeddings: %s", config.EMBED_MODEL)
    _embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return _embeddings


def chroma_is_ready() -> bool:
    persist = Path(config.CHROMA_DIR)
    if not persist.exists():
        return False
    return any(persist.iterdir())


def get_document_name() -> Optional[str]:
    names = get_indexed_document_names()
    if not names:
        return None
    return ", ".join(names)


def get_indexed_document_names() -> List[str]:
    _load_doc_index()
    return sorted(_document_names)


def _load_doc_index() -> None:
    global _document_names
    path = config.DOC_INDEX_PATH
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _document_names = set(data.get("documents") or [])
        except Exception as e:
            logger.warning("Failed to load doc index: %s", e)


def _save_doc_index() -> None:
    config.ensure_dirs()
    config.DOC_INDEX_PATH.write_text(
        json.dumps({"documents": sorted(_document_names)}, indent=2),
        encoding="utf-8",
    )


def _make_chunk_id(document_name: str, raw_text: str) -> str:
    digest = hashlib.sha256(
        f"{document_name}\n{raw_text}".encode("utf-8", errors="ignore")
    ).hexdigest()
    return digest[:24]


def _bm25_text(raw_text: str, tables: List[str]) -> str:
    parts = [raw_text or ""]
    for table in tables or []:
        parts.append(table or "")
    return "\n\n".join(p for p in parts if p).strip()


def _load_bm25_docs() -> None:
    global _bm25_docs, _bm25_retriever
    path = config.BM25_PATH
    if not path.exists():
        _bm25_docs = []
        _bm25_retriever = None
        return
    try:
        with path.open("rb") as f:
            _bm25_docs = pickle.load(f)
        _rebuild_bm25_retriever()
    except Exception as e:
        logger.warning("Failed to load BM25 corpus: %s", e)
        _bm25_docs = []
        _bm25_retriever = None


def _save_bm25_docs() -> None:
    config.ensure_dirs()
    with config.BM25_PATH.open("wb") as f:
        pickle.dump(_bm25_docs, f)


def _rebuild_bm25_retriever() -> None:
    global _bm25_retriever
    if not _bm25_docs:
        _bm25_retriever = None
        return
    from langchain_community.retrievers import BM25Retriever

    # BM25 over raw text + tables stored on each doc as page_content for the retriever
    search_docs = []
    for doc in _bm25_docs:
        search_docs.append(
            Document(
                page_content=doc.metadata.get("bm25_text") or doc.page_content,
                metadata=doc.metadata,
            )
        )
    retriever = BM25Retriever.from_documents(search_docs)
    retriever.k = config.BM25_K
    _bm25_retriever = retriever


def get_bm25_retriever():
    global _bm25_retriever
    if _bm25_retriever is None and config.BM25_PATH.exists():
        _load_bm25_docs()
    return _bm25_retriever


def load_vectorstore() -> Optional[Chroma]:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    if not chroma_is_ready():
        return None
    _vectorstore = Chroma(
        persist_directory=str(config.CHROMA_DIR),
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )
    _load_bm25_docs()
    _load_doc_index()
    return _vectorstore


def reset_vectorstore_cache() -> None:
    global _vectorstore, _bm25_retriever
    _vectorstore = None
    _bm25_retriever = None


def clear_index() -> None:
    """Wipe Chroma + BM25 corpus + document index."""
    global _vectorstore, _bm25_docs, _bm25_retriever, _document_names
    reset_vectorstore_cache()
    persist = Path(config.CHROMA_DIR)
    if persist.exists():
        shutil.rmtree(persist, ignore_errors=True)
    persist.mkdir(parents=True, exist_ok=True)
    if config.BM25_PATH.exists():
        config.BM25_PATH.unlink()
    if config.DOC_INDEX_PATH.exists():
        config.DOC_INDEX_PATH.unlink()
    _bm25_docs = []
    _bm25_retriever = None
    _document_names = set()
    _vectorstore = None


def _delete_document_from_stores(document_name: str) -> None:
    """Remove existing chunks for document_name from Chroma and BM25 (replace-on-reingest)."""
    global _bm25_docs, _vectorstore
    vs = load_vectorstore()
    if vs is not None:
        try:
            data = vs.get(where={"document_name": document_name})
            ids = data.get("ids") or []
            if ids:
                vs.delete(ids=ids)
                logger.info(
                    "Removed %s existing Chroma chunks for %s",
                    len(ids),
                    document_name,
                )
        except Exception as e:
            logger.warning("Chroma delete for %s failed: %s", document_name, e)

    before = len(_bm25_docs)
    _bm25_docs = [
        d for d in _bm25_docs if d.metadata.get("document_name") != document_name
    ]
    if len(_bm25_docs) != before:
        _save_bm25_docs()
        _rebuild_bm25_retriever()


def partition_document(file_path: str, on_progress: ProgressCallback = None):
    if on_progress:
        on_progress("parsing", f"Partitioning {Path(file_path).name}", 0.1)
    logger.info("Partitioning document: %s", file_path)
    elements = partition_pdf(
        filename=file_path,
        strategy=config.PARTITION_STRATEGY,
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )
    logger.info("Extracted %s elements", len(elements))
    return elements


def create_chunks_by_title(elements, on_progress: ProgressCallback = None):
    if on_progress:
        on_progress("parsing", "Creating title-based chunks", 0.25)
    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500,
    )
    logger.info("Created %s chunks", len(chunks))
    return chunks


def separate_content_types(chunk) -> Dict[str, Any]:
    content_data: Dict[str, Any] = {
        "text": chunk.text,
        "tables": [],
        "images": [],
        "types": ["text"],
    }

    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            if element_type == "Table":
                content_data["types"].append("table")
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)
            elif element_type == "Image":
                if hasattr(element, "metadata") and hasattr(
                    element.metadata, "image_base64"
                ):
                    content_data["types"].append("image")
                    content_data["images"].append(element.metadata.image_base64)

    content_data["types"] = list(set(content_data["types"]))
    return content_data


def _build_text_message(prompt_text: str) -> HumanMessage:
    """gpt-oss on Ollama Cloud is text-only — never attach image_url parts."""
    return HumanMessage(content=prompt_text)


def _chunk_text_and_tables(chunk: Document) -> tuple[str, List[str], int]:
    """Return (text, tables_html, image_count) from a stored chunk."""
    if "original_content" not in chunk.metadata:
        return chunk.page_content, [], 0
    try:
        original = json.loads(chunk.metadata["original_content"])
    except json.JSONDecodeError:
        return chunk.page_content, [], 0
    text = original.get("raw_text") or chunk.page_content or ""
    tables = original.get("tables_html") or []
    images = original.get("images_base64") or []
    return text, tables, len(images)


def create_ai_enhanced_summary(
    text: str, tables: List[str], images: List[str]
) -> str:
    try:
        llm = get_llm()
        prompt_text = f"""You are creating a searchable description for document content retrieval.

CONTENT TO ANALYZE:
TEXT CONTENT:
{text}

"""
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i + 1}:\n{table}\n\n"

        if images:
            prompt_text += (
                f"FIGURES: {len(images)} image(s) are present in this chunk "
                "(pixels are not sent to the text model; note that figures exist).\n\n"
            )

        prompt_text += """
YOUR TASK:
Generate a comprehensive, searchable description that covers:

1. Key facts, numbers, and data points from text and tables
2. Main topics and concepts discussed
3. Questions this content could answer
4. Mentions of figures/diagrams when noted above
5. Alternative search terms users might use

Make it detailed and searchable - prioritize findability over brevity.

SEARCHABLE DESCRIPTION:"""

        response = llm.invoke([_build_text_message(prompt_text)])
        return response.content
    except Exception as e:
        logger.warning("AI summary failed: %s", e)
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary


def summarise_chunks(
    chunks,
    document_name: str,
    on_progress: ProgressCallback = None,
) -> List[Document]:
    langchain_documents: List[Document] = []
    total = len(chunks) or 1

    for i, chunk in enumerate(chunks):
        if on_progress:
            frac = 0.3 + 0.45 * ((i + 1) / total)
            on_progress(
                "summarising",
                f"Summarising chunk {i + 1}/{len(chunks)}",
                frac,
            )

        content_data = separate_content_types(chunk)
        if content_data["tables"] or content_data["images"]:
            enhanced_content = create_ai_enhanced_summary(
                content_data["text"],
                content_data["tables"],
                content_data["images"],
            )
        else:
            enhanced_content = content_data["text"]

        raw_text = content_data["text"] or ""
        tables = content_data["tables"] or []
        chunk_id = _make_chunk_id(document_name, raw_text)
        bm25_body = _bm25_text(raw_text, tables)

        doc = Document(
            page_content=enhanced_content,
            metadata={
                "document_name": document_name,
                "source": document_name,
                "chunk_id": chunk_id,
                "bm25_text": bm25_body,
                "original_content": json.dumps(
                    {
                        "raw_text": raw_text,
                        "tables_html": tables,
                        "images_base64": content_data["images"],
                    }
                ),
            },
        )
        langchain_documents.append(doc)

    return langchain_documents


def _ensure_vectorstore() -> Chroma:
    global _vectorstore
    vs = load_vectorstore()
    if vs is not None:
        return vs
    config.ensure_dirs()
    Path(config.CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    _vectorstore = Chroma(
        persist_directory=str(config.CHROMA_DIR),
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )
    return _vectorstore


def append_documents_to_stores(
    documents: List[Document],
    on_progress: ProgressCallback = None,
) -> Chroma:
    """Append chunks to Chroma + BM25 without wiping the whole index."""
    global _bm25_docs

    if on_progress:
        on_progress("embedding", "Creating embeddings and updating stores", 0.85)

    vs = _ensure_vectorstore()
    ids = [d.metadata["chunk_id"] for d in documents]
    vs.add_documents(documents=documents, ids=ids)

    for doc in documents:
        _bm25_docs.append(doc)
    _save_bm25_docs()
    _rebuild_bm25_retriever()
    return vs


def run_ingestion(
    pdf_path: str,
    document_name: Optional[str] = None,
    on_progress: ProgressCallback = None,
) -> Chroma:
    global _document_names

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    config.ensure_dirs()
    _load_bm25_docs()
    _load_doc_index()

    name = document_name or path.name
    # Replace-on-reingest for the same PDF name
    if name in _document_names or chroma_is_ready():
        _delete_document_from_stores(name)

    elements = partition_document(str(path), on_progress=on_progress)
    chunks = create_chunks_by_title(elements, on_progress=on_progress)
    summarised = summarise_chunks(chunks, document_name=name, on_progress=on_progress)
    db = append_documents_to_stores(summarised, on_progress=on_progress)

    _document_names.add(name)
    _save_doc_index()

    if on_progress:
        on_progress("ready", f"Indexed {get_document_name()}", 1.0)
    return db


def _source_preview(chunk: Document, index: int) -> Dict[str, Any]:
    preview = chunk.page_content[:280].replace("\n", " ").strip()
    has_tables = False
    has_images = False
    document_name = chunk.metadata.get("document_name") or ""
    if "original_content" in chunk.metadata:
        try:
            original = json.loads(chunk.metadata["original_content"])
            has_tables = bool(original.get("tables_html"))
            has_images = bool(original.get("images_base64"))
            raw = (original.get("raw_text") or "").replace("\n", " ").strip()
            if raw:
                preview = raw[:280]
        except json.JSONDecodeError:
            pass
    if len(chunk.page_content) > 280 or len(preview) >= 280:
        preview = preview.rstrip() + "…"
    return {
        "index": index,
        "preview": preview,
        "has_tables": has_tables,
        "has_images": has_images,
        "document_name": document_name,
        "chunk_id": chunk.metadata.get("chunk_id") or "",
    }


def generate_final_answer(chunks: List[Document], query: str) -> str:
    try:
        llm = get_llm()
        prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""
        max_table = config.TABLE_PROMPT_MAX_CHARS
        for i, chunk in enumerate(chunks):
            doc_name = chunk.metadata.get("document_name") or "unknown"
            prompt_text += f"--- Document {i + 1} ({doc_name}) ---\n"
            raw_text, tables_html, image_count = _chunk_text_and_tables(chunk)
            if raw_text:
                prompt_text += f"TEXT:\n{raw_text}\n\n"
            if tables_html:
                prompt_text += "TABLES:\n"
                for j, table in enumerate(tables_html):
                    clipped = table
                    if len(clipped) > max_table:
                        clipped = clipped[:max_table] + "\n…[table truncated]"
                    prompt_text += f"Table {j + 1}:\n{clipped}\n\n"
            if image_count:
                prompt_text += (
                    f"[{image_count} figure(s) present in source; "
                    "not sent to text model]\n\n"
                )
            prompt_text += "\n"

        prompt_text += """
Please provide a clear, comprehensive answer using the text and tables above.
If figures are noted but not shown, answer from the available text/tables only.
If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

ANSWER:"""

        response = llm.invoke([_build_text_message(prompt_text)])
        return response.content
    except Exception as e:
        logger.exception("Answer generation failed: %s", e)
        return f"Answer generation failed: {e}"


def ask_question(question: str) -> Dict[str, Any]:
    db = load_vectorstore()
    if db is None:
        raise RuntimeError(
            "No document is indexed yet. Ingest the demo PDF or upload a file first."
        )

    bm25 = get_bm25_retriever()
    chunks, trace = retrieve_with_trace(
        question,
        vectorstore=db,
        bm25_retriever=bm25,
        llm=get_llm(),
    )

    if not chunks:
        answer = (
            "I don't have enough information to answer that question based on "
            "the provided documents."
        )
        return {
            "answer": answer,
            "sources": [],
            "trace": trace,
            "trace_markdown": format_trace_markdown(trace),
        }

    answer = generate_final_answer(chunks, question)
    sources = [_source_preview(chunk, i + 1) for i, chunk in enumerate(chunks)]
    return {
        "answer": answer,
        "sources": sources,
        "trace": trace,
        "trace_markdown": format_trace_markdown(trace),
    }
