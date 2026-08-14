"""Multimodal RAG pipeline extracted from multi_modal_rag.ipynb."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

from app import config

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str, str, float], None]]

_vectorstore: Optional[Chroma] = None
_document_name: Optional[str] = None
_embeddings: Optional[Embeddings] = None


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
    return _document_name


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
    return _vectorstore


def reset_vectorstore_cache() -> None:
    global _vectorstore
    _vectorstore = None


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


def summarise_chunks(chunks, on_progress: ProgressCallback = None) -> List[Document]:
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

        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps(
                    {
                        "raw_text": content_data["text"],
                        "tables_html": content_data["tables"],
                        "images_base64": content_data["images"],
                    }
                )
            },
        )
        langchain_documents.append(doc)

    return langchain_documents


def create_vector_store(
    documents: List[Document],
    persist_directory: Optional[str] = None,
    on_progress: ProgressCallback = None,
) -> Chroma:
    global _vectorstore

    persist = Path(persist_directory or config.CHROMA_DIR)
    if on_progress:
        on_progress("embedding", "Creating embeddings and vector store", 0.85)

    if persist.exists():
        shutil.rmtree(persist, ignore_errors=True)
    persist.mkdir(parents=True, exist_ok=True)

    embedding_model = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(persist),
        collection_metadata={"hnsw:space": "cosine"},
    )
    _vectorstore = vectorstore
    return vectorstore


def run_ingestion(
    pdf_path: str,
    document_name: Optional[str] = None,
    on_progress: ProgressCallback = None,
) -> Chroma:
    global _document_name

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    config.ensure_dirs()
    elements = partition_document(str(path), on_progress=on_progress)
    chunks = create_chunks_by_title(elements, on_progress=on_progress)
    summarised = summarise_chunks(chunks, on_progress=on_progress)
    db = create_vector_store(summarised, on_progress=on_progress)

    _document_name = document_name or path.name
    if on_progress:
        on_progress("ready", f"Indexed {_document_name}", 1.0)
    return db


def _source_preview(chunk: Document, index: int) -> Dict[str, Any]:
    preview = chunk.page_content[:280].replace("\n", " ").strip()
    has_tables = False
    has_images = False
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
    }


def generate_final_answer(chunks: List[Document], query: str) -> str:
    try:
        llm = get_llm()
        prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i + 1} ---\n"
            raw_text, tables_html, image_count = _chunk_text_and_tables(chunk)
            if raw_text:
                prompt_text += f"TEXT:\n{raw_text}\n\n"
            if tables_html:
                prompt_text += "TABLES:\n"
                for j, table in enumerate(tables_html):
                    prompt_text += f"Table {j + 1}:\n{table}\n\n"
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

    retriever = db.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
    chunks = retriever.invoke(question)
    answer = generate_final_answer(chunks, question)
    sources = [_source_preview(chunk, i + 1) for i, chunk in enumerate(chunks)]
    return {"answer": answer, "sources": sources}
