"""ModalRAG — Gradio UI for Hugging Face Spaces (hybrid RAG + retrieval log)."""

from __future__ import annotations

import inspect
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

# Avoid CUDA init in the parent process (ZeroGPU / numba).
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
# HF Spaces enables Gradio SSR by default; disable for stable Spaces.
os.environ.setdefault("GRADIO_SSR_MODE", "False")

import gradio as gr

try:
    import spaces
except ImportError:  # local runs without the HF spaces package
    class spaces:  # type: ignore[no-redef]
        @staticmethod
        def GPU(*dargs, **dkwargs):
            if dargs and callable(dargs[0]) and not dkwargs:
                return dargs[0]

            def _wrap(fn):
                return fn

            return _wrap

from app import config, pipeline

EXAMPLE_QUESTIONS = [
    "What are the two main components of the Transformer architecture?",
    "How many layers does the base Transformer model use in both encoder and decoder?",
    "What is the formula for Scaled Dot-Product Attention?",
    "How many attention heads does the Transformer use, and what is the dimension of each head?",
]


def _status_ready() -> str:
    if pipeline.chroma_is_ready():
        name = pipeline.get_document_name() or "indexed PDF(s)"
        return f"Ready — {name}"
    return "No document indexed yet. Use demo PDF or upload one."


def _resolve_upload_path(file_obj: Any) -> Optional[Path]:
    if file_obj is None:
        return None
    if isinstance(file_obj, (str, Path)):
        return Path(file_obj)
    if isinstance(file_obj, dict) and file_obj.get("path"):
        return Path(file_obj["path"])
    name = getattr(file_obj, "name", None)
    if name:
        return Path(name)
    return None


@spaces.GPU(duration=120)
def ingest_demo() -> str:
    if not config.OLLAMA_API_KEY:
        return "Error: set OLLAMA_API_KEY as a Space secret (or in local .env)."
    try:
        pdf_path = config.ensure_demo_pdf()
        pipeline.reset_vectorstore_cache()
        pipeline.run_ingestion(
            str(pdf_path),
            document_name=pdf_path.name,
        )
        return _status_ready()
    except Exception as e:
        return f"Ingest failed: {e}"


@spaces.GPU(duration=120)
def ingest_upload(file_obj: Any) -> str:
    if file_obj is None:
        return "Please upload a PDF."
    if not config.OLLAMA_API_KEY:
        return "Error: set OLLAMA_API_KEY as a Space secret (or in local .env)."

    src = _resolve_upload_path(file_obj)
    if src is None or not src.exists():
        return "Could not read the uploaded file."
    if src.suffix.lower() != ".pdf":
        return "Only PDF files are supported."

    try:
        config.ensure_dirs()
        dest = config.UPLOAD_DIR / src.name
        shutil.copy2(src, dest)
        pipeline.reset_vectorstore_cache()
        pipeline.run_ingestion(str(dest), document_name=dest.name)
        return _status_ready()
    except Exception as e:
        return f"Ingest failed: {e}"


def clear_index() -> str:
    try:
        pipeline.clear_index()
        return "Index cleared. Upload or use demo PDF to start again."
    except Exception as e:
        return f"Clear failed: {e}"


@spaces.GPU(duration=120)
def chat(
    message: str,
    history: Optional[List[dict]],
    log_md: Optional[str],
) -> Tuple[Union[List[dict], list], str]:
    history = list(history or [])
    log_md = log_md or ""
    if not message or not str(message).strip():
        return history, log_md

    user_text = str(message).strip()
    if not config.OLLAMA_API_KEY:
        history.append({"role": "user", "content": user_text})
        history.append(
            {
                "role": "assistant",
                "content": "OLLAMA_API_KEY is not configured on the server.",
            }
        )
        return history, log_md

    trace_md = ""
    try:
        result = pipeline.ask_question(user_text)
        sources = result.get("sources") or []
        answer = result.get("answer") or ""
        if sources:
            lines = []
            for s in sources:
                doc = s.get("document_name") or "?"
                lines.append(f"- [{doc}] Chunk {s['index']}: {s['preview']}")
            answer = f"{answer}\n\n**Sources**\n" + "\n".join(lines)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trace_md = f"### Ask @ {stamp}\n\n" + (
            result.get("trace_markdown") or ""
        )
    except Exception as e:
        answer = f"Error: {e}"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trace_md = f"### Ask @ {stamp}\n\n**Error:** `{e}`\n\n---\n"

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})
    new_log = (log_md + "\n" + trace_md).strip() + "\n"
    return history, new_log


def clear_question() -> str:
    return ""


def clear_log() -> str:
    return "_Retrieval log cleared. Ask a question to see multi-query, BM25, vector, RRF, and rerank steps._\n"


with gr.Blocks(title="ModalRAG") as demo:
    gr.Markdown(
        """
# ModalRAG
Multimodal RAG over PDFs — **hybrid retrieval** (multi-query + BM25 + vector + RRF + rerank).
LLM: **Ollama Cloud** · Embeddings: local · Rerank: Cohere with local BGE fallback.
        """
    )
    status = gr.Textbox(
        label="Document status",
        value=_status_ready(),
        interactive=False,
    )

    with gr.Row():
        demo_btn = gr.Button("Use demo PDF", variant="primary")
        upload = gr.File(
            label="Upload PDF",
            file_types=[".pdf"],
            type="filepath",
        )
        clear_btn = gr.Button("Clear index", variant="stop")

    demo_btn.click(ingest_demo, outputs=status)
    upload.upload(ingest_upload, inputs=upload, outputs=status)
    clear_btn.click(clear_index, outputs=status)

    with gr.Tabs():
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(label="Chat", height=440, type="messages")
            question = gr.Textbox(
                label="Question",
                placeholder="Ask about the indexed document(s)…",
                lines=2,
            )
            ask_btn = gr.Button("Ask", variant="primary")
            gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=question)

        with gr.Tab("Retrieval log"):
            retrieval_log = gr.Markdown(
                value=(
                    "_Ask a question in the Chat tab. This log shows multi-query "
                    "variations, BM25 vs vector hits, RRF fusion, and rerank "
                    "(including Cohere → local fallback)._"
                ),
            )
            clear_log_btn = gr.Button("Clear log")
            clear_log_btn.click(clear_log, outputs=retrieval_log)

    ask_btn.click(
        chat,
        inputs=[question, chatbot, retrieval_log],
        outputs=[chatbot, retrieval_log],
    ).then(clear_question, outputs=question)
    question.submit(
        chat,
        inputs=[question, chatbot, retrieval_log],
        outputs=[chatbot, retrieval_log],
    ).then(clear_question, outputs=question)


def _launch_demo() -> None:
    kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7860,
    }
    params = inspect.signature(demo.launch).parameters
    if "ssr_mode" in params:
        kwargs["ssr_mode"] = False
    elif "ssr" in params:
        kwargs["ssr"] = False
    demo.launch(**kwargs)


if __name__ == "__main__":
    _launch_demo()
