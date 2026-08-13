"""ModalRAG — Gradio UI for Hugging Face Spaces."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, List, Optional, Union

import gradio as gr

from app import config, pipeline

EXAMPLE_QUESTIONS = [
    "What are the two main components of the Transformer architecture?",
    "How many layers does the base Transformer model use in both encoder and decoder?",
    "What is the formula for Scaled Dot-Product Attention?",
    "How many attention heads does the Transformer use, and what is the dimension of each head?",
]


def _status_ready() -> str:
    if pipeline.chroma_is_ready():
        name = pipeline.get_document_name() or "indexed PDF"
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


def chat(
    message: str, history: Optional[List[dict]]
) -> Union[List[dict], tuple]:
    history = list(history or [])
    if not message or not str(message).strip():
        return history

    user_text = str(message).strip()
    if not config.OLLAMA_API_KEY:
        history.append({"role": "user", "content": user_text})
        history.append(
            {
                "role": "assistant",
                "content": "OLLAMA_API_KEY is not configured on the server.",
            }
        )
        return history

    try:
        result = pipeline.ask_question(user_text)
        sources = result.get("sources") or []
        answer = result.get("answer") or ""
        if sources:
            lines = [
                f"- Chunk {s['index']}: {s['preview']}" for s in sources
            ]
            answer = f"{answer}\n\n**Sources**\n" + "\n".join(lines)
    except Exception as e:
        answer = f"Error: {e}"

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})
    return history


def clear_question() -> str:
    return ""


with gr.Blocks(title="ModalRAG") as demo:
    gr.Markdown(
        """
# ModalRAG
Multimodal RAG chat over PDFs — grounded answers from text, tables, and figures.
Powered by **Ollama Cloud** + ChromaDB.
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

    demo_btn.click(ingest_demo, outputs=status)
    upload.upload(ingest_upload, inputs=upload, outputs=status)

    chatbot = gr.Chatbot(label="Chat", height=440, type="messages")
    question = gr.Textbox(
        label="Question",
        placeholder="Ask about the indexed document…",
        lines=2,
    )
    ask_btn = gr.Button("Ask", variant="primary")

    gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=question)

    ask_btn.click(chat, inputs=[question, chatbot], outputs=chatbot).then(
        clear_question, outputs=question
    )
    question.submit(chat, inputs=[question, chatbot], outputs=chatbot).then(
        clear_question, outputs=question
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
