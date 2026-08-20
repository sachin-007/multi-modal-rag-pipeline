"""ModalRAG — Gradio UI for Hugging Face Spaces (dashboard-style hybrid RAG)."""

from __future__ import annotations

import html
import inspect
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    "What are the MODEL VARIANTS?",
    "What is ColPali and how does it work?",
    "What datasets or benchmarks are used for evaluation?",
    "What are the main results or conclusions of the paper?",
]


def _esc(text: Any) -> str:
    return html.escape(str(text or ""), quote=True)


def _status_ready() -> str:
    if pipeline.chroma_is_ready():
        name = pipeline.get_document_name() or "indexed PDF(s)"
        return f"Ready — {name}"
    return "No document indexed yet. Use demo PDF or upload one."


def _docs_panel_html() -> str:
    names = pipeline.get_indexed_document_names()
    if not names:
        body = '<p class="mr-muted">No documents yet.</p>'
    else:
        items = []
        for n in names:
            items.append(
                f'<div class="mr-doc"><span class="mr-doc-name">{_esc(n)}</span>'
                f'<span class="mr-pill ready">Ready</span></div>'
            )
        body = "".join(items)
    return f'<div class="mr-section"><div class="mr-section-title">Documents</div>{body}</div>'


def _system_panel_html() -> str:
    llm = _esc(config.LLM_MODEL)
    embed = _esc(config.EMBED_MODEL.split("/")[-1])
    rerank = _esc(config.RERANK_BACKEND)
    key_ok = "Connected" if config.OLLAMA_API_KEY else "Missing key"
    key_cls = "ready" if config.OLLAMA_API_KEY else "warn"
    idx = "Ready" if pipeline.chroma_is_ready() else "Empty"
    idx_cls = "ready" if pipeline.chroma_is_ready() else "warn"
    return f"""
<div class="mr-section">
  <div class="mr-section-title">System status</div>
  <div class="mr-sys"><span>LLM (Ollama)</span><span class="mr-pill {key_cls}">{key_ok}</span></div>
  <div class="mr-sys"><span>Model</span><span class="mr-muted">{llm}</span></div>
  <div class="mr-sys"><span>Embeddings</span><span class="mr-muted">{embed}</span></div>
  <div class="mr-sys"><span>Reranker</span><span class="mr-muted">{rerank}</span></div>
  <div class="mr-sys"><span>Index</span><span class="mr-pill {idx_cls}">{idx}</span></div>
</div>
"""


def _banner_html(status_text: Optional[str] = None) -> str:
    text = status_text or _status_ready()
    ready = pipeline.chroma_is_ready() and not str(text).lower().startswith(
        ("error", "ingest failed", "clear failed", "please", "could not", "only pdf")
    )
    pill = "Ready" if ready else "Idle"
    pill_cls = "ready" if ready else "warn"
    names = pipeline.get_indexed_document_names()
    title = names[0] if len(names) == 1 else (
        f"{len(names)} documents" if names else "No document indexed"
    )
    return f"""
<div class="mr-banner">
  <div class="mr-banner-top">
    <div class="mr-pdf-icon">PDF</div>
    <div>
      <div class="mr-banner-title">{_esc(title)}</div>
      <div class="mr-muted">{_esc(text)}</div>
    </div>
    <span class="mr-pill {pill_cls}">{pill}</span>
  </div>
</div>
"""


def _sources_html(sources: Optional[List[Dict[str, Any]]] = None) -> str:
    sources = sources or []
    if not sources:
        return """
<div class="mr-section">
  <div class="mr-section-title">Sources</div>
  <p class="mr-muted">Ask a question to see retrieved chunks here.</p>
</div>
"""
    cards = []
    for s in sources:
        score = s.get("score")
        score_txt = f"{float(score):.2f}" if isinstance(score, (int, float)) else "—"
        cards.append(
            f"""
<div class="mr-source">
  <div class="mr-source-head">
    <span class="mr-source-doc">{_esc(s.get('document_name') or 'chunk')}</span>
    <span class="mr-pill ready">#{_esc(s.get('index'))}</span>
  </div>
  <div class="mr-muted mr-preview">{_esc(s.get('preview') or '')}</div>
  <div class="mr-source-meta">tables={_esc(s.get('has_tables'))} · images={_esc(s.get('has_images'))} · score={_esc(score_txt)}</div>
</div>
"""
        )
    return (
        '<div class="mr-section"><div class="mr-section-title">Sources</div>'
        + "".join(cards)
        + "</div>"
    )


def _pipeline_html(trace: Optional[Dict[str, Any]] = None) -> str:
    trace = trace or {}
    if not trace.get("original_query") and not trace.get("query_variations"):
        return """
<div class="mr-section">
  <div class="mr-section-title">Retrieval pipeline</div>
  <p class="mr-muted">Pipeline steps appear after each ask.</p>
</div>
"""
    variations = trace.get("query_variations") or []
    n_var = max(0, len(variations) - 1)
    per_q = trace.get("per_query") or []
    bm25_n = sum(len(b.get("bm25") or []) for b in per_q)
    vec_n = sum(len(b.get("vector") or []) for b in per_q)
    rrf_n = len(trace.get("rrf") or [])
    rerank = trace.get("rerank") or {}
    kept = trace.get("retrieved") or 0
    timings = trace.get("timings_ms") or {}
    backend = rerank.get("backend_used") or rerank.get("backend_requested") or "—"
    fallback = "yes" if rerank.get("fallback") else "no"

    steps = [
        ("Query", "1", f"{_esc((trace.get('original_query') or '')[:80])}"),
        ("Query expansion", str(1 + n_var), f"+{n_var} paraphrases · {timings.get('multi_query_ms', '—')} ms"),
        ("BM25 retrieval", str(bm25_n), f"hits across queries · {timings.get('search_ms', '—')} ms search"),
        ("Dense vectors", str(vec_n), "semantic hits"),
        ("RRF fusion", str(rrf_n), f"fused candidates · {timings.get('rrf_ms', '—')} ms"),
        ("Rerank", str(kept), f"{_esc(backend)} · fallback={fallback} · {timings.get('rerank_ms', '—')} ms"),
        ("Final context", str(kept), "chunks sent to LLM"),
    ]
    items = []
    for title, badge, detail in steps:
        items.append(
            f"""
<div class="mr-step">
  <div class="mr-step-dot"></div>
  <div>
    <div class="mr-step-title">{title} <span class="mr-pill ready">{badge}</span></div>
    <div class="mr-muted">{detail}</div>
  </div>
</div>
"""
        )
    return (
        '<div class="mr-section"><div class="mr-section-title">Retrieval pipeline</div>'
        '<div class="mr-timeline">'
        + "".join(items)
        + "</div></div>"
    )


def _topbar_html() -> str:
    return """
<div class="mr-topbar">
  <div class="mr-brand">
    <div class="mr-logo">M</div>
    <div>
      <div class="mr-brand-name">ModalRAG</div>
      <div class="mr-muted">Multimodal Document Intelligence</div>
    </div>
  </div>
  <div class="mr-pill ready mr-pulse">All systems operational</div>
  <div class="mr-muted">Ollama Cloud · local embed · hybrid retrieval</div>
</div>
"""


def _footer_html() -> str:
    n_docs = len(pipeline.get_indexed_document_names())
    return f"""
<div class="mr-footer">
  <span>Documents <b>{n_docs}</b></span>
  <span>Rerank <b>{_esc(config.RERANK_BACKEND)}</b></span>
  <span>Embed <b>{_esc(config.EMBED_BACKEND)}</b></span>
  <span>ModalRAG v1.0</span>
</div>
"""


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


def _refresh_shell(status_text: str) -> Tuple[str, str, str, str]:
    return (
        _banner_html(status_text),
        _docs_panel_html(),
        _system_panel_html(),
        _footer_html(),
    )


@spaces.GPU(duration=120)
def ingest_demo() -> Tuple[str, str, str, str]:
    if not config.OLLAMA_API_KEY:
        msg = "Error: set OLLAMA_API_KEY as a Space secret (or in local .env)."
        return _refresh_shell(msg)
    try:
        pdf_path = config.ensure_demo_pdf()
        pipeline.reset_vectorstore_cache()
        pipeline.run_ingestion(
            str(pdf_path),
            document_name=pdf_path.name,
        )
        return _refresh_shell(_status_ready())
    except Exception as e:
        return _refresh_shell(f"Ingest failed: {e}")


@spaces.GPU(duration=120)
def ingest_upload(file_obj: Any) -> Tuple[str, str, str, str]:
    if file_obj is None:
        return _refresh_shell("Please upload a PDF.")
    if not config.OLLAMA_API_KEY:
        return _refresh_shell(
            "Error: set OLLAMA_API_KEY as a Space secret (or in local .env)."
        )

    src = _resolve_upload_path(file_obj)
    if src is None or not src.exists():
        return _refresh_shell("Could not read the uploaded file.")
    if src.suffix.lower() != ".pdf":
        return _refresh_shell("Only PDF files are supported.")

    try:
        config.ensure_dirs()
        dest = config.UPLOAD_DIR / src.name
        shutil.copy2(src, dest)
        pipeline.reset_vectorstore_cache()
        pipeline.run_ingestion(str(dest), document_name=dest.name)
        return _refresh_shell(_status_ready())
    except Exception as e:
        return _refresh_shell(f"Ingest failed: {e}")


def clear_index() -> Tuple[str, str, str, str, str, str]:
    try:
        pipeline.clear_index()
        msg = "Index cleared. Upload or use demo PDF to start again."
    except Exception as e:
        msg = f"Clear failed: {e}"
    banner, docs, system, footer = _refresh_shell(msg)
    return (
        banner,
        docs,
        system,
        footer,
        _sources_html([]),
        _pipeline_html({}),
    )


@spaces.GPU(duration=120)
def chat(
    message: str,
    history: Optional[List[dict]],
    log_md: Optional[str],
) -> Tuple[Union[List[dict], list], str, str, str]:
    history = list(history or [])
    log_md = log_md or ""
    empty_src = _sources_html([])
    empty_pipe = _pipeline_html({})

    if not message or not str(message).strip():
        return history, log_md, empty_src, empty_pipe

    user_text = str(message).strip()
    if not config.OLLAMA_API_KEY:
        history.append({"role": "user", "content": user_text})
        history.append(
            {
                "role": "assistant",
                "content": "OLLAMA_API_KEY is not configured on the server.",
            }
        )
        return history, log_md, empty_src, empty_pipe

    sources_html = empty_src
    pipeline_html = empty_pipe
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
        trace_md = f"### Ask @ {stamp}\n\n" + (result.get("trace_markdown") or "")
        sources_html = _sources_html(sources)
        pipeline_html = _pipeline_html(result.get("trace") or {})
    except Exception as e:
        answer = f"Error: {e}"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trace_md = f"### Ask @ {stamp}\n\n**Error:** `{e}`\n\n---\n"

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})
    new_log = (log_md + "\n" + trace_md).strip() + "\n"
    return history, new_log, sources_html, pipeline_html


def clear_question() -> str:
    return ""


def clear_log() -> str:
    return "_Retrieval log cleared. Ask a question to see multi-query, BM25, vector, RRF, and rerank steps._\n"


_APP_CSS = """
.gradio-container {
  max-width: 1440px !important;
  margin: 0 auto !important;
  font-family: Inter, ui-sans-serif, system-ui, Segoe UI, sans-serif !important;
}
body, .gradio-container, .main {
  background: #0b1220 !important;
  color: #e8eef9 !important;
}
footer { display: none !important; }

.mr-topbar {
  display: flex; align-items: center; justify-content: space-between; gap: 1rem;
  padding: 0.85rem 1rem; margin-bottom: 0.75rem;
  background: #111a2b; border: 1px solid #243147; border-radius: 14px;
}
.mr-brand { display: flex; align-items: center; gap: 0.75rem; }
.mr-logo {
  width: 36px; height: 36px; border-radius: 10px;
  background: linear-gradient(135deg, #ff7a18, #ff5a00);
  display:flex; align-items:center; justify-content:center; font-weight:800; color:white;
}
.mr-brand-name { font-weight: 700; font-size: 1.1rem; color: #fff; }
.mr-muted { color: #93a0b8; font-size: 0.82rem; line-height: 1.35; }
.mr-pill {
  display: inline-flex; align-items: center; gap: 0.35rem;
  padding: 0.15rem 0.55rem; border-radius: 999px; font-size: 0.72rem; font-weight: 600;
}
.mr-pill.ready { background: rgba(34,197,94,0.15); color: #4ade80; }
.mr-pill.warn { background: rgba(251,146,60,0.15); color: #fb923c; }
.mr-pulse::before {
  content: ""; width: 7px; height: 7px; border-radius: 50%;
  background: #4ade80; box-shadow: 0 0 0 0 rgba(74,222,128,0.7);
  animation: mrpulse 1.6s infinite;
}
@keyframes mrpulse {
  0% { box-shadow: 0 0 0 0 rgba(74,222,128,0.55); }
  70% { box-shadow: 0 0 0 8px rgba(74,222,128,0); }
  100% { box-shadow: 0 0 0 0 rgba(74,222,128,0); }
}

.mr-section {
  background: #111a2b; border: 1px solid #243147; border-radius: 14px;
  padding: 0.85rem; margin-bottom: 0.75rem;
}
.mr-section-title {
  font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: #9fb0cc; margin-bottom: 0.65rem; font-weight: 700;
}
.mr-doc, .mr-sys {
  display: flex; justify-content: space-between; align-items: center; gap: 0.5rem;
  padding: 0.45rem 0.5rem; border-radius: 10px; margin-bottom: 0.35rem;
  background: #0d1524; border: 1px solid #1e2b42;
}
.mr-doc-name { font-size: 0.85rem; color: #e8eef9; word-break: break-all; }

.mr-banner {
  background: #111a2b; border: 1px solid #243147; border-radius: 14px;
  padding: 0.9rem 1rem; margin-bottom: 0.75rem;
}
.mr-banner-top { display: flex; align-items: center; gap: 0.85rem; }
.mr-pdf-icon {
  width: 42px; height: 42px; border-radius: 10px; background: #b91c1c;
  display:flex; align-items:center; justify-content:center; font-weight:800; font-size: 0.75rem;
}
.mr-banner-title { font-weight: 700; color: #fff; margin-bottom: 0.15rem; }

.mr-source {
  background: #0d1524; border: 1px solid #1e2b42; border-radius: 12px;
  padding: 0.65rem; margin-bottom: 0.5rem;
}
.mr-source-head { display:flex; justify-content: space-between; gap: 0.5rem; margin-bottom: 0.35rem; }
.mr-source-doc { font-weight: 600; font-size: 0.85rem; color: #fff; }
.mr-preview { max-height: 4.2em; overflow: hidden; }
.mr-source-meta { margin-top: 0.35rem; font-size: 0.72rem; color: #7f8da6; }

.mr-timeline { display: flex; flex-direction: column; gap: 0.55rem; }
.mr-step { display: flex; gap: 0.65rem; align-items: flex-start; }
.mr-step-dot {
  width: 10px; height: 10px; margin-top: 0.35rem; border-radius: 50%;
  background: #ff7a18; box-shadow: 0 0 0 3px rgba(255,122,24,0.2); flex-shrink: 0;
}
.mr-step-title { color: #fff; font-weight: 600; font-size: 0.88rem; margin-bottom: 0.15rem; }

.mr-footer {
  display: flex; flex-wrap: wrap; gap: 1rem; justify-content: space-between;
  padding: 0.65rem 0.9rem; margin-top: 0.5rem;
  background: #111a2b; border: 1px solid #243147; border-radius: 12px;
  color: #93a0b8; font-size: 0.8rem;
}
.mr-footer b { color: #e8eef9; font-weight: 600; }

#left-col, #right-col, #center-col {
  background: transparent !important;
}
button.primary, .primary {
  background: #ff7a18 !important;
  border-color: #ff7a18 !important;
}
button.stop, .stop {
  background: #7f1d1d !important;
  border-color: #7f1d1d !important;
}
"""


with gr.Blocks(
    title="ModalRAG",
    css=_APP_CSS,
    theme=gr.themes.Base(
        primary_hue="orange",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
) as demo:
    gr.HTML(_topbar_html())

    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=220, elem_id="left-col"):
            demo_btn = gr.Button("+ Use demo PDF", variant="primary")
            upload = gr.File(
                label="Add document",
                file_types=[".pdf"],
                type="filepath",
                height=110,
            )
            clear_btn = gr.Button("Clear index", variant="stop")
            docs_html = gr.HTML(value=_docs_panel_html())
            system_html = gr.HTML(value=_system_panel_html())

        with gr.Column(scale=5, min_width=420, elem_id="center-col"):
            banner = gr.HTML(value=_banner_html())
            chatbot = gr.Chatbot(
                label="Chat",
                height=420,
                type="messages",
                show_copy_button=True,
            )
            with gr.Row():
                question = gr.Textbox(
                    placeholder="Ask anything about your documents…",
                    lines=2,
                    scale=5,
                    show_label=False,
                    container=False,
                )
                ask_btn = gr.Button("Ask", variant="primary", scale=1)
            gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=question, label="Quick actions")

        with gr.Column(scale=3, min_width=280, elem_id="right-col"):
            sources_panel = gr.HTML(value=_sources_html([]))
            pipeline_panel = gr.HTML(value=_pipeline_html({}))

    footer = gr.HTML(value=_footer_html())

    with gr.Accordion("Full retrieval log (debug)", open=False):
        retrieval_log = gr.Markdown(
            value=(
                "_Ask a question to populate multi-query, BM25 vs vector, RRF, and rerank details._"
            ),
        )
        clear_log_btn = gr.Button("Clear log", size="sm")
        clear_log_btn.click(clear_log, outputs=retrieval_log)

    shell_outs = [banner, docs_html, system_html, footer]
    demo_btn.click(ingest_demo, outputs=shell_outs)
    upload.upload(ingest_upload, inputs=upload, outputs=shell_outs)
    clear_btn.click(
        clear_index,
        outputs=[banner, docs_html, system_html, footer, sources_panel, pipeline_panel],
    )

    ask_btn.click(
        chat,
        inputs=[question, chatbot, retrieval_log],
        outputs=[chatbot, retrieval_log, sources_panel, pipeline_panel],
    ).then(clear_question, outputs=question)
    question.submit(
        chat,
        inputs=[question, chatbot, retrieval_log],
        outputs=[chatbot, retrieval_log, sources_panel, pipeline_panel],
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
