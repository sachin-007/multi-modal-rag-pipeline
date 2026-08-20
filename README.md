---
title: ModalRAG
emoji: 📄
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.22.0
app_file: app.py
pinned: false
license: mit
short_description: Hybrid multimodal RAG over PDFs with retrieval log
---

# ModalRAG — Multimodal hybrid RAG (Gradio)

Ask questions over one or more PDFs (text, tables, and figures) using a Gradio UI, ChromaDB, **hybrid retrieval** (multi-query + BM25 + dense vectors + RRF + rerank), **Ollama Cloud** for the LLM, and **local sentence-transformers** for embeddings.

This is **enterprise-shaped retrieval** (hybrid + RRF + rerank + traces), not full enterprise (no tenant ACL, eval harness, or OpenSearch).

Figures are extracted and stored in chunk metadata; answers use **text + tables only** because the default Cloud chat model (`gpt-oss:120b`) is text-only.

The one-click demo uses `docs/demo_pdf.pdf` (local file). Optional `DEMO_PDF_URL` downloads it only if that path is missing.

## Features

- Gradio **Chat** + **Retrieval log** tabs (multi-query, BM25 vs vector, RRF, rerank)
- Multi-PDF index (append; re-ingest same file replaces; **Clear index**)
- Hybrid: BM25 on raw text/tables + vector on summaries → RRF → Cohere rerank (local BGE fallback)
- One-click demo ingest / PDF upload
- Answers with per-document source previews
- Free deploy on Hugging Face Spaces

## Space setup

1. Create a **Gradio** Space — **CPU basic** is enough (LLM / Cohere are remote).
2. Secrets:
   - `OLLAMA_API_KEY` (required)
   - `RERANK_API_KEY` (optional Cohere key; falls back to local BGE if missing/429/billing)
3. Optional variables:
   - `GRADIO_SSR_MODE=False`
   - `RERANK_BACKEND=cohere` (or `local`)
   - `RERANK_MODEL=rerank-english-v3.0`
   - `EMBED_BACKEND=local`
   - `LLM_MODEL=gpt-oss:120b`
4. Push this repo to the Space

## Local run

Full post-clone guide: [`setup.md`](setup.md).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
# set OLLAMA_API_KEY (and optional RERANK_API_KEY) in .env
# Windows: Poppler + Tesseract on PATH
python app.py
```

## Project layout

- `app.py` — Gradio entry (Chat + Retrieval log)
- `app/pipeline.py` — ingest + ask
- `app/retrieval.py` — multi-query, BM25, vector, RRF, rerank
- `packages.txt` — Poppler/Tesseract for Spaces
- `requirements.txt` — Python deps
