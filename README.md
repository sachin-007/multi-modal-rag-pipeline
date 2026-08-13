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
short_description: Multimodal RAG chat over PDFs with Ollama Cloud
---

# ModalRAG — Multimodal RAG (Gradio)

Ask questions over PDF documents (text, tables, and figures) using a Gradio chat UI, ChromaDB, **Ollama Cloud** for the LLM, and **local sentence-transformers** for embeddings (Ollama Cloud has no embeddings API).

**Short description:** Multimodal RAG chat over PDFs with Ollama Cloud.

The demo paper (*Attention Is All You Need*) is **downloaded automatically** on first use (not stored in Git), so Hugging Face Spaces accepts the push.

## Features

- Gradio chat UI
- One-click demo ingest (arXiv PDF auto-download)
- PDF upload + indexing
- Answers with source previews
- Free deploy on Hugging Face Spaces

## Space setup

1. Create a **Gradio** Space — **CPU basic** is enough (LLM runs on Ollama Cloud).
   If you use **ZeroGPU**, handlers are already decorated with `@spaces.GPU`.
2. Add secret: `OLLAMA_API_KEY`
3. Optional variables:
   - `GRADIO_SSR_MODE=False` (recommended)
   - `OPENAI_BASE_URL=https://ollama.com/v1`
   - `LLM_MODEL=gpt-oss:120b`
   - `EMBED_BACKEND=local` (default; required on Ollama Cloud)
   - `EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2`
4. Push this repo to the Space

**Note:** Delete Space secret `EMBED_MODEL=nomic-embed-text` if you set that earlier — Cloud cannot embed with it.

## Local run

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
# set OLLAMA_API_KEY in .env
# Windows: Poppler + Tesseract on PATH
python app.py
```

## Project layout

- `app.py` — Gradio entry
- `app/` — RAG pipeline (+ optional FastAPI)
- `packages.txt` — Poppler/Tesseract for Spaces
- `requirements.txt` — Python deps
