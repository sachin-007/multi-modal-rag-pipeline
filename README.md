---
title: ModalRAG
emoji: 📄
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Multimodal RAG chat over PDFs with Ollama Cloud
---

# ModalRAG — Multimodal RAG (Gradio)

Ask questions over PDF documents (text, tables, and figures) using a Gradio chat UI, ChromaDB, and **Ollama Cloud** for the LLM and embeddings.

**Short description:** Multimodal RAG chat over PDFs with Ollama Cloud.

The demo paper (*Attention Is All You Need*) is **downloaded automatically** on first use (not stored in Git), so Hugging Face Spaces accepts the push.

## Features

- Gradio chat UI
- One-click demo ingest (arXiv PDF auto-download)
- PDF upload + indexing
- Answers with source previews
- Free deploy on Hugging Face Spaces

## Space setup

1. Create a **Gradio** Space (CPU basic)
2. Add secret: `OLLAMA_API_KEY`
3. Optional variables: `OPENAI_BASE_URL=https://ollama.com/v1`, `LLM_MODEL=gpt-oss:120b`, `EMBED_MODEL=nomic-embed-text`
4. Push this repo to the Space

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
