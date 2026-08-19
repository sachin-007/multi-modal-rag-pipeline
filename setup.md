# ModalRAG — Complete setup (after clone)

Step-by-step guide to run this project locally after cloning.

## 1) Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python 3.10–3.12** | **3.11 recommended.** Avoid 3.8 and 3.14. |
| **Git** | To clone the repo |
| **Poppler** | Required for PDF → image (`pdftoppm`) |
| **Tesseract OCR** | Required for `hi_res` PDF parsing |
| **Ollama Cloud API key** | Free key from [ollama.com/settings/keys](https://ollama.com/settings/keys) |
| **Cohere API key (optional)** | Rerank default; local BGE used if missing or API fails |

Disk: allow a few GB for `torch`, Unstructured, MiniLM embeddings, and (on Cohere fallback) `bge-reranker-base`.

---

## 2) Clone

```bash
git clone https://github.com/sachin-007/multi-modal-rag-pipeline.git
cd multi-modal-rag-pipeline
```

---

## 3) System dependencies (Poppler + Tesseract)

### Windows

1. **Poppler** — download a Windows build (e.g. [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases)), unzip (example: `C:\poppler-26.02.0`).
2. **Tesseract** — install from [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki) (often `C:\Program Files\Tesseract-OCR`).

**Session PATH** (run in the same terminal before starting the app):

```powershell
$env:Path = "C:\poppler-26.02.0\Library\bin;" + $env:Path
$env:Path = "C:\Program Files\Tesseract-OCR;" + $env:Path

where.exe pdftoppm
where.exe tesseract
```

Both commands should print a real `.exe` path. Adjust folders to match your install.

**Permanent PATH:** Windows → Environment Variables → Path → add Poppler `Library\bin` and Tesseract folders → open a **new** terminal.

### macOS

```bash
brew install poppler tesseract
```

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr libmagic1 libgl1 libglib2.0-0
```

---

## 4) Python virtualenv + packages

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

First install can take several minutes (torch, unstructured, etc.).

Optional lighter app set (if you only need FastAPI path): `requirements-app.txt` — for Gradio + HF-style local run, prefer `requirements.txt`.

---

## 5) Environment file

```powershell
# Windows
copy .env.example .env
```

```bash
# macOS / Linux
cp .env.example .env
```

Edit `.env` and set at least:

```env
OLLAMA_API_KEY=your_real_ollama_cloud_key

OPENAI_BASE_URL=https://ollama.com/v1
LLM_MODEL=gpt-oss:120b

EMBED_BACKEND=local
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Hybrid retrieval + rerank
RERANK_BACKEND=cohere
RERANK_MODEL=rerank-english-v3.0
RERANK_API_KEY=your_cohere_key_optional
RERANK_FALLBACK_BACKEND=local
RERANK_FALLBACK_MODEL=BAAI/bge-reranker-base
```

Notes:

- **Never commit `.env`** (it is gitignored).
- Ollama Cloud has **chat**, not `/v1/embeddings` — keep `EMBED_BACKEND=local`.
- If `RERANK_API_KEY` is empty or Cohere returns 429/billing/errors, rerank **falls back to local BGE**.
- BM25 uses raw text + tables; vectors use AI summaries; RRF (`k=60`) fuses lists; top 5 go to the LLM.
- Multi-PDF: uploads **append**; same filename **replaces**; use **Clear index** to wipe.

---

## 6) Run the Gradio app (default)

From the **repo root**, with venv active and Poppler/Tesseract on PATH:

```powershell
python app.py
```

Open the URL printed in the terminal (usually `http://127.0.0.1:7860`).

### Use the UI

1. Click **Use demo PDF** or **upload** one or more PDFs (same name replaces).
2. Wait until status is **Ready**.
3. Ask in the **Chat** tab; open **Retrieval log** to see multi-query, BM25, vector, RRF, and rerank (including Cohere → local fallback).
4. **Clear index** wipes Chroma + BM25 corpus.

---

## 7) Optional: FastAPI + custom UI

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000` (serves `static/`).

---

## 8) Optional: Jupyter notebook

```powershell
python -m pip install -r requirements-notebook.txt
jupyter notebook multi_modal_rag.ipynb
```

Still needs Poppler, Tesseract, and a valid `.env` / API key for LLM cells.

---

## 9) Verify setup checklist

| Check | Command / action |
|--------|------------------|
| Python | `python --version` → 3.10–3.12 |
| Venv | prompt shows `(.venv)` |
| Poppler | `where.exe pdftoppm` / `which pdftoppm` |
| Tesseract | `where.exe tesseract` / `which tesseract` |
| API key | `.env` has `OLLAMA_API_KEY` (no quotes) |
| Chat API | see curl below |

Quick chat test (replace with your key; do not paste keys into chat/issues):

```powershell
curl.exe -sS https://ollama.com/v1/chat/completions ^
  -H "Authorization: Bearer YOUR_KEY" ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"gpt-oss:120b\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}]}"
```

`/api/tags` alone is **not** enough proof — test `/v1/chat/completions`.

---

## 10) Common errors

| Error | Fix |
|--------|-----|
| `Unable to get page count. Is poppler installed and in PATH?` | Add Poppler `bin` to PATH in **this** shell; restart app |
| `TesseractNotFoundError` | Install Tesseract; add to PATH |
| `Answer generation failed: 401 Unauthorized` | Invalid/expired key → new key in `.env`, restart app. On HF Space, set secret + restart Space |
| `path "/v1/embeddings" not found` | Keep `EMBED_BACKEND=local` (do not use Ollama Cloud for embeddings) |
| Pip / numpy build fails | Use Python **3.11** venv, not 3.14 |
| Module not found | Activate `.venv` and `pip install -r requirements.txt` |
| App ignores `.env` | Run from repo root; restart after editing `.env` |

---

## 11) Push to GitHub + Hugging Face (optional)

Remotes (example):

```bash
git remote -v
# origin → GitHub
# hf     → https://huggingface.co/spaces/SachNDev/multimodal-rag
```

After commits:

```bash
git push origin main
git push hf main
```

**HF Space secrets** (Settings → Variables and secrets):

- Secret: `OLLAMA_API_KEY`
- Secret (optional): `RERANK_API_KEY` (Cohere; falls back to local BGE)
- Optional: `GRADIO_SSR_MODE=False`, `EMBED_BACKEND=local`, `RERANK_BACKEND=cohere`, `LLM_MODEL=gpt-oss:120b`

Then **Restart** / **Factory reboot** the Space. Spaces do **not** read your local `.env`.

---

## 12) Project layout (quick)

| Path | Role |
|------|------|
| `app.py` | Gradio entry (local + HF Spaces) |
| `app/` | Config, RAG pipeline, optional FastAPI |
| `static/` | Custom UI for FastAPI |
| `requirements.txt` | Main Python deps |
| `packages.txt` | Apt packages for HF Gradio Spaces |
| `.env.example` | Env template |
| `Dockerfile` | Optional Docker deploy |

---

## One-shot local flow (Windows)

```powershell
cd multi-modal-rag-pipeline
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
# Edit .env → set OLLAMA_API_KEY

$env:Path = "C:\poppler-26.02.0\Library\bin;C:\Program Files\Tesseract-OCR;" + $env:Path
python app.py
```

Then open `http://127.0.0.1:7860` → **Use demo PDF** → ask a question.
