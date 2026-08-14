FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHROMA_DIR=/app/data/chroma_db \
    UPLOAD_DIR=/app/data/uploads

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --upgrade pip && pip install -r requirements-app.txt

COPY app ./app
COPY static ./static
COPY docs ./docs

RUN mkdir -p /app/data/chroma_db /app/data/uploads

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
