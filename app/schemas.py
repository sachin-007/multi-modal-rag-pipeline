"""Pydantic request/response models."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)


class SourceItem(BaseModel):
    index: int
    preview: str
    has_tables: bool = False
    has_images: bool = False
    document_name: str = ""
    chunk_id: str = ""


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = []
    trace: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    ready: bool
    document_name: Optional[str] = None
    llm_model: str
    embed_model: str
    has_api_key: bool


class StatusResponse(BaseModel):
    state: str
    message: str
    progress: float = 0.0
    document_name: Optional[str] = None
    error: Optional[str] = None
    ready: bool = False


class IngestResponse(BaseModel):
    started: bool
    message: str
