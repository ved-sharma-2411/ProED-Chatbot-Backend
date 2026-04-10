"""
Pydantic request / response models for the RAG API.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from config import DEFAULT_TOP_K


# ── Requests ──────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    """Full RAG request: retrieval + LLM answer."""

    question: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=25)
    namespace: str = Field(default="default")


class QueryRequest(BaseModel):
    """Retrieval-only request — returns citations, no LLM call."""

    question: str = Field(..., min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=25)
    namespace: str = Field(default="default")


# ── Shared sub-models ─────────────────────────────────────────────────────────

class Citation(BaseModel):
    source_name: str
    topic_heading: str
    section_ref: str
    node_id: str
    source_url: str
    score: float
    snippet: str


# ── Responses ─────────────────────────────────────────────────────────────────

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    out_of_scope: bool = False


class QueryResponse(BaseModel):
    citations: List[Citation]


class HealthResponse(BaseModel):
    status: str


class RootResponse(BaseModel):
    service: str
    status: str
    docs: str
    health: str
    endpoints: dict
