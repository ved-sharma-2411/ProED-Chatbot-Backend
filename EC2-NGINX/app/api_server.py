from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_pinecone import (
    DEFAULT_TOP_K,
    UNKNOWN_ANSWER,
    build_source_label,
    extract_part_hint,
    infer_part_from_metadata,
    infer_source_url,
    load_dotenv,
    rag_answer,
    semantic_query,
    setup_logging,
    warmup_runtime,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=25)
    namespace: str = Field(default="default")
    force_llm: bool = Field(default=False)


class Citation(BaseModel):
    source_name: str
    topic_heading: str
    section_ref: str
    node_id: str
    source_url: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    out_of_scope: bool = False
    from_cache: bool = False
    llm_skipped: bool = False
    retry_after_seconds: Optional[int] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=25)
    namespace: str = Field(default="default")


class QueryResponse(BaseModel):
    citations: List[Citation]


def _to_citation(item: Dict[str, Any]) -> Citation:
    md = item.get("metadata", {}) or {}
    source_url = infer_source_url(md)
    source_name = build_source_label(md)
    content = str(md.get("content") or "").strip()
    snippet = content[:300] + ("..." if len(content) > 300 else "")
    return Citation(
        source_name=source_name,
        topic_heading=str(md.get("document_title") or md.get("path_label") or ""),
        section_ref=str(md.get("section_ref") or ""),
        node_id=str(md.get("node_id") or ""),
        source_url=source_url,
        score=float(item.get("score") or 0.0),
        snippet=snippet,
    )


app = FastAPI(title="ProEd RAG API", version="1.0.0")


def _enforce_part_scope(question: str, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    strict = str(os.getenv("STRICT_PART_SCOPE", "1")).strip().lower() not in {"0", "false", "no"}
    if not strict:
        return matches

    part_hint = extract_part_hint(question)
    if not part_hint:
        return matches

    filtered: List[Dict[str, Any]] = []
    for m in matches:
        md = m.get("metadata", {}) or {}
        doc_part = infer_part_from_metadata(md)
        if doc_part == part_hint:
            filtered.append(m)

    return filtered if filtered else matches


def _warmup_async() -> None:
    try:
        warmup_runtime()
    except Exception as exc:
        logging.warning("Warmup failed: %s", exc)


@app.on_event("startup")
def _startup() -> None:
    load_dotenv()
    setup_logging(verbose=False)
    if str(os.getenv("API_WARMUP_ON_START", "1")).strip().lower() not in {"0", "false", "no"}:
        threading.Thread(target=_warmup_async, daemon=True).start()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "service": "ProEd RAG API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "ask": "/ask",
        "query": "/query",
    }


@app.post("/query", response_model=QueryResponse)
def query_only(request: QueryRequest) -> QueryResponse:
    try:
        matches = semantic_query(
            query=request.question,
            top_k=request.top_k,
            namespace=request.namespace,
        )
        matches = _enforce_part_scope(request.question, matches)
        citations = [_to_citation(m) for m in matches]
        return QueryResponse(citations=citations)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        matches = semantic_query(
            query=request.question,
            top_k=request.top_k,
            namespace=request.namespace,
        )
        matches = _enforce_part_scope(request.question, matches)
        result = rag_answer(
            user_query=request.question,
            top_k=request.top_k,
            namespace=request.namespace,
            force_llm=request.force_llm,
            prefetched_matches=matches,
        )

        answer = str(result.get("answer") or "").strip() or UNKNOWN_ANSWER
        citations = [_to_citation(m) for m in result.get("sources", [])]
        if not citations or bool(result.get("out_of_scope", False)):
            answer = UNKNOWN_ANSWER

        return AskResponse(
            answer=answer,
            citations=citations,
            out_of_scope=bool(result.get("out_of_scope", False)),
            from_cache=bool(result.get("from_cache", False)),
            llm_skipped=bool(result.get("llm_skipped", False)),
            retry_after_seconds=result.get("retry_after_seconds"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
