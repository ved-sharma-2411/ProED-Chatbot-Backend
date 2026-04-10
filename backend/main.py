"""
ProEd RAG — standalone FastAPI backend.

Start with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Or via gunicorn (production):
    gunicorn main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
import rag_engine
from models import (
    AskRequest,
    AskResponse,
    Citation,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ProEd RAG API",
    version="2.0.0",
    description="Retrieval-Augmented Generation over eCFR Title 34 regulations.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins by default; restrict in production via CORS_ORIGINS env var.
_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def _startup() -> None:
    config.load_dotenv()
    rag_engine.setup_logging(verbose=False)
    if config.API_WARMUP_ON_START:
        threading.Thread(target=_warmup, daemon=True).start()


def _warmup() -> None:
    try:
        rag_engine.warmup_runtime()
        logging.info("Warmup complete.")
    except Exception as exc:
        logging.warning("Warmup failed: %s", exc)


# ── Helper: match -> Citation ─────────────────────────────────────────────────

def _to_citation(item: Dict[str, Any]) -> Citation:
    md = item.get("metadata", {}) or {}
    content = str(md.get("content") or "").strip()
    return Citation(
        source_name=rag_engine.build_source_label(md),
        topic_heading=str(md.get("document_title") or md.get("path_label") or ""),
        section_ref=str(md.get("section_ref") or ""),
        node_id=str(md.get("node_id") or ""),
        source_url=rag_engine.infer_source_url(md),
        score=float(item.get("score") or 0.0),
        snippet=content[:300] + ("..." if len(content) > 300 else ""),
    )


# ── Part-scope filter (optional strict mode) ──────────────────────────────────

def _enforce_part_scope(
    question: str, matches: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not config.STRICT_PART_SCOPE:
        return matches
    part_hint = rag_engine.extract_part_hint(question)
    if not part_hint:
        return matches
    filtered = [
        m for m in matches
        if rag_engine.infer_part(m.get("metadata", {}) or {}) == part_hint
    ]
    return filtered if filtered else matches


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Meta"])
def root() -> Dict[str, Any]:
    """Service info and available endpoints."""
    return {
        "service": "ProEd RAG API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "GET  /health": "Liveness probe",
            "POST /ask": "Full RAG — retrieval + LLM answer + citations",
            "POST /query": "Retrieval-only — citations without LLM",
        },
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health() -> HealthResponse:
    """Kubernetes / load-balancer liveness probe."""
    return HealthResponse(status="ok")


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_only(request: QueryRequest) -> QueryResponse:
    """
    Retrieval-only endpoint.

    Embeds the question, runs hybrid search (semantic + BM25 + RRF + rerank),
    and returns the top-k citations. No LLM call is made.
    """
    try:
        matches = rag_engine.semantic_query(
            query=request.question,
            top_k=request.top_k,
            namespace=request.namespace,
        )
        matches = _enforce_part_scope(request.question, matches)
        return QueryResponse(citations=[_to_citation(m) for m in matches])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(request: AskRequest) -> AskResponse:
    """
    Full RAG endpoint.

    1. Embeds the question using OpenAI text-embedding-3-large.
    2. Runs hybrid retrieval (Pinecone semantic + BM25 keyword + RRF merge + rerank).
    3. Applies part-scope filter if STRICT_PART_SCOPE=1.
    4. Calls GPT-4o-mini with retrieved context to generate a grounded answer.
    5. Returns answer + structured citations.
    """
    try:
        matches = rag_engine.semantic_query(
            query=request.question,
            top_k=request.top_k,
            namespace=request.namespace,
        )
        matches = _enforce_part_scope(request.question, matches)

        result = rag_engine.rag_answer(
            user_query=request.question,
            top_k=request.top_k,
            namespace=request.namespace,
            prefetched_matches=matches,
        )

        out_of_scope = bool(result.get("out_of_scope", False))
        citations = [_to_citation(m) for m in result.get("sources", [])]
        answer = str(result.get("answer") or "").strip() or config.UNKNOWN_ANSWER

        if not citations or out_of_scope:
            answer = config.UNKNOWN_ANSWER

        return AskResponse(
            answer=answer,
            citations=citations,
            out_of_scope=out_of_scope,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
