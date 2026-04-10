# Complete Backend Breakdown of Our RAG Chatbot

## Overview

This is the standalone FastAPI backend for the **ProEd RAG Chatbot** — a Retrieval-Augmented Generation system built over eCFR Title 34 federal education regulations.

The backend sits between the Vercel frontend and Pinecone vector database. It handles everything from receiving a user question to returning a grounded answer with citations. Scraping, chunking, and ingestion are completely separate (run on AWS Batch) and are not part of this server.

---

## Architecture

```
AWS Batch
(scrape → parse → chunk → embed → push to Pinecone + upload BM25 to S3)
                    ↓
            ┌───────────────┐
            │   Pinecone    │  ← vector store (AWS us-east-1)
            │   (vectors)   │
            └───────┬───────┘
                    │
            ┌───────────────┐       ┌──────────────┐
            │  S3 Bucket    │       │   Railway    │
            │  (BM25 cache) │──────▶│   Backend    │◀──── Vercel Frontend
            └───────────────┘       │  (this repo) │      (user chat UI)
                                    └──────────────┘
```

**Key principle:** The backend is read-only at runtime. It never scrapes, never writes to Pinecone, never writes to S3. AWS Batch does all the writing. The backend only reads.

---

## File Structure

```
backend/
├── main.py          ← FastAPI app — routes, CORS, startup
├── config.py        ← All settings and environment variables
├── rag_engine.py    ← All RAG logic — retrieval, ranking, LLM
├── models.py        ← Pydantic request/response shapes
└── requirements.txt ← Runtime dependencies
```

---

## File by File Breakdown

---

### `config.py` — Settings & Environment

The single source of truth for all configuration. No logic runs here — it only holds values read from environment variables.

**What it manages:**
- Pinecone index name, cloud region, metric type
- OpenAI embedding model (`text-embedding-3-large`) and chat model (`gpt-4o-mini`)
- S3 bucket name, BM25 prefix key, AWS region — for loading the BM25 keyword index on startup
- BM25 tuning parameters (k1, b)
- Hybrid search tuning (candidate multiplier, RRF k value)
- Relevance threshold — minimum score a result must hit before the LLM is called
- CORS origin list and warmup flag
- Pre-compiled regex patterns shared across the app (CFR part detection, node ID extraction, non-KB query detection)

**Rule:** If you need to change a model, a threshold, a timeout, or any tunable value — you only ever touch this file.

---

### `models.py` — Request & Response Shapes

Defines what data comes into the API and what goes out. No logic, pure data contracts. Pydantic validates all incoming requests automatically — malformed requests are rejected before touching any logic.

**Request models:**
- `AskRequest` — sent by frontend to `/ask`: question text, how many results to retrieve (`top_k`), which Pinecone namespace
- `QueryRequest` — sent to `/query`: same fields, no LLM call made

**Response models:**
- `Citation` — one source result: `source_name`, `section_ref`, `node_id`, `source_url`, `score`, `snippet` (first 300 chars of content)
- `AskResponse` — returned by `/ask`: `answer` (LLM-generated text), `citations` (list of Citation), `out_of_scope` (boolean, true if no relevant results found)
- `QueryResponse` — returned by `/query`: `citations` only, no answer field
- `HealthResponse` — returned by `/health`: `{ "status": "ok" }`

---

### `rag_engine.py` — The Brain

All the intelligence lives here. No HTTP, no routes — pure business logic called by `main.py`. This is the file that makes the chatbot produce accurate, cited answers.

#### Client Management
- `build_openai_client()` — creates a singleton OpenAI client using `OPENAI_API_KEY`. Reused across all embedding and LLM calls. Never recreated on each request.
- `build_pinecone_client()` — creates a singleton Pinecone client using `PINECONE_API_KEY`.
- `get_pinecone_index()` — connects to the named Pinecone index, validates dimension matches, caches the index object in memory.

#### BM25 Keyword Index (S3-backed)
The BM25 index is a JSON file uploaded to S3 by AWS Batch after every ingestion run. On first use, the server downloads it from S3 into memory and keeps it cached for the lifetime of the process.

- `_download_bm25_store_from_s3()` — downloads `s3://{S3_BUCKET_NAME}/{S3_BM25_PREFIX}/{index}__{namespace}.json` into memory. Logs a warning and disables keyword search gracefully if the file doesn't exist yet.
- `_get_bm25_index()` — checks in-memory cache first, calls S3 download if not cached, builds the BM25 scoring structures (term frequencies, IDF, average doc length).
- `keyword_bm25_query()` — scores all documents in the BM25 index against the query using the BM25 formula. Supports part-level filtering. Returns top-k results normalized to 0–1.

#### Embedding
- `embed_texts()` — sends a list of text strings to OpenAI's embedding API (`text-embedding-3-large`), returns a list of float vectors. Has exponential backoff retry (up to 4 attempts).
- `embed_query()` — thin wrapper for single-string embedding.

#### Semantic Search (Pinecone)
- `semantic_query()` — the full retrieval pipeline:
  1. Detects non-knowledge-base queries (greetings, identity questions) and returns empty immediately
  2. Embeds the query into a vector
  3. Extracts any explicit node ID reference from the query (e.g. `668.32(a)(1)`) — if found, filters Pinecone by exact node
  4. Extracts any CFR part hint (e.g. "Part 682", "direct loan" → Part 685) — if found, filters Pinecone by part
  5. Falls back to unrestricted Pinecone search if filtered results are empty
  6. Runs parallel BM25 keyword search
  7. Merges both result sets via RRF
  8. Re-ranks the merged list
  9. Applies relevance gate — returns empty list if top score is below threshold

#### Hybrid Merge — Reciprocal Rank Fusion (RRF)
- `rrf_merge()` — combines the semantic result list and keyword result list into one unified ranking using the RRF formula: `score = 1 / (k + rank)`. A document that appears in both lists gets contributions from both. The final list is sorted by RRF score, normalized, and returned.

#### Custom Reranker
- `rerank_matches()` — re-scores every result from the RRF merge using a weighted formula:
  - `0.70 × base score` (from Pinecone cosine similarity / RRF)
  - `0.25 × keyword overlap` (query tokens found in section ref + content)
  - `+0.07` citation boost if the query references a specific CFR citation that appears in the chunk
  - `+0.05` authority boost for regulations, `+0.06` for statutes
  - `+0.02` source URL presence boost
  - `+0.35` explicit node ID exact match boost
  - `+0.22` part match boost / `−0.08` part mismatch penalty
  - `−up to 0.08` length penalty for chunks over 2600 characters
- `should_skip()` — checks if the top result's final score is below `RAG_MIN_RELEVANCE_SCORE`. If yes, the LLM is never called and the unknown-answer response is returned.

#### Context Builder & Citation Helpers
- `build_context()` — formats the top-k matches into a structured text block fed to the LLM. Each block includes: section ref, source name, source type, authority level, source URL, and full content.
- `_format_citation()` — formats a single metadata dict into a human-readable citation string with URL.
- `_replace_source_tags()` — if the LLM output contains `[SOURCE 1]` style placeholders, replaces them with real section references and URLs.
- `_add_citation_footer()` — if the LLM answer contains no URLs or section symbols (§), appends a citations block at the end automatically.

#### RAG Answer
- `rag_answer()` — the main function called by the `/ask` route:
  1. Uses pre-fetched matches (passed from the route) or runs `semantic_query()` itself
  2. Returns the unknown-answer string immediately if `should_skip()` is true
  3. Calls GPT-4o-mini with a strict system prompt: answer only from provided sources, no hallucination, no placeholder citations
  4. Post-processes the answer through citation formatting
  5. Returns answer, source matches, and `out_of_scope` flag

---

### `main.py` — FastAPI Server

The entry point and HTTP layer. Handles routing only — all logic is delegated to `rag_engine.py`.

#### Startup
On server start:
1. Loads `.env` file (for local dev; Railway injects env vars directly)
2. Configures logging
3. Fires a background thread that runs `warmup_runtime()` — makes a probe embedding call and connects to Pinecone so the first real user request is not cold

#### CORS
Configured via `CORS_ORIGINS` env var. Defaults to `*` for development. Set to your Vercel domain in production (e.g. `https://proed-chatbot.vercel.app`).

#### Routes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info — lists all endpoints, version, status |
| `GET` | `/health` | Liveness probe — Railway uses this to confirm the server is running |
| `POST` | `/query` | Retrieval only — returns top-k citations, no LLM call made |
| `POST` | `/ask` | Full RAG — retrieval + rerank + GPT answer + citations |

#### Part-scope filter (`_enforce_part_scope`)
If `STRICT_PART_SCOPE=1` (default), and the query mentions a specific CFR part, results from other parts are filtered out after retrieval. Falls back to unfiltered results if filtering leaves nothing.

---

## Full Request Flow — `/ask`

```
Frontend sends POST /ask { question, top_k, namespace }
        │
        ▼
main.py receives request, validates via Pydantic
        │
        ▼
rag_engine.semantic_query()
    ├── Detect non-KB query? → return [] immediately
    ├── embed_query() → OpenAI text-embedding-3-large
    ├── extract node ID hint from query
    ├── extract CFR part hint from query
    ├── Pinecone query (filtered or unrestricted)
    ├── keyword_bm25_query() → BM25 from S3-loaded index
    ├── rrf_merge() → combine semantic + keyword
    └── rerank_matches() → final scored list
        │
        ▼
main.py _enforce_part_scope() → filter by CFR part if needed
        │
        ▼
rag_engine.rag_answer()
    ├── should_skip()? → return UNKNOWN_ANSWER if score too low
    ├── build_context() → format top-k chunks for LLM
    ├── OpenAI GPT-4o-mini chat completion
    └── _add_citation_footer() → inject real citations into answer
        │
        ▼
main.py builds AskResponse { answer, citations, out_of_scope }
        │
        ▼
Frontend receives response
Frontend stores answer in localStorage (per-user cache)
```

---

## Current Features

| Feature | Detail |
|---------|--------|
| Semantic vector search | Pinecone cosine similarity, `text-embedding-3-large` |
| BM25 keyword search | S3-backed, loaded into memory on first request |
| Hybrid ranking | Reciprocal Rank Fusion (RRF) merging both result sets |
| Custom reranker | 8-factor scoring — keyword, citation, authority, part, node match |
| CFR part-scope filtering | Auto-detects part from query, filters results |
| Exact node/paragraph lookup | Detects references like `668.32(a)(1)`, queries by exact node ID |
| Out-of-scope detection | Greetings/identity queries short-circuited before any API call |
| Relevance gate | Returns unknown-answer if top score < threshold, never hallucinates |
| LLM answer generation | GPT-4o-mini, strict grounding prompt, temperature 0 |
| Structured citations | Every answer includes section ref, source URL, content snippet |
| Citation auto-formatting | Replaces `[SOURCE N]` placeholders with real references |
| CORS for Vercel | Configurable via `CORS_ORIGINS` env var |
| Railway health check | `GET /health` endpoint |
| Auto warmup on startup | Background thread pre-loads OpenAI + Pinecone connections |
| Per-user LLM cache | Handled in browser localStorage on the frontend |
| All config via env vars | Zero hardcoded secrets or values |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ | — | OpenAI API key (embeddings + GPT) |
| `PINECONE_API_KEY` | ✅ | — | Pinecone API key |
| `S3_BUCKET_NAME` | ✅ | — | S3 bucket holding the BM25 cache |
| `AWS_ACCESS_KEY_ID` | ✅ | — | AWS credentials for S3 access |
| `AWS_SECRET_ACCESS_KEY` | ✅ | — | AWS credentials for S3 access |
| `AWS_REGION` | ⬜ | `us-east-1` | AWS region |
| `S3_BM25_PREFIX` | ⬜ | `bm25` | S3 key prefix for BM25 files |
| `PINECONE_INDEX_NAME` | ⬜ | `proed-chatbot` | Pinecone index name |
| `EMBEDDING_MODEL` | ⬜ | `text-embedding-3-large` | Must match ingestion model |
| `CHAT_MODEL` | ⬜ | `gpt-4o-mini` | OpenAI chat model |
| `CORS_ORIGINS` | ⬜ | `*` | Comma-separated allowed origins |
| `STRICT_PART_SCOPE` | ⬜ | `1` | Filter results to CFR part in query |
| `RAG_MIN_RELEVANCE_SCORE` | ⬜ | `0.23` | Minimum score to trigger LLM |
| `API_WARMUP_ON_START` | ⬜ | `1` | Pre-warm connections on startup |

---

## Running Locally

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: `http://localhost:8000/docs`

## Production (Railway)

Railway auto-runs `pip install -r requirements.txt` on deploy. Set all required environment variables in the Railway dashboard. No other configuration needed.

```bash
# Production start command
uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API framework | FastAPI + Uvicorn |
| Data validation | Pydantic v2 |
| Vector database | Pinecone (Serverless, AWS us-east-1) |
| Embeddings | OpenAI `text-embedding-3-large` |
| LLM | OpenAI `gpt-4o-mini` |
| Keyword search | BM25 (in-process, S3-backed) |
| Object storage | AWS S3 (BM25 cache) |
| Frontend cache | Browser localStorage |
| Hosting | Railway |
