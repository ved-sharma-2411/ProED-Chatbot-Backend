"""
Configuration & environment settings for the RAG backend.
All tuneable values live here — override via environment variables or .env file.
"""

from __future__ import annotations

import os
import re


def load_dotenv(dotenv_path: str = ".env") -> None:
    """Minimal .env loader — no third-party deps required."""
    if not os.path.exists(dotenv_path):
        # try one level up (running from backend/)
        parent = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(parent):
            dotenv_path = parent
        else:
            return

    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


# ── Pinecone ──────────────────────────────────────────────────────────────────
INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "proed-chatbot")
INDEX_METRIC: str = "cosine"
INDEX_CLOUD: str = "aws"
INDEX_REGION: str = "us-east-1"

# ── Embedding (OpenAI) ────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# ── LLM (OpenAI) ──────────────────────────────────────────────────────────────
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# ── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K: int = 10
UPSERT_BATCH_SIZE: int = 100

# ── BM25 ──────────────────────────────────────────────────────────────────────
BM25_K1: float = float(os.getenv("BM25_K1", "1.5"))
BM25_B: float = float(os.getenv("BM25_B", "0.75"))

# ── S3 (BM25 cache storage) ───────────────────────────────────────────────────
S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "")
S3_BM25_PREFIX: str = os.getenv("S3_BM25_PREFIX", "bm25")
AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")

# ── Hybrid / Rerank ───────────────────────────────────────────────────────────
HYBRID_CANDIDATE_MULTIPLIER: int = int(os.getenv("HYBRID_CANDIDATE_MULTIPLIER", "4"))
RRF_K: int = int(os.getenv("RRF_K", "60"))
RAG_MIN_RELEVANCE_SCORE: float = float(os.getenv("RAG_MIN_RELEVANCE_SCORE", "0.23"))

# ── Misc ──────────────────────────────────────────────────────────────────────
STRICT_PART_SCOPE: bool = os.getenv("STRICT_PART_SCOPE", "1").strip().lower() not in {
    "0", "false", "no"
}
API_WARMUP_ON_START: bool = os.getenv(
    "API_WARMUP_ON_START", "1"
).strip().lower() not in {"0", "false", "no"}

UNKNOWN_ANSWER: str = (
    "I'm unable to provide a reliable answer from the available knowledge base. "
    "Please share more context or a source-specific question."
)

# ── Pre-compiled regexes (shared across modules) ──────────────────────────────
NODE_ID_QUERY_RE = re.compile(
    r"p-(\d+(?:\.\d+)+(?:\([^)]+\))+)", re.IGNORECASE
)
BARE_NODE_ID_QUERY_RE = re.compile(
    r"(\d+(?:\.\d+)+(?:\([^)]+\))+)", re.IGNORECASE
)
PART_QUERY_RE = re.compile(
    r"\b(?:34\s*cfr\s*)?part\s*(\d{3})\b", re.IGNORECASE
)
NON_KB_QUERY_RE = re.compile(
    r"\b(what\s+is\s+your\s+name|who\s+are\s+you|your\s+name|hello|hi|hey"
    r"|how\s+are\s+you|good\s+morning|good\s+evening)\b",
    re.IGNORECASE,
)
