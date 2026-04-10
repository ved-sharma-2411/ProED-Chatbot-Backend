"""
Core RAG engine — retrieval + answering.

Covers:
- Embedding generation (OpenAI text-embedding-3-small by default)
- Pinecone semantic search
- BM25 keyword search (in-process, JSON-backed cache)
- Hybrid merge via Reciprocal Rank Fusion (RRF)
- Custom reranker (keyword overlap + citation + authority boosts)
- LLM answer via OpenAI (with rate-limit guard + response cache)

NOT included here: scraping, HTML parsing, chunking, or ingestion.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

import config

# ── Module-level caches ────────────────────────────────────────────────────────
_OPENAI_CLIENT_CACHE: Optional[OpenAI] = None
_PINECONE_CLIENT_CACHE: Optional[Pinecone] = None
_INDEX_OBJ_CACHE: Dict[str, Any] = {}
_INDEX_DIM_VALIDATED: Dict[str, int] = {}
_BM25_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}
_QUIET_RUNTIME: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# Logging / silence helpers
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(verbose: bool = False) -> None:
    global _QUIET_RUNTIME
    _QUIET_RUNTIME = not verbose
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")
    if not verbose:
        for noisy in ["httpx", "httpcore", "urllib3"]:
            logging.getLogger(noisy).setLevel(logging.ERROR)


# ═══════════════════════════════════════════════════════════════════════════════
# Client builders (cached singletons)
# ═══════════════════════════════════════════════════════════════════════════════

def build_openai_client() -> OpenAI:
    global _OPENAI_CLIENT_CACHE
    if _OPENAI_CLIENT_CACHE is None:
        _OPENAI_CLIENT_CACHE = OpenAI(
            api_key=config.get_required_env("OPENAI_API_KEY"),
        )
    return _OPENAI_CLIENT_CACHE


def build_pinecone_client() -> Pinecone:
    global _PINECONE_CLIENT_CACHE
    if _PINECONE_CLIENT_CACHE is None:
        _PINECONE_CLIENT_CACHE = Pinecone(
            api_key=config.get_required_env("PINECONE_API_KEY")
        )
    return _PINECONE_CLIENT_CACHE


# ═══════════════════════════════════════════════════════════════════════════════
# Pinecone index management
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_index(pc: Pinecone, index_name: str, dimension: int) -> None:
    listed = pc.list_indexes()
    existing: set[str] = (
        set(listed.names())
        if hasattr(listed, "names")
        else {
            (idx.get("name") if isinstance(idx, dict) else getattr(idx, "name", None))
            for idx in listed
        }
    )
    existing.discard(None)  # type: ignore

    if index_name in existing:
        desc = pc.describe_index(index_name)
        idx_dim = getattr(desc, "dimension", None) or (
            desc.get("dimension") if isinstance(desc, dict) else None
        )
        if idx_dim is not None and int(idx_dim) != int(dimension):
            raise ValueError(
                f"Index '{index_name}' dimension mismatch: existing={idx_dim}, "
                f"expected={dimension}."
            )
        return

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=config.INDEX_METRIC,
        spec=ServerlessSpec(cloud=config.INDEX_CLOUD, region=config.INDEX_REGION),
    )
    for _ in range(60):
        desc = pc.describe_index(index_name)
        status_obj = getattr(desc, "status", None)
        ready = (
            bool(status_obj.get("ready", False))
            if isinstance(status_obj, dict)
            else bool(getattr(status_obj, "ready", False))
        )
        if ready:
            return
        time.sleep(2)
    raise TimeoutError(f"Index creation timed out: {index_name}")


def get_pinecone_index(index_name: str, dimension: int):
    """Return a cached Pinecone index object, ensuring the index exists."""
    if _INDEX_DIM_VALIDATED.get(index_name) == dimension:
        cached = _INDEX_OBJ_CACHE.get(index_name)
        if cached is not None:
            return cached

    pc = build_pinecone_client()
    _ensure_index(pc, index_name, dimension)
    index = pc.Index(index_name)
    _INDEX_DIM_VALIDATED[index_name] = dimension
    _INDEX_OBJ_CACHE[index_name] = index
    return index


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding
# ═══════════════════════════════════════════════════════════════════════════════

def embed_texts(
    texts: List[str],
    model: str = config.EMBEDDING_MODEL,
    max_retries: int = 4,
    retry_base_seconds: float = 1.5,
) -> List[List[float]]:
    client = build_openai_client()
    for attempt in range(max_retries + 1):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(retry_base_seconds * (2 ** attempt))
    raise RuntimeError("embed_texts: unreachable")


def embed_query(query: str, model: str = config.EMBEDDING_MODEL) -> List[float]:
    return embed_texts([query], model=model)[0]


def warmup_runtime(
    index_name: str = config.INDEX_NAME,
    embedding_model: str = config.EMBEDDING_MODEL,
) -> None:
    """Pre-load model + verify Pinecone connection so first request is fast."""
    probe = embed_query("warmup probe", model=embedding_model)
    get_pinecone_index(index_name, dimension=len(probe))


# ═══════════════════════════════════════════════════════════════════════════════
# BM25 keyword index (S3-backed, in-process memory cache)
# ═══════════════════════════════════════════════════════════════════════════════

def _s3_bm25_key(namespace: str, index_name: str) -> str:
    ns = (namespace or "default").strip()
    return f"{config.S3_BM25_PREFIX}/{index_name}__{ns}.json"


def _download_bm25_store_from_s3(namespace: str, index_name: str) -> Dict[str, Dict[str, Any]]:
    """Download the BM25 JSON store from S3. Returns empty dict if not found."""
    bucket = config.S3_BUCKET_NAME
    if not bucket:
        logging.warning("S3_BUCKET_NAME not set — BM25 keyword search disabled.")
        return {}

    key = _s3_bm25_key(namespace, index_name)
    try:
        s3 = boto3.client("s3", region_name=config.AWS_REGION)
        buf = io.BytesIO()
        s3.download_fileobj(bucket, key, buf)
        buf.seek(0)
        obj = json.loads(buf.read().decode("utf-8"))
        logging.info("BM25 store loaded from s3://%s/%s", bucket, key)
        return {k: v for k, v in obj.items() if isinstance(k, str) and isinstance(v, dict)}
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("NoSuchKey", "404"):
            logging.warning("BM25 store not found at s3://%s/%s — keyword search disabled.", bucket, key)
        else:
            logging.error("S3 error loading BM25 store: %s", e)
        return {}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)?", (text or "").lower())


def _bm25_text(md: Dict[str, Any]) -> str:
    return " ".join(
        [
            str(md.get("section_ref") or ""),
            str(md.get("node_id") or ""),
            str(md.get("document_title") or ""),
            str(md.get("part") or ""),
            str(md.get("content") or ""),
        ]
    ).strip()


def _build_bm25_index(doc_store: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    ids = list(doc_store.keys())
    doc_tfs: List[Dict[str, int]] = []
    doc_lens: List[int] = []
    df: Dict[str, int] = defaultdict(int)

    for doc_id in ids:
        md = doc_store[doc_id].get("metadata", {}) or {}
        tokens = _tokenize(_bm25_text(md))
        tf = Counter(tokens)
        doc_tfs.append(dict(tf))
        doc_lens.append(len(tokens))
        for tok in tf:
            df[tok] += 1

    N = max(len(ids), 1)
    avgdl = sum(doc_lens) / N if N else 0.0
    idf = {
        tok: math.log(1.0 + ((N - f + 0.5) / (f + 0.5)))
        for tok, f in df.items()
    }
    return {"ids": ids, "doc_tfs": doc_tfs, "doc_lens": doc_lens, "avgdl": avgdl, "idf": idf}


def _get_bm25_index(namespace: str, index_name: str) -> Dict[str, Any]:
    cache_key = f"{index_name}::{namespace}"
    if cache_key not in _BM25_INDEX_CACHE:
        doc_store = _download_bm25_store_from_s3(namespace, index_name)
        idx = _build_bm25_index(doc_store)
        idx["doc_store"] = doc_store
        _BM25_INDEX_CACHE[cache_key] = idx
    return _BM25_INDEX_CACHE[cache_key]


def keyword_bm25_query(
    query: str,
    namespace: str,
    top_k: int,
    index_name: str = config.INDEX_NAME,
    part_hint: str = "",
) -> List[Dict[str, Any]]:
    bm25 = _get_bm25_index(namespace, index_name)
    ids = bm25.get("ids", [])
    if not ids:
        return []

    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    idf = bm25["idf"]
    doc_tfs = bm25["doc_tfs"]
    doc_lens = bm25["doc_lens"]
    avgdl = float(bm25.get("avgdl") or 0.0) or 1.0
    doc_store = bm25["doc_store"]
    k1, b = config.BM25_K1, config.BM25_B

    scored: List[tuple[int, float]] = []
    for i, doc_id in enumerate(ids):
        md = doc_store.get(doc_id, {}).get("metadata", {}) or {}
        if part_hint and infer_part(md) and infer_part(md) != part_hint:
            continue
        tf = doc_tfs[i] if i < len(doc_tfs) else {}
        dl = float(doc_lens[i]) if i < len(doc_lens) else 0.0
        score = 0.0
        for tok in q_tokens:
            f = float(tf.get(tok, 0.0))
            if f <= 0:
                continue
            den = f + k1 * (1.0 - b + b * (dl / avgdl))
            score += float(idf.get(tok, 0.0)) * ((f * (k1 + 1.0)) / max(den, 1e-9))
        if score > 0:
            scored.append((i, score))

    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)
    max_score = max(s for _, s in scored) or 1.0
    return [
        {
            "id": ids[idx],
            "score": score / max_score,
            "metadata": doc_store.get(ids[idx], {}).get("metadata", {}) or {},
            "_keyword_rank": rank,
        }
        for rank, (idx, score) in enumerate(scored[:top_k], start=1)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid merge — Reciprocal Rank Fusion
# ═══════════════════════════════════════════════════════════════════════════════

def rrf_merge(
    semantic: List[Dict[str, Any]],
    keyword: List[Dict[str, Any]],
    top_k: int,
    rrf_k: int = config.RRF_K,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for rank, item in enumerate(semantic, start=1):
        doc_id = str(item.get("id") or "")
        if not doc_id:
            continue
        rec = merged.setdefault(
            doc_id,
            {"id": doc_id, "metadata": item.get("metadata", {}) or {}, "_rrf": 0.0},
        )
        rec["_rrf"] += 1.0 / (rrf_k + rank)
        rec["_sem_rank"] = rank

    for rank, item in enumerate(keyword, start=1):
        doc_id = str(item.get("id") or "")
        if not doc_id:
            continue
        rec = merged.setdefault(
            doc_id,
            {"id": doc_id, "metadata": item.get("metadata", {}) or {}, "_rrf": 0.0},
        )
        rec["_rrf"] += 1.0 / (rrf_k + rank)
        rec["_key_rank"] = rank

    if not merged:
        return []

    items = sorted(merged.values(), key=lambda x: x.get("_rrf", 0.0), reverse=True)
    max_rrf = max(float(x.get("_rrf") or 0.0) for x in items) or 1.0

    return [
        {
            "id": rec["id"],
            "score": float(rec.get("_rrf", 0.0)) / max_rrf,
            "metadata": rec.get("metadata", {}) or {},
        }
        for rec in items[: max(top_k * 2, top_k)]
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Metadata helpers
# ═══════════════════════════════════════════════════════════════════════════════

def infer_part(md: Dict[str, Any]) -> str:
    part = str(md.get("part") or "").strip()
    if part:
        return part
    for field in ("section_ref", "node_id", "source_url"):
        m = re.search(r"(\d+)\.(\d+)", str(md.get(field) or ""))
        if m:
            return m.group(1)
    return ""


def extract_part_hint(query: str) -> str:
    q = (query or "").lower().strip()
    m = config.PART_QUERY_RE.search(q)
    if m:
        return m.group(1)
    if "federal family education loan" in q or "ffel" in q:
        return "682"
    if "direct loan" in q:
        return "685"
    return ""


def extract_node_id(query: str) -> str:
    q_compact = re.sub(r"\s+", "", (query or "").strip())
    m = config.NODE_ID_QUERY_RE.search(q_compact)
    if m:
        return f"p-{m.group(1)}"
    m2 = config.BARE_NODE_ID_QUERY_RE.search(q_compact)
    return f"p-{m2.group(1)}" if m2 else ""


def is_non_kb_query(query: str) -> bool:
    q = (query or "").strip()
    if not q or not config.NON_KB_QUERY_RE.search(q):
        return False
    stripped = re.sub(r"[^a-zA-Z0-9]+", " ", config.NON_KB_QUERY_RE.sub(" ", q)).strip()
    return not stripped


def infer_source_url(md: Dict[str, Any]) -> str:
    url = str(md.get("source_url") or "").strip()
    if url:
        return url
    for field in ("node_id", "section_ref"):
        val = str(md.get(field) or "")
        m = re.search(r"p-(\d+(?:\.\d+)+)" if field == "node_id" else r"(\d+(?:\.\d+)+)", val)
        if m:
            sec = m.group(1)
            part = sec.split(".")[0]
            return f"https://www.ecfr.gov/current/title-34/subtitle-B/chapter-VI/part-{part}/section-{sec}"
    return ""


def build_source_label(md: Dict[str, Any]) -> str:
    source_name = str(md.get("source_name") or "").strip()
    if source_name:
        return source_name
    url = str(md.get("source_url") or "").strip()
    if url:
        m = re.search(r"https?://([^/]+)", url)
        if m:
            return m.group(1).replace("www.", "")
    return "source"


# ═══════════════════════════════════════════════════════════════════════════════
# Custom reranker
# ═══════════════════════════════════════════════════════════════════════════════

_AUTHORITY_BOOST = {
    "statute": 0.06, "regulation": 0.05, "sub_regulatory": 0.02,
    "high": 0.05, "medium": 0.03, "low": 0.01, "unknown": 0.0,
}

_CFR_REF_RE = re.compile(r"\b\d+\s*cfr\s*\d+(?:\.\d+)+\b")
_NUM_REF_RE = re.compile(r"\b\d+(?:\.\d+)+\b")


def _tok(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9\.]+", (text or "").lower()))


def _extract_cfr_ref(query: str) -> str:
    m = _CFR_REF_RE.search(query.lower())
    if m:
        return m.group(0)
    m2 = _NUM_REF_RE.search(query.lower())
    return m2.group(0) if m2 else ""


def rerank_matches(
    query: str, matches: List[Dict[str, Any]], top_k: int
) -> List[Dict[str, Any]]:
    q_tokens = _tok(query)
    cfr_ref = _extract_cfr_ref(query)
    explicit_node = extract_node_id(query)
    part_hint = extract_part_hint(query)

    ranked = []
    for item in matches:
        md = item.get("metadata", {}) or {}
        content = str(md.get("content", ""))
        section_ref = str(md.get("section_ref", ""))
        node_id = str(md.get("node_id", ""))
        doc_part = infer_part(md)
        authority = str(md.get("authority_level", "unknown")).strip().lower()
        base = float(item.get("score") or 0.0)

        keyword_score = len(q_tokens & _tok(f"{section_ref} {content}")) / max(len(q_tokens), 1)
        early_boost = 0.05 * (len(q_tokens & _tok(f"{section_ref} {content[:500]}")) / max(len(q_tokens), 1))

        citation_boost = 0.0
        if cfr_ref:
            haystack = (content + section_ref).lower()
            if cfr_ref in haystack:
                citation_boost = 0.07
            elif cfr_ref.replace(" ", "") in haystack.replace(" ", ""):
                citation_boost = 0.05

        meta_boost = (0.02 if md.get("source_url") else 0.0) + _AUTHORITY_BOOST.get(authority, 0.0)

        explicit_boost = 0.0
        if explicit_node:
            if node_id.lower() == explicit_node.lower():
                explicit_boost = 0.35
            elif explicit_node.lower() in (content + section_ref).lower():
                explicit_boost = 0.10
            if "merged" in section_ref.lower():
                explicit_boost -= 0.05

        long_penalty = min((len(content) - 2600) / 10000.0, 0.08) if len(content) > 2600 else 0.0

        part_boost = 0.0
        if part_hint:
            part_boost = 0.22 if doc_part == part_hint else -0.08

        final = (
            0.70 * base
            + 0.25 * keyword_score
            + citation_boost
            + meta_boost
            + explicit_boost
            + early_boost
            + part_boost
            - long_penalty
        )
        ranked.append({**item, "_score_final": final})

    ranked.sort(key=lambda x: x.get("_score_final", 0.0), reverse=True)
    return ranked[:top_k]


def should_skip(matches: List[Dict[str, Any]]) -> bool:
    if not matches:
        return True
    top_score = float(matches[0].get("_score_final") or matches[0].get("score") or 0.0)
    return top_score < config.RAG_MIN_RELEVANCE_SCORE


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic search (Pinecone)
# ═══════════════════════════════════════════════════════════════════════════════

def semantic_query(
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    namespace: str = "default",
    index_name: str = config.INDEX_NAME,
    embedding_model: str = config.EMBEDDING_MODEL,
) -> List[Dict[str, Any]]:
    if is_non_kb_query(query):
        return []

    qvec = embed_query(query, model=embedding_model)
    index = get_pinecone_index(index_name, dimension=len(qvec))

    part_hint = extract_part_hint(query)
    explicit_node = extract_node_id(query)
    candidate_k = max(top_k * 3, 20)

    def _pinecone_query(filter_dict=None) -> List[Dict[str, Any]]:
        kwargs: Dict[str, Any] = dict(
            vector=qvec,
            top_k=candidate_k,
            include_values=False,
            include_metadata=True,
            namespace=namespace,
        )
        if filter_dict:
            kwargs["filter"] = filter_dict
        result = index.query(**kwargs)
        raw = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", [])
        out = []
        for m in raw:
            m_id = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
            m_score = m.get("score") if isinstance(m, dict) else getattr(m, "score", None)
            m_md = (m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {})) or {}
            if not str(m_md.get("part") or "").strip():
                m_md["part"] = infer_part(m_md)
            out.append({"id": m_id, "score": m_score, "metadata": m_md})
        return out

    # Try exact node_id first, then part-filtered, then unrestricted
    if explicit_node:
        sem_matches = _pinecone_query({"node_id": {"$eq": explicit_node}})
        if not sem_matches:
            sem_matches = _pinecone_query({"part": {"$eq": part_hint}} if part_hint else None)
    elif part_hint:
        sem_matches = _pinecone_query({"part": {"$eq": part_hint}}) or _pinecone_query()
    else:
        sem_matches = _pinecone_query()

    # BM25 keyword pass
    keyword_k = max(top_k * config.HYBRID_CANDIDATE_MULTIPLIER, 20)
    kw_matches = keyword_bm25_query(
        query=query,
        namespace=namespace,
        top_k=keyword_k,
        index_name=index_name,
        part_hint=part_hint,
    )

    hybrid = rrf_merge(semantic=sem_matches, keyword=kw_matches, top_k=max(top_k * 2, top_k))
    ranked = rerank_matches(query=query, matches=hybrid, top_k=top_k)

    if should_skip(ranked):
        return []
    return ranked


# ═══════════════════════════════════════════════════════════════════════════════
# Context builder + citation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_context(matches: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, item in enumerate(matches, start=1):
        md = item.get("metadata", {})
        blocks.append(
            "\n".join(
                [
                    f"[SOURCE {i}]",
                    f"Section: {md.get('section_ref', '')}",
                    f"Source Name: {md.get('source_name', '')}",
                    f"Source Type: {md.get('source_type', '')}",
                    f"Authority Level: {md.get('authority_level', '')}",
                    f"Source URL: {md.get('source_url', '')}",
                    f"Content: {md.get('content', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _format_citation(md: Dict[str, Any]) -> str:
    section = str(md.get("section_ref") or "").strip()
    label = section or build_source_label(md) or "source"
    url = infer_source_url(md)
    return f"{label} ({url})" if url else label


def _replace_source_tags(answer: str, matches: List[Dict[str, Any]]) -> str:
    def _repl(m: re.Match) -> str:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(matches):
            return _format_citation(matches[idx].get("metadata", {}) or {})
        return "source"
    return re.sub(r"\[\s*source\s*(\d+)\s*\]", _repl, answer, flags=re.IGNORECASE)


def _add_citation_footer(answer: str, matches: List[Dict[str, Any]]) -> str:
    text = (answer or "").strip()
    if not text:
        return text
    text = _replace_source_tags(text, matches)
    if not ("http" in text.lower() or "§" in text):
        footer_lines = [
            f"- {i}. {_format_citation(m.get('metadata', {}) or {})}"
            for i, m in enumerate(matches[:3], start=1)
        ]
        if footer_lines:
            text += "\n\nCitations:\n" + "\n".join(footer_lines)
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Main RAG answer function
# ═══════════════════════════════════════════════════════════════════════════════

def rag_answer(
    user_query: str,
    top_k: int = config.DEFAULT_TOP_K,
    namespace: str = "default",
    prefetched_matches: Optional[List[Dict[str, Any]]] = None,
    index_name: str = config.INDEX_NAME,
    embedding_model: str = config.EMBEDDING_MODEL,
) -> Dict[str, Any]:
    matches = (
        prefetched_matches
        if prefetched_matches is not None
        else semantic_query(user_query, top_k=top_k, namespace=namespace,
                            index_name=index_name, embedding_model=embedding_model)
    )
    context = build_context(matches)

    if should_skip(matches):
        return {
            "answer": config.UNKNOWN_ANSWER,
            "sources": matches,
            "out_of_scope": True,
        }

    client = build_openai_client()
    completion = client.chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer ONLY using provided sources. Do not hallucinate. "
                    "Do NOT output placeholders like [SOURCE 1]. "
                    "Cite using real section references and URLs from retrieved sources."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{user_query}\n\n"
                    f"Retrieved Sources:\n{context}\n\n"
                    "Provide a concise, accurate answer with citations."
                ),
            },
        ],
    )
    answer = _add_citation_footer(
        completion.choices[0].message.content or "", matches
    )

    return {
        "answer": answer,
        "sources": matches,
        "out_of_scope": False,
    }
