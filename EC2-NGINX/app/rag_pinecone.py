"""
Production-ready RAG pipeline using Groq + Pinecone (no LangChain).

Features
- Serverless Pinecone index bootstrap
- Batch upsert of vectors with metadata
- Embedding generation (sentence-transformers by default)
- Semantic retrieval
- Context building with source citations
- Final RAG answer generation (Groq chat model)

Environment variables required
- PINECONE_API_KEY
- GROQ_API_KEY (for chat and optional API embeddings)

Example usage
1) Ingest chunks into Pinecone:
   python rag_pinecone.py ingest --input data/chunks.json

2) Ask a question with RAG:
   python rag_pinecone.py ask --question "What is student eligibility?"
"""

from __future__ import annotations

import argparse
import contextlib
import math
import hashlib
import io
import json
import logging
import os
import re
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

try:
	from sentence_transformers import SentenceTransformer
except ImportError:
	SentenceTransformer = None


# ---------------------------
# Configuration
# ---------------------------

INDEX_NAME = "proed-chatbot"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.3-70b-versatile")
INDEX_METRIC = "cosine"
INDEX_CLOUD = "aws"
INDEX_REGION = "us-east-1"
UPSERT_BATCH_SIZE = 100
DEFAULT_TOP_K = 10
_ST_MODEL_CACHE: Dict[str, Any] = {}
_GROQ_CLIENT_CACHE: Optional[OpenAI] = None
_PINECONE_CLIENT_CACHE: Optional[Pinecone] = None
_INDEX_OBJ_CACHE: Dict[str, Any] = {}
_INDEX_DIM_VALIDATED: Dict[str, int] = {}
_BM25_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}
_QUIET_RUNTIME = True
LLM_STATE_PATH = os.getenv("LLM_STATE_PATH", "data/rag_llm_state.json")
LLM_MIN_INTERVAL_SECONDS = int(os.getenv("LLM_MIN_INTERVAL_SECONDS", "45"))
LLM_REQUEST_DELAY_SECONDS = float(os.getenv("LLM_REQUEST_DELAY_SECONDS", "3"))
LLM_CACHE_TTL_SECONDS = int(os.getenv("LLM_CACHE_TTL_SECONDS", "1800"))
NODE_ID_QUERY_RE = re.compile(r"p-(\d+(?:\.\d+)+(?:\([^)]+\))+)", re.IGNORECASE)
BARE_NODE_ID_QUERY_RE = re.compile(r"(\d+(?:\.\d+)+(?:\([^)]+\))+)", re.IGNORECASE)
PART_QUERY_RE = re.compile(r"\b(?:34\s*cfr\s*)?part\s*(\d{3})\b", re.IGNORECASE)
UNKNOWN_ANSWER = "I’m unable to provide a reliable answer from the available knowledge base. Please share more context or a source-specific question."
RAG_MIN_RELEVANCE_SCORE = float(os.getenv("RAG_MIN_RELEVANCE_SCORE", "0.23"))
BM25_CACHE_DIR = os.getenv("BM25_CACHE_DIR", "data/bm25_cache")
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))
HYBRID_CANDIDATE_MULTIPLIER = int(os.getenv("HYBRID_CANDIDATE_MULTIPLIER", "4"))
RRF_K = int(os.getenv("RRF_K", "60"))
NON_KB_QUERY_RE = re.compile(
	r"\b(what\s+is\s+your\s+name|who\s+are\s+you|your\s+name|hello|hi|hey|how\s+are\s+you|good\s+morning|good\s+evening)\b",
	re.IGNORECASE,
)


@dataclass
class ChunkRecord:
	content: str
	section_ref: str
	source_name: str
	source_type: str
	authority_level: str
	source_url: str
	document_title: str = ""
	effective_date: str = ""
	ingestion_date: str = ""
	volume: str = ""
	chapter: str = ""
	title: str = ""
	part: str = ""
	chunk_id: Optional[str] = None
	node_id: str = ""
	path_label: str = ""


def setup_logging(verbose: bool = False) -> None:
	global _QUIET_RUNTIME
	_QUIET_RUNTIME = not verbose

	level = logging.DEBUG if verbose else logging.WARNING
	logging.basicConfig(
		level=level,
		format="%(asctime)s | %(levelname)s | %(message)s",
	)
	if not verbose:
		os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
		os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
		os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
		os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
		warnings.filterwarnings("ignore", message=r".*You are sending unauthenticated requests to the HF Hub.*")
		for noisy in [
			"httpx",
			"httpcore",
			"urllib3",
			"sentence_transformers",
			"transformers",
			"huggingface_hub",
			"huggingface_hub.file_download",
			"model2vec",
		]:
			logging.getLogger(noisy).setLevel(logging.ERROR)


def _run_quietly(func, *args, **kwargs):
	if _QUIET_RUNTIME:
		with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
			return func(*args, **kwargs)
	return func(*args, **kwargs)


def get_required_env(name: str) -> str:
	value = os.getenv(name, "").strip()
	if not value:
		raise EnvironmentError(f"Missing required environment variable: {name}")
	return value


def ensure_parent_dir(file_path: str) -> None:
	parent = os.path.dirname(file_path)
	if parent:
		os.makedirs(parent, exist_ok=True)


def load_llm_state(path: str = LLM_STATE_PATH) -> Dict[str, Any]:
	if not os.path.exists(path):
		return {"last_llm_call_ts": 0.0, "query_cache": {}}
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, dict):
		return {"last_llm_call_ts": 0.0, "query_cache": {}}
	data.setdefault("last_llm_call_ts", 0.0)
	data.setdefault("query_cache", {})
	return data


def save_llm_state(state: Dict[str, Any], path: str = LLM_STATE_PATH) -> None:
	ensure_parent_dir(path)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(state, f, indent=2, ensure_ascii=False)


def normalize_query_for_cache(query: str) -> str:
	return " ".join((query or "").lower().split())


def _namespace_key(namespace: str, index_name: str = INDEX_NAME) -> str:
	return f"{index_name}::{(namespace or 'default').strip()}"


def _bm25_store_path(namespace: str, index_name: str = INDEX_NAME) -> str:
	name = (namespace or "default").strip() or "default"
	os.makedirs(BM25_CACHE_DIR, exist_ok=True)
	return os.path.join(BM25_CACHE_DIR, f"{index_name}__{name}.json")


def _tokenize_bm25(text: str) -> List[str]:
	return re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)?", (text or "").lower())


def _bm25_text_for_doc(md: Dict[str, Any]) -> str:
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
		text = _bm25_text_for_doc(md)
		tokens = _tokenize_bm25(text)
		tf = Counter(tokens)
		doc_tfs.append(dict(tf))
		doc_lens.append(len(tokens))
		for tok in tf.keys():
			df[tok] += 1

	N = max(len(ids), 1)
	avgdl = (sum(doc_lens) / N) if N else 0.0
	idf = {
		tok: math.log(1.0 + ((N - f + 0.5) / (f + 0.5)))
		for tok, f in df.items()
	}

	return {
		"ids": ids,
		"doc_tfs": doc_tfs,
		"doc_lens": doc_lens,
		"avgdl": avgdl,
		"idf": idf,
	}


def _load_bm25_store(namespace: str, index_name: str = INDEX_NAME) -> Dict[str, Dict[str, Any]]:
	path = _bm25_store_path(namespace=namespace, index_name=index_name)
	if not os.path.exists(path):
		return {}
	with open(path, "r", encoding="utf-8") as f:
		obj = json.load(f)
	if not isinstance(obj, dict):
		return {}
	out: Dict[str, Dict[str, Any]] = {}
	for k, v in obj.items():
		if isinstance(k, str) and isinstance(v, dict):
			out[k] = v
	return out


def _save_bm25_store(namespace: str, doc_store: Dict[str, Dict[str, Any]], index_name: str = INDEX_NAME) -> None:
	path = _bm25_store_path(namespace=namespace, index_name=index_name)
	ensure_parent_dir(path)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(doc_store, f, ensure_ascii=False)


def _update_bm25_store(namespace: str, vectors: List[Dict[str, Any]], index_name: str = INDEX_NAME) -> None:
	doc_store = _load_bm25_store(namespace=namespace, index_name=index_name)
	for item in vectors:
		doc_id = str(item.get("id") or "").strip()
		if not doc_id:
			continue
		md = item.get("metadata", {}) or {}
		doc_store[doc_id] = {"metadata": md}

	_save_bm25_store(namespace=namespace, doc_store=doc_store, index_name=index_name)
	_BM25_INDEX_CACHE.pop(_namespace_key(namespace, index_name=index_name), None)


def _get_bm25_index(namespace: str, index_name: str = INDEX_NAME) -> Dict[str, Any]:
	key = _namespace_key(namespace, index_name=index_name)
	if key in _BM25_INDEX_CACHE:
		return _BM25_INDEX_CACHE[key]

	doc_store = _load_bm25_store(namespace=namespace, index_name=index_name)
	index = _build_bm25_index(doc_store)
	index["doc_store"] = doc_store
	_BM25_INDEX_CACHE[key] = index
	return index


def keyword_bm25_query(
	query: str,
	namespace: str,
	top_k: int,
	index_name: str = INDEX_NAME,
	part_hint: str = "",
) -> List[Dict[str, Any]]:
	bm25 = _get_bm25_index(namespace=namespace, index_name=index_name)
	ids = bm25.get("ids", [])
	if not ids:
		return []

	q_tokens = _tokenize_bm25(query)
	if not q_tokens:
		return []

	idf = bm25.get("idf", {})
	doc_tfs = bm25.get("doc_tfs", [])
	doc_lens = bm25.get("doc_lens", [])
	avgdl = float(bm25.get("avgdl") or 0.0) or 1.0
	doc_store = bm25.get("doc_store", {})

	scored: List[tuple[int, float]] = []
	for i, doc_id in enumerate(ids):
		rec = doc_store.get(doc_id, {})
		md = rec.get("metadata", {}) or {}
		doc_part = infer_part_from_metadata(md)
		if part_hint and doc_part and doc_part != part_hint:
			continue

		tf = doc_tfs[i] if i < len(doc_tfs) else {}
		dl = float(doc_lens[i]) if i < len(doc_lens) else 0.0
		score = 0.0
		for tok in q_tokens:
			f = float(tf.get(tok, 0.0))
			if f <= 0:
				continue
			tok_idf = float(idf.get(tok, 0.0))
			den = f + BM25_K1 * (1.0 - BM25_B + BM25_B * (dl / avgdl))
			score += tok_idf * ((f * (BM25_K1 + 1.0)) / max(den, 1e-9))

		if score > 0:
			scored.append((i, score))

	if not scored:
		return []

	scored.sort(key=lambda x: x[1], reverse=True)
	max_score = max(sc for _, sc in scored) or 1.0

	out: List[Dict[str, Any]] = []
	for rank, (idx, score) in enumerate(scored[:top_k], start=1):
		doc_id = ids[idx]
		rec = doc_store.get(doc_id, {})
		md = rec.get("metadata", {}) or {}
		out.append(
			{
				"id": doc_id,
				"score": float(score / max_score),
				"metadata": md,
				"_keyword_rank": rank,
			}
		)

	return out


def rrf_merge(
	semantic_matches: List[Dict[str, Any]],
	keyword_matches: List[Dict[str, Any]],
	top_k: int,
	rrf_k: int = RRF_K,
) -> List[Dict[str, Any]]:
	merged: Dict[str, Dict[str, Any]] = {}

	for rank, item in enumerate(semantic_matches, start=1):
		doc_id = str(item.get("id") or "")
		if not doc_id:
			continue
		rec = merged.setdefault(
			doc_id,
			{
				"id": doc_id,
				"metadata": item.get("metadata", {}) or {},
				"_rrf": 0.0,
				"_sem_rank": None,
				"_key_rank": None,
			},
		)
		rec["_rrf"] += 1.0 / (rrf_k + rank)
		rec["_sem_rank"] = rank

	for rank, item in enumerate(keyword_matches, start=1):
		doc_id = str(item.get("id") or "")
		if not doc_id:
			continue
		rec = merged.setdefault(
			doc_id,
			{
				"id": doc_id,
				"metadata": item.get("metadata", {}) or {},
				"_rrf": 0.0,
				"_sem_rank": None,
				"_key_rank": None,
			},
		)
		rec["_rrf"] += 1.0 / (rrf_k + rank)
		rec["_key_rank"] = rank

	if not merged:
		return []

	items = list(merged.values())
	items.sort(key=lambda x: x.get("_rrf", 0.0), reverse=True)
	max_rrf = max(float(x.get("_rrf") or 0.0) for x in items) or 1.0

	out: List[Dict[str, Any]] = []
	for rec in items[: max(top_k * 2, top_k)]:
		out.append(
			{
				"id": rec["id"],
				"score": float(rec.get("_rrf", 0.0)) / max_rrf,
				"metadata": rec.get("metadata", {}) or {},
				"_rrf_debug": {
					"rrf": round(float(rec.get("_rrf", 0.0)), 8),
					"sem_rank": rec.get("_sem_rank"),
					"key_rank": rec.get("_key_rank"),
				},
			}
		)

	return out


def build_retrieval_only_answer(matches: List[Dict[str, Any]]) -> str:
	if not matches:
		return UNKNOWN_ANSWER

	lines = ["LLM response is rate-limited. Retrieval-only result:"]
	for i, item in enumerate(matches[:5], start=1):
		md = item.get("metadata", {})
		citation = _format_citation_from_metadata(md)
		content = (md.get("content", "") or "").strip()
		snippet = content[:280] + ("..." if len(content) > 280 else "")
		lines.append(f"[{i}] {citation}")
		lines.append(f"{snippet}")

	return "\n".join(lines)


def should_return_unknown_answer(matches: List[Dict[str, Any]]) -> bool:
	if not matches:
		return True

	top = matches[0]
	debug_rank = top.get("_debug_rank", {}) if isinstance(top, dict) else {}
	if isinstance(debug_rank, dict) and "final" in debug_rank:
		top_score = float(debug_rank.get("final") or 0.0)
	else:
		top_score = float(top.get("score") or 0.0)

	return top_score < RAG_MIN_RELEVANCE_SCORE


def is_non_kb_query(query: str) -> bool:
	q = (query or "").strip()
	if not q:
		return True
	if not NON_KB_QUERY_RE.search(q):
		return False
	stripped = NON_KB_QUERY_RE.sub(" ", q)
	stripped = re.sub(r"[^a-zA-Z0-9]+", " ", stripped).strip().lower()
	if stripped:
		return False
	return True


def build_source_label(md: Dict[str, Any]) -> str:
	source_name = str(md.get("source_name") or "").strip()
	if source_name:
		return source_name

	source_url = str(md.get("source_url") or "").strip()
	if source_url:
		match = re.search(r"https?://([^/]+)", source_url)
		if match:
			return match.group(1).replace("www.", "")

	return "source"


def infer_source_url(md: Dict[str, Any]) -> str:
	source_url = str(md.get("source_url") or "").strip()
	if source_url:
		return source_url

	node_id = str(md.get("node_id") or "")
	match = re.search(r"p-(\d+(?:\.\d+)+)", node_id)
	if match:
		section = match.group(1)
		return f"https://www.ecfr.gov/current/title-34/subtitle-B/chapter-VI/part-668/section-{section}"

	section_ref = str(md.get("section_ref") or "")
	match2 = re.search(r"(\d+(?:\.\d+)+)", section_ref)
	if match2:
		section = match2.group(1)
		return f"https://www.ecfr.gov/current/title-34/subtitle-B/chapter-VI/part-668/section-{section}"

	return ""


def infer_part_from_metadata(md: Dict[str, Any]) -> str:
	part = str(md.get("part") or "").strip()
	if part:
		return part

	section_ref = str(md.get("section_ref") or "")
	m = re.search(r"(\d+)\.(\d+)", section_ref)
	if m:
		return m.group(1)

	node_id = str(md.get("node_id") or "")
	m2 = re.search(r"p-(\d+)\.(\d+)", node_id)
	if m2:
		return m2.group(1)

	source_url = str(md.get("source_url") or "")
	m3 = re.search(r"section-(\d+)\.(\d+)", source_url)
	if m3:
		return m3.group(1)

	return ""


def extract_part_hint(query: str) -> str:
	q = (query or "").lower().strip()
	if not q:
		return ""

	m = PART_QUERY_RE.search(q)
	if m:
		return m.group(1)

	if "federal family education loan" in q or "ffel" in q:
		return "682"
	if "direct loan" in q:
		return "685"

	return ""


def load_dotenv(dotenv_path: str = ".env") -> None:
	if not os.path.exists(dotenv_path):
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


def build_groq_client() -> OpenAI:
	global _GROQ_CLIENT_CACHE
	if _GROQ_CLIENT_CACHE is not None:
		return _GROQ_CLIENT_CACHE

	api_key = get_required_env("GROQ_API_KEY")
	base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip()
	_GROQ_CLIENT_CACHE = OpenAI(api_key=api_key, base_url=base_url)
	return _GROQ_CLIENT_CACHE


def build_chat_client_and_model() -> tuple[OpenAI, str]:
	chat_model = CHAT_MODEL
	return build_groq_client(), chat_model


def build_pinecone_client() -> Pinecone:
	global _PINECONE_CLIENT_CACHE
	if _PINECONE_CLIENT_CACHE is not None:
		return _PINECONE_CLIENT_CACHE

	api_key = get_required_env("PINECONE_API_KEY")
	_PINECONE_CLIENT_CACHE = Pinecone(api_key=api_key)
	return _PINECONE_CLIENT_CACHE


def ensure_index(pc: Pinecone, index_name: str = INDEX_NAME, dimension: int = 3072) -> None:
	listed = pc.list_indexes()
	if hasattr(listed, "names"):
		existing = set(listed.names())
	else:
		existing = {
			(idx.get("name") if isinstance(idx, dict) else getattr(idx, "name", None))
			for idx in listed
		}
		existing.discard(None)

	if index_name in existing:
		desc = pc.describe_index(index_name)
		idx_dim = getattr(desc, "dimension", None)
		if idx_dim is None and isinstance(desc, dict):
			idx_dim = desc.get("dimension")
		if idx_dim is not None and int(idx_dim) != int(dimension):
			raise ValueError(
				f"Index '{index_name}' dimension mismatch: existing={idx_dim}, expected={dimension}. "
				"Use a different index name or recreate index with correct dimension."
			)
		logging.info("Pinecone index exists: %s", index_name)
		return

	logging.info("Creating Pinecone index: %s", index_name)
	pc.create_index(
		name=index_name,
		dimension=dimension,
		metric=INDEX_METRIC,
		spec=ServerlessSpec(cloud=INDEX_CLOUD, region=INDEX_REGION),
	)

	# Wait for readiness
	for _ in range(60):
		desc = pc.describe_index(index_name)
		status_obj = getattr(desc, "status", None)
		if isinstance(status_obj, dict):
			status = bool(status_obj.get("ready", False))
		else:
			status = bool(getattr(status_obj, "ready", False))
		if status:
			logging.info("Index is ready: %s", index_name)
			return
		time.sleep(2)

	raise TimeoutError(f"Index creation timed out for: {index_name}")


def get_index(pc: Pinecone, index_name: str = INDEX_NAME):
	return pc.Index(index_name)


def ensure_index_once(index_name: str, dimension: int) -> Any:
	validated_dim = _INDEX_DIM_VALIDATED.get(index_name)
	if validated_dim == int(dimension):
		cached = _INDEX_OBJ_CACHE.get(index_name)
		if cached is not None:
			return cached

	pc = build_pinecone_client()
	ensure_index(pc, index_name=index_name, dimension=dimension)
	index = get_index(pc, index_name)
	_INDEX_DIM_VALIDATED[index_name] = int(dimension)
	_INDEX_OBJ_CACHE[index_name] = index
	return index


def is_sentence_transformer_model(model: str) -> bool:
	return model.strip().lower().startswith("sentence-transformers/")


def get_sentence_transformer(model: str):
	if SentenceTransformer is None:
		raise ImportError(
			"sentence-transformers is required for this embedding model. "
			"Install with: pip install sentence-transformers"
		)
	if model not in _ST_MODEL_CACHE:
		_ST_MODEL_CACHE[model] = _run_quietly(SentenceTransformer, model)
	return _ST_MODEL_CACHE[model]


def build_embedding_client(model: str):
	if is_sentence_transformer_model(model):
		return None
	return build_groq_client()


# ---------------------------
# Input parsing
# ---------------------------

def _stable_chunk_id(content: str, section_ref: str, source_url: str) -> str:
	base = f"{section_ref}|{source_url}|{content}".encode("utf-8")
	return hashlib.sha256(base).hexdigest()[:32]


def _to_chunk_record(item: Dict[str, Any]) -> ChunkRecord:
	content = (item.get("embedding_text") or item.get("content") or item.get("text_no_overlap") or item.get("text") or "").strip()
	section_ref = (item.get("section_ref") or item.get("path_label") or item.get("section") or "").strip()
	path_label = str(item.get("path_label") or "")
	node_id = str(item.get("node_id") or "")
	source_name = str(item.get("source_name") or item.get("metadata", {}).get("source_name") or "")
	source_type = str(item.get("source_type") or item.get("metadata", {}).get("source_type") or "unknown")
	authority_level = str(item.get("authority_level") or item.get("metadata", {}).get("authority_level") or "unknown")
	source_url = str(item.get("source_url") or item.get("metadata", {}).get("source_url") or "")
	document_title = str(item.get("document_title") or item.get("metadata", {}).get("document_title") or "")
	effective_date = str(item.get("effective_date") or item.get("metadata", {}).get("effective_date") or "")
	ingestion_date = str(item.get("ingestion_date") or item.get("metadata", {}).get("ingestion_date") or "")
	volume = str(item.get("volume") or item.get("metadata", {}).get("volume") or "")
	chapter = str(item.get("chapter") or item.get("metadata", {}).get("chapter") or "")
	title = str(item.get("title") or item.get("metadata", {}).get("title") or "")
	part = str(item.get("part") or item.get("metadata", {}).get("part") or "")
	provided_id = item.get("id") or item.get("chunk_id")

	if not content:
		raise ValueError("Chunk content is required")

	chunk_id = str(provided_id) if provided_id else _stable_chunk_id(content, section_ref, source_url)
	return ChunkRecord(
		content=content,
		section_ref=section_ref,
		source_name=source_name,
		source_type=source_type,
		authority_level=authority_level,
		source_url=source_url,
		document_title=document_title,
		effective_date=effective_date,
		ingestion_date=ingestion_date,
		volume=volume,
		chapter=chapter,
		title=title,
		part=part,
		chunk_id=chunk_id,
		node_id=node_id,
		path_label=path_label,
	)


def load_chunk_records(input_path: str, chunk_source: str = "base_chunks") -> List[ChunkRecord]:
	with open(input_path, "r", encoding="utf-8") as f:
		raw = json.load(f)

	if isinstance(raw, list):
		items = raw
	elif isinstance(raw, dict):
		if chunk_source == "auto":
			items = raw.get("level_chunks") or raw.get("base_chunks") or raw.get("chunks") or []
		else:
			items = raw.get(chunk_source) or []
			if not items:
				items = raw.get("level_chunks") or raw.get("base_chunks") or raw.get("chunks") or []
	else:
		raise ValueError("Unsupported input JSON structure")

	if not isinstance(items, list) or not items:
		raise ValueError("No chunks found in input file")

	records: List[ChunkRecord] = []
	skipped = 0
	blocked_markers = (
		"aggressive automated scraping",
		"complete the captcha",
		"request access",
		"programmatic access to these sites is limited",
	)
	for i, item in enumerate(items):
		try:
			rec = _to_chunk_record(item)
			low = rec.content.lower()
			if any(m in low for m in blocked_markers):
				skipped += 1
				logging.warning("Skipping blocked/anti-bot chunk %s", i)
				continue
			records.append(rec)
		except Exception as exc:
			skipped += 1
			logging.warning("Skipping invalid chunk %s: %s", i, exc)

	if not records:
		raise ValueError("No valid chunks after validation (input may be blocked/captcha HTML)")

	logging.info("Loaded %s chunks (skipped=%s)", len(records), skipped)
	return records


# ---------------------------
# Embeddings
# ---------------------------

def embed_texts(
	api_client: Optional[Any],
	texts: List[str],
	model: str = EMBEDDING_MODEL,
	max_retries: int = 4,
	retry_base_seconds: float = 1.5,
) -> List[List[float]]:
	if is_sentence_transformer_model(model):
		st_model = get_sentence_transformer(model)
		vectors = _run_quietly(st_model.encode, texts, convert_to_numpy=True, show_progress_bar=False)
		return [v.tolist() for v in vectors]

	if api_client is None:
		raise ValueError("api_client is required for non sentence-transformers embedding models")

	for attempt in range(max_retries + 1):
		try:
			resp = api_client.embeddings.create(model=model, input=texts)
			vectors = [d.embedding for d in resp.data]
			if len(vectors) != len(texts):
				raise RuntimeError("Embedding response size mismatch")
			return vectors
		except Exception:
			if attempt >= max_retries:
				raise
			sleep_s = retry_base_seconds * (2 ** attempt)
			logging.warning("Embedding retry %s/%s in %.1fs", attempt + 1, max_retries, sleep_s)
			time.sleep(sleep_s)


def embed_query(api_client: Optional[Any], query: str, model: str = EMBEDDING_MODEL) -> List[float]:
	return embed_texts(api_client, [query], model=model)[0]


# ---------------------------
# Pinecone upsert
# ---------------------------

def _iter_batches(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
	for i in range(0, len(items), batch_size):
		yield items[i : i + batch_size]


def prepare_vectors(records: List[ChunkRecord], vectors: List[List[float]]) -> List[Dict[str, Any]]:
	if len(records) != len(vectors):
		raise ValueError("records and vectors length mismatch")

	out: List[Dict[str, Any]] = []
	for rec, vec in zip(records, vectors):
		if not vec:
			raise ValueError(f"Empty vector for id={rec.chunk_id}")

		out.append(
			{
				"id": rec.chunk_id,
				"values": vec,
				"metadata": {
					"content": rec.content,
					"section_ref": rec.section_ref,
					"node_id": rec.node_id,
					"path_label": rec.path_label,
					"source_name": rec.source_name,
					"source_type": rec.source_type,
					"authority_level": rec.authority_level,
					"source_url": rec.source_url,
					"document_title": rec.document_title,
					"effective_date": rec.effective_date,
					"ingestion_date": rec.ingestion_date,
					"volume": rec.volume,
					"chapter": rec.chapter,
					"title": rec.title,
					"part": rec.part,
				},
			}
		)
	return out


def upsert_vectors(
	index,
	vectors: List[Dict[str, Any]],
	batch_size: int = UPSERT_BATCH_SIZE,
	namespace: str = "default",
) -> None:
	total = len(vectors)
	done = 0
	for batch in _iter_batches(vectors, batch_size):
		index.upsert(vectors=batch, namespace=namespace)
		done += len(batch)
		logging.info("Upserted %s/%s vectors", done, total)


def ingest_chunks(
	input_path: str,
	namespace: str = "default",
	chunk_source: str = "base_chunks",
	index_name: str = INDEX_NAME,
	embedding_model: str = EMBEDDING_MODEL,
) -> None:
	api_client = build_embedding_client(embedding_model)
	pc = build_pinecone_client()

	records = load_chunk_records(input_path, chunk_source=chunk_source)
	probe_vec = embed_texts(api_client, [records[0].content], model=embedding_model)[0]
	vector_dim = len(probe_vec)
	ensure_index(pc, index_name, dimension=vector_dim)
	index = get_index(pc, index_name)

	all_vectors: List[Dict[str, Any]] = []
	for batch in _iter_batches(records, UPSERT_BATCH_SIZE):
		texts = [r.content for r in batch]
		embeddings = embed_texts(api_client, texts, model=embedding_model)
		all_vectors.extend(prepare_vectors(batch, embeddings))

	upsert_vectors(
		index=index,
		vectors=all_vectors,
		batch_size=UPSERT_BATCH_SIZE,
		namespace=namespace,
	)
	_update_bm25_store(namespace=namespace, vectors=all_vectors, index_name=index_name)
	logging.info("Ingestion complete. Namespace=%s, vectors=%s", namespace, len(all_vectors))


def extract_explicit_node_id(query: str) -> str:
	q = (query or "").strip()
	q_compact = re.sub(r"\s+", "", q)

	match = NODE_ID_QUERY_RE.search(q_compact)
	if match:
		return f"p-{match.group(1)}"

	bare = BARE_NODE_ID_QUERY_RE.search(q_compact)
	if bare:
		return f"p-{bare.group(1)}"
	return ""


# ---------------------------
# Retrieval + RAG
# ---------------------------

def semantic_query(
	query: str,
	top_k: int = DEFAULT_TOP_K,
	namespace: str = "default",
	index_name: str = INDEX_NAME,
	embedding_model: str = EMBEDDING_MODEL,
) -> List[Dict[str, Any]]:
	if is_non_kb_query(query):
		logging.info("Detected non-knowledge-base query; returning no matches.")
		return []

	api_client = build_embedding_client(embedding_model)
	qvec = embed_query(api_client, query, model=embedding_model)
	index = ensure_index_once(index_name, dimension=len(qvec))
	explicit_node_id = extract_explicit_node_id(query)
	part_hint = extract_part_hint(query)

	if explicit_node_id:
		exact_result = index.query(
			vector=qvec,
			top_k=max(top_k, 3),
			include_values=False,
			include_metadata=True,
			namespace=namespace,
			filter={"node_id": {"$eq": explicit_node_id}},
		)
		exact_matches = exact_result.get("matches", []) if isinstance(exact_result, dict) else getattr(exact_result, "matches", [])
		if exact_matches:
			result_matches = exact_matches
		else:
			candidate_k = max(top_k * 3, 20)
			if part_hint:
				part_result = index.query(
					vector=qvec,
					top_k=candidate_k,
					include_values=False,
					include_metadata=True,
					namespace=namespace,
					filter={"part": {"$eq": part_hint}},
				)
				part_matches = part_result.get("matches", []) if isinstance(part_result, dict) else getattr(part_result, "matches", [])
				if part_matches:
					result_matches = part_matches
				else:
					result = index.query(
						vector=qvec,
						top_k=candidate_k,
						include_values=False,
						include_metadata=True,
						namespace=namespace,
					)
					result_matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", [])
			else:
				result = index.query(
					vector=qvec,
					top_k=candidate_k,
					include_values=False,
					include_metadata=True,
					namespace=namespace,
				)
				result_matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", [])
	else:
		candidate_k = max(top_k * 3, 20)
		if part_hint:
			part_result = index.query(
				vector=qvec,
				top_k=candidate_k,
				include_values=False,
				include_metadata=True,
				namespace=namespace,
				filter={"part": {"$eq": part_hint}},
			)
			part_matches = part_result.get("matches", []) if isinstance(part_result, dict) else getattr(part_result, "matches", [])
			if part_matches:
				result_matches = part_matches
			else:
				result = index.query(
					vector=qvec,
					top_k=candidate_k,
					include_values=False,
					include_metadata=True,
					namespace=namespace,
				)
				result_matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", [])
		else:
			result = index.query(
				vector=qvec,
				top_k=candidate_k,
				include_values=False,
				include_metadata=True,
				namespace=namespace,
			)
			result_matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", [])

	matches = []
	for m in result_matches:
		m_id = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
		m_score = m.get("score") if isinstance(m, dict) else getattr(m, "score", None)
		m_md = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {})
		if isinstance(m_md, dict) and not str(m_md.get("part") or "").strip():
			m_md["part"] = infer_part_from_metadata(m_md)
		matches.append(
			{
				"id": m_id,
				"score": m_score,
				"metadata": m_md or {},
			}
		)
	keyword_k = max(top_k * HYBRID_CANDIDATE_MULTIPLIER, 20)
	keyword_matches = keyword_bm25_query(
		query=query,
		namespace=namespace,
		top_k=keyword_k,
		index_name=index_name,
		part_hint=part_hint,
	)

	hybrid_matches = rrf_merge(
		semantic_matches=matches,
		keyword_matches=keyword_matches,
		top_k=max(top_k * 2, top_k),
	)

	ranked = rerank_matches(query=query, matches=hybrid_matches, top_k=top_k)
	if should_return_unknown_answer(ranked):
		return []
	return ranked


def warmup_runtime(
	index_name: str = INDEX_NAME,
	embedding_model: str = EMBEDDING_MODEL,
) -> None:
	"""Warm caches to reduce first-request latency in API mode."""
	api_client = build_embedding_client(embedding_model)
	probe = embed_query(api_client, "warmup", model=embedding_model)
	ensure_index_once(index_name, dimension=len(probe))


def build_context(matches: List[Dict[str, Any]]) -> str:
	blocks: List[str] = []
	for i, item in enumerate(matches, start=1):
		md = item.get("metadata", {})
		section = md.get("section_ref", "")
		source_name = md.get("source_name", "")
		content = md.get("content", "")
		source_type = md.get("source_type", "")
		authority_level = md.get("authority_level", "")
		source_url = md.get("source_url", "")

		blocks.append(
			"\n".join(
				[
					f"[SOURCE {i}]",
					f"Section: {section}",
					f"Source Name: {source_name}",
					f"Source Type: {source_type}",
					f"Authority Level: {authority_level}",
					f"Source URL: {source_url}",
					f"Content: {content}",
				]
			)
		)
	return "\n\n".join(blocks)


def _format_citation_from_metadata(md: Dict[str, Any]) -> str:
	section = str(md.get("section_ref") or "").strip()
	source_name = build_source_label(md)
	source_url = infer_source_url(md)

	label = section or source_name or "source"
	if source_url:
		return f"{label} ({source_url})"
	return label


def _replace_source_placeholders(answer: str, matches: List[Dict[str, Any]]) -> str:
	if not answer:
		return answer

	def _repl(m: re.Match) -> str:
		idx = int(m.group(1)) - 1
		if 0 <= idx < len(matches):
			md = matches[idx].get("metadata", {}) or {}
			return _format_citation_from_metadata(md)
		return "source"

	return re.sub(r"\[\s*source\s*(\d+)\s*\]", _repl, answer, flags=re.IGNORECASE)


def _build_citation_footer(matches: List[Dict[str, Any]], max_items: int = 3) -> str:
	if not matches:
		return ""

	lines = []
	for i, item in enumerate(matches[:max_items], start=1):
		md = item.get("metadata", {}) or {}
		lines.append(f"- {i}. {_format_citation_from_metadata(md)}")

	return "\n".join(lines)


def _normalize_answer_with_citations(answer: str, matches: List[Dict[str, Any]]) -> str:
	text = (answer or "").strip()
	if not text:
		return text

	text = _replace_source_placeholders(text, matches)

	# If the model did not include concrete citations, append deterministic citations.
	has_url = "http://" in text.lower() or "https://" in text.lower()
	has_section = "§" in text
	if not (has_url or has_section):
		footer = _build_citation_footer(matches)
		if footer:
			text = f"{text}\n\nCitations:\n{footer}"

	return text


def _tokenize_for_rank(text: str) -> List[str]:
	return re.findall(r"[a-z0-9\.]+", (text or "").lower())


def _extract_cfr_ref(query: str) -> str:
	q = (query or "").lower()
	match = re.search(r"\b\d+\s*cfr\s*\d+(?:\.\d+)+\b", q)
	if match:
		return match.group(0)

	match2 = re.search(r"\b\d+(?:\.\d+)+\b", q)
	return match2.group(0) if match2 else ""


def rerank_matches(query: str, matches: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
	q_tokens = set(_tokenize_for_rank(query))
	cfr_ref = _extract_cfr_ref(query)
	explicit_node_id = extract_explicit_node_id(query)
	part_hint = extract_part_hint(query)
	authority_boost = {
		"statute": 0.06,
		"regulation": 0.05,
		"sub_regulatory": 0.02,
		"high": 0.05,
		"medium": 0.03,
		"low": 0.01,
		"unknown": 0.0,
	}

	ranked = []
	for item in matches:
		md = item.get("metadata", {}) or {}
		content = str(md.get("content", ""))
		section_ref = str(md.get("section_ref", ""))
		node_id = str(md.get("node_id", ""))
		source_url = str(md.get("source_url", ""))
		doc_part = infer_part_from_metadata(md)
		authority = str(md.get("authority_level", "unknown")).strip().lower()
		base = float(item.get("score") or 0.0)

		doc_tokens = set(_tokenize_for_rank(f"{section_ref} {content}"))
		overlap = len(q_tokens & doc_tokens)
		keyword_score = overlap / max(len(q_tokens), 1)

		early_text = content[:500]
		early_tokens = set(_tokenize_for_rank(f"{section_ref} {early_text}"))
		early_overlap = len(q_tokens & early_tokens)
		early_boost = 0.05 * (early_overlap / max(len(q_tokens), 1))

		citation_boost = 0.0
		if cfr_ref:
			if cfr_ref in content.lower() or cfr_ref in section_ref.lower():
				citation_boost += 0.07
			elif cfr_ref.replace(" ", "") in (content.lower() + section_ref.lower()).replace(" ", ""):
				citation_boost += 0.05

		meta_boost = 0.0
		if source_url:
			meta_boost += 0.02
		meta_boost += authority_boost.get(authority, 0.0)

		explicit_boost = 0.0
		if explicit_node_id:
			if node_id.lower() == explicit_node_id.lower():
				explicit_boost += 0.35
			elif explicit_node_id.lower() in content.lower() or explicit_node_id.lower() in section_ref.lower():
				explicit_boost += 0.10
			if "merged" in section_ref.lower():
				explicit_boost -= 0.05

		long_penalty = 0.0
		if len(content) > 2600:
			long_penalty = min((len(content) - 2600) / 10000.0, 0.08)

		part_boost = 0.0
		if part_hint:
			if doc_part == part_hint:
				part_boost += 0.22
			else:
				part_boost -= 0.08

		final_score = (0.70 * base) + (0.25 * keyword_score) + citation_boost + meta_boost + explicit_boost + early_boost + part_boost - long_penalty
		ranked.append(
			{
				**item,
				"_debug_rank": {
					"base": round(base, 6),
					"keyword": round(keyword_score, 6),
					"early_boost": round(early_boost, 6),
					"citation_boost": round(citation_boost, 6),
					"meta_boost": round(meta_boost, 6),
					"explicit_boost": round(explicit_boost, 6),
					"part_boost": round(part_boost, 6),
					"long_penalty": round(long_penalty, 6),
					"final": round(final_score, 6),
				},
			}
		)

	ranked.sort(key=lambda x: x.get("_debug_rank", {}).get("final", 0.0), reverse=True)
	return ranked[:top_k]


def rag_answer(
	user_query: str,
	top_k: int = DEFAULT_TOP_K,
	namespace: str = "default",
	force_llm: bool = False,
	prefetched_matches: Optional[List[Dict[str, Any]]] = None,
	index_name: str = INDEX_NAME,
	embedding_model: str = EMBEDDING_MODEL,
) -> Dict[str, Any]:
	matches = prefetched_matches if prefetched_matches is not None else semantic_query(
		query=user_query,
		top_k=top_k,
		namespace=namespace,
		index_name=index_name,
		embedding_model=embedding_model,
	)
	context = build_context(matches)
	if should_return_unknown_answer(matches):
		return {
			"answer": UNKNOWN_ANSWER,
			"sources": matches,
			"context": context,
			"from_cache": False,
			"out_of_scope": True,
		}

	state = load_llm_state()
	now_ts = time.time()
	cache_key = normalize_query_for_cache(user_query)
	cached = state.get("query_cache", {}).get(cache_key)

	if cached:
		cached_ts = float(cached.get("ts", 0))
		if now_ts - cached_ts <= LLM_CACHE_TTL_SECONDS:
			return {
				"answer": cached.get("answer", ""),
				"sources": matches,
				"context": context,
				"from_cache": True,
			}

	last_llm_call_ts = float(state.get("last_llm_call_ts", 0.0))
	elapsed = now_ts - last_llm_call_ts
	if not force_llm and elapsed < LLM_MIN_INTERVAL_SECONDS:
		fallback_answer = build_retrieval_only_answer(matches)
		return {
			"answer": fallback_answer,
			"sources": matches,
			"context": context,
			"from_cache": False,
			"llm_skipped": True,
			"retry_after_seconds": int(LLM_MIN_INTERVAL_SECONDS - elapsed),
		}

	chat_client, chat_model = build_chat_client_and_model()
	system_prompt = (
		"Answer ONLY using provided sources. Do not hallucinate. "
		"Do NOT output placeholders like [SOURCE 1]. "
		"Cite using real section references and URLs from retrieved sources."
	)

	user_prompt = (
		f"Question:\n{user_query}\n\n"
		f"Retrieved Sources:\n{context}\n\n"
		"Provide a concise, accurate answer with citations."
	)

	if LLM_REQUEST_DELAY_SECONDS > 0:
		logging.info("Applying LLM delay: %.1fs", LLM_REQUEST_DELAY_SECONDS)
		time.sleep(LLM_REQUEST_DELAY_SECONDS)

	try:
		completion = chat_client.chat.completions.create(
			model=chat_model,
			temperature=0,
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
		)
		answer = completion.choices[0].message.content or ""
		answer = _normalize_answer_with_citations(answer, matches)

		state["last_llm_call_ts"] = time.time()
		state.setdefault("query_cache", {})[cache_key] = {
			"answer": answer,
			"ts": state["last_llm_call_ts"],
		}
		save_llm_state(state)
	except Exception as exc:
		logging.warning("LLM call failed, returning retrieval-only answer: %s", exc)
		state["last_llm_call_ts"] = time.time()
		save_llm_state(state)
		answer = build_retrieval_only_answer(matches)
	return {
		"answer": answer,
		"sources": matches,
		"context": context,
	}


# ---------------------------
# CLI
# ---------------------------

def build_cli() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Production RAG with Groq + Pinecone")
	parser.add_argument("--verbose", action="store_true", help="Enable debug logs")

	sub = parser.add_subparsers(dest="command", required=True)

	ingest_p = sub.add_parser("ingest", help="Embed and upsert chunks into Pinecone")
	ingest_p.add_argument("--input", required=True, help="Path to chunks JSON")
	ingest_p.add_argument("--namespace", default="default", help="Pinecone namespace")
	ingest_p.add_argument(
		"--chunk-source",
		choices=["level_chunks", "base_chunks", "auto"],
		default="base_chunks",
		help="Chunk list to ingest from input JSON",
	)

	query_p = sub.add_parser("query", help="Run semantic search only")
	query_p.add_argument("--question", required=True, help="User query")
	query_p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
	query_p.add_argument("--namespace", default="default")

	ask_p = sub.add_parser("ask", help="Run full RAG answer")
	ask_p.add_argument("--question", required=True, help="User query")
	ask_p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
	ask_p.add_argument("--namespace", default="default")
	ask_p.add_argument(
		"--force-llm",
		action="store_true",
		help="Force LLM call even if cooldown is active",
	)

	return parser


def main() -> None:
	load_dotenv()
	parser = build_cli()
	args = parser.parse_args()
	setup_logging(verbose=args.verbose)

	if args.command == "ingest":
		ingest_chunks(input_path=args.input, namespace=args.namespace, chunk_source=args.chunk_source)
		return

	if args.command == "query":
		print(f"Searching for: '{args.question}'")
		matches = semantic_query(query=args.question, top_k=args.top_k, namespace=args.namespace)
		print("\nRetrieved sources:")
		if not matches:
			print("- None")
			print("\nAnswer:")
			print(UNKNOWN_ANSWER)
			return

		for i, item in enumerate(matches[: args.top_k], start=1):
			md = item.get("metadata", {}) or {}
			label = md.get("node_id") or md.get("section_ref") or str(item.get("id") or "unknown")
			source_name = build_source_label(md)
			source_url = infer_source_url(md)
			score = float(item.get("score") or 0.0)
			if source_url:
				print(f"  {i}. {source_name} | {label} | Score: {score:.2f} | {source_url}")
			else:
				print(f"  {i}. {source_name} | {label} | Score: {score:.2f}")

		top_md = matches[0].get("metadata", {}) or {}
		top_text = str(top_md.get("content") or "").strip()
		print("\nAnswer:")
		print(top_text if top_text else UNKNOWN_ANSWER)
		return

	if args.command == "ask":
		print(f"Searching for: {args.question}")
		matches = semantic_query(query=args.question, top_k=args.top_k, namespace=args.namespace)

		print("Retrieved sources:")
		if not matches:
			print("- None")
			print("\nAnswer:")
			print(UNKNOWN_ANSWER)
			return

		for i, item in enumerate(matches[: args.top_k], start=1):
			md = item.get("metadata", {}) or {}
			label = md.get("node_id") or md.get("section_ref") or str(item.get("id") or "unknown")
			source_name = build_source_label(md)
			source_url = infer_source_url(md)
			if source_url:
				print(f"- [{i}] {source_name} | {label} | {source_url}")
			else:
				print(f"- [{i}] {source_name} | {label}")

		print(f"\nAsking Groq ({CHAT_MODEL})...")
		result = rag_answer(
			user_query=args.question,
			top_k=args.top_k,
			namespace=args.namespace,
			force_llm=args.force_llm,
			prefetched_matches=matches,
		)
		print("\nAnswer:")
		print(result["answer"])
		return

	raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
	main()
