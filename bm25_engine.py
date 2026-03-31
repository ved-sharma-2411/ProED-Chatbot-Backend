from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


Tokenizer = Callable[[str], List[str]]


@dataclass(frozen=True)
class BM25Hit:
    doc_id: str
    score: float
    rank: int
    metadata: Dict[str, Any]


class BM25Index:
    """
    Production-friendly BM25 index.

    Features
    - Configurable tokenizer and BM25 params (k1, b)
    - Fast top-k retrieval over in-memory postings
    - Optional metadata filters at query time
    - Add / update / remove documents
    - Save / load snapshot to JSON
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.tokenizer: Tokenizer = tokenizer or self.default_tokenizer

        # doc_id -> raw text
        self._docs: Dict[str, str] = {}
        # doc_id -> metadata
        self._meta: Dict[str, Dict[str, Any]] = {}

        # doc_id -> term frequency dict
        self._tf: Dict[str, Dict[str, int]] = {}
        # doc_id -> length in tokens
        self._doc_len: Dict[str, int] = {}

        # token -> set(doc_id)
        self._postings: Dict[str, set[str]] = defaultdict(set)
        # token -> document frequency
        self._df: Dict[str, int] = {}
        # token -> inverse document frequency
        self._idf: Dict[str, float] = {}

        self._avgdl: float = 0.0
        self._dirty_stats: bool = True

    @staticmethod
    def default_tokenizer(text: str) -> List[str]:
        # Keeps legal style refs like 668.32 as one token
        return re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)?", (text or "").lower())

    @property
    def size(self) -> int:
        return len(self._docs)

    def add(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not doc_id:
            raise ValueError("doc_id is required")
        if text is None:
            raise ValueError("text is required")

        # Upsert semantics
        if doc_id in self._docs:
            self.remove(doc_id)

        tokens = self.tokenizer(text)
        tf = Counter(tokens)

        self._docs[doc_id] = text
        self._meta[doc_id] = dict(metadata or {})
        self._tf[doc_id] = dict(tf)
        self._doc_len[doc_id] = len(tokens)

        for term in tf.keys():
            self._postings[term].add(doc_id)

        self._dirty_stats = True

    def add_many(self, rows: Iterable[Tuple[str, str, Optional[Dict[str, Any]]]]) -> None:
        for doc_id, text, meta in rows:
            self.add(doc_id=doc_id, text=text, metadata=meta)

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._docs:
            return

        tf = self._tf.get(doc_id, {})
        for term in tf.keys():
            bucket = self._postings.get(term)
            if bucket:
                bucket.discard(doc_id)
                if not bucket:
                    self._postings.pop(term, None)

        self._docs.pop(doc_id, None)
        self._meta.pop(doc_id, None)
        self._tf.pop(doc_id, None)
        self._doc_len.pop(doc_id, None)
        self._dirty_stats = True

    def _recompute_stats_if_needed(self) -> None:
        if not self._dirty_stats:
            return

        N = max(len(self._docs), 1)
        total_len = sum(self._doc_len.values())
        self._avgdl = (total_len / N) if N else 0.0

        self._df = {term: len(doc_ids) for term, doc_ids in self._postings.items()}

        # BM25 idf variant with +1 for stability
        self._idf = {
            term: math.log(1.0 + ((N - df + 0.5) / (df + 0.5)))
            for term, df in self._df.items()
        }

        self._dirty_stats = False

    def _score_doc(self, query_terms: List[str], doc_id: str) -> float:
        tf = self._tf.get(doc_id, {})
        dl = float(self._doc_len.get(doc_id, 0))
        avgdl = self._avgdl if self._avgdl > 0 else 1.0

        score = 0.0
        for term in query_terms:
            f = float(tf.get(term, 0.0))
            if f <= 0.0:
                continue

            idf = float(self._idf.get(term, 0.0))
            denom = f + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
            score += idf * ((f * (self.k1 + 1.0)) / max(denom, 1e-9))

        return score

    def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
        normalize_scores: bool = True,
    ) -> List[BM25Hit]:
        if top_k <= 0:
            return []

        self._recompute_stats_if_needed()

        terms = self.tokenizer(query)
        if not terms or self.size == 0:
            return []

        # Candidate docs = union of postings for all query terms
        candidates: set[str] = set()
        for t in terms:
            candidates |= self._postings.get(t, set())

        if not candidates:
            return []

        def _passes_filter(doc_id: str) -> bool:
            if not metadata_filter:
                return True
            md = self._meta.get(doc_id, {})
            for k, v in metadata_filter.items():
                if md.get(k) != v:
                    return False
            return True

        scored: List[Tuple[str, float]] = []
        for doc_id in candidates:
            if not _passes_filter(doc_id):
                continue
            s = self._score_doc(terms, doc_id)
            if s >= min_score:
                scored.append((doc_id, s))

        if not scored:
            return []

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        max_score = scored[0][1] if scored else 1.0
        max_score = max(max_score, 1e-12)

        hits: List[BM25Hit] = []
        for i, (doc_id, raw_score) in enumerate(scored, start=1):
            score = (raw_score / max_score) if normalize_scores else raw_score
            hits.append(
                BM25Hit(
                    doc_id=doc_id,
                    score=float(score),
                    rank=i,
                    metadata=dict(self._meta.get(doc_id, {})),
                )
            )

        return hits

    def to_dict(self) -> Dict[str, Any]:
        # stats are derivable; persist compactly
        return {
            "k1": self.k1,
            "b": self.b,
            "docs": self._docs,
            "meta": self._meta,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], tokenizer: Optional[Tokenizer] = None) -> "BM25Index":
        idx = cls(k1=float(payload.get("k1", 1.5)), b=float(payload.get("b", 0.75)), tokenizer=tokenizer)

        docs = payload.get("docs", {}) or {}
        meta = payload.get("meta", {}) or {}

        rows = []
        for doc_id, text in docs.items():
            rows.append((str(doc_id), str(text), dict(meta.get(doc_id, {}))))
        idx.add_many(rows)
        return idx

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str, tokenizer: Optional[Tokenizer] = None) -> "BM25Index":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload, tokenizer=tokenizer)


if __name__ == "__main__":
    # Minimal demo
    index = BM25Index()

    docs = [
        (
            "d1",
            "34 CFR 682 covers Federal Family Education Loan program requirements.",
            {"part": "682", "authority_level": "regulation"},
        ),
        (
            "d2",
            "34 CFR 685 covers William D. Ford Direct Loan program.",
            {"part": "685", "authority_level": "regulation"},
        ),
        (
            "d3",
            "Institutional eligibility appears in 34 CFR Part 600.",
            {"part": "600", "authority_level": "regulation"},
        ),
    ]

    index.add_many(docs)

    print("Query: federal family education loan")
    for h in index.search("federal family education loan", top_k=5):
        print(h)

    print("\nQuery with filter: part=682")
    for h in index.search("loan program", top_k=5, metadata_filter={"part": "682"}):
        print(h)
