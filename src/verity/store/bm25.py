"""BM25 Okapi lexical index. The sparse half of hybrid retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from verity.models import Chunk, ScoredChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_freq: dict[str, int] = defaultdict(int)
        self._avg_len: float = 0.0

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._doc_tokens = [tokenize(c.text) for c in chunks]
        self._doc_freq = defaultdict(int)
        for tokens in self._doc_tokens:
            for term in set(tokens):
                self._doc_freq[term] += 1
        total = sum(len(t) for t in self._doc_tokens)
        self._avg_len = total / len(self._doc_tokens) if self._doc_tokens else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._doc_tokens)
        df = self._doc_freq.get(term, 0)
        return max(0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0))

    def _score(self, query_terms: list[str], tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        freqs = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for term in query_terms:
            tf = freqs.get(term, 0)
            if tf == 0:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1.0))
            score += self._idf(term) * (tf * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        query_terms = tokenize(query)
        scored = [
            ScoredChunk(chunk=chunk, score=self._score(query_terms, tokens), source="bm25")
            for chunk, tokens in zip(self._chunks, self._doc_tokens, strict=True)
        ]
        scored = [s for s in scored if s.score > 0]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]
