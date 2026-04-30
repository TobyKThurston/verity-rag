"""Rerank the fused candidate pool.

CrossEncoderReranker scores each (query, chunk) pair jointly, which is more
accurate than the first-stage bi-encoder but only affordable on a small pool.
LexicalReranker is a model-free Jaccard fallback for tests.
"""

from __future__ import annotations

from verity.models import ScoredChunk
from verity.store.bm25 import tokenize
from verity.telemetry import span


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        with span("rerank.cross_encoder", candidates=len(candidates), top_n=top_n):
            pairs = [[query, c.chunk.text] for c in candidates]
            scores = self._model.predict(pairs)
            reranked = [
                ScoredChunk(chunk=c.chunk, score=float(s), source="reranked")
                for c, s in zip(candidates, scores, strict=True)
            ]
            reranked.sort(key=lambda s: s.score, reverse=True)
            return reranked[:top_n]


class LexicalReranker:
    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        q_terms = set(tokenize(query))
        with span("rerank.lexical", candidates=len(candidates), top_n=top_n):
            reranked = []
            for c in candidates:
                c_terms = set(tokenize(c.chunk.text))
                union = q_terms | c_terms
                jaccard = len(q_terms & c_terms) / len(union) if union else 0.0
                reranked.append(
                    ScoredChunk(chunk=c.chunk, score=jaccard, source="reranked")
                )
            reranked.sort(key=lambda s: s.score, reverse=True)
            return reranked[:top_n]
