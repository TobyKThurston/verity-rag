"""Brute-force cosine search over an in-memory matrix.

Fine for a few thousand chunks. PgVectorStore is the ANN backend for larger
corpora.
"""

from __future__ import annotations

import numpy as np

from verity.models import Chunk, ScoredChunk


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk]) -> None:
        embedded = [c for c in chunks if c.embedding is not None]
        if not embedded:
            return
        self._chunks.extend(embedded)
        vectors = np.array([c.embedding for c in self._chunks], dtype=np.float64)
        # normalize up front so search is one matmul
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = vectors / norms

    def search(self, query_vector: list[float], top_k: int) -> list[ScoredChunk]:
        if self._matrix is None or not self._chunks:
            return []
        q = np.array(query_vector, dtype=np.float64)
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        sims = self._matrix @ (q / qn)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [
            ScoredChunk(chunk=self._chunks[i], score=float(sims[i]), source="dense")
            for i in top_idx
        ]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)
