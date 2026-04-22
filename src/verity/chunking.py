"""Group sentences into chunks while they stay similar to the running centroid.

Breaks to a new chunk when similarity drops below the threshold or the chunk
hits max_chars, which keeps related sentences together better than fixed windows.
"""

from __future__ import annotations

import re

from verity.embeddings import cosine
from verity.interfaces import Embedder
from verity.models import Chunk, Document

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


class SemanticChunker:
    def __init__(
        self,
        embedder: Embedder,
        similarity_threshold: float = 0.5,
        max_chars: int = 1200,
    ) -> None:
        self._embedder = embedder
        self._threshold = similarity_threshold
        self._max_chars = max_chars

    def chunk(self, doc: Document) -> list[Chunk]:
        sentences = split_sentences(doc.text)
        if not sentences:
            return []

        vectors = self._embedder.embed(sentences)
        chunks: list[Chunk] = []
        current: list[str] = [sentences[0]]
        centroid = vectors[0]

        for sent, vec in zip(sentences[1:], vectors[1:], strict=True):
            joined_len = sum(len(s) for s in current) + len(sent)
            coherent = cosine(centroid, vec) >= self._threshold
            if coherent and joined_len <= self._max_chars:
                current.append(sent)
                centroid = _running_mean(centroid, vec, len(current))
            else:
                chunks.append(self._emit(doc, len(chunks), current))
                current = [sent]
                centroid = vec

        chunks.append(self._emit(doc, len(chunks), current))
        return chunks

    @staticmethod
    def _emit(doc: Document, idx: int, sentences: list[str]) -> Chunk:
        return Chunk(
            id=f"{doc.id}::{idx}",
            doc_id=doc.id,
            text=" ".join(sentences),
            metadata=doc.metadata,
        )


def _running_mean(centroid: list[float], new: list[float], n: int) -> list[float]:
    return [c + (x - c) / n for c, x in zip(centroid, new, strict=True)]
