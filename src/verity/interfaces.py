"""Protocols for the swappable backends (embedder, stores, reranker, model)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from verity.models import Chunk, ScoredChunk


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...

    def search(self, query_vector: list[float], top_k: int) -> list[ScoredChunk]: ...

    def all_chunks(self) -> list[Chunk]: ...


@runtime_checkable
class SparseIndex(Protocol):
    def index(self, chunks: list[Chunk]) -> None: ...

    def search(self, query: str, top_k: int) -> list[ScoredChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class ChatModel(Protocol):
    def chat(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]: ...
