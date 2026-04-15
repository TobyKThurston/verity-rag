"""Pydantic models passed between the ingestion, retrieval, and eval layers."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)
    embedding: list[float] | None = None


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    source: str = "unknown"  # which retriever/reranker produced it

    @property
    def id(self) -> str:
        return self.chunk.id


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    snippet: str


class Answer(BaseModel):
    query: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[ScoredChunk] = Field(default_factory=list)
    trace_id: str | None = None


class EvalCase(BaseModel):
    """One row of a golden dataset."""

    id: str
    question: str
    ground_truth: str
    relevant_ids: list[str] = Field(default_factory=list)


class MetricResult(BaseModel):
    name: str
    score: float
    detail: str = ""


class EvalReport(BaseModel):
    n_cases: int
    metrics: dict[str, float]
    per_case: list[dict[str, float]] = Field(default_factory=list)

    def passes(self, thresholds: dict[str, float]) -> bool:
        return all(self.metrics.get(name, 0.0) >= floor for name, floor in thresholds.items())
