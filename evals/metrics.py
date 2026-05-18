"""Retrieval metrics, scored at document granularity.

A chunk counts as relevant if its doc_id is in the case's relevant_ids. Scoring
by document (not chunk) keeps the labels stable when chunking changes.
"""

from __future__ import annotations

from verity.models import ScoredChunk


def _relevant_hits(retrieved: list[ScoredChunk], relevant: set[str]) -> list[bool]:
    return [sc.chunk.doc_id in relevant for sc in retrieved]


def context_precision(retrieved: list[ScoredChunk], relevant: set[str]) -> float:
    """Fraction of retrieved contexts that are relevant."""

    if not retrieved:
        return 0.0
    hits = _relevant_hits(retrieved, relevant)
    return sum(hits) / len(hits)


def context_recall(retrieved: list[ScoredChunk], relevant: set[str]) -> float:
    """Fraction of relevant documents that were retrieved."""

    if not relevant:
        return 1.0
    retrieved_docs = {sc.chunk.doc_id for sc in retrieved}
    return len(retrieved_docs & relevant) / len(relevant)


def mrr(retrieved: list[ScoredChunk], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant context (0 if none)."""

    for rank, sc in enumerate(retrieved, start=1):
        if sc.chunk.doc_id in relevant:
            return 1.0 / rank
    return 0.0


def hit_rate(retrieved: list[ScoredChunk], relevant: set[str]) -> float:
    """1.0 if any relevant document is present in the retrieved set."""

    return 1.0 if any(_relevant_hits(retrieved, relevant)) else 0.0


RETRIEVAL_METRICS = {
    "context_precision": context_precision,
    "context_recall": context_recall,
    "mrr": mrr,
    "hit_rate": hit_rate,
}
