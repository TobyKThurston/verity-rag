"""Dense + sparse retrieval fused with Reciprocal Rank Fusion.

RRF combines rankings rather than raw scores: score(d) = sum(1 / (k + rank)).
It needs no score calibration between the two retrievers.
"""

from __future__ import annotations

from verity.interfaces import Embedder, SparseIndex, VectorStore
from verity.models import ScoredChunk
from verity.telemetry import span


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]],
    k: int = 60,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    # source is set to whichever retrievers contributed, e.g. "bm25+dense"
    fused: dict[str, float] = {}
    contributors: dict[str, set[str]] = {}
    by_id: dict[str, ScoredChunk] = {}

    for ranked in ranked_lists:
        for rank, sc in enumerate(ranked):
            fused[sc.id] = fused.get(sc.id, 0.0) + 1.0 / (k + rank + 1)
            contributors.setdefault(sc.id, set()).add(sc.source)
            by_id.setdefault(sc.id, sc)

    results = [
        ScoredChunk(
            chunk=by_id[cid].chunk,
            score=score,
            source="+".join(sorted(contributors[cid])),
        )
        for cid, score in fused.items()
    ]
    results.sort(key=lambda s: s.score, reverse=True)
    return results[:top_k] if top_k is not None else results


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        sparse_index: SparseIndex,
        rrf_k: int = 60,
    ) -> None:
        self._embedder = embedder
        self._vectors = vector_store
        self._sparse = sparse_index
        self._rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        with span("retrieve.hybrid", query=query, top_k=top_k):
            query_vec = self._embedder.embed([query])[0]
            with span("retrieve.dense"):
                dense = self._vectors.search(query_vec, top_k)
            with span("retrieve.sparse"):
                sparse = self._sparse.search(query, top_k)
            with span("retrieve.fuse", dense=len(dense), sparse=len(sparse)):
                return reciprocal_rank_fusion([dense, sparse], k=self._rrf_k, top_k=top_k)
