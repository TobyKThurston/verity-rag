from verity.models import Chunk, ScoredChunk
from verity.retrieval.hybrid import reciprocal_rank_fusion


def _sc(doc: str, score: float, source: str) -> ScoredChunk:
    chunk = Chunk(id=f"{doc}::0", doc_id=doc, text=doc)
    return ScoredChunk(chunk=chunk, score=score, source=source)


def test_rrf_rewards_agreement_across_lists():
    dense = [_sc("a", 0.9, "dense"), _sc("b", 0.8, "dense")]
    sparse = [_sc("b", 5.0, "bm25"), _sc("c", 1.0, "bm25")]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    # b appears in both lists, so it should win despite not being rank-1 in dense.
    assert fused[0].chunk.doc_id == "b"
    assert fused[0].source == "bm25+dense"


def test_rrf_is_rank_based_not_score_based():
    # Sparse scores are huge but RRF only uses position, so a rank-1 dense-only
    # doc ties with a rank-1 sparse-only doc.
    dense = [_sc("a", 0.01, "dense")]
    sparse = [_sc("b", 999.0, "bm25")]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    assert {f.chunk.doc_id for f in fused} == {"a", "b"}
    assert fused[0].score == fused[1].score


def test_rrf_top_k_truncates():
    lists = [[_sc(c, 1.0, "dense") for c in "abcde"]]
    assert len(reciprocal_rank_fusion(lists, top_k=2)) == 2
