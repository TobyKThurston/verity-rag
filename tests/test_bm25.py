from verity.models import Chunk
from verity.store.bm25 import BM25Index, tokenize


def _chunks() -> list[Chunk]:
    return [
        Chunk(id="a::0", doc_id="a", text="the quick brown fox jumps"),
        Chunk(id="b::0", doc_id="b", text="a lazy dog sleeps all day"),
        Chunk(id="c::0", doc_id="c", text="quick foxes are quick and clever"),
    ]


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello, World-123!") == ["hello", "world", "123"]


def test_bm25_ranks_term_frequency():
    idx = BM25Index()
    idx.index(_chunks())
    results = idx.search("quick", top_k=10)
    # "quick" appears twice in c and once in a; c should rank first.
    assert results[0].chunk.doc_id == "c"
    assert {r.chunk.doc_id for r in results} == {"a", "c"}


def test_bm25_drops_zero_score_docs():
    idx = BM25Index()
    idx.index(_chunks())
    results = idx.search("nonexistent token", top_k=10)
    assert results == []


def test_bm25_source_label():
    idx = BM25Index()
    idx.index(_chunks())
    assert idx.search("fox", top_k=1)[0].source == "bm25"
