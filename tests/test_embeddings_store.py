from verity.embeddings import HashingEmbedder, cosine
from verity.models import Chunk
from verity.store.memory import InMemoryVectorStore


def test_hashing_embedder_is_deterministic_and_normalised():
    emb = HashingEmbedder(128)
    a = emb.embed(["hello world"])[0]
    b = emb.embed(["hello world"])[0]
    assert a == b
    assert abs(cosine(a, a) - 1.0) < 1e-9


def test_cosine_orthogonal_and_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_vector_store_search_ranks_by_similarity():
    emb = HashingEmbedder(256)
    chunks = [
        Chunk(id="a::0", doc_id="a", text="rust ownership borrow checker"),
        Chunk(id="b::0", doc_id="b", text="sourdough bread hydration starter"),
    ]
    for c in chunks:
        c.embedding = emb.embed([c.text])[0]
    store = InMemoryVectorStore()
    store.add(chunks)

    q = emb.embed(["rust borrow checker"])[0]
    results = store.search(q, top_k=2)
    assert results[0].chunk.doc_id == "a"
    assert results[0].source == "dense"
    assert len(store.all_chunks()) == 2


def test_empty_store_returns_nothing():
    assert InMemoryVectorStore().search([0.1, 0.2], top_k=5) == []
