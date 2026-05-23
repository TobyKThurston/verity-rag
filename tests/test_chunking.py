from verity.chunking import SemanticChunker, split_sentences
from verity.embeddings import HashingEmbedder
from verity.models import Document


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_chunker_produces_chunks_with_stable_ids():
    chunker = SemanticChunker(HashingEmbedder(64), similarity_threshold=0.0)
    doc = Document(id="doc1", text="Alpha beta gamma. Delta epsilon zeta. Eta theta iota.")
    chunks = chunker.chunk(doc)
    assert chunks
    assert all(c.doc_id == "doc1" for c in chunks)
    assert chunks[0].id == "doc1::0"
    # threshold 0.0 keeps everything coherent -> single chunk
    assert len(chunks) == 1


def test_chunker_splits_on_low_similarity():
    # threshold 1.0 means nothing is "coherent enough" -> one chunk per sentence
    chunker = SemanticChunker(HashingEmbedder(64), similarity_threshold=1.0)
    doc = Document(id="d", text="Cats purr softly. Quantum entanglement is spooky.")
    chunks = chunker.chunk(doc)
    assert len(chunks) == 2


def test_chunker_respects_max_chars():
    chunker = SemanticChunker(HashingEmbedder(64), similarity_threshold=0.0, max_chars=20)
    doc = Document(id="d", text="aaaa bbbb cccc. dddd eeee ffff. gggg hhhh iiii.")
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1


def test_empty_document_yields_no_chunks():
    chunker = SemanticChunker(HashingEmbedder(64))
    assert chunker.chunk(Document(id="d", text="   ")) == []
