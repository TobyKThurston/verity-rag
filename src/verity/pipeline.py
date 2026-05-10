"""Wires the subsystems into one pipeline.

    documents -> chunks -> {dense store, bm25 index}
    query -> hybrid retrieve -> rerank -> agent -> answer
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.agent.loop import RagAgent
from verity.chunking import SemanticChunker
from verity.config import Settings, get_settings
from verity.embeddings import HashingEmbedder
from verity.interfaces import ChatModel, Embedder, Reranker, SparseIndex, VectorStore
from verity.models import Answer, Chunk, Document, ScoredChunk
from verity.retrieval.hybrid import HybridRetriever
from verity.retrieval.rerank import LexicalReranker
from verity.store.bm25 import BM25Index
from verity.store.memory import InMemoryVectorStore
from verity.telemetry import span


@dataclass
class Pipeline:
    embedder: Embedder
    vector_store: VectorStore
    sparse_index: SparseIndex
    reranker: Reranker
    chat_model: ChatModel
    settings: Settings

    def __post_init__(self) -> None:
        self._chunker = SemanticChunker(self.embedder)
        self._retriever = HybridRetriever(
            self.embedder, self.vector_store, self.sparse_index, rrf_k=self.settings.rrf_k
        )
        self._agent = RagAgent(
            self.chat_model, self.retrieve, max_steps=self.settings.max_agent_steps
        )

    def ingest(self, documents: list[Document]) -> list[Chunk]:
        with span("ingest", n_docs=len(documents)):
            chunks: list[Chunk] = []
            for doc in documents:
                chunks.extend(self._chunker.chunk(doc))
            vectors = self.embedder.embed([c.text for c in chunks])
            for chunk, vec in zip(chunks, vectors, strict=True):
                chunk.embedding = vec
            self.vector_store.add(chunks)
            self.sparse_index.index(self.vector_store.all_chunks())
            return chunks

    def retrieve(self, query: str) -> list[ScoredChunk]:
        candidates = self._retriever.retrieve(query, top_k=self.settings.retrieval_top_k)
        return self.reranker.rerank(query, candidates, top_n=self.settings.rerank_top_n)

    def ask(self, question: str) -> Answer:
        return self._agent.answer(question)


def build_pipeline(
    settings: Settings | None = None,
    *,
    use_models: bool = True,
    chat_model: ChatModel | None = None,
) -> Pipeline:
    # use_models=True: sentence-transformers + cross-encoder + Ollama.
    # use_models=False: deterministic fallbacks (tests, CI, --no-models).
    settings = settings or get_settings()

    embedder: Embedder
    reranker: Reranker
    if use_models:
        from verity.embeddings import SentenceTransformerEmbedder
        from verity.retrieval.rerank import CrossEncoderReranker

        embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dim)
        reranker = CrossEncoderReranker(settings.reranker_model)
    else:
        embedder = HashingEmbedder(settings.embedding_dim)
        reranker = LexicalReranker()

    vector_store: VectorStore
    if settings.pg_dsn:
        from verity.store.pgvector import PgVectorStore

        vector_store = PgVectorStore(settings.pg_dsn, settings.embedding_dim)
    else:
        vector_store = InMemoryVectorStore()

    if chat_model is None:
        if use_models:
            from verity.llm.ollama import OllamaChatModel

            chat_model = OllamaChatModel()
        else:
            from verity.llm.extractive import ExtractiveChatModel

            chat_model = ExtractiveChatModel()

    return Pipeline(
        embedder=embedder,
        vector_store=vector_store,
        sparse_index=BM25Index(),
        reranker=reranker,
        chat_model=chat_model,
        settings=settings,
    )
