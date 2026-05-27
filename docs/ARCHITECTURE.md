# Architecture

Notes on why the pieces are built the way they are.

## Request lifecycle

A single `ask` flows through the pipeline as one OpenTelemetry trace:

```
ask(question)
└─ agent.answer                         (agent/loop.py)
   ├─ chat()  → model decides to search
   ├─ agent.tool.search
   │  └─ retrieve(query)                (pipeline.py)
   │     ├─ retrieve.hybrid             (retrieval/hybrid.py)
   │     │  ├─ retrieve.dense  → vector store ANN   (store/)
   │     │  ├─ retrieve.sparse → BM25                (store/bm25.py)
   │     │  └─ retrieve.fuse   → Reciprocal Rank Fusion
   │     └─ rerank.cross_encoder        (retrieval/rerank.py)
   └─ chat()  → grounded answer + citations
```

## Why these choices

### RRF instead of weighted score fusion
Dense cosine scores (roughly 0 to 1) and BM25 scores (unbounded) aren't on the
same scale, and a fixed weighting between them tends to drift from corpus to
corpus. RRF combines ranks instead of scores: `score(d) = sum(1 / (k + rank))`.
No calibration needed. `k=60` follows Cormack et al. (2009).

### Retrieve, then rerank
First-stage retrieval uses a bi-encoder (embeddings computed once at ingest) for
cheap recall over the whole corpus. The cross-encoder reranker reads the query
and chunk together, which is more accurate but too slow to run on everything, so
it only sees the fused candidate pool. Recall first, precision second.

### Semantic chunking
Fixed-size windows tend to cut sentences in half and spread one idea across two
chunks, which hurts retrieval. The chunker keeps adding sentences while they
stay similar to the running centroid, then starts a new chunk, with a hard
character cap as a backstop.

### Document-level eval labels
Relevance in the golden set is labelled by document, not by chunk id. That way
re-tuning the chunker doesn't change the labels, so the eval measures retrieval
quality rather than breaking every time chunk ids shift.

### Protocol seams
`Embedder`, `VectorStore`, `SparseIndex`, `Reranker`, and `ChatModel` are
`typing.Protocol`s. Real runs wire in sentence-transformers, a cross-encoder,
pgvector, and Ollama; tests and CI wire in a hashing embedder, a lexical
reranker, an in-memory store, and a scripted model. Same wiring, different
leaves, which is what lets the eval suite run without any models.

### Bounded, single-tool agent
The agent gets one tool (`search_knowledge_base`) and a step cap. One tool means
it has to ground its answer in retrieved notes rather than its own training
data. The cap stops a misbehaving model from looping forever. The faithfulness
metric then checks, after the fact, that the answer actually stuck to the notes.

## Production swap-ins

| Concern | Dev / CI default | Production |
|---------|------------------|------------|
| Embeddings | `HashingEmbedder` (deterministic) | `SentenceTransformerEmbedder` |
| Reranker | `LexicalReranker` (Jaccard) | `CrossEncoderReranker` |
| Vector store | `InMemoryVectorStore` (exact) | `PgVectorStore` (HNSW ANN) |
| Chat model | `ScriptedChatModel` | `OllamaChatModel` |
| Judge | `HeuristicJudge` | `LlmJudge` |
| Trace export | console | OTLP to a collector |
