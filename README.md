# Verity

A local question-answering tool for your own notes. Point it at a folder of
notes and ask questions in plain language; it answers from the notes, cites the
file each fact came from, and says it doesn't know when the notes don't cover
something. Everything runs locally through [Ollama](https://ollama.com), so the
notes never leave your machine.

[![CI](https://github.com/TobyKThurston/verity-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/TobyKThurston/verity-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Why

I built this to study for finals and pull quotes for essays without scrolling
through a semester of lecture notes. It's the same idea as NotebookLM, with two
differences I cared about:

- It runs locally, so my notes aren't uploaded anywhere.
- Retrieval quality is measured. There's an eval suite that runs in CI, and the
  build fails if a change makes retrieval or grounding worse. That mattered to
  me because the failure mode of these tools is confidently making things up.

It's a single-user command-line tool. No accounts, no server to run, no users.

The sample notes under `evals/sample_notes/` are lecture notes for an intro
South Asia course. They're used for the demo and as the eval fixture.

## How it works

```
   notes/*.md
       │  semantic chunking
       ▼
   embeddings ──► dense vector store (pgvector / in-memory)
       │
       └────────► BM25 sparse index
                       │
            both rankings fused with RRF
                       │
            cross-encoder reranker (top N)
                       │
            tool-calling agent loop (Ollama)
                       │
            answer + citations
```

Each stage emits an OpenTelemetry span, so one query is one trace you can read
top to bottom.

| Part | File | Approach |
|------|------|----------|
| Chunking | `chunking.py` | sentence grouping by embedding similarity |
| Hybrid retrieval | `retrieval/hybrid.py` | Reciprocal Rank Fusion of dense + BM25 |
| Reranking | `retrieval/rerank.py` | cross-encoder over the fused pool |
| Vector store | `store/` | pgvector + HNSW, in-memory fallback |
| Sparse index | `store/bm25.py` | BM25 Okapi |
| Agent | `agent/` | bounded tool-calling loop |
| Evaluation | `evals/` | retrieval metrics + LLM-as-judge |
| Tracing | `telemetry.py` | OpenTelemetry spans |

There's more detail on the design decisions in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Evaluation

The eval suite scores the pipeline against a labelled golden set
(`evals/golden.jsonl`) on two groups of metrics:

Retrieval (no model needed):
- `context_recall`, `context_precision`, `mrr`, `hit_rate`

Generation (LLM-as-judge, or a token-overlap fallback for CI):
- `faithfulness` (is the answer supported by the retrieved notes)
- `answer_relevancy` (does it address the question)

By default the suite runs with the deterministic backend, so there are no model
downloads and the numbers are reproducible. It runs on every push and fails if
any metric drops below the floors in `evals/thresholds.py`.

```bash
python -m evals.run_evals            # offline, deterministic (what CI runs)
python -m evals.run_evals --models   # real local models + LLM-as-judge
```

## Quickstart

```bash
# install (core deps are enough to run the pipeline and eval suite)
pip install -e ".[dev]"

# run the eval suite, no models required
python -m evals.run_evals

# for real answers, install the model deps and pull a model
pip install -e ".[ml,store]"
ollama pull llama3.1

# ask your notes
verity ask "How did the emperor Akbar show religious tolerance?" --notes evals/sample_notes

# or run the API
verity serve   # POST /ingest and /ask; OpenAPI docs at /docs
```

Pass `--no-models` to force the deterministic backend on any command.

## Stack

Python 3.11+, FastAPI, Pydantic v2, sentence-transformers, Postgres + pgvector,
BM25, Ollama, OpenTelemetry, pytest, ruff, mypy, GitHub Actions, Docker.

## License

MIT
