"""CLI: verity ask | serve | eval."""

from __future__ import annotations

import argparse
import sys

from verity.corpus import load_directory
from verity.pipeline import build_pipeline


def _cmd_ingest_and_ask(directory: str, question: str, use_models: bool) -> int:
    pipeline = build_pipeline(use_models=use_models)
    docs = load_directory(directory)
    chunks = pipeline.ingest(docs)
    print(f"Ingested {len(docs)} documents -> {len(chunks)} chunks", file=sys.stderr)
    answer = pipeline.ask(question)
    print(answer.text)
    if answer.citations:
        print("\nCitations:")
        for c in answer.citations:
            print(f"  - {c.chunk_id} ({c.doc_id})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verity", description="Local agentic RAG over your notes."
    )
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="Use deterministic fallbacks (no sentence-transformers/Ollama).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ask = sub.add_parser("ask", help="Ingest a directory then ask one question.")
    p_ask.add_argument("question")
    p_ask.add_argument("--notes", default="./notes", help="Notes directory to ingest.")

    p_serve = sub.add_parser("serve", help="Run the FastAPI server.")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    p_eval = sub.add_parser("eval", help="Run the offline evaluation suite.")
    p_eval.add_argument("--dataset", default=None)

    args = parser.parse_args(argv)
    use_models = not args.no_models

    if args.command == "ask":
        return _cmd_ingest_and_ask(args.notes, args.question, use_models)

    if args.command == "serve":
        import uvicorn

        from verity.api import create_app

        uvicorn.run(create_app(use_models=use_models), host=args.host, port=args.port)
        return 0

    if args.command == "eval":
        from evals.run_evals import run

        report = run(dataset_path=args.dataset, use_models=use_models)
        return 0 if report.passes(get_settings_thresholds()) else 1

    return 2


def get_settings_thresholds() -> dict[str, float]:
    from evals.thresholds import THRESHOLDS

    return THRESHOLDS


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
