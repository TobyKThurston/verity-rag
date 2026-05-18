"""Run the eval suite. Exits non-zero if any metric is below threshold.

    python -m evals.run_evals            # offline, deterministic (CI default)
    python -m evals.run_evals --models   # real local models + LLM judge
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.answerer import extractive_answer
from evals.judge import HeuristicJudge, Judge, LlmJudge
from evals.metrics import RETRIEVAL_METRICS
from evals.thresholds import THRESHOLDS
from verity.config import get_settings
from verity.corpus import load_directory
from verity.models import EvalCase, EvalReport
from verity.pipeline import build_pipeline

_HERE = Path(__file__).parent
DEFAULT_DATASET = _HERE / "golden.jsonl"
SAMPLE_CORPUS = _HERE / "sample_notes"


def load_dataset(path: str | Path) -> list[EvalCase]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(EvalCase(**json.loads(line)))
    return cases


def run(dataset_path: str | None = None, *, use_models: bool = False) -> EvalReport:
    settings = get_settings()
    pipeline = build_pipeline(settings, use_models=use_models)
    pipeline.ingest(load_directory(SAMPLE_CORPUS))

    cases = load_dataset(dataset_path or DEFAULT_DATASET)
    judge: Judge = LlmJudge(pipeline.chat_model) if use_models else HeuristicJudge()

    per_case: list[dict[str, float]] = []
    for case in cases:
        relevant = set(case.relevant_ids)
        contexts = pipeline.retrieve(case.question)
        answer = pipeline.ask(case.question) if use_models else extractive_answer(
            case.question, contexts
        )

        row: dict[str, float] = {
            name: fn(contexts, relevant) for name, fn in RETRIEVAL_METRICS.items()
        }
        row["faithfulness"] = judge.faithfulness(answer.text, contexts)
        row["answer_relevancy"] = judge.answer_relevancy(case.question, answer.text)
        per_case.append(row)

    metrics = _aggregate(per_case)
    report = EvalReport(n_cases=len(cases), metrics=metrics, per_case=per_case)
    _print_report(report)
    return report


def _aggregate(per_case: list[dict[str, float]]) -> dict[str, float]:
    if not per_case:
        return {}
    keys = per_case[0].keys()
    return {k: sum(row[k] for row in per_case) / len(per_case) for k in keys}


def _print_report(report: EvalReport) -> None:
    print(f"\n=== Verity eval report ({report.n_cases} cases) ===")
    print(f"{'metric':<20}{'score':>8}{'floor':>8}{'':>4}")
    for name, score in sorted(report.metrics.items()):
        floor = THRESHOLDS.get(name)
        mark = "" if floor is None else ("PASS" if score >= floor else "FAIL")
        floor_s = "-" if floor is None else f"{floor:.2f}"
        print(f"{name:<20}{score:>8.3f}{floor_s:>8}{mark:>6}")
    verdict = "PASS" if report.passes(THRESHOLDS) else "FAIL"
    print(f"\noverall: {verdict}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Verity eval suite.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--models", action="store_true", help="Use real local models + LLM judge."
    )
    args = parser.parse_args(argv)
    report = run(dataset_path=args.dataset, use_models=args.models)
    return 0 if report.passes(THRESHOLDS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
