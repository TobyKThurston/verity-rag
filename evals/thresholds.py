"""Minimum metric scores the eval suite must clear. Tuned on the golden set."""

from __future__ import annotations

THRESHOLDS: dict[str, float] = {
    "context_recall": 0.90,
    "hit_rate": 0.95,
    "mrr": 0.60,
    "context_precision": 0.30,
    "faithfulness": 0.80,
    "answer_relevancy": 0.30,
}
