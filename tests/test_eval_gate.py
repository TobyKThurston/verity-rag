"""The golden suite must pass its own thresholds, so CI catches regressions."""

from evals.run_evals import run
from evals.thresholds import THRESHOLDS


def test_golden_suite_passes_thresholds():
    report = run(use_models=False)
    assert report.n_cases >= 6
    assert report.passes(THRESHOLDS), report.metrics


def test_every_gated_metric_is_reported():
    report = run(use_models=False)
    for metric in THRESHOLDS:
        assert metric in report.metrics
