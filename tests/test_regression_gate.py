"""
Unit tests for the regression gate (evals/metric_registry.py + evals/compare.py).

These exercise the DECISION logic on hand-built snapshots -- no pipeline, no LLM
judge, no API key -- so they run free and fast in CI:

    pytest -q
"""

import math

import pytest

from evals.compare import classify, compare
from evals.metric_registry import (
    GATE_HIGHER_AVG,
    GATE_LOWER_AVG,
    INFO,
    LATENCY_GUARD,
    QUALITY_GUARD,
    rule_for,
)


def snap(**metrics):
    """Build a snapshot from dotted ids written with '__' in place of '.'."""
    return {"metadata": {}, "metrics": {k.replace("__", "."): v for k, v in metrics.items()}}


# ------------------------------------------------------------------ registry
@pytest.mark.parametrize("mid, rule", [
    ("safety.scope.avg_score", GATE_HIGHER_AVG),
    ("safety.leakage.pii_avg_score", GATE_HIGHER_AVG),
    ("safety.toxicity.avg_toxicity", GATE_LOWER_AVG),
    ("application.correctness.avg_score", QUALITY_GUARD),
    ("retriever.contextual_recall.avg_score", QUALITY_GUARD),
    ("ops.latency.e2e_p95_ms", LATENCY_GUARD),
    ("application.correctness.pass_rate", INFO),     # pass rate is info, not gated
    ("ops.latency.ttft_p95_ms", INFO),               # too noisy to gate on
])
def test_rule_resolution(mid, rule):
    assert rule_for(mid) is rule


# ------------------------------------------------------------------ classify
def test_safety_drop_beyond_tolerance_blocks():
    row = classify("safety.scope.avg_score", 0.90, 0.85, GATE_HIGHER_AVG)
    assert row["status"] == "blocked"


def test_safety_drop_within_noise_is_flat():
    row = classify("safety.scope.avg_score", 0.90, 0.885, GATE_HIGHER_AVG)
    assert row["status"] == "flat"


def test_toxicity_direction_is_lower_better():
    assert classify("safety.toxicity.avg_toxicity", 0.02, 0.10, GATE_LOWER_AVG)["status"] == "blocked"
    assert classify("safety.toxicity.avg_toxicity", 0.10, 0.02, GATE_LOWER_AVG)["status"] == "improved"


def test_quality_regression_needs_review_not_block():
    row = classify("application.correctness.avg_score", 0.88, 0.80, QUALITY_GUARD)
    assert row["status"] == "regressed"


def test_latency_uses_relative_tolerance():
    # 25% band on a 3000 ms baseline = 750 ms
    assert classify("ops.latency.e2e_p95_ms", 3000, 3700, LATENCY_GUARD)["status"] == "flat"
    assert classify("ops.latency.e2e_p95_ms", 3000, 3800, LATENCY_GUARD)["status"] == "regressed"


def test_slo_boolean_flip_is_a_regression():
    rule = rule_for("ops.cost.budget_pass")
    assert classify("ops.cost.budget_pass", True, False, rule)["status"] == "regressed"
    assert classify("ops.cost.budget_pass", False, True, rule)["status"] == "improved"


def test_info_metrics_never_affect_status():
    row = classify("application.correctness.pass_rate", 93.3, 40.0, INFO)
    assert row["status"] == "info" and math.isclose(row["delta"], -53.3)


def test_new_and_dropped_metrics_are_surfaced():
    assert classify("x.new.avg_score", None, 0.9, QUALITY_GUARD)["status"] == "new"
    assert classify("x.old.avg_score", 0.9, None, QUALITY_GUARD)["status"] == "dropped"


# ------------------------------------------------------------------ verdicts
BASE = dict(
    safety__scope__avg_score=0.90,
    safety__toxicity__avg_toxicity=0.02,
    application__correctness__avg_score=0.88,
    ops__latency__e2e_p95_ms=3000.0,
)


def test_identical_snapshots_pass():
    verdict, _ = compare(snap(**BASE), snap(**BASE))
    assert verdict == "PASS"


def test_quality_regression_gives_review():
    cand = {**BASE, "application__correctness__avg_score": 0.78}
    verdict, _ = compare(snap(**BASE), snap(**cand))
    assert verdict == "REVIEW"


def test_safety_regression_fails_even_if_quality_improves():
    cand = {**BASE, "safety__scope__avg_score": 0.80, "application__correctness__avg_score": 0.95}
    verdict, _ = compare(snap(**BASE), snap(**cand))
    assert verdict == "FAIL"
