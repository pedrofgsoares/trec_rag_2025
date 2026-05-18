"""Eval metrics tests — task 6.3.

The metric methodology now matches the published BioGEN 2025 protocol:

  * F1 macro = **mean of per-cell F1s** (not F1 of macro P / R)
  * P, R macro = mean of per-cell P / R
  * In sentence-level mode, cells with predictions but no qrels positives
    contribute F1 = 0 (``unjudged_as_zero=True``); pass
    ``unjudged_as_zero=False`` to keep only judged cells in the macro.
"""

from __future__ import annotations

from pathlib import Path

from trec_biogen.eval.metrics import evaluate
from trec_biogen.io.qrels import load_qrels

FIXTURES = Path(__file__).parent / "fixtures"


def test_strict_vs_relaxed_differs_on_partial() -> None:
    qrels = load_qrels(FIXTURES / "mini_qrels.jsonl")
    # unjudged_as_zero=False so we only average over the 2 judged cells
    # per (setting, class); the fixture predicts only there.
    report = evaluate(FIXTURES / "mini_submission.jsonl", qrels, unjudged_as_zero=False)

    # === Strict support ===
    # Q001 s0: positives={10000001}, predicted={10000001,10000004,99999999}
    #   P = 1/3, R = 1, F1 = 2*(1/3)*1/((1/3)+1) = 0.5
    # Q002 s0: positives={10000005}, predicted={10000005}
    #   F1 = 1
    # mean F1 = (0.5 + 1) / 2 = 0.75
    sup_strict = report["strict"]["support"]
    assert 0.74 < sup_strict["F1"] < 0.76
    assert 0.66 < sup_strict["P"] < 0.67     # macro P = (1/3 + 1)/2
    assert sup_strict["R"] == 1.0            # macro R

    # === Relaxed support ===
    # Q001 s0: positives={10000001,10000004}, predicted same as above
    #   P = 2/3, R = 1, F1 = 0.8
    # Q002 s0: F1 = 1
    # mean F1 = (0.8 + 1) / 2 = 0.9
    sup_relaxed = report["relaxed"]["support"]
    assert 0.89 < sup_relaxed["F1"] < 0.91

    # === Strict contradict ===
    # Q001 s1: predicted={10000003}, positives={10000003}: F1 = 1
    # Q002 s1: predicted={10000003,10000006}, positives={10000003}: P=0.5, R=1, F1=2/3
    # mean F1 = (1 + 2/3) / 2 ≈ 0.833
    con_strict = report["strict"]["contradict"]
    assert 0.82 < con_strict["F1"] < 0.84

    # === Relaxed contradict ===
    # Both cells perfect overlap -> mean F1 = 1.0
    con_relaxed = report["relaxed"]["contradict"]
    assert con_relaxed["F1"] == 1.0


def test_unjudged_as_zero_defaults_true_for_sentence_level() -> None:
    """Default behaviour: cells with predictions but no positives drag F1 down to 0."""
    qrels = load_qrels(FIXTURES / "mini_qrels.jsonl")
    default_report = evaluate(FIXTURES / "mini_submission.jsonl", qrels)
    explicit_report = evaluate(
        FIXTURES / "mini_submission.jsonl", qrels, unjudged_as_zero=True
    )
    # Same numbers (default == True)
    for setting in ("strict", "relaxed"):
        for cls in ("support", "contradict"):
            for k in ("P", "R", "F1"):
                assert default_report[setting][cls][k] == explicit_report[setting][cls][k]


def test_all_metrics_finite_and_in_unit_interval() -> None:
    qrels = load_qrels(FIXTURES / "mini_qrels.jsonl")
    report = evaluate(FIXTURES / "mini_submission.jsonl", qrels)
    for setting in ("strict", "relaxed"):
        for cls in ("support", "contradict"):
            for k, v in report[setting][cls].items():
                assert 0.0 <= v <= 1.0, f"{setting}/{cls}/{k}={v}"
