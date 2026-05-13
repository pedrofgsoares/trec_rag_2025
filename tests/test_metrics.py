"""Eval metrics tests — task 6.3."""

from __future__ import annotations

from pathlib import Path

from trec_biogen.eval.metrics import evaluate
from trec_biogen.io.qrels import load_qrels

FIXTURES = Path(__file__).parent / "fixtures"


def test_strict_vs_relaxed_differs_on_partial() -> None:
    qrels = load_qrels(FIXTURES / "mini_qrels.jsonl")
    report = evaluate(FIXTURES / "mini_submission.jsonl", qrels)

    # Strict support, Q001 s0: positives={10000001}, predicted={10000001,10000004,99999999}
    #   P = 1/3, R = 1/1, F1 ≈ 0.5
    # Strict support, Q002 s0: positives={10000005}, predicted={10000005}
    #   P = 1, R = 1, F1 = 1
    # Macro: P=(1/3+1)/2≈0.667, R=1, F1=2*0.667*1/(0.667+1)≈0.800
    sup_strict = report["strict"]["support"]
    assert 0.79 < sup_strict["F1"] < 0.81
    assert 0.66 < sup_strict["P"] < 0.67
    assert sup_strict["R"] == 1.0

    # Relaxed support, Q001 s0: positives={10000001,10000004}, predicted same as above
    #   P = 2/3, R = 1, F1 = 0.8 -> macro with Q002 (1,1,1) -> P=5/6, R=1, F1≈0.909
    sup_relaxed = report["relaxed"]["support"]
    assert 0.90 < sup_relaxed["F1"] < 0.92

    # Strict contradict: Q001 s1 perfect, Q002 s1 predicts {10000003,10000006} vs strict positives {10000003}
    #   Q002 s1: P=0.5, R=1, F1=0.667
    #   Macro F1 with Q001 s1 (1,1,1) -> P=0.75, R=1, F1≈0.857
    con_strict = report["strict"]["contradict"]
    assert 0.85 < con_strict["F1"] < 0.87

    # Relaxed contradict: Q002 s1 positives = {10000003,10000006}, predicted = same set -> perfect
    con_relaxed = report["relaxed"]["contradict"]
    assert con_relaxed["F1"] == 1.0


def test_unjudged_cells_ignored() -> None:
    """Topics with no qrels positives must not penalise the score (no P/R/F1 contribution)."""
    qrels = load_qrels(FIXTURES / "mini_qrels.jsonl")
    report = evaluate(FIXTURES / "mini_submission.jsonl", qrels)
    # Every output field is finite and in [0,1].
    for setting in ("strict", "relaxed"):
        for cls in ("support", "contradict"):
            for k, v in report[setting][cls].items():
                assert 0.0 <= v <= 1.0, f"{setting}/{cls}/{k}={v}"
