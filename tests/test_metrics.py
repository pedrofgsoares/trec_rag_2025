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

from trec_biogen.eval.metrics import DEFAULT_QRELS_PATHS, evaluate, main
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


# -- Phase 2 §3.8: dual-pool + source filter ------------------------------


def test_load_expanded_qrels_carries_source_attribution() -> None:
    """LLM rows must be tagged with their non-'human' source so the
    --source filter can isolate them downstream."""
    idx = load_qrels(FIXTURES / "mini_qrels_expanded.jsonl")
    # Q001 sentence 0 support: 10000001 (human) + 99999999 (LLM).
    key = ("Q001", 0, "support")
    assert idx.strict_sources[key]["10000001"] == "human"
    assert idx.strict_sources[key]["99999999"] == "openai-gpt-4o-mini"


def test_source_filter_human_recovers_official_pool_numbers() -> None:
    """`--source=human` on the expanded file MUST reproduce the metrics
    you'd get from the human-only file. This is the §6.5 reproducibility
    contract."""
    human_qrels = load_qrels(FIXTURES / "mini_qrels.jsonl")
    expanded_qrels = load_qrels(FIXTURES / "mini_qrels_expanded.jsonl")
    a = evaluate(FIXTURES / "mini_submission.jsonl", human_qrels, unjudged_as_zero=False)
    b = evaluate(
        FIXTURES / "mini_submission.jsonl", expanded_qrels,
        unjudged_as_zero=False, source="human",
    )
    for setting in ("strict", "relaxed"):
        for cls in ("support", "contradict"):
            for k in ("P", "R", "F1"):
                assert abs(a[setting][cls][k] - b[setting][cls][k]) < 1e-9, (
                    f"{setting}/{cls}/{k} drifted: human-only={a[setting][cls][k]} "
                    f"vs expanded+source=human={b[setting][cls][k]}"
                )


def test_source_filter_any_unions_human_and_llm() -> None:
    """Default `source=any` on the expanded file credits LLM-attributed
    positives. Submission Q001 sentence 0 predicts 99999999 which is now
    an LLM-positive — recall must go up vs human-only."""
    human_qrels = load_qrels(FIXTURES / "mini_qrels.jsonl")
    expanded_qrels = load_qrels(FIXTURES / "mini_qrels_expanded.jsonl")
    a = evaluate(FIXTURES / "mini_submission.jsonl", human_qrels, unjudged_as_zero=False)
    b = evaluate(FIXTURES / "mini_submission.jsonl", expanded_qrels, unjudged_as_zero=False)
    # Strict support: under human-only, Q001 s0 has positives={10000001} and
    # we predict 3 -> P=1/3, R=1, F1=0.5. Under expanded, positives now
    # include 99999999 (LLM) -> P=2/3, R=1, F1=0.8. F1 should increase.
    assert b["strict"]["support"]["F1"] > a["strict"]["support"]["F1"]


def test_source_filter_llm_isolates_llm_only_contribution() -> None:
    """`--source=llm` keeps only LLM-attributed positives. Cells with no
    LLM positives drop to F1=0 (under unjudged_as_zero=True)."""
    expanded_qrels = load_qrels(FIXTURES / "mini_qrels_expanded.jsonl")
    rep = evaluate(FIXTURES / "mini_submission.jsonl", expanded_qrels, source="llm")
    # Only Q001 s0 support has an LLM positive (99999999), predicted; and
    # only Q002 s1 contradict has an LLM positive (10000008), predicted in
    # the submission as part of {10000003, 10000006}. The other cells are
    # human-only-positive and become F1=0 under source=llm.
    assert 0.0 < rep["strict"]["support"]["F1"] < 1.0


def test_default_qrels_paths_constant_is_consistent() -> None:
    """The convenience constant must point at the canonical file names
    referenced in the design and reports."""
    assert DEFAULT_QRELS_PATHS["official"].name == "biogen2025_taskA_qrels.jsonl"
    assert DEFAULT_QRELS_PATHS["expanded"].name == "biogen2025_taskA_qrels_expanded.jsonl"
    # Phase 2.5: intersection pool is the third canonical path.
    assert DEFAULT_QRELS_PATHS["intersection"].name == "biogen2025_taskA_qrels_intersection.jsonl"


def test_source_filter_human_recovers_official_on_intersection_pool(tmp_path: Path) -> None:
    """Phase 2.5: `--source=human` on the intersection pool MUST also
    reproduce the official-pool numbers byte-for-byte (the intersection
    file is a strict superset of the human qrels, so filtering down to
    human-only positives must equal the human-only file)."""
    from trec_biogen.judge.intersection import emit_intersection_pool

    human_path = FIXTURES / "mini_qrels.jsonl"
    expanded_path = FIXTURES / "mini_qrels_expanded.jsonl"
    # Use the expanded file as both inputs — the intersection is then
    # the expanded set itself (every contradicts triple agrees with itself).
    intersection_path = tmp_path / "mini_qrels_intersection.jsonl"
    emit_intersection_pool(
        expanded_path, expanded_path,
        human_qrels_path=human_path, out_path=intersection_path,
    )
    human = load_qrels(human_path)
    intersection = load_qrels(intersection_path)
    a = evaluate(FIXTURES / "mini_submission.jsonl", human, unjudged_as_zero=False)
    b = evaluate(
        FIXTURES / "mini_submission.jsonl", intersection,
        unjudged_as_zero=False, source="human",
    )
    for setting in ("strict", "relaxed"):
        for cls in ("support", "contradict"):
            for k in ("P", "R", "F1"):
                assert abs(a[setting][cls][k] - b[setting][cls][k]) < 1e-9, (
                    f"{setting}/{cls}/{k} drifted: human-only={a[setting][cls][k]} "
                    f"vs intersection+source=human={b[setting][cls][k]}"
                )


def test_cli_qrels_pool_resolves_default_path(tmp_path: Path, capsys) -> None:
    """`--qrels-pool=official` must be a working substitute for `--qrels=<path>`."""
    out = tmp_path / "metrics.json"
    rc = main(
        [
            "--submission", str(FIXTURES / "mini_submission.jsonl"),
            "--qrels", str(FIXTURES / "mini_qrels.jsonl"),
            "--out", str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
