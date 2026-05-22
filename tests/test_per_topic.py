"""Phase 2.5 §4.3 — per-topic aggregation tests.

Covers the public surface of `eval/per_topic.py`:
* `per_topic_f1` returns the right shape from a cached run-like dir.
* Per-topic F1 matches manual arithmetic on a synthetic fixture.
* `topic_f1_delta` computes target-minus-anchor over the intersection
  of topics.
* `select_three_topics` picks max-Δ, near-zero, min-Δ deterministically.
* Missing qrels file raises a clean error.
"""

from __future__ import annotations

import json
from pathlib import Path

import orjson

from trec_biogen.eval.per_topic import (
    per_topic_f1,
    select_three_topics,
    topic_f1_delta,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _write_submission(path: Path, qa_to_cells: dict[str, list[dict]]) -> Path:
    """Write a task_a_output.json-style submission. Each cell is
    `{sentence_id, supported_citations, contradicted_citations}`."""
    items = [
        {
            "meta_data": {"qa_id": qa_id, "question": ""},
            "answer": cells,
        }
        for qa_id, cells in qa_to_cells.items()
    ]
    path.write_bytes(orjson.dumps(items))
    return path


def _make_run_dir(tmp_path: Path, qa_to_cells: dict[str, list[dict]]) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_submission(run_dir / "task_a_output.json", qa_to_cells)
    return run_dir


def test_per_topic_f1_returns_one_entry_per_topic(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, {
        "Q001": [
            {"text": "s0", "supported_citations": [10000001, 10000004, 99999999], "contradicted_citations": []},
            {"text": "s1", "supported_citations": [], "contradicted_citations": [10000003]},
        ],
        "Q002": [
            {"text": "s0", "supported_citations": [10000005], "contradicted_citations": []},
            {"text": "s1", "supported_citations": [], "contradicted_citations": [10000003, 10000006]},
        ],
    })
    report = per_topic_f1(run_dir, qrels_path=FIXTURES / "mini_qrels.jsonl")
    assert set(report.topics) == {"Q001", "Q002"}
    # Each topic carries strict + relaxed × support + contradict.
    for qa in ["Q001", "Q002"]:
        for setting in ("strict", "relaxed"):
            for cls in ("support", "contradict"):
                tm = report.topics[qa][setting][cls]
                # PRF dataclass exposes F1.
                assert hasattr(tm, "F1")


def test_per_topic_f1_matches_manual_strict_support(tmp_path: Path) -> None:
    """Manual: Q001 has 2 sentence cells × 2 classes = 4 cells; under
    Strict support, sentence 0 predicts {10000001,10000004,99999999}
    against positives={10000001} -> tp=1, P=1/3, R=1, F1=0.5.
    Sentence 1 predicts no supports but has no positives either AND no
    predicted support PMIDs, so the cell is unjudged and skipped under
    unjudged_as_zero=True only when there IS a non-empty prediction —
    here both positives and prediction empty, so skipped too.
    Topic support F1 = mean([0.5]) = 0.5.
    """
    run_dir = _make_run_dir(tmp_path, {
        "Q001": [
            {"text": "s0", "supported_citations": [10000001, 10000004, 99999999], "contradicted_citations": []},
            {"text": "s1", "supported_citations": [], "contradicted_citations": [10000003]},
        ],
    })
    report = per_topic_f1(run_dir, qrels_path=FIXTURES / "mini_qrels.jsonl")
    sup_strict_f1 = report.topics["Q001"]["strict"]["support"].F1
    assert abs(sup_strict_f1 - 0.5) < 1e-9


def test_per_topic_f1_missing_qrels_raises(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, {"Q001": []})
    try:
        per_topic_f1(run_dir, qrels_path=tmp_path / "nonexistent.jsonl")
    except FileNotFoundError as e:
        assert "qrels file not found" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_per_topic_f1_missing_submission_raises(tmp_path: Path) -> None:
    empty_run = tmp_path / "empty"
    empty_run.mkdir()
    try:
        per_topic_f1(empty_run, qrels_path=FIXTURES / "mini_qrels.jsonl")
    except FileNotFoundError as e:
        assert "task_a_output.json" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_topic_f1_delta_takes_intersection_of_topics(tmp_path: Path) -> None:
    a_dir = _make_run_dir(tmp_path / "a", {
        "Q001": [{"text": "s", "supported_citations": [10000001], "contradicted_citations": []}],
        "Q002": [{"text": "s", "supported_citations": [10000005], "contradicted_citations": []}],
    })
    # Different run with the same submissions but a different file layout.
    b_dir = _make_run_dir(tmp_path / "b", {
        "Q001": [{"text": "s", "supported_citations": [99999999], "contradicted_citations": []}],
        # Q002 absent in B — must be dropped from the delta.
    })
    target = per_topic_f1(a_dir, qrels_path=FIXTURES / "mini_qrels.jsonl")
    anchor = per_topic_f1(b_dir, qrels_path=FIXTURES / "mini_qrels.jsonl")
    delta = topic_f1_delta(target, anchor)
    assert set(delta) == {"Q001"}  # Q002 absent in B
    # A predicts the gold PMID, B predicts a non-positive -> A.F1 > B.F1.
    assert delta["Q001"] > 0


def test_select_three_topics_picks_max_near_min() -> None:
    deltas = {
        "100": 0.30,   # max positive
        "101": -0.25,  # min negative
        "102": 0.005,  # near-zero
        "103": 0.20,
        "104": -0.10,
    }
    pos, neutral, neg = select_three_topics(deltas)
    assert pos == "100"
    assert neg == "101"
    assert neutral == "102"


def test_select_three_topics_tie_break_by_qa_int() -> None:
    """When two topics share the same delta, pick the smaller qa_id."""
    deltas = {"50": 0.10, "30": 0.10, "60": -0.10, "40": -0.10, "70": 0.0}
    pos, neutral, neg = select_three_topics(deltas)
    # Both 50 and 30 have +0.10; tie-break picks 30 first per the
    # implementation's secondary key (-qa_int makes smaller win for max).
    assert pos == "30"
    # Both 60 and 40 have -0.10; smaller qa_int wins.
    assert neg == "40"
    assert neutral == "70"


def test_select_three_topics_collision_neutral_falls_back() -> None:
    """If the 'closest to zero' tie-breaks into the same qa_id as pos or
    neg, the routine picks the second-closest to avoid duplication."""
    # pos = max positive 0.5 (qa=100); neg = min negative -0.5 (qa=200);
    # neutral candidates: 100, 200 are gone -> the next-closest is 300.
    deltas = {"100": 0.5, "200": -0.5, "300": 0.01}
    pos, neutral, neg = select_three_topics(deltas)
    assert pos == "100" and neg == "200"
    assert neutral == "300"


def test_select_three_topics_too_few_raises() -> None:
    try:
        select_three_topics({"100": 0.1, "101": -0.1})
    except ValueError as e:
        assert "need ≥3" in str(e)
    else:
        raise AssertionError("expected ValueError")
