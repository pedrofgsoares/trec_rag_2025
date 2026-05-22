"""Concordance validator unit tests (task 2.14).

Drives :func:`run_validation` with a :class:`RecordedBackend` so the test
is fully offline. The fixture mixes correct, wrong, and out-of-class
predictions so each row of the confusion matrix gets exercised.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from trec_biogen.judge.backends import Judge, JudgeRecord, RecordedBackend
from trec_biogen.judge.validator import (
    Triple,
    bootstrap_ci,
    load_validation_triples,
    render_report,
    run_validation,
    score,
)


def _rec(label: str, conf: float = 0.9) -> JudgeRecord:
    return JudgeRecord(
        label=label,
        confidence=conf,
        input_tokens=10,
        output_tokens=5,
        backend="recorded",
        cost_usd=0.0,
    )


def test_score_perfect_pairs() -> None:
    pairs = [("Supports", "Supports")] * 5 + [("Contradicts", "Contradicts")] * 3
    r = score(pairs)
    assert r.total == 8
    assert r.per_class["Supports"].f1 == pytest.approx(1.0)
    assert r.per_class["Contradicts"].f1 == pytest.approx(1.0)
    assert r.macro_weighted_f1 == pytest.approx(1.0)
    assert r.passes(0.85) is True


def test_score_partial_disagreement_weighted_macro() -> None:
    # 5 supports: 3 correct, 1 -> Contradicts, 1 -> Neutral.
    # 3 contradicts: 2 correct, 1 -> Supports.
    pairs = (
        [("Supports", "Supports")] * 3
        + [("Supports", "Contradicts")]
        + [("Supports", "Neutral")]
        + [("Contradicts", "Contradicts")] * 2
        + [("Contradicts", "Supports")]
    )
    r = score(pairs)
    # Supports: TP=3, FP=1 (from contradict→supports), FN=2 → P=3/4, R=3/5 → F1≈0.6667
    sup = r.per_class["Supports"]
    assert sup.precision == pytest.approx(0.75)
    assert sup.recall == pytest.approx(0.6)
    assert sup.f1 == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))
    # Contradicts: TP=2, FP=1 (from support→contradicts), FN=1 → P=2/3, R=2/3 → F1=2/3
    con = r.per_class["Contradicts"]
    assert con.precision == pytest.approx(2 / 3)
    assert con.recall == pytest.approx(2 / 3)
    assert con.f1 == pytest.approx(2 / 3)
    expected_macro = (sup.f1 * 5 + con.f1 * 3) / 8
    assert r.macro_weighted_f1 == pytest.approx(expected_macro)
    assert r.passes(0.85) is False
    assert r.passes(0.5) is True


def test_score_zero_support_classes_do_not_break() -> None:
    pairs = [("Supports", "Supports")] * 4
    r = score(pairs)
    assert r.per_class["Neutral"].support == 0
    assert r.per_class["Neutral"].f1 == 0.0
    assert r.macro_weighted_f1 == pytest.approx(1.0)


def test_run_validation_uses_judge_and_returns_records() -> None:
    triples = [
        Triple(qa_id="1", sentence_id=0, pmid="P1", human_label="Supports"),
        Triple(qa_id="1", sentence_id=0, pmid="P2", human_label="Supports"),
        Triple(qa_id="2", sentence_id=1, pmid="P3", human_label="Contradicts"),
    ]
    abstracts = {"P1": "abstract one", "P2": "abstract two", "P3": "abstract three"}
    sentences = {("1", 0): "sentence one", ("2", 1): "sentence two"}

    recorded = {
        RecordedBackend.key_for("sentence one", "abstract one"): _rec("Supports"),
        RecordedBackend.key_for("sentence one", "abstract two"): _rec("Contradicts"),
        RecordedBackend.key_for("sentence two", "abstract three"): _rec("Contradicts"),
    }
    judge = Judge(RecordedBackend(recorded))

    result, records = run_validation(
        judge, triples,
        abstract_lookup=lambda p: abstracts[p],
        answer_sentence_lookup=lambda q, s: sentences[(q, s)],
    )
    assert len(records) == 3
    assert result.per_class["Supports"].support == 2
    assert result.per_class["Contradicts"].support == 1
    # 1 of 2 supports correct, 1/1 contradict correct.
    assert result.per_class["Supports"].recall == pytest.approx(0.5)
    assert result.per_class["Contradicts"].recall == pytest.approx(1.0)


def test_empty_abstract_short_circuits_to_not_relevant() -> None:
    backend = RecordedBackend({})  # empty mapping: any hit would raise
    judge = Judge(backend)
    rec = judge.classify("does X cause Y?", "P0", "   ")
    assert rec.label == "Not relevant"
    assert rec.skip_reason == "empty_abstract"
    assert rec.cost_usd == 0.0


def test_load_validation_triples_skips_zero_relevance(tmp_path: Path) -> None:
    qrels = tmp_path / "q.jsonl"
    qrels.write_text(
        "\n".join(
            [
                orjson.dumps({"qa_id": "1", "sentence_id": 0, "pmid": "P1",
                              "class": "support", "relevance": 1}).decode(),
                orjson.dumps({"qa_id": "1", "sentence_id": 0, "pmid": "P2",
                              "class": "partial_support", "relevance": 1}).decode(),
                orjson.dumps({"qa_id": "1", "sentence_id": 0, "pmid": "P3",
                              "class": "contradict", "relevance": 1}).decode(),
                orjson.dumps({"qa_id": "1", "sentence_id": 0, "pmid": "P4",
                              "class": "support", "relevance": 0}).decode(),
                orjson.dumps({"qa_id": "1", "sentence_id": 0, "pmid": "P5",
                              "class": "unknown", "relevance": 1}).decode(),
            ]
        )
    )
    triples = load_validation_triples(qrels)
    assert [t.pmid for t in triples] == ["P1", "P2", "P3"]
    assert [t.human_label for t in triples] == ["Supports", "Supports", "Contradicts"]


def test_render_report_includes_verdict_and_confusion(tmp_path: Path) -> None:
    pairs = [("Supports", "Supports"), ("Contradicts", "Supports")]
    r = score(pairs)
    body = render_report(r, backend_name="recorded", qrels_path=Path("q.jsonl"), threshold=0.85)
    assert "LLM-Judge Concordance Validation" in body
    assert "Confusion matrix" in body
    assert "FAIL" in body  # macro weighted F1 = 0.5


# -- §12.1 bootstrap CI ----------------------------------------------------


def test_bootstrap_ci_perfect_pairs_returns_unity_with_tight_interval() -> None:
    pairs = [("Supports", "Supports")] * 50 + [("Contradicts", "Contradicts")] * 10
    point, lo, hi = bootstrap_ci(pairs, n_iter=200, seed=0)
    assert point == pytest.approx(1.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_bootstrap_ci_contains_point_estimate() -> None:
    # A noisy sample: 80% correct supports, 70% correct contradicts.
    pairs = (
        [("Supports", "Supports")] * 40
        + [("Supports", "Neutral")] * 10
        + [("Contradicts", "Contradicts")] * 7
        + [("Contradicts", "Supports")] * 3
    )
    point, lo, hi = bootstrap_ci(pairs, n_iter=500, seed=42)
    assert 0 <= lo <= point <= hi <= 1
    # CI width should be well within [0, 0.5] for this sample size.
    assert hi - lo < 0.5


def test_bootstrap_ci_deterministic_with_seed() -> None:
    pairs = [("Supports", "Supports")] * 10 + [("Contradicts", "Supports")] * 5
    a = bootstrap_ci(pairs, n_iter=100, seed=7)
    b = bootstrap_ci(pairs, n_iter=100, seed=7)
    assert a == b


def test_bootstrap_ci_empty_input() -> None:
    assert bootstrap_ci([], n_iter=100, seed=0) == (0.0, 0.0, 0.0)


def test_run_validation_persists_records_to_disk(tmp_path: Path) -> None:
    """run_validation with records_out= must write one JSONL row per triple
    with the gold/pred/confidence columns the bootstrap-CI and calibration
    analyses depend on."""
    import json
    triples = [
        Triple(qa_id="1", sentence_id=0, pmid="P1", human_label="Supports"),
        Triple(qa_id="2", sentence_id=1, pmid="P2", human_label="Contradicts"),
    ]
    abstracts = {"P1": "first abstract", "P2": "second abstract"}
    sentences = {("1", 0): "sentence one", ("2", 1): "sentence two"}
    recorded = {
        RecordedBackend.key_for("sentence one", "first abstract"): _rec("Supports", 0.91),
        RecordedBackend.key_for("sentence two", "second abstract"): _rec("Neutral", 0.55),
    }
    judge = Judge(RecordedBackend(recorded))
    out = tmp_path / "records.jsonl"

    run_validation(
        judge, triples,
        abstract_lookup=lambda p: abstracts[p],
        answer_sentence_lookup=lambda q, s: sentences[(q, s)],
        records_out=out,
    )

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 2
    assert rows[0]["qa_id"] == "1"
    assert rows[0]["gold"] == "Supports"
    assert rows[0]["pred"] == "Supports"
    assert rows[0]["confidence"] == pytest.approx(0.91)
    assert rows[1]["pred"] == "Neutral"
