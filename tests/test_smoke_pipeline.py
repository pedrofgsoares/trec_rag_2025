"""End-to-end smoke test (task 11.5).

Exercises the structural bottom half of the pipeline — selection → submission
writer → validator → metrics — using the 2-topic fixture and *synthetic* NLI
Parquets so it can run on CI without GPU, models, or a Lucene index.

The full pipeline (with real models) is exercised by the operator via
``python -m trec_biogen.pipeline.run_task_a`` once §1–6 setup is complete.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from trec_biogen.eval.metrics import evaluate
from trec_biogen.io.qrels import load_qrels
from trec_biogen.io.submission import (
    validate,
    validate_official,
    write_official_submission,
    write_submission,
)
from trec_biogen.io.topics import load_topics
from trec_biogen.pipeline.selection import SelectionConfig, select

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def synth_nli(tmp_path: Path) -> tuple[Path, Path]:
    """Build NLI parquets matching what real phases would emit, against the 2-topic fixture."""
    topics = load_topics(FIXTURES / "mini_input.json")
    sup_rows: list[dict] = []
    con_rows: list[dict] = []
    pmid_pool = {"Q001": ["10000001", "10000004", "99999999"], "Q002": ["10000005"]}
    con_pool = {"Q001": ["10000003"], "Q002": ["10000003", "10000006"]}

    for t in topics:
        for sid in range(len(t.sentences)):
            for pmid in pmid_pool[t.qa_id]:
                sup_rows.append({
                    "qa_id": t.qa_id, "sentence_id": sid,
                    "candidate_pmid": pmid,
                    "entailment_prob": 0.9 if sid == 0 else 0.1,
                    "neutral_prob": 0.05,
                    "contradiction_prob": 0.05,
                })
            for pmid in con_pool[t.qa_id]:
                con_rows.append({
                    "qa_id": t.qa_id, "sentence_id": sid,
                    "candidate_pmid": pmid,
                    "contradict_score": 0.9 if sid == 1 else 0.1,
                    "bm25_rank": 1 + len(con_rows),
                    "bm25_score": 10.0,
                })

    sup = tmp_path / "nli_support.parquet"
    con = tmp_path / "nli_contradict.parquet"
    pl.DataFrame(sup_rows).write_parquet(sup)
    pl.DataFrame(con_rows).write_parquet(con)
    return sup, con


def test_legacy_jsonl_submission_end_to_end(
    tmp_path: Path, synth_nli: tuple[Path, Path]
) -> None:
    sup, con = synth_nli
    topics = load_topics(FIXTURES / "mini_input.json")
    selection = select(
        nli_support=sup, nli_contradict=con,
        topics=topics,
        config=SelectionConfig(tau_sup=0.5, tau_con=0.5, cap=3),
    )
    out = tmp_path / "submission.jsonl"
    write_submission(selection, out)
    validate(out)

    qrels = load_qrels(FIXTURES / "mini_qrels.jsonl")
    report = evaluate(out, qrels)
    for setting in ("strict", "relaxed"):
        for cls in ("support", "contradict"):
            f1 = report[setting][cls]["F1"]
            assert 0.0 <= f1 <= 1.0


def test_official_submission_validates_and_scores(
    tmp_path: Path, synth_nli: tuple[Path, Path]
) -> None:
    sup, con = synth_nli
    topics = load_topics(FIXTURES / "mini_input.json")
    selection = select(
        nli_support=sup, nli_contradict=con, topics=topics, config=SelectionConfig(),
    )
    out = tmp_path / "task_a_output.json"
    write_official_submission(selection, topics, out)
    validate_official(out)

    # Eval reads either form; the official file should score the same as the
    # legacy form would for equivalent selections.
    qrels = load_qrels(FIXTURES / "mini_qrels.jsonl")
    report = evaluate(out, qrels)
    assert report["strict"]["support"]["F1"] >= 0.0


def test_existing_supported_citations_are_excluded(tmp_path: Path) -> None:
    """Track rule: existing_supported_citations must never appear in supported/contradicted."""
    topics = load_topics(FIXTURES / "mini_input.json")
    # Q002 sentence 0 has existing {99000123}. If retrieval surfaces it, selection must skip it.
    sup = pl.DataFrame([{
        "qa_id": "Q002", "sentence_id": 0, "candidate_pmid": "99000123",
        "entailment_prob": 0.99, "neutral_prob": 0.0, "contradiction_prob": 0.01,
    }, {
        "qa_id": "Q002", "sentence_id": 0, "candidate_pmid": "55555555",
        "entailment_prob": 0.95, "neutral_prob": 0.0, "contradiction_prob": 0.05,
    }])
    con = pl.DataFrame([{
        "qa_id": "Q002", "sentence_id": 0, "candidate_pmid": "DUMMY",
        "contradict_score": 0.0, "bm25_rank": 1, "bm25_score": 0.0,
    }])
    sup_p = tmp_path / "s.parquet"
    con_p = tmp_path / "c.parquet"
    sup.write_parquet(sup_p)
    con.write_parquet(con_p)

    selection = select(
        nli_support=sup_p, nli_contradict=con_p, topics=topics, config=SelectionConfig(),
    )
    chosen = selection["Q002"][0]["support"]
    assert "99000123" not in chosen, "existing citation must be excluded"
    assert "55555555" in chosen


def test_selection_applies_global_dedup(tmp_path: Path) -> None:
    """Same PMID strong in two sentences must appear only once per topic (D9)."""
    sup = pl.DataFrame([
        {"qa_id": "QX", "sentence_id": 0, "candidate_pmid": "P1",
         "entailment_prob": 0.95, "neutral_prob": 0.0, "contradiction_prob": 0.05},
        {"qa_id": "QX", "sentence_id": 1, "candidate_pmid": "P1",
         "entailment_prob": 0.95, "neutral_prob": 0.0, "contradiction_prob": 0.05},
        {"qa_id": "QX", "sentence_id": 1, "candidate_pmid": "P2",
         "entailment_prob": 0.90, "neutral_prob": 0.0, "contradiction_prob": 0.10},
    ])
    con = pl.DataFrame([
        {"qa_id": "QX", "sentence_id": 0, "candidate_pmid": "C1",
         "contradict_score": 0.8, "bm25_rank": 1, "bm25_score": 10.0},
    ])
    sup_p = tmp_path / "s.parquet"
    con_p = tmp_path / "c.parquet"
    sup.write_parquet(sup_p)
    con.write_parquet(con_p)

    # No topics passed -> no existing exclusions; dedup still applies.
    selection = select(nli_support=sup_p, nli_contradict=con_p, config=SelectionConfig())
    assert selection["QX"][0]["support"] == ["P1"]
    assert selection["QX"][1]["support"] == ["P2"]
