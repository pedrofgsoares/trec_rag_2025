"""Phase 2.5 §5.4 — scripts/per_topic_diff.py smoke tests.

Importable smoke: the two mode entry-points (`cmd_qa_id_diff` and
`cmd_select_3`) run cleanly against synthetic fixtures and produce
non-empty output, exercising the qrels lookup and the rejudge-record
join paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import orjson
import pytest

import sys
SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from per_topic_diff import (  # type: ignore  # noqa: E402
    cmd_qa_id_diff, cmd_select_3, _resolve_pool_path,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _write_submission(path: Path, qa_to_cells: dict) -> Path:
    items = [
        {"meta_data": {"qa_id": qa, "question": ""}, "answer": cells}
        for qa, cells in qa_to_cells.items()
    ]
    path.write_bytes(orjson.dumps(items))
    return path


def _make_run(tmp_path: Path, name: str, qa_to_cells: dict) -> Path:
    run = tmp_path / name
    run.mkdir(parents=True, exist_ok=True)
    _write_submission(run / "task_a_output.json", qa_to_cells)
    return run


def test_qa_id_diff_smoke_runs_and_finds_set_difference(tmp_path: Path, capsys) -> None:
    a = _make_run(tmp_path, "a", {
        "Q001": [
            {"text": "s0", "supported_citations": [10000001, 99999999],
             "contradicted_citations": []},
            {"text": "s1", "supported_citations": [], "contradicted_citations": [10000003]},
        ],
    })
    b = _make_run(tmp_path, "b", {
        "Q001": [
            {"text": "s0", "supported_citations": [10000001, 10000004],
             "contradicted_citations": []},
            {"text": "s1", "supported_citations": [], "contradicted_citations": [10000003]},
        ],
    })
    ns = argparse.Namespace(
        pool="official",
        qrels_path=FIXTURES / "mini_qrels.jsonl",
        qa_id="Q001",
        a=a, b=b,
    )
    exit_code = cmd_qa_id_diff(ns)
    assert exit_code == 0
    out = capsys.readouterr().out
    # Identifies set difference on sentence 0 support.
    assert "A \\ B" in out
    assert "99999999" in out  # in A but not B
    assert "B \\ A" in out
    assert "10000004" in out  # in B but not A
    # And same-set sentences print "sets identical".
    assert "(sets identical)" in out


def test_select_3_smoke_runs_and_picks(tmp_path: Path, capsys) -> None:
    # Build three topics whose target-vs-anchor strict-support F1s give
    # us a clean +, ~0, - spread.
    target = _make_run(tmp_path, "target", {
        "Q001": [{"text": "", "supported_citations": [10000001], "contradicted_citations": []}],  # +
        "Q002": [{"text": "", "supported_citations": [10000005], "contradicted_citations": []}],  # ~0
        "Q003": [{"text": "", "supported_citations": [99999999], "contradicted_citations": []}],  # -
    })
    anchor = _make_run(tmp_path, "anchor", {
        "Q001": [{"text": "", "supported_citations": [99999999], "contradicted_citations": []}],
        "Q002": [{"text": "", "supported_citations": [10000005], "contradicted_citations": []}],
        "Q003": [{"text": "", "supported_citations": [10000001], "contradicted_citations": []}],
    })
    # Mini qrels has positives for Q001 s0 (10000001) and Q002 s0 (10000005);
    # Q003 needs a positive too — synthesise a small qrels.
    qrels_path = tmp_path / "q.jsonl"
    qrels_path.write_text(
        '\n'.join([
            orjson.dumps({"qa_id": "Q001", "sentence_id": 0, "pmid": "10000001", "class": "support", "relevance": 1}).decode(),
            orjson.dumps({"qa_id": "Q002", "sentence_id": 0, "pmid": "10000005", "class": "support", "relevance": 1}).decode(),
            orjson.dumps({"qa_id": "Q003", "sentence_id": 0, "pmid": "10000001", "class": "support", "relevance": 1}).decode(),
        ])
    )
    ns = argparse.Namespace(
        pool="official", qrels_path=qrels_path,
        target=target, anchor=anchor,
        setting="strict", cls="support",
        json_out=None,
    )
    exit_code = cmd_select_3(ns)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Mechanical 3-topic selection" in out
    assert "largest positive Δ" in out
    assert "closest-to-zero Δ" in out
    assert "largest negative Δ" in out
    # The full sorted appendix prints all three qa_ids.
    assert "Q001" in out and "Q002" in out and "Q003" in out


def test_resolve_pool_path_explicit_wins(tmp_path: Path) -> None:
    explicit = tmp_path / "q.jsonl"
    explicit.write_text("")
    p = _resolve_pool_path(explicit, pool="intersection")
    assert p == explicit


def test_resolve_pool_path_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _resolve_pool_path(tmp_path / "nope.jsonl", pool="intersection")
