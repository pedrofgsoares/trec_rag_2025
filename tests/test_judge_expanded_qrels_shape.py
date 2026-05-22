"""Expanded qrels file-shape tests (task 2.14).

Drives :func:`emit_expanded_qrels` directly with a small canned LLM-record
mapping. Verifies that human rows are preserved verbatim with ``source:
human`` appended, LLM rows carry ``source`` + ``confidence``, the JSONL
parses back through :func:`trec_biogen.io.qrels.load_qrels` without error,
and the ``incomplete`` flag is captured in the sidecar.
"""

from __future__ import annotations

import json
from pathlib import Path

import orjson

from trec_biogen.io.qrels import load_qrels
from trec_biogen.judge.backends import JudgeRecord
from trec_biogen.judge.rejudge import (
    emit_expanded_qrels,
    novel_pmids_from_submission,
)


def _write_human_qrels(path: Path) -> None:
    rows = [
        {"qa_id": "116", "sentence_id": 1, "pmid": "29517065",
         "class": "support", "relevance": 1},
        {"qa_id": "116", "sentence_id": 1, "pmid": "29797754",
         "class": "support", "relevance": 1},
        {"qa_id": "127", "sentence_id": 0, "pmid": "37763239",
         "class": "contradict", "relevance": 1},
    ]
    path.write_text("\n".join(orjson.dumps(r).decode() for r in rows) + "\n")


def test_emit_expanded_qrels_preserves_human_rows_and_appends_llm(tmp_path: Path) -> None:
    human = tmp_path / "qrels.jsonl"
    _write_human_qrels(human)

    llm_records = {
        ("127", 0, "31813888"): JudgeRecord(
            label="Supports", confidence=0.84, input_tokens=120,
            output_tokens=15, backend="recorded", cost_usd=0.0001,
        ),
        ("127", 0, "15638715"): JudgeRecord(
            label="Contradicts", confidence=0.72, input_tokens=130,
            output_tokens=15, backend="recorded", cost_usd=0.00012,
        ),
        ("127", 0, "9999999"): JudgeRecord(
            label="Neutral", confidence=0.55, input_tokens=110,
            output_tokens=15, backend="recorded", cost_usd=0.00011,
        ),
    }
    out = tmp_path / "expanded.jsonl"
    emit_expanded_qrels(
        human_qrels_path=human,
        llm_records=llm_records,
        out_path=out,
        incomplete=False,
        abort_reason="",
    )

    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    # 3 human + 2 LLM (Neutral is dropped — it's neither support nor contradict).
    assert len(lines) == 5
    human_lines = [r for r in lines if r["source"] == "human"]
    assert len(human_lines) == 3
    assert all("confidence" not in r for r in human_lines)
    llm_lines = [r for r in lines if r["source"] == "recorded"]
    assert {r["pmid"] for r in llm_lines} == {"31813888", "15638715"}
    assert all("confidence" in r for r in llm_lines)
    assert all(r["class"] in ("support", "contradict") for r in llm_lines)

    sidecar = out.with_suffix(out.suffix + ".meta.json")
    meta = json.loads(sidecar.read_text())
    assert meta["incomplete"] is False
    assert meta["llm_record_count"] == 3


def test_emit_expanded_qrels_marks_incomplete_in_sidecar(tmp_path: Path) -> None:
    human = tmp_path / "qrels.jsonl"
    _write_human_qrels(human)

    out = tmp_path / "expanded.jsonl"
    emit_expanded_qrels(
        human_qrels_path=human,
        llm_records={
            ("127", 0, "12345"): JudgeRecord(
                label="Supports", confidence=0.9, input_tokens=10,
                output_tokens=5, backend="recorded", cost_usd=0.0,
            ),
        },
        out_path=out,
        incomplete=True,
        abort_reason="cost_cap_reached:$5.00",
    )
    meta = json.loads(out.with_suffix(out.suffix + ".meta.json").read_text())
    assert meta["incomplete"] is True
    assert meta["abort_reason"] == "cost_cap_reached:$5.00"


def test_expanded_qrels_loads_through_load_qrels(tmp_path: Path) -> None:
    """The expanded JSONL must remain parseable by the existing QrelsIndex loader."""
    human = tmp_path / "qrels.jsonl"
    _write_human_qrels(human)
    out = tmp_path / "expanded.jsonl"
    emit_expanded_qrels(
        human_qrels_path=human,
        llm_records={
            ("127", 0, "31813888"): JudgeRecord(
                label="Supports", confidence=0.84, input_tokens=10,
                output_tokens=5, backend="recorded", cost_usd=0.0,
            ),
        },
        out_path=out,
    )
    idx = load_qrels(out)
    assert "31813888" in idx.positives("127", 0, "support")
    assert "29517065" in idx.positives("116", 1, "support")


def test_novel_pmids_from_submission_excludes_already_judged(tmp_path: Path) -> None:
    human = tmp_path / "qrels.jsonl"
    _write_human_qrels(human)
    submission = tmp_path / "task_a_output.json"
    submission.write_text(
        json.dumps(
            [
                {
                    "meta_data": {"qa_id": "116", "question": "x"},
                    "answer": [
                        {"text": "s0", "supported_citations": [], "contradicted_citations": []},
                        {
                            "text": "s1",
                            "supported_citations": [29517065, 11111111],  # one in qrels, one novel
                            "contradicted_citations": [22222222],
                        },
                    ],
                },
                {
                    "meta_data": {"qa_id": "127", "question": "y"},
                    "answer": [
                        {
                            "text": "s0",
                            "supported_citations": [31813888],
                            "contradicted_citations": [37763239],  # already in qrels as contradict
                        }
                    ],
                },
            ]
        )
    )
    novel = novel_pmids_from_submission(submission, human)
    triples = set(novel)
    assert ("116", 1, "11111111") in triples
    assert ("116", 1, "22222222") in triples
    assert ("127", 0, "31813888") in triples
    # Already in qrels:
    assert ("116", 1, "29517065") not in triples
    assert ("127", 0, "37763239") not in triples
