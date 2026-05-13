"""Submission writer + validator tests (9.3, 9.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trec_biogen.io.submission import (
    CAP,
    SubmissionValidationError,
    validate,
    write_submission,
)


def _selection() -> dict:
    return {
        "Q001": {
            0: {"support": ["10000001", "10000004"], "contradict": []},
            1: {"support": [], "contradict": ["10000003"]},
        },
        "Q002": {
            0: {"support": ["10000005"], "contradict": []},
            1: {"support": [], "contradict": ["10000006"]},
        },
    }


def test_write_submission_shape(tmp_path: Path) -> None:
    out = tmp_path / "submission.jsonl"
    write_submission(_selection(), out)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)
        assert {"qa_id", "sentences"} <= rec.keys()
        for sent in rec["sentences"]:
            assert {"sentence_id", "contradict_pmids", "support_pmids"} <= sent.keys()


def test_validate_passes_on_clean_submission(tmp_path: Path) -> None:
    out = tmp_path / "submission.jsonl"
    write_submission(_selection(), out)
    validate(out)  # should not raise


def test_validate_rejects_cap_exceeded(tmp_path: Path) -> None:
    bad = _selection()
    bad["Q001"][0]["support"] = ["a", "b", "c", "d"]  # 4 > CAP
    out = tmp_path / "bad.jsonl"
    write_submission(bad, out)
    # The writer trims to CAP; emulate the original by writing directly.
    out.write_text(
        '{"qa_id":"Q001","sentences":[{"sentence_id":0,"contradict_pmids":[],'
        '"support_pmids":["a","b","c","d"]}]}\n'
    )
    with pytest.raises(SubmissionValidationError, match="cap exceeded"):
        validate(out)


def test_validate_rejects_global_dup(tmp_path: Path) -> None:
    out = tmp_path / "dup.jsonl"
    out.write_text(
        '{"qa_id":"Q1","sentences":['
        '{"sentence_id":0,"contradict_pmids":[],"support_pmids":["X"]},'
        '{"sentence_id":1,"contradict_pmids":["X"],"support_pmids":[]}'
        ']}\n'
    )
    with pytest.raises(SubmissionValidationError, match="already used in topic"):
        validate(out)


def test_validate_rejects_unordered_sentence_ids(tmp_path: Path) -> None:
    out = tmp_path / "ord.jsonl"
    out.write_text(
        '{"qa_id":"Q1","sentences":['
        '{"sentence_id":1,"contradict_pmids":[],"support_pmids":["X"]},'
        '{"sentence_id":0,"contradict_pmids":[],"support_pmids":["Y"]}'
        ']}\n'
    )
    with pytest.raises(SubmissionValidationError, match="not strictly increasing"):
        validate(out)


def test_validate_index_membership(tmp_path: Path) -> None:
    out = tmp_path / "m.jsonl"
    write_submission(_selection(), out)
    validate(out, index_pmids={"10000001", "10000004", "10000003", "10000005", "10000006"})
    with pytest.raises(SubmissionValidationError, match="not in corpus"):
        validate(out, index_pmids={"10000001"})


def test_write_trims_excess_pmids(tmp_path: Path) -> None:
    sel = {"Q1": {0: {"support": ["a", "b", "c", "d", "e"], "contradict": []}}}
    out = tmp_path / "trim.jsonl"
    write_submission(sel, out)
    rec = json.loads(out.read_text().strip())
    assert rec["sentences"][0]["support_pmids"] == ["a", "b", "c"]
    assert CAP == 3
