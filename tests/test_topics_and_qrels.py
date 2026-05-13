"""Topic + qrels loader tests. Tasks 5.1, 5.2, 5.3, 5.4."""

from __future__ import annotations

from pathlib import Path

import pytest

from trec_biogen.io.qrels import load_qrels
from trec_biogen.io.topics import TopicLoadError, load_topics

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_topics_basic() -> None:
    topics = load_topics(FIXTURES / "mini_input.jsonl")
    assert len(topics) == 2
    assert topics[0].qa_id == "Q001"
    assert topics[0].topic_id == "Q001"  # defaults to qa_id
    assert "Aspirin" in topics[0].answer


def test_topics_fail_fast_missing_qa_id(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(
        '{"metadata": {}, "answer": "x", "question": "y"}\n'
        '{"metadata": {"qa_id": "ok"}, "answer": "a", "question": "q"}\n'
    )
    with pytest.raises(TopicLoadError) as exc:
        load_topics(p)
    assert "line 1" in str(exc.value)
    assert "qa_id" in str(exc.value)


def test_topics_fail_fast_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text('{"metadata": {"qa_id": "ok"}, "answer": "a"}\n{not json\n')
    with pytest.raises(TopicLoadError) as exc:
        load_topics(p)
    assert "line 2" in str(exc.value)


def test_qrels_strict_vs_relaxed() -> None:
    idx = load_qrels(FIXTURES / "mini_qrels.jsonl")

    # Q001 sentence 0 support: strict has only 10000001, relaxed adds 10000004 (partial_support).
    assert idx.positives("Q001", 0, "support", setting="strict") == {"10000001"}
    assert idx.positives("Q001", 0, "support", setting="relaxed") == {"10000001", "10000004"}

    # Q001 sentence 1 contradict: only 10000003 in both.
    assert idx.positives("Q001", 1, "contradict", setting="strict") == {"10000003"}
    assert idx.positives("Q001", 1, "contradict", setting="relaxed") == {"10000003"}

    # Q002 sentence 1 contradict: strict {10000003}, relaxed adds 10000006 (partial_contradict).
    assert idx.positives("Q002", 1, "contradict", setting="strict") == {"10000003"}
    assert idx.positives("Q002", 1, "contradict", setting="relaxed") == {"10000003", "10000006"}


def test_qrels_unknown_key_returns_empty() -> None:
    idx = load_qrels(FIXTURES / "mini_qrels.jsonl")
    assert idx.positives("NONEXISTENT", 0, "support") == set()
