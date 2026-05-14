"""Topic + qrels loader tests. Tasks 5.1, 5.2, 5.3, 5.4."""

from __future__ import annotations

from pathlib import Path

import pytest

from trec_biogen.io.qrels import load_qrels
from trec_biogen.io.topics import TopicLoadError, load_topics

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_topics_official_format() -> None:
    topics = load_topics(FIXTURES / "mini_input.json")
    assert len(topics) == 2
    t = topics[0]
    assert t.qa_id == "Q001"
    assert t.topic_id == "Q001"
    assert len(t.sentences) == 2
    assert "Aspirin" in t.sentences[0]
    assert "no evidence" in t.sentences[1].lower()


def test_existing_per_sentence_carried() -> None:
    topics = load_topics(FIXTURES / "mini_input.json")
    by_qa = {t.qa_id: t for t in topics}
    # Q002 sentence 0 has one existing citation; sentence 1 has none.
    assert by_qa["Q002"].existing_per_sentence[0] == {"99000123"}
    assert by_qa["Q002"].existing_per_sentence[1] == set()
    # Q001 has none on either sentence.
    assert all(s == set() for s in by_qa["Q001"].existing_per_sentence)


def test_answer_property_joins_sentences() -> None:
    topics = load_topics(FIXTURES / "mini_input.json")
    t = topics[0]
    assert t.answer == " ".join(t.sentences)


def test_topics_fail_fast_missing_qa_id(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(
        '[{"meta_data": {}, "answer": [{"text": "x"}]},'
        '{"meta_data": {"qa_id": "ok"}, "answer": [{"text": "a"}]}]'
    )
    with pytest.raises(TopicLoadError) as exc:
        load_topics(p)
    assert "qa_id" in str(exc.value)


def test_topics_fail_fast_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text('[{"meta_data": {"qa_id": "ok"}, "answer": [{"text": "a"}]}, {not json}]')
    with pytest.raises(TopicLoadError):
        load_topics(p)


def test_topics_fail_fast_empty_answer(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text('[{"meta_data": {"qa_id": "X"}, "answer": []}]')
    with pytest.raises(TopicLoadError, match="non-empty list"):
        load_topics(p)


def test_qrels_strict_vs_relaxed() -> None:
    idx = load_qrels(FIXTURES / "mini_qrels.jsonl")
    assert idx.positives("Q001", 0, "support", setting="strict") == {"10000001"}
    assert idx.positives("Q001", 0, "support", setting="relaxed") == {"10000001", "10000004"}
    assert idx.positives("Q001", 1, "contradict", setting="strict") == {"10000003"}
    assert idx.positives("Q002", 1, "contradict", setting="strict") == {"10000003"}
    assert idx.positives("Q002", 1, "contradict", setting="relaxed") == {"10000003", "10000006"}


def test_qrels_unknown_key_returns_empty() -> None:
    idx = load_qrels(FIXTURES / "mini_qrels.jsonl")
    assert idx.positives("NONEXISTENT", 0, "support") == set()
