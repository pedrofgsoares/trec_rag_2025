"""Load the official BioGen Task A input JSONL.

Expected per-line schema (one topic per line)::

    {
      "metadata": {"qa_id": "<question id>"},
      "question": "<natural-language question>",
      "answer":   "<multi-sentence candidate answer>",
      "topic_id": "<optional, defaults to qa_id>"
    }

The pipeline segments ``answer`` into sentences downstream (scispaCy). Topics
are returned as plain dataclasses to keep callers free of pydantic /
omegaconf at the boundary.

Task: 5.1, 5.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import orjson


class TopicLoadError(ValueError):
    """Raised on malformed Task A input. Always carries the offending line number."""


@dataclass(slots=True)
class Topic:
    qa_id: str
    question: str
    answer: str
    topic_id: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.topic_id:
            self.topic_id = self.qa_id


def _validate(rec: dict, lineno: int) -> Topic:
    meta = rec.get("metadata") or {}
    qa_id = meta.get("qa_id")
    if not qa_id or not isinstance(qa_id, str):
        raise TopicLoadError(f"line {lineno}: missing or non-string metadata.qa_id")
    answer = rec.get("answer")
    if not answer or not isinstance(answer, str):
        raise TopicLoadError(f"line {lineno}: missing or empty 'answer' string")
    question = rec.get("question") or ""
    if not isinstance(question, str):
        raise TopicLoadError(f"line {lineno}: 'question' must be string if present")
    return Topic(
        qa_id=qa_id,
        question=question,
        answer=answer,
        topic_id=str(rec.get("topic_id") or qa_id),
        metadata=meta,
    )


def iter_topics(path: Path) -> Iterator[Topic]:
    """Stream topics from a JSONL file; fail-fast with line number on error."""
    with path.open("rb") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = orjson.loads(raw)
            except orjson.JSONDecodeError as e:
                raise TopicLoadError(f"line {lineno}: invalid JSON: {e}") from e
            yield _validate(rec, lineno)


def load_topics(path: Path) -> list[Topic]:
    return list(iter_topics(path))
