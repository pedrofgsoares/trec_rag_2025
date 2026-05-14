"""Load the official BioGen Task A input.

The 2025 distribution is a single JSON file (list of items), each with::

    {
      "meta_data": {"qa_id": "116", "question": "..."},
      "answer": [
        {"text": "<sentence>", "existing_supported_citations": null|[int,...]},
        ...
      ]
    }

Notes on conventions in the upstream file:
* Top-level key is ``meta_data`` (with underscore), not ``metadata``.
* ``answer`` is a pre-segmented list of sentence dicts — we use this list
  directly instead of re-segmenting with scispaCy, since the official
  segmentation is what the qrels and the starter-kit baseline align to.
* ``existing_supported_citations`` carries PMIDs that are already cited in
  the source answer; the track rule (and starter-kit code) excludes these
  from newly-predicted supports.

We also keep backward compatibility with the legacy JSONL form used in
existing tests::

    {"metadata": {"qa_id": "..."}, "question": "...", "answer": "..."}

In that form the ``answer`` is a single string and is split via scispaCy at
read time.

Task: 5.1, 5.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import orjson


class TopicLoadError(ValueError):
    """Raised on malformed Task A input. Carries the offending line/item number."""


@dataclass(slots=True)
class Topic:
    qa_id: str
    question: str
    sentences: list[str]
    existing_per_sentence: list[set[str]] = field(default_factory=list)
    topic_id: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.topic_id:
            self.topic_id = self.qa_id
        if len(self.existing_per_sentence) < len(self.sentences):
            self.existing_per_sentence += [set()] * (
                len(self.sentences) - len(self.existing_per_sentence)
            )

    @property
    def answer(self) -> str:
        """Joined view of the answer sentences (for legacy callers / logs)."""
        return " ".join(self.sentences)


def _coerce_existing(raw: Any) -> set[str]:
    if raw is None:
        return set()
    if not isinstance(raw, (list, tuple)):
        return set()
    return {str(x) for x in raw}


def _topic_from_official(item: dict, where: str) -> Topic:
    meta = item.get("meta_data") or item.get("metadata") or {}
    qa_id = meta.get("qa_id")
    if not qa_id or not isinstance(qa_id, str):
        raise TopicLoadError(f"{where}: missing or non-string meta_data.qa_id")
    answers = item.get("answer")
    if not isinstance(answers, list) or not answers:
        raise TopicLoadError(f"{where}: 'answer' must be a non-empty list of sentence dicts")

    sentences: list[str] = []
    existing: list[set[str]] = []
    for i, a in enumerate(answers):
        if not isinstance(a, dict):
            raise TopicLoadError(f"{where}/answer[{i}]: expected dict, got {type(a).__name__}")
        text = a.get("text")
        if not text or not isinstance(text, str):
            raise TopicLoadError(f"{where}/answer[{i}]: missing or empty 'text'")
        sentences.append(text.strip())
        existing.append(_coerce_existing(a.get("existing_supported_citations")))

    return Topic(
        qa_id=qa_id,
        question=str(meta.get("question") or ""),
        sentences=sentences,
        existing_per_sentence=existing,
        topic_id=str(item.get("topic_id") or qa_id),
        metadata=meta,
    )


def _topic_from_legacy(rec: dict, lineno: int) -> Topic:
    """Legacy JSONL form: single answer string, scispaCy splits at read time."""
    meta = rec.get("metadata") or rec.get("meta_data") or {}
    qa_id = meta.get("qa_id")
    if not qa_id or not isinstance(qa_id, str):
        raise TopicLoadError(f"line {lineno}: missing or non-string metadata.qa_id")
    answer = rec.get("answer")
    if not answer or not isinstance(answer, str):
        raise TopicLoadError(f"line {lineno}: legacy form requires 'answer' as a string")
    from trec_biogen.pipeline.sentences import split_sentences

    sentences = split_sentences(answer) or [answer]
    return Topic(
        qa_id=qa_id,
        question=str(rec.get("question") or ""),
        sentences=sentences,
        existing_per_sentence=[set() for _ in sentences],
        topic_id=str(rec.get("topic_id") or qa_id),
        metadata=meta,
    )


def iter_topics(path: Path) -> Iterator[Topic]:
    """Stream topics from either the official ``.json`` (list) or legacy ``.jsonl`` form."""
    path = Path(path)
    raw = path.read_bytes().lstrip()
    if not raw:
        raise TopicLoadError(f"{path}: empty file")

    if raw[:1] == b"[":
        # Official JSON list.
        try:
            items = orjson.loads(raw)
        except orjson.JSONDecodeError as e:
            raise TopicLoadError(f"{path}: invalid JSON: {e}") from e
        for i, item in enumerate(items):
            yield _topic_from_official(item, where=f"{path.name} item {i + 1}")
        return

    # JSONL (legacy).
    with path.open("rb") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = orjson.loads(line)
            except orjson.JSONDecodeError as e:
                raise TopicLoadError(f"{path}:line {lineno}: invalid JSON: {e}") from e
            # If the line already has the official shape, use that loader.
            if isinstance(rec.get("answer"), list):
                yield _topic_from_official(rec, where=f"{path.name}:line {lineno}")
            else:
                yield _topic_from_legacy(rec, lineno)


def load_topics(path: Path) -> list[Topic]:
    return list(iter_topics(path))
