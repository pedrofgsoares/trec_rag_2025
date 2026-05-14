"""Submission writer + validator (9.3, 9.4).

Two output forms:

* **Official** (matches the starter-kit's ``task_a_output.json`` shape and
  passes ``task_a_validation.py``): a top-level JSON list, one entry per
  topic, with ``meta_data``, ``answer[]`` carrying each sentence's
  ``text``, ``existing_supported_citations``, ``supported_citations``,
  ``contradicted_citations``. PMIDs are emitted as **integers**.

* **Legacy JSONL** (used by the CI smoke test and historical fixtures):
  one record per line, with ``qa_id`` + ``sentences[]`` carrying
  ``contradict_pmids`` / ``support_pmids``.

The validator enforces:
* per-sentence caps ≤ 3 per class,
* no PMID re-used across (sentences, classes) within a topic (global dedup),
* every PMID present in the corpus (when ``index_pmids`` is supplied),
* no overlap between supported_citations and existing_supported_citations
  (the official rule explicitly forbids re-suggesting an existing citation).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import orjson

from trec_biogen.io.topics import Topic

CAP = 3


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_official_submission(
    selection: dict[str, dict[int, dict[str, list[str]]]],
    topics: Iterable[Topic],
    out_path: Path,
) -> Path:
    """Write the official ``task_a_output.json`` shape (single JSON list)."""
    by_qa = {t.qa_id: t for t in topics}
    out: list[dict[str, Any]] = []
    for qa_id, sentences in selection.items():
        topic = by_qa.get(qa_id)
        if topic is None:
            # Should not happen in normal use; we keep going so the writer
            # is robust to partial selections during ablations.
            continue
        answers = []
        for sid, text in enumerate(topic.sentences):
            picks = sentences.get(sid, {"support": [], "contradict": []})
            answers.append(
                {
                    "text": text,
                    "existing_supported_citations": (
                        sorted(int(p) for p in topic.existing_per_sentence[sid] if p.isdigit())
                        or None
                    ),
                    "supported_citations": _to_int_pmids(picks.get("support", []))[:CAP],
                    "contradicted_citations": _to_int_pmids(picks.get("contradict", []))[:CAP],
                }
            )
        out.append(
            {
                "meta_data": {
                    "qa_id": topic.qa_id,
                    "question": topic.question,
                },
                "answer": answers,
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out_path


def write_submission(
    selection: dict[str, dict[int, dict[str, list[str]]]],
    out_path: Path,
) -> Path:
    """Legacy JSONL writer — kept for the CI smoke test + historical fixtures."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        for qa_id, sentences in selection.items():
            record = {
                "qa_id": qa_id,
                "sentences": [
                    {
                        "sentence_id": sid,
                        "contradict_pmids": sentences[sid]["contradict"][:CAP],
                        "support_pmids": sentences[sid]["support"][:CAP],
                    }
                    for sid in sorted(sentences)
                ],
            }
            fh.write(orjson.dumps(record))
            fh.write(b"\n")
    return out_path


def _to_int_pmids(pmids: Iterable[str]) -> list[int]:
    out: list[int] = []
    for p in pmids:
        try:
            out.append(int(p))
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

class SubmissionValidationError(ValueError):
    """Raised on the first rule violation."""


def validate_official(
    submission_path: Path,
    *,
    index_pmids: set[str] | None = None,
) -> None:
    """Validate the official JSON-list submission. Mirrors task_a_validation.py + our rules."""
    data = json.loads(Path(submission_path).read_text())
    if not isinstance(data, list):
        raise SubmissionValidationError("top-level must be a JSON list of items")
    for idx, item in enumerate(data, start=1):
        meta = item.get("meta_data") or {}
        qa_id = meta.get("qa_id")
        if not qa_id:
            raise SubmissionValidationError(f"item {idx}: missing meta_data.qa_id")
        answers = item.get("answer", [])
        if not isinstance(answers, list):
            raise SubmissionValidationError(f"{qa_id}: 'answer' must be a list")

        used_in_topic: set[str] = set()
        for a_idx, ans in enumerate(answers):
            for field in ("text", "supported_citations", "contradicted_citations"):
                if field not in ans:
                    raise SubmissionValidationError(
                        f"{qa_id}/answer[{a_idx}]: missing field '{field}'"
                    )
            existing = set(ans.get("existing_supported_citations") or [])
            supported = ans.get("supported_citations") or []
            contradicted = ans.get("contradicted_citations") or []
            for pmids, cls in ((supported, "supported"), (contradicted, "contradicted")):
                _check_pmid_list(qa_id, a_idx, cls, pmids, used_in_topic, index_pmids)
                overlap = set(pmids) & existing
                if overlap:
                    raise SubmissionValidationError(
                        f"{qa_id}/answer[{a_idx}]/{cls}: overlap with existing: {overlap}"
                    )


def validate(submission_path: Path, *, index_pmids: set[str] | None = None) -> None:
    """Validate the legacy JSONL submission (used by tests and the smoke test)."""
    with submission_path.open("rb") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            rec = orjson.loads(raw)
            qa_id = rec.get("qa_id")
            if not qa_id or not isinstance(qa_id, str):
                raise SubmissionValidationError(f"line {lineno}: missing qa_id")
            sentences = rec.get("sentences")
            if not isinstance(sentences, list):
                raise SubmissionValidationError(f"{qa_id}: 'sentences' must be a list")

            seen_sid = -1
            used_in_topic: set[str] = set()
            for sent in sentences:
                sid = sent.get("sentence_id")
                if not isinstance(sid, int):
                    raise SubmissionValidationError(f"{qa_id}: sentence_id must be int")
                if sid <= seen_sid:
                    raise SubmissionValidationError(
                        f"{qa_id}/sid={sid}: sentence_id not strictly increasing"
                    )
                seen_sid = sid
                for cls in ("contradict_pmids", "support_pmids"):
                    pmids = sent.get(cls, [])
                    _check_pmid_list(qa_id, sid, cls, pmids, used_in_topic, index_pmids)


def _check_pmid_list(
    qa_id: str,
    sid: int | str,
    cls: str,
    pmids: Iterable,
    used_in_topic: set[str],
    index_pmids: set[str] | None,
) -> None:
    pmids_list = list(pmids)
    if len(pmids_list) > CAP:
        raise SubmissionValidationError(
            f"{qa_id}/sid={sid}/{cls}: cap exceeded ({len(pmids_list)}>{CAP})"
        )
    seen_here: set[str] = set()
    for pmid in pmids_list:
        if pmid is None or pmid == "":
            raise SubmissionValidationError(f"{qa_id}/sid={sid}/{cls}: empty pmid")
        pmid_s = str(pmid)
        if pmid_s in seen_here:
            raise SubmissionValidationError(f"{qa_id}/sid={sid}/{cls}: duplicate pmid {pmid_s}")
        seen_here.add(pmid_s)
        if pmid_s in used_in_topic:
            raise SubmissionValidationError(
                f"{qa_id}/sid={sid}/{cls}: pmid {pmid_s} already used in topic"
            )
        if index_pmids is not None and pmid_s not in index_pmids:
            raise SubmissionValidationError(
                f"{qa_id}/sid={sid}/{cls}: pmid {pmid_s} not in corpus"
            )
        used_in_topic.add(pmid_s)
