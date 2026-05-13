"""Submission writer + validator (9.3, 9.4).

Output JSONL, one record per topic, sentences in answer order, contradicting
PMIDs emitted before supporting PMIDs per the track rule (design D9)::

    {
      "qa_id": "...",
      "sentences": [
        {"sentence_id": 0,
         "contradict_pmids": ["...", "...", "..."],
         "support_pmids":    ["...", "...", "..."]},
        ...
      ]
    }

The validator (9.4) enforces:
* per-sentence caps ≤ 3 for each class,
* PMID strings only, no duplicates within a (sentence, class) list,
* no PMID re-used across sentences within a topic,
* every PMID a member of the index (when an ``index_pmids`` set is supplied),
* sentence order strictly increasing.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import orjson


CAP = 3


def write_submission(
    selection: dict[str, dict[int, dict[str, list[str]]]],
    out_path: Path,
) -> Path:
    """Write the JSONL submission. ``selection`` is the return of ``selection.select``."""
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


class SubmissionValidationError(ValueError):
    """Raised by ``validate`` on any rule violation. Carries qa_id + sentence_id when known."""


def validate(
    submission_path: Path,
    *,
    index_pmids: set[str] | None = None,
) -> None:
    """Raise ``SubmissionValidationError`` on the first rule violation."""
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
    sid: int,
    cls: str,
    pmids: Iterable[str],
    used_in_topic: set[str],
    index_pmids: set[str] | None,
) -> None:
    pmids = list(pmids)
    if len(pmids) > CAP:
        raise SubmissionValidationError(f"{qa_id}/sid={sid}/{cls}: cap exceeded ({len(pmids)}>{CAP})")
    seen_here: set[str] = set()
    for pmid in pmids:
        if not isinstance(pmid, str) or not pmid:
            raise SubmissionValidationError(f"{qa_id}/sid={sid}/{cls}: non-string pmid")
        if pmid in seen_here:
            raise SubmissionValidationError(f"{qa_id}/sid={sid}/{cls}: duplicate pmid {pmid}")
        seen_here.add(pmid)
        if pmid in used_in_topic:
            raise SubmissionValidationError(
                f"{qa_id}/sid={sid}/{cls}: pmid {pmid} already used in topic"
            )
        if index_pmids is not None and pmid not in index_pmids:
            raise SubmissionValidationError(
                f"{qa_id}/sid={sid}/{cls}: pmid {pmid} not in corpus"
            )
        used_in_topic.add(pmid)
