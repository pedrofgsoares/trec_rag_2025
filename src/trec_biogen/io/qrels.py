"""Load BioGen Task A qrels.

Expected per-line JSONL schema (one judgment per line)::

    {
      "qa_id":       "<question id>",
      "sentence_id": <int, 0-based position of the answer sentence>,
      "pmid":        "<PubMed id, string>",
      "class":       "support" | "partial_support" | "contradict" | "partial_contradict",
      "relevance":   <int, 0 or 1>
    }

Returned indexes:

* ``positives[(qa_id, sentence_id, "support")]`` -> set of PMIDs with class
  ``support`` and relevance == 1.  (Dsup — Strict)
* ``positives_relaxed[(qa_id, sentence_id, "support")]`` -> Dsup ∪ Dpsup
  (Strict + partial). Same keys for the ``contradict`` class.

Task: 5.3
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import orjson

Class = Literal["support", "contradict"]
_STRICT_CLASSES = {"support", "contradict"}
_PARTIAL_TO_STRICT = {"partial_support": "support", "partial_contradict": "contradict"}


@dataclass(slots=True)
class QrelsIndex:
    """Per-(qa_id, sentence_id, class) positive PMID lookups under both settings."""

    strict: dict[tuple[str, int, str], set[str]] = field(default_factory=dict)
    relaxed: dict[tuple[str, int, str], set[str]] = field(default_factory=dict)

    def positives(
        self, qa_id: str, sentence_id: int, cls: Class, *, setting: str = "strict"
    ) -> set[str]:
        store = self.strict if setting == "strict" else self.relaxed
        return store.get((qa_id, sentence_id, cls), set())


def load_qrels(path: Path) -> QrelsIndex:
    strict: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    relaxed: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    with path.open("rb") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                r = orjson.loads(raw)
            except orjson.JSONDecodeError as e:
                raise ValueError(f"qrels line {lineno}: invalid JSON: {e}") from e

            qa_id = str(r["qa_id"])
            sid = int(r["sentence_id"])
            pmid = str(r["pmid"])
            cls = str(r["class"])
            rel = int(r.get("relevance", 1))
            if rel <= 0:
                continue

            base_cls = cls if cls in _STRICT_CLASSES else _PARTIAL_TO_STRICT.get(cls)
            if base_cls is None:
                continue  # unknown class — ignore

            relaxed[(qa_id, sid, base_cls)].add(pmid)
            if cls in _STRICT_CLASSES:
                strict[(qa_id, sid, base_cls)].add(pmid)

    return QrelsIndex(strict=dict(strict), relaxed=dict(relaxed))
