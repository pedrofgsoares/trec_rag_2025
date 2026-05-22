"""Load BioGen Task A qrels.

Expected per-line JSONL schema (one judgment per line)::

    {
      "qa_id":       "<question id>",
      "sentence_id": <int, 0-based position of the answer sentence>,
      "pmid":        "<PubMed id, string>",
      "class":       "support" | "partial_support" | "contradict" | "partial_contradict",
      "relevance":   <int, 0 or 1>,
      "source":      "<human | openai-gpt-4o-mini | ...>",       # optional
      "confidence":  <float 0..1>                                  # optional
    }

The ``source`` and ``confidence`` fields are introduced by the Phase 2 §2
LLM-judge rejudge pipeline. Records that omit ``source`` are treated as
``"human"`` so the official-pool path is unchanged. ``confidence`` is parsed
and stored on the index for downstream use but is not consulted by the
default ``positives()`` filter.

Returned indexes:

* ``positives[(qa_id, sentence_id, "support")]`` -> set of PMIDs with class
  ``support`` and relevance == 1.  (Dsup — Strict)
* ``positives_relaxed[(qa_id, sentence_id, "support")]`` -> Dsup ∪ Dpsup
  (Strict + partial). Same keys for the ``contradict`` class.

Task: 5.3, Phase 2 §3.3
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import orjson

Class = Literal["support", "contradict"]
Source = Literal["human", "llm", "any"]
_STRICT_CLASSES = {"support", "contradict"}
_PARTIAL_TO_STRICT = {"partial_support": "support", "partial_contradict": "contradict"}

Key = tuple[str, int, str]


@dataclass(slots=True)
class QrelsIndex:
    """Per-(qa_id, sentence_id, class) positive PMID lookups under both settings.

    ``strict_sources`` / ``relaxed_sources`` carry the per-PMID source
    attribution introduced in Phase 2 §3.3 (LLM-judge expanded qrels).
    They are populated regardless of whether the qrels file includes
    ``source`` fields — records without one default to ``"human"`` —
    so ``positives(..., source="human")`` works against both
    legacy-shape and expanded-shape qrels uniformly.
    """

    strict: dict[Key, set[str]] = field(default_factory=dict)
    relaxed: dict[Key, set[str]] = field(default_factory=dict)
    strict_sources: dict[Key, dict[str, str]] = field(default_factory=dict)
    relaxed_sources: dict[Key, dict[str, str]] = field(default_factory=dict)

    def positives(
        self,
        qa_id: str,
        sentence_id: int,
        cls: Class,
        *,
        setting: str = "strict",
        source: Source = "any",
    ) -> set[str]:
        store = self.strict if setting == "strict" else self.relaxed
        all_pmids = store.get((qa_id, sentence_id, cls), set())
        if source == "any":
            return all_pmids
        src_store = self.strict_sources if setting == "strict" else self.relaxed_sources
        sources = src_store.get((qa_id, sentence_id, cls), {})
        if source == "human":
            return {p for p in all_pmids if sources.get(p, "human") == "human"}
        # source == "llm"
        return {p for p in all_pmids if sources.get(p, "human") != "human"}

    def question_positives(
        self,
        qa_id: str,
        cls: Class,
        *,
        setting: str = "strict",
        source: Source = "any",
    ) -> set[str]:
        """Union of positives across every sentence_id for ``(qa_id, cls)``."""
        store = self.strict if setting == "strict" else self.relaxed
        src_store = self.strict_sources if setting == "strict" else self.relaxed_sources
        out: set[str] = set()
        for (q, _sid, c), pmids in store.items():
            if q != qa_id or c != cls:
                continue
            if source == "any":
                out |= pmids
                continue
            sources = src_store.get((q, _sid, c), {})
            if source == "human":
                out |= {p for p in pmids if sources.get(p, "human") == "human"}
            else:  # llm
                out |= {p for p in pmids if sources.get(p, "human") != "human"}
        return out


def load_qrels(path: Path) -> QrelsIndex:
    strict: dict[Key, set[str]] = defaultdict(set)
    relaxed: dict[Key, set[str]] = defaultdict(set)
    strict_sources: dict[Key, dict[str, str]] = defaultdict(dict)
    relaxed_sources: dict[Key, dict[str, str]] = defaultdict(dict)
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
            source = str(r.get("source", "human")) or "human"

            base_cls = cls if cls in _STRICT_CLASSES else _PARTIAL_TO_STRICT.get(cls)
            if base_cls is None:
                continue  # unknown class — ignore

            relaxed_key: Key = (qa_id, sid, base_cls)
            relaxed[relaxed_key].add(pmid)
            relaxed_sources[relaxed_key][pmid] = source
            if cls in _STRICT_CLASSES:
                strict[relaxed_key].add(pmid)
                strict_sources[relaxed_key][pmid] = source

    return QrelsIndex(
        strict=dict(strict),
        relaxed=dict(relaxed),
        strict_sources=dict(strict_sources),
        relaxed_sources=dict(relaxed_sources),
    )
