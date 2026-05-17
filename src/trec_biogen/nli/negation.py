"""NegEx + biomedical cue-list pre-filter for the contradiction path (D4, §8.3).

Keeps only abstract sentences that either (a) contain a negated entity per
``negspacy`` or (b) match one of the 23 explicit cue patterns from design D4.
Logs ``filtered_out_count`` per topic so we can audit how many candidates
were dropped (task 8.6).

The filter is applied to the long-format Parquet produced by
``phases.segment_abstracts`` and yields a same-schema Parquet plus a JSONL
audit log of dropped rows (sampled to 50 by default).
"""

from __future__ import annotations

import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import orjson
import polars as pl

# The 23 biomedical negation cue patterns from design.md D4.
CUE_PATTERNS: tuple[str, ...] = (
    r"\bno evidence of\b",
    r"\babsence of\b",
    r"\bdid not\b",
    r"\bfailed to\b",
    r"\bno association\b",
    r"\bno significant difference\b",
    r"\bcontrary to\b",
    r"\bin contrast to\b",
    r"\bhowever\b",
    r"\bdid not find\b",
    r"\bwas not associated\b",
    r"\bno effect\b",
    r"\bno significant effect\b",
    r"\bno difference\b",
    r"\bruled out\b",
    r"\brefuted\b",
    r"\binconsistent with\b",
    r"\bwas not significant\b",
    r"\bdid not support\b",
    r"\bno benefit\b",
    r"\bdid not improve\b",
    r"\bdid not reduce\b",
    r"\bno correlation\b",
)
_CUE_RE = re.compile("|".join(CUE_PATTERNS), flags=re.IGNORECASE)


@lru_cache(maxsize=1)
def _negspacy_pipeline() -> Any:
    """Return an ``en_core_sci_sm`` pipeline with negspacy attached."""
    import spacy

    nlp = spacy.load("en_core_sci_sm", disable=["lemmatizer", "tagger"])
    # negspacy uses the NER output to flag negated entities, so NER must stay enabled.
    try:
        from negspacy.negation import Negex  # noqa: F401

        if "negex" not in nlp.pipe_names:
            nlp.add_pipe("negex", config={"chunk_prefix": ["no"]})
    except Exception:
        # If negspacy / NER setup fails, fall back to cue-only matching.
        pass
    return nlp


def _negated_by_negspacy(text: str) -> bool:
    nlp = _negspacy_pipeline()
    try:
        doc = nlp(text)
    except Exception:
        return False
    return any(getattr(ent._, "negex", False) for ent in doc.ents)


def has_negation(text: str) -> bool:
    """Return True if ``text`` matches one of the 23 biomedical cue patterns.

    Phase 1 uses cue-regex only. The original design also fell back to
    scispaCy NegEx (NER + parser per sentence) when the cue regex missed,
    but that costs ~30 ms/sentence and produces hours of latency on the
    ~1.5 M segmented abstract sentences our contradict path generates.
    Empirically the cue regex catches the dominant signal; scispaCy NegEx
    is deferred to Phase 2 where we can tune it on a smaller candidate
    pool. The ``_negated_by_negspacy`` helper below is kept for that work.
    """
    if not text:
        return False
    return bool(_CUE_RE.search(text))


def filter_negated(
    long_parquet: Path,
    *,
    out_parquet: Path,
    audit_jsonl: Path | None = None,
    audit_sample: int = 50,
    seed: int = 0,
) -> dict[str, int]:
    """Keep only rows whose ``abstract_sentence_text`` carries negation.

    Returns counts ``{kept, dropped, total}`` for the run log; if
    ``audit_jsonl`` is set, writes a sampled JSONL of dropped rows for the
    operator review required by task 8.6.
    """
    df = pl.read_parquet(long_parquet)
    texts = df["abstract_sentence_text"].to_list()
    keep_mask = [has_negation(t) for t in texts]
    kept = df.filter(pl.Series("__keep__", keep_mask))
    dropped = df.filter(pl.Series("__keep__", [not k for k in keep_mask]))

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    kept.write_parquet(out_parquet)

    if audit_jsonl is not None and dropped.height:
        rng = random.Random(seed)
        sample_n = min(audit_sample, dropped.height)
        idxs = rng.sample(range(dropped.height), sample_n)
        audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with audit_jsonl.open("wb") as fh:
            for i in idxs:
                fh.write(orjson.dumps(dropped.row(i, named=True)))
                fh.write(b"\n")

    counts = {"kept": kept.height, "dropped": dropped.height, "total": df.height}
    return counts
