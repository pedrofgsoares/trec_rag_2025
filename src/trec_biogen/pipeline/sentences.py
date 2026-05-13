"""Answer-sentence and abstract-sentence segmentation via scispaCy.

Shared by both retrieval paths and the contradiction NLI step (§8.2).
The scispaCy model is loaded lazily on first call and held as a module-level
singleton so we pay the load cost once per process.

Task: shared helper for §7.1, §8.1, §8.2
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _nlp() -> Any:
    import spacy  # local import: heavy

    nlp = spacy.load("en_core_sci_sm", disable=["ner", "tagger", "lemmatizer", "parser"])
    # Use the lightweight sentence segmenter rather than the full parser.
    if "senter" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
        nlp.add_pipe("senter")
    return nlp


def split_sentences(text: str) -> list[str]:
    """Return ``text`` split into trimmed sentence strings (empty list on '')."""
    text = (text or "").strip()
    if not text:
        return []
    doc = _nlp()(text)
    out = [s.text.strip() for s in doc.sents]
    return [s for s in out if s]
