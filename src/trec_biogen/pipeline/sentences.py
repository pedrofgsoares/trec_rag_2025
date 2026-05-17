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

    # Disable everything heavy. en_core_sci_sm doesn't ship a trained
    # `senter` component, so adding one without initialize() raises E109.
    # `sentencizer` is rule-based (no weights), fast, and adequate for
    # feeding biomedical sentences to NLI.
    nlp = spacy.load("en_core_sci_sm", disable=["ner", "tagger", "lemmatizer", "parser"])
    if not any(p in nlp.pipe_names for p in ("sentencizer", "senter", "parser")):
        nlp.add_pipe("sentencizer")
    return nlp


def split_sentences(text: str) -> list[str]:
    """Return ``text`` split into trimmed sentence strings (empty list on '')."""
    text = (text or "").strip()
    if not text:
        return []
    doc = _nlp()(text)
    out = [s.text.strip() for s in doc.sents]
    return [s for s in out if s]
