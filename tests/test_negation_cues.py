"""Cue-list regex tests for the NegEx pre-filter (8.3). Pure-regex; no spaCy needed."""

from __future__ import annotations

import pytest

from trec_biogen.nli.negation import CUE_PATTERNS, _CUE_RE


@pytest.mark.parametrize(
    "sentence",
    [
        "There was no evidence of harm.",
        "We did not find a difference.",
        "Drug X failed to reduce mortality.",
        "Contrary to prior reports, this study refuted the hypothesis.",
        "There was no significant difference between groups.",
        "No benefit observed in the trial cohort.",
        "Inconsistent with earlier findings, the effect did not improve outcomes.",
    ],
)
def test_cue_matches_negative_sentence(sentence: str) -> None:
    assert _CUE_RE.search(sentence) is not None


@pytest.mark.parametrize(
    "sentence",
    [
        "Aspirin reduces myocardial infarction.",
        "The drug significantly improved survival.",
        "Patients had a clear benefit from treatment.",
    ],
)
def test_cue_skips_positive_sentence(sentence: str) -> None:
    assert _CUE_RE.search(sentence) is None


def test_cue_patterns_count_matches_design() -> None:
    # Design D4 enumerates exactly 23 explicit biomedical cue patterns.
    assert len(CUE_PATTERNS) == 23
