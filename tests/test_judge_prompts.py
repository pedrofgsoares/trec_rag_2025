"""Prompt builder unit tests (task 2.14)."""

from __future__ import annotations

from trec_biogen.judge.prompts import (
    COT_SYSTEM_PROMPT,
    LABELS,
    MAX_ABSTRACT_TOKENS,
    SYSTEM_PROMPT,
    build_prompt,
    system_prompt_for,
    truncate_abstract,
)


def test_labels_are_the_four_canonical_classes() -> None:
    assert LABELS == ("Supports", "Contradicts", "Neutral", "Not relevant")


def test_truncate_abstract_under_cap_keeps_text() -> None:
    short = "Vitamin D reduces parathyroid hormone in dialysis patients."
    assert truncate_abstract(short) == short


def test_truncate_abstract_caps_at_max_tokens() -> None:
    long_text = " ".join(["token"] * (MAX_ABSTRACT_TOKENS + 50))
    out = truncate_abstract(long_text)
    assert len(out.split()) == MAX_ABSTRACT_TOKENS


def test_truncate_abstract_handles_empty() -> None:
    assert truncate_abstract("") == ""


def test_build_prompt_returns_system_then_user() -> None:
    msgs = build_prompt("the sky is blue", "Wavelength scattering explains blue sky.")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[1]["role"] == "user"


def test_build_prompt_user_message_contains_both_inputs() -> None:
    msgs = build_prompt("metformin lowers HbA1c", "Trial: metformin reduced HbA1c by 1.2%.")
    user = msgs[1]["content"]
    assert "metformin lowers HbA1c" in user
    assert "metformin reduced HbA1c by 1.2%" in user


def test_build_prompt_clips_long_abstract() -> None:
    long_text = " ".join(["word"] * (MAX_ABSTRACT_TOKENS + 50))
    msgs = build_prompt("a sentence", long_text)
    # The abstract section trails the user prompt; its body word count
    # should be bounded by MAX_ABSTRACT_TOKENS regardless of input length.
    abstract_section = msgs[1]["content"].split("Abstract:\n", 1)[-1]
    assert len(abstract_section.split()) <= MAX_ABSTRACT_TOKENS


def test_system_prompt_for_switches_on_mode() -> None:
    assert system_prompt_for("strict") == SYSTEM_PROMPT
    assert system_prompt_for("cot") == COT_SYSTEM_PROMPT
    assert "reasoning" in COT_SYSTEM_PROMPT  # CoT prompt asks for a reasoning chain
    assert "reasoning" not in SYSTEM_PROMPT


def test_build_prompt_cot_uses_cot_system() -> None:
    msgs = build_prompt("sentence", "abstract", mode="cot")
    assert msgs[0]["content"] == COT_SYSTEM_PROMPT


def test_build_prompt_default_mode_is_strict() -> None:
    assert build_prompt("s", "a")[0]["content"] == SYSTEM_PROMPT
