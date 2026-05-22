"""MedNLI-style prompt for the LLM-as-judge re-judgement pipeline.

Returns a chat-style messages payload (system + user) plus the structured
``LABELS`` the judge can emit. Designed for any OpenAI-compatible chat
completions backend (OpenAI, Together.ai, etc.).

The abstract is capped at ``MAX_ABSTRACT_TOKENS`` whitespace tokens to keep
judge cost predictable and to fit comfortably inside any backend's prompt
budget. The cap is approximate — PubMed abstracts already cluster well
below 300 whitespace tokens, and the difference between whitespace and
sub-word tokenisation is small enough not to matter for cost-bounding.

Task: 2.2
"""

from __future__ import annotations

from typing import Final, Literal

LABELS: Final[tuple[str, str, str, str]] = (
    "Supports", "Contradicts", "Neutral", "Not relevant",
)

MAX_ABSTRACT_TOKENS: Final[int] = 300

PromptMode = Literal["strict", "cot"]

SYSTEM_PROMPT: Final[str] = (
    "You are a careful biomedical evidence assessor. Given an answer sentence and a "
    "PubMed abstract, classify the abstract's stance toward the sentence into exactly "
    "one of: Supports, Contradicts, Neutral, Not relevant.\n"
    "- Supports: the abstract provides evidence consistent with the sentence's claim.\n"
    "- Contradicts: the abstract provides evidence inconsistent with the sentence's claim.\n"
    "- Neutral: the abstract is about the same topic but neither supports nor contradicts.\n"
    "- Not relevant: the abstract is about a different topic.\n"
    "Respond with a strict JSON object: "
    '{"label": "<one of the four labels>", "confidence": <float between 0 and 1>}. '
    "Do not include any other text."
)

# Chain-of-thought variant. The model emits a short ``reasoning`` field
# before committing to a label, which gives it room to chain biomedical
# inference (J-curve → harm, institutional training → recommendation, etc.).
# Empirically this resolved every disagreement in the
# ``scripts/judge_cot_probe.py`` 4-case probe — the strict-mode failures
# were inferential-chain failures, not label-space mismatches.
COT_SYSTEM_PROMPT: Final[str] = (
    "You are a careful biomedical evidence assessor. Given an answer sentence "
    "and a PubMed abstract, decide whether the abstract supports, contradicts, "
    "or is neutral / not relevant to the sentence's claim.\n"
    "\n"
    "Support can be implicit. An abstract supports the sentence if its content "
    "(including domain mechanisms it cites such as J-curves, established "
    "side-effect profiles, or institutional behaviour like nationwide training "
    "programs) is logically consistent with the sentence's claim, even when "
    "the abstract does not state the claim verbatim. You may chain 1-3 "
    "inferential steps using widely-known biomedical knowledge.\n"
    "\n"
    "Labels:\n"
    "- Supports: abstract's evidence is consistent with the claim, directly or "
    "via short inference.\n"
    "- Contradicts: abstract's evidence is inconsistent with the claim.\n"
    "- Neutral: abstract is about the same topic but provides no evidence either "
    "way after a fair attempt at inference.\n"
    "- Not relevant: abstract is about a different topic.\n"
    "\n"
    "Output a strict JSON object with three fields and NOTHING else:\n"
    '{"reasoning": "<2-3 sentence inferential chain>", '
    '"label": "<one of: Supports | Contradicts | Neutral | Not relevant>", '
    '"confidence": <float 0..1>}'
)


def system_prompt_for(mode: PromptMode) -> str:
    if mode == "cot":
        return COT_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def truncate_abstract(abstract_text: str, *, max_tokens: int = MAX_ABSTRACT_TOKENS) -> str:
    if not abstract_text:
        return ""
    tokens = abstract_text.split()
    if len(tokens) <= max_tokens:
        return abstract_text.strip()
    return " ".join(tokens[:max_tokens])


def build_prompt(
    answer_sentence: str,
    abstract_text: str,
    *,
    mode: PromptMode = "strict",
) -> list[dict[str, str]]:
    """Return the chat-completions messages list for one judge call.

    ``mode="cot"`` swaps in the chain-of-thought system prompt, which asks
    the model to emit a ``reasoning`` field alongside ``label`` and
    ``confidence``. The JSON parser in :mod:`trec_biogen.judge.backends`
    ignores extra fields, so the two modes share the same output handler.
    """
    clipped = truncate_abstract(abstract_text)
    user = (
        "Answer sentence:\n"
        + answer_sentence.strip()
        + "\n\nAbstract:\n"
        + clipped
    )
    return [
        {"role": "system", "content": system_prompt_for(mode)},
        {"role": "user", "content": user},
    ]
