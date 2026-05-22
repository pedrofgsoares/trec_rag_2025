"""LLM-driven query rewriting for biomedical claim retrieval (Phase 2 §12.10).

Attacks the lexical-mismatch problem at *source*: instead of expanding the
BM25 query with pseudo-relevant feedback (RM3, §7.4 — *negative result*) or
filtering the pseudo-relevant set with an LLM (§12.7 — *partial recovery*),
ask the LLM directly to *paraphrase* the claim into 3 short PubMed-style
queries. Run BM25 over each variant, fuse via RRF.

The mechanism is fundamentally different from §7.4 / §12.7:

* RM3 builds the expanded query from words inside the *retrieved* documents
  (which are typically topical but not claim-bearing).
* §12.10 builds query variants from the LLM's *generative* understanding of
  the claim's medical entities, using biomedical paraphrases / synonyms
  / MeSH-style terms.

Cost: 1 LLM call per (qa_id, sentence_id) cell — 194 calls for the 2025
input. At ~170 tokens/call on `gpt-4o-mini` ≈ $0.00006/call ≈ $0.012 per
full run.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from trec_biogen.judge.backends import HTTPBackend


REWRITE_SYSTEM_PROMPT = (
    "You are a biomedical literature search assistant. The user has an answer "
    "sentence (a claim) about a medical topic, and wants to find PubMed "
    "abstracts that address it.\n"
    "Given the question and the answer sentence, write SHORT search queries "
    "that, when run against a PubMed BM25 index, will surface abstracts "
    "likely to support or contradict the claim. Each query should be 4 to 12 "
    "words, focused on the key medical entities, interventions, and outcomes "
    "in the claim. Use VARIED biomedical terminology across the queries — "
    "generic + brand drug names, MeSH-style terms, abbreviations, synonyms — "
    "so the BM25 retrieval has multiple paths to the same evidence.\n"
    "Respond with a strict JSON object and nothing else: "
    '{"queries": ["query 1", "query 2", "query 3"]}.'
)

REWRITE_COT_SYSTEM_PROMPT = (
    "You are a biomedical literature search assistant. The user has an answer "
    "sentence (a claim) about a medical topic, and wants to find PubMed "
    "abstracts that address it. Work in two steps before answering.\n"
    "\n"
    "STEP 1 — reasoning (mandatory, do this first):\n"
    "  * Identify the key medical entities in the claim: drugs (generic AND "
    "    common brand names), diseases / phenotypes (and their MeSH terms or "
    "    common abbreviations like HTN, T2DM, AKI), interventions, anatomy, "
    "    and outcomes.\n"
    "  * For each entity, think about ALTERNATIVE terminology a PubMed paper "
    "    might use: synonyms, mechanism-based descriptors, common acronyms, "
    "    related-but-distinct concepts.\n"
    "  * Decide on 2-3 ANGLES OF ATTACK that the queries should cover (e.g. "
    "    intervention + outcome; mechanism + outcome; disease + diagnostic).\n"
    "\n"
    "STEP 2 — emit queries:\n"
    "  * Write SHORT BM25 queries (4 to 12 words each) covering different "
    "    terminology angles. Each must stand alone as a PubMed-natural search.\n"
    "  * The queries together should give the BM25 retrieval multiple "
    "    independent paths to the same evidence.\n"
    "\n"
    "Respond with a strict JSON object and nothing else: "
    '{"reasoning": "<your step-1 analysis, 2-4 sentences>", '
    '"queries": ["query 1", "query 2", "query N"]}.'
)


def _system_prompt_for(mode: str) -> str:
    if mode == "cot":
        return REWRITE_COT_SYSTEM_PROMPT
    return REWRITE_SYSTEM_PROMPT


@dataclass(slots=True)
class RewriteRecord:
    """Outcome of one LLM-rewrite call."""

    queries: list[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _rewrite_payload_messages(
    question: str, sentence: str, n_variants: int, *, mode: str = "strict",
) -> list[dict[str, str]]:
    user = (
        f"Question: {question.strip()}\n"
        f"Answer sentence: {sentence.strip()}\n\n"
        f"Write {n_variants} varied search queries that target the claim."
    )
    return [
        {"role": "system", "content": _system_prompt_for(mode)},
        {"role": "user", "content": user},
    ]


def parse_rewrite_response(content: str, *, n_expected: int) -> list[str]:
    """Extract the ``queries`` list from a model response. Tolerant of minor
    JSON deviations. Returns an empty list on unrecoverable parse failures
    so the caller can fall back to the original query."""
    if not content:
        return []
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if 0 <= start < end:
            try:
                obj = json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []
    raw_queries = obj.get("queries") if isinstance(obj, dict) else None
    if not isinstance(raw_queries, list):
        return []
    out: list[str] = []
    for q in raw_queries:
        if isinstance(q, str) and q.strip():
            out.append(q.strip())
    return out[:n_expected]


class LLMQueryRewriter:
    """Per-cell biomedical query rewriter backed by an LLM HTTP backend."""

    def __init__(
        self,
        backend: HTTPBackend,
        *,
        n_variants: int = 3,
        max_concurrent: int = 4,
        prompt_mode: str = "cot",
    ) -> None:
        self._backend = backend
        self._n_variants = max(1, n_variants)
        self._max_concurrent = max(1, max_concurrent)
        if prompt_mode not in ("strict", "cot"):
            raise ValueError(f"prompt_mode must be 'strict' or 'cot', got {prompt_mode!r}")
        self._prompt_mode = prompt_mode
        self._calls = 0
        self._total_input = 0
        self._total_output = 0
        self._total_cost = 0.0

    @property
    def stats(self) -> dict[str, float | int | str]:
        return {
            "calls": self._calls,
            "input_tokens": self._total_input,
            "output_tokens": self._total_output,
            "cost_usd": round(self._total_cost, 6),
            "n_variants": self._n_variants,
            "prompt_mode": self._prompt_mode,
        }

    def rewrite_one(self, question: str, sentence: str) -> RewriteRecord:
        """Rewrite a single (question, sentence) into ``n_variants`` queries."""
        messages = _rewrite_payload_messages(
            question, sentence, self._n_variants, mode=self._prompt_mode,
        )
        # CoT mode emits a reasoning chain before the queries; bump max_tokens
        # so the JSON output doesn't get truncated mid-reasoning.
        max_tokens = 400 if self._prompt_mode == "cot" else 200
        payload: dict[str, Any] = {
            "model": self._backend._model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        resp = self._backend._post_with_retry(payload)
        body = resp.json()
        content = body["choices"][0]["message"]["content"] or ""
        queries = parse_rewrite_response(content, n_expected=self._n_variants)
        usage = body.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens", 0))
        out_tok = int(usage.get("completion_tokens", 0))
        from trec_biogen.judge.backends import _compute_cost
        cost = _compute_cost(self._backend.name, in_tok, out_tok)
        rec = RewriteRecord(
            queries=queries,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )
        self._calls += 1
        self._total_input += in_tok
        self._total_output += out_tok
        self._total_cost += cost
        return rec

    def rewrite_many(
        self,
        items: list[tuple[str, str]],
    ) -> list[RewriteRecord]:
        """Rewrite a batch of (question, sentence) pairs in parallel."""
        if not items:
            return []
        out: list[RewriteRecord | None] = [None] * len(items)

        def _job(i: int) -> tuple[int, RewriteRecord]:
            q, s = items[i]
            return i, self.rewrite_one(q, s)

        with ThreadPoolExecutor(max_workers=self._max_concurrent) as ex:
            for i, rec in ex.map(_job, range(len(items))):
                out[i] = rec
        return [r for r in out if r is not None]
