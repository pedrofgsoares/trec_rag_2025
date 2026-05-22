"""LLM-filtered pseudo-relevance feedback (Phase 2 §12.7).

Addresses the §7.4 negative finding that *blind* RM3 hurts on
biomedical evidence retrieval because the top-k BM25 hits used as
pseudo-relevant are typically *topically related but not
evidence-bearing*. Mechanism here:

1. Run BM25 to get top-30 candidates for the query.
2. Ask an LLM (binary: ``relevant?`` yes/no per candidate) to filter
   the top-30. The filter prompt is deliberately *minimal* — no CoT,
   no per-candidate reasoning — to keep per-call latency and cost
   low. We are not asking for evidence judgement; we are asking
   *"is this on-topic for the claim?"*.
3. Compute RM3 expansion terms over the LLM-accepted subset only.
   If zero docs accepted, fall back to no expansion (plain BM25).
4. Re-run BM25 with the expanded query.

The implementation uses Pyserini's ``LuceneIndexReader`` for the
per-document term vectors; the RM3-style scoring is a small custom
function rather than Pyserini's built-in ``set_rm3`` because the
latter does not expose the option of restricting the
pseudo-relevant set to a custom list.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from trec_biogen.judge.backends import Backend, HTTPBackend, _empty_abstract_record
from trec_biogen.judge.prompts import truncate_abstract
from trec_biogen.retrieval.bm25 import BM25Index, Hit


# Binary, no-CoT system prompt. Optimised for cost: ~60 token system
# prompt, ~15 token output. Per call ≈ $0.00006 at gpt-4o-mini pricing
# given a 250-token abstract truncation.
RELEVANCE_FILTER_PROMPT: str = (
    "You are a relevance assessor. Given an answer sentence and a PubMed "
    "abstract, decide whether the abstract is *relevant* to the sentence's "
    "claim — i.e., whether reading the abstract could help determine if "
    "the sentence is true or false.\n"
    "A relevant abstract discusses the same topic, intervention, disease, "
    "or mechanism. It does NOT need to directly state the claim — being "
    "on-topic is enough.\n"
    "Respond with a strict JSON object and nothing else: "
    '{"relevant": true} or {"relevant": false}.'
)


@dataclass(slots=True, frozen=True)
class FilterDecision:
    """One LLM filter outcome for a (sentence, abstract) pair."""

    pmid: str
    relevant: bool
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _filter_payload(answer_sentence: str, abstract_text: str) -> list[dict[str, str]]:
    clipped = truncate_abstract(abstract_text, max_tokens=250)
    user = (
        "Answer sentence:\n" + answer_sentence.strip()
        + "\n\nAbstract:\n" + clipped
    )
    return [
        {"role": "system", "content": RELEVANCE_FILTER_PROMPT},
        {"role": "user", "content": user},
    ]


class LLMRelevanceFilter:
    """Per-candidate binary relevance filter backed by an LLM HTTP backend.

    Encapsulates the LLM call so the BM25/RM3 logic above doesn't have to
    know about HTTP, JSON, or retries. Honours the backend's existing
    quota-exhaustion / retry semantics (see
    :class:`trec_biogen.judge.backends.HTTPBackend`).
    """

    def __init__(self, backend: HTTPBackend, *, max_concurrent: int = 8) -> None:
        self._backend = backend
        self._max_concurrent = max(1, max_concurrent)
        self._total_input = 0
        self._total_output = 0
        self._total_cost = 0.0
        self._calls = 0

    @property
    def stats(self) -> dict[str, float | int]:
        return {
            "calls": self._calls,
            "input_tokens": self._total_input,
            "output_tokens": self._total_output,
            "cost_usd": round(self._total_cost, 6),
        }

    def _classify_one(
        self, answer_sentence: str, pmid: str, abstract_text: str,
    ) -> FilterDecision:
        if not abstract_text.strip():
            # Empty abstract: treat as not relevant. Cheap deterministic skip.
            return FilterDecision(
                pmid=pmid, relevant=False,
                input_tokens=0, output_tokens=0, cost_usd=0.0,
            )
        messages = _filter_payload(answer_sentence, abstract_text)
        payload: dict[str, Any] = {
            "model": self._backend._model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 20,
            "response_format": {"type": "json_object"},
        }
        resp = self._backend._post_with_retry(payload)
        body = resp.json()
        content = body["choices"][0]["message"]["content"] or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {}
        relevant = bool(parsed.get("relevant", False))
        usage = body.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens", 0))
        out_tok = int(usage.get("completion_tokens", 0))
        # Cost computed via the same _PRICES table as Judge backends.
        from trec_biogen.judge.backends import _compute_cost
        cost = _compute_cost(self._backend.name, in_tok, out_tok)
        return FilterDecision(
            pmid=pmid, relevant=relevant,
            input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost,
        )

    def filter_many(
        self,
        answer_sentence: str,
        candidates: list[tuple[str, str]],  # list of (pmid, abstract_text)
    ) -> list[FilterDecision]:
        """Classify a batch of candidates in parallel; return decisions in input order."""
        if not candidates:
            return []
        out: list[FilterDecision | None] = [None] * len(candidates)

        def _job(i: int) -> tuple[int, FilterDecision]:
            pmid, abstract = candidates[i]
            return i, self._classify_one(answer_sentence, pmid, abstract)

        with ThreadPoolExecutor(max_workers=self._max_concurrent) as ex:
            for i, decision in ex.map(_job, range(len(candidates))):
                out[i] = decision

        decisions = [d for d in out if d is not None]
        for d in decisions:
            self._calls += 1
            self._total_input += d.input_tokens
            self._total_output += d.output_tokens
            self._total_cost += d.cost_usd
        return decisions


# ---------------------------------------------------------------------------
# Manual RM3 over a custom pseudo-relevant set
# ---------------------------------------------------------------------------


def manual_rm3_terms(
    reader,
    pseudo_relevant_pmids: Iterable[str],
    *,
    fb_terms: int = 10,
    exclude: Iterable[str] = (),
) -> list[tuple[str, float]]:
    """Compute top-N RM1-style expansion terms over a custom set of docs.

    Standard RM1: for each pseudo-relevant doc d, term-probability is
    ``P(t | d) = tf(t, d) / |d|``. Aggregate over the set::

        weight(t) = (1 / |R|) * Σ_{d ∈ R}  P(t | d)

    Then take the top-``fb_terms`` by weight, optionally excluding any
    terms in ``exclude`` (typically the original query terms, so we
    don't trivially "expand" with the words already in the query).
    Sortable by weight descending; weights are not used downstream
    (we only need the term identities for the BM25 re-retrieval).
    """
    exclude_lower = {t.lower() for t in exclude}
    accumulator: dict[str, float] = {}
    n_docs = 0
    for pmid in pseudo_relevant_pmids:
        try:
            tv = reader.get_document_vector(pmid)
        except Exception:
            continue
        if not tv:
            continue
        n_docs += 1
        doc_len = sum(tv.values())
        if doc_len <= 0:
            continue
        for term, tf in tv.items():
            if term.lower() in exclude_lower:
                continue
            accumulator[term] = accumulator.get(term, 0.0) + tf / doc_len
    if not accumulator or n_docs == 0:
        return []
    for term in accumulator:
        accumulator[term] /= n_docs
    return sorted(accumulator.items(), key=lambda kv: -kv[1])[:fb_terms]


def build_expanded_query(
    original: str,
    expansion_terms: list[tuple[str, float]],
) -> str:
    """Concatenate the original query with the top expansion terms.

    A simpler alternative to Pyserini's term-weighted RM3 (which we
    cannot drive with a custom pseudo-relevant set): just append the
    expansion terms as additional bag-of-words query tokens. BM25 will
    score the expanded query naturally; the result is monotone in
    pseudo-relevance overlap by construction.
    """
    if not expansion_terms:
        return original
    expansion_str = " ".join(t for t, _ in expansion_terms)
    return f"{original} {expansion_str}".strip()
