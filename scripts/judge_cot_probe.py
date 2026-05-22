"""Probe: does a chain-of-thought prompt fix the substantive reasoning
errors found in the 4 disagreements identified by judge_disagreement_examples.py?

Keeps the 4-label space; adds a `reasoning` field to the JSON output that
forces the model to articulate the inferential chain (e.g. J-curve →
harm) before committing to a label.

Disposable.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from trec_biogen.judge.prompts import LABELS, truncate_abstract
from trec_biogen.judge.rejudge import (
    bm25_abstract_lookup,
    load_answer_sentence_lookup,
)

REPO = Path(__file__).resolve().parents[1]
TOPICS = REPO / "data/topics/biogen2025_taskA_input.json"
INDEX = REPO / "data/indexes/pubmed_bm25"

# The four disagreements from judge_disagreement_examples.py (seed=17).
CASES = [
    ("146", 0, "28287769"),  # VA recommends PE/CPT/EMDR
    ("143", 0, "32705582"),  # HD coping → dependence
    ("125", 1, "28894841"),  # ureteral stones
    ("144", 3, "19785385"),  # BP <120/70 may cause problems
]

COT_SYSTEM = (
    "You are a careful biomedical evidence assessor. Given an answer sentence "
    "and a PubMed abstract, decide whether the abstract supports, contradicts, "
    "or is neutral / not relevant to the sentence's claim.\n"
    "\n"
    "Crucially: support can be *implicit*. An abstract supports the sentence "
    "if its content (including domain mechanisms it cites, like J-curves, "
    "established side-effect profiles, or institutional behavior like "
    "nationwide training programs) is logically consistent with the "
    "sentence's claim, even if the abstract does not state the claim "
    "verbatim. You may chain 1-3 inferential steps using widely-known "
    "biomedical knowledge.\n"
    "\n"
    "Labels:\n"
    "- Supports: abstract's evidence is consistent with the claim, directly "
    "or via short inference.\n"
    "- Contradicts: abstract's evidence is inconsistent with the claim.\n"
    "- Neutral: abstract is about the same topic but provides no evidence "
    "either way after a fair attempt at inference.\n"
    "- Not relevant: abstract is about a different topic.\n"
    "\n"
    "Output a strict JSON object with three fields and NOTHING else:\n"
    '{"reasoning": "<2-3 sentence inferential chain>", '
    '"label": "<one of: Supports | Contradicts | Neutral | Not relevant>", '
    '"confidence": <float 0..1>}'
)


def cot_call(model: str, sentence: str, abstract: str) -> dict[str, Any]:
    abstract_clipped = truncate_abstract(abstract)
    body = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": COT_SYSTEM},
                {"role": "user",
                 "content": "Answer sentence:\n" + sentence.strip()
                            + "\n\nAbstract:\n" + abstract_clipped},
            ],
            "temperature": 0.0,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
        },
        timeout=60.0,
    ).json()
    return json.loads(body["choices"][0]["message"]["content"])


def main() -> int:
    if "OPENAI_API_KEY" not in os.environ:
        print("set OPENAI_API_KEY first", file=sys.stderr)
        return 1
    answer_lookup = load_answer_sentence_lookup(TOPICS)
    abstract_lookup = bm25_abstract_lookup(INDEX)

    for i, (qa_id, sid, pmid) in enumerate(CASES, 1):
        sentence = answer_lookup(qa_id, sid)
        abstract = abstract_lookup(pmid)
        print(f"\n=== Case {i}: qa={qa_id} sent={sid} pmid={pmid} (human: Supports) ===")
        print(f"Sentence: {sentence}")
        for model in ("gpt-4o-mini", "gpt-4o"):
            try:
                r = cot_call(model, sentence, abstract)
            except Exception as e:  # noqa: BLE001
                print(f"  {model}: ERROR {e}")
                continue
            print(f"  {model}: label={r.get('label')!r}  conf={r.get('confidence')}")
            print(f"    reasoning: {r.get('reasoning')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
