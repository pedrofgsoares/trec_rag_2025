"""Ad-hoc: print concrete (prompt, response) pairs where the LLM judge
disagreed with the human qrels.

Samples a handful of human-"Supports" triples, runs both OpenAI backends,
and dumps the full system+user prompt and raw model response for every
disagreement, so we can eyeball whether the disagreement is a label-space
artefact or a substantive call.

Not part of the test suite — disposable.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import orjson

from trec_biogen.judge.backends import OpenAI4o, OpenAIMini
from trec_biogen.judge.prompts import build_prompt
from trec_biogen.judge.rejudge import (
    bm25_abstract_lookup,
    load_answer_sentence_lookup,
)
from trec_biogen.judge.validator import load_validation_triples

REPO = Path(__file__).resolve().parents[1]
QRELS = REPO / "data/qrels/biogen2025_taskA_qrels.jsonl"
TOPICS = REPO / "data/topics/biogen2025_taskA_input.json"
INDEX = REPO / "data/indexes/pubmed_bm25"
SAMPLE_SIZE = 12
SEED = 17


def main() -> int:
    if "OPENAI_API_KEY" not in os.environ:
        print("set OPENAI_API_KEY first", file=sys.stderr)
        return 1
    triples = [t for t in load_validation_triples(QRELS) if t.human_label == "Supports"]
    random.seed(SEED)
    sample = random.sample(triples, SAMPLE_SIZE)

    answer_lookup = load_answer_sentence_lookup(TOPICS)
    abstract_lookup = bm25_abstract_lookup(INDEX)
    mini = OpenAIMini()
    big = OpenAI4o()

    disagreements: list[dict] = []
    for t in sample:
        sentence = answer_lookup(t.qa_id, t.sentence_id)
        abstract = abstract_lookup(t.pmid)
        if not abstract.strip():
            continue
        r_mini = mini.classify(sentence, abstract)
        r_big = big.classify(sentence, abstract)
        if r_mini.label == "Supports" and r_big.label == "Supports":
            continue
        disagreements.append({
            "qa_id": t.qa_id,
            "sentence_id": t.sentence_id,
            "pmid": t.pmid,
            "human": t.human_label,
            "mini": (r_mini.label, r_mini.confidence),
            "big":  (r_big.label,  r_big.confidence),
            "sentence": sentence,
            "abstract": abstract,
        })

    print(f"# {len(disagreements)} / {SAMPLE_SIZE} disagreements\n")
    for i, d in enumerate(disagreements, 1):
        print(f"\n## Example {i}: qa={d['qa_id']} sent={d['sentence_id']} pmid={d['pmid']}")
        print(f"- human:   Supports")
        print(f"- 4o-mini: {d['mini'][0]} (conf={d['mini'][1]:.2f})")
        print(f"- 4o:      {d['big'][0]} (conf={d['big'][1]:.2f})")
        print(f"\n### Prompt sent (literal):\n")
        msgs = build_prompt(d["sentence"], d["abstract"])
        for m in msgs:
            print(f"--- role: {m['role']} ---")
            print(m["content"])
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
