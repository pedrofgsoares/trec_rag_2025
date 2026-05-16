"""Convert BioGEN 2024 ``biogen_collection_v3.json`` into our qrels JSONL shape.

The 2024 collection nests judgements as
``[*].machine_generated_answers[*].answer_sentences[*].citation_assessment[*]``.
We aggregate to **question-level** because the ``answer_sentence_id`` in the
collection refers to *each 2024 team's generated answer sentences*, which do
not align with our 2025 input's fixed sentences. We use ``sentence_id=-1``
as a sentinel that ``eval.metrics`` interprets as "any sentence in topic".

* ``evidence_relation="supporting"``     -> class="support"
* ``evidence_relation="contradicting"``  -> class="contradict"
* anything else                          -> ignored (negatives are implicit)

Output: one line per unique (qa_id, pmid, class) tuple; if the same PMID
appears as both supporting and contradicting across runs/sentences, both
records are emitted (the per-class set logic in qrels.py handles this).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import orjson

REL_TO_CLASS = {
    "supporting": "support",
    "contradicting": "contradict",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path, help="biogen_collection_v3.json")
    p.add_argument("--output", required=True, type=Path, help="output qrels JSONL")
    a = p.parse_args(argv)

    collection = json.loads(a.input.read_text())

    # (qa_id, pmid, class) -> count of judgements (debug only)
    triples: dict[tuple[str, str, str], int] = defaultdict(int)

    for item in collection:
        qa_id = str(item["question_id"])
        runs = item.get("machine_generated_answers") or {}
        for run in runs.values():
            for s in run.get("answer_sentences") or []:
                for j in s.get("citation_assessment") or []:
                    rel = j.get("evidence_relation")
                    cls = REL_TO_CLASS.get(rel)
                    pmid = j.get("cited_pmid")
                    if not cls or not pmid:
                        continue
                    triples[(qa_id, str(pmid), cls)] += 1

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("wb") as fh:
        for (qa_id, pmid, cls), count in sorted(triples.items()):
            rec = {
                "qa_id": qa_id,
                "sentence_id": -1,        # sentinel: question-level
                "pmid": pmid,
                "class": cls,
                "relevance": 1,
                # carry the judgement count for traceability (multiple runs may
                # cite the same PMID for the same question — stronger signal)
                "n_judgements": count,
            }
            fh.write(orjson.dumps(rec))
            fh.write(b"\n")

    print(f"wrote {len(triples)} unique (qa_id, pmid, class) records to {a.output}")
    # Quick stats
    sup = sum(1 for (_, _, c) in triples if c == "support")
    con = sum(1 for (_, _, c) in triples if c == "contradict")
    qa_ids = sorted({q for (q, _, _) in triples}, key=int)
    print(f"  support: {sup}  contradict: {con}")
    print(f"  qa_ids covered: {qa_ids[0]}..{qa_ids[-1]} ({len(qa_ids)} unique)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
