"""Convert the BioGEN 2025 ``baseline_labels.json`` into our qrels JSONL shape.

The 2025 labels file (shared by Deepak Gupta) holds, per (qa_id, answer
sentence index, PMID), one of four assessor labels:

  ``Supports``       -> class="support", relevance=1
  ``Contradicts``    -> class="contradict", relevance=1
  ``Neutral``        -> ignored (negative implicit)
  ``Not relevant``   -> ignored (negative implicit)

Labels appear in two nested fields:
  * ``supported_citations_labels`` (labels for PMIDs the baseline put in
    ``supported_citations``)
  * ``contradicted_citations_labels`` (labels for the baseline's
    ``contradicted_citations``)

A PMID labeled ``Supports`` in either bucket is a support positive for that
cell. Same for contradict. The qrels we emit is a flat sentence-level
JSONL — directly consumable by ``eval/metrics.py --level sentence``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import orjson

REL_TO_CLASS = {
    "supports": "support",
    "contradicts": "contradict",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path, help="baseline_labels.json")
    p.add_argument("--output", required=True, type=Path, help="output qrels JSONL")
    a = p.parse_args(argv)

    data = json.loads(a.input.read_text())

    # (qa_id, sentence_id, pmid, class) -> seen flag
    positives: dict[tuple[str, int, str, str], bool] = {}

    for item in data:
        meta = item.get("meta_data") or item.get("metadata") or {}
        qa_id = str(meta["qa_id"])
        for sid, ans in enumerate(item.get("answer", [])):
            for bucket in ("supported_citations_labels", "contradicted_citations_labels"):
                for pmid, label in (ans.get(bucket) or {}).items():
                    cls = REL_TO_CLASS.get(str(label).lower())
                    if cls is None:
                        continue
                    positives[(qa_id, sid, str(pmid), cls)] = True

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("wb") as fh:
        for (qa_id, sid, pmid, cls), _ in sorted(positives.items()):
            fh.write(orjson.dumps({
                "qa_id": qa_id,
                "sentence_id": sid,
                "pmid": pmid,
                "class": cls,
                "relevance": 1,
            }))
            fh.write(b"\n")

    n_sup = sum(1 for k in positives if k[3] == "support")
    n_con = sum(1 for k in positives if k[3] == "contradict")
    qa_ids = sorted({k[0] for k in positives}, key=int)
    print(f"wrote {len(positives)} qrels records to {a.output}")
    print(f"  support: {n_sup}  contradict: {n_con}")
    print(f"  qa_ids: {qa_ids[0]}..{qa_ids[-1]} ({len(qa_ids)} unique)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
