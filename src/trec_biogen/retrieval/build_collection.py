"""Convert parsed PubMed JSONL into Pyserini ``JsonCollection`` shape.

Pyserini's ``JsonCollection`` requires one JSON object per line with ``id``
and ``contents`` fields. We concatenate title + abstract as ``contents`` —
the format expected by the starter-kit baseline.

Task: 4.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import orjson


def convert(input_jsonl: Path, output_jsonl: Path) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with input_jsonl.open("rb") as fin, output_jsonl.open("wb") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = orjson.loads(line)
            title = rec.get("title", "") or ""
            abstract = rec.get("abstract", "") or ""
            contents = (title + " " + abstract).strip()
            fout.write(orjson.dumps({"id": rec["pmid"], "contents": contents}))
            fout.write(b"\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path, help="parsed PubMed JSONL")
    p.add_argument("--output", required=True, type=Path, help="Pyserini-shaped JSONL")
    a = p.parse_args(argv)
    n = convert(a.input, a.output)
    print(f"wrote {n} docs to {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
