"""Per-class P/R/F1 under Strict and Relaxed settings.

A submission is a JSONL with one record per topic::

    {
      "qa_id": "Q001",
      "sentences": [
        {
          "sentence_id": 0,
          "contradict_pmids": ["..."],   # ≤3, contradictions first
          "support_pmids":    ["..."],   # ≤3
        },
        ...
      ]
    }

For each (qa_id, sentence_id, class) we compute set-based precision, recall,
F1 against the qrels positives (Dsup for Strict; Dsup ∪ Dpsup for Relaxed),
then macro-average across judged (qa_id, sentence_id) cells.

Output (JSON, also printed): six numbers per qrels file::

    {
      "strict":  {"support":  {"P":..,"R":..,"F1":..},
                  "contradict":{"P":..,"R":..,"F1":..}},
      "relaxed": {"support":  ..., "contradict": ...}
    }

Task: 6.3, 10.1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Literal

import orjson

from trec_biogen.io.qrels import QrelsIndex, load_qrels

Class = Literal["support", "contradict"]
Setting = Literal["strict", "relaxed"]
_CLASSES: tuple[Class, ...] = ("support", "contradict")
_SETTINGS: tuple[Setting, ...] = ("strict", "relaxed")


@dataclass(slots=True)
class PRF:
    P: float
    R: float
    F1: float

    def as_dict(self) -> dict[str, float]:
        return {"P": self.P, "R": self.R, "F1": self.F1}


def _prf(predicted: Iterable[str], positives: set[str]) -> PRF | None:
    """Return per-cell P/R/F1 or None if the cell is unjudged (no positives)."""
    if not positives:
        return None
    pred = set(predicted)
    if not pred:
        return PRF(0.0, 0.0, 0.0)
    tp = len(pred & positives)
    p = tp / len(pred)
    r = tp / len(positives)
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return PRF(p, r, f1)


def _iter_submission_cells(submission_path: Path) -> Iterable[tuple[str, int, Class, list[str]]]:
    with submission_path.open("rb") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            rec = orjson.loads(raw)
            qa_id = str(rec["qa_id"])
            for sent in rec.get("sentences", []):
                sid = int(sent["sentence_id"])
                yield qa_id, sid, "support", list(sent.get("support_pmids", []))
                yield qa_id, sid, "contradict", list(sent.get("contradict_pmids", []))


def evaluate(submission_path: Path, qrels: QrelsIndex) -> dict[str, dict[str, dict[str, float]]]:
    """Return {setting: {class: {P,R,F1}}} macro-averaged across judged cells."""
    per_cell: dict[Setting, dict[Class, list[PRF]]] = {
        s: {c: [] for c in _CLASSES} for s in _SETTINGS
    }
    for qa_id, sid, cls, predicted in _iter_submission_cells(submission_path):
        for setting in _SETTINGS:
            positives = qrels.positives(qa_id, sid, cls, setting=setting)
            res = _prf(predicted, positives)
            if res is not None:
                per_cell[setting][cls].append(res)

    out: dict[str, dict[str, dict[str, float]]] = {}
    for setting in _SETTINGS:
        out[setting] = {}
        for cls in _CLASSES:
            cells = per_cell[setting][cls]
            if not cells:
                out[setting][cls] = {"P": 0.0, "R": 0.0, "F1": 0.0}
            else:
                # Macro-average P and R; compute F1 from averaged P/R to match
                # the published convention (per-class P/R/F1, not mean F1).
                avg_p = mean(c.P for c in cells)
                avg_r = mean(c.R for c in cells)
                avg_f1 = (2 * avg_p * avg_r / (avg_p + avg_r)) if (avg_p + avg_r) else 0.0
                out[setting][cls] = {"P": avg_p, "R": avg_r, "F1": avg_f1}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--submission", required=True, type=Path)
    p.add_argument("--qrels", required=True, type=Path)
    p.add_argument("--out", type=Path, help="Optional JSON output path")
    a = p.parse_args(argv)

    qrels = load_qrels(a.qrels)
    report = evaluate(a.submission, qrels)
    text = json.dumps(report, indent=2)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
