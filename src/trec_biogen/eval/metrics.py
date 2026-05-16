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
    """Yield ``(qa_id, sentence_id, class, predicted_pmids)`` from either submission shape.

    Accepts both:
      * official ``task_a_output.json`` — a single JSON list with ``meta_data.qa_id``
        and ``answer[]`` carrying ``supported_citations`` / ``contradicted_citations``
        as integer PMIDs;
      * legacy JSONL — ``{qa_id, sentences:[{sentence_id, support_pmids, contradict_pmids}]}``.
    """
    raw = Path(submission_path).read_bytes().lstrip()
    if raw[:1] == b"[":
        items = orjson.loads(raw)
        for item in items:
            meta = item.get("meta_data") or item.get("metadata") or {}
            qa_id = str(meta.get("qa_id") or "")
            for sid, ans in enumerate(item.get("answer", [])):
                yield qa_id, sid, "support", [str(p) for p in ans.get("supported_citations") or []]
                yield qa_id, sid, "contradict", [str(p) for p in ans.get("contradicted_citations") or []]
        return

    with submission_path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = orjson.loads(line)
            qa_id = str(rec["qa_id"])
            for sent in rec.get("sentences", []):
                sid = int(sent["sentence_id"])
                yield qa_id, sid, "support", [str(p) for p in sent.get("support_pmids", [])]
                yield qa_id, sid, "contradict", [str(p) for p in sent.get("contradict_pmids", [])]


def evaluate(
    submission_path: Path,
    qrels: QrelsIndex,
    *,
    level: Literal["sentence", "question"] = "sentence",
) -> dict[str, dict[str, dict[str, float]]]:
    """Return {setting: {class: {P,R,F1}}} macro-averaged across judged cells.

    Level:
      * ``sentence`` (default) — score each (qa_id, sentence_id, class) cell
        against the qrels positives for that exact triple. Used with qrels
        whose sentence_ids match the submission's input (e.g. 2025).
      * ``question`` — union predictions across sentences per (qa_id, class)
        and score against ``qrels.question_positives``. Used with qrels
        whose sentence_ids do not align with the submission (e.g. BioGEN
        2024 where the source qrels references each team's generated
        answer-sentences). Aggregates per qa_id rather than per cell.
    """
    if level == "question":
        return _evaluate_question_level(submission_path, qrels)

    per_cell: dict[Setting, dict[Class, list[PRF]]] = {
        s: {c: [] for c in _CLASSES} for s in _SETTINGS
    }
    for qa_id, sid, cls, predicted in _iter_submission_cells(submission_path):
        for setting in _SETTINGS:
            positives = qrels.positives(qa_id, sid, cls, setting=setting)
            res = _prf(predicted, positives)
            if res is not None:
                per_cell[setting][cls].append(res)
    return _macro_average(per_cell)


def _evaluate_question_level(
    submission_path: Path, qrels: QrelsIndex
) -> dict[str, dict[str, dict[str, float]]]:
    # Union predictions per (qa_id, class) across sentences.
    by_qa: dict[tuple[str, Class], set[str]] = {}
    for qa_id, _sid, cls, predicted in _iter_submission_cells(submission_path):
        key = (qa_id, cls)
        by_qa.setdefault(key, set()).update(predicted)

    per_cell: dict[Setting, dict[Class, list[PRF]]] = {
        s: {c: [] for c in _CLASSES} for s in _SETTINGS
    }
    seen: set[tuple[str, Class]] = set()
    for (qa_id, cls), predicted in by_qa.items():
        seen.add((qa_id, cls))
        for setting in _SETTINGS:
            positives = qrels.question_positives(qa_id, cls, setting=setting)
            res = _prf(predicted, positives)
            if res is not None:
                per_cell[setting][cls].append(res)
    return _macro_average(per_cell)


def _macro_average(
    per_cell: dict[Setting, dict[Class, list[PRF]]],
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for setting in _SETTINGS:
        out[setting] = {}
        for cls in _CLASSES:
            cells = per_cell[setting][cls]
            if not cells:
                out[setting][cls] = {"P": 0.0, "R": 0.0, "F1": 0.0}
            else:
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
    p.add_argument(
        "--level",
        choices=["sentence", "question"],
        default="sentence",
        help="sentence (default, for 2025 qrels) or question (for 2024 collection)",
    )
    a = p.parse_args(argv)

    qrels = load_qrels(a.qrels)
    report = evaluate(a.submission, qrels, level=a.level)
    text = json.dumps(report, indent=2)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
