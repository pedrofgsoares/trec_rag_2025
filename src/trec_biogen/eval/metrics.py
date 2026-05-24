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

from trec_biogen.io.qrels import QrelsIndex, Source, load_qrels

Class = Literal["support", "contradict"]
Setting = Literal["strict", "relaxed"]
_CLASSES: tuple[Class, ...] = ("support", "contradict")
_SETTINGS: tuple[Setting, ...] = ("strict", "relaxed")

# Default qrels path per pool. ``--qrels-pool`` is a sugar that fills
# ``--qrels`` when the latter is omitted, so the eval CLI can be invoked
# without remembering the file name. ``--qrels`` always wins if passed
# explicitly.
DEFAULT_QRELS_PATHS: dict[str, Path] = {
    "official": Path("data/qrels/biogen2025_taskA_qrels.jsonl"),
    "expanded": Path("data/qrels/biogen2025_taskA_qrels_expanded.jsonl"),
    # Phase 2.5: two-judge intersection-on-contradicts pool.
    "intersection": Path("data/qrels/biogen2025_taskA_qrels_intersection.jsonl"),
    # Phase 2.6: three-judge intersection-on-contradicts pool
    # (mini-cot ∩ Llama-3.3-70B-cot ∩ Qwen-2.5-72B-cot).
    "intersection-3way": Path("data/qrels/biogen2025_taskA_qrels_intersection_3way.jsonl"),
}


def krippendorff_alpha(
    labels_per_coder: list[list[str | None]],
    *,
    classes: tuple[str, ...],
    missing_marker: str | None = None,
) -> float:
    """Krippendorff's α for nominal data with N≥2 coders.

    Standard formula (Krippendorff 1980, 2011; Hayes & Krippendorff 2007):

        α = 1 − D_observed / D_expected

    with the nominal disagreement function ``δ(c1, c2) = 0 if c1 == c2 else 1``.
    Per-unit terms are weighted by ``m_u * (m_u - 1)`` where ``m_u`` is the
    number of non-missing judgements on unit ``u`` (Krippendorff 2011 eq. 1) —
    when every unit has the same number of judgements this reduces to the
    simple formulation in Hayes & Krippendorff (2007).

    ``labels_per_coder`` is a list of N lists (one per coder), all of equal
    length U (one entry per unit). Entries equal to ``missing_marker``
    (default ``None``) are treated as missing and dropped from the unit's
    counts; units with fewer than 2 non-missing labels contribute zero to
    both numerator and denominator (they carry no within-unit disagreement
    information). In our primary use case (every backend judges every
    triple) no missing markers appear; the parameter is exposed to support
    published reference fixtures (e.g. Hayes & Krippendorff 2007 Table 1
    contains 3 missing values).

    Reference fixture: Hayes & Krippendorff (2007) Table 1 yields α = 0.7434
    with three ``"*"`` entries treated as missing. The test suite anchors
    against that value to detect any algorithmic drift.
    """
    if not labels_per_coder:
        raise ValueError("labels_per_coder must contain at least one coder")
    n_coders = len(labels_per_coder)
    if n_coders < 2:
        raise ValueError(f"need ≥2 coders for an inter-coder reliability metric, got {n_coders}")
    n_units = len(labels_per_coder[0])
    for i, lst in enumerate(labels_per_coder):
        if len(lst) != n_units:
            raise ValueError(
                f"all coders must have the same number of units; "
                f"coder 0 has {n_units}, coder {i} has {len(lst)}"
            )
    if n_units == 0:
        return 1.0  # vacuously perfect agreement
    class_set = set(classes)
    for ci, lst in enumerate(labels_per_coder):
        for ui, lab in enumerate(lst):
            if lab == missing_marker:
                continue
            if lab not in class_set:
                raise ValueError(
                    f"coder {ci} unit {ui} has label {lab!r} not in classes "
                    f"{classes} and not the missing marker {missing_marker!r}"
                )

    # Krippendorff (2011) eq. 5 + coincidence-matrix formulation: per-unit
    # contribution to observed disagreement is (m_u² − Σ_c n_uc²) / (m_u − 1)
    # (units with m_u < 2 contribute 0); total observed disagreement
    # D_obs = (sum of per-unit contributions) / N where N is the total
    # number of non-missing labels across all units.
    obs_numerator = 0.0
    total_label_counts: dict[str, int] = {c: 0 for c in classes}
    for ui in range(n_units):
        per_unit_counts: dict[str, int] = {c: 0 for c in classes}
        for lst in labels_per_coder:
            lab = lst[ui]
            if lab == missing_marker:
                continue
            per_unit_counts[lab] += 1
            total_label_counts[lab] += 1
        m_u = sum(per_unit_counts.values())
        if m_u < 2:
            continue  # carries no within-unit disagreement information
        m_sq = m_u * m_u
        same_sq = sum(c * c for c in per_unit_counts.values())
        obs_numerator += (m_sq - same_sq) / (m_u - 1)

    N = sum(total_label_counts.values())
    if N < 2:
        return 1.0

    D_obs = obs_numerator / N

    # Expected disagreement under random pairing across all non-missing labels.
    # D_exp = (N² − Σ_c n_c²) / (N · (N − 1))
    sum_sq = sum(c * c for c in total_label_counts.values())
    D_exp = (N * N - sum_sq) / (N * (N - 1))

    if D_exp == 0:
        return 1.0  # all non-missing labels are the same class
    return 1.0 - (D_obs / D_exp)


@dataclass(slots=True)
class PRF:
    P: float
    R: float
    F1: float

    def as_dict(self) -> dict[str, float]:
        return {"P": self.P, "R": self.R, "F1": self.F1}


def _prf(
    predicted: Iterable[str],
    positives: set[str],
    *,
    unjudged_as_zero: bool = True,
) -> PRF | None:
    """Return per-cell P/R/F1.

    ``unjudged_as_zero``:
      * ``True`` (default, matches the published BioGEN 2025 methodology):
        a cell with empty ``positives`` AND a non-empty ``predicted`` set
        contributes ``F1=0`` to the macro. Reproduces the published
        organizers-baseline numbers (Support F1 = 44.34, Contradict F1 ≈ 4.67).
      * ``False`` (legacy): unjudged cells are dropped, so the macro
        averages only over judged cells. Useful when comparing across
        qrels of different judgement coverage (e.g. partial 2024 question-
        level qrels).
    """
    pred = set(predicted)
    if not positives:
        if unjudged_as_zero and pred:
            return PRF(0.0, 0.0, 0.0)
        return None
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
    unjudged_as_zero: bool | None = None,
    source: Source = "any",
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
    # Sentence-level: by default mirror the published BioGEN 2025 protocol
    # (unjudged cells count as F1=0 if the submission predicts something).
    # Question-level: by default drop unjudged cells (cross-year fallback).
    if unjudged_as_zero is None:
        unjudged_as_zero = (level == "sentence")

    if level == "question":
        return _evaluate_question_level(
            submission_path, qrels, unjudged_as_zero=unjudged_as_zero, source=source,
        )

    per_cell: dict[Setting, dict[Class, list[PRF]]] = {
        s: {c: [] for c in _CLASSES} for s in _SETTINGS
    }
    for qa_id, sid, cls, predicted in _iter_submission_cells(submission_path):
        for setting in _SETTINGS:
            positives = qrels.positives(qa_id, sid, cls, setting=setting, source=source)
            res = _prf(predicted, positives, unjudged_as_zero=unjudged_as_zero)
            if res is not None:
                per_cell[setting][cls].append(res)
    return _macro_average(per_cell)


def _evaluate_question_level(
    submission_path: Path,
    qrels: QrelsIndex,
    *,
    unjudged_as_zero: bool = False,
    source: Source = "any",
) -> dict[str, dict[str, dict[str, float]]]:
    # Union predictions per (qa_id, class) across sentences.
    by_qa: dict[tuple[str, Class], set[str]] = {}
    for qa_id, _sid, cls, predicted in _iter_submission_cells(submission_path):
        key = (qa_id, cls)
        by_qa.setdefault(key, set()).update(predicted)

    per_cell: dict[Setting, dict[Class, list[PRF]]] = {
        s: {c: [] for c in _CLASSES} for s in _SETTINGS
    }
    for (qa_id, cls), predicted in by_qa.items():
        for setting in _SETTINGS:
            positives = qrels.question_positives(qa_id, cls, setting=setting, source=source)
            res = _prf(predicted, positives, unjudged_as_zero=unjudged_as_zero)
            if res is not None:
                per_cell[setting][cls].append(res)
    return _macro_average(per_cell)


def _macro_average(
    per_cell: dict[Setting, dict[Class, list[PRF]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Macro-average per (setting, class). F1 is mean-of-per-cell-F1s, matching
    the published BioGEN 2025 methodology. P and R are macro means."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for setting in _SETTINGS:
        out[setting] = {}
        for cls in _CLASSES:
            cells = per_cell[setting][cls]
            if not cells:
                out[setting][cls] = {"P": 0.0, "R": 0.0, "F1": 0.0}
            else:
                out[setting][cls] = {
                    "P":  mean(c.P  for c in cells),
                    "R":  mean(c.R  for c in cells),
                    "F1": mean(c.F1 for c in cells),
                }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--submission", required=True, type=Path)
    p.add_argument(
        "--qrels", type=Path, default=None,
        help="Explicit qrels JSONL path. Overrides --qrels-pool when set.",
    )
    p.add_argument(
        "--qrels-pool",
        choices=sorted(DEFAULT_QRELS_PATHS),
        default="official",
        help="Convenience: pick the canonical qrels file for the given pool. "
             "'official' = data/qrels/biogen2025_taskA_qrels.jsonl. "
             "'expanded' = data/qrels/biogen2025_taskA_qrels_expanded.jsonl. "
             "'intersection' = data/qrels/biogen2025_taskA_qrels_intersection.jsonl "
             "(Phase 2.5: two-judge intersection-on-contradicts). "
             "'intersection-3way' = data/qrels/biogen2025_taskA_qrels_intersection_3way.jsonl "
             "(Phase 2.6: three-judge unanimity on contradicts). "
             "Ignored when --qrels is set.",
    )
    p.add_argument(
        "--source", choices=["human", "llm", "any"], default="any",
        help="Filter qrels positives by source attribution. Default 'any' "
             "unions human + LLM judgements. 'human' restricts to the "
             "official-pool subset for §6.5 reproducibility (works on the "
             "expanded file too — it filters out LLM rows). 'llm' isolates "
             "the LLM-only contribution.",
    )
    p.add_argument("--out", type=Path, help="Optional JSON output path")
    p.add_argument(
        "--level",
        choices=["sentence", "question"],
        default="sentence",
        help="sentence (default, for 2025 qrels) or question (for 2024 collection)",
    )
    a = p.parse_args(argv)

    qrels_path = a.qrels if a.qrels is not None else DEFAULT_QRELS_PATHS[a.qrels_pool]
    if not qrels_path.exists():
        hint = {
            "expanded": "Run `python -m trec_biogen.judge.rejudge expand-pool` first.",
            "intersection": (
                "Run `python -m trec_biogen.judge.intersection` after producing "
                "both backends' expanded qrels (mini-cot canonical + Together / HF-Llama)."
            ),
            "intersection-3way": (
                "Run `python -m trec_biogen.judge.intersection --records-paths "
                "biogen2025_taskA_qrels_expanded.jsonl biogen2025_taskA_qrels_expanded_hf-llama.jsonl "
                "biogen2025_taskA_qrels_expanded_qwen.jsonl --out data/qrels/biogen2025_taskA_qrels_intersection_3way.jsonl` "
                "after producing all three backends' expanded qrels."
            ),
        }.get(a.qrels_pool, "")
        msg = f"qrels file not found: {qrels_path}"
        if hint:
            msg += f"\n{hint}"
        raise SystemExit(msg)
    qrels = load_qrels(qrels_path)
    report = evaluate(a.submission, qrels, level=a.level, source=a.source)
    text = json.dumps(report, indent=2)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
