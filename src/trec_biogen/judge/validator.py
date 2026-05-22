"""LLM-judge concordance validator.

Classifies the human-labeled triples in
``data/qrels/biogen2025_taskA_qrels.jsonl`` with the configured backend,
computes per-class P/R/F1 and a support-weighted macro F1, and emits
``reports/llm_judge_validation.md``. The concordance gate (task 2.8)
fails the validation run if macro weighted F1 < ``threshold`` (default 0.85
via the CLI; configurable).

Task: 2.7, 2.8
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import orjson

from trec_biogen.judge.backends import Judge, JudgeRecord
from trec_biogen.judge.prompts import LABELS

# Human qrels classes ↔ judge labels. Partial classes collapse to their
# strict counterpart, matching the QrelsIndex loader.
_QRELS_TO_LABEL: dict[str, str] = {
    "support": "Supports",
    "partial_support": "Supports",
    "contradict": "Contradicts",
    "partial_contradict": "Contradicts",
}


@dataclass(slots=True, frozen=True)
class Triple:
    qa_id: str
    sentence_id: int
    pmid: str
    human_label: str


def load_validation_triples(qrels_path: Path) -> list[Triple]:
    out: list[Triple] = []
    with qrels_path.open("rb") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            rec = orjson.loads(raw)
            if int(rec.get("relevance", 1)) <= 0:
                continue
            label = _QRELS_TO_LABEL.get(str(rec.get("class", "")))
            if label is None:
                continue
            out.append(
                Triple(
                    qa_id=str(rec["qa_id"]),
                    sentence_id=int(rec["sentence_id"]),
                    pmid=str(rec["pmid"]),
                    human_label=label,
                )
            )
    return out


@dataclass(slots=True)
class ClassMetric:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(slots=True)
class ValidationResult:
    per_class: dict[str, ClassMetric]
    confusion: dict[str, dict[str, int]]
    macro_weighted_f1: float
    total: int

    def passes(self, threshold: float) -> bool:
        return self.macro_weighted_f1 >= threshold


def _weighted_f1(per_class: dict[str, ClassMetric]) -> float:
    total = sum(m.support for m in per_class.values())
    if total == 0:
        return 0.0
    return sum(m.f1 * m.support for m in per_class.values()) / total


def bootstrap_ci(
    pairs: list[tuple[str, str]],
    *,
    n_iter: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap a 95% CI on macro-weighted-F1 over (gold, pred) pairs.

    Phase 2 §12.1. Resamples ``pairs`` with replacement ``n_iter`` times,
    recomputes macro-weighted-F1 for each sample, and returns
    ``(point_estimate, low, high)`` where ``low`` / ``high`` are the
    ``alpha/2`` / ``1 - alpha/2`` empirical percentiles. Promotes
    "0.85 PASS" to "0.85 with 95% CI [a, b]" — paper-grade statistical
    defensibility for the concordance gate claim.
    """
    import random

    rng = random.Random(seed)
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0.0
    samples: list[float] = []
    for _ in range(n_iter):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        samples.append(score(sample).macro_weighted_f1)
    samples.sort()
    point = score(pairs).macro_weighted_f1
    lo_idx = max(0, int((alpha / 2) * n_iter))
    hi_idx = min(n_iter - 1, int((1 - alpha / 2) * n_iter))
    return point, samples[lo_idx], samples[hi_idx]


def score(pairs: Iterable[tuple[str, str]]) -> ValidationResult:
    """Per-class P/R/F1 + support-weighted macro F1 from ``(gold, predicted)`` pairs.

    Labels outside :data:`LABELS` get their own zero-support row added to
    the confusion matrix; they contribute zero to the weighted macro since
    weighted-F1 weighs by gold support.
    """
    pairs = list(pairs)
    total = len(pairs)
    confusion: dict[str, dict[str, int]] = {g: {p: 0 for p in LABELS} for g in LABELS}
    for gold, pred in pairs:
        confusion.setdefault(gold, {p: 0 for p in LABELS})
        confusion[gold].setdefault(pred, 0)
        confusion[gold][pred] += 1

    gold_counts = Counter(g for g, _ in pairs)
    pred_counts = Counter(p for _, p in pairs)
    per_class: dict[str, ClassMetric] = {}
    for label in LABELS:
        tp = confusion.get(label, {}).get(label, 0)
        fp = pred_counts.get(label, 0) - tp
        fn = gold_counts.get(label, 0) - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        per_class[label] = ClassMetric(
            precision=p, recall=r, f1=f1, support=gold_counts.get(label, 0),
        )
    return ValidationResult(
        per_class=per_class,
        confusion=confusion,
        macro_weighted_f1=_weighted_f1(per_class),
        total=total,
    )


def render_report(
    result: ValidationResult,
    *,
    backend_name: str,
    qrels_path: Path,
    threshold: float,
) -> str:
    verdict = "PASS" if result.passes(threshold) else "FAIL"
    lines = [
        "# LLM-Judge Concordance Validation",
        "",
        f"- Backend: `{backend_name}`",
        f"- Qrels: `{qrels_path}`",
        f"- Threshold (macro weighted F1): `{threshold:.2f}`",
        f"- Triples scored: {result.total}",
        f"- **Macro weighted F1: {result.macro_weighted_f1:.4f}** ({verdict})",
        "",
        "## Per-class metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for label in LABELS:
        m = result.per_class[label]
        lines.append(
            f"| {label} | {m.precision:.4f} | {m.recall:.4f} | {m.f1:.4f} | {m.support} |"
        )
    lines += [
        "",
        "## Confusion matrix",
        "",
        "Rows: human label. Columns: judge label.",
        "",
        "| | " + " | ".join(LABELS) + " |",
        "|---|" + "|".join(["---"] * len(LABELS)) + "|",
    ]
    for gold in LABELS:
        row = [gold] + [str(result.confusion.get(gold, {}).get(p, 0)) for p in LABELS]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def run_validation(
    judge: Judge,
    triples: list[Triple],
    *,
    abstract_lookup: Callable[[str], str],
    answer_sentence_lookup: Callable[[str, int], str],
    records_out: Path | None = None,
) -> tuple[ValidationResult, list[JudgeRecord]]:
    """Classify every triple with ``judge`` and return the validation result + raw records.

    When ``records_out`` is set, also writes one JSONL row per triple with
    columns: ``qa_id, sentence_id, pmid, gold, pred, confidence,
    input_tokens, output_tokens, cost_usd, skip_reason``. Phase 2 §12.1
    consumes this file for the bootstrap-CI analysis.
    """
    import orjson

    records: list[JudgeRecord] = []
    pairs: list[tuple[str, str]] = []
    rows: list[dict] = []
    for t in triples:
        sentence = answer_sentence_lookup(t.qa_id, t.sentence_id)
        abstract = abstract_lookup(t.pmid)
        rec = judge.classify(sentence, t.pmid, abstract)
        records.append(rec)
        pairs.append((t.human_label, rec.label))
        if records_out is not None:
            rows.append(
                {
                    "qa_id": t.qa_id, "sentence_id": t.sentence_id, "pmid": t.pmid,
                    "gold": t.human_label, "pred": rec.label,
                    "confidence": rec.confidence,
                    "input_tokens": rec.input_tokens, "output_tokens": rec.output_tokens,
                    "cost_usd": rec.cost_usd, "backend": rec.backend,
                    "skip_reason": rec.skip_reason,
                }
            )
    if records_out is not None:
        records_out.parent.mkdir(parents=True, exist_ok=True)
        with records_out.open("wb") as fh:
            for row in rows:
                fh.write(orjson.dumps(row) + b"\n")
    return score(pairs), records
