"""Phase 2.5 §5 — cross-run topic-level diff CLI.

Two modes:

1. ``--a <run> --b <run> --qa-id <int>`` — print the per-class PMID set
   diffs for a single topic. For each PMID in ``A \\ B`` and ``B \\ A``,
   look up the LLM-judge label and confidence in the chosen qrels pool
   (default: intersection); if rejudge records exist next to the qrels,
   print a one-line excerpt of the model's reasoning chain.

2. ``--select-3 --target <run> --anchor <run>`` — print the three topics
   that maximise (a) positive Δ F1(target − anchor), (b) absolute Δ
   closest to zero, (c) negative Δ. The selection is mechanical and
   prints the full sorted appendix so a reader can verify the choice.

Pool source for both modes defaults to the intersection pool
(``data/qrels/biogen2025_taskA_qrels_intersection.jsonl``) when present,
falling back to the expanded pool then the official pool.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import orjson

from trec_biogen.eval.metrics import DEFAULT_QRELS_PATHS, _iter_submission_cells
from trec_biogen.eval.per_topic import (
    per_topic_f1,
    select_three_topics,
    topic_f1_delta,
)
from trec_biogen.io.qrels import load_qrels


# ---------------------------------------------------------------------------
# Pool resolution
# ---------------------------------------------------------------------------


def _resolve_pool_path(explicit: Path | None, pool: str) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"qrels file not found at {explicit} (--qrels-path)")
        return explicit
    if pool == "auto":
        for candidate_pool in ("intersection", "expanded", "official"):
            p = DEFAULT_QRELS_PATHS[candidate_pool]
            if p.exists():
                return p
        raise SystemExit(
            "no qrels file found at any canonical pool path "
            f"({list(DEFAULT_QRELS_PATHS.values())})"
        )
    p = DEFAULT_QRELS_PATHS[pool]
    if not p.exists():
        raise SystemExit(f"qrels file not found at {p} (pool={pool!r})")
    return p


# ---------------------------------------------------------------------------
# Rejudge records lookup (for reasoning-chain excerpts)
# ---------------------------------------------------------------------------


def _candidate_records_paths() -> list[Path]:
    """Persisted per-call rejudge records consulted by `--qa-id` mode
    when surfacing model reasoning. Best-effort: missing files just drop
    the reasoning column."""
    return [
        Path("data/interim/validate_cot_records.jsonl"),
        Path("data/interim/validate_cot_records_together.jsonl"),
    ]


def _load_rejudge_records(paths: Iterable[Path]) -> dict[tuple[str, int, str], dict]:
    """Build `(qa_id, sentence_id, pmid) -> record` from any rejudge
    records JSONL files we can find. The records may not exist (it's
    populated by §12.1's `--records-out`); absence is fine."""
    out: dict[tuple[str, int, str], dict] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("rb") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                rec = orjson.loads(raw)
                key = (str(rec.get("qa_id", "")),
                       int(rec.get("sentence_id", -1)),
                       str(rec.get("pmid", "")))
                # Keep first record per key (mini-cot is the canonical).
                out.setdefault(key, rec)
    return out


def _reasoning_excerpt(rec: dict | None, max_chars: int = 140) -> str:
    if rec is None:
        return ""
    text = (
        rec.get("raw_response")
        or rec.get("reasoning")
        or rec.get("chain")
        or ""
    )
    text = str(text).replace("\n", " ").strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return text


# ---------------------------------------------------------------------------
# Submission readers
# ---------------------------------------------------------------------------


def _submission_predictions(
    run_dir: Path, qa_id: str,
) -> dict[int, dict[str, set[str]]]:
    """Return `{sentence_id: {"support": {pmid,...}, "contradict": {pmid,...}}}`
    for the requested qa_id from `<run_dir>/task_a_output.json`."""
    submission = run_dir / "task_a_output.json"
    if not submission.exists():
        raise SystemExit(f"no task_a_output.json under {run_dir}")
    out: dict[int, dict[str, set[str]]] = defaultdict(
        lambda: {"support": set(), "contradict": set()}
    )
    for q, sid, cls, predicted in _iter_submission_cells(submission):
        if q != qa_id:
            continue
        out[sid][cls].update(predicted)
    if not out:
        raise SystemExit(
            f"qa_id {qa_id!r} not found in {submission}"
        )
    return dict(out)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def _format_pmid_line(
    pmid: str, cls: str, qrels, qa_id: str, sid: int,
    rejudge_records: dict[tuple[str, int, str], dict],
) -> str:
    # Class label and source via qrels.strict / strict_sources.
    key = (qa_id, sid, cls)
    label = "<unjudged>"
    source = "—"
    confidence = "—"
    if key in qrels.strict and pmid in qrels.strict[key]:
        label = cls
        source = qrels.strict_sources.get(key, {}).get(pmid, "?")
    elif key in qrels.relaxed and pmid in qrels.relaxed[key]:
        label = f"{cls} (relaxed)"
        source = qrels.relaxed_sources.get(key, {}).get(pmid, "?")
    rec = rejudge_records.get((qa_id, sid, pmid))
    if rec is not None:
        confidence = f"{float(rec.get('confidence', 0)):.2f}"
    excerpt = _reasoning_excerpt(rec)
    parts = [f"  - {pmid:>10}  [{label}]  source={source}  conf={confidence}"]
    if excerpt:
        parts.append(f"      reasoning: {excerpt}")
    return "\n".join(parts)


def cmd_qa_id_diff(args: argparse.Namespace) -> int:
    qrels_path = _resolve_pool_path(args.qrels_path, args.pool)
    qrels = load_qrels(qrels_path)
    rejudge_records = _load_rejudge_records(_candidate_records_paths())

    a_preds = _submission_predictions(args.a, args.qa_id)
    b_preds = _submission_predictions(args.b, args.qa_id)

    print(f"# Per-topic diff for qa_id={args.qa_id}")
    print(f"  A: {args.a}")
    print(f"  B: {args.b}")
    print(f"  pool qrels: {qrels_path}")
    print()

    sids = sorted(set(a_preds) | set(b_preds))
    for sid in sids:
        print(f"## sentence_id = {sid}")
        for cls in ("support", "contradict"):
            ap = a_preds.get(sid, {}).get(cls, set())
            bp = b_preds.get(sid, {}).get(cls, set())
            print(f"### class = {cls}")
            print(f"  A predicts ({len(ap)}): {sorted(ap)}")
            print(f"  B predicts ({len(bp)}): {sorted(bp)}")
            only_a = sorted(ap - bp)
            only_b = sorted(bp - ap)
            if only_a:
                print(f"  A \\ B ({len(only_a)}):")
                for pmid in only_a:
                    print(_format_pmid_line(pmid, cls, qrels, args.qa_id, sid, rejudge_records))
            if only_b:
                print(f"  B \\ A ({len(only_b)}):")
                for pmid in only_b:
                    print(_format_pmid_line(pmid, cls, qrels, args.qa_id, sid, rejudge_records))
            if not only_a and not only_b:
                print("  (sets identical)")
            print()
    return 0


def cmd_select_3(args: argparse.Namespace) -> int:
    qrels_path = _resolve_pool_path(args.qrels_path, args.pool)
    target_report = per_topic_f1(args.target, qrels_path=qrels_path, pool=args.pool)
    anchor_report = per_topic_f1(args.anchor, qrels_path=qrels_path, pool=args.pool)
    deltas = topic_f1_delta(
        target_report, anchor_report, setting=args.setting, cls=args.cls,
    )
    if len(deltas) < 3:
        raise SystemExit(
            f"only {len(deltas)} topics present in both runs — cannot pick 3."
        )
    pos, neutral, neg = select_three_topics(deltas)

    print(f"# Mechanical 3-topic selection")
    print(f"  target: {args.target}")
    print(f"  anchor: {args.anchor}")
    print(f"  pool: {qrels_path}")
    print(f"  setting={args.setting}, class={args.cls}")
    print()
    print("## Picks")
    print(f"  largest positive Δ : qa_id={pos:>4}  Δ={deltas[pos]:+.4f}")
    print(f"  closest-to-zero Δ  : qa_id={neutral:>4}  Δ={deltas[neutral]:+.4f}")
    print(f"  largest negative Δ : qa_id={neg:>4}  Δ={deltas[neg]:+.4f}")
    print()
    print("## Full sorted appendix (largest positive first)")
    print()
    sorted_items = sorted(deltas.items(), key=lambda kv: -kv[1])
    print("| qa_id | Δ F1 |")
    print("|---|---|")
    for qa, d in sorted_items:
        print(f"| {qa} | {d:+.4f} |")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "target": str(args.target),
                    "anchor": str(args.anchor),
                    "pool": str(qrels_path),
                    "setting": args.setting,
                    "class": args.cls,
                    "picks": {"positive": pos, "neutral": neutral, "negative": neg},
                    "deltas": deltas,
                },
                indent=2, sort_keys=True,
            ),
            encoding="utf-8",
        )
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts/per_topic_diff.py",
        description="Per-topic PMID-level diff or mechanical 3-topic selection.",
    )
    p.add_argument(
        "--pool", choices=("auto", "official", "expanded", "intersection"),
        default="auto",
        help="Canonical qrels pool to consult. 'auto' = intersection > expanded > official.",
    )
    p.add_argument(
        "--qrels-path", type=Path, default=None,
        help="Override the qrels file directly.",
    )
    p.add_argument("--qa-id", type=str, default=None,
                   help="If set, run the single-topic diff mode.")
    p.add_argument("--a", type=Path, default=None, help="Run A (single-topic mode).")
    p.add_argument("--b", type=Path, default=None, help="Run B (single-topic mode).")
    p.add_argument("--select-3", action="store_true",
                   help="Mechanical 3-topic selection mode.")
    p.add_argument("--target", type=Path, default=None,
                   help="Target run for --select-3.")
    p.add_argument("--anchor", type=Path, default=None,
                   help="Anchor (comparison baseline) run for --select-3.")
    p.add_argument("--setting", choices=("strict", "relaxed"), default="strict")
    p.add_argument("--cls", choices=("support", "contradict"), default="support")
    p.add_argument("--json-out", type=Path, default=None,
                   help="Optional JSON dump of the selection result.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.select_3:
        if args.target is None or args.anchor is None:
            raise SystemExit("--select-3 requires --target and --anchor")
        return cmd_select_3(args)
    if args.qa_id is not None:
        if args.a is None or args.b is None:
            raise SystemExit("--qa-id mode requires --a and --b")
        return cmd_qa_id_diff(args)
    raise SystemExit(
        "pick a mode: either --qa-id <int> (with --a, --b) or --select-3 "
        "(with --target, --anchor)"
    )


if __name__ == "__main__":
    sys.exit(main())
