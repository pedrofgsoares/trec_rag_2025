"""Phase 2.5 §2 — Two-judge intersection-pool emitter.

§12.4 multi-backend concordance established that LLM-positive Supports
are robust across judges (κ high on the Supports class) but LLM-positive
Contradicts carry meaningful judge-dependent variance (κ ≈ 0.34 overall;
Together-Llama-70B is materially more conservative on contradicts than
gpt-4o-mini). This module materialises a stricter pool that intersects
*only* the Contradicts class across two judges, while passing Supports
through from the canonical judge unchanged.

Algorithm (per design §D3):

1. Read both backends' expanded qrels (each a superset of the human
   qrels).
2. Read the human qrels separately to know which records are human.
3. Emit a third qrels file containing:
   * Every human record from `records_a` copied verbatim (their
     `source` field stays `"human"`).
   * Every LLM Supports record from `records_a` copied verbatim
     (Supports are not intersected — §12.4 shows them stable).
   * Every LLM Contradicts record from `records_a` that is also
     present (matching `(qa_id, sentence_id, pmid, class)`) in
     `records_b`. Its emitted `source` is `"llm-intersection"`.
4. Write a sidecar `<out>.meta.json` with SHA256s of both inputs,
   timestamps, before/after positive counts per class, and the
   `incomplete` flag propagated from either input.

The output is a strict superset of the human qrels and a strict
subset of the canonical `records_a` expanded qrels. NOTE: re-scoring
against it *can* in principle inflate an individual variant's F1 —
removing a positive that a variant did not predict raises that
variant's recall denominator-side fairness without adding false
positives. In practice on our data every variant's Contradicts F1
*decreased* because the pipelines did predict some of the dropped
contradicts, but the general claim "cannot inflate" is incorrect.
Reviewers (Codex, 2026-05-22) flagged the original wording; corrected
here.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import orjson


_CANONICAL_CONTRADICT_CLASSES = {"contradict", "partial_contradict"}
_CANONICAL_SUPPORT_CLASSES = {"support", "partial_support"}
_INTERSECTION_SOURCE = "llm-intersection"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            yield orjson.loads(raw)


def _meta_sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def _is_incomplete(qrels_path: Path) -> bool:
    sidecar = _meta_sidecar_path(qrels_path)
    if not sidecar.exists():
        return False
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(meta.get("incomplete", False))


def _record_is_human(rec: dict[str, Any]) -> bool:
    return str(rec.get("source", "human")) == "human"


def _record_key(rec: dict[str, Any]) -> tuple[str, int, str, str]:
    """Identity key for a single qrels record.

    Class is included because the intersection contract is
    (qa_id, sentence_id, pmid, class) — a triple could in principle
    be labelled differently by two judges, in which case we do not
    intersect them.
    """
    return (
        str(rec["qa_id"]),
        int(rec["sentence_id"]),
        str(rec["pmid"]),
        str(rec["class"]),
    )


def _contradict_keys(records: Iterable[dict[str, Any]]) -> set[tuple[str, int, str, str]]:
    out: set[tuple[str, int, str, str]] = set()
    for rec in records:
        if int(rec.get("relevance", 1)) <= 0:
            continue
        if str(rec["class"]) in _CANONICAL_CONTRADICT_CLASSES and not _record_is_human(rec):
            out.add(_record_key(rec))
    return out


def emit_intersection_pool(
    records_a_path: Path,
    records_b_path: Path,
    *,
    human_qrels_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Emit a two-judge-intersection qrels file derived from `records_a`
    and `records_b`. Returns the sidecar metadata as a dict; the file and
    its sidecar are written to disk as side-effects.

    Args:
      records_a_path: canonical (first-arg) judge's expanded qrels. Its
        Supports records are passed through; its Contradicts records are
        kept only when also present in `records_b_path`.
      records_b_path: second judge's expanded qrels. Only its Contradicts
        records affect the output (via the intersection check).
      human_qrels_path: the human-labelled qrels. Used to derive
        SHA256 and for the metadata sidecar; the actual human records
        are emitted from `records_a` (which copies them verbatim).
      out_path: destination JSONL path. A sidecar
        ``<out_path>.meta.json`` is written next to it.
    """
    # Build the set of Contradicts keys that B agrees on.
    b_contradict_keys = _contradict_keys(_iter_jsonl(records_b_path))

    # First pass: count A's pre-intersection positives per class for the
    # sidecar — cheap and gives an exact "dropped" figure.
    a_pre_counts = {"support": 0, "contradict": 0, "human": 0}
    for rec in _iter_jsonl(records_a_path):
        if int(rec.get("relevance", 1)) <= 0:
            continue
        if _record_is_human(rec):
            a_pre_counts["human"] += 1
        else:
            cls = str(rec["class"])
            if cls in _CANONICAL_SUPPORT_CLASSES:
                a_pre_counts["support"] += 1
            elif cls in _CANONICAL_CONTRADICT_CLASSES:
                a_pre_counts["contradict"] += 1

    out_counts = {"support": 0, "contradict": 0, "human": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        for rec in _iter_jsonl(records_a_path):
            if int(rec.get("relevance", 1)) <= 0:
                continue
            if _record_is_human(rec):
                # Human records pass through verbatim.
                fh.write(orjson.dumps(rec) + b"\n")
                out_counts["human"] += 1
                continue
            cls = str(rec["class"])
            if cls in _CANONICAL_SUPPORT_CLASSES:
                # Supports pass through unchanged (not intersected per §D2).
                fh.write(orjson.dumps(rec) + b"\n")
                out_counts["support"] += 1
            elif cls in _CANONICAL_CONTRADICT_CLASSES:
                # Contradicts only kept if B also agrees.
                if _record_key(rec) in b_contradict_keys:
                    emitted = dict(rec)
                    emitted["source"] = _INTERSECTION_SOURCE
                    fh.write(orjson.dumps(emitted) + b"\n")
                    out_counts["contradict"] += 1

    a_contradicts = a_pre_counts["contradict"]
    contradicts_dropped = a_contradicts - out_counts["contradict"]
    dropped_pct = (
        (contradicts_dropped / a_contradicts) if a_contradicts > 0 else 0.0
    )

    incomplete = _is_incomplete(records_a_path) or _is_incomplete(records_b_path)

    meta = {
        "out_path": str(out_path),
        "records_a_path": str(records_a_path),
        "records_b_path": str(records_b_path),
        "human_qrels_path": str(human_qrels_path),
        "records_a_sha256": _sha256(records_a_path),
        "records_b_sha256": _sha256(records_b_path),
        "human_qrels_sha256": _sha256(human_qrels_path),
        "intersection_rule": (
            "support: pass-through from records_a; "
            "contradict: keep iff (qa_id, sentence_id, pmid, class) "
            "matches a positive in records_b"
        ),
        "before_intersection": a_pre_counts,
        "after_intersection": out_counts,
        "contradicts_dropped": contradicts_dropped,
        "contradicts_dropped_pct": round(dropped_pct, 4),
        "incomplete": incomplete,
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    }
    _meta_sidecar_path(out_path).write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    return meta


# ---------------------------------------------------------------------------
# CLI: emit one intersection file from two backends' rejudge outputs.
# ---------------------------------------------------------------------------


def _build_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="trec_biogen.judge.intersection",
        description="Emit the two-judge intersection-on-contradicts qrels.",
    )
    p.add_argument("--records-a", type=Path, required=True,
                   help="Canonical (first-arg) judge expanded qrels JSONL.")
    p.add_argument("--records-b", type=Path, required=True,
                   help="Second judge expanded qrels JSONL.")
    p.add_argument("--human-qrels", type=Path,
                   default=Path("data/qrels/biogen2025_taskA_qrels.jsonl"))
    p.add_argument(
        "--out", type=Path,
        default=Path("data/qrels/biogen2025_taskA_qrels_intersection.jsonl"),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    meta = emit_intersection_pool(
        args.records_a, args.records_b,
        human_qrels_path=args.human_qrels, out_path=args.out,
    )
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
