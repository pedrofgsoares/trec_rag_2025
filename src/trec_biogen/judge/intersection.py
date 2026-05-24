"""Phase 2.5 §2 / Phase 2.6 §3 — N-judge intersection-pool emitter.

§12.4 multi-backend concordance established that LLM-positive Supports
are robust across judges (κ high on the Supports class) but LLM-positive
Contradicts carry meaningful judge-dependent variance (κ ≈ 0.34 overall;
Together-Llama-70B is materially more conservative on contradicts than
gpt-4o-mini). This module materialises a stricter pool that intersects
*only* the Contradicts class across N≥2 judges, while passing Supports
through from one canonical judge unchanged.

Algorithm:

1. Read N backends' expanded qrels (each a superset of the human qrels).
2. Read the human qrels separately to know which records are human.
3. Emit a single qrels file containing:
   * Every human record from `records_paths[0]` copied verbatim (their
     `source` field stays `"human"`).
   * Every LLM Supports record from
     `records_paths[supports_source_index]` (default 0) copied verbatim
     (Supports are not intersected — §12.4 shows them stable).
   * Every LLM Contradicts record from `records_paths[0]` whose
     `(qa_id, sentence_id, pmid, class)` triple is present in **all
     other N-1** inputs. Its emitted `source` is `"llm-intersection"`.
4. Write a sidecar `<out>.meta.json` with SHA256s of every input,
   timestamps, before/after positive counts per class (including all
   pairwise intersections for N≥3), and the `incomplete` flag
   propagated from any input.

Phase 2.6 added the N-way generalisation; the original two-judge
function signature remains supported (the dispatch detects whether
the first positional arg is a Path or a list of Paths).

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
    records_a_or_paths,
    records_b_path: Path | None = None,
    *,
    human_qrels_path: Path,
    out_path: Path,
    supports_source_index: int = 0,
) -> dict[str, Any]:
    """Emit an N-judge intersection-on-contradicts qrels file.

    Two call shapes are supported:

    * **Phase 2.5 (two-judge)**: ``emit_intersection_pool(a, b, human_qrels_path=h, out_path=o)``.
      The first positional arg is a single Path; ``records_b_path`` is the
      second. The output is byte-for-byte equivalent to what Phase 2.5
      produced (Phase 2.6 §3.5 regression test asserts this).
    * **Phase 2.6 (N-judge, N ≥ 2)**: ``emit_intersection_pool([a, b, c], human_qrels_path=h, out_path=o)``.
      The first positional arg is a list of N Paths. ``supports_source_index``
      (default 0) picks which input's Supports records pass through.

    The contradict intersection requires the same
    ``(qa_id, sentence_id, pmid, class)`` triple to appear as a positive in
    **all N** inputs. Human positives are copied verbatim from
    ``records_paths[0]`` (which itself copies them from the human qrels).
    """
    # Dispatch: detect old vs new call shape.
    if isinstance(records_a_or_paths, (list, tuple)):
        records_paths = [Path(p) for p in records_a_or_paths]
        if records_b_path is not None:
            raise ValueError(
                "pass either records_paths=[a, b, ...] OR (records_a, records_b), not both"
            )
    else:
        if records_b_path is None:
            raise ValueError(
                "legacy two-arg form requires both records_a_path and records_b_path"
            )
        records_paths = [Path(records_a_or_paths), Path(records_b_path)]

    if len(records_paths) < 2:
        raise ValueError(f"need at least 2 inputs, got {len(records_paths)}")
    if not (0 <= supports_source_index < len(records_paths)):
        raise ValueError(
            f"supports_source_index={supports_source_index} out of range "
            f"for {len(records_paths)} inputs"
        )

    canonical_path = records_paths[0]
    supports_path = records_paths[supports_source_index]

    # For each non-canonical input, build the set of Contradicts keys it
    # endorses. The intersection is then "key in all of these sets".
    other_contradict_key_sets = [
        _contradict_keys(_iter_jsonl(p))
        for p in records_paths
        if p != canonical_path
    ]
    # When N=2 with canonical==index 0 (the common case), this collapses
    # to the single set the Phase 2.5 code used; when supports_source_index
    # != 0 it includes more sets but still includes the canonical one's
    # "agree with itself" set implicitly (a key trivially agrees with itself,
    # so canonical's own contradicts are not added explicitly).
    if not other_contradict_key_sets:
        # Pathological case: every input == canonical (cannot happen given
        # the len>=2 check + path comparison, but keep defensive).
        other_contradict_key_sets = [set()]

    # First pass: pre-intersection counts on the canonical input.
    a_pre_counts = {"support": 0, "contradict": 0, "human": 0}
    for rec in _iter_jsonl(canonical_path):
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

    # When supports come from a different input than the canonical, we need
    # two streams: human + contradicts from canonical, supports from the
    # supports_path. To keep the file deterministic and similar to Phase 2.5,
    # we iterate canonical first (emit human + contradicts), then supports_path
    # if different (emit only its supports). When supports_source_index == 0
    # (the typical case) the single-pass over canonical does everything.
    seen_keys: set[tuple[str, int, str, str]] = set()
    with out_path.open("wb") as fh:
        # Pass 1: canonical input emits human + intersected contradicts.
        # Also emits supports if supports_source_index == 0.
        emit_supports_from_canonical = (supports_source_index == 0)
        for rec in _iter_jsonl(canonical_path):
            if int(rec.get("relevance", 1)) <= 0:
                continue
            if _record_is_human(rec):
                fh.write(orjson.dumps(rec) + b"\n")
                out_counts["human"] += 1
                seen_keys.add(_record_key(rec))
                continue
            cls = str(rec["class"])
            if cls in _CANONICAL_SUPPORT_CLASSES:
                if emit_supports_from_canonical:
                    fh.write(orjson.dumps(rec) + b"\n")
                    out_counts["support"] += 1
                    seen_keys.add(_record_key(rec))
            elif cls in _CANONICAL_CONTRADICT_CLASSES:
                key = _record_key(rec)
                # Keep iff every *other* input also endorses this contradict.
                if all(key in ks for ks in other_contradict_key_sets):
                    emitted = dict(rec)
                    emitted["source"] = _INTERSECTION_SOURCE
                    fh.write(orjson.dumps(emitted) + b"\n")
                    out_counts["contradict"] += 1
                    seen_keys.add(key)

        # Pass 2 (only if supports come from a different input): emit
        # supports from the chosen supports_path. Skip records already seen.
        if not emit_supports_from_canonical:
            for rec in _iter_jsonl(supports_path):
                if int(rec.get("relevance", 1)) <= 0:
                    continue
                if _record_is_human(rec):
                    continue  # human passes through canonical pass 1
                cls = str(rec["class"])
                if cls in _CANONICAL_SUPPORT_CLASSES and _record_key(rec) not in seen_keys:
                    fh.write(orjson.dumps(rec) + b"\n")
                    out_counts["support"] += 1

    a_contradicts = a_pre_counts["contradict"]
    contradicts_dropped = a_contradicts - out_counts["contradict"]
    dropped_pct = (
        (contradicts_dropped / a_contradicts) if a_contradicts > 0 else 0.0
    )

    incomplete = any(_is_incomplete(p) for p in records_paths)

    # Per-file SHA256 + (for N ≥ 3) pairwise intersection counts for
    # diagnostic comparison against the Phase 2.5 two-judge pool.
    sha256s = {str(p): _sha256(p) for p in records_paths}
    pairwise = {}
    if len(records_paths) >= 3:
        for i, pi in enumerate(records_paths):
            for j, pj in enumerate(records_paths):
                if i >= j:
                    continue
                a_keys = _contradict_keys(_iter_jsonl(pi))
                b_keys = _contradict_keys(_iter_jsonl(pj))
                pairwise[f"{pi.name} ∩ {pj.name}"] = len(a_keys & b_keys)

    meta = {
        "out_path": str(out_path),
        "records_paths": [str(p) for p in records_paths],
        "supports_source_index": supports_source_index,
        "supports_source_path": str(supports_path),
        # Phase 2.5 compatibility keys (legacy consumers may read these).
        "records_a_path": str(records_paths[0]),
        "records_b_path": str(records_paths[1]) if len(records_paths) == 2 else None,
        "human_qrels_path": str(human_qrels_path),
        "records_sha256": sha256s,
        # Phase 2.5 compatibility for the two-input case.
        "records_a_sha256": sha256s[str(records_paths[0])],
        "records_b_sha256": sha256s[str(records_paths[1])] if len(records_paths) == 2 else None,
        "human_qrels_sha256": _sha256(human_qrels_path),
        "intersection_rule": (
            f"support: pass-through from records_paths[{supports_source_index}]; "
            f"contradict: keep iff (qa_id, sentence_id, pmid, class) "
            f"is a positive in all {len(records_paths)} inputs"
        ),
        "before_intersection": a_pre_counts,
        "after_intersection": out_counts,
        "pairwise_contradict_intersections": pairwise,  # empty when N==2
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
        description=(
            "Emit the N-judge intersection-on-contradicts qrels. "
            "Either pass --records-a / --records-b (Phase 2.5 two-judge form) "
            "or --records-paths <file1> <file2> [file3 ...] (Phase 2.6 N-judge form)."
        ),
    )
    p.add_argument("--records-a", type=Path,
                   help="Canonical (first-arg) judge expanded qrels JSONL (two-judge form).")
    p.add_argument("--records-b", type=Path,
                   help="Second judge expanded qrels JSONL (two-judge form).")
    p.add_argument("--records-paths", type=Path, nargs="+",
                   help="N≥2 judge expanded qrels JSONL files (N-judge form). "
                        "First file is the canonical source for Supports and human pass-through.")
    p.add_argument("--supports-source-index", type=int, default=0,
                   help="Index into --records-paths for Supports pass-through (default 0).")
    p.add_argument("--human-qrels", type=Path,
                   default=Path("data/qrels/biogen2025_taskA_qrels.jsonl"))
    p.add_argument(
        "--out", type=Path,
        default=Path("data/qrels/biogen2025_taskA_qrels_intersection.jsonl"),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.records_paths:
        if args.records_a or args.records_b:
            raise SystemExit(
                "pass either --records-paths OR --records-a/--records-b, not both"
            )
        meta = emit_intersection_pool(
            args.records_paths,
            human_qrels_path=args.human_qrels, out_path=args.out,
            supports_source_index=args.supports_source_index,
        )
    else:
        if not (args.records_a and args.records_b):
            raise SystemExit(
                "two-judge form requires both --records-a and --records-b"
            )
        meta = emit_intersection_pool(
            args.records_a, args.records_b,
            human_qrels_path=args.human_qrels, out_path=args.out,
        )
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
