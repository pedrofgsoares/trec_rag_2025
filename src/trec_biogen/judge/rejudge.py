"""LLM-judge CLI.

Subcommands:

- ``validate``         — run concordance gate against the human qrels.
- ``rejudge``          — classify novel PMIDs from a Phase 1 submission and
                         emit ``data/qrels/biogen2025_taskA_qrels_expanded.jsonl``.
- ``compare-backends`` — classify a fixed 200-pair sample with each named
                         backend and emit pairwise concordance numbers.

All subcommands honour ``--cost-cap`` (USD) and write cost / token
accounting into ``metadata.yaml`` when ``--metadata-run-dir`` is set. The
``rejudge`` subcommand emits a sidecar ``<out>.meta.json`` carrying the
``incomplete`` flag and abort reason, so the expanded qrels JSONL stays a
strict JSONL parseable by :mod:`trec_biogen.io.qrels`.

Task: 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 2.13
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import orjson
import yaml

from trec_biogen.judge.backends import (
    BACKEND_REGISTRY,
    Backend,
    Judge,
    JudgeRecord,
    QuotaExhausted,
    make_backend,
)
from trec_biogen.judge.validator import (
    Triple,
    load_validation_triples,
    render_report,
    run_validation,
    score,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def load_answer_sentence_lookup(topics_path: Path) -> Callable[[str, int], str]:
    """Build a ``(qa_id, sentence_id) -> sentence text`` lookup from the BioGen input."""
    raw = orjson.loads(Path(topics_path).read_bytes())
    by_key: dict[tuple[str, int], str] = {}
    for item in raw:
        qa_id = str(item["meta_data"]["qa_id"])
        for sid, sent in enumerate(item.get("answer", [])):
            text = sent.get("text", "") if isinstance(sent, dict) else str(sent)
            by_key[(qa_id, sid)] = text

    def lookup(qa_id: str, sentence_id: int) -> str:
        return by_key.get((qa_id, sentence_id), "")

    return lookup


def bm25_abstract_lookup(index_dir: Path) -> Callable[[str], str]:
    """Build a ``pmid -> abstract text`` lookup backed by the BM25 index."""
    from trec_biogen.retrieval.bm25 import BM25Index

    bm = BM25Index(index_dir)
    return bm.doc_text


def record_to_metadata_payload(records: list[JudgeRecord]) -> dict:
    """Aggregate per-call records into a metadata.yaml-ready payload."""
    total_cost = round(sum(r.cost_usd for r in records), 6)
    by_backend: dict[str, int] = {}
    for r in records:
        by_backend[r.backend] = by_backend.get(r.backend, 0) + 1
    return {
        "judge_cost_usd": total_cost,
        "judge_token_breakdown": {
            "input_tokens": sum(r.input_tokens for r in records),
            "output_tokens": sum(r.output_tokens for r in records),
            "cache_hit_rate": 0.0,
        },
        "judge_calls": len(records),
        "judge_calls_skipped": sum(1 for r in records if r.skip_reason),
        "judge_calls_by_backend": by_backend,
    }


def write_run_metadata(run_dir: Path, payload: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "metadata.yaml"
    existing = yaml.safe_load(out.read_text()) if out.exists() else {}
    existing.update(payload)
    out.write_text(yaml.safe_dump(existing, sort_keys=False))
    return out


def _label_to_class(label: str) -> str | None:
    if label == "Supports":
        return "support"
    if label == "Contradicts":
        return "contradict"
    return None


# ---------------------------------------------------------------------------
# Expanded qrels emitter (task 2.12)
# ---------------------------------------------------------------------------


def emit_expanded_qrels(
    *,
    human_qrels_path: Path,
    llm_records: dict[tuple[str, int, str], JudgeRecord],
    out_path: Path,
    incomplete: bool = False,
    abort_reason: str = "",
) -> Path:
    """Write expanded qrels: human rows verbatim + LLM rows with source/confidence.

    LLM records whose label is neither Supports nor Contradicts are dropped
    (they do not contribute to either positive class). The sidecar
    ``<out>.meta.json`` carries the ``incomplete`` flag and the abort
    reason so the JSONL itself remains parseable by :func:`load_qrels`.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for line in human_qrels_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rec.setdefault("source", "human")
            fh.write(json.dumps(rec) + "\n")
        for (qa_id, sid, pmid), r in sorted(llm_records.items()):
            cls = _label_to_class(r.label)
            if cls is None:
                continue
            payload = {
                "qa_id": qa_id,
                "sentence_id": sid,
                "pmid": pmid,
                "class": cls,
                "relevance": 1,
                "source": r.backend,
                "confidence": round(r.confidence, 4),
            }
            fh.write(json.dumps(payload) + "\n")

    sidecar = out_path.with_suffix(out_path.suffix + ".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "incomplete": bool(incomplete),
                "abort_reason": abort_reason,
                "llm_record_count": len(llm_records),
            },
            indent=2,
        )
        + "\n"
    )
    return out_path


# ---------------------------------------------------------------------------
# Submission diffing
# ---------------------------------------------------------------------------


def load_existing_llm_judgements(out_path: Path) -> dict[tuple[str, int, str], JudgeRecord]:
    """Read prior LLM rows from an existing expanded-qrels file.

    Used by the rejudge resume path: re-invoking with the same ``--out``
    after a quota/cost-cap halt skips triples already judged. The reloaded
    records carry ``skip_reason="resumed"`` and zero cost so this session's
    cost accounting only reflects new spend.
    """
    if not out_path.exists():
        return {}
    class_to_label = {"support": "Supports", "contradict": "Contradicts"}
    out: dict[tuple[str, int, str], JudgeRecord] = {}
    with out_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = rec.get("source")
            if not src or src == "human":
                continue
            label = class_to_label.get(str(rec.get("class", "")))
            if label is None:
                continue
            key = (str(rec["qa_id"]), int(rec["sentence_id"]), str(rec["pmid"]))
            out[key] = JudgeRecord(
                label=label,
                confidence=float(rec.get("confidence", 0.0)),
                input_tokens=0,
                output_tokens=0,
                backend=str(src),
                cost_usd=0.0,
                skip_reason="resumed",
            )
    return out


def human_qrels_keys(qrels_path: Path) -> set[tuple[str, int, str]]:
    """Return the set of (qa_id, sentence_id, pmid) already in the human qrels."""
    out: set[tuple[str, int, str]] = set()
    with qrels_path.open("rb") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            rec = orjson.loads(raw)
            out.add((str(rec["qa_id"]), int(rec["sentence_id"]), str(rec["pmid"])))
    return out


def top_k_triples_from_retrieval_parquets(
    retrieval_paths: list[Path],
    *,
    top_k: int = 30,
    exclude: set[tuple[str, int, str]] | None = None,
) -> list[tuple[str, int, str]]:
    """Enumerate ``(qa_id, sentence_id, candidate_pmid)`` for the BM25 top-K of every cell.

    Reads each parquet in ``retrieval_paths`` (typically the support and
    contradict retrieval parquets from a Phase 1 run), filters
    ``rank <= top_k``, deduplicates per (qa_id, sentence_id, pmid)
    across both paths, and optionally subtracts ``exclude``. Phase 2
    §2.17 — the broader expanded-qrels pool that decouples cross-variant
    comparisons from the §2.16 circular-bias artefact discovered after
    `phase2_no_rerank`.
    """
    import polars as pl

    triples: set[tuple[str, int, str]] = set()
    for p in retrieval_paths:
        df = pl.read_parquet(p).filter(pl.col("rank") <= top_k)
        for row in df.iter_rows(named=True):
            triples.add(
                (str(row["qa_id"]), int(row["sentence_id"]), str(row["candidate_pmid"]))
            )
    if exclude:
        triples -= exclude
    return sorted(triples)


def novel_pmids_from_submission(
    submission_path: Path,
    qrels_path: Path,
) -> list[tuple[str, int, str]]:
    """Return ``(qa_id, sentence_id, pmid)`` triples in the submission but NOT the qrels."""
    qrels_keys: set[tuple[str, int, str]] = set()
    with qrels_path.open("rb") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            rec = orjson.loads(raw)
            qrels_keys.add(
                (str(rec["qa_id"]), int(rec["sentence_id"]), str(rec["pmid"]))
            )

    novel: set[tuple[str, int, str]] = set()
    items = orjson.loads(submission_path.read_bytes())
    for item in items:
        qa_id = str(item.get("meta_data", {}).get("qa_id", ""))
        for sid, ans in enumerate(item.get("answer", [])):
            for key in ("supported_citations", "contradicted_citations"):
                for p in ans.get(key) or []:
                    triple = (qa_id, sid, str(p))
                    if triple not in qrels_keys:
                        novel.add(triple)
    return sorted(novel)


# ---------------------------------------------------------------------------
# Subcommand: validate (2.7, 2.8)
# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace, *, backend: Backend | None = None) -> int:
    judge = Judge(backend or make_backend(args.backend, prompt_mode=args.prompt))
    triples = load_validation_triples(args.qrels)
    answer_lookup = load_answer_sentence_lookup(args.topics)
    abstract_lookup = bm25_abstract_lookup(args.index)

    result, records = run_validation(
        judge, triples,
        abstract_lookup=abstract_lookup,
        answer_sentence_lookup=answer_lookup,
        records_out=args.records_out,
    )
    report = render_report(
        result, backend_name=judge.name, qrels_path=args.qrels, threshold=args.threshold,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    if args.metadata_run_dir:
        write_run_metadata(args.metadata_run_dir, record_to_metadata_payload(records))
    if not result.passes(args.threshold):
        print(
            f"::error::concordance gate failed: macro weighted F1 = "
            f"{result.macro_weighted_f1:.4f} < {args.threshold:.2f}",
            file=sys.stderr,
        )
        return 1
    return 0


# ---------------------------------------------------------------------------
# Subcommand: rejudge (2.9, 2.10, 2.11, 2.12)
# ---------------------------------------------------------------------------


def _judge_triples_and_emit(
    *,
    judge: Judge,
    all_triples: list[tuple[str, int, str]],
    answer_lookup,
    abstract_lookup,
    out_path: Path,
    human_qrels_path: Path,
    max_concurrent: int,
    cost_cap: float | None,
    metadata_run_dir: Path | None = None,
    label: str = "rejudge",
) -> int:
    """Run the judge over ``all_triples``, write expanded qrels, return exit code.

    Resume mode kicks in automatically if ``out_path`` already exists —
    prior LLM judgements are reused and only the remaining triples are
    submitted. Cost-cap and ``QuotaExhausted`` both trigger a graceful
    halt that still emits a partial expanded qrels with ``incomplete:
    true`` in the sidecar. Shared by :func:`cmd_rejudge` (§2.16) and
    :func:`cmd_expand_pool` (§2.17).
    """
    resumed = load_existing_llm_judgements(out_path)
    results: dict[tuple[str, int, str], JudgeRecord] = dict(resumed)
    remaining = [t for t in all_triples if t not in resumed]
    print(
        f"[{label}] triples: {len(all_triples)} "
        f"(resumed {len(resumed)}, to classify {len(remaining)})"
    )

    aborted = False
    abort_reason = ""
    running_cost = 0.0
    session_records: list[JudgeRecord] = []

    def task(t: tuple[str, int, str]) -> tuple[tuple[str, int, str], JudgeRecord]:
        qa_id, sid, pmid = t
        sentence = answer_lookup(qa_id, sid)
        abstract = abstract_lookup(pmid)
        rec = judge.classify(sentence, pmid, abstract)
        return t, rec

    try:
        with ThreadPoolExecutor(max_workers=max(1, max_concurrent)) as ex:
            futures = [ex.submit(task, t) for t in remaining]
            for fut in as_completed(futures):
                try:
                    t, rec = fut.result()
                except QuotaExhausted as exc:
                    aborted = True
                    abort_reason = f"quota_exhausted: {exc}"
                    for f in futures:
                        f.cancel()
                    break
                results[t] = rec
                session_records.append(rec)
                running_cost += rec.cost_usd
                if cost_cap is not None and running_cost >= cost_cap:
                    aborted = True
                    abort_reason = f"cost_cap_reached:${cost_cap:.2f}"
                    for f in futures:
                        f.cancel()
                    break
    except KeyboardInterrupt:
        aborted = True
        abort_reason = "interrupted"

    incomplete = aborted or (len(results) < len(all_triples))
    emit_expanded_qrels(
        human_qrels_path=human_qrels_path,
        llm_records=results,
        out_path=out_path,
        incomplete=incomplete,
        abort_reason=abort_reason,
    )

    n_sup = sum(1 for r in results.values() if r.label == "Supports")
    n_con = sum(1 for r in results.values() if r.label == "Contradicts")
    print(
        f"[{label}] wrote {out_path} — {len(results)} judged total "
        f"({n_sup} support / {n_con} contradict; {len(session_records)} this session), "
        f"cost ${running_cost:.4f}, incomplete={incomplete}"
        + (f", reason={abort_reason}" if abort_reason else "")
    )
    if metadata_run_dir:
        meta = record_to_metadata_payload(session_records)
        meta["judge_calls_total"] = len(results)
        meta["judge_calls_resumed"] = len(resumed)
        meta["incomplete"] = incomplete
        if abort_reason:
            meta["abort_reason"] = abort_reason
        write_run_metadata(metadata_run_dir, meta)
    return 0 if not incomplete else 2


def cmd_rejudge(args: argparse.Namespace, *, backend: Backend | None = None) -> int:
    judge = Judge(backend or make_backend(args.backend, prompt_mode=args.prompt))
    answer_lookup = load_answer_sentence_lookup(args.topics)
    abstract_lookup = bm25_abstract_lookup(args.index)
    all_triples = novel_pmids_from_submission(args.submission, args.qrels)
    return _judge_triples_and_emit(
        judge=judge, all_triples=all_triples,
        answer_lookup=answer_lookup, abstract_lookup=abstract_lookup,
        out_path=args.out, human_qrels_path=args.qrels,
        max_concurrent=args.max_concurrent, cost_cap=args.cost_cap,
        metadata_run_dir=args.metadata_run_dir,
        label="rejudge",
    )


def cmd_expand_pool(args: argparse.Namespace, *, backend: Backend | None = None) -> int:
    """§2.17: rejudge BM25 top-K per (qa_id, sentence_id) cell into a
    broader expanded pool, decoupling cross-variant comparisons from the
    Phase-1-shaped pool produced by §2.16."""
    judge = Judge(backend or make_backend(args.backend, prompt_mode=args.prompt))
    answer_lookup = load_answer_sentence_lookup(args.topics)
    abstract_lookup = bm25_abstract_lookup(args.index)
    excluded = human_qrels_keys(args.qrels)
    retrieval_paths = [args.retrieval_support, args.retrieval_contradict]
    all_triples = top_k_triples_from_retrieval_parquets(
        retrieval_paths, top_k=args.top_k, exclude=excluded,
    )
    return _judge_triples_and_emit(
        judge=judge, all_triples=all_triples,
        answer_lookup=answer_lookup, abstract_lookup=abstract_lookup,
        out_path=args.out, human_qrels_path=args.qrels,
        max_concurrent=args.max_concurrent, cost_cap=args.cost_cap,
        metadata_run_dir=args.metadata_run_dir,
        label="expand-pool",
    )


# ---------------------------------------------------------------------------
# Subcommand: compare-backends (2.13)
# ---------------------------------------------------------------------------


def cmd_compare_backends(
    args: argparse.Namespace,
    *,
    backend_overrides: dict[str, Backend] | None = None,
) -> int:
    sample = orjson.loads(args.sample.read_bytes())
    if len(sample) > 200:
        sample = sample[:200]
    backend_names: list[str] = list(args.backends)
    overrides = backend_overrides or {}

    per_backend_labels: dict[str, list[str]] = {}
    per_backend_records: dict[str, list[JudgeRecord]] = {}
    for name in backend_names:
        judge = Judge(overrides.get(name) or make_backend(name))
        labels: list[str] = []
        recs: list[JudgeRecord] = []
        for item in sample:
            r = judge.classify(item["answer_sentence"], item["pmid"], item["abstract"])
            labels.append(r.label)
            recs.append(r)
        per_backend_labels[name] = labels
        per_backend_records[name] = recs

    rows: list[str] = [
        "| A | B | weighted F1 (A vs B) | agreement |",
        "|---|---|---|---|",
    ]
    for i, a in enumerate(backend_names):
        for b in backend_names[i + 1:]:
            pairs = list(zip(per_backend_labels[a], per_backend_labels[b]))
            result = score(pairs)
            agree = sum(1 for x, y in pairs if x == y) / len(pairs) if pairs else 0.0
            rows.append(
                f"| {a} | {b} | {result.macro_weighted_f1:.4f} | {agree:.4f} |"
            )

    cost_lines = [
        "",
        "## Cost per backend",
        "",
        "| backend | calls | cost USD |",
        "|---|---|---|",
    ]
    for name in backend_names:
        recs = per_backend_records[name]
        cost = sum(r.cost_usd for r in recs)
        cost_lines.append(f"| {name} | {len(recs)} | {cost:.4f} |")

    body = "\n".join(
        ["# Backend comparison", "", f"- Sample size: {len(sample)}", ""]
        + rows
        + cost_lines
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(body, encoding="utf-8")
    print(body)
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trec_biogen.judge.rejudge")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _add_backend(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--backend", choices=sorted(BACKEND_REGISTRY), default="together",
            help="LLM-judge backend.",
        )
        sp.add_argument(
            "--prompt", choices=("strict", "cot"), default="strict",
            help="Prompt mode. 'strict' = 4-label JSON only. 'cot' = adds a "
                 "reasoning chain before the label (recommended; resolved the "
                 "Phase-2 §2.15 concordance-gate failure).",
        )

    v = sub.add_parser("validate", help="Run concordance validation against the human qrels.")
    _add_backend(v)
    v.add_argument("--qrels", type=Path, required=True)
    v.add_argument("--topics", type=Path, required=True)
    v.add_argument("--index", type=Path, required=True, help="BM25 index dir.")
    v.add_argument("--threshold", type=float, default=0.85)
    v.add_argument("--report", type=Path, default=Path("reports/llm_judge_validation.md"))
    v.add_argument("--metadata-run-dir", type=Path, default=None)
    v.add_argument(
        "--records-out", type=Path, default=None,
        help="If set, dump per-call (gold, pred, confidence, cost) JSONL — "
             "consumed by §12.1 bootstrap-CI and §12.2 calibration analyses.",
    )
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("rejudge", help="Classify novel PMIDs from a Phase 1 submission.")
    _add_backend(r)
    r.add_argument("--submission", type=Path, required=True)
    r.add_argument("--qrels", type=Path, required=True)
    r.add_argument("--topics", type=Path, required=True)
    r.add_argument("--index", type=Path, required=True, help="BM25 index dir.")
    r.add_argument(
        "--out", type=Path,
        default=Path("data/qrels/biogen2025_taskA_qrels_expanded.jsonl"),
    )
    r.add_argument("--cost-cap", type=float, default=None,
                   help="USD spend cap; aborts gracefully when reached.")
    r.add_argument("--max-concurrent", type=int, default=4)
    r.add_argument("--metadata-run-dir", type=Path, default=None)
    r.set_defaults(func=cmd_rejudge)

    e = sub.add_parser(
        "expand-pool",
        help="§2.17: rejudge BM25 top-K per (qa_id, sentence_id) into a broader expanded qrels.",
    )
    _add_backend(e)
    e.add_argument("--retrieval-support", type=Path, required=True,
                   help="retrieval_support.parquet from a prior run (BM25 k=100).")
    e.add_argument("--retrieval-contradict", type=Path, required=True,
                   help="retrieval_contradict.parquet from a prior run (BM25 k=1000).")
    e.add_argument("--top-k", type=int, default=30,
                   help="Per-cell top-K to enumerate from each retrieval parquet.")
    e.add_argument("--qrels", type=Path, required=True)
    e.add_argument("--topics", type=Path, required=True)
    e.add_argument("--index", type=Path, required=True, help="BM25 index dir.")
    e.add_argument(
        "--out", type=Path,
        default=Path("data/qrels/biogen2025_taskA_qrels_expanded.jsonl"),
    )
    e.add_argument("--cost-cap", type=float, default=None)
    e.add_argument("--max-concurrent", type=int, default=4)
    e.add_argument("--metadata-run-dir", type=Path, default=None)
    e.set_defaults(func=cmd_expand_pool)

    c = sub.add_parser(
        "compare-backends",
        help="Pairwise concordance on a fixed 200-pair sample (task 2.13).",
    )
    c.add_argument(
        "--sample", type=Path, required=True,
        help="JSON list of {answer_sentence, pmid, abstract} records.",
    )
    c.add_argument(
        "--backends", nargs="+", required=True, choices=sorted(BACKEND_REGISTRY),
    )
    c.add_argument("--report", type=Path, default=Path("reports/backend_comparison.md"))
    c.set_defaults(func=cmd_compare_backends)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
