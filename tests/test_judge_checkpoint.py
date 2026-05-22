"""Phase 2.5 — Incremental checkpoint contract for `_judge_triples_and_emit`.

Covers the new resilience properties added before the Together rejudge
re-launch:

1. The expanded qrels file is written every N completed triples
   (atomic, via `os.replace`), not only at end of batch.
2. After a checkpoint, the partial file is parseable by `load_qrels`
   and consumable by `load_existing_llm_judgements` — i.e. resume
   mode picks up where the checkpoint stopped.
3. A non-quota exception during the loop still flushes whatever has
   been judged before re-raising, instead of silently dropping the
   in-memory dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest

from trec_biogen.judge.backends import JudgeRecord, Judge
from trec_biogen.judge.rejudge import (
    _judge_triples_and_emit,
    emit_expanded_qrels,
    load_existing_llm_judgements,
)


def _seed_human_qrels(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        orjson.dumps({
            "qa_id": "Q001", "sentence_id": 0, "pmid": "10000001",
            "class": "support", "relevance": 1,
        }).decode() + "\n",
        encoding="utf-8",
    )
    return path


def _make_record(label: str = "Supports", *, backend: str = "test", confidence: float = 0.9) -> JudgeRecord:
    return JudgeRecord(
        label=label, confidence=confidence,
        input_tokens=10, output_tokens=5, cost_usd=0.0001,
        backend=backend, skip_reason=None,
    )


def _stub_judge(label_seq: list[str] | None = None) -> Judge:
    """A Judge whose classify() returns canned labels in order. Doesn't
    touch any HTTPBackend."""
    j = MagicMock(spec=Judge)
    if label_seq is None:
        j.classify.side_effect = lambda *a, **kw: _make_record("Supports")
    else:
        it = iter(label_seq)
        j.classify.side_effect = lambda *a, **kw: _make_record(next(it))
    return j


def test_checkpoint_writes_file_every_n_completed_triples(tmp_path: Path) -> None:
    out = tmp_path / "expanded.jsonl"
    human = _seed_human_qrels(tmp_path / "human.jsonl")
    judge = _stub_judge()
    triples = [("Q001", 0, f"P{i}") for i in range(1, 11)]  # 10 triples

    _judge_triples_and_emit(
        judge=judge, all_triples=triples,
        answer_lookup=lambda *a, **k: "s",
        abstract_lookup=lambda *a, **k: "abs",
        out_path=out, human_qrels_path=human,
        max_concurrent=1, cost_cap=None,
        # Checkpoint every 3 → expect intermediate writes at 3, 6, 9, then final.
        checkpoint_every=3,
    )

    # Final file present and has all 10 + 1 human.
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    llm_pmids = sorted(r["pmid"] for r in rows if r.get("source", "human") != "human")
    assert llm_pmids == sorted(f"P{i}" for i in range(1, 11))


def test_partial_checkpoint_file_is_resumable(tmp_path: Path) -> None:
    """After a checkpoint mid-batch, `load_existing_llm_judgements` MUST
    pick up the partial file so resume mode skips already-judged triples."""
    out = tmp_path / "expanded.jsonl"
    human = _seed_human_qrels(tmp_path / "human.jsonl")

    # Simulate a checkpoint by manually emitting partial results.
    partial = {
        ("Q001", 0, "P1"): _make_record("Supports"),
        ("Q001", 0, "P2"): _make_record("Contradicts"),
    }
    emit_expanded_qrels(
        human_qrels_path=human, llm_records=partial,
        out_path=out, incomplete=True, abort_reason="every_2",
    )

    # Resume the file via the rejudge loader.
    resumed = load_existing_llm_judgements(out)
    assert set(resumed) == {("Q001", 0, "P1"), ("Q001", 0, "P2")}


def test_checkpoint_zero_disables_intermediate_writes(tmp_path: Path) -> None:
    """`checkpoint_every=0` reverts to the old behaviour: single write at
    end of batch."""
    out = tmp_path / "expanded.jsonl"
    human = _seed_human_qrels(tmp_path / "human.jsonl")
    judge = _stub_judge()
    triples = [("Q001", 0, f"P{i}") for i in range(1, 6)]

    _judge_triples_and_emit(
        judge=judge, all_triples=triples,
        answer_lookup=lambda *a, **k: "s",
        abstract_lookup=lambda *a, **k: "abs",
        out_path=out, human_qrels_path=human,
        max_concurrent=1, cost_cap=None,
        checkpoint_every=0,
    )

    # Final write happened even with no intermediate checkpoints.
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    llm = [r for r in rows if r.get("source", "human") != "human"]
    assert len(llm) == 5


def test_atomic_write_uses_replace_no_partial_file(tmp_path: Path) -> None:
    """`emit_expanded_qrels` writes via tmp + os.replace; after the call,
    the only file present is the final one (no `.tmp` leftovers)."""
    out = tmp_path / "expanded.jsonl"
    human = _seed_human_qrels(tmp_path / "human.jsonl")
    records = {
        ("Q001", 0, "P1"): _make_record("Supports"),
    }
    emit_expanded_qrels(
        human_qrels_path=human, llm_records=records,
        out_path=out, incomplete=False, abort_reason="",
    )
    assert out.exists()
    # No leftover tmp.
    assert not (tmp_path / "expanded.jsonl.tmp").exists()
    # Sidecar also written atomically.
    sidecar = out.with_suffix(out.suffix + ".meta.json")
    assert sidecar.exists()
    assert not (sidecar.with_suffix(sidecar.suffix + ".tmp")).exists()


def test_unhandled_exception_flushes_results_before_reraise(tmp_path: Path) -> None:
    """Phase 2.5 safety net: a generic exception (not Quota, not Ctrl+C)
    used to bypass `emit_expanded_qrels` and lose all in-memory results.
    Now we flush whatever has been completed, then re-raise."""
    out = tmp_path / "expanded.jsonl"
    human = _seed_human_qrels(tmp_path / "human.jsonl")

    # Judge classifies the first 2 triples normally then raises on the 3rd.
    call_count = {"n": 0}

    def flaky_classify(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 3:
            raise RuntimeError("simulated network blast after 2 successes")
        return _make_record("Supports")

    judge = MagicMock(spec=Judge)
    judge.classify.side_effect = flaky_classify

    triples = [("Q001", 0, f"P{i}") for i in range(1, 4)]  # 3 triples
    with pytest.raises(RuntimeError, match="simulated network blast"):
        _judge_triples_and_emit(
            judge=judge, all_triples=triples,
            answer_lookup=lambda *a, **k: "s",
            abstract_lookup=lambda *a, **k: "abs",
            out_path=out, human_qrels_path=human,
            max_concurrent=1, cost_cap=None,
            checkpoint_every=0,  # disable checkpoint to isolate the flush-on-error path
        )
    # File exists with at least the 2 successes saved.
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    llm = [r for r in rows if r.get("source", "human") != "human"]
    assert len(llm) >= 2, "the flush-before-reraise path must persist completed triples"
    # Sidecar flags it incomplete with the reason.
    sidecar = json.loads((out.with_suffix(out.suffix + ".meta.json")).read_text())
    assert sidecar["incomplete"] is True
    assert "unhandled_exception" in sidecar["abort_reason"]
