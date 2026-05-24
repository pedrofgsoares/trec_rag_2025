"""Phase 2.5 §2.4 — Two-judge intersection pool emitter.

The contract under test (per the llm-judge delta spec):
1. Human positives copied bit-identical from records_a.
2. Supports positives passed through from records_a (no intersection
   on the Supports class — §12.4 shows them robust to judge choice).
3. Contradicts positives kept iff (qa_id, sentence_id, pmid, class)
   matches a positive in records_b.
4. Sidecar `<out>.meta.json` carries SHA256 of both inputs, timestamp,
   per-class before/after counts, dropped % for contradicts, and the
   `incomplete` flag OR'd from both inputs' sidecars.
"""

from __future__ import annotations

import json
from pathlib import Path

import orjson

from trec_biogen.judge.intersection import emit_intersection_pool


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    with path.open("wb") as fh:
        for r in rows:
            fh.write(orjson.dumps(r) + b"\n")
    return path


def _human_row(qa_id: str, sid: int, pmid: str, cls: str) -> dict:
    return {"qa_id": qa_id, "sentence_id": sid, "pmid": pmid,
            "class": cls, "relevance": 1, "source": "human"}


def _llm_row(qa_id: str, sid: int, pmid: str, cls: str,
             source: str, confidence: float = 0.85) -> dict:
    return {"qa_id": qa_id, "sentence_id": sid, "pmid": pmid,
            "class": cls, "relevance": 1, "source": source,
            "confidence": confidence}


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("rb") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                out.append(orjson.loads(raw))
    return out


def test_human_records_pass_through_verbatim(tmp_path: Path) -> None:
    human = _write_jsonl(tmp_path / "human.jsonl", [
        _human_row("Q1", 0, "P1", "support"),
        _human_row("Q1", 0, "P2", "contradict"),
    ])
    a = _write_jsonl(tmp_path / "a.jsonl", [
        _human_row("Q1", 0, "P1", "support"),
        _human_row("Q1", 0, "P2", "contradict"),
        _llm_row("Q1", 0, "P3", "support", "llm-openai-mini"),
    ])
    b = _write_jsonl(tmp_path / "b.jsonl", [
        _human_row("Q1", 0, "P1", "support"),
        _human_row("Q1", 0, "P2", "contradict"),
    ])
    out = tmp_path / "intersection.jsonl"
    emit_intersection_pool(a, b, human_qrels_path=human, out_path=out)
    rows = _read_jsonl(out)
    human_rows = [r for r in rows if r.get("source") == "human"]
    # Both human rows preserved, bit-identical to what was in records_a.
    assert {(r["pmid"], r["class"]) for r in human_rows} == {("P1", "support"), ("P2", "contradict")}


def test_supports_pass_through_from_records_a(tmp_path: Path) -> None:
    """Supports are NOT intersected per §D2; they come from A only."""
    human = _write_jsonl(tmp_path / "human.jsonl", [])
    a = _write_jsonl(tmp_path / "a.jsonl", [
        _llm_row("Q1", 0, "P1", "support", "llm-openai-mini"),
        _llm_row("Q1", 0, "P2", "support", "llm-openai-mini"),
    ])
    # B has no Supports at all — intersection on contradicts only, so
    # Supports MUST still come through.
    b = _write_jsonl(tmp_path / "b.jsonl", [])
    out = tmp_path / "intersection.jsonl"
    emit_intersection_pool(a, b, human_qrels_path=human, out_path=out)
    rows = _read_jsonl(out)
    support_pmids = {r["pmid"] for r in rows if r["class"] == "support"}
    assert support_pmids == {"P1", "P2"}


def test_contradicts_intersect_on_qa_sid_pmid_class(tmp_path: Path) -> None:
    human = _write_jsonl(tmp_path / "human.jsonl", [])
    a = _write_jsonl(tmp_path / "a.jsonl", [
        _llm_row("Q1", 0, "P1", "contradict", "llm-openai-mini"),
        _llm_row("Q1", 0, "P2", "contradict", "llm-openai-mini"),
        _llm_row("Q1", 0, "P3", "contradict", "llm-openai-mini"),
    ])
    b = _write_jsonl(tmp_path / "b.jsonl", [
        _llm_row("Q1", 0, "P1", "contradict", "llm-together"),
        # P2 missing in B
        _llm_row("Q1", 0, "P3", "contradict", "llm-together"),
        # P4 in B only — should not appear in output (A didn't agree)
        _llm_row("Q1", 0, "P4", "contradict", "llm-together"),
    ])
    out = tmp_path / "intersection.jsonl"
    emit_intersection_pool(a, b, human_qrels_path=human, out_path=out)
    rows = _read_jsonl(out)
    contradict_pmids = {r["pmid"] for r in rows if r["class"] == "contradict"}
    assert contradict_pmids == {"P1", "P3"}


def test_contradict_in_a_supports_in_b_does_not_intersect(tmp_path: Path) -> None:
    """The class is part of the intersection key — A:Contradict ∩ B:Support
    does NOT count as a contradicts intersection."""
    human = _write_jsonl(tmp_path / "human.jsonl", [])
    a = _write_jsonl(tmp_path / "a.jsonl", [
        _llm_row("Q1", 0, "P1", "contradict", "llm-openai-mini"),
    ])
    b = _write_jsonl(tmp_path / "b.jsonl", [
        _llm_row("Q1", 0, "P1", "support", "llm-together"),
    ])
    out = tmp_path / "intersection.jsonl"
    emit_intersection_pool(a, b, human_qrels_path=human, out_path=out)
    rows = _read_jsonl(out)
    assert not [r for r in rows if r["class"] == "contradict"]


def test_intersection_records_carry_intersection_source_tag(tmp_path: Path) -> None:
    human = _write_jsonl(tmp_path / "human.jsonl", [])
    a = _write_jsonl(tmp_path / "a.jsonl", [
        _llm_row("Q1", 0, "P1", "contradict", "llm-openai-mini", confidence=0.9),
    ])
    b = _write_jsonl(tmp_path / "b.jsonl", [
        _llm_row("Q1", 0, "P1", "contradict", "llm-together", confidence=0.7),
    ])
    out = tmp_path / "intersection.jsonl"
    emit_intersection_pool(a, b, human_qrels_path=human, out_path=out)
    rows = [r for r in _read_jsonl(out) if r["class"] == "contradict"]
    assert len(rows) == 1
    assert rows[0]["source"] == "llm-intersection"
    # Other fields preserved from A.
    assert rows[0]["confidence"] == 0.9


def test_sidecar_metadata_is_exhaustive(tmp_path: Path) -> None:
    human = _write_jsonl(tmp_path / "human.jsonl", [
        _human_row("Q1", 0, "PH", "support"),
    ])
    a = _write_jsonl(tmp_path / "a.jsonl", [
        _human_row("Q1", 0, "PH", "support"),
        _llm_row("Q1", 0, "P1", "contradict", "llm-openai-mini"),
        _llm_row("Q1", 0, "P2", "contradict", "llm-openai-mini"),
        _llm_row("Q1", 0, "P3", "support", "llm-openai-mini"),
    ])
    b = _write_jsonl(tmp_path / "b.jsonl", [
        _llm_row("Q1", 0, "P1", "contradict", "llm-together"),
    ])
    out = tmp_path / "intersection.jsonl"
    emit_intersection_pool(a, b, human_qrels_path=human, out_path=out)
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text())
    assert set(meta) >= {
        "records_a_sha256", "records_b_sha256", "human_qrels_sha256",
        "intersection_rule", "before_intersection", "after_intersection",
        "contradicts_dropped", "contradicts_dropped_pct",
        "incomplete", "timestamp_utc",
    }
    assert meta["before_intersection"]["contradict"] == 2
    assert meta["after_intersection"]["contradict"] == 1
    assert meta["contradicts_dropped"] == 1
    assert meta["contradicts_dropped_pct"] == 0.5
    assert meta["after_intersection"]["support"] == 1  # from A
    assert meta["after_intersection"]["human"] == 1
    assert meta["incomplete"] is False


# ---------------------------------------------------------------------------
# Phase 2.6 §3 — N-way generalisation tests
# ---------------------------------------------------------------------------


def test_three_input_intersection_keeps_only_unanimous_contradicts(tmp_path: Path) -> None:
    """Contradict on (Q1,0,P1) is endorsed by all 3 → kept.
    Contradict on (Q1,0,P2) is endorsed by only 2/3 → dropped.
    Contradict on (Q1,0,P3) is endorsed by only 1/3 → dropped.
    Supports come from records_paths[0] (canonical).
    """
    human = _write_jsonl(tmp_path / "human.jsonl", [])
    a = _write_jsonl(tmp_path / "a.jsonl", [
        _llm_row("Q1", 0, "P1", "contradict", "llm-mini"),
        _llm_row("Q1", 0, "P2", "contradict", "llm-mini"),
        _llm_row("Q1", 0, "P3", "contradict", "llm-mini"),
        _llm_row("Q1", 0, "P4", "support", "llm-mini"),
    ])
    b = _write_jsonl(tmp_path / "b.jsonl", [
        _llm_row("Q1", 0, "P1", "contradict", "llm-llama"),
        _llm_row("Q1", 0, "P2", "contradict", "llm-llama"),
    ])
    c = _write_jsonl(tmp_path / "c.jsonl", [
        _llm_row("Q1", 0, "P1", "contradict", "llm-qwen"),
    ])
    out = tmp_path / "intersection_3way.jsonl"
    meta = emit_intersection_pool([a, b, c], human_qrels_path=human, out_path=out)
    rows = _read_jsonl(out)
    contradict_pmids = {r["pmid"] for r in rows if r["class"] == "contradict"}
    support_pmids = {r["pmid"] for r in rows if r["class"] == "support"}
    assert contradict_pmids == {"P1"}, "only unanimously-endorsed contradicts survive"
    assert support_pmids == {"P4"}, "support from records_paths[0] passes through"
    assert meta["before_intersection"]["contradict"] == 3
    assert meta["after_intersection"]["contradict"] == 1
    # Pairwise intersection diagnostics are populated for N≥3.
    assert len(meta["pairwise_contradict_intersections"]) == 3  # (a∩b, a∩c, b∩c)


def test_supports_source_index_swaps_supports_input(tmp_path: Path) -> None:
    """With supports_source_index=1 the Supports come from records_paths[1]
    instead of records_paths[0]."""
    human = _write_jsonl(tmp_path / "human.jsonl", [])
    a = _write_jsonl(tmp_path / "a.jsonl", [
        _llm_row("Q1", 0, "P-from-A", "support", "llm-mini"),
        _llm_row("Q1", 0, "P1", "contradict", "llm-mini"),
    ])
    b = _write_jsonl(tmp_path / "b.jsonl", [
        _llm_row("Q1", 0, "P-from-B", "support", "llm-llama"),
        _llm_row("Q1", 0, "P1", "contradict", "llm-llama"),
    ])
    out = tmp_path / "intersection.jsonl"
    emit_intersection_pool([a, b], human_qrels_path=human, out_path=out,
                           supports_source_index=1)
    rows = _read_jsonl(out)
    support_pmids = {r["pmid"] for r in rows if r["class"] == "support"}
    assert support_pmids == {"P-from-B"}, (
        "with supports_source_index=1, supports come from records_paths[1]"
    )


def test_supports_source_index_out_of_range_raises(tmp_path: Path) -> None:
    """Out-of-range index must raise ValueError *before* any file I/O."""
    import pytest
    # Non-existent paths — if validation happens after file read, this would
    # blow up with FileNotFoundError instead of ValueError.
    a = tmp_path / "does_not_exist_a.jsonl"
    b = tmp_path / "does_not_exist_b.jsonl"
    human = tmp_path / "does_not_exist_human.jsonl"
    out = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match="supports_source_index"):
        emit_intersection_pool([a, b], human_qrels_path=human,
                               out_path=out, supports_source_index=5)
    with pytest.raises(ValueError, match="supports_source_index"):
        emit_intersection_pool([a, b], human_qrels_path=human,
                               out_path=out, supports_source_index=-1)


def test_incomplete_flag_propagates_from_any_of_n_inputs(tmp_path: Path) -> None:
    """With N=3, an incomplete sidecar on any one input flips the output."""
    human = _write_jsonl(tmp_path / "human.jsonl", [])
    a = _write_jsonl(tmp_path / "a.jsonl", [
        _llm_row("Q1", 0, "P1", "contradict", "llm-mini"),
    ])
    b = _write_jsonl(tmp_path / "b.jsonl", [
        _llm_row("Q1", 0, "P1", "contradict", "llm-llama"),
    ])
    c = _write_jsonl(tmp_path / "c.jsonl", [
        _llm_row("Q1", 0, "P1", "contradict", "llm-qwen"),
    ])
    # Mark C's sidecar incomplete.
    c_sidecar = c.with_suffix(c.suffix + ".meta.json")
    c_sidecar.write_text(json.dumps({"incomplete": True}), encoding="utf-8")

    out = tmp_path / "intersection_3way.jsonl"
    emit_intersection_pool([a, b, c], human_qrels_path=human, out_path=out)
    meta = json.loads((out.with_suffix(out.suffix + ".meta.json")).read_text())
    assert meta["incomplete"] is True


def test_two_input_call_reproduces_phase25_pool_byte_for_byte(tmp_path: Path) -> None:
    """Regenerating the Phase 2.5 two-judge intersection pool with the new
    list-form API must produce the exact same JSONL file as the archived
    one. This is the structural invariant: Phase 2.6's N-way generalisation
    must not silently shift any byte of the Phase 2.5 output.
    """
    import os
    repo_root = Path(__file__).resolve().parents[1]
    canonical = repo_root / "data/qrels/biogen2025_taskA_qrels_expanded.jsonl"
    llama = repo_root / "data/qrels/biogen2025_taskA_qrels_expanded_hf-llama.jsonl"
    human = repo_root / "data/qrels/biogen2025_taskA_qrels.jsonl"
    archived = repo_root / "data/qrels/biogen2025_taskA_qrels_intersection.jsonl"
    if not all(p.exists() for p in (canonical, llama, human, archived)):
        import pytest
        pytest.skip("Phase 2.5 archived inputs not available in this checkout")

    out = tmp_path / "intersection_regen.jsonl"
    # New list-form API.
    emit_intersection_pool([canonical, llama], human_qrels_path=human, out_path=out)
    assert out.read_bytes() == archived.read_bytes(), (
        "N-way generalisation broke the Phase 2.5 byte-for-byte invariant"
    )


def test_incomplete_flag_propagates_from_either_input(tmp_path: Path) -> None:
    human = _write_jsonl(tmp_path / "human.jsonl", [])
    a = _write_jsonl(tmp_path / "a.jsonl", [
        _llm_row("Q1", 0, "P1", "contradict", "llm-openai-mini"),
    ])
    b = _write_jsonl(tmp_path / "b.jsonl", [
        _llm_row("Q1", 0, "P1", "contradict", "llm-together"),
    ])
    # Mark B's sidecar incomplete.
    b_sidecar = b.with_suffix(b.suffix + ".meta.json")
    b_sidecar.write_text(json.dumps({"incomplete": True}), encoding="utf-8")

    out = tmp_path / "intersection.jsonl"
    emit_intersection_pool(a, b, human_qrels_path=human, out_path=out)
    meta = json.loads((out.with_suffix(out.suffix + ".meta.json")).read_text())
    assert meta["incomplete"] is True
