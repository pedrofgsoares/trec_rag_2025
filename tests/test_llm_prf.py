"""Tests for the LLM-filtered RM3 retrieval helper (Phase 2 §12.7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from trec_biogen.retrieval.llm_prf import (
    RELEVANCE_FILTER_PROMPT,
    LLMRelevanceFilter,
    build_expanded_query,
    manual_rm3_terms,
)


def test_relevance_filter_prompt_is_binary() -> None:
    assert "relevant" in RELEVANCE_FILTER_PROMPT
    assert "JSON" in RELEVANCE_FILTER_PROMPT
    # Must NOT mention chain-of-thought / reasoning — the filter is binary.
    assert "reasoning" not in RELEVANCE_FILTER_PROMPT.lower()


def test_build_expanded_query_appends_terms() -> None:
    q = build_expanded_query(
        "blood pressure lowering risks",
        [("hypertension", 0.5), ("diastolic", 0.3), ("j-curve", 0.2)],
    )
    assert q.startswith("blood pressure lowering risks ")
    assert "hypertension" in q
    assert "diastolic" in q
    assert "j-curve" in q


def test_build_expanded_query_empty_expansion_returns_original() -> None:
    assert build_expanded_query("query", []) == "query"


def test_manual_rm3_terms_excludes_original_query_terms() -> None:
    """Mock reader returning fixed term vectors; verify exclude list works."""
    class _Reader:
        DOCS = {
            "P1": {"hypertension": 5, "diastolic": 3, "metformin": 1, "blood": 2},
            "P2": {"hypertension": 4, "j-curve": 2, "trial": 3, "pressure": 1},
        }
        def get_document_vector(self, pmid: str) -> dict[str, int]:
            return self.DOCS.get(pmid, {})

    reader = _Reader()
    terms = manual_rm3_terms(
        reader, ["P1", "P2"], fb_terms=5, exclude=["blood", "pressure"],
    )
    pmids = [t for t, _ in terms]
    assert "blood" not in pmids
    assert "pressure" not in pmids
    # Highest weight should be "hypertension" (appears in both with high tf).
    assert pmids[0] == "hypertension"


def test_manual_rm3_terms_empty_inputs() -> None:
    class _EmptyReader:
        def get_document_vector(self, pmid: str) -> dict[str, int]:
            return {}

    assert manual_rm3_terms(_EmptyReader(), [], fb_terms=5) == []
    assert manual_rm3_terms(_EmptyReader(), ["X"], fb_terms=5) == []


def test_manual_rm3_terms_orders_by_weight() -> None:
    class _Reader:
        def get_document_vector(self, pmid: str) -> dict[str, int]:
            # Single doc, three terms with different frequencies.
            return {"a": 10, "b": 5, "c": 2}

    reader = _Reader()
    terms = manual_rm3_terms(reader, ["P1"], fb_terms=3)
    assert [t for t, _ in terms] == ["a", "b", "c"]
    # Weights should be P(t|d) for a single doc: tf/doc_len.
    a_weight = next(w for t, w in terms if t == "a")
    assert a_weight == pytest.approx(10 / 17)


def test_phase2_bm25_rm3_llm_filtered_config(tmp_path: Path) -> None:
    """Hydra compose check for the new variant config."""
    from hydra import compose, initialize_config_dir

    repo = Path(__file__).resolve().parent.parent
    with initialize_config_dir(version_base=None, config_dir=str(repo / "configs")):
        cfg = compose(config_name="run/phase2_bm25_rm3_llm_filtered")
    assert cfg.phase2_variant == "bm25_rm3_llm_filtered"
    assert cfg.retrieval.flavour == "bm25_rm3_llm_filtered"
    assert cfg.retrieval.support_k == 100
    assert cfg.retrieval.contradict_k == 1000
    assert cfg.retrieval.llm_prf.backend == "openai-mini"
    assert cfg.retrieval.llm_prf.initial_k == 30
    assert cfg.retrieval.llm_prf.fb_terms == 10


def test_llm_filter_records_stats(monkeypatch) -> None:
    """Stub backend; verify the filter aggregates stats correctly."""
    import httpx
    from trec_biogen.judge.backends import HTTPBackend

    def handler(request: httpx.Request) -> httpx.Response:
        body = {
            "choices": [{"message": {"content": '{"relevant": true}'}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5},
        }
        return httpx.Response(200, json=body)

    monkeypatch.setenv("FAKE", "test")
    backend = HTTPBackend(
        name="openai-gpt-4o-mini", model="gpt-4o-mini",
        base_url="https://api.example.invalid/v1", api_key_env="FAKE",
        transport=httpx.MockTransport(handler),
    )
    flt = LLMRelevanceFilter(backend, max_concurrent=2)
    decisions = flt.filter_many(
        "sentence", [("P1", "abstract one"), ("P2", "abstract two")],
    )
    assert len(decisions) == 2
    assert all(d.relevant for d in decisions)
    stats = flt.stats
    assert stats["calls"] == 2
    assert stats["input_tokens"] == 200
    assert stats["output_tokens"] == 10
    assert stats["cost_usd"] > 0
