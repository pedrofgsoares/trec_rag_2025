"""Tests for the Phase 2 ablation variants (§§4–8).

Covers:

* ``passthrough_rerank`` — phase2_no_rerank (§4)
* ``select(exclude_existing=False)`` — phase2_allow_existing (§6)
* ``BM25Index._set_rm3`` toggle — phase2_bm25_rm3 (§8)
* ``_t5_label_token_ids`` — phase2_scifive_large (§7)
* Hydra composition for every new run config:
  phase2_{no_rerank, no_negex, allow_existing, bm25_rm3, scifive_large}
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from trec_biogen.io.topics import Topic
from trec_biogen.nli.stance import _t5_label_token_ids
from trec_biogen.pipeline.selection import SelectionConfig, select
from trec_biogen.rerank.cross_encoder import passthrough_rerank
from trec_biogen.retrieval.bm25 import BM25Index


# ---------------------------------------------------------------------------
# §4 passthrough_rerank
# ---------------------------------------------------------------------------


def _fake_retrieval_parquet(path: Path) -> Path:
    pl.DataFrame(
        {
            "qa_id": ["Q1"] * 5 + ["Q2"] * 5,
            "sentence_id": [0] * 5 + [0] * 5,
            "sentence_text": ["sent A"] * 5 + ["sent B"] * 5,
            "query": ["q A"] * 5 + ["q B"] * 5,
            "candidate_pmid": [f"P{i}" for i in range(1, 11)],
            "rank": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "bm25_score": [50.0, 40.0, 30.0, 20.0, 10.0, 55.0, 45.0, 35.0, 25.0, 15.0],
        }
    ).write_parquet(path)
    return path


def test_passthrough_rerank_keeps_top_k_per_cell(tmp_path: Path) -> None:
    src = _fake_retrieval_parquet(tmp_path / "retrieval.parquet")
    out = passthrough_rerank(src, out_path=tmp_path / "rerank.parquet", top_k=3)
    df = pl.read_parquet(out)
    assert set(df.columns) >= {
        "qa_id", "sentence_id", "sentence_text",
        "candidate_pmid", "ce_score", "rank_after_rerank",
    }
    # Top-3 by rank per (qa_id, sentence_id) — 6 rows total.
    assert df.shape[0] == 6
    q1 = df.filter(pl.col("qa_id") == "Q1").sort("rank_after_rerank")
    assert q1["candidate_pmid"].to_list() == ["P1", "P2", "P3"]
    # ce_score is filled in from bm25_score so downstream sorts work.
    assert q1["ce_score"].to_list() == [50.0, 40.0, 30.0]


def test_passthrough_rerank_compatible_with_score_support_schema(tmp_path: Path) -> None:
    """The columns ``score_support`` reads must be present after passthrough."""
    src = _fake_retrieval_parquet(tmp_path / "retrieval.parquet")
    out = passthrough_rerank(src, out_path=tmp_path / "rerank.parquet", top_k=5)
    df = pl.read_parquet(out)
    # score_support consumes candidate_pmid + sentence_text per the
    # phase-2 design contract; both must be present.
    assert "candidate_pmid" in df.columns
    assert "sentence_text" in df.columns


# ---------------------------------------------------------------------------
# §6 exclude_existing toggle
# ---------------------------------------------------------------------------


def _selection_fixtures(tmp_path: Path) -> tuple[Path, Path, list[Topic]]:
    """Two cells with a clear existing-citation candidate; selection
    should drop or keep depending on ``exclude_existing``."""
    sup = pl.DataFrame(
        {
            "qa_id": ["Q1", "Q1", "Q1"],
            "sentence_id": [0, 0, 0],
            "candidate_pmid": ["EXISTING", "NEW1", "NEW2"],
            "entailment_prob": [0.95, 0.85, 0.75],
        }
    )
    sup_path = tmp_path / "nli_support.parquet"
    sup.write_parquet(sup_path)

    con = pl.DataFrame(
        {
            "qa_id": ["Q1"],
            "sentence_id": [0],
            "candidate_pmid": ["NEW3"],
            "contradict_score": [0.9],
            "bm25_rank": [1],
        }
    )
    con_path = tmp_path / "nli_contradict.parquet"
    con.write_parquet(con_path)

    topics = [Topic(
        qa_id="Q1",
        question="Q?",
        sentences=["only sentence"],
        existing_per_sentence=[{"EXISTING"}],
    )]
    return sup_path, con_path, topics


def test_select_drops_existing_by_default(tmp_path: Path) -> None:
    sup_path, con_path, topics = _selection_fixtures(tmp_path)
    result = select(
        nli_support=sup_path,
        nli_contradict=con_path,
        topics=topics,
        config=SelectionConfig(tau_sup=0.5, tau_con=0.5, cap=3),
    )
    assert "EXISTING" not in result["Q1"][0]["support"]
    assert result["Q1"][0]["support"] == ["NEW1", "NEW2"]


def test_select_keeps_existing_when_disabled(tmp_path: Path) -> None:
    sup_path, con_path, topics = _selection_fixtures(tmp_path)
    result = select(
        nli_support=sup_path,
        nli_contradict=con_path,
        topics=topics,
        config=SelectionConfig(tau_sup=0.5, tau_con=0.5, cap=3, exclude_existing=False),
    )
    # The highest-scoring support PMID — EXISTING — is now kept.
    assert "EXISTING" in result["Q1"][0]["support"]
    assert result["Q1"][0]["support"][0] == "EXISTING"


# ---------------------------------------------------------------------------
# §§4.1 / 5.1 / 6.1 — Hydra config composition
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = str(REPO_ROOT / "configs")


@pytest.fixture
def hydra_compose():
    with initialize_config_dir(version_base=None, config_dir=CONFIGS_DIR):
        yield compose


def test_phase2_no_rerank_config_sets_rerank_null(hydra_compose) -> None:
    cfg = hydra_compose(config_name="run/phase2_no_rerank")
    assert cfg.phase2_variant == "no_rerank"
    assert cfg.run.label == "phase2_no_rerank"
    assert OmegaConf.is_missing(cfg, "rerank") is False
    assert cfg.rerank is None
    # Defaults from phase1_baseline still in effect:
    assert cfg.retrieval.support_k == 100 or cfg.retrieval.support_k > 0


def test_phase2_no_negex_config_sets_contradict_negex_false(hydra_compose) -> None:
    cfg = hydra_compose(config_name="run/phase2_no_negex")
    assert cfg.phase2_variant == "no_negex"
    assert cfg.nli.contradict.negex is False
    # Support side untouched.
    assert cfg.nli.support.model  # non-empty


def test_phase2_allow_existing_config_sets_selection_exclude_false(hydra_compose) -> None:
    cfg = hydra_compose(config_name="run/phase2_allow_existing")
    assert cfg.phase2_variant == "allow_existing"
    assert cfg.selection.exclude_existing is False
    assert cfg.selection.tau_sup == 0.5  # default carried over


def test_phase1_baseline_config_keeps_default_rerank(hydra_compose) -> None:
    """Sanity: the baseline config — the parent — is unaffected by the new
    variants. ``rerank`` is a dict, ``negex`` defaults true."""
    cfg = hydra_compose(config_name="run/phase1_baseline")
    assert cfg.rerank is not None
    assert cfg.rerank.model
    assert cfg.nli.contradict.get("negex", True) is True
    # Phase 2 §7: the default contradict NLI head is DeBERTa, not T5.
    assert cfg.nli.contradict.get("type", "deberta") == "deberta"


# ---------------------------------------------------------------------------
# §8 BM25 RM3 toggle
# ---------------------------------------------------------------------------


def _build_mocked_bm25(rm3_params: tuple[int, int, float] = (10, 10, 0.5)) -> tuple[BM25Index, MagicMock]:
    """Construct a BM25Index without touching the JVM / Lucene index."""
    bm = BM25Index.__new__(BM25Index)
    bm.index_dir = Path("/tmp/fake-index")
    mock = MagicMock()
    mock.search.return_value = []
    bm._searcher = mock
    bm._rm3_params = rm3_params
    bm._rm3_active = False
    return bm, mock


def test_search_default_does_not_set_rm3() -> None:
    bm, mock = _build_mocked_bm25()
    bm.search("q", k=10)
    mock.search.assert_called_once_with("q", k=10)
    mock.set_rm3.assert_not_called()
    mock.unset_rm3.assert_not_called()


def test_search_with_rm3_true_sets_rm3_once() -> None:
    bm, mock = _build_mocked_bm25(rm3_params=(12, 8, 0.3))
    bm.search("q1", k=5, rm3=True)
    mock.set_rm3.assert_called_once_with(
        fb_terms=12, fb_docs=8, original_query_weight=0.3,
    )
    # Second call with same rm3=True must not re-toggle.
    bm.search("q2", k=5, rm3=True)
    assert mock.set_rm3.call_count == 1


def test_search_back_to_no_rm3_unsets_once() -> None:
    bm, mock = _build_mocked_bm25()
    bm.search("q1", k=5, rm3=True)
    bm.search("q2", k=5, rm3=False)
    mock.unset_rm3.assert_called_once()
    # And toggling back enables again.
    bm.search("q3", k=5, rm3=True)
    assert mock.set_rm3.call_count == 2


def test_phase2_bm25_rm3_config(hydra_compose) -> None:
    cfg = hydra_compose(config_name="run/phase2_bm25_rm3")
    assert cfg.phase2_variant == "bm25_rm3"
    assert cfg.retrieval.rm3.enabled is True
    assert cfg.retrieval.rm3.fb_terms == 10
    assert cfg.retrieval.rm3.fb_docs == 10
    assert cfg.retrieval.rm3.original_query_weight == 0.5
    # k1/b inherited from bm25_rm3.yaml (not bm25_default).
    assert cfg.retrieval.support_k == 100
    assert cfg.retrieval.contradict_k == 1000


# ---------------------------------------------------------------------------
# §7 SciFive T5 contradict
# ---------------------------------------------------------------------------


def test_t5_label_token_ids_uses_first_token_per_label() -> None:
    """Constrained decoding requires one token ID per label; the helper
    must call ``encode(label, add_special_tokens=False)`` and pick [0]."""
    mock_tok = MagicMock()
    mock_tok.encode.side_effect = lambda label, add_special_tokens: {
        "entailment":    [101, 5],
        "neutral":       [202],
        "contradiction": [303, 7, 9],
    }[label]
    ids = _t5_label_token_ids(mock_tok)
    assert ids == {"entailment": 101, "neutral": 202, "contradiction": 303}
    # Must be called with add_special_tokens=False (no <s>/<bos> noise).
    for call in mock_tok.encode.call_args_list:
        assert call.kwargs.get("add_special_tokens") is False


def test_t5_label_token_ids_raises_on_empty_tokenization() -> None:
    mock_tok = MagicMock()
    mock_tok.encode.return_value = []  # pathological tokenizer
    with pytest.raises(ValueError):
        _t5_label_token_ids(mock_tok)


def test_phase2_scifive_large_config(hydra_compose) -> None:
    cfg = hydra_compose(config_name="run/phase2_scifive_large")
    assert cfg.phase2_variant == "scifive_large"
    # Contradict path swapped to SciFive seq2seq.
    assert cfg.nli.contradict.type == "t5"
    assert "scifive" in cfg.nli.contradict.model.lower()
    assert cfg.nli.contradict.fp16 is True
    assert cfg.nli.contradict.chunk_size == 4
    # Support path untouched (still DeBERTa).
    assert "deberta" in cfg.nli.support.model.lower()
    # NegEx pre-filter still on (decoupled from the §5 ablation).
    assert cfg.nli.contradict.negex is True


# ---------------------------------------------------------------------------
# §9 hybrid retrieval config + RRF wiring
# ---------------------------------------------------------------------------


def test_phase2_hybrid_config_selects_rrf_flavour(hydra_compose) -> None:
    cfg = hydra_compose(config_name="run/phase2_hybrid")
    assert cfg.phase2_variant == "hybrid"
    assert cfg.retrieval.flavour == "hybrid_rrf"
    assert cfg.retrieval.rrf.k == 60
    assert cfg.retrieval.dense.index_dir == "data/indexes/medcpt_5m"
    assert cfg.retrieval.dense.query_model == "ncbi/MedCPT-Query-Encoder"
    # Inherited from hybrid_rrf retrieval config:
    assert cfg.retrieval.support_k == 100
    assert cfg.retrieval.contradict_k == 1000


def test_phase1_baseline_has_no_flavour_field(hydra_compose) -> None:
    """Backwards compat: the default BM25 path must work without ``flavour``
    — the orchestrator falls back to ``"bm25"`` when the key is absent."""
    cfg = hydra_compose(config_name="run/phase1_baseline")
    assert cfg.retrieval.get("flavour", "bm25") == "bm25"
