"""Unit tests for retrieval.llm_rewrite (Phase 2 §12.10)."""

from __future__ import annotations

import pytest

from trec_biogen.retrieval.llm_rewrite import (
    REWRITE_COT_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    _system_prompt_for,
    parse_rewrite_response,
)


def test_parse_strict_json() -> None:
    out = parse_rewrite_response(
        '{"queries": ["metformin HbA1c", "type 2 diabetes glycemic", "biguanide outcomes"]}',
        n_expected=3,
    )
    assert out == ["metformin HbA1c", "type 2 diabetes glycemic", "biguanide outcomes"]


def test_parse_recovers_from_surrounding_text() -> None:
    """A model that emits explanatory text around the JSON still parses."""
    out = parse_rewrite_response(
        'Sure, here are 3 queries:\n{"queries": ["a", "b", "c"]}\nHope this helps.',
        n_expected=3,
    )
    assert out == ["a", "b", "c"]


def test_parse_truncates_to_n_expected() -> None:
    out = parse_rewrite_response(
        '{"queries": ["q1", "q2", "q3", "q4", "q5"]}', n_expected=3,
    )
    assert out == ["q1", "q2", "q3"]


def test_parse_drops_empty_and_whitespace_strings() -> None:
    out = parse_rewrite_response(
        '{"queries": ["good query", "", "   ", "another one"]}', n_expected=4,
    )
    assert out == ["good query", "another one"]


def test_parse_rejects_non_string_entries() -> None:
    out = parse_rewrite_response(
        '{"queries": ["valid", 42, null, "also valid"]}', n_expected=4,
    )
    assert out == ["valid", "also valid"]


def test_parse_empty_string_returns_empty() -> None:
    assert parse_rewrite_response("", n_expected=3) == []


def test_parse_invalid_json_returns_empty() -> None:
    assert parse_rewrite_response("not json at all", n_expected=3) == []


def test_parse_wrong_shape_returns_empty() -> None:
    """Object that doesn't have a 'queries' list returns empty."""
    assert parse_rewrite_response('{"results": ["a", "b"]}', n_expected=3) == []


def test_parse_queries_not_list_returns_empty() -> None:
    assert parse_rewrite_response('{"queries": "single string"}', n_expected=3) == []


def test_system_prompt_mentions_pubmed_and_bm25() -> None:
    """The prompt should be transparent about what the rewrites are for."""
    assert "PubMed" in REWRITE_SYSTEM_PROMPT
    assert "BM25" in REWRITE_SYSTEM_PROMPT
    assert "VARIED" in REWRITE_SYSTEM_PROMPT


def test_cot_prompt_demands_reasoning_step() -> None:
    """The CoT variant should require an explicit pre-query reasoning step."""
    assert "STEP 1" in REWRITE_COT_SYSTEM_PROMPT
    assert "STEP 2" in REWRITE_COT_SYSTEM_PROMPT
    assert "reasoning" in REWRITE_COT_SYSTEM_PROMPT
    assert "MeSH" in REWRITE_COT_SYSTEM_PROMPT
    assert "synonyms" in REWRITE_COT_SYSTEM_PROMPT


def test_system_prompt_for_dispatch() -> None:
    assert _system_prompt_for("strict") == REWRITE_SYSTEM_PROMPT
    assert _system_prompt_for("cot") == REWRITE_COT_SYSTEM_PROMPT
    # Unknown mode → defaults to strict (defensive default).
    assert _system_prompt_for("unknown") == REWRITE_SYSTEM_PROMPT


def test_parser_ignores_reasoning_field() -> None:
    """The parser only reads `queries`; a CoT response's `reasoning` field is silently dropped."""
    cot_response = (
        '{"reasoning": "The claim is about metformin and HbA1c. '
        'Key entities: metformin (biguanide), T2DM, HbA1c. Variants: '
        'brand name Glucophage, MeSH term Diabetes Mellitus Type 2.", '
        '"queries": ["metformin HbA1c reduction", "Glucophage T2DM", "biguanide glycemic control"]}'
    )
    out = parse_rewrite_response(cot_response, n_expected=3)
    assert out == ["metformin HbA1c reduction", "Glucophage T2DM", "biguanide glycemic control"]


def test_hydra_compose_bm25_llm_rewrite() -> None:
    """Compose the new variant config and verify the override resolves."""
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="run/phase2_bm25_llm_rewrite")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["retrieval"]["flavour"] == "bm25_llm_rewrite"
    assert resolved["retrieval"]["llm_rewrite"]["backend"] == "openai-mini"
    assert resolved["retrieval"]["llm_rewrite"]["prompt"] == "cot"
    assert resolved["retrieval"]["llm_rewrite"]["n_variants"] == 3
    assert resolved["retrieval"]["llm_rewrite"]["include_original"] is True
    assert resolved["phase2_variant"] == "bm25_llm_rewrite"
