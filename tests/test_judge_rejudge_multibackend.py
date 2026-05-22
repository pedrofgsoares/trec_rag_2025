"""Phase 2.5 §1.5 — Multi-backend rejudge file independence.

The contract under test (per the llm-judge delta spec):
* `--out` omitted on `expand-pool` resolves to the per-backend default
  via `_default_expanded_out_for_backend(name)`.
* `openai-mini` → canonical `biogen2025_taskA_qrels_expanded.jsonl`
  (back-compat).
* Every other backend (e.g. `together`, `openai`) → `expanded_<name>.jsonl`.
* A second-backend rejudge with `--out` omitted MUST NOT touch the
  canonical file — the two backends' artefacts must be independent.
"""

from __future__ import annotations

from pathlib import Path

from trec_biogen.judge.rejudge import _default_expanded_out_for_backend


def test_openai_mini_uses_canonical_filename() -> None:
    out = _default_expanded_out_for_backend("openai-mini")
    assert out == Path("data/qrels/biogen2025_taskA_qrels_expanded.jsonl"), (
        "openai-mini must keep writing to the canonical filename so prior "
        "Phase 2 invocations remain bit-identical"
    )


def test_together_uses_per_backend_filename() -> None:
    out = _default_expanded_out_for_backend("together")
    assert out == Path("data/qrels/biogen2025_taskA_qrels_expanded_together.jsonl")


def test_openai_4o_uses_per_backend_filename() -> None:
    out = _default_expanded_out_for_backend("openai")
    assert out == Path("data/qrels/biogen2025_taskA_qrels_expanded_openai.jsonl")


def test_unknown_backend_still_gets_per_backend_filename() -> None:
    # Defensive: even an unregistered name resolves to a non-canonical path.
    out = _default_expanded_out_for_backend("some-future-backend")
    assert (
        out == Path("data/qrels/biogen2025_taskA_qrels_expanded_some-future-backend.jsonl")
    )


def test_two_backends_resolve_to_distinct_paths() -> None:
    a = _default_expanded_out_for_backend("openai-mini")
    b = _default_expanded_out_for_backend("together")
    assert a != b, (
        "openai-mini and together must default to different filenames so a "
        "second-backend rejudge cannot silently overwrite the canonical pool"
    )


def test_hf_llama_uses_per_backend_filename() -> None:
    out = _default_expanded_out_for_backend("hf-llama")
    assert out == Path("data/qrels/biogen2025_taskA_qrels_expanded_hf-llama.jsonl")


def test_hf_llama_backend_is_registered() -> None:
    """Phase 2.5 §1: `hf-llama` must be selectable via --backend so the
    Together fallback path is actually wired through to the CLI."""
    from trec_biogen.judge.backends import BACKEND_REGISTRY, HFLlama70B
    assert "hf-llama" in BACKEND_REGISTRY
    assert BACKEND_REGISTRY["hf-llama"] is HFLlama70B


def test_hf_llama_backend_uses_hf_router_endpoint() -> None:
    """Make sure we're hitting the HF router, not some other Llama host."""
    import os
    from trec_biogen.judge.backends import HFLlama70B
    # Construction does not call the network; safe without HF_TOKEN.
    os.environ.setdefault("HF_TOKEN", "test")
    b = HFLlama70B(prompt_mode="cot")
    assert b._base_url == "https://router.huggingface.co/v1"
    assert b._api_key_env == "HF_TOKEN"
    assert "Llama-3.3-70B-Instruct" in b._model
    assert b.name == "hf-llama-3.3-70b"


def test_hf_llama_backend_honours_provider_pin(monkeypatch) -> None:
    """`HF_PROVIDER=sambanova` should be appended to the model id so HF
    routes to that specific provider rather than auto-routing."""
    from trec_biogen.judge.backends import HFLlama70B
    monkeypatch.setenv("HF_PROVIDER", "sambanova")
    monkeypatch.setenv("HF_TOKEN", "test")
    b = HFLlama70B(prompt_mode="cot")
    assert b._model.endswith(":sambanova")
