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


# ---------------------------------------------------------------------------
# Phase 2.6 §2.6 — Qwen2.5-72B backend smoke (pivot from Mixtral per D1)
# ---------------------------------------------------------------------------


def test_qwen_uses_per_backend_filename() -> None:
    """`expand-pool --backend qwen` MUST default to its own qrels file so
    a qwen rejudge does not overwrite the canonical mini-cot expanded
    qrels (the byte-for-byte Phase 2.5 invariant)."""
    out = _default_expanded_out_for_backend("qwen")
    assert out == Path("data/qrels/biogen2025_taskA_qrels_expanded_qwen.jsonl"), (
        "qwen must default to its own per-backend filename so it doesn't "
        "overwrite the canonical mini-cot or any other backend's pool"
    )


def test_qwen_backend_resolves_and_pins_model() -> None:
    """The `qwen` registry entry resolves to the Qwen2.5-72B-Instruct model
    via the HF Inference Providers router. Pivot from D1's original
    Mixtral-8x7B (HF removed Mistral family from the chat-routable roster)."""
    from trec_biogen.judge.backends import HFQwen72B, make_backend
    import os
    saved = os.environ.pop("HF_TOKEN", None)
    try:
        os.environ["HF_TOKEN"] = "test"
        b = make_backend("qwen", prompt_mode="cot")
        assert isinstance(b, HFQwen72B)
        assert b.name == "hf-qwen-2.5-72b"
        assert "Qwen2.5-72B-Instruct" in b._model
        assert b._base_url == "https://router.huggingface.co/v1"
    finally:
        if saved is not None:
            os.environ["HF_TOKEN"] = saved
        else:
            os.environ.pop("HF_TOKEN", None)


def test_qwen_backend_honours_provider_pin(monkeypatch) -> None:
    """Same `HF_PROVIDER` env-var convention as `hf-llama` — appended to
    the model id."""
    from trec_biogen.judge.backends import HFQwen72B
    monkeypatch.setenv("HF_PROVIDER", "openrouter")
    monkeypatch.setenv("HF_TOKEN", "test")
    b = HFQwen72B(prompt_mode="cot")
    assert b._model.endswith(":openrouter")
