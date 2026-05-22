"""Robustness tests for the LLM-judge backend and rejudge resume path.

Three independent scenarios:

1. Transient HTTP 429 is retried with back-off — the second response
   wins and ``classify`` returns a normal record. Verifies the
   ``Retry-After`` header is honoured (we pin the back-off sleep to 0
   to keep the test fast).

2. HTTP 429 with ``insufficient_quota`` (or HTTP 402) raises
   :class:`QuotaExhausted` — the rejudge loop must distinguish this
   from a transient rate-limit.

3. The rejudge resume path: prior LLM rows in an existing expanded-qrels
   file are picked up by :func:`load_existing_llm_judgements` and the
   re-emitted file is consistent (no duplicates, no human row lost).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import orjson
import pytest

from trec_biogen.judge.backends import (
    HTTPBackend,
    JudgeRecord,
    QuotaExhausted,
)
from trec_biogen.judge.rejudge import (
    emit_expanded_qrels,
    load_existing_llm_judgements,
)


def _ok_response(label: str = "Supports", confidence: float = 0.9) -> httpx.Response:
    body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"label": label, "confidence": confidence}),
                }
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 8},
    }
    return httpx.Response(200, json=body)


def _build_backend(
    transport: httpx.BaseTransport, monkeypatch: pytest.MonkeyPatch,
) -> HTTPBackend:
    monkeypatch.setenv("FAKE_KEY", "test")
    monkeypatch.setattr("time.sleep", lambda *_: None)
    return HTTPBackend(
        name="openai-gpt-4o-mini",  # uses a real price row so cost > 0
        model="gpt-4o-mini",
        base_url="https://api.example.invalid/v1",
        api_key_env="FAKE_KEY",
        transport=transport,
    )


# ---------------------------------------------------------------------------
# 1. transient 429 -> retry succeeds
# ---------------------------------------------------------------------------


def test_cot_mode_sends_cot_system_and_higher_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CoT mode swaps the system prompt and bumps max_tokens. The model's
    extra ``reasoning`` field must be silently dropped by the parser."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        body = {
            "choices": [{
                "message": {"content": json.dumps({
                    "reasoning": "abstract reports nationwide VA training rollouts, implying institutional recommendation",
                    "label": "Supports",
                    "confidence": 0.88,
                })},
            }],
            "usage": {"prompt_tokens": 150, "completion_tokens": 80},
        }
        return httpx.Response(200, json=body)

    monkeypatch.setenv("FAKE_KEY", "test")
    monkeypatch.setattr("time.sleep", lambda *_: None)
    backend = HTTPBackend(
        name="openai-gpt-4o-mini", model="gpt-4o-mini",
        base_url="https://api.example.invalid/v1", api_key_env="FAKE_KEY",
        transport=httpx.MockTransport(handler),
        prompt_mode="cot",
    )
    rec = backend.classify("VA recommends PE/CPT", "nationwide training program")
    payload = captured["payload"]
    assert payload["max_tokens"] == 300  # CoT cap, not strict 80
    system_msg = payload["messages"][0]["content"]
    assert "reasoning" in system_msg  # CoT system asks for reasoning chain
    assert rec.label == "Supports"
    assert rec.confidence == pytest.approx(0.88)


def test_transient_429_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "0"},
                json={"error": {"code": "rate_limit_exceeded", "message": "slow down"}},
            )
        return _ok_response()

    backend = _build_backend(httpx.MockTransport(handler), monkeypatch)
    rec = backend.classify("sentence", "non-empty abstract")
    assert len(calls) == 2
    assert rec.label == "Supports"
    assert rec.cost_usd > 0


# ---------------------------------------------------------------------------
# 2. quota exhaustion -> QuotaExhausted
# ---------------------------------------------------------------------------


def test_429_insufficient_quota_raises_quota_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"code": "insufficient_quota", "message": "out of credit"}},
        )

    backend = _build_backend(httpx.MockTransport(handler), monkeypatch)
    with pytest.raises(QuotaExhausted):
        backend.classify("sentence", "non-empty abstract")


def test_402_raises_quota_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": {"message": "payment required"}})

    backend = _build_backend(httpx.MockTransport(handler), monkeypatch)
    with pytest.raises(QuotaExhausted):
        backend.classify("sentence", "non-empty abstract")


def test_transient_timeout_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A one-shot ``httpx.ReadTimeout`` is treated as transient and retried."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("read timed out", request=request)
        return _ok_response()

    backend = _build_backend(httpx.MockTransport(handler), monkeypatch)
    rec = backend.classify("sentence", "non-empty abstract")
    assert len(calls) == 2
    assert rec.label == "Supports"


def test_persistent_timeout_eventually_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the read keeps timing out past the retry budget, the exception
    surfaces — so the caller can decide what to do (validator aborts,
    rejudge moves on to the next future)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    backend = _build_backend(httpx.MockTransport(handler), monkeypatch)
    with pytest.raises(httpx.ReadTimeout):
        backend.classify("sentence", "non-empty abstract")


def test_persistent_429_eventually_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient 429 that never recovers must surface as an HTTPStatusError,
    not silently return — otherwise the caller can't distinguish failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "0"},
            json={"error": {"code": "rate_limit_exceeded"}},
        )

    backend = _build_backend(httpx.MockTransport(handler), monkeypatch)
    with pytest.raises(httpx.HTTPStatusError):
        backend.classify("sentence", "non-empty abstract")


def test_transient_503_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 2.5: a single HTTP 503 (Service Unavailable — Together's 70B
    endpoint returns this under load) must be retried with backoff, not
    propagate as a fatal error that kills the rejudge."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                503,
                headers={"retry-after": "0"},
                content=b"Service Unavailable",
            )
        return _ok_response()

    backend = _build_backend(httpx.MockTransport(handler), monkeypatch)
    rec = backend.classify("sentence", "non-empty abstract")
    assert len(calls) == 2
    assert rec.label == "Supports"


def test_persistent_503_eventually_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 503 that never recovers must surface — otherwise the caller can't
    know that the run is stuck on a wedged upstream."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, headers={"retry-after": "0"}, content=b"down")

    backend = _build_backend(httpx.MockTransport(handler), monkeypatch)
    with pytest.raises(httpx.HTTPStatusError):
        backend.classify("sentence", "non-empty abstract")


def test_transient_502_504_500_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """502 Bad Gateway, 504 Gateway Timeout, and 500 Internal Server Error
    are all upstream-transient classes that warrant a retry — same policy
    as 503."""
    for code in (500, 502, 504):
        calls: list[int] = []

        def handler(request: httpx.Request, _code: int = code) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(_code, headers={"retry-after": "0"}, content=b"")
            return _ok_response()

        backend = _build_backend(httpx.MockTransport(handler), monkeypatch)
        rec = backend.classify("sentence", "non-empty abstract")
        assert len(calls) == 2, f"code={code} should have been retried once"
        assert rec.label == "Supports"


# ---------------------------------------------------------------------------
# 3. resume mode
# ---------------------------------------------------------------------------


def _write_human_qrels(path: Path) -> None:
    rows = [
        {"qa_id": "1", "sentence_id": 0, "pmid": "P_HUM",
         "class": "support", "relevance": 1},
    ]
    path.write_text("\n".join(orjson.dumps(r).decode() for r in rows) + "\n")


def test_load_existing_llm_judgements_picks_only_llm_rows(tmp_path: Path) -> None:
    human = tmp_path / "qrels.jsonl"
    _write_human_qrels(human)
    out = tmp_path / "expanded.jsonl"
    emit_expanded_qrels(
        human_qrels_path=human,
        llm_records={
            ("1", 0, "P_LLM1"): JudgeRecord(
                label="Supports", confidence=0.91, input_tokens=10,
                output_tokens=5, backend="openai-gpt-4o-mini", cost_usd=0.0001,
            ),
            ("1", 0, "P_LLM2"): JudgeRecord(
                label="Contradicts", confidence=0.65, input_tokens=10,
                output_tokens=5, backend="openai-gpt-4o-mini", cost_usd=0.0001,
            ),
        },
        out_path=out,
        incomplete=True,
        abort_reason="quota_exhausted: ...",
    )
    resumed = load_existing_llm_judgements(out)
    assert set(resumed) == {("1", 0, "P_LLM1"), ("1", 0, "P_LLM2")}
    assert resumed[("1", 0, "P_LLM1")].label == "Supports"
    assert resumed[("1", 0, "P_LLM1")].skip_reason == "resumed"
    assert resumed[("1", 0, "P_LLM1")].cost_usd == 0.0  # this-session cost only
    # Human row must not bleed into the LLM-resumed set.
    assert ("1", 0, "P_HUM") not in resumed


def test_resume_round_trip_preserves_all_rows(tmp_path: Path) -> None:
    """Emit, resume, emit again with one new row — final file has 1 human + 3 LLM."""
    human = tmp_path / "qrels.jsonl"
    _write_human_qrels(human)
    out = tmp_path / "expanded.jsonl"

    emit_expanded_qrels(
        human_qrels_path=human,
        llm_records={
            ("1", 0, "P_LLM1"): JudgeRecord(
                label="Supports", confidence=0.91, input_tokens=10,
                output_tokens=5, backend="openai-gpt-4o-mini", cost_usd=0.0,
            ),
            ("1", 0, "P_LLM2"): JudgeRecord(
                label="Contradicts", confidence=0.65, input_tokens=10,
                output_tokens=5, backend="openai-gpt-4o-mini", cost_usd=0.0,
            ),
        },
        out_path=out,
        incomplete=True,
    )

    resumed = load_existing_llm_judgements(out)
    combined = dict(resumed)
    combined[("1", 0, "P_LLM3")] = JudgeRecord(
        label="Supports", confidence=0.8, input_tokens=10,
        output_tokens=5, backend="openai-gpt-4o-mini", cost_usd=0.0,
    )
    emit_expanded_qrels(
        human_qrels_path=human, llm_records=combined, out_path=out, incomplete=False,
    )

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert sum(1 for r in rows if r.get("source") == "human") == 1
    llm_rows = [r for r in rows if r.get("source") != "human"]
    assert {r["pmid"] for r in llm_rows} == {"P_LLM1", "P_LLM2", "P_LLM3"}
    sidecar = out.with_suffix(out.suffix + ".meta.json")
    assert json.loads(sidecar.read_text())["incomplete"] is False
