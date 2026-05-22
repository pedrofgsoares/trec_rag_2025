"""LLM-judge backends.

All backends speak the OpenAI Chat Completions API. Together.ai exposes the
same shape behind a different base URL, so a single :class:`HTTPBackend`
parameterised by ``base_url`` / ``model`` / ``api_key_env`` covers
:class:`TogetherLlama70B`, :class:`OpenAIMini`, and :class:`OpenAI4o`.

:class:`RecordedBackend` lets the test suite replay canned responses keyed
by a SHA-1 of the prompt, with no network access.

:class:`Judge` is the user-facing wrapper. It carries the canonical
``classify(answer_sentence, pmid, abstract_text) -> JudgeRecord`` contract
mandated by task 2.5; ``pmid`` is metadata-only (the backend never needs
it) but threading it through makes per-call logging trivial.

Task: 2.3, 2.4, 2.5, 2.6
"""

from __future__ import annotations

import abc
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from trec_biogen.judge.prompts import LABELS, PromptMode, build_prompt


# Per-1M-token list prices in USD (input, output). Sources: provider list
# prices, May 2026. The cost cap mechanism uses these to halt mid-run before
# spend exceeds budget; off-by-one decimal places will not change behaviour
# materially. Update if a backend rate moves.
_PRICES: dict[str, tuple[float, float]] = {
    # Together Llama-3.3-70B-Instruct-Turbo: $0.88/M tokens
    # (input + output combined, serverless tier). The design's
    # original choice was 3.1-70B-Turbo but Together has since
    # moved that model to dedicated-endpoint-only access; 3.3 is
    # the serverless successor of the same Llama-70B family.
    "together-llama-3.3-70b": (0.88, 0.88),
    "openai-gpt-4o-mini":     (0.15, 0.60),
    "openai-gpt-4o":          (2.50, 10.00),
}


@dataclass(slots=True)
class JudgeRecord:
    """Outcome of one judge call."""

    label: str
    confidence: float
    input_tokens: int
    output_tokens: int
    backend: str
    cost_usd: float
    skip_reason: str | None = None


class QuotaExhausted(RuntimeError):
    """Raised when the backend rejects calls because the account has no remaining credit.

    Distinct from generic HTTP errors so the rejudge loop can halt gracefully
    and emit a partial expanded-qrels with ``incomplete:true``, rather than
    crashing mid-run and losing every judgement already paid for.
    """


class Backend(abc.ABC):
    """Abstract base for an LLM-judge backend."""

    name: str

    @abc.abstractmethod
    def classify(self, answer_sentence: str, abstract_text: str) -> JudgeRecord:
        """Empty abstracts must short-circuit to ``Not relevant`` without a network call."""


def _is_quota_error(resp: httpx.Response) -> bool:
    """Return True iff the response indicates the *account* is out of credit
    (as opposed to a transient rate-limit). HTTP 402 is treated as quota
    unconditionally; HTTP 429 only when the body's ``error.code`` is
    ``insufficient_quota`` — plain 429 is a transient rate-limit and
    eligible for retry/back-off.
    """
    if resp.status_code == 402:
        return True
    if resp.status_code == 429:
        try:
            body = resp.json()
        except (json.JSONDecodeError, ValueError):
            return False
        code = (body.get("error") or {}).get("code") or ""
        return code == "insufficient_quota"
    return False


def _normalize_label(raw: str) -> str:
    """Map raw model output to the canonical LABELS tuple; on miss return ``Neutral``."""
    if not raw:
        return "Neutral"
    s = raw.strip().strip(".").strip('"').strip("'").lower()
    for label in LABELS:
        if s == label.lower():
            return label
    for label in LABELS:
        head = label.split()[0].lower()
        if s.startswith(head):
            return label
    return "Neutral"


def _parse_response_content(content: str) -> tuple[str, float]:
    """Parse ``{"label": ..., "confidence": ...}`` out of a model response."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if 0 <= start < end:
            try:
                obj = json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return "Neutral", 0.0
        else:
            return "Neutral", 0.0
    label = _normalize_label(str(obj.get("label", "")))
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return label, max(0.0, min(1.0, conf))


def _empty_abstract_record(backend_name: str) -> JudgeRecord:
    return JudgeRecord(
        label="Not relevant",
        confidence=1.0,
        input_tokens=0,
        output_tokens=0,
        backend=backend_name,
        cost_usd=0.0,
        skip_reason="empty_abstract",
    )


def _compute_cost(backend_name: str, input_tokens: int, output_tokens: int) -> float:
    price = _PRICES.get(backend_name)
    if price is None:
        return 0.0
    p_in, p_out = price
    return round((input_tokens * p_in + output_tokens * p_out) / 1_000_000, 6)


class HTTPBackend(Backend):
    """OpenAI-compatible chat-completions HTTP backend.

    Transient HTTP 429s are retried with exponential back-off honouring the
    server's ``Retry-After`` header. HTTP 402 and HTTP 429 with
    ``insufficient_quota`` raise :class:`QuotaExhausted` immediately — the
    rejudge loop catches this and emits a partial expanded-qrels rather
    than retrying through a billing failure.
    """

    _MAX_RETRIES = 20
    _INITIAL_BACKOFF_SEC = 2.0
    _MAX_BACKOFF_SEC = 30.0
    # Strict-mode output is a 2-field JSON object (~20 tokens). CoT adds a
    # short reasoning chain (~150-200 tokens). Bump the cap accordingly.
    _MAX_TOKENS_STRICT = 80
    _MAX_TOKENS_COT = 300

    def __init__(
        self,
        *,
        name: str,
        model: str,
        base_url: str,
        api_key_env: str,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        prompt_mode: PromptMode = "strict",
    ) -> None:
        self.name = name
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._timeout = timeout
        self._transport = transport  # injected for tests
        self._prompt_mode: PromptMode = prompt_mode

    def _client(self) -> httpx.Client:
        token = os.environ.get(self._api_key_env)
        if not token:
            raise RuntimeError(
                f"environment variable {self._api_key_env} is required for backend {self.name}"
            )
        kwargs: dict[str, Any] = dict(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=self._timeout,
        )
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def _post_with_retry(self, payload: dict[str, Any]) -> httpx.Response:
        """POST with retry on transient 429s, timeouts, and network errors.

        Distinguishes:
        * HTTP 200 → return.
        * Quota error (402 or 429 with ``insufficient_quota``) → raise
          :class:`QuotaExhausted` immediately so the rejudge loop can halt
          gracefully and emit a partial expanded-qrels.
        * Plain HTTP 429 → back off (honour ``Retry-After``) and retry.
        * :class:`httpx.TimeoutException` /
          :class:`httpx.TransportError` (DNS, connection-reset, etc.) →
          treat as transient, back off and retry.
        * Anything else → ``raise_for_status``.
        """
        backoff = self._INITIAL_BACKOFF_SEC
        last_resp: httpx.Response | None = None
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                with self._client() as client:
                    resp = client.post("/chat/completions", json=payload)
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt < self._MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise
            last_resp = resp
            if resp.status_code == 200:
                return resp
            if _is_quota_error(resp):
                raise QuotaExhausted(
                    f"backend {self.name} reports insufficient_quota (HTTP {resp.status_code}); "
                    f"top up the account and re-invoke with the same --out to resume."
                )
            if resp.status_code == 429 and attempt < self._MAX_RETRIES:
                retry_after = self._parse_retry_after(resp, default=backoff)
                time.sleep(min(retry_after, self._MAX_BACKOFF_SEC))
                backoff = min(backoff * 2, self._MAX_BACKOFF_SEC)
                continue
            resp.raise_for_status()
            return resp
        assert last_resp is not None
        last_resp.raise_for_status()
        return last_resp

    @staticmethod
    def _parse_retry_after(resp: httpx.Response, *, default: float) -> float:
        header = resp.headers.get("retry-after")
        if not header:
            return default
        try:
            return max(0.0, float(header))
        except ValueError:
            return default

    def classify(self, answer_sentence: str, abstract_text: str) -> JudgeRecord:
        if not abstract_text.strip():
            return _empty_abstract_record(self.name)
        messages = build_prompt(answer_sentence, abstract_text, mode=self._prompt_mode)
        max_tokens = (
            self._MAX_TOKENS_COT if self._prompt_mode == "cot" else self._MAX_TOKENS_STRICT
        )
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        resp = self._post_with_retry(payload)
        body = resp.json()
        content = body["choices"][0]["message"]["content"] or ""
        label, confidence = _parse_response_content(content)
        usage = body.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens", 0))
        out_tok = int(usage.get("completion_tokens", 0))
        return JudgeRecord(
            label=label,
            confidence=confidence,
            input_tokens=in_tok,
            output_tokens=out_tok,
            backend=self.name,
            cost_usd=_compute_cost(self.name, in_tok, out_tok),
        )


class TogetherLlama70B(HTTPBackend):
    def __init__(self, *, prompt_mode: PromptMode = "strict") -> None:
        # Together moved Llama-3.1-70B-Turbo to dedicated-endpoint-only
        # serving (May 2026); the same-family successor on the serverless
        # tier is Llama-3.3-70B-Turbo. Same pricing (~$0.88/M combined),
        # comparable MedNLI accuracy. Class name kept stable
        # (TogetherLlama70B) since the parameter count is unchanged.
        super().__init__(
            name="together-llama-3.3-70b",
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            base_url="https://api.together.xyz/v1",
            api_key_env="TOGETHER_API_KEY",
            prompt_mode=prompt_mode,
        )


class OpenAIMini(HTTPBackend):
    def __init__(self, *, prompt_mode: PromptMode = "strict") -> None:
        super().__init__(
            name="openai-gpt-4o-mini",
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            prompt_mode=prompt_mode,
        )


class OpenAI4o(HTTPBackend):
    def __init__(self, *, prompt_mode: PromptMode = "strict") -> None:
        super().__init__(
            name="openai-gpt-4o",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            prompt_mode=prompt_mode,
        )


class RecordedBackend(Backend):
    """In-memory backend used by the test suite.

    Holds a ``{prompt_key -> JudgeRecord}`` mapping; the key is the SHA-1 of
    ``(answer_sentence, abstract_text)`` so tests can record real outputs
    once and replay them deterministically.
    """

    def __init__(
        self,
        records: dict[str, JudgeRecord],
        *,
        name: str = "recorded",
        default: JudgeRecord | None = None,
    ) -> None:
        self.name = name
        self._records = records
        self._default = default

    @staticmethod
    def key_for(answer_sentence: str, abstract_text: str) -> str:
        h = hashlib.sha1()
        h.update(answer_sentence.encode("utf-8"))
        h.update(b"\0")
        h.update(abstract_text.encode("utf-8"))
        return h.hexdigest()

    def classify(self, answer_sentence: str, abstract_text: str) -> JudgeRecord:
        if not abstract_text.strip():
            return _empty_abstract_record(self.name)
        key = self.key_for(answer_sentence, abstract_text)
        rec = self._records.get(key)
        if rec is None:
            if self._default is not None:
                return self._default
            raise KeyError(f"no recorded response for prompt key={key}")
        return rec


BACKEND_REGISTRY: dict[str, type[Backend]] = {
    "together": TogetherLlama70B,
    "openai-mini": OpenAIMini,
    "openai": OpenAI4o,
}


def make_backend(name: str, *, prompt_mode: PromptMode = "strict") -> Backend:
    cls = BACKEND_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"unknown backend {name!r}; choose from {sorted(BACKEND_REGISTRY)}"
        )
    return cls(prompt_mode=prompt_mode)


class Judge:
    """User-facing wrapper around a :class:`Backend`."""

    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        self.name = backend.name

    def classify(
        self, answer_sentence: str, pmid: str, abstract_text: str,
    ) -> JudgeRecord:
        return self._backend.classify(answer_sentence, abstract_text)
