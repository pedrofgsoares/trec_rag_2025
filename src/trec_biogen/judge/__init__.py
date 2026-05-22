"""LLM-judge subsystem (phase2-pool-aware-pipeline §2).

Public surface::

    from trec_biogen.judge import Judge, JudgeRecord, make_backend
"""

from trec_biogen.judge.backends import (
    BACKEND_REGISTRY,
    Backend,
    HTTPBackend,
    Judge,
    JudgeRecord,
    OpenAI4o,
    OpenAIMini,
    QuotaExhausted,
    RecordedBackend,
    TogetherLlama70B,
    make_backend,
)
from trec_biogen.judge.prompts import (
    COT_SYSTEM_PROMPT,
    LABELS,
    MAX_ABSTRACT_TOKENS,
    SYSTEM_PROMPT,
    PromptMode,
    build_prompt,
    system_prompt_for,
    truncate_abstract,
)

__all__ = [
    "BACKEND_REGISTRY",
    "Backend",
    "COT_SYSTEM_PROMPT",
    "HTTPBackend",
    "Judge",
    "JudgeRecord",
    "LABELS",
    "MAX_ABSTRACT_TOKENS",
    "OpenAI4o",
    "OpenAIMini",
    "PromptMode",
    "QuotaExhausted",
    "RecordedBackend",
    "SYSTEM_PROMPT",
    "TogetherLlama70B",
    "build_prompt",
    "make_backend",
    "system_prompt_for",
    "truncate_abstract",
]
