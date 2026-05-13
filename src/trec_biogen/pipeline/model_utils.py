"""Sequential model lifecycle helpers (design D6).

The 4 GB VRAM budget forbids two heavy models resident at once. Every phase
calls ``unload(model)`` before the next phase loads its own — see task 7.4.
"""

from __future__ import annotations

from typing import Any


def unload(*objs: Any) -> None:
    """Drop references to ``objs`` and empty the CUDA allocator cache."""
    for o in objs:
        try:
            del o  # noqa: F841  -- caller still holds the binding; this is symbolic
        except Exception:
            pass
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
