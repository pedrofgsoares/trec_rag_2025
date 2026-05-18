"""Per-run metadata snapshot (11.1) + structured JSONL logging (11.2).

Phase 2 additions:
* ``PhaseTimer`` context manager — captures wall-clock seconds and CUDA
  peak VRAM per phase. The orchestrator accumulates these into the
  ``wall_clock_seconds_per_phase`` and ``vram_peak_gb_per_phase`` dicts
  written to ``metadata.yaml``.
* ``update_run_metadata`` — appends Phase 2 totals
  (``phase2_variant``, ``wall_clock_seconds_total``, ``vram_peak_gb_total``,
  ``judge_cost_usd``) after the run finishes.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def run_id(label: str) -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{label}"


def git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def hardware_fingerprint() -> dict[str, Any]:
    fp: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        fp["ram_total_gib"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except ImportError:
        pass
    try:
        import torch

        # torch.__version__ is a TorchVersion (str subclass); yaml.safe_dump
        # matches by exact type, so we coerce to plain str.
        fp["torch"] = str(torch.__version__)
        if torch.cuda.is_available():
            fp["cuda_device"] = str(torch.cuda.get_device_name(0))
            fp["cuda_total_gib"] = round(
                int(torch.cuda.get_device_properties(0).total_memory) / 1024**3, 2
            )
    except ImportError:
        pass
    return fp


def snapshot(
    *,
    run_dir: Path,
    resolved_config: dict[str, Any],
    repo_root: Path,
    preflight: dict[str, Any] | None = None,
) -> Path:
    """Write ``runs/<id>/metadata.yaml`` with resolved config + git SHA + hardware fingerprint."""
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "git_sha": git_sha(repo_root),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware_fingerprint(),
        "preflight": preflight or {},
        "config": resolved_config,
    }
    out = run_dir / "metadata.yaml"
    out.write_text(yaml.safe_dump(meta, sort_keys=False))
    return out


@contextmanager
def phase_timer(name: str, results: dict[str, Any]):
    """Context manager: time a pipeline phase and record peak VRAM under ``name``.

    Adds two entries to ``results``:
      * ``results["wall_clock_seconds_per_phase"][name]`` — float seconds
      * ``results["vram_peak_gb_per_phase"][name]`` — float GiB

    Resets ``torch.cuda.max_memory_allocated`` before the phase so the captured
    peak is for *this phase only*, not since process start.
    """
    try:
        import torch

        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        cuda = False

    results.setdefault("wall_clock_seconds_per_phase", {})
    results.setdefault("vram_peak_gb_per_phase", {})
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - t0
        results["wall_clock_seconds_per_phase"][name] = round(elapsed, 2)
        if cuda:
            import torch

            peak_bytes = int(torch.cuda.max_memory_allocated())
            results["vram_peak_gb_per_phase"][name] = round(peak_bytes / 1024**3, 3)
        else:
            results["vram_peak_gb_per_phase"][name] = 0.0


def update_run_metadata(
    run_dir: Path,
    *,
    phase_results: dict[str, Any] | None = None,
    phase2_variant: str | None = None,
    judge_cost_usd: float = 0.0,
    judge_token_breakdown: dict[str, Any] | None = None,
) -> Path:
    """Append per-run Phase 2 fields to ``metadata.yaml`` after run completion.

    Idempotent: re-running with new values overwrites only the keys it touches.
    """
    out = run_dir / "metadata.yaml"
    meta = yaml.safe_load(out.read_text()) if out.exists() else {}

    if phase_results:
        for key in ("wall_clock_seconds_per_phase", "vram_peak_gb_per_phase"):
            if key in phase_results:
                meta[key] = phase_results[key]
        per_phase = phase_results.get("wall_clock_seconds_per_phase", {})
        peaks = phase_results.get("vram_peak_gb_per_phase", {})
        meta["wall_clock_seconds_total"] = round(sum(per_phase.values()), 2)
        meta["vram_peak_gb_total"] = round(max(peaks.values()) if peaks else 0.0, 3)

    if phase2_variant is not None:
        meta["phase2_variant"] = phase2_variant
    meta["judge_cost_usd"] = float(judge_cost_usd)
    meta["judge_token_breakdown"] = judge_token_breakdown or {
        "input_tokens": 0, "output_tokens": 0, "cache_hit_rate": 0.0,
    }

    out.write_text(yaml.safe_dump(meta, sort_keys=False))
    return out


def configure_logger(run_dir: Path) -> None:
    """Attach a JSONL sink to loguru at ``runs/<id>/log.jsonl`` (task 11.2)."""
    try:
        from loguru import logger
    except ImportError:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        run_dir / "log.jsonl",
        serialize=True,
        level="INFO",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
