"""Per-run metadata snapshot (11.1) + structured JSONL logging (11.2)."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
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
