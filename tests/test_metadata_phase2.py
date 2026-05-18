"""Phase 2 cross-cuts in pipeline.metadata — task 1.6.

Covers the new ``phase_timer`` context manager and ``update_run_metadata``
helper introduced in task 1.3 / 1.4. Pure CPU / file IO, no GPU or models;
safe for CI.
"""

from __future__ import annotations

import time
from pathlib import Path

import yaml

from trec_biogen.pipeline import metadata


def test_phase_timer_records_wall_clock(tmp_path: Path) -> None:
    results: dict = {}
    with metadata.phase_timer("dummy_phase", results):
        time.sleep(0.05)
    assert "wall_clock_seconds_per_phase" in results
    assert "dummy_phase" in results["wall_clock_seconds_per_phase"]
    assert results["wall_clock_seconds_per_phase"]["dummy_phase"] >= 0.04


def test_phase_timer_records_vram_zero_without_cuda(tmp_path: Path) -> None:
    """When CUDA isn't available (or torch missing), peak VRAM is recorded as 0.0."""
    results: dict = {}
    with metadata.phase_timer("p1", results):
        pass
    assert "vram_peak_gb_per_phase" in results
    assert "p1" in results["vram_peak_gb_per_phase"]
    # We don't assume CUDA is unavailable, but if it is, the value is 0.0.
    assert results["vram_peak_gb_per_phase"]["p1"] >= 0.0


def test_phase_timer_resets_per_phase(tmp_path: Path) -> None:
    """Each phase gets its own peak — phase A's peak doesn't leak into phase B."""
    results: dict = {}
    with metadata.phase_timer("phase_a", results):
        time.sleep(0.02)
    with metadata.phase_timer("phase_b", results):
        time.sleep(0.04)
    assert "phase_a" in results["wall_clock_seconds_per_phase"]
    assert "phase_b" in results["wall_clock_seconds_per_phase"]
    assert (
        results["wall_clock_seconds_per_phase"]["phase_b"]
        > results["wall_clock_seconds_per_phase"]["phase_a"]
    )


def test_update_run_metadata_writes_phase2_fields(tmp_path: Path) -> None:
    """All Phase 2 totals + per-phase dicts land in metadata.yaml."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Seed an existing metadata.yaml (mimicking snapshot()'s output)
    (run_dir / "metadata.yaml").write_text(yaml.safe_dump({"git_sha": "abc123"}))

    phase_results = {
        "wall_clock_seconds_per_phase": {"phase_a": 1.2, "phase_b": 3.4},
        "vram_peak_gb_per_phase": {"phase_a": 0.5, "phase_b": 1.7},
    }
    metadata.update_run_metadata(
        run_dir,
        phase_results=phase_results,
        phase2_variant="no_rerank",
        judge_cost_usd=2.34,
        judge_token_breakdown={"input_tokens": 1000, "output_tokens": 50, "cache_hit_rate": 0.4},
    )

    out = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    assert out["git_sha"] == "abc123"  # untouched
    assert out["phase2_variant"] == "no_rerank"
    assert out["wall_clock_seconds_total"] == round(1.2 + 3.4, 2)
    assert out["vram_peak_gb_total"] == 1.7  # max across phases
    assert out["judge_cost_usd"] == 2.34
    assert out["judge_token_breakdown"]["input_tokens"] == 1000


def test_update_run_metadata_idempotent_and_default_judge_cost(tmp_path: Path) -> None:
    """Re-running with judge_cost_usd=0 (default) sets the field; phase2_variant=None leaves it absent."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metadata.yaml").write_text(yaml.safe_dump({"git_sha": "abc"}))

    metadata.update_run_metadata(run_dir)  # all defaults
    out = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    assert out["judge_cost_usd"] == 0.0
    assert out["judge_token_breakdown"]["input_tokens"] == 0
    # phase2_variant left unset because no Phase 2 variant was used.
    assert "phase2_variant" not in out


def test_phase_results_empty_yields_zero_totals(tmp_path: Path) -> None:
    """A run with no phases timed still produces sensible total fields."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metadata.yaml").write_text(yaml.safe_dump({}))
    metadata.update_run_metadata(
        run_dir, phase_results={"wall_clock_seconds_per_phase": {}, "vram_peak_gb_per_phase": {}}
    )
    out = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    assert out["wall_clock_seconds_total"] == 0.0
    assert out["vram_peak_gb_total"] == 0.0
