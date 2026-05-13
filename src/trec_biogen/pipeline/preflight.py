"""Pipeline preflight — fail-fast before any GPU work (design risks section).

* RAM ≥ 11 GiB (the WSL2 .wslconfig bump).
* CUDA visible (the Quadro T1000 with 4 GB).
* BM25 index present and large enough.

Called once from ``run_task_a`` before models load.
"""

from __future__ import annotations

from pathlib import Path

from trec_biogen.retrieval.bm25 import verify_index

MIN_RAM_BYTES = 11 * (1024**3)


def check_ram(min_bytes: int = MIN_RAM_BYTES) -> int:
    """Return total RAM in bytes; raise if below ``min_bytes``."""
    import psutil

    total = psutil.virtual_memory().total
    if total < min_bytes:
        raise RuntimeError(
            f"RAM={total / 1024**3:.1f} GiB < required {min_bytes / 1024**3:.0f} GiB — "
            "bump WSL2 via host .wslconfig per SETUP.md §1.1"
        )
    return total


def check_cuda() -> dict[str, str | int]:
    """Return summary dict; raise if CUDA is configured but not visible."""
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("torch not installed — see SETUP.md §1.4") from e

    if not torch.cuda.is_available():
        # CPU-only is allowed but the design assumes GPU; warn loudly.
        return {"cuda": False, "device_count": 0}
    return {
        "cuda": True,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    }


def run(index_dir: Path) -> dict:
    """Run all preflight checks, return a summary dict for the run metadata."""
    return {
        "ram_total_bytes": check_ram(),
        "cuda": check_cuda(),
        "index_dir": str(index_dir),
        "index_doc_count": _index_doc_count(index_dir),
    }


def _index_doc_count(index_dir: Path) -> int:
    """Verify the index and return its doc count for the metadata snapshot."""
    verify_index(index_dir)
    from trec_biogen.retrieval.bm25 import BM25Index

    bm = BM25Index(index_dir)
    try:
        return bm.doc_count()
    finally:
        bm.close()
