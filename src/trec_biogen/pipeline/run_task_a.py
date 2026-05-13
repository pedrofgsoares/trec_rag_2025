"""End-to-end Task A pipeline orchestrator (Hydra entry point).

Runs the five-phase pipeline (design D6), writing every intermediate to
``runs/<id>/`` and producing a validated submission JSONL plus a leaderboard
comparison report.

Phases:
    1. Retrieve k=100 (support)               -> retrieval_support.parquet
    1'. Retrieve k=1000 (contradict)          -> retrieval_contradict.parquet
    2. Rerank with MedCPT-CE (support, top-30)-> rerank_support.parquet
    3. Support NLI (DeBERTa-MNLI)             -> nli_support.parquet
    3'. Segment abstracts                     -> segmented_contradict.parquet
    3''. NegEx + cue filter                   -> negex_contradict.parquet
    4. Contradict NLI (SciFive-MedNLI)        -> nli_contradict_pairs.parquet
    5. Aggregate max-pool                     -> nli_contradict.parquet
    6. Selection + submission                 -> submission.jsonl
    7. Evaluate against 2024 + 2025 qrels     -> metrics_*.json, report.md

Tasks: 11.1 (Hydra + metadata), 11.3 (MLflow).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from trec_biogen.eval.metrics import evaluate
from trec_biogen.eval.report import phase1_pass, write_report
from trec_biogen.io.qrels import load_qrels
from trec_biogen.io.submission import validate, write_submission
from trec_biogen.io.topics import load_topics
from trec_biogen.nli.negation import filter_negated
from trec_biogen.nli.stance import score_contradict_pairs, score_support
from trec_biogen.pipeline import metadata, phases, preflight
from trec_biogen.pipeline.model_utils import unload
from trec_biogen.pipeline.selection import SelectionConfig, select
from trec_biogen.rerank.cross_encoder import rerank_support
from trec_biogen.retrieval.bm25 import BM25Index


def _repo_root() -> Path:
    """The project root — Hydra changes CWD, so we resolve via this file's path."""
    return Path(__file__).resolve().parents[3]


def _start_mlflow_run(run_dir: Path, resolved: dict[str, Any]) -> Any:
    try:
        import mlflow
    except ImportError:
        return None
    mlflow.set_tracking_uri((_repo_root() / "mlruns").as_uri())
    mlflow.set_experiment("biogen_task_a")
    run = mlflow.start_run(run_name=run_dir.name)
    mlflow.log_params(_flatten(resolved))
    return run


def _flatten(d: dict, parent: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _log_metrics(report: dict[str, dict[str, dict[str, float]]], year: str) -> None:
    try:
        import mlflow
    except ImportError:
        return
    for setting in report:
        for cls in report[setting]:
            for metric, val in report[setting][cls].items():
                mlflow.log_metric(f"{year}_{setting}_{cls}_{metric}", val)


@hydra.main(version_base=None, config_path="../../../configs/run", config_name="phase1_baseline")
def main(cfg: DictConfig) -> None:
    repo = _repo_root()
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)

    run_dir = repo / cfg.paths.runs_dir / metadata.run_id(cfg.run.label)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata.configure_logger(run_dir)
    try:
        from loguru import logger
    except ImportError:  # logger may be absent; degrade gracefully
        class _L:
            def info(self, *a, **k): print(*a)  # noqa: T201
        logger = _L()  # type: ignore[assignment]

    logger.info(f"run_dir={run_dir}")
    pf = preflight.run(repo / cfg.paths.index_dir)
    metadata.snapshot(run_dir=run_dir, resolved_config=resolved, repo_root=repo, preflight=pf)
    mlflow_run = _start_mlflow_run(run_dir, resolved)

    bm = BM25Index(
        repo / cfg.paths.index_dir,
        k1=cfg.retrieval.k1,
        b=cfg.retrieval.b,
    )
    topics = load_topics(repo / cfg.paths.topics)

    try:
        logger.info("phase 1: BM25 k=100 (support)")
        retrieval_sup = phases.retrieve(
            topics, bm,
            k=cfg.retrieval.support_k,
            out_path=run_dir / "retrieval_support.parquet",
        )
        logger.info("phase 1': BM25 k=1000 (contradict)")
        retrieval_con = phases.retrieve(
            topics, bm,
            k=cfg.retrieval.contradict_k,
            out_path=run_dir / "retrieval_contradict.parquet",
        )

        logger.info("phase 2: MedCPT-CE rerank (support)")
        rerank_sup = rerank_support(
            retrieval_sup, bm,
            out_path=run_dir / "rerank_support.parquet",
            batch_size=cfg.rerank.batch_size,
            top_k=cfg.rerank.top_k,
        )
        unload()  # release MedCPT-CE before DeBERTa loads

        logger.info("phase 3: DeBERTa-MNLI entailment (support)")
        nli_sup = score_support(
            rerank_sup, bm,
            out_path=run_dir / "nli_support.parquet",
            model_name=cfg.nli.support.model,
            batch_size=cfg.nli.support.batch_size,
        )

        logger.info("phase 3': abstract segmentation (contradict)")
        seg = phases.segment_abstracts(
            retrieval_con, bm,
            out_path=run_dir / "segmented_contradict.parquet",
        )
        logger.info("phase 3'': NegEx + cue-list filter")
        counts = filter_negated(
            seg,
            out_parquet=run_dir / "negex_contradict.parquet",
            audit_jsonl=run_dir / "negation_audit.jsonl",
            audit_sample=cfg.nli.contradict.audit_sample,
        )
        logger.info(f"negex counts: {counts}")

        logger.info("phase 4: SciFive-MedNLI contradiction NLI")
        # The contradict NLI also needs sentence_text per row; join in.
        import polars as pl

        negex_df = pl.read_parquet(run_dir / "negex_contradict.parquet")
        ret_df = pl.read_parquet(retrieval_con).select(
            ["qa_id", "sentence_id", "sentence_text"]
        ).unique()
        pair_df = negex_df.join(ret_df, on=["qa_id", "sentence_id"], how="left")
        pair_df.write_parquet(run_dir / "pairs_contradict.parquet")

        nli_con_pairs = score_contradict_pairs(
            run_dir / "pairs_contradict.parquet",
            out_path=run_dir / "nli_contradict_pairs.parquet",
            model_name=cfg.nli.contradict.model,
            batch_size=cfg.nli.contradict.batch_size,
        )
        logger.info("phase 5: aggregate contradiction max-pool")
        nli_con = phases.aggregate_contradict(
            nli_con_pairs, out_path=run_dir / "nli_contradict.parquet"
        )

        logger.info("phase 6: selection + submission")
        selection = select(
            nli_support=nli_sup,
            nli_contradict=nli_con,
            config=SelectionConfig(
                tau_sup=cfg.selection.tau_sup,
                tau_con=cfg.selection.tau_con,
                cap=cfg.selection.cap,
            ),
        )
        submission_path = write_submission(selection, run_dir / "submission.jsonl")
        validate(submission_path)
        logger.info(f"submission validated: {submission_path}")

        logger.info("phase 7: evaluation")
        qrels_2025_path = repo / cfg.paths.qrels_2025
        qrels_2024_path = repo / cfg.paths.qrels_2024
        report_2025 = evaluate(submission_path, load_qrels(qrels_2025_path))
        (run_dir / "metrics_2025.json").write_text(json.dumps(report_2025, indent=2))
        _log_metrics(report_2025, "2025")

        report_2024 = None
        if qrels_2024_path.exists():
            report_2024 = evaluate(submission_path, load_qrels(qrels_2024_path))
            (run_dir / "metrics_2024.json").write_text(json.dumps(report_2024, indent=2))
            _log_metrics(report_2024, "2024")

        write_report(
            report_2025_json=run_dir / "metrics_2025.json",
            report_2024_json=run_dir / "metrics_2024.json" if report_2024 else None,
            out_md=run_dir / "report.md",
            run_label=cfg.run.label,
        )

        verdict = phase1_pass(report_2025)
        logger.info(
            f"phase1: passed={verdict.passed} "
            f"sup_f1={verdict.support_f1:.2f} con_f1={verdict.contradict_f1:.2f}"
        )
    finally:
        bm.close()
        if mlflow_run is not None:
            import mlflow

            mlflow.end_run()


if __name__ == "__main__":
    main()
