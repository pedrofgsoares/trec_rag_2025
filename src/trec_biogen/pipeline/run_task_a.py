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
import os
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from trec_biogen.eval.metrics import evaluate
from trec_biogen.eval.report import phase1_pass, write_report
from trec_biogen.io.qrels import load_qrels
from trec_biogen.io.submission import validate_official, write_official_submission
from trec_biogen.io.topics import load_topics
from trec_biogen.nli.negation import filter_negated
from trec_biogen.nli.stance import (
    score_contradict_pairs,
    score_contradict_pairs_t5,
    score_support,
)
from trec_biogen.pipeline import metadata, phases, preflight
from trec_biogen.pipeline.model_utils import unload
from trec_biogen.pipeline.selection import SelectionConfig, select
from trec_biogen.rerank.cross_encoder import passthrough_rerank, rerank_support
from trec_biogen.retrieval.bm25 import BM25Index


def _repo_root() -> Path:
    """The project root — Hydra changes CWD, so we resolve via this file's path."""
    return Path(__file__).resolve().parents[3]


def _maybe_run(out_path: Path, fn, *args, _logger=None, **kwargs) -> Path:
    """Run ``fn(*args, out_path=out_path, **kwargs)`` unless ``out_path`` already exists.

    Lets a crashed run resume by setting ``BIOGEN_RUN_DIR=<dir>`` to point
    at the previous run directory: phases whose output Parquet survived are
    skipped and we pick up from the first missing one.
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        if _logger is not None:
            _logger.info(f"reusing existing {out_path.name} ({out_path.stat().st_size} bytes)")
        return out_path
    return fn(*args, out_path=out_path, **kwargs)


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


@hydra.main(version_base=None, config_path="../../../configs", config_name="run/phase1_baseline")
def main(cfg: DictConfig) -> None:
    repo = _repo_root()
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)

    # BIOGEN_RUN_DIR lets you point at an existing run_dir to resume a crashed
    # pipeline — existing intermediate parquets are reused.
    reuse_dir = os.environ.get("BIOGEN_RUN_DIR")
    if reuse_dir:
        run_dir = Path(reuse_dir)
    else:
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

    # --reuse-from=<path>: symlink intermediate parquets from a prior run dir
    # into this run dir so _maybe_run() skips the upstream phases. Used by
    # Phase 2 ablation variants that only change a downstream phase.
    reuse_from = cfg.get("reuse_from") if hasattr(cfg, "get") else None
    if reuse_from:
        src = Path(reuse_from)
        if not src.is_dir():
            raise FileNotFoundError(f"--reuse-from path does not exist: {src}")
        linked = 0
        for parquet in src.glob("*.parquet"):
            link = run_dir / parquet.name
            if link.exists() or link.is_symlink():
                continue
            link.symlink_to(parquet.resolve())
            linked += 1
        logger.info(f"reuse-from {src}: symlinked {linked} parquet files into {run_dir.name}")

    pf = preflight.run(repo / cfg.paths.index_dir)
    metadata.snapshot(run_dir=run_dir, resolved_config=resolved, repo_root=repo, preflight=pf)
    mlflow_run = _start_mlflow_run(run_dir, resolved)

    # Phase 2 per-phase timing + VRAM accumulator.
    phase_results: dict[str, Any] = {}

    # Phase 2 §8: RM3 params (when the retrieval config opts in).
    rm3_cfg = cfg.retrieval.get("rm3") if hasattr(cfg.retrieval, "get") else None
    rm3_enabled = bool(rm3_cfg and rm3_cfg.get("enabled", False))
    bm = BM25Index(
        repo / cfg.paths.index_dir,
        k1=cfg.retrieval.k1,
        b=cfg.retrieval.b,
        rm3_fb_terms=int(rm3_cfg.get("fb_terms", 10)) if rm3_cfg else 10,
        rm3_fb_docs=int(rm3_cfg.get("fb_docs", 10)) if rm3_cfg else 10,
        rm3_original_query_weight=float(
            rm3_cfg.get("original_query_weight", 0.5)
        ) if rm3_cfg else 0.5,
    )
    topics = load_topics(repo / cfg.paths.topics)

    try:
        # Phase 2 §9.6 / §12.7: dispatch on retrieval flavour.
        #   bm25                 — plain BM25, optionally + RM3
        #   hybrid_rrf           — BM25 + dense MedCPT FAISS, fused via RRF
        #   bm25_rm3_llm_filtered — BM25 → LLM relevance filter → manual RM3
        flavour = cfg.retrieval.get("flavour", "bm25")
        dense_index = None
        if flavour == "bm25_llm_rewrite":
            from trec_biogen.judge.backends import make_backend, HTTPBackend
            from trec_biogen.retrieval.llm_rewrite import LLMQueryRewriter

            rw_cfg = cfg.retrieval.llm_rewrite
            backend = make_backend(
                rw_cfg.backend, prompt_mode=rw_cfg.get("prompt", "strict"),
            )
            if not isinstance(backend, HTTPBackend):
                raise TypeError(
                    f"llm_rewrite backend must be HTTPBackend, got {type(backend).__name__}"
                )
            rewriter = LLMQueryRewriter(
                backend,
                n_variants=int(rw_cfg.get("n_variants", 3)),
                max_concurrent=int(rw_cfg.get("max_concurrent", 4)),
                prompt_mode=str(rw_cfg.get("prompt", "cot")),
            )
            include_orig = bool(rw_cfg.get("include_original", True))
            rrf_k = int(cfg.retrieval.rrf.k)
            logger.info(
                f"phase 1: BM25 over LLM-rewritten queries (backend={rw_cfg.backend}, "
                f"prompt={rewriter._prompt_mode}, n_variants={rewriter._n_variants}, "
                f"include_original={include_orig}, rrf_k={rrf_k})"
            )
            with metadata.phase_timer("retrieve_support", phase_results):
                retrieval_sup = _maybe_run(
                    run_dir / "retrieval_support.parquet",
                    phases.retrieve_llm_rewrite, topics, bm, rewriter,
                    k=cfg.retrieval.support_k, rrf_k=rrf_k,
                    include_original=include_orig, _logger=logger,
                )
            logger.info("phase 1': BM25 over LLM-rewritten queries (contradict)")
            with metadata.phase_timer("retrieve_contradict", phase_results):
                retrieval_con = _maybe_run(
                    run_dir / "retrieval_contradict.parquet",
                    phases.retrieve_llm_rewrite, topics, bm, rewriter,
                    k=cfg.retrieval.contradict_k, rrf_k=rrf_k,
                    include_original=include_orig, _logger=logger,
                )
            logger.info(
                f"[llm_rewrite] total cost: ${rewriter.stats['cost_usd']:.4f} "
                f"over {rewriter.stats['calls']} calls"
            )
        elif flavour == "bm25_rm3_llm_filtered":
            from trec_biogen.judge.backends import make_backend, HTTPBackend
            from trec_biogen.retrieval.llm_prf import LLMRelevanceFilter

            prf_cfg = cfg.retrieval.llm_prf
            backend = make_backend(
                prf_cfg.backend, prompt_mode=prf_cfg.get("prompt", "strict"),
            )
            if not isinstance(backend, HTTPBackend):
                raise TypeError(
                    f"llm_prf backend must be HTTPBackend, got {type(backend).__name__}"
                )
            llm_filter = LLMRelevanceFilter(
                backend, max_concurrent=int(prf_cfg.get("max_concurrent", 8)),
            )
            initial_k = int(prf_cfg.get("initial_k", 30))
            fb_terms = int(prf_cfg.get("fb_terms", 10))
            apply_sup = bool(prf_cfg.get("apply_to_support", True))
            apply_con = bool(prf_cfg.get("apply_to_contradict", True))

            if apply_sup:
                logger.info(
                    f"phase 1: BM25 + LLM-filtered RM3 (support) "
                    f"backend={prf_cfg.backend} initial_k={initial_k} fb_terms={fb_terms}"
                )
                with metadata.phase_timer("retrieve_support", phase_results):
                    retrieval_sup = _maybe_run(
                        run_dir / "retrieval_support.parquet",
                        phases.retrieve_llm_filtered_rm3, topics, bm, llm_filter,
                        k=cfg.retrieval.support_k,
                        initial_k=initial_k, fb_terms=fb_terms, _logger=logger,
                    )
            else:
                logger.info("phase 1: BM25 (support) — LLM filter disabled for this path")
                with metadata.phase_timer("retrieve_support", phase_results):
                    retrieval_sup = _maybe_run(
                        run_dir / "retrieval_support.parquet",
                        phases.retrieve, topics, bm,
                        k=cfg.retrieval.support_k, _logger=logger,
                    )
            if apply_con:
                logger.info("phase 1': BM25 + LLM-filtered RM3 (contradict)")
                with metadata.phase_timer("retrieve_contradict", phase_results):
                    retrieval_con = _maybe_run(
                        run_dir / "retrieval_contradict.parquet",
                        phases.retrieve_llm_filtered_rm3, topics, bm, llm_filter,
                        k=cfg.retrieval.contradict_k,
                        initial_k=initial_k, fb_terms=fb_terms, _logger=logger,
                    )
            else:
                logger.info("phase 1': BM25 (contradict) — LLM filter disabled for this path")
                with metadata.phase_timer("retrieve_contradict", phase_results):
                    retrieval_con = _maybe_run(
                        run_dir / "retrieval_contradict.parquet",
                        phases.retrieve, topics, bm,
                        k=cfg.retrieval.contradict_k, _logger=logger,
                    )
            logger.info(
                f"[llm_prf] total filter cost: ${llm_filter.stats['cost_usd']:.4f} "
                f"over {llm_filter.stats['calls']} calls"
            )
        elif flavour == "hybrid_rrf":
            from trec_biogen.retrieval.dense import DenseIndex

            dense_cfg = cfg.retrieval.dense
            dense_index = DenseIndex(
                repo / dense_cfg.index_dir,
                query_model=dense_cfg.query_model,
            )
            rrf_k = int(cfg.retrieval.rrf.k)
            leg_k = dense_cfg.get("k_leg")
            logger.info(
                f"phase 1: hybrid retrieval (BM25 + Dense MedCPT, RRF k={rrf_k})"
            )
            with metadata.phase_timer("retrieve_support", phase_results):
                retrieval_sup = _maybe_run(
                    run_dir / "retrieval_support.parquet",
                    phases.retrieve_hybrid, topics, bm, dense_index,
                    k=cfg.retrieval.support_k, rrf_k=rrf_k, leg_k=leg_k,
                    _logger=logger,
                )
            logger.info(
                f"phase 1': hybrid retrieval (BM25 + Dense MedCPT, RRF k={rrf_k})"
            )
            with metadata.phase_timer("retrieve_contradict", phase_results):
                retrieval_con = _maybe_run(
                    run_dir / "retrieval_contradict.parquet",
                    phases.retrieve_hybrid, topics, bm, dense_index,
                    k=cfg.retrieval.contradict_k, rrf_k=rrf_k, leg_k=leg_k,
                    _logger=logger,
                )
        else:
            rm3_note = " (RM3 enabled)" if rm3_enabled else ""
            logger.info(f"phase 1: BM25 k=100 (support){rm3_note}")
            with metadata.phase_timer("retrieve_support", phase_results):
                retrieval_sup = _maybe_run(
                    run_dir / "retrieval_support.parquet",
                    phases.retrieve, topics, bm,
                    k=cfg.retrieval.support_k, rm3=rm3_enabled, _logger=logger,
                )
            logger.info(f"phase 1': BM25 k=1000 (contradict){rm3_note}")
            with metadata.phase_timer("retrieve_contradict", phase_results):
                retrieval_con = _maybe_run(
                    run_dir / "retrieval_contradict.parquet",
                    phases.retrieve, topics, bm,
                    k=cfg.retrieval.contradict_k, rm3=rm3_enabled, _logger=logger,
                )

        # Phase 2 §4: ``rerank: null`` disables the MedCPT-CE forward pass
        # and falls through to a passthrough that keeps the BM25 top-K in
        # the same parquet shape ``score_support`` expects.
        rerank_cfg = cfg.get("rerank") if hasattr(cfg, "get") else cfg.rerank
        if rerank_cfg is None:
            logger.info("phase 2: rerank disabled (passthrough BM25 top-K)")
            with metadata.phase_timer("rerank_support", phase_results):
                rerank_sup = _maybe_run(
                    run_dir / "rerank_support.parquet",
                    passthrough_rerank, retrieval_sup,
                    top_k=30, _logger=logger,
                )
        else:
            logger.info("phase 2: MedCPT-CE rerank (support)")
            with metadata.phase_timer("rerank_support", phase_results):
                rerank_sup = _maybe_run(
                    run_dir / "rerank_support.parquet",
                    rerank_support, retrieval_sup, bm,
                    model_name=cfg.rerank.model,
                    batch_size=cfg.rerank.batch_size,
                    top_k=cfg.rerank.top_k, _logger=logger,
                )
        unload()  # release MedCPT-CE before DeBERTa loads (no-op if passthrough)

        logger.info("phase 3: DeBERTa-MNLI entailment (support)")
        with metadata.phase_timer("nli_support", phase_results):
            nli_sup = _maybe_run(
                run_dir / "nli_support.parquet",
                score_support, rerank_sup, bm,
                model_name=cfg.nli.support.model,
                batch_size=cfg.nli.support.batch_size, _logger=logger,
            )

        logger.info("phase 3': abstract segmentation (contradict)")
        with metadata.phase_timer("segment_contradict", phase_results):
            seg = _maybe_run(
                run_dir / "segmented_contradict.parquet",
                phases.segment_abstracts, retrieval_con, bm,
                _logger=logger,
            )
        # Phase 2 §5: when ``nli.contradict.negex`` is false, skip the
        # NegEx + cue-list pre-filter and use the segmented sentences
        # directly. The contradict NLI step then runs over ~23× more
        # pairs (~1.9M vs ~83k in Phase 1 on the same input) — overnight
        # job. Schemas match (verified empirically), so the downstream
        # join is unchanged.
        negex_enabled = cfg.nli.contradict.get("negex", True)
        if not negex_enabled:
            logger.info("phase 3'': NegEx disabled — using segmented_contradict directly")
            negex_out = seg
        else:
            logger.info("phase 3'': NegEx + cue-list filter")
            negex_out = run_dir / "negex_contradict.parquet"
            with metadata.phase_timer("negex_filter", phase_results):
                if negex_out.exists() and negex_out.stat().st_size > 0:
                    logger.info(f"reusing existing {negex_out.name}")
                else:
                    counts = filter_negated(
                        seg,
                        out_parquet=negex_out,
                        audit_jsonl=run_dir / "negation_audit.jsonl",
                        audit_sample=cfg.nli.contradict.audit_sample,
                    )
                    logger.info(f"negex counts: {counts}")

        import polars as pl

        pairs_path = run_dir / "pairs_contradict.parquet"
        if not (pairs_path.exists() and pairs_path.stat().st_size > 0):
            negex_df = pl.read_parquet(negex_out)
            ret_df = pl.read_parquet(retrieval_con).select(
                ["qa_id", "sentence_id", "sentence_text"]
            ).unique()
            pair_df = negex_df.join(ret_df, on=["qa_id", "sentence_id"], how="left")
            pair_df.write_parquet(pairs_path)

        # Phase 2 §7: dispatch between the DeBERTa classification head
        # (default) and the SciFive seq2seq constrained-decoding head.
        nli_type = cfg.nli.contradict.get("type", "deberta")
        if nli_type == "t5":
            logger.info("phase 4: contradiction NLI (T5 seq2seq, constrained decoding)")
            with metadata.phase_timer("nli_contradict_pairs", phase_results):
                nli_con_pairs = _maybe_run(
                    run_dir / "nli_contradict_pairs.parquet",
                    score_contradict_pairs_t5, pairs_path,
                    model_name=cfg.nli.contradict.model,
                    batch_size=cfg.nli.contradict.batch_size,
                    fp16=cfg.nli.contradict.get("fp16", True),
                    chunk_size=cfg.nli.contradict.get("chunk_size", 4),
                    _logger=logger,
                )
        else:
            logger.info("phase 4: contradiction NLI (DeBERTa, sentence-level pairs)")
            with metadata.phase_timer("nli_contradict_pairs", phase_results):
                nli_con_pairs = _maybe_run(
                    run_dir / "nli_contradict_pairs.parquet",
                    score_contradict_pairs, pairs_path,
                    model_name=cfg.nli.contradict.model,
                    batch_size=cfg.nli.contradict.batch_size, _logger=logger,
                )
        logger.info("phase 5: aggregate contradiction max-pool")
        with metadata.phase_timer("aggregate_contradict", phase_results):
            nli_con = _maybe_run(
                run_dir / "nli_contradict.parquet",
                phases.aggregate_contradict, nli_con_pairs, _logger=logger,
            )

        logger.info("phase 6: selection + submission")
        selection = select(
            nli_support=nli_sup,
            nli_contradict=nli_con,
            topics=topics,
            config=SelectionConfig(
                tau_sup=cfg.selection.tau_sup,
                tau_con=cfg.selection.tau_con,
                cap=cfg.selection.cap,
                # Phase 2 §6: phase2_allow_existing variant flips this.
                exclude_existing=cfg.selection.get("exclude_existing", True),
            ),
        )
        submission_path = write_official_submission(
            selection, topics, run_dir / "task_a_output.json"
        )
        # Phase 2 §6: the official validator enforces the existing-citations
        # track rule. When the variant explicitly relaxes the rule
        # (``selection.exclude_existing=False``) we expect the validator to
        # reject the submission — log the rejection and continue to eval
        # so the dual-pool numbers are still captured. Any other variant
        # must still validate.
        try:
            validate_official(submission_path)
            logger.info(f"submission validated: {submission_path}")
        except Exception as exc:
            if cfg.selection.get("exclude_existing", True) is False:
                logger.warning(
                    "submission validator rejected the run, as expected for "
                    f"phase2_allow_existing: {exc}"
                )
            else:
                raise

        logger.info("phase 7: evaluation")
        qrels_2025_path = repo / cfg.paths.qrels_2025
        qrels_2024_path = repo / cfg.paths.qrels_2024

        report_2025 = None
        if qrels_2025_path.exists():
            report_2025 = evaluate(submission_path, load_qrels(qrels_2025_path))
            (run_dir / "metrics_2025.json").write_text(json.dumps(report_2025, indent=2))
            _log_metrics(report_2025, "2025")
        else:
            logger.info(f"qrels_2025 missing at {qrels_2025_path} — skipping 2025 eval")

        report_2024 = None
        if qrels_2024_path.exists():
            report_2024 = evaluate(submission_path, load_qrels(qrels_2024_path))
            (run_dir / "metrics_2024.json").write_text(json.dumps(report_2024, indent=2))
            _log_metrics(report_2024, "2024")
        else:
            logger.info(f"qrels_2024 missing at {qrels_2024_path} — skipping 2024 eval")

        if report_2025 is not None:
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
        else:
            logger.info("no qrels available — submission produced, evaluation deferred")
    finally:
        bm.close()
        if mlflow_run is not None:
            import mlflow

            mlflow.end_run()
        # Append Phase 2 totals + per-phase timings to metadata.yaml.
        # phase2_variant is read from the resolved Hydra config; default None.
        phase2_variant = (
            resolved.get("phase2_variant") if isinstance(resolved, dict) else None
        )
        metadata.update_run_metadata(
            run_dir,
            phase_results=phase_results,
            phase2_variant=phase2_variant,
            judge_cost_usd=0.0,
        )


if __name__ == "__main__":
    main()
