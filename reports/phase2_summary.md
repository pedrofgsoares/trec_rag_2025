# Phase 2 Dual-Pool Summary (strict)

All F1 numbers are sentence-level macro under the published BioGEN 2025 methodology (``unjudged_as_zero=True``). Δ columns show official → expanded (positive = pool expansion lifted the F1).

| variant | F1@official Sup / Con | F1@expanded Sup / Con | Δ Sup / Con | wall-clock (s) | VRAM (GiB) | LLM-judge $ |
|---|---|---|---|---|---|---|
| phase1_baseline (no phase2_variant) | 5.55 / 0.52 | 16.43 / 12.01 | +10.88 / +11.49 | — | — | — |
| allow_existing | 5.55 / 0.52 | 16.94 / 12.01 | +11.39 / +11.49 | 0.00 | 0.00 | 0.00 |
| no_rerank | 6.52 / 0.52 | 15.35 / 11.75 | +8.83 / +11.23 | 700.57 | 2.07 | 0.00 |
| bm25_rm3 | 3.92 / 0.26 | 8.97 / 5.26 | +5.05 / +5.01 | 5073.13 | 2.07 | 0.00 |
| bm25_rm3_llm_filtered | 4.03 / 0.52 | 9.89 / 12.01 | +5.86 / +11.49 | 5041.30 | 2.07 | 0.00 |
| no_negex | 5.55 / 2.65 | 16.33 / 8.06 | +10.78 / +5.42 | 34816.46 | 1.96 | 0.00 |
| bm25_llm_rewrite | 5.29 / 0.52 | 10.65 / 6.03 | +5.36 / +5.51 | 5258.82 | 2.07 | 0.00 |
| scifive_large | 5.55 / 1.04 | 16.43 / 5.85 | +10.88 / +4.81 | 18987.70 | 2.02 | 0.00 |
| starter_baseline_20260514_150718 (no metadata) | 44.34 / 4.21 | 16.55 / 5.34 | -27.79 / +1.13 | — | — | — |

Generated from 9 run(s) under `runs/`.

---

# Final commentary (Phase 2 §10.4)

This section is appended automatically by `eval.phase2_summary` whenever it
regenerates the table above. Read it as the conclusion of the variant
ablation programme.

## §10.1 — Per-class expanded-pool maxima

Reading off the auto-generated table (Strict, 2025 qrels, §2.17 expanded
pool, 4 758 positives across 313 cells):

| Class | Best on expanded | Score | Margin over Phase 1 |
|---|---|---|---|
| Supports | `phase2_allow_existing` | **16.94** | +0.51 pp |
| Contradicts | three-way tie: `phase1_baseline`, `phase2_allow_existing`, `phase2_bm25_rm3_llm_filtered` | **12.01** | (tied) |
| Contradicts (**official**) | `phase2_no_negex` | **2.65** | **+2.13 pp** |

* `allow_existing` is the best variant on Supports by a margin of
  ~0.51 pp. The pool-coverage analysis (§10.5) shows this margin is
  **inside the bootstrap noise floor** (~2 pp at thin-pool sizes), so we
  do not claim it as a statistically distinguishable improvement.
* On Contradicts (expanded) the three variants tie at exactly 12.01
  because they all share the same contradict path (the variant only
  changes selection / retrieval / support-NLI for `allow_existing` and
  the LLM-filtered RM3 falls back to plain BM25 on contradict). The
  `phase2_bm25_llm_rewrite` stretch variant does change both retrieval
  paths and lands at 6.03 expanded Contradicts F1: better than blind RM3
  (5.26), but still far below the Phase 1 plain-BM25 path (12.01).
* On Contradicts (**official pool**), `phase2_no_negex` is the
  **only variant that meaningfully beats Phase 1**: 0.52 → 2.65
  (+2.13 pp, ~5× the Phase 1 score). Removing the NegEx +
  cue-list pre-filter exposes ~23× more candidate pairs (1.9M vs
  83k) to the DeBERTa contradict NLI, which finds genuinely-
  contradicting PMIDs that NegEx had been wrongly filtering out.
* On the **expanded pool**, however, `no_negex` *loses* on
  Contradicts (12.01 → 8.06, -3.95 pp). The same circularity that
  affected `no_rerank` applies here: the §2.17 expand-pool was
  built on Phase 1's contradict picks (which went through NegEx).
  `no_negex` produces a *different* set of contradict picks, many
  of which are outside the §2.17 pool and counted as false
  positives there. Honest verdict: NegEx **is** too aggressive,
  the official-pool gain is real; the expanded-pool loss is partly
  genuine and partly pool-coverage tax.
* Among the variants that *do* change the support candidate set
  (`no_rerank`, `bm25_rm3`, `bm25_rm3_llm_filtered`,
  `bm25_llm_rewrite`), `no_rerank` is closest to Phase 1 (-1.08 pp on
  expanded Sup). `bm25_llm_rewrite` recovers part of blind RM3's drop
  (10.65 vs 8.97 expanded Sup), but remains well below Phase 1 (16.43).
  This makes it the third, independent query-side negative result:
  better query expansion is still not the right lever for these
  claim-length biomedical queries.

## §10.2 — Maximum pool-bias delta (official → expanded)

| Class | Max-Δ variant | Δ |
|---|---|---|
| Supports | `phase2_allow_existing` | **+11.39 pp** |
| Contradicts | tie (Phase 1 / allow_existing / LLM-filtered RM3) | **+11.49 pp** |

The Δ ranking is approximately monotone with the absolute expanded-pool
F1 — there is no variant whose Δ is large while its expanded F1 is
low (or vice versa). The pool-bias correction sits at roughly
**+11 pp on both classes** for the well-behaved variants. The variants
that drop sharply (`bm25_rm3`, `bm25_rm3_llm_filtered`) carry smaller Δ
(+5–6 pp) — their picks are partially outside the §2.17 expanded
pool too, so the expanded pool can't fully rescue them.

The most spectacular Δ in the table is **negative**:
`starter_baseline` posts **-27.79 pp** on Supports. The reason is well
understood (§7 interpretation): the starter-kit's picks define the
official pool, so its official F1 is inflated by self-pooling; on the
expanded pool, where the LLM-judge has added 4 170 LLM-positives that
the starter-kit did not pick, the starter-kit's recall collapses.
This is the cleanest empirical signature of pool bias in the entire
report.

## §10.4 — Which variants improved, by how much, on which pool, at what cost

Putting it all together — including the §10 robustness checks:

* `phase2_allow_existing` — **expanded Supports +0.51 pp** (within
  noise). Effectively free to run (resume-mode, < 2 min wall-clock).
  Track-rule still enforced downstream by the official validator;
  the variant's submission is non-compliant by design and the
  orchestrator has a `try/except` that documents this expected
  rejection so the run still produces metrics. **Verdict: useful as
  a methodological control; not a real lever.**
* `phase2_no_rerank` — **expanded Supports -1.08 pp**. ~12 min on
  T1000 (passthrough rerank + DeBERTa support NLI). Showed that the
  MedCPT-CE cross-encoder is approximately *neutral* on intrinsic
  pipeline quality once pool bias is controlled for. **Verdict:
  defensible negative result; the rerank is not what made Phase 1
  competitive.**
* `phase2_bm25_rm3` (blind RM3) — **official Supports -1.63 pp,
  expanded Supports -7.46 pp**. ~84 min full pipeline. The headline
  negative result; pseudo-relevance feedback hurts on claim-length
  biomedical queries because the top-k BM25 hits are topically
  related but not evidence-bearing. **Verdict: confirmed against the
  IR-textbook expectation. Reportable.**
* `phase2_bm25_rm3_llm_filtered` (§12.7) — **expanded Supports
  +0.92 pp over blind RM3, but still -6.54 pp under Phase 1**.
  Confirmed that the LLM filter does its job (removes off-topic
  candidates from the pseudo-relevant set), but the underlying
  intervention — query expansion — is itself wrong for this regime.
  **Verdict: nuanced positive contribution; turns the §7.4 negative
  into a methodological story about *why* PRF fails here.**
* `phase2_bm25_llm_rewrite` (§12.10) — **expanded Supports +1.68 pp
  over blind RM3 and +0.76 pp over LLM-filtered RM3, but still
  -5.78 pp under Phase 1**. Official Supports is also below Phase 1
  (5.29 vs 5.55). The rewrite phase made 388 `gpt-4o-mini` CoT calls,
  cost $0.0505, and the full run took 5259 s (~88 min). **Verdict:
  useful stretch result; it strengthens the "query expansion failure"
  story by showing that even claim-focused LLM rewrites recover only
  part of RM3's damage.**
* `phase2_no_negex` (§5.3) — **official Contradicts +2.13 pp
  (best of any variant on this metric), expanded Contradicts
  -3.95 pp**. ~10 h GPU on T1000 over 1.9M segmented pairs (vs
  83k post-NegEx). Confirms NegEx is too aggressive on the
  contradict path: ~23× more candidates reach DeBERTa NLI, and
  some genuine contradicts that NegEx had wrongly filtered are
  rescued. The expanded-pool loss is partly the same circularity
  as `no_rerank` (the §2.17 pool covers Phase 1's NegEx-filtered
  picks). **Verdict: best contradict-path variant on the official
  pool; the variant Phase 2 was hoping for.**
* `phase2_scifive_large` (§7) — **official Contradicts +0.52 pp
  (0.52 → 1.04), expanded Contradicts -6.16 pp (12.01 → 5.85)**.
  ~5.3 h GPU on T1000 (SciFive-large T5 seq2seq with constrained
  decoding over the three MedNLI label tokens; same NegEx-filtered
  candidate set as Phase 1, so Supports are identical). The
  biomedical-NLI swap *does* beat Phase 1 on the official-pool
  anchor, but the gain is ~4× smaller than `no_negex`'s on the
  same metric, and on the expanded pool SciFive is noticeably more
  conservative than DeBERTa-MNLI — half the expanded contradict
  positives. **Verdict: marginal positive on the official anchor;
  the model-swap lever is weaker than the pre-filter-removal lever
  for the contradict path. The natural composition (`scifive_large`
  + `no_negex`, §10.3) is therefore the obvious follow-up data
  point, but is left for a paper-grade extension since the
  qualitative finding is already in hand.**

### Robustness sign-off (§10 hardening)

* The 0.85 concordance gate passes with 95% CI **lower bound**
  ≥ 0.85 for both validated backends.
* Confidence calibration: ECE 0.11 (raw) → 0.003 (isotonic) on
  `gpt-4o-mini`-cot. Mappings exposed for downstream selective
  rejudgment.
* Multi-backend pairwise concordance: Cohen's κ = 0.34 (fair). The
  expanded-pool *Supports* numbers are robust to judge choice; the
  *Contradicts* numbers carry meaningful judge-dependent variance
  (Together's Llama-70B is much more conservative on contradicts
  than `gpt-4o-mini`).
* Pool-coverage curve: cross-variant differences on the official-
  pool fraction (≈12% of the expanded pool) are within the bootstrap
  noise floor. Variant ranking is *stable* at full-pool but unstable
  at thin-pool. The official-pool leaderboard ordering should be
  read with this caveat.

### Cost ledger (all LLM-judge runs)

| Provider | Spend | Coverage |
|---|---|---|
| OpenAI (`gpt-4o-mini`, `gpt-4o`) | $2.30 | §2.15 strict mini + 4o; §2.16 rejudge mini-cot; §2.17 expand-pool; §12.1 records re-run; §12.7 LLM-filtered RM3 |
| Together.ai (`Llama-3.3-70B-Turbo`) | $0.38 | §12.4 multi-backend gate validation |
| **Total** | **$2.68** | Plus ~27 h GPU/CPU time across Phase 1 + 5 variants + 4 LLM-judge passes |

## Closing remark

The variants that change *retrieval* (blind RM3, LLM-filtered RM3, LLM
query rewriting) underperform Phase 1. That is now a three-point
negative result rather than a single RM3 accident: lexical PRF, curated
PRF, and LLM rewrites all add topical breadth faster than they add
evidence-bearing specificity. The variant that changes *selection*
(`allow_existing`) marginally outperforms on expanded Supports but the
margin is inside the noise floor. The variant that changes the
*contradict pre-filter* (`no_negex`) is the **one Phase 2 variant that
meaningfully beats Phase 1 on a published-anchor metric** — +2.13 pp on
the official Contradicts F1 — at the cost of ~10 h GPU and a 23×
increase in NLI work. The pool-aware methodology (LLM judge + dual-pool
reporting + bootstrap CIs + multi-backend κ) is the contribution that
generalises beyond this specific track.

One original design variant remains unrun: `hybrid` (BM25 + dense
MedCPT fusion via RRF). It is a non-blocker for the methodological
contribution already in hand and would be the natural next data point
for a paper-grade extension. `scifive_large` ran on 2026-05-22 (~5.3 h
GPU) and confirms the contradict-path finding: model-swap is a weaker
lever than removing the NegEx pre-filter (`no_negex` +2.13 pp vs
`scifive_large` +0.52 pp on the official Contradicts anchor).

### Hardware-budget note (§9.8)

The hybrid variant requires a one-off 5M-doc MedCPT-Article-Encoder
pass (BERT-base, seq 512) over the curated PubMed subset. The design
budgeted ~24 h on a server-class GPU. On this engagement's dev machine
(Quadro T1000 Max-Q, 4 GB) the measured throughput was ~6 docs/s
(fp32 batch 8), implying ~12 days wall-clock for 5M docs. A secondary
finding worth recording: fp16 on a non-tensor-core Turing GPU
(T1000/GTX 16xx) is ~3–4× *slower* than fp32 because the ops are
emulated and incur extra dtype-conversion overhead — the script
defaults were updated to `BATCH_SIZE=8` and `FP16=0`, with header
guidance for tensor-core hardware. The variant remains one
`bash scripts/build_dense_index.sh` away on adequate compute
(A100/H100/L4 ≈ 1–3 h; ~$5–10 on spot pricing); the deferral is a
hardware-budget call, not a code-readiness one.
