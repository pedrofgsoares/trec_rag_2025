# Phase 2 Dual-Pool Summary (strict)

All F1 numbers are sentence-level macro under the published BioGEN 2025 methodology (``unjudged_as_zero=True``). Δ columns show official → expanded (positive = pool expansion lifted the F1). The `intersection` pool is the Phase 2.5 two-judge intersection-on-contradicts pool (Supports come from the canonical mini-cot; Contradicts kept only when mini-cot and Together-Llama-70B both label them positive). The `intersection-3way` pool is the Phase 2.6 three-judge pool (adds Qwen-2.5-72B-cot to the intersection — Contradicts require unanimous agreement across all three judges). `n/a` = the corresponding qrels file is absent.

| variant | F1@official Sup / Con | F1@expanded Sup / Con | F1@intersection Sup / Con | F1@intersection-3way Sup / Con | Δ Sup / Con (official→expanded) | wall-clock (s) | VRAM (GiB) | LLM-judge $ |
|---|---|---|---|---|---|---|---|---|
| phase1_baseline (no phase2_variant) | 5.55 / 0.52 | 16.43 / 12.01 | 16.43 / 1.07 | 16.43 / 1.12 | +10.88 / +11.49 | — | — | — |
| allow_existing | 5.55 / 0.52 | 16.94 / 12.01 | 16.94 / 1.07 | 16.94 / 1.12 | +11.39 / +11.49 | 0.00 | 0.00 | 0.00 |
| no_rerank | 6.52 / 0.52 | 15.35 / 11.75 | 15.35 / 1.07 | 15.35 / 1.12 | +8.83 / +11.23 | 700.57 | 2.07 | 0.00 |
| bm25_rm3 | 3.92 / 0.26 | 8.97 / 5.26 | 8.97 / 0.55 | 8.97 / 0.60 | +5.05 / +5.01 | 5073.13 | 2.07 | 0.00 |
| bm25_rm3_llm_filtered | 4.03 / 0.52 | 9.89 / 12.01 | 9.89 / 1.07 | 9.89 / 1.12 | +5.86 / +11.49 | 5041.30 | 2.07 | 0.00 |
| no_negex | 5.55 / 2.65 | 16.33 / 8.06 | 16.33 / 3.63 | 16.33 / 3.70 | +10.78 / +5.42 | 34816.46 | 1.96 | 0.00 |
| bm25_llm_rewrite | 5.29 / 0.52 | 10.65 / 6.03 | 10.65 / 0.81 | 10.65 / 0.86 | +5.36 / +5.51 | 5258.82 | 2.07 | 0.00 |
| scifive_large | 5.55 / 1.04 | 16.43 / 5.85 | 16.43 / 2.21 | 16.43 / 2.00 | +10.88 / +4.81 | 18987.70 | 2.02 | 0.00 |
| starter_baseline_20260514_150718 (no metadata) | 44.34 / 4.21 | 16.55 / 5.34 | 16.55 / 4.01 | 16.55 / 4.08 | -27.79 / +1.13 | — | — | — |

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
| HuggingFace Inference Providers (Llama-3.3-70B via Groq) | $2.47 | Phase 2.5 §1.3 second-judge rejudge (5398 triples, CoT) |
| **Total** | **$5.16** | Plus ~27 h GPU/CPU time across Phase 1 + 5 variants + 5 LLM-judge passes |

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

## §13 — Phase 2.5 judge-robustness closure

The Phase 2.5 work added a second LLM judge (`Llama-3.3-70B --prompt cot`
via HuggingFace Inference Providers), a two-judge intersection-pool, a
third `--qrels-pool=intersection` flag end-to-end, per-topic F1 reporting,
a mechanical 3-topic qualitative analysis, and bootstrap CIs at the cell
level on the intersection-pool numbers. Two reports were produced:
[`reports/judge_intersection_analysis.md`](judge_intersection_analysis.md)
and [`reports/per_topic_error_analysis.md`](per_topic_error_analysis.md).

The intersection pool is aggressively conservative: 88 % of the Contradicts
positives in the mini-cot pool are *not* ratified by Llama. The two judges
agree on Supports (Jaccard ~0.93) but rarely on Contradicts (Jaccard
~0.12). This isn't noise — it's a calibration asymmetry: small models
trained on RLHF data are more permissive on contradictions than the
larger Llama-70B, at the same CoT temperature.

The §11.3 structural Phase 2 finding (no_negex beats Phase 1 on
Contradicts) **survives the conservative pool**: 3.63 [1.98, 5.38] vs
1.07 [0.26, 2.04] is directionally clear and 3.4× the midpoint. But
the original "no_negex >>> starter on Contradicts" reading from the
expanded pool (12.01 vs 5.34) **does not** survive: 3.63 vs 4.01 sits
inside the CI overlap. Honest re-reading: on the conservative pool,
no_negex and starter are statistically indistinguishable on Contradicts,
and both clearly beat Phase 1. The bootstrap CIs in
[`reports/judge_intersection_analysis.md`](judge_intersection_analysis.md)
flag where the cross-variant ordering is signal vs noise.

The three retrieval-side negative results (blind RM3, LLM-filtered RM3,
LLM rewrites) survive the pool tightening unchanged — their Supports F1
sits several CIs below the ~16-17 band where every selection-side
variant clusters.

The per-topic qualitative analysis (qa_ids 150, 120, 131 picked
mechanically) reveals **three distinct patterns** behind the aggregate
F1 numbers: pool-expansion gains (qa=150, MedCPT-CE surfaces
LLM-confirmed-but-pool-invisible PMIDs), trade-offs (qa=120, both
pipelines converge on the same topical understanding but cite disjoint
valid evidence), and reranker losses (qa=131, MedCPT-CE demotes
sub-population-specific human-pool golds on sentences 4-5). The
aggregate ~0.1 pp difference between Phase 1 and starter on supports
hides these compositional effects, which the per-topic view makes
visible.

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

## §14 — Phase 2.6: three-judge intersection + held-out calibration ECE

Phase 2.6 closed the two methodological frontiers that Phase 2.5
sign-off explicitly flagged as open: (a) the in-sample isotonic-PAV
ECE in §10.2 is replaced by a k=5 held-out CV estimate with folds
split at `qa_id` boundaries (mini-cot 0.0476 ± 0.0225; Together-cot
0.0329 ± 0.0278 — both still ≤ the Guo et al. "substantial" threshold
of 0.05, just much closer to it than the in-sample numbers suggested);
and (b) a third independent judge (Qwen2.5-72B-Instruct via HF
Inference Providers, pivot from the originally-spec'd Mixtral-8x7B
because HF Providers had dropped the Mistral family from its
chat-routable roster at implementation time) was added to enable a
three-way Krippendorff α and a three-judge intersection-on-contradicts
pool.

The empirical headline from the α computation is decisive: on the 5 398
candidate set, **Llama-70B-cot and Qwen-72B-cot agree at α = 0.6013**
("substantial"), but both agree with mini-cot at only α = 0.12 / 0.20
("slight"). This resolves the Phase 2.5 ambiguity "mini over-emits OR
Llama over-strips" cleanly in favour of "**mini is the outlier**".
Llama and Qwen — trained by Meta and Alibaba respectively on different
data with different architectures — converge on a much smaller
contradicts set than mini ratifies.

The 3-way intersection narrowed Contradicts from 363 (mini-only) → 43
(mini ∩ Llama) → **31** (mini ∩ Llama ∩ Qwen, 91.5 % drop). Llama ∩ Qwen
alone is 32 — essentially identical to the three-way 31, confirming the
α reading. Re-scoring all run dirs against the 3-way pool produced
±0.07 pp shifts on Contradicts F1 vs the 2-way pool; the Phase 2.5
qualitative findings carry over unchanged (`no_negex` still beats Phase 1
by ~3.3×; `no_negex` still statistically indistinguishable from `starter`).

The Phase 2.6 §1-§3 spend was $4.00 (~$0.36 gold + ~$3.64 expand-pool +
resume, after one HF Router 400 mid-run that the per-200-triple
checkpoint absorbed cleanly). Inside the $5 hard ceiling defined in
the openspec change. Cumulative LLM-judge spend across the whole
programme (Phase 2 §2 $1.77 + Phase 2 §10 $0.91 + Phase 2.5 $2.49 +
Phase 2.6 $4.00) ≈ **$9.17 total** — about three coffees, for a
defensible three-judge methodology over a 26.8 M-doc corpus.
