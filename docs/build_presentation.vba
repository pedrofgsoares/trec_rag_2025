' =========================================================================
'  BuildBioGENPresentation.vba
'  ------------------------------------------------------------------------
'  Builds the full course-report presentation for the TREC BioGEN 2025 work
'  documented in docs/phase2_report.md. ~54 slides covering: task, IR theory,
'  Phase 1 baseline, the pool-bias problem, Phase 2 methodology (CoT pivot,
'  §2.16 + §2.17 expansion), eight variant ablations including the LLM-
'  filtered RM3 and LLM query-rewrite negative results, Phase 2.5 two-judge
'  intersection pool + bootstrap CIs + per-topic error analysis, Phase 2.6
'  three-judge Krippendorff α (Llama ↔ Qwen α=0.60 — mini-cot is the
'  contradict-class outlier) and held-out k=5 ECE; positioning against the
'  full BioGEN 2025 leaderboard (Table 5 from Gupta et al. 2026,
'  arXiv:2603.21582), the 7-team pipeline architecture survey, the BioACE
'  / Llama-3.3 disclosure (organisers already use LLM-as-judge — our
'  contribution is the infrastructure around it), and the code-availability
'  gap (0 of 7 teams published code, so we are the first publicly
'  reproducible Task A pipeline if we submit in 2026).
'
'  How to use:
'    1. Open PowerPoint (a blank presentation is fine; the macro creates a new
'       one anyway, so close any deck you don't want to keep first).
'    2. Press Alt+F11 to open the VBA editor.
'    3. In the Project pane, right-click the presentation node and choose
'       Insert -> Module.
'    4. Paste this entire file into the new module.
'    5. Place the cursor inside the BuildBioGENPresentation sub and press F5
'       (or Run -> Run Sub/UserForm). The macro creates a new presentation
'       named "TREC BioGEN 2025 - Phase 2.pptx" in memory; Save As to disk.
'
'  Notes:
'    * The macro is idempotent only in the sense that each call creates a
'       fresh presentation. To rebuild, close the previous deck and re-run.
'    * Speaker notes are added on the key methodological slides.
'    * Formatting is Calibri throughout, light theme. Adjust constants in the
'       Constants section below if your house style differs.
' =========================================================================

Option Explicit

' ----- Constants ---------------------------------------------------------
Const TITLE_FONT As String = "Calibri"
Const BODY_FONT As String = "Calibri"
Const TITLE_SIZE As Single = 32
Const SECTION_SIZE As Single = 40
Const BODY_SIZE As Single = 20
Const SMALL_SIZE As Single = 16
Const TABLE_SIZE As Single = 14

' Layout constants (PpSlideLayout). 1 = Title, 2 = TitleAndContent,
' 11 = TitleOnly, 12 = Blank.
Const LAYOUT_TITLE As Integer = 1
Const LAYOUT_TITLE_CONTENT As Integer = 2
Const LAYOUT_SECTION As Integer = 11
Const LAYOUT_BLANK As Integer = 12

' ----- Main routine ------------------------------------------------------
Public Sub BuildBioGENPresentation()
    Dim pres As Presentation
    Set pres = Application.Presentations.Add

    Application.DisplayAlerts = ppAlertsNone

    pres.PageSetup.SlideSize = ppSlideSizeOnScreen16x9

    ' --- Section 0 : Title -----------------------------------------------
    AddTitleSlide pres, _
        "TREC BioGEN 2025 — Task A", _
        "A Local-First, Pool-Aware Sentence-Level Biomedical Grounding Pipeline" & _
            vbCrLf & "Pedro Soares — Information Retrieval, 2026"

    AddBulletSlide pres, "Outline", Array( _
        "1. The TREC BioGEN 2025 task", _
        "2. Information Retrieval background", _
        "3. Phase 1 baseline pipeline and the §6.5 gate", _
        "4. The TREC pool-bias problem", _
        "5. Phase 2 — pool-aware methodology (LLM judge + CoT pivot)", _
        "6. Eight variant ablations and findings", _
        "7. Phase 2.5 + 2.6 — judge robustness (3 backends, Krippendorff α)", _
        "8. Positioning vs the BioGEN 2025 field (7 teams, BioACE, code gap)", _
        "9. State of the art, limitations, future work")

    ' --- Section 1 : Task & Background -----------------------------------
    AddSectionSlide pres, "Task & Background"

    AddBulletSlide pres, "TREC BioGEN 2025 — what & why", Array( _
        "Annual evaluation forum run by NIST since 1992 (trec.nist.gov)", _
        "BioGEN track: biomedical answer grounding over 26.8M PubMed abstracts", _
        "Task A: per-sentence emit ≤3 supporting + ≤3 contradicting PMIDs", _
        "Sits at the intersection of IR + NLI + biomedical NLP + evaluation methodology", _
        "Our design constraint: single laptop (WSL2, 12 GB RAM, Quadro T1000 4 GB VRAM)")

    AddBulletSlide pres, "Task specification (formal)", Array( _
        "Input: 40 topics, each with question + pre-segmented answer sentences", _
        "Each sentence may carry ""existing_supported_citations"" (track rule excludes these)", _
        "Output: task_a_output.json — supported_citations + contradicted_citations per sentence", _
        "Metric: per-cell precision / recall / F1, macro-averaged", _
        "Strict = positives only; Relaxed = also partial_support / partial_contradict", _
        "Anchor: published baseline = Supports F1 44.34 / Contradicts F1 4.67 (Strict, 2025 qrels)")

    AddBulletSlide pres, "IR theory — sparse retrieval (BM25)", Array( _
        "Okapi BM25 (Robertson & Walker 1994): bag-of-words probabilistic ranker", _
        "Anserini biomedical presets: k1 = 0.9, b = 0.4 (shorter, denser texts)", _
        "Pyserini (Lin et al. 2021): Python wrapper around Anserini / Lucene", _
        "Why sparse here: full-corpus dense encoding is ~80 GB, infeasible on 4 GB VRAM", _
        "Output: top-100 for support path, top-1000 for contradict (rarer signal)")

    AddBulletSlide pres, "IR theory — rerank + dense + hybrid", Array( _
        "Cross-encoder rerank (Nogueira & Cho 2019): scores (query, doc) jointly with a transformer", _
        "MedCPT-Cross-Encoder (Jin et al. 2023, Bioinformatics): domain-adapted on PubMed search logs", _
        "Dense bi-encoder (Karpukhin 2020 DPR; Khattab 2020 ColBERT): encode independently, FAISS ANN", _
        "MedCPT also ships matched query / article encoders for hybrid retrieval", _
        "Hybrid fusion via RRF (Cormack et al. 2009): RRF_score(d) = Σ 1/(k + rank_r(d)), k = 60")

    AddBulletSlide pres, "IR theory — NLI, negation, pool bias", Array( _
        "Textual entailment (Bowman 2015 SNLI; Romanov 2018 MedNLI): label {supports, contradicts, neutral}", _
        "DeBERTa-v3-MNLI-FEVER-ANLI (He 2021): general-domain NLI, strong MedNLI transfer", _
        "NegEx (Chapman 2001): rule-based negation detector; cheap pre-filter for contradict path", _
        "TREC pooled qrels (Voorhees 1998): only pooled PMIDs are judged; outside-pool = false positive", _
        "Pool bias: any retrospective system retrieving different-but-correct PMIDs is structurally penalised")

    AddBulletSlide pres, "IR theory — LLM-as-judge", Array( _
        "Methodology family crystallised by Zheng et al. 2023 (MT-Bench, arxiv 2306.05685)", _
        "Two-step protocol: concordance validation vs human pool → expansion to novel candidates", _
        "Gate threshold: ≥ 0.85 macro-weighted-F1 (per design D3; D3 risk register handles failures)", _
        "Risk family (Wang 2024): positional bias, verbosity bias, calibration drift", _
        "Standard mitigations: chain-of-thought prompts, two-judge agreement floor, manual spot-review")

    ' --- Section 2 : Phase 1 Baseline ------------------------------------
    AddSectionSlide pres, "Phase 1 — Baseline Pipeline"

    AddBulletSlide pres, "Phase 1 architecture (five phases, sequential model loading)", Array( _
        "1. Pyserini BM25: k=100 (support) + k=1000 (contradict)", _
        "2. MedCPT-CE rerank: top-30 per cell, ~30 min on T1000", _
        "3. DeBERTa-MNLI entailment (support path)", _
        "3'. scispaCy segmentation + NegEx + 23-cue regex pre-filter (1.9M → 83k pairs)", _
        "4. DeBERTa-MNLI contradiction-probability scoring", _
        "5. Max-pool aggregation + selection (cap=3, τ=0.5, dedup, exclude existing)", _
        "Hand-off: every phase writes a Parquet, none cached in RAM")

    AddBulletSlide pres, "Hardware constraint — 4 GB VRAM", Array( _
        "At most one transformer is resident at a time", _
        "unload() releases the prior model + torch.cuda.empty_cache() between phases", _
        "Cross-phase state on disk (Parquet), never in process memory", _
        "Resume mode: ""BIOGEN_RUN_DIR=<path>"" picks up where a crash left off", _
        "Phase 2 extension: ""--reuse-from=<run>"" symlinks parquets for cheap ablations")

    AddResultsSlide pres, "Phase 1 results vs published baseline", _
        Array("Pipeline", "Sup F1", "Con F1", "Notes"), _
        Array( _
            Array("Published baseline (organisers')", "44.34", "4.67", "Table 5, official overview"), _
            Array("CLaC (leaderboard top)", "67.74", "—", "support only reported"), _
            Array("InfoLab", "—", "14.15", "contradict only reported"), _
            Array("Starter-kit (organisers' code)", "44.34", "4.21", "reproduced by us within ±2 F1"), _
            Array("Our Phase 1", "5.55", "0.52", "Strict, 2025 qrels — gap = 38.79 / 4.15"))

    AddBulletSlide pres, "Calibration: ±2 F1 vs published baseline (gate PASS)", Array( _
        "Vendored organisers' starter-kit via vendor_starter_kit.sh", _
        "Symlink against our BM25 index + official 2025 inputs", _
        "Re-scored their task_a_output.json through our eval/metrics.py:", _
        "  Supports F1 = 44.34 (Δ = 0.00 vs published 44.34)", _
        "  Contradicts F1 = 4.21 (Δ = 0.46 vs published 4.67)", _
        "Conclusion: our evaluator is correct → gap is in our pipeline, not the score")

    AddBulletSlide pres, "The 38.79 pp gap — hypothesis", Array( _
        "(A) Pool bias dominates: official qrels were built from organisers' baseline picks", _
        "(B) Phase 1 pipeline is algorithmically weak", _
        "Sanity numbers: Phase 1 emits 555 distinct support PMIDs", _
        "  only ~30 of those appear in the 588-triple human qrels", _
        "  the other ~525 were never shown to a human assessor", _
        "Phase 2 mission: test (A) vs (B) with an LLM-judge expanded qrels")

    ' --- Section 3 : Phase 2 -- Pool-Aware ----------------------------------
    AddSectionSlide pres, "Phase 2 — Pool-Aware Methodology"

    AddBulletSlide pres, "LLM-judge design (D2 + D3)", Array( _
        "Single Backend abstract base; three concrete adapters via BACKEND_REGISTRY", _
        "  together: Llama-3.1-70B-Instruct-Turbo, $0.88/M (OSS-default)", _
        "  openai-mini: gpt-4o-mini, $0.15 in / $0.60 out", _
        "  openai: gpt-4o, $2.50 in / $10.00 out", _
        "Single Judge.classify(answer_sentence, pmid, abstract_text) → JudgeRecord", _
        "Robustness: QuotaExhausted on HTTP 402/429, retry-with-backoff on transient errors", _
        "Resume mode: skip already-judged triples on re-invocation (idempotent)")

    AddBulletSlide pres, "Concordance gate — strict mode FAILED", Array( _
        "First attempt: tight JSON prompt {""label"": ""..."", ""confidence"": 0..1}", _
        "openai-gpt-4o-mini: macro w-F1 = 0.7497 (gate threshold 0.85)", _
        "Escalation to gpt-4o: 0.7443 — actually slightly WORSE", _
        "Same structural failure: 171/549 (or 167/549) human-""Supports"" → judge ""Neutral""", _
        "Cost: $0.048 + $0.792 = $0.84 on validation alone, no progress on the gate", _
        "Tempting interpretation: ""humans generous, LLM strict"" → wrong. See next slide.")

    AddSpeakerNotesSlide pres, _
        "Diagnostic disagreement — J-curve case (qa=144 sent=3)", _
        Array( _
            "Sentence: ""Lowering blood pressure below 120/70 mmHg may cause heart and other problems.""", _
            "Abstract: J-curve in ischemic heart disease, lack of randomised studies for sub-80 mmHg," & _
                " guideline target <140/90.", _
            "Human label: Supports.", _
            "GPT-4o (strict): Neutral, confidence 0.80.", _
            "Three implicit inferential steps the LLM did not perform:", _
            "  1. J-curve ⇒ low BP can be harmful (canonical mechanism)", _
            "  2. No RCTs for sub-80 mmHg ⇒ safety unknown ⇒ ""may cause"" is consistent", _
            "  3. Guideline target <140/90, not <120/70 ⇒ no professional support for going lower", _
            "Diagnosis: the LLM has the medical knowledge — but the strict-JSON prompt gives it no surface to articulate inference."), _
        "Walk through the J-curve example slowly. The point is that the model knows the medicine — what it can't do under a strict prompt is articulate the inferential chain. This was the methodological turning point of Phase 2."

    AddBulletSlide pres, "CoT pivot — gate PASSES at 0.8944", Array( _
        "Same model (gpt-4o-mini), same triples — only the prompt changes", _
        "New prompt asks for a 2-3 sentence ""reasoning"" field BEFORE the label", _
        "max_tokens 80 → 300; output now includes inferential chain + label + confidence", _
        "Macro w-F1: 0.7497 → 0.8944 (PASS)", _
        "Supports F1: 0.7723 → 0.9238 (485/549 correct, was 351/549)", _
        "Supports → Neutral confusion: 171 → 45 (CoT unlocks ~75% of false-Neutrals)", _
        "Cost: $0.080 for the full 588-triple re-validation", _
        "Lesson: when an experiment hits an unexpected ceiling, sample concrete cases first")

    AddBulletSlide pres, "§2.16 — rejudge Phase 1 novel PMIDs", Array( _
        "Input: 1074 (qa_id, sentence_id, pmid) triples Phase 1 emitted but not in human qrels", _
        "Backend: openai-mini --prompt cot. --max-concurrent 4", _
        "Output: 605 Supports + 104 Contradicts (the rest dropped as Neutral / Not relevant)", _
        "Cost: $0.149. Wall-clock: ~6 min", _
        "Expanded qrels file: 588 human rows verbatim + 709 LLM rows = 1297 positives", _
        "Sidecar .meta.json carries incomplete flag, abort reason, token counts")

    AddBulletSlide pres, "§2.17 — broader BM25-top-30 expansion", Array( _
        "Trigger: phase2_no_rerank dropped 27 pp on the §2.16 pool — circularity exposed", _
        "Cause: §2.16 pool was built on Phase 1's picks (MedCPT-CE), no_rerank picks BM25 top-30 directly", _
        "Most no_rerank picks were outside the §2.16 pool → scored as wrong even though LLM agrees", _
        "Fix: rejudge BM25 top-30 per (qa_id, sentence_id) cell across both retrieval paths", _
        "5169 new classifications, $0.704 cost, ~16 min at concurrency=8", _
        "Expanded pool now 4758 positives (3.7× larger), cross-variant comparison is honest")

    ' --- Section 4 : Variants & Findings ---------------------------------
    AddSectionSlide pres, "Variant Ablations & Findings"

    AddResultsSlide pres, "Dual-pool summary — eight variants on §2.17 pool", _
        Array("Variant", "Off Sup", "Off Con", "Exp Sup", "Exp Con", "Δ Sup", "Δ Con"), _
        Array( _
            Array("starter_baseline (organisers')", "44.34", "4.21", "16.55", "5.34", "-27.79", "+1.13"), _
            Array("phase1_baseline", "5.55", "0.52", "16.43", "12.01", "+10.88", "+11.49"), _
            Array("phase2_allow_existing", "5.55", "0.52", "16.94", "12.01", "+11.39", "+11.49"), _
            Array("phase2_no_rerank", "6.52", "0.52", "15.35", "11.75", "+8.83", "+11.23"), _
            Array("phase2_bm25_rm3", "3.92", "0.26", "8.97", "5.26", "+5.05", "+5.01"), _
            Array("phase2_bm25_rm3_llm_filtered", "4.03", "0.52", "9.89", "12.01", "+5.86", "+11.49"), _
            Array("phase2_bm25_llm_rewrite", "5.29", "0.52", "10.65", "6.03", "+5.36", "+5.51"), _
            Array("phase2_no_negex", "5.55", "2.65", "16.33", "8.06", "+10.78", "+5.42"), _
            Array("phase2_scifive_large", "5.55", "1.04", "16.43", "5.85", "+10.88", "+4.81"))

    AddBulletSlide pres, "Headline — pool bias is real but bounded", Array( _
        "Published baseline support F1 = 44.34 on the official pool", _
        "Same baseline on the §2.17 expanded pool = 16.55", _
        "= 27.8 pp of the published 44.34 was pool overlap with itself, not intrinsic quality", _
        "Our Phase 1 pipeline on the expanded pool = 16.43 — competitive with starter-kit", _
        "Δ official → expanded converges to ~+10 pp for every internal variant", _
        "= true pool-bias contribution to Phase 1's apparent gap is ~10 pp, not 38.79")

    AddBulletSlide pres, "Negative result — RM3 hurts on biomedical evidence retrieval", Array( _
        "Variant: phase2_bm25_rm3, Pyserini RM3 (fb_terms=10, fb_docs=10, w=0.5)", _
        "Official Sup F1: 5.55 → 3.92 (-1.63 pp). Expanded Sup: 16.43 → 8.97 (-7.46 pp)", _
        "Hypothesised mechanism (matches Pal et al. 2020 on biomedical IR):", _
        "  Queries (question + sentence) are already very specific", _
        "  Top BM25 hits are topically related but typically not evidence-bearing", _
        "  RM3 draws terms from those hits → query drift toward general disease literature", _
        "Textbook lesson: PRF is only as good as the top-k relevance signal it bootstraps from")

    AddBulletSlide pres, "LLM-filtered RM3 — partial recovery, still loses to no-RM3", Array( _
        "Literature fix (Mackie 2023, Pal 2020): filter the pseudo-relevant set with an LLM first", _
        "Implementation: BM25 top-30 → gpt-4o-mini binary 'relevant?' → custom RM1 over accepted subset", _
        "Beats blind RM3 by +0.11 pp official / +0.92 pp expanded support — the filter works as designed", _
        "But still -6.54 pp below plain Phase 1 BM25 on expanded support (9.89 vs 16.43)", _
        "Conclusion: query expansion — lexical OR LLM-curated — is the wrong intervention", _
        "Claim-length biomedical queries are already specific enough; expansion adds noise either way")

    AddBulletSlide pres, "LLM query rewriting — the third query-side falsification", Array( _
        "Hypothesis: maybe RM3 was the wrong expansion mechanism, not expansion itself", _
        "Variant: gpt-4o-mini emits 3 PubMed-style rewrites per (qa_id, sentence_id) cell", _
        "Original query + 3 rewrites → BM25 × 4 rankings → RRF fusion", _
        "Cost: $0.0505 (388 rewrite calls). Wall-clock: ~88 min", _
        "Expanded support: 10.65 — beats blind RM3 (+1.68 pp) and LLM-filtered RM3 (+0.76 pp)", _
        "Still loses to plain Phase 1 by -5.78 pp → query expansion is structurally wrong here", _
        "= three independent query-side experiments converge on the same negative result")

    AddBulletSlide pres, "Defensible win — the contradict path", Array( _
        "Across every internal pipeline: contradict F1 on §2.17 pool ≈ 11.7 - 12.0 pp", _
        "Starter-kit (organisers') on the same pool: 5.34 pp", _
        "Our pipeline more than DOUBLES contradict F1 on the honest pool", _
        "Architectural source: NegEx + cue-list pre-filter + DeBERTa contradiction max-pool", _
        "Result is independent of which support-side variant we run", _
        "= the contribution to flag in the paper / report")

    AddBulletSlide pres, "Other findings", Array( _
        "MedCPT-CE rerank: official +0.97 pp NEGATIVE; expanded +1.08 pp marginal", _
        "  the reranker is roughly neutral on intrinsic quality once pool bias is removed", _
        "allow_existing: +0.51 pp on expanded support; +0.00 on contradict — neutral", _
        "  validates that the existing-citations exclusion is a track-compliance hook, not a lever", _
        "no_negex: structural contradict gain on expanded (8.06 vs Phase 1's 12.01 — flips under intersection)", _
        "scifive_large: 5.85 expanded contradict — biomedical NLI doesn't beat DeBERTa-MNLI on this task", _
        "Total Phase 2 §2 LLM-judge spend: $1.77 across four runs", _
        "GPU time: ~22 h across Phase 1 baseline + 7 Phase 2 variants (phase2_hybrid unrun)")

    ' --- Section 5 : Judge Robustness (Phase 2.5 + 2.6) ------------------
    AddSectionSlide pres, "Phase 2.5 + 2.6 — Judge Robustness"

    AddBulletSlide pres, "§10.1 — Bootstrap 95% CI on the concordance gate", Array( _
        "Reviewer concern: does the 0.85 gate-pass survive 588-triple sampling noise?", _
        "Non-parametric bootstrap on (gold, pred) pairs, B=1000, seed=0:", _
        "  openai-gpt-4o-mini × cot: 0.8982, 95% CI [0.8776, 0.9196] — PASS (lower bound > 0.85)", _
        "  together-llama-3.3-70b × cot: 0.9112, 95% CI [0.8861, 0.9355] — PASS", _
        "Implementation: validate --records-out path.jsonl persists per-call records", _
        "Records also unlock §10.2 calibration and the multi-backend analysis")

    AddBulletSlide pres, "§10.2 — Confidence calibration (ECE, isotonic, held-out CV)", Array( _
        "Raw ECE: mini 0.1136, Llama 0.0961 (both > 0.05 Guo et al. 'substantial' threshold)", _
        "Pattern: confidence 0.6 → empirical accuracy 0% (wildly over-confident at low end)", _
        "         confidence 0.85 → empirical accuracy ~94% (under-confident at the middle)", _
        "PAV isotonic regression (pool-adjacent-violators) recovers most of the gap", _
        "In-sample post-isotonic: 0.0032 (mini), 0.0000 (Llama) — too optimistic", _
        "Phase 2.6 §1 held-out k=5 CV (folded at qa_id): 0.0476 ± 0.0225 (mini), 0.0329 ± 0.0278 (Llama)", _
        "= calibrator generalises across topics, but ~15× the in-sample fit; still ≤ 0.05")

    AddBulletSlide pres, "§10.4 — Multi-backend gate (second backend, Llama-3.3-70B)", Array( _
        "Design D10 mandates backend sensitivity — TOGETHER_API_KEY enabled the third backend", _
        "Substitution: Llama-3.3-70B (Together moved 3.1 to dedicated-endpoint-only)", _
        "Llama-3.3-70B × cot: macro-w-F1 = 0.9112 (PASS), Supports F1 0.958 (beats mini's 0.924)", _
        "Pairwise mini ↔ Llama: raw agreement 0.867, Cohen κ = 0.338 (only 'fair')", _
        "κ-correction reveals: Supports robust to judge; Contradicts carries judge-dependent variance", _
        "Honest claim: expanded Supports F1 is judge-robust; expanded Contradicts is NOT")

    AddBulletSlide pres, "§10.8 (Phase 2.5) — Two-judge intersection-on-contradicts pool", Array( _
        "Full §2.17 re-judge: Llama-3.3-70B via HF Inference Providers (Groq-routed), $2.47", _
        "Cross-judge Jaccard: 0.93 on Supports, 0.12 on Contradicts (43 of 378 union)", _
        "Llama emits 6.3× fewer Contradicts than mini under the same CoT temperature", _
        "Conservative pool: humans verbatim + Supports passthrough + Contradicts intersected", _
        "4 758 → 4 438 positives — 88% drop entirely on Contradicts (363 → 43)", _
        "Wired through eval/metrics.py as --qrels-pool=intersection with cell-level bootstrap CIs")

    AddResultsSlide pres, "Intersection-pool bootstrap CIs (B=1000, 95% percentile)", _
        Array("Variant", "Supports F1 (95% CI)", "Contradicts F1 (95% CI)"), _
        Array( _
            Array("starter_baseline", "16.55 [15.01, 18.25]", "4.01 [2.13, 6.13]"), _
            Array("phase1_baseline", "16.43 [15.15, 17.80]", "1.07 [0.26, 2.04]"), _
            Array("phase2_no_negex", "16.33 [15.15, 17.59]", "3.63 [1.98, 5.38]"), _
            Array("phase2_scifive_large", "16.43 [15.09, 17.73]", "2.21 [0.88, 3.79]"), _
            Array("phase2_allow_existing", "16.94 [15.59, 18.26]", "1.07 [0.21, 2.10]"), _
            Array("phase2_bm25_rm3", "8.97 [7.79, 10.21]", "0.55 [0.00, 1.32]"))

    AddBulletSlide pres, "Two findings on the conservative pool", Array( _
        "(1) Structural Phase 2 claim SURVIVES: no_negex beats Phase 1 on Contradicts", _
        "    3.63 vs 1.07 (~3.4× midpoint), CI overlap marginal", _
        "(2) Apparent 'no_negex >>> starter' on Contradicts DOES NOT survive intersection", _
        "    no_negex 3.63 [1.98, 5.38] vs starter 4.01 [2.13, 6.13] — CIs overlap heavily", _
        "Honest reading: no_negex ≈ starter on conservative Contradicts; both clearly > Phase 1", _
        "Retrieval-side negatives (rm3, rm3_llm_filtered, llm_rewrite) survive unchanged")

    AddBulletSlide pres, "§10.9 (Phase 2.6) — Three-judge Krippendorff α", Array( _
        "Open question after Phase 2.5: is mini over-emitting OR is Llama over-stripping Contradicts?", _
        "Third judge from a different family — Mixtral (HF dropped Mistral) → Qwen2.5-72B-Instruct", _
        "Qwen gate: macro-w-F1 = 0.8980 (between mini and Llama), Contradicts conservatism matches Llama", _
        "Three-way α on the 5398 candidates (full label space):", _
        "  mini ↔ Llama: 0.1166", _
        "  mini ↔ Qwen:  0.2041", _
        "  Llama ↔ Qwen: 0.6013 ('substantial' per Landis & Koch 1977)", _
        "Resolution: mini-cot is the Contradicts-class OUTLIER; the other two backends agree")

    AddBulletSlide pres, "Three-judge intersection pool — Phase 2.5 conclusions hold", Array( _
        "363 mini-only contradicts → 43 mini ∩ Llama (Phase 2.5) → 31 mini ∩ Llama ∩ Qwen (Phase 2.6)", _
        "Pairwise Llama ∩ Qwen alone: 32 — essentially identical to the three-way 31", _
        "31 positives clear design-D2 floor of 20 (small-sample caveat applies)", _
        "Variant rankings unchanged: no_negex 3.70 vs Phase 1 1.12 (~3.3×, structural finding holds)", _
        "no_negex still statistically indistinguishable from starter on Contradicts", _
        "Total Phase 2.6 spend: $4.00 (gate validation + full expand-pool, with HF Router 400 retry)")

    AddBulletSlide pres, "§10.3 + §10.5 — Topical-bias and pool-size sensitivity", Array( _
        "Per-topic LLM-positive distribution: no systematic topical bias (mean 95.2 sup / topic, IQR [62,116])", _
        "Global LLM-support / LLM-contradict ratio = 10.5 — matches PubMed prior toward affirmatives", _
        "Pool thinning: jackknife-style sub-sampling at fractions {0.05, 0.10, ..., 1.00}, N=200 each", _
        "Phase 1 vs starter ranking SWAPS between thin pool (frac=0.10) and full pool", _
        "Differences at thin-pool sizes (frac=0.10) are within sampling-noise width (~2 pp)", _
        "bm25_rm3 is the LEAST pool-dependent variant — Δ +5.20 pp vs +9-10 pp = genuine algorithmic weakness")

    AddBulletSlide pres, "Per-topic error analysis (Phase 2.5 §10.8 appendix)", Array( _
        "Mechanical topic selection (no cherry-picking): rank by Phase1 − starter on intersection pool", _
        "qa=150 (Phase 1 wins, +13.94 pp): MedCPT-CE surfaces LLM-confirmed-but-pool-invisible PMIDs", _
        "qa=120 (tie, +0.00 pp): both pipelines converge semantically but cite disjoint valid evidence", _
        "qa=131 (Phase 1 loses, -19.49 pp): sub-population-specific human-pool golds (paediatric, cancer)", _
        "  MedCPT-CE demotes the small curated literature in favour of broader topical relevance", _
        "Aggregate ~0.1 pp difference Phase 1 vs starter hides these compositional effects")

    ' --- Section 6 : SOTA & Outlook --------------------------------------
    AddSectionSlide pres, "State of the Art, Limitations, Outlook"

    AddResultsSlide pres, "BioGEN 2025 Task A — Table 5 (Strict, official pool, top runs per team)", _
        Array("Team", "Run", "Sup F1", "Con F1"), _
        Array( _
            Array("CLaC", "LLM_NLI_BM25", "67.74", "4.57"), _
            Array("InfoLab", "task_a_run6_A", "67.23", "14.15"), _
            Array("InfoLab", "task_a_run2", "53.41", "15.67"), _
            Array("SIB", "SIB-task-a-1", "58.87", "0.00"), _
            Array("polito", "scifive-ft-512CL-lex", "55.81", "4.79"), _
            Array("dal", "emotional_prompt", "55.53", "1.20"), _
            Array("GEHC-HTIC", "gehc_htic_task_a", "53.53", "8.57"), _
            Array("uniud", "run1_no-rerank_sparse", "39.10", "1.87"), _
            Array("Baseline", "TEST (organisers')", "44.34", "4.67"), _
            Array("Ours", "phase1_baseline", "5.55", "0.52"))

    AddBulletSlide pres, "Pipeline architectures across the seven Task A teams", Array( _
        "CLaC: BM25 + ColBERT + NLI + LLM in decision (top Sup F1 67.74)", _
        "InfoLab: BM25 + strong reranker + SciFive-MedNLI variants (top Con F1 15.67)", _
        "GEHC-HTIC: BM25 'Decoupled Lexical' + Narrative-Aware Rerank + One-Shot ICL", _
        "  (arXiv:2603.17580, GE Healthcare Bangalore)", _
        "dal: BM25 + RAG variants + Llama-3-70B / GPT-3.5 with emotional / expert prompts", _
        "polito: BM25 + SciFive-large fine-tuned on MedNLI", _
        "SIB: BM25 / SIBiLS + Bio-Medical-Llama-3-8B (per HF card)", _
        "uniud: sparse + dense passage indexes, four rerank-on/off ablations", _
        "Baseline TEST: BM25 + ms-marco-MiniLM + SciFive-MedNLI", _
        "Ours: BM25 + MedCPT-CE + DeBERTa-MNLI + NegEx — LLM kept OUT of pipeline (only in judge)")

    AddBulletSlide pres, "Field finding — LLM-in-decision is where the Supports gap lives", Array( _
        "Every top-Supports system places an LLM in the DECISION path (CLaC, GEHC-HTIC, dal)", _
        "NLI-only pipelines (baseline TEST, polito, ours) cap around 44-55 Sup F1", _
        "We deliberately kept the LLM out of the pipeline (cleaner attribution, cheaper iteration)", _
        "  → structural cost: bounded Sup F1 that §10 methodological work does NOT close", _
        "scifive_large ablation: 1.04 Con F1 official, 5.85 expanded — domain NLI is not a free win", _
        "Phase 4 candidate: phase2_llm_decision — CoT-prompted gpt-4o-mini over MedCPT-CE top-30", _
        "  expected lift: ~10+ pp on Sup expanded; cost: ~$1-2 per full run at concurrency 8")

    AddBulletSlide pres, "Pool construction (overview §6) — pool bias quantified", Array( _
        "Official pool: 10 topics (not 40), 244 PubMed abstracts manually assessed", _
        "ONE top-priority run per team contributes — 7 teams + baseline = 8 runs feeding the pool", _
        "588 human triples in our qrels = the 244 PMIDs × answer-sentence × class expansion", _
        "= the structural mechanism our §5 / §8.1 analysis already diagnosed, now confirmed in print", _
        "Pool bias inflates the baseline TEST run by 27.79 pp (44.34 official → 16.55 §2.17)", _
        "Non-pooled retrospective systems pay the smaller-but-same-direction gap")

    AddBulletSlide pres, "BioACE / Llama-3.3 disclosure — repositioning our contribution", Array( _
        "Overview §6: 'For all submitted runs, we used the BioACE evaluation framework' (Llama-3.3)", _
        "Same prompt task: classify (answer-sentence, document) as Supports / Contradicts / Neutral / NR", _
        "Overview defers: 'detailed analysis of correlation between expert and automated evaluation in future work'", _
        "→ we are NOT the first to use LLM-as-judge here; the organisers already do", _
        "Our differentiation is what we layer on top:", _
        "  three backends + Krippendorff α; bootstrap CI on the gate; held-out k=5 CV ECE", _
        "  CoT pivot diagnosed via 4-case probe; conservative-pool reporting with cell-level CIs", _
        "= the infrastructure that makes single-backend LLM-as-judge defensible")

    AddBulletSlide pres, "Code availability — 0 of 7 teams published", Array( _
        "Audited all 7 Task A notebook papers + author personal GitHubs + institutional orgs", _
        "  papers: trec.nist.gov/pubs/trec34/papers/<team>.biogen.pdf", _
        "  orgs checked: CLaC-Lab, sib-swiss, ailab-uniud", _
        "  personal profiles: jknafou, jarobyte91 (dal), aman2000jaiswal14 (dal)", _
        "Zero participant repositories. Only public code: organisers' starter-kit-2025", _
        "Nearest precedent: webis-de/trec24-biogen (2024, Webis did not submit in 2025)", _
        "Implication: independent verification of Table 5 at the implementation level is IMPOSSIBLE", _
        "Opportunity: submit this repo to 2026 as the first publicly reproducible Task A reference", _
        "  independently of where pipeline ranks — the methodological infra IS the contribution")

    AddBulletSlide pres, "Where this work sits in the broader IR literature", Array( _
        "Sparse: BM25 + Anserini biomed presets = competitive baseline at our scale", _
        "  SPLADE / uniCOIL beat BM25 on MS MARCO but need bigger VRAM at indexing", _
        "Dense: MedCPT bi-encoder is SOTA for PubMed (Jin 2023, Bioinformatics)", _
        "  late-interaction (ColBERT-v2) would 10× recall but 10× storage", _
        "NLI: DeBERTa-v3-MNLI ~85% MedNLI; SciFive-large ~85.6%; both used in 2025 field", _
        "LLM-judge closest external parallel: TREC Health Misinformation 2022 (Clarke 2023, 0.80 gate)", _
        "LLM-judge closest internal parallel: BioGEN organisers' BioACE / Llama-3.3 (single backend, no CI)", _
        "Multi-juror corroboration with Krippendorff α: new to the BioGEN family of tracks")

    AddBulletSlide pres, "Limitations", Array( _
        "Hardware: 4 GB VRAM rules out full-corpus dense, late-interaction, joint training", _
        "LLM-in-decision gap vs the field (CLaC 67.74, GEHC-HTIC 53.53) is NOT closed by §10 work", _
        "  → structural cap on our Sup F1 until Phase 4 lifts the LLM-out-of-pipeline restriction", _
        "Contradicts class is small-sample after intersection (31 positives on 3-way pool)", _
        "Expanded pool local to Phase 1 retrieval shape: phase2_hybrid would need own expand-pool pass", _
        "Single domain-expert review of 12 disagreement cases — paper-grade wants 2 reviewers, 50+", _
        "No team-level head-to-head possible: 0 of 7 teams published task_a_output.json or code", _
        "phase2_hybrid (BM25 + Dense MedCPT + RRF) wired but unrun (~24 h CPU encoding)")

    AddBulletSlide pres, "Future work", Array( _
        "Phase 4 — phase2_llm_decision: CoT-prompted gpt-4o-mini over MedCPT-CE top-30", _
        "  expected lift: 10+ pp on Sup expanded; cost ~$1-2/run at concurrency 8", _
        "  this is the variant that would close the LLM-in-decision gap vs CLaC / GEHC-HTIC / dal", _
        "Phase 3 — NLI fine-tuning: QLoRA-tuned DeBERTa on SciFact/HealthVer/BioNLI (~6 h Colab GPU)", _
        "Reproduce CLaC / InfoLab / GEHC-HTIC on §2.17 pool if any team publishes task_a_output.json", _
        "  one email away — closes the published-vs-honest gap for that team", _
        "Phase 2.5 / 2.6 fourth-juror (Mixtral dedicated endpoint, MedPaLM) — 4-way Krippendorff α", _
        "Submit to BioGEN 2026 as the FIRST publicly reproducible Task A pipeline", _
        "  Hydra configs + Phase 2.5/2.6 qrels artefacts + three-judge α infrastructure = the contribution")

    AddBulletSlide pres, "Methodological contributions", Array( _
        "Pool-bias diagnosis and quantification for a specific 2025 TREC track", _
        "  10 topics / 244 PMIDs / one run per team → 27.79 pp baseline inflation, now quantified", _
        "LLM-as-judge defensibility infrastructure beyond BioACE / Llama-3.3 baseline:", _
        "  three backends + Krippendorff α (mini ↔ Llama ↔ Qwen = 0.12 / 0.20 / 0.60)", _
        "  bootstrap CI on the 0.85 gate; held-out k=5 CV ECE; CoT pivot via 4-case probe", _
        "Dual + intersection pool reporting recovers §6.5 anchor while exposing pool bias honestly", _
        "Three independent query-side falsifications (RM3 / LLM-RM3 / LLM-rewrite) all fail", _
        "Cost-bounded LLM expansion: $5.15 total across §2 ($1.77), §10 ($0.91), Phase 2.5+2.6 ($6.47)", _
        "Engineering: reproducible Hydra configs, resume mode, atomic checkpoint writes, 5xx retries", _
        "First publicly reproducible Task A pipeline (0 of 7 BioGEN 2025 teams published code)")

    AddBulletSlide pres, "Key sources", Array( _
        "TREC BioGEN 2025 overview — Gupta et al. 2026, arXiv:2603.21582 (Table 5 anchor)", _
        "TREC 34 proceedings — trec.nist.gov/pubs/trec34/ (7 team notebook papers)", _
        "GEHC-HTIC preprint — Sahoo et al. 2026, arXiv:2603.17580 (GE Healthcare Bangalore)", _
        "Starter-kit-2025 — github.com/trec-biogen/starter-kit-2025 (baseline TEST anchor)", _
        "Pyserini — Lin et al. 2021; MedCPT — Jin et al. 2023, Bioinformatics", _
        "DeBERTa-v3 — He et al. 2021; SciFive — Phan et al. 2021; MedNLI — Romanov & Shivade 2018", _
        "RRF — Cormack et al. 2009; LLM-as-judge — Zheng et al. 2023; CoT — Wei et al. 2022", _
        "Calibration / ECE — Guo et al. 2017; Krippendorff α — Krippendorff 2011; Landis & Koch 1977", _
        "LLM-filtered PRF — Mackie et al. 2023, SIGIR; Pal et al. 2020 (biomedical PRF)", _
        "Full bibliography: docs/phase2_report.md §14")

    AddTitleSlide pres, "Questions?", _
        "Repository: github.com/<user>/trec_rag_2025" & vbCrLf & _
        "Full report: docs/phase2_report.md (§11.5 for field positioning)" & vbCrLf & _
        "Total spend: ~$5.15 + ~22 GPU-hours + 1 laptop" & vbCrLf & _
        "First publicly reproducible Task A pipeline (0 of 7 teams published code)"

    Application.DisplayAlerts = ppAlertsAll

    MsgBox "Built " & pres.Slides.Count & " slides. Save As to disk.", _
        vbInformation, "Done"
End Sub

' =========================================================================
'  Helper routines
' =========================================================================

' --- Title slide (Slide 1, dividers, closing) ---------------------------
Private Sub AddTitleSlide(pres As Presentation, title As String, subtitle As String)
    Dim s As Slide
    Set s = pres.Slides.Add(pres.Slides.Count + 1, LAYOUT_TITLE)
    With s.Shapes.Title.TextFrame2.TextRange
        .Text = title
        .Font.Name = TITLE_FONT
        .Font.Size = 44
        .Font.Bold = msoTrue
    End With
    ' Layout 1 has an explicit subtitle placeholder at index 2.
    If s.Shapes.Count >= 2 Then
        With s.Shapes(2).TextFrame2.TextRange
            .Text = subtitle
            .Font.Name = BODY_FONT
            .Font.Size = 22
        End With
    End If
End Sub

' --- Section divider ----------------------------------------------------
Private Sub AddSectionSlide(pres As Presentation, title As String)
    Dim s As Slide
    Set s = pres.Slides.Add(pres.Slides.Count + 1, LAYOUT_SECTION)
    With s.Shapes.Title.TextFrame2.TextRange
        .Text = title
        .Font.Name = TITLE_FONT
        .Font.Size = SECTION_SIZE
        .Font.Bold = msoTrue
    End With
End Sub

' --- Bullet content slide -----------------------------------------------
Private Sub AddBulletSlide(pres As Presentation, title As String, bullets As Variant)
    Dim s As Slide
    Set s = pres.Slides.Add(pres.Slides.Count + 1, LAYOUT_TITLE_CONTENT)

    With s.Shapes.Title.TextFrame2.TextRange
        .Text = title
        .Font.Name = TITLE_FONT
        .Font.Size = TITLE_SIZE
        .Font.Bold = msoTrue
    End With

    Dim placeholder As Shape
    Set placeholder = GetContentPlaceholder(s)
    If placeholder Is Nothing Then
        ' Fallback: inject a textbox if the layout didn't supply one.
        Set placeholder = s.Shapes.AddTextbox( _
            msoTextOrientationHorizontal, 50, 110, 860, 380)
    End If

    Dim tr As TextRange2
    Set tr = placeholder.TextFrame2.TextRange
    tr.Text = JoinArray(bullets, vbCrLf)
    tr.Font.Name = BODY_FONT
    tr.Font.Size = BODY_SIZE
    tr.ParagraphFormat.Bullet.Type = msoBulletUnnumbered
End Sub

' --- Slide with bullets + speaker notes ---------------------------------
Private Sub AddSpeakerNotesSlide( _
    pres As Presentation, title As String, bullets As Variant, notes As String _
)
    AddBulletSlide pres, title, bullets
    Dim s As Slide
    Set s = pres.Slides(pres.Slides.Count)
    s.NotesPage.Shapes.Placeholders(2).TextFrame2.TextRange.Text = notes
End Sub

' --- Table slide for results --------------------------------------------
'  headers : Array of column headers (Strings)
'  rows    : Array of Array of cell values per row (Strings)
Private Sub AddResultsSlide( _
    pres As Presentation, title As String, headers As Variant, rows As Variant _
)
    Dim s As Slide
    Set s = pres.Slides.Add(pres.Slides.Count + 1, LAYOUT_TITLE_CONTENT)
    With s.Shapes.Title.TextFrame2.TextRange
        .Text = title
        .Font.Name = TITLE_FONT
        .Font.Size = TITLE_SIZE
        .Font.Bold = msoTrue
    End With

    ' Remove any default content placeholder — we draw a table instead.
    Dim ph As Shape
    Set ph = GetContentPlaceholder(s)
    If Not ph Is Nothing Then ph.Delete

    Dim n_cols As Integer, n_rows As Integer
    n_cols = UBound(headers) - LBound(headers) + 1
    n_rows = UBound(rows) - LBound(rows) + 2   ' +1 header row

    Dim tbl As Shape
    Set tbl = s.Shapes.AddTable(n_rows, n_cols, _
        Left:=30, Top:=110, Width:=900, Height:=320)

    Dim col As Integer, row As Integer
    For col = 0 To n_cols - 1
        With tbl.Table.Cell(1, col + 1).Shape.TextFrame2.TextRange
            .Text = headers(LBound(headers) + col)
            .Font.Name = BODY_FONT
            .Font.Size = TABLE_SIZE + 2
            .Font.Bold = msoTrue
        End With
    Next col

    For row = 0 To UBound(rows) - LBound(rows)
        Dim r As Variant
        r = rows(LBound(rows) + row)
        For col = 0 To n_cols - 1
            With tbl.Table.Cell(row + 2, col + 1).Shape.TextFrame2.TextRange
                .Text = CStr(r(LBound(r) + col))
                .Font.Name = BODY_FONT
                .Font.Size = TABLE_SIZE
            End With
        Next col
    Next row
End Sub

' --- Find the standard content placeholder of a slide -------------------
Private Function GetContentPlaceholder(s As Slide) As Shape
    Dim sh As Shape
    For Each sh In s.Shapes
        If sh.Type = msoPlaceholder Then
            If sh.PlaceholderFormat.Type <> ppPlaceholderTitle And _
               sh.PlaceholderFormat.Type <> ppPlaceholderCenterTitle Then
                Set GetContentPlaceholder = sh
                Exit Function
            End If
        End If
    Next sh
    Set GetContentPlaceholder = Nothing
End Function

' --- Variant array join (handles 0-based and 1-based) -------------------
Private Function JoinArray(arr As Variant, sep As String) As String
    Dim out As String
    Dim i As Long
    For i = LBound(arr) To UBound(arr)
        If i > LBound(arr) Then out = out & sep
        out = out & CStr(arr(i))
    Next i
    JoinArray = out
End Function
