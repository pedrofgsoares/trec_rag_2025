' =========================================================================
'  BuildBioGENPresentation.vba
'  ------------------------------------------------------------------------
'  Builds the full course-report presentation for the TREC BioGEN 2025 work
'  documented in docs/phase2_report.md. ~28 slides covering: task, IR theory,
'  Phase 1 baseline, the pool-bias problem, Phase 2 methodology, variant
'  ablations, findings, state of the art, limitations, future work.
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
        "5. Phase 2 — pool-aware methodology (LLM judge)", _
        "6. Variant ablations and findings", _
        "7. State of the art, limitations, future work")

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

    AddResultsSlide pres, "Dual-pool summary — all runs to date", _
        Array("Variant", "Off Sup", "Off Con", "Exp Sup", "Exp Con", "Δ Sup", "Δ Con"), _
        Array( _
            Array("starter_baseline (organisers')", "44.34", "4.21", "16.55", "5.34", "-27.79", "+1.13"), _
            Array("phase1_baseline", "5.55", "0.52", "16.43", "12.01", "+10.88", "+11.49"), _
            Array("phase2_allow_existing", "5.55", "0.52", "16.94", "12.01", "+11.39", "+11.49"), _
            Array("phase2_no_rerank", "6.52", "0.52", "15.35", "11.75", "+8.83", "+11.23"), _
            Array("phase2_bm25_rm3", "3.92", "0.26", "8.97", "5.26", "+5.05", "+5.01"))

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
        "Cost of the entire LLM-judge methodology: $1.77 total across 4 runs", _
        "GPU time consumed: ~7.5 h across Phase 1 baseline + 3 Phase 2 variants")

    ' --- Section 5 : SOTA & Outlook --------------------------------------
    AddSectionSlide pres, "State of the Art, Limitations, Outlook"

    AddBulletSlide pres, "Where this work sits", Array( _
        "Sparse: BM25 + Anserini biomed presets = competitive baseline at our scale", _
        "  SPLADE / uniCOIL beat BM25 on MS MARCO but need bigger VRAM at indexing", _
        "Dense: MedCPT bi-encoder is the SOTA for PubMed (Jin 2023)", _
        "  late-interaction (ColBERT-v2) would 10x recall but 10x storage", _
        "NLI: DeBERTa-v3-MNLI is ~85% on MedNLI; SciFive-large reaches ~85.6%", _
        "  our phase2_scifive_large variant is wired in (constrained-decoding T5), not yet run", _
        "LLM-judge: closest parallel is TREC Health Misinformation 2022 (Clarke 2023)", _
        "  they used 0.80 concordance threshold; we use 0.85 (stricter, cleaner labels)")

    AddBulletSlide pres, "Limitations", Array( _
        "Hardware: 4 GB VRAM rules out full-corpus dense, late-interaction, joint training", _
        "  pipeline scales linearly to 24 GB+ GPUs without redesign", _
        "No NLI fine-tuning (Phase 3 deferred). SciFact + HealthVer + BioNLI ≈ 50k pairs", _
        "LLM-judge backend coverage: gpt-4o-mini only on the production expanded qrels", _
        "  Llama-3.1-70B Together backend wired but not yet run (TOGETHER_API_KEY required)", _
        "Expanded pool local to Phase 1 retrieval shape: variants that radically change retrieval", _
        "  (phase2_hybrid) would need their own expand-pool pass", _
        "Single domain-expert review of 12 disagreement cases — paper-grade would want 2 reviewers, 50+ cases")

    AddBulletSlide pres, "Future work", Array( _
        "Phase 3 — NLI fine-tuning: QLoRA-tuned DeBERTa on SciFact/HealthVer/BioNLI", _
        "  ~6 h on a free Colab GPU; expected 2-5 pp on contradict F1", _
        "Phase 4 — agentic retrieval: LLM in the first-stage loop for query rewriting", _
        "  Addresses BM25 vocabulary mismatch at source; cost / latency trade-off", _
        "Backend-sensitivity experiment (design D10): run compare-backends with mini + 4o + Llama-70B", _
        "  defensible claim for the paper: F1@expanded numbers are robust to judge choice", _
        "Submit to TREC BioGEN 2026: the only way to fully escape pool bias for system numbers", _
        "  calendar problem (track call usually March)")

    AddBulletSlide pres, "Methodological contributions", Array( _
        "Pool-bias diagnosis and quantification for a specific 2025 TREC track", _
        "LLM-as-judge with concordance gate, CoT prompt, OSS-default + paid escalation", _
        "Dual-pool reporting (--qrels-pool, --source) recovers §6.5 anchor while exposing pool bias", _
        "Engineering: reproducible Hydra configs, resume mode, per-phase VRAM/timing in metadata", _
        "Cost-bounded LLM expansion: $1.77 total, $10 cost-cap with graceful quota-exhaustion", _
        "Findings: pool bias ~10 pp (not 40), RM3 hurts on biomed, contradict path > starter-kit")

    AddBulletSlide pres, "Key sources", Array( _
        "TREC BioGEN — trec.nist.gov", _
        "Pyserini — Lin et al. 2021; github.com/castorini/pyserini", _
        "MedCPT — Jin et al. 2023, Bioinformatics; HuggingFace ncbi/MedCPT-*", _
        "DeBERTa-v3 — He et al. 2021, arxiv 2111.09543", _
        "SciFive — Phan et al. 2021, arxiv 2106.03598", _
        "MedNLI — Romanov & Shivade 2018", _
        "RRF — Cormack et al. 2009, SIGIR", _
        "LLM-as-judge — Zheng et al. 2023, arxiv 2306.05685", _
        "CoT prompting — Wei et al. 2022, arxiv 2201.11903", _
        "Full bibliography: docs/phase2_report.md §13")

    AddTitleSlide pres, "Questions?", _
        "Repository: github.com/<user>/trec_rag_2025" & vbCrLf & _
        "Full report: docs/phase2_report.md" & vbCrLf & _
        "Total spend: $1.77 + ~7.5 GPU-hours + 1 laptop"

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
