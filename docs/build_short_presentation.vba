' =========================================================================
'  BuildBioGENShortPresentation.vba
'  ------------------------------------------------------------------------
'  Short / talk-length companion to BuildBioGENPresentation.vba.
'
'  Target: a 15-slide ceiling for a ~15-minute course presentation of the
'  TREC BioGEN 2025 Task A work (docs/phase2_report.md). Structured around
'  a narrative arc — "how do we know a retrieval system is good?" — with
'  two epistemic corrections (pool bias, CoT prompt) and a meta-question
'  ("who watches the watchmen?") resolved by three-juror Krippendorff α.
'
'  Slide 5 places our Phase 1 directly against the full BioGEN 2025
'  leaderboard (Table 5 of Gupta et al. 2026, arXiv:2603.21582). Slide 13
'  repositions the methodological contribution against the organisers'
'  BioACE / Llama-3.3 framework and the code-availability gap (0 of 7
'  participants published code — we are the first publicly reproducible
'  Task A pipeline if submitted in 2026).
'
'  Slide count (15, hard cap):
'    1.  Title
'    2.  Framing question — "How do we know a retrieval system is good?"
'    3.  TREC BioGEN 2025 Task A — what & why
'    4.  Phase 1 baseline pipeline (five phases, one diagram)
'    5.  The first defeat — Phase 1 loses by 38.79 pp
'    6.  First correction — the ruler is bent (pool bias)
'    7.  Second correction — the judge needed reading time (CoT pivot)
'    8.  Honest comparison — §2.17 expanded qrels methodology
'    9.  Eight variants on the honest pool (dual-pool table)
'    10. Three query-side expansions all hurt (RM3, LLM-RM3, rewrite)
'    11. The third correction — who watches the watchmen? (Phase 2.5 + 2.6)
'    12. Headline findings (numbers in one slide)
'    13. What this work claims for the field
'    14. Methodological contributions + limitations
'    15. Closing — two corrections, three judges, one laptop
'
'  How to use:
'    1. Open PowerPoint, Alt+F11, Insert -> Module, paste this file, F5 on
'       BuildBioGENShortPresentation. The macro creates a new presentation
'       called "TREC BioGEN 2025 - Short.pptx" in memory; Save As to disk.
'    2. Helpers are local to this file (no dependency on the long deck's
'       module) so you can paste either one independently.
' =========================================================================

Option Explicit

' ----- Constants ---------------------------------------------------------
Const SHORT_TITLE_FONT As String = "Calibri"
Const SHORT_BODY_FONT As String = "Calibri"
Const SHORT_TITLE_SIZE As Single = 30
Const SHORT_BODY_SIZE As Single = 18
Const SHORT_TABLE_SIZE As Single = 12

Const SHORT_LAYOUT_TITLE As Integer = 1
Const SHORT_LAYOUT_TITLE_CONTENT As Integer = 2
Const SHORT_LAYOUT_BLANK As Integer = 12

' ----- Main routine ------------------------------------------------------
Public Sub BuildBioGENShortPresentation()
    Dim pres As Presentation
    Set pres = Application.Presentations.Add

    Application.DisplayAlerts = ppAlertsNone
    pres.PageSetup.SlideSize = ppSlideSizeOnScreen16x9

    ' 1 -- Title -----------------------------------------------------------
    AddShortTitleSlide pres, _
        "TREC BioGEN 2025 — Task A", _
        "Two Corrections, Three Judges, One Laptop" & vbCrLf & _
        "Pedro Soares — Information Retrieval, 2026"

    ' 2 -- Framing question (narrative opener) -----------------------------
    AddShortBulletSlide pres, "How do we know a retrieval system is good?", Array( _
        "Every IR result rests on an answer that's rarely interrogated: the leaderboard number", _
        "TREC measures retrieval quality through pool-based qrels — only judged PMIDs count", _
        "BioGEN 2025: 588 human judgements over 26.8M abstracts ≈ 0.0022% corpus coverage", _
        "This talk is structured around two epistemic corrections to that answer:", _
        "   First — ""our system loses by 40 pp"" → no, the ruler is bent (pool bias)", _
        "   Second — ""the LLM judge is too strict"" → no, the prompt was bent (CoT pivot)", _
        "And one open meta-question: who watches the watchmen? (multi-juror Krippendorff α)", _
        "The technical work below is in service of these three questions")

    ' 3 -- Task ------------------------------------------------------------
    AddShortBulletSlide pres, "TREC BioGEN 2025 Task A — what & why", Array( _
        "Corpus: 26.8M PubMed abstracts. 40 topics × pre-segmented answer sentences", _
        "Per sentence emit ≤3 supporting + ≤3 contradicting PMIDs", _
        "Per-cell precision / recall / F1, macro over cells, Strict + Relaxed", _
        "Anchor: published 2025 organisers' baseline = Sup F1 44.34 / Con F1 4.67 (Strict)", _
        "Constraint: ONE WSL2 laptop — 12 GB RAM, Quadro T1000 4 GB VRAM, 37 GB index", _
        "Sits at the intersection of IR + NLI + biomedical NLP + evaluation methodology")

    ' 4 -- Phase 1 pipeline ------------------------------------------------
    AddShortBulletSlide pres, "Phase 1 baseline — five phases, sequential model loading", Array( _
        "1. Pyserini BM25 (k1=0.9, b=0.4): top-100 support + top-1000 contradict", _
        "2. MedCPT-Cross-Encoder rerank (Jin 2023, biomedical PubMed-tuned)", _
        "3. DeBERTa-v3-MNLI-FEVER-ANLI entailment scoring (support path)", _
        "3'. scispaCy segmentation + NegEx + 23-cue regex pre-filter (1.9M → 83k pairs)", _
        "4. DeBERTa-MNLI contradiction-probability scoring (contradict path)", _
        "5. Max-pool aggregation + selection (cap=3, τ=0.5, dedup, exclude existing)", _
        "Hand-off via parquet between phases; 4 GB VRAM = ONE transformer resident at a time")

    ' 5 -- The first defeat — in the context of the full BioGEN 2025 field
    AddShortResultsSlide pres, "The first defeat — Phase 1 vs the BioGEN 2025 field", _
        Array("Team / Run", "Sup F1", "Con F1", "Comment"), _
        Array( _
            Array("CLaC LLM_NLI_BM25", "67.74", "4.57", "top Sup (LLM in decision)"), _
            Array("InfoLab task_a_run6_A", "67.23", "14.15", "top combined"), _
            Array("InfoLab task_a_run2", "53.41", "15.67", "top Con F1"), _
            Array("GEHC-HTIC gehc_htic_task_a", "53.53", "8.57", "arXiv 2603.17580"), _
            Array("polito scifive-ft", "55.81", "4.79", "SciFive-MedNLI"), _
            Array("dal emotional_prompt", "55.53", "1.20", "Llama-3-70B prompt"), _
            Array("Baseline TEST (organisers')", "44.34", "4.67", "starter-kit anchor"), _
            Array("uniud run1_no-rerank_sparse", "39.10", "1.87", "—"), _
            Array("Our Phase 1", "5.55", "0.52", "gap = 38.79 / 4.15"))

    ' 6 -- First correction: pool bias (retitled) -------------------------
    AddShortBulletSlide pres, "First correction — the ruler is bent (pool bias)", Array( _
        "TREC pooled qrels (Voorhees 1998): only PMIDs in the pool get human judgement", _
        "2025 BioGEN pool was built from the organisers' baseline picks — small, self-referential", _
        "Phase 1 emits 1124 distinct PMIDs → ~30 in the 588-triple pool → ~1094 unjudged → false-positive", _
        "The 38.79 pp gap is not algorithmic weakness — it's a measurement artefact", _
        "Mechanism diagnosed; the fix is to extend the ruler, not to rebuild the system", _
        "Fix: LLM-as-judge expanded qrels, validated against the human pool ≥ 0.85 macro-w-F1", _
        "Methodology: Zheng et al. 2023 MT-Bench; TREC Health Misinformation 2022 used 0.80")

    ' 7 -- Second correction: CoT pivot (retitled) ------------------------
    AddShortBulletSlide pres, "Second correction — the judge needed reading time (CoT)", Array( _
        "Strict JSON prompt: gpt-4o-mini = 0.7497, gpt-4o = 0.7443 — gate FAILS (≥ 0.85)", _
        "171/549 human-Supports → judge-Neutral. GPT-4o is WORSE than mini = not a scale problem", _
        "J-curve case: LLM has the medical knowledge — strict JSON gives it no surface to reason", _
        "Domain-expert review of 12 disagreements overturned the obvious quantitative interpretation", _
        "CoT prompt: 2-3 sentence reasoning BEFORE the label, max_tokens 80 → 300", _
        "Same gpt-4o-mini, same triples: 0.7497 → 0.8944 macro-w-F1 — gate PASS, cost $0.080", _
        "Lesson: when an experiment hits an unexpected ceiling, read concrete cases FIRST")

    ' 8 -- Expanded pool methodology --------------------------------------
    AddShortBulletSlide pres, "Honest comparison — §2.17 expanded qrels", Array( _
        "§2.16: rejudge Phase 1's 1074 novel triples with gpt-4o-mini --prompt cot", _
        "  → 605 Supports + 104 Contradicts. Cost $0.149, ~6 min", _
        "Problem: phase2_no_rerank dropped 27 pp on §2.16 pool — pool was Phase-1-shaped (circular)", _
        "§2.17: rejudge BM25 top-30 per (qa_id, sentence_id) cell across both retrieval paths", _
        "  5169 new classifications, $0.704, ~16 min, --max-concurrent 8", _
        "Final expanded qrels: 588 human + 4170 LLM = 4 758 positives, 3.7× human pool", _
        "Dual-pool eval: --qrels-pool {official, expanded} + --source {human, llm, any}")

    ' 9 -- Variants results table -----------------------------------------
    AddShortResultsSlide pres, "Eight variants on the honest pool (Strict, §2.17)", _
        Array("Variant", "Off Sup", "Off Con", "Exp Sup", "Exp Con"), _
        Array( _
            Array("starter_baseline (organisers')", "44.34", "4.21", "16.55", "5.34"), _
            Array("phase1_baseline", "5.55", "0.52", "16.43", "12.01"), _
            Array("phase2_allow_existing", "5.55", "0.52", "16.94", "12.01"), _
            Array("phase2_no_rerank", "6.52", "0.52", "15.35", "11.75"), _
            Array("phase2_no_negex", "5.55", "2.65", "16.33", "8.06"), _
            Array("phase2_scifive_large", "5.55", "1.04", "16.43", "5.85"), _
            Array("phase2_bm25_rm3", "3.92", "0.26", "8.97", "5.26"), _
            Array("phase2_bm25_rm3_llm_filtered", "4.03", "0.52", "9.89", "12.01"), _
            Array("phase2_bm25_llm_rewrite", "5.29", "0.52", "10.65", "6.03"))

    ' 10 -- Negative result: query expansion ------------------------------
    AddShortBulletSlide pres, "Three query-side expansions all hurt — converging negative", Array( _
        "Phase 1 (no expansion): expanded Sup 16.43 — the baseline to beat", _
        "phase2_bm25_rm3 (blind RM3, Lavrenko & Croft 2001): 8.97 (-7.46 pp) — RM3 HURTS on biomed", _
        "phase2_bm25_rm3_llm_filtered (Mackie 2023, Pal 2020): 9.89 (+0.92 pp on blind, still -6.54)", _
        "phase2_bm25_llm_rewrite (3 PubMed-style rewrites + RRF fusion): 10.65 (still -5.78 pp)", _
        "Mechanism: claim-length queries (question + sentence) are already specific", _
        "Top-k BM25 hits are topically related but typically not evidence-bearing", _
        "Generalisation: query expansion — lexical OR LLM-curated — is the wrong intervention here")

    ' 11 -- The third correction (merged Phase 2.5 + 2.6) ----------------
    AddShortBulletSlide pres, "Third correction — who watches the watchmen?", Array( _
        "Meta-question: how do we know OUR ruler (the LLM judge) is reading right?", _
        "Phase 2.5 — second judge Llama-3.3-70B (HF Providers, $2.47): Jaccard 0.93/0.12 sup/con", _
        "  Llama emits 6.3× fewer Contradicts than mini — judges DISAGREE precisely on the small class", _
        "  Two-judge intersection on Contradicts: 363 → 43 positives (88% drop)", _
        "  Open: is mini over-emitting, OR is Llama over-stripping? Two judges cannot tell", _
        "Phase 2.6 — third judge Qwen2.5-72B (different org, different training data, $4.00)", _
        "  Krippendorff α: mini↔Llama 0.12, mini↔Qwen 0.20, Llama↔Qwen 0.60 (SUBSTANTIAL)", _
        "Resolution: mini-cot is the OUTLIER. Independent jurors agree → 31-positive intersection", _
        "Three-judge intersection holds: no_negex 3.70 vs Phase 1 1.12 (structural Phase 2 win survives)")

    ' 12 -- Headline findings ---------------------------------------------
    AddShortBulletSlide pres, "Headline findings — the numbers in one slide", Array( _
        "Pool bias contribution to Phase 1's apparent 38.79 pp gap: ~10 pp (NOT 40)", _
        "Published 44.34 baseline is inflated by 27.8 pp of pool overlap with itself", _
        "Honest comparable on expanded pool: starter 16.55 vs Phase 1 16.43 (≈ tie on Sup)", _
        "Phase 1 contradict 12.01 vs starter 5.34 on expanded pool — Phase 1 wins (>2×)", _
        "Three-judge intersection: no_negex 3.70 vs Phase 1 1.12 (3.3×, structural finding survives)", _
        "Negative: query expansion — lexical OR LLM-curated — fails on claim-length biomedical queries", _
        "Methodological: CoT prompt is essential; Krippendorff α is the right judge-agreement metric")

    ' 13 -- Field positioning + reproducibility opportunity ---------------
    AddShortBulletSlide pres, "What this work claims for the field", Array( _
        "Organisers ALREADY use LLM-as-judge (BioACE / Llama-3.3, single backend, no CI)", _
        "  → our novelty is not LLM-as-judge; it's the DEFENSIBILITY infrastructure around it", _
        "Pool construction (overview §6): 10 topics, 244 PMIDs, ONE top-priority run per team", _
        "  = the structural mechanism behind the 27.79 pp baseline inflation we diagnose", _
        "Multi-juror Krippendorff α across different organisations is the right agreement metric", _
        "Conservative intersection > liberal single-judge pool for cross-system claims", _
        "0 of 7 BioGEN 2025 teams published code → if we submit in 2026 we are the", _
        "  FIRST publicly reproducible Task A pipeline. The infra IS the contribution.", _
        "Total bill: $5.15 + ~22 GPU-hours + ONE consumer laptop")

    ' 14 -- Contributions + limitations (folded) -------------------------
    AddShortBulletSlide pres, "Methodological contributions + limitations", Array( _
        "+ Pool-bias diagnosis quantified (27.79 pp inflation) — confirmed by overview §6 pool spec", _
        "+ LLM-as-judge defensibility infra beyond BioACE: 3 backends + α + bootstrap CI + held-out ECE", _
        "+ Dual-pool + intersection-pool reporting (recovers anchor, exposes bias, tightens claims)", _
        "+ Eight variants, three independent query-side falsifications, cell-level bootstrap CIs", _
        "+ First publicly reproducible Task A pipeline if submitted to 2026 (0 of 7 in 2025)", _
        "− LLM-in-decision gap vs CLaC / GEHC-HTIC / dal — our Sup F1 is structurally capped", _
        "  Phase 4 (phase2_llm_decision) would close this; expected +10 pp expanded Sup", _
        "− No team-level head-to-head possible: cannot re-score CLaC's 67.74 on §2.17 (no code)", _
        "− Contradicts class small-sample post-intersection (31 positives over 313 cells)")

    ' 15 -- Closing narrative (was: Questions) ---------------------------
    AddShortTitleSlide pres, "Two corrections, three judges, one laptop.", _
        "We started with a 40-point defeat and ended with a competitive system." & vbCrLf & _
        "We didn't change the team — we checked the scoreboard, twice." & vbCrLf & _
        "Full report: docs/phase2_report.md  |  ~$5.15 + ~22 GPU-hours + 1 laptop"

    Application.DisplayAlerts = ppAlertsAll

    MsgBox "Built " & pres.Slides.Count & " slides (target ≤ 15).", _
        vbInformation, "Done"
End Sub

' =========================================================================
'  Helper routines (prefixed Short* to avoid clashing with the long deck's
'  module if both are pasted into the same VBProject).
' =========================================================================

Private Sub AddShortTitleSlide(pres As Presentation, title As String, subtitle As String)
    Dim s As Slide
    Set s = pres.Slides.Add(pres.Slides.Count + 1, SHORT_LAYOUT_TITLE)
    With s.Shapes.Title.TextFrame2.TextRange
        .Text = title
        .Font.Name = SHORT_TITLE_FONT
        .Font.Size = 40
        .Font.Bold = msoTrue
    End With
    If s.Shapes.Count >= 2 Then
        With s.Shapes(2).TextFrame2.TextRange
            .Text = subtitle
            .Font.Name = SHORT_BODY_FONT
            .Font.Size = 20
        End With
    End If
End Sub

Private Sub AddShortBulletSlide(pres As Presentation, title As String, bullets As Variant)
    Dim s As Slide
    Set s = pres.Slides.Add(pres.Slides.Count + 1, SHORT_LAYOUT_TITLE_CONTENT)

    With s.Shapes.Title.TextFrame2.TextRange
        .Text = title
        .Font.Name = SHORT_TITLE_FONT
        .Font.Size = SHORT_TITLE_SIZE
        .Font.Bold = msoTrue
    End With

    Dim placeholder As Shape
    Set placeholder = GetShortContentPlaceholder(s)
    If placeholder Is Nothing Then
        Set placeholder = s.Shapes.AddTextbox( _
            msoTextOrientationHorizontal, 50, 110, 860, 400)
    End If

    Dim tr As TextRange2
    Set tr = placeholder.TextFrame2.TextRange
    tr.Text = ShortJoinArray(bullets, vbCrLf)
    tr.Font.Name = SHORT_BODY_FONT
    tr.Font.Size = SHORT_BODY_SIZE
    tr.ParagraphFormat.Bullet.Type = msoBulletUnnumbered
End Sub

Private Sub AddShortResultsSlide( _
    pres As Presentation, title As String, headers As Variant, rows As Variant _
)
    Dim s As Slide
    Set s = pres.Slides.Add(pres.Slides.Count + 1, SHORT_LAYOUT_TITLE_CONTENT)
    With s.Shapes.Title.TextFrame2.TextRange
        .Text = title
        .Font.Name = SHORT_TITLE_FONT
        .Font.Size = SHORT_TITLE_SIZE
        .Font.Bold = msoTrue
    End With

    Dim ph As Shape
    Set ph = GetShortContentPlaceholder(s)
    If Not ph Is Nothing Then ph.Delete

    Dim n_cols As Integer, n_rows As Integer
    n_cols = UBound(headers) - LBound(headers) + 1
    n_rows = UBound(rows) - LBound(rows) + 2

    Dim tbl As Shape
    Set tbl = s.Shapes.AddTable(n_rows, n_cols, _
        Left:=30, Top:=110, Width:=900, Height:=380)

    Dim col As Integer, row As Integer
    For col = 0 To n_cols - 1
        With tbl.Table.Cell(1, col + 1).Shape.TextFrame2.TextRange
            .Text = headers(LBound(headers) + col)
            .Font.Name = SHORT_BODY_FONT
            .Font.Size = SHORT_TABLE_SIZE + 2
            .Font.Bold = msoTrue
        End With
    Next col

    For row = 0 To UBound(rows) - LBound(rows)
        Dim r As Variant
        r = rows(LBound(rows) + row)
        For col = 0 To n_cols - 1
            With tbl.Table.Cell(row + 2, col + 1).Shape.TextFrame2.TextRange
                .Text = CStr(r(LBound(r) + col))
                .Font.Name = SHORT_BODY_FONT
                .Font.Size = SHORT_TABLE_SIZE
            End With
        Next col
    Next row
End Sub

Private Function GetShortContentPlaceholder(s As Slide) As Shape
    Dim sh As Shape
    For Each sh In s.Shapes
        If sh.Type = msoPlaceholder Then
            If sh.PlaceholderFormat.Type <> ppPlaceholderTitle And _
               sh.PlaceholderFormat.Type <> ppPlaceholderCenterTitle Then
                Set GetShortContentPlaceholder = sh
                Exit Function
            End If
        End If
    Next sh
    Set GetShortContentPlaceholder = Nothing
End Function

Private Function ShortJoinArray(arr As Variant, sep As String) As String
    Dim out As String
    Dim i As Long
    For i = LBound(arr) To UBound(arr)
        If i > LBound(arr) Then out = out & sep
        out = out & CStr(arr(i))
    Next i
    ShortJoinArray = out
End Function
