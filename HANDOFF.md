# Phase-1 Handoff — what remains

After this session, 42 of 58 tasks in `openspec/changes/add-baseline-pipeline/tasks.md`
are done. All Python modules, configs, scripts, and CI-safe tests are committed.
The remaining 16 tasks all need either (a) sudo/host-Windows access, (b) the full
PubMed corpus, or (c) GPU + downloaded models — none of which fit in a session.

Run order (top to bottom; later steps assume earlier ones passed):

| Task | Command / Action | Wall-clock | Gate? |
|---|---|---:|---|
| 1.1, 1.2 | Bump WSL2 RAM (see [`SETUP.md`](SETUP.md) §1.1) | restart | hard — preflight fails below 11 GiB |
| 1.3 | `sudo apt install openjdk-21-jdk-headless` | <5 min | |
| 1.4 | pyenv 3.11 + `uv venv` + `uv pip install -e .` | ~10 min | |
| 2.3 | `uv pip install <scispacy wheel>` (SETUP.md §2.3) | <5 min | |
| 3.5 | `BIOGEN_CORPUS_URL=… bash scripts/download_pubmed.sh` | 30–60 min download + 1–2 h parse | |
| 4.2 | `bash scripts/build_indexes.sh` (background overnight) | ~12 h | |
| 6.5 | `bash scripts/baseline_check.sh` | hours | **HARD GATE** — ±2 F1 of (44.34, 4.67); do not proceed past on failure |
| 7.5 | Run pipeline on fixture (`+paths.topics=tests/fixtures/mini_input.jsonl`); check `nli_support.parquet` top-1 entailment > 0.5 | 5–10 min | |
| 8.6 | Open `runs/<id>/negation_audit.jsonl`, manually review 50 sampled rows, write `runs/<id>/negation_audit.md` | ~30 min | |
| 9.5 | Pipeline run on fixture passes `validate()` (already done by orchestrator) | covered by 7.5 | |
| 10.4 | `python -m trec_biogen.pipeline.run_task_a` against the full 2025 input | 6–10 h | |
| 10.5 | Open `runs/<id>/report.md`; compare to published Table 5 | <10 min | |
| 12.1 | Verify `metrics_2024.json` + `metrics_2025.json` contain all six (P,R,F1)×(strict,relaxed)×(support,contradict) numbers | <2 min | |
| 12.2 | If `phase1_pass` failed: write `reports/phase1_gap_analysis.md` from the template; enumerate which thresholds missed and the proposed Phase-2 mitigation | 30–60 min | |
| 12.3 | `git tag phase1-baseline && /opsx:archive add-baseline-pipeline` | <5 min | |

## Hard gates (per design)

1. **WSL2 RAM ≥ 11 GiB** — [`preflight.run()`](src/trec_biogen/pipeline/preflight.py) raises if not.
2. **Baseline replication within ±2 F1** — [`scripts/baseline_check.sh`](scripts/baseline_check.sh). Do not proceed past on failure (design D10).
3. **Phase-1 gate** — `Supports F1 ≥ 60` AND `Contradicts F1 ≥ 10`, strict, 2025 qrels. If missed, write gap analysis before tagging.

## Quick-start (operator)

```bash
# 1. Bump RAM (Windows host), restart WSL, then in WSL:
sudo apt install -y openjdk-21-jdk-headless
curl https://pyenv.run | bash && exec "$SHELL"
pyenv install 3.11.9 && pyenv local 3.11.9
uv venv --python 3.11.9 && source .venv/bin/activate
uv pip install -e .
uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

# 2. Verify (this session's code) passes CI-safe tests:
pytest -q

# 3. Long-running operator gates:
export BIOGEN_CORPUS_URL=<url from starter-kit-2025 README>
bash scripts/download_pubmed.sh
bash scripts/build_indexes.sh        # overnight
bash scripts/vendor_starter_kit.sh
bash scripts/baseline_check.sh       # HARD GATE
python -m trec_biogen.pipeline.run_task_a
```

Every run writes to `runs/<id>/` and is fully self-describing
(`metadata.yaml`, `log.jsonl`, all intermediate Parquets, `submission.jsonl`,
`metrics_*.json`, `report.md`).
