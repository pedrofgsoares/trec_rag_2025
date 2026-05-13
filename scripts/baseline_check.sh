#!/usr/bin/env bash
# Reproduce the official BioGen 2025 starter baseline numbers within ±2 F1.
# Hard gate per design D10: do NOT proceed to optimisation until this exits 0.
#
# Published baseline (Table 5, official overview):
#   Supports F1   = 44.34   (strict, 2025 qrels)
#   Contradicts F1 = 4.67   (strict, 2025 qrels)
#
# Task: 6.4, 6.5

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QRELS="${REPO_ROOT}/data/qrels/biogen2025_taskA_qrels.jsonl"
TOL_F1="${BASELINE_TOL_F1:-2.0}"
EXPECTED_SUPPORT_F1="44.34"
EXPECTED_CONTRADICT_F1="4.67"

echo "[1/2] running starter baseline"
SUB=$(bash "${REPO_ROOT}/scripts/run_starter_baseline.sh" | tail -1 | awk '{print $2}')
[[ -f "${SUB}" ]] || { echo "submission not found: ${SUB}" >&2; exit 2; }

echo "[2/2] evaluating against ${QRELS}"
REPORT=$(python -m trec_biogen.eval.metrics --submission "${SUB}" --qrels "${QRELS}")
echo "${REPORT}"

python - <<PY
import json, sys
report = json.loads("""${REPORT}""")
strict = report["strict"]
sup_f1 = strict["support"]["F1"] * 100.0  # report is in [0,1]; published is in %
con_f1 = strict["contradict"]["F1"] * 100.0
exp_s, exp_c, tol = ${EXPECTED_SUPPORT_F1}, ${EXPECTED_CONTRADICT_F1}, ${TOL_F1}
ds, dc = abs(sup_f1 - exp_s), abs(con_f1 - exp_c)
print(f"support_F1  = {sup_f1:.2f}  (expected {exp_s}; |delta|={ds:.2f}; tol={tol})")
print(f"contradict_F1 = {con_f1:.2f}  (expected {exp_c}; |delta|={dc:.2f}; tol={tol})")
fail = (ds > tol) or (dc > tol)
sys.exit(1 if fail else 0)
PY
