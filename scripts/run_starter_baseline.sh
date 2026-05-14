#!/usr/bin/env bash
# Run the unmodified starter-kit Task A baseline against the official 2025 input.
#
# The starter-kit's src/task_a.py:
#   * loads its index from ../data/indexes/pubmed_baseline_collection_jsonl
#     (we symlink that to data/indexes/pubmed_bm25 in vendor_starter_kit.sh)
#   * reads ../data/task_a.json (symlinked from data/topics/biogen2025_taskA_input.json)
#   * downloads razent/SciFive-large-Pubmed_PMC-MedNLI from HF on first run (~3 GB)
#   * writes ../data/task_a_output.json
#
# We copy that output into runs/starter_baseline_<ts>/ for archival and to
# feed our own evaluator.
#
# Task: 6.2

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STARTER_DIR="${REPO_ROOT}/external/starter-kit-2025"
INDEX_LINK="${STARTER_DIR}/data/indexes/pubmed_baseline_collection_jsonl"
INPUT_LINK="${STARTER_DIR}/data/task_a.json"
OUTPUT_JSON="${STARTER_DIR}/data/task_a_output.json"
OUT_DIR="${REPO_ROOT}/runs/starter_baseline_$(date +%Y%m%d_%H%M%S)"

[[ -e "${STARTER_DIR}" ]] || { echo "missing ${STARTER_DIR}; run vendor_starter_kit.sh first" >&2; exit 2; }
[[ -e "${INDEX_LINK}" ]] || { echo "index symlink missing: ${INDEX_LINK}" >&2; exit 2; }
[[ -e "${INPUT_LINK}" ]] || { echo "input symlink missing: ${INPUT_LINK}" >&2; exit 2; }

mkdir -p "${OUT_DIR}"
echo "[starter] running baseline (this loads SciFive-large; ~3 GB first time)"
echo "[starter] output will land at: ${OUTPUT_JSON}"

# The starter-kit script reads cwd-relative paths and runs at import time;
# invoking via -m would be cleaner but the file isn't a package, so we cd in.
(
  cd "${STARTER_DIR}/src" && \
  PYTHONPATH=. python task_a.py
)

cp "${OUTPUT_JSON}" "${OUT_DIR}/task_a_output.json"
echo "[done] ${OUT_DIR}/task_a_output.json"
