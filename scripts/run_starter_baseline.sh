#!/usr/bin/env bash
# Run the unmodified starter-kit baseline against the official 2025 input,
# producing a submission JSONL that the eval module can score.
#
# Assumes:
#   - external/starter-kit-2025/ exists (run scripts/vendor_starter_kit.sh)
#   - data/indexes/pubmed_bm25/ exists (run scripts/build_indexes.sh)
#   - data/topics/biogen2025_taskA_input.jsonl exists (operator-supplied)
#
# Task: 6.2

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STARTER_DIR="${REPO_ROOT}/external/starter-kit-2025"
INPUT="${REPO_ROOT}/data/topics/biogen2025_taskA_input.jsonl"
INDEX="${REPO_ROOT}/data/indexes/pubmed_bm25"
OUT_DIR="${REPO_ROOT}/runs/starter_baseline_$(date +%Y%m%d_%H%M%S)"

[[ -d "${STARTER_DIR}" ]] || { echo "missing ${STARTER_DIR}; run vendor_starter_kit.sh" >&2; exit 2; }
[[ -f "${INPUT}" ]] || { echo "missing ${INPUT}; place official 2025 input there" >&2; exit 2; }
[[ -d "${INDEX}" ]] || { echo "missing ${INDEX}; run build_indexes.sh" >&2; exit 2; }

mkdir -p "${OUT_DIR}"
echo "[starter] running baseline -> ${OUT_DIR}/submission.jsonl"

# The starter kit's CLI entry varies between revisions; assume the canonical
# `python -m starter_kit.baseline` form documented in their README. Adjust here
# if the upstream repo changes its CLI.
( cd "${STARTER_DIR}" && \
  PYTHONPATH=. python -m starter_kit.baseline \
      --input "${INPUT}" \
      --index "${INDEX}" \
      --output "${OUT_DIR}/submission.jsonl" )

echo "[done] ${OUT_DIR}/submission.jsonl"
