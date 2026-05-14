#!/usr/bin/env bash
# Download the official BioGen 2025 PubMed document collection.
# Set $BIOGEN_CORPUS_URL to the URL published by the TREC BioGEN organisers
# (see https://github.com/trec-biogen/starter-kit-2025 for the canonical link).
#
# Task: 3.1, 3.2

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# RAW_DIR can be overridden, e.g. for a Windows-mounted drive on WSL:
#   BIOGEN_RAW_DIR=/mnt/c/Users/<you>/Downloads/TREC bash scripts/download_pubmed.sh
# Recommended: keep raw corpus on /mnt/c if disk-space-constrained, but keep
# data/interim and data/indexes on native WSL FS for indexing throughput.
RAW_DIR="${BIOGEN_RAW_DIR:-${REPO_ROOT}/data/raw/pubmed_baseline}"
ARCHIVE="${RAW_DIR}/biogen-2025-document-collection.zip"

EXPECTED_DOCS=26805982
TOLERANCE_PCT="0.1"

mkdir -p "${RAW_DIR}"

URL="${BIOGEN_CORPUS_URL:-}"
if [[ -z "${URL}" ]]; then
    echo "ERROR: set BIOGEN_CORPUS_URL to the published corpus URL." >&2
    echo "       See the starter-kit repo for the canonical link." >&2
    exit 2
fi

if [[ -f "${ARCHIVE}" ]]; then
    echo "[skip] archive already present: ${ARCHIVE}"
else
    echo "[download] ${URL} -> ${ARCHIVE}"
    curl -L --fail --retry 3 --retry-delay 5 -o "${ARCHIVE}" "${URL}"
fi

echo "[extract] ${ARCHIVE} -> ${RAW_DIR}"
unzip -n -q "${ARCHIVE}" -d "${RAW_DIR}"

# 3.2 — doc count sanity check. The BioGen corpus is distributed as Pyserini
# JsonCollection JSONL (one {"id","contents"} record per line), so the count
# is just the total line count across all .jsonl files under RAW_DIR.
echo "[verify] counting documents (this may take a few minutes)..."
JSONL_COUNT=$(find "${RAW_DIR}" -type f -name '*.jsonl' | wc -l)
if [[ "${JSONL_COUNT}" -eq 0 ]]; then
    echo "ERROR: no .jsonl files found under ${RAW_DIR}" >&2
    exit 2
fi
ACTUAL=$(find "${RAW_DIR}" -type f -name '*.jsonl' -print0 | xargs -0 cat | wc -l)
echo "[verify] actual=${ACTUAL} expected=${EXPECTED_DOCS}  (across ${JSONL_COUNT} jsonl files)"

python - <<PY
actual = ${ACTUAL}
expected = ${EXPECTED_DOCS}
tol = expected * ${TOLERANCE_PCT} / 100.0
delta = abs(actual - expected)
if delta > tol:
    raise SystemExit(
        f"FAIL: doc count delta {delta} exceeds {tol:.0f} ({TOLERANCE_PCT}% of {expected})"
    )
print(f"OK: doc count within ±${TOLERANCE_PCT}% of {expected}")
PY
