#!/usr/bin/env bash
# Vendor the official BioGen 2025 starter kit under external/starter-kit-2025/.
# Used by run_starter_baseline.sh to reproduce the published baseline numbers.
#
# Task: 6.1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/external/starter-kit-2025"
REPO_URL="${STARTER_KIT_URL:-https://github.com/trec-biogen/starter-kit-2025.git}"
REF="${STARTER_KIT_REF:-main}"

mkdir -p "${REPO_ROOT}/external"

LOCAL_ZIP="${REPO_ROOT}/../../../mnt/c/Users/up746872/Downloads/starter-kit-2025-master.zip"
LOCAL_ZIP="$(realpath -m "${LOCAL_ZIP}" 2>/dev/null || true)"

if [[ -d "${DEST}/src" ]]; then
    echo "[skip] starter kit already present at ${DEST}"
elif [[ -f "${LOCAL_ZIP}" ]]; then
    echo "[unzip] ${LOCAL_ZIP} -> ${DEST}"
    tmp=$(mktemp -d)
    unzip -q "${LOCAL_ZIP}" -d "${tmp}"
    mv "${tmp}/starter-kit-2025-master" "${DEST}"
    rm -rf "${tmp}"
else
    echo "[clone] ${REPO_URL}@${REF} -> ${DEST}"
    git clone --depth 1 --branch "${REF}" "${REPO_URL}" "${DEST}"
fi

# Wire the starter-kit's hard-coded paths to our project layout.
mkdir -p "${DEST}/data/indexes"
ln -sfn "${REPO_ROOT}/data/indexes/pubmed_bm25" \
        "${DEST}/data/indexes/pubmed_baseline_collection_jsonl"
if [[ -f "${REPO_ROOT}/data/topics/biogen2025_taskA_input.json" ]]; then
    ln -sfn "${REPO_ROOT}/data/topics/biogen2025_taskA_input.json" \
            "${DEST}/data/task_a.json"
    echo "[link] ${DEST}/data/task_a.json -> data/topics/biogen2025_taskA_input.json"
fi
echo "[link] ${DEST}/data/indexes/pubmed_baseline_collection_jsonl -> data/indexes/pubmed_bm25"
