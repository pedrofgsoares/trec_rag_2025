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

if [[ -d "${DEST}/.git" ]]; then
    echo "[skip] starter kit already cloned at ${DEST}"
    (cd "${DEST}" && git fetch --quiet && git checkout --quiet "${REF}")
else
    echo "[clone] ${REPO_URL}@${REF} -> ${DEST}"
    git clone --depth 1 --branch "${REF}" "${REPO_URL}" "${DEST}"
fi
