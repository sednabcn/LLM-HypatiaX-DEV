#!/usr/bin/env bash
# .github/scripts/verify_results.sh
# Runs verify_results.py + hash_lock.py after the main pipeline.
# Called by full-pipeline and slow-pipeline jobs.
#
# Required env:
#   GITHUB_WORKSPACE, ANTHROPIC_API_KEY (inherited from job env)

set -euo pipefail

RESULTS_DIR="${GITHUB_WORKSPACE}/hypatiax/data/results"

export PATCHED_DATA_DIR="${RESULTS_DIR}"
export VERIFY_RESULTS_DIR="${RESULTS_DIR}"
export FAST="0"

echo "── Verify results against paper targets ─────────────────────────────"
python3 scripts/patches/verify_results.py --report

echo "── Hash-lock check ──────────────────────────────────────────────────"
python3 hypatiax/reproducibility/hash_lock.py --check

echo "✓ Verify + hash-lock passed"
