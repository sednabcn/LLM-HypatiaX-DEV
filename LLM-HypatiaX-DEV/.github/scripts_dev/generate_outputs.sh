#!/usr/bin/env bash
# .github/scripts/generate_outputs.sh
# Generates figures and LaTeX tables after a successful pipeline run.
# Called by full-pipeline and slow-pipeline jobs.
#
# Required env: GITHUB_WORKSPACE, ANTHROPIC_API_KEY (inherited from job env)

set -euo pipefail

RESULTS_DIR="${GITHUB_WORKSPACE}/hypatiax/data/results"

export FAST="0"
export TABLE_OUTDIR="${RESULTS_DIR}/tables"
export VERIFY_RESULTS_DIR="${RESULTS_DIR}"

echo "── Generate figures ─────────────────────────────────────────────────"
python3 figures/generate_figures.py --outdir "${RESULTS_DIR}/figures"

echo "── Generate tables ──────────────────────────────────────────────────"
python3 scripts/patches/generate_tables.py --outdir "${RESULTS_DIR}/tables"

echo "✓ Figures → ${RESULTS_DIR}/figures"
echo "✓ Tables  → ${RESULTS_DIR}/tables"
