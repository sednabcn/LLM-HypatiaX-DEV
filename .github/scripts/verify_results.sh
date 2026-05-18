#!/usr/bin/env bash
# .github/scripts/verify_results.sh
# Runs verify_results.py + hash_lock.py after the main pipeline.
# Called by full-pipeline and slow-pipeline jobs.
#
# Required env:
#   GITHUB_WORKSPACE, ANTHROPIC_API_KEY (inherited from job env)
#
# FIX-VERIFY: verify_results.py derives its PATCHED_DATA_DIR from its own
# __file__ location (scripts/patches/), producing the path:
#   ${GITHUB_WORKSPACE}/scripts/hypatiax/data/patched/
# The env-var PATCHED_DATA_DIR was previously set to RESULTS_DIR which
# verify_results.py ignores when constructing benchmark-specific paths.
# This script now stages merged outputs into the expected patched tree
# BEFORE calling verify_results.py so the file-not-found warnings are
# resolved for every experiment whose merge step has already run.

set -euo pipefail

# Use RESULTS_BASE when set by ci_experiment.yml (FIX-VERIFY); otherwise
# default to the canonical repo-relative path so local runs still work.
RESULTS_DIR="${RESULTS_BASE:-${GITHUB_WORKSPACE}/hypatiax/data/results}"

# verify_results.py derives its patched base as:
#   Path(__file__).resolve().parent.parent / "hypatiax/data/patched"
# which equals ${GITHUB_WORKSPACE}/scripts/hypatiax/data/patched/
PATCHED_DIR="${GITHUB_WORKSPACE}/scripts/hypatiax/data/patched"

export PATCHED_DATA_DIR="${PATCHED_DIR}"
export VERIFY_RESULTS_DIR="${RESULTS_DIR}"
export FAST="0"

# ── Stage merged outputs → patched directory tree ──────────────────────────
# Maps each experiment's merge output (written by merge_shards.py) to the
# sub-path that verify_results.py expects inside PATCHED_DIR.
# Only copies if the source file/dir exists; missing experiments are skipped
# silently here (verify_results.py will still warn about them, as intended).

echo "── Stage merged results into patched tree ───────────────────────────────"
echo "   RESULTS_DIR : ${RESULTS_DIR}"
echo "   PATCHED_DIR : ${PATCHED_DIR}"

# exp1 — DeFi benchmark (§10.2–10.4)
# merge_shards writes _merged.json + _stats.json to the noiseless subdir.
# Stage _stats.json as benchmark_results.json so verify_results.py finds it.
EXP1_DIR="${RESULTS_DIR}/comparison_results/noise-noiseless/noiseless"
if [[ -f "${EXP1_DIR}/_stats.json" ]]; then
    mkdir -p "${PATCHED_DIR}/defi"
    cp "${EXP1_DIR}/_stats.json" "${PATCHED_DIR}/defi/benchmark_results.json"
    echo "   ✓ defi/benchmark_results.json staged (_stats.json)"
elif [[ -f "${EXP1_DIR}/_merged.json" ]]; then
    mkdir -p "${PATCHED_DIR}/defi"
    cp "${EXP1_DIR}/_merged.json" "${PATCHED_DIR}/defi/benchmark_results.json"
    echo "   ✓ defi/benchmark_results.json staged (_merged.json)"
fi

# exp2_feynman — Feynman benchmark (§10.7)
EXP2F_DIR="${RESULTS_DIR}/comparison_results/feynman-tests/exp2"
if [[ -f "${EXP2F_DIR}/_stats.json" ]]; then
    mkdir -p "${PATCHED_DIR}/feynman"
    cp "${EXP2F_DIR}/_stats.json" "${PATCHED_DIR}/feynman/benchmark_results.json"
    echo "   ✓ feynman/benchmark_results.json staged (_stats.json)"
elif [[ -f "${EXP2F_DIR}/_merged.json" ]]; then
    mkdir -p "${PATCHED_DIR}/feynman"
    cp "${EXP2F_DIR}/_merged.json" "${PATCHED_DIR}/feynman/benchmark_results.json"
    echo "   ✓ feynman/benchmark_results.json staged (_merged.json)"
fi

# exp1_ablation — Core-15 ablation (§10.6)
EXP1A_SRC="${RESULTS_DIR}/comparison_results/feynman-tests/exp1_ablation"
if [[ -d "${EXP1A_SRC}" ]]; then
    mkdir -p "${PATCHED_DIR}/exp1_ablation"
    cp -r "${EXP1A_SRC}/." "${PATCHED_DIR}/exp1_ablation/"
    echo "   ✓ exp1_ablation/ staged"
fi

# instability — Instability benchmark (§10.9)
# merge_shards._merge_instability_csvs() writes _stats.json with n_tasks.
# Stage _stats.json so find_result("instability","*.json") finds it, which
# is check_instability()'s primary path.  Stage _merged.csv as the CSV
# fallback in case _stats.json is absent.
INST_SRC="${RESULTS_DIR}/figures"
if [[ -f "${INST_SRC}/_stats.json" ]]; then
    mkdir -p "${PATCHED_DIR}/instability"
    cp "${INST_SRC}/_stats.json" "${PATCHED_DIR}/instability/_stats.json"
    echo "   ✓ instability/_stats.json staged"
fi
if [[ -f "${INST_SRC}/_merged.csv" ]]; then
    mkdir -p "${PATCHED_DIR}/instability"
    cp "${INST_SRC}/_merged.csv" "${PATCHED_DIR}/instability/_merged.csv"
    echo "   ✓ instability/_merged.csv staged"
fi

echo ""

echo "── Verify results against paper targets ─────────────────────────────"
python3 scripts/patches/verify_results.py --report

echo "── Hash-lock check ──────────────────────────────────────────────────"
python3 hypatiax/reproducibility/hash_lock.py --check

echo "✓ Verify + hash-lock passed"
