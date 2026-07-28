#!/usr/bin/env bash
# .github/scripts/build_pipeline_flags.sh
# Translates workflow_dispatch inputs into run_all_checkpoint.py CLI flags.
#
# Environment variables consumed (all optional, match workflow input names):
#   INPUT_ONLY_STEP        → --only <id>
#   INPUT_SEED             → --seed <n>    (skipped when value == '42')
#   INPUT_PYSR_TIMEOUT     → --pysr-timeout <n>  (skipped when value == '1100')
#   INPUT_SKIP_PAPER       → --skip-paper
#   INPUT_CONTINUE_ON_FAIL → --continue-on-fail  (always appended by full/slow jobs)
#   INPUT_SKIP_SLOW        → --skip-slow          (slow-pipeline: always false)
#   CHECKPOINT_FILE        → path checked for --resume (default: logs/pipeline_checkpoint.json)
#
# Exports: PIPELINE_FLAGS  (space-separated string, safe for direct shell interpolation)
#
# Usage in a workflow step:
#   source .github/scripts/build_pipeline_flags.sh
#   python3 run_all_checkpoint.py $PIPELINE_FLAGS

set -euo pipefail

CHECKPOINT_FILE="${CHECKPOINT_FILE:-logs/pipeline_checkpoint.json}"
FLAGS=()

# ── Resume ──────────────────────────────────────────────────────────────────
if [[ -f "${CHECKPOINT_FILE}" ]]; then
  FLAGS+=("--resume")
  echo "  ↩  Checkpoint found — adding --resume"
fi

# ── --only ──────────────────────────────────────────────────────────────────
if [[ -n "${INPUT_ONLY_STEP:-}" ]]; then
  FLAGS+=("--only" "${INPUT_ONLY_STEP}")
  echo "  ── --only ${INPUT_ONLY_STEP}"
fi

# ── --seed (skip default 42) ─────────────────────────────────────────────────
if [[ -n "${INPUT_SEED:-}" && "${INPUT_SEED}" != "42" ]]; then
  FLAGS+=("--seed" "${INPUT_SEED}")
  echo "  ── --seed ${INPUT_SEED}"
fi

# ── --pysr-timeout (skip paper default 1100) ──────────────────────────────
if [[ -n "${INPUT_PYSR_TIMEOUT:-}" && "${INPUT_PYSR_TIMEOUT}" != "1100" ]]; then
  FLAGS+=("--pysr-timeout" "${INPUT_PYSR_TIMEOUT}")
  echo "  ── --pysr-timeout ${INPUT_PYSR_TIMEOUT}"
fi

# ── --skip-paper ─────────────────────────────────────────────────────────────
if [[ "${INPUT_SKIP_PAPER:-false}" == "true" ]]; then
  FLAGS+=("--skip-paper")
  echo "  ── --skip-paper"
fi

# ── --continue-on-fail ───────────────────────────────────────────────────────
if [[ "${INPUT_CONTINUE_ON_FAIL:-false}" == "true" ]]; then
  FLAGS+=("--continue-on-fail")
  echo "  ── --continue-on-fail"
fi

# ── --skip-slow (only honoured for fast/smoke runs, not paper) ───────────────
if [[ "${INPUT_SKIP_SLOW:-false}" == "true" ]]; then
  FLAGS+=("--skip-slow")
  echo "  ── --skip-slow"
fi

PIPELINE_FLAGS="${FLAGS[*]:-}"
export PIPELINE_FLAGS
echo "PIPELINE_FLAGS=${PIPELINE_FLAGS}" >> "${GITHUB_ENV}"
echo "  ✓ PIPELINE_FLAGS='${PIPELINE_FLAGS}'"
