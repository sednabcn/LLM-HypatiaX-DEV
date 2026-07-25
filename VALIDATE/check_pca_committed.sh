#!/usr/bin/env bash
# =============================================================================
# check_pca_committed.sh
#
# Verifies that exp1_pca / exp1b_pca / exp2_feynman_pca results are properly
# committed to the repository.  Run from REPO_ROOT.
#
# Usage:
#   bash check_pca_committed.sh [RESULTS_DIR]
#
# Default RESULTS_DIR: hypatiax/data/results
#
# Exit codes:
#   0 — all checks passed
#   1 — one or more checks failed (details printed inline)
# =============================================================================
set -euo pipefail

RESULTS_DIR="${1:-${RESULTS_DIR:-hypatiax/data/results}}"

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; RST='\033[0m'
ok()   { echo -e "${GRN}  [OK]${RST}  $*"; }
warn() { echo -e "${YLW}  [WARN]${RST} $*"; }
fail() { echo -e "${RED}  [FAIL]${RST} $*"; FAILURES=$((FAILURES+1)); }

FAILURES=0

# ── helpers ──────────────────────────────────────────────────────────────────

# Count real result JSONs in a directory (excludes meta files).
count_real_jsons() {
  local dir="$1"
  if [ ! -d "${dir}" ]; then
    echo 0
    return
  fi
  find "${dir}" -maxdepth 1 -name '*.json' \
    ! -name 'checkpoint*' \
    ! -name '*disclosure*' \
    ! -name '*summary*' \
    ! -name 'fixc3_baseline*' \
    2>/dev/null | wc -l
}

# Count how many of those real JSONs are tracked by git (committed).
count_committed_jsons() {
  local dir="$1"
  # git ls-files lists only tracked paths; count the non-meta ones
  { git ls-files --error-unmatch "${dir}" 2>/dev/null | \
    grep '\.json$' | \
    grep -v -E '(checkpoint|disclosure|summary|fixc3_baseline)' || true; } | \
    wc -l
}

# Check for files that are on disk but NOT yet committed (untracked or modified).
untracked_in_dir() {
  local dir="$1"
  { git status --porcelain "${dir}" 2>/dev/null | grep '^[?MA]' || true; } | wc -l
}

# =============================================================================
# PITFALL-1 GUARD: skip-guard uses any *.json, so a dir with only meta files
# can suppress the benchmark re-run.  We verify the real-JSON count directly.
# =============================================================================
check_pitfall1() {
  local exp="$1" dir="$2"
  local n_real n_any
  if [ -d "${dir}" ]; then
    n_any=$(find "${dir}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
  else
    n_any=0
  fi
  n_real=$(count_real_jsons "${dir}")
  if [ "${n_real}" -eq 0 ] && [ "${n_any}" -gt 0 ]; then
    fail "PITFALL-1 [${exp}]: directory has ${n_any} JSON(s) but 0 are real results — only meta files present. The skip guard would fire and suppress the benchmark run."
    echo "        Dir: ${dir}"
    echo "        Files: $(ls "${dir}"/*.json 2>/dev/null | xargs -I{} basename {} | tr '\n' ' ')"
  elif [ "${n_real}" -eq 0 ]; then
    fail "PITFALL-1 [${exp}]: no real result JSONs found in ${dir} — experiment has not produced output yet."
  else
    ok "PITFALL-1 [${exp}]: ${n_real} real result JSON(s) in ${dir}"
  fi
}

# =============================================================================
# PITFALL-2 GUARD: git diff --cached --quiet is silent when files are already
# committed (idempotent OK) OR when the benchmark produced nothing and git add
# staged nothing.  We distinguish by checking on-disk vs committed counts.
# =============================================================================
check_pitfall2() {
  local exp="$1" dir="$2"
  local n_real n_committed n_uncommitted
  n_real=$(count_real_jsons "${dir}")
  # git ls-files: count tracked result JSONs under this dir
  n_committed=$( { git ls-files "${dir}" 2>/dev/null | \
    grep '\.json$' | \
    grep -v -E '(checkpoint|disclosure|summary|fixc3_baseline)' || true; } | \
    wc -l )
  n_uncommitted=$((n_real - n_committed))

  if [ "${n_real}" -eq 0 ]; then
    fail "PITFALL-2 [${exp}]: 0 result JSONs on disk — commit step would report 'Nothing new to commit' while silently missing all results."
  elif [ "${n_committed}" -eq 0 ]; then
    fail "PITFALL-2 [${exp}]: ${n_real} result JSON(s) on disk but NONE are committed to git — commit step may have silently no-oped."
    echo "        Run: git add -f '${dir}'/ && git commit -m 'manual: ${exp} results'"
  elif [ "${n_uncommitted}" -gt 0 ]; then
    warn "PITFALL-2 [${exp}]: ${n_uncommitted} result JSON(s) on disk but not yet committed (${n_committed} committed, ${n_real} total on disk). A re-run may not have been committed yet."
  else
    ok "PITFALL-2 [${exp}]: ${n_committed} result JSON(s) committed to git (on-disk count matches: ${n_real})"
  fi
}

# =============================================================================
# PITFALL-3 GUARD: exp2_feynman_pca comparison table files missing.
# The generator may run successfully but produce no output, or may not be
# committed to the repo at all.  Both cases leave the CI commit incomplete.
# =============================================================================
check_pitfall3() {
  local pca4060_dir="${RESULTS_DIR}/comparison_results/feynman-tests/exp2_pca_4060"
  local generator_script="scripts/patches/generate_exp2_pca_comparison_table.py"
  local -a expected_exts=("tex" "csv" "md")
  local any_missing=0

  # Check generator script is committed
  if ! git ls-files --error-unmatch "${generator_script}" &>/dev/null; then
    fail "PITFALL-3: ${generator_script} is NOT tracked by git — the comparison table step will [ERROR] and the CI commit will lack .tex/.csv/.md artefacts."
    any_missing=1
  else
    ok "PITFALL-3: ${generator_script} is committed to git"
  fi

  # Check each output file exists on disk AND is committed
  for ext in "${expected_exts[@]}"; do
    local f="${pca4060_dir}/exp2_pca_comparison.${ext}"
    if [ ! -f "${f}" ]; then
      fail "PITFALL-3: ${f} not found on disk — comparison table was not generated."
      any_missing=1
    elif ! git ls-files --error-unmatch "${f}" &>/dev/null; then
      fail "PITFALL-3: ${f} exists on disk but is NOT committed — git add missed it."
      any_missing=1
    else
      ok "PITFALL-3: exp2_pca_comparison.${ext} committed"
    fi
  done
  [ "${any_missing}" -eq 0 ] && ok "PITFALL-3: all three comparison table artefacts (.tex .csv .md) committed"
}

# =============================================================================
# ADDITIONAL: verify required meta/summary files are committed for each exp
# =============================================================================
check_meta() {
  local exp="$1" dir="$2"
  shift 2
  local file
  for file in "$@"; do
    local fpath="${dir}/${file}"
    if [ ! -f "${fpath}" ]; then
      fail "META [${exp}]: ${file} not found on disk at ${fpath}"
    elif ! git ls-files --error-unmatch "${fpath}" &>/dev/null; then
      fail "META [${exp}]: ${file} exists on disk but NOT committed to git"
    else
      ok "META [${exp}]: ${file} committed"
    fi
  done
}

# =============================================================================
# MAIN
# =============================================================================
echo ""
echo "======================================================================="
echo "  PCA experiment commit verification"
echo "  RESULTS_DIR: ${RESULTS_DIR}"
echo "  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '(not a git repo)')  $(git rev-parse --short HEAD 2>/dev/null || true)"
echo "======================================================================="

EXP1_DIR="${RESULTS_DIR}/comparison_results/noise-noiseless/noiseless/defi_pca"
EXP1B_DIR="${RESULTS_DIR}/comparison_results/noise-noiseless/15_pca"
EXP2_DIR="${RESULTS_DIR}/comparison_results/feynman-tests/exp2_pca_4060"

echo ""
echo "── exp1_pca ─────────────────────────────────────────────────────────────"
check_pitfall1 "exp1_pca"  "${EXP1_DIR}"
check_pitfall2 "exp1_pca"  "${EXP1_DIR}"
check_meta     "exp1_pca"  "${EXP1_DIR}" \
  "split_protocol_disclosure.json" \
  "exp1_pca_summary.json"

echo ""
echo "── exp1b_pca ────────────────────────────────────────────────────────────"
check_pitfall1 "exp1b_pca" "${EXP1B_DIR}"
check_pitfall2 "exp1b_pca" "${EXP1B_DIR}"

echo ""
echo "── exp2_feynman_pca ─────────────────────────────────────────────────────"
check_pitfall1 "exp2_feynman_pca" "${EXP2_DIR}"
check_pitfall2 "exp2_feynman_pca" "${EXP2_DIR}"
check_meta     "exp2_feynman_pca" "${EXP2_DIR}" \
  "split_protocol_disclosure.json" \
  "exp2_pca_4060_summary.json"
# fixc3_baseline.json sits at RESULTS_DIR root, not inside exp2_pca_4060/
BASELINE="${RESULTS_DIR}/fixc3_baseline.json"
if [ ! -f "${BASELINE}" ]; then
  fail "META [exp2_feynman_pca]: fixc3_baseline.json not found at ${BASELINE}"
elif ! git ls-files --error-unmatch "${BASELINE}" &>/dev/null; then
  fail "META [exp2_feynman_pca]: fixc3_baseline.json exists but NOT committed"
else
  ok  "META [exp2_feynman_pca]: fixc3_baseline.json committed"
fi
check_pitfall3

# =============================================================================
echo ""
echo "======================================================================="
if [ "${FAILURES}" -eq 0 ]; then
  echo -e "${GRN}  ALL CHECKS PASSED — PCA results look correctly committed.${RST}"
  echo "======================================================================="
  exit 0
else
  echo -e "${RED}  ${FAILURES} CHECK(S) FAILED — see details above.${RST}"
  echo "======================================================================="
  exit 1
fi
