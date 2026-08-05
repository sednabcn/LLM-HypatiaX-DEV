#!/usr/bin/env bash
# =============================================================================
# move_results.sh — Move stray root-level result files into hypatiax/data/results/
#
# Run from the repo root.  Safe to run multiple times (skips files already in
# the right place).  Use --dry-run to preview without moving anything.
#
# Files covered (from root ls output):
#   exp1_ablation_checkpoint.json   → results/
#   exp1_ablation_results.json      → results/
#   exp1_ablation_table.tex         → results/tables/
#   exp1_instability_stats.json     → results/figures/
#   exp1_rf01_mannwhitney.json      → results/
#   provenance_map_exp1.json        → results/
#   defi_v3_*.json                  → results/         (exp1b outputs)
#   ablation_*.json                 → results/         (exp1 outputs)
#   hypatiax_defi_benchmark_v3*.json→ results/         (exp1 outputs)
#
# Additional rules added v2:
#   instability_extrapolation_v2.csv            → results/figures/
#   instability_extrapolation*.csv (all)        → results/figures/
#   exp2_all30_checkpoint.json                  → results/comparison_results/
#   noise_sweep_*.{json,csv} inside             → results/comparison_results/feynman-tests/noise-sweep/
#     noise-sweep/*/ subdirs (suppB flatten)
#   sample_complexity_*.{json,csv} inside       → results/comparison_results/feynman-tests/sample-complexity/
#     feynman-tests/*/ subdirs (suppB_sc flatten)
#   tree_results.txt at results root            → warned and skipped
# =============================================================================
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/hypatiax/data/results}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY_RUN=true; shift ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
moved=0; skipped=0; warned=0

move() {
  local src="$1" dst_dir="$2"
  local dst="${dst_dir}/$(basename "$src")"
  if [[ ! -f "$src" ]]; then
    return 0  # already gone or never existed
  fi
  if [[ -f "$dst" ]]; then
    echo -e "${YELLOW}[SKIP]${NC}  $(basename "$src")  →  already at ${dst_dir#$REPO_ROOT/}"
    (( skipped++ )) || true
    return 0
  fi
  if [[ "$DRY_RUN" == true ]]; then
    echo -e "${GREEN}[DRY]${NC}   $(basename "$src")  →  ${dst_dir#$REPO_ROOT/}/"
  else
    mkdir -p "$dst_dir"
    mv -v "$src" "$dst"
    echo -e "${GREEN}[MOVE]${NC}  $(basename "$src")  →  ${dst_dir#$REPO_ROOT/}/"
    (( moved++ )) || true
  fi
}

move_glob() {
  local pattern="$1" dst_dir="$2"
  local f
  for f in "${REPO_ROOT}"/${pattern}; do
    [[ -f "$f" ]] && move "$f" "$dst_dir"
  done
}

# Flatten all files matching a glob pattern found *anywhere under* a given
# directory tree into a single flat destination directory.
# Usage: flatten_into <search_root> <glob_pattern> <dst_dir>
flatten_into() {
  local search_root="$1" pattern="$2" dst_dir="$3"
  local f
  while IFS= read -r -d '' f; do
    # Skip files that are already directly inside dst_dir
    if [[ "$(dirname "$f")" == "$dst_dir" ]]; then
      continue
    fi
    move "$f" "$dst_dir"
  done < <(find "$search_root" -maxdepth 5 -name "$pattern" -type f -print0 2>/dev/null)
}

warn_stray() {
  local path="$1" reason="$2"
  if [[ -f "$path" ]]; then
    echo -e "${RED}[WARN]${NC}  $(basename "$path")  — ${reason}"
    (( warned++ )) || true
  fi
}

echo ""
echo "=== move_results.sh ==="
echo "  REPO_ROOT   : $REPO_ROOT"
echo "  RESULTS_DIR : $RESULTS_DIR"
[[ "$DRY_RUN" == true ]] && echo "  MODE        : DRY RUN (no files will be moved)"
echo ""

cd "$REPO_ROOT"

# ── Ensure destination directories exist ──────────────────────────────────────
if [[ "$DRY_RUN" == false ]]; then
  mkdir -p \
    "${RESULTS_DIR}" \
    "${RESULTS_DIR}/tables" \
    "${RESULTS_DIR}/figures" \
    "${RESULTS_DIR}/comparison_results" \
    "${RESULTS_DIR}/comparison_results/feynman-tests/noise-sweep" \
    "${RESULTS_DIR}/comparison_results/feynman-tests/sample-complexity"
fi

# ── exp1 outputs → results/ ───────────────────────────────────────────────────
move "${REPO_ROOT}/exp1_ablation_checkpoint.json"   "${RESULTS_DIR}"
move "${REPO_ROOT}/exp1_ablation_results.json"       "${RESULTS_DIR}"
move "${REPO_ROOT}/exp1_rf01_mannwhitney.json"       "${RESULTS_DIR}"
move_glob "ablation_*.json"                          "${RESULTS_DIR}"
move_glob "hypatiax_defi_benchmark_v3*.json"         "${RESULTS_DIR}"

# ── exp1 table → results/tables/ ─────────────────────────────────────────────
move "${REPO_ROOT}/exp1_ablation_table.tex"          "${RESULTS_DIR}/tables"

# ── instability outputs → results/figures/ ───────────────────────────────────
# Handles both the original name and any versioned variants (e.g. _v2).
move "${REPO_ROOT}/exp1_instability_stats.json"      "${RESULTS_DIR}/figures"
move_glob "instability_analysis.csv"                 "${RESULTS_DIR}/figures"
move_glob "instability_extrapolation*.csv"           "${RESULTS_DIR}/figures"   # covers _v2 and beyond

# ── exp1b outputs → results/ ─────────────────────────────────────────────────
move_glob "defi_v3_*.json"                           "${RESULTS_DIR}"
move_glob "*portfolio*variance*.json"                "${RESULTS_DIR}"

# ── provenance / audit → results/ ────────────────────────────────────────────
move "${REPO_ROOT}/provenance_map_exp1.json"         "${RESULTS_DIR}"

# ── exp2 top-level checkpoint → results/comparison_results/ ──────────────────
move "${RESULTS_DIR}/comparison_results/exp2_all30_checkpoint.json" \
     "${RESULTS_DIR}/comparison_results"   # no-op if already there
move "${REPO_ROOT}/exp2_all30_checkpoint.json" \
     "${RESULTS_DIR}/comparison_results"

# ── suppB flatten: noise-sweep per-equation subdirs → noise-sweep/ ───────────
# run_instability_suite.py (suppB step) writes files like:
#   comparison_results/feynman-tests/noise-sweep/<equation>/noise_sweep_*.{json,csv}
# Flatten them one level up so the directory is a flat collection.
NOISE_SWEEP_ROOT="${RESULTS_DIR}/comparison_results/feynman-tests/noise-sweep"
NOISE_SWEEP_DST="${NOISE_SWEEP_ROOT}"
flatten_into "${NOISE_SWEEP_ROOT}" "noise_sweep_*.json" "${NOISE_SWEEP_DST}"
flatten_into "${NOISE_SWEEP_ROOT}" "noise_sweep_*.csv"  "${NOISE_SWEEP_DST}"

# ── suppB_sc flatten: sample-complexity subdirs → sample-complexity/ ─────────
# Similar pattern for sample_complexity outputs produced alongside noise sweeps.
SAMPLE_COMPLEXITY_ROOT="${RESULTS_DIR}/comparison_results/feynman-tests"
SAMPLE_COMPLEXITY_DST="${SAMPLE_COMPLEXITY_ROOT}/sample-complexity"
flatten_into "${SAMPLE_COMPLEXITY_ROOT}" "sample_complexity_*.json" "${SAMPLE_COMPLEXITY_DST}"
flatten_into "${SAMPLE_COMPLEXITY_ROOT}" "sample_complexity_*.csv"  "${SAMPLE_COMPLEXITY_DST}"

# ── stray / unexpected files — warn but do not move ──────────────────────────
warn_stray "${RESULTS_DIR}/tree_results.txt" \
  "stray file — not a result artifact; remove manually if unneeded"

echo ""
if [[ "$DRY_RUN" == true ]]; then
  echo "=== Dry run complete — no files moved ==="
else
  echo "=== Done: ${moved} file(s) moved, ${skipped} already in place, ${warned} warning(s) ==="
fi
echo ""
