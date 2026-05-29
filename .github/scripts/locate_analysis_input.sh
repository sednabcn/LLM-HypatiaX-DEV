#!/usr/bin/env bash
# ==============================================================================
#  locate_analysis_input.sh
#  Single source of truth for locating analysis input across all workflows.
#
#  Called by:
#    ci_analysis.yml              — writes to $GITHUB_ENV
#    ci_pipeline_analysis.yml    — writes to $GITHUB_OUTPUT
#
#  Required env vars (set by the calling workflow step):
#    EXPERIMENT      e.g. exp1, exp1b, exp2_feynman, exp3b ...
#    RESULT_DIR      absolute or repo-relative path to the result directory
#    RESULT_SUBDIR   relative subdir under OUT_BASE (for merge_shards.py)
#    OUTPUT_TARGET   "env" (ci_analysis) | "output" (ci_pipeline_analysis)
#
#  Output (written to $GITHUB_ENV or $GITHUB_OUTPUT depending on OUTPUT_TARGET):
#    INPUT_MODE      merged | direct | shards
#    INPUT_JSON      path to single input file  (merged or direct mode)
#    SHARD_MANIFEST  path to manifest file       (shards mode)
#
#  Architecture:
#    MERGE_REQUIRED_EXPERIMENTS (merge_shards.py) — dynamically detected:
#      Fast path  — committed _merged.json exists → INPUT_MODE=merged
#      Fallback   — run merge_shards.py on committed shard/CSV files → INPUT_MODE=merged
#                   instability: CSV→_merged.json via _merge_instability_csvs()
#                   exp1b/exp3b: JSON shards → _merged.json via standard path
#    All others (REQUIRE_MERGE=false):
#      DIRECT     — exactly 1 shard file → INPUT_MODE=direct
#      SHARDS     — N>1 shard files      → INPUT_MODE=shards + manifest
#
#  NO inline benchmark merge for non-merge experiments.  This prevents
#  field-name mismatches (e.g. exp2/Feynman records don't have far-R²).
# ==============================================================================

set -euo pipefail

# ── Validate required env vars ────────────────────────────────────────────────
: "${EXPERIMENT:?EXPERIMENT must be set}"
: "${RESULT_DIR:?RESULT_DIR must be set}"
: "${RESULT_SUBDIR:?RESULT_SUBDIR must be set}"
OUTPUT_TARGET="${OUTPUT_TARGET:-env}"

# ── Helper: write a key=value to the correct GitHub output channel ────────────
emit() {
  local key="$1" val="$2"
  if [[ "$OUTPUT_TARGET" == "output" ]]; then
    echo "${key}=${val}" >> "$GITHUB_OUTPUT"
  else
    echo "${key}=${val}" >> "$GITHUB_ENV"
  fi
}

echo "=== RESULT DIRECTORY ==="
echo "$RESULT_DIR"

echo
echo "=== TREE ==="
if [[ -d "$RESULT_DIR" ]]; then
  find "$RESULT_DIR" -maxdepth 2 -type f | sort
else
  echo "Directory does not exist: $RESULT_DIR"
fi

echo
echo "=== DETERMINE INPUT MODE ==="

# ── Determine whether this experiment requires a merge ───────────────────────
# MERGE_REQUIRED_EXPERIMENTS in merge_shards.py is the single source of truth.
# Reading it here avoids duplicating the list in the shell script.
REQUIRE_MERGE=$(python3 -c "
import sys
sys.path.insert(0, '.github/scripts')
try:
    from merge_shards import MERGE_REQUIRED_EXPERIMENTS
    print('true' if '$EXPERIMENT' in MERGE_REQUIRED_EXPERIMENTS else 'false')
except Exception as e:
    # Fallback: known merge experiments hard-coded as a safety net
    print('true' if '$EXPERIMENT' in ('exp1b', 'exp3b', 'instability') else 'false',
          file=sys.stdout)
    print(f'::warning::Could not import MERGE_REQUIRED_EXPERIMENTS: {e}', file=sys.stderr)
")
echo "REQUIRE_MERGE=$REQUIRE_MERGE"

# ==============================================================================
#  MERGED MODE (exp1b / exp3b only)
# ==============================================================================

if [[ "$REQUIRE_MERGE" == "true" ]]; then

  echo
  echo "Merged mode activated"

  # ── Fast path: committed _merged.json (from ci_consolidate_experiment.yml) ──
  CANDIDATES=()
  while IFS= read -r path; do
    CANDIDATES+=("$path")
  done < <(
    find consolidated_artifact "$RESULT_DIR" \
      -type f \
      -name '_merged.json' \
      2>/dev/null \
      | sort
  )

  if [[ ${#CANDIDATES[@]} -gt 0 ]]; then
    INPUT_JSON="${CANDIDATES[0]}"
    emit "INPUT_MODE" "merged"
    emit "INPUT_JSON" "$INPUT_JSON"
    emit "SHARD_MANIFEST" ""
    echo
    echo "Selected merged input: $INPUT_JSON"
    exit 0
  fi

  # ── Fallback: run merge_shards.py against committed shard files ─────────────
  # For JSON-shard experiments (exp1b, exp3b): collects *.json files.
  # For CSV-only experiments (instability): collects *.csv files.
  # merge_shards.py reads EXP_CONFIG[exp_id].shard_globs so it always finds
  # the right files regardless of extension — we just need at least one file
  # present to confirm the directory isn't empty before invoking it.
  echo
  echo "No _merged.json found — falling back to merge_shards.py."

  mapfile -t SHARD_FILES < <(
    find "$RESULT_DIR" \
      -maxdepth 2 \
      -type f \
      \( -name '*.json' -o -name '*.csv' \) \
      ! -name '_*.json' \
      ! -name '_*.csv' \
      | sort
  )

  if [[ ${#SHARD_FILES[@]} -eq 0 ]]; then
    echo "::error::No shard files (*.json or *.csv) found in ${RESULT_DIR} and no _merged.json."
    echo "         Ensure workers have committed result files."
    exit 1
  fi

  echo "  Found ${#SHARD_FILES[@]} shard file(s) — merging via merge_shards.py..."

  python3 .github/scripts/merge_shards.py \
    --experiment    "$EXPERIMENT" \
    --input-root    "$RESULT_DIR" \
    --output-dir    "$RESULT_DIR" \
    --result-subdir "$RESULT_SUBDIR"

  emit "INPUT_MODE" "merged"
  emit "INPUT_JSON" "${RESULT_DIR}/_merged.json"
  emit "SHARD_MANIFEST" ""
  echo
  echo "Merge complete → ${RESULT_DIR}/_merged.json"
  exit 0

fi

# ==============================================================================
#  SHARD / DIRECT MODE (all other experiments)
# ==============================================================================

echo
echo "Shard mode activated"

# Collect ALL non-meta JSON files.
# - No *_shard*_run*.json exclusion: those ARE the final committed results
#   for many experiments (re-run worker naming pattern).
# - maxdepth 2: catches both flat layouts (depth 1) and one-level-nested
#   layouts like suppB's noise-sweep/noise-sweep/*.json.
mapfile -t SHARD_FILES < <(
  find "$RESULT_DIR" \
    -maxdepth 2 \
    -type f \
    -name '*.json' \
    ! -name '_*.json' \
    ! -name 'benchmark_results.json' \
    | sort
)

if [[ ${#SHARD_FILES[@]} -eq 0 ]]; then
  echo "::error::No shard JSON files found in ${RESULT_DIR}."
  echo "  Searched: ${RESULT_DIR}/**/*.json (maxdepth 2)"
  echo "  Excluded: _*.json, benchmark_results.json"
  exit 1
fi

N_SHARDS=${#SHARD_FILES[@]}

# ── DIRECT: single committed result file ─────────────────────────────────────
if [[ $N_SHARDS -eq 1 ]]; then
  echo "  DIRECT mode: single result file — ${SHARD_FILES[0]}"
  emit "INPUT_MODE" "direct"
  emit "INPUT_JSON" "${SHARD_FILES[0]}"
  emit "SHARD_MANIFEST" ""
  exit 0
fi

# ── SHARDS: N>1 flat/list-format shard files ─────────────────────────────────
# No inline benchmark merge here — non-merge experiments use their shard files
# directly so run_analysis.py receives the correct field layout for each exp.
MANIFEST="${RESULT_DIR}/_shard_manifest.txt"
printf '%s\n' "${SHARD_FILES[@]}" > "$MANIFEST"

echo "  SHARDS mode: ${N_SHARDS} file(s) → ${MANIFEST}"
cat "$MANIFEST"

emit "INPUT_MODE" "shards"
emit "INPUT_JSON" ""
emit "SHARD_MANIFEST" "$MANIFEST"
