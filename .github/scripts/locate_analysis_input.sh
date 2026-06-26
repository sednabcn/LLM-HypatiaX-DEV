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
#      Fast path  — committed _merged.json exists AND contains ≥1 record → INPUT_MODE=merged
#                   (FIX-EMPTY-MERGED-FASTPATH: a _merged.json with 0 records is rejected
#                    and falls through to merge_shards.py rather than producing a downstream
#                    FATAL: EMPTY DATASET.  Shape S files count per_equation entries across
#                    all sweep points; Shape A/B count top-level task-row keys.)
#      Fallback   — run merge_shards.py on committed shard files → INPUT_MODE=merged
#                   exp1b/exp1_ablation/exp3b: JSON shards → _merged.json via standard path
#                   suppB/suppB_sc: Shape S sweep shards → _merged.json via merge_sweep_files()
#    All others (REQUIRE_MERGE=false) — including "instability", which is CSV/figures-only
#    and has no task-row JSON to merge (run_analysis.py short-circuits for it directly):
#      DIRECT     — exactly 1 shard file → INPUT_MODE=direct
#      SHARDS     — N>1 shard files      → INPUT_MODE=shards + manifest
#
#  NO inline benchmark merge for non-merge experiments.  This prevents
#  field-name mismatches (e.g. exp2/Feynman records don't have far-R²).
#
#  METADATA FILE EXCLUSIONS (shard/direct mode):
#    The following filename patterns are excluded from the shard manifest
#    because they are metadata / disclosure / summary files that contain
#    no per-record experiment data.  validate_analysis_input.py uses the
#    same list (METADATA_FILENAME_PATTERNS) as its single source of truth.
#
#    *_disclosure.json         split_protocol_disclosure.json etc.
#    *_summary.json            exp2_pca_4060_summary.json etc.
#    *_pca_comparison.*        PCA comparison tables
#    *_comparison.*            generic comparison output files
#    ablation_paired.json      exp2_feynman_extrap ablation output
#    symbolic_equivalence*     symbolic equivalence report files
#
#    If a new metadata filename pattern is introduced, add it to BOTH:
#      • the METADATA_EXCLUSIONS array below
#      • METADATA_FILENAME_PATTERNS in validate_analysis_input.py
# ==============================================================================

set -euo pipefail

# ── Validate required env vars ────────────────────────────────────────────────
: "${EXPERIMENT:?EXPERIMENT must be set}"
: "${RESULT_DIR:?RESULT_DIR must be set}"
: "${RESULT_SUBDIR:?RESULT_SUBDIR must be set}"
OUTPUT_TARGET="${OUTPUT_TARGET:-env}"

# ── Self-heal known doubled-path bug (FIX-suppB-DOUBLED-PATH lineage) ────────
# Some callers of this script (e.g. an analysis workflow with a stale
# RESULT_SUBDIR/MAPPING entry) may still pass a RESULT_DIR whose final path
# segment is duplicated, e.g.:
#   .../comparison_results/feynman-tests/noise-sweep/noise-sweep
# instead of the canonical single-level path that run_noise_sweep_benchmark.py
# (and run_all.sh / ci_runner.yml) actually write to / expect:
#   .../comparison_results/feynman-tests/noise-sweep
# (see run_all.sh's FIX-suppB-DOUBLED-PATH comment for the full history —
# the benchmark script itself writes single-level; doubling has only ever
# been introduced by a caller appending RESULT_SUBDIR a second time.)
#
# If the directory as given does not exist, but stripping one duplicated
# trailing path segment resolves to a directory that DOES exist, use the
# corrected path instead of failing outright. This keeps locate_analysis_input.sh
# the single source of truth even when an upstream caller still carries the bug.
if [[ ! -d "$RESULT_DIR" ]]; then
  _rd_parent="$(dirname "$RESULT_DIR")"
  _rd_base="$(basename "$RESULT_DIR")"
  _rd_grandparent_base="$(basename "$_rd_parent")"
  if [[ "$_rd_base" == "$_rd_grandparent_base" && -d "$_rd_parent" ]]; then
    echo "::warning::RESULT_DIR has a duplicated trailing path segment ('${_rd_base}/${_rd_base}') — falling back to '${_rd_parent}'. Fix the caller's RESULT_SUBDIR/MAPPING entry for ${EXPERIMENT} so this doesn't recur."
    RESULT_DIR="$_rd_parent"
  fi
fi

# ── Metadata filename patterns excluded from the shard manifest ───────────────
# These are known non-record files committed alongside result shards.
# Keep in sync with METADATA_FILENAME_PATTERNS in validate_analysis_input.py.
METADATA_EXCLUSIONS=(
  '*_disclosure.json'
  '*_summary.json'
  '*_pca_comparison.json'
  '*_comparison.json'
  '*_comparison.csv'
  '*_comparison.md'
  'ablation_paired.json'
  'symbolic_equivalence*.json'
  'symbolic_equivalence*.csv'
  'symbolic_equivalence*.txt'
  'benchmark_results*.json'   # benchmark_results.json + _extrap + _legacy + _shard* variants
  '*_checkpoint_*.json'       # domain checkpoint files (pca4060_checkpoint_*, feynman_exp2_checkpoint_*, etc.)
)

# ── Helper: write a key=value to the correct GitHub output channel ────────────
emit() {
  local key="$1" val="$2"
  if [[ "$OUTPUT_TARGET" == "output" ]]; then
    echo "${key}=${val}" >> "$GITHUB_OUTPUT"
  else
    echo "${key}=${val}" >> "$GITHUB_ENV"
  fi
}

# ── Helper: build the -not -name ... exclusion args for find ─────────────────
# Returns a sequence of:  ! -name '<pattern>' ! -name '<pattern>' ...
metadata_exclusion_args() {
  local args=()
  for pat in "${METADATA_EXCLUSIONS[@]}"; do
    args+=( '!' '-name' "$pat" )
  done
  printf '%s\0' "${args[@]}"
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
echo "=== METADATA EXCLUSIONS ==="
printf '  %s\n' "${METADATA_EXCLUSIONS[@]}"

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
    print('true' if '$EXPERIMENT' in ('exp1b', 'exp1_ablation', 'exp3b') else 'false',
          file=sys.stdout)
    print(f'::warning::Could not import MERGE_REQUIRED_EXPERIMENTS: {e}', file=sys.stderr)
")
echo "REQUIRE_MERGE=$REQUIRE_MERGE"

# ==============================================================================
#  MERGED MODE (exp1b / exp1_ablation / exp3b / suppB / suppB_sc)
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

  # FIX-EMPTY-MERGED-FASTPATH: reject a _merged.json that was committed with
  # zero records (e.g. every upstream benchmark run failed/timed out, leaving
  # empty per_equation blocks inside the Shape S sweep file).  If we accept it
  # as-is the analysis step will hit "FATAL: EMPTY DATASET" with no pointer
  # back to the cause.  Instead, discard it here and fall through to the
  # merge_shards.py fallback, which will either raise a clear RuntimeError with
  # full file-path context or produce a fresh _merged.json from the shard files.
  #
  # Record-count heuristic (mirrors merge_shards.py merge_sweep_files logic):
  #   Shape S (suppB / suppB_sc): sum len(per_equation) across all sweep points
  #                                in per_noise / per_n.
  #   Shape A/B (task-row):       count top-level keys that are not underscore
  #                                meta-keys (_checkpoint, _stats, …).
  if [[ ${#CANDIDATES[@]} -gt 0 ]]; then
    INPUT_JSON="${CANDIDATES[0]}"
    _n_records=$(python3 - <<PYEOF 2>/dev/null
import json, sys
try:
    d = json.load(open('${INPUT_JSON}'))
    total = 0
    # Shape S: count equation-level entries across all sweep-axis points.
    for axis_key in ('per_noise', 'per_n'):
        for pt in d.get(axis_key, {}).values():
            if isinstance(pt, dict):
                total += len(pt.get('per_equation', {}))
    # Shape A/B: if no sweep axis found, count non-underscore top-level keys.
    if total == 0 and isinstance(d, dict):
        total = sum(1 for k in d if isinstance(k, str) and not k.startswith('_'))
    print(total)
except Exception:
    print(0)
PYEOF
)
    if [[ "${_n_records:-0}" -eq 0 ]]; then
      echo "::warning::Fast-path _merged.json at '${INPUT_JSON}' has 0 records — rejecting stale/empty file and falling back to merge_shards.py. If shard files are also missing, delete '${INPUT_JSON}' from the repo and re-run the workers."
      CANDIDATES=()   # fall through to merge_shards.py fallback below
    fi
  fi

  if [[ ${#CANDIDATES[@]} -gt 0 ]]; then
    INPUT_JSON="${CANDIDATES[0]}"
    emit "INPUT_MODE" "merged"
    emit "INPUT_JSON" "$INPUT_JSON"
    emit "SHARD_MANIFEST" ""
    echo
    echo "Selected merged input: $INPUT_JSON  (${_n_records} record(s))"
    exit 0
  fi

  # ── Fallback: run merge_shards.py against committed shard files ─────────────
  # For JSON-shard experiments (exp1b, exp1_ablation, exp3b): collects *.json files.
  # For Shape-S sweep experiments (suppB, suppB_sc): collects *.json sweep shards.
  # "instability" is NOT in MERGE_REQUIRED_EXPERIMENTS and never reaches this
  # branch — its CSV outputs (instability_analysis.csv, etc.) are not merged
  # here; run_analysis.py short-circuits for it before any input is needed.
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

# ── Build the find exclusion args from METADATA_EXCLUSIONS ───────────────────
# We need: ! -name 'pat1' ! -name 'pat2' ...
# Bash arrays can't be passed to find as a single arg safely, so we build
# the command as an array and expand it with "${FIND_ARGS[@]}".
FIND_ARGS=(
  find "$RESULT_DIR"
  -maxdepth 2
  -type f
  -name '*.json'
  '!' -name '_*.json'
)
# Add every METADATA_EXCLUSIONS pattern as a find ! -name predicate.
# The outer find is already restricted to *.json, so non-json patterns
# (*.csv, *.md, *.txt) are harmless to include — they simply never match.
# Previously a case filter skipped non-json patterns, which accidentally
# omitted 'benchmark_results*.json' and '*_checkpoint_*.json' from the
# exclusion list when they were added as plain *.json patterns above.
for pat in "${METADATA_EXCLUSIONS[@]}"; do
  FIND_ARGS+=( '!' '-name' "$pat" )
done

# Collect non-meta JSON shard files
mapfile -t SHARD_FILES < <(
  "${FIND_ARGS[@]}" | sort
)

if [[ ${#SHARD_FILES[@]} -eq 0 ]]; then
  echo "::error::No shard JSON files found in ${RESULT_DIR}."
  echo "  Searched:  ${RESULT_DIR}/**/*.json (maxdepth 2)"
  echo "  Excluded:  _*.json  ${METADATA_EXCLUSIONS[*]}"
  exit 1
fi

echo "  Found ${#SHARD_FILES[@]} candidate shard file(s):"
printf '    %s\n' "${SHARD_FILES[@]}"

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




