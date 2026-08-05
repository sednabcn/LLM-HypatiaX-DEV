#!/usr/bin/env bash
#
# apply_shard_patch.sh
#
# Implements the full nshards-suffix -> shard_id rename from patching.txt
# using sed, against ONLY the canonical copies of each file:
#
#   .github/workflows/ci_runner.yml   (step 1 -- the actual "CI matrix
#       generation" file; patching.txt's "ci_runner.py" doesn't exist
#       anywhere in tree_.txt, this .yml is what it was describing)
#   hypatiax/run_all.sh
#   hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_v2.py
#   hypatiax/experiments/benchmarks/run_noise_sweep_benchmark.py
#   hypatiax/experiments/benchmarks/run_sample_complexity_benchmark.py
#
# Per your instruction, the duplicate copies under RUN_ALL_OLD/, RUN_ALL_INJECTED/,
# hypatiax-all-cloud/, hypatiax-orchestrator/, hypatiax-windows/, and the bare repo
# root run_all.sh are intentionally NOT touched.
#
# Step 1 detail (ci_runner.yml, verified against the actual uploaded file):
#   - plan job's per-shard matrix dict gets a new "shard_id": j + 1 field
#     (1-based), alongside the existing 0-based "shard": j.
#   - plan job's fh.write(f"n_shards={N_SHARDS:02d}\n") loses its zero-pad
#     (:02d) since n_shards is now just the total count, not an artifact
#     suffix -- the zero-padding moves to shard_id via run_all.sh's
#     `printf -v SHARD_ID "%02d" ...` (step 2).
#   - worker job env gets a new SHARD_ID: ${{ matrix.shard_id }} line next
#     to the existing SHARD_INDEX: ${{ matrix.shard }} line, so run_all.sh
#     receives the 1-based shard id it needs for artifact naming.
#
# Usage:
#   ./apply_shard_patch.sh /path/to/repo/root
#   ./apply_shard_patch.sh /path/to/repo/root --dry-run   # show diffs, change nothing
#
set -euo pipefail

REPO_ROOT="${1:-.}"
DRY_RUN=false
if [[ "${2:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

CI_RUNNER="$REPO_ROOT/.github/workflows/ci_runner.yml"
RUN_ALL="$REPO_ROOT/hypatiax/run_all.sh"
BENCH_DIR="$REPO_ROOT/hypatiax/experiments/benchmarks"
COMPARATIVE="$BENCH_DIR/run_comparative_suite_benchmark_v2.py"
NOISE_SWEEP="$BENCH_DIR/run_noise_sweep_benchmark.py"
SAMPLE_COMPLEXITY="$BENCH_DIR/run_sample_complexity_benchmark.py"

TARGETS=("$CI_RUNNER" "$RUN_ALL" "$COMPARATIVE" "$NOISE_SWEEP" "$SAMPLE_COMPLEXITY")

for f in "${TARGETS[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: expected canonical file not found: $f" >&2
    echo "       Check REPO_ROOT or re-confirm the path against tree_.txt." >&2
    exit 1
  fi
done

run_sed() {
  local file="$1"; shift
  if $DRY_RUN; then
    diff -u "$file" <(sed "$@" "$file") || true
  else
    cp "$file" "$file.bak"
    sed -i "$@" "$file"
    echo "patched: $file (backup: $file.bak)"
  fi
}

# ---------------------------------------------------------------------------
# 1. .github/workflows/ci_runner.yml
# ---------------------------------------------------------------------------
# 1a. Give each matrix entry its own 1-based shard_id, alongside "shard": j.
#     Anchored on the exact "shard":   j, line (verified present, line 739).
run_sed "$CI_RUNNER" \
  -e '/^                  "shard":   j,$/a\
                  "shard_id": j + 1,'

# 1b. Drop the :02d zero-pad on the n_shards output (it is the resolved shard
#     COUNT now, not an artifact suffix -- zero-padding moves to shard_id via
#     run_all.sh's printf -v SHARD_ID "%02d" in step 2 below).
run_sed "$CI_RUNNER" \
  -e 's/N_SHARDS:02d/N_SHARDS/'

# 1c. Forward the new shard_id to worker jobs, next to the existing
#     SHARD_INDEX line (verified present, line 801).
run_sed "$CI_RUNNER" \
  -e '/^      SHARD_INDEX:    \${{ matrix.shard }}$/a\
      SHARD_ID:       ${{ matrix.shard_id }}'

# ---------------------------------------------------------------------------
# 2. hypatiax/run_all.sh
# ---------------------------------------------------------------------------
# Remove the old suffix export, then insert the three new lines in its place.
run_sed "$RUN_ALL" \
  -e '/^export HYPATIAX_NSHARDS_SUFFIX=/{
        c\
export HYPATIAX_NSHARDS="${n_shards}"\
\
printf -v SHARD_ID "%02d" "${shard_id}"\
\
export HYPATIAX_SHARD_ID="${SHARD_ID}"
      }'

# ---------------------------------------------------------------------------
# 3. hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_v2.py
# ---------------------------------------------------------------------------
# 3a. Rename the env var the script reads from (NSHARDS_SUFFIX -> SHARD_ID env)
run_sed "$COMPARATIVE" \
  -e 's/HYPATIAX_NSHARDS_SUFFIX/HYPATIAX_SHARD_ID/g'

# 3b. Rename the internal identifiers _NSHARDS_SUFFIX -> _SHARD_ID, _NSHARDS_TAG -> _SHARD_TAG
run_sed "$COMPARATIVE" \
  -e 's/_NSHARDS_SUFFIX/_SHARD_ID/g' \
  -e 's/_NSHARDS_TAG/_SHARD_TAG/g'

# 3c. Rename the literal f-string prefix used to build the actual filename tag,
#     e.g. f"_nshards{_SHARD_ID}" -> f"_shrd{_SHARD_ID}" (this is the part that
#     turns benchmark_results_nshards05.json into benchmark_results_shrd03.json --
#     the identifier renames above don't touch this literal text).
#     NOTE: this only targets the `_nshards{` f-string-prefix pattern, not the
#     standalone word "_nshards" elsewhere (comments, unrelated strings), since
#     this file is presumed to be where _SHARD_TAG is actually defined.
run_sed "$COMPARATIVE" \
  -e 's/_nshards{/_shrd{/g'

# ---------------------------------------------------------------------------
# 4. hypatiax/experiments/benchmarks/run_noise_sweep_benchmark.py
#    (N_SHARDS = 5 is left untouched -- only the tag identifier is renamed)
# ---------------------------------------------------------------------------
run_sed "$NOISE_SWEEP" \
  -e 's/_NSHARDS_TAG/_SHARD_TAG/g'

# ---------------------------------------------------------------------------
# 5. hypatiax/experiments/benchmarks/run_sample_complexity_benchmark.py
#    (N_SHARDS = 6 is left untouched -- only the tag identifier is renamed)
# ---------------------------------------------------------------------------
run_sed "$SAMPLE_COMPLEXITY" \
  -e 's/_NSHARDS_TAG/_SHARD_TAG/g'

# ---------------------------------------------------------------------------
# 6. Result aggregation scripts (matched by content, anywhere under hypatiax/,
#    since the patch plan doesn't name them specifically). Updates the glob
#    pattern used to discover per-shard result files.
# ---------------------------------------------------------------------------
echo
echo "Scanning for aggregation scripts using the old '*_nshards*.json' glob..."
mapfile -t AGG_FILES < <(grep -rl '_nshards' "$REPO_ROOT/hypatiax" --include='*.py' 2>/dev/null || true)

if [[ ${#AGG_FILES[@]} -eq 0 ]]; then
  echo "  none found."
else
  for af in "${AGG_FILES[@]}"; do
    # Skip the four files we already handled above so they aren't double-patched.
    skip=false
    for t in "${TARGETS[@]}"; do
      if [[ "$(realpath "$af")" == "$(realpath "$t")" ]]; then
        skip=true
        break
      fi
    done
    $skip && continue

    run_sed "$af" \
      -e 's/\*_nshards\*\.json/*_shrd*.json/g' \
      -e 's/_nshards[0-9][0-9]*/_shrd/g'
  done
fi

echo
if $DRY_RUN; then
  echo "Dry run complete. No files were modified."
else
  echo "Done. Review the diffs (git diff, or compare against the .bak files), then remove the .bak files once verified."
fi

# ---------------------------------------------------------------------------
# 7. .github/workflows/ci_runner.yml -- remove the "Move results to
#    RESULTS_DIR" step, which is the only place ci_runner.yml shells out to
#    move_results.sh (verified: single occurrence, line 1309 of the uploaded
#    file). Deletes the step's header comment block, the step itself, and
#    the blank line right after it, so the next step ("Store results in
#    canonical RESULT_SUBDIR") butts up cleanly against the prior step.
#    NOTE: that next step's own comment references the deleted step's
#    behavior for context ("The move_matching function in 'Move results to
#    RESULTS_DIR' excludes files already inside OUT_BASE...") -- that prose
#    is now stale and may be worth a manual follow-up edit.
# ---------------------------------------------------------------------------
run_sed "$CI_RUNNER" \
  -e '/^      # -- Move results to canonical RESULTS_DIR after each experiment/,/^$/d'
