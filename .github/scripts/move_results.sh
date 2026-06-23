#!/usr/bin/env bash
# ==============================================================================
#  .github/scripts/move_results.sh
#
#  Extracted from ci_runner.yml "Move results to RESULTS_DIR" step to work
#  around GitHub Actions' 21 000-character run: block limit.
#
#  Required env vars (all injected by the workflow step's env: block):
#    EXP           – experiment ID (e.g. exp1, exp2_feynman, suppB …)
#    RESULT_SUBDIR – canonical result sub-directory relative to OUT_BASE
#    RESULTS_DIR   – absolute path: $GITHUB_WORKSPACE/$OUT_BASE
#    SHARD_INDEX   – zero-based shard index (from matrix.shard, set at job level)
#
#  $GITHUB_RUN_ID is provided automatically by the Actions runner — no need
#  to pass it explicitly.  All occurrences of ${{ github.run_id }} in the
#  original inline script have been replaced with ${GITHUB_RUN_ID}.
# ==============================================================================
set -euo pipefail

TARGET="${RESULTS_DIR}/${RESULT_SUBDIR}"
mkdir -p "${TARGET}"

# FIX-CROSS-EXPERIMENT-CAPTURE (2026-06-23, confirmed from a real CI
# run's logs — logs_75542162448.zip, Worker shard 0-4, step "Move
# results to RESULTS_DIR"): move_matching's `find . -name "$pattern"`
# searches the ENTIRE workspace with no awareness of which
# experiment a matched file actually belongs to. Several
# experiments share generic output filenames written by the same
# underlying script (run_comparative_suite_benchmark_v2.py) --
# protocol_core_*.json and benchmark_results.json are written by
# exp2, exp2_extrap, exp2_multi, suppB, AND suppB_sc alike. A
# confirmed production run showed suppB's move step reaching into
# exp2_extrap/, exp2/, and exp2_multi/'s own already-committed
# directories and relocating THEIR files into noise-sweep/ --
# silently merging unrelated experiments' historical results
# together. This is data corruption, not just a cosmetic path bug.
#
# Fix: before considering a match, skip it if it's already sitting
# inside ANY OTHER experiment's canonical RESULT_SUBDIR (the full
# known list, kept in sync with every RESULT_SUBDIR="..." assignment
# in the plan job's meta step above). A file inside its OWN
# experiment's dir is still handled correctly by the existing
# same-dir-as-dest skip just below; this additional check only
# blocks pulling a DIFFERENT experiment's already-placed file.
_OTHER_EXPERIMENT_SUBDIRS=(
  "comparison_results/noise-noiseless/noiseless/defi"
  "comparison_results/noise-noiseless/15"
  "comparison_results/feynman-tests/exp2"
  "comparison_results/feynman-tests/exp2_extrap"
  "comparison_results/feynman-tests/exp2_multi"
  "comparison_results/feynman-tests/noise-sweep"
  "comparison_results/feynman-tests/sample-complexity"
  "comparison_results/extrapolation"
)

move_matching() {
  local pattern="$1" dest="$2"
  mkdir -p "$dest"
  local realdest_check
  realdest_check="$(realpath "$dest" 2>/dev/null || echo "$dest")"
  while IFS= read -r src; do
    [[ -e "$src" ]] || continue
    local realsrc
    realsrc="$(realpath "$src" 2>/dev/null || echo "$src")"
    # Skip if src is already inside dest (file already in canonical location)
    [[ "$realsrc" == "$realdest_check"/* ]] && continue
    # FIX-CROSS-EXPERIMENT-CAPTURE: skip if src is already inside a
    # DIFFERENT experiment's canonical subdir (not this call's own
    # dest -- already excluded above).
    local _skip=0
    for _other in "${_OTHER_EXPERIMENT_SUBDIRS[@]}"; do
      local _other_real
      _other_real="$(realpath "${RESULTS_DIR}/${_other}" 2>/dev/null || echo "${RESULTS_DIR}/${_other}")"
      [[ "$_other_real" == "$realdest_check" ]] && continue  # that's this call's own dest, already handled
      if [[ "$realsrc" == "$_other_real"/* ]]; then
        _skip=1
        break
      fi
    done
    if [[ "$_skip" -eq 1 ]]; then
      echo "  SKIP (belongs to another experiment's dir): $src"
      continue
    fi
    fname=$(basename "$src")
    dst="${dest}/${fname}"
    if [[ -e "$dst" ]]; then
      # BUG A FIX (universal): collision on same filename across shards/runs.
      # Insert _shard<N>_run<run_id> before the extension so no result is lost.
      stem="${fname%.*}"; ext="${fname##*.}"
      if [[ "$stem" == "$fname" ]]; then
        dst="${dest}/${fname}_shard${SHARD_INDEX}_run${GITHUB_RUN_ID}"
      else
        dst="${dest}/${stem}_shard${SHARD_INDEX}_run${GITHUB_RUN_ID}.${ext}"
      fi
      echo "  COLLISION: ${fname} already in ${dest} — writing to $(basename "$dst")"
    fi
    mv -v "$src" "$dst" 2>/dev/null || true
  done < <(find . -maxdepth 8 \
               -name "$pattern" 2>/dev/null)
}

# prune_old: remove previously-committed (git-tracked) result files
# matching a glob from a destination directory before the current
# run's fresh outputs are moved in.  Without this, timestamp-named
# files from past runs accumulate in TARGET on every workflow run
# because move_matching only renames exact-name collisions, never
# removes older timestamped variants.
#
# Strategy:
#   1. find all files in dest matching the pattern (maxdepth 1).
#   2. For each, check git ls-files — only remove if git-tracked
#      (i.e. committed by a prior run, not a local-only file that
#      another step in this run might still need).
#   3. git rm --cached unstages the file; rm -f removes it from disk.
#      The subsequent "Commit and push" step will then stage the
#      deletion (D) alongside the newly moved files (A), keeping
#      the repo tree in sync with exactly one run's outputs.
prune_old() {
  # Removes files matching pattern from dest before a fresh run moves
  # new outputs in.  ONLY removes git-tracked (committed) files.
  #
  # FIX-E2: removed untracked-file deletion.  The original code also
  # deleted untracked files as "stale partial-run leftovers", but this
  # is indistinguishable from fresh outputs the current run just wrote
  # directly into TARGET (e.g. via --output-dir in the Python script).
  # For exp2_feynman_extrap this caused all 9+ protocol_core_extrap_*.json
  # files to be silently wiped immediately after creation — they were
  # present in TARGET but untracked, so prune_old deleted them before
  # move_matching could stage them for the artifact upload and git commit.
  #
  # Correct semantics: only remove files that are git-tracked (i.e.
  # committed by a prior workflow run).  Untracked files are either
  # (a) fresh outputs of the current run — must NOT be deleted, or
  # (b) leftover workspace noise — harmless to leave; move_matching
  #     will skip them because their destination already exists.
  local pattern="$1" dest="$2"
  local pruned=0
  while IFS= read -r f; do
    [[ -f "$f" ]] || continue
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      echo "  prune_old: removing stale committed result: $f"
      git rm --cached --quiet "$f" 2>/dev/null || true
      rm -f "$f"
      pruned=$((pruned + 1))
    else
      echo "  prune_old: skipping untracked file (may be current-run output): $f"
    fi
  done < <(find "$dest" -maxdepth 1 -name "$pattern" -type f 2>/dev/null)
  [[ $pruned -gt 0 ]] && echo "  prune_old: removed ${pruned} stale committed file(s) matching '${pattern}' from ${dest}"
  return 0
}

case "${EXP}" in

  # -- exp1: noiseless DeFi benchmark -------------------------------
  # hypatiax_defi_benchmark_v3c.py writes:
  #   hypatiax_defi_benchmark_v3_results_<TS>.json   ← primary output
  #   protocol_core_noiseless_<TS>.json              ← if protocol wrapper used
  #   defi_v3_<TS>.json                              ← legacy name
  # All three patterns are moved so the artifact upload finds them.
  #
  # STALE-FILE FIX: prune previously-committed output files from
  # TARGET before moving in the current run's outputs, so that
  # timestamp-named files from past runs do not accumulate in the
  # repo tree (each run should own exactly one result set).
  exp1)
    prune_old "hypatiax_defi_benchmark_v3*results*.json" "${TARGET}"
    prune_old "protocol_core_noiseless_*.json"           "${TARGET}"
    prune_old "defi_v3_*.json"                           "${TARGET}"
    move_matching "hypatiax_defi_benchmark_v3*results*.json" "${TARGET}"
    move_matching "protocol_core_noiseless_*.json"           "${TARGET}"
    move_matching "defi_v3_*.json"                           "${RESULTS_DIR}"
    ;;

  # -- exp1b: noisy DeFi benchmark (noise=15) -----------------------
  # BUG A FIX: comparison_FIXED_<TS>.json files from concurrent or
  # repeated shards collide if two runners produce the same timestamp.
  # Rename each file to comparison_FIXED_<TS>_shard<N>_<runid>.json
  # before moving so every shard's output has a unique repo path.
  # The same logic applies to .txt companion files.
  #
  # STALE-FILE FIX: prune previously-committed outputs before moving
  # in the current run's files (same pattern as exp1).
  exp1b)
    prune_old "comparison_FIXED_*.json"                  "${TARGET}"
    prune_old "comparison_FIXED_*.txt"                   "${TARGET}"
    prune_old "hypatiax_defi_benchmark_v3*results*.json" "${TARGET}"
    prune_old "defi_v3_*.json"                           "${TARGET}"
    prune_old "*portfolio*variance*.json"                 "${TARGET}"
    DEST_15="${TARGET}"
    mkdir -p "${DEST_15}"
    while IFS= read -r src; do
      [[ -e "$src" ]] || continue
      base=$(basename "$src")
      stem="${base%.*}"; ext="${base##*.}"
      # Insert _shard<N>_run<run_id> before the extension so the filename
      # is unique across shards and across repeated workflow runs.
      dest="${DEST_15}/${stem}_shard${SHARD_INDEX}_run${GITHUB_RUN_ID}.${ext}"
      mv -v "$src" "$dest" || true
    done < <(find . -maxdepth 8 \
                 \( -name "comparison_FIXED_*.json" -o \
                    -name "comparison_FIXED_*.txt"  -o \
                    -name "hypatiax_defi_benchmark_v3*results*.json" -o \
                    -name "defi_v3_*.json" -o \
                    -name "*portfolio*variance*.json" \) 2>/dev/null)
    ;;

  # -- exp2_feynman: comparative Feynman suite -----------------------
  # Now runs all 6 methods (--skip-pysr removed) on 1 shard.
  # Per-domain outputs: protocol_core_*.json, benchmark_results.json,
  # protocol_core_*_checkpoint.json (one per domain from --checkpoint-name).
  # All are saved to TARGET so the verify step and artifact upload find them.
  exp2_feynman)
    prune_old "protocol_core_*.json"            "${TARGET}"
    prune_old "protocol_core_*_checkpoint.json" "${TARGET}"
    prune_old "benchmark_results.json"          "${TARGET}"
    move_matching "exp2_feynman_checkpoint_shard*.json" "${TARGET}"
    move_matching "exp2_feynman_merged.*"               "${TARGET}"
    move_matching "exp2_feynman_stats.json"             "${TARGET}"
    move_matching "exp2_all*_checkpoint.json"           "${RESULTS_DIR}/comparison_results"
    # Primary per-domain outputs written by run_comparative_suite_benchmark_v2.py
    move_matching "protocol_core_*.json"                "${TARGET}"
    move_matching "protocol_core_*_checkpoint.json"     "${TARGET}"
    move_matching "benchmark_results.json"              "${TARGET}"
    move_matching "feynman_exp2_checkpoint_*.json"      "${TARGET}"
    ;;

  # exp2_feynman_extrap: outputs use protocol_core_extrap_* prefix and
  # go into their own feynman-tests/exp2_extrap dir (RESULT_SUBDIR).
  # prune_old is scoped to the extrap prefix only — never touches
  # the main exp2_feynman results in feynman-tests/exp2.
  #
  # benchmark_results_extrap.json is the new flat file written by
  # run_comparative_suite_benchmark_v2.py v2.2 when --extrap is active.
  # It carries both r2 (train) AND extrap_r2_far in one place and is
  # the primary input to merge_extrap_into_benchmark.py --extrap-benchmark-dir.
  # Without this move it lives only in the workspace root, misses the
  # artifact upload, and is never committed to the repo.
  exp2_feynman_extrap)
    # FIX-NOISELESS_EXTRAP: benchmark_results_extrap.json must NOT be pruned.
    # prune_old "benchmark_results_extrap.json" was here and deleted the file
    # immediately after it was created, before move_matching could pick it up.
    # This is the primary input to merge_extrap_into_benchmark.py
    # --extrap-benchmark-dir; purging it caused ablation_paired.json to be
    # empty and the Mann-Whitney test to report TOO_FEW_MW_PAIRS. REMOVED.
    move_matching "protocol_core_extrap_*.json"         "${TARGET}"
    move_matching "feynman_extrap_checkpoint_*.json"    "${TARGET}"
    # FIX-E6: benchmark_results_extrap.json and benchmark_results.json have no
    # timestamp or shard suffix, so concurrent shards or re-runs silently
    # overwrite each other on git push.  Rename them to include the shard
    # index and run_id so every run's output has a unique repo path.
    # The merge script (merge_extrap_into_benchmark.py) reads the extrap dir
    # via glob, so any _shard*_run*.json file is picked up correctly.
    for _stable_name in "benchmark_results_extrap.json" "benchmark_results.json"; do
      _stable_src=$(find . -maxdepth 8 -name "${_stable_name}" 2>/dev/null | head -1)
      if [[ -n "${_stable_src}" && -e "${_stable_src}" ]]; then
        _stable_stem="${_stable_name%.json}"
        _stable_dst="${TARGET}/${_stable_stem}_shard${SHARD_INDEX}_run${GITHUB_RUN_ID}.json"
        # Skip if already inside TARGET
        [[ "$(realpath "${_stable_src}" 2>/dev/null || echo "${_stable_src}")" == "$(realpath "${TARGET}" 2>/dev/null || echo "${TARGET}")"/* ]] && continue
        mv -v "${_stable_src}" "${_stable_dst}" 2>/dev/null || true
        echo "  FIX-E6: renamed ${_stable_name} → $(basename "${_stable_dst}")"
      fi
    done
    move_matching "exp2_extrap_run.log"                 "${TARGET}"
    ;;

  # -- exp2: multi-seed comparative suite ---------------------------
  exp2)
    prune_old "protocol_core_*.json"            "${TARGET}"
    prune_old "protocol_core_*_checkpoint.json" "${TARGET}"
    prune_old "benchmark_results.json"          "${TARGET}"
    move_matching "exp2_checkpoint_shard*.json"        "${TARGET}"
    move_matching "exp2_merged.*"                      "${TARGET}"
    move_matching "exp2_stats.json"                    "${TARGET}"
    # Primary per-domain outputs written by run_comparative_suite_benchmark_v2.py
    move_matching "protocol_core_*.json"               "${TARGET}"
    move_matching "protocol_core_*_checkpoint.json"    "${TARGET}"
    move_matching "benchmark_results.json"             "${TARGET}"
    move_matching "exp2_checkpoint_*.json"             "${TARGET}"
    ;;

  # -- exp3: Nguyen12 extrapolation (single seed) -------------------
  exp3)
    prune_old "full_run_*.json"          "${TARGET}"
    prune_old "report_hybrid_*.json"     "${TARGET}"
    prune_old "exp3_nguyen12_*.json"     "${TARGET}"
    prune_old "*nguyen*seed*.json"        "${TARGET}"
    move_matching "full_run_*.json"          "${TARGET}"
    move_matching "report_hybrid_*.json"     "${TARGET}"
    move_matching "hybrid_defi_*.json"       "${TARGET}"
    move_matching "experiment_registry.json" "${TARGET}"
    # ISSUE 2 FIX: xref lists exp3_nguyen12_*.json and *nguyen*seed*.json as
    # expected outputs (tab:nguyen12) but these were never moved — files stayed
    # at workspace root and were silently lost before artifact upload.
    move_matching "exp3_nguyen12_*.json"     "${TARGET}"
    move_matching "*nguyen*seed*.json"       "${TARGET}"
    ;;

  # BUG 2 FIX: exp3b outputs now go to extrapolation/multi_seed to avoid
  # colliding with exp3 outputs in extrapolation/.
  exp3b)
    prune_old "full_run_*.json"          "${TARGET}"
    prune_old "report_hybrid_*.json"     "${TARGET}"
    prune_old "exp3_nguyen12_*.json"     "${TARGET}"
    prune_old "*nguyen*seed*.json"        "${TARGET}"
    move_matching "full_run_*.json"          "${TARGET}"
    move_matching "report_hybrid_*.json"     "${TARGET}"
    move_matching "hybrid_defi_*.json"       "${TARGET}"
    move_matching "experiment_registry.json" "${TARGET}"
    # ISSUE 2 FIX: same missing nguyen globs as exp3 — add them here too.
    move_matching "exp3_nguyen12_*.json"     "${TARGET}"
    move_matching "*nguyen*seed*.json"       "${TARGET}"
    ;;

  # -- suppA: hybrid PySR/DeFi benchmark ----------------------------
  # RESULT_SUBDIR=hybrid_pysr/defi (set in meta step) - move must
  # match so the artifact upload and consolidate job find the files.
  suppA)
    move_matching "consolidated_hybrid_*.json"          "${TARGET}"
    # ISSUE 3 FIX: xref lists hybrid_system*.json and extrapolation_73cases*.json
    # / *73cases*.json as primary suppA outputs (Tab 11-13) but these patterns
    # were missing from the move block — files were never relocated to the
    # canonical subdir and would be absent from the artifact upload.
    move_matching "hybrid_system*.json"                 "${TARGET}"
    move_matching "extrapolation_73cases*.json"         "${TARGET}"
    move_matching "*73cases*.json"                      "${TARGET}"
    move_matching "hybrid_llm_nn_all_domains_*.json"    "${RESULTS_DIR}/hybrid_llm_nn/all_domains"
    move_matching "ablation_exp1_*.json"                "${RESULTS_DIR}"
    move_matching "hypatiax_defi_benchmark_v3_results*" "${RESULTS_DIR}"
    ;;

  # -- suppB: noise sweep --------------------------------------------
  # Top-level files + per-equation sub-directories (e.g. I.12.1-correction/)
  suppB)
    # STALE-FILE FIX (mirrors exp1/exp1b/exp2_feynman/exp2/exp3/exp3b above):
    # this block had NO prune_old calls, so timestamp-named outputs from past
    # runs accumulated unbounded in noise-sweep/ on every workflow run.
    # All 5 noise-level shards (one per matrix.shard) write into this SAME
    # shared TARGET — pruning is therefore dir/pattern-wide, not per-shard.
    # Each shard's job starts from the same pre-run git checkout, so all 5
    # concurrently-running shards prune the identical set of git-tracked
    # stale files; idempotent across the matrix, same pattern exp1b already
    # relies on across its own 4 concurrent shards.
    prune_old "noise_sweep_*.json"              "${TARGET}"
    prune_old "noise_sweep_*.csv"               "${TARGET}"
    prune_old "protocol_core_*.json"            "${TARGET}"
    prune_old "protocol_core_*_checkpoint.json" "${TARGET}"
    prune_old "benchmark_results*.json"         "${TARGET}"
    move_matching "noise_sweep_*.json" "${TARGET}"
    move_matching "noise_sweep_*.csv"  "${TARGET}"
    # FIX-suppB-MOVE-PROTOCOL: protocol_core_*.json and benchmark_results.json
    # (written by run_comparative_suite_benchmark_v2.py via --output-dir, which
    # now resolves to this same TARGET once OUT_BASE/RESULT_SUBDIR are no longer
    # doubled — see run_all.sh STEP 10 and RESULT_SUBDIR comment above) were
    # previously left for the per-equation find/mv rescue below to catch
    # incidentally. Move them explicitly, mirroring the exp2/exp2_extrap cases,
    # so they're never missed if the inner runner's CWD or output path changes.
    #
    # FIX-NSHARDS-SUFFIX-MOVE: run_comparative_suite_benchmark_v2.py
    # now writes benchmark_results{_nshardsNN}.json /
    # benchmark_results_extrap{_nshardsNN}.json because
    # HYPATIAX_NSHARDS_SUFFIX is always set for suppB (run_all.sh's
    # suppB step exports it per-shard) — the bare "benchmark_results.json"
    # literal this used to match never matches that filename anymore.
    # Glob-matching also still catches the bare form for any local
    # run made outside this CI path, where the suffix env var is unset.
    move_matching "protocol_core_*.json"            "${TARGET}"
    move_matching "protocol_core_*_checkpoint.json" "${TARGET}"
    move_matching "benchmark_results*.json"         "${TARGET}"
    # Rescue per-equation sub-dirs preserving directory name
    #
    # FIX-suppB-RESCUE-LOOP-DOUBLED-PATH (2026-06-23, confirmed from
    # a real CI run's logs — logs_75542162448.zip, Worker shard 0-4,
    # step "Move results to RESULTS_DIR"):
    #
    # This loop is meant to rescue files sitting in a per-equation
    # SUBdirectory of noise-sweep/ (e.g.
    # .../noise-sweep/I.12.1-correction/result.json), preserving
    # that subdirectory name when moving it to the canonical
    # location. It previously matched ANY *.json/*.csv under any
    # path containing the literal string "noise-sweep" --
    # including files the move_matching calls just above had
    # ALREADY correctly placed directly in noise-sweep/ itself
    # (not in a per-equation subdirectory). For those files,
    # dirname(f) is exactly ".../noise-sweep", so
    # basename(dirname(f)) evaluates to the literal string
    # "noise-sweep" -- and dest becomes
    # ".../noise-sweep/noise-sweep", manufacturing the doubled
    # directory this codebase has hit repeatedly across multiple
    # "FIX-suppB-DOUBLED-PATH" attempts elsewhere (RESULT_SUBDIR,
    # SUPPB_SUBDIR, MAPPING fallbacks) — none of which were the
    # actual cause. THIS loop was. A confirmed production run
    # showed exactly this: 217 files moved correctly to the
    # single-level path, then control reached this loop and the
    # remaining ~218 files (already sitting in noise-sweep/ from
    # the move_matching calls just above) all got "rescued" one
    # level deeper.
    #
    # Fixed by skipping any match whose immediate parent directory
    # IS noise-sweep itself (dir == "noise-sweep") -- those files
    # are already in the canonical location and need no rescue.
    # Only files in a genuine per-equation subdirectory (dir !=
    # "noise-sweep") still get rescued, preserving the original
    # intent.
    find . -maxdepth 8 \
      -path "*/noise-sweep/*" \( -name "*.json" -o -name "*.csv" \) \
      | while IFS= read -r f; do
          dir=$(basename "$(dirname "$f")")
          [[ "$dir" == "noise-sweep" ]] && continue
          dest="${RESULTS_DIR}/comparison_results/feynman-tests/noise-sweep/${dir}"
          mkdir -p "$dest"
          mv -v "$f" "$dest/" 2>/dev/null || true
        done
    ;;

  # -- suppB_sc: sample-complexity sweep ----------------------------
  suppB_sc)
    # STALE-FILE FIX (mirrors exp1/exp1b/exp2_feynman/exp2/exp3/exp3b above):
    # same gap and same fix as suppB's block — this case had NO prune_old
    # calls. All 6 sample-size shards share this TARGET (sample-complexity/);
    # pruning is dir/pattern-wide, idempotent across the concurrent matrix.
    prune_old "sample_complexity_*.json"        "${TARGET}"
    prune_old "sample_complexity_*.csv"         "${TARGET}"
    prune_old "protocol_core_*.json"            "${TARGET}"
    prune_old "protocol_core_*_checkpoint.json" "${TARGET}"
    prune_old "benchmark_results*.json"         "${TARGET}"
    move_matching "sample_complexity_*.json" "${TARGET}"
    move_matching "sample_complexity_*.csv"  "${TARGET}"
    # FIX-suppB_sc-MOVE-PROTOCOL: protocol_core_*.json and benchmark_results.json
    # (written by run_comparative_suite_benchmark_v2.py via --output-dir, which
    # now resolves to this same TARGET once OUT_BASE is no longer doubled — see
    # run_all.sh STEP 10b FIX-suppB_sc-DOUBLED-PATH comment) — mirrors the same
    # fix applied to the suppB block above.
    #
    # FIX-NSHARDS-SUFFIX-MOVE: same fix as suppB's block — once
    # run_all.sh's suppB_sc step exports HYPATIAX_NSHARDS_SUFFIX,
    # run_comparative_suite_benchmark_v2.py writes
    # benchmark_results{_nshardsNN}.json instead of the bare name,
    # so the literal match below would find nothing.
    move_matching "protocol_core_*.json"            "${TARGET}"
    move_matching "protocol_core_*_checkpoint.json" "${TARGET}"
    move_matching "benchmark_results*.json"         "${TARGET}"
    ;;

  # -- hybrid_all_domains: LLM+NN all-domains run -------------------
  hybrid_all_domains)
    move_matching "hybrid_llm_nn_all_domains_*.json" "${TARGET}"
    ;;

  # -- instability: instability index figures + CSVs -----------------
  instability)
    move_matching "instability_analysis.csv"     "${TARGET}"
    # ISSUE 5a FIX: instability_extrapolation.csv (Stage 2 output, tab:instability)
    # was never moved — add it alongside instability_analysis.csv.
    move_matching "instability_extrapolation.csv" "${TARGET}"
    # ISSUE 5b FIX: *.png and *.pdf globs were workspace-wide (maxdepth 8),
    # which risks capturing runner debug images or Julia precompile artifacts.
    # Scope to the canonical figure name stems produced by run_instability_suite.py.
    move_matching "fig_paper_*.pdf"              "${TARGET}"
    move_matching "fig_paper_*.png"              "${TARGET}"
    move_matching "hypatiax_instability_*.pdf"   "${TARGET}"
    move_matching "hypatiax_instability_*.png"   "${TARGET}"
    ;;

  # -- extrap: OOD extrapolation comparative -------------------------
  # Also rescues standalone_llm_nn outputs written by some script variants
  extrap)
    move_matching "all_domains_extrap_v4_*.json" "${TARGET}"
    move_matching "all_domains_extrap_v4_*.txt"  "${TARGET}"
    # ISSUE 4 FIX: xref lists protocol_core_*.json as an extrap output
    # (Tab 9 OOD columns) but this pattern was absent from the move block.
    # The verify step (line ~1840) already checks for it; now actually move it.
    move_matching "protocol_core_*.json"         "${TARGET}"
    move_matching "standalone_llm_nn_*.json"      "${RESULTS_DIR}/standalone_llm_nn"
    move_matching "standalone_real_methods_*.json" "${RESULTS_DIR}/standalone_llm_nn"
    ;;

esac

echo "=== RESULTS_DIR contents after move ==="
find "${RESULTS_DIR}/${RESULT_SUBDIR}" -type f | sort || true
