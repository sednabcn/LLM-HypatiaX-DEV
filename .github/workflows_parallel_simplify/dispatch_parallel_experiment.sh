#!/usr/bin/env bash
# ==============================================================================
#  dispatch_parallel_experiment.sh  —  HypatiaX multi-shard workflow dispatcher
#
#  Dispatches N parallel runs of ci_experiment_simplify.yml (one per shard),
#  locates each run's GitHub run-ID, then emits:
#
#      PARALLEL_RUN_IDS=<id0>,<id1>,...,<idN-1>
#
#  on stdout for the calling step in ci_parallel_schedule_simplify.yml to
#  capture and pass to its Wait step.
#
#  Usage:
#      dispatch_parallel_experiment.sh <experiment> <n_shards> \
#          <task_ids_override> <workflow_file>
#
#  Environment (must be set by caller):
#      GH_TOKEN            — GitHub token with actions:write permission
#      GITHUB_REPOSITORY   — "owner/repo"
#      GITHUB_REF_NAME     — branch to dispatch on (e.g. "main")
#
#  Fixes vs original:
#    FIX-D1  shard_index and n_shards are now passed as --field arguments to
#            gh workflow run.  They were previously omitted, so every dispatched
#            run had inputs.shard_index = "" (default 0) and inputs.n_shards = 1,
#            causing ALL shard runs to execute the FULL task list.
#    FIX-D2  run-name in ci_experiment_simplify.yml now embeds "(shard N/M)".
#            locate_run_id() matches on that suffix via displayTitle.  The
#            original script matched on a suffix that the workflow never emitted,
#            so Strategy 1 always failed and the script burned 400-600 s in
#            fallback retries before giving up.
#    FIX-D3  Pre-dispatch timestamp safety margin extended from 5 s to 30 s to
#            absorb GitHub API propagation delay and inter-runner clock skew.
#    FIX-D4  gh run list --limit raised from 20 to 100.  A busy repo can have
#            >20 runs queued between dispatch and lookup, causing the shard run
#            to fall off the list before locate_run_id() ever sees it.
#    FIX-D5  poll_run_to_completion() removed.  Polling is the caller's
#            responsibility (the Wait step in ci_parallel_schedule_simplify.yml).
#            The original dead-code poll loop consumed the job's 355-min timeout
#            budget while duplicating work already done by the Wait step.
# ==============================================================================

set -euo pipefail

# -- Arguments -----------------------------------------------------------------
EXP="${1:?usage: dispatch_parallel_experiment.sh <experiment> <n_shards> <task_ids_override> <workflow_file>}"
N_SHARDS="${2:-1}"
TASK_IDS_OVERRIDE="${3:-}"
WORKFLOW_FILE="${4:-ci_experiment_simplify.yml}"

# -- Environment ---------------------------------------------------------------
REPO="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY must be set}"
BRANCH="${GITHUB_REF_NAME:-main}"

# -- Tunables ------------------------------------------------------------------
# FIX-D3: 30 s margin absorbs API propagation delay + inter-runner clock skew.
DISPATCH_MARGIN_SECS=30

# FIX-D4: 100 gives headroom even on a busy main branch with concurrent runs.
GH_LIST_LIMIT=100

LOCATE_MAX_ATTEMPTS=12     # × LOCATE_SLEEP = up to 120 s per shard
LOCATE_SLEEP=10            # seconds between locate retries
INTER_SHARD_SLEEP=5        # seconds between successive shard dispatches

# ==============================================================================
#  Helper: portable epoch timestamp (Linux + macOS)
# ==============================================================================
_epoch() { date +%s; }

_iso_from_epoch() {
  local ep="$1"
  date -u -d "@${ep}" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null \
    || date -u -r "${ep}" +%Y-%m-%dT%H:%M:%SZ   # macOS fallback
}

# ==============================================================================
#  locate_run_id  <shard_idx>
#
#  Finds the GitHub run-ID of the workflow run for shard <shard_idx> by matching
#  its displayTitle suffix "(shard <shard_idx>/<N_SHARDS>)".
#
#  FIX-D2: ci_experiment_simplify.yml now sets:
#      run-name: "HypatiaX - <exp> (shard <shard_index>/<n_shards>) . ci_experiment_simplify"
#  so this suffix is reliably present in the displayTitle field.
#
#  FIX-D3/D4: filters by createdAt >= BEFORE_ISO and --limit 100.
#
#  Writes the run-ID to stdout on success.  Returns 1 on failure.
# ==============================================================================
locate_run_id() {
  local shard_idx="$1"
  local suffix="(shard ${shard_idx}/${N_SHARDS})"
  local attempt rid

  for attempt in $(seq 1 $LOCATE_MAX_ATTEMPTS); do
    rid=$(gh run list \
      --repo     "${REPO}" \
      --workflow "${WORKFLOW_FILE}" \
      --branch   "${BRANCH}" \
      --limit    "${GH_LIST_LIMIT}" \
      --json     "databaseId,displayTitle,createdAt" \
      --jq       ".[] |
                    select(
                      (.displayTitle | test(\"\\\\Q${suffix}\\\\E\"; \"\")) and
                      .createdAt >= \"${BEFORE_ISO}\"
                    ) | .databaseId" \
      2>/dev/null | head -1 || true)

    if [[ -n "$rid" ]]; then
      echo "  shard ${shard_idx}: located run ${rid} (attempt ${attempt}/${LOCATE_MAX_ATTEMPTS})" >&2
      echo "$rid"
      return 0
    fi

    echo "  shard ${shard_idx}: run not visible yet (attempt ${attempt}/${LOCATE_MAX_ATTEMPTS})" \
         "— retrying in ${LOCATE_SLEEP}s ..." >&2
    sleep $LOCATE_SLEEP
  done

  echo "::error::Could not locate run for shard ${shard_idx} of '${EXP}'" \
       "after ${LOCATE_MAX_ATTEMPTS} attempts." >&2
  echo "  Expected displayTitle containing: '${suffix}'" >&2
  echo "  Workflow : ${WORKFLOW_FILE}  Branch : ${BRANCH}  Since : ${BEFORE_ISO}" >&2
  echo "  Searched ${GH_LIST_LIMIT} most recent runs on the branch." >&2
  echo "  Checklist:" >&2
  echo "    1. ci_experiment_simplify.yml run-name includes '(shard N/M)' — see FIX-D2" >&2
  echo "    2. workflow was dispatched with --field shard_index=${shard_idx}" >&2
  echo "    3. GH_LIST_LIMIT (${GH_LIST_LIMIT}) is large enough for the current queue depth" >&2
  return 1
}

# ==============================================================================
#  Main
# ==============================================================================

echo "======================================================================"
echo "  dispatch_parallel_experiment.sh"
echo "  experiment   : ${EXP}"
echo "  n_shards     : ${N_SHARDS}"
echo "  workflow     : ${WORKFLOW_FILE}"
echo "  branch       : ${BRANCH}"
echo "  repo         : ${REPO}"
echo "  task_override: ${TASK_IDS_OVERRIDE:-<none>}"
echo "======================================================================"

# Validate n_shards
if ! [[ "$N_SHARDS" =~ ^[1-9][0-9]*$ ]] || (( N_SHARDS > 16 )); then
  echo "::error::n_shards must be a positive integer ≤16 (got '${N_SHARDS}')" >&2
  exit 1
fi

# -- Pre-dispatch timestamp --------------------------------------------------
# FIX-D3: capture BEFORE dispatch with a margin to absorb clock skew and API
# propagation delay.  locate_run_id() filters runs with createdAt >= BEFORE_ISO,
# so any run created before the dispatch is excluded even if its name matches
# (e.g. a stale retried run from a previous invocation).
BEFORE_EPOCH=$(( $(_epoch) - DISPATCH_MARGIN_SECS ))
BEFORE_ISO=$(_iso_from_epoch "$BEFORE_EPOCH")

echo "Pre-dispatch timestamp : ${BEFORE_ISO}  (margin=${DISPATCH_MARGIN_SECS}s)"
echo ""

# -- Dispatch each shard -----------------------------------------------------
declare -a RUN_IDS=()

for shard_idx in $(seq 0 $(( N_SHARDS - 1 ))); do
  echo "--- Dispatching shard ${shard_idx}/${N_SHARDS} for '${EXP}' ---"

  # Build dispatch arguments.
  # FIX-D1: shard_index and n_shards are passed so ci_experiment_simplify.yml
  # can embed them in run-name AND the plan step can select the correct task
  # slice.  The original script omitted both, so every shard ran all tasks.
  DISPATCH_ARGS=(
    --repo  "${REPO}"
    --ref   "${BRANCH}"
    --field "experiment=${EXP}"
    --field "n_shards=${N_SHARDS}"
    --field "shard_index=${shard_idx}"
    --field "resume=true"
    --field "dry_run=false"
  )

  if [[ -n "${TASK_IDS_OVERRIDE}" ]]; then
    DISPATCH_ARGS+=( --field "task_ids_override=${TASK_IDS_OVERRIDE}" )
  fi

  if ! gh workflow run "${WORKFLOW_FILE}" "${DISPATCH_ARGS[@]}"; then
    echo "::error::gh workflow run failed for shard ${shard_idx} of '${EXP}'." >&2
    exit 1
  fi

  echo "  shard ${shard_idx}: dispatched OK."

  # Brief pause before locate: give the GitHub API time to register the run.
  sleep 3

  # Locate this shard's run-ID.
  rid=$(locate_run_id "${shard_idx}")
  if [[ -z "$rid" ]]; then
    echo "::error::locate_run_id returned empty string for shard ${shard_idx}." >&2
    exit 1
  fi

  RUN_IDS+=("$rid")
  echo "  shard ${shard_idx} → run ${rid}"
  echo ""

  # Stagger dispatches to reduce API burst and avoid race on run-list lookup.
  if (( shard_idx < N_SHARDS - 1 )); then
    sleep $INTER_SHARD_SLEEP
  fi
done

# -- Emit summary and run IDs ------------------------------------------------
echo "======================================================================"
echo "  All ${N_SHARDS} shard(s) dispatched for '${EXP}'"
for i in "${!RUN_IDS[@]}"; do
  echo "    shard ${i} → run ${RUN_IDS[$i]}"
done
echo "======================================================================"

# FIX-D5: this script no longer contains a poll loop (dead code removed).
# Polling is the caller's responsibility via the Wait step in
# ci_parallel_schedule_simplify.yml.  That step receives RUN_IDS and runs
# its own round-robin poll loop with a per-experiment deadline.

IDS_CSV=$(IFS=','; echo "${RUN_IDS[*]}")
echo "PARALLEL_RUN_IDS=${IDS_CSV}"
