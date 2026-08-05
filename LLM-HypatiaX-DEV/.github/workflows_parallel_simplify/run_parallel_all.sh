#!/usr/bin/env bash
# ==============================================================================
#  run_parallel_all.sh  —  HypatiaX JMLR v3.0 local parallel reproduction pipeline
#
#  Mirrors the wave structure of ci_parallel_schedule_simplify.yml, running all
#  experiments via run_all.sh subprocesses launched in background where there
#  are no data dependencies, then waiting for them before proceeding.
#
#  Wave structure (matches parallel_dispatch_dag.svg):
#
#    Wave 1  ── exp1               (core noiseless DeFi benchmark)
#    Wave 2+3 ─ exp1b              (DeFi seed sweep + portfolio variance)
#               exp2_feynman       (Feynman SR noisy benchmark, per domain)
#               exp2               (combined five-system comparison)
#               hybrid_all_domains (LLM+NN all-domains, 10 domains)
#               exp3               (Nguyen-12 seed=42)
#               exp3b              (Nguyen-12 multi-seed 99/123/777/2024)
#               extrap             (OOD extrapolation comparative)
#               suppA              (DeFi routing improvement)
#               suppB              (noise sweep)
#               suppB_sc           (sample-complexity sweep)
#    Wave 4  ── instability        (Instability Index — reads Wave 1 exp1 JSON)
#    Wave 5  ── tables + figures   (post-processing — reads all JSON outputs)
#    Wave 6  ── validate           (cross-check against paper-reported values)
#
#  Usage:
#      run_parallel_all.sh [OPTIONS]
#
#  Options:
#      --from <step>           Resume from a named wave or step
#                              (wave1 | wave23 | wave4 | wave5 | validate)
#      --only <step>           Run a single wave/step and exit
#      --skip-wave1            Skip Wave 1 (exp1 results already present)
#      --skip-wave23           Skip Wave 2+3 (all independent results present)
#      --no-continue           Do NOT run tables/figures/validate after experiments
#      --dry-run               Pass --dry-run through to every run_all.sh call
#      --jobs <N>              Cap simultaneous Wave 2+3 background jobs (default: 11)
#      -h, --help              Print this message and exit
#
#  Environment (inherited by all child processes):
#      REPO_ROOT               Repo root (default: git rev-parse --show-toplevel)
#      RESULTS_DIR             Output tree root (default: <REPO_ROOT>/hypatiax/data/results)
#      EXPERIMENTS_DIR         Benchmarks dir
#      ANTHROPIC_API_KEY       Required for LLM-backed experiments
#      JOB_DEADLINE            Per-experiment wall-clock budget in seconds (default: 19800)
#      PYSR_POPULATIONS        Number of PySR populations (default: 30)
#      JULIA_NUM_THREADS       Julia thread count (default: 4)
#      ... (all run_all.sh env vars are transparently forwarded)
#
#  Exit codes:
#      0   All waves completed successfully (or were skipped).
#      1   One or more waves reported failure.
#
#  Log files:
#      <RESULTS_DIR>/parallel_run_<wave>_<exp>.log   — per-experiment log
#      <RESULTS_DIR>/parallel_run_summary.log        — final summary
# ==============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/hypatiax/data/results}"
RUN_ALL="${REPO_ROOT}/run_all.sh"

# ── Defaults ──────────────────────────────────────────────────────────────────
FROM_WAVE=""
ONLY_WAVE=""
SKIP_WAVE1=false
SKIP_WAVE23=false
NO_CONTINUE=false
DRY_RUN=false
MAX_JOBS=11          # cap on simultaneous Wave 2+3 background processes

# ── CLI parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --from)         FROM_WAVE="$2";  shift 2 ;;
    --only)         ONLY_WAVE="$2";  shift 2 ;;
    --skip-wave1)   SKIP_WAVE1=true; shift ;;
    --skip-wave23)  SKIP_WAVE23=true; shift ;;
    --no-continue)  NO_CONTINUE=true; shift ;;
    --dry-run)      DRY_RUN=true;    shift ;;
    --jobs)         MAX_JOBS="$2";   shift 2 ;;
    -h|--help)
      sed -n '2,50p' "$0" | sed 's/^#  \?//'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()    { echo -e "${GREEN}[parallel]${NC} $*"; }
warn()   { echo -e "${YELLOW}[WARN]${NC}    $*"; }
die()    { echo -e "${RED}[ERROR]${NC}   $*" >&2; exit 1; }
header() { echo -e "\n${CYAN}══════════════════════════════════════════════════════════════${NC}"; \
           echo -e "${CYAN}  $*${NC}"; \
           echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"; }

# _epoch — portable epoch
_epoch() { date +%s; }

# _elapsed <start_epoch> — print human-readable HH:MM:SS
_elapsed() {
  local secs=$(( $(_epoch) - $1 ))
  printf '%02dh %02dm %02ds' $(( secs/3600 )) $(( (secs%3600)/60 )) $(( secs%60 ))
}

# _run_all_step <wave_label> <step_id> <log_file>
#   Runs `run_all.sh --step <step_id>` (or dry-run equivalent).
#   Streams output to log_file AND to the terminal.
_run_all_step() {
  local wave="$1" step="$2" logfile="$3"
  local extra_args=()
  [[ "$DRY_RUN" == true ]] && extra_args+=( --dry-run )

  log "[${wave}] Starting: ${step}"
  if bash "${RUN_ALL}" --step "${step}" "${extra_args[@]}" \
       2>&1 | tee "${logfile}"; then
    log "[${wave}] Completed OK: ${step}  (log: ${logfile})"
    return 0
  else
    warn "[${wave}] FAILED: ${step}  (log: ${logfile})"
    return 1
  fi
}

# _should_run <wave>
#   Returns 0 (true) if the named wave should execute given --from / --only.
#   Wave ordering: wave1 < wave23 < wave4 < wave5 < validate
_WAVE_ORDER="wave1 wave23 wave4 wave5 validate"
_should_run() {
  local wave="$1"
  [[ -n "$ONLY_WAVE" && "$ONLY_WAVE" != "$wave" ]] && return 1
  if [[ -n "$FROM_WAVE" ]]; then
    local skip=true
    for w in $_WAVE_ORDER; do
      [[ "$w" == "$FROM_WAVE" ]] && skip=false
      [[ "$w" == "$wave"      ]] && break
    done
    [[ "$skip" == true ]] && return 1
  fi
  return 0
}

# ── Pre-flight checks ─────────────────────────────────────────────────────────
[[ -f "${RUN_ALL}" ]] || die "run_all.sh not found at: ${RUN_ALL}"
mkdir -p "${RESULTS_DIR}"

SUMMARY_LOG="${RESULTS_DIR}/parallel_run_summary.log"
PIPELINE_START=$(_epoch)

header "HypatiaX — local parallel reproduction pipeline"
log "REPO_ROOT   : ${REPO_ROOT}"
log "RESULTS_DIR : ${RESULTS_DIR}"
log "run_all.sh  : ${RUN_ALL}"
log "MAX_JOBS    : ${MAX_JOBS}"
log "DRY_RUN     : ${DRY_RUN}"
log "SKIP_WAVE1  : ${SKIP_WAVE1}"
log "SKIP_WAVE23 : ${SKIP_WAVE23}"
[[ -n "$FROM_WAVE"  ]] && log "FROM_WAVE   : ${FROM_WAVE}"
[[ -n "$ONLY_WAVE"  ]] && log "ONLY_WAVE   : ${ONLY_WAVE}"
echo ""

# Track overall success
OVERALL_OK=true

# ==============================================================================
#  Wave 1 — exp1 (sequential; instability depends on its JSON output)
# ==============================================================================
if _should_run wave1 && [[ "$SKIP_WAVE1" != true ]]; then
  header "Wave 1 — exp1 (core noiseless DeFi benchmark)"
  W1_LOG="${RESULTS_DIR}/parallel_run_wave1_exp1.log"
  W1_START=$(_epoch)

  if _run_all_step "Wave 1" exp1 "${W1_LOG}"; then
    log "Wave 1 completed in $(_elapsed $W1_START)"
  else
    warn "Wave 1 FAILED — instability (Wave 4) will be skipped."
    OVERALL_OK=false
    # Do not exit: Wave 2+3 is independent and can still run.
  fi
elif _should_run wave1 && [[ "$SKIP_WAVE1" == true ]]; then
  log "Wave 1 skipped (--skip-wave1)."
fi

# ==============================================================================
#  Wave 2+3 — all independent experiments (fully parallel)
#
#  Experiments are launched as background subprocesses up to MAX_JOBS at a
#  time.  Each writes its own log file.  We wait() for all of them before
#  continuing to Wave 4.
# ==============================================================================
# List of (experiment_id, descriptive_label) pairs.
WAVE23_EXPERIMENTS=(
  "exp1b:DeFi seed sweep + portfolio variance"
  "exp2_feynman:Feynman SR noisy benchmark"
  "exp2:Combined five-system comparison"
  "hybrid_all_domains:Hybrid LLM+NN all-domains"
  "exp3:Nguyen-12 seed=42"
  "exp3b:Nguyen-12 multi-seed"
  "extrap:OOD extrapolation"
  "suppA:DeFi routing improvement"
  "suppB:Noise sweep"
  "suppB_sc:Sample-complexity sweep"
)

if _should_run wave23 && [[ "$SKIP_WAVE23" != true ]]; then
  header "Wave 2+3 — ${#WAVE23_EXPERIMENTS[@]} independent experiments (parallel)"
  W23_START=$(_epoch)

  declare -A W23_PIDS=()      # experiment → background PID
  declare -A W23_LOGS=()      # experiment → log file path
  declare -A W23_STATUS=()    # experiment → "running" | "ok" | "fail"
  RUNNING=0

  for entry in "${WAVE23_EXPERIMENTS[@]}"; do
    exp="${entry%%:*}"
    desc="${entry#*:}"
    logfile="${RESULTS_DIR}/parallel_run_wave23_${exp}.log"
    W23_LOGS[$exp]="${logfile}"

    # Throttle to MAX_JOBS simultaneous background processes.
    # If we are already at the cap, wait for the oldest slot to free up.
    while (( RUNNING >= MAX_JOBS )); do
      for e in "${!W23_PIDS[@]}"; do
        [[ "${W23_STATUS[$e]}" != "running" ]] && continue
        pid="${W23_PIDS[$e]}"
        if ! kill -0 "$pid" 2>/dev/null; then
          # Process finished — harvest exit code
          if wait "$pid" 2>/dev/null; then
            W23_STATUS[$e]="ok"
            log "[Wave 2+3] Finished OK: ${e}"
          else
            W23_STATUS[$e]="fail"
            warn "[Wave 2+3] FAILED: ${e}  (log: ${W23_LOGS[$e]})"
            OVERALL_OK=false
          fi
          RUNNING=$(( RUNNING - 1 ))
        fi
      done
      (( RUNNING >= MAX_JOBS )) && sleep 5
    done

    # Launch the experiment in the background.
    log "[Wave 2+3] Launching: ${exp}  (${desc})"
    (
      extra_args=()
      [[ "$DRY_RUN" == true ]] && extra_args+=( --dry-run )
      bash "${RUN_ALL}" --step "${exp}" "${extra_args[@]}" \
        2>&1 | tee "${logfile}"
    ) &
    W23_PIDS[$exp]=$!
    W23_STATUS[$exp]="running"
    RUNNING=$(( RUNNING + 1 ))

    # Brief stagger to avoid filesystem / API burst at start.
    sleep 2
  done

  # Wait for all remaining background jobs.
  log "[Wave 2+3] All experiments launched — waiting for completion ..."
  for exp in "${!W23_PIDS[@]}"; do
    [[ "${W23_STATUS[$exp]}" != "running" ]] && continue
    pid="${W23_PIDS[$exp]}"
    if wait "$pid" 2>/dev/null; then
      W23_STATUS[$exp]="ok"
      log "[Wave 2+3] Finished OK: ${exp}"
    else
      W23_STATUS[$exp]="fail"
      warn "[Wave 2+3] FAILED: ${exp}  (log: ${W23_LOGS[$exp]})"
      OVERALL_OK=false
    fi
  done

  echo ""
  log "Wave 2+3 summary ($(_elapsed $W23_START) total):"
  for entry in "${WAVE23_EXPERIMENTS[@]}"; do
    exp="${entry%%:*}"
    status="${W23_STATUS[$exp]:-not_started}"
    if [[ "$status" == "ok" ]]; then
      echo -e "  ${GREEN}OK${NC}   ${exp}"
    else
      echo -e "  ${RED}FAIL${NC} ${exp}  (see ${W23_LOGS[$exp]})"
    fi
  done

elif _should_run wave23 && [[ "$SKIP_WAVE23" == true ]]; then
  log "Wave 2+3 skipped (--skip-wave23)."
fi

# ==============================================================================
#  Wave 4 — instability  (requires Wave 1 exp1 JSON)
# ==============================================================================
if _should_run wave4; then
  header "Wave 4 — instability (Regime A/B/C + 12 figures)"
  W4_LOG="${RESULTS_DIR}/parallel_run_wave4_instability.log"
  W4_START=$(_epoch)

  if _run_all_step "Wave 4" instability "${W4_LOG}"; then
    log "Wave 4 completed in $(_elapsed $W4_START)"
  else
    warn "Wave 4 FAILED."
    OVERALL_OK=false
  fi
fi

# ==============================================================================
#  Wave 5 — tables + figures  (post-processing; reads all JSON outputs)
# ==============================================================================
if _should_run wave5 && [[ "$NO_CONTINUE" != true ]]; then
  header "Wave 5 — tables + figures (post-processing)"

  W5_TABLE_LOG="${RESULTS_DIR}/parallel_run_wave5_tables.log"
  W5_FIG_LOG="${RESULTS_DIR}/parallel_run_wave5_figures.log"
  W5_START=$(_epoch)

  TABLES_OK=true
  FIGURES_OK=true

  log "[Wave 5] Launching: tables (background) ..."
  (
    extra_args=()
    [[ "$DRY_RUN" == true ]] && extra_args+=( --dry-run )
    bash "${RUN_ALL}" --step tables "${extra_args[@]}" 2>&1 | tee "${W5_TABLE_LOG}"
  ) &
  TABLES_PID=$!

  log "[Wave 5] Launching: figures (background) ..."
  (
    extra_args=()
    [[ "$DRY_RUN" == true ]] && extra_args+=( --dry-run )
    bash "${RUN_ALL}" --step figures "${extra_args[@]}" 2>&1 | tee "${W5_FIG_LOG}"
  ) &
  FIGURES_PID=$!

  wait "$TABLES_PID"  || { warn "[Wave 5] tables FAILED.";  TABLES_OK=false;  OVERALL_OK=false; }
  wait "$FIGURES_PID" || { warn "[Wave 5] figures FAILED."; FIGURES_OK=false; OVERALL_OK=false; }

  [[ "$TABLES_OK"  == true ]] && log "[Wave 5] tables  OK"
  [[ "$FIGURES_OK" == true ]] && log "[Wave 5] figures OK"
  log "Wave 5 completed in $(_elapsed $W5_START)"

elif _should_run wave5 && [[ "$NO_CONTINUE" == true ]]; then
  log "Wave 5 skipped (--no-continue)."
fi

# ==============================================================================
#  Wave 6 — validate  (cross-check against paper-reported values)
# ==============================================================================
if _should_run validate && [[ "$NO_CONTINUE" != true ]]; then
  header "Wave 6 — validate (numerical cross-check)"
  W6_LOG="${RESULTS_DIR}/parallel_run_wave6_validate.log"
  W6_START=$(_epoch)

  if _run_all_step "Wave 6" validate "${W6_LOG}"; then
    log "Wave 6 completed in $(_elapsed $W6_START)"
  else
    warn "Wave 6 FAILED — some numerical checks did not pass."
    OVERALL_OK=false
  fi

elif _should_run validate && [[ "$NO_CONTINUE" == true ]]; then
  log "Wave 6 skipped (--no-continue)."
fi

# ==============================================================================
#  Final summary
# ==============================================================================
PIPELINE_ELAPSED=$(_elapsed $PIPELINE_START)

header "Pipeline complete — ${PIPELINE_ELAPSED}"

{
  echo "======================================================================"
  echo "  HypatiaX parallel reproduction pipeline — summary"
  echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)   elapsed: ${PIPELINE_ELAPSED}"
  echo "======================================================================"
  echo "  Wave 1:   exp1                        ${W1_LOG:-skipped}"
  echo "  Wave 2+3: (see individual logs above)"
  for entry in "${WAVE23_EXPERIMENTS[@]:-}"; do
    [[ -z "$entry" ]] && continue
    exp="${entry%%:*}"
    echo "            ${exp}  ${W23_LOGS[$exp]:-skipped}"
  done
  echo "  Wave 4:   instability                 ${W4_LOG:-skipped}"
  echo "  Wave 5:   tables + figures            ${W5_TABLE_LOG:-skipped}  ${W5_FIG_LOG:-skipped}"
  echo "  Wave 6:   validate                    ${W6_LOG:-skipped}"
  echo ""
  echo "  Key outputs:"
  echo "    Results JSON : ${RESULTS_DIR}/"
  echo "    LaTeX tables : ${RESULTS_DIR}/tables/*.tex"
  echo "    Figures PDF  : ${RESULTS_DIR}/figures/*.pdf"
  echo ""
  echo "  Overall status: $( [[ "$OVERALL_OK" == true ]] && echo OK || echo FAILED )"
  echo "======================================================================"
} | tee "${SUMMARY_LOG}"

log "Summary log: ${SUMMARY_LOG}"

if [[ "$OVERALL_OK" != true ]]; then
  die "One or more waves FAILED.  Check the per-experiment logs above."
fi

log "All waves completed successfully."
