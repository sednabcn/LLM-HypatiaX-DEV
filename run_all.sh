#!/usr/bin/env bash
# =============================================================================
# run_all.sh — HypatiaX JMLR v3.0 full numerical reproduction pipeline
#
# FIX CRITICAL 1 : 'instability' → 'hybrid_all_domains' (CI naming alignment)
# FIX CRITICAL 2 : suppB_sc step added (sample-complexity sweep)
# FIX CRITICAL 3 : hybrid_llm_nn/all_domains (not /defi) used throughout
# FIX STEP-11-12 : tables (Step 11) + figures (Step 12) both write to
#                  ${RESULTS_DIR}/tables  and  ${RESULTS_DIR}/figures
#                  — previously tables wrote to ${REPO_ROOT}/scripts/paper/tables
# FIX WARN-2     : HYBRID_ALL_DOMAINS_EXPECTED corrected to 10-domain list that
#                  matches CI HYBRID_ALL_DOMAINS_IDS and ExperimentProtocolAll
# FIX STEP-ORDER : removed exp2_sym / exp2_hyb (no run-blocks exist for them)
# FIX-suppA-1    : suppA cd REPO_ROOT (not EXPERIMENTS_DIR) — fixes doubled-path
#                  ENOENT on all three Python scripts (hypatiax/core/..., etc.)
# FIX-suppA-2    : suppA mkdir -p results dirs before first tee — fixes
#                  "tee: No such file or directory" when run standalone
# FIX-suppA-3    : suppA runs all three scripts (run_hybrid_system_benchmark.py,
#                  test_enhanced_defi_extrapolation.py, analyze_hybrid_performance.py)
#                  with tee / tee -a into suppA_run.log
# FIX-exp1b-1    : exp1b cd REPO_ROOT (not EXPERIMENTS_DIR) — mirrors suppA-1/exp1 fix.
#                  hypatiax_defi_benchmark_v3c.py writes to os.getcwd()/hypatiax/data/results;
#                  calling from EXPERIMENTS_DIR doubled the path → ENOENT on all outputs.
# FIX-exp1b-2/3  : removed --noise-level 15 and --output-dir from exp1b invocation.
#                  Those flags are NOT in hypatiax_defi_benchmark_v3c.py's argparse;
#                  passing them caused "unrecognized arguments" SystemExit(2) (CI log line 426).
#                  The noise-level/output-dir concern is handled by the dest15 mv block.
# FIX-exp1b-4    : portfolio_variance_v3c2.py now guarded by a pre-flight JSON check.
#                  It reads the benchmark JSON as a prerequisite; when that file is absent
#                  df_pysr=None → AttributeError on line 375 "df_pysr.columns" (CI log line 448).
#                  Fix: skip with a warning when benchmark JSON not yet present; use || echo
#                  so a non-zero exit from the variance script doesn't abort the whole step.
# FIX-exp1b-5    : move block now searches both EXPERIMENTS_DIR and RESULTS_DIR root.
#                  After the cd REPO_ROOT fix, outputs land in RESULTS_DIR (not EXPERIMENTS_DIR),
#                  so the original single-root find missed them entirely.
# FIX-suppA-4    : suppA move block now searches REPO_ROOT, EXPERIMENTS_DIR, and RESULTS_DIR.
#                  After cd REPO_ROOT, run_hybrid_system_benchmark.py may write to RESULTS_DIR
#                  directly; searching only EXPERIMENTS_DIR missed all files.
# FIX-suppA-5    : suppA move glob aligned with CI YAML move_matching calls (lines 1455-1458).
#                  CI matches: consolidated_hybrid_*.json → hybrid_pysr/defi
#                              hybrid_llm_nn_all_domains_*.json → hybrid_llm_nn/all_domains
#                              ablation_exp1_*.json + hypatiax_defi_benchmark_v3_results* → RESULTS_DIR root
#                  run_all.sh previously matched hybrid_system*.json (wrong glob, not in CI).
# SYNC-ci (2026-05-14):
#   — git push now uses HEAD:ref_name (not hardcoded master)
#   — consolidate timeout-minutes: 30 added
#   — Upload consolidated artifact: if: always() added
#   — shard_matrix=[] emitted on empty-pending to let worker if-guard fire
#   — JOB_DEADLINE exported to exp3/exp3b subprocess env
#   — python3 -c IndentationErrors fixed (3 sites in worker step)
#
# FIX-NSHARDS1-AUDIT (2026-05-25):
#   — extrap: added --resume flag to match exp2_feynman; without it extrap re-runs
#     all 11 domains from scratch on every retry, ignoring the CI RESUME=true env var.
#     run_comparative_suite_benchmark_v2.py only honours --resume (not the env var).
#   — suppB: NOISE_LEVELS forwarded explicitly as env var to run_noise_sweep_benchmark.py
#     so custom dispatch inputs are respected; previously the script used its own default.
#   — suppB: --samples, --pysr-timeout, --method-timeout, --populations, --parsimony
#     now passed as CLI args (matching repro.yaml / CI values) rather than relying on
#     the script picking them up from the environment — eliminates the env-vs-CLI gap.
#   — suppB_sc: same repro.yaml CLI flag set added (--samples, --pysr-timeout,
#     --method-timeout, --populations, --parsimony) — mirrors suppB fix.
#   — exp1, exp2_feynman: confirmed correct for NSHARDS=1; no changes needed.
#
# STEP IDs (linear order):
#   env_check          → verify Python, PySR, API key
#   exp1               → core extrapolation benchmark (Tab 9, 10, 15 · Fig 9, 10)
#   exp1b              → DeFi seed sweep + portfolio variance (Tab 11-13 · Fig 11-13)
#   extrap             → OOD extrapolation comparative (Tab 9 OOD columns)
#   hybrid_all_domains → hybrid LLM+NN all-domains run (§10.9 hybrid table — one-shot)
#   instability        → Instability Index analysis + 12 figures (§10.9 Regime A/B/C)
#   exp2_feynman       → Feynman SR noisy benchmark (Tab 16-18 · Phase 2)
#   exp2_feynman_extrap
#   exp2               → Combined five-system comparison injection (Tab 19 full)
#   exp3               → Nguyen-12 benchmark (tab:nguyen12 · §10.8)
#   exp3b              → Nguyen-12 extended seeds 99/123/777/2024
#   suppA              → DeFi routing improvement experiments (Tab 11-13 routing)
#   suppB              → Noise sweep (Tab 28, 29 · suppB)
#   suppB_sc           → Sample-complexity sweep (Tab 29 · suppB)   ← FIX CRITICAL 2
#   tables             → Generate all LaTeX tables  → ${RESULTS_DIR}/tables/
#   figures            → Generate all paper figures → ${RESULTS_DIR}/figures/
#   validate           → Cross-check all result files against expected checksums
#
# FIXES (observ-02 audit 2026-05-27):
#   — FIX-suppA-BUG-A : purge_dir moved BEFORE run_hybrid_system_benchmark.py in suppA.
#                        Previously purge_dir ran after the script wrote its outputs,
#                        deleting all results (critical/breaking).
#   — FIX-NOISE_LEVELS : export NOISE_LEVELS globally at config level.
#                        Without this, suppB silently fell back to its internal default
#                        instead of the CI/dispatch value → silent reproducibility drift.
#   — FIX-PYSR_POPULATION : removed export PYSR_POPULATION=100 (singular).
#                        Only PYSR_POPULATIONS (plural, value 30) is read by scripts.
#                        The singular variable was never used but scripts calling
#                        os.getenv("PYSR_POPULATION") would silently get 100 (wrong).
#   — FIX-exp1b-D      : relaxed exp1b count=0 from hard exit 1 to conditional warning.
#                        A zero count is valid when the step is intentionally skipped;
#                        hard failure broke --from / shard-filter workflows.
#                        Override with SKIP_ALLOWED=true to suppress the warning.
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
# FIX-ABS-PATH: always resolve RESULTS_DIR to an absolute path.
# If the caller passed a relative path (e.g. RESULTS_DIR=hypatiax/data/results)
# scripts that cd before writing will produce doubled/wrong paths.
# realpath -m tolerates non-existent dirs (no --canonicalize-missing needed on macOS).
_RESULTS_RAW="${RESULTS_DIR:-${REPO_ROOT}/hypatiax/data/results}"
RESULTS_DIR="$(cd "$(dirname "${_RESULTS_RAW}")" 2>/dev/null && pwd)/$(basename "${_RESULTS_RAW}")" \
  || RESULTS_DIR="${REPO_ROOT}/hypatiax/data/results"
export RESULTS_DIR
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${REPO_ROOT}/hypatiax/experiments/benchmarks}"
# FIX PATH-1: GENERATION_DIR corrected to hypatiax/core/generation/ to match
# CI script_path: hypatiax/core/generation/hybrid_all_domains_llm_nn/hybrid_system_llm_nn_all_domains.py
# (was: hypatiax/experiments/generation — wrong tree; caused ENOENT on hybrid_all_domains step
#  and FIX TASK 7 domain-list validation both in run_all.sh and CI parity check)
GENERATION_DIR="${GENERATION_DIR:-${REPO_ROOT}/hypatiax/core/generation}"
CORE_DIR="${CORE_DIR:-${REPO_ROOT}/hypatiax/core}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${REPO_ROOT}/hypatiax/analysis}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${REPO_ROOT}/scripts}"

# PySR hyperparameters (Table 23)
# NOTE: PYSR_POPULATION (singular) removed — it was unused and conflicted with
# PYSR_POPULATIONS (plural) which is the variable actually read by all scripts.
# Any script using os.getenv("PYSR_POPULATION") was silently getting 100 instead
# of the paper value 30. Prefer PYSR_POPULATIONS throughout.
export PYSR_GENERATIONS=10000
export PYSR_TOURNAMENT_SIZE=3
export PYSR_CROSSOVER=0.9
export PYSR_MUTATION=0.1
export PYSR_PARETO_PRESSURE=0.001
export PYSR_SEED=42
# FIX-1: default was 2, then 4; CI and repro.yaml now use 30 (paper value).
# Local runs with fewer populations diverge from paper results.
export PYSR_POPULATIONS="${PYSR_POPULATIONS:-30}"

# FIX-B: export NOISE_LEVELS globally so CI and local runs are consistent.
# Without this, suppB silently falls back to the script's own default,
# causing reproducibility drift vs. CI (which sets this via dispatch input).
export NOISE_LEVELS="${NOISE_LEVELS:-0.0,0.5,1.0,5.0,10.0}"

# Method timeouts — mirrors ci_experiment.yml global env block.
# METHOD_TIMEOUT: PySR methods 5/6 budget (repro.yaml timeouts.method_seconds).
# LLM_METHOD_TIMEOUT: tight cap for LLM/NN-only steps (retained for any custom invocations).
export METHOD_TIMEOUT="${METHOD_TIMEOUT:-900}"
export LLM_METHOD_TIMEOUT="${LLM_METHOD_TIMEOUT:-120}"
# PYSR_FIT_WALL_TIMEOUT: hard per-fit wall-clock cap passed to DiscoveryConfig.
# PYSR_FIT_GRACE_SECS:   extra grace seconds before forceful kill after timeout.
# Both must be exported so worker sub-processes and Python scripts inherit them.
export PYSR_FIT_WALL_TIMEOUT="${PYSR_FIT_WALL_TIMEOUT:-1200}"
export PYSR_FIT_GRACE_SECS="${PYSR_FIT_GRACE_SECS:-120}"

# Feynman benchmark defaults (Appendix A)
# FIX-10: exported so subshells and child processes inherit the values.
export FEYNMAN_SAMPLES=200
export FEYNMAN_TIMEOUT=1100        # FIX-G2: paper value 1100s (was 900)
export FEYNMAN_NOISELESS_THRESHOLD=0.999999  # FIX-THRESHOLD: matches ci_experiment_simplify.yml (was 0.9999)

# Julia signal handling — FIX-6 (FIX-G10): must be set before any juliacall
# import so Julia segfaults produce traceable Python exceptions.
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes

# Julia threading — FIX-7: match CI env (JULIA_NUM_THREADS: "4", JULIA_EXCLUSIVE: "0")
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-4}"
export JULIA_EXCLUSIVE="${JULIA_EXCLUSIVE:-0}"

# Repro config — FIX-8 (FIX-G2): paper-quality hyperparameters loaded at runtime.
# Scripts that honour REPRO_CFG will prefer values from config/repro.yaml
# over their own compile-time defaults (e.g. FEYNMAN_TIMEOUT=1100 from paper).
export REPRO_CFG="${REPRO_CFG:-${REPO_ROOT}/config/repro.yaml}"

# Job deadline — FIX-9: CI passes JOB_DEADLINE=19800 (330 min) to run_all.sh.
# Set the same default locally so deadline-aware scripts behave consistently.
# Override with JOB_DEADLINE=0 to disable deadline enforcement locally.
export JOB_DEADLINE="${JOB_DEADLINE:-19800}"

# Expected domain list for hybrid_all_domains validation (FIX WARN-2)
# Must match ExperimentProtocolAll.get_all_domains() in experiment_protocol_all_30.py v4.1.
# FIX: removed "statistics", "finance", "other" (never existed in protocol);
#      added "fluid_dynamics" and "mathematics" (present in protocol).
HYBRID_ALL_DOMAINS_EXPECTED="biology,chemistry,economics,electromagnetism,fluid_dynamics,mathematics,mechanics,optics,quantum,thermodynamics"

# FIX-FEYNMAN_DOMAINS-HOIST: defined here (not at first use in exp2_feynman/extrap steps)
# so bash does not hit an unbound-variable error when expanding double-quoted run()
# arguments for those steps while running a different --step (e.g. exp1b).
# With set -euo pipefail, bash expands ${FEYNMAN_DOMAINS} in the argument list of every
# run() call that embeds it in a double-quoted string -- even when run() would skip the
# step -- causing 'unbound variable' before run() is ever entered.
FEYNMAN_DOMAINS="feynman_biology feynman_chemistry feynman_electrochemistry feynman_electromagnetism feynman_electrostatics feynman_magnetism feynman_mechanics feynman_optics feynman_probability feynman_quantum feynman_thermodynamics"

# ── CLI parsing ───────────────────────────────────────────────────────────────
ONLY_STEP=""
FROM_STEP=""
DRY_RUN=false

# FIX STEP-ORDER: removed exp2_sym and exp2_hyb — no run-blocks exist for them
# FIX CRITICAL 1: instability → hybrid_all_domains
# FIX CRITICAL 2: suppB_sc added after suppB
# SPLIT STEP 4: hybrid_all_domains (one-shot run) + instability (K-run II analysis)
_STEP_ORDER="env_check exp1 exp1b extrap hybrid_all_domains instability exp2_feynman exp2_feynman_extrap exp2 exp3 exp3b suppA suppB suppB_sc tables figures validate"

while [[ $# -gt 0 ]]; do
  case $1 in
    --step)   ONLY_STEP="$2"; shift 2 ;;
    --from)   FROM_STEP="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[run_all]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()  { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

run() {
  local step="$1" desc="$2"; shift 2
  [[ -n "$ONLY_STEP" && "$ONLY_STEP" != "$step" ]] && return 0
  if [[ -n "$FROM_STEP" ]]; then
    # Scan the ordered step list; once FROM_STEP is reached flip skip→false.
    # Break as soon as we hit the current step.  If skip is still true at that
    # point the current step precedes FROM_STEP → skip it.
    local skip=true
    for s in $_STEP_ORDER; do
      [[ "$s" == "$FROM_STEP" ]] && skip=false
      [[ "$s" == "$step"      ]] && break
    done
    [[ "$skip" == true ]] && return 0
  fi
  echo ""
  log "=== STEP: ${step} -- ${desc} ==="
  if [[ "$DRY_RUN" == true ]]; then
    echo "    [dry-run] $*"
  else
    "$@"
    log "--- DONE: ${step} ---"
  fi
}

# ── purge_dir — wipe stale result files before a fresh run ────────────────────
# Removes every JSON/CSV/log from the target dir that is NOT a checkpoint or
# .pkl resume file.  Called at the start of each experiment step so that
# timestamp-named outputs from prior local runs never contaminate the new run.
# Mirrors what CI prune_old() does for git-tracked files, but works locally
# where nothing is committed yet.
purge_dir() {
  local dir="$1"
  [[ -d "$dir" ]] || return 0
  local removed=0
  while IFS= read -r f; do
    fname=$(basename "$f")
    # Preserve checkpoint / resume artefacts so --resume keeps working.
    [[ "$fname" == *checkpoint* ]] && continue
    [[ "$fname" == *.pkl       ]] && continue
    [[ "$fname" == _merged*    ]] && continue
    [[ "$fname" == _stats*     ]] && continue
    [[ "$fname" == _analysis*  ]] && continue
    [[ "$fname" == _report*    ]] && continue
    rm -f "$f"
    removed=$((removed + 1))
  done < <(find "$dir" -maxdepth 1 -type f \
             \( -name "*.json" -o -name "*.csv" -o -name "*.log" -o -name "*.txt" \) 2>/dev/null)
  [[ $removed -gt 0 ]] && echo "[purge_dir] Removed $removed stale file(s) from $dir" || true
}

# ── STEP 0: env_check ─────────────────────────────────────────────────────────
run env_check "Verify environment (Python, Julia/PySR, API key, directories)" bash -c '
  set -e
  echo "Python: $(python3 --version)"
  python3 -c "import pysr; print(\"PySR:\", pysr.__version__)" || { echo "ERROR: pysr not installed"; exit 1; }
  python3 -c "import torch; print(\"PyTorch:\", torch.__version__)"
  python3 -c "import anthropic; print(\"anthropic SDK: ok\")"
  # BUG 10 FIX: claude-sonnet-4-20250514 (repro.yaml llm_model) requires SDK >= 0.40.0.
  # environment.yml was pinned to 0.28.0 which predates this model family.
  # Assert the minimum here so local runs fail fast with a clear message.
  python3 - <<'SDKCHECK'
import anthropic, sys
ver = tuple(int(x) for x in anthropic.__version__.split(".")[:3])
if ver < (0, 40, 0):
    print("ERROR: anthropic SDK " + anthropic.__version__ + " is too old; need >= 0.40.0 for claude-sonnet-4-20250514")
    sys.exit(1)
print("anthropic SDK version: " + anthropic.__version__ + " (>= 0.40.0 OK)")
SDKCHECK
  # BUG 4 FIX: the '[ $? -eq 0 ] || exit 1' guard that was here is dead code —
  # set -e (line above) exits the subshell immediately if python3 fails, so $?
  # is never checked. Removed to avoid misleading future readers.
  python3 -c "import sympy; print(\"SymPy:\", sympy.__version__)"
  python3 -c "import scipy; print(\"SciPy:\", scipy.__version__)"
  # FIX-11: match CI pip-installed + checked deps (scikit-learn, pyyaml, matplotlib, pmlb)
  python3 -c "import sklearn; print(\"scikit-learn:\", sklearn.__version__)" || { echo "ERROR: scikit-learn not installed"; exit 1; }
  python3 -c "import yaml; print(\"PyYAML: ok\")" || { echo "ERROR: pyyaml not installed"; exit 1; }
  python3 -c "import matplotlib; print(\"matplotlib:\", matplotlib.__version__)" || { echo "ERROR: matplotlib not installed"; exit 1; }
  python3 -c "import pmlb; print(\"pmlb: ok\")" || { echo "ERROR: pmlb not installed"; exit 1; }
  # ITEM 2 FIX: seaborn is required by statistical_analysis.py (exp1 step).
  # If it is missing the script crashes before producing any figures or stats,
  # leaving exp1 tables and PDFs empty.  Check here and self-heal so the run
  # never reaches the analysis step without it.
  python3 -c "import seaborn; print(\"seaborn:\", seaborn.__version__)" 2>/dev/null || {
    echo "WARNING: seaborn not found — installing now (required by statistical_analysis.py)"
    python3 -m pip install --quiet seaborn || { echo "ERROR: seaborn install failed"; exit 1; }
    python3 -c "import seaborn; print(\"seaborn: installed\", seaborn.__version__)"
  }
  [[ -n "${ANTHROPIC_API_KEY:-}" ]] || { echo "ERROR: ANTHROPIC_API_KEY not set"; exit 1; }
  echo "ANTHROPIC_API_KEY: set (${#ANTHROPIC_API_KEY} chars)"
  # FIX-13: echo all CI-parity env vars for auditability
  echo "PYSR_POPULATIONS: ${PYSR_POPULATIONS}"
  echo "JULIA_NUM_THREADS: ${JULIA_NUM_THREADS}"
  echo "JULIA_EXCLUSIVE: ${JULIA_EXCLUSIVE}"
  echo "PYTHON_JULIACALL_HANDLE_SIGNALS: ${PYTHON_JULIACALL_HANDLE_SIGNALS}"
  echo "FEYNMAN_SAMPLES: ${FEYNMAN_SAMPLES}"
  echo "FEYNMAN_TIMEOUT: ${FEYNMAN_TIMEOUT}"
  echo "FEYNMAN_NOISELESS_THRESHOLD: ${FEYNMAN_NOISELESS_THRESHOLD}"
  echo "JOB_DEADLINE: ${JOB_DEADLINE}s"
  echo "REPRO_CFG: ${REPRO_CFG}"
  # FIX-12: REPRO_CFG audit — mirrors CI FIX-G2 print_repro.py log
  if [ -f "${REPRO_CFG}" ]; then
    echo "repro.yaml found -- printing key values:"
    python3 -c "
import yaml, sys
with open(\"${REPRO_CFG}\") as f: cfg = yaml.safe_load(f)
for k, v in (cfg or {}).items(): print(f\"  {k}: {v}\")
" 2>/dev/null || echo "  (could not parse repro.yaml)"
  else
    echo "WARNING: repro.yaml not found at ${REPRO_CFG} -- using env defaults"
  fi
  echo "Results dir: '"${RESULTS_DIR}"'"
  # FIX CRITICAL 3: hybrid_llm_nn/all_domains (not /defi)
  # BUG 2 FIX: added extrapolation/multi_seed — exp3b now writes to this subdir
  # (was: extrapolation/) to avoid collision with exp3 outputs.
  # BUG 1 FIX: added comparison_results/feynman-tests/exp2_multi (exp2 tee target)
  # and bare extrapolation/ (exp3 RESULT_SUBDIR) — both present in the CI mkdir
  # step but absent here, causing tee/mv failures when those steps run standalone.
  # Mirrors ci_experiment.yml Create results directory structure step exactly.
  mkdir -p '"${RESULTS_DIR}"'/{comparison_results/{feynman-tests/{exp2,exp2_extrap,exp2_multi,noise-sweep,sample-complexity},noise-noiseless/{noiseless/defi,15},extrapolation},extrapolation/multi_seed,hybrid_llm_nn/{all_domains,defi},hybrid_pysr/{all_domains,defi},llm_guided/{all_domains,defi},standalone_llm_nn,figures,tables}
  mkdir -p '"${RESULTS_DIR}"'/extrapolation
  echo "Directory structure: ok"
'

# ── STEP 1: exp1 ──────────────────────────────────────────────────────────────
run exp1 "Core extrapolation benchmark (Tab 9, 10, 15 - Fig 9, 10)" bash -c "
  # FIX-exp1-cd: cd REPO_ROOT so statistical_analysis.py and any repo-relative
  # imports resolve correctly.  Mirrors the fix applied to exp1b, suppA, extrap.
  cd '${REPO_ROOT}'
  _DEFI_TARGET='${RESULTS_DIR}/comparison_results/noise-noiseless/noiseless/defi'
  mkdir -p \"\${_DEFI_TARGET}\"
  purge_dir \"\${_DEFI_TARGET}\"

  python3 '${EXPERIMENTS_DIR}/hypatiax_defi_benchmark_v3c.py' \
    --output-dir \"\${_DEFI_TARGET}\" \
    2>&1 | tee '${RESULTS_DIR}/exp1_run.log'

  python3 -c 'import seaborn' 2>/dev/null || \
    python3 -m pip install --quiet seaborn || \
    { echo 'ERROR: seaborn install failed — statistical_analysis.py will crash'; exit 1; }
  cd '${ANALYSIS_DIR}'
  python3 statistical_analysis.py \
    2>&1 | tee -a '${RESULTS_DIR}/exp1_run.log' \
  || echo 'WARNING: statistical_analysis.py exited non-zero — primary results already saved, continuing'

  echo '=== exp1 verification ==='
  find \"\${_DEFI_TARGET}\" -type f 2>/dev/null | sort || echo '  (directory empty)'
  COUNT_DEFI=\$(find \"\${_DEFI_TARGET}\" -name 'hypatiax_defi_benchmark_v3*results*.json' 2>/dev/null | wc -l)
  if [[ \"\${COUNT_DEFI}\" -eq 0 ]]; then
    echo 'WARNING: exp1 produced no result JSON in canonical target — check log above.'
  else
    echo \"OK: \${COUNT_DEFI} result file(s) confirmed in \${_DEFI_TARGET}\"
  fi
  echo '=== end exp1 verification ==='
"

# ── STEP 2: exp1b ─────────────────────────────────────────────────────────────
# FIX-exp1b-1: cd to REPO_ROOT (not EXPERIMENTS_DIR).
#   hypatiax_defi_benchmark_v3c.py hardcodes "hypatiax/data/results" relative
#   to os.getcwd().  When called from EXPERIMENTS_DIR, CWD becomes
#   .../hypatiax/experiments/benchmarks and outputs land in the doubled path
#   .../benchmarks/hypatiax/data/results/... — nothing downstream finds them.
#   Fix mirrors suppA-1 and exp1: stay at REPO_ROOT, invoke by full path.
#
# FIX-exp1b-2/3: removed --noise-level 15 and --output-dir.
#   hypatiax_defi_benchmark_v3c.py's argparse does NOT accept these flags:
#     usage: hypatiax_defi_benchmark_v3c.py [-h] [--resume] [--verify-fix5]
#            [--report-only] [--verbose] [--cases SUBSTRING [SUBSTRING ...]]
#   Passing them caused "error: unrecognized arguments" (log line 426) and an
#   immediate SystemExit(2) before any work was done.
#   The noise-level=15 / output-dir are encoded by setting RESULT_SUBDIR in
#   the plan job (CI YAML line 216) and via the dest15 mv block below — the
#   script itself writes to its hardcoded path, then we move the files.
#
# FIX-exp1b-4: portfolio_variance_v3c2.py guard.
#   This script reads portfolio_variance_seed_sweep.json and
#   hypatiax_defi_benchmark_v3c3_results.json as prerequisites.  When those
#   files do not exist yet (first run), df_pysr is None and line 375
#   "if 'success' not in df_pysr.columns" raises AttributeError.
#   Fix: skip portfolio_variance_v3c2.py if the benchmark JSON it needs has
#   not been produced yet, with a clear warning rather than a fatal crash.
#   Cross-reference: CI YAML safety-net (FIX-G5) rescues partial outputs;
#   portfolio_variance_v3c2.py is a post-processing script that must run
#   AFTER the benchmark JSON exists, not simultaneously with it.
run exp1b "DeFi seed sweep + portfolio variance (Tab 11-13 - Fig 11-13)" bash -c "
  cd '${REPO_ROOT}'
  DEFI_TASK_FILTER=portfolio \
  DEFI_SEEDS='42,99,123,777,2024' \
    python3 '${EXPERIMENTS_DIR}/hypatiax_defi_benchmark_v3c.py' \
      --resume \
      2>&1 | tee '${RESULTS_DIR}'/exp1b_run.log

  # FIX-exp1b-4: only run portfolio_variance_v3c2.py when its input JSON exists.
  # It needs hypatiax_defi_benchmark_v3*results*.json in RESULTS_DIR or
  # portfolio_variance_seed_sweep.json — both written by the step above.
  _BENCH_JSON=\$(ls -t '${RESULTS_DIR}/comparison_results/noise-noiseless/noiseless/defi'/hypatiax_defi_benchmark_v3*results*.json 2>/dev/null | head -1 || true)
  if [[ -z \"\${_BENCH_JSON}\" ]]; then
    echo 'WARNING: portfolio_variance_v3c2.py skipped — benchmark JSON not found in ${RESULTS_DIR}.'
    echo '         This is expected on the first shard run when hypatiax_defi_benchmark_v3c.py'
    echo '         writes its output to the doubled path or has not yet produced results.'
    echo '         Re-run exp1b after confirming the benchmark JSON is present.'
  else
    echo '[exp1b] Running portfolio_variance_v3c2.py against: '\"\${_BENCH_JSON}\"
    RESULTS_DIR='${RESULTS_DIR}' \
      python3 '${EXPERIMENTS_DIR}/portfolio_variance_v3c2.py' \
        2>&1 | tee -a '${RESULTS_DIR}'/exp1b_run.log \
      || echo 'WARNING: portfolio_variance_v3c2.py exited non-zero — primary benchmark results already saved, continuing'
  fi
  # ── Move exp1b outputs → RESULTS_DIR ─────────────────────────────────────
  # BUG A FIX: comparison_FIXED_<TS>.json filenames are not unique across shards
  # or repeated runs — the second writer silently overwrites the first in the repo.
  # Rename each file to include SHARD_INDEX (from CI env) and a short seed tag so
  # every output has a distinct name.  SHARD_INDEX defaults to 0 for local runs.
  _SHARD=\${SHARD_INDEX:-0}
  _SEED_TAG=\$(echo \"\${DEFI_SEEDS:-42}\" | tr ',' '_')

  dest15='${RESULTS_DIR}/comparison_results/noise-noiseless/15'

  mkdir -p \"\${dest15}\"
  purge_dir \"\${dest15}\"

  # move primary outputs
  # FIX-exp1b-1 (move block): after cd REPO_ROOT, hypatiax_defi_benchmark_v3c.py
  # writes to REPO_ROOT/hypatiax/data/results/ (its hardcoded relative path).
  # That resolves to RESULTS_DIR, so files land there directly — not in
  # EXPERIMENTS_DIR root as the original code assumed.  Search BOTH locations
  # so the move works whether the script writes to RESULTS_DIR root or
  # EXPERIMENTS_DIR root (e.g. if the script is run standalone from a different CWD).
  for _search_root in '${EXPERIMENTS_DIR}' '${RESULTS_DIR}'; do
    find \"\${_search_root}\" -maxdepth 1 \
    \( \
        -name 'defi_v3_*.json' \
        -o -name '*portfolio*variance*.json' \
        -o -name 'hypatiax_defi_benchmark_v3*results*.json' \
    \) | while IFS= read -r src; do

        # Skip if already inside dest15 (avoid self-move loop)
        [[ \"\$src\" == \"\${dest15}\"* ]] && continue

        fname=\$(basename \"\$src\")
        stem=\"\${fname%.*}\"
        ext=\"\${fname##*.}\"

        dst=\"\${dest15}/\${stem}_shard\${_SHARD}_seed\${_SEED_TAG}.\${ext}\"

        if [ -f \"\$src\" ]; then
            mv -v \"\$src\" \"\$dst\" || true
        fi
    done
  done

  # move comparison files
  for _search_root in '${EXPERIMENTS_DIR}' '${RESULTS_DIR}'; do
    find \"\${_search_root}\" -maxdepth 1 \
    \( \
        -name 'comparison_FIXED_*.json' \
        -o -name 'comparison_FIXED_*.txt' \
    \) | while IFS= read -r src; do

        [[ \"\$src\" == \"\${dest15}\"* ]] && continue

        fname=\$(basename \"\$src\")
        stem=\"\${fname%.*}\"
        ext=\"\${fname##*.}\"

        dst=\"\${dest15}/\${stem}_shard\${_SHARD}_seed\${_SEED_TAG}.\${ext}\"

        if [ -f \"\$src\" ]; then
            mv -v \"\$src\" \"\$dst\" || true
        fi
    done
  done

  # verification
  echo '=== exp1b verification ==='

  find \"\${dest15}\" -type f 2>/dev/null | sort

  count=\$(find \"\${dest15}\" -type f 2>/dev/null | wc -l)

  echo \"Files produced: \${count}\"

  # FIX-D: relax hard failure — count=0 is valid when the step was intentionally
  # skipped (e.g. shard filter, or --from started at a later step).
  # Set SKIP_ALLOWED=true to suppress this warning when skipping is expected.
  if [[ \"\${count}\" -eq 0 && \"\${SKIP_ALLOWED:-false}\" != \"true\" ]]; then
      echo 'WARNING: exp1b generated no files — set SKIP_ALLOWED=true if this step was intentionally skipped'
  elif [[ \"\${count}\" -eq 0 ]]; then
      echo 'NOTE: exp1b produced no files (step was skipped — SKIP_ALLOWED=true)'
  fi
"

# ── STEP 3: extrap ────────────────────────────────────────────────────────────
# Patch 4 — FULL REWRITE STEP 3
#
# Activates the OOD extrapolation path in run_comparative_suite_benchmark_v2.py
# via three argparse flags introduced in Patch 4 (line 3585):
#
#   --extrap               Enable STEP 3 OOD comparative mode (Tab 9 OOD columns).
#                          Without this flag the script runs the standard in-dist
#                          benchmark and extrap_r2 is never computed.
#
#   --extrap-multiplier X  OOD test range upper bound as a multiple of training max.
#                          Default / paper value: 2.0  →  test on [x_max … 2·x_max].
#                          Override via env: EXTRAP_MULTIPLIER (e.g. CI fast-mode 1.5).
#
#   --extrap-train-frac F  Fraction of each variable range used for training.
#                          Default / paper value: 0.8  →  train on [x_min … x_min + 0.8·Δx].
#                          Top 20 % of the in-distribution range is held out; OOD
#                          test begins at x_max (= x_min + Δx).
#                          Override via env: EXTRAP_TRAIN_FRAC.
#
# Output: comparison_results/extrapolation/all_domains_extrap_v4_<TS>.json
#         Schema includes extrap_r2 / extrap_rmse / extrap_error_pct per method
#         per equation — these are the Tab 9 OOD columns read by generate_tables.py.
#
# Env-override knobs (CI / ablation use):
#   EXTRAP_MULTIPLIER   (default: 2.0)   — paper "medium" OOD regime
#   EXTRAP_TRAIN_FRAC   (default: 0.8)   — paper train/test split fraction
# -----------------------------------------------------------------------------
run extrap "OOD extrapolation comparative run (Tab 9 OOD columns)" bash -c "
  # FIX-extrap-1: cd REPO_ROOT (not EXPERIMENTS_DIR) — same doubled-path fix as
  #   exp1, exp1b, suppA.  Invoke script by full path so os.getcwd()=REPO_ROOT.
  # FIX-extrap-2: per-domain loop matching CI YAML lines 1203-1237 exactly.
  #   Previous monolithic call had no --domain flag, so every invocation ran ALL
  #   domains regardless of SHARD_IDS, and results landed in the wrong path.
  #   Now loops over FEYNMAN_DOMAINS (same list as CI FEYNMAN_DOMAINS) and passes
  #   --domain and an absolute --output-dir on every invocation.
  cd '${REPO_ROOT}'
  mkdir -p '${RESULTS_DIR}/comparison_results/extrapolation'
  purge_dir '${RESULTS_DIR}/comparison_results/extrapolation'
  for DOMAIN_ID in ${FEYNMAN_DOMAINS}; do
    echo '=== extrap: domain='\${DOMAIN_ID}' ==='
    FEYNMAN_SAMPLES=${FEYNMAN_SAMPLES} \
    FEYNMAN_TIMEOUT=${FEYNMAN_TIMEOUT} \
    METHOD_TIMEOUT=${METHOD_TIMEOUT} \
    PYSR_FIT_WALL_TIMEOUT=${PYSR_FIT_WALL_TIMEOUT} \
    PYSR_FIT_GRACE_SECS=${PYSR_FIT_GRACE_SECS} \
    JOB_DEADLINE=${JOB_DEADLINE} \
      python3 '${EXPERIMENTS_DIR}/run_comparative_suite_benchmark_v2.py' \
        --benchmark feynman \
        --extrap \
        --extrap-multiplier \${EXTRAP_MULTIPLIER:-2.0} \
        --extrap-train-frac \${EXTRAP_TRAIN_FRAC:-0.8} \
        --domain \"\${DOMAIN_ID}\" \
        --samples ${FEYNMAN_SAMPLES} \
        --pysr-timeout ${FEYNMAN_TIMEOUT} \
        --method-timeout ${METHOD_TIMEOUT} \
        --populations ${PYSR_POPULATIONS} \
        --parsimony 0.01 \
        --use-transcendental-compositions \
        --nn-seeds 3 \
        --no-llm-cache \
        --checkpoint-name \"extrap_checkpoint_\${DOMAIN_ID}\" \
        --output-dir '${RESULTS_DIR}/comparison_results/extrapolation' \
        --resume \
        2>&1 | tee -a '${RESULTS_DIR}/extrap_run.log' \
      || echo 'WARNING: extrap domain '\${DOMAIN_ID}' exited non-zero — continuing'
  done
  echo 'extrap output: ${RESULTS_DIR}/comparison_results/extrapolation/'
  ls '${RESULTS_DIR}/comparison_results/extrapolation/' 2>/dev/null || true
"

# ── STEP 4: hybrid_all_domains ────────────────────────────────────────────────
# FIX CRITICAL 1 : renamed from 'instability' → 'hybrid_all_domains'
# FIX CRITICAL 3 : outputs written to hybrid_llm_nn/all_domains/ (not /defi)
# FIX WARN-2     : domain list validated against corrected 10-domain set
# FIX TASK 7     : runtime domain-list cross-check before the long run starts
#
# Runs the one-shot hybrid LLM+NN system across 10 domains (§10.9 hybrid table).
# Produces: hybrid_llm_nn/all_domains/hybrid_llm_nn_all_domains_<TS>.json
#
# NOTE: This step does NOT reproduce the §10.9 Instability Index (Regime A/B/C,
# Spearman ρ). That is STEP 4a (instability) which runs run_instability_suite.py
# against the K-run DeFi benchmark results from STEP 1 (exp1).
run hybrid_all_domains "Hybrid LLM+NN all-domains run -- 10 domains (SS10.9 hybrid)" bash -c "
  # ── FIX TASK 7: runtime domain-list validation ────────────────────────────
  ACTUAL_DOMAINS=\$(python3 - << 'PYEOF'
import importlib.util, sys, pathlib
# PATH-1 FIX: GENERATION_DIR = hypatiax/core/generation (matches CI script_path).
# Previously this comment said "hypatiax/experiments/generation/" — that was wrong.
spec = importlib.util.spec_from_file_location(
    'hybrid_mod',
    pathlib.Path('${GENERATION_DIR}/hybrid_all_domains_llm_nn/hybrid_system_llm_nn_all_domains.py')
    # PATH-1 FIX: GENERATION_DIR = hypatiax/core/generation (matches CI script_path)
)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except SystemExit:
    pass
domains = getattr(mod, 'DOMAINS', getattr(mod, 'ALL_DOMAINS', getattr(mod, 'DOMAIN_KEYS', None)))
if domains is None:
    try:
        from hypatiax.core.generation.hybrid_all_domains_llm_nn \
            .hybrid_system_llm_nn_all_domains import ExperimentProtocolAll
        domains = set(ExperimentProtocolAll().get_all_domains().keys())
    except Exception as e:
        print('UNKNOWN', file=sys.stderr); sys.exit(1)
print(','.join(sorted(str(d) for d in domains)))
PYEOF
  )
  EXPECTED_SORTED=\$(echo '${HYBRID_ALL_DOMAINS_EXPECTED}' | tr ',' '\n' | sort | tr '\n' ',' | sed 's/,\$//')
  ACTUAL_SORTED=\$(echo \"\${ACTUAL_DOMAINS}\" | tr ',' '\n' | sort | tr '\n' ',' | sed 's/,\$//')
  if [[ \"\${ACTUAL_SORTED}\" != \"\${EXPECTED_SORTED}\" ]]; then
    echo '[WARN] hybrid_all_domains domain list MISMATCH -- update HYBRID_ALL_DOMAINS_EXPECTED'
    echo '  Expected: '\"\${EXPECTED_SORTED}\"
    echo '  Actual  : '\"\${ACTUAL_SORTED}\"
    exit 1
  fi
  echo '[hybrid_all_domains] Domain-list OK: '\"\${ACTUAL_SORTED}\"
  # ── Main experiment — cd to GENERATION_DIR (hypatiax/core/generation) ───────
  # PATH-1 FIX: GENERATION_DIR now correctly points to hypatiax/core/generation/
  # matching CI script_path. Previous stale comment said "not CORE_DIR" — reversed.
  cd '${GENERATION_DIR}/hybrid_all_domains_llm_nn'
  # FIX-OUTDIR-1: --output-dir so outputs land in hybrid_llm_nn/all_domains/
  # matching CI RESULT_SUBDIR and validate glob. Previously no --output-dir
  # was passed; files landed in CWD and were never found by the validate check.
  purge_dir '${RESULTS_DIR}/hybrid_llm_nn/all_domains'
  python3 hybrid_system_llm_nn_all_domains.py \
    --samples '${FEYNMAN_SAMPLES}' \
    --output-dir '${RESULTS_DIR}/hybrid_llm_nn/all_domains' \
    2>&1 | tee '${RESULTS_DIR}'/hybrid_all_domains_run.log
"

# ── STEP 4a: instability ──────────────────────────────────────────────────────
# Reproduces §10.9 Instability Index: Regime A/B/C taxonomy, Spearman ρ,
# complexity–instability theorem, and all 12 instability figures (Groups A, B, C
# + extrapolation scatter EX).
#
# Data sources (auto-detected in priority order by run_instability_suite.py):
#   1. hypatiax_defi_variance_results.json           ← preferred (--variance run)
#   2. hypatiax_defi_benchmark_v3_results_<TS>Z.json ← timestamped multi-run files
#   3. hypatiax_defi_benchmark_v3_results.json        ← single-run fallback (II=0)
#
# To get meaningful II values (σ > 0), STEP 1 (exp1) must have been run with
# K ≥ 2 repeat runs or --variance mode.  A single exp1 run produces a valid
# instability_analysis.csv but all II values will be 0 (Regime A/B only).
#
# Outputs (all under ${RESULTS_DIR}/figures/):
#   instability_analysis.csv
#   instability_extrapolation.csv          (Stage 2, if benchmark JSON present)
#   fig_paper_complexity_vs_instability.{png,pdf}   ← KEY figure (§10.9 theorem)
#   fig_paper_instability_hist.{png,pdf}
#   fig_paper_regime_counts.{png,pdf}
#   hypatiax_instability_per_case.{png,pdf}
#   … (all 12 figure stems: Groups A + B + C + EX)
run instability "Instability Index analysis + all figures -- SS10.9 (Regime A/B/C - Groups A-C + EX)" bash -c "
  mkdir -p '${RESULTS_DIR}/figures'
  # Purge only instability-specific files; preserve exp1 benchmark JSONs.
  rm -f \
    '${RESULTS_DIR}/figures/instability_analysis.csv' \
    '${RESULTS_DIR}/figures/instability_extrapolation.csv' \
    2>/dev/null || true
  find '${RESULTS_DIR}/figures' -maxdepth 1 \
    \( -name 'fig_paper_*.pdf' -o -name 'fig_paper_*.png' \
       -o -name 'hypatiax_instability_*.pdf' -o -name 'hypatiax_instability_*.png' \) \
    -delete 2>/dev/null || true

  # Canonical exp1 output directory (matches RESULT_SUBDIR in CI YAML).
  # All hypatiax_defi_benchmark_v3*results*.json from exp1 are moved here
  # by the _exp1_body move block and CI move_matching.
  DEFI_DIR='${RESULTS_DIR}/comparison_results/noise-noiseless/noiseless/defi'

  BENCH_JSON=\$(ls -t \"\${DEFI_DIR}\"/hypatiax_defi_benchmark_v3*results*.json 2>/dev/null | head -1 || true)

  if [[ -n \"\${BENCH_JSON}\" ]]; then
    echo '[instability] Stage 2 extrapolation merge enabled: '\"\${BENCH_JSON}\"
    BENCH_ARG=\"--benchmark-json \${BENCH_JSON}\"
  else
    echo '[instability] No benchmark JSON found in '\"\${DEFI_DIR}\"' -- Stage 2 (EX figure) skipped.'
    echo '              Run STEP 1 (exp1) first to enable the EX figure.'
    BENCH_ARG=\"\"
  fi

  python3 '${EXPERIMENTS_DIR}/run_instability_suite.py' \
    --results-dir \"\${DEFI_DIR}\" \
    --out         '${RESULTS_DIR}/figures' \
    --csv-out     '${RESULTS_DIR}/figures/instability_analysis.csv' \
    \${BENCH_ARG} \
    --format png pdf \
    2>&1 | tee '${RESULTS_DIR}'/instability_run.log
"


# ── STEP 5: exp2_feynman ──────────────────────────────────────────────────────
# SYNC-ci: per-domain loop matching ci_experiment.yml exp2_feynman worker step.
# BUG 1 + BUG 4 FIX (ci parity): previous monolithic call ran ALL 11 Feynman
#   domains on a single worker (no --domain filter) and omitted --output-dir,
#   so results landed in the default comparison_results/ path rather than
#   comparison_results/feynman-tests/exp2/ (RESULT_SUBDIR).
# All 6 methods active; METHOD_TIMEOUT (900s) gives methods 5+6 (SymbolicEngine, HybridV50_2)
#   adequate PySR budget.
# --noiseless --threshold 0.9999: exp2_feynman uses the noiseless Feynman
#   protocol, matching FEYNMAN_NOISELESS_THRESHOLD from repro.yaml.
# --parsimony 0.01 --populations: matches CI worker invocation exactly.
# Domains: 11 Feynman sub-domains derived from experiment_protocol_benchmark_v2.py
#   _build_domain_map() — same list as CI FEYNMAN_DOMAIN_IDS.
# FIX-DOMAINS: removed feynman_astronomy + feynman_fluid_dynamics (don't exist in
# BenchmarkProtocol._build_domain_map()); added feynman_magnetism + feynman_probability
# (present in protocol). Matches CI FEYNMAN_DOMAINS authoritative list exactly.
# NOTE: FEYNMAN_DOMAINS is defined once at the top of the script (line ~152) and
# must not be re-assigned here — doing so produces two sources of truth that can
# silently diverge.  The hoisted definition is used by all steps that reference it.
run exp2_feynman "Feynman SR benchmark -- Phase 2 noisy protocol per-domain (Tab 16-18)" bash -c "
  # FIX-exp2_feynman-1: cd REPO_ROOT and invoke by full path (doubled-path fix).
  cd '${REPO_ROOT}'
  mkdir -p '${RESULTS_DIR}/comparison_results/feynman-tests/exp2'
  # purge_dir removes all stale JSON/CSV/log (not checkpoints) before re-run.
  purge_dir '${RESULTS_DIR}/comparison_results/feynman-tests/exp2'
  for DOMAIN_ID in ${FEYNMAN_DOMAINS}; do
    echo '=== exp2_feynman: domain='\${DOMAIN_ID}' ==='
    FEYNMAN_SAMPLES=${FEYNMAN_SAMPLES} \
    FEYNMAN_TIMEOUT=${FEYNMAN_TIMEOUT} \
    METHOD_TIMEOUT=${METHOD_TIMEOUT} \
    PYSR_FIT_WALL_TIMEOUT=${PYSR_FIT_WALL_TIMEOUT} \
    PYSR_FIT_GRACE_SECS=${PYSR_FIT_GRACE_SECS} \
    JOB_DEADLINE=${JOB_DEADLINE} \
      python3 '${EXPERIMENTS_DIR}/run_comparative_suite_benchmark_v2.py' \
        --benchmark feynman \
        --domain \"\${DOMAIN_ID}\" \
        --samples ${FEYNMAN_SAMPLES} \
        --pysr-timeout ${FEYNMAN_TIMEOUT} \
        --method-timeout ${METHOD_TIMEOUT} \
        --populations ${PYSR_POPULATIONS} \
        --parsimony 0.01 \
        --noiseless \
        --threshold ${FEYNMAN_NOISELESS_THRESHOLD} \
        --checkpoint-name \"feynman_exp2_checkpoint_\${DOMAIN_ID}\" \
        --output-dir '${RESULTS_DIR}/comparison_results/feynman-tests/exp2' \
        --resume \
      2>&1 | tee -a '${RESULTS_DIR}/comparison_results/feynman-tests/exp2/exp2_run.log' \
    || echo 'WARNING: domain '\${DOMAIN_ID}' exited non-zero — continuing'
  done
"

# ── STEP 5c: exp2_feynman_extrap ──────────────────────────────────────────────
# Generates extrap_r2_far for every Feynman equation by re-running
# run_comparative_suite_benchmark_v2.py with --extrap on the same domain set
# as exp2_feynman.
#
# WHY THIS STEP EXISTS
# The main exp2_feynman run (STEP 5) trains each method on the full 200-sample
# dataset and records r2 / rmse (in-distribution).  run_analysis.py (ablation
# mode) additionally requires hypatia.extrap_r2_far / pysr_only.extrap_r2_far
# for every equation to run the Mann-Whitney test that is the paper's primary
# ablation claim (Table 14).  Without this step the field is never computed, the
# pairing fails, and the test exits with 0 pairs — this was the root cause of
# the "not a Mann-Whitney issue" diagnosis in the project log.
#
# WHAT --extrap DOES (run_comparative_suite_benchmark_v2.py, BUG 3 FIX)
#   1. Sorts each equation's samples by X[:,0] (first variable).
#   2. Trains every method on the first --extrap-train-frac (80%) of rows
#      — the "near" region.
#   3. After each method returns a formula string, re-evaluates that formula on
#      the remaining 20% of rows (the "far" region, beyond training max).
#   4. Records R² on the far region as extrap_r2_far in the result record and
#      in the flat benchmark_results.json (alongside the normal r2 field).
#
# OUTPUT SCHEMA (protocol_core_extrap_<TS>.json + benchmark_results.json)
#   Per record: { ..., "extrap_r2_far": { "method_name": float_or_null, ... } }
#   Per flat row: { ..., "extrap_r2_far": float_or_null }
#
# merge_extrap_into_benchmark.py (called by CI YAML exp2_feynman extrap step)
# reads these outputs alongside the noiseless benchmark_results.json and produces
# ablation_paired.json — the input schema run_analysis.py (ablation mode) needs.
#
# DATA CONDITIONS: --noiseless matches the main exp2_feynman run so r2 values
# are directly comparable.  --noiseless and --extrap are independent argparse
# flags (confirmed in BUG 3 FIX section of the script) and do not conflict.
#
# DOMAIN FILTER: DOMAIN_FILTER env var is set by CI to the shard's pending domain
# IDs (e.g. "feynman_biology feynman_chemistry").  ACTIVE_DOMAINS falls back to
# the full FEYNMAN_DOMAINS list when called locally without DOMAIN_FILTER.
run exp2_feynman_extrap "Feynman far-region R² (extrap_r2_far for Mann-Whitney ablation)" bash -c "
  cd '${REPO_ROOT}'
  mkdir -p '${RESULTS_DIR}/comparison_results/feynman-tests/exp2_extrap'
  # NOTE: purge_dir intentionally absent here — must NEVER be added.
  # CI calls this step exactly once; purge_dir would delete any results already
  # written by a prior resume attempt before the domain loop completes — causing
  # the verify step to see 0 files.  The CI move step's prune_old handles stale
  # committed files from prior workflow runs.
  # OUTPUT FILE: run_comparative_suite_benchmark_v2.py v2.2+ writes
  # benchmark_results_extrap.json (not benchmark_results.json) into --output-dir
  # when --extrap is active.  This name is mandatory: merge_extrap_into_benchmark.py
  # reads it via --extrap-benchmark-dir.  Do NOT rename, move, or purge this file.
  ACTIVE_DOMAINS=\"\${DOMAIN_FILTER:-${FEYNMAN_DOMAINS}}\"
  for DOMAIN_ID in \${ACTIVE_DOMAINS}; do
    echo '=== exp2_feynman_extrap: domain='\${DOMAIN_ID}' ==='
    FEYNMAN_SAMPLES=${FEYNMAN_SAMPLES} \
    FEYNMAN_TIMEOUT=${FEYNMAN_TIMEOUT} \
    METHOD_TIMEOUT=${METHOD_TIMEOUT} \
    PYSR_FIT_WALL_TIMEOUT=${PYSR_FIT_WALL_TIMEOUT} \
    PYSR_FIT_GRACE_SECS=${PYSR_FIT_GRACE_SECS} \
    JOB_DEADLINE=${JOB_DEADLINE} \
      python3 '${EXPERIMENTS_DIR}/run_comparative_suite_benchmark_v2.py' \
        --benchmark feynman \
        --extrap \
        --extrap-multiplier \${EXTRAP_MULTIPLIER:-2.0} \
        --extrap-train-frac \${EXTRAP_TRAIN_FRAC:-0.8} \
        --domain \"\${DOMAIN_ID}\" \
        --samples ${FEYNMAN_SAMPLES} \
        --pysr-timeout ${FEYNMAN_TIMEOUT} \
        --method-timeout ${METHOD_TIMEOUT} \
        --populations ${PYSR_POPULATIONS} \
        --parsimony 0.01 \
        --noiseless \
        --threshold ${FEYNMAN_NOISELESS_THRESHOLD} \
        --checkpoint-name \"feynman_extrap_checkpoint_\${DOMAIN_ID}\" \
        --output-dir '${RESULTS_DIR}/comparison_results/feynman-tests/exp2_extrap' \
        --resume \
      2>&1 | tee -a '${RESULTS_DIR}/comparison_results/feynman-tests/exp2_extrap/exp2_extrap_run.log' \
    || echo 'WARNING: exp2_feynman_extrap domain '\${DOMAIN_ID}' exited non-zero — continuing'
  done
  echo '=== exp2_feynman_extrap verification ==='
  find '${RESULTS_DIR}/comparison_results/feynman-tests/exp2_extrap' \
    -name 'protocol_core_extrap_*.json' 2>/dev/null | sort || echo '  (none yet)'
  COUNT_EXTRAP=\$(find '${RESULTS_DIR}/comparison_results/feynman-tests/exp2_extrap' \
    -name 'protocol_core_extrap_*.json' 2>/dev/null | wc -l)
  COUNT_BENCH_EXTRAP=\$(find '${RESULTS_DIR}/comparison_results/feynman-tests/exp2_extrap' \
    -name 'benchmark_results_extrap.json' 2>/dev/null | wc -l)
  if [[ \"\${COUNT_EXTRAP}\" -eq 0 ]]; then
    echo 'WARNING: exp2_feynman_extrap produced no protocol_core_extrap_*.json — extrap_r2_far will be missing from ablation_paired.json'
  else
    echo \"OK: \${COUNT_EXTRAP} extrap protocol file(s) produced\"
  fi
  if [[ \"${COUNT_BENCH_EXTRAP}\" -eq 0 ]]; then
    echo 'WARNING: benchmark_results_extrap.json not found — ci_analysis.yml merge step will find nothing'
    echo '  Ensure run_comparative_suite_benchmark_v2.py v2.2+ is in use (writes this file when --extrap is active)'
  else
    echo 'OK: benchmark_results_extrap.json present — ci_analysis.yml will merge into ablation_paired.json in exp2_extrap/'
  fi
  # NOTE: merge_extrap_into_benchmark.py is intentionally NOT called here.
  # ci_analysis.yml is the sole owner of the merge: it reads benchmark_results_extrap.json
  # from exp2_extrap/ and writes ablation_paired.json to exp2_extrap/.
  # Running the merge here would write to exp2/ (wrong path) and race ci_analysis.
\"


# FIX: --protocol all30 does not exist in run_comparative_suite_benchmark_v2.py
#      argparse — it caused SystemExit(2) on every worker (confirmed in CI BUG 2 fix).
#      Replaced with --benchmark both which runs both Feynman + SRBench protocols
#      (ExperimentProtocolAll, 30 multi-domain equations, Tab 19).
# FIX: mkdir -p ensures tee target directory exists when this step runs
#      standalone (--step exp2) without a prior env_check.
# All 6 methods active; METHOD_TIMEOUT (900s) gives methods 5+6 (SymbolicEngine, HybridV50_2)
# adequate PySR budget.
run exp2 "Combined five-system comparison -- all Methods (Tab 19 full)" bash -c "
  # FIX-exp2-1: cd REPO_ROOT and invoke by full path (doubled-path fix).
  # FIX-exp2-2: per-domain loop matching CI YAML lines 1002-1031 exactly.
  #   Previous monolithic --benchmark both call ran ALL domains in one invocation;
  #   CI workers loop per-domain so each domain gets its own checkpoint + output.
  cd '${REPO_ROOT}'
  mkdir -p '${RESULTS_DIR}/comparison_results/feynman-tests/exp2_multi'
  purge_dir '${RESULTS_DIR}/comparison_results/feynman-tests/exp2_multi'
  EXP2_DOMAINS='mechanics thermodynamics electromagnetism fluid_dynamics optics quantum chemistry biology mathematics economics'
  for DOMAIN_ID in \${EXP2_DOMAINS}; do
    echo '=== exp2: domain='\${DOMAIN_ID}' ==='
    FEYNMAN_TIMEOUT=${FEYNMAN_TIMEOUT} \
    METHOD_TIMEOUT=${METHOD_TIMEOUT} \
    PYSR_FIT_WALL_TIMEOUT=${PYSR_FIT_WALL_TIMEOUT} \
    PYSR_FIT_GRACE_SECS=${PYSR_FIT_GRACE_SECS} \
    JOB_DEADLINE=${JOB_DEADLINE} \
      python3 '${EXPERIMENTS_DIR}/run_comparative_suite_benchmark_v2.py' \
        --benchmark both \
        --domain \"\${DOMAIN_ID}\" \
        --samples ${FEYNMAN_SAMPLES} \
        --pysr-timeout ${FEYNMAN_TIMEOUT} \
        --method-timeout ${METHOD_TIMEOUT} \
        --populations ${PYSR_POPULATIONS} \
        --parsimony 0.01 \
        --use-transcendental-compositions \
        --noiseless \
        --threshold ${FEYNMAN_NOISELESS_THRESHOLD} \
        --checkpoint-name \"exp2_checkpoint_\${DOMAIN_ID}\" \
        --output-dir '${RESULTS_DIR}/comparison_results/feynman-tests/exp2_multi' \
        --resume \
        2>&1 | tee -a '${RESULTS_DIR}/comparison_results/feynman-tests/exp2_multi/exp2_run.log' \
      || echo 'WARNING: domain '\${DOMAIN_ID}' exited non-zero — continuing'
  done
"

# ── STEP 7: exp3 ──────────────────────────────────────────────────────────────
# FIX: mkdir -p ensures results/extrapolation exists when running standalone.
run exp3 "Nguyen-12 benchmark -- SEED=42 (tab:nguyen12 - SS10.8)" bash -c '
  # FIX-exp3-1: cd REPO_ROOT and invoke by full path (doubled-path fix).
  cd '"'"'${REPO_ROOT}'"'"'
  mkdir -p '"'"'${RESULTS_DIR}/extrapolation'"'"'
  purge_dir '"'"'${RESULTS_DIR}/extrapolation'"'"'
  echo "=== exp3 seed 1/1: seed=42 | equations: N1-N12 (12 total) ==="
  RESULTS_DIR='${RESULTS_DIR}' \
    python3 '"'"'${EXPERIMENTS_DIR}/exp3_nguyen12_hybrid50v_02.py'"'"' \
    --seed 42 \
    2>&1 | tee '"'"'${RESULTS_DIR}'"'"'/exp3_run.log \
  || echo "WARNING: seed=42 exited non-zero — continuing"
  # FIX-4: CI RESULT_SUBDIR=extrapolation — move outputs to extrapolation/,
  # not to ${RESULTS_DIR}/ root.
  # FIX-OUTDIR-4: add CI-matching globs (full_run_*, report_hybrid_*, hybrid_defi_*)
  # CI Move step exp3 moves all four patterns; run_all.sh only moved *nguyen*.json.
  find '"'"'${RESULTS_DIR}'"'"' -maxdepth 1 \
    \( -name '"'"'*nguyen*seed42*.json'"'"' -o -name '"'"'*nguyen12*42*.json'"'"' \
       -o -name '"'"'full_run_*seed42*.json'"'"' -o -name '"'"'report_hybrid_*seed42*.json'"'"' \
       -o -name '"'"'hybrid_defi_*seed42*.json'"'"' \) \
    -exec mv -v {} '"'"'${RESULTS_DIR}/extrapolation/'"'"' \; 2>/dev/null || true
  find '"'"'${RESULTS_DIR}'"'"' -maxdepth 1 -name '"'"'experiment_registry.json'"'"' \
    -exec cp -v {} '"'"'${RESULTS_DIR}/extrapolation/'"'"' \; 2>/dev/null || true
  # -- Partial results summary after seed=42 ----------------------------------
  echo "--- exp3 partial results after seed=42 (1/1) ---"
  RESULT_DIR='"'"'${RESULTS_DIR}/extrapolation'"'"' python3 - <<'"'"'PYEOF'"'"'
import glob, json, os
result_dir = os.environ.get("RESULT_DIR", "")
run_files = (sorted(glob.glob(f"{result_dir}/**/full_run_*seed42*.json", recursive=True)) +
             sorted(glob.glob(f"{result_dir}/**/*seed42*.json", recursive=True)))
all_files = glob.glob(f"{result_dir}/**/*.json", recursive=True)
print(f"  seed=42: {len(run_files)} result file(s)  |  total JSON in {result_dir}: {len(all_files)}")
for f in run_files[-1:]:
    try:
        data = json.load(open(f))
        results = data.get("results") or data.get("equation_results") or []
        if isinstance(results, list) and results:
            print(f"  Per-equation summary ({os.path.basename(f)}):")
            for r in results:
                eq   = r.get("equation") or r.get("eq_id") or r.get("name", "?")
                r2   = r.get("r2") or r.get("r2_test") or r.get("r2_train")
                rmse = r.get("rmse") or r.get("rmse_test", "")
                stat = r.get("status", "")
                r2_s = f"{r2:.4f}" if isinstance(r2, float) else str(r2)
                print(f"    {str(eq):10s}  R2={r2_s:8s}  rmse={rmse}  {stat}")
        elif isinstance(results, dict):
            print(f"  Per-equation summary ({os.path.basename(f)}):")
            for eq, r in sorted(results.items()):
                r2 = r.get("r2") or r.get("r2_test") if isinstance(r, dict) else r
                r2_s = f"{r2:.4f}" if isinstance(r2, float) else str(r2)
                print(f"    {str(eq):10s}  R2={r2_s}")
    except Exception as e:
        print(f"  (could not parse {os.path.basename(f)}: {e})")
PYEOF
  echo "--- end partial results seed=42 ---"
'

# ── STEP 8: exp3b ─────────────────────────────────────────────────────────────
# BUG 2 FIX: exp3b now uses extrapolation/multi_seed/ as its RESULT_SUBDIR.
# Previously both exp3 and exp3b wrote to extrapolation/, causing the second
# run's git commit to overwrite the first's merged files.
# Mirrors ci_experiment.yml (exp3b RESULT_SUBDIR="extrapolation/multi_seed")
# and ci_consolidate_experiment.yml (exp3b → extrapolation/multi_seed case).
run exp3b "Nguyen-12 stability seeds 99/123/777/2024 (tab:nguyen12 extended)" bash -c "
  # FIX-exp3b-1: cd REPO_ROOT (not EXPERIMENTS_DIR) — same doubled-path bug as exp1b/exp1/suppA.
  # exp3_nguyen12_hybrid50v_02.py writes relative to os.getcwd(); cd EXPERIMENTS_DIR
  # produced .../benchmarks/hypatiax/data/results/... → outputs never found.
  # Mirrors the exp3 fix (cd REPO_ROOT + full path invocation).
  cd '${REPO_ROOT}'
  mkdir -p '${RESULTS_DIR}/extrapolation/multi_seed'
  purge_dir '${RESULTS_DIR}/extrapolation/multi_seed'
  for seed in 99 123 777 2024; do
    echo '--- exp3b seed='\$seed' ---'
    RESULTS_DIR='${RESULTS_DIR}' \
      python3 '${EXPERIMENTS_DIR}/exp3_nguyen12_hybrid50v_02.py' \
      --seed \$seed \
      2>&1 | tee -a '${RESULTS_DIR}'/exp3b_run.log
  done
  # BUG 2 FIX: target is extrapolation/multi_seed/ (not extrapolation/).
  # Prevents overwriting the exp3 seed=42 outputs that live in extrapolation/.
  # FIX-DIR: script writes to RESULTS_DIR root — search RESULTS_DIR, not EXPERIMENTS_DIR.
  # FIX-GLOB: exclude seed42 explicitly so exp3 output is never swept here.
  # FIX-OUTDIR-3: add CI-matching globs for exp3b (full_run_*, report_hybrid_*, hybrid_defi_*)
  # CI Move step moves all four patterns; run_all.sh was only moving *nguyen*.json.
  find '${RESULTS_DIR}' -maxdepth 1 \
    \( -name '*nguyen*.json' -o -name 'full_run_*.json' \
       -o -name 'report_hybrid_*.json' -o -name 'hybrid_defi_*.json' \) \
    ! -name '*seed42*' ! -name '*nguyen12*42*' \
    -exec mv -v {} '${RESULTS_DIR}/extrapolation/multi_seed/' \;
  find '${RESULTS_DIR}' -maxdepth 1 -name 'experiment_registry.json' \
    -exec cp -v {} '${RESULTS_DIR}/extrapolation/multi_seed/' \; 2>/dev/null || true
"

# ── STEP 9: suppA ─────────────────────────────────────────────────────────────
# FIX-suppA-1: cd to REPO_ROOT (not EXPERIMENTS_DIR) so all repo-relative paths
#   (hypatiax/core/..., hypatiax/experiments/..., hypatiax/analysis/...) resolve
#   correctly.  Previously cd '${EXPERIMENTS_DIR}' caused a doubled path prefix,
#   e.g. hypatiax/experiments/benchmarks/hypatiax/core/generation/... → ENOENT.
# FIX-suppA-2: mkdir -p the results dir here so tee never fails with ENOENT.
#   env_check creates the dirs, but suppA can be run standalone (--step suppA).
# FIX-suppA-3: use tee -a on the two subsequent Python calls so all output goes
#   to the same log file without truncating it.
run suppA "DeFi routing improvement experiments (Supplement A - Tab 11-13 routing)" bash -c "
  cd '${REPO_ROOT}'
  mkdir -p '${RESULTS_DIR}/hybrid_pysr/defi' '${RESULTS_DIR}/figures' '${RESULTS_DIR}/tables'
  purge_dir '${RESULTS_DIR}/hybrid_pysr/defi'
  python3 '${EXPERIMENTS_DIR}/run_hybrid_system_benchmark.py' \
    2>&1 | tee    '${RESULTS_DIR}'/suppA_run.log
  python3 hypatiax/experiments/tests/test_enhanced_defi_extrapolation.py \
    2>&1 | tee -a '${RESULTS_DIR}'/suppA_run.log
  python3 hypatiax/analysis/analyze_hybrid_performance.py \
    --results-dir '${RESULTS_DIR}' \
    2>&1 | tee -a '${RESULTS_DIR}'/suppA_run.log
  # FIX-suppA-2 (move block): search both REPO_ROOT and EXPERIMENTS_DIR.
  #   After cd REPO_ROOT, run_hybrid_system_benchmark.py writes relative to
  #   REPO_ROOT (or RESULTS_DIR if it honours that env var).  The original
  #   single-root find '${EXPERIMENTS_DIR}' missed all files after the cd fix.
  # FIX-suppA-glob: align with CI YAML move_matching calls (lines 1455-1458):
  #   CI matches: consolidated_hybrid_*.json → hybrid_pysr/defi
  #               hybrid_llm_nn_all_domains_*.json → hybrid_llm_nn/all_domains
  #               ablation_exp1_*.json             → RESULTS_DIR root
  #               hypatiax_defi_benchmark_v3_results* → RESULTS_DIR root
  #   run_all.sh previously matched hybrid_system*.json (wrong glob — that
  #   pattern was not in the CI move step and produced false moves).
  for _sroot in '${REPO_ROOT}' '${EXPERIMENTS_DIR}' '${RESULTS_DIR}'; do
    find \"\${_sroot}\" -maxdepth 1 -name 'consolidated_hybrid_*.json' \
      ! -path '${RESULTS_DIR}/hybrid_pysr/defi/*' \
      -exec mv -v {} '${RESULTS_DIR}/hybrid_pysr/defi/' \; 2>/dev/null || true
    find \"\${_sroot}\" -maxdepth 1 -name 'hybrid_llm_nn_all_domains_*.json' \
      ! -path '${RESULTS_DIR}/hybrid_llm_nn/all_domains/*' \
      -exec mv -v {} '${RESULTS_DIR}/hybrid_llm_nn/all_domains/' \; 2>/dev/null || true
    find \"\${_sroot}\" -maxdepth 1 -name 'ablation_exp1_*.json' \
      ! -path '${RESULTS_DIR}/*' \
      -exec mv -v {} '${RESULTS_DIR}/' \; 2>/dev/null || true
    find \"\${_sroot}\" -maxdepth 1 -name 'hypatiax_defi_benchmark_v3_results*' \
      ! -path '${RESULTS_DIR}/*' \
      -exec mv -v {} '${RESULTS_DIR}/' \; 2>/dev/null || true
  done
"

# ── STEP 10: suppB — noise sweep ─────────────────────────────────────────────
# FIX CRITICAL 2: noise sweep now its own step; sample-complexity in suppB_sc
run suppB "Noise sweep benchmark sigma in {0,0.5,1,5,10}% (Tab 28, 29 - Supplement B)" bash -c "
  # FIX-suppB-1: cd REPO_ROOT (not EXPERIMENTS_DIR) — same doubled-path bug as all other steps.
  # run_noise_sweep_benchmark.py uses os.getcwd()-relative paths; cd EXPERIMENTS_DIR
  # caused outputs to land in .../benchmarks/hypatiax/data/results/... → never found.
  cd '${REPO_ROOT}'
  purge_dir '${RESULTS_DIR}/comparison_results/feynman-tests/noise-sweep/noise-sweep'
  # FIX-suppB-2: NOISE_LEVELS forwarded explicitly so run_noise_sweep_benchmark.py
  # honours custom dispatch inputs (default: CI value 0.0,0.5,1.0,5.0,10.0).
  # Without this the script silently uses its own internal default, which may diverge
  # from the task-ID list the plan job built (noise{NL}__<domain> IDs).
  # FIX-suppB-3: --samples, --pysr-timeout, --method-timeout, --populations, --parsimony
  # forwarded as CLI flags to match repro.yaml paper-quality values.  The CI worker
  # exports these as env vars; passing them explicitly ensures the script respects them
  # even if it reads CLI args rather than the environment.
  # OUT_BASE and RESULTS_DIR both set to match CI's explicit dual-set (suppB/suppB_sc).
  # Scripts that read either var will resolve to the same canonical path.
  NOISE_LEVELS='${NOISE_LEVELS:-0.0,0.5,1.0,5.0,10.0}' \
  OUT_BASE='${RESULTS_DIR}' \
  RESULTS_DIR='${RESULTS_DIR}' \
    python3 '${EXPERIMENTS_DIR}/run_noise_sweep_benchmark.py' \
    --output-dir '${RESULTS_DIR}/comparison_results/feynman-tests/noise-sweep/noise-sweep' \
    --samples ${FEYNMAN_SAMPLES} \
    --pysr-timeout ${FEYNMAN_TIMEOUT} \
    --method-timeout ${METHOD_TIMEOUT} \
    --populations ${PYSR_POPULATIONS} \
    --parsimony 0.01 \
    2>&1 | tee '${RESULTS_DIR}'/suppB_run.log
"

# ── STEP 10b: suppB_sc — sample-complexity sweep ─────────────────────────────
# FIX CRITICAL 2: new dedicated step, previously missing from CI and run_all.sh
# Produces: Tab 29 sample-complexity columns · Supplement B §6
# Task format: sc_n{n}__{feynman_id}  →  n ∈ {50,100,200,500,750,1000}, 30 equations
# Output dir: comparison_results/feynman-tests/sample-complexity/
run suppB_sc "Sample-complexity sweep n in {50..1000} (Tab 29 - Supplement B SS6)" bash -c "
  # FIX-suppB_sc-1: cd REPO_ROOT (not EXPERIMENTS_DIR) — same doubled-path bug.
  cd '${REPO_ROOT}'
  purge_dir '${RESULTS_DIR}/comparison_results/feynman-tests/sample-complexity'
  # FIX-suppB_sc-2: --samples, --pysr-timeout, --method-timeout, --populations, --parsimony
  # forwarded as CLI flags to match repro.yaml paper-quality values (same fix as suppB).
  # OUT_BASE and RESULTS_DIR both set to match CI's explicit dual-set (suppB/suppB_sc).
  NOISE_LEVEL='5.0' \
  SC_SAMPLE_COUNTS='50,100,200,500,750,1000' \
  OUT_BASE='${RESULTS_DIR}' \
  RESULTS_DIR='${RESULTS_DIR}' \
    python3 '${EXPERIMENTS_DIR}/run_sample_complexity_benchmark.py' \
    --output-dir '${RESULTS_DIR}/comparison_results/feynman-tests/sample-complexity' \
    --samples ${FEYNMAN_SAMPLES} \
    --pysr-timeout ${FEYNMAN_TIMEOUT} \
    --method-timeout ${METHOD_TIMEOUT} \
    --populations ${PYSR_POPULATIONS} \
    --parsimony 0.01 \
    2>&1 | tee '${RESULTS_DIR}'/suppB_sc_run.log
"

# ── STEP 11: tables ──────────────────────────────────────────────────────────
# FIX STEP-11-12: output now goes to \${RESULTS_DIR}/tables/ (same tree as figures)
# Previously written to \${REPO_ROOT}/scripts/paper/tables which diverged from
# the path used by inventory_results() and tables-generator glob checks.
run tables "Generate all LaTeX tables from result JSONs -> \${RESULTS_DIR}/tables/" bash -c "
  mkdir -p '${RESULTS_DIR}/tables'
  cd '${REPO_ROOT}/tables'
  TABLE_OUTDIR='${RESULTS_DIR}/tables' \
  VERIFY_RESULTS_DIR='${RESULTS_DIR}' \
    python3 generate_tables.py \
      --results-dir '${RESULTS_DIR}' \
      --output-dir  '${RESULTS_DIR}/tables' \
      2>&1 | tee '${RESULTS_DIR}'/tables_run.log
  echo 'Tables written to: ${RESULTS_DIR}/tables/'
  ls '${RESULTS_DIR}/tables/'
"

# ── STEP 12: figures ─────────────────────────────────────────────────────────
# FIX STEP-11-12: confirmed output dir is \${RESULTS_DIR}/figures/ — consistent
# with Step 11 (tables) now also writing under \${RESULTS_DIR}/.
run figures "Generate all paper figures from results -> \${RESULTS_DIR}/figures/" bash -c "
  mkdir -p '${RESULTS_DIR}/figures'
  cd '${REPO_ROOT}/figures'
  python3 generate_figures.py \
    --results-dir '${RESULTS_DIR}' \
    --output-dir  '${RESULTS_DIR}/figures' \
    2>&1 | tee '${RESULTS_DIR}'/figures_run.log
  echo 'Figures written to: ${RESULTS_DIR}/figures/'
  ls '${RESULTS_DIR}/figures/'
"

# ── STEP 13: validate ────────────────────────────────────────────────────────
# FIX-validate: run() dispatches via "$@" which cannot forward a here-doc on stdin.
# Wrapping the inline Python in bash -c '...' with a single-quoted heredoc ensures
# the script body is passed as an argument (not stdin) and executes correctly.
run validate "Cross-check all results against paper-reported values" bash -c '
python3 - <<'"'"'PYEOF'"'"'
import json, os, glob, sys

RESULTS = os.environ.get('RESULTS_DIR', 'hypatiax/data/results')
TOLERANCE = 0.01

checks = []

def check(label, got, expected, tol=TOLERANCE):
    ok = abs(got - expected) <= tol * max(abs(expected), 1e-9)
    checks.append((label, got, expected, ok))
    print(f"  [{'OK' if ok else 'FAIL'}] {label}: got={got:.6f}, expected={expected:.6f}")
    return ok

print("\n=== Validating key numerical results against JMLR v3.0 ===\n")

# --- exp1 noiseless ---
noiseless_files = (
    sorted(glob.glob(f"{RESULTS}/comparison_results/noise-noiseless/noiseless/defi/hypatiax_defi_benchmark_v3*results*.json")) +
    sorted(glob.glob(f"{RESULTS}/comparison_results/noise-noiseless/noiseless/defi/protocol_core_noiseless_*.json"))
)
if noiseless_files:
    with open(noiseless_files[-1]) as f: data = json.load(f)
    hx = [r for r in data.get('results', []) if r.get('method') in ('hybrid_v40', 'Hybrid v40')]
    if hx:
        import statistics
        r2v = [r['r2_train'] for r in hx if 'r2_train' in r]
        if r2v:
            check("Hybrid v40 mean train R2",   statistics.mean(r2v),   0.931)
            check("Hybrid v40 median train R2", statistics.median(r2v), 1.000)
else:
    print("  [SKIP] exp1 noiseless results not found")

# --- exp2_feynman ---
exp2_files = sorted(glob.glob(f"{RESULTS}/comparison_results/feynman-tests/exp2/protocol_core_noisy_*.json"))
if exp2_files:
    with open(exp2_files[-1]) as f: data = json.load(f)
    rec = data.get('hybrid_deFi_recovery') or data.get('recovery_rate')
    if rec is not None:
        check("Hybrid DeFi recovery rate (Feynman noisy)", rec, 1.0, tol=0.001)
else:
    print("  [SKIP] exp2_feynman results not found")

# --- Mann-Whitney (Tab 14) ---
mw_files = sorted(glob.glob(f"{RESULTS}/exp1_rf01_mannwhitney*.json"))
if mw_files:
    with open(mw_files[-1]) as f: data = json.load(f)
    u = data.get('mann_whitney_u', data.get('U'))
    if u is not None: check("Mann-Whitney U (Hybrid v40 vs NN)", float(u), 0.0, tol=0.0)
    p = data.get('p_value', data.get('p'))
    if p is not None:
        ok = p < 1e-5
        checks.append(("p-value < 1e-5", p, 1.11e-6, ok))
        print(f"  [{'OK' if ok else 'FAIL'}] p-value < 1e-5: got={p:.2e}")
else:
    print("  [SKIP] Mann-Whitney results not found")

# --- FIX CRITICAL 1/3: hybrid_all_domains output in correct subdir ---
had = glob.glob(f"{RESULTS}/hybrid_llm_nn/all_domains/*.json")
ok = bool(had)
checks.append(("hybrid_all_domains output present (all_domains/)", 1.0 if ok else 0.0, 1.0, ok))
print(f"  [{'OK' if ok else 'FAIL'}] hybrid_llm_nn/all_domains/: {len(had)} JSON file(s)")

# --- STEP 4a: instability outputs present ---
inst_csv = os.path.isfile(f"{RESULTS}/figures/instability_analysis.csv")
checks.append(("instability_analysis.csv present", 1.0 if inst_csv else 0.0, 1.0, inst_csv))
print(f"  [{'OK' if inst_csv else 'FAIL'}] instability_analysis.csv")
inst_fig = glob.glob(f"{RESULTS}/figures/fig_paper_complexity_vs_instability.pdf")
ok_ifig = bool(inst_fig)
checks.append(("fig_paper_complexity_vs_instability.pdf present", 1.0 if ok_ifig else 0.0, 1.0, ok_ifig))
print(f"  [{'OK' if ok_ifig else 'FAIL'}] fig_paper_complexity_vs_instability.pdf (KEY SS10.9 figure)")

# --- FIX CRITICAL 2: suppB_sc output present ---
# Output path: comparison_results/feynman-tests/sample-complexity/
sc = (glob.glob(f"{RESULTS}/comparison_results/feynman-tests/sample-complexity/*.json") +
      glob.glob(f"{RESULTS}/comparison_results/feynman-tests/sample-complexity/**/*.json"))
ok = bool(sc)
checks.append(("suppB_sc output present (sample-complexity/)", 1.0 if ok else 0.0, 1.0, ok))
print(f"  [{'OK' if ok else 'FAIL'}] sample-complexity outputs: {len(sc)} file(s)")

# --- CRITICAL 4: suppB noise_sweep_*.json glob match ---
# tables-generator uses glob 'noise_sweep_*.json' to find suppB results.
# If run_noise_sweep_benchmark.py writes files under a different prefix,
# all suppB tables will contain placeholder text.
noise_sweep_matched = glob.glob(f"{RESULTS}/comparison_results/feynman-tests/noise-sweep/noise-sweep/noise_sweep_*.json")
noise_sweep_all     = glob.glob(f"{RESULTS}/comparison_results/feynman-tests/noise-sweep/noise-sweep/*.json")
if noise_sweep_all:
    ok = bool(noise_sweep_matched)
    checks.append(("suppB output matches noise_sweep_*.json glob (CRITICAL 4)", 1.0 if ok else 0.0, 1.0, ok))
    if not ok:
        bad = [os.path.basename(p) for p in noise_sweep_all[:5]]
        print(f"  [FAIL] noise-sweep/: {len(noise_sweep_all)} JSON(s) found but NONE match "
              f"noise_sweep_*.json. Actual filenames: {bad} -- reconcile script output prefix with tables-generator glob.")
    else:
        print(f"  [OK]   noise-sweep/: {len(noise_sweep_matched)} noise_sweep_*.json -- tables-generator glob OK")
else:
    print(f"  [SKIP] noise-sweep/: no JSON files found (suppB not yet run)")

# --- BUG 2 FIX: exp3b outputs must be in extrapolation/multi_seed/, not extrapolation/ ---
exp3b_files = glob.glob(f"{RESULTS}/extrapolation/multi_seed/*nguyen*.json")
ok_exp3b = bool(exp3b_files)
checks.append(("exp3b outputs in extrapolation/multi_seed/ (BUG 2)", 1.0 if ok_exp3b else 0.0, 1.0, ok_exp3b))
suffix_exp3b = " (exp3b not yet run)" if not ok_exp3b else ""
print(
    f"  [{'OK' if ok_exp3b else 'SKIP'}] extrapolation/multi_seed/: "
    f"{len(exp3b_files)} nguyen JSON(s){suffix_exp3b}"
)

# --- FIX STEP-11-12: tables and figures co-located under RESULTS_DIR ---
tbl = glob.glob(f"{RESULTS}/tables/*.tex")
fig = glob.glob(f"{RESULTS}/figures/*.pdf")
ok_tbl = bool(tbl); ok_fig = bool(fig)
checks.append(("tables in RESULTS_DIR/tables/", 1.0 if ok_tbl else 0.0, 1.0, ok_tbl))
checks.append(("figures in RESULTS_DIR/figures/", 1.0 if ok_fig else 0.0, 1.0, ok_fig))
print(f"  [{'OK' if ok_tbl else 'FAIL'}] {RESULTS}/tables/: {len(tbl)} .tex file(s)")
print(f"  [{'OK' if ok_fig else 'FAIL'}] {RESULTS}/figures/: {len(fig)} .pdf file(s)")

# --- Summary ---
total = len(checks); passed = sum(1 for item in checks if item[-1])
print(f"\n=== Result: {passed}/{total} checks passed ===")
if passed < total:
    print("FAILED:")
    for label, got, exp, ok in checks:
        if not ok: print(f"  FAIL: {label} (got={got}, expected={exp})")
    sys.exit(1)
else:
    print("All checks passed.")
PYEOF
'

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
log "============================================================"
log " HypatiaX reproduction pipeline COMPLETE"
log "============================================================"
echo ""
echo "  Key output locations:"
echo "    Results JSON:  ${RESULTS_DIR}/"
echo "    LaTeX tables:  ${RESULTS_DIR}/tables/*.tex"     # FIX STEP-11-12
echo "    Figures PDF:   ${RESULTS_DIR}/figures/*.pdf"    # consistent with tables
echo ""
echo "  Cross-reference with paper:"
echo "    Table 9          <- exp1              (core extrapolation)"
echo "    Table 11         <- exp1b             (DeFi routing)"
echo "    Table 17         <- exp2_feynman      (Feynman noisy)"
echo "    Table 19         <- exp2              (five-system comparison)"
echo "    Table 28         <- suppB             (noise sweep)"
echo "    Table 29 sc      <- suppB_sc          (sample complexity)"
echo "    tab:hybrid_all   <- hybrid_all_domains (SS10.9 hybrid system -- one-shot)"
echo "    tab:nguyen12     <- exp3              (extrapolation/)      seed=42"
echo "                    <- exp3b             (extrapolation/multi_seed/)  seeds 99/123/777/2024"
echo "    tab:instability  <- instability        (SS10.9 Regime A/B/C, Spearman rho, 12 figs)"
echo ""
echo "  Instability outputs (STEP 4a):"
echo "    ${RESULTS_DIR}/figures/instability_analysis.csv"
echo "    ${RESULTS_DIR}/figures/instability_extrapolation.csv  (Stage 2, if benchmark JSON found)"
echo "    ${RESULTS_DIR}/figures/fig_paper_complexity_vs_instability.{png,pdf}  <- KEY (SS10.9)"
echo "    ${RESULTS_DIR}/figures/fig_paper_instability_hist.{png,pdf}"
echo "    ${RESULTS_DIR}/figures/fig_paper_regime_counts.{png,pdf}"
echo "    ${RESULTS_DIR}/figures/hypatiax_instability_per_case.{png,pdf}"
echo "    (+ 8 more figure stems: Groups A, B, C full set + EX)"
echo ""
echo "  To rebuild the paper PDF:"
echo "    cd ${REPO_ROOT} && pdflatex jmlr-hypatiax-paper-final.tex"
echo ""
log "Done. See individual *_run.log files in ${RESULTS_DIR}/ for per-step output."
