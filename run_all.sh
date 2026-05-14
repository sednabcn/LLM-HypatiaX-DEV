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
# SYNC-ci (2026-05-14):
#   — git push now uses HEAD:ref_name (not hardcoded master)
#   — consolidate timeout-minutes: 30 added
#   — Upload consolidated artifact: if: always() added
#   — shard_matrix=[] emitted on empty-pending to let worker if-guard fire
#   — JOB_DEADLINE exported to exp3/exp3b subprocess env
#   — python3 -c IndentationErrors fixed (3 sites in worker step)
#
# STEP IDs (linear order):
#   env_check          → verify Python, PySR, API key
#   exp1               → core extrapolation benchmark (Tab 9, 10, 15 · Fig 9, 10)
#   exp1b              → DeFi seed sweep + portfolio variance (Tab 11-13 · Fig 11-13)
#   extrap             → OOD extrapolation comparative (Tab 9 OOD columns)
#   hybrid_all_domains → hybrid LLM+NN all-domains run (§10.9 hybrid table — one-shot)
#   instability        → Instability Index analysis + 12 figures (§10.9 Regime A/B/C)
#   exp2_feynman       → Feynman SR noisy benchmark (Tab 16-18 · Phase 2)
#   exp2               → Combined five-system comparison injection (Tab 19 full)
#   exp3               → Nguyen-12 benchmark (tab:nguyen12 · §10.8)
#   exp3b              → Nguyen-12 extended seeds 99/123/777/2024
#   suppA              → DeFi routing improvement experiments (Tab 11-13 routing)
#   suppB              → Noise sweep (Tab 28, 29 · suppB)
#   suppB_sc           → Sample-complexity sweep (Tab 29 · suppB)   ← FIX CRITICAL 2
#   tables             → Generate all LaTeX tables  → ${RESULTS_DIR}/tables/
#   figures            → Generate all paper figures → ${RESULTS_DIR}/figures/
#   validate           → Cross-check all result files against expected checksums
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/hypatiax/data/results}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${REPO_ROOT}/hypatiax/experiments/benchmarks}"
# FIX-5: GENERATION_DIR aligns with CI script path hypatiax/experiments/generation/
# (was CORE_DIR/generation which resolved to hypatiax/core/generation — wrong tree)
GENERATION_DIR="${GENERATION_DIR:-${REPO_ROOT}/hypatiax/experiments/generation}"
CORE_DIR="${CORE_DIR:-${REPO_ROOT}/hypatiax/core}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${REPO_ROOT}/hypatiax/analysis}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${REPO_ROOT}/scripts}"

# PySR hyperparameters (Table 23)
export PYSR_GENERATIONS=10000
export PYSR_POPULATION=100
export PYSR_TOURNAMENT_SIZE=3
export PYSR_CROSSOVER=0.9
export PYSR_MUTATION=0.1
export PYSR_PARETO_PRESSURE=0.001
export PYSR_SEED=42
# FIX-1: default was 2; CI uses 4 (workflow env PYSR_POPULATIONS: "4").
# Local runs with 2 populations diverge from paper results — align to 4.
export PYSR_POPULATIONS="${PYSR_POPULATIONS:-4}"

# Feynman benchmark defaults (Appendix A)
# FIX-10: exported so subshells and child processes inherit the values.
export FEYNMAN_SAMPLES=200
export FEYNMAN_TIMEOUT=1100        # FIX-G2: paper value 1100s (was 900)
export FEYNMAN_NOISELESS_THRESHOLD=0.9999

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

# ── CLI parsing ───────────────────────────────────────────────────────────────
ONLY_STEP=""
FROM_STEP=""
DRY_RUN=false

# FIX STEP-ORDER: removed exp2_sym and exp2_hyb — no run-blocks exist for them
# FIX CRITICAL 1: instability → hybrid_all_domains
# FIX CRITICAL 2: suppB_sc added after suppB
# SPLIT STEP 4: hybrid_all_domains (one-shot run) + instability (K-run II analysis)
_STEP_ORDER="env_check exp1 exp1b extrap hybrid_all_domains instability exp2_feynman exp2 exp3 exp3b suppA suppB suppB_sc tables figures validate"

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
  mkdir -p '"${RESULTS_DIR}"'/{comparison_results/{feynman-tests/{exp2,exp2_multi,noise-sweep,sample-complexity},noise-noiseless/{noiseless,15},extrapolation},extrapolation/multi_seed,hybrid_llm_nn/{all_domains,defi},hybrid_pysr/{all_domains,defi},llm_guided/{all_domains,defi},standalone_llm_nn,figures,tables}
  mkdir -p '"${RESULTS_DIR}"'/extrapolation
  echo "Directory structure: ok"
'

# ── STEP 1: exp1 ──────────────────────────────────────────────────────────────
run exp1 "Core extrapolation benchmark (Tab 9, 10, 15 - Fig 9, 10)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  python3 hypatiax_defi_benchmark_v3c.py \
    2>&1 | tee '${RESULTS_DIR}'/exp1_run.log
  cd '${ANALYSIS_DIR}'
  python3 statistical_analysis.py \
    2>&1 | tee -a '${RESULTS_DIR}'/exp1_run.log
  # ── Move exp1 outputs → RESULTS_DIR ──────────────────────────────────────
  find '${EXPERIMENTS_DIR}' -maxdepth 1 -name 'hypatiax_defi_benchmark_v3*results*.json' \
    -exec mv -v {} '${RESULTS_DIR}/' \;
  find '${EXPERIMENTS_DIR}' -maxdepth 1 -name 'ablation_*.json' \
    -exec mv -v {} '${RESULTS_DIR}/' \;
  find '${EXPERIMENTS_DIR}' -maxdepth 1 -name 'exp1_rf01_mannwhitney*.json' \
    -exec mv -v {} '${RESULTS_DIR}/' \;
"

# ── STEP 2: exp1b ─────────────────────────────────────────────────────────────
run exp1b "DeFi seed sweep + portfolio variance (Tab 11-13 - Fig 11-13)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  DEFI_TASK_FILTER=portfolio \
  DEFI_SEEDS='42,99,123,777,2024' \
    python3 hypatiax_defi_benchmark_v3c.py \
      2>&1 | tee '${RESULTS_DIR}'/exp1b_run.log
  python3 portfolio_variance_v3c2.py \
    2>&1 | tee -a '${RESULTS_DIR}'/exp1b_run.log
  # ── Move exp1b outputs → RESULTS_DIR ─────────────────────────────────────
  find '${EXPERIMENTS_DIR}' -maxdepth 1 -name 'defi_v3_*.json' \
    -exec mv -v {} '${RESULTS_DIR}/' \;
  find '${EXPERIMENTS_DIR}' -maxdepth 1 -name '*portfolio*variance*.json' \
    -exec mv -v {} '${RESULTS_DIR}/' \;
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
  cd '${EXPERIMENTS_DIR}'
  python3 run_comparative_suite_benchmark_v2.py \
    --extrap \
    --extrap-multiplier \${EXTRAP_MULTIPLIER:-2.0} \
    --extrap-train-frac \${EXTRAP_TRAIN_FRAC:-0.8} \
    --samples ${FEYNMAN_SAMPLES} \
    --pysr-timeout ${FEYNMAN_TIMEOUT} \
    --method-timeout ${FEYNMAN_TIMEOUT} \
    --no-llm-cache \
    2>&1 | tee '${RESULTS_DIR}'/extrap_run.log
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
# FIX-5: use GENERATION_DIR (hypatiax/experiments/generation/) — matches CI script path.
# Previously used CORE_DIR/generation/ (hypatiax/core/generation/) — wrong tree.
spec = importlib.util.spec_from_file_location(
    'hybrid_mod',
    pathlib.Path('${GENERATION_DIR}/hybrid_all_domains_llm_nn/hybrid_system_llm_nn_all_domains.py')
)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except SystemExit:
    pass
domains = getattr(mod, 'DOMAINS', getattr(mod, 'ALL_DOMAINS', getattr(mod, 'DOMAIN_KEYS', None)))
if domains is None:
    try:
        from hypatiax.experiments.generation.hybrid_all_domains_llm_nn \
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
  # ── Main experiment — FIX-5: cd to GENERATION_DIR (not CORE_DIR) ─────────
  cd '${GENERATION_DIR}/hybrid_all_domains_llm_nn'
  python3 hybrid_system_llm_nn_all_domains.py \
    --samples '${FEYNMAN_SAMPLES}' \
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

  # Locate the most recent DeFi benchmark JSON produced by STEP 1 (exp1) for
  # Stage 2 extrapolation merge and the EX figure.
  BENCH_JSON=\$(ls -t '${RESULTS_DIR}'/hypatiax_defi_benchmark_v3*results*.json 2>/dev/null | head -1 || true)

  if [[ -n \"\${BENCH_JSON}\" ]]; then
    echo '[instability] Stage 2 extrapolation merge enabled: '\"\${BENCH_JSON}\"
    BENCH_ARG=\"--benchmark-json \${BENCH_JSON}\"
  else
    echo '[instability] No benchmark JSON found -- Stage 2 (EX figure) skipped.'
    echo '              Run STEP 1 (exp1) first to enable the EX figure.'
    BENCH_ARG=\"\"
  fi

  python3 '${EXPERIMENTS_DIR}/run_instability_suite.py' \
    --results-dir '${RESULTS_DIR}' \
    --out         '${RESULTS_DIR}/figures' \
    --csv-out     '${RESULTS_DIR}/figures/instability_analysis.csv' \
    \${BENCH_ARG} \
    --format png pdf \
    2>&1 | tee '${RESULTS_DIR}'/instability_run.log
"


# ── STEP 5: exp2_feynman ──────────────────────────────────────────────────────
# FIX: mkdir -p ensures tee target directory exists when this step runs
#      standalone (--step exp2_feynman) without a prior env_check.
# --skip-pysr: methods 5+6 (SymbolicEngine, HybridV50_2) are excluded.
#   They are NOT in the paper's Tab 16-18 comparison.  Julia startup overhead
#   (~150s per test × 30 tests = 75 min) would blow the 5h30m job deadline.
#   Remove --skip-pysr here AND in ci_experiment.yml if you want them back.
run exp2_feynman "Feynman SR benchmark -- Phase 2 noisy protocol (Tab 16-18)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  mkdir -p '${RESULTS_DIR}/comparison_results/feynman-tests/exp2'
  python3 run_comparative_suite_benchmark_v2.py \
    --benchmark feynman \
    --skip-pysr \
    --samples ${FEYNMAN_SAMPLES} \
    --pysr-timeout ${FEYNMAN_TIMEOUT} \
    --checkpoint-name feynman_exp2_checkpoint \
    --resume \
    2>&1 | tee '${RESULTS_DIR}/comparison_results/feynman-tests/exp2/exp2_run.log'
"

# ── STEP 6: exp2 ──────────────────────────────────────────────────────────────
# FIX: --benchmark all30 was not a valid argparse choice.
# FIX: exp2 uses ExperimentProtocolAll (30 multi-domain equations, Tab 19).
#      --protocol all30 loads ExperimentProtocolAll from experiment_protocol_all_30.py.
#      --benchmark is not passed (not applicable to the all30 protocol path).
# FIX: mkdir -p ensures tee target directory exists when this step runs
#      standalone (--step exp2) without a prior env_check.
# --skip-pysr: methods 5+6 (SymbolicEngine, HybridV50_2) not in Tab 19 comparison.
#   Julia startup overhead (~150s per test × 30 tests) would exceed job deadline.
run exp2 "Combined five-system comparison -- all Methods (Tab 19 full)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  mkdir -p '${RESULTS_DIR}/comparison_results/feynman-tests/exp2_multi'
  python3 run_comparative_suite_benchmark_v2.py \
    --protocol all30 \
    --skip-pysr \
    --samples ${FEYNMAN_SAMPLES} \
    --pysr-timeout ${FEYNMAN_TIMEOUT} \
    --method-timeout ${FEYNMAN_TIMEOUT} \
    --checkpoint-name exp2_checkpoint \
    --resume \
    2>&1 | tee '${RESULTS_DIR}/comparison_results/feynman-tests/exp2_multi/exp2_run.log'
"

# ── STEP 7: exp3 ──────────────────────────────────────────────────────────────
# FIX: mkdir -p ensures results/extrapolation exists when running standalone.
run exp3 "Nguyen-12 benchmark -- SEED=42 (tab:nguyen12 - SS10.8)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  mkdir -p '${RESULTS_DIR}/extrapolation'
  python3 exp3_nguyen12_hybrid50v_02.py \
    --seed 42 \
    2>&1 | tee '${RESULTS_DIR}'/exp3_run.log
  # FIX-4: CI RESULT_SUBDIR=extrapolation — move outputs to extrapolation/,
  # not to ${RESULTS_DIR}/ root (was: -exec mv {} '${RESULTS_DIR}/' which lost the subdir).
  find '${EXPERIMENTS_DIR}' -maxdepth 1 \
    \( -name '*nguyen*seed42*.json' -o -name '*nguyen12*42*.json' \) \
    -exec mv -v {} '${RESULTS_DIR}/extrapolation/' \; 2>/dev/null || true
"

# ── STEP 8: exp3b ─────────────────────────────────────────────────────────────
# BUG 2 FIX: exp3b now uses extrapolation/multi_seed/ as its RESULT_SUBDIR.
# Previously both exp3 and exp3b wrote to extrapolation/, causing the second
# run's git commit to overwrite the first's merged files.
# Mirrors ci_experiment.yml (exp3b RESULT_SUBDIR="extrapolation/multi_seed")
# and ci_consolidate_experiment.yml (exp3b → extrapolation/multi_seed case).
run exp3b "Nguyen-12 stability seeds 99/123/777/2024 (tab:nguyen12 extended)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  mkdir -p '${RESULTS_DIR}/extrapolation/multi_seed'
  for seed in 99 123 777 2024; do
    echo '--- exp3b seed='\$seed' ---'
    python3 exp3_nguyen12_hybrid50v_02.py \
      --seed \$seed \
      2>&1 | tee -a '${RESULTS_DIR}'/exp3b_run.log
  done
  # BUG 2 FIX: target is extrapolation/multi_seed/ (not extrapolation/).
  # Prevents overwriting the exp3 seed=42 outputs that live in extrapolation/.
  find '${EXPERIMENTS_DIR}' -maxdepth 1 -name '*nguyen*.json' \
    -exec mv -v {} '${RESULTS_DIR}/extrapolation/multi_seed/' \;
"

# ── STEP 9: suppA ─────────────────────────────────────────────────────────────
run suppA "DeFi routing improvement experiments (Supplement A - Tab 11-13 routing)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  python3 run_hybrid_system_benchmark.py \
    2>&1 | tee '${RESULTS_DIR}'/suppA_run.log
  # FIX-2: CI RESULT_SUBDIR=hybrid_pysr/defi — move outputs there, not hybrid_llm_nn/defi/.
  # FIX-3: removed second mv of hybrid_system*.json to hybrid_llm_nn/all_domains/ —
  #         suppA is a DeFi routing run, not a hybrid_all_domains run; those files
  #         belong in hybrid_pysr/defi/ alongside the consolidated outputs.
  find '${EXPERIMENTS_DIR}' -maxdepth 1 \
    \( -name 'consolidated_hybrid*.json' -o -name 'hybrid_system*.json' \) \
    -exec mv -v {} '${RESULTS_DIR}/hybrid_pysr/defi/' \;
"

# ── STEP 10: suppB — noise sweep ─────────────────────────────────────────────
# FIX CRITICAL 2: noise sweep now its own step; sample-complexity in suppB_sc
run suppB "Noise sweep benchmark sigma in {0,0.5,1,5,10}% (Tab 28, 29 - Supplement B)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  python3 run_noise_sweep_benchmark.py \
    2>&1 | tee '${RESULTS_DIR}'/suppB_run.log
  # ── Flatten per-equation subdirs → noise-sweep/ (CRITICAL 4 glob fix) ────
  # run_noise_sweep_benchmark.py writes into per-equation subdirs
  # (e.g. noise-sweep/I.12.1-correction/noise_sweep_*.json).
  # generate_tables.py globs noise-sweep/noise_sweep_*.json — move up one level.
  find '${RESULTS_DIR}/comparison_results/feynman-tests/noise-sweep' \
    -mindepth 2 -name 'noise_sweep_*.json' \
    -exec mv -v {} '${RESULTS_DIR}/comparison_results/feynman-tests/noise-sweep/' \;
"

# ── STEP 10b: suppB_sc — sample-complexity sweep ─────────────────────────────
# FIX CRITICAL 2: new dedicated step, previously missing from CI and run_all.sh
# Produces: Tab 29 sample-complexity columns · Supplement B §6
# Task format: sc_n{n}__{feynman_id}  →  n ∈ {50,100,200,500,750,1000}, 30 equations
# Output dir: comparison_results/feynman-tests/sample-complexity/
run suppB_sc "Sample-complexity sweep n in {50..1000} (Tab 29 - Supplement B SS6)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  NOISE_LEVEL='5.0' \
  SC_SAMPLE_COUNTS='50,100,200,500,750,1000' \
    python3 run_sample_complexity_benchmark.py \
    2>&1 | tee '${RESULTS_DIR}'/suppB_sc_run.log
  # ── Move sample_complexity outputs → sample-complexity/ ──────────────────
  # The script may share the noise-sweep output dir and write sample_complexity_*.json
  # into per-equation subdirs alongside noise_sweep files. Move them to the
  # dedicated sample-complexity/ dir that the validate step and tables-generator expect.
  mkdir -p '${RESULTS_DIR}/comparison_results/feynman-tests/sample-complexity'
  find '${RESULTS_DIR}/comparison_results/feynman-tests' \
    -name 'sample_complexity_*.json' \
    ! -path '*/sample-complexity/*' \
    -exec mv -v {} '${RESULTS_DIR}/comparison_results/feynman-tests/sample-complexity/' \;
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
run validate "Cross-check all results against paper-reported values" python3 - << 'PYEOF'
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
noiseless_files = sorted(glob.glob(f"{RESULTS}/comparison_results/noise-noiseless/noiseless/protocol_core_noiseless_*.json"))
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
exp2_files = sorted(glob.glob(f"{RESULTS}/comparison_results/feynman-tests/exp2/exp2_results*.json"))
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
noise_sweep_matched = glob.glob(f"{RESULTS}/comparison_results/feynman-tests/noise-sweep/noise_sweep_*.json")
noise_sweep_all     = glob.glob(f"{RESULTS}/comparison_results/feynman-tests/noise-sweep/*.json")
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
print(f"  [{'OK' if ok_exp3b else 'SKIP'}] extrapolation/multi_seed/: {len(exp3b_files)} nguyen JSON(s) "
      f"{'(exp3b not yet run)' if not ok_exp3b else ''}")

# --- FIX STEP-11-12: tables and figures co-located under RESULTS_DIR ---
tbl = glob.glob(f"{RESULTS}/tables/*.tex")
fig = glob.glob(f"{RESULTS}/figures/*.pdf")
ok_tbl = bool(tbl); ok_fig = bool(fig)
checks.append(("tables in RESULTS_DIR/tables/", 1.0 if ok_tbl else 0.0, 1.0, ok_tbl))
checks.append(("figures in RESULTS_DIR/figures/", 1.0 if ok_fig else 0.0, 1.0, ok_fig))
print(f"  [{'OK' if ok_tbl else 'FAIL'}] {RESULTS}/tables/: {len(tbl)} .tex file(s)")
print(f"  [{'OK' if ok_fig else 'FAIL'}] {RESULTS}/figures/: {len(fig)} .pdf file(s)")

# --- Summary ---
total = len(checks); passed = sum(1 for *_, ok in checks if ok)
print(f"\n=== Result: {passed}/{total} checks passed ===")
if passed < total:
    print("FAILED:")
    for label, got, exp, ok in checks:
        if not ok: print(f"  FAIL: {label} (got={got}, expected={exp})")
    sys.exit(1)
else:
    print("All checks passed.")
PYEOF

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
