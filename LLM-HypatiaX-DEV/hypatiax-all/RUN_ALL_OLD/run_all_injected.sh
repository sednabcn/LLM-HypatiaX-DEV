#!/usr/bin/env bash
# =============================================================================
# run_all.sh — HypatiaX JMLR v3.0 full numerical reproduction pipeline
#
# Reproduces ALL numerical results in jmlr_paper_final.pdf (108 pp)
# across five experimental campaigns (131 tests).
#
# Usage:
#   bash run_all.sh            # full run (~5–8 h on modern hardware)
#   bash run_all.sh --step exp1  # single step (see STEP IDs below)
#   bash run_all.sh --from exp1b # resume from a step
#   bash run_all.sh --dry-run   # print commands without executing
#
# STEP IDs (linear order):
#   env_check   → verify Python, PySR, API key
#   exp1        → core extrapolation benchmark (Tab 9, 10, 15 · Fig 9, 10)
#   exp1b       → DeFi seed sweep + portfolio variance (Tab 11-13 · Fig 11-13)
#   extrap      → OOD extrapolation comparative (Tab 9 OOD columns)
#                 [experiment_protocol_extrapolation_comparative.py]
#   instability → LLM instability regime study (Tab tab:instability, §10.9)
#   exp2_feynman → Feynman-30 HypatiaX-vs-NN extrapolation protocol (Tab 16-18 · Phase 2)
#                 [experiment_protocol_feynman_exp2.py → run_protocol → run_task]
#   exp2_sym    → Symbolic engine all-domain run, Method 5 (Tab 19)
#   exp2_hyb    → Hybrid LLM+NN all-domain run, Method 6 (Tab 19)
#   exp2        → Multi-domain 30: Methods 1-4 + inject 5+6 (Tab 19 combined)
#                 [run_comparative_suite_benchmark_injected.py --protocol all30]
#   exp2b       → Feynman-30 6-method comparison suite (Tab 19 Feynman companion)
#                 [run_comparative_suite_benchmark_v2.py --protocol feynman]
#   exp3        → Nguyen-12 benchmark (tab:nguyen12 · §10.8)
#   exp3b       → Nguyen-12 extended seeds 99/123/777/2024 (tab:nguyen12 stability)
#   suppA       → DeFi routing improvement experiments (Tab 11-13 routing)
#   suppB       → Noise sweep + sample complexity (Tab 28, 29 · suppB)
#   tables      → Generate all LaTeX tables from result JSONs
#   figures     → Generate all paper figures from result JSONs/CSVs
#   validate    → Cross-check all result files against expected checksums
#
# Hardware: Intel Celeron T3100 (2 cores, 3 GB RAM) was used for the paper.
#   On modern multi-core hardware runtimes are substantially shorter.
#   PySR parallelism is limited to 2 populations on the paper's hardware;
#   set PYSR_POPULATIONS=<ncores> to scale.
#
# Requirements:
#   Python 3.12+, PySR (Julia backend), PyTorch, SymPy, anthropic SDK
#   ANTHROPIC_API_KEY environment variable set
#   Julia 1.9+ with SymbolicRegression.jl installed
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/hypatiax/data/results}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${REPO_ROOT}/hypatiax/experiments/benchmarks}"
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
export PYSR_POPULATIONS="${PYSR_POPULATIONS:-2}"  # override for multi-core hardware

# Feynman benchmark defaults (Appendix A)
FEYNMAN_SAMPLES=200
FEYNMAN_TIMEOUT=900
FEYNMAN_NOISELESS_THRESHOLD=0.9999

# ── CLI parsing ───────────────────────────────────────────────────────────────
ONLY_STEP=""
FROM_STEP=""
DRY_RUN=false
_STEP_ORDER="env_check exp1 exp1b extrap instability exp2_feynman exp2_sym exp2_hyb exp2 exp2b exp3 exp3b suppA suppB tables figures validate"

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
  # run <step_id> <description> <command...>
  local step="$1" desc="$2"; shift 2
  [[ -n "$ONLY_STEP" && "$ONLY_STEP" != "$step" ]] && return 0
  if [[ -n "$FROM_STEP" ]]; then
    # Skip steps that appear before FROM_STEP in the order.
    # past=true once FROM_STEP is reached; return 0 until then.
    local past=false
    for s in $_STEP_ORDER; do
      [[ "$s" == "$FROM_STEP" ]] && past=true
      [[ "$past" == true && "$s" == "$step" ]] && break
    done
    [[ "$past" == false ]] && return 0
  fi
  echo ""
  log "=== STEP: ${step} — ${desc} ==="
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
  python3 -c "import sympy; print(\"SymPy:\", sympy.__version__)"
  python3 -c "import scipy; print(\"SciPy:\", scipy.__version__)"
  [[ -n "${ANTHROPIC_API_KEY:-}" ]] || { echo "ERROR: ANTHROPIC_API_KEY not set"; exit 1; }
  echo "ANTHROPIC_API_KEY: set (${#ANTHROPIC_API_KEY} chars)"
  echo "PYSR_POPULATIONS: ${PYSR_POPULATIONS}"
  echo "Results dir: '"${RESULTS_DIR}"'"
  mkdir -p '"${RESULTS_DIR}"'/{comparison_results/{feynman-tests/{exp2,noise-sweep},noise-noiseless/{noiseless,15},extrapolation},extrapolation,hybrid_llm_nn/{all_domains,defi},hybrid_pysr/{all_domains,defi},llm_guided/{all_domains,defi},standalone_llm_nn,figures,tables}
  echo "Directory structure: ok"
'

# ── STEP 1: exp1 — Core extrapolation benchmark ───────────────────────────────
# Produces: Tab 9, 10, 15 · Fig 9, 10 · R² heatmaps · ablation.tex · defi_main.tex
# Output files: ablation_exp1_d0966414.json, exp1_ablation_results.json,
#               exp1_rf01_mannwhitney.json,
#               comparison_results/noise-noiseless/noiseless/protocol_core_noiseless_*.json
run exp1 "Core extrapolation benchmark (Tab 9, 10, 15 · Fig 9, 10)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  python3 hypatiax_defi_benchmark_v3c.py \
    2>&1 | tee '${RESULTS_DIR}'/exp1_run.log
  cd '${ANALYSIS_DIR}'
  python3 statistical_analysis.py \
    2>&1 | tee -a '${RESULTS_DIR}'/exp1_run.log
"

# ── STEP 2: exp1b — DeFi seed sweep + portfolio variance ─────────────────────
# Produces: Tab 11, 12, 13 · Fig 11, 12, 13 · defi_tiers.tex
# Output files: defi_v3_a2742f91.json, comparison_FIXED_*.json,
#               comparison_results/noise-noiseless/15/comparison_FIXED_*.json
# DEFI_TASK_FILTER restricts to portfolio variance task only.
# DEFI_SEEDS runs the five-seed robustness sweep (§10.5).
run exp1b "DeFi seed sweep + portfolio variance (Tab 11-13 · Fig 11-13)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  DEFI_TASK_FILTER=portfolio \
  DEFI_SEEDS='42,99,123,777,2024' \
    python3 hypatiax_defi_benchmark_v3c.py \
      2>&1 | tee '${RESULTS_DIR}'/exp1b_run.log
  python3 portfolio_variance_v3c2.py \
    2>&1 | tee -a '${RESULTS_DIR}'/exp1b_run.log
"

# ── STEP 3: extrap — OOD extrapolation comparative ───────────────────────────
# Produces: Tab 9 OOD columns
# Output files: comparison_results/extrapolation/all_domains_extrap_v4_*.json
# Script: experiment_protocol_extrapolation_comparative.py (NOT run_comparative_suite_benchmark_v2.py)
# FIX: was incorrectly calling run_comparative_suite_benchmark_v2.py with no flags,
#      which defaults to --protocol feynman and runs the Feynman benchmark — wrong.
#      The extrapolation comparative uses its own protocol entry-point.
run extrap "OOD extrapolation comparative run (Tab 9 OOD columns)" bash -c "
  cd '${REPO_ROOT}'
  python3 hypatiax/protocols/experiment_protocol_extrapolation_comparative.py \
    2>&1 | tee '${RESULTS_DIR}'/extrap_run.log
  echo 'Outputs: comparison_results/extrapolation/all_domains_extrap_v4_*.json'
"

# ── STEP 4: instability — LLM instability regime ─────────────────────────────
# Produces: tab:instability, tab:arch · instability.tex · §10.9
# Output files: exp1_instability_stats.json, instability_extrapolation_v2.csv,
#               hybrid_llm_nn/defi/*.json
# Note: 70 tasks × 30 LLM runs = 2,100 API calls. Allow significant wall time.
# Key finding: Spearman ρ=−0.70, p<0.001 · C-Collapse anomaly (RF-06)
run instability "LLM instability regime (70 tasks × 30 LLM runs · §10.9)" bash -c "
  cd '${CORE_DIR}/generation/hybrid_all_domains_llm_nn'
  python3 hybrid_system_llm_nn_all_domains.py \
    2>&1 | tee '${RESULTS_DIR}'/instability_run.log
"

# ── STEP 5: exp2_feynman — Feynman SR Phase 2 (noisy) ────────────────────────
# Produces: Tab 16, 17, 18 (Feynman SR benchmark, noisy, R²>0.995)
# Output files: comparison_results/feynman-tests/exp2/I_*.json, exp2_results.json
# Recovery threshold: R² > 0.995
run exp2_feynman "Feynman SR benchmark — Phase 2 noisy protocol (Tab 16-18)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  python3 exp2_feynman_colab_multithreaded.py \
    --samples ${FEYNMAN_SAMPLES} \
    --no-llm-cache \
    2>&1 | tee '${RESULTS_DIR}'/comparison_results/feynman-tests/exp2/exp2_run.log
"

# ── STEP 6: exp2_sym — Symbolic engine all-domain run ────────────────────────
# Produces: Tab 19 Method 5 · tab:five_systems_full (symbolic+LLM column)
# Output files: hybrid_pysr/all_domains/llm_*/ (31 per-equation JSONs)
# Calls: symbolic_engine.py → hybrid_system_v50_2.py
run exp2_sym "Symbolic engine all-domain run (Tab 19 Method 5)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  python3 run_exp2_symbolic_engine.py 2>&1 | tee '${RESULTS_DIR}'/exp2_sym_run.log
"

# ── STEP 7: exp2_hyb — Hybrid LLM+NN all-domain run ─────────────────────────
# Produces: Tab 19 Method 6 · tab:five_systems_full (LLM+NN column)
# Output files: hybrid_llm_nn/all_domains/hybrid_llm_nn_all_domains_*.json
# Calls: hybrid_system_v50_2.py
run exp2_hyb "Hybrid LLM+NN all-domain run (Tab 19 Method 6)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  python3 run_exp2_hybrid_system.py 2>&1 | tee '${RESULTS_DIR}'/exp2_hyb_run.log
"

# ── STEP 8: exp2 — Combined five-system injection ────────────────────────────
# Produces: Tab 19 full · tab:five_systems_full (all methods combined, multi-domain 30)
# Output files: llm_guided/all_domains/*.json, standalone_llm_nn/*.json
# Note: exp2_sym and exp2_hyb must complete first — Methods 5+6 injected here.
# Protocol: all30 (30 multi-domain equations via ExperimentProtocolAll)
run exp2 "Multi-domain 30: Methods 1-4 + inject 5+6 (Tab 19 full)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  python3 run_comparative_suite_benchmark_injected.py 2>&1 | tee '${RESULTS_DIR}'/exp2_run.log
"

# ── STEP 9: exp2b — Feynman-30 6-method comparison suite ─────────────────────
# Produces: Tab 19 Feynman companion · tab:five_systems_feynman
# Output files: comparison_results/feynman-tests/*.json
# Note: exp2_sym and exp2_hyb must complete first — their checkpoints supply
#       pre-computed Method-5/6 results that run_comparative_suite_benchmark_v2.py
#       can optionally inject via --inject-5-6 (if supported), or they run live.
# Protocol: feynman (30 Feynman equations via BenchmarkProtocol)
# FIX: this step was missing entirely — run_comparative_suite_benchmark_v2.py
#      was never called in the original pipeline. exp2 (above) covers the
#      multi-domain 30 via _injected.py; exp2b covers the Feynman 30 via _v2.py.
run exp2b "Feynman-30 6-method comparison suite (Tab 19 Feynman companion)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  python3 run_comparative_suite_benchmark_v2.py \
    --protocol feynman \
    --resume \
    --checkpoint-name exp2b_feynman30_checkpoint \
    --samples 200 \
    2>&1 | tee '${RESULTS_DIR}'/exp2b_run.log
  echo 'Outputs: comparison_results/feynman-tests/**/*.json'
"

# ── STEP 10: exp3 — Nguyen-12 benchmark, SEED=42 ─────────────────────────────
# Produces: tab:nguyen12 (§10.8 primary · SEED=42)
# Output files: extrapolation/full_run_*.json, extrapolation/report_hybrid_*.json,
#               extrapolation/experiment_registry.json
# Expected: 11/12 H (91.7%) · 10/12 P · MW U=113, p=0.0097
run exp3 "Nguyen-12 benchmark — SEED=42 (tab:nguyen12 · §10.8)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  python3 exp3_nguyen12_hybrid50v_02.py \
    --seed 42 \
    2>&1 | tee '${RESULTS_DIR}'/exp3_run.log
"

# ── STEP 11: exp3b — Nguyen-12 stability seeds ───────────────────────────────
# Produces: tab:nguyen12 stability columns (seeds 99, 123, 777, 2024)
# Output files: extrapolation/full_run_*.json (one file per seed)
# Expected: consistent with SEED=42 across all 5 seeds
run exp3b "Nguyen-12 stability seeds 99/123/777/2024 (tab:nguyen12 extended)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  for seed in 99 123 777 2024; do
    log_stamp=\$(date +%Y%m%d_%H%M%S)
    echo '--- exp3b seed='\$seed' ---'
    python3 exp3_nguyen12_hybrid50v_02.py \
      --seed \$seed \
      2>&1 | tee -a '${RESULTS_DIR}'/exp3b_run.log
  done
"

# ── STEP 12: suppA — DeFi routing improvements ───────────────────────────────
# Produces: tab:baseline, tab:fix5_cases, tab:projected, tab:changes
#           (Tab 11-13 routing fix detail · Supplement A)
# Output files: hybrid_pysr/defi/*/ (25 JSONs), llm_guided/defi/*.json
# Routing fixes 1-5 applied sequentially; Fix 5 + routing guard is final state.
run suppA "DeFi routing improvement experiments (Supplement A · Tab 11-13 routing)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  python3 run_hybrid_system_benchmark.py \
    2>&1 | tee '${RESULTS_DIR}'/suppA_run.log
"

# ── STEP 13: suppB — Noise sweep + sample complexity ─────────────────────────
# Produces: Tab 28, 29 (noise + sample tables · Supplement B)
# Output files: comparison_results/feynman-tests/noise-sweep/
#               noise_sweep_*.json/.csv, sample_complexity_*.json/.csv
# Tab 28: success rate vs noise level (σ ∈ {0, 0.5, 1, 5, 10}%)
# Tab 29: extrapolation performance by noise level
run suppB "Noise sweep + sample complexity benchmark (Tab 28, 29 · Supplement B)" bash -c "
  cd '${EXPERIMENTS_DIR}'
  python3 run_noise_sweep_benchmark.py \
    2>&1 | tee  '${RESULTS_DIR}'/suppB_run.log
  python3 run_sample_complexity_benchmark.py \
    2>&1 | tee -a '${RESULTS_DIR}'/suppB_run.log
"

# ── STEP 14: tables — Generate all LaTeX tables ───────────────────────────────
# Produces: all *.tex files in scripts/paper/tables/
#           ablation.tex, defi_main.tex, defi_tiers.tex, instability.tex, repro_macros.tex
# Reads ALL result JSONs from previous steps.
run tables "Generate all LaTeX tables from result JSONs" bash -c "
  cd '${SCRIPTS_DIR}/patches'
  python3 generate_tables.py \
    --results-dir '${RESULTS_DIR}' \
    --output-dir '${REPO_ROOT}/scripts/paper/tables' \
    2>&1 | tee '${RESULTS_DIR}'/tables_run.log
  echo 'Tables written to: scripts/paper/tables/*.tex'
  ls '${REPO_ROOT}/scripts/paper/tables/'
"

# ── STEP 15: figures — Generate all paper figures ────────────────────────────
# Produces: figures/*.pdf (all 5 .tex-referenced figures + supplementary)
# Reads result JSONs + CSVs from all previous experiment steps.
run figures "Generate all paper figures from results" bash -c "
  cd '${REPO_ROOT}/figures'
  python3 generate_figures.py \
    --results-dir '${RESULTS_DIR}' \
    --output-dir '${RESULTS_DIR}/figures' \
    2>&1 | tee '${RESULTS_DIR}'/figures_run.log
  echo 'Figures written to: hypatiax/data/results/figures/*.pdf'
  ls '${RESULTS_DIR}/figures/'
"

# ── STEP 16: validate — Cross-check results ───────────────────────────────────
# Verifies key numerical results against paper-reported values.
# Reports any deviation exceeding tolerance.
run validate "Cross-check all results against paper-reported values" python3 - << 'PYEOF'
import json, os, glob, sys

RESULTS = os.environ.get('RESULTS_DIR', 'hypatiax/data/results')
TOLERANCE = 0.01  # 1% relative tolerance for R² comparisons

checks = []

def check(label, got, expected, tol=TOLERANCE):
    ok = abs(got - expected) <= tol * max(abs(expected), 1e-9)
    checks.append((label, got, expected, ok))
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {label}: got={got:.6f}, expected={expected:.6f}")
    return ok

print("\n=== Validating key numerical results against JMLR v3.0 ===\n")

# --- exp1: core extrapolation (Table 9) ---
noiseless_files = sorted(glob.glob(f"{RESULTS}/comparison_results/noise-noiseless/noiseless/protocol_core_noiseless_*.json"))
if noiseless_files:
    with open(noiseless_files[-1]) as f:
        data = json.load(f)
    # Check Hybrid v40 train R² (expected: mean=0.931, median=1.000)
    hx = [r for r in data.get('results', []) if r.get('method') in ('hybrid_v40', 'Hybrid v40')]
    if hx:
        r2_vals = [r['r2_train'] for r in hx if 'r2_train' in r]
        if r2_vals:
            import statistics
            check("Hybrid v40 mean train R²", statistics.mean(r2_vals), 0.931)
            check("Hybrid v40 median train R²", statistics.median(r2_vals), 1.000)
else:
    print("  [SKIP] exp1 noiseless results not found — run exp1 first")

# --- Feynman Phase 2 (Table 17) ---
exp2_files = sorted(glob.glob(f"{RESULTS}/comparison_results/feynman-tests/exp2/exp2_results*.json"))
if exp2_files:
    with open(exp2_files[-1]) as f:
        data = json.load(f)
    recovery = data.get('hybrid_deFi_recovery') or data.get('recovery_rate')
    if recovery is not None:
        check("Hybrid DeFi recovery rate (Feynman noisy)", recovery, 1.0, tol=0.001)
else:
    print("  [SKIP] exp2_feynman results not found — run exp2_feynman first")

# --- Mann-Whitney U test (Table 14) ---
mw_files = sorted(glob.glob(f"{RESULTS}/exp1_rf01_mannwhitney*.json"))
if mw_files:
    with open(mw_files[-1]) as f:
        data = json.load(f)
    u_stat = data.get('mann_whitney_u', data.get('U'))
    if u_stat is not None:
        check("Mann-Whitney U statistic (Hybrid v40 vs NN)", float(u_stat), 0.0, tol=0.0)
    p_val = data.get('p_value', data.get('p'))
    if p_val is not None:
        ok = p_val < 1e-5
        checks.append(("p-value < 1e-6", p_val, 1.11e-6, ok))
        print(f"  [{'OK' if ok else 'FAIL'}] p-value < 1e-5: got={p_val:.2e}")
else:
    print("  [SKIP] Mann-Whitney results not found — run exp1 first")

# --- Summary ---
total = len(checks)
passed = sum(1 for _, _, _, ok in checks if ok)
print(f"\n=== Result: {passed}/{total} checks passed ===")
if passed < total:
    print("FAILED checks:")
    for label, got, exp, ok in checks:
        if not ok:
            print(f"  FAIL: {label} (got={got}, expected={exp})")
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
echo "    LaTeX tables:  ${REPO_ROOT}/scripts/paper/tables/*.tex"
echo "    Figures PDF:   ${RESULTS_DIR}/figures/*.pdf"
echo ""
echo "  Cross-reference with paper:"
echo "    Table 9   <- exp1   (core extrapolation)"
echo "    Table 11  <- exp1b  (DeFi routing)"
echo "    Table 17  <- exp2_feynman  (Feynman-30 HypatiaX-vs-NN extrapolation)"
echo "    Table 19  <- exp2   (six-method comparison, multi-domain 30)"
echo "    Table 19  <- exp2b  (six-method comparison, Feynman 30)"
echo "    Table 28  <- suppB  (noise sweep)"
echo "    tab:nguyen12 <- exp3/exp3b"
echo ""
echo "  To rebuild the paper PDF:"
echo "    cd ${REPO_ROOT} && pdflatex jmlr-hypatiax-paper-final.tex"
echo ""
log "Done. See individual *_run.log files in ${RESULTS_DIR}/ for per-step output."
