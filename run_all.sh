#!/usr/bin/env bash
# =============================================================================
# HypatiaX JMLR — Full Reproducibility Pipeline
# Version: v5.0 (Apr 2026)
#
# Usage:
#   ./run_all.sh                    # full pipeline
#   ./run_all.sh --skip-slow        # skip Feynman, noise sweep, instability
#   ./run_all.sh --only exp3        # run one step by id
#   ./run_all.sh --skip-paper       # skip pdflatex compile
#   ./run_all.sh --verify-only      # re-check existing results without re-running
#   ./run_all.sh --continue-on-fail # log failures but keep going
#
# Prerequisites:
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   pip install -r requirements.txt
#
# Step IDs (use with --only):
#   Setup   : deps  patches-gen  patches-apply  patches-verify  validate
#   Phase 1 : exp1  exp1b  exp2  exp3  exp3b
#   Phase 2 : suppB  suppA  instability  extrap
#   Phase 3 : provenance  verify  hashlock
#   Phase 4 : figures  tables
# =============================================================================
set -euo pipefail

# ── CLI flags ──────────────────────────────────────────────────────────────────
SKIP_SLOW=0
SKIP_PAPER=0
VERIFY_ONLY=0
CONTINUE_ON_FAIL=0
ONLY=""

for arg in "$@"; do
    case $arg in
        --skip-slow)        SKIP_SLOW=1 ;;
        --skip-paper)       SKIP_PAPER=1 ;;
        --verify-only)      VERIFY_ONLY=1 ;;
        --continue-on-fail) CONTINUE_ON_FAIL=1 ;;
        --only=*)           ONLY="${arg#--only=}" ;;
        --only)             shift; ONLY="$1" ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO="$ROOT/protocols"
SCRIPTS="$ROOT/scripts/patches"
RESULTS="$ROOT/hypatiax/data/results"
LOGDIR="$ROOT/logs"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export REPRO_ROOT="$ROOT"
export PYTHON_JULIACALL_HANDLE_SIGNALS="yes"

# ── Reproducibility seeds & model ─────────────────────────────────────────────
export NN_SEED="${NN_SEED:-42}"
export PYSR_SEED="${PYSR_SEED:-42}"
export LLM_MODEL="${LLM_MODEL:-claude-sonnet-4-6}"
export LLM_RETRIES="${LLM_RETRIES:-3}"
export LLM_K_RUNS="${LLM_K_RUNS:-1}"   # set to 30 for full §10.9 sweep
export N_TASKS_DEFI=74
export N_TASKS_INSTABILITY=70           # FIX-T1: must be 70, NOT 71
export PCA_TRAIN_FRAC=0.40
export NN_TIME_LIMIT=120
export ENGINE_NAME=hybrid_system_v50_2  # FIX-C2: never v40

mkdir -p "$RESULTS" "$LOGDIR"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; NC='\033[0m'

phase() {
    echo ""
    echo -e "${YLW}══════════════════════════════════════════════════${NC}"
    echo -e "${YLW}  Phase $1${NC}"
    echo -e "${YLW}══════════════════════════════════════════════════${NC}"
}

ok()   { echo -e "${GRN}✓ $1${NC}"; }
warn() { echo -e "${YLW}⚠  $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; exit 1; }

# run_step ID LABEL CMD...
# Honours --only, --skip-slow (if 4th arg is "slow"), --continue-on-fail
STEP_RESULTS=()
run_step() {
    local id="$1" label="$2" slow="${3:-}" ; shift 3
    local cmd=("$@")
    local logfile="$LOGDIR/${id}.log"

    # --only filter
    if [[ -n "$ONLY" && "$id" != "$ONLY" ]]; then
        echo "  ── skip [$id]  (--only $ONLY)"
        STEP_RESULTS+=("skip:$id:$label")
        return 0
    fi

    # --skip-slow filter
    if [[ "$slow" == "slow" && "$SKIP_SLOW" -eq 1 ]]; then
        warn "skip [$id]  (--skip-slow)"
        STEP_RESULTS+=("skip:$id:$label")
        return 0
    fi

    echo ""
    echo "  ┌─── [$id] $label"
    echo "  │    $(date '+%H:%M:%S')"
    echo "  │    cmd: ${cmd[*]}"

    local t0=$SECONDS
    if "${cmd[@]}" 2>&1 | tee "$logfile"; then
        local elapsed=$(( SECONDS - t0 ))
        ok "[$id] done  (${elapsed}s)  → logs/${id}.log"
        STEP_RESULTS+=("pass:$id:$label")
    else
        local elapsed=$(( SECONDS - t0 ))
        echo -e "  └─── ${RED}✗ FAILED${NC}  (${elapsed}s)  → logs/${id}.log"
        STEP_RESULTS+=("fail:$id:$label")
        if [[ "$CONTINUE_ON_FAIL" -eq 0 ]]; then
            fail "Pipeline aborted at [$id]. Re-run with --continue-on-fail to keep going."
        fi
    fi
}

# ── Preflight ─────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  HypatiaX · Reproducibility Pipeline v5.0           ║"
echo "║  JMLR Apr 2026                                       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  ROOT      : $ROOT"
echo "  NN_SEED   : $NN_SEED   PYSR_SEED: $PYSR_SEED"
echo "  LLM_MODEL : $LLM_MODEL"
echo "  LLM_K_RUNS: $LLM_K_RUNS  (set to 30 for full §10.9)"
echo "  ENGINE    : $ENGINE_NAME"
echo ""

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    fail "ANTHROPIC_API_KEY is not set.\n  export ANTHROPIC_API_KEY='sk-ant-...'"
fi
ok "API key set (${#ANTHROPIC_API_KEY} chars)"

# Scan for exposed keys in source
echo "  Scanning for exposed API keys..."
if grep -r "sk-ant-api" "$ROOT" --include="*.py" --include="*.ipynb" -l 2>/dev/null \
        | grep -v ".git"; then
    fail "Exposed API key found — revoke at console.anthropic.com, then remove from source"
fi
ok "No exposed API keys"

# Scan for stale v40 engine references (FIX-C2)
echo "  Checking for stale v40 engine imports..."
STALE=$(grep -r "hybrid_system_v40[^_]" "$ROOT" --include="*.py" \
    --exclude="hybrid_system_v40.py" --exclude="hybrid_system_v40fix.py" \
    -l 2>/dev/null || true)
if [[ -n "$STALE" ]]; then
    warn "Stale v40 imports found — auto-patching:"
    echo "$STALE"
    echo "$STALE" | xargs sed -i 's/hybrid_system_v40\([^_]\)/hybrid_system_v50_2\1/g'
    ok "v40 → v50_2 patched"
fi

# Verify hypatiax/protocols/ input-data modules are present
echo "  Checking hypatiax/protocols/ input-data modules..."
HYPATIAX_PROTO="$ROOT/hypatiax/protocols"
REQUIRED_HYPATIAX_PROTOCOLS=(
    "experiment_protocol_defi.py"
    "experiment_protocol_defi_20.py"
    "experiment_protocol_nguyen12.py"
    "experiment_protocol_all_18_a.py"
    "experiment_protocol_all_20.py"
    "experiment_protocol_all_30.py"
    "experiment_protocol_benchmark.py"
    "experiment_protocol_benchmark_v2.py"
    "experiment_protocol_comparative.py"
)
MISSING_PROTO=0
for f in "${REQUIRED_HYPATIAX_PROTOCOLS[@]}"; do
    if [[ ! -f "$HYPATIAX_PROTO/$f" ]]; then
        warn "Missing: hypatiax/protocols/$f"
        MISSING_PROTO=$(( MISSING_PROTO + 1 ))
    fi
done
if [[ "$MISSING_PROTO" -eq 0 ]]; then
    ok "hypatiax/protocols/ — all ${#REQUIRED_HYPATIAX_PROTOCOLS[@]} input-data modules present"
else
    fail "$MISSING_PROTO input-data module(s) missing from hypatiax/protocols/. Copy them from your source and re-run."
fi

# ── Phase 0-B: Code quality audit (NB-06) ─────────────────────────────────────
# Runs before experiments to catch duplicate case names, stale v40 imports,
# split-protocol mismatches, and exposed API keys in .py files.
# NB-06 is diagnostic-only; apply_patches.py (patches-apply step) does the actual fixes.
echo ""
echo "  Running NB-06 code quality pre-audit..."
if command -v jupyter &>/dev/null; then
    jupyter nbconvert --to notebook --execute --inplace         --ExecutePreprocessor.timeout=120         "$ROOT/notebooks/NB-06_Code_Quality_Pipeline_Integrity.ipynb"         2>&1 | tail -3
    ok "NB-06 code quality audit complete  (see notebooks/NB-06_*.ipynb)"
else
    warn "jupyter not found — skipping NB-06 pre-audit (install with: pip install notebook)"
fi

# Note: Feynman benchmark uses 80/20 random split; DeFi uses PCA 40/60 split.
# See §10.7 disclosure and run_comparative_suite_benchmark_v2.py docstring.
# These are NOT comparable — do not aggregate across benchmarks without normalisation.

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
    phase "Verify-only mode"
    python3 "$SCRIPTS/verify_results.py" --report --json
    python3 "$ROOT/reproducibility/hash_lock.py" --check
    exit 0
fi

# ═════════════════════════════════════════════════════════════════════════════
phase "0 · Setup"
# ═════════════════════════════════════════════════════════════════════════════

run_step "deps"          "Install dependencies"            "" \
    pip install -q -r requirements.txt

run_step "patches-gen"   "Generate patches"                "" \
    python3 "$SCRIPTS/generate_patches.py"

run_step "patches-apply" "Apply patches (FIX-C1…FIX-5b)"  "" \
    python3 "$SCRIPTS/apply_patches.py"

run_step "patches-verify" "Verify patches (import scan)"         "" \
    python3 "$SCRIPTS/apply_patches.py" --verify

run_step "validate"      "Validate patched source"         "" \
    python3 "$SCRIPTS/validate_code.py"

# ═════════════════════════════════════════════════════════════════════════════
phase "1 · Core experiments"
# ═════════════════════════════════════════════════════════════════════════════

# Exp 1 — DeFi 74-task benchmark v3.0  (§10.2–10.4, §10.6)
# Expected: 89.2% R²>0.99  ·  0 catastrophic  ·  1.73× speedup
# Wall time: 2–4 h
run_step "exp1" \
    "Exp 1 · DeFi 74-task benchmark v3.0 (§10.2–10.4, §10.6)" "" \
    python3 "$PROTO/experiment_protocol_ablation_exp1.py"

# Exp 1b — Portfolio Variance seed sweep  (§10.5)
# Expected: P(HypatiaX > PureLLM) ≈ 0.76 across seeds 42/99/123/777/2024
# Wall time: 20–40 min
run_step "exp1b" \
    "Exp 1b · Portfolio Variance seed sweep (§10.5)" "" \
    python3 "$PROTO/experiment_protocol_defi_v3.py" \
        --task portfolio_variance \
        --seeds 42 99 123 777 2024

# Exp 2 — Feynman 30-equation extrapolation  (§10.7)
# Expected: 9/30 (30%)  ·  40/60 PCA split  ·  Kaggle 4-vCPU is primary run
# Wall time: 4–8 h
run_step "exp2" \
    "Exp 2 · Feynman 30-equation extrapolation (§10.7)" "slow" \
    python3 "$PROTO/experiment_protocol_feynman_exp2.py"

# Exp 3 — Nguyen-12 SR suite, SEED=42  (§10.8 primary)
# Expected: 11/12 H (91.7%)  ·  10/12 P (83.3%)  ·  MW P>NN U=113, p=0.0097
# Wall time: 30–90 min
run_step "exp3" \
    "Exp 3 · Nguyen-12 SEED=42 (§10.8 primary)" "" \
    python3 "$PROTO/experiment_protocol_nguyen12_exp3.py"

# Exp 3b — Nguyen-12 stability check, SEED=123  (§10.8)
# Expected: consistent with SEED=42
# Wall time: 30–90 min
run_step "exp3b" \
    "Exp 3b · Nguyen-12 SEED=123 (§10.8 stability check)" "" \
    python3 "$PROTO/experiment_protocol_nguyen12_exp3.py" --seed 123

# ═════════════════════════════════════════════════════════════════════════════
phase "2 · Supplementary benchmarks"
# ═════════════════════════════════════════════════════════════════════════════

# Supp B — Noise & sample-complexity sweep
# Expected: EHD 100% at all σ levels  ·  plateau ≈ N=500
# Wall time: 6–12 h
run_step "suppB" \
    "Supp B · Noise & sample-complexity sweep" "slow" \
    python3 "$PROTO/experiment_protocol_noise_sweep.py"

# Supp A — Hybrid routing improvements (Fix 1–5b)
# Expected: +6pp Fix1, +5pp Fix2, +1pp Fix3
# Wall time: 30–60 min
run_step "suppA" \
    "Supp A · Hybrid routing improvements (Fix 1–5b)" "" \
    python3 "$PROTO/experiment_protocol_hybrid_routing.py"

# §10.9 — Stability under stochastic inference (K=30 runs × 70 tasks)
# Expected: Spearman ρ=−0.70, p<0.001  ·  C-Collapse Portfolio ES anomaly (RF-06)
# Wall time: 3–6 h  (LLM_K_RUNS=30 set below)
LLM_K_RUNS=30 \
run_step "instability" \
    "§10.9 · Stability under stochastic inference (K=30)" "slow" \
    python3 "$PROTO/experiment_protocol_instability_rf02_04.py"

# §10.8 — Extrapolation comparative (cross-method R² across OOD regimes)
# Wall time: 20–40 min
run_step "extrap" \
    "§10.8 · Extrapolation comparative" "" \
    python3 "$PROTO/experiment_protocol_extrapolation_comparative.py"

# ═════════════════════════════════════════════════════════════════════════════
phase "3 · Audit & verification"
# ═════════════════════════════════════════════════════════════════════════════

# §11 — Provenance audit: (a) protocol orchestration, (b) discover_provenance, (c) import scan
# discover_provenance.py links every result file to its source/family/patch chain
# scan_internal_imports.py maps the import DAG for repro verification
run_step "provenance" \
    "§11 · Provenance audit (run after all experiments)" "" \
    python3 "$PROTO/experiment_protocol_provenance_audit.py"

if [[ -f "$ROOT/provenance_map.json" ]]; then
    run_step "discover-provenance" \
        "§11 · discover_provenance.py — link result files to families" "" \
        python3 "$ROOT/discover_provenance.py" \
            --root "$ROOT" \
            --map  "$ROOT/provenance_map.json" \
            --out  "$LOGDIR/provenance_audit"
else
    mkdir -p "$LOGDIR/provenance_audit"
    warn "provenance_map.json absent — skipping discover_provenance (public repo)"
    STEP_RESULTS+=("skip:discover-provenance:§11 · discover_provenance.py — link result files to families")
fi

run_step "scan-imports" \
    "§11 · scan_internal_imports.py — internal import DAG" "" \
    python3 "$ROOT/scan_internal_imports.py" \
        --root "$ROOT" \
        --out  "$LOGDIR/repro_output"

run_step "verify" \
    "Verify results against paper targets" "" \
    python3 "$SCRIPTS/verify_results.py" --report --json

run_step "hashlock" \
    "Hash lock check" "" \
    python3 "$ROOT/reproducibility/hash_lock.py" --check

# ═════════════════════════════════════════════════════════════════════════════
phase "4 · Figures & tables"
# ═════════════════════════════════════════════════════════════════════════════

run_step "figures" \
    "Generate all figures" "" \
    python3 "$ROOT/figures/generate_figures.py"

run_step "tables" \
    "Generate all tables" "" \
    python3 "$SCRIPTS/generate_tables.py"

# ═════════════════════════════════════════════════════════════════════════════
phase "4-B · Paper audit notebooks (NB-01 through NB-05)"
# ═════════════════════════════════════════════════════════════════════════════
# Scans jmlr-hypatiax-paper-final.tex and reports:
#   NB-01: FIX-B1/B2/B3   — missing/duplicate bibliography entries
#   NB-02: FIX-XR1–XR4    — undefined refs, duplicate labels, Supp A cross-refs
#   NB-03: (diagnostic)    — section structure & equation inventory
#   NB-04: FIX-N1/N2/N3   — numerical consistency (70 vs 71, terminology)
#   NB-05: FIX-F1–F4       — missing figure files and fbox placeholders
# Run before pdflatex so issues are visible before the compile attempt.

NOTEBOOKS_DIR="$ROOT/notebooks"
TEX_PAPER="$ROOT/paper/jmlr-hypatiax-paper-final.tex"

if command -v jupyter &>/dev/null && [[ -f "$TEX_PAPER" ]]; then
    cp "$TEX_PAPER" "$NOTEBOOKS_DIR/jmlr-hypatiax-paper-final.tex"
    for NB_ID in NB-01 NB-02 NB-03 NB-04 NB-05; do
        NB_FILE=$(ls "$NOTEBOOKS_DIR/${NB_ID}_"*.ipynb 2>/dev/null | head -1)
        if [[ -n "$NB_FILE" ]]; then
            run_step "audit-${NB_ID}" \
                "Paper audit · $(basename "$NB_FILE" .ipynb)" "" \
                jupyter nbconvert --to notebook --execute --inplace \
                    --ExecutePreprocessor.timeout=300 \
                    "$NB_FILE"
        else
            warn "Not found: $NOTEBOOKS_DIR/${NB_ID}_*.ipynb — skipping"
        fi
    done
    cp "$NOTEBOOKS_DIR"/NB-0*.ipynb "$LOGDIR/" 2>/dev/null || true
    ok "Paper audit notebooks executed — see logs/NB-0*.ipynb"
elif ! command -v jupyter &>/dev/null; then
    warn "jupyter not found — skipping paper audit (pip install notebook)"
else
    warn "$TEX_PAPER not found — skipping paper audit notebooks"
fi

# ═════════════════════════════════════════════════════════════════════════════
phase "5 · Paper compile (optional)"
# ═════════════════════════════════════════════════════════════════════════════

if [[ "$SKIP_PAPER" -eq 0 ]] && command -v pdflatex &>/dev/null; then
    (
        cd "$ROOT/paper"
        pdflatex -interaction=nonstopmode jmlr-hypatiax-full-paper.tex | tail -5
        bibtex   jmlr-hypatiax-full-paper
        pdflatex -interaction=nonstopmode jmlr-hypatiax-full-paper.tex | tail -5
        pdflatex -interaction=nonstopmode jmlr-hypatiax-full-paper.tex | tail -5
        pdflatex -interaction=nonstopmode supp_routing_improvements.tex | tail -5
        pdflatex -interaction=nonstopmode supp_benchmark_report.tex     | tail -5
    )
    ok "PDFs compiled → paper/"
else
    warn "Skipping PDF compile (pdflatex not found or --skip-paper set)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║                  PIPELINE SUMMARY                   ║"
echo "╚══════════════════════════════════════════════════════╝"

PASSED=0; FAILED=0; SKIPPED=0
for entry in "${STEP_RESULTS[@]}"; do
    status="${entry%%:*}"
    rest="${entry#*:}"
    id="${rest%%:*}"
    label="${rest#*:}"
    case "$status" in
        pass)  echo -e "  ${GRN}✓${NC} [$id] $label"; (( PASSED++  )) || true ;;
        fail)  echo -e "  ${RED}✗${NC} [$id] $label  → logs/${id}.log"; (( FAILED++ )) || true ;;
        skip)  echo "  ─ [$id] $label  (skipped)"; (( SKIPPED++ )) || true ;;
    esac
done

echo ""
echo "  ✓ passed : $PASSED"
echo "  ✗ failed : $FAILED"
echo "  ─ skipped: $SKIPPED"
echo ""

# Provenance map coverage check (always run, even when some steps failed)
if [[ -f "$ROOT/provenance_map.json" && -f "$LOGDIR/provenance_audit/provenance_audit_summary.txt" ]]; then
    echo ""
    echo "  Provenance map summary:"
    grep -E "AUTHORITATIVE|ORPHAN|Total" "$LOGDIR/provenance_audit/provenance_audit_summary.txt" | head -8 | sed "s/^/    /"
fi

if [[ "$FAILED" -gt 0 ]]; then
    echo -e "  ${RED}Some steps FAILED. Check logs/ for details.${NC}"
    echo "  Re-run a single step:  ./run_all.sh --only <id>"
    exit 1
else
    echo -e "  ${GRN}All steps passed. Results ready for paper verification.${NC}"
    echo "  Results    : $RESULTS/"
    echo "  Provenance : logs/provenance_audit/"
    echo "  Import DAG : logs/repro_output/import_graph.dot"
    echo "  Logs       : logs/"
    echo "  Re-check   : ./run_all.sh --verify-only"
fi
