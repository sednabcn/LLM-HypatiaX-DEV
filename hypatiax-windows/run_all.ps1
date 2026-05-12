# =============================================================================
# run_all.ps1 — HypatiaX JMLR v3.0 full numerical reproduction pipeline
#               Windows / PowerShell port of run_all.sh
#
# STEP IDs (linear order):
#   env_check          → verify Python, PySR, API key
#   exp1               → core extrapolation benchmark (Tab 9, 10, 15 · Fig 9, 10)
#   exp1b              → DeFi seed sweep + portfolio variance (Tab 11-13 · Fig 11-13)
#   extrap             → OOD extrapolation comparative (Tab 9 OOD columns)
#   hybrid_all_domains → hybrid LLM+NN all-domains run (§10.9 hybrid table)
#   instability        → Instability Index analysis + 12 figures (§10.9 Regime A/B/C)
#   exp2_feynman       → Feynman SR noisy benchmark (Tab 16-18 · Phase 2)
#   exp2               → Combined five-system comparison (Tab 19 full)
#   exp3               → Nguyen-12 benchmark (tab:nguyen12 · §10.8)
#   exp3b              → Nguyen-12 extended seeds 99/123/777/2024
#   suppA              → DeFi routing improvement experiments (Tab 11-13 routing)
#   suppB              → Noise sweep (Tab 28, 29 · suppB)
#   suppB_sc           → Sample-complexity sweep (Tab 29 · suppB)
#   tables             → Generate all LaTeX tables → $RESULTS_DIR\tables\
#   figures            → Generate all paper figures → $RESULTS_DIR\figures\
#   validate           → Cross-check all result files against expected checksums
#
# Usage:
#   .\run_all.ps1                          # run all steps
#   .\run_all.ps1 -Step exp1              # run one step
#   .\run_all.ps1 -From exp2_feynman      # run from a step onwards
#   .\run_all.ps1 -DryRun                 # preview commands without executing
#
# Requirements:
#   • Python 3.12 on PATH (python or python3)
#   • Git on PATH (used to locate repo root)
#   • $env:ANTHROPIC_API_KEY set before running
# =============================================================================

param(
    [string]$Step     = "",        # run only this step
    [string]$From     = "",        # run from this step onwards
    [switch]$DryRun   = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Python executable ─────────────────────────────────────────────────────────
$PYTHON = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } `
          elseif (Get-Command python3 -ErrorAction SilentlyContinue) { "python3" } `
          else { throw "Python not found on PATH" }

# ── Paths ─────────────────────────────────────────────────────────────────────
$REPO_ROOT = if ($env:REPO_ROOT) { $env:REPO_ROOT } `
             else {
               try { (git rev-parse --show-toplevel 2>$null).Trim() }
               catch { $PSScriptRoot }
             }

$RESULTS_DIR     = if ($env:RESULTS_DIR)     { $env:RESULTS_DIR }     else { Join-Path $REPO_ROOT "hypatiax\data\results" }
$EXPERIMENTS_DIR = if ($env:EXPERIMENTS_DIR) { $env:EXPERIMENTS_DIR } else { Join-Path $REPO_ROOT "hypatiax\experiments\benchmarks" }
$CORE_DIR        = if ($env:CORE_DIR)        { $env:CORE_DIR }        else { Join-Path $REPO_ROOT "hypatiax\core" }
$ANALYSIS_DIR    = if ($env:ANALYSIS_DIR)    { $env:ANALYSIS_DIR }    else { Join-Path $REPO_ROOT "hypatiax\analysis" }
$SCRIPTS_DIR     = if ($env:SCRIPTS_DIR)     { $env:SCRIPTS_DIR }     else { Join-Path $REPO_ROOT "scripts" }

# ── PySR hyperparameters (Table 23) ──────────────────────────────────────────
$env:PYSR_GENERATIONS    = "10000"
$env:PYSR_POPULATION     = "100"
$env:PYSR_TOURNAMENT_SIZE = "3"
$env:PYSR_CROSSOVER      = "0.9"
$env:PYSR_MUTATION       = "0.1"
$env:PYSR_PARETO_PRESSURE = "0.001"
$env:PYSR_SEED           = "42"
if (-not $env:PYSR_POPULATIONS) { $env:PYSR_POPULATIONS = "2" }

# ── Feynman benchmark defaults (Appendix A) ───────────────────────────────────
$FEYNMAN_SAMPLES              = "200"
$FEYNMAN_TIMEOUT              = "1100"    # paper value 1100 s
$FEYNMAN_NOISELESS_THRESHOLD  = "0.9999"

# ── Domain list for hybrid_all_domains validation ────────────────────────────
$HYBRID_ALL_DOMAINS_EXPECTED = "biology,chemistry,electromagnetism,finance,mechanics,optics,other,quantum,statistics,thermodynamics"

# ── Step order ────────────────────────────────────────────────────────────────
$STEP_ORDER = @(
  "env_check","exp1","exp1b","extrap","hybrid_all_domains","instability",
  "exp2_feynman","exp2","exp3","exp3b","suppA","suppB","suppB_sc",
  "tables","figures","validate"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
function Log  { Write-Host "[run_all] $args" -ForegroundColor Green }
function Warn { Write-Host "[WARN]    $args" -ForegroundColor Yellow }
function Die  { Write-Error "[ERROR]   $args"; exit 1 }

function Invoke-Step {
    param([string]$StepName, [string]$Desc, [scriptblock]$Body)

    # --step filter
    if ($Step -and $Step -ne $StepName) { return }

    # --from filter
    if ($From) {
        $fromIdx = $STEP_ORDER.IndexOf($From)
        $thisIdx = $STEP_ORDER.IndexOf($StepName)
        if ($fromIdx -lt 0) { Die "Unknown --from step: $From" }
        if ($thisIdx -lt $fromIdx) { return }
    }

    Write-Host ""
    Log "=== STEP: $StepName — $Desc ==="

    if ($DryRun) {
        Write-Host "    [dry-run] $StepName" -ForegroundColor Cyan
    } else {
        & $Body
        Log "--- DONE: $StepName ---"
    }
}

function Run-Python {
    param([string[]]$Args)
    & $PYTHON @Args
    if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE" }
}

function Move-Outputs {
    # Move files matching a wildcard from $src_dir to $dst_dir (one level deep).
    param([string]$Pattern, [string]$SrcDir, [string]$DstDir)
    New-Item -ItemType Directory -Force -Path $DstDir | Out-Null
    Get-ChildItem -Path $SrcDir -MaxDepth 1 -Filter $Pattern -File -ErrorAction SilentlyContinue |
        ForEach-Object { Move-Item -Path $_.FullName -Destination $DstDir -Force -Verbose }
}

# ── Ensure result directories exist ───────────────────────────────────────────
@(
  $RESULTS_DIR,
  "$RESULTS_DIR\comparison_results\feynman-tests\exp2",
  "$RESULTS_DIR\comparison_results\feynman-tests\noise-sweep",
  "$RESULTS_DIR\comparison_results\feynman-tests\sample-complexity",
  "$RESULTS_DIR\comparison_results\noise-noiseless\noiseless",
  "$RESULTS_DIR\comparison_results\noise-noiseless\15",
  "$RESULTS_DIR\comparison_results\extrapolation",
  "$RESULTS_DIR\extrapolation",
  "$RESULTS_DIR\hybrid_llm_nn\all_domains",
  "$RESULTS_DIR\hybrid_llm_nn\defi",
  "$RESULTS_DIR\hybrid_pysr\all_domains",
  "$RESULTS_DIR\hybrid_pysr\defi",
  "$RESULTS_DIR\llm_guided\all_domains",
  "$RESULTS_DIR\llm_guided\defi",
  "$RESULTS_DIR\standalone_llm_nn",
  "$RESULTS_DIR\figures",
  "$RESULTS_DIR\tables"
) | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

# =============================================================================
# STEP 0: env_check
# =============================================================================
Invoke-Step "env_check" "Verify environment (Python, Julia/PySR, API key, directories)" {
    Run-Python -Args @("-c", "import pysr; print('PySR:', pysr.__version__)")
    Run-Python -Args @("-c", "import torch; print('PyTorch:', torch.__version__)")
    Run-Python -Args @("-c", "import anthropic; print('anthropic SDK: ok')")
    Run-Python -Args @("-c", "import sympy; print('SymPy:', sympy.__version__)")
    Run-Python -Args @("-c", "import scipy; print('SciPy:', scipy.__version__)")
    if (-not $env:ANTHROPIC_API_KEY) { Die "ANTHROPIC_API_KEY not set" }
    Write-Host "ANTHROPIC_API_KEY: set ($($env:ANTHROPIC_API_KEY.Length) chars)"
    Write-Host "PYSR_POPULATIONS: $($env:PYSR_POPULATIONS)"
    Write-Host "Results dir: $RESULTS_DIR"
    Write-Host "Directory structure: ok"
}

# =============================================================================
# STEP 1: exp1
# =============================================================================
Invoke-Step "exp1" "Core extrapolation benchmark (Tab 9, 10, 15 · Fig 9, 10)" {
    Push-Location $EXPERIMENTS_DIR
    try {
        Run-Python @("hypatiax_defi_benchmark_v3c.py") 2>&1 |
            Tee-Object -FilePath "$RESULTS_DIR\exp1_run.log"
        Push-Location $ANALYSIS_DIR
        try {
            Run-Python @("statistical_analysis.py") 2>&1 |
                Add-Content -Path "$RESULTS_DIR\exp1_run.log"
        } finally { Pop-Location }
        # Move exp1 outputs → RESULTS_DIR
        Move-Outputs "hypatiax_defi_benchmark_v3*results*.json" $EXPERIMENTS_DIR $RESULTS_DIR
        Move-Outputs "ablation_*.json"                          $EXPERIMENTS_DIR $RESULTS_DIR
        Move-Outputs "exp1_rf01_mannwhitney*.json"              $EXPERIMENTS_DIR $RESULTS_DIR
    } finally { Pop-Location }
}

# =============================================================================
# STEP 2: exp1b
# =============================================================================
Invoke-Step "exp1b" "DeFi seed sweep + portfolio variance (Tab 11-13 · Fig 11-13)" {
    Push-Location $EXPERIMENTS_DIR
    try {
        $env:DEFI_TASK_FILTER = "portfolio"
        $env:DEFI_SEEDS       = "42,99,123,777,2024"
        Run-Python @("hypatiax_defi_benchmark_v3c.py") 2>&1 |
            Tee-Object -FilePath "$RESULTS_DIR\exp1b_run.log"
        Run-Python @("portfolio_variance_v3c2.py") 2>&1 |
            Add-Content -Path "$RESULTS_DIR\exp1b_run.log"
        Remove-Item Env:\DEFI_TASK_FILTER -ErrorAction SilentlyContinue
        Remove-Item Env:\DEFI_SEEDS       -ErrorAction SilentlyContinue
        # Move exp1b outputs → RESULTS_DIR
        Move-Outputs "defi_v3_*.json"              $EXPERIMENTS_DIR $RESULTS_DIR
        Move-Outputs "*portfolio*variance*.json"   $EXPERIMENTS_DIR $RESULTS_DIR
    } finally { Pop-Location }
}

# =============================================================================
# STEP 3: extrap
# =============================================================================
Invoke-Step "extrap" "OOD extrapolation comparative run (Tab 9 OOD columns)" {
    Push-Location $EXPERIMENTS_DIR
    try {
        $mult  = if ($env:EXTRAP_MULTIPLIER) { $env:EXTRAP_MULTIPLIER } else { "2.0" }
        $frac  = if ($env:EXTRAP_TRAIN_FRAC) { $env:EXTRAP_TRAIN_FRAC } else { "0.8" }
        Run-Python @(
            "run_comparative_suite_benchmark_v2.py",
            "--extrap",
            "--extrap-multiplier", $mult,
            "--extrap-train-frac", $frac
        ) 2>&1 | Tee-Object -FilePath "$RESULTS_DIR\extrap_run.log"
        Write-Host "extrap output: $RESULTS_DIR\comparison_results\extrapolation\"
        Get-ChildItem "$RESULTS_DIR\comparison_results\extrapolation\" -ErrorAction SilentlyContinue
    } finally { Pop-Location }
}

# =============================================================================
# STEP 4: hybrid_all_domains
# =============================================================================
Invoke-Step "hybrid_all_domains" "Hybrid LLM+NN all-domains run — 10 domains (§10.9 hybrid)" {
    # Runtime domain-list validation
    $domainScript = @"
import importlib.util, sys, pathlib
spec = importlib.util.spec_from_file_location(
    'hybrid_mod',
    pathlib.Path(r'$($CORE_DIR -replace "\\","/")/generation/hybrid_all_domains_llm_nn/hybrid_system_llm_nn_all_domains.py')
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
"@
    $actualDomains = (& $PYTHON -c $domainScript).Trim()
    $expectedSorted = ($HYBRID_ALL_DOMAINS_EXPECTED -split "," | Sort-Object) -join ","
    $actualSorted   = ($actualDomains -split ","             | Sort-Object) -join ","
    if ($actualSorted -ne $expectedSorted) {
        Warn "hybrid_all_domains domain list MISMATCH — update HYBRID_ALL_DOMAINS_EXPECTED"
        Warn "  Expected: $expectedSorted"
        Warn "  Actual  : $actualSorted"
        Die "Domain list mismatch"
    }
    Write-Host "[hybrid_all_domains] Domain-list OK: $actualSorted"

    $hybridDir = Join-Path $CORE_DIR "generation\hybrid_all_domains_llm_nn"
    Push-Location $hybridDir
    try {
        Run-Python @(
            "hybrid_system_llm_nn_all_domains.py",
            "--samples", $FEYNMAN_SAMPLES
        ) 2>&1 | Tee-Object -FilePath "$RESULTS_DIR\hybrid_all_domains_run.log"
    } finally { Pop-Location }
}

# =============================================================================
# STEP 4a: instability
# =============================================================================
Invoke-Step "instability" "Instability Index analysis + all figures — §10.9 (Regime A/B/C)" {
    New-Item -ItemType Directory -Force -Path "$RESULTS_DIR\figures" | Out-Null
    $benchJson = Get-ChildItem "$RESULTS_DIR" -Filter "hypatiax_defi_benchmark_v3*results*.json" |
                 Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $benchArg = if ($benchJson) {
        Write-Host "[instability] Stage 2 extrapolation merge enabled: $($benchJson.FullName)"
        @("--benchmark-json", $benchJson.FullName)
    } else {
        Write-Host "[instability] No benchmark JSON found — Stage 2 (EX figure) skipped."
        @()
    }
    Run-Python @(
        "$EXPERIMENTS_DIR\run_instability_suite.py",
        "--results-dir", $RESULTS_DIR,
        "--out",         "$RESULTS_DIR\figures",
        "--csv-out",     "$RESULTS_DIR\figures\instability_analysis.csv",
        "--format", "png", "pdf"
    ) + $benchArg 2>&1 | Tee-Object -FilePath "$RESULTS_DIR\instability_run.log"
}

# =============================================================================
# STEP 5: exp2_feynman
# =============================================================================
Invoke-Step "exp2_feynman" "Feynman SR benchmark — Phase 2 noisy protocol (Tab 16-18)" {
    Push-Location $EXPERIMENTS_DIR
    try {
        Run-Python @(
            "run_comparative_suite_benchmark_v2.py",
            "--benchmark",        "feynman",
            "--samples",          $FEYNMAN_SAMPLES,
            "--pysr-timeout",     $FEYNMAN_TIMEOUT,
            "--checkpoint-name",  "feynman_exp2_checkpoint",
            "--resume"
        ) 2>&1 | Tee-Object -FilePath "$RESULTS_DIR\comparison_results\feynman-tests\exp2\exp2_run.log"
    } finally { Pop-Location }
}

# =============================================================================
# STEP 6: exp2
# =============================================================================
Invoke-Step "exp2" "Combined five-system comparison — all Methods (Tab 19 full)" {
    Push-Location $EXPERIMENTS_DIR
    try {
        Run-Python @(
            "run_comparative_suite_benchmark_v2.py",
            "--benchmark",       "all30",
            "--samples",         $FEYNMAN_SAMPLES,
            "--pysr-timeout",    $FEYNMAN_TIMEOUT,
            "--checkpoint-name", "exp2_checkpoint",
            "--resume"
        ) 2>&1 | Tee-Object -FilePath "$RESULTS_DIR\exp2_run.log"
    } finally { Pop-Location }
}

# =============================================================================
# STEP 7: exp3
# =============================================================================
Invoke-Step "exp3" "Nguyen-12 benchmark — SEED=42 (tab:nguyen12 · §10.8)" {
    Push-Location $EXPERIMENTS_DIR
    try {
        Run-Python @("exp3_nguyen12_hybrid50v_02.py", "--seed", "42") 2>&1 |
            Tee-Object -FilePath "$RESULTS_DIR\exp3_run.log"
        # Move exp3 outputs → RESULTS_DIR
        Move-Outputs "*nguyen*seed42*.json" $EXPERIMENTS_DIR $RESULTS_DIR
        Move-Outputs "*nguyen12*42*.json"   $EXPERIMENTS_DIR $RESULTS_DIR
    } finally { Pop-Location }
}

# =============================================================================
# STEP 8: exp3b
# =============================================================================
Invoke-Step "exp3b" "Nguyen-12 stability seeds 99/123/777/2024" {
    Push-Location $EXPERIMENTS_DIR
    try {
        foreach ($seed in @(99, 123, 777, 2024)) {
            Write-Host "--- exp3b seed=$seed ---"
            Run-Python @("exp3_nguyen12_hybrid50v_02.py", "--seed", "$seed") 2>&1 |
                Add-Content -Path "$RESULTS_DIR\exp3b_run.log"
        }
        Move-Outputs "*nguyen*.json" $EXPERIMENTS_DIR $RESULTS_DIR
    } finally { Pop-Location }
}

# =============================================================================
# STEP 9: suppA
# =============================================================================
Invoke-Step "suppA" "DeFi routing improvement experiments (Supplement A)" {
    Push-Location $EXPERIMENTS_DIR
    try {
        Run-Python @("run_hybrid_system_benchmark.py") 2>&1 |
            Tee-Object -FilePath "$RESULTS_DIR\suppA_run.log"
        Move-Outputs "consolidated_hybrid*.json" $EXPERIMENTS_DIR "$RESULTS_DIR\hybrid_llm_nn\defi"
        Move-Outputs "hybrid_system*.json"       $EXPERIMENTS_DIR "$RESULTS_DIR\hybrid_llm_nn\all_domains"
    } finally { Pop-Location }
}

# =============================================================================
# STEP 10: suppB — noise sweep
# =============================================================================
Invoke-Step "suppB" "Noise sweep benchmark (Tab 28, 29 · Supplement B)" {
    Push-Location $EXPERIMENTS_DIR
    try {
        Run-Python @("run_noise_sweep_benchmark.py") 2>&1 |
            Tee-Object -FilePath "$RESULTS_DIR\suppB_run.log"
        # Flatten per-equation subdirs → noise-sweep/ so tables-generator glob works
        $noiseSweepDir = "$RESULTS_DIR\comparison_results\feynman-tests\noise-sweep"
        Get-ChildItem -Path $noiseSweepDir -Recurse -Depth 2 -Filter "noise_sweep_*.json" |
            Where-Object { $_.DirectoryName -ne $noiseSweepDir } |
            ForEach-Object { Move-Item -Path $_.FullName -Destination $noiseSweepDir -Force -Verbose }
    } finally { Pop-Location }
}

# =============================================================================
# STEP 10b: suppB_sc — sample-complexity sweep
# =============================================================================
Invoke-Step "suppB_sc" "Sample-complexity sweep n ∈ {50…1000} (Tab 29 · Supplement B §6)" {
    Push-Location $EXPERIMENTS_DIR
    try {
        Run-Python @("run_sample_complexity_benchmark.py") 2>&1 |
            Tee-Object -FilePath "$RESULTS_DIR\suppB_sc_run.log"
        # Move sample_complexity_*.json to dedicated sample-complexity/ dir
        $scDir = "$RESULTS_DIR\comparison_results\feynman-tests\sample-complexity"
        New-Item -ItemType Directory -Force -Path $scDir | Out-Null
        $feynmanTestsDir = "$RESULTS_DIR\comparison_results\feynman-tests"
        Get-ChildItem -Path $feynmanTestsDir -Recurse -Filter "sample_complexity_*.json" |
            Where-Object { $_.DirectoryName -ne $scDir } |
            ForEach-Object { Move-Item -Path $_.FullName -Destination $scDir -Force -Verbose }
    } finally { Pop-Location }
}

# =============================================================================
# STEP 11: tables
# =============================================================================
Invoke-Step "tables" "Generate all LaTeX tables → $RESULTS_DIR\tables\" {
    New-Item -ItemType Directory -Force -Path "$RESULTS_DIR\tables" | Out-Null
    Push-Location "$REPO_ROOT\tables"
    try {
        $env:TABLE_OUTDIR        = "$RESULTS_DIR\tables"
        $env:VERIFY_RESULTS_DIR  = $RESULTS_DIR
        Run-Python @(
            "generate_tables.py",
            "--results-dir", $RESULTS_DIR,
            "--output-dir",  "$RESULTS_DIR\tables"
        ) 2>&1 | Tee-Object -FilePath "$RESULTS_DIR\tables_run.log"
        Write-Host "Tables written to: $RESULTS_DIR\tables\"
        Get-ChildItem "$RESULTS_DIR\tables\"
    } finally { Pop-Location }
}

# =============================================================================
# STEP 12: figures
# =============================================================================
Invoke-Step "figures" "Generate all paper figures → $RESULTS_DIR\figures\" {
    New-Item -ItemType Directory -Force -Path "$RESULTS_DIR\figures" | Out-Null
    Push-Location "$REPO_ROOT\figures"
    try {
        Run-Python @(
            "generate_figures.py",
            "--results-dir", $RESULTS_DIR,
            "--output-dir",  "$RESULTS_DIR\figures"
        ) 2>&1 | Tee-Object -FilePath "$RESULTS_DIR\figures_run.log"
        Write-Host "Figures written to: $RESULTS_DIR\figures\"
        Get-ChildItem "$RESULTS_DIR\figures\"
    } finally { Pop-Location }
}

# =============================================================================
# STEP 13: validate
# =============================================================================
Invoke-Step "validate" "Cross-check all results against paper-reported values" {
    $validateScript = @'
import json, os, glob, sys, statistics

RESULTS   = os.environ.get("RESULTS_DIR", r"hypatiax\data\results")
TOLERANCE = 0.01
checks    = []

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
    hx = [r for r in data.get("results", []) if r.get("method") in ("hybrid_v40", "Hybrid v40")]
    if hx:
        r2v = [r["r2_train"] for r in hx if "r2_train" in r]
        if r2v:
            check("Hybrid v40 mean train R²",   statistics.mean(r2v),   0.931)
            check("Hybrid v40 median train R²", statistics.median(r2v), 1.000)
else:
    print("  [SKIP] exp1 noiseless results not found")

# --- exp2_feynman ---
exp2_files = sorted(glob.glob(f"{RESULTS}/comparison_results/feynman-tests/exp2/exp2_results*.json"))
if exp2_files:
    with open(exp2_files[-1]) as f: data = json.load(f)
    rec = data.get("hybrid_deFi_recovery") or data.get("recovery_rate")
    if rec is not None:
        check("Hybrid DeFi recovery rate (Feynman noisy)", rec, 1.0, tol=0.001)
else:
    print("  [SKIP] exp2_feynman results not found")

# --- Mann-Whitney (Tab 14) ---
mw_files = sorted(glob.glob(f"{RESULTS}/exp1_rf01_mannwhitney*.json"))
if mw_files:
    with open(mw_files[-1]) as f: data = json.load(f)
    u = data.get("mann_whitney_u", data.get("U"))
    if u is not None: check("Mann-Whitney U (Hybrid v40 vs NN)", float(u), 0.0, tol=0.0)
    p = data.get("p_value", data.get("p"))
    if p is not None:
        ok = p < 1e-5
        checks.append(("p-value < 1e-5", p, 1.11e-6, ok))
        print(f"  [{'OK' if ok else 'FAIL'}] p-value < 1e-5: got={p:.2e}")
else:
    print("  [SKIP] Mann-Whitney results not found")

# --- hybrid_all_domains output ---
had = glob.glob(f"{RESULTS}/hybrid_llm_nn/all_domains/*.json")
ok  = bool(had)
checks.append(("hybrid_all_domains output present (all_domains/)", 1.0 if ok else 0.0, 1.0, ok))
print(f"  [{'OK' if ok else 'FAIL'}] hybrid_llm_nn/all_domains/: {len(had)} JSON file(s)")

# --- instability outputs ---
inst_csv = os.path.isfile(f"{RESULTS}/figures/instability_analysis.csv")
checks.append(("instability_analysis.csv present", 1.0 if inst_csv else 0.0, 1.0, inst_csv))
print(f"  [{'OK' if inst_csv else 'FAIL'}] instability_analysis.csv")
inst_fig = glob.glob(f"{RESULTS}/figures/fig_paper_complexity_vs_instability.pdf")
ok_ifig  = bool(inst_fig)
checks.append(("fig_paper_complexity_vs_instability.pdf present", 1.0 if ok_ifig else 0.0, 1.0, ok_ifig))
print(f"  [{'OK' if ok_ifig else 'FAIL'}] fig_paper_complexity_vs_instability.pdf (KEY §10.9)")

# --- suppB_sc output ---
sc = (glob.glob(f"{RESULTS}/comparison_results/feynman-tests/sample-complexity/*.json") +
      glob.glob(f"{RESULTS}/comparison_results/feynman-tests/sample-complexity/**/*.json"))
ok = bool(sc)
checks.append(("suppB_sc output present (sample-complexity/)", 1.0 if ok else 0.0, 1.0, ok))
print(f"  [{'OK' if ok else 'FAIL'}] sample-complexity outputs: {len(sc)} file(s)")

# --- suppB noise_sweep_*.json glob match ---
noise_sweep_matched = glob.glob(f"{RESULTS}/comparison_results/feynman-tests/noise-sweep/noise_sweep_*.json")
noise_sweep_all     = glob.glob(f"{RESULTS}/comparison_results/feynman-tests/noise-sweep/*.json")
if noise_sweep_all:
    ok = bool(noise_sweep_matched)
    checks.append(("suppB output matches noise_sweep_*.json glob", 1.0 if ok else 0.0, 1.0, ok))
    if not ok:
        bad = [os.path.basename(p) for p in noise_sweep_all[:5]]
        print(f"  [FAIL] noise-sweep/: {len(noise_sweep_all)} JSON(s) found but NONE match "
              f"noise_sweep_*.json. Actual: {bad}")
    else:
        print(f"  [OK]   noise-sweep/: {len(noise_sweep_matched)} noise_sweep_*.json — tables glob OK")
else:
    print("  [SKIP] noise-sweep/: no JSON files found (suppB not yet run)")

# --- tables and figures co-located ---
tbl    = glob.glob(f"{RESULTS}/tables/*.tex")
fig    = glob.glob(f"{RESULTS}/figures/*.pdf")
ok_tbl = bool(tbl); ok_fig = bool(fig)
checks.append(("tables in RESULTS_DIR/tables/", 1.0 if ok_tbl else 0.0, 1.0, ok_tbl))
checks.append(("figures in RESULTS_DIR/figures/", 1.0 if ok_fig else 0.0, 1.0, ok_fig))
print(f"  [{'OK' if ok_tbl else 'FAIL'}] {RESULTS}/tables/: {len(tbl)} .tex file(s)")
print(f"  [{'OK' if ok_fig else 'FAIL'}] {RESULTS}/figures/: {len(fig)} .pdf file(s)")

total  = len(checks)
passed = sum(1 for *_, ok in checks if ok)
print(f"\n=== Result: {passed}/{total} checks passed ===")
if passed < total:
    print("FAILED:")
    for label, got, exp, ok in checks:
        if not ok: print(f"  FAIL: {label} (got={got}, expected={exp})")
    sys.exit(1)
else:
    print("All checks passed.")
'@
    $env:RESULTS_DIR = $RESULTS_DIR
    Run-Python -Args @("-c", $validateScript)
}

# =============================================================================
# Final summary
# =============================================================================
Write-Host ""
Log "============================================================"
Log " HypatiaX reproduction pipeline COMPLETE"
Log "============================================================"
Write-Host ""
Write-Host "  Key output locations:"
Write-Host "    Results JSON:  $RESULTS_DIR\"
Write-Host "    LaTeX tables:  $RESULTS_DIR\tables\*.tex"
Write-Host "    Figures PDF:   $RESULTS_DIR\figures\*.pdf"
Write-Host ""
Write-Host "  Cross-reference with paper:"
Write-Host "    Table 9          <- exp1              (core extrapolation)"
Write-Host "    Table 11         <- exp1b             (DeFi routing)"
Write-Host "    Table 17         <- exp2_feynman      (Feynman noisy)"
Write-Host "    Table 19         <- exp2              (five-system comparison)"
Write-Host "    Table 28         <- suppB             (noise sweep)"
Write-Host "    Table 29 sc      <- suppB_sc          (sample complexity)"
Write-Host "    tab:hybrid_all   <- hybrid_all_domains (§10.9 hybrid system)"
Write-Host "    tab:nguyen12     <- exp3/exp3b"
Write-Host "    tab:instability  <- instability        (§10.9 Regime A/B/C)"
Write-Host ""
Write-Host "  To rebuild the paper PDF (requires MiKTeX or TeX Live on PATH):"
Write-Host "    cd $REPO_ROOT"
Write-Host "    pdflatex jmlr-hypatiax-paper-final.tex"
Write-Host ""
Log "Done. See individual *_run.log files in $RESULTS_DIR\ for per-step output."
