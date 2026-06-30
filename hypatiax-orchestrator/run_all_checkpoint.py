#!/usr/bin/env python3
"""
run_all_checkpoint.py  —  HypatiaX · Full reproducibility pipeline (Python)
Paper: "HypatiaX: A Hybrid Symbolic-Neural Framework for
        Extrapolation-Reliable Analytical Discovery"  (JMLR v3.0, Apr 2026)

Usage:
    python3 run_all.py                      # full pipeline
    python3 run_all.py --skip-slow          # skip slow steps
    python3 run_all.py --only exp3          # run one step by id
    python3 run_all.py --resume             # resume from last checkpoint
    python3 run_all.py --resume --from exp2 # resume, force-rerun from step
    python3 run_all.py --clear-checkpoint   # delete checkpoint and exit
    python3 run_all.py --continue-on-fail   # log failures but keep going
    python3 run_all.py --verify-only        # re-check results without re-running
    python3 run_all.py --seed 123           # override seed for all steps
    python3 run_all.py --only exp3 --seed 777
    python3 run_all.py --dry-run
    python3 run_all.py --dry-run --only exp1 --case-range 1-4
    python3 run_all.py --skip-paper
    python3 run_all.py --pysr-timeout 900
    python3 run_all.py --one-equation       # smoke-test: 1 equation per experiment
    python3 run_all.py --one-equation-paper # reviewer probe: paper-quality values

Step IDs (use with --only / --from):
    Setup    : deps  patches-gen  patches-apply  fixup-init  fixup-tex
               validate-patches  validate-paper-config  check-hypatiax-protocols
    Phase 1  : exp1  exp1_analysis  exp1b  extrap  hybrid_all_domains
               instability  exp2_feynman  exp2  exp3  exp3b
    Phase 2  : suppA  suppB  suppB_sc
    Phase 3  : provenance  discover-provenance  scan-imports  verify  hashlock
    Phase 4  : tables  figures
    Phase 4B : audit-setup  audit-NB-01 ... audit-NB-05

Notes:
    --from requires --resume to have any effect; alone it is a no-op.
    validate-patches (Phase 0) checks patched source code.
    verify (Phase 3) cross-checks numerical results — equivalent to run_all.sh validate.

Changelog v7.1 (2026-05-08):
    SYNC-run_all.sh:
      exp1        — cmd changed to hypatiax_defi_benchmark_v3c.py (direct script,
                    matches run_all.sh STEP 1; statistical_analysis.py split into
                    new exp1_analysis step, also Phase 1).
      suppA       — cmd changed to run_hybrid_system_benchmark.py (direct script,
                    matches run_all.sh STEP 9; was experiment_protocol_hybrid_routing.py).
      suppB       — cmd changed to run_noise_sweep_benchmark.py (direct script,
                    matches run_all.sh STEP 10; was experiment_protocol_noise_sweep.py
                    whose wrapper equivalence was unverified — aligning avoids
                    output-prefix divergence that breaks noise_sweep_*.json glob).
      instability — RESTORED as a real Phase 2 step (run_instability_suite.py,
                    --results-dir/--out/--csv-out/--format flags, matches STEP 4a
                    in run_all.sh). Was incorrectly absent since v6.0/v7.0.
      extrap      — cmd changed to run_comparative_suite_benchmark_v2.py with
                    --extrap / --extrap-multiplier / --extrap-train-frac flags
                    (matches run_all.sh STEP 3; was experiment_protocol_extrapolation_comparative.py).
      figures     — --outdir replaced with --results-dir + --output-dir (matches
                    run_all.sh STEP 12 invocation of generate_figures.py).
      tables      — --outdir replaced with --results-dir + --output-dir (matches
                    run_all.sh STEP 11 invocation of generate_tables.py).

Changelog v7.0 (2026-05-08):
    BLOCKER-1 RESOLVED: Renamed CI 'instability' step → 'hybrid_all_domains'.
              result_glob corrected to hybrid_llm_nn/all_domains/**/*.json.
              Step now passes --domains CLI flag and TASK_IDS/SHARD_IDS env vars
              to hybrid_system_llm_nn_all_domains.py, exactly mirroring the CI
              worker dispatch. suppA result_glob fixed to hybrid_pysr/defi/**/*.json
              (matching the CI RESULT_SUBDIR hybrid_pysr/defi).
    BLOCKER-2 RESOLVED: Added suppB_sc step (sample-complexity sweep,
              n ∈ {50,100,200,500,750,1000} × 30 Feynman equations) using
              run_sample_complexity_benchmark.py. Result subdir:
              comparison_results/feynman-tests/sample-complexity.
              Produces suppb_sc_metrics.tex, suppb_winrate.tex inputs.
    BLOCKER-3 RESOLVED: hybrid_all_domains result_subdir is now
              hybrid_llm_nn/all_domains (not hybrid_llm_nn/defi).
              ensure_output_dirs() creates both; artifact upload and
              tables-generator will now find files correctly.
    BLOCKER-4 RESOLVED: suppB filename glob updated to noise_sweep_*.json
              with fallback glob suppB_*.json via _suppb_result_glob() helper.
              Note: run_noise_sweep_benchmark.py MUST produce noise_sweep_*.json
              prefix — see WARN-3 in audit.
    BLOCKER-5 RESOLVED: Domain-list validation for hybrid_all_domains runs
              before the main experiment subprocess (WARN-2 / task 7 in audit).
              validate_hybrid_all_domains_ids() checks the 10 expected domain
              keys against ExperimentProtocolAll.get_all_domains() at runtime
              and aborts with a clear diff if they diverge.
    WARN-5 RESOLVED: Nguyen-12 91.7% caveat now printed prominently in the
              pipeline summary alongside the strict 33.3% figure.
    EXP2-ALIGN (v6.0): Removed exp2_sym / exp2_hyb — not in run_all.sh.
    FIX-EXP1B-ARGS (v4.8), FIX-EXP2-FUTURE (v4.8) retained.

Changelog v6.0 (2026-05-07):
    EXP2-ALIGN: Removed exp2_sym and exp2_hyb steps — not in run_all.sh.

Changelog v4.8 (2026-04-23):
    FIX-EXP1B-ARGS, FIX-EXP2-FUTURE.

Prerequisites:
    export ANTHROPIC_API_KEY="sk-ant-..."
    pip install -r requirements.txt
"""

import argparse
import importlib.util as _ilu
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path


# ── Load API key (env → Kaggle → .env → Colab) ─────────────────────────────
def load_repro_config() -> dict:
    """Load configuration from repro.yaml, with environment variable overrides."""
    import yaml  # type: ignore

    for config_path in [
        REPO_ROOT / "config" / "repro.yaml",
        REPO_ROOT / "repro.yaml",
    ]:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"  ⚠ Failed to load {config_path}: {e}")
    print("  ⚠ repro.yaml not found — using defaults")
    return {}


def _load_api_key() -> None:
    """Load ANTHROPIC_API_KEY via hypatiax/config_secrets.py, or fall back to .env."""
    _repo = Path(__file__).resolve().parent
    _config_secrets_path = _repo / "hypatiax" / "config_secrets.py"
    if _config_secrets_path.exists():
        try:
            _spec = _ilu.spec_from_file_location(
                "hypatiax._config_secrets_standalone", _config_secrets_path
            )
            if _spec and _spec.loader:
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)  # type: ignore[arg-type]
                if os.environ.get("ANTHROPIC_API_KEY"):
                    print("✅ ANTHROPIC_API_KEY loaded from hypatiax/config_secrets.py")
                    return
        except Exception as _e:
            print(f"  ⚠  config_secrets.py direct-load failed ({_e}); falling back")

    if os.environ.get("ANTHROPIC_API_KEY"):
        print("✅ ANTHROPIC_API_KEY already set in environment")
        return
    for _env_path in [
        _repo / "hypatiax" / ".env",
        _repo / ".env",
        Path.home() / ".env",
    ]:
        if _env_path.exists():
            for _line in _env_path.read_text().splitlines():
                _line = _line.strip()
                if _line.startswith("#") or "=" not in _line:
                    continue
                _k, _, _v = _line.partition("=")
                _k = _k.strip()
                _v = _v.strip().strip('"').strip("'")
                if _k == "ANTHROPIC_API_KEY" and _v:
                    os.environ["ANTHROPIC_API_KEY"] = _v
                    print(f"✅ ANTHROPIC_API_KEY loaded from {_env_path}")
                    return


_load_api_key()

# ── Canonical paths ──────────────────────────────────────────────────────────
REPO_ROOT       = Path(__file__).resolve().parent
RESULTS_DIR     = REPO_ROOT / "hypatiax" / "data" / "results"
EXPERIMENTS_DIR = REPO_ROOT / "hypatiax" / "experiments" / "benchmarks"
LOG_DIR         = REPO_ROOT / "logs"
CHECKPOINT  = LOG_DIR / "pipeline_checkpoint.json"
EXP2_EQ_CHECKPOINT = LOG_DIR / "exp2_eq_checkpoint.json"

# ── Strip incompatible deps from requirements.txt ───────────────────────────
_REQUIREMENTS  = REPO_ROOT / "requirements.txt"
_STRIP_PATTERNS = ["defi-risk", "optimum-onnx"]
if _REQUIREMENTS.exists():
    _lines    = _REQUIREMENTS.read_text().splitlines(keepends=True)
    _filtered = [l for l in _lines if not any(p in l for p in _STRIP_PATTERNS)]
    if len(_filtered) < len(_lines):
        _REQUIREMENTS.write_text("".join(_filtered))
        print(
            f"  ✂  Removed {len(_lines)-len(_filtered)} incompatible dep(s): "
            f"{_STRIP_PATTERNS}"
        )

# ── Stage paper .tex files into paper/ if they live at repo root ────────────
import shutil as _shutil  # noqa: E402

_PAPER_DIR   = REPO_ROOT / "paper"
_TEX_PATTERNS = [
    "jmlr_paper*.tex",
    "jmlr-hypatiax*.tex",
    "supp_routing_improvements.tex",
    "supp_benchmark_report.tex",
]
_staged: list[str] = []
for _pat in _TEX_PATTERNS:
    for _src in REPO_ROOT.glob(_pat):
        _dst = _PAPER_DIR / _src.name
        if not _dst.exists():
            _PAPER_DIR.mkdir(exist_ok=True)
            _shutil.copy2(_src, _dst)
            _staged.append(_src.name)
if _staged:
    print(f"  📄 Staged {len(_staged)} .tex file(s) into paper/: {_staged}")

_PAPER_STEP_IDS = {
    "audit-NB-01", "audit-NB-02", "audit-NB-03",
    "audit-NB-04", "audit-NB-05", "audit-setup",
    # validate-patches is a code-only check; keep it out of --skip-paper scope
}

# ── Domain registry ──────────────────────────────────────────────────────────
# BLOCKER-1 / WARN-2: canonical 10-domain list for hybrid_all_domains.
# validate_hybrid_all_domains_ids() checks this against the script at runtime.
HYBRID_ALL_DOMAINS_IDS: list[str] = [
    "mechanics", "electromagnetism", "thermodynamics", "quantum",
    "optics",    "chemistry",        "biology",        "statistics",
    "finance",   "other",
]

# ── suppB sample-complexity sweep parameters (BLOCKER-2) ───────────────────
SUPPB_SC_SAMPLE_COUNTS: list[str] = ["50", "100", "200", "500", "750", "1000"]


# ════════════════════════════════════════════════════════════════════════════
#  BLOCKER-1 / WARN-2 — Runtime domain-list validation
# ════════════════════════════════════════════════════════════════════════════
def validate_hybrid_all_domains_ids() -> bool:
    """
    Compare HYBRID_ALL_DOMAINS_IDS against what hybrid_system_llm_nn_all_domains.py
    actually exports (DOMAINS / ALL_DOMAINS / ExperimentProtocolAll.get_all_domains()).

    Returns True if lists match; prints a diff and returns False on mismatch.
    This runs BEFORE the 5.5-hour experiment so failures surface immediately.
    Mirrors the CI 'Validate hybrid_all_domains domain list' step (FIX TASK 7).
    """
    expected = set(HYBRID_ALL_DOMAINS_IDS)
    script = (
        REPO_ROOT
        / "hypatiax" / "experiments" / "generation"
        / "hybrid_all_domains_llm_nn"
        / "hybrid_system_llm_nn_all_domains.py"
    )
    if not script.exists():
        print(f"  ⚠  validate_hybrid_all_domains_ids: script not found at {script}")
        print("      Skipping domain-list validation (non-blocking).")
        return True  # tolerate absent script (CI may have it; local dev may not)

    spec = _ilu.spec_from_file_location("hybrid_mod", script)
    if spec is None or spec.loader is None:
        print("  ⚠  validate_hybrid_all_domains_ids: could not create module spec")
        return True

    mod = _ilu.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except SystemExit:
        pass  # script may call sys.exit() at module level — tolerate

    actual = (
        getattr(mod, "DOMAINS",     None)
        or getattr(mod, "ALL_DOMAINS", None)
        or getattr(mod, "DOMAIN_KEYS", None)
    )
    if actual is None:
        # Fallback: import ExperimentProtocolAll directly
        try:
            from hypatiax.experiments.generation.hybrid_all_domains_llm_nn\
                .hybrid_system_llm_nn_all_domains import ExperimentProtocolAll  # type: ignore
            actual = set(ExperimentProtocolAll().get_all_domains().keys())
        except Exception as e:
            print(f"  ⚠  Could not resolve domain list from script: {e}")
            return True  # non-blocking locally

    actual_set = {str(d) for d in actual}
    missing = expected - actual_set
    extra   = actual_set - expected

    if missing or extra:
        print("  ✗  DOMAIN LIST MISMATCH — update HYBRID_ALL_DOMAINS_IDS!")
        if missing:
            print(f"     In pipeline registry but NOT in script : {sorted(missing)}")
        if extra:
            print(f"     In script but NOT in pipeline registry : {sorted(extra)}")
        return False

    print(f"  ✓  Domain-list validation OK: {sorted(actual_set)}")
    return True


# ════════════════════════════════════════════════════════════════════════════
#  BLOCKER-4 — suppB result glob helper
# ════════════════════════════════════════════════════════════════════════════
def _suppb_result_glob() -> str:
    """
    Return the result glob for suppB.  Primary pattern: noise_sweep_*.json
    (the prefix tables-generator.py expects).  If none found, fall back to
    suppB_*.json (alternative script naming convention).

    NOTE: run_noise_sweep_benchmark.py MUST produce files with prefix
    'noise_sweep_' for tables-generator to pick them up automatically.
    If it uses a different prefix, align the script or update this helper.
    """
    primary = "comparison_results/feynman-tests/noise-sweep/noise_sweep_*.json"
    fallback = "comparison_results/feynman-tests/noise-sweep/suppB_*.json"
    if list(RESULTS_DIR.glob(primary)):
        return primary
    return fallback  # best-effort fallback


# ════════════════════════════════════════════════════════════════════════════
#  EXP2 isolated-runner (unchanged from v4.x — kept verbatim)
# ════════════════════════════════════════════════════════════════════════════
EXP2_PASS_THRESHOLD = 9
EXP2_KILL_GRACE     = 300

_EXP2_WORKER_SCRIPT = textwrap.dedent("""\
import json, os, sys, time, pathlib, traceback
import numpy as np

spec     = json.loads(os.environ["EXP2_EQUATION_JSON"])
out_path = pathlib.Path(os.environ["EXP2_RESULT_PATH"])
out_path.parent.mkdir(parents=True, exist_ok=True)

eq_name  = spec["name"]
seed     = int(os.environ.get("PYSR_SEED", "42"))
np.random.seed(seed)

repo_root = os.environ.get("REPRO_ROOT", str(pathlib.Path(__file__).resolve().parent))
for _p in [repo_root, os.path.join(repo_root, "hypatiax")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

t0 = time.perf_counter()
try:
    from hypatiax.tools.symbolic.symbolic_engine import DiscoveryConfig, SymbolicEngine

    cfg = DiscoveryConfig(
        pysr_timeout    = int(os.environ.get("PYSR_TIMEOUT",    "1100")),
        niterations     = int(os.environ.get("N_ITERATIONS",    "1000")),
        populations     = int(os.environ.get("POPULATIONS",     "30")),
        population_size = int(os.environ.get("PYSR_POPULATION_SIZE", "33")),
        parsimony       = float(os.environ.get("PYSR_PARSIMONY", "0.01")),
        maxsize         = int(os.environ.get("PYSR_MAXSIZE",    "30")),
        binary_operators = ["+", "-", "*", "/"],
        unary_operators  = ["exp", "log", "sin", "cos", "sqrt"],
    )

    N   = spec["n_samples"]
    rng = np.random.default_rng(seed)
    cols = []
    for vname, (lo, hi) in zip(spec["variable_names"], spec["variable_ranges"]):
        cols.append(rng.uniform(lo, hi, N))
    X = np.column_stack(cols)

    local_ns = {v: cols[i] for i, v in enumerate(spec["variable_names"])}
    local_ns["np"] = np
    y = eval(spec["numpy_expr"], {"__builtins__": {}},
             {**local_ns, "np": np,
              "exp": np.exp, "log": np.log, "sin": np.sin,
              "cos": np.cos, "sqrt": np.sqrt, "pi": np.pi})

    engine = SymbolicEngine(cfg, domain="physics")
    result = engine.discover(X, y, variable_names=spec["variable_names"])

    elapsed = time.perf_counter() - t0
    expr = result.get("expression", result.get("best_expression", "N/A"))
    r2   = float(result.get("r2_score", result.get("r2", float("nan"))))

    payload = {
        "equation": eq_name, "status": "ok",
        "expression": expr,  "r2": r2,
        "elapsed_s": elapsed, "ground_truth": spec["ground_truth"],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  ✅ [{eq_name}] R²={r2:.4f}  expr={expr}  ({elapsed:.1f}s)")
    sys.exit(0)

except Exception:
    elapsed = time.perf_counter() - t0
    tb = traceback.format_exc()
    payload = {"equation": eq_name, "status": "error", "error": tb, "elapsed_s": elapsed}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  ❌ [{eq_name}] FAILED after {elapsed:.1f}s:", file=sys.stderr)
    print(tb, file=sys.stderr)
    sys.exit(1)
""")

FEYNMAN_30 = [
    {"name": "I.6.2a",   "variable_names": ["theta"],
     "variable_ranges": [[-3.0, 3.0]],
     "numpy_expr": "np.exp(-theta**2/2) / np.sqrt(2*np.pi)",
     "ground_truth": "exp(-theta^2/2)/sqrt(2*pi)"},
    {"name": "I.9.18",   "variable_names": ["F","m","t1","t2"],
     "variable_ranges": [[1,10],[1,5],[2,10],[11,20]],
     "numpy_expr": "F / (m * (1/t1 - 1/t2))", "ground_truth": "F/(m*(1/t1-1/t2))"},
    {"name": "I.12.1",   "variable_names": ["F1","F2","eps","r"],
     "variable_ranges": [[1,5],[1,5],[0.5,2],[1,10]],
     "numpy_expr": "F1*F2 / (4*np.pi*eps*r**2)", "ground_truth": "F1*F2/(4*pi*eps*r^2)"},
    {"name": "I.12.2",   "variable_names": ["q1","q2","eps","r"],
     "variable_ranges": [[1,5],[1,5],[0.5,2],[1,10]],
     "numpy_expr": "q1*q2 / (4*np.pi*eps*r**2)", "ground_truth": "q1*q2/(4*pi*eps*r^2)"},
    {"name": "I.12.4",   "variable_names": ["q1","eps","r"],
     "variable_ranges": [[1,5],[0.5,2],[1,10]],
     "numpy_expr": "q1 / (4*np.pi*eps*r**2)", "ground_truth": "q1/(4*pi*eps*r^2)"},
    {"name": "I.15.1",   "variable_names": ["x","u","t","c"],
     "variable_ranges": [[1,10],[0.1,0.9],[1,5],[1,1]],
     "numpy_expr": "(x - u*t) / np.sqrt(1 - u**2/c**2)",
     "ground_truth": "(x-u*t)/sqrt(1-u^2/c^2)"},
    {"name": "I.18.4",   "variable_names": ["m1","m2","r1"],
     "variable_ranges": [[1,5],[1,5],[1,10]],
     "numpy_expr": "m1*r1 / (m1+m2)", "ground_truth": "m1*r1/(m1+m2)"},
    {"name": "I.24.6",   "variable_names": ["m","omega","omega0","x"],
     "variable_ranges": [[1,5],[1,5],[1,5],[1,5]],
     "numpy_expr": "0.25 * m * (omega**2 + omega0**2) * x**2",
     "ground_truth": "0.25*m*(omega^2+omega0^2)*x^2"},
    {"name": "I.26.2",   "variable_names": ["n","theta2"],
     "variable_ranges": [[0.5,1.0],[0.1,1.0]],
     "numpy_expr": "np.arcsin(n * np.sin(theta2))",
     "ground_truth": "arcsin(n*sin(theta2))"},
    {"name": "I.34.8",   "variable_names": ["omega","v","c"],
     "variable_ranges": [[1,10],[0.1,0.9],[1,1]],
     "numpy_expr": "omega / (1 - v/c)", "ground_truth": "omega/(1-v/c)"},
    {"name": "I.34.14",  "variable_names": ["omega0","v","c"],
     "variable_ranges": [[1,10],[0.1,0.9],[1,1]],
     "numpy_expr": "omega0 / (1 - v/c)", "ground_truth": "omega0/(1-v/c)"},
    {"name": "I.34.27",  "variable_names": ["h","omega"],
     "variable_ranges": [[0.5,2],[1,10]],
     "numpy_expr": "h * omega", "ground_truth": "h*omega"},
    {"name": "I.37.4",   "variable_names": ["I1","I2","delta"],
     "variable_ranges": [[1,5],[1,5],[0,3.14159]],
     "numpy_expr": "I1 + I2 + 2*np.sqrt(I1*I2)*np.cos(delta)",
     "ground_truth": "I1+I2+2*sqrt(I1*I2)*cos(delta)"},
    {"name": "I.41.16",  "variable_names": ["h","omega","c","kb","T"],
     "variable_ranges": [[0.5,2],[1,5],[1,3],[0.5,2],[100,1000]],
     "numpy_expr": "h*omega**3 / (np.pi**2 * c**3 * (np.exp(h*omega/(kb*T)) - 1))",
     "ground_truth": "h*omega^3/(pi^2*c^3*(exp(h*omega/(kb*T))-1))"},
    {"name": "I.43.31",  "variable_names": ["mob","kb","T"],
     "variable_ranges": [[0.5,2],[0.5,2],[100,1000]],
     "numpy_expr": "mob * kb * T", "ground_truth": "mob*kb*T"},
    {"name": "I.43.43",  "variable_names": ["kappa","T1","T2","A","d"],
     "variable_ranges": [[0.5,2],[200,500],[501,800],[1,5],[0.1,1]],
     "numpy_expr": "kappa * (T2-T1) * A / d",
     "ground_truth": "kappa*(T2-T1)*A/d"},
    {"name": "I.50.26",  "variable_names": ["x1","x2","omega","t"],
     "variable_ranges": [[1,5],[1,5],[1,5],[0,2]],
     "numpy_expr": "x1 + x2 * np.cos(omega * t)",
     "ground_truth": "x1+x2*cos(omega*t)"},
    {"name": "II.2.42",  "variable_names": ["kappa","T1","T2","A","d"],
     "variable_ranges": [[0.5,2],[200,500],[501,800],[1,5],[0.1,1]],
     "numpy_expr": "kappa * (T2 - T1) * A / d",
     "ground_truth": "kappa*(T2-T1)*A/d"},
    {"name": "II.11.27", "variable_names": ["n","alpha"],
     "variable_ranges": [[0.1,0.9],[0.1,1.0]],
     "numpy_expr": "n*alpha / (1 - n*alpha/3)",
     "ground_truth": "n*alpha/(1-n*alpha/3)"},
    {"name": "II.11.28", "variable_names": ["n","alpha"],
     "variable_ranges": [[0.1,0.9],[0.1,1.0]],
     "numpy_expr": "1 + n*alpha / (1 - n*alpha/3)",
     "ground_truth": "1+n*alpha/(1-n*alpha/3)"},
    {"name": "II.34.2a", "variable_names": ["q","v","r"],
     "variable_ranges": [[1,5],[1,10],[1,10]],
     "numpy_expr": "q*v / (2*np.pi*r)", "ground_truth": "q*v/(2*pi*r)"},
    {"name": "II.34.29b","variable_names": ["q","h","m","me"],
     "variable_ranges": [[1,3],[0.5,2],[1,5],[1,5]],
     "numpy_expr": "q*h*m / (4*np.pi*me)",
     "ground_truth": "q*h*m/(4*pi*me)"},
    {"name": "II.35.18", "variable_names": ["n0","m","g","x","kb","T"],
     "variable_ranges": [[1,5],[0.1,1],[5,15],[0,5],[0.5,2],[200,500]],
     "numpy_expr": "n0 * np.exp(-m*g*x / (kb*T))",
     "ground_truth": "n0*exp(-m*g*x/(kb*T))"},
    {"name": "II.36.38", "variable_names": ["mu","Ef","v"],
     "variable_ranges": [[0.1,1],[1,10],[10,50]],
     "numpy_expr": "mu*Ef / (1 + mu*Ef/v)",
     "ground_truth": "mu*Ef/(1+mu*Ef/v)"},
    {"name": "III.4.32", "variable_names": ["h","omega","kb","T"],
     "variable_ranges": [[0.5,2],[1,5],[0.5,2],[100,1000]],
     "numpy_expr": "h*omega / (np.exp(h*omega/(kb*T)) - 1)",
     "ground_truth": "h*omega/(exp(h*omega/(kb*T))-1)"},
    {"name": "III.4.33", "variable_names": ["h","omega","kb","T"],
     "variable_ranges": [[0.5,2],[1,5],[0.5,2],[100,1000]],
     "numpy_expr": ("h*omega * np.exp(h*omega/(kb*T)) / "
                    "(kb * T**2 * (np.exp(h*omega/(kb*T)) - 1)**2)"),
     "ground_truth": "h*omega*exp(h*omega/(kb*T))/(kb*T^2*(exp(h*omega/(kb*T))-1)^2)"},
    {"name": "III.12.4", "variable_names": ["n","h"],
     "variable_ranges": [[1,10],[0.5,2]],
     "numpy_expr": "n*h / (2*np.pi)", "ground_truth": "n*h/(2*pi)"},
    {"name": "III.14.14","variable_names": ["I0","q","V","kb","T"],
     "variable_ranges": [[0.1,2],[1,2],[0.1,1],[0.5,2],[200,500]],
     "numpy_expr": "I0 * (np.exp(q*V/(kb*T)) - 1)",
     "ground_truth": "I0*(exp(q*V/(kb*T))-1)"},
    {"name": "III.19.51","variable_names": ["m","q","eps","h","n"],
     "variable_ranges": [[0.5,2],[1,2],[0.5,2],[0.5,2],[1,5]],
     "numpy_expr": "-m * q**4 / (2 * (4*np.pi*eps)**2 * h**2) / n**2",
     "ground_truth": "-m*q^4/(2*(4*pi*eps)^2*h^2*n^2)"},
    {"name": "III.21.20","variable_names": ["rho","q","Ef","m"],
     "variable_ranges": [[0.5,2],[1,3],[1,10],[1,5]],
     "numpy_expr": "rho*q*Ef / m", "ground_truth": "rho*q*Ef/m"},
]


def _load_exp2_eq_checkpoint() -> dict:
    if EXP2_EQ_CHECKPOINT.exists():
        try:
            return json.loads(EXP2_EQ_CHECKPOINT.read_text())
        except Exception:
            pass
    return {}


def _save_exp2_eq_checkpoint(state: dict) -> None:
    EXP2_EQ_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    tmp = EXP2_EQ_CHECKPOINT.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(EXP2_EQ_CHECKPOINT)


def run_exp2_feynman(env: dict, args, log_fh) -> bool:
    """Per-equation isolated runner for exp2.  Returns True if ≥ EXP2_PASS_THRESHOLD solved."""
    n_tasks = int(env.get("N_FEYNMAN_TASKS", len(FEYNMAN_30)))
    equations = FEYNMAN_30[:n_tasks]
    n_samples = 300

    pysr_timeout  = int(env.get("PYSR_TIMEOUT", "1100"))
    kill_grace    = getattr(args, "kill_grace", None) or EXP2_KILL_GRACE
    kill_deadline = pysr_timeout + kill_grace

    out_dir = RESULTS_DIR / "comparison_results" / "feynman-tests" / "exp2"
    out_dir.mkdir(parents=True, exist_ok=True)

    worker_path = LOG_DIR / "_exp2_worker.py"
    worker_path.write_text(_EXP2_WORKER_SCRIPT)

    eq_checkpoint = _load_exp2_eq_checkpoint()
    results = []
    t_total = time.time()
    SEP  = "=" * 68
    SSEP = "-" * 68

    def _log(msg: str) -> None:
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    _log(f"\n{SEP}")
    _log(f"  exp2 · Feynman {n_tasks}-equation extrapolation (per-equation isolation)")
    _log(f"  PYSR_TIMEOUT={pysr_timeout}s  kill_grace={kill_grace}s  samples={n_samples}")
    _log(f"  pass_threshold={EXP2_PASS_THRESHOLD}/{n_tasks}")
    _log(SEP)

    for idx, spec in enumerate(equations):
        eq_name = spec["name"]
        if eq_name in eq_checkpoint and eq_checkpoint[eq_name].get("status") == "ok":
            cached = eq_checkpoint[eq_name]
            _log(f"\n  ↩  [{idx+1}/{n_tasks}] {eq_name}  "
                 f"(checkpoint: R²={cached.get('r2', '?'):.4f})  — skipping")
            results.append(cached)
            continue

        _log(f"\n{SSEP}")
        _log(f"  [{idx+1}/{n_tasks}] {eq_name}  gt={spec['ground_truth']}")

        run_spec    = {**spec, "n_samples": n_samples}
        result_path = out_dir / f"{eq_name.replace('.', '_')}.json"
        child_env   = {**env, "EXP2_EQUATION_JSON": json.dumps(run_spec),
                       "EXP2_RESULT_PATH": str(result_path)}

        t0   = time.time()
        proc = None
        deadline = t0 + kill_deadline
        status = "error"
        try:
            proc = subprocess.Popen(
                [sys.executable, str(worker_path)],
                env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, preexec_fn=os.setsid,
            )
            import queue as _queue
            import threading as _threading

            assert proc.stdout is not None
            _line_q: _queue.Queue = _queue.Queue()

            def _stdout_reader(stream, q):
                try:
                    for line in stream:
                        q.put(line)
                finally:
                    q.put(None)

            _reader_thread = _threading.Thread(
                target=_stdout_reader, args=(proc.stdout, _line_q), daemon=True
            )
            _reader_thread.start()
            timed_out = False
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    _log(f"\n  ⏱  [{eq_name}] wall-clock limit ({kill_deadline}s) — killing")
                    try:
                        import signal as _signal
                        os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
                    except Exception:
                        proc.kill()
                    timed_out = True
                    break
                try:
                    line = _line_q.get(timeout=min(remaining, 5.0))
                except _queue.Empty:
                    continue
                if line is None:
                    break
                log_fh.write(line)
                log_fh.flush()
                print(f"│  {line}", end="")
            proc.wait(timeout=30)
            elapsed = time.time() - t0
            if result_path.exists():
                try:
                    payload = json.loads(result_path.read_text())
                    status  = payload.get("status", "error")
                except Exception:
                    status = "error"
            else:
                status = "timeout" if timed_out else "error"

        except KeyboardInterrupt:
            if proc is not None:
                try:
                    proc.terminate(); proc.wait(timeout=5)
                except Exception:
                    try: proc.kill()
                    except Exception: pass
            _log(f"\n  ⚠  [{eq_name}] interrupted — saving checkpoint")
            _save_exp2_eq_checkpoint(eq_checkpoint)
            raise
        except Exception as exc:
            _log(f"\n  ❌ [{eq_name}] subprocess error: {exc}")
            status = "error"

        elapsed = time.time() - t0
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
            except Exception:
                result = {"equation": eq_name, "status": status, "elapsed_s": elapsed}
        else:
            result = {"equation": eq_name, "status": status, "elapsed_s": elapsed}

        results.append(result)
        eq_checkpoint[eq_name] = result
        _save_exp2_eq_checkpoint(eq_checkpoint)

        sym = "✅" if status == "ok" else ("⏱" if status == "timeout" else "❌")
        r2_str = f"R²={result.get('r2', float('nan')):.4f}" if status == "ok" else ""
        _log(f"\n  {sym} [{eq_name}] {status}  {r2_str}  ({elapsed:.0f}s)")

    total_elapsed = time.time() - t_total
    solved   = [r for r in results if r.get("status") == "ok"]
    timeouts = [r for r in results if r.get("status") == "timeout"]
    errors   = [r for r in results if r.get("status") not in ("ok","timeout")]

    _log(f"\n{SEP}")
    _log(f"  exp2 SUMMARY  —  {len(solved)}/{n_tasks} solved  "
         f"({len(timeouts)} timeouts  {len(errors)} errors)  "
         f"total {total_elapsed/60:.1f} min")
    _log(f"  {'#':<4} {'Name':<14} {'Status':<10} {'R²':>8}  Expression")
    _log("  " + "-" * 60)
    for i, r in enumerate(results):
        st  = r.get("status", "?")
        r2s = f"{r['r2']:.4f}" if st == "ok" and "r2" in r else "—"
        exp = r.get("expression", r.get("error", ""))[:40]
        _log(f"  {i+1:<4} {r.get('equation','?'):<14} {st:<10} {r2s:>8}  {exp}")

    consolidated = {
        "experiment": "exp2_feynman_30", "n_equations": n_tasks,
        "n_solved": len(solved), "solve_rate": len(solved) / n_tasks,
        "results": results,
    }
    consolidated_path = (
        RESULTS_DIR / "comparison_results" / "feynman-tests" / "exp2" / "exp2_results.json"
    )
    consolidated_path.write_text(json.dumps(consolidated, indent=2))
    _log(f"\n  Results → {consolidated_path}")
    _log(SEP)

    passed = len(solved) >= EXP2_PASS_THRESHOLD
    _log(f"\n  exp2 {'✅ PASS' if passed else '❌ FAIL'}  "
         f"({len(solved)}/{n_tasks} solved, threshold={EXP2_PASS_THRESHOLD})")
    return passed


# ════════════════════════════════════════════════════════════════════════════
#  Step dataclass & registry
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class PostMove:
    """Describes one file-move operation to execute after a step succeeds.

    Mirrors the ``find … -exec mv`` blocks in run_all.sh.

    Attributes
    ----------
    src_dir   : Directory to search for files.
    glob      : Filename glob pattern (e.g. ``'*nguyen*.json'``).
    dest_dir  : Directory to move matched files into (created if absent).
    recursive : If True, search src_dir recursively (rglob); otherwise
                only the immediate children are considered (like -maxdepth 1).
    subdir_only : If True *and* recursive is True, skip files that already
                  live directly inside src_dir (equivalent to -mindepth 2).
                  Used by the suppB noise-sweep flatten step.
    exclude   : Skip any path whose string representation contains this
                substring (e.g. ``'sample-complexity'`` to avoid moving files
                that are already in the right place).
    """
    src_dir: Path
    glob: str
    dest_dir: Path
    recursive: bool = False
    subdir_only: bool = False
    exclude: str = ""


@dataclass
class Step:
    id: str
    label: str
    cmd: list[str]
    phase: str
    slow: bool = False
    paper: bool = False
    env_extra: dict = field(default_factory=dict)
    expected: str = ""
    result_glob: str = ""
    inline_runner: bool = False
    post_move: list = field(default_factory=list)  # list[PostMove]


# ── helper: build the hybrid_all_domains domain string for --domains flag ──
def _hybrid_domain_args() -> list[str]:
    """Return ['--domains', 'mechanics', 'electromagnetism', ...]."""
    return ["--domains"] + HYBRID_ALL_DOMAINS_IDS


STEPS: list[Step] = [
    # ── Phase 0: Setup ─────────────────────────────────────────────────────
    # env_check mirrors run_all.sh STEP 0: verifies Python, PySR, API key, dirs.
    Step("env_check",
         "Verify environment (Python, PySR, API key, output directories)",
         [sys.executable, "-c", "\n".join([
             "import sys, os",
             "print('Python:', sys.version)",
             "try:",
             "    import pysr; print('PySR:', pysr.__version__)",
             "except ImportError:",
             "    print('ERROR: pysr not installed'); sys.exit(1)",
             "import torch; print('PyTorch:', torch.__version__)",
             "import anthropic; print('anthropic SDK: ok')",
             "import sympy; print('SymPy:', sympy.__version__)",
             "import scipy; print('SciPy:', scipy.__version__)",
             "key = os.environ.get('ANTHROPIC_API_KEY', '')",
             "if not key:",
             "    print('ERROR: ANTHROPIC_API_KEY not set'); sys.exit(1)",
             "print(f'ANTHROPIC_API_KEY: set ({len(key)} chars)')",
             "print('PYSR_POPULATIONS:', os.environ.get('PYSR_POPULATIONS','2'))",
             "from pathlib import Path",
             "results = Path(os.environ.get('RESULTS_DIR',",
             "               'hypatiax/data/results'))",
             "for sub in [",
             "    'comparison_results/feynman-tests/exp2',",
             "    'comparison_results/feynman-tests/noise-sweep',",
             "    'comparison_results/feynman-tests/sample-complexity',",
             "    'comparison_results/noise-noiseless/noiseless',",
             "    'comparison_results/noise-noiseless/15',",
             "    'comparison_results/extrapolation',",
             "    'extrapolation',",
             "    'hybrid_llm_nn/all_domains', 'hybrid_llm_nn/defi',",
             "    'hybrid_pysr/all_domains',   'hybrid_pysr/defi',",
             "    'llm_guided/all_domains',    'llm_guided/defi',",
             "    'standalone_llm_nn', 'figures', 'tables',",
             "]:",
             "    (results / sub).mkdir(parents=True, exist_ok=True)",
             "print('Directory structure: ok')",
         ])],
         phase="0 · Setup"),

    Step("deps", "Install dependencies",
         ["pip", "install", "-q", "-r", "requirements.txt"],
         phase="0 · Setup"),

    Step("patches-gen", "Generate patches",
         ["python3", "scripts/patches/generate_patches.py"],
         phase="0 · Setup"),

    Step("patches-apply", "Apply patches (FIX-C1…FIX-5b)",
         ["python3", "scripts/patches/apply_patches.py"],
         phase="0 · Setup"),

    Step("fixup-init",
         "Guard hypatiax/__init__.py broken HypatiaX import (FIX-INIT-PY)",
         ["python3", "-c", "\n".join([
             "from pathlib import Path",
             "init = Path('hypatiax') / '__init__.py'",
             "if not init.exists():",
             "    print('  ⚠ fixup-init: not found — skipping'); raise SystemExit(0)",
             "src = init.read_text(encoding='utf-8')",
             "BAD  = 'from hypatiax.core import HypatiaX'",
             "GOOD = ('try:\\n'",
             "        '    from hypatiax.core import HypatiaX  # noqa: F401\\n'",
             "        'except Exception:\\n'",
             "        '    HypatiaX = None  # type: ignore')",
             "if BAD in src and 'except Exception:' not in src:",
             "    init.write_text(src.replace(BAD, GOOD), encoding='utf-8')",
             "    print('  ✓ fixup-init: patched')",
             "else:",
             "    print('  ✓ fixup-init: already clean')",
         ])],
         phase="0 · Setup"),

    Step("fixup-tex",
         "Apply FIX-T2 (Five-Stage) + FIX-B2/B3 (rename dup bibkeys)",
         ["python3", "-c", "\n".join([
             "from pathlib import Path",
             "TEX = Path('paper') / 'jmlr-hypatiax-paper-final.tex'",
             "if not TEX.exists(): print(f'  ⚠ {TEX} not found'); raise SystemExit(0)",
             "src = TEX.read_text(encoding='utf-8'); orig = src",
             "src = src.replace('Five-Layer Architecture Overview',",
             "                  'Five-Stage Architecture Overview')",
             "for old, new in [('cranmer2023interpretable','cranmer2023interp'),",
             "                  ('udrescu2020aifeynman','udrescu2020feynman')]:",
             "    if old in src: src = src.replace(old, new); print(f'  ✓ renamed {old}')",
             "    else: print(f'  ✓ {old} absent')",
             "if src != orig: TEX.write_text(src, encoding='utf-8'); print('  ✓ patched')",
             "else: print('  ✓ already clean')",
             "final = TEX.read_text(encoding='utf-8')",
             "bad = {'FIX-T2':'Five-Layer Architecture Overview',",
             "       'FIX-B2':'cranmer2023interpretable',",
             "       'FIX-B3':'udrescu2020aifeynman'}",
             "fails = [f'{k}: still present' for k,v in bad.items() if v in final]",
             "if fails: print('\\n'.join(fails)); raise SystemExit(1)",
             "print('fixup-tex: done')",
         ])],
         phase="0 · Setup"),

    Step("validate-patches", "Validate patched source",
         ["python3", "scripts/patches/validate_code.py"],
         phase="0 · Setup"),

    Step("validate-paper-config",
         "Validate paper-quality configuration (repro.yaml v3.0)",
         ["python3", "validation_paper_config.py"],
         phase="0 · Setup",
         expected="All PAPER_CONFIG vars match repro.yaml v3.0 values"),

    Step("check-hypatiax-protocols",
         "Verify hypatiax/protocols/ input-data modules",
         ["python3", "scripts/patches/check_hypatiax_protocols.py"],
         phase="0 · Setup",
         expected="All 9 hypatiax/protocols/ input-data modules present"),

    # ── Phase 1: Core experiments ──────────────────────────────────────────
    Step("exp1",
         "Exp 1 · DeFi 74-task benchmark v3.0 (§10.2–10.4, §10.6)",
         [sys.executable,
          "hypatiax/experiments/benchmarks/hypatiax_defi_benchmark_v3c.py"],
         phase="1 · Core experiments",
         expected="89.2% R²>0.99 · 0 catastrophic · 1.73× speedup",
         result_glob="comparison_results/noise-noiseless/noiseless/*.json",
         # run_all.sh STEP 1: move benchmark result JSONs → RESULTS_DIR/
         post_move=[
             PostMove(EXPERIMENTS_DIR, "hypatiax_defi_benchmark_v3*results*.json", RESULTS_DIR),
             PostMove(EXPERIMENTS_DIR, "ablation_*.json", RESULTS_DIR),
             PostMove(EXPERIMENTS_DIR, "exp1_rf01_mannwhitney*.json", RESULTS_DIR),
         ]),

    Step("exp1_analysis",
         "Exp 1 · Statistical analysis (Tab 9 significance tests)",
         [sys.executable,
          "hypatiax/analysis/statistical_analysis.py"],
         phase="1 · Core experiments",
         expected="Mann-Whitney p<1e-5, effect-size tables written",
         result_glob="comparison_results/noise-noiseless/noiseless/*.json"),

    Step("exp1b",
         "Exp 1b · Portfolio Variance seed sweep (§10.5)",
         # run_all.sh STEP 2: benchmark with DEFI_TASK_FILTER=portfolio,
         # then portfolio_variance_v3c2.py for the variance analysis.
         # Both scripts run from EXPERIMENTS_DIR; pipeline runs them sequentially.
         [sys.executable, "-c",
          "import subprocess, sys, pathlib, os;"
          "bd = pathlib.Path('hypatiax/experiments/benchmarks');"
          "e = {**os.environ};"
          "e['DEFI_TASK_FILTER'] = 'portfolio';"
          "e['DEFI_SEEDS'] = '42,99,123,777,2024';"
          "r1 = subprocess.run([sys.executable, str(bd / 'hypatiax_defi_benchmark_v3c.py')], env=e);"
          "r2 = subprocess.run([sys.executable, str(bd / 'portfolio_variance_v3c2.py')], env=e);"
          "sys.exit(r1.returncode or r2.returncode)"],
         phase="1 · Core experiments",
         expected="P(H>P) ≈ 0.76",
         result_glob="defi_v3_*.json",
         env_extra={
             "DEFI_V3C_NO_TIMEOUT_FLAGS": "1",
             "DEFI_TASK_FILTER":          "portfolio",
             "DEFI_SEEDS":                "42,99,123,777,2024",
         },
         # run_all.sh STEP 2: move exp1b outputs → RESULTS_DIR/
         post_move=[
             PostMove(EXPERIMENTS_DIR, "defi_v3_*.json", RESULTS_DIR),
             PostMove(EXPERIMENTS_DIR, "*portfolio*variance*.json", RESULTS_DIR),
         ]),

    Step("extrap",
         "OOD extrapolation comparative — Tab 9 OOD columns (§10.8)",
         [sys.executable,
          "hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_v2.py",
          "--extrap",
          "--extrap-multiplier", str(float(os.environ.get("EXTRAP_MULTIPLIER", "2.0"))),
          "--extrap-train-frac", str(float(os.environ.get("EXTRAP_TRAIN_FRAC",  "0.8"))),
          ],
         phase="1 · Core experiments",
         result_glob="comparison_results/extrapolation/*.json",
         env_extra={
             # Env-override knobs (CI / ablation use) — match run_all.sh behaviour
             "EXTRAP_MULTIPLIER":      os.environ.get("EXTRAP_MULTIPLIER", "2.0"),
             "EXTRAP_TRAIN_FRAC":      os.environ.get("EXTRAP_TRAIN_FRAC",  "0.8"),
             "HYPATIAX_CORE_OPTIONAL": "1",
         }),

    # ── Phase 1 continued: hybrid_all_domains ──────────────────────────────
    Step("hybrid_all_domains",
         "Hybrid LLM+NN all-domains run · 10 domains (§results, was: instability)",
         [sys.executable,
          "hypatiax/experiments/generation/hybrid_all_domains_llm_nn/"
          "hybrid_system_llm_nn_all_domains.py",
          "--samples", str(int(os.environ.get("FEYNMAN_SAMPLES", "200"))),
          ] + _hybrid_domain_args(),
         phase="1 · Core experiments",
         slow=True,
         # LLM_K_RUNS=1 for one-shot run; instability_analysis step (future) uses K=30
         env_extra={
             "LLM_K_RUNS":    "1",
             "TASK_IDS":      " ".join(HYBRID_ALL_DOMAINS_IDS),
             "SHARD_IDS":     " ".join(HYBRID_ALL_DOMAINS_IDS),
             "HYPATIAX_CORE_OPTIONAL": "1",
         },
         expected=(
             "One-shot inference across 10 domains written to hybrid_llm_nn/all_domains/. "
             "NOTE: NOT the §10.9 Instability Index (K=30) — see instability_analysis step."
         ),
         # BLOCKER-1 + BLOCKER-3: correct subdir hybrid_llm_nn/all_domains
         result_glob="hybrid_llm_nn/all_domains/**/*.json"),

    # ── instability — §10.9 Instability Index (separate from hybrid_all_domains) ──
    # run_all.sh STEP 4a: runs run_instability_suite.py against K-run DeFi JSON
    # from exp1, producing Regime A/B/C taxonomy, Spearman ρ, and 12 figures.
    # Outputs land under ${RESULTS_DIR}/figures/.
    # NOTE: Meaningful II values (σ>0) require exp1 to have been run with K≥2
    # repeat runs or --variance mode; a single exp1 run yields II=0 (Regime A/B only).
    # --benchmark-json is auto-detected (most-recent DeFi benchmark JSON from exp1)
    # to enable Stage 2 extrapolation merge and the EX figure; omitted if not found.
    Step("instability",
         "Instability Index analysis + 12 figures — §10.9 (Regime A/B/C · Groups A–C + EX)",
         # run_all.sh STEP 4a: detects most-recent DeFi benchmark JSON for --benchmark-json.
         # Reproduced here as a wrapper that builds the arg list at runtime.
         [sys.executable, "-c", "\n".join([
             "import subprocess, sys, pathlib, glob, os",
             "results_dir = pathlib.Path(os.environ.get('RESULTS_DIR',"
             "    str(pathlib.Path('hypatiax/data/results').resolve())))",
             "figures_dir = results_dir / 'figures'",
             "figures_dir.mkdir(parents=True, exist_ok=True)",
             # Locate most-recent DeFi benchmark JSON (Stage 2 source)
             "bench_jsons = sorted(results_dir.glob('hypatiax_defi_benchmark_v3*results*.json'),"
             "                     key=lambda p: p.stat().st_mtime, reverse=True)",
             "bench_arg = ['--benchmark-json', str(bench_jsons[0])] if bench_jsons else []",
             "if bench_arg:",
             "    print(f'[instability] Stage 2 enabled: {bench_jsons[0].name}')",
             "else:",
             "    print('[instability] No benchmark JSON found — Stage 2 / EX figure skipped.')",
             "cmd = [sys.executable,"
             "       'hypatiax/experiments/benchmarks/run_instability_suite.py',"
             "       '--results-dir', str(results_dir),"
             "       '--out',         str(figures_dir),"
             "       '--csv-out',     str(figures_dir / 'instability_analysis.csv'),"
             "       '--format', 'png', 'pdf'] + bench_arg",
             "sys.exit(subprocess.run(cmd, env=os.environ).returncode)",
         ])],
         phase="1 · Core experiments",
         slow=True,
         expected=(
             "instability_analysis.csv + fig_paper_complexity_vs_instability.{png,pdf} "
             "written; Regime A/B/C counts; Spearman ρ printed"
         ),
         result_glob="figures/instability_analysis.csv",
         env_extra={"HYPATIAX_CORE_OPTIONAL": "1"}),

    Step("exp2_feynman",
         "Exp 2 · Feynman-30 SR benchmark — Phase 2 noisy protocol (§10.7)",
         [sys.executable,
          "hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_v2.py",
          "--benchmark", "feynman",
          "--samples",   str(int(os.environ.get("FEYNMAN_SAMPLES", "200"))),
          "--pysr-timeout", str(int(os.environ.get("PYSR_TIMEOUT", "1100"))),
          "--checkpoint-name", "feynman_exp2_checkpoint",
          "--resume"],
         phase="1 · Core experiments",
         slow=True,
         expected="stats.json written; ≥1/30 solved  [~15 min smoke / 8-24 h full]",
         result_glob="comparison_results/feynman-tests/exp2/*.json",
         env_extra={
             "N_FEYNMAN_TASKS": (
                 "1" if os.environ.get("ONE_EQUATION") == "1"
                 else str(int(os.environ.get("N_FEYNMAN_TASKS", "30")))
             ),
             "PYSR_TIMEOUT":  str(int(os.environ.get("PYSR_TIMEOUT", "1100"))),
             "POPULATIONS":   str(int(os.environ.get("POPULATIONS",  "30"))),
             "N_ITERATIONS":  str(int(os.environ.get("N_ITERATIONS", "1000"))),
         }),

    Step("exp2",
         "Exp 2 · Combined five-system comparison — all methods (§10.7 combined)",
         [sys.executable,
          "hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_v2.py",
          "--benchmark", "all30",
          "--samples",   str(int(os.environ.get("FEYNMAN_SAMPLES", "200"))),
          "--pysr-timeout", str(int(os.environ.get("PYSR_TIMEOUT", "1100"))),
          "--checkpoint-name", "exp2_checkpoint",
          "--resume"],
         phase="1 · Core experiments",
         expected="9/30 (30%)  [fast after method-5/6 checkpoints ready]",
         result_glob="comparison_results/**/*.json"),

    Step("exp3",
         "Exp 3 · Nguyen-12 SEED=42 (§10.8 primary)",
         # run_all.sh STEP 7: direct script in EXPERIMENTS_DIR (not -m module).
         [sys.executable,
          "hypatiax/experiments/benchmarks/exp3_nguyen12_hybrid50v_02.py",
          "--seed", "42"]
         + (["--n-tasks", "1"] if os.environ.get("ONE_EQUATION") == "1" else []),
         phase="1 · Core experiments",
         expected=(
             "11/12 (91.7% by 4-decimal rounding) · strict R²≥0.9999: 4/12 (33.3%) · "
             "MW U=113, p=0.0097"
         ),
         result_glob="hypatiax/data/results/nguyen12_exp3_*.json",
         env_extra={"SKIP_PKG_CHECK": "1"},
         # run_all.sh STEP 7: move nguyen*seed42*.json → RESULTS_DIR/
         post_move=[
             PostMove(EXPERIMENTS_DIR, "*nguyen*seed42*.json", RESULTS_DIR),
             PostMove(EXPERIMENTS_DIR, "*nguyen12*42*.json",   RESULTS_DIR),
         ]),

    Step("exp3b",
         "Exp 3b · Nguyen-12 seeds 99/123/777/2024 (§10.8 stability)",
         # run_all.sh STEP 8: direct script looped over seeds sequentially.
         [sys.executable, "-c",
          "import subprocess, sys, pathlib, os;"
          "s = pathlib.Path('hypatiax/experiments/benchmarks/exp3_nguyen12_hybrid50v_02.py');"
          "extra = ['--n-tasks','1'] if os.environ.get('ONE_EQUATION')=='1' else [];"
          "rc = 0;"
          "[rc := rc or subprocess.run([sys.executable, str(s), '--seed', seed] + extra,"
          " env=os.environ).returncode"
          " for seed in ('99','123','777','2024')];"
          "sys.exit(rc)"],
         phase="1 · Core experiments",
         expected="consistent with SEED=42 across all 5 seeds",
         result_glob="extrapolation/full_run_*.json",
         env_extra={"SKIP_PKG_CHECK": "1"},
         # run_all.sh STEP 8: move all *nguyen*.json → RESULTS_DIR/
         post_move=[
             PostMove(EXPERIMENTS_DIR, "*nguyen*.json", RESULTS_DIR),
         ]),

    # ── Phase 2: Supplementary benchmarks ─────────────────────────────────
    # Order mirrors run_all.sh _STEP_ORDER: suppA → suppB → suppB_sc
    #
    # suppA — hybrid-PySR DeFi benchmark (STEP 9 in run_all.sh) ───────────
    # WARN-1 DOCUMENTED: suppA runs run_hybrid_system_benchmark.py.
    # result_glob corrected to hybrid_pysr/defi (matches CI RESULT_SUBDIR).
    # suppB — noise sweep (STEP 10, σ ∈ {0,0.5,1,5,10}%, n=200, 30 eq) ──
    # BLOCKER-4: result_glob uses noise_sweep_*.json (tables-generator expectation).
    # run_noise_sweep_benchmark.py MUST output files with this prefix.
    # WARN-3: NEEDS_JULIA assumed false here; set env_extra NEEDS_JULIA=true
    #         if run_noise_sweep_benchmark.py invokes PySR (EHD/M3 symbolic path).
    Step("suppA",
         "Supp A · Hybrid-PySR DeFi benchmark (standalone run_hybrid_system_benchmark.py)",
         [sys.executable,
          "hypatiax/experiments/benchmarks/run_hybrid_system_benchmark.py"],
         phase="2 · Supplementary benchmarks",
         expected="+6pp Fix1, +5pp Fix2, +1pp Fix3",
         # BLOCKER-3 / WARN-1: result written to hybrid_pysr/defi (CI RESULT_SUBDIR).
         # Distinct from hybrid_all_domains output (hybrid_llm_nn/all_domains).
         result_glob="hybrid_pysr/defi/**/*.json",
         env_extra={
             "SKIP_PERF_ANALYSIS":    "1",
             "HYPATIAX_CORE_OPTIONAL": "1",
         },
         # run_all.sh STEP 9: move consolidated_hybrid* → hybrid_llm_nn/defi/
         #                     move hybrid_system*       → hybrid_llm_nn/all_domains/
         post_move=[
             PostMove(EXPERIMENTS_DIR, "consolidated_hybrid*.json",
                      RESULTS_DIR / "hybrid_llm_nn" / "defi"),
             PostMove(EXPERIMENTS_DIR, "hybrid_system*.json",
                      RESULTS_DIR / "hybrid_llm_nn" / "all_domains"),
         ]),

    # ── BLOCKER-1 RESOLVED: hybrid_all_domains (was: instability) ──────────
    # WHAT CHANGED:
    #   • Step id: instability → hybrid_all_domains
    #   • result_glob: hybrid_llm_nn/defi → hybrid_llm_nn/all_domains/**/*.json
    #   • cmd: passes --domains flag with all 10 domain keys (FIX TASK 7)
    #   • env_extra: TASK_IDS / SHARD_IDS set for _resolve_domains_from_env() fallback
    #   • Domain-list validated at pipeline startup via validate_hybrid_all_domains_ids()
    #
    # NAMING CLARIFICATION (audit DISCONNECT):
    #   This step = one-shot hybrid LLM+NN inference across 10 domains.
    #   Paper §10.9 Instability Index (σ over K=30 runs, 70 tasks) is produced by
    #   hypatiax_instability_analysis_pipeline.py (generate_all_figures.py GROUP A).
    #   That 30-run analysis has NO CI equivalent — it is a separate local pipeline.
    #   See task 2 in the audit for the planned 'instability_analysis' CI step.
    #
    # SYNC-run_all.sh (v7.1): suppB cmd changed to run_noise_sweep_benchmark.py
    #   (direct script, matches run_all.sh STEP 10).  Was experiment_protocol_noise_sweep.py
    #   whose wrapper equivalence was unverified — aligning avoids output-prefix divergence
    #   that would break the noise_sweep_*.json glob used by tables-generator and validate.
    Step("suppB",
         "Supp B · Noise sweep σ ∈ {0,0.5,1,5,10}% × 30 equations (§SuppB §5–7)",
         [sys.executable,
          "hypatiax/experiments/benchmarks/run_noise_sweep_benchmark.py"],
         phase="2 · Supplementary benchmarks",
         slow=True,
         expected=(
             "EHD 100% at all σ · HSL 90% noiseless, 100% at σ>0 · "
             "M3 avg 841.4s · M4 avg 11.1s · speedup 75.8×"
         ),
         result_glob=(
             "comparison_results/feynman-tests/noise-sweep/noise_sweep_*.json"
         ),
         # run_all.sh STEP 10: flatten per-equation subdirs → noise-sweep/
         # Files written to noise-sweep/<eq_id>/noise_sweep_*.json must be moved
         # up one level so tables-generator glob noise_sweep_*.json finds them.
         post_move=[
             PostMove(RESULTS_DIR / "comparison_results" / "feynman-tests" / "noise-sweep",
                      "noise_sweep_*.json",
                      RESULTS_DIR / "comparison_results" / "feynman-tests" / "noise-sweep",
                      recursive=True, subdir_only=True),
         ]),

    # ── BLOCKER-2 RESOLVED: suppB_sc — sample-complexity sweep ─────────────
    # Task format: sc_n{n}__{feynman_id}  e.g. sc_n200__I.6.20
    # Script: run_sample_complexity_benchmark.py
    # Result subdir: comparison_results/feynman-tests/sample-complexity
    # Produces: suppb_sc_metrics.tex, partial suppb_winrate.tex
    # n ∈ {50,100,200,500,750,1000} × 30 equations = 180 tasks
    # Default: σ=5% (Supplementary B §6 design)
    Step("suppB_sc",
         "Supp B-SC · Sample-complexity sweep n ∈ {50…1000} × 30 eq (§SuppB §6)",
         [sys.executable,
          "hypatiax/experiments/benchmarks/run_sample_complexity_benchmark.py"],
         phase="2 · Supplementary benchmarks",
         slow=True,
         expected=(
             "Both M3 & M4 plateau at n≈500 · convergence at n=50 visible · "
             "180 task results in sample-complexity/"
         ),
         result_glob=(
             "comparison_results/feynman-tests/sample-complexity/*.json"
         ),
         env_extra={
             # σ=5% fixed for sample-complexity sweep (Supp B §6)
             "NOISE_LEVEL":    "5.0",
             # Pass sample counts as comma-separated list; script iterates internally
             "SC_SAMPLE_COUNTS": ",".join(SUPPB_SC_SAMPLE_COUNTS),
             # N_FEYNMAN_TASKS honours --one-equation flag
             "N_FEYNMAN_TASKS": (
                 "1" if os.environ.get("ONE_EQUATION") == "1" else "30"
             ),
         },
         # run_all.sh STEP 10b: move sample_complexity_*.json that are NOT already
         # inside sample-complexity/ into that dedicated subdir.
         post_move=[
             PostMove(RESULTS_DIR / "comparison_results" / "feynman-tests",
                      "sample_complexity_*.json",
                      RESULTS_DIR / "comparison_results" / "feynman-tests" / "sample-complexity",
                      recursive=True,
                      exclude="sample-complexity"),
         ]),

    # ── suppA — hybrid-PySR DeFi benchmark ─────────────────────────────────
    # WARN-1 DOCUMENTED: suppA runs run_hybrid_system_benchmark.py (standalone
    # hybrid-PySR DeFi run), NOT the routing-improvements script from Supp A.
    # The routing improvements (Fix 0–5b, 66.2%→89.2%) are baked into exp1.
    # result_glob corrected to hybrid_pysr/defi (matches CI RESULT_SUBDIR).
    # BLOCKER-3 NOTE: suppA and hybrid_all_domains now use distinct output dirs:
    #   suppA            → hybrid_pysr/defi/**/*.json
    #   hybrid_all_domains → hybrid_llm_nn/all_domains/**/*.json
    Step("provenance",
         "§11 · Provenance audit — protocol orchestration",
         ["python3", "-c",
          "import subprocess, sys, pathlib; "
          "s = pathlib.Path('hypatiax/protocols/experiment_protocol_provenance_audit.py'); "
          "sys.exit(subprocess.run([sys.executable, str(s)]).returncode) "
          "if s.exists() else "
          "(print('  ⚠  not found — skipping') or sys.exit(0))"],
         phase="3 · Audit & verification"),

    Step("discover-provenance",
         "§11 · discover_provenance.py — link result files to families",
         ["python3", "-c",
          "import subprocess, sys, pathlib; "
          "m = pathlib.Path('provenance_map.json'); "
          "pathlib.Path('logs/provenance_audit').mkdir(parents=True, exist_ok=True); "
          "(print('INFO: provenance_map.json absent — skipping') or sys.exit(0)) "
          "if not m.exists() else "
          "sys.exit(subprocess.run([sys.executable, 'discover_provenance.py', "
          "'--root', '.', '--map', str(m), '--out', 'logs/provenance_audit']).returncode)"],
         phase="3 · Audit & verification"),

    Step("scan-imports",
         "§11 · scan_internal_imports.py — internal import DAG",
         [sys.executable, "scan_internal_imports.py",
          "--root", ".", "--out", "logs/repro_output"],
         phase="3 · Audit & verification"),

    Step("verify",
         "Verify results against paper targets",
         [sys.executable, "scripts/patches/verify_results.py", "--report"],
         phase="3 · Audit & verification",
         env_extra={
             "PATCHED_DATA_DIR":   str(REPO_ROOT / "hypatiax" / "data" / "results"),
             "VERIFY_RESULTS_DIR": str(RESULTS_DIR),
         }),

    Step("hashlock",
         "Hash lock check",
         [sys.executable, "hypatiax/reproducibility/hash_lock.py", "--check"],
         phase="3 · Audit & verification"),

    # ── Phase 4: Outputs ────────────────────────────────────────────────────
    Step("tables",
         "Generate all tables",
         [sys.executable, "tables/generate_tables.py",
          "--results-dir", str(RESULTS_DIR),
          "--output-dir",  str(RESULTS_DIR / "tables")],
         phase="4 · Outputs",
         result_glob="tables/*.tex",
         env_extra={
             "TABLE_OUTDIR":       str(RESULTS_DIR / "tables"),
             "VERIFY_RESULTS_DIR": str(RESULTS_DIR),
         }),

    Step("figures",
         "Generate all figures",
         [sys.executable, "figures/generate_figures.py",
          "--results-dir", str(RESULTS_DIR),
          "--output-dir",  str(RESULTS_DIR / "figures")],
         phase="4 · Outputs",
         result_glob="figures/*.pdf"),

    # ── Phase 4-B: Paper audit notebooks ───────────────────────────────────
    Step("audit-setup",
         "Paper audit · Copy main paper + supplements into notebooks/",
         ["python3", "-c", "\n".join([
             "import shutil, pathlib",
             "nb = pathlib.Path('notebooks'); nb.mkdir(exist_ok=True)",
             "search_dirs = [pathlib.Path('paper'), pathlib.Path('.'),",
             "               pathlib.Path('paper') / 'tables', pathlib.Path('logs')]",
             "copied = []; missing = []",
             "main = next((f for d in search_dirs",
             "             for pat in ('jmlr-hypatiax*.tex','jmlr_paper*.tex')",
             "             for f in d.glob(pat) if f.is_file()), None)",
             "if main: shutil.copy(main, nb / main.name); copied.append(main.name)",
             "else: print('WARNING: main paper .tex not found')",
             "for name in ('supp_routing_improvements.tex','supp_benchmark_report.tex'):",
             "    src = next((d/name for d in search_dirs if (d/name).is_file()), None)",
             "    if src: shutil.copy(src, nb/name); copied.append(name)",
             "    else: missing.append(name); print(f'WARNING: {name} not found')",
             "print(f'audit-setup: copied {len(copied)} file(s): {copied}')",
             "if missing: print(f'Missing: {missing}')",
         ])],
         phase="4-B · Paper audit", paper=True),

    Step("audit-NB-01", "Paper audit · NB-01 Citation & Bibliography",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-01_Citation_Bibliography_Audit.ipynb"],
         phase="4-B · Paper audit", paper=True),

    Step("audit-NB-02", "Paper audit · NB-02 Cross-Reference & Label",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-02_CrossReference_Label_Audit.ipynb"],
         phase="4-B · Paper audit", paper=True),

    Step("audit-NB-03", "Paper audit · NB-03 Section Structure & Numbering",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-03_Section_Structure_Numbering.ipynb"],
         phase="4-B · Paper audit", paper=True),

    Step("audit-NB-04", "Paper audit · NB-04 Numerical Consistency",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-04_Numerical_Consistency_Checker.ipynb"],
         phase="4-B · Paper audit", paper=True),

    Step("audit-NB-05", "Paper audit · NB-05 Figure & Image Dependencies",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-05_Figure_Image_Dependency_Checker.ipynb"],
         phase="4-B · Paper audit", paper=True),
]

STEP_IDS = [s.id for s in STEPS]


# ════════════════════════════════════════════════════════════════════════════
#  Checkpoint helpers
# ════════════════════════════════════════════════════════════════════════════
def load_checkpoint() -> dict:
    state: dict[str, str] = {}
    root_cp = REPO_ROOT / "pipeline_checkpoint.json"
    for path in [root_cp, CHECKPOINT]:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for k, v in data.items():
                    if state.get(k) != "pass":
                        state[k] = v
            except Exception:
                pass
    if state and not CHECKPOINT.exists():
        save_checkpoint(state)
    return state


def save_checkpoint(state: dict) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, str] = {}
    if CHECKPOINT.exists():
        try:
            merged = json.loads(CHECKPOINT.read_text())
        except Exception:
            pass
    for k, v in state.items():
        if merged.get(k) != "pass":
            merged[k] = v
    tmp = CHECKPOINT.with_suffix(".tmp")
    tmp.write_text(json.dumps(merged, indent=2))
    tmp.replace(CHECKPOINT)


def clear_checkpoint() -> None:
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print(f"  Checkpoint cleared: {CHECKPOINT}")
    else:
        print("  No checkpoint file found.")


# ════════════════════════════════════════════════════════════════════════════
#  Result-file helpers
# ════════════════════════════════════════════════════════════════════════════
def ensure_output_dirs() -> None:
    """Create all canonical output subdirs under hypatiax/data/results/."""
    for sub in [
        "comparison_results/extrapolation",
        "comparison_results/feynman-tests/exp2",
        "comparison_results/feynman-tests/noise-sweep",
        # BLOCKER-2: new subdir for suppB_sc
        "comparison_results/feynman-tests/sample-complexity",
        "comparison_results/noise-noiseless/noiseless",
        "comparison_results/noise-noiseless/15",
        "extrapolation",
        # BLOCKER-1 + BLOCKER-3: hybrid_all_domains output (not /defi)
        "hybrid_llm_nn/all_domains",
        # kept for suppA and other scripts that may write here
        "hybrid_llm_nn/defi",
        "hybrid_pysr/all_domains",
        # BLOCKER-3: suppA writes to hybrid_pysr/defi
        "hybrid_pysr/defi",
        "llm_guided/all_domains",
        "llm_guided/defi",
        "standalone_llm_nn",
        "figures",
        "tables",
    ]:
        (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)


def move_step_outputs(step: Step) -> None:
    """Execute post_move operations for a step, mirroring run_all.sh mv blocks.

    Called immediately after a step succeeds, before archive_step_results().
    Each PostMove entry maps to one ``find … -exec mv`` block in run_all.sh.
    """
    if not step.post_move:
        return
    for pm in step.post_move:
        pm.dest_dir.mkdir(parents=True, exist_ok=True)
        if pm.recursive:
            candidates = list(pm.src_dir.rglob(pm.glob))
        else:
            candidates = list(pm.src_dir.glob(pm.glob))
        moved = 0
        for src in candidates:
            if not src.is_file():
                continue
            if pm.subdir_only and src.parent == pm.src_dir:
                continue  # already at top level — mindepth 2 semantics
            if pm.exclude and pm.exclude in str(src):
                continue
            dst = pm.dest_dir / src.name
            if src == dst:
                continue  # nothing to do
            shutil.move(str(src), dst)
            print(f"│    mv {src.name} → {pm.dest_dir.relative_to(REPO_ROOT)}/")
            moved += 1
        if moved:
            print(f"│    post-move [{pm.glob}]: {moved} file(s) → {pm.dest_dir.relative_to(REPO_ROOT)}")


def archive_step_results(step: Step) -> None:
    if not step.result_glob:
        return
    pattern = step.result_glob
    if "**" in pattern:
        parts   = Path(pattern).parts
        star_i  = next(i for i, p in enumerate(parts) if "**" in p)
        base_d  = RESULTS_DIR / Path(*parts[:star_i])
        sub_pat = str(Path(*parts[star_i:]))
        matches = list(base_d.rglob(sub_pat)) if base_d.exists() else []
    else:
        matches = list(RESULTS_DIR.glob(pattern))
    if not matches:
        return
    dest = LOG_DIR / f"{step.id}_results"
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in matches:
        dst = dest / src.name
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            shutil.copy2(src, dst)
            count += 1
    if count:
        print(f"  📁  {count} result file(s) archived → logs/{step.id}_results/")


def inventory_results() -> tuple[int, int, int]:
    jsons = sum(1 for _ in RESULTS_DIR.rglob("*.json"))
    csvs  = sum(1 for _ in RESULTS_DIR.rglob("*.csv"))
    pdfs  = (
        sum(1 for _ in (RESULTS_DIR / "figures").glob("*.pdf"))
        if (RESULTS_DIR / "figures").exists() else 0
    )
    tables_dir = RESULTS_DIR / "tables"
    if not tables_dir.exists() or not any(tables_dir.glob("*.tex")):
        tables_dir = REPO_ROOT / "paper" / "tables"
    texs = sum(1 for _ in tables_dir.glob("*.tex")) if tables_dir.exists() else 0
    return jsons + csvs, pdfs, texs


# ════════════════════════════════════════════════════════════════════════════
#  Step result & runner
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class StepResult:
    id: str
    label: str
    status: str
    elapsed: float = 0.0
    log_path: Path | None = None
    returncode: int = 0


def run_step(step: Step, env: dict, args) -> StepResult:
    log_path   = LOG_DIR / f"{step.id}.log"
    merged_env = {**env, **step.env_extra}

    if args.case_range:
        try:
            _s, _e = args.case_range.split("-")
            merged_env["CASE_RANGE_START"] = _s.strip()
            merged_env["CASE_RANGE_END"]   = _e.strip()
        except ValueError:
            print(f"[CI] WARNING: --case-range '{args.case_range}' malformed — ignoring")

    print(f"\n┌─── [{step.id}] {step.label}")
    print(f"│    {time.strftime('%H:%M:%S')}")
    if step.expected:
        print(f"│    Expected : {step.expected}")
    if step.env_extra:
        for k, v in step.env_extra.items():
            print(f"│    env+  {k}={v}")
    if merged_env.get("CASE_RANGE_START"):
        print(f"│    case-range: {merged_env['CASE_RANGE_START']}-{merged_env.get('CASE_RANGE_END','?')}")
    print(f"│    cmd: {' '.join(str(x) for x in step.cmd)}")

    if getattr(args, "dry_run", False):
        _dry = {k: v for k, v in merged_env.items()
                if k not in os.environ or os.environ[k] != v}
        if _dry:
            print("│    env overrides:")
            for k, v in sorted(_dry.items()):
                print(f"│      {k}={v}")
        print("└─── (dry-run — not executed)\n")
        return StepResult(step.id, step.label, "skip")

    t0 = time.time()

    if step.inline_runner:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log_path, "w") as log_fh:
                ok = run_exp2_feynman(merged_env, args, log_fh)
            elapsed = time.time() - t0
            sym = "✓" if ok else "✗"
            print(f"\n└─── {sym} {'done' if ok else 'FAILED'}  ({elapsed:.0f}s)"
                  + (f"  — see {log_path}" if not ok else ""))
            if ok:
                move_step_outputs(step)
                archive_step_results(step)
            return StepResult(step.id, step.label,
                              "pass" if ok else "fail",
                              elapsed, log_path, 0 if ok else 1)
        except KeyboardInterrupt:
            elapsed = time.time() - t0
            print(f"\n└─── ✗ INTERRUPTED  ({elapsed:.0f}s)")
            raise
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"└─── ✗ ERROR: {exc}")
            return StepResult(step.id, step.label, "fail", elapsed, log_path)

    proc: subprocess.Popen | None = None
    try:
        with open(log_path, "w") as log_fh:
            proc = subprocess.Popen(
                step.cmd, env=merged_env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log_fh.write(line)
                print(f"│  {line}", end="")
            proc.wait()

        elapsed = time.time() - t0
        ok  = proc.returncode == 0
        sym = "✓" if ok else "✗"
        print(f"\n└─── {sym} {'done' if ok else 'FAILED'}  ({elapsed:.0f}s)"
              + (f"  — see {log_path}" if not ok else ""))
        if ok:
            move_step_outputs(step)
            archive_step_results(step)
        return StepResult(step.id, step.label,
                          "pass" if ok else "fail",
                          elapsed, log_path, proc.returncode)

    except KeyboardInterrupt:
        elapsed = time.time() - t0
        if proc is not None:
            try:
                proc.terminate(); proc.wait(timeout=5)
            except Exception:
                try: proc.kill()
                except Exception: pass
        print(f"\n└─── ✗ INTERRUPTED  ({elapsed:.0f}s)")
        raise
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"└─── ✗ ERROR: {exc}")
        return StepResult(step.id, step.label, "fail", elapsed, log_path)


def banner(msg: str) -> None:
    print("\n" + "═" * 68)
    print(f"  {msg}")
    print("═" * 68)


# ════════════════════════════════════════════════════════════════════════════
#  Stale-lock cleanup (unchanged)
# ════════════════════════════════════════════════════════════════════════════
def _clear_stale_locks() -> None:
    _cleared: list[str] = []
    _failed:  list[str] = []

    def _try_unlink(p: Path) -> None:
        if p.exists():
            try:
                p.unlink(); _cleared.append(str(p))
            except Exception as e:
                _failed.append(f"{p} ({e})")

    for lf in RESULTS_DIR.glob(".lock_*"):
        _try_unlink(lf)

    _exe = Path(sys.executable).resolve()
    _julia_roots = [
        _exe.parent.parent,
        Path.home() / ".local",
        Path.home() / ".julia" / "environments",
    ]
    _FS_ROOT      = Path("/")
    _BLOCKED_ROOTS = {_FS_ROOT, Path("/usr"), Path("/usr/local")}
    for _root in _julia_roots:
        if not _root.exists() or _root in _BLOCKED_ROOTS:
            continue
        try:
            for _pid in _root.rglob("julia_env/lock.pid"):
                _try_unlink(_pid)
        except OSError:
            pass

    _julia_home = Path.home() / ".julia"
    if _julia_home.exists():
        _locks_dir = _julia_home / "locks"
        if _locks_dir.exists():
            for lf in _locks_dir.iterdir():
                if lf.is_file():
                    _try_unlink(lf)
        _reg = _julia_home / "registries"
        if _reg.exists():
            for lf in _reg.rglob("*.lock"):
                _try_unlink(lf)

    try:
        for lf in REPO_ROOT.rglob("lock.pid"):
            _try_unlink(lf)
    except OSError:
        pass

    if _cleared:
        print(f"  🔓 Cleared {len(_cleared)} stale lock file(s):")
        for lf in _cleared:
            print(f"       {lf}")
    else:
        print("  🔓 No stale lock files found")
    if _failed:
        print(f"  ⚠  Could not remove {len(_failed)} lock(s):")
        for lf in _failed:
            print(f"       {lf}")


# ════════════════════════════════════════════════════════════════════════════
#  main()
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="HypatiaX reproducibility pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skip-slow",         action="store_true")
    parser.add_argument("--only",              metavar="ID")
    parser.add_argument("--resume",            action="store_true")
    parser.add_argument("--from",              dest="from_step", metavar="ID")
    parser.add_argument("--clear-checkpoint",  action="store_true")
    parser.add_argument("--continue-on-fail",  action="store_true")
    parser.add_argument("--verify-only",       action="store_true")
    parser.add_argument("--skip-paper",        action="store_true")
    parser.add_argument("--seed",              type=int, default=None, metavar="N")
    parser.add_argument("--pysr-timeout",      type=int, default=None, metavar="SECS")
    parser.add_argument("--kill-grace",        type=int, default=None, metavar="SECS")
    parser.add_argument("--one-equation",      action="store_true")
    parser.add_argument("--one-equation-paper",action="store_true")
    parser.add_argument("--case-range",        metavar="START-END", default=None)
    parser.add_argument("--dry-run",           action="store_true")
    args = parser.parse_args()

    if args.case_range and not args.only:
        parser.error("--case-range requires --only <STEP_ID>")
    if args.from_step and not args.resume:
        print("  WARNING: --from has no effect without --resume.", file=sys.stderr)

    os.chdir(REPO_ROOT)
    LOG_DIR.mkdir(exist_ok=True)
    ensure_output_dirs()
    _clear_stale_locks()

    if args.clear_checkpoint:
        clear_checkpoint(); sys.exit(0)

    banner(
        "HypatiaX · Reproducibility Pipeline v7.1"
        + ("  [DRY-RUN]"          if args.dry_run            else "")
        + ("  [SMOKE-TEST]"       if args.one_equation        else "")
        + ("  [PAPER-QUALITY-1]"  if args.one_equation_paper  else "")
    )
    print(f"  Repo      : {REPO_ROOT}")
    print(f"  Python    : {sys.version.split()[0]}")
    print(f"  Date      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results   : {RESULTS_DIR}")
    print(f"  Logs      : {LOG_DIR}")
    print(f"  Checkpoint: {CHECKPOINT}")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("\n  ERROR: ANTHROPIC_API_KEY is not set.")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)
    print(f"\n  API key   : set ({len(api_key)} chars)")

    # ── BLOCKER-1 / WARN-2: validate hybrid_all_domains domain list ─────────
    print("\n  Validating hybrid_all_domains domain list …")
    if not validate_hybrid_all_domains_ids():
        print("\n  ERROR: Domain-list validation failed. "
              "Update HYBRID_ALL_DOMAINS_IDS before running.")
        sys.exit(1)

    # ── hypatiax/protocols/ check ───────────────────────────────────────────
    hypatiax_proto = REPO_ROOT / "hypatiax" / "protocols"
    required_hp = [
        "experiment_protocol_defi.py",
        "experiment_protocol_defi_20.py",
        "experiment_protocol_nguyen12.py",
        "experiment_protocol_all_18_a.py",
        "experiment_protocol_all_20.py",
        "experiment_protocol_all_30.py",
        "experiment_protocol_benchmark.py",
        "experiment_protocol_benchmark_v2.py",
        "experiment_protocol_comparative.py",
    ]
    missing_hp = [f for f in required_hp if not (hypatiax_proto / f).exists()]
    if missing_hp:
        print(f"\n  ERROR: {len(missing_hp)} module(s) missing from hypatiax/protocols/:")
        for f in missing_hp:
            print(f"    ✗  {f}")
        sys.exit(1)
    print(f"  Protocols : all {len(required_hp)} hypatiax/protocols/ modules ✓")

    if args.verify_only:
        banner("Verify-only mode")
        subprocess.run([sys.executable, "scripts/patches/verify_results.py", "--report"],
                       check=False)
        subprocess.run([sys.executable, "hypatiax/reproducibility/hash_lock.py", "--check"],
                       check=False)
        sys.exit(0)

    # ── Load repro.yaml ─────────────────────────────────────────────────────
    _repro_config   = load_repro_config()
    _timeout_config = _repro_config.get("timeouts", {})
    _pysr_config    = _repro_config.get("pysr", {})

    DEFAULT_PYSR_TIMEOUT   = _timeout_config.get("pysr_attempt_seconds", 1100)
    DEFAULT_METHOD_TIMEOUT = _timeout_config.get("method_seconds",        900)
    DEFAULT_KILL_GRACE     = _timeout_config.get("kill_grace_seconds",    300)

    _seed_str = str(args.seed) if args.seed is not None else "42"

    env = {**os.environ}
    env["PYTHONWARNINGS"]  = "ignore"
    env["NN_SEED"]         = os.environ.get("NN_SEED",        _seed_str)
    env["PYSR_SEED"]       = os.environ.get("PYSR_SEED",      _seed_str)
    env["PYTHONHASHSEED"]  = os.environ.get("PYTHONHASHSEED", _seed_str)

    if args.seed is not None:
        env["NN_SEED"] = env["PYSR_SEED"] = env["PYTHONHASHSEED"] = _seed_str

    env.setdefault("LLM_MODEL",   _repro_config.get("llm_model",   "claude-sonnet-4-6"))
    env.setdefault("LLM_RETRIES", str(_repro_config.get("llm_retries", 3)))
    env.setdefault("LLM_K_RUNS",  "1")   # instability_analysis step overrides to 30

    env.setdefault("N_TASKS_DEFI",         str(_repro_config.get("n_tasks_defi",        74)))
    env.setdefault("N_TASKS_INSTABILITY",   str(_repro_config.get("n_tasks_instability", 70)))
    env.setdefault("PCA_TRAIN_FRAC",        str(_repro_config.get("pca_train_frac",      0.40)))
    env.setdefault("NN_TIME_LIMIT",         str(_repro_config.get("nn_time_limit",       120)))
    env.setdefault("ENGINE_NAME",
                   _repro_config.get("engine", {}).get("name", "hybrid_system_v50_2"))
    env.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    env.setdefault("JULIA_NUM_THREADS", "1")
    env.setdefault("FEYNMAN_SAMPLES",   str(_repro_config.get("feynman_samples", 200)))

    # Timeout priority: CLI flag > repro.yaml > env var > hard default
    if args.pysr_timeout is not None:
        env["PYSR_TIMEOUT"]   = str(args.pysr_timeout)
        env["METHOD_TIMEOUT"] = str(DEFAULT_METHOD_TIMEOUT)
        print(f"  PYSR_TIMEOUT={args.pysr_timeout}s  (--pysr-timeout override)")
    else:
        pysr_timeout   = DEFAULT_PYSR_TIMEOUT
        method_timeout = DEFAULT_METHOD_TIMEOUT
        if env_pysr := os.environ.get("PYSR_TIMEOUT"):
            pysr_timeout   = int(env_pysr)
            method_timeout = DEFAULT_METHOD_TIMEOUT
            print(f"  ⚠ PYSR_TIMEOUT={pysr_timeout}s from env")
        env["PYSR_TIMEOUT"]   = str(pysr_timeout)
        env["METHOD_TIMEOUT"] = str(method_timeout)
        print(f"  PYSR_TIMEOUT={pysr_timeout}s  METHOD_TIMEOUT={method_timeout}s")

    env.setdefault("POPULATIONS",          str(_pysr_config.get("populations",    30)))
    env.setdefault("N_ITERATIONS",         str(_pysr_config.get("niterations",  1000)))
    env.setdefault("PYSR_POPULATIONS",     env["POPULATIONS"])
    env.setdefault("PYSR_NITERATIONS",     env["N_ITERATIONS"])
    env.setdefault("PYSR_PARALLELISM",     _pysr_config.get("parallelism", "multithreading"))
    env.setdefault("EQUATION_WALL_CLOCK",
                   str(_timeout_config.get("equation_wall_clock", 1200)))
    env.setdefault("PYSR_POPULATION_SIZE", str(_pysr_config.get("population_size", 33)))
    env.setdefault("PYSR_PARSIMONY",       str(_pysr_config.get("parsimony",    0.01)))
    env.setdefault("PYSR_MAXSIZE",         str(_pysr_config.get("maxsize",        30)))

    env["PYTHONPATH"]      = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["RESULTS_DIR"]     = str(RESULTS_DIR)
    env["PIPELINE_PYTHON"] = sys.executable
    env["REPRO_ROOT"]      = str(REPO_ROOT)

    print(f"\n  Seeds: NN={env['NN_SEED']}  PYSR={env['PYSR_SEED']}  "
          f"HASH={env['PYTHONHASHSEED']}")
    print(f"  LLM_MODEL={env['LLM_MODEL']}")
    print(f"  PySR: iters={env['N_ITERATIONS']} pops={env['POPULATIONS']} "
          f"pop_sz={env['PYSR_POPULATION_SIZE']}")
    print(f"  FEYNMAN_SAMPLES={env['FEYNMAN_SAMPLES']}")

    # ── --one-equation smoke-test ────────────────────────────────────────────
    if args.one_equation:
        env.update({
            "ONE_EQUATION":        "1",
            "N_TASKS_DEFI":        "1",
            "N_CORE15_TASKS":      "1",
            "N_FEYNMAN_TASKS":     "1",
            "N_TASKS_INSTABILITY": "1",
            "N_NGUYEN_TASKS":      "1",
            "N_NOISE_EQUATIONS":   "1",
            "LLM_K_RUNS":          "1",
            "N_ITERATIONS":        "200",
            "POPULATIONS":         "10",
        })
        if args.pysr_timeout is None:
            env["PYSR_TIMEOUT"] = "60"
        print("\n" + "▲" * 68)
        print("  ▲▲  SMOKE-TEST MODE  (--one-equation) — NOT paper-quality")
        print("▲" * 68)

    # ── --one-equation-paper reviewer-probe ─────────────────────────────────
    if args.one_equation_paper:
        env.update({
            "ONE_EQUATION":        "1",
            "N_TASKS_DEFI":        "1",
            "N_CORE15_TASKS":      "1",
            "N_FEYNMAN_TASKS":     "1",
            "N_TASKS_INSTABILITY": "1",
            "N_NGUYEN_TASKS":      "1",
            "N_NOISE_EQUATIONS":   "1",
            "N_ITERATIONS":        "1000",
            "POPULATIONS":         "30",
            "PYSR_POPULATION_SIZE":"33",
            "PYSR_PARSIMONY":      "0.01",
            "PYSR_MAXSIZE":        "30",
            "PYSR_PARALLELISM":    "multithreading",
            "LLM_K_RUNS":          "30",
            "METHOD_TIMEOUT":      "900",
            "EQUATION_WALL_CLOCK": "1200",
        })
        if args.pysr_timeout is None:
            env["PYSR_TIMEOUT"] = "1100"
        print("\n" + "★" * 68)
        print("  ★★  PAPER-QUALITY PROBE  (--one-equation-paper)")
        print("★" * 68)

    if args.only and args.only not in STEP_IDS:
        print(f"\n  ERROR: unknown step id '{args.only}'.")
        print(f"  Valid ids: {', '.join(STEP_IDS)}")
        sys.exit(1)
    if args.from_step and args.from_step not in STEP_IDS:
        print(f"\n  ERROR: unknown step id '{args.from_step}'.")
        sys.exit(1)

    # ── Load checkpoint ─────────────────────────────────────────────────────
    checkpoint_state: dict[str, str] = {}
    if args.resume:
        root_cp = REPO_ROOT / "pipeline_checkpoint.json"
        for _cp in [root_cp, CHECKPOINT]:
            if _cp.exists():
                try:
                    for k, v in json.loads(_cp.read_text()).items():
                        if checkpoint_state.get(k) != "pass":
                            checkpoint_state[k] = v
                except Exception:
                    pass
        save_checkpoint(checkpoint_state)

        _done    = [s for s in STEPS if checkpoint_state.get(s.id) == "pass"]
        _pending = [s for s in STEPS if checkpoint_state.get(s.id) != "pass"]
        print(f"\n  Pipeline status ({len(_done)}/{len(STEPS)} done):")
        _cur_phase = ""
        for _s in STEPS:
            if _s.phase != _cur_phase:
                print(f"    Phase {_s.phase}")
                _cur_phase = _s.phase
            _st  = checkpoint_state.get(_s.id, "todo")
            _ico = {"pass": "✓", "fail": "✗", "todo": "·"}.get(_st, "·")
            print(f"      {_ico}  {_s.id}")
        if _pending:
            print(f"  Next: [{_pending[0].id}]")

    # ── Run pipeline ─────────────────────────────────────────────────────────
    results: list[StepResult] = []
    current_phase = ""
    t_total   = time.time()
    past_from = False

    try:
        for step in STEPS:
            if args.from_step and step.id == args.from_step:
                past_from = True
            if args.only and step.id != args.only:
                results.append(StepResult(step.id, step.label, "skip"))
                continue
            if args.resume and checkpoint_state.get(step.id) == "pass" and not past_from:
                results.append(StepResult(step.id, step.label, "resume-skip"))
                continue
            if step.phase != current_phase:
                banner(f"Phase {step.phase}")
                current_phase = step.phase
            if args.skip_slow and step.slow:
                results.append(StepResult(step.id, step.label, "skip"))
                print(f"  ── skip [{step.id}]  (--skip-slow)")
                continue
            if args.skip_paper and step.paper:
                results.append(StepResult(step.id, step.label, "skip"))
                print(f"  ── skip [{step.id}]  (--skip-paper)")
                continue

            result = run_step(step, env, args)
            results.append(result)
            checkpoint_state[step.id] = result.status
            save_checkpoint(checkpoint_state)

            if result.status == "fail" and not args.continue_on_fail:
                print(f"\n  Pipeline aborted at [{step.id}].")
                print(f"  Checkpoint saved → {CHECKPOINT}")
                print("  To resume:  python3 run_all.py --resume")
                _print_summary(results, time.time() - t_total)
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n  ⚠  Interrupted by user (Ctrl+C).")
        for r in results:
            if r.id not in checkpoint_state:
                checkpoint_state[r.id] = r.status
        save_checkpoint(checkpoint_state)
        print(f"  Checkpoint saved → {CHECKPOINT}")
        _print_summary(results, time.time() - t_total)
        sys.exit(130)

    _print_summary(results, time.time() - t_total)
    failed = [r for r in results if r.status == "fail"]
    if not failed and not args.only:
        clear_checkpoint()
    sys.exit(1 if failed else 0)


def _print_summary(results: list[StepResult], elapsed: float) -> None:
    passed       = [r for r in results if r.status == "pass"]
    failed       = [r for r in results if r.status == "fail"]
    skipped      = [r for r in results if r.status == "skip"]
    resume_skips = [r for r in results if r.status == "resume-skip"]

    hh, rem = divmod(int(elapsed), 3600)
    mm, ss  = divmod(rem, 60)

    banner("Pipeline summary")
    col = {"pass": "✓", "fail": "✗", "skip": "─", "resume-skip": "↩"}
    for r in results:
        t = f"  {r.elapsed:6.0f}s" if r.status in ("pass","fail") else "        "
        print(f"  {col[r.status]} [{r.id:30s}] {r.label[:46]:46s}{t}")

    print()
    print(f"  ✓ passed      : {len(passed)}")
    print(f"  ✗ failed      : {len(failed)}")
    print(f"  ─ skipped     : {len(skipped)}")
    print(f"  ↩ resume-skip : {len(resume_skips)}")
    print(f"  Wall time     : {hh:02d}:{mm:02d}:{ss:02d}")

    # WARN-5 RESOLVED: Nguyen-12 caveat printed in every summary
    print("\n  ⚠  Nguyen-12 caveat (exp3/exp3b):")
    print("       Paper abstract: 11/12 (91.7%) uses 4-decimal rounding (Uy et al. benchmark).")
    print("       Strict R²≥0.9999 threshold: 4/12 (33.3%).")
    print("       Both figures should appear in the abstract & §10.8 for transparency.")

    data_files, fig_files, tbl_files = inventory_results()
    print(f"\n  Results → {RESULTS_DIR}")
    print(f"    Data files (JSON+CSV) : {data_files}")
    print(f"    Figures (PDF)         : {fig_files}")
    print(f"    Tables  (TeX)         : {tbl_files}")

    if failed:
        print("\n  Failed steps:")
        for r in failed:
            print(f"    [{r.id}] → {r.log_path}")
        print(f"\n  Checkpoint : {CHECKPOINT}")
        print("  Resume     : python3 run_all.py --resume")
    else:
        print("\n  ✓ All steps passed.")
        print(f"  Results    : {RESULTS_DIR}/")
        print(f"  Figures    : {RESULTS_DIR}/figures/")
        print(f"  Tables     : {RESULTS_DIR}/tables/")
        print("  Checkpoint : cleared")


if __name__ == "__main__":
    main()
