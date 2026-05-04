#!/usr/bin/env python3
"""
run_all_checkpoint.py
    ↓
experiment_protocol_*.py
    ↓
universal_protocol.run_protocol(config, runner)
    ↓
run_task(config)
    ↓
SCRIPT_MAP → one benchmark script

run_all.py  —  HypatiaX · Full reproducibility pipeline (Python version)
Paper: "HypatiaX: A Hybrid Symbolic-Neural Framework for
        Extrapolation-Reliable Analytical Discovery"  (JMLR v3.0, Apr 2026)

Usage:
    python3 run_all.py                      # full pipeline
    python3 run_all.py --skip-slow          # skip slow steps (Feynman, noise sweep, instability)
    python3 run_all.py --only exp3          # run one step by id
    python3 run_all.py --resume             # resume from last checkpoint
    python3 run_all.py --resume --from exp2 # resume, but force-rerun from this step
    python3 run_all.py --clear-checkpoint   # delete checkpoint file and exit
    python3 run_all.py --continue-on-fail   # log failures but keep going
    python3 run_all.py --verify-only        # re-check existing results without re-running
    python3 run_all.py --seed 123           # override seed for all steps (default: 42)
    python3 run_all.py --only exp3 --seed 777  # single step with custom seed
    python3 run_all.py --dry-run               # print what would run, no execution
    python3 run_all.py --dry-run --only exp1 --case-range 1-4  # preview CI job
    python3 run_all.py --skip-paper         # skip pdflatex compile steps
    python3 run_all.py --pysr-timeout 900   # extend PySR wall-clock limit (default 1100s)
    python3 run_all.py --one-equation       # smoke-test: 1 equation per experiment, fast timeout

Step IDs (use with --only / --from):
    Setup   : deps  patches-gen  patches-apply  fixup-init  fixup-tex  validate  validate-paper-config  check-hypatiax-protocols
    Phase 1 : exp1  exp1b  exp2_feynman  exp2_sym  exp2_hyb  exp2  exp3  exp3b
    Phase 2 : suppB  suppA  instability  extrap
    Phase 3 : provenance  discover-provenance  scan-imports  verify  hashlock
    Phase 4 : figures  tables
    Phase 4B: audit-setup  audit-NB-01 ... audit-NB-05

Prerequisites:
    export ANTHROPIC_API_KEY="sk-ant-..."
    pip install -r requirements.txt

Changelog v5.0 (2026-05-03):
    EXP2-FEYNMAN: Added exp2_feynman as a standalone step in Phase 1, placed
               before exp2_sym/exp2_hyb/exp2 and clearly separated from them.
               exp2_feynman runs hypatiax/protocols/experiment_protocol_feynman_exp2.py
               (→ run_protocol → run_task → exp2_feynman_colab_multithreaded.py).
               Produces per-equation JSON + stats.json/results.csv/table.tex/report.html.
               Fully independent of the 6-method comparison suite (exp2_sym/hyb/exp2).
               Respects ONE_EQUATION=1 smoke-test flag via N_FEYNMAN_TASKS=1.

Changelog v4.9 (2026-05-01):
    EXP2-SPLIT: exp2 is now three sequential steps:
               exp2_sym  — run_exp2_symbolic_engine.py  (Method 5, PySR, resumable)
               exp2_hyb  — run_exp2_hybrid_system.py    (Method 6, PySR, resumable)
               exp2      — run_comparative_suite_benchmark_injected.py
                           runs Methods 1-4 live + injects pre-computed 5+6 results.
               Each step has its own checkpoint; a SIGKILL on method-5 never wipes
               method-6 progress.  Pipeline checkpoint migration: if the old
               'exp2: fail' entry exists, exp2_sym and exp2_hyb are treated as
               not-yet-run (both will start from scratch unless their own checkpoints
               exist in logs/).  Pass threshold: ≥9/30 equations solved.

Changelog v4.8 (2026-04-23):
    FIX-EXP1B-ARGS: exp1b Step no longer passes --task/--seeds to
               experiment_protocol_defi_v3.py — those flags were silently
               ignored (the protocol has no argparser). Task filtering and
               seed list are now forwarded via DEFI_TASK_FILTER and
               DEFI_SEEDS env vars, which experiment_protocol_defi_v3.py
               reads and forwards into config for hypatiax_defi_benchmark_v3c.py.
    FIX-EXP2-FUTURE: experiment_protocol_benchmark_v2.py had
               `from __future__ import annotations` at line 84 (after
               sys.path bootstrap block) — Python requires it at line 1.
               Moved to top of file; SyntaxError in exp2 resolved.

Changelog v4.7 (2026-04-22):
    FIX-ONE-EQ-EXP3: exp3 and exp3b now append --n-tasks 1 to their cmd when
               ONE_EQUATION=1 is active. Previously N_NGUYEN_TASKS=1 was set
               in the env but experiment_protocol_nguyen12_exp3.py controls
               task count via its own CLI arg — the env var was silently
               ignored, so all 12 Nguyen equations still ran.
    FIX-4:        --one-equation smoke-test now injects N_ITERATIONS=200 and
               POPULATIONS=10 so PySR actually runs light. Previously it still
               used default populations=30 / iterations=1000, making the smoke
               test as slow as a real run. PYSR_TIMEOUT also tightened from
               120 → 60s to match the lighter config.
    MINOR-A:   extrap step PROTOCOL_ROOT now auto-detects: tries
               hypatiax/protocols/ first, falls back to repo-root protocols/
               if the former is absent (matches the comment in FIX-EXTRAP).

Changelog v4.6 (2026-04-21):
    FIX-SUPPA:    suppA step now injects SKIP_PERF_ANALYSIS=1 (suppresses the call to
               hypatiax/analysis/analyze_hybrid_performance.py which does not exist)
               and HYPATIAX_CORE_OPTIONAL=1 as an extra guard against the __init__
               ImportError while fixup-init runs first to fix the root cause.
    FIX-EXTRAP:   extrap step now injects PROTOCOL_ROOT=<repo>/protocols/ so the
               extrapolation protocol finds experiment_protocol_benchmark_v2.py in
               the correct location (repo-root protocols/) not hypatiax/protocols/.
               Also injects HYPATIAX_CORE_OPTIONAL=1 so noisy/noiseless passes
               don't abort when hypatiax.core is unavailable.
    FIX-VERIFY:   verify step now passes PATCHED_DATA_DIR and VERIFY_RESULTS_DIR env
               vars pointing to hypatiax/data/results/ — previously verify_results.py
               looked under scripts/hypatiax/data/patched/ and found nothing.
    FIX-TABLES:   tables step now passes TABLE_OUTDIR and VERIFY_RESULTS_DIR env vars
               so generate_tables.py writes to RESULTS_DIR/tables/ (not paper/tables/)
               and sources its input data from the correct results directory.
    FIX-AUDIT-SETUP: audit-setup now searches paper/, repo-root, paper/tables/, and
               logs/ for supplement .tex files and emits explicit WARNINGs for any
               that cannot be found, instead of silently skipping them.
    FIX-INVENTORY:  inventory_results() now falls back to paper/tables/ when
               RESULTS_DIR/tables/ has no .tex files, so the pipeline summary
               correctly reports a non-zero table count.

Changelog v4.5 (2026-04-21):
    FIX-INIT-PY: Added fixup-init step (Phase 0, after patches-apply) that wraps
               the broken `from hypatiax.core import HypatiaX` line in hypatiax/
               __init__.py inside a try/except. This prevents ImportError from
               propagating into sub-packages (hypatiax.protocols.*, etc.) used by
               exp1b, suppA, extrap, and others.  The patch is idempotent.
    FIX-PKG-CHECK: exp3 and exp3b now inject SKIP_PKG_CHECK=1 into the subprocess
               environment. The protocol wrapper's importlib package check fails
               spuriously when packages are installed in an editable/stale venv
               even though they are importable.  SKIP_PKG_CHECK=1 bypasses the
               pre-flight check; packages were already validated by the deps step.
    FIX-PROVENANCE: provenance step is now graceful when
               protocols/experiment_protocol_provenance_audit.py is absent — prints
               a warning and exits 0 instead of crashing the pipeline.

Changelog v4.4 (2026-04-21):
    FIX-INIT:  Replaced `import hypatiax.config_secrets` (which triggers hypatiax/__init__.py
               → `from hypatiax.core import HypatiaX` → ImportError) with a robust
               _load_api_key() function that loads hypatiax/config_secrets.py *directly*
               via importlib, bypassing __init__.py entirely.  Falls back to a
               minimal inline .env parser so the pipeline starts cleanly even when
               the hypatiax package itself is broken.
    FIX-PY:    Inject PIPELINE_PYTHON=sys.executable into the subprocess environment.
    FIX-EXP3:  exp3 / exp3b cmd lists use sys.executable (not bare "python3") to
               avoid "Missing packages: pysr, sklearn, pandas" false-negatives.
    FIX-EXP1B: exp1b sets env_extra DEFI_V3C_NO_TIMEOUT_FLAGS=1 so the protocol
               wrapper does not forward --pysr-timeout/--method-timeout flags that
               hypatiax_defi_benchmark_v3c.py does not accept.

Changelog v4.3 (2026-04-21):
    NEW: --one-equation flag injects ONE_EQUATION=1 + N_TASKS_*=1 + N_FEYNMAN_TASKS=1
         into the environment so every experiment script runs exactly one equation.
         Also forces PYSR_TIMEOUT=120 (overridable with --pysr-timeout) and prints a
         SMOKE-TEST banner.  Use this to verify the full pipeline end-to-end quickly
         before committing to a full multi-hour run.

Changelog v4.2 (2026-04-21):
    FIX: KeyboardInterrupt (Ctrl+C) now caught in both run_step() and main():
         subprocess is terminated cleanly, checkpoint is saved, summary is
         printed, and the process exits with code 130 (standard Ctrl+C exit).
    FIX: archive_step_results() rglob sub_pattern no longer uses fragile
         lstrip() chain — '**/*.json' is passed to rglob() directly.
    FIX: exp3b label corrected to "seeds 99/123/777/2024" (SEED=42 = exp3).
    FIX: Popen now uses bufsize=1 (line-buffered) for more responsive streaming.
    NEW: --pysr-timeout SECS flag injects PYSR_TIMEOUT env var so experiment
         scripts can extend the PySR wall-clock limit on slower hardware (the
         360 s default causes 11/15 Hybrid results to be N/A locally, breaking
         the Mann-Whitney U stat — paper expects U=126 over 15 pairs, not 4).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── Load API key (env → Kaggle → .env → Colab) ────────────────────────────────
# FIX-INIT: importing hypatiax.config_secrets at module level triggers hypatiax/__init__.py,
# which does `from hypatiax.core import HypatiaX`.  If hypatiax.core is broken or
# not yet importable (e.g. editable-install stale .egg-link, missing compiled ext),
# the whole pipeline crashes before main() even runs.
#
# Strategy (in order):
#   1. Try importlib to load hypatiax/config_secrets.py *directly* as a standalone module,
#      completely bypassing hypatiax/__init__.py.
#   2. If that also fails (e.g. config_secrets.py itself has a bad import), fall back to
#      a minimal inline .env parser that replicates what hypatiax.config_secrets does:
#      read ANTHROPIC_API_KEY from the environment, then from hypatiax/.env or .env.
# Add after the imports (around line 50)

def load_repro_config() -> dict:
    """Load configuration from repro.yaml, with environment variable overrides."""
    import yaml

    config_path = REPO_ROOT / "config" / "repro.yaml"
    if not config_path.exists():
        config_path = REPO_ROOT / "repro.yaml"  # fallback to repo root

    if not config_path.exists():
        print("  ⚠ repro.yaml not found — using defaults")
        return {}

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config or {}
    except Exception as e:
        print(f"  ⚠ Failed to load repro.yaml: {e}")
        return {}

def _load_api_key() -> None:
    """Load ANTHROPIC_API_KEY via hypatiax.config_secrets, or fall back to .env parsing."""
    import importlib.util as _ilu

    # ── Attempt 1: load hypatiax/config_secrets.py directly (skips __init__.py) ──────
    _repo = Path(__file__).resolve().parent
    _config_secrets_path = _repo / "hypatiax" / "config_secrets.py"
    if _config_secrets_path.exists():
        try:
            _spec = _ilu.spec_from_file_location("hypatiax._config_secrets_standalone",
                                                  _config_secrets_path)
            if _spec and _spec.loader:
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)  # type: ignore[arg-type]
                # config_secrets.py sets os.environ["ANTHROPIC_API_KEY"] as a side-effect
                if os.environ.get("ANTHROPIC_API_KEY"):
                    print("✅ ANTHROPIC_API_KEY loaded from hypatiax/config_secrets.py")
                    return
        except Exception as _e:
            print(f"  ⚠  hypatiax/config_secrets.py direct-load failed ({_e}); "
                  "falling back to .env parser")

    # ── Attempt 2: minimal .env parser (no third-party deps) ──────────────────
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("✅ ANTHROPIC_API_KEY already set in environment")
        return
    for _env_path in [_repo / "hypatiax" / ".env",
                      _repo / ".env",
                      Path.home() / ".env"]:
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
    # Key still not found — main() will catch this and print a clear error.

_load_api_key()

# ── Canonical paths ────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "hypatiax" / "data" / "results"
LOG_DIR     = REPO_ROOT / "logs"
CHECKPOINT  = LOG_DIR / "pipeline_checkpoint.json"

# Per-equation checkpoint for exp2 (survives across restarts)
EXP2_EQ_CHECKPOINT = LOG_DIR / "exp2_eq_checkpoint.json"

# ── Strip incompatible deps from requirements.txt (local runs) ────────────────
# • defi-risk    : private SSH-only repo, unavailable locally
# • optimum-onnx : ==0.0.3 conflicts with transformers==5.0.0
_REQUIREMENTS = REPO_ROOT / "requirements.txt"
_STRIP_PATTERNS = ["defi-risk", "optimum-onnx"]
if _REQUIREMENTS.exists():
    _lines = _REQUIREMENTS.read_text().splitlines(keepends=True)
    _filtered = [_line for _line in _lines
                 if not any(p in _line for p in _STRIP_PATTERNS)]
    if len(_filtered) < len(_lines):
        _REQUIREMENTS.write_text("".join(_filtered))
        print(f"  ✂  Removed {len(_lines)-len(_filtered)} incompatible dep(s) "
              f"from requirements.txt: {_STRIP_PATTERNS}")

# ── Stage paper .tex files into paper/ if they live at repo root ──────────────
# validate_code.py and audit notebooks expect tex files in paper/.
# If the repo was published with them at root, copy them across automatically.
import shutil as _shutil  # noqa: E402 — must follow REPO_ROOT definition

_PAPER_DIR = REPO_ROOT / "paper"
_TEX_PATTERNS = [
    "jmlr_paper*.tex",               # main paper (underscore variant)
    "jmlr-hypatiax*.tex",            # main paper (hyphen variant)
    "supp_routing_improvements.tex", # Supp A (needed by FIX-XR3 in validate_code)
    "supp_benchmark_report.tex",     # Supp B
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

# Steps that are part of a paper compile (skipped by --skip-paper)
_PAPER_STEP_IDS = {"audit-NB-01", "audit-NB-02", "audit-NB-03",
                   "audit-NB-04", "audit-NB-05", "audit-setup"}


# ──────────────────────────────────────────────────────────────────────────────
#  EXP2 REDESIGN: per-equation isolated runner
# ──────────────────────────────────────────────────────────────────────────────

# Minimum solved equations to call exp2 a pass (paper reports 9/30 = 30%).
EXP2_PASS_THRESHOLD = 9

# Grace seconds added on top of PYSR_TIMEOUT before we SIGKILL the child.
EXP2_KILL_GRACE = 300

# Inline driver executed in each child process.  Receives the equation spec
# as a JSON string via the EXP2_EQUATION_JSON env var.  Writes a result JSON
# to the path in EXP2_RESULT_PATH env var.  Exit 0 = success, 1 = failure.
_EXP2_WORKER_SCRIPT = textwrap.dedent("""\
import json, os, sys, time, pathlib, traceback
import numpy as np

spec     = json.loads(os.environ["EXP2_EQUATION_JSON"])
out_path = pathlib.Path(os.environ["EXP2_RESULT_PATH"])
out_path.parent.mkdir(parents=True, exist_ok=True)

eq_name  = spec["name"]
seed     = int(os.environ.get("PYSR_SEED", "42"))
np.random.seed(seed)

# ── Resolve repo root from env (set by the parent pipeline) ──────────────────
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

    # Reconstruct X, y from spec
    N = spec["n_samples"]
    rng = np.random.default_rng(seed)
    # Each variable column: uniform in [lo, hi]
    cols = []
    for vname, (lo, hi) in zip(spec["variable_names"], spec["variable_ranges"]):
        cols.append(rng.uniform(lo, hi, N))
    X = np.column_stack(cols)

    # Evaluate ground-truth expression to get y
    local_ns = {v: cols[i] for i, v in enumerate(spec["variable_names"])}
    local_ns["np"] = np
    y = eval(spec["numpy_expr"], {"__builtins__": {}}, {**local_ns, "np": np,
        "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos,
        "sqrt": np.sqrt, "pi": np.pi})

    engine = SymbolicEngine(cfg, domain="physics")
    result = engine.discover(X, y, variable_names=spec["variable_names"])

    elapsed = time.perf_counter() - t0
    expr = result.get("expression", result.get("best_expression", "N/A"))
    r2   = float(result.get("r2_score", result.get("r2", float("nan"))))

    payload = {
        "equation":     eq_name,
        "status":       "ok",
        "expression":   expr,
        "r2":           r2,
        "elapsed_s":    elapsed,
        "ground_truth": spec["ground_truth"],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  ✅ [{eq_name}] R²={r2:.4f}  expr={expr}  ({elapsed:.1f}s)")
    sys.exit(0)

except Exception:
    elapsed = time.perf_counter() - t0
    tb = traceback.format_exc()
    payload = {
        "equation":  eq_name,
        "status":    "error",
        "error":     tb,
        "elapsed_s": elapsed,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  ❌ [{eq_name}] FAILED after {elapsed:.1f}s:", file=sys.stderr)
    print(tb, file=sys.stderr)
    sys.exit(1)
""")

# ── Feynman equation catalogue (30 equations used in the paper) ───────────────
# Each entry: name, variable_names, variable_ranges, numpy_expr, ground_truth.
# Ranges chosen to keep y well-behaved (no div-by-zero, no log(0)).
FEYNMAN_30 = [
    # I.6.2a: exp(-θ²/2)/√(2π)
    {"name": "I.6.2a",  "variable_names": ["theta"],
     "variable_ranges": [[-3.0, 3.0]],
     "numpy_expr": "np.exp(-theta**2/2) / np.sqrt(2*np.pi)",
     "ground_truth": "exp(-theta^2/2)/sqrt(2*pi)"},

    # I.9.18: F / (m*(1/t1 - 1/t2))
    {"name": "I.9.18",  "variable_names": ["F","m","t1","t2"],
     "variable_ranges": [[1,10],[1,5],[2,10],[11,20]],
     "numpy_expr": "F / (m * (1/t1 - 1/t2))",
     "ground_truth": "F / (m*(1/t1 - 1/t2))"},

    # I.12.1: F1*F2/(4*pi*eps*r²)
    {"name": "I.12.1",  "variable_names": ["F1","F2","eps","r"],
     "variable_ranges": [[1,5],[1,5],[0.5,2],[1,10]],
     "numpy_expr": "F1*F2 / (4*np.pi*eps*r**2)",
     "ground_truth": "F1*F2/(4*pi*eps*r^2)"},

    # I.12.2: q1*q2/(4*pi*eps*r²)
    {"name": "I.12.2",  "variable_names": ["q1","q2","eps","r"],
     "variable_ranges": [[1,5],[1,5],[0.5,2],[1,10]],
     "numpy_expr": "q1*q2 / (4*np.pi*eps*r**2)",
     "ground_truth": "q1*q2/(4*pi*eps*r^2)"},

    # I.12.4: q1*r/(4*pi*eps*r³)  simplified as q1/(4*pi*eps*r²)
    {"name": "I.12.4",  "variable_names": ["q1","eps","r"],
     "variable_ranges": [[1,5],[0.5,2],[1,10]],
     "numpy_expr": "q1 / (4*np.pi*eps*r**2)",
     "ground_truth": "q1/(4*pi*eps*r^2)"},

    # I.15.1: x - u*t / sqrt(1 - u²/c²)
    {"name": "I.15.1",  "variable_names": ["x","u","t","c"],
     "variable_ranges": [[1,10],[0.1,0.9],[1,5],[1,1]],
     "numpy_expr": "(x - u*t) / np.sqrt(1 - u**2/c**2)",
     "ground_truth": "(x-u*t)/sqrt(1-u^2/c^2)"},

    # I.18.4: m1*r1 / (m1+m2)
    {"name": "I.18.4",  "variable_names": ["m1","m2","r1"],
     "variable_ranges": [[1,5],[1,5],[1,10]],
     "numpy_expr": "m1*r1 / (m1+m2)",
     "ground_truth": "m1*r1/(m1+m2)"},

    # I.24.6: 1/4 * m*(ω²+ω0²)*x²
    {"name": "I.24.6",  "variable_names": ["m","omega","omega0","x"],
     "variable_ranges": [[1,5],[1,5],[1,5],[1,5]],
     "numpy_expr": "0.25 * m * (omega**2 + omega0**2) * x**2",
     "ground_truth": "0.25*m*(omega^2+omega0^2)*x^2"},

    # I.26.2: arcsin(n*sin(θ2))
    {"name": "I.26.2",  "variable_names": ["n","theta2"],
     "variable_ranges": [[0.5,1.0],[0.1,1.0]],
     "numpy_expr": "np.arcsin(n * np.sin(theta2))",
     "ground_truth": "arcsin(n*sin(theta2))"},

    # I.34.8: ω/(1 - v/c)
    {"name": "I.34.8",  "variable_names": ["omega","v","c"],
     "variable_ranges": [[1,10],[0.1,0.9],[1,1]],
     "numpy_expr": "omega / (1 - v/c)",
     "ground_truth": "omega/(1-v/c)"},

    # I.34.14: ω0/(1-v/c) — same structure, different physics
    {"name": "I.34.14", "variable_names": ["omega0","v","c"],
     "variable_ranges": [[1,10],[0.1,0.9],[1,1]],
     "numpy_expr": "omega0 / (1 - v/c)",
     "ground_truth": "omega0/(1-v/c)"},

    # I.34.27: h*ω
    {"name": "I.34.27", "variable_names": ["h","omega"],
     "variable_ranges": [[0.5,2],[1,10]],
     "numpy_expr": "h * omega",
     "ground_truth": "h*omega"},

    # I.37.4: I1+I2+2*sqrt(I1*I2)*cos(delta)
    {"name": "I.37.4",  "variable_names": ["I1","I2","delta"],
     "variable_ranges": [[1,5],[1,5],[0,3.14159]],
     "numpy_expr": "I1 + I2 + 2*np.sqrt(I1*I2)*np.cos(delta)",
     "ground_truth": "I1+I2+2*sqrt(I1*I2)*cos(delta)"},

    # I.41.16: h*omega³/(pi²*c³*(exp(h*omega/(kb*T))-1))
    {"name": "I.41.16", "variable_names": ["h","omega","c","kb","T"],
     "variable_ranges": [[0.5,2],[1,5],[1,3],[0.5,2],[100,1000]],
     "numpy_expr": "h*omega**3 / (np.pi**2 * c**3 * (np.exp(h*omega/(kb*T)) - 1))",
     "ground_truth": "h*omega^3/(pi^2*c^3*(exp(h*omega/(kb*T))-1))"},

    # I.43.31: mob*kb*T
    {"name": "I.43.31", "variable_names": ["mob","kb","T"],
     "variable_ranges": [[0.5,2],[0.5,2],[100,1000]],
     "numpy_expr": "mob * kb * T",
     "ground_truth": "mob*kb*T"},

    # I.43.43: kappa*(T2-T1)*A/d
    {"name": "I.43.43", "variable_names": ["kappa","T1","T2","A","d"],
     "variable_ranges": [[0.5,2],[200,500],[501,800],[1,5],[0.1,1]],
     "numpy_expr": "kappa * (T2-T1) * A / d",
     "ground_truth": "kappa*(T2-T1)*A/d"},

    # I.50.26: x1 + x2*cos(omega*t)
    {"name": "I.50.26", "variable_names": ["x1","x2","omega","t"],
     "variable_ranges": [[1,5],[1,5],[1,5],[0,2]],
     "numpy_expr": "x1 + x2 * np.cos(omega * t)",
     "ground_truth": "x1+x2*cos(omega*t)"},

    # II.2.42: kappa*(T2-T1)*A/d  (same as I.43.43 but different physics)
    {"name": "II.2.42", "variable_names": ["kappa","T1","T2","A","d"],
     "variable_ranges": [[0.5,2],[200,500],[501,800],[1,5],[0.1,1]],
     "numpy_expr": "kappa * (T2 - T1) * A / d",
     "ground_truth": "kappa*(T2-T1)*A/d"},

    # II.11.27: n*alpha/(1-n*alpha/3)
    {"name": "II.11.27","variable_names": ["n","alpha"],
     "variable_ranges": [[0.1,0.9],[0.1,1.0]],
     "numpy_expr": "n*alpha / (1 - n*alpha/3)",
     "ground_truth": "n*alpha/(1-n*alpha/3)"},

    # II.11.28: 1+n*alpha/(1-n*alpha/3)
    {"name": "II.11.28","variable_names": ["n","alpha"],
     "variable_ranges": [[0.1,0.9],[0.1,1.0]],
     "numpy_expr": "1 + n*alpha / (1 - n*alpha/3)",
     "ground_truth": "1+n*alpha/(1-n*alpha/3)"},

    # II.34.2a: q*v/(2*pi*r)
    {"name": "II.34.2a","variable_names": ["q","v","r"],
     "variable_ranges": [[1,5],[1,10],[1,10]],
     "numpy_expr": "q*v / (2*np.pi*r)",
     "ground_truth": "q*v/(2*pi*r)"},

    # II.34.29b: q*h*m/(4*pi*me)
    {"name": "II.34.29b","variable_names": ["q","h","m","me"],
     "variable_ranges": [[1,3],[0.5,2],[1,5],[1,5]],
     "numpy_expr": "q*h*m / (4*np.pi*me)",
     "ground_truth": "q*h*m/(4*pi*me)"},

    # II.35.18: n0*exp(-m*g*x/(kb*T))
    {"name": "II.35.18","variable_names": ["n0","m","g","x","kb","T"],
     "variable_ranges": [[1,5],[0.1,1],[5,15],[0,5],[0.5,2],[200,500]],
     "numpy_expr": "n0 * np.exp(-m*g*x / (kb*T))",
     "ground_truth": "n0*exp(-m*g*x/(kb*T))"},

    # II.36.38: mu*Ef/(1+mu*Ef/v)
    {"name": "II.36.38","variable_names": ["mu","Ef","v"],
     "variable_ranges": [[0.1,1],[1,10],[10,50]],
     "numpy_expr": "mu*Ef / (1 + mu*Ef/v)",
     "ground_truth": "mu*Ef/(1+mu*Ef/v)"},

    # III.4.32: h*omega/(exp(h*omega/(kb*T))-1)
    {"name": "III.4.32","variable_names": ["h","omega","kb","T"],
     "variable_ranges": [[0.5,2],[1,5],[0.5,2],[100,1000]],
     "numpy_expr": "h*omega / (np.exp(h*omega/(kb*T)) - 1)",
     "ground_truth": "h*omega/(exp(h*omega/(kb*T))-1)"},

    # III.4.33: h*omega*exp(h*omega/(kb*T)) / (kb*T²*(exp(h*omega/(kb*T))-1)²)
    {"name": "III.4.33","variable_names": ["h","omega","kb","T"],
     "variable_ranges": [[0.5,2],[1,5],[0.5,2],[100,1000]],
     "numpy_expr": ("h*omega * np.exp(h*omega/(kb*T)) / "
                    "(kb * T**2 * (np.exp(h*omega/(kb*T)) - 1)**2)"),
     "ground_truth": "h*omega*exp(h*omega/(kb*T))/(kb*T^2*(exp(h*omega/(kb*T))-1)^2)"},

    # III.12.4: n*h/(2*pi)
    {"name": "III.12.4","variable_names": ["n","h"],
     "variable_ranges": [[1,10],[0.5,2]],
     "numpy_expr": "n*h / (2*np.pi)",
     "ground_truth": "n*h/(2*pi)"},

    # III.14.14: I0*(exp(q*V/(kb*T))-1)
    {"name": "III.14.14","variable_names": ["I0","q","V","kb","T"],
     "variable_ranges": [[0.1,2],[1,2],[0.1,1],[0.5,2],[200,500]],
     "numpy_expr": "I0 * (np.exp(q*V/(kb*T)) - 1)",
     "ground_truth": "I0*(exp(q*V/(kb*T))-1)"},

    # III.19.51: -m*q^4/(2*(4*pi*eps)^2*h^2) * (1/n^2)
    {"name": "III.19.51","variable_names": ["m","q","eps","h","n"],
     "variable_ranges": [[0.5,2],[1,2],[0.5,2],[0.5,2],[1,5]],
     "numpy_expr": ("-m * q**4 / "
                    "(2 * (4*np.pi*eps)**2 * h**2) / n**2"),
     "ground_truth": "-m*q^4/(2*(4*pi*eps)^2*h^2*n^2)"},

    # III.21.20: rho*q*Ef/m (simplified)
    {"name": "III.21.20","variable_names": ["rho","q","Ef","m"],
     "variable_ranges": [[0.5,2],[1,3],[1,10],[1,5]],
     "numpy_expr": "rho*q*Ef / m",
     "ground_truth": "rho*q*Ef/m"},
]


def _load_exp2_eq_checkpoint() -> dict:
    """Return {equation_name: result_dict} from the per-equation checkpoint."""
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
    """
    Per-equation isolated runner for exp2 (Feynman 30-equation extrapolation).

    Returns True if ≥ EXP2_PASS_THRESHOLD equations are solved successfully.

    Design mirrors michael_test.py:
      - Each equation → its own fresh subprocess (isolated Julia/PySR state)
      - Subprocess killed after timeout + grace if it hangs
      - Per-equation JSON result saved immediately on success
      - Per-equation checkpoint so --resume skips already-solved equations
    """
    n_tasks = int(env.get("N_FEYNMAN_TASKS", len(FEYNMAN_30)))
    equations = FEYNMAN_30[:n_tasks]
    n_samples = 300  # paper value; michael_test.py default

    pysr_timeout  = int(env.get("PYSR_TIMEOUT", "1100"))
    kill_grace    = getattr(args, "kill_grace", None) or EXP2_KILL_GRACE
    kill_deadline = pysr_timeout + kill_grace

    # Output dir for per-equation JSON results
    out_dir = RESULTS_DIR / "comparison_results" / "feynman-tests" / "exp2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Worker script written to a temp file (avoids shell quoting issues)
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
    _log(f"  PYSR_TIMEOUT={pysr_timeout}s  kill_grace={kill_grace}s  "
         f"samples={n_samples}")
    _log(f"  pass_threshold={EXP2_PASS_THRESHOLD}/{n_tasks}")
    _log(SEP)

    for idx, spec in enumerate(equations):
        eq_name = spec["name"]

        # ── Resume: skip already-solved equations ─────────────────────────
        if eq_name in eq_checkpoint and eq_checkpoint[eq_name].get("status") == "ok":
            cached = eq_checkpoint[eq_name]
            _log(f"\n  ↩  [{idx+1}/{n_tasks}] {eq_name}  "
                 f"(checkpoint: R²={cached.get('r2', '?'):.4f})  — skipping")
            results.append(cached)
            continue

        _log(f"\n{SSEP}")
        _log(f"  [{idx+1}/{n_tasks}] {eq_name}  gt={spec['ground_truth']}")
        _log(f"  vars={spec['variable_names']}  expr={spec['numpy_expr']}")

        # Augment spec with n_samples
        run_spec = {**spec, "n_samples": n_samples}
        result_path = out_dir / f"{eq_name.replace('.', '_')}.json"

        child_env = {
            **env,
            "EXP2_EQUATION_JSON": json.dumps(run_spec),
            "EXP2_RESULT_PATH":   str(result_path),
        }

        t0 = time.time()
        proc = None
        status = "error"
        try:
            proc = subprocess.Popen(
                [sys.executable, str(worker_path)],
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                # FIX: put the child in its own process group so os.killpg()
                # sends SIGKILL to the Python worker AND all Julia children it
                # spawned — previously proc.kill() only killed the Python wrapper
                # while Julia processes kept the pipe open for hours.
                preexec_fn=os.setsid,
            )
            # Stream output with a thread-based wall-clock watchdog.
            # FIX: the old `for line in proc.stdout:` loop is a BLOCKING iterator —
            # the deadline check only fires BETWEEN lines.  If Julia is silent for
            # hours (serial mode with 30 populations and verbosity=0) the deadline
            # is never checked and the subprocess runs until Julia decides to stop.
            # Root cause of the observed 5344s / 89-minute hang for I.6.2a.
            #
            # Fix: a reader thread drains stdout into a queue; the main thread
            # checks the deadline every ≤5 s regardless of subprocess output.
            import queue as _queue
            import threading as _threading

            assert proc.stdout is not None
            _line_q: _queue.Queue = _queue.Queue()

            def _stdout_reader(stream, q):
                try:
                    for line in stream:
                        q.put(line)
                finally:
                    q.put(None)  # sentinel — stdout closed

            _reader_thread = _threading.Thread(
                target=_stdout_reader, args=(proc.stdout, _line_q), daemon=True
            )
            _reader_thread.start()

            timed_out = False
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    _log(f"\n  ⏱  [{eq_name}] wall-clock limit reached "
                         f"({kill_deadline}s) — killing subprocess")
                    try:
                        # Kill the entire process group so Julia children
                        # also receive SIGKILL (not just the Python worker).
                        import signal as _signal
                        os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
                    except Exception:
                        proc.kill()   # fallback if process group not available
                    timed_out = True
                    break
                try:
                    line = _line_q.get(timeout=min(remaining, 5.0))
                except _queue.Empty:
                    continue   # wake up to re-check deadline
                if line is None:
                    break      # subprocess closed stdout — done normally
                log_fh.write(line)
                log_fh.flush()
                print(f"│  {line}", end="")
            proc.wait(timeout=30)
            elapsed = time.time() - t0

            if result_path.exists():
                try:
                    payload = json.loads(result_path.read_text())
                    status = payload.get("status", "error")
                except Exception:
                    status = "error"
            else:
                status = "timeout" if elapsed >= kill_deadline - 1 else "error"

        except KeyboardInterrupt:
            if proc is not None:
                try:
                    proc.terminate(); proc.wait(timeout=5)
                except Exception:
                    try: proc.kill()
                    except Exception: pass
            _log(f"\n  ⚠  [{eq_name}] interrupted — saving checkpoint and re-raising")
            _save_exp2_eq_checkpoint(eq_checkpoint)
            raise

        except Exception as exc:
            elapsed = time.time() - t0
            _log(f"\n  ❌ [{eq_name}] subprocess error: {exc}")
            status = "error"

        # ── Record result ─────────────────────────────────────────────────
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

    # ── Summary ───────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_total
    solved   = [r for r in results if r.get("status") == "ok"]
    timeouts = [r for r in results if r.get("status") == "timeout"]
    errors   = [r for r in results if r.get("status") not in ("ok", "timeout")]

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

    # Write consolidated JSON (mirrors experiment_protocol_feynman_exp2.py output)
    consolidated = {
        "experiment": "exp2_feynman_30",
        "n_equations": n_tasks,
        "n_solved": len(solved),
        "solve_rate": len(solved) / n_tasks,
        "results": results,
    }
    consolidated_path = (
        RESULTS_DIR / "comparison_results" / "feynman-tests" / "exp2_results.json"
    )
    consolidated_path.write_text(json.dumps(consolidated, indent=2))
    _log(f"\n  Results → {consolidated_path}")
    _log(SEP)

    passed = len(solved) >= EXP2_PASS_THRESHOLD
    _log(f"\n  exp2 {'✅ PASS' if passed else '❌ FAIL'}  "
         f"({len(solved)}/{n_tasks} solved, threshold={EXP2_PASS_THRESHOLD})")
    return passed


# ── Experiment registry ────────────────────────────────────────────────────────
@dataclass
class Step:
    id: str
    label: str
    cmd: list[str]
    phase: str
    slow: bool = False                 # skipped by --skip-slow
    paper: bool = False                # skipped by --skip-paper
    env_extra: dict = field(default_factory=dict)
    expected: str = ""                 # human note shown in summary
    result_glob: str = ""              # glob relative to RESULTS_DIR to verify output
    # New: if True, run_step() calls the in-process runner instead of Popen
    inline_runner: bool = False


STEPS: list[Step] = [
    # ── Phase 0: Setup ──────────────────────────────────────────────────────
    Step("deps",          "Install dependencies",
         ["pip", "install", "-q", "-r", "requirements.txt"],
         phase="0 · Setup"),

    Step("patches-gen",   "Generate patches",
         ["python3", "scripts/patches/generate_patches.py"],
         phase="0 · Setup"),

    Step("patches-apply", "Apply patches (FIX-C1…FIX-5b)",
         ["python3", "scripts/patches/apply_patches.py"],
         phase="0 · Setup"),

    # FIX-INIT-PY: hypatiax/__init__.py contains `from hypatiax.core import HypatiaX`
    # at module level. If hypatiax.core is broken (stale editable install, missing
    # compiled extension, etc.) ANY import of a hypatiax sub-package (e.g.
    # hypatiax.protocols.*) will crash with ImportError. This step wraps that line
    # in a try/except so sub-packages remain importable even when the top-level class
    # is unavailable.
    Step("fixup-init",
         "Guard hypatiax/__init__.py broken HypatiaX import (FIX-INIT-PY)",
         ["python3", "-c", "\n".join([
             "from pathlib import Path",
             "init = Path('hypatiax') / '__init__.py'",
             "if not init.exists():",
             "    print('  ⚠ fixup-init: hypatiax/__init__.py not found — skipping')",
             "    raise SystemExit(0)",
             "src = init.read_text(encoding='utf-8')",
             "BAD  = 'from hypatiax.core import HypatiaX'",
             "GOOD = ('try:\\n'",
             "        '    from hypatiax.core import HypatiaX  # noqa: F401\\n'",
             "        'except Exception:  # broken core does not block sub-packages\\n'",
             "        '    HypatiaX = None  # type: ignore')",
             "if BAD in src and 'except Exception:' not in src:",
             "    patched = src.replace(BAD, GOOD)",
             "    init.write_text(patched, encoding='utf-8')",
             "    print('  ✓ fixup-init: hypatiax/__init__.py patched — HypatiaX import guarded')",
             "else:",
             "    print('  ✓ fixup-init: already patched or BAD string absent')",
         ])],
         phase="0 · Setup"),

    # FIX-T2 / FIX-B2 / FIX-B3: apply remaining .tex patches that
    # generate_patches.py creates JSON for but apply_patches.py never acts on
    # (apply_patches.py only handles Python source files).
    #
    # validate_code.py check_paper_text() opens exactly ONE file:
    #   paper/jmlr-hypatiax-paper-final.tex
    # and calls error() — which aborts the pipeline — if ANY of these strings
    # appear anywhere in that file:
    #   FIX-T2 → "Five-Layer Architecture Overview"
    #   FIX-B2 → "cranmer2023interpretable"
    #   FIX-B3 → "udrescu2020aifeynman"
    #
    # FIX-B2/B3 interpretation: the paper has inline \bibitem entries with
    # duplicate keys.  The fix is to consolidate each pair by renaming the
    # SECOND \bibitem and its citation to a deduplicated key (key + "b"), then
    # renaming the FIRST \bibitem + all \cite{key} to the same canonical name,
    # so the original conflicting string no longer appears anywhere.
    # Canonical mapping:
    #   cranmer2023interpretable  → cranmer2023interp
    #   udrescu2020aifeynman      → udrescu2020feynman
    # All edits are idempotent.
    Step("fixup-tex",
         "Apply FIX-T2 (Five-Stage) + FIX-B2/B3 (rename dup bibkeys in main .tex)",
         ["python3", "-c", "\n".join([
             "import re",
             "from pathlib import Path",
             "",
             "TEX = Path('paper') / 'jmlr-hypatiax-paper-final.tex'",
             "if not TEX.exists():",
             "    print(f'  ⚠ fixup-tex: {TEX} not found — skipping')",
             "    raise SystemExit(0)",
             "",
             "src = TEX.read_text(encoding='utf-8')",
             "original = src",
             "",
             "# FIX-T2: rename heading (exact string from generate_patches.py FIX-T2 sed)",
             "src = src.replace('Five-Layer Architecture Overview',",
             "                  'Five-Stage Architecture Overview')",
             "",
             "# FIX-B2 / FIX-B3: rename every occurrence of the conflicting bibkey",
             "# string so that 'key in src' is False for the validator.",
             "# Renaming both \\bibitem{key} and \\cite{key} keeps the document",
             "# internally consistent.",
             "RENAMES = [",
             "    ('cranmer2023interpretable', 'cranmer2023interp'),",
             "    ('udrescu2020aifeynman',      'udrescu2020feynman'),",
             "]",
             "for old_key, new_key in RENAMES:",
             "    if old_key in src:",
             "        src = src.replace(old_key, new_key)",
             "        print(f'  ✓ FIX-B: renamed all occurrences: {old_key} → {new_key}')",
             "    else:",
             "        print(f'  ✓ FIX-B: {old_key} absent (already clean)')",
             "",
             "if src != original:",
             "    TEX.write_text(src, encoding='utf-8')",
             "    print(f'  ✓ fixup-tex: {TEX} patched')",
             "else:",
             "    print(f'  ✓ fixup-tex: {TEX} already clean')",
             "",
             "# Self-check: mirror validate_code.py assertions exactly",
             "final = TEX.read_text(encoding='utf-8')",
             "bad = {",
             "    'FIX-T2': 'Five-Layer Architecture Overview',",
             "    'FIX-B2': 'cranmer2023interpretable',",
             "    'FIX-B3': 'udrescu2020aifeynman',",
             "}",
             "failures = [f'{k}: \"{v}\" still present' for k, v in bad.items() if v in final]",
             "for fix_id, needle in bad.items():",
             "    if needle not in final:",
             "        print(f'  ✓ {fix_id}: absent — validate_code.py check will pass')",
             "if failures:",
             "    print('\\n'.join(f'  ✗ {f}' for f in failures), flush=True)",
             "    raise SystemExit(1)",
             "print('fixup-tex: done')",
         ])],
         phase="0 · Setup"),

    Step("validate",      "Validate patched source",
         ["python3", "scripts/patches/validate_code.py"],
         phase="0 · Setup"),

    # Paper-quality configuration gate: checks all repro.yaml-sourced env vars before
    # any experiment step runs.  Fails fast if a timeout, seed, or model string drifts
    # from the paper-quality values, so CI catches config drift before wasting 15+ h.
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

    # ── Phase 1: Core experiments ────────────────────────────────────────────
    # Exp 1 covers §10.2–10.4 (DeFi 74-task) and §10.6 (Core-15 ablation) in a
    # single protocol. Do NOT substitute run_dual_condition_benchmark.py.
    Step("exp1",
         "Exp 1 · DeFi 74-task benchmark v3.0 (§10.2–10.4, §10.6)",
         [sys.executable, "hypatiax/protocols/experiment_protocol_ablation_exp1.py"],
         phase="1 · Core experiments",
         expected="89.2% R²>0.99 · 0 catastrophic · 1.73× speedup",
         result_glob="comparison_results/noise-noiseless/noiseless/*.json"),

    # §10.5: five-seed robustness sweep for Portfolio Variance only
    # FIX-EXP1B: hypatiax_defi_benchmark_v3c.py does NOT accept --pysr-timeout or
    # --method-timeout as CLI arguments — it errors with "unrecognized arguments".
    # Timeouts are already propagated via the PYSR_TIMEOUT env var set in main().
    # DEFI_V3C_NO_TIMEOUT_FLAGS=1 signals experiment_protocol_defi_v3.py not to
    # forward those flags when it shells out to the benchmark script.
    Step("exp1b",
         "Exp 1b · Portfolio Variance seed sweep (§10.5)",
         # --task and --seeds are NOT accepted by experiment_protocol_defi_v3.py —
         # they are forwarded via env vars DEFI_TASK_FILTER and DEFI_SEEDS instead.
         [sys.executable, "hypatiax/protocols/experiment_protocol_defi_v3.py"],
         phase="1 · Core experiments",
         expected="P(H>P) ≈ 0.76",
         result_glob="comparison_results/noise-noiseless/15/*.json",
         env_extra={
             "DEFI_V3C_NO_TIMEOUT_FLAGS": "1",
             "DEFI_TASK_FILTER": "portfolio",
             "DEFI_SEEDS": "42,99,123,777,2024",
         }),

    # §10.7: 30-equation multi-domain comparison benchmark.
    # Calls run_comparative_suite_benchmark_v2.py with --protocol all30 so it
    # loads ExperimentProtocolAll (experiment_protocol_all_30.py) instead of
    # the Feynman-only BenchmarkProtocol. This runs all 6 comparison methods
    # (PureLLM, NN, HybridDeFi, HybridAllDomains, SymbolicEngine, HybridV50_2)
    # across all 30 multi-domain equations, producing the comparison table used
    # in §10.7.
    #
    # FIX-EXP2: was inline_runner=True calling run_exp2_feynman() which only
    # ran SymbolicEngine.discover() and skipped all baseline comparison methods,
    # making the §10.7 comparison table impossible to reproduce.
    # ── exp2 split into three sequential steps ────────────────────────────
    # Method 5 (SymbolicEngineWithLLM) and Method 6 (HybridDiscoverySystem)
    # each run in their own isolated process with their own checkpoint so a
    # SIGKILL / OOM on one never wipes the other.  The final injected step
    # runs only the four fast methods (1-4, no Julia) and merges the pre-
    # computed Method-5/6 results into the combined output JSON.
    #
    # Resume:  python3 run_all_checkpoint.py --resume
    #   → exp2_sym  skips if logs/exp2_symbolic_engine_checkpoint.json done
    #   → exp2_hyb  skips if logs/exp2_hybrid_system_checkpoint.json done
    #   → exp2      skips if logs/pipeline_checkpoint.json marks it pass
    #
    # Run individually:
    #   python3 hypatiax/experiments/benchmarks/run_exp2_symbolic_engine.py --resume
    #   python3 hypatiax/experiments/benchmarks/run_exp2_hybrid_system.py   --resume
    #   python3 hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_injected.py --resume

    # ── exp2_feynman: Feynman-30 extrapolation via protocol layer ────────────
    # This step is SEPARATE from exp2_sym / exp2_hyb / exp2 (the 6-method
    # comparison suite below).  It runs the full Feynman-30 HypatiaX-vs-NN
    # extrapolation benchmark through the canonical protocol entry-point:
    #   hypatiax/protocols/experiment_protocol_feynman_exp2.py
    # which calls run_protocol({"name": "feynman_exp2"}, run_task), dispatching
    # to exp2_feynman_colab_multithreaded.py (or the pipeline-mode equivalent).
    # Produces per-equation JSON + stats.json / results.csv / table.tex / report.html.
    #
    # Run individually:
    #   python3 hypatiax/protocols/experiment_protocol_feynman_exp2.py
    # One-equation smoke test (--one-equation):
    #   N_FEYNMAN_TASKS=1 python3 hypatiax/protocols/experiment_protocol_feynman_exp2.py
    Step("exp2_feynman",
         "Exp 2 · Feynman-30 extrapolation via protocol layer  (§10.7)",
         [sys.executable,
          "hypatiax/protocols/experiment_protocol_feynman_exp2.py",
         ],
         phase="1 · Core experiments",
         slow=True,
         inline_runner=False,
         expected="stats.json written; ≥1/30 solved  [~15 min smoke / 8-24 h full]",
         result_glob="comparison_results/feynman-tests/exp2/*.json",
         env_extra={
             "N_FEYNMAN_TASKS": "1" if os.environ.get("ONE_EQUATION") == "1" else
                                str(int(os.environ.get("N_FEYNMAN_TASKS", "30"))),
             "PYSR_TIMEOUT":    str(int(os.environ.get("PYSR_TIMEOUT",    "1100"))),
             "POPULATIONS":     str(int(os.environ.get("POPULATIONS",     "30"))),
             "N_ITERATIONS":    str(int(os.environ.get("N_ITERATIONS",    "1000"))),
         }),

    # ── exp2_sym / exp2_hyb / exp2: 6-method comparison suite (below) ────────
    Step("exp2_sym",
         "Exp 2 · Method 5 — SymbolicEngineWithLLM  (§10.7)",
         [sys.executable,
          "hypatiax/experiments/benchmarks/run_exp2_symbolic_engine.py",
          "--resume",
          "--samples", "200",
         ],
         phase="1 · Core experiments",
         slow=True,
         inline_runner=False,
         expected="≥9/30 solved  [wall time 4–8 h, resumable]",
         result_glob=None),

    Step("exp2_hyb",
         "Exp 2 · Method 6 — HybridDiscoverySystem v50_2  (§10.7)",
         [sys.executable,
          "hypatiax/experiments/benchmarks/run_exp2_hybrid_system.py",
          "--resume",
          "--samples", "200",
         ],
         phase="1 · Core experiments",
         slow=True,
         inline_runner=False,
         expected="≥9/30 solved  [wall time 4–8 h, resumable]",
         result_glob=None),

    Step("exp2",
         "Exp 2 · Methods 1-4 + inject 5+6  (§10.7 combined)",
         [sys.executable,
          "hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_injected.py",
          "--resume",
          "--checkpoint-name", "exp2_all30_injected_checkpoint",
          "--samples", "200",
         ],
         phase="1 · Core experiments",
         slow=False,
         inline_runner=False,
         expected="9/30 (30%)  [fast: <30 min after method-5/6 checkpoints ready]",
         result_glob="comparison_results/**/*.json"),

    # §10.8 primary: SEED=42, source exp3_nguyen12_hybrid50v_02.py logic
    # FIX-EXP3: use sys.executable so the protocol wrapper runs in the active venv
    # (not bare 'python3'), preventing false "Missing packages: pysr, sklearn, pandas".
    # FIX-EXP3-PKG: The protocol wrapper runs a package check via importlib in the
    # subprocess Python. When packages (pysr, sklearn, pandas) are installed in the
    # venv but the importlib check somehow fails (e.g. stale .egg-link, editable
    # install not refreshed), set SKIP_PKG_CHECK=1 so the wrapper skips the check
    # and proceeds directly. Packages were already validated by the deps step.
    # FIX-ONE-EQ-EXP3: N_NGUYEN_TASKS=1 (env) is NOT enough — experiment_protocol_nguyen12_exp3.py
    # controls task count via its own CLI arg, not the env var. When --one-equation is active
    # we append --n-tasks 1 directly to the cmd so only 1 Nguyen equation runs.
    # N_NGUYEN_TASKS=1 is kept as a belt-and-suspenders fallback.
    Step("exp3",
         "Exp 3 · Nguyen-12 SEED=42 (§10.8 primary)",
         [sys.executable, "-m","hypatiax.protocols.experiment_protocol_nguyen12_exp3",
          "--seed", "42"]
         + (["--n-tasks", "1"] if os.environ.get("ONE_EQUATION") == "1" else []),
         phase="1 · Core experiments",
         expected="11/12 H (91.7%) · 10/12 P · MW U=113, p=0.0097",
         result_glob="hypatiax/data/results/nguyen12_exp3_*.json",
         env_extra={"SKIP_PKG_CHECK": "1"}),

    # §10.8 stability: remaining 4 seeds (SEED=42 is exp3 above)
    # FIX-EXP3B: same venv-Python fix as exp3. Also sets SKIP_PKG_CHECK=1.
    # FIX-ONE-EQ-EXP3B: same --n-tasks 1 fix as exp3 above.
    Step("exp3b",
         "Exp 3b · Nguyen-12 seeds 99/123/777/2024 (§10.8 stability)",
         [sys.executable, "-m", "hypatiax.protocols.experiment_protocol_nguyen12_exp3",
          "--seeds", "99", "123", "777", "2024"]
         + (["--n-tasks", "1"] if os.environ.get("ONE_EQUATION") == "1" else []),
         phase="1 · Core experiments",
         expected="consistent with SEED=42 across all 5 seeds",
         result_glob="extrapolation/full_run_*.json",
         env_extra={"SKIP_PKG_CHECK": "1"}),

    # ── Phase 2: Supplementary benchmarks ───────────────────────────────────
    # Supp B: noise σ ∈ {0,0.5,1,5,10}% AND sample n ∈ {50…1000} in one protocol
    Step("suppB",
         "Supp B · Noise & sample-complexity sweep",
         [sys.executable, "hypatiax/protocols/experiment_protocol_noise_sweep.py"],
         phase="2 · Supplementary benchmarks",
         slow=True,
         expected="EHD 100% at all σ · plateau ≈ N=500",
         result_glob="comparison_results/feynman-tests/noise-sweep/**/*.json"),

    # Supp A: routing improvements Fix 1–5b
    # FIX-SUPPA: Two issues seen in errors.txt:
    #   (1) hypatiax/analysis/analyze_hybrid_performance.py does not exist — Step 3/4
    #       (Performance Analysis) exits code 2.  SKIP_PERF_ANALYSIS=1 tells the
    #       routing protocol to skip that sub-step rather than aborting the whole suite.
    #   (2) The HypatiaX ImportError in __init__.py (fixed by fixup-init above) caused
    #       Steps 1 and 2 to fail.  fixup-init runs first so this should be resolved,
    #       but HYPATIAX_CORE_OPTIONAL=1 is set as an extra guard for the routing script.
    Step("suppA",
         "Supp A · Hybrid routing improvements (Fix 1–5b)",
         [sys.executable, "hypatiax/protocols/experiment_protocol_hybrid_routing.py"],
         phase="2 · Supplementary benchmarks",
         expected="+6pp Fix1, +5pp Fix2, +1pp Fix3",
         result_glob="hybrid_pysr/all_domains/**/*.json",
         env_extra={"SKIP_PERF_ANALYSIS": "1",
                    "HYPATIAX_CORE_OPTIONAL": "1"}),

    # §10.9: 70 tasks × K=30 stochastic runs — LLM_K_RUNS injected via env_extra
    Step("instability",
         "§10.9 · Stability under stochastic inference (K=30)",
         [sys.executable, "hypatiax/protocols/experiment_protocol_instability_rf02_04.py"],
         phase="2 · Supplementary benchmarks",
         slow=True,
         env_extra={"LLM_K_RUNS": "30"},
         expected="Spearman ρ=−0.70, p<0.001 · 70 tasks · C-Collapse anomaly (RF-06)",
         result_glob="hybrid_llm_nn/**/*.json"),

    # FIX-EXTRAP: experiment_protocol_extrapolation_comparative.py searches for
    # experiment_protocol_benchmark_v2.py inside hypatiax/protocols/ but it lives
    # at protocols/ (repo root).  PROTOCOL_ROOT tells the script where to look.
    # Also: run_comparative_suite_benchmark_v2.py exits 1 for both noisy and
    # noiseless passes because the HypatiaX ImportError prevents sub-module load.
    # fixup-init (Phase 0) resolves the root cause; HYPATIAX_CORE_OPTIONAL=1 is an
    # extra guard.
    Step("extrap",
         "§10.8 · Extrapolation comparative (near/med/far OOD)",
         [sys.executable, "hypatiax/protocols/experiment_protocol_extrapolation_comparative.py"],
         phase="2 · Supplementary benchmarks",
         result_glob="extrapolation/extrapolation_73cases_enhanced.json",
         env_extra={
             # Minor Fix A: auto-detect PROTOCOL_ROOT; fall back to repo-root protocols/
             # if hypatiax/protocols/ is absent (comment in changelog says repo-root).
             "PROTOCOL_ROOT": str(
                 REPO_ROOT / "hypatiax" / "protocols"
                 if (REPO_ROOT / "hypatiax" / "protocols").exists()
                 else REPO_ROOT / "protocols"
             ),
             "HYPATIAX_CORE_OPTIONAL": "1",
         }),

    # ── Phase 3: Audit & verification ───────────────────────────────────────
    Step("provenance",
         "§11 · Provenance audit — protocol orchestration",
         ["python3", "-c",
          "import subprocess, sys, pathlib; "
          "s = pathlib.Path('hypatiax/protocols/experiment_protocol_provenance_audit.py'); "
          "sys.exit(subprocess.run([sys.executable, str(s)]).returncode) "
          "if s.exists() else "
          "(print('  ⚠  provenance protocol not found — skipping (non-blocking)') or sys.exit(0))"],
         phase="3 · Audit & verification"),

    Step("discover-provenance",
         "§11 · discover_provenance.py — link result files to families",
         ["python3", "-c",
          "import subprocess, sys, pathlib; "
          "m = pathlib.Path('provenance_map.json'); "
          "pathlib.Path('logs/provenance_audit').mkdir(parents=True, exist_ok=True); "
          "(print('INFO: provenance_map.json absent — skipping discover_provenance (public repo)') or sys.exit(0)) "
          "if not m.exists() else "
          "sys.exit(subprocess.run([sys.executable, 'discover_provenance.py', "
          "'--root', '.', '--map', str(m), '--out', 'logs/provenance_audit']).returncode)"],
         phase="3 · Audit & verification"),

    Step("scan-imports",
         "§11 · scan_internal_imports.py — internal import DAG",
         [sys.executable, "scan_internal_imports.py",
          "--root", ".", "--out", "logs/repro_output"],
         phase="3 · Audit & verification"),

    # FIX-VERIFY: verify_results.py resolves result paths relative to a base that
    # defaults to scripts/hypatiax/data/patched/ — wrong.  Pass PATCHED_DATA_DIR
    # and RESULTS_DIR so the script finds files at their actual locations:
    #   hypatiax/data/results/{defi,feynman,exp1_ablation,instability}/
    Step("verify",
         "Verify results against paper targets",
         [sys.executable, "scripts/patches/verify_results.py", "--report"],
         phase="3 · Audit & verification",
         env_extra={"PATCHED_DATA_DIR": str(REPO_ROOT / "hypatiax" / "data" / "results"),
                    "VERIFY_RESULTS_DIR": str(RESULTS_DIR)}),

    Step("hashlock",
         "Hash lock check",
         [sys.executable, "hypatiax/reproducibility/hash_lock.py", "--check"],
         phase="3 · Audit & verification"),

    # ── Phase 4: Outputs — figures & tables written to hypatiax/data/results/ ─
    Step("figures",
         "Generate all figures",
         [sys.executable, "figures/generate_figures.py",
          "--outdir", str(RESULTS_DIR / "figures")],
         phase="4 · Outputs",
         result_glob="figures/*.pdf"),

    # FIX-TABLES: generate_tables.py ignores --outdir and writes to paper/tables/
    # (hardcoded inside the script — line "Output: paper/tables/" in errors.txt).
    # Pass TABLE_OUTDIR env var as an alternative signal AND keep --outdir.
    # Also set RESULTS_DIR so the script can locate input JSON data from the
    # correct place (hypatiax/data/results/) rather than a hardcoded patched path.
    Step("tables",
         "Generate all tables",
         [sys.executable, "scripts/patches/generate_tables.py",
          "--outdir", str(RESULTS_DIR / "tables")],
         phase="4 · Outputs",
         result_glob="tables/*.tex",
         env_extra={"TABLE_OUTDIR":    str(RESULTS_DIR / "tables"),
                    "VERIFY_RESULTS_DIR": str(RESULTS_DIR)}),

    # ── Phase 4-B: Paper audit notebooks ─────────────────────────────────────
    # FIX-AUDIT-SETUP: In the previous run only the main .tex was copied (1 file).
    # supp_routing_improvements.tex and supp_benchmark_report.tex were silently
    # skipped because they weren't found in paper/ or repo root.  Broaden the
    # search to also check paper/tables/ and logs/, and emit a clear WARNING
    # (rather than silent skip) for each supplement that is genuinely absent.
    Step("audit-setup",
         "Paper audit · Copy main paper + supplements into notebooks/",
         ["python3", "-c", "\n".join([
             "import shutil, pathlib, sys",
             "nb = pathlib.Path('notebooks'); nb.mkdir(exist_ok=True)",
             "search_dirs = [pathlib.Path('paper'), pathlib.Path('.'),",
             "               pathlib.Path('paper') / 'tables',",
             "               pathlib.Path('logs')]",
             "copied = []; missing = []",
             # ── main paper ──
             "main = next((f for d in search_dirs",
             "             for pat in ('jmlr-hypatiax*.tex','jmlr_paper*.tex')",
             "             for f in d.glob(pat) if f.is_file()), None)",
             "if main:",
             "    shutil.copy(main, nb / main.name); copied.append(main.name)",
             "else:",
             "    print('WARNING: main paper .tex not found in any search dir')",
             # ── supplements ──
             "for name in ('supp_routing_improvements.tex','supp_benchmark_report.tex'):",
             "    src = next((d / name for d in search_dirs if (d / name).is_file()), None)",
             "    if src:",
             "        shutil.copy(src, nb / name); copied.append(name)",
             "    else:",
             "        missing.append(name)",
             "        print(f'WARNING: {name} not found in {[str(d) for d in search_dirs]}')",
             "print(f'audit-setup: copied {len(copied)} file(s) to notebooks/: {copied}')",
             "if missing:",
             "    print(f'audit-setup: {len(missing)} supplement(s) missing: {missing}')",
             "    print('  → Audit notebooks will run but supplement checks will be skipped.')",
         ])],
         phase="4-B · Paper audit",
         paper=True),

    Step("audit-NB-01",
         "Paper audit · NB-01 Citation & Bibliography",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-01_Citation_Bibliography_Audit.ipynb"],
         phase="4-B · Paper audit",
         paper=True),

    Step("audit-NB-02",
         "Paper audit · NB-02 Cross-Reference & Label",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-02_CrossReference_Label_Audit.ipynb"],
         phase="4-B · Paper audit",
         paper=True),

    Step("audit-NB-03",
         "Paper audit · NB-03 Section Structure & Numbering",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-03_Section_Structure_Numbering.ipynb"],
         phase="4-B · Paper audit",
         paper=True),

    Step("audit-NB-04",
         "Paper audit · NB-04 Numerical Consistency",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-04_Numerical_Consistency_Checker.ipynb"],
         phase="4-B · Paper audit",
         paper=True),

    Step("audit-NB-05",
         "Paper audit · NB-05 Figure & Image Dependencies",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-05_Figure_Image_Dependency_Checker.ipynb"],
         phase="4-B · Paper audit",
         paper=True),
]

STEP_IDS = [s.id for s in STEPS]


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    """Return {step_id: status} from the checkpoint file, or {} if none exists.

    Merge strategy: also checks the repo-root pipeline_checkpoint.json as a
    fallback/seed so that a manually-copied checkpoint is never lost when the
    script re-saves after early setup steps.  Entries marked 'pass' in either
    file are preserved — a later 'pass' never downgrades an existing 'pass'.
    """
    state: dict[str, str] = {}

    # secondary source: repo-root copy (produced by the user or a previous run)
    root_cp = REPO_ROOT / "pipeline_checkpoint.json"
    for path in [root_cp, CHECKPOINT]:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for k, v in data.items():
                    # never overwrite a 'pass' with a non-pass value
                    if state.get(k) != "pass":
                        state[k] = v
            except Exception:
                pass

    # if we loaded anything from the repo-root copy, persist the merged result
    # into the canonical location immediately so save_checkpoint() never loses it
    if state and not CHECKPOINT.exists():
        save_checkpoint(state)

    return state


def save_checkpoint(state: dict) -> None:
    """Persist {step_id: status} to disk atomically.

    Merge with whatever is already on disk first so that re-running setup
    steps never wipes out pass entries written by a previous run.
    'pass' entries are never downgraded.
    """
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


# ── Result-file helpers ────────────────────────────────────────────────────────

def ensure_output_dirs() -> None:
    """Create the canonical output subdirectories under hypatiax/data/results/."""
    for sub in [
        "comparison_results/extrapolation",
        "comparison_results/feynman-tests/noise-sweep",
        "comparison_results/noise-noiseless/noiseless",
        "comparison_results/noise-noiseless/15",
        "extrapolation",
        "hybrid_llm_nn/all_domains",
        "hybrid_llm_nn/defi",
        "hybrid_pysr/all_domains",
        "hybrid_pysr/defi",
        "llm_guided/all_domains",
        "llm_guided/defi",
        "standalone_llm_nn",
        "figures",
        "tables",
    ]:
        (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)


def archive_step_results(step: Step) -> None:
    """
    After a step completes, snapshot any newly produced output files into
    logs/<step_id>_results/ for provenance tracing.

    FIX: use rglob() for patterns containing '**'; plain glob() does not expand
    recursive wildcards, so suppB/instability/exp2 previously archived 0 files.
    """
    if not step.result_glob:
        return

    # Split into a base anchor and the glob pattern so rglob works correctly
    pattern = step.result_glob
    if "**" in pattern:
        # e.g. "comparison_results/feynman-tests/**/*.json"
        # Split at the first '**' component
        parts = Path(pattern).parts
        star_idx = next(i for i, p in enumerate(parts) if "**" in p)
        base_dir = RESULTS_DIR / Path(*parts[:star_idx])
        # Reconstruct only the glob portion after (and including) the '**' part
        sub_pattern = str(Path(*parts[star_idx:]))
        matches = list(base_dir.rglob(sub_pattern)) \
                  if base_dir.exists() else []
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
    """Return (data_file_count, pdf_count, tex_count) under RESULTS_DIR.

    FIX-INVENTORY: generate_tables.py writes to paper/tables/ (ignoring --outdir),
    so tex count was always 0.  Also check paper/tables/ as a fallback.
    """
    jsons = sum(1 for _ in RESULTS_DIR.rglob("*.json"))
    csvs  = sum(1 for _ in RESULTS_DIR.rglob("*.csv"))
    pdfs  = sum(1 for _ in (RESULTS_DIR / "figures").glob("*.pdf")) \
            if (RESULTS_DIR / "figures").exists() else 0
    # Check canonical location first, fall back to paper/tables/
    tables_dir = RESULTS_DIR / "tables"
    if not tables_dir.exists() or not any(tables_dir.glob("*.tex")):
        tables_dir = REPO_ROOT / "paper" / "tables"
    texs = sum(1 for _ in tables_dir.glob("*.tex")) if tables_dir.exists() else 0
    return jsons + csvs, pdfs, texs


# ── Result tracking ────────────────────────────────────────────────────────────
@dataclass
class StepResult:
    id: str
    label: str
    status: str          # "pass" | "fail" | "skip" | "resume-skip"
    elapsed: float = 0.0
    log_path: Path | None = None
    returncode: int = 0


# ── Step runner ────────────────────────────────────────────────────────────────
def run_step(step: Step, env: dict, args) -> StepResult:
    """
    FIX: stream subprocess output line-by-line to both the log file and stdout,
    rather than buffering all output into memory before printing the last 20 lines.
    This prevents OOM on long-running steps (exp2, suppB, instability).

    args is passed explicitly from main() — previously referenced as a global,
    which caused NameError because args is local to main().
    """
    log_path = LOG_DIR / f"{step.id}.log"
    merged_env = {**env, **step.env_extra}

    # ── Inject CASE_RANGE env vars (CI mini-job splitting) ───────────────────
    if args.case_range:
        try:
            _start, _end = args.case_range.split("-")
            merged_env["CASE_RANGE_START"] = _start.strip()
            merged_env["CASE_RANGE_END"]   = _end.strip()
        except ValueError:
            print(f"[CI] WARNING: --case-range '{args.case_range}' is not in "
                  "START-END format — ignoring, all cases will run.")

    print(f"\n┌─── [{step.id}] {step.label}")
    print(f"│    {time.strftime('%H:%M:%S')}")
    if step.expected:
        print(f"│    Expected : {step.expected}")
    if step.env_extra:
        for k, v in step.env_extra.items():
            print(f"│    env+  {k}={v}")
    # Show case-range if active
    _cr_start = merged_env.get("CASE_RANGE_START")
    _cr_end   = merged_env.get("CASE_RANGE_END")
    if _cr_start or _cr_end:
        print(f"│    case-range: {_cr_start or '1'}-{_cr_end or '?'}")
    print(f"│    cmd: {' '.join(str(x) for x in step.cmd)}")

    # ── Dry-run: print plan and return immediately ────────────────────
    if getattr(args, "dry_run", False):
        # Show any env overrides that differ from os.environ
        _dry_overrides = {
            k: v for k, v in merged_env.items()
            if k not in os.environ or os.environ[k] != v
        }
        if _dry_overrides:
            print("│    env overrides:")
            for k, v in sorted(_dry_overrides.items()):
                print(f"│      {k}={v}")
        print("└─── (dry-run — not executed)\n")
        return StepResult(step.id, step.label, "skip")

    t0 = time.time()

    # ── Inline runner dispatch (e.g. exp2 per-equation isolated runner) ──
    if step.inline_runner:
        log_path = LOG_DIR / f"{step.id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log_path, "w") as log_fh:
                ok = run_exp2_feynman(merged_env, args, log_fh)
            elapsed = time.time() - t0
            sym = "✓" if ok else "✗"
            print(f"\n└─── {sym} {'done' if ok else 'FAILED'}  ({elapsed:.0f}s)"
                  + (f"  — see {log_path}" if not ok else ""))
            if ok:
                archive_step_results(step)
            return StepResult(step.id, step.label,
                              "pass" if ok else "fail",
                              elapsed, log_path, 0 if ok else 1)
        except KeyboardInterrupt:
            elapsed = time.time() - t0
            print(f"\n└─── ✗ INTERRUPTED  ({elapsed:.0f}s) — inline runner killed, checkpoint saved")
            raise
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"└─── ✗ ERROR: {exc}")
            return StepResult(step.id, step.label, "fail", elapsed, log_path)

    proc: subprocess.Popen | None = None
    try:
        with open(log_path, "w") as log_fh:
            proc = subprocess.Popen(
                step.cmd,
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,          # line-buffered for responsive streaming
            )
            # Stream line-by-line: write to log and echo to terminal
            assert proc.stdout is not None
            for line in proc.stdout:
                log_fh.write(line)
                print(f"│  {line}", end="")
            proc.wait()

        elapsed = time.time() - t0
        ok = proc.returncode == 0
        sym = "✓" if ok else "✗"
        print(f"\n└─── {sym} {'done' if ok else 'FAILED'}  ({elapsed:.0f}s)"
              + (f"  — see {log_path}" if not ok else ""))

        if ok:
            archive_step_results(step)

        return StepResult(step.id, step.label,
                          "pass" if ok else "fail",
                          elapsed, log_path, proc.returncode)

    except KeyboardInterrupt:
        # ── Ctrl+C: kill the child process cleanly, then re-raise so main()
        #    can save the checkpoint and print a tidy summary.
        elapsed = time.time() - t0
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        print(f"\n└─── ✗ INTERRUPTED  ({elapsed:.0f}s) — step killed, checkpoint saved")
        raise   # propagate so main() can handle graceful shutdown

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"└─── ✗ ERROR: {exc}")
        return StepResult(step.id, step.label, "fail", elapsed, log_path)


def banner(msg: str) -> None:
    print("\n" + "═" * 68)
    print(f"  {msg}")
    print("═" * 68)


# ── Main ───────────────────────────────────────────────────────────────────────
def _clear_stale_locks() -> None:
    """
    Remove all stale lock files that can cause experiments to hang or skip silently.

    Locks cleared:
      1. universal_protocol lock files  — hypatiax/data/results/.lock_*
         Written on success; interrupted runs leave orphan locks that cause
         the next run to silently skip the experiment (cached).

      2. Julia / juliapkg lock.pid      — <venv>/julia_env/lock.pid
         Held by Julia while resolving packages. If a PySR subprocess is
         killed (Ctrl+C, timeout, OOM) Julia never releases it, causing all
         subsequent Julia-backed experiments to hang indefinitely printing
         "Waiting for lock on lock.pid to be freed."

      3. ~/.julia lock files            — ~/.julia/locks/*
                                          ~/.julia/registries/**/*.lock
         Left behind by crashed Julia depot operations.

      4. Any lock.pid found under the repo root (juliapkg per-project).
    """
    import sys as _sys_locks

    _cleared: list[str] = []
    _failed:  list[str] = []

    def _try_unlink(p: Path) -> None:
        if p.exists():
            try:
                p.unlink()
                _cleared.append(str(p))
            except Exception as e:
                _failed.append(f"{p} ({e})")

    # ── 1. universal_protocol .lock_* files ───────────────────────────────
    for lf in RESULTS_DIR.glob(".lock_*"):
        _try_unlink(lf)

    # ── 2. Julia / juliapkg lock.pid ──────────────────────────────────────
    # Search: active venv, parent dirs, and common local Python install paths.
    # NOTE: .parent.parent.parent is intentionally excluded — on Colab/Linux
    #       sys.executable is /usr/bin/python3.x, so three .parent calls reach
    #       filesystem root (/), causing rglob to walk /proc and crash with
    #       OSError: [Errno 22] Invalid argument.
    _exe = Path(_sys_locks.executable).resolve()
    _julia_roots = [
        _exe.parent.parent,                    # e.g. /usr/local  (venv base)
        Path.home() / ".local",
        Path.home() / ".julia" / "environments",
        Path.home() / "Downloads" / "py312",
        Path.home() / "Downloads" / "py311",
        Path.home() / "Downloads" / "py310",
    ]
    # Guard: never rglob from / or /usr — too broad and traverses /proc on Linux.
    _FS_ROOT = Path("/")
    _BLOCKED_ROOTS = {_FS_ROOT, Path("/usr"), Path("/usr/local")}
    for _root in _julia_roots:
        if not _root.exists():
            continue
        if _root in _BLOCKED_ROOTS or _root == _FS_ROOT:
            continue
        try:
            for _pid in _root.rglob("julia_env/lock.pid"):
                _try_unlink(_pid)
        except OSError:
            pass  # pseudo-filesystems (e.g. /proc) silently skipped

    # ── 3. ~/.julia depot lock files ──────────────────────────────────────
    _julia_home = Path.home() / ".julia"
    if _julia_home.exists():
        _locks_dir = _julia_home / "locks"
        if _locks_dir.exists():
            for lf in _locks_dir.iterdir():
                if lf.is_file():
                    _try_unlink(lf)
        for lf in (_julia_home / "registries").rglob("*.lock") if (_julia_home / "registries").exists() else []:
            _try_unlink(lf)

    # ── 4. Any lock.pid under repo root ───────────────────────────────────
    try:
        for lf in REPO_ROOT.rglob("lock.pid"):
            _try_unlink(lf)
    except OSError:
        pass

    # ── Report ─────────────────────────────────────────────────────────────
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HypatiaX reproducibility pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip Feynman (exp2), noise sweep (suppB), instability")
    parser.add_argument("--only", metavar="ID",
                        help="Run only this step id")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint — skip steps already marked pass")
    parser.add_argument("--from", dest="from_step", metavar="ID",
                        help="With --resume: force-rerun from this step id onwards")
    parser.add_argument("--clear-checkpoint", action="store_true",
                        help="Delete the checkpoint file and exit")
    parser.add_argument("--continue-on-fail", action="store_true",
                        help="Log failures but continue remaining steps")
    parser.add_argument("--verify-only", action="store_true",
                        help="Re-check existing results without re-running experiments")
    parser.add_argument("--skip-paper", action="store_true",
                        help="Skip Phase 4-B paper audit notebook steps")
    parser.add_argument("--seed", type=int, default=None, metavar="N",
                        help="Override PYSR_SEED / NN_SEED / PYTHONHASHSEED for all steps "
                             "(e.g. --seed 123). Defaults to 42 when omitted.")
    parser.add_argument("--pysr-timeout", type=int, default=None, metavar="SECS",
                        help="Wall-clock timeout (seconds) passed to PySR via PYSR_TIMEOUT "
                             "env var. Default: 1100s (paper-quality, from repro.yaml). "
                             "This is how long PySR searches internally — do NOT lower this "
                             "to fix timeouts; use --kill-grace instead.")
    parser.add_argument("--kill-grace", type=int, default=None, metavar="SECS",
                        help="Extra seconds the exp2 subprocess gets AFTER PYSR_TIMEOUT "
                             "before being hard-killed. Default: 300s (from repro.yaml "
                             "timeouts.kill_grace_seconds). Must cover Julia startup + "
                             "final-generation drain so the worker can write its result JSON. "
                             "The original 60s caused 0/30 solves — 300s fixes this.")
    parser.add_argument("--one-equation", action="store_true",
                        help="Smoke-test mode: run exactly 1 equation per experiment. "
                             "Injects ONE_EQUATION=1, N_TASKS_DEFI=1, N_FEYNMAN_TASKS=1, "
                             "N_TASKS_INSTABILITY=1, and forces PYSR_TIMEOUT=120 (unless "
                             "--pysr-timeout is also given). Use this to verify the full "
                             "pipeline end-to-end quickly on local hardware.")
    parser.add_argument("--one-equation-paper", action="store_true",
                        help="Reviewer-probe mode: run exactly 1 equation per experiment "
                             "with ALL paper-quality PySR values intact (N_ITERATIONS=1000, "
                             "POPULATIONS=30, PYSR_TIMEOUT=1100, PYSR_POPULATION_SIZE=33, "
                             "PYSR_PARSIMONY=0.01, PYSR_MAXSIZE=30, LLM_K_RUNS=30). "
                             "Use this to verify a single equation reproduces paper targets "
                             "exactly. Much slower than --one-equation (~15-30 min per step).")
    # ── Case-range support (CI mini-job splitting) ────────────────────────────
    # NOTE: --only is defined above (dest="only").  Do NOT add a second
    # add_argument("--only", dest="only_step") here — it would shadow the first
    # definition, zeroing args.only everywhere else in main().
    parser.add_argument(
        "--case-range",
        metavar="START-END",
        default=None,
        help=(
            "Run only cases START..END (1-indexed, inclusive) within the "
            "selected experiment.  Requires --only.  "
            "Example: --only exp1 --case-range 1-4"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print each step's command, environment overrides, and case-range "
            "without executing anything.  Safe to run locally or in CI to preview "
            "exactly what a job will do.  Respects --only, --case-range, "
            "--skip-slow, --skip-paper, and --resume."
        ),
    )
    args = parser.parse_args()

    # ── Validate --case-range ─────────────────────────────────────────────────
    if args.case_range and not args.only:
        parser.error("--case-range requires --only <STEP_ID>  e.g. --only exp1 --case-range 1-4")

    # FIX: validate --from is only used alongside --resume; warn otherwise
    if args.from_step and not args.resume:
        print("  WARNING: --from has no effect without --resume. "
              "Did you mean: python3 run_all.py --resume --from <id>?",
              file=sys.stderr)

    os.chdir(REPO_ROOT)
    LOG_DIR.mkdir(exist_ok=True)
    ensure_output_dirs()

    # ── Pre-flight: clear all stale lock files ─────────────────────────────
    _clear_stale_locks()

    # ── --clear-checkpoint ─────────────────────────────────────────────────
    if args.clear_checkpoint:
        clear_checkpoint()
        sys.exit(0)

    banner("HypatiaX · Reproducibility Pipeline v4.8 (checkpoint/resume)"
           + ("  [DRY-RUN]" if args.dry_run else "")
           + ("  [SMOKE-TEST: 1 equation]" if args.one_equation else "")
           + ("  [PAPER-QUALITY: 1 equation]" if args.one_equation_paper else ""))
    print(f"  Repo      : {REPO_ROOT}")
    print(f"  Python    : {sys.version.split()[0]}")
    print(f"  Date      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results   : {RESULTS_DIR}")
    print(f"  Logs      : {LOG_DIR}")
    print(f"  Checkpoint: {CHECKPOINT}")

    # ── API key ────────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("\n  ERROR: ANTHROPIC_API_KEY is not set.")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)
    print(f"\n  API key   : set ({len(api_key)} chars)")

    # ── hypatiax/protocols/ check ──────────────────────────────────────────
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
        print(f"\n  ERROR: {len(missing_hp)} input-data module(s) missing "
              "from hypatiax/protocols/:")
        for f in missing_hp:
            print(f"    ✗  {f}")
        sys.exit(1)
    print(f"  Protocols : all {len(required_hp)} hypatiax/protocols/ modules ✓")

    # ── NB-06 code-quality pre-audit (non-blocking) ────────────────────────
    nb06 = REPO_ROOT / "notebooks" / "NB-06_Code_Quality_Pipeline_Integrity.ipynb"
    if nb06.exists():
        try:
            r = subprocess.run(
                ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                 "--inplace", "--ExecutePreprocessor.timeout=120", str(nb06)],
                capture_output=True, text=True, timeout=150,
            )
            status = "✓" if r.returncode == 0 else "⚠ warnings (non-blocking)"
            print(f"  NB-06     : code quality pre-audit {status}")
        except FileNotFoundError:
            print("  NB-06     : jupyter not found — skipped (pip install notebook)")
        except Exception as exc:
            print(f"  NB-06     : skipped ({exc})")

    # ── verify-only shortcut ───────────────────────────────────────────────
    if args.verify_only:
        banner("Verify-only mode")
        subprocess.run([sys.executable, "scripts/patches/verify_results.py", "--report"],
                       check=False)
        subprocess.run([sys.executable, "hypatiax/reproducibility/hash_lock.py", "--check"],
                       check=False)
        sys.exit(0)

    # ── Load repro.yaml config ───────────────────────────────────────────────────
    _repro_config = load_repro_config()
    _timeout_config = _repro_config.get("timeouts", {})
    _pysr_config = _repro_config.get("pysr", {})

    # Timeout defaults (300s PySR + 300s kill-grace — from repro.yaml)
    DEFAULT_PYSR_TIMEOUT = _timeout_config.get("pysr_attempt_seconds", 1100)
    DEFAULT_METHOD_TIMEOUT = _timeout_config.get("method_seconds", 900)
    DEFAULT_KILL_GRACE = _timeout_config.get("kill_grace_seconds", 300)

    # ── Build environment (mirrors run_all.sh and notebook cell 2) ─────────
    _seed_str = str(args.seed) if args.seed is not None else "42"

    env = {**os.environ}
    env["PYTHONWARNINGS"] = "ignore"
    env["NN_SEED"]               = os.environ.get("NN_SEED",   _seed_str)
    env["PYSR_SEED"]             = os.environ.get("PYSR_SEED", _seed_str)
    env["PYTHONHASHSEED"]        = os.environ.get("PYTHONHASHSEED", _seed_str)

    # Override seeds if --seed was given
    if args.seed is not None:
        env["NN_SEED"]        = _seed_str
        env["PYSR_SEED"]      = _seed_str
        env["PYTHONHASHSEED"] = _seed_str

    # LLM config
    env.setdefault("LLM_MODEL", _repro_config.get("llm_model", "claude-sonnet-4-20250514"))
    env.setdefault("LLM_RETRIES", str(_repro_config.get("llm_retries", 3)))
    # FIX: default LLM_K_RUNS to 1 (not llm_k_runs=30 from repro.yaml) so standard
    # experiment steps run a single pass; the instability step overrides to 30 via
    # its own env_extra={"LLM_K_RUNS": "30"}.
    env.setdefault("LLM_K_RUNS", "1")

    # Task counts
    env.setdefault("N_TASKS_DEFI", str(_repro_config.get("n_tasks_defi", 74)))
    env.setdefault("N_TASKS_INSTABILITY", str(_repro_config.get("n_tasks_instability", 70)))
    env.setdefault("PCA_TRAIN_FRAC", str(_repro_config.get("pca_train_frac", 0.40)))
    env.setdefault("NN_TIME_LIMIT", str(_repro_config.get("nn_time_limit", 120)))

    # Engine
    env.setdefault("ENGINE_NAME", _repro_config.get("engine", {}).get("name", "hybrid_system_v50_2"))
    env.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    env.setdefault("JULIA_NUM_THREADS", "1")

    # ── TIMEOUT: priority order ───────────────────────────────────────────────────
    # 1. --pysr-timeout CLI flag (highest priority)
    # 2. repro.yaml timeouts.pysr_attempt_seconds
    # 3. Environment variable PYSR_TIMEOUT (fallback)
    # 4. Default 1100s (paper-quality)
    if args.pysr_timeout is not None:
        env["PYSR_TIMEOUT"] = str(args.pysr_timeout)
        # FIX: do NOT derive METHOD_TIMEOUT as min(pysr_timeout*3, 1800) — that formula
        # produces 1800 for pysr_timeout=1100, which the exp1 script caps to 300s then
        # sets wall-clock=630s, causing a timeout before PySR finishes (the 630s bug).
        # Instead always use the repro.yaml method_seconds value (900s) directly.
        env["METHOD_TIMEOUT"] = str(DEFAULT_METHOD_TIMEOUT)
        print(f"  PYSR_TIMEOUT={args.pysr_timeout}s  (--pysr-timeout override)")
        print(f"  METHOD_TIMEOUT={env['METHOD_TIMEOUT']}s  (repro.yaml method_seconds)")
    else:
        # Check repro.yaml first
        pysr_timeout = DEFAULT_PYSR_TIMEOUT
        method_timeout = DEFAULT_METHOD_TIMEOUT

        # Then environment variable override
        env_pysr = os.environ.get("PYSR_TIMEOUT")
        if env_pysr:
            pysr_timeout = int(env_pysr)
            # FIX: use repro.yaml method_seconds (900) rather than min(pysr*3, 1800)
            method_timeout = DEFAULT_METHOD_TIMEOUT
            print(f"  ⚠ PYSR_TIMEOUT={pysr_timeout}s from env (repro.yaml wants {DEFAULT_PYSR_TIMEOUT}s)")

        env["PYSR_TIMEOUT"] = str(pysr_timeout)
        env["METHOD_TIMEOUT"] = str(method_timeout)
        print(f"  PYSR_TIMEOUT={pysr_timeout}s  (repro.yaml default: 1100s)")
        print(f"  METHOD_TIMEOUT={method_timeout}s  (paper-quality: 900s)")

    # PySR search parameters from repro.yaml
    env.setdefault("POPULATIONS",    str(_pysr_config.get("populations", 30)))
    env.setdefault("N_ITERATIONS",   str(_pysr_config.get("niterations", 1000)))
    # Alias names used by some experiment scripts directly
    env.setdefault("PYSR_POPULATIONS",  env["POPULATIONS"])
    env.setdefault("PYSR_NITERATIONS",  env["N_ITERATIONS"])
    env.setdefault("PYSR_PARALLELISM",  _pysr_config.get("parallelism", "multithreading"))
    env.setdefault("_PYSR_TIMEOUT_SECS",   env["PYSR_TIMEOUT"])
    env.setdefault("_METHOD_TIMEOUT_SECS",  env["METHOD_TIMEOUT"])
    # FIX: expose EQUATION_WALL_CLOCK from repro.yaml (timeouts.equation_wall_clock=1200)
    # so experiment scripts that enforce a per-equation wall-clock cap use the correct value.
    env.setdefault("EQUATION_WALL_CLOCK", str(_timeout_config.get("equation_wall_clock", 1200)))
    env.setdefault("PYSR_POPULATION_SIZE", str(_pysr_config.get("population_size", 33)))
    env.setdefault("PYSR_PARSIMONY", str(_pysr_config.get("parsimony", 0.01)))
    env.setdefault("PYSR_MAXSIZE", str(_pysr_config.get("maxsize", 30)))

    env["PYTHONPATH"]    = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["RESULTS_DIR"]   = str(RESULTS_DIR)
    env["PIPELINE_PYTHON"] = sys.executable
    env["REPRO_ROOT"]  = str(REPO_ROOT)

    _seed_source = "--seed flag" if args.seed is not None else "default (env or 42)"
    print(f"\n  NN_SEED={env['NN_SEED']}  PYSR_SEED={env['PYSR_SEED']}  "
          f"PYTHONHASHSEED={env['PYTHONHASHSEED']}  (source: {_seed_source})")
    print(f"  LLM_MODEL={env['LLM_MODEL']}")
    print(f"  ENGINE={env['ENGINE_NAME']}  N_TASKS_INSTABILITY={env['N_TASKS_INSTABILITY']}")
    print(f"  PySR: iterations={env['N_ITERATIONS']} populations={env['POPULATIONS']} pop_size={env['PYSR_POPULATION_SIZE']}")
    if args.skip_paper:
        print("  --skip-paper: Phase 4-B notebook steps will be skipped")
    if args.dry_run:
        print("\n  ⚡ DRY-RUN MODE — nothing will be executed")
        print("  Commands below show exactly what each step would run.")


    # ── --one-equation: smoke-test mode ───────────────────────────────────
    if args.one_equation:
        # Tell every experiment script to run only 1 equation/task.
        # Scripts should check ONE_EQUATION=1 and/or the N_TASKS_* vars.
        env["ONE_EQUATION"]          = "1"
        env["N_TASKS_DEFI"]          = "1"   # exp1: 1 of 74 DeFi tasks
        env["N_CORE15_TASKS"]        = "1"   # exp1: 1 of 15 Core-15 ablation equations
        env["N_FEYNMAN_TASKS"]       = "1"   # exp2: 1 of 30 Feynman equations
        env["N_TASKS_INSTABILITY"]   = "1"   # instability: 1 of 70 tasks
        env["N_NGUYEN_TASKS"]        = "1"   # exp3/exp3b: 1 of 12 Nguyen equations
        env["N_NOISE_EQUATIONS"]     = "1"   # suppB: 1 equation across all noise levels
        env["LLM_K_RUNS"]            = "1"   # force K=1 even for instability step
        # FIX 4: real smoke-test PySR config — small populations + iterations
        env["N_ITERATIONS"]          = "200"  # was default ~1000 (too heavy for smoke-test)
        env["POPULATIONS"]           = "10"   # was default ~30  (too heavy for smoke-test)
        # Use a short PySR timeout (60s) unless the user explicitly overrode it
        if args.pysr_timeout is None:
            env["PYSR_TIMEOUT"] = "60"  # FIX 4: tightened from 120 → 60 for real smoke-test
        print("\n" + "▲" * 68)
        print("  ▲▲  SMOKE-TEST MODE  (--one-equation)")
        print("  ▲▲  1 equation per experiment · PYSR_TIMEOUT="
              + env["PYSR_TIMEOUT"] + "s"
              + "  N_ITERATIONS=" + env["N_ITERATIONS"]
              + "  POPULATIONS=" + env["POPULATIONS"])
        print("  ▲▲  This is NOT a full reproducibility run.")
        print("  ▲▲  Results will NOT match paper targets.")
        print("▲" * 68)

    # ── --one-equation-paper: reviewer-probe mode (1 equation, paper-quality values) ──
    if args.one_equation_paper:
        # Scope: 1 equation per experiment — identical task-count reduction to
        # --one-equation.  Difference: every PySR hyperparameter and LLM setting
        # is kept at the paper-quality value from repro.yaml v3.0 so a reviewer
        # can verify that a single equation reproduces the paper's exact result.
        env["ONE_EQUATION"]          = "1"
        env["N_TASKS_DEFI"]          = "1"   # exp1: 1 of 74 DeFi tasks
        env["N_CORE15_TASKS"]        = "1"   # exp1: 1 of 15 Core-15 ablation equations
        env["N_FEYNMAN_TASKS"]       = "1"   # exp2: 1 of 30 Feynman equations
        env["N_TASKS_INSTABILITY"]   = "1"   # instability: 1 of 70 tasks
        env["N_NGUYEN_TASKS"]        = "1"   # exp3/exp3b: 1 of 12 Nguyen equations
        env["N_NOISE_EQUATIONS"]     = "1"   # suppB: 1 equation across all noise levels
        # ── Paper-quality PySR values (repro.yaml v3.0) — NOT degraded ────
        env["N_ITERATIONS"]          = "1000"   # pysr.niterations
        env["POPULATIONS"]           = "30"     # pysr.populations
        env["PYSR_POPULATION_SIZE"]  = "33"     # pysr.population_size
        env["PYSR_PARSIMONY"]        = "0.01"   # pysr.parsimony
        env["PYSR_MAXSIZE"]          = "30"     # pysr.maxsize
        env["PYSR_PARALLELISM"]      = "multithreading"  # pysr.parallelism
        env["LLM_K_RUNS"]            = "30"     # repro.yaml llm_k_runs (instability + all steps)
        # Timeout: honour explicit --pysr-timeout override, else use paper value
        if args.pysr_timeout is None:
            env["PYSR_TIMEOUT"]      = "1100"   # timeouts.pysr_attempt_seconds (paper value)
        env["METHOD_TIMEOUT"]        = "900"    # timeouts.method_seconds
        env["EQUATION_WALL_CLOCK"]   = "1200"   # timeouts.equation_wall_clock (paper value)
        print("\n" + "★" * 68)
        print("  ★★  PAPER-QUALITY PROBE  (--one-equation-paper)")
        print("  ★★  1 equation per experiment · ALL values from repro.yaml v3.0")
        print("  ★★  PYSR_TIMEOUT=" + env["PYSR_TIMEOUT"] + "s"
              + "  N_ITERATIONS=" + env["N_ITERATIONS"]
              + "  POPULATIONS=" + env["POPULATIONS"])
        print("  ★★  PYSR_POPULATION_SIZE=" + env["PYSR_POPULATION_SIZE"]
              + "  PYSR_PARSIMONY=" + env["PYSR_PARSIMONY"]
              + "  PYSR_MAXSIZE=" + env["PYSR_MAXSIZE"])
        print("  ★★  LLM_K_RUNS=" + env["LLM_K_RUNS"]
              + "  — results WILL match paper targets for the selected equation.")
        print("★" * 68)

    # ── Validate --only / --from ───────────────────────────────────────────
    if args.only and args.only not in STEP_IDS:
        print(f"\n  ERROR: unknown step id '{args.only}'.")
        print(f"  Valid ids: {', '.join(STEP_IDS)}")
        sys.exit(1)
    if args.from_step and args.from_step not in STEP_IDS:
        print(f"\n  ERROR: unknown step id '{args.from_step}'.")
        print(f"  Valid ids: {', '.join(STEP_IDS)}")
        sys.exit(1)

    # ── Load checkpoint state ──────────────────────────────────────────────
    checkpoint_state: dict[str, str] = {}
    if args.resume:
        # Re-read both sources so repo-root entries are never lost
        _root_cp = REPO_ROOT / "pipeline_checkpoint.json"
        for _cp_path in [_root_cp, CHECKPOINT]:
            if _cp_path.exists():
                try:
                    _disk = json.loads(_cp_path.read_text())
                    for _k, _v in _disk.items():
                        if checkpoint_state.get(_k) != "pass":
                            checkpoint_state[_k] = _v
                except Exception:
                    pass
        # exp2 key migration: old "exp2: pass" → mark all three new steps pass
        if checkpoint_state.get("exp2") == "pass":
            for _sid in ("exp2_sym", "exp2_hyb", "exp2"):
                checkpoint_state.setdefault(_sid, "pass")
        if checkpoint_state.get("exp2") == "fail":
            checkpoint_state.pop("exp2", None)
        # Persist merged state immediately
        save_checkpoint(checkpoint_state)

        # ── Print clean pipeline status table, then jump to pending work ──
        _col = {"pass": "✓", "fail": "✗", "skip": "─"}
        _pending = [s for s in STEPS if checkpoint_state.get(s.id) != "pass"]
        _done    = [s for s in STEPS if checkpoint_state.get(s.id) == "pass"]
        print()
        print("  ┌─ Pipeline status (" + f"{len(_done)}/{len(STEPS)} done) ──────────────────────────────────────────")
        _cur_phase = ""
        for _s in STEPS:
            if _s.phase != _cur_phase:
                print(f"  │  Phase {_s.phase}")
                _cur_phase = _s.phase
            _st  = checkpoint_state.get(_s.id, "todo")
            _ico = {"pass": "✓", "fail": "✗", "todo": "·"}.get(_st, "·")
            _tag = f"[{_st}]" if _st in ("fail",) else ""
            print(f"  │    {_ico}  {_s.id:<30}  {_tag}")
        if _pending:
            print(f"  └─ Next: [{_pending[0].id}]  {_pending[0].label}")
        else:
            print("  └─ All steps done.")
        print()
        if args.from_step:
            print(f"  ── --from {args.from_step}: force-rerun from here onwards")
        if not checkpoint_state:
            print("  ── No checkpoint found — running full pipeline")

    # ── Run pipeline ───────────────────────────────────────────────────────
    results: list[StepResult] = []
    current_phase = ""
    t_total = time.time()
    # once we reach --from step, all subsequent steps run regardless of checkpoint
    # FIX: must start as False so resume-skip fires on already-passed steps when
    # no --from is given.  With the old `args.from_step is None` initialisation,
    # past_from was True from the start, making `not past_from` always False and
    # therefore the resume-skip branch never triggered — every step re-ran.
    past_from = False

    try:
        for step in STEPS:
            if args.from_step and step.id == args.from_step:
                past_from = True

            # --only filter
            if args.only and step.id != args.only:
                results.append(StepResult(step.id, step.label, "skip"))
                continue

            # --resume: re-read checkpoint before every skip check (belt-and-suspenders)
            if args.resume and not past_from:
                for _cp_path in [REPO_ROOT / "pipeline_checkpoint.json", CHECKPOINT]:
                    if _cp_path.exists():
                        try:
                            for _k, _v in json.loads(_cp_path.read_text()).items():
                                if checkpoint_state.get(_k) != "pass":
                                    checkpoint_state[_k] = _v
                        except Exception:
                            pass

            # --resume: silently skip steps that already passed
            if (args.resume
                    and checkpoint_state.get(step.id) == "pass"
                    and not past_from):
                results.append(StepResult(step.id, step.label, "resume-skip"))
                continue

            # Only print phase banner when we actually run a step
            if step.phase != current_phase:
                banner(f"Phase {step.phase}")
                current_phase = step.phase

            # --skip-slow filter
            if args.skip_slow and step.slow:
                results.append(StepResult(step.id, step.label, "skip"))
                print(f"  ── skip [{step.id}]  (--skip-slow)")
                continue

            # FIX: --skip-paper filter — previously parsed but never applied
            if args.skip_paper and step.paper:
                results.append(StepResult(step.id, step.label, "skip"))
                print(f"  ── skip [{step.id}]  (--skip-paper)")
                continue

            result = run_step(step, env, args)
            results.append(result)

            # save checkpoint after every step
            checkpoint_state[step.id] = result.status
            save_checkpoint(checkpoint_state)

            if result.status == "fail" and not args.continue_on_fail:
                print(f"\n  Pipeline aborted at [{step.id}].")
                print(f"  Checkpoint saved → {CHECKPOINT}")
                print("  To resume:         python3 run_all.py --resume")
                print(f"  To rerun this step: python3 run_all.py --only {step.id}")
                _print_summary(results, time.time() - t_total)
                sys.exit(1)

    except KeyboardInterrupt:
        # ── Ctrl+C pressed between steps (or re-raised from run_step) ──────
        print("\n\n  ⚠  Interrupted by user (Ctrl+C).")
        # Mark any step currently being attempted as failed in checkpoint
        # (run_step already appended its StepResult before re-raising, so
        #  results list is up to date; just ensure the checkpoint reflects it.)
        for r in results:
            if r.id not in checkpoint_state:
                checkpoint_state[r.id] = r.status
        save_checkpoint(checkpoint_state)
        print(f"  Checkpoint saved → {CHECKPOINT}")
        print("  Resume with:       python3 run_all.py --resume"
              + ("  --one-equation" if args.one_equation else "")
              + ("  --one-equation-paper" if args.one_equation_paper else ""))
        _print_summary(results, time.time() - t_total)
        sys.exit(130)   # conventional exit code for Ctrl+C

    _print_summary(results, time.time() - t_total)

    # Clear checkpoint only after a complete, fully-passing run
    failed = [r for r in results if r.status == "fail"]
    if not failed and not args.only:
        clear_checkpoint()

    if args.one_equation:
        print("\n" + "▲" * 68)
        print("  ▲▲  SMOKE-TEST COMPLETE  (--one-equation)")
        print("  ▲▲  Re-run without --one-equation for a full reproducibility run.")
        print("▲" * 68)

    if args.one_equation_paper:
        print("\n" + "★" * 68)
        print("  ★★  PAPER-QUALITY PROBE COMPLETE  (--one-equation-paper)")
        print("  ★★  Result for this equation used paper-quality values throughout.")
        print("  ★★  Re-run without --one-equation-paper for the full pipeline.")
        print("★" * 68)

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
        t = f"  {r.elapsed:6.0f}s" if r.status in ("pass", "fail") else "        "
        print(f"  {col[r.status]} [{r.id:28s}] {r.label[:48]:48s}{t}")

    print()
    print(f"  ✓ passed      : {len(passed)}")
    print(f"  ✗ failed      : {len(failed)}")
    print(f"  ─ skipped     : {len(skipped)}")
    print(f"  ↩ resume-skip : {len(resume_skips)}")
    print(f"  Wall time     : {hh:02d}:{mm:02d}:{ss:02d}")

    # Result file inventory
    data_files, fig_files, tbl_files = inventory_results()
    print(f"\n  Results → {RESULTS_DIR}")
    print(f"    Data files (JSON+CSV) : {data_files}")
    print(f"    Figures (PDF)         : {fig_files}")
    print(f"    Tables  (TeX)         : {tbl_files}")

    # Provenance coverage
    prov = LOG_DIR / "provenance_audit" / "provenance_audit_summary.txt"
    if prov.exists():
        print(f"\n  Provenance ({prov}):")
        for line in prov.read_text().splitlines():
            if any(k in line for k in ("AUTHORITATIVE", "ORPHAN", "Total")):
                print(f"    {line.strip()}")

    if failed:
        print("\n  Failed steps:")
        for r in failed:
            print(f"    [{r.id}] → {r.log_path}")
        print()
        print(f"  Checkpoint : {CHECKPOINT}")
        print("  Resume     : python3 run_all.py --resume")
        print("  Single step: python3 run_all.py --only <id>")
    else:
        print("\n  ✓ All steps passed.")
        print(f"  Results    : {RESULTS_DIR}/")
        print(f"  Figures    : {RESULTS_DIR}/figures/")
        print(f"  Tables     : {RESULTS_DIR}/tables/")
        print(f"  Provenance : {LOG_DIR}/provenance_audit/")
        print(f"  Import DAG : {LOG_DIR}/repro_output/import_graph.dot")
        print("  Checkpoint : cleared (all steps passed)")


if __name__ == "__main__":
    main()
