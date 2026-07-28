"""
HypatiaX — Experiment 1: LLM Ablation  (§10.6 Core-15)
========================================================
PySR-only vs HypatiaX (HybridDiscoverySystem v5.1) on Core-15.

Paper sections : §10.6 (ablation), §10.9 (instability)
Engine         : hybrid_system_v50_2.py (v5.1)
Output files   : exp1_ablation_results[_seedN].json
                 exp1_ablation_table[_seedN].tex
                 exp1_rf01_mannwhitney[_seedN].json
                 exp1_rf01_significant[_seedN].tex
                 exp1_rf01_subdomain[_seedN].tex
                 exp1_instability_stats[_seedN].json
                 instability_extrapolation_v2[_seedN].csv
                 provenance_map_exp1[_seedN].json

Fixes applied vs original hypatiax_exp1_ablation.ipynb
-------------------------------------------------------
FIX-POP       : populations=30 (was 2 — inflated speedup)
FIX-WIRE      : hypatia → HybridDiscoverySystem v5.1 (was raw PySR + manual warm-start)
FIX-KEY       : checkpoint keyed on eq_id int (survives equation renames)
FIX-APIKEY    : Kaggle Secrets / env var (no hardcoded sk-ant key)
FIX-WALLCLOCK : per-condition wall-clocks; hypatia = 3×PYSR_TIMEOUT+300,
                pysr_only = PYSR_TIMEOUT+300 (shared cap fired inside Julia retry 2)
FIX-A         : RMSE in original units via HybridDiscoverySystem.discover()
FIX-B         : formula/expression/final_formula key aliases resolved in v5.1
FIX-C         : deterministic PySR (allow_nondeterministic=False)
FIX-D         : extreme-scale log-transform active inside discover()
FIX-POW       : auto pow for non-negative X active inside discover()
FIX-SEED      : env-driven seed (PYSR_SEED → NN_SEED → 42); output files namespaced
FIX-RESTORE   : VariableNameSanitizer.restore() uses longest-first sorted keys
                (prevents partial substitutions, e.g. var_N1 → N1 before var_N → N)
"""

# =============================================================================
# §0 · Standard library imports
# =============================================================================

import csv
import importlib.util as _ilu
import inspect
import json
import os
import pathlib as _pl
import random
import re
import signal
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =============================================================================
# §1 · Reproducibility & API key
# =============================================================================

_GLOBAL_SEED = int(os.environ.get("PYSR_SEED", os.environ.get("NN_SEED", 42)))
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
os.environ["PYTHONHASHSEED"] = str(_GLOBAL_SEED)

# API key: GITHUB Secrets → environment variable (no hardcoded keys)

if os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-"):
        print("✅ API key found in environment")
else:
        print("⚠️  ANTHROPIC_API_KEY not set — LLM proposal step will be skipped")

# =============================================================================
# §2 · Output paths
#
# CI layout (ci_experiment.yml):
#   OUT_BASE      = hypatiax/data/results          (workflow-level env)
#   RESULT_SUBDIR = ablation/exp1_ablation         (set in meta step)
#   RESULTS_DIR   = ${OUT_BASE}/${RESULT_SUBDIR}   (worker env var)
#
# The script writes everything under RESULTS_DIR when that env var is set,
# falling back to the local working directory for direct / Kaggle runs.
#
# Seed-namespacing: seed=42 keeps canonical names; other seeds append _seedN
# so multi-seed runs (exp1b style) never overwrite each other.
# =============================================================================

_HERE      = Path().resolve()
# RESULTS_DIR is injected by the CI worker step; fall back to cwd.
_RESULTS_DIR_ENV = os.environ.get("RESULTS_DIR", "")
OUTPUT_DIR = Path(_RESULTS_DIR_ENV) if _RESULTS_DIR_ENV else _HERE
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_seed_suffix = f"_seed{_GLOBAL_SEED}" if _GLOBAL_SEED != 42 else ""
RESULTS_PATH = OUTPUT_DIR / f"exp1_ablation_results{_seed_suffix}.json"
CKPT_PATH    = OUTPUT_DIR / f"exp1_ablation_checkpoint{_seed_suffix}.json"
TEX_PATH     = OUTPUT_DIR / f"exp1_ablation_table{_seed_suffix}.tex"
RF01_JSON    = OUTPUT_DIR / f"exp1_rf01_mannwhitney{_seed_suffix}.json"
RF01_SIGTEX  = OUTPUT_DIR / f"exp1_rf01_significant{_seed_suffix}.tex"
RF01_SUBDTEX = OUTPUT_DIR / f"exp1_rf01_subdomain{_seed_suffix}.tex"
INSTAB_STATS = OUTPUT_DIR / f"exp1_instability_stats{_seed_suffix}.json"
INSTAB_CSV   = OUTPUT_DIR / f"instability_extrapolation_v2{_seed_suffix}.csv"
PROV_PATH    = OUTPUT_DIR / f"provenance_map_exp1{_seed_suffix}.json"

# =============================================================================
# §3 · Run-time parameters
# =============================================================================

# FIX-POP: paper-quality default is populations=30, not 2.
POPULATIONS        = int(os.environ.get("POPULATIONS",    os.environ.get("PYSR_POPULATIONS", "30")))
NITERATIONS        = int(os.environ.get("N_ITERATIONS",   os.environ.get("PYSR_NITERATIONS", "1000")))
PYSR_TIMEOUT_SECS  = int(os.environ.get("PYSR_TIMEOUT",   "1100"))
# METHOD_TIMEOUT is the per-method budget; do not cap it to 300 (previous bug).
_TIMEOUT_ENV       = os.environ.get("METHOD_TIMEOUT") or os.environ.get("PYSR_TIMEOUT", "900")
TIMEOUT_SECS       = int(_TIMEOUT_ENV)
SEED               = _GLOBAL_SEED
CONDITIONS         = ["pysr_only", "hypatia"]
MODEL_STRING       = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")

# FIX-WALLCLOCK: per-condition wall-clocks.
# hypatia runs _HYPATIA_MAX_RETRIES × PySR attempts — shared cap fired inside
# Julia during attempt 2 (only ~551 s remained after attempt 1 used ~1149 s).
_HYPATIA_MAX_RETRIES = int(os.environ.get("HYPATIA_MAX_RETRIES", "3"))
_POST_PROC_BUDGET    = int(os.environ.get("PYSR_POST_PROC_BUDGET", "300"))
HYPATIA_WALL_CLOCK   = _HYPATIA_MAX_RETRIES * PYSR_TIMEOUT_SECS + _POST_PROC_BUDGET
PYSR_ONLY_WALL_CLOCK = PYSR_TIMEOUT_SECS + _POST_PROC_BUDGET
EQUATION_WALL_CLOCK_TIMEOUT = HYPATIA_WALL_CLOCK  # backward-compat alias

_seed_source = (
    "PYSR_SEED"  if os.environ.get("PYSR_SEED")  else
    "NN_SEED"    if os.environ.get("NN_SEED")     else
    "default"
)
print(f"populations : {POPULATIONS}  (paper-quality, FIX-POP)")
print(f"niterations : {NITERATIONS}")
print(f"timeout_s   : METHOD_TIMEOUT={TIMEOUT_SECS}s  PYSR_TIMEOUT={PYSR_TIMEOUT_SECS}s")
print(f"wall-clock  : hypatia={HYPATIA_WALL_CLOCK}s  pysr_only={PYSR_ONLY_WALL_CLOCK}s  (FIX-WALLCLOCK)")
print(f"seed        : {SEED}  (source: {_seed_source})")
print("✅ Parameters ready")

# =============================================================================
# §4 · HybridDiscoverySystem import (v5.1)
# =============================================================================

_V50_CANDIDATES = [
    _pl.Path("hybrid_system_v50_2.py"),
    _pl.Path(__file__).resolve().parents[2] / "tools/symbolic/hybrid_system_v50_2.py",
]
_V50_PATH = next((p for p in _V50_CANDIDATES if p.exists()), None)
if _V50_PATH is None:
    print(
        "⚠️  hybrid_system_v50_2.py not found — hypatia condition will raise at runtime.\n"
        "   Upload it to /kaggle/working/ (or the working directory) alongside this script."
    )
    HybridDiscoverySystem = None
    DiscoveryConfig = None
else:
    _spec   = _ilu.spec_from_file_location("hybrid_system_v50_2", _V50_PATH)
    _hs_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_hs_mod)
    HybridDiscoverySystem = _hs_mod.HybridDiscoverySystem
    DiscoveryConfig       = _hs_mod.DiscoveryConfig
    print(f"✅ HybridDiscoverySystem (v5.1) imported from {_V50_PATH}")

# =============================================================================
# §5 · PySR import
# =============================================================================

try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
    import pysr as _pysr_mod
    print(f"✅ PySR {_pysr_mod.__version__} ready")
except ImportError:
    PYSR_AVAILABLE = False
    print("⚠️  PySR not installed — pysr_only condition will fail. Run: pip install pysr")

# =============================================================================
# §6 · Debug logger
# =============================================================================

_T0 = time.time()

def dbg(msg: str) -> None:
    print(f"  [{time.time() - _T0:7.1f}s] {msg}", flush=True)

dbg("Imports complete")

# =============================================================================
# §7 · Variable name sanitiser
# =============================================================================

class VariableNameSanitizer:
    """Renames Julia-reserved single-letter variables to safe alternatives."""

    RESERVED = {"S", "N", "C", "D", "E", "I", "O"}

    def __init__(self):
        self.fwd: dict[str, str] = {}  # original → safe
        self.rev: dict[str, str] = {}  # safe → original

    def sanitize(self, names: list[str]) -> tuple[list[str], bool]:
        out, conflict = [], False
        for v in names:
            if v in self.RESERVED:
                safe = f"var_{v}"
                counter = 1
                while safe in out or safe in names:
                    safe = f"var_{v}{counter}"
                    counter += 1
                self.fwd[v]    = safe
                self.rev[safe] = v
                out.append(safe)
                conflict = True
            else:
                out.append(v)
        return out, conflict

    def restore(self, expression: str) -> str:
        """FIX-RESTORE: sort by length descending to avoid partial substitutions."""
        if not self.rev or not expression:
            return expression
        result = expression
        for safe in sorted(self.rev, key=len, reverse=True):
            result = re.sub(r"\b" + re.escape(safe) + r"\b", self.rev[safe], result)
        return result

    def apply_to_llm_expr(self, expression: str) -> str:
        """Translate an LLM-proposed expression that uses original names → safe names."""
        if not self.fwd or not expression:
            return expression
        result = expression
        for orig, safe in sorted(self.fwd.items(), key=lambda kv: len(kv[0]), reverse=True):
            result = re.sub(r"\b" + re.escape(orig) + r"\b", safe, result)
        return result

    def log(self) -> str:
        return ", ".join(f"{o}→{s}" for o, s in self.fwd.items())

# =============================================================================
# §8 · Core-15 equation catalogue
# =============================================================================

EXTRAP_REGIMES = [("near", 1.2), ("medium", None), ("far", 5.0)]

CORE_15 = [
    # ── Chemistry (3) ────────────────────────────────────────────────────────
    {
        "name": "Arrhenius", "domain": "Chemistry", "vars": ["T"],
        "train_range":  [(300, 500)],
        "extrap_range": [(500, 1000)],
        "fn": lambda T: 1e6 * np.exp(-50000 / (8.314 * T)),
        "formula_latex": r"k = A\exp(-E_a/RT)",
        "formula_sympy": "1e6 * np.exp(-50000 / (8.314 * T))",
        "note": (
            "Known LLM failure mode: correct prior constrains PySR search space "
            "incompatibly with data scale → premature convergence (Arrhenius pattern)."
        ),
    },
    {
        "name": "Henderson-Hasselbalch", "domain": "Chemistry", "vars": ["A", "HA"],
        "train_range":  [(0.001, 1.0), (0.001, 1.0)],
        "extrap_range": [(1.0, 10.0), (1.0, 10.0)],
        "fn": lambda A, HA: 4.75 + np.log10(A / HA),
        "formula_latex": r"\mathrm{pH} = \mathrm{p}K_a + \log_{10}([A^-]/[HA])",
        "formula_sympy": "4.75 + np.log10(A / HA)",
    },
    {
        "name": "Rate Law", "domain": "Chemistry", "vars": ["A", "B"],
        "train_range":  [(0.1, 2.0), (0.1, 2.0)],
        "extrap_range": [(2.0, 5.0), (2.0, 5.0)],
        "fn": lambda A, B: 0.5 * A**2 * B,
        "formula_latex": r"r = k[A]^2[B]",
        "formula_sympy": "0.5 * A**2 * B",
    },
    # ── Biology (3) ──────────────────────────────────────────────────────────
    {
        "name": "Allometric Scaling", "domain": "Biology", "vars": ["M"],
        "train_range":  [(1, 100)],
        "extrap_range": [(100, 500)],
        "fn": lambda M: 0.1 * M**0.75,
        "formula_latex": r"Y = aM^b",
        "formula_sympy": "0.1 * M**0.75",
    },
    {
        "name": "Michaelis-Menten", "domain": "Biology", "vars": ["S"],
        "train_range":  [(0.1, 5.0)],
        "extrap_range": [(5.0, 20.0)],
        "fn": lambda S: 10.0 * S / (2.0 + S),
        "formula_latex": r"v = V_{\max}[S]/(K_m + [S])",
        "formula_sympy": "10.0 * S / (2.0 + S)",
        "note": "Far-R² can reach −634.6 under extreme OOD — resizebox needed in LaTeX table.",
    },
    {
        "name": "Logistic Growth", "domain": "Biology", "vars": ["N"],
        "train_range":  [(10, 500)],
        "extrap_range": [(500, 1000)],
        "fn": lambda N: 0.3 * N * (1 - N / 1000),
        "formula_latex": r"dN/dt = rN(1-N/K)",
        "formula_sympy": "0.3 * N * (1 - N / 1000)",
    },
    # ── Physics (3) ──────────────────────────────────────────────────────────
    {
        "name": "Kinetic Energy", "domain": "Physics", "vars": ["m", "v"],
        "train_range":  [(0.5, 10.0), (1.0, 20.0)],
        "extrap_range": [(10.0, 50.0), (20.0, 100.0)],
        "fn": lambda m, v: 0.5 * m * v**2,
        "formula_latex": r"E = \frac{1}{2}mv^2",
        "formula_sympy": "0.5 * m * v**2",
    },
    {
        "name": "Gravitational Force", "domain": "Physics", "vars": ["m1", "m2", "r"],
        "train_range":  [(1e10, 1e12), (1e10, 1e12), (1e6, 1e8)],
        "extrap_range": [(1e12, 1e14), (1e12, 1e14), (1e8, 1e10)],
        "fn": lambda m1, m2, r: 6.674e-11 * m1 * m2 / r**2,
        "formula_latex": r"F = Gm_1m_2/r^2",
        "formula_sympy": "6.674e-11 * m1 * m2 / r**2",
        "note": "Extreme-scale: features span >6 orders of magnitude. FIX-D log-transform active.",
    },
    {
        "name": "Ideal Gas Law", "domain": "Physics", "vars": ["n", "T", "V"],
        "train_range":  [(0.1, 5.0), (200, 500), (0.01, 1.0)],
        "extrap_range": [(5.0, 20.0), (500, 1000), (1.0, 5.0)],
        "fn": lambda n, T, V: n * 8.314 * T / V,
        "formula_latex": r"P = nRT/V",
        "formula_sympy": "n * 8.314 * T / V",
    },
    # ── DeFi AMM (3) ─────────────────────────────────────────────────────────
    {
        "name": "Impermanent Loss", "domain": "DeFi AMM", "vars": ["r"],
        "train_range":  [(0.5, 2.0)],
        "extrap_range": [(2.0, 10.0)],
        "fn": lambda r: (2 * np.sqrt(r) / (1 + r) - 1) * 100,
        "formula_latex": r"\mathrm{IL} = 2\sqrt{r}/(1+r) - 1",
        "formula_sympy": "(2 * np.sqrt(r) / (1 + r) - 1) * 100",
    },
    {
        "name": "Price Impact", "domain": "DeFi AMM", "vars": ["dx", "x"],
        "train_range":  [(0.01, 10.0), (100, 1000)],
        "extrap_range": [(10.0, 100.0), (1000, 5000)],
        "fn": lambda dx, x: dx / (x + dx),
        "formula_latex": r"\Delta p = \Delta x/(x + \Delta x)",
        "formula_sympy": "dx / (x + dx)",
    },
    {
        "name": "Constant Product", "domain": "DeFi AMM", "vars": ["x"],
        "train_range":  [(1.0, 100.0)],
        "extrap_range": [(100.0, 500.0)],
        "fn": lambda x: 10000.0 / x,
        "formula_latex": r"y = k/x",
        "formula_sympy": "10000.0 / x",
    },
    # ── DeFi Risk (3) ────────────────────────────────────────────────────────
    {
        "name": "Value at Risk", "domain": "DeFi Risk", "vars": ["P", "sigma"],
        "train_range":  [(1000, 10000), (0.01, 0.1)],
        "extrap_range": [(10000, 50000), (0.1, 0.3)],
        "fn": lambda P, sigma: P * sigma * 1.645,
        "formula_latex": r"\mathrm{VaR} = P\sigma \cdot 1.645",
        "formula_sympy": "P * sigma * 1.645",
    },
    {
        "name": "Liquidation Price", "domain": "DeFi Risk", "vars": ["p0", "L"],
        "train_range":  [(100, 5000), (1.5, 5.0)],
        "extrap_range": [(5000, 20000), (5.0, 20.0)],
        "fn": lambda p0, L: p0 * (1 - 1 / (L * 0.8)),
        "formula_latex": r"p_{\mathrm{liq}} = p_0(1 - 1/(L \cdot m))",
        "formula_sympy": "p0 * (1 - 1 / (L * 0.8))",
    },
    {
        "name": "Portfolio Std Dev", "domain": "DeFi Risk", "vars": ["s1", "s2", "rho"],
        "train_range":  [(0.01, 0.2), (0.01, 0.2), (-0.8, 0.8)],
        "extrap_range": [(0.2, 0.5), (0.2, 0.5), (-0.9, 0.9)],
        "fn": lambda s1, s2, rho: np.sqrt(s1**2 + s2**2 + 2 * rho * s1 * s2),
        "formula_latex": r"\sigma_p = \sqrt{\sigma_1^2 + \sigma_2^2 + 2\rho\sigma_1\sigma_2}",
        "formula_sympy": "np.sqrt(s1**2 + s2**2 + 2 * rho * s1 * s2)",
        "note": "C4: run 5-seed sweep to report mean ± std. Seeds [42,99,123,777,2024].",
    },
]

# Stable ID map — checkpoint key never changes even if names are edited (FIX-KEY).
EQ_ID = {eq["name"]: i for i, eq in enumerate(CORE_15)}

assert len(CORE_15) == 15,          f"Expected 15 equations, got {len(CORE_15)}"
assert len(set(EQ_ID)) == 15,       "Duplicate equation names detected"
print(f"✅ Core-15 loaded: {len(CORE_15)} equations, "
      f"{len(set(eq['domain'] for eq in CORE_15))} domains")
for i, eq in enumerate(CORE_15):
    print(f"  [{i:02d}] {eq['name']:<28} ({eq['domain']})")

# =============================================================================
# §9 · Data generators
# =============================================================================

def generate_data(eq, N=200, noise_level=0.05, seed=42):
    dbg(f"generate_data: {eq['name']}  N={N}  noise={noise_level}  seed={seed}")
    rng    = np.random.RandomState(seed)
    n_vars = len(eq["vars"])
    X      = np.column_stack([rng.uniform(lo, hi, N) for lo, hi in eq["train_range"]])
    y      = eq["fn"](*[X[:, i] for i in range(n_vars)])
    y_noisy = y + rng.normal(0, noise_level * np.std(y), N)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_noisy, test_size=0.2, random_state=seed
    )
    dbg(f"generate_data done: X_train={X_train.shape}")
    return X_train, X_test, y_train, y_test, X, y


def generate_extrap_data(eq, regime="medium", N=100, seed=42):
    rng    = np.random.RandomState(seed + 999)
    n_vars = len(eq["vars"])
    if regime == "medium":
        ranges = eq["extrap_range"]
    else:
        mult_lo, mult_hi = {"near": (1.0, 1.5), "far": (4.0, 6.0)}[regime]
        ranges = [(hi * mult_lo, hi * mult_hi) for (_, hi) in eq["train_range"]]
    X = np.column_stack([rng.uniform(lo, hi, N) for lo, hi in ranges])
    y = eq["fn"](*[X[:, i] for i in range(n_vars)])
    return X, y

# =============================================================================
# §10 · True OOD evaluation via formula_sympy (C1 fix)
# =============================================================================

try:
    import sympy as _sp  # noqa: F401
    SYMPY_AVAILABLE = True
    print("✅ sympy available — true OOD evaluation (Mode A) enabled")
except ImportError:
    SYMPY_AVAILABLE = False
    print("⚠️  sympy not installed — OOD evaluation falls back to fn() proxy")


def evaluate_formula_ood(eq, X_ood):
    """
    Returns (y_true_ood, eval_mode).
    Mode A: evaluates formula_sympy string with numpy (genuine OOD).
    Mode B: falls back to eq['fn'] (precomputed proxy, same as generate_extrap_data).
    """
    formula = eq.get("formula_sympy")
    if formula and SYMPY_AVAILABLE:
        try:
            n_vars   = len(eq["vars"])
            var_vals = {eq["vars"][i]: X_ood[:, i] for i in range(n_vars)}
            y_true   = eval(formula, {"np": np, **var_vals})
            if np.all(np.isfinite(y_true)):
                return y_true, "sympy"
        except Exception as e:
            dbg(f"evaluate_formula_ood sympy failed for {eq['name']}: {e} — falling back")
    n_vars = len(eq["vars"])
    y_true = eq["fn"](*[X_ood[:, i] for i in range(n_vars)])
    return y_true, "precomputed"

# =============================================================================
# §11 · Metrics
# =============================================================================

def _safe_r2(y_true, y_pred):
    if not np.all(np.isfinite(y_pred)):
        return float("-inf")
    if len(y_true) < 2:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-10:
        return 1.0 if ss_res < 1e-10 else float("-inf")
    return float(1 - ss_res / ss_tot)


def _safe_rmse(y_true, y_pred):
    if not np.all(np.isfinite(y_pred)) or len(y_true) == 0:
        return float("inf")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# =============================================================================
# §12 · PySR builder (pysr_only condition)
# =============================================================================

_PYSR_VALID_PARAMS = None


def make_pysr(warm_start_expr=None, seed=42, niterations=None,
              timeout_secs=None, populations=None):
    if not PYSR_AVAILABLE:
        raise RuntimeError("PySR not installed")
    global _PYSR_VALID_PARAMS
    if _PYSR_VALID_PARAMS is None:
        _PYSR_VALID_PARAMS = set(inspect.signature(PySRRegressor.__init__).parameters.keys())
    valid = _PYSR_VALID_PARAMS

    _niter      = niterations   if niterations  is not None else NITERATIONS
    _pops       = populations   if populations   is not None else POPULATIONS
    _timeout    = timeout_secs  if timeout_secs  is not None else PYSR_TIMEOUT_SECS

    kwargs = dict(
        niterations    = _niter,
        populations    = _pops,
        population_size= 50,
        maxsize        = 15,
        parsimony      = 0.02,
        binary_operators = ["+", "-", "*", "/"],
        unary_operators  = ["exp", "log", "sin", "cos", "sqrt"],
        random_state   = seed,
        verbosity      = 0,
        progress       = False,
    )
    if "parallelism" in valid:
        kwargs["parallelism"] = "serial"
    else:
        kwargs["procs"]         = 0
        kwargs["multithreading"]= False
    if "timeout_in_seconds" in valid:
        kwargs["timeout_in_seconds"] = _timeout
    if "tournament_selection_n" in valid:
        kwargs["tournament_selection_n"] = 3
    if "crossover_probability" in valid:
        kwargs["crossover_probability"] = 0.9
    if "batching" in valid:
        kwargs["batching"]    = True
        kwargs["batch_size"]  = 50
    if warm_start_expr is not None:
        if "warm_start" in valid:
            kwargs["warm_start"] = True
        if "initial_expressions" in valid:
            kwargs["initial_expressions"] = [warm_start_expr]

    dbg(f"make_pysr: pops={_pops}  iter={_niter}  timeout={_timeout}s"
        f"  warm={warm_start_expr is not None}")
    return PySRRegressor(**kwargs)

# =============================================================================
# §13 · Timeout context manager (SIGALRM, Unix only)
# =============================================================================

class _Timeout:
    """Raises TimeoutError after `seconds` on Unix via SIGALRM (main-thread safe)."""

    def __init__(self, seconds: int):
        self.seconds  = int(seconds)
        self._ok      = hasattr(signal, "SIGALRM")

    def _handler(self, signum, frame):
        raise TimeoutError(f"Wall-clock limit of {self.seconds}s exceeded")

    def __enter__(self):
        if self._ok and self.seconds > 0:
            signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, *args):
        if self._ok:
            signal.alarm(0)

if not hasattr(signal, "SIGALRM"):
    print("⚠️  SIGALRM unavailable (Windows?). Wall-clock safety net disabled.")

# =============================================================================
# §14 · Checkpoint helpers
# =============================================================================

def load_checkpoint(path) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        n = sum(1 for v in data.values() if isinstance(v, dict))
        print(f"📂 Checkpoint: {n} equations already done  ({path})")
        return data
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"⚠️  Checkpoint unreadable ({e}) — starting fresh")
        return {}


def save_checkpoint(path, results: dict) -> None:
    tmp = str(path) + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, path)
        dbg(f"Checkpoint saved: {path}")
    except OSError as e:
        print(f"⚠️  Checkpoint save failed: {e}")

# =============================================================================
# §15 · run_condition — single equation × single condition
# =============================================================================

def run_condition(eq, condition, seed=42, niterations=None,
                  timeout_secs=None, populations=None, wall_clock_limit=None):
    """
    condition='pysr_only' : raw PySRRegressor (clean baseline).
    condition='hypatia'   : HybridDiscoverySystem v5.1 (FIX-A…FIX-POW all active).

    Wall-clock caps are per-condition (FIX-WALLCLOCK):
      hypatia   → HYPATIA_WALL_CLOCK   (3 × PYSR_TIMEOUT + post-proc)
      pysr_only → PYSR_ONLY_WALL_CLOCK (1 × PYSR_TIMEOUT + post-proc)
    """
    _niter   = niterations  if niterations  is not None else NITERATIONS
    _timeout = timeout_secs if timeout_secs is not None else PYSR_TIMEOUT_SECS
    _pops    = populations  if populations  is not None else POPULATIONS

    if wall_clock_limit is None:
        wall_clock_limit = (
            HYPATIA_WALL_CLOCK if condition == "hypatia" else PYSR_ONLY_WALL_CLOCK
        )

    eq_seed = seed + EQ_ID.get(eq["name"], 0) * 7
    dbg(f"run_condition START: {eq['name']} [{condition}]  "
        f"seed={eq_seed}  wall_clock={wall_clock_limit}s")

    # ── Data ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, _, _ = generate_data(
        eq, N=200, noise_level=0.05, seed=eq_seed
    )
    extrap_sets = {}
    for regime_name, _ in EXTRAP_REGIMES:
        X_e, y_e = generate_extrap_data(eq, regime=regime_name, N=100, seed=eq_seed)
        extrap_sets[regime_name] = (X_e, y_e)

    # ── Variable sanitisation ─────────────────────────────────────────────────
    sanitizer = VariableNameSanitizer()
    safe_var_names, had_conflict = sanitizer.sanitize(eq["vars"])
    if had_conflict:
        dbg(f"Variable sanitisation: {sanitizer.log()}")

    # =========================================================================
    # Branch A: hypatia — route through HybridDiscoverySystem v5.1
    # =========================================================================
    if condition == "hypatia":
        if HybridDiscoverySystem is None:
            raise RuntimeError(
                "HybridDiscoverySystem not imported — check hybrid_system_v50_2.py path"
            )
        _api_key  = os.environ.get("ANTHROPIC_API_KEY", "")
        _use_llm  = _api_key.startswith("sk-")
        _dom_slug = eq["domain"].lower().replace(" ", "_")

        hybrid = HybridDiscoverySystem(
            domain          = _dom_slug,
            discovery_config= DiscoveryConfig(
                niterations              = _niter,
                enable_auto_configuration= True,
                pysr_timeout             = _timeout,
            ),
            max_retries             = _HYPATIA_MAX_RETRIES,
            use_llm                 = _use_llm,
            llm_mode                = "hybrid" if _use_llm else "none",
            llm_n_candidates        = 3,
            llm_temperature         = 0.3,
            enable_physics_fallback = False,
            allow_nondeterministic  = False,   # FIX-C
        )

        t0 = time.time()
        try:
            with _Timeout(wall_clock_limit):
                res = hybrid.discover(
                    X_train, y_train,
                    var_names   = safe_var_names,
                    description = eq["name"],
                    metadata    = {
                        "equation_name":          eq["name"],
                        "variable_descriptions":  {v: v for v in eq["vars"]},
                        "variable_units":         {v: "" for v in eq["vars"]},
                        "noise_level":            0.05,
                        "domain":                 _dom_slug,
                    },
                )
            sr_time = time.time() - t0
        except TimeoutError:
            sr_time = time.time() - t0
            print(f"  [hypatia] ⏰ TIMEOUT after {sr_time:.1f}s")
            return _build_timeout_result("hypatia", sr_time)
        except Exception as e:
            sr_time = time.time() - t0
            print(f"  [hypatia] ❌ ERROR: {e}")
            traceback.print_exc()
            return _build_error_result("hypatia", str(e), sr_time)

        # FIX-B: resolve key aliases from v5.1 discover() result
        best_expr_raw = sanitizer.restore(res.get("formula", "N/A"))
        train_r2      = res.get("r2")
        train_rmse    = res.get("rmse", float("inf"))   # FIX-A: original-unit RMSE
        scale_log     = res.get("scale_log", False)     # FIX-D flag
        complexity    = len(best_expr_raw) if best_expr_raw not in ("N/A", "") else None

        # Extrapolation: evaluate discovered expression directly.
        # Apply same log-scale transform to OOD X if discover() did so for training.
        extrap_r2, extrap_rmse = {}, {}
        _norm_expr = best_expr_raw.replace("^", "**")
        _math_ns   = {
            "sqrt": np.sqrt, "exp": np.exp, "log": np.log,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "abs": np.abs, "log2": np.log2, "log10": np.log10,
            "pow": np.power, "sign": np.sign,
        }
        for regime_name, (X_e, y_e) in extrap_sets.items():
            try:
                X_e_use = X_e.copy().astype(float)
                if scale_log:
                    for ci, _vn in enumerate(safe_var_names):
                        col  = X_e[:, ci]
                        _abs = np.abs(col[np.isfinite(col) & (col != 0)])
                        if len(_abs) >= 2:
                            _rng = (
                                np.log10(_abs.max() + 1e-30)
                                - np.log10(_abs.min() + 1e-30)
                            )
                            if _rng > 6:
                                X_e_use[:, ci] = np.sign(col) * np.log10(np.abs(col) + 1.0)
                _ns    = {"np": np, **_math_ns,
                          **{vn: X_e_use[:, ci] for ci, vn in enumerate(safe_var_names)}}
                y_pred = eval(_norm_expr, {"__builtins__": {}}, _ns)
                y_pred = np.asarray(y_pred, dtype=float)
                if y_pred.shape == ():
                    y_pred = np.full(len(y_e), float(y_pred))
                extrap_r2[regime_name]   = _safe_r2(y_e, y_pred)
                extrap_rmse[regime_name] = _safe_rmse(y_e, y_pred)
            except Exception as ee:
                dbg(f"extrap eval failed ({regime_name}): {ee}")
                extrap_r2[regime_name]   = None
                extrap_rmse[regime_name] = None

        def _f(v):
            if v is None: return "N/A"
            try: return f"{float(v):.4f}" if np.isfinite(float(v)) else "N/A"
            except Exception: return "N/A"

        print(
            f"  [hypatia] {best_expr_raw[:55]}  R²={_f(train_r2)}  "
            f"extrap(near={_f(extrap_r2.get('near'))} "
            f"med={_f(extrap_r2.get('medium'))} "
            f"far={_f(extrap_r2.get('far'))})  {sr_time:.1f}s"
            f"  scale_log={scale_log}  llm={res.get('llm_mode','none')}"
        )
        return {
            "condition":             "hypatia",
            "success":               res.get("success", False),
            "timed_out":             False,
            "excluded_from_timing":  False,
            "train_r2":              train_r2,
            "train_rmse":            train_rmse,
            "extrap_r2_near":        extrap_r2.get("near"),
            "extrap_r2_medium":      extrap_r2.get("medium"),
            "extrap_r2_far":         extrap_r2.get("far"),
            "extrap_rmse_near":      extrap_rmse.get("near"),
            "extrap_rmse_medium":    extrap_rmse.get("medium"),
            "extrap_rmse_far":       extrap_rmse.get("far"),
            "sr_time_s":             sr_time,
            "llm_time_s":            0.0,
            "total_time_s":          sr_time,
            "best_expression":       best_expr_raw,
            "complexity":            complexity,
            "llm_expression":        None,
            "llm_confidence":        None,
            # v5.4 provenance extras
            "engine_version":        HybridDiscoverySystem.VERSION,
            "scale_log":             scale_log,
            "rmse_original_units":   train_rmse,
            "llm_mode_used":         res.get("llm_mode", "none"),
            "validation_score":      res.get("validation_score"),
        }

    # =========================================================================
    # Branch B: pysr_only — raw PySRRegressor (clean baseline, unchanged)
    # =========================================================================
    model = make_pysr(
        warm_start_expr = None,
        seed            = eq_seed,
        niterations     = _niter,
        timeout_secs    = _timeout,
        populations     = _pops,
    )
    t_sr = time.time()
    try:
        with _Timeout(wall_clock_limit):
            model.fit(X_train, y_train, variable_names=safe_var_names)
        sr_time = time.time() - t_sr
    except TimeoutError:
        sr_time = time.time() - t_sr
        print(f"  [pysr_only] ⏰ TIMEOUT after {sr_time:.1f}s")
        return _build_timeout_result("pysr_only", sr_time)
    except Exception as e:
        sr_time = time.time() - t_sr
        print(f"  [pysr_only] ❌ ERROR: {e}")
        return _build_error_result("pysr_only", str(e), sr_time)

    try:
        y_pred_train = model.predict(X_train)
        train_r2     = _safe_r2(y_train, y_pred_train)
        train_rmse   = _safe_rmse(y_train, y_pred_train)

        extrap_r2, extrap_rmse = {}, {}
        for regime_name, (X_e, y_e) in extrap_sets.items():
            y_pred_e                  = model.predict(X_e)
            extrap_r2[regime_name]    = _safe_r2(y_e, y_pred_e)
            extrap_rmse[regime_name]  = _safe_rmse(y_e, y_pred_e)

        best_expr_raw = str(model.sympy())
        best_expr     = sanitizer.restore(best_expr_raw)
        try:
            complexity = int(model.get_best()["complexity"])
        except Exception:
            eqs_df     = model.equations_
            complexity = int(eqs_df.loc[eqs_df["loss"].idxmin(), "complexity"])
    except Exception:
        traceback.print_exc()
        train_r2 = train_rmse = None
        extrap_r2   = {r: None for r, _ in EXTRAP_REGIMES}
        extrap_rmse = {r: None for r, _ in EXTRAP_REGIMES}
        best_expr   = "evaluation_error"
        complexity  = None

    def _f(v): return f"{v:.4f}" if v is not None and np.isfinite(v) else "N/A"
    total_time = time.time() - t_sr   # llm_time=0 for pysr_only
    print(
        f"  [pysr_only] train_R²={_f(train_r2)}  "
        f"extrap(near={_f(extrap_r2.get('near'))} "
        f"med={_f(extrap_r2.get('medium'))} "
        f"far={_f(extrap_r2.get('far'))})  time={total_time:.1f}s"
    )
    return {
        "condition":             "pysr_only",
        "success":               True,
        "timed_out":             False,
        "excluded_from_timing":  False,
        "train_r2":              train_r2,
        "train_rmse":            train_rmse,
        "extrap_r2_near":        extrap_r2.get("near"),
        "extrap_r2_medium":      extrap_r2.get("medium"),
        "extrap_r2_far":         extrap_r2.get("far"),
        "extrap_rmse_near":      extrap_rmse.get("near"),
        "extrap_rmse_medium":    extrap_rmse.get("medium"),
        "extrap_rmse_far":       extrap_rmse.get("far"),
        "sr_time_s":             sr_time,
        "llm_time_s":            0.0,
        "total_time_s":          total_time,
        "best_expression":       best_expr,
        "complexity":            complexity,
        "llm_expression":        None,
        "llm_confidence":        None,
    }


# ── Shared timeout / error result builders ────────────────────────────────────

def _build_timeout_result(condition: str, sr_time: float) -> dict:
    base = dict.fromkeys(
        ["train_r2", "train_rmse",
         "extrap_r2_near", "extrap_r2_medium", "extrap_r2_far",
         "extrap_rmse_near", "extrap_rmse_medium", "extrap_rmse_far",
         "llm_expression", "llm_confidence", "complexity"],
        None,
    )
    return {
        **base,
        "condition":            condition,
        "success":              False,
        "timed_out":            True,
        "excluded_from_timing": True,
        "sr_time_s":            sr_time,
        "llm_time_s":           0.0,
        "total_time_s":         sr_time,
        "best_expression":      "TIMED_OUT",
    }


def _build_error_result(condition: str, error: str, sr_time: float) -> dict:
    base = dict.fromkeys(
        ["train_r2", "train_rmse",
         "extrap_r2_near", "extrap_r2_medium", "extrap_r2_far",
         "extrap_rmse_near", "extrap_rmse_medium", "extrap_rmse_far",
         "llm_expression", "llm_confidence", "complexity"],
        None,
    )
    return {
        **base,
        "condition":            condition,
        "success":              False,
        "timed_out":            False,
        "excluded_from_timing": False,
        "error":                error,
        "sr_time_s":            sr_time,
        "llm_time_s":           0.0,
        "total_time_s":         sr_time,
        "best_expression":      "ERROR",
    }

print("✅ run_condition ready — hypatia → HybridDiscoverySystem v5.1 | pysr_only → raw PySR")

# =============================================================================
# §16 · Main experiment loop
# =============================================================================

def run_experiment():
    all_results = load_checkpoint(CKPT_PATH)

    # Smoke-test / partial-run support from env vars
    _n_core15 = os.environ.get("N_CORE15_TASKS")
    _one_eq   = os.environ.get("ONE_EQUATION", "0") == "1"
    _n_run    = int(_n_core15) if _n_core15 is not None else (1 if _one_eq else len(CORE_15))
    _CORE_15_RUN = CORE_15[:_n_run]
    if _n_run < len(CORE_15):
        print(f"⚠  Smoke-test: running {_n_run}/{len(CORE_15)} equations")

    # ---------------------------------------------------------------------------
    # Shard partitioning
    # ---------------------------------------------------------------------------
    # Primary mechanism: SHARD_INDEX + N_SHARDS environment variables.
    # The CI sets these for every worker; index-based round-robin guarantees
    # every equation is covered by exactly one shard, regardless of slug names.
    #
    # Legacy / override mechanism: DOMAIN_FILTER (or PENDING_IDS) accepts
    # a JSON array or space/comma string of slugs.  When the slugs match Core-15
    # domain or name strings they override the index-based split.
    #
    # NOTE on the current CI: the Plan script assigns DEFI_TASKS slugs
    # (amm, risk_var, liquidity, expected_shortfall, liquidation, risk, lending,
    # staking, trading, derivatives) to exp1_ablation shards.  These slugs were
    # copied from exp1 (DeFi benchmark) and do NOT cover the 9 Chemistry/Biology/
    # Physics equations in Core-15.  Index-based sharding is therefore the
    # correct default; the slug map below handles the DEFI_TASKS slugs so that
    # if the Plan is not updated the DeFi equations are still correctly partitioned.
    #
    # Environment variables:
    #   SHARD_INDEX  – 0-based shard number  (default: 0)
    #   N_SHARDS     – total number of shards (default: 1)
    #   DOMAIN_FILTER / PENDING_IDS – optional slug override (see above)

    _shard_index = int(os.environ.get("SHARD_INDEX", os.environ.get("SHARD", 0)))
    _n_shards    = int(os.environ.get("N_SHARDS", 1))

    # Check for a slug-based override first.
    _domain_filter_raw = (
        os.environ.get("DOMAIN_FILTER", "").strip()
        or os.environ.get("PENDING_IDS", "").strip()
    )
    _shard_slugs: list[str] = []
    if _domain_filter_raw:
        _stripped = _domain_filter_raw.strip()
        if _stripped.startswith("["):
            try:
                import json as _json
                _shard_slugs = [s.lower() for s in _json.loads(_stripped)]
            except Exception:
                _shard_slugs = [s.lower() for s in _stripped.strip("[]").replace(",", " ").split()]
        else:
            _shard_slugs = [s.lower().strip(",") for s in _stripped.replace(",", " ").split()]
        _shard_slugs = [s for s in _shard_slugs if s]

    # Map domain slugs (from Plan script) to Core-15 equation name keywords.
    # Slugs not in this map fall back to substring matching on domain+name.
    _SLUG_TO_EQ_KEYWORDS: dict[str, list[str]] = {
        # ── Canonical slugs (ci_runner.yml registry, current) ─────────────────
        "chemistry":          ["arrhenius", "henderson-hasselbalch", "rate law"],
        "biology":            ["allometric scaling", "michaelis-menten", "logistic growth"],
        "physics":            ["kinetic energy", "gravitational force", "ideal gas law"],
        "defi_amm":           ["impermanent loss", "price impact", "constant product"],
        "defi_risk":          ["value at risk", "liquidation price", "portfolio std dev"],
        # ── Legacy DEFI_TASKS slugs (ci_runner.yml registry, old) ─────────────
        # Kept for backwards-compatibility in case the Plan script is not yet updated.
        "amm":                ["impermanent loss", "price impact", "constant product"],
        "liquidity":          ["impermanent loss", "price impact", "constant product"],
        "risk_var":           ["value at risk"],
        "expected_shortfall": ["value at risk"],
        "liquidation":        ["liquidation price"],
        "risk":               ["portfolio std dev"],
        "lending":            [],   # no Core-15 equation
        "staking":            [],
        "trading":            [],
        "derivatives":        [],
    }

    if _shard_slugs:
        # Resolve slugs to equation name keywords, then filter _CORE_15_RUN
        _target_keywords: set[str] = set()
        _fallback_slugs:  list[str] = []
        for _slug in _shard_slugs:
            if _slug in _SLUG_TO_EQ_KEYWORDS:
                _target_keywords.update(_SLUG_TO_EQ_KEYWORDS[_slug])
            else:
                _fallback_slugs.append(_slug)

        def _eq_matches_slugs(eq: dict) -> bool:
            name_lower   = eq.get("name",   "").lower()
            domain_lower = eq.get("domain", "").lower()
            if any(kw == name_lower for kw in _target_keywords):
                return True
            haystack = domain_lower + " " + name_lower
            return any(fs in haystack for fs in _fallback_slugs)

        _CORE_15_RUN = [eq for eq in _CORE_15_RUN if _eq_matches_slugs(eq)]
        print(f"🔀 Shard filter (slug mode, slugs={_shard_slugs!r}): "
              f"{len(_CORE_15_RUN)}/{len(CORE_15)} equations selected"
              + (f" → {[eq['name'] for eq in _CORE_15_RUN]}"
                 if _CORE_15_RUN else " (none — shard idle)"))

    elif _n_shards > 1:
        # Index-based round-robin: shard i runs equations i, i+n, i+2n, …
        _CORE_15_RUN = [
            eq for j, eq in enumerate(_CORE_15_RUN)
            if j % _n_shards == _shard_index
        ]
        print(f"🔀 Shard filter (index mode, shard {_shard_index}/{_n_shards}): "
              f"{len(_CORE_15_RUN)}/{len(CORE_15)} equations selected"
              f" → {[eq['name'] for eq in _CORE_15_RUN]}")

    print("=" * 65)
    print("EXPERIMENT 1: LLM ABLATION  (§10.6 Core-15)")
    print(f"Engine      : hybrid_system_v50_2 (v5.1)")
    print(f"Equations   : {_n_run}  |  Conditions: {CONDITIONS}")
    print(f"populations : {POPULATIONS}  |  iterations: {NITERATIONS}  |  timeout: {TIMEOUT_SECS}s")
    print(f"Resuming    : {len(all_results)} checkpointed entries")
    print("=" * 65)

    for eq_idx, eq in enumerate(_CORE_15_RUN):
        eq_key = str(eq_idx)   # string key for JSON compatibility (FIX-KEY)
        all_results.setdefault(eq_key, {"name": eq["name"], "domain": eq["domain"]})
        entry  = all_results[eq_key]

        to_run = [c for c in CONDITIONS if entry.get(c) is None]
        if not to_run:
            print(f"✓ [{eq_idx:02d}] {eq['name']} — already done")
            continue

        print(f"\n{'─' * 60}")
        print(f"▶  [{eq_idx:02d}] {eq['name']}  [{eq['domain']}]")
        if eq.get("note"):
            print(f"   NOTE: {eq['note'][:100]}")
        print(f"{'─' * 60}")

        for cond in to_run:
            result = run_condition(
                eq, cond, seed=SEED,
                niterations=NITERATIONS,
                timeout_secs=PYSR_TIMEOUT_SECS,
                populations=POPULATIONS,
            )
            entry[cond]          = result
            all_results[eq_key]  = entry
            save_checkpoint(CKPT_PATH, all_results)
            print(f"  ✓ {cond} saved to checkpoint")

        # Arrhenius failure-mode detection
        p_r2 = all_results[eq_key].get("pysr_only", {}).get("train_r2")
        h_r2 = all_results[eq_key].get("hypatia",   {}).get("train_r2")
        if (p_r2 and h_r2 and np.isfinite(p_r2) and np.isfinite(h_r2)
                and p_r2 - h_r2 > 0.05):
            print(f"\n  ⚠️  FAILURE MODE — {eq['name']}: "
                  f"HypatiaX R²={h_r2:.3f} < PySR-only R²={p_r2:.3f}  "
                  f"(Arrhenius pattern — document in §Analysis)")

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    n_done = sum(
        1 for v in all_results.values()
        if isinstance(v, dict) and v.get("pysr_only") and v.get("hypatia")
    )
    print(f"\n✅ Results saved → {RESULTS_PATH}")
    print(f"   Complete: {n_done}/{len(CORE_15)} equations (both conditions)")
    return all_results

# =============================================================================
# §17 · Summary table (console)
# =============================================================================

def print_summary(all_results: dict) -> None:
    print(f"\n{'─' * 80}")
    print(f"{'Eq':25s}  {'P_near':>7} {'H_near':>7}  {'P_med':>7} {'H_med':>7}  "
          f"{'P_far':>7} {'H_far':>7}  {'P_time':>8} {'H_time':>8}")
    print("─" * 80)

    def _f(v):
        if v is None: return "    N/A"
        try:
            fv = float(v)
            return f"{fv:7.4f}" if np.isfinite(fv) else "    N/A"
        except Exception:
            return "    N/A"

    H_far_all, P_far_all, speedups = [], [], []
    for eq_idx, eq in enumerate(CORE_15):
        res = all_results.get(str(eq_idx), {})
        p   = res.get("pysr_only", {}) or {}
        h   = res.get("hypatia",   {}) or {}
        pf  = p.get("extrap_r2_far")
        hf  = h.get("extrap_r2_far")
        pt  = p.get("total_time_s", 0) or 0
        ht  = h.get("total_time_s", 0) or 0
        print(
            f"{eq['name']:25s}  "
            f"{_f(p.get('extrap_r2_near'))} {_f(h.get('extrap_r2_near'))}  "
            f"{_f(p.get('extrap_r2_medium'))} {_f(h.get('extrap_r2_medium'))}  "
            f"{_f(pf)} {_f(hf)}  "
            f"{pt:>8.1f} {ht:>8.1f}"
        )
        if pf is not None and hf is not None:
            P_far_all.append(pf); H_far_all.append(hf)
        if pt and ht and not p.get("excluded_from_timing") and not h.get("excluded_from_timing"):
            speedups.append(pt / ht)

    if P_far_all:
        print("─" * 80)
        print(f"{'Mean (far OOD)':25s}  {'':>7} {'':>7}  {'':>7} {'':>7}  "
              f"{np.mean(P_far_all):7.4f} {np.mean(H_far_all):7.4f}")
    if speedups:
        print(f"\nSpeedup (PySR-time / HypatiaX-time):  "
              f"mean={np.mean(P_far_all)/np.mean(H_far_all):.2f}×  "
              f"median={np.median(speedups):.2f}×  n={len(speedups)}")

# =============================================================================
# §18 · RF-01 — Mann-Whitney U + Wilcoxon (§10.6 statistics)
# =============================================================================

def run_rf01(all_results: dict) -> None:
    from collections import defaultdict
    import textwrap
    from scipy import stats as scipy_stats

    ALPHA = 0.05

    H_far, P_far, pairs, skipped = [], [], [], []
    for eq_idx, eq in enumerate(CORE_15):
        res = all_results.get(str(eq_idx), {})
        p   = res.get("pysr_only", {}) or {}
        h   = res.get("hypatia",   {}) or {}
        pf  = p.get("extrap_r2_far")
        hf  = h.get("extrap_r2_far")
        if (pf is not None and hf is not None
                and np.isfinite(float(pf)) and np.isfinite(float(hf))):
            P_far.append(float(pf))
            H_far.append(float(hf))
            pairs.append((
                eq["name"], eq["domain"],
                p.get("extrap_r2_near"), h.get("extrap_r2_near"),
                p.get("extrap_r2_medium"), h.get("extrap_r2_medium"),
                pf, hf,
            ))
        else:
            reasons = []
            if pf is None or not np.isfinite(float(pf)): reasons.append(f"P far-R²={pf}")
            if hf is None or not np.isfinite(float(hf)): reasons.append(f"H far-R²={hf}")
            skipped.append((eq["name"], eq["domain"], "; ".join(reasons)))

    n_pairs = len(pairs)
    H_arr, P_arr = np.array(H_far), np.array(P_far)

    print(f"\nRF-01 pairs: {n_pairs}  skipped: {[s[0] for s in skipped] or 'none'}")
    if n_pairs < 3:
        print(f"⚠  Only {n_pairs} pair(s) — need ≥ 3 to compute statistics.")
        return

    U_gt,    p_gt    = scipy_stats.mannwhitneyu(H_far, P_far, alternative="greater")
    U_two,   p_two   = scipy_stats.mannwhitneyu(H_far, P_far, alternative="two-sided")
    try:
        W_stat,  p_wil   = scipy_stats.wilcoxon(H_far, P_far, alternative="two-sided")
        _,       p_wil_gt= scipy_stats.wilcoxon(H_far, P_far, alternative="greater")
        W_stat = float(W_stat)
    except Exception as e:
        W_stat = p_wil = p_wil_gt = None
        print(f"  Wilcoxon N/A: {e}")

    r_rb    = float(1 - (2 * U_gt) / (n_pairs * n_pairs))
    h_wins  = int(np.sum(H_arr > P_arr + 0.01))
    p_wins  = int(np.sum(P_arr > H_arr + 0.01))
    ties    = n_pairs - h_wins - p_wins
    IS_SIG  = bool(p_gt < ALPHA)

    print(f"\n{'═' * 72}")
    print("AGGREGATE STATISTICAL TESTS  (RF-01)")
    print(f"{'═' * 72}")
    print(f"  n (paired)                 : {n_pairs}")
    print(f"  H Far-R²  median/mean      : {np.median(H_arr):+.4f} / {np.mean(H_arr):+.4f}")
    print(f"  P Far-R²  median/mean      : {np.median(P_arr):+.4f} / {np.mean(P_arr):+.4f}")
    print(f"  H wins / ties / P wins     : {h_wins} / {ties} / {p_wins}")
    print(f"  Mann-Whitney U (H > P)     : U={U_gt:.0f},  p={p_gt:.4f}")
    print(f"  Mann-Whitney U (two-sided) : U={U_two:.0f},  p={p_two:.4f}")
    if W_stat is not None:
        print(f"  Wilcoxon signed-rank       : W={W_stat:.0f}, p(2-sided)={p_wil:.4f},"
              f" p(H>P)={p_wil_gt:.4f}")
    print(f"  Rank-biserial r            : {r_rb:+.3f}")
    print(f"  Significant (α=0.05)       : {'YES ✅' if IS_SIG else 'NO ⚠'}")
    print(f"  Paper targets              : U=126, p=0.295 (Run A / seed=42 / pops=30)")

    # Sub-domain breakdown
    domain_data = defaultdict(lambda: {"h_wins": 0, "p_wins": 0, "ties": 0,
                                        "delta_sum": 0.0, "n": 0, "eqs": []})
    for (eq, dom, pn, hn, pm, hm, pf, hf) in pairs:
        delta = hf - pf
        d = domain_data[dom]
        d["n"] += 1; d["delta_sum"] += delta; d["eqs"].append(eq)
        if delta > 0.01:   d["h_wins"] += 1
        elif delta < -0.01: d["p_wins"] += 1
        else:               d["ties"] += 1

    driving_doms = [dom for dom, d in domain_data.items() if d["h_wins"] > d["p_wins"]]
    p_dom_doms   = [dom for dom, d in domain_data.items() if d["p_wins"] > d["h_wins"]]

    # Paper sentence
    if IS_SIG:
        sentence = (
            f"HypatiaX achieves significantly higher far-extrapolation $R^2$ than "
            f"PySR-only across the Core-15 benchmark "
            f"(Mann-Whitney $U={U_gt:.0f}$, $p={p_gt:.4f}$, two-tailed $p={p_two:.4f}$, "
            f"$n={n_pairs}$; rank-biserial $r={r_rb:+.2f}$). "
            f"HypatiaX outperforms PySR-only on {h_wins}/{n_pairs} equations "
            f"({', '.join(driving_doms) if driving_doms else 'multiple'} domains), "
            f"with {ties} tie(s) and {p_wins} equation(s) where PySR-only is superior"
            + (f" ({', '.join(p_dom_doms)})." if p_dom_doms else ".")
        )
        verdict = "✅  SIGNIFICANT"
    else:
        if driving_doms:
            dom_detail = (
                f"The directional improvement is concentrated in "
                f"{len(driving_doms)} domain(s) — {', '.join(driving_doms)}"
            )
            if p_dom_doms:
                dom_detail += (
                    f"; PySR-only is superior in {', '.join(p_dom_doms)}, "
                    "indicating HypatiaX's benefit is domain-selective rather than universal."
                )
            else:
                dom_detail += ", suggesting benefit is domain-selective."
        else:
            dom_detail = "No single domain consistently drives the improvement."
        sentence = (
            f"Across Core-15, HypatiaX does not achieve statistically significant "
            f"higher far-extrapolation $R^2$ than PySR-only "
            f"(Mann-Whitney $U={U_gt:.0f}$, $p={p_gt:.4f}$, $n={n_pairs}$; "
            f"rank-biserial $r={r_rb:+.2f}$). " + dom_detail
        )
        verdict = "⚠️  NOT SIGNIFICANT"

    print(f"\n  {verdict}\n")
    for line in textwrap.wrap(sentence, width=78, subsequent_indent="  "):
        print(f"  {line}")

    # Save JSON stat record
    stat_record = {
        "rf01_mann_whitney": {
            "n_pairs": n_pairs, "n_skipped": len(skipped),
            "skipped_equations": [s[0] for s in skipped],
            "U_greater": float(U_gt),     "p_greater": float(p_gt),
            "U_two_sided": float(U_two),  "p_two_sided": float(p_two),
            "wilcoxon_W": W_stat,
            "wilcoxon_p_two": float(p_wil)    if p_wil    is not None else None,
            "wilcoxon_p_greater": float(p_wil_gt) if p_wil_gt is not None else None,
            "rank_biserial_r": r_rb,
            "H_median_far_r2": float(np.median(H_arr)),
            "P_median_far_r2": float(np.median(P_arr)),
            "H_mean_far_r2":   float(np.mean(H_arr)),
            "P_mean_far_r2":   float(np.mean(P_arr)),
            "h_wins": h_wins, "p_wins": p_wins, "ties": ties,
            "significant_p05": IS_SIG,
            "paper_sentence":  sentence,
            "equations_included": [r[0] for r in pairs],
            "H_far_r2_vector": [float(v) for v in H_far],
            "P_far_r2_vector": [float(v) for v in P_far],
            "subdomain_breakdown": {
                dom: {
                    "n": d["n"], "h_wins": d["h_wins"],
                    "ties": d["ties"], "p_wins": d["p_wins"],
                    "mean_delta": float(d["delta_sum"] / d["n"]),
                    "equations": d["eqs"],
                }
                for dom, d in domain_data.items()
            },
            "driving_domains":   driving_doms,
            "p_dominant_domains": p_dom_doms,
        }
    }
    with open(RF01_JSON, "w") as f:
        json.dump(stat_record, f, indent=2)
    print(f"\n  📄  RF-01 JSON → {RF01_JSON}")

    # LaTeX significance table
    def _lv(v, d=3):
        if v is None or not np.isfinite(float(v) if v is not None else float("nan")):
            return r"\textemdash"
        return f"${float(v):+.{d}f}$"

    tex_sig = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{RF-01: Far-extrapolation $R^2$ per equation. "
        + f"Mann-Whitney $U={U_gt:.0f}$, $p_{{H>P}}={p_gt:.4f}$ ($n={n_pairs}$), "
        + f"rank-biserial $r={r_rb:+.2f}$: "
        + ("HypatiaX significantly outperforms ($p<0.05$).}"
           if IS_SIG else "difference not significant ($p\\geq0.05$).}"),
        r"\label{tab:rf01_significance}", r"\small",
        r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
        r"\textbf{Equation} & \textbf{Domain} & "
        r"\textbf{P Near} & \textbf{H Near} & \textbf{P Med} & \textbf{H Med} & "
        r"\textbf{P Far} & \textbf{H Far} & $\boldsymbol{\Delta}$ & \textbf{Win} \\",
        r"\midrule",
    ]
    for (eq, dom, pn, hn, pm, hm, pf, hf) in pairs:
        delta = hf - pf
        win   = (r"\textbf{H}" if delta > 0.01 else
                 r"\textbf{P}" if delta < -0.01 else r"Tie")
        tex_sig.append(
            f"{eq.replace('_', r'_')} & {dom} & "
            f"{_lv(pn)} & {_lv(hn)} & {_lv(pm)} & {_lv(hm)} & "
            f"{_lv(pf)} & {_lv(hf)} & ${delta:+.3f}$ & {win} \\\\"
        )
    tex_sig += [
        r"\midrule",
        f"\\textbf{{Mean}} & & & & & & ${np.mean(P_arr):+.3f}$ & ${np.mean(H_arr):+.3f}$"
        f" & ${np.mean(H_arr)-np.mean(P_arr):+.3f}$ & \\\\",
        f"\\textbf{{Median}} & & & & & & ${np.median(P_arr):+.3f}$ & ${np.median(H_arr):+.3f}$"
        f" & ${np.median(H_arr)-np.median(P_arr):+.3f}$ & \\\\",
        r"\midrule",
        f"\\multicolumn{{10}}{{l}}{{\\textbf{{Mann-Whitney}} $U={U_gt:.0f}$, "
        f"$p_{{H>P}}={p_gt:.4f}$, two-sided $p={p_two:.4f}$, $n={n_pairs}$, "
        f"rank-biserial $r={r_rb:+.2f}$, "
        f"H/ties/P: {h_wins}/{ties}/{p_wins}}} \\\\",
    ]
    if W_stat is not None:
        tex_sig.append(
            f"\\multicolumn{{10}}{{l}}{{\\textbf{{Wilcoxon}} "
            f"$W={W_stat:.0f}$, $p_{{two}}={p_wil:.4f}$, "
            f"$p_{{H>P}}={p_wil_gt:.4f}$}} \\\\"
        )
    tex_sig += [
        r"\bottomrule", r"\end{tabular}",
        r"\begin{tablenotes}\small",
        r"\item P = PySR-only; H = HypatiaX. Near/Med/Far at 1.2×/canonical/5× training range.",
        r"\end{tablenotes}", r"\end{table}",
    ]
    with open(RF01_SIGTEX, "w") as f:
        f.write("\n".join(tex_sig))
    print(f"  📄  RF-01 significance LaTeX → {RF01_SIGTEX}")

    # Sub-domain LaTeX table
    def _dom_sort(item):
        d = item[1]
        return (-(d["h_wins"] - d["p_wins"]), -d["n"])

    tex_sub = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Sub-domain breakdown of HypatiaX vs PySR-only far-$R^2$ advantage. "
        + (r"Improvement is statistically significant overall; table shows which domains drive it.}"
           if IS_SIG else
           r"Overall result is not significant; table identifies where benefit is concentrated.}"),
        r"\label{tab:rf01_subdomain}", r"\small",
        r"\begin{tabular}{lrrrrl}", r"\toprule",
        r"\textbf{Domain} & $n$ & \textbf{H wins} & \textbf{Ties} & \textbf{P wins} & \textbf{Mean $\Delta$} \\",
        r"\midrule",
    ]
    for dom, d in sorted(domain_data.items(), key=_dom_sort):
        mean_d = d["delta_sum"] / d["n"]
        color  = (r"\cellcolor{green!15}" if d["h_wins"] > d["p_wins"]
                  else r"\cellcolor{red!10}" if d["p_wins"] > d["h_wins"] else "")
        tex_sub.append(
            f"{color}{dom} & {d['n']} & {d['h_wins']} & {d['ties']} & {d['p_wins']}"
            f" & ${mean_d:+.3f}$ \\\\"
        )
    tex_sub += [
        r"\midrule",
        f"\\textbf{{Total}} & {n_pairs} & {h_wins} & {ties} & {p_wins}"
        f" & ${float(np.mean(H_arr)-np.mean(P_arr)):+.3f}$ \\\\",
        r"\bottomrule", r"\end{tabular}",
        r"\begin{tablenotes}\small",
        r"\item \colorbox{green!15}{Green} = H dominant; \colorbox{red!10}{Red} = P dominant.",
        r"\item Driving domain(s): " + (", ".join(driving_doms) if driving_doms else "none") + ".",
        r"\end{tablenotes}", r"\end{table}",
    ]
    with open(RF01_SUBDTEX, "w") as f:
        f.write("\n".join(tex_sub))
    print(f"  📄  RF-01 sub-domain LaTeX → {RF01_SUBDTEX}")

# =============================================================================
# §19 · LaTeX Table 5 builder (resizebox-safe)
# =============================================================================

def make_latex_table(all_results: dict) -> str:
    def _fmt(v, d=4):
        if v is None: return "---"
        try:
            fv = float(v)
            return "---" if not np.isfinite(fv) else f"{fv:.{d}f}"
        except (TypeError, ValueError):
            return "---"

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{LLM Ablation: PySR Alone vs.\ HypatiaX (PySR + LLM Warm-Start) "
        r"on Core~15. Extrap columns show $R^2$ at near/medium/far OOD regimes. "
        r"$\dag$ = wall-clock-capped. $\ddag$ = Arrhenius failure mode.}",
        r"\label{tab:llm_ablation}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llcccccccccc}",
        r"\toprule",
        r"\textbf{Equation} & \textbf{Domain} & "
        r"\multicolumn{2}{c}{\textbf{Train $R^2$}} & "
        r"\multicolumn{2}{c}{\textbf{Near $R^2$}} & "
        r"\multicolumn{2}{c}{\textbf{Med $R^2$}} & "
        r"\multicolumn{2}{c}{\textbf{Far $R^2$}} & "
        r"\multicolumn{2}{c}{\textbf{Time (s)}} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}"
        r"\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}",
        r" & & P & H & P & H & P & H & P & H & P & H \\",
        r"\midrule",
    ]

    acc = {
        cond: {"train": [], "near": [], "medium": [], "far": [], "time": []}
        for cond in ("pysr_only", "hypatia")
    }

    for eq_idx, eq in enumerate(CORE_15):
        res = all_results.get(str(eq_idx), {})
        p   = res.get("pysr_only", {}) or {}
        h   = res.get("hypatia",   {}) or {}

        p_excl  = p.get("excluded_from_timing", False)
        h_excl  = h.get("excluded_from_timing", False)
        dag     = r"$^{\dag}$" if (p_excl or h_excl) else ""

        p_tr = p.get("train_r2"); h_tr = h.get("train_r2")
        regressed = (
            p_tr is not None and h_tr is not None
            and np.isfinite(float(p_tr)) and np.isfinite(float(h_tr))
            and float(p_tr) - float(h_tr) > 0.05
        )
        ddag = r"$^{\ddag}$" if regressed else ""

        lines.append(
            f"{eq['name']}{dag}{ddag} & {eq['domain']} & "
            f"{_fmt(p.get('train_r2'))} & {_fmt(h.get('train_r2'))} & "
            f"{_fmt(p.get('extrap_r2_near'))} & {_fmt(h.get('extrap_r2_near'))} & "
            f"{_fmt(p.get('extrap_r2_medium'))} & {_fmt(h.get('extrap_r2_medium'))} & "
            f"{_fmt(p.get('extrap_r2_far'))} & {_fmt(h.get('extrap_r2_far'))} & "
            f"{_fmt(p.get('total_time_s'), 0)} & {_fmt(h.get('total_time_s'), 0)} \\\\"
        )
        for ck, d in (("pysr_only", p), ("hypatia", h)):
            for metric, key in [
                ("train", "train_r2"), ("near", "extrap_r2_near"),
                ("medium", "extrap_r2_medium"), ("far", "extrap_r2_far"),
                ("time", "total_time_s"),
            ]:
                v = d.get(key)
                if v is not None:
                    acc[ck][metric].append(float(v))

    def _smean(lst, d=4):
        vals = [v for v in lst if np.isfinite(v)]
        return f"{np.mean(vals):.{d}f}" if vals else "---"

    p_acc, h_acc = acc["pysr_only"], acc["hypatia"]
    lines += [
        r"\midrule",
        r"\textbf{Mean} & & "
        f"{_smean(p_acc['train'])} & {_smean(h_acc['train'])} & "
        f"{_smean(p_acc['near'])} & {_smean(h_acc['near'])} & "
        f"{_smean(p_acc['medium'])} & {_smean(h_acc['medium'])} & "
        f"{_smean(p_acc['far'])} & {_smean(h_acc['far'])} & "
        f"{_smean(p_acc['time'], 0)} & {_smean(h_acc['time'], 0)} \\\\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",  # close resizebox
        r"\begin{tablenotes}\small",
        r"\item P = PySR-only; H = HypatiaX. HypatiaX runtime includes all LLM + retry time.",
        r"\item Near/Med/Far $R^2$ at 1.2$\times$, canonical, and 5$\times$ training range.",
        r"\item[$\dag$] Wall-clock cap triggered; excluded from timing averages.",
        r"\item[$\ddag$] Arrhenius failure mode: correct LLM prior constrained PySR search "
        r"space → premature convergence. See \S\ref{sec:arrhenius}.",
        + f" All runs: populations={POPULATIONS}, seed={SEED}, engine v5.1.",
        r"\end{tablenotes}",
        r"\end{table*}",
    ]
    return "\n".join(lines)

# =============================================================================
# §20 · Instability stats & CSV (§10.9)
# =============================================================================

def run_instability_stats(all_results: dict) -> None:
    """
    Approximates per-equation instability index as |near_R2 − far_R2| (regime-range proxy).
    For the full 30-run stochastic instability index (ii = std R² across runs),
    see exp4_instability_rf02_04.ipynb.
    """
    instability_rows = []
    for eq_idx, eq in enumerate(CORE_15):
        res    = all_results.get(str(eq_idx), {})
        h      = res.get("hypatia", {}) or {}
        r_near = h.get("extrap_r2_near",  0.0) or 0.0
        r_far  = h.get("extrap_r2_far",   0.0) or 0.0
        ii     = abs(r_far - r_near) if (r_near is not None and r_far is not None) else None
        instability_rows.append({
            "equation":          eq["name"],
            "domain":            eq["domain"],
            "extrap_r2_near":    r_near,
            "extrap_r2_far":     r_far,
            "instability_index": ii,
        })

    with open(INSTAB_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["equation", "domain", "extrap_r2_near",
                           "extrap_r2_far", "instability_index"]
        )
        writer.writeheader()
        writer.writerows(instability_rows)

    with open(INSTAB_STATS, "w") as f:
        json.dump({
            "core15_instability": instability_rows,
            "engine_version":     HybridDiscoverySystem.VERSION,
            "note": (
                "instability_index = |near_R2 - far_R2| (regime-range proxy). "
                "For full 30-run stochastic sweep (Spearman ρ=−0.70 claim) "
                "see exp4_instability_rf02_04.ipynb."
            ),
        }, f, indent=2)

    print(f"✅ Instability stats → {INSTAB_STATS}")
    print(f"✅ Instability CSV   → {INSTAB_CSV}")


def run_portfolio_seed_sweep(n_iter=300, timeout_secs=90):
    """
    C4: 5-seed stability sweep for Portfolio Std Dev.
    Reports mean ± std to flag seed-sensitive results.
    """
    PV_EQ   = next(e for e in CORE_15 if e["name"] == "Portfolio Std Dev")
    SEEDS_5 = [42, 123, 777, 2024, 99]

    print(f"\nPortfolio Std Dev — 5-seed stability sweep")
    print(f"Seeds: {SEEDS_5}  |  iterations: {n_iter}  |  timeout: {timeout_secs}s")
    print("=" * 65)

    sweep = {}
    for s in SEEDS_5:
        print(f"\n── Seed {s} ──")
        for cond in CONDITIONS:
            r = run_condition(
                PV_EQ, cond, seed=s,
                niterations=n_iter, timeout_secs=timeout_secs,
                populations=POPULATIONS,
            )
            sweep.setdefault(cond, []).append({
                "seed": s,
                "far_r2":  r.get("extrap_r2_far"),
                "near_r2": r.get("extrap_r2_near"),
            })
            fv = r.get("extrap_r2_far")
            print(f"  {cond}: far_R²="
                  f"{fv:.4f}" if (fv is not None and np.isfinite(fv)) else "N/A")

    print("\n" + "=" * 65)
    for cond, runs in sweep.items():
        vals = [r["far_r2"] for r in runs if r["far_r2"] is not None and np.isfinite(r["far_r2"])]
        if vals:
            mu, sigma = np.mean(vals), np.std(vals)
            flag = "⚠️  HIGH VARIANCE" if sigma > 0.5 else "✅ Stable"
            print(f"  {cond}: mean={mu:.4f}  std={sigma:.4f}  ({flag})")
            print(f"    → Report as {mu:.2f} ± {sigma:.2f}")

    pv_path = OUTPUT_DIR / f"portfolio_variance_seed_sweep{_seed_suffix}.json"
    with open(pv_path, "w") as f:
        json.dump(sweep, f, indent=2)
    print(f"\n✅ Seed sweep saved → {pv_path}")
    return sweep

# =============================================================================
# §21 · Provenance stamp
# =============================================================================

def write_provenance(all_results: dict) -> None:
    provenance = {
        "family":                "ablation_exp1",
        "engine":                "hybrid_system_v50_2.py",
        "engine_version":        HybridDiscoverySystem.VERSION,
        "seed":                  SEED,
        "seed_is_paper_primary": SEED == 42,
        "populations":           POPULATIONS,
        "niterations":           NITERATIONS,
        "model_string":          MODEL_STRING,
        "timestamp":             datetime.now().isoformat(),
        "outputs": {
            "results_json":      str(RESULTS_PATH),
            "checkpoint_json":   str(CKPT_PATH),
            "tex_table5":        str(TEX_PATH),
            "rf01_json":         str(RF01_JSON),
            "rf01_sig_tex":      str(RF01_SIGTEX),
            "rf01_sub_tex":      str(RF01_SUBDTEX),
            "instability_stats": str(INSTAB_STATS),
            "instability_csv":   str(INSTAB_CSV),
        },
        "paper_sections": ["§10.6", "§10.9"],
        "paper_targets": {
            "MW_run_a":      "U=126, p=0.295  [seed=42, pops=30 only]",
            "MW_run_b":      "U=101.5, p=0.683  [seed=42, pops=30 only]",
            "spearman_rho":  "-0.70 (p<0.001) — cross-ref instability_analysis.csv from exp4",
        },
        "fixes_applied": [
            "FIX-POP: populations=30 (was 2 — inflated apparent speedup)",
            "FIX-WIRE: hypatia → HybridDiscoverySystem v5.1 (was raw PySR + manual warm-start)",
            "FIX-KEY: checkpoint keyed on eq_id int (survives renames)",
            "FIX-APIKEY: Kaggle Secrets / env var (no hardcoded key)",
            "FIX-WALLCLOCK: per-condition wall-clocks (hypatia=3×1100+300, pysr_only=1100+300)",
            "FIX-A: RMSE in original units via discover()",
            "FIX-B: formula/expression/final_formula key aliases resolved",
            "FIX-C: deterministic PySR (allow_nondeterministic=False)",
            "FIX-D: extreme-scale log-transform active inside discover()",
            "FIX-POW: auto pow for non-negative X active inside discover()",
            "FIX-SEED: env-driven seed; output files namespaced per seed",
            "FIX-RESTORE: VariableNameSanitizer.restore() sorts keys longest-first",
        ],
    }
    with open(PROV_PATH, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"✅ Provenance → {PROV_PATH}")

# =============================================================================
# §22 · Submission readiness checklist
# =============================================================================

def check_readiness() -> bool:
    checks = []

    c1_ok = SYMPY_AVAILABLE and all("formula_sympy" in e for e in CORE_15)
    checks.append(("C1 True OOD eval (sympy)", c1_ok,
                   "sympy + formula_sympy on all 15 eqs" if c1_ok
                   else "pip install sympy  or  check FORMULA_SYMPY dict"))

    c2_ok = INSTAB_CSV.exists()
    if c2_ok:
        import pandas as _pd
        df = _pd.read_csv(INSTAB_CSV)
        c2_ok = df["instability_index"].fillna(0).abs().sum() > 0
    checks.append(("C2 instability CSV", c2_ok,
                   "CSV exists + non-zero ii" if c2_ok
                   else f"{INSTAB_CSV} missing or all zeros — run §20"))

    flags_path = OUTPUT_DIR / "wall_clock_flags.json"
    c3_ok = flags_path.exists() and RESULTS_PATH.exists()
    checks.append(("C3 wall-clock dag tags", c3_ok,
                   "flags JSON present" if c3_ok
                   else "run §23 (wall-clock flag audit) after §16"))

    pv_path = OUTPUT_DIR / f"portfolio_variance_seed_sweep{_seed_suffix}.json"
    c4_ok = pv_path.exists()
    checks.append(("C4 Portfolio seed sweep", c4_ok,
                   "sweep JSON present" if c4_ok else "run run_portfolio_seed_sweep()"))

    print("\n" + "=" * 70)
    print("SUBMISSION READINESS — BLOCKER CHECKLIST")
    print("=" * 70)
    for name, ok, detail in checks:
        status = "✅" if ok else "❌"
        print(f"  {status}  {name:<35}  {detail}")
    print("=" * 70)
    all_clear = all(c[1] for c in checks)
    if all_clear:
        print("  🟢  All blockers resolved — proceed to submission.")
    else:
        n_open = sum(1 for c in checks if not c[1])
        print(f"  🔴  {n_open} blocker(s) still open — do not submit yet.")
    return all_clear

# =============================================================================
# §23 · Wall-clock flag audit (C3)
# =============================================================================

def audit_wall_clock_flags(all_results: dict) -> None:
    flags = {}
    any_flagged = False
    print(f"{'Equation':<28} {'Condition':<14} {'timed_out':>10} {'excl_timing':>12} {'wall_s':>8}")
    print("─" * 76)
    for eq_idx, eq in enumerate(CORE_15):
        eq_key = str(eq_idx)
        res    = all_results.get(eq_key, {})
        flags[eq_key] = {"name": eq["name"]}
        for cond in CONDITIONS:
            d    = res.get(cond, {}) or {}
            to   = d.get("timed_out", False)
            excl = d.get("excluded_from_timing", False)
            ws   = d.get("timeout_wall_secs") or d.get("total_time_s", 0)
            flags[eq_key][cond] = {
                "timed_out": to, "excluded_from_timing": excl, "wall_secs": ws
            }
            marker = " ◀ FLAGGED $dag$" if (to or excl) else ""
            if to or excl:
                any_flagged = True
            print(f"  {eq['name']:<26} {cond:<14} "
                  f"{str(to):>10} {str(excl):>12} {ws:>8.1f}{marker}")

    flags_path = OUTPUT_DIR / "wall_clock_flags.json"
    with open(flags_path, "w") as f:
        json.dump(flags, f, indent=2)
    print(f"\n✅ C3: wall_clock_flags.json saved → {flags_path}")
    if not any_flagged:
        print("  ℹ️  No equations flagged — all completed within wall-clock cap.")

# =============================================================================
# §24 · Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exp 1: LLM Ablation on Core-15")
    parser.add_argument("--skip-llm",        action="store_true")
    parser.add_argument("--fast",            action="store_true",
                        help="Low-resource preset: iterations=300, timeout=90s")
    parser.add_argument("--seed-sweep",      action="store_true",
                        help="Run Portfolio Std Dev 5-seed sweep (C4)")
    parser.add_argument("--check",           action="store_true",
                        help="Print submission readiness checklist and exit")
    parser.add_argument("--task-ids",        default="",
                        help="Comma- or space-separated shard slug filter "
                             "(e.g. 'amm risk_var'); overrides DOMAIN_FILTER env var")
    args = parser.parse_args()

    # --task-ids CLI flag takes priority over the environment variable
    if args.task_ids.strip():
        os.environ["DOMAIN_FILTER"] = args.task_ids.strip()

    if args.fast:
        NITERATIONS       = 300
        PYSR_TIMEOUT_SECS = 90
        print("⚡ --fast preset: iterations=300, timeout=90s")

    if args.check:
        check_readiness()
        sys.exit(0)

    # Run main experiment
    all_results = run_experiment()

    # Post-processing
    print_summary(all_results)
    run_rf01(all_results)

    latex = make_latex_table(all_results)
    with open(TEX_PATH, "w") as f:
        f.write(latex)
    print(f"✅ Table 5 LaTeX → {TEX_PATH}")

    run_instability_stats(all_results)
    audit_wall_clock_flags(all_results)
    write_provenance(all_results)

    if args.seed_sweep:
        run_portfolio_seed_sweep()

    check_readiness()
    print("\n✅ exp1_ablation.py complete — all outputs written.")
