"""
michael_test.py — HypatiaX SymbolicEngine: Biology Domain (Paper-Quality)
==========================================================================
Drives SymbolicEngine against all three biology test cases from
ExperimentProtocolAll using the exact hyperparameters published in
repro.yaml (run_id="paper_v3", llm_model="claude-sonnet-4-5",
seed=42, niterations=1000, populations=30, population_size=33 …).

Biology test cases (from experiment_protocol_all_30.py):
  1. Michaelis-Menten kinetics  — Vmax*S / (Km + S)
  2. Logistic Growth            — r*N*(1 - N/K)
  3. Allometric Scaling         — a * M^b

Usage
-----
    # Minimal smoke-test (short timeout, overrides paper values):
    PYSR_TIMEOUT=10 python michael_test.py --fast

    # Full paper-quality run (honours repro.yaml timeouts):
    python michael_test.py

    # Single case by index (0-based):
    python michael_test.py --case 0

Environment variables (all optional — repro.yaml defaults are used):
    PYSR_TIMEOUT          override pysr_attempt_seconds   (default 1100)
    PYSR_FIT_WALL_TIMEOUT override fit_wall_timeout        (default 1200)
    PYSR_FIT_GRACE_SECS   override fit_grace_secs          (default  120)
    METHOD_TIMEOUT        override method_seconds           (default  900)
"""

import argparse
import os
import pathlib
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# sys.path: resolve hypatiax.* regardless of cwd
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = pathlib.Path(os.environ.get("REPRO_ROOT", str(_HERE.parent)))
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "hypatiax")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# repro.yaml paper-quality constants (hard-coded to avoid yaml dependency)
# ---------------------------------------------------------------------------
REPRO = {
    "run_id":           "paper_v3",
    "run_version":      "3.0",
    "llm_model":        "claude-sonnet-4-5",
    "llm_retries":      3,
    "llm_k_runs":       30,
    "seed":             42,
    # timeouts
    "pysr_attempt_seconds":     int(os.environ.get("PYSR_TIMEOUT",          1100)),
    "fit_wall_timeout":         int(os.environ.get("PYSR_FIT_WALL_TIMEOUT", 1200)),
    "fit_grace_secs":           int(os.environ.get("PYSR_FIT_GRACE_SECS",   120)),
    "method_seconds":           int(os.environ.get("METHOD_TIMEOUT",         900)),
    # PySR settings
    "niterations":      1000,
    "populations":      30,
    "population_size":  33,
    "parsimony":        0.01,
    "maxsize":          30,
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators":  ["exp", "log", "sin", "cos", "sqrt"],
    "deterministic":    True,
    "parallelism":      "multithreading",
}

# ---------------------------------------------------------------------------
# --fast flag: short-circuit timeouts for smoke-testing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Michael test — biology, paper quality")
parser.add_argument("--fast",  action="store_true",
                    help="Override timeouts to 10 s for a quick smoke-test")
parser.add_argument("--case",  type=int, default=None,
                    help="Run only case N (0=Michaelis-Menten, 1=Logistic Growth, 2=Allometric)")
parser.add_argument("--samples", type=int, default=300,
                    help="Number of data points per test case (default 300, paper value)")
args = parser.parse_args()

if args.fast:
    REPRO["pysr_attempt_seconds"] = 10
    REPRO["fit_wall_timeout"]     = 10
    REPRO["fit_grace_secs"]       = 5
    REPRO["method_seconds"]       = 15
    print("⚡ --fast mode: timeouts overridden to 10 s")

# ---------------------------------------------------------------------------
# Propagate key env-vars so DiscoveryConfig / PySRRegressor pick them up
# ---------------------------------------------------------------------------
os.environ["PYSR_TIMEOUT"]          = str(REPRO["pysr_attempt_seconds"])
os.environ["PYSR_FIT_WALL_TIMEOUT"] = str(REPRO["fit_wall_timeout"])
os.environ["PYSR_FIT_GRACE_SECS"]   = str(REPRO["fit_grace_secs"])
os.environ["METHOD_TIMEOUT"]        = str(REPRO["method_seconds"])

# ---------------------------------------------------------------------------
# Biology test-case data (mirrors experiment_protocol_all_30.py exactly)
# ---------------------------------------------------------------------------
np.random.seed(REPRO["seed"])
N = args.samples


def _michaelis_menten(N: int) -> tuple:
    """Michaelis-Menten: v = Vmax*Sub / (Km + Sub)
    NOTE: 'S' is a reserved sympy function name → renamed to 'Sub' (substrate).
    """
    Vmax = np.random.uniform(1.0, 10.0,  N)
    Km   = np.random.uniform(0.1, 5.0,   N)
    Sub  = np.random.uniform(0.1, 20.0,  N)   # was S — reserved by sympy
    X    = np.column_stack([Vmax, Km, Sub])
    y    = Vmax * Sub / (Km + Sub)
    meta = {
        "equation_name":        "michaelis_menten",
        "difficulty":           "medium",
        "formula_type":         "rational",
        "ground_truth":         "Vmax * Sub / (Km + Sub)",
        "original_ground_truth": "Vmax * S / (Km + S)",   # paper notation
        "units":                {"Vmax": "mM/s", "Km": "mM", "Sub": "mM", "v": "mM/s"},
        "variable_descriptions": {
            "Vmax": "Maximum reaction rate",
            "Km":   "Michaelis constant (substrate at half-Vmax)",
            "Sub":  "Substrate concentration (renamed from S: sympy reserved)",
        },
        "variable_roles":       {"Vmax": "varying", "Km": "varying", "Sub": "varying"},
        "structure_hints":      {"rational_form": True, "saturation_curve": True},
        "protocol":             "B",
    }
    return "Michaelis-Menten Kinetics: v = Vmax*Sub / (Km+Sub)", X, y, ["Vmax", "Km", "Sub"], meta


def _logistic_growth(N: int) -> tuple:
    """Logistic Growth: dPop/dt = r*Pop*(1 - Pop/K)
    NOTE: 'N' is a reserved sympy function name → renamed to 'Pop' (population).
    """
    r   = np.random.uniform(0.1, 0.5,   N)
    Pop = np.random.uniform(10,  900,    N)   # was N — reserved by sympy
    K   = np.random.uniform(1000, 2000,  N)
    X   = np.column_stack([r, Pop, K])
    y   = r * Pop * (1 - Pop / K)
    meta = {
        "equation_name":        "logistic_growth",
        "difficulty":           "medium",
        "formula_type":         "nonlinear",
        "ground_truth":         "r * Pop * (1 - Pop / K)",
        "original_ground_truth": "r * N * (1 - N / K)",   # paper notation
        "units":                {"r": "1/s", "Pop": "dimensionless", "K": "dimensionless", "dNdt": "1/s"},
        "variable_descriptions": {
            "r":   "Intrinsic growth rate",
            "Pop": "Current population size (renamed from N: sympy reserved)",
            "K":   "Carrying capacity",
        },
        "variable_roles":       {"r": "constant", "Pop": "varying", "K": "constant"},
        "structure_hints":      {"multiplicative_terms": True, "subtraction_in_factor": True},
        "protocol":             "B",
    }
    return "Logistic Growth: dN/dt = r*Pop*(1 - Pop/K)", X, y, ["r", "Pop", "K"], meta


def _allometric_scaling(N: int) -> tuple:
    """Allometric Scaling: Y = 3.5 * M^0.75
    NOTE: a and b are constants (not varying), so passing them as columns gives
    PySR zero signal — it just memorises their values as literals.
    Fix: expose only M as the input variable; PySR must discover the coefficient
    and exponent on its own.  The ground truth is stated in terms of the paper
    symbols for reference.
    """
    M = np.random.uniform(0.1, 100, N)
    X = M.reshape(-1, 1)                 # single feature
    y = 3.5 * M ** 0.75
    meta = {
        "equation_name":        "allometric_scaling",
        "difficulty":           "easy",
        "formula_type":         "power_law",
        "ground_truth":         "3.5 * M**0.75",
        "original_ground_truth": "a * M**b  (a=3.5, b=0.75)",  # paper notation
        "units":                {"M": "kg", "Y": "W"},
        "variable_descriptions": {
            "M": "Body mass (a=3.5 and b=0.75 are fixed constants, not inputs)",
        },
        "variable_roles":       {"M": "varying"},
        "structure_hints":      {"M": "power_law"},
        "protocol":             "B",
    }
    return "Allometric Scaling: Y = 3.5*M^0.75", X, y, ["M"], meta


ALL_CASES = [_michaelis_menten, _logistic_growth, _allometric_scaling]

# ---------------------------------------------------------------------------
# Import HypatiaX engine
# ---------------------------------------------------------------------------
try:
    from hypatiax.tools.symbolic.symbolic_engine import DiscoveryConfig, SymbolicEngine
    ENGINE_AVAILABLE = True
except ImportError as exc:
    ENGINE_AVAILABLE = False
    _IMPORT_ERR = exc


# ---------------------------------------------------------------------------
# DiscoveryConfig factory — paper-quality values
# ---------------------------------------------------------------------------
def make_config() -> "DiscoveryConfig":
    """Build a DiscoveryConfig from repro.yaml paper values.

    Only fields that exist in DiscoveryConfig.__init__ are passed.
    repro.yaml extras (fit_wall_timeout, fit_grace_secs, deterministic,
    parallelism, seed) are propagated via env-vars above instead.
    """
    return DiscoveryConfig(
        # Timeout — the one DiscoveryConfig field for PySR wall-clock cap
        pysr_timeout       = REPRO["pysr_attempt_seconds"],
        # Search budget
        niterations        = REPRO["niterations"],
        populations        = REPRO["populations"],
        population_size    = REPRO["population_size"],
        parsimony          = REPRO["parsimony"],
        maxsize            = REPRO["maxsize"],
        # Operators (paper set)
        binary_operators   = REPRO["binary_operators"],
        unary_operators    = REPRO["unary_operators"],
    )


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------
SEP  = "=" * 72
SSEP = "-" * 72

def _banner(text: str) -> None:
    print(f"\n{SEP}")
    print(f"  {text}")
    print(SEP)

def _section(text: str) -> None:
    print(f"\n{SSEP}")
    print(f"  {text}")
    print(SSEP)

def _kv(key: str, value) -> None:
    print(f"  {key:<30s} {value}")


# ---------------------------------------------------------------------------
# Run a single case
# ---------------------------------------------------------------------------
def run_case(idx: int, case_fn, domain: str = "biology") -> dict:
    desc, X, y, var_names, meta = case_fn(N)

    _section(f"Case {idx+1}/3 · {desc}")
    _kv("Equation",    meta["ground_truth"])
    _kv("Variables",   ", ".join(var_names))
    _kv("Difficulty",  meta["difficulty"])
    _kv("Samples",     X.shape[0])
    _kv("Domain",      domain)
    _kv("pysr_timeout",    f"{REPRO['pysr_attempt_seconds']} s")
    _kv("fit_wall_timeout",f"{REPRO['fit_wall_timeout']} s")
    _kv("niterations", REPRO["niterations"])
    _kv("populations", REPRO["populations"])

    if not ENGINE_AVAILABLE:
        print(f"\n  ⚠  SymbolicEngine not importable: {_IMPORT_ERR}")
        return {"case": desc, "status": "import_error", "error": str(_IMPORT_ERR)}

    config = make_config()
    engine = SymbolicEngine(config, domain=domain)

    t0 = time.perf_counter()
    try:
        result = engine.discover(X, y, variable_names=var_names)
        elapsed = time.perf_counter() - t0
        expr    = result.get("expression", result.get("best_expression", "N/A"))
        r2      = result.get("r2_score",   result.get("r2",              float("nan")))
        print(f"\n  ✅ Discovered in {elapsed:.1f} s")
        _kv("Expression",  expr)
        _kv("R²",          f"{r2:.6f}")
        _kv("Ground truth",meta["ground_truth"])
        return {
            "case":       desc,
            "status":     "ok",
            "expression": expr,
            "r2":         r2,
            "elapsed_s":  elapsed,
            "ground_truth": meta["ground_truth"],
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"\n  ❌ FAILED after {elapsed:.1f} s: {exc}")
        return {"case": desc, "status": "error", "error": str(exc), "elapsed_s": elapsed}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _banner(
        f"HypatiaX · Michael Test · Biology Domain · paper_v3\n"
        f"  Engine: {REPRO['llm_model']}  |  seed={REPRO['seed']}  |"
        f"  samples={N}"
    )

    print("\n  repro.yaml hyperparameters in use:")
    for k in ("run_id", "run_version", "seed", "niterations", "populations",
               "population_size", "parsimony", "maxsize",
               "pysr_attempt_seconds", "fit_wall_timeout",
               "fit_grace_secs", "method_seconds"):
        _kv(k, REPRO[k])

    cases_to_run = (
        [ALL_CASES[args.case]] if args.case is not None else ALL_CASES
    )
    indices = (
        [args.case] if args.case is not None else list(range(len(ALL_CASES)))
    )

    results = []
    t_total = time.perf_counter()
    for idx, fn in zip(indices, cases_to_run):
        results.append(run_case(idx, fn))

    total_elapsed = time.perf_counter() - t_total

    # ── Summary table ──────────────────────────────────────────────────────
    _banner("SUMMARY")
    ok   = [r for r in results if r["status"] == "ok"]
    fail = [r for r in results if r["status"] != "ok"]

    print(f"  Ran {len(results)} case(s)  |  "
          f"✅ {len(ok)} passed  |  ❌ {len(fail)} failed  |  "
          f"total {total_elapsed:.1f} s\n")

    header = f"  {'#':<4} {'Status':<8} {'R²':>10}  {'Expression'}"
    print(header)
    print("  " + "-" * 68)
    for i, r in enumerate(results):
        status = "✅ ok" if r["status"] == "ok" else "❌ err"
        r2_str = f"{r['r2']:.4f}" if r.get("r2") is not None and r["status"] == "ok" else "—"
        expr   = r.get("expression", r.get("error", ""))[:50]
        print(f"  {i:<4} {status:<8} {r2_str:>10}  {expr}")

    if fail:
        print("\n  Failures:")
        for r in fail:
            print(f"    • {r['case']}: {r.get('error', 'unknown error')}")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
