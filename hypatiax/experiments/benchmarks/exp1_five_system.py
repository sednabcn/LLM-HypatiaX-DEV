"""
HypatiaX — Experiment 1-Five: Five-System Comparison  (§10.1)
================================================================
Extends exp1_ablation.py's Core-15 protocol from 2 conditions
(pysr_only, hypatia) to the 5-method comparison used elsewhere in this
codebase (matches generate_tables.py's _EXP2_METHOD_TO_ROW mapping and
the exp2/exp2_extrap "five-system" concept — see that file's dead
_load_exp2_five_system_rows() loader for the prior, never-wired-in
version of this same idea).

Methods (5 of run_comparative_suite_benchmark_v2.py's 6 METHOD_REGISTRY
entries; index 3, HybridDeFiMethod, is excluded — it is DeFi-domain-scoped
and does not correspond to any of the five paper row names, matching the
exclusion already documented in generate_tables.py):

    idx  class                  → row name
    ---  ---------------------  --------------------
     1   PureLLMBaselineMethod  → Pure LLM
     2   ImprovedNNMethod       → Neural Network
     4   HybridAllDomainsMethod → System 3 LLM+Fallback
     5   SymbolicEngineMethod   → System 2 Symbolic
     6   HybridSystemV50_2Method → Hybrid v50\\_2

IMPORTANT — NOT a reproduction of exp1_ablation's "hypatia" condition:
HybridSystemV50_2Method.run() calls hybrid_system_v50_2 through
_run_pysr_in_subprocess(method="hybrid_v50_2", ...) with its own internal
3-retry adaptive-iteration scheme and NO LLM proposal step. exp1_ablation's
"hypatia" condition instead calls HybridDiscoverySystem(...).discover()
directly, with use_llm/llm_mode explicitly configured from
ANTHROPIC_API_KEY. These are two different call paths through the same
underlying engine and are not expected to produce identical numbers for
the same equation. This experiment's "Hybrid v50_2" row is therefore a
genuinely different (fifth) system, not a duplicate/rename of
exp1_ablation's "hypatia" row — do not merge or compare them directly
without accounting for this.

Nothing in this file reimplements method logic: PureLLMBaselineMethod,
ImprovedNNMethod, HybridAllDomainsMethod, SymbolicEngineMethod, and
HybridSystemV50_2Method are imported directly from
run_comparative_suite_benchmark_v2.py (same file, loaded via
importlib.util the same way exp1_ablation.py loads hybrid_system_v50_2.py).
Equation suite (CORE_15), data generators, checkpointing, and the
wall-clock _Timeout context manager are imported the same way from
exp1_ablation.py itself, so the two experiments cannot drift apart on
what "Core-15" or "extrapolation regime" means.

Output files (RESULTS_DIR/five_systems/exp1_five/):
    exp1_five_results.json         — {eq_idx: {method_name: {...}}}
    exp1_five_checkpoint.json      — resumable checkpoint (same shape)
    exp1_five_performance.json     — performance sub-table source (§ metric 1)
    exp1_five_extrapolation.json   — extrapolation sub-table source (§ metric 2)
    provenance_map_exp1_five.json
    exp1_five_run.log              — written by run_all.sh's tee, not this script
"""

# =============================================================================
# §0 · Standard library / third-party imports
# =============================================================================

import importlib.util as _ilu
import json
import os
import pathlib as _pl
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

# =============================================================================
# §1 · Load exp1_ablation.py and run_comparative_suite_benchmark_v2.py as modules
#
# Both files are expected to sit alongside this one in
# hypatiax/experiments/benchmarks/ (same directory convention already used
# by run_all.sh / ci_runner_repro.yml for every other experiment script).
# =============================================================================

_HERE = Path(__file__).resolve().parent


def _load_module(filename: str, modname: str):
    path = _HERE / filename
    if not path.exists():
        raise RuntimeError(
            f"{filename} not found next to exp1_five_system.py at {path}. "
            f"exp1_five_system.py reuses its equation suite / method classes "
            f"from this file rather than reimplementing them — it must be "
            f"present."
        )
    spec = _ilu.spec_from_file_location(modname, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


print("Loading exp1_ablation.py (Core-15 equation suite, data generators, checkpoint helpers)...")
_abl = _load_module("exp1_ablation.py", "exp1_ablation_shared")

print("Loading run_comparative_suite_benchmark_v2.py (method classes, extrap eval helpers)...")
_rcsb = _load_module("run_comparative_suite_benchmark_v2.py", "run_comparative_suite_benchmark_v2_shared")

# Reused, not reimplemented:
CORE_15             = _abl.CORE_15
EQ_ID               = _abl.EQ_ID
EXTRAP_REGIMES      = _abl.EXTRAP_REGIMES
generate_data        = _abl.generate_data
generate_extrap_data = _abl.generate_extrap_data
load_checkpoint      = _abl.load_checkpoint
save_checkpoint      = _abl.save_checkpoint
_Timeout             = _abl._Timeout

MethodResult          = _rcsb.MethodResult
_runner_eval_formula   = _rcsb._runner_eval_formula
_far_r2                = _rcsb._far_r2
_far_rmse              = _rcsb._far_rmse

# =============================================================================
# §2 · The five-method registry (METHOD_REGISTRY minus index 3, HybridDeFiMethod)
# =============================================================================

_FIVE_METHOD_INDICES = (1, 2, 4, 5, 6)

_ALL_METHOD_ROW_NAMES = {
    "PureLLMBaselineMethod":    "Pure LLM",
    "ImprovedNNMethod":         "Neural Network",
    "HybridAllDomainsMethod":   "System 3 LLM+Fallback",
    "SymbolicEngineMethod":     "System 2 Symbolic",
    "HybridSystemV50_2Method":  "Hybrid v50_2",
}

_DESIGN_FOCUS = {
    "Pure LLM":              "Recognition",
    "Neural Network":        "Baseline",
    "System 2 Symbolic":     "Validation",
    "System 3 LLM+Fallback": "Robustness",
    "Hybrid v50_2":          "Extrapolation",
}

# =============================================================================
# §3 · Run-time parameters (same env-var names as exp1_ablation.py, so a
# single run_all.sh stage's env block works for both)
# =============================================================================

_GLOBAL_SEED = int(os.environ.get("PYSR_SEED", os.environ.get("NN_SEED", 42)))
np.random.seed(_GLOBAL_SEED)

POPULATIONS       = int(os.environ.get("POPULATIONS", os.environ.get("PYSR_POPULATIONS", "30")))
_TIMEOUT_ENV       = os.environ.get("METHOD_TIMEOUT") or os.environ.get("PYSR_TIMEOUT", "900")
METHOD_TIMEOUT_SECS = int(_TIMEOUT_ENV)
PYSR_TIMEOUT_SECS   = int(os.environ.get("PYSR_TIMEOUT", "1100"))
_WALL_CLOCK_BUDGET  = int(os.environ.get("JOB_DEADLINE", str(METHOD_TIMEOUT_SECS + 300)))

# run_comparative_suite_benchmark_v2.py reads these as module-level globals
# at *its own* __main__ time (from --method-timeout / --pysr-timeout CLI
# flags). Since we're loading it as a library, not running it as __main__,
# those flags never fire — so we set the globals directly here to keep the
# adaptive-iteration heuristics inside SymbolicEngineMethod/HybridSystemV50_2Method
# consistent with our env-var configuration instead of silently using their
# hardcoded module defaults (900s / 1100s).
_rcsb._METHOD_TIMEOUT_SECS = METHOD_TIMEOUT_SECS
_rcsb._PYSR_TIMEOUT        = PYSR_TIMEOUT_SECS

print(f"populations   : {POPULATIONS}")
print(f"method_timeout: {METHOD_TIMEOUT_SECS}s  pysr_timeout: {PYSR_TIMEOUT_SECS}s")
print(f"seed          : {_GLOBAL_SEED}")

# =============================================================================
# §4 · Output paths
# =============================================================================

_RESULTS_DIR_ENV = os.environ.get("RESULTS_DIR", "")
OUTPUT_DIR = Path(_RESULTS_DIR_ENV) if _RESULTS_DIR_ENV else Path().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "exp1_five_results.json"
CKPT_PATH    = OUTPUT_DIR / "exp1_five_checkpoint.json"
PERF_PATH    = OUTPUT_DIR / "exp1_five_performance.json"
EXTRAP_PATH  = OUTPUT_DIR / "exp1_five_extrapolation.json"
PROV_PATH    = OUTPUT_DIR / "provenance_map_exp1_five.json"

# =============================================================================
# §5 · Method instantiation (mirrors ProtocolBenchmarkSuite.__init__'s own
# per-class kwarg dispatch, so construction matches how exp2/exp2_five's
# ProtocolBenchmarkSuite-based path builds the same classes)
# =============================================================================

def _build_methods(verbose: bool = False, no_llm_cache: bool = False):
    methods = []
    for idx, cls, src in _rcsb.ProtocolBenchmarkSuite.METHOD_REGISTRY:
        if idx not in _FIVE_METHOD_INDICES:
            continue
        if cls is _rcsb.PureLLMBaselineMethod:
            m = cls(verbose=verbose, no_cache=no_llm_cache)
        elif cls is _rcsb.ImprovedNNMethod:
            m = cls(verbose=verbose, nn_seeds=1)
        elif cls is _rcsb.HybridAllDomainsMethod:
            m = cls(verbose=verbose, no_cache=no_llm_cache)
        else:
            m = cls(verbose=verbose)
        methods.append((_ALL_METHOD_ROW_NAMES.get(cls.__name__, cls.__name__), m))
    return methods


# =============================================================================
# §6 · Single (equation, method) run, including extrapolation evaluation
# =============================================================================

def run_method_on_equation(eq: dict, method_name: str, method, seed: int = 42) -> dict:
    """
    Runs one method on one Core-15 equation and returns a result dict in the
    same field shape exp1_ablation.py's run_condition() uses, so
    generate_tables.py's existing per-equation loaders don't need a second
    schema.
    """
    eq_seed = seed + EQ_ID.get(eq["name"], 0) * 7
    X_train, X_test, y_train, y_test, _, _ = generate_data(
        eq, N=200, noise_level=0.05, seed=eq_seed
    )
    extrap_sets = {}
    for regime_name, _ in EXTRAP_REGIMES:
        X_e, y_e = generate_extrap_data(eq, regime=regime_name, N=100, seed=eq_seed)
        extrap_sets[regime_name] = (X_e, y_e)

    metadata = {
        "equation_name": eq["name"],
        "domain":        eq["domain"].lower().replace(" ", "_"),
        "difficulty":    eq.get("difficulty", "medium"),
    }

    t0 = time.time()
    try:
        with _Timeout(_WALL_CLOCK_BUDGET):
            result: MethodResult = method.run(
                description=eq["name"],
                X=X_train, y=y_train,
                var_names=eq["vars"],
                metadata=metadata,
                verbose=False,
            )
        elapsed = time.time() - t0
    except TimeoutError:
        elapsed = time.time() - t0
        print(f"  [{method_name}] ⏰ TIMEOUT after {elapsed:.1f}s")
        return _timeout_result(method_name, elapsed)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{method_name}] ❌ ERROR: {e}")
        traceback.print_exc()
        return _error_result(method_name, str(e), elapsed)

    if not result.success:
        print(f"  [{method_name}] unavailable/failed: {result.error}")
        return _error_result(method_name, result.error or "unavailable", elapsed)

    # ── Extrapolation: re-evaluate the returned formula on near/medium/far ──
    # Uses the SAME evaluator (_runner_eval_formula) and R²/RMSE functions
    # (_far_r2/_far_rmse) that compute_extrap_r2_far() uses in
    # run_comparative_suite_benchmark_v2.py itself, so exp1_five's
    # extrapolation numbers are computed identically to exp2/exp2_five's,
    # not by a second, possibly-divergent implementation.
    extrap_r2, extrap_rmse = {}, {}
    _predict_fn = getattr(result, "_predict_fn", None)
    for regime_name, (X_e, y_e) in extrap_sets.items():
        if _predict_fn is not None:
            # e.g. Neural Network: no evaluable formula string exists, so
            # replay the method's actual trained predictor on the
            # extrapolation inputs instead of trying to parse a formula.
            try:
                y_pred = np.asarray(_predict_fn(X_e), dtype=float)
                if not np.all(np.isfinite(y_pred)):
                    y_pred = None
            except Exception as _pred_exc:
                print(f"  [{method_name}] extrap predict_fn failed ({regime_name}): {_pred_exc}")
                y_pred = None
        else:
            y_pred = _runner_eval_formula(result.formula_full or result.formula, X_e, eq["vars"])
        if y_pred is None:
            extrap_r2[regime_name] = None
            extrap_rmse[regime_name] = None
        else:
            extrap_r2[regime_name]   = _far_r2(y_e, y_pred)
            extrap_rmse[regime_name] = _far_rmse(y_e, y_pred)

    print(
        f"  [{method_name}] {result.formula[:50]!r}  R²={result.r2:.4f}  "
        f"extrap(near={extrap_r2.get('near')}, medium={extrap_r2.get('medium')}, "
        f"far={extrap_r2.get('far')})  {elapsed:.1f}s"
    )

    return {
        "condition":            method_name,
        "success":              True,
        "timed_out":            False,
        "excluded_from_timing": False,
        "train_r2":             result.r2,
        "train_rmse":           result.rmse,
        "extrap_r2_near":       extrap_r2.get("near"),
        "extrap_r2_medium":     extrap_r2.get("medium"),
        "extrap_r2_far":        extrap_r2.get("far"),
        "extrap_rmse_near":     extrap_rmse.get("near"),
        "extrap_rmse_medium":   extrap_rmse.get("medium"),
        "extrap_rmse_far":      extrap_rmse.get("far"),
        "sr_time_s":            elapsed,
        "llm_time_s":           0.0,
        "total_time_s":         elapsed,
        "best_expression":      result.formula,
        "complexity":           None,
        "design_focus":         _DESIGN_FOCUS.get(method_name, "---"),
    }


def _timeout_result(method_name: str, elapsed: float) -> dict:
    base = dict.fromkeys(
        ["train_r2", "train_rmse",
         "extrap_r2_near", "extrap_r2_medium", "extrap_r2_far",
         "extrap_rmse_near", "extrap_rmse_medium", "extrap_rmse_far",
         "complexity"], None,
    )
    return {**base, "condition": method_name, "success": False, "timed_out": True,
            "excluded_from_timing": True, "sr_time_s": elapsed, "llm_time_s": 0.0,
            "total_time_s": elapsed, "best_expression": "TIMED_OUT",
            "design_focus": _DESIGN_FOCUS.get(method_name, "---")}


def _error_result(method_name: str, error: str, elapsed: float) -> dict:
    base = dict.fromkeys(
        ["train_r2", "train_rmse",
         "extrap_r2_near", "extrap_r2_medium", "extrap_r2_far",
         "extrap_rmse_near", "extrap_rmse_medium", "extrap_rmse_far",
         "complexity"], None,
    )
    return {**base, "condition": method_name, "success": False, "timed_out": False,
            "excluded_from_timing": False, "error": error, "sr_time_s": elapsed,
            "llm_time_s": 0.0, "total_time_s": elapsed, "best_expression": "ERROR",
            "design_focus": _DESIGN_FOCUS.get(method_name, "---")}


# =============================================================================
# §7 · Main experiment loop
# =============================================================================

def run_experiment() -> dict:
    all_results = load_checkpoint(CKPT_PATH)
    methods = _build_methods(verbose=False, no_llm_cache=os.environ.get("NO_LLM_CACHE", "") == "1")
    method_names = [name for name, _ in methods]

    _shard_index = int(os.environ.get("SHARD_INDEX", os.environ.get("SHARD", 0)))
    _n_shards    = int(os.environ.get("N_SHARDS", 1))
    _core15_run  = CORE_15
    if _n_shards > 1:
        _core15_run = [eq for j, eq in enumerate(CORE_15) if j % _n_shards == _shard_index]
        print(f"🔀 Shard {_shard_index}/{_n_shards}: {len(_core15_run)}/{len(CORE_15)} equations")

    print("=" * 65)
    print("EXPERIMENT 1-FIVE: Five-System Comparison  (§10.1 Core-15)")
    print(f"Methods   : {method_names}")
    print(f"Equations : {len(_core15_run)}")
    print("=" * 65)

    for eq in _core15_run:
        eq_idx = EQ_ID[eq["name"]]
        eq_key = str(eq_idx)
        all_results.setdefault(eq_key, {"name": eq["name"], "domain": eq["domain"]})
        entry = all_results[eq_key]

        to_run = [(name, m) for name, m in methods if entry.get(name) is None]
        if not to_run:
            print(f"✓ [{eq_idx:02d}] {eq['name']} — already done (all 5 methods)")
            continue

        print(f"\n{'─'*60}\n▶  [{eq_idx:02d}] {eq['name']}  [{eq['domain']}]\n{'─'*60}")
        for method_name, method in to_run:
            result = run_method_on_equation(eq, method_name, method, seed=_GLOBAL_SEED)
            entry[method_name] = result
            all_results[eq_key] = entry
            save_checkpoint(CKPT_PATH, all_results)
            print(f"  ✓ {method_name} saved to checkpoint")

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    n_done = sum(
        1 for v in all_results.values()
        if isinstance(v, dict) and all(v.get(n) for n in method_names)
    )
    print(f"\n✅ Results saved → {RESULTS_PATH}")
    print(f"   Complete: {n_done}/{len(CORE_15)} equations (all 5 methods)")
    return all_results


# =============================================================================
# §8 · Metric definition + two derived sub-tables (performance, extrapolation)
#
# Same statistical convention as evaluator.txt / fix_ci_statistics.py earlier
# in this pipeline: n / mean / median / sd / se / t-based 95% CI, computed
# from these raw per-equation numbers at generation time — never hand-typed.
# =============================================================================

def _compute_stats(values: list[float]) -> dict | None:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return None
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    mean, median = float(np.mean(arr)), float(np.median(arr))
    sd = float(np.std(arr, ddof=1))
    se = sd / np.sqrt(n)
    try:
        from scipy.stats import t as _t
        tcrit = float(_t.ppf(0.975, df=n - 1))
    except ImportError:
        tcrit = 1.96  # normal-approx fallback if scipy unavailable
    return {
        "n": n, "mean": mean, "median": median, "sd": sd, "se": se,
        "ci_low": mean - tcrit * se, "ci_high": mean + tcrit * se,
    }


def build_performance_table(all_results: dict, method_names: list[str]) -> dict:
    """Performance sub-table: train R² / train RMSE per method, across all
    equations that method ran successfully."""
    table = {}
    for method_name in method_names:
        r2s   = [e.get(method_name, {}).get("train_r2")   for e in all_results.values() if isinstance(e, dict)]
        rmses = [e.get(method_name, {}).get("train_rmse") for e in all_results.values() if isinstance(e, dict)]
        table[method_name] = {
            "train_r2":   _compute_stats(r2s),
            "train_rmse": _compute_stats(rmses),
            "design_focus": _DESIGN_FOCUS.get(method_name, "---"),
        }
    with open(PERF_PATH, "w") as f:
        json.dump(table, f, indent=2)
    print(f"✅ Performance sub-table → {PERF_PATH}")
    return table


def build_extrapolation_table(all_results: dict, method_names: list[str]) -> dict:
    """Extrapolation sub-table: R²/RMSE per method for each of the three
    regimes (near/medium/far), same regimes exp1_ablation.py defines."""
    table = {}
    for method_name in method_names:
        regimes = {}
        for regime_name, _ in EXTRAP_REGIMES:
            r2s   = [e.get(method_name, {}).get(f"extrap_r2_{regime_name}")   for e in all_results.values() if isinstance(e, dict)]
            rmses = [e.get(method_name, {}).get(f"extrap_rmse_{regime_name}") for e in all_results.values() if isinstance(e, dict)]
            regimes[regime_name] = {
                "extrap_r2":   _compute_stats(r2s),
                "extrap_rmse": _compute_stats(rmses),
            }
        table[method_name] = {**regimes, "design_focus": _DESIGN_FOCUS.get(method_name, "---")}
    with open(EXTRAP_PATH, "w") as f:
        json.dump(table, f, indent=2)
    print(f"✅ Extrapolation sub-table → {EXTRAP_PATH}")
    return table


# =============================================================================
# §9 · Provenance
# =============================================================================

def write_provenance(all_results: dict) -> None:
    provenance = {
        "family":         "five_systems_exp1_five",
        "engine":         "exp1_five_system.py",
        "reused_from": {
            "equation_suite_and_data_generators": "exp1_ablation.py (CORE_15, generate_data, generate_extrap_data)",
            "method_classes":                     "run_comparative_suite_benchmark_v2.py (METHOD_REGISTRY indices 1,2,4,5,6)",
            "extrapolation_evaluator":             "run_comparative_suite_benchmark_v2.py (_runner_eval_formula, _far_r2, _far_rmse — same functions compute_extrap_r2_far() uses)",
        },
        "excluded_method": {
            "index": 3, "class": "HybridDeFiMethod",
            "reason": "DeFi-domain-scoped; not one of the five paper row names (see generate_tables.py's _EXP2_METHOD_TO_ROW comment for the same exclusion rationale)",
        },
        "seed":           _GLOBAL_SEED,
        "populations":    POPULATIONS,
        "method_timeout": METHOD_TIMEOUT_SECS,
        "pysr_timeout":   PYSR_TIMEOUT_SECS,
        "timestamp":      datetime.now().isoformat(),
        "outputs": {
            "results_json":       str(RESULTS_PATH),
            "checkpoint_json":    str(CKPT_PATH),
            "performance_json":   str(PERF_PATH),
            "extrapolation_json": str(EXTRAP_PATH),
        },
        "paper_sections": ["§10.1"],
        "known_caveat": (
            "The 'Hybrid v50_2' row here is NOT expected to reproduce "
            "exp1_ablation's 'hypatia' row — different call path through "
            "the same underlying engine (subprocess/no-LLM here vs. direct "
            "discover()/LLM-enabled there). See module docstring."
        ),
    }
    with open(PROV_PATH, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"✅ Provenance → {PROV_PATH}")


# =============================================================================
# §10 · Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exp 1-Five: Five-System Comparison on Core-15")
    parser.add_argument("--task-ids", default="", help="(reserved; not yet wired to a shard filter)")
    args = parser.parse_args()

    all_results = run_experiment()
    methods = _build_methods()
    method_names = [name for name, _ in methods]

    build_performance_table(all_results, method_names)
    build_extrapolation_table(all_results, method_names)
    write_provenance(all_results)

    print("\n✅ exp1_five_system.py complete — all outputs written.")
