#!/usr/bin/env python3
"""
run_sample_complexity_benchmark.py
====================================

Sample-complexity experiment for the HypatiaX benchmark suite.

Sweeps training-set size n ∈ {50, 100, 200, 500} for the top two methods
— EnhancedHybridSystemDeFi (method 3) and HybridSystemLLMNN all-domains
(method 4) — and produces a report addressing data-efficiency, a standard
reviewer question for symbolic regression papers.

The existing n=200 run (protocol_core_noisy_20260313_094752.json) can be
merged via --existing-results to avoid re-running that sample size.

What it does
------------
1. For each n in [50, 100, 200, 500] it launches
   ``run_comparative_suite_benchmark_v2.py`` as a subprocess with
   ``--samples n``.
2. Collects the four result JSONs and builds a cross-n comparison table:
   per-method median R², mean R², std R², and recovery rate, all as a
   function of n.
3. Also computes a "data efficiency score" — the smallest n at which a
   method first exceeds the recovery-rate threshold — to give a single
   headline number per method.
4. Saves
       data/results/comparison_results/sample_complexity_<TS>.json
       data/results/comparison_results/sample_complexity_<TS>.csv

Usage
-----
    # Default: all methods, all Feynman equations, n ∈ {50,100,200,500}
    python run_sample_complexity_benchmark.py

    # Custom sample sizes
    python run_sample_complexity_benchmark.py --sample-sizes 50 100 200 500

    # Only methods 1 and 2 (fastest smoke-test)
    python run_sample_complexity_benchmark.py --methods 1 2

    # Single equation
    python run_sample_complexity_benchmark.py --test arrhenius

    # Subset of equations
    python run_sample_complexity_benchmark.py --equations 1 2 3

    # Skip PySR-backed methods
    python run_sample_complexity_benchmark.py --skip-pysr

    # Use noiseless mode (σ = 0)
    python run_sample_complexity_benchmark.py --noiseless

    # Abort on any method failure
    python run_sample_complexity_benchmark.py --fail-fast

    # Verbose output
    python run_sample_complexity_benchmark.py --verbose

Outputs
-------
  data/results/comparison_results/sample_complexity_<TS>.json
  data/results/comparison_results/sample_complexity_<TS>.csv
  data/results/comparison_results/sample_complexity_<TS>.log  (if --log)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE        = Path(__file__).resolve().parent
_PKG_ROOT    = _HERE.parent.parent          # hypatiax/
_RUNNER      = _HERE / "run_comparative_suite_benchmark_v2.py"
_RESULTS_DIR = _PKG_ROOT / "data/results/comparison_results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default sample sizes (training points per equation)
_DEFAULT_SAMPLE_SIZES: List[int] = [50, 100, 200, 500, 750, 1000]

# Top-two methods from protocol_core_noisy_20260313_094752.json:
#   3 -> EnhancedHybridSystemDeFi (core)       median R2=0.9999998  wins=19/30
#   4 -> HybridSystemLLMNN all-domains (core)  median R2=0.9999998  wins=11/30
# n=200 already completed — merge it via --existing-results.
_DEFAULT_METHODS: List[int] = [3, 4]


# ============================================================================
# TEE LOGGER
# ============================================================================

class _TeeLogger:
    """Mirror every write to both a real stream and an open log file."""

    def __init__(self, stream, log_file):
        self._stream   = stream
        self._log_file = log_file

    def write(self, data):
        self._stream.write(data)
        self._log_file.write(data)
        self._log_file.flush()

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    def fileno(self):
        return self._stream.fileno()

    def isatty(self):
        return hasattr(self._stream, "isatty") and self._stream.isatty()


# ============================================================================
# RESULT HELPERS
# ============================================================================

def _find_result_written_after(mode: str, t_start: float) -> Optional[Path]:
    """Return the newest result JSON for mode whose mtime >= t_start."""
    candidates = [
        p for p in _RESULTS_DIR.glob(f"protocol_core_{mode}_*.json")
        if p.stat().st_mtime >= t_start
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_latest_result(mode: str) -> Optional[Path]:
    """Return the most-recently-modified JSON for protocol_core_{mode}_*.json."""
    pattern    = f"protocol_core_{mode}_*.json"
    candidates = sorted(
        _RESULTS_DIR.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _extract_per_test(data: Dict) -> Dict[str, Dict]:
    """Return {equation_name: {method_name: result_dict}}."""
    out: Dict[str, Dict] = {}
    for test in data.get("tests", []):
        eq_name = (
            test.get("metadata", {}).get("equation_name", "")
            or test.get("description", "unknown")
        )
        key = test.get("metadata", {}).get("equation_name", eq_name) or eq_name[:50]
        out[key] = {name: res for name, res in test.get("results", {}).items()}
    return out


def _load_results(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


# ============================================================================
# SUBPROCESS BUILDER
# ============================================================================

def _build_runner_cmd(
    n_samples:  int,
    args:       argparse.Namespace,
    runner:     Path,
) -> List[str]:
    """Build the subprocess command for one sample-size configuration."""
    cmd = [sys.executable, str(runner)]

    # ── Noise / noiseless mode ────────────────────────────────────────────────
    noiseless = getattr(args, "noiseless", False)
    if noiseless:
        cmd.append("--noiseless")
        cmd += ["--threshold", str(args.threshold_noiseless)]
    else:
        cmd += ["--threshold", str(args.threshold_noisy)]

    # ── Sample count — the variable being swept ───────────────────────────────
    cmd += ["--samples", str(n_samples)]

    # ── Always-emitted stable flags ───────────────────────────────────────────
    cmd += ["--nn-seeds",       str(args.nn_seeds)]
    cmd += ["--method-timeout", str(args.method_timeout)]
    cmd += ["--pysr-timeout",   str(args.pysr_timeout)]

    # ── Optional method selection ─────────────────────────────────────────────
    if args.methods:
        cmd += ["--methods"] + [str(m) for m in args.methods]

    # ── Optional filters ──────────────────────────────────────────────────────
    if args.skip_pysr:
        cmd.append("--skip-pysr")
    if getattr(args, "test", None):
        cmd += ["--test", args.test]
    if getattr(args, "equations", None):
        cmd += ["--equations"] + [str(e) for e in args.equations]
    if getattr(args, "domain", "all_domains") != "all_domains":
        cmd += ["--domain", args.domain]
    if getattr(args, "series", None):
        cmd += ["--series", args.series]
    if getattr(args, "benchmark", "feynman") != "feynman":
        cmd += ["--benchmark", args.benchmark]
    if args.verbose:
        cmd.append("--verbose")
    if getattr(args, "quiet", False):
        cmd.append("--quiet")
    if getattr(args, "no_llm_cache", False):
        cmd.append("--no-llm-cache")

    # ── Give each n its own checkpoint to prevent run collisions ─────────────
    cmd += ["--checkpoint-name", f"sample_complexity_n{n_samples:04d}_checkpoint"]

    return cmd


# ============================================================================
# RUN ONE SAMPLE SIZE
# ============================================================================

def _run_sample_size(
    n_samples:  int,
    args:       argparse.Namespace,
    runner:     Path,
) -> Optional[Path]:
    """
    Run the inner benchmark for one training-set size.

    Returns the Path to the result JSON, or None on failure.
    """
    noiseless  = getattr(args, "noiseless", False)
    noise_val  = 0.0 if noiseless else getattr(args, "fixed_noise", 0.05)
    mode_str   = "noiseless" if noiseless else f"n={n_samples}  σ={noise_val:.4g}"
    mode_file  = "noiseless" if noiseless else "noisy"   # filename-safe mode for glob
    print(f"\n{'='*80}")
    print(f"  SAMPLE COMPLEXITY  ──  n = {n_samples}  (mode: {mode_str})".center(80))
    print(f"{'='*80}\n")

    cmd = _build_runner_cmd(n_samples, args, runner)
    print(f"  Command: {' '.join(cmd)}\n")

    # Inject fixed noise level — noiseless mode uses 0.0, otherwise
    # use args.fixed_noise (default 0.05 = σ=5%).
    child_env = os.environ.copy()
    noise_val = 0.0 if noiseless else getattr(args, "fixed_noise", 0.05)
    child_env["HYPATIAX_NOISE_LEVEL"] = str(noise_val)

    t0 = time.time()
    # EINTR-safe subprocess wrapper — Python 3.12 does not always retry
    # os.waitpid() on EINTR/SIGCHLD, so a bare subprocess.run() can raise
    # ChildProcessError or InterruptedError and crash the entire sweep.
    # We retry up to 3 times on transient OS-level interrupts; genuine
    # non-zero exit codes are still handled normally below.
    _max_retries = 3
    result = None
    for _attempt in range(1, _max_retries + 1):
        try:
            result = subprocess.run(cmd, env=child_env)
            break  # completed (success or non-zero exit) — stop retrying
        except (InterruptedError, ChildProcessError, OSError) as _exc:
            _errno = getattr(_exc, "errno", None)
            import errno as _errno_mod
            if _errno == _errno_mod.EINTR and _attempt < _max_retries:
                print(
                    f"\n  ⚠️   subprocess interrupted (EINTR) for n={n_samples} "
                    f"— retry {_attempt}/{_max_retries - 1} …"
                )
                time.sleep(0.5 * _attempt)  # brief back-off before retry
                continue
            # Non-EINTR OS error or retries exhausted — treat as failure
            print(
                f"\n  ❌  subprocess raised {type(_exc).__name__} "
                f"(errno={_errno}) for n={n_samples}: {_exc}"
            )
            result = type("_FakeResult", (), {"returncode": 1})()
            break
    if result is None:
        result = type("_FakeResult", (), {"returncode": 1})()
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(
            f"\n  ❌  Runner exited with code {result.returncode} "
            f"for n={n_samples}  ({elapsed/60:.1f} min)."
        )
        if getattr(args, "fail_fast", False):
            print("  ⛔  FAIL-FAST: aborting sample-complexity sweep.")
            sys.exit(result.returncode)
        print("      Attempting to locate partial results…")

    json_path = _find_result_written_after(mode_file, t0)
    if json_path is None:
        json_path = _find_latest_result(mode_file)   # fallback

    if json_path is None:
        print(f"\n  ❌ No result JSON found for n={n_samples}.")
        if getattr(args, "fail_fast", False):
            sys.exit(1)
        return None

    print(f"\n  ✅ n={n_samples} completed in {elapsed/60:.1f} min")
    print(f"     Results: {json_path}")
    return json_path


# ============================================================================
# AGGREGATION
# ============================================================================

def _aggregate_results(
    sample_sizes:    List[int],
    result_paths:    Dict[int, Optional[Path]],
    noiseless:       bool,
    args_thresholds: Dict = {},
) -> Dict:
    """
    Build a unified cross-n comparison object.

    Schema
    ------
    {
      "generated":    ISO timestamp,
      "sample_sizes": [50, 100, 200, 500],
      "mode":         "noiseless" | "noisy",
      "methods":      ["PureLLMBaseline", ...],
      "per_n": {
        "50":  {
          "method_summary": {
            "<method>": {
              "median_r2": float, "mean_r2": float, "std_r2": float,
              "recovery_rate": float,
              "n_success": int, "n_total": int,
              "threshold_used": float
            }, ...
          },
          "per_equation": { "<eq>": {"<method>": {"r2", "rmse", "success"}}, ... }
        }, ...
      },
      "data_efficiency": {
        "<method>": {
          "min_n_above_threshold": int | null,   # smallest n where recovery_rate >= target
          "recovery_curve": {"50": float, "100": float, ...}
        }, ...
      }
    }
    """
    all_methods: set = set()
    loaded: Dict[int, Optional[Dict]] = {}

    for n in sample_sizes:
        path = result_paths.get(n)
        if path is None:
            loaded[n] = None
            continue
        try:
            data = _load_results(path)
        except Exception as exc:
            print(f"  ⚠️  Cannot load {path}: {exc}")
            loaded[n] = None
            continue
        loaded[n] = data
        for eq_methods in _extract_per_test(data).values():
            all_methods.update(eq_methods.keys())

    all_methods_sorted = sorted(all_methods)
    # Per-n threshold: tighter as sample size grows — more data means
    # we expect better recovery. Noiseless mode uses higher baselines.
    if noiseless:
        _default_thresholds = {
            50:  args_thresholds.get(50,  0.9990),
            100: args_thresholds.get(100, 0.9995),
            200: args_thresholds.get(200, 0.9999),
            500: args_thresholds.get(500, 0.9999),
        }
    else:
        # Fixed σ=0.05 → flat 0.995 across all n values
        _default_thresholds = {
            50:   args_thresholds.get(50,   0.995),
            100:  args_thresholds.get(100,  0.995),
            200:  args_thresholds.get(200,  0.995),
            500:  args_thresholds.get(500,  0.995),
            750:  args_thresholds.get(750,  0.995),
            1000: args_thresholds.get(1000, 0.995),
        }

    per_n_data: Dict[str, Optional[Dict]] = {}
    for n in sample_sizes:
        n_str     = str(n)
        data      = loaded.get(n)
        if data is None:
            per_n_data[n_str] = None
            continue

        # Threshold for this specific n
        threshold = _default_thresholds.get(
            n,
            args_thresholds.get(n, 0.9999 if noiseless else 0.995)
        )

        per_eq = _extract_per_test(data)
        method_summary: Dict[str, Dict] = {}

        for method in all_methods_sorted:
            r2_vals:   List[float] = []
            n_success  = 0
            n_total    = 0
            n_recovery = 0

            for eq_results in per_eq.values():
                n_total += 1
                res = eq_results.get(method, {})
                if res.get("success", False):
                    n_success += 1
                    r2 = res.get("r2")
                    if r2 is not None:
                        try:
                            r2f = float(r2)
                            if np.isfinite(r2f):
                                r2_vals.append(r2f)
                                if r2f >= threshold:
                                    n_recovery += 1
                        except (TypeError, ValueError):
                            pass

            method_summary[method] = {
                "median_r2":     float(np.median(r2_vals))       if r2_vals             else None,
                "mean_r2":       float(np.mean(r2_vals))         if r2_vals             else None,
                "std_r2":        float(np.std(r2_vals, ddof=1))  if len(r2_vals) > 1   else 0.0,
                "recovery_rate": n_recovery / n_total             if n_total > 0         else None,
                "n_success":     n_success,
                "n_total":       n_total,
                "threshold_used": threshold,
            }

        per_n_data[n_str] = {
            "method_summary": method_summary,
            "per_equation": {
                eq: {
                    m: {
                        "r2":      r.get("r2"),
                        "rmse":    r.get("rmse"),
                        "success": r.get("success", False),
                    }
                    for m, r in mdict.items()
                }
                for eq, mdict in per_eq.items()
            },
        }

    # ── Data-efficiency summary ───────────────────────────────────────────────
    # min_n_above_threshold: smallest n where recovery_rate >= 0.5 (majority of eqs recovered)
    _efficiency_target = 0.5

    data_efficiency: Dict[str, Dict] = {}
    for method in all_methods_sorted:
        curve: Dict[str, Optional[float]] = {}
        min_n: Optional[int] = None
        for n in sample_sizes:
            n_str = str(n)
            pnd   = per_n_data.get(n_str)
            if pnd is None:
                curve[n_str] = None
                continue
            rec = pnd["method_summary"].get(method, {}).get("recovery_rate")
            curve[n_str] = rec
            if rec is not None and rec >= _efficiency_target and min_n is None:
                min_n = n
        data_efficiency[method] = {
            "min_n_above_threshold": min_n,
            "efficiency_target":     _efficiency_target,
            "recovery_curve":        curve,
        }

    return {
        "generated":        datetime.now().isoformat(),
        "sample_sizes":     sample_sizes,
        "mode":             "noiseless" if noiseless else "noisy",
        "threshold":        _default_thresholds,
        "methods":          all_methods_sorted,
        "per_n":            per_n_data,
        "data_efficiency":  data_efficiency,
    }


# ============================================================================
# REPORTING
# ============================================================================

def _print_sample_complexity_table(agg: Dict) -> None:
    """Pretty-print the cross-n comparison tables to stdout."""
    methods      = agg["methods"]
    sample_sizes = agg["sample_sizes"]

    if not methods or not sample_sizes:
        print("  (no data to display)")
        return

    col_w = 11

    # ── Median R² table ────────────────────────────────────────────────────────
    header_n = "".join(f"  n={n}".rjust(col_w) for n in sample_sizes)

    print(f"\n{'='*100}")
    print(f"  SAMPLE COMPLEXITY SUMMARY  —  Median R²  (mode: {agg['mode']})".center(100))
    print(f"{'='*100}")
    print(f"  {'Method':<42}" + header_n)
    print("  " + "-" * 98)

    for method in sorted(methods):
        row = f"  {method:<42}"
        for n in sample_sizes:
            pnd = agg["per_n"].get(str(n))
            if pnd is None:
                row += f"{'N/A':>{col_w}}"
                continue
            ms  = pnd["method_summary"].get(method, {})
            med = ms.get("median_r2")
            row += f"{(f'{med:.4f}' if med is not None else 'N/A'):>{col_w}}"
        print(row)

    # ── Recovery rate table ────────────────────────────────────────────────────
    threshold = agg.get("threshold", 0.995)
    print(f"\n{'='*100}")
    print(
        f"  SAMPLE COMPLEXITY SUMMARY  —  Recovery Rate  "
        f"(R² ≥ {threshold},  mode: {agg['mode']})".center(100)
    )
    print(f"{'='*100}")
    print(f"  {'Method':<42}" + header_n)
    print("  " + "-" * 98)

    for method in sorted(methods):
        row = f"  {method:<42}"
        for n in sample_sizes:
            pnd = agg["per_n"].get(str(n))
            if pnd is None:
                row += f"{'N/A':>{col_w}}"
                continue
            ms  = pnd["method_summary"].get(method, {})
            rec = ms.get("recovery_rate")
            row += f"{(f'{rec*100:.1f}%' if rec is not None else 'N/A'):>{col_w}}"
        print(row)

    # ── Data-efficiency summary ────────────────────────────────────────────────
    eff_target = list(agg["data_efficiency"].values())[0]["efficiency_target"] \
        if agg["data_efficiency"] else 0.5

    print(f"\n{'='*80}")
    print(
        f"  DATA EFFICIENCY  —  smallest n where recovery rate ≥ "
        f"{eff_target*100:.0f}%".center(80)
    )
    print(f"{'='*80}")
    print(f"  {'Method':<42} {'Min n':>8}  {'Recovery curve'}")
    print("  " + "-" * 78)

    for method in sorted(methods):
        de  = agg["data_efficiency"].get(method, {})
        min_n  = de.get("min_n_above_threshold")
        min_n_s = str(min_n) if min_n is not None else "never"
        curve  = de.get("recovery_curve", {})
        curve_s = "  ".join(
            f"n={n}:{(f'{curve[str(n)]*100:.0f}%' if curve.get(str(n)) is not None else 'N/A')}"
            for n in sample_sizes
        )
        print(f"  {method:<42} {min_n_s:>8}  {curve_s}")

    print(f"{'='*100}\n")


def _save_complexity_json(agg: Dict, ts: str) -> Path:
    path = _RESULTS_DIR / f"sample_complexity_{ts}.json"
    with open(path, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"  💾 Sample complexity JSON → {path}")
    return path


def _save_complexity_csv(agg: Dict, ts: str) -> Path:
    """
    Two-section flat CSV:
    Section 1 — per (method, n) aggregate metrics.
    Section 2 — per (method, n, equation) individual R² values.
    """
    path = _RESULTS_DIR / f"sample_complexity_{ts}.csv"

    # Unified fieldnames covering both sections.  Aggregate-only columns are
    # empty strings in per-equation rows; equation-level columns are empty in
    # aggregate rows.  Previously fieldnames_eq was defined but never used and
    # r2/rmse/success were silently dropped from the per-equation output.
    fieldnames = [
        "section", "method", "n_samples",
        "median_r2", "mean_r2", "std_r2",
        "recovery_rate", "n_success", "n_total", "threshold_used",
        "min_n_above_threshold",
        "equation", "r2", "rmse", "success",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        # Section 1: aggregate
        for n in agg["sample_sizes"]:
            n_str = str(n)
            pnd   = agg["per_n"].get(n_str)
            for method in agg["methods"]:
                min_n = agg["data_efficiency"].get(method, {}).get("min_n_above_threshold")
                if pnd is None:
                    writer.writerow({
                        "section": "aggregate", "method": method, "n_samples": n,
                        "median_r2": None, "mean_r2": None, "std_r2": None,
                        "recovery_rate": None, "n_success": None, "n_total": None,
                        "threshold_used": None, "min_n_above_threshold": min_n,
                        "equation": "", "r2": "", "rmse": "", "success": "",
                    })
                    continue
                ms = pnd["method_summary"].get(method, {})
                writer.writerow({
                    "section":               "aggregate",
                    "method":                method,
                    "n_samples":             n,
                    "median_r2":             ms.get("median_r2"),
                    "mean_r2":               ms.get("mean_r2"),
                    "std_r2":                ms.get("std_r2"),
                    "recovery_rate":         ms.get("recovery_rate"),
                    "n_success":             ms.get("n_success"),
                    "n_total":               ms.get("n_total"),
                    "threshold_used":        ms.get("threshold_used"),
                    "min_n_above_threshold": min_n,
                    "equation": "", "r2": "", "rmse": "", "success": "",
                })

        # Section 2: per-equation detail
        for n in agg["sample_sizes"]:
            n_str = str(n)
            pnd   = agg["per_n"].get(n_str)
            if pnd is None:
                continue
            for eq, eq_methods in pnd.get("per_equation", {}).items():
                for method, res in eq_methods.items():
                    writer.writerow({
                        "section":               "per_equation",
                        "method":                method,
                        "n_samples":             n,
                        "median_r2":             "",
                        "mean_r2":               "",
                        "std_r2":                "",
                        "recovery_rate":         "",
                        "n_success":             "",
                        "n_total":               "",
                        "threshold_used":        "",
                        "min_n_above_threshold": "",
                        "equation":              eq,
                        "r2":                    res.get("r2"),
                        "rmse":                  res.get("rmse"),
                        "success":               res.get("success"),
                    })

    print(f"  📊 Sample complexity CSV  → {path}")
    return path


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample-complexity sweep benchmark  n ∈ {50,100,200,500}  "
            "for the HypatiaX suite.  Addresses data-efficiency reviewer questions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Sample sizes ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--sample-sizes",
        type=int, nargs="+",
        default=_DEFAULT_SAMPLE_SIZES,
        dest="sample_sizes",
        metavar="N",
        help=(
            "Training-set sizes to sweep. "
            f"Default: {_DEFAULT_SAMPLE_SIZES}"
        ),
    )

    # ── Method selection ──────────────────────────────────────────────────────
    # ── Merge pre-existing results (e.g. n=200 already done) ─────────────────
    parser.add_argument(
        "--existing-results",
        nargs="+", default=[], dest="existing_results", metavar="N:PATH",
        help=(
            "Pre-existing result JSONs to merge without re-running. "
            "Format: n_value:path  e.g.  200:results/noisy_200.json"
        ),
    )

    parser.add_argument(
        "--methods",
        type=int, nargs="+",
        default=_DEFAULT_METHODS,
        metavar="N",
        help=(
            "Core method indices (1-6). "
            "Default: [3, 4]  (EnhancedHybridDeFi + HybridLLMNN-all-domains)."
        ),
    )

    # ── Noise / noiseless mode ────────────────────────────────────────────────
    parser.add_argument(
        "--noiseless",
        action="store_true",
        help=(
            "Run all passes with noise_level=0.0 (directly comparable to "
            "published SR figures). Default: noisy (noise_level=0.05)."
        ),
    )
    parser.add_argument(
        "--fixed-noise", type=float, default=0.05, dest="fixed_noise",
        metavar="SIGMA",
        help=(
            "Fixed noise level (σ) injected as HYPATIAX_NOISE_LEVEL for every "
            "n run. Ignored when --noiseless is set. Default: 0.05 (σ=5%%)."
        ),
    )
    parser.add_argument(
        "--threshold-noisy",
        type=float, default=0.995,
        dest="threshold_noisy",
        help="R² recovery threshold for noisy passes at fixed σ=0.05 (default: 0.995).",
    )
    parser.add_argument(
        "--threshold-noiseless",
        type=float, default=0.9999,
        dest="threshold_noiseless",
        help="R² recovery threshold for noiseless passes (default: 0.9999).",
    )
    parser.add_argument(
        "--threshold-per-n", nargs="+", default=[], dest="threshold_per_n",
        metavar="N:VALUE",
        help=(
            "Per-n threshold overrides. "
            "Format: n:value  e.g.  50:0.988  100:0.992  500:0.998. "
            "Noisy defaults (flat, fixed σ=0.05): all n ∈ {50,100,200,500,750,1000} -> 0.995. "
            "Noiseless defaults: 50->0.9990 100->0.9995 200->0.9999 500->0.9999"
        ),
    )

    # ── Standard runner flags ─────────────────────────────────────────────────
    parser.add_argument(
        "--nn-seeds",      type=int,  default=3,   dest="nn_seeds",
        help="NN ensemble seeds per equation (default: 3).",
    )
    parser.add_argument(
        "--method-timeout", type=int, default=900,  dest="method_timeout",
        help="Per-method wall-clock timeout in seconds (default: 900).",
    )
    parser.add_argument(
        "--pysr-timeout",  type=int,  default=1100, dest="pysr_timeout",
        help="PySR wall-clock timeout in seconds (default: 1100).",
    )
    parser.add_argument("--skip-pysr",    action="store_true", dest="skip_pysr")
    parser.add_argument("--test",         type=str,  default=None)
    parser.add_argument("--equations",    type=int,  nargs="+", metavar="N", default=None)
    parser.add_argument("--domain",       type=str,  default="all_domains")
    parser.add_argument("--series",       choices=["I", "II", "III", "crossover"], default=None)
    parser.add_argument("--benchmark",    choices=["feynman", "srbench", "both"], default="feynman")
    parser.add_argument("--verbose",      action="store_true")
    parser.add_argument("--quiet",        action="store_true")
    parser.add_argument("--no-llm-cache", action="store_true", dest="no_llm_cache")

    # ── Fail-fast ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--fail-fast", action="store_true", dest="fail_fast",
        help="Abort the sweep if any runner subprocess exits non-zero.",
    )

    # ── Log file ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--log", type=str, default=None, metavar="FILE",
        help="Append orchestrator-level output to FILE (tee-style).",
    )

    # ── Runner path override ──────────────────────────────────────────────────
    parser.add_argument(
        "--runner", type=str, default=None,
        help="Path to run_comparative_suite_benchmark_v2.py (auto-detected).",
    )

    args = parser.parse_args()

    # ── Optional tee logging ──────────────────────────────────────────────────
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _log_fh    = open(log_path, "a", buffering=1)
        sys.stdout = _TeeLogger(sys.stdout, _log_fh)
        sys.stderr = _TeeLogger(sys.stderr, _log_fh)
        print(f"  📝 Logging to: {log_path}  (append mode)")

    # ── Locate runner script ──────────────────────────────────────────────────
    runner_path = Path(args.runner) if args.runner else _RUNNER
    if not runner_path.exists():
        print(
            f"❌  Cannot find runner script: {runner_path}\n"
            f"    Pass --runner /path/to/run_comparative_suite_benchmark_v2.py"
        )
        sys.exit(1)

    # Parse --existing-results tokens
    existing_map: Dict[int, Path] = {}
    for token in getattr(args, "existing_results", []):
        try:
            n_s, path_s = token.split(":", 1)
            n = int(n_s)
            p = Path(path_s)
            if not p.exists():
                print(f"  WARNING: --existing-results file not found for n={n}: {p}")
            else:
                existing_map[n] = p
                print(f"  Merging existing result  n={n}  -> {p.name}")
        except ValueError:
            print(f"  WARNING: Cannot parse '{token}'  (expected n:path)")

    # Deduplicate and sort sample sizes
    sample_sizes = sorted(set(args.sample_sizes))

    noiseless   = getattr(args, "noiseless", False)
    mode_label  = "noiseless (σ=0)" if noiseless else "noisy (σ=0.05)"
    threshold   = args.threshold_noiseless if noiseless else args.threshold_noisy
    methods_str = str(args.methods) if args.methods else "all available"

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SAMPLE COMPLEXITY BENCHMARK RUNNER".center(80))
    print(f"{'='*80}")
    print(f"  Runner script   : {runner_path}")
    print(f"  Results dir     : {_RESULTS_DIR}")
    sizes_to_show = sorted(set(sample_sizes) | set(existing_map.keys()))
    print(f"  Sample sizes    : n ∈ {sample_sizes}  (run)")
    if existing_map:
        print(f"  Sizes MERGED    : n ∈ {sorted(existing_map.keys())}  (pre-existing)")
    fixed_noise_val = 0.0 if noiseless else getattr(args, "fixed_noise", 0.05)
    print(f"  Mode            : {mode_label}  (HYPATIAX_NOISE_LEVEL={fixed_noise_val:.4g})")
    if noiseless:
        _thresh_display = {50: 0.9990, 100: 0.9995, 200: 0.9999, 500: 0.9999}
    else:
        _thresh_display = {50: 0.995, 100: 0.995, 200: 0.995, 500: 0.995, 750: 0.995, 1000: 0.995}
    for tok in getattr(args, "threshold_per_n", []):
        try:
            _n, _v = tok.split(":")
            _thresh_display[int(_n)] = float(_v)
        except ValueError:
            pass
    print(f"  R² threshold    : per-n → {_thresh_display}")
    print(f"  Methods         : {methods_str}  (3=EnhancedHybridDeFi, 4=HybridLLMNN)")
    print(f"  NN seeds        : {args.nn_seeds}")
    print(f"  Method timeout  : {args.method_timeout}s")
    print(f"  PySR timeout    : {args.pysr_timeout}s")
    print(f"  Fail-fast       : {'ON  ⛔' if args.fail_fast else 'OFF (warn and continue)'}")
    if args.log:
        print(f"  Log file        : {args.log}")
    print(f"{'='*80}\n")

    ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_paths: Dict[int, Optional[Path]] = dict(existing_map)
    sweep_start   = time.time()

    # ── SWEEP LOOP ────────────────────────────────────────────────────────────
    sizes_to_run = [n for n in sample_sizes if n not in existing_map]
    for i, n in enumerate(sizes_to_run, 1):
        print(f"\n  [{i}/{len(sizes_to_run)}]  n = {n} samples")
        result_paths[n] = _run_sample_size(n, args, runner_path)

    total_elapsed = time.time() - sweep_start
    print(f"\n  ⏱  Total sweep time: {total_elapsed/60:.1f} min\n")

    # ── AGGREGATION & REPORT ──────────────────────────────────────────────────
    n_available = sum(1 for v in result_paths.values() if v is not None)
    if n_available == 0:
        print("❌  No result JSONs available — cannot build comparison report.")
        sys.exit(1)

    if n_available < len(sample_sizes):
        missing = [str(n) for n, p in result_paths.items() if p is None]
        print(f"  ⚠️  Some sample sizes produced no results: n ∈ {{{', '.join(missing)}}}")
        print(f"      Proceeding with the {n_available} available result(s).\n")

    print(f"\n{'='*80}")
    print("  GENERATING SAMPLE COMPLEXITY REPORT".center(80))
    print(f"{'='*80}")

    # Build per-n threshold map
    args_thresholds: Dict = {}
    for token in getattr(args, "threshold_per_n", []):
        try:
            n_str, v = token.split(":")
            args_thresholds[int(n_str)] = float(v)
        except ValueError:
            print(f"  WARNING: bad --threshold-per-n token: {token!r}")
    agg = _aggregate_results(sample_sizes, result_paths,
                             noiseless=noiseless,
                             args_thresholds=args_thresholds)
    _print_sample_complexity_table(agg)
    _save_complexity_json(agg, ts)
    _save_complexity_csv(agg, ts)

    print(f"\n✅  Sample complexity sweep complete.\n")


if __name__ == "__main__":
    main()
