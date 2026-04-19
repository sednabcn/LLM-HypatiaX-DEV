#!/usr/bin/env python3
"""
run_noise_sweep_benchmark.py
============================

Noise-level sweep experiment for the HypatiaX benchmark suite.

Runs the top two methods — EnhancedHybridSystemDeFi (method 3) and
HybridSystemLLMNN all-domains (method 4) — across the remaining noise
levels sigma in {1%, 5%, 10%}.

Background
----------
sigma=0% (noiseless) and sigma=0.5% are already completed and on disk.
Pass those result files via --existing-results so they are merged into
the final report without re-running.

The two target methods were identified from the noisy benchmark results
(protocol_core_noisy_20260313_094752.json):

    EnhancedHybridSystemDeFi (core)        median R2 = 0.9999998   wins 18/30  avg 19.7s
    HybridSystemLLMNN all-domains (core)   median R2 = 0.9999998   wins 11/30  avg 10.8s

Both are in a separate tier from the remaining four methods
(median R2 approx 0.9977).

Important per-equation notes
-----------------------------
  - M4 (HybridSystemLLMNN) has a catastrophic failure on "Newton's
    gravitational force" at sigma=5% in the baseline run: R2=0.643.
    This is flagged in the report and should be verified across seeds.
  - M4 is on average 1.8x faster than M3 (10.8s vs 19.7s per equation).

What it does
------------
1. Optionally merges pre-existing result JSONs via --existing-results
   (list of  sigma_value:path  pairs, no spaces around the colon).
2. For each sigma in --noise-levels that is NOT already covered, launches
   run_comparative_suite_benchmark_v2.py as a subprocess.
3. Uses mtime-gating to match each run to its result file, preventing
   collisions between the three noisy passes that all write
   protocol_core_noisy_*.json.
4. Saves
       data/results/comparison_results/noise_sweep_<TS>.json
       data/results/comparison_results/noise_sweep_<TS>.csv

Usage
-----
    # Default: methods 3 & 4, sigma in {1%,5%,10%}
    python run_noise_sweep_benchmark.py

    # Merge the already-completed sigma=0% and sigma=0.5% runs:
    python run_noise_sweep_benchmark.py \\
        --existing-results 0.0:path/to/noiseless.json 0.005:path/to/sig0005.json

    # Full five-level sweep (re-running sigma=0 and sigma=0.5% too):
    python run_noise_sweep_benchmark.py --noise-levels 0.0 0.005 0.01 0.05 0.10

    # Quick smoke-test on one equation
    python run_noise_sweep_benchmark.py --test arrhenius

    # Abort on any failure
    python run_noise_sweep_benchmark.py --fail-fast

Outputs
-------
  data/results/comparison_results/noise_sweep_<TS>.json
  data/results/comparison_results/noise_sweep_<TS>.csv
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
from typing import Dict, List, Optional, Tuple

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
_PKG_ROOT    = _HERE.parent.parent
_RUNNER      = _HERE / "run_comparative_suite_benchmark_v2.py"
_RESULTS_DIR = _PKG_ROOT / "data/results/comparison_results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default noise levels: sigma=0% and sigma=0.5% already completed.
_DEFAULT_NOISE_LEVELS: List[float] = [0.01, 0.05, 0.10]

# Top-two methods from protocol_core_noisy_20260313_094752.json:
#   3 -> EnhancedHybridSystemDeFi (core)       median R2=0.9999998  wins=18/30
#   4 -> HybridSystemLLMNN all-domains (core)  median R2=0.9999998  wins=11/30
_DEFAULT_METHODS: List[int] = [3, 4]

# R2 below this triggers a catastrophic-failure flag in the report
_CATASTROPHIC_R2_THRESHOLD: float = 0.90


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
    """
    Return the newest result JSON for *mode* whose mtime is >= t_start.

    Prevents the collision where three sequential sigma>0 passes all write
    protocol_core_noisy_*.json and a naive find-latest would return the
    same file for all three.
    """
    candidates = [
        p for p in _RESULTS_DIR.glob(f"protocol_core_{mode}_*.json")
        if p.stat().st_mtime >= t_start
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_latest_result(mode: str) -> Optional[Path]:
    """Return the most-recently-modified JSON for protocol_core_{mode}_*.json."""
    candidates = sorted(
        _RESULTS_DIR.glob(f"protocol_core_{mode}_*.json"),
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
    noise_level: float,
    args:        argparse.Namespace,
    runner:      Path,
) -> Tuple[List[str], str]:
    """
    Build the subprocess command for one noise level.

    Returns (cmd_list, sigma_label) where sigma_label encodes the sigma
    as an integer number of thousandths, giving unique checkpoint names:
      0.000 -> sig0000
      0.005 -> sig0005
      0.010 -> sig0010
      0.050 -> sig0050
      0.100 -> sig0100
    """
    sigma_label = f"sig{int(round(noise_level * 1000)):04d}"
    cmd = [sys.executable, str(runner)]

    if noise_level == 0.0:
        cmd.append("--noiseless")
        cmd += ["--threshold", str(args.threshold_noiseless)]
    else:
        cmd += ["--threshold", str(args.threshold_noisy)]

    cmd += ["--samples",        str(args.samples)]
    cmd += ["--nn-seeds",       str(args.nn_seeds)]
    cmd += ["--method-timeout", str(args.method_timeout)]
    cmd += ["--pysr-timeout",   str(args.pysr_timeout)]
    cmd += ["--methods"] + [str(m) for m in args.methods]

    if args.skip_pysr:
        cmd.append("--skip-pysr")
    if getattr(args, "test", None):
        # Normalize common human-readable aliases to canonical Feynman IDs.
        # The inner runner (run_comparative_suite_benchmark_v2.py) only
        # accepts canonical IDs (e.g. "I.12.1"); passing a bare name like
        # "newton" causes an exit-code-1 crash with no result JSON written.
        _TEST_ALIASES: dict = {
            # Newton's gravitational force  F = G*m1*m2/r^2
            "newton":          "I.12.1",
            "newton_gravity":  "I.12.1",
            "gravity":         "I.12.1",
            # Coulomb's law  F = q1*q2/(4*pi*eps0*r^2)
            "coulomb":         "I.12.2",
            # Kinetic energy  E = 0.5*m*v^2
            "kinetic_energy":  "I.12.4",
            "kinetic":         "I.12.4",
            # Arrhenius  k = A*exp(-Ea/(R*T))
            "arrhenius":       "II.11.27",
            # Ideal gas  P*V = n*R*T
            "ideal_gas":       "II.11.28",
        }
        canonical = _TEST_ALIASES.get(args.test.lower().replace(" ", "_"), args.test)
        if canonical != args.test:
            print(f"  [INFO] --test alias '{args.test}' resolved to canonical ID '{canonical}'")
        cmd += ["--test", canonical]
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

    # Unique checkpoint per sigma — prevents noisy passes colliding.
    cmd += ["--checkpoint-name", f"noise_sweep_{sigma_label}_checkpoint"]

    return cmd, sigma_label


# ============================================================================
# RUN ONE NOISE LEVEL
# ============================================================================

def _run_noise_level(
    noise_level: float,
    args:        argparse.Namespace,
    runner:      Path,
) -> Optional[Path]:
    """
    Run the inner benchmark for one noise level and return the result JSON path.

    Noise is injected via HYPATIAX_NOISE_LEVEL env var.
    The result JSON is identified by mtime >= run-start time to avoid
    the collision bug where all three noisy passes write to the same
    protocol_core_noisy_*.json filename pattern.
    """
    pct = f"{noise_level*100:.4g}%"
    print(f"\n{'='*80}")
    print(f"  NOISE SWEEP  --  sigma={pct}  (noise_level={noise_level:.4f})".center(80))
    print(f"{'='*80}\n")

    cmd, _label = _build_runner_cmd(noise_level, args, runner)
    child_env   = os.environ.copy()
    child_env["HYPATIAX_NOISE_LEVEL"] = str(noise_level)

    print(f"  Command: {' '.join(cmd)}\n")
    t_start = time.time()
    result  = subprocess.run(cmd, env=child_env)
    elapsed = time.time() - t_start

    if result.returncode != 0:
        print(f"\n  ERROR: Runner exited with code {result.returncode} "
              f"for noise_level={noise_level:.4f}  ({elapsed/60:.1f} min).")
        if getattr(args, "fail_fast", False):
            print("  FAIL-FAST: aborting noise sweep.")
            sys.exit(result.returncode)
        print("  Attempting to locate partial results...")

    mode      = "noiseless" if noise_level == 0.0 else "noisy"
    json_path = _find_result_written_after(mode, t_start)
    if json_path is None:
        json_path = _find_latest_result(mode)   # fallback

    if json_path is None:
        print(f"\n  ERROR: No result JSON found for noise_level={noise_level:.4f}.")
        if getattr(args, "fail_fast", False):
            sys.exit(1)
        return None

    print(f"\n  OK: noise_level={noise_level:.4f} completed in {elapsed/60:.1f} min")
    print(f"     Results: {json_path}")
    return json_path


# ============================================================================
# AGGREGATION
# ============================================================================

def _aggregate_results(
    noise_levels:    List[float],
    result_paths:    Dict[float, Optional[Path]],
    args_thresholds: Dict = {},
) -> Dict:
    """
    Build a unified cross-noise comparison object.

    per_noise schema per sigma level:
      method_summary  : median_r2, mean_r2, std_r2, recovery_rate,
                        n_success, n_total, threshold_used, n_catastrophic
      per_equation    : {eq: {method: {r2, rmse, success, catastrophic}}}
      catastrophic_failures: [{equation, method, r2}]  (R2 < 0.90)

    cross_noise_summary: {method: {sigma_str: {median_r2, recovery_rate,
                                               std_r2, n_catastrophic}}}
    """
    loaded: Dict[float, Optional[Dict]] = {}
    all_methods: set = set()

    for sigma in noise_levels:
        path = result_paths.get(sigma)
        if path is None:
            loaded[sigma] = None
            continue
        try:
            data = _load_results(path)
        except Exception as exc:
            print(f"  WARNING: Cannot load {path}: {exc}")
            loaded[sigma] = None
            continue
        loaded[sigma] = data
        for eq_methods in _extract_per_test(data).values():
            all_methods.update(eq_methods.keys())

    all_methods_sorted = sorted(all_methods)
    per_noise_data: Dict[str, Optional[Dict]] = {}

    for sigma in noise_levels:
        sigma_str = f"{sigma:.4f}"
        data      = loaded.get(sigma)
        if data is None:
            per_noise_data[sigma_str] = None
            continue

        per_eq    = _extract_per_test(data)
        # Per-sigma threshold: tighter for low-noise, looser as noise grows.
        _sigma_thresholds = {
            0.0:   args_thresholds.get(0.0,   0.999999),  # noiseless — near-perfect
            0.005: args_thresholds.get(0.005, 0.995),     # σ=0.5%
            0.01:  args_thresholds.get(0.01,  0.990),     # σ=1%
            0.05:  args_thresholds.get(0.05,  0.950),     # σ=5%
            0.10:  args_thresholds.get(0.10,  0.900),     # σ=10%
        }
        threshold = _sigma_thresholds.get(sigma, args_thresholds.get(sigma, 0.995))
        method_summary: Dict[str, Dict] = {}
        catastrophic_failures: List[Dict] = []

        for method in all_methods_sorted:
            r2_vals: List[float] = []
            n_success = n_total = n_recovery = n_catastrophic = 0

            for eq_name, eq_results in per_eq.items():
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
                                if r2f < _CATASTROPHIC_R2_THRESHOLD:
                                    n_catastrophic += 1
                                    catastrophic_failures.append(
                                        {"equation": eq_name, "method": method, "r2": r2f}
                                    )
                        except (TypeError, ValueError):
                            pass

            method_summary[method] = {
                "median_r2":      float(np.median(r2_vals))         if r2_vals          else None,
                "mean_r2":        float(np.mean(r2_vals))           if r2_vals          else None,
                "std_r2":         float(np.std(r2_vals, ddof=1))    if len(r2_vals) > 1 else 0.0,
                "recovery_rate":  n_recovery / n_total              if n_total > 0      else None,
                "n_success":      n_success,
                "n_total":        n_total,
                "threshold_used": threshold,
                "n_catastrophic": n_catastrophic,
            }

        per_noise_data[sigma_str] = {
            "method_summary":       method_summary,
            "catastrophic_failures": sorted(catastrophic_failures, key=lambda x: x["r2"]),
            "per_equation": {
                eq: {
                    m: {
                        "r2":          eq_res.get("r2"),
                        "rmse":        eq_res.get("rmse"),
                        "success":     eq_res.get("success", False),
                        "catastrophic": (
                            np.isfinite(float(eq_res["r2"]))
                            and float(eq_res["r2"]) < _CATASTROPHIC_R2_THRESHOLD
                        ) if eq_res.get("r2") is not None else False,
                    }
                    for m, eq_res in methods_dict.items()
                }
                for eq, methods_dict in per_eq.items()
            },
        }

    # Cross-noise summary
    cross_noise: Dict[str, Dict] = {}
    for method in all_methods_sorted:
        cross_noise[method] = {}
        for sigma in noise_levels:
            sigma_str = f"{sigma:.4f}"
            pnd = per_noise_data.get(sigma_str)
            if pnd is None:
                cross_noise[method][sigma_str] = None
                continue
            ms = pnd["method_summary"].get(method, {})
            cross_noise[method][sigma_str] = {
                "median_r2":      ms.get("median_r2"),
                "recovery_rate":  ms.get("recovery_rate"),
                "std_r2":         ms.get("std_r2"),
                "n_catastrophic": ms.get("n_catastrophic", 0),
            }

    return {
        "generated":           datetime.now().isoformat(),
        "noise_levels":        noise_levels,
        "methods":             all_methods_sorted,
        "per_noise":           per_noise_data,
        "cross_noise_summary": cross_noise,
    }


# ============================================================================
# REPORTING
# ============================================================================

def _print_noise_sweep_table(agg: Dict) -> None:
    """Pretty-print cross-noise comparison tables and flag catastrophic failures."""
    methods      = agg["methods"]
    noise_levels = agg["noise_levels"]

    if not methods or not noise_levels:
        print("  (no data to display)")
        return

    col_w   = 13
    headers = "".join(f"sigma={s*100:.4g}%".rjust(col_w) for s in noise_levels)

    # Median R2 table
    print(f"\n{'='*100}")
    print("  NOISE SWEEP  --  Median R2 across sigma levels".center(100))
    print(f"{'='*100}")
    print(f"  {'Method':<44}" + headers)
    print("  " + "-" * 98)
    for method in sorted(methods):
        row = f"  {method:<44}"
        for sigma in noise_levels:
            pnd = agg["per_noise"].get(f"{sigma:.4f}")
            if pnd is None:
                row += f"{'N/A':>{col_w}}"
                continue
            ms  = pnd["method_summary"].get(method, {})
            med = ms.get("median_r2")
            cat = ms.get("n_catastrophic", 0)
            val = (f"{med:.5f}" if med is not None else "N/A") + (" (!)  " if cat > 0 else "")
            row += f"{val:>{col_w}}"
        print(row)

    # Recovery rate table
    print(f"\n{'='*100}")
    print("  NOISE SWEEP  --  Recovery Rate  (R2 >= threshold)".center(100))
    print(f"  threshold: sigma=0% -> 0.9999    sigma>0% -> 0.995".center(100))
    print(f"{'='*100}")
    print(f"  {'Method':<44}" + headers)
    print("  " + "-" * 98)
    for method in sorted(methods):
        row = f"  {method:<44}"
        for sigma in noise_levels:
            pnd = agg["per_noise"].get(f"{sigma:.4f}")
            if pnd is None:
                row += f"{'N/A':>{col_w}}"
                continue
            ms  = pnd["method_summary"].get(method, {})
            rec = ms.get("recovery_rate")
            val = f"{rec*100:.1f}%" if rec is not None else "N/A"
            row += f"{val:>{col_w}}"
        print(row)
    print(f"  (!) = one or more equations with R2 < {_CATASTROPHIC_R2_THRESHOLD} (catastrophic)")

    # Catastrophic failures block
    all_cats = []
    for sigma in noise_levels:
        pnd = agg["per_noise"].get(f"{sigma:.4f}")
        if pnd:
            for cf in pnd.get("catastrophic_failures", []):
                all_cats.append({**cf, "sigma": sigma})

    if all_cats:
        print(f"\n{'='*100}")
        print("  CATASTROPHIC FAILURES  (R2 < {:.2f})".format(_CATASTROPHIC_R2_THRESHOLD).center(100))
        print(f"{'='*100}")
        print(f"  {'sigma':>8}  {'Method':<44}  {'Equation':<38}  {'R2':>8}")
        print("  " + "-" * 96)
        for cf in sorted(all_cats, key=lambda x: x["r2"]):
            eq_short = cf["equation"][:38]
            print(f"  {cf['sigma']*100:>7.4g}%  {cf['method']:<44}  {eq_short:<38}  {cf['r2']:>8.4f}")
        print()
        print("  NOTE: M4 (HybridSystemLLMNN) catastrophically failed on 'Newton's gravitational")
        print("  force' at sigma=5% in the baseline run (R2=0.643). Verify this is reproducible")
        print("  across multiple seeds and report it explicitly in the paper -- it is a genuine")
        print("  failure mode that reviewers will ask about.")

    print(f"{'='*100}\n")


def _save_sweep_json(agg: Dict, ts: str) -> Path:
    path = _RESULTS_DIR / f"noise_sweep_{ts}.json"
    with open(path, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"  Saved noise sweep JSON  -> {path}")
    return path


def _save_sweep_csv(agg: Dict, ts: str) -> Path:
    """
    Two-section CSV:
      section=aggregate    : one row per (method, sigma) -- summary stats
      section=per_equation : one row per (method, sigma, equation) -- individual R2
    """
    path = _RESULTS_DIR / f"noise_sweep_{ts}.csv"
    fieldnames = [
        "section", "method", "noise_level_fraction", "noise_level_pct", "equation",
        "median_r2", "mean_r2", "std_r2", "recovery_rate",
        "n_success", "n_total", "threshold_used", "n_catastrophic",
        "r2", "rmse", "success", "catastrophic",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for sigma in agg["noise_levels"]:
            sigma_str = f"{sigma:.4f}"
            pnd       = agg["per_noise"].get(sigma_str)
            for method in agg["methods"]:
                ms = (pnd or {}).get("method_summary", {}).get(method, {}) if pnd else {}
                # Aggregate row
                writer.writerow({
                    "section": "aggregate", "method": method,
                    "noise_level_fraction": sigma,
                    "noise_level_pct":      f"{sigma*100:.4g}%",
                    "equation": "",
                    "median_r2":      ms.get("median_r2"),
                    "mean_r2":        ms.get("mean_r2"),
                    "std_r2":         ms.get("std_r2"),
                    "recovery_rate":  ms.get("recovery_rate"),
                    "n_success":      ms.get("n_success"),
                    "n_total":        ms.get("n_total"),
                    "threshold_used": ms.get("threshold_used"),
                    "n_catastrophic": ms.get("n_catastrophic", 0),
                    "r2": "", "rmse": "", "success": "", "catastrophic": "",
                })
                if pnd is None:
                    continue
                # Per-equation rows
                for eq, eq_methods in pnd.get("per_equation", {}).items():
                    res = eq_methods.get(method, {})
                    writer.writerow({
                        "section": "per_equation", "method": method,
                        "noise_level_fraction": sigma,
                        "noise_level_pct":      f"{sigma*100:.4g}%",
                        "equation": eq,
                        "median_r2": "", "mean_r2": "", "std_r2": "",
                        "recovery_rate": "", "n_success": "", "n_total": "",
                        "threshold_used": "", "n_catastrophic": "",
                        "r2":          res.get("r2"),
                        "rmse":        res.get("rmse"),
                        "success":     res.get("success"),
                        "catastrophic": res.get("catastrophic", False),
                    })
    print(f"  Saved noise sweep CSV   -> {path}")
    return path


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Noise-level sweep  sigma in {1%,5%,10%}  for "
            "EnhancedHybridDeFi and HybridSystemLLMNN (top two from HypatiaX "
            "noisy benchmark).  Merge pre-existing results with --existing-results."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--noise-levels", type=float, nargs="+",
        default=_DEFAULT_NOISE_LEVELS, dest="noise_levels", metavar="SIGMA",
        help=(
            "Noise levels to run (fractions of signal std). "
            "sigma=0%% and sigma=0.5%% already completed. "
            f"Default: {_DEFAULT_NOISE_LEVELS}"
        ),
    )
    parser.add_argument(
        "--existing-results", nargs="+", default=[], dest="existing_results",
        metavar="SIGMA:PATH",
        help=(
            "Pre-existing result JSONs to merge without re-running. "
            "Format: sigma_value:path  e.g.  0.0:noiseless.json  0.005:sig0005.json"
        ),
    )
    parser.add_argument(
        "--methods", type=int, nargs="+", default=_DEFAULT_METHODS, metavar="N",
        help=(
            "Core method indices (1-6). "
            "Default: [3, 4]  (EnhancedHybridDeFi + HybridLLMNN-all-domains)."
        ),
    )
    parser.add_argument("--threshold-noisy",     type=float, default=0.950,  dest="threshold_noisy",
                        help="R² threshold for σ=5%% (default: 0.950).")
    parser.add_argument("--threshold-noiseless", type=float, default=0.999999, dest="threshold_noiseless",
                        help="R² threshold for σ=0%% (default: 0.999999).")
    parser.add_argument(
        "--threshold-per-sigma", nargs="+", default=[], dest="threshold_per_sigma",
        metavar="SIGMA:VALUE",
        help=(
            "Per-sigma threshold overrides. "
            "Format: sigma:value  e.g.  0.005:0.999  0.01:0.998  0.10:0.990. "
            "Defaults: 0.0->0.999999  0.005->0.995  0.01->0.990  0.05->0.950  0.10->0.900"
        ),
    )
    parser.add_argument("--samples",             type=int,   default=200)
    parser.add_argument("--nn-seeds",            type=int,   default=3,   dest="nn_seeds")
    parser.add_argument("--method-timeout",      type=int,   default=900, dest="method_timeout")
    parser.add_argument("--pysr-timeout",        type=int,   default=1100,dest="pysr_timeout")
    parser.add_argument("--skip-pysr",           action="store_true", dest="skip_pysr")
    parser.add_argument("--test",                type=str,   default=None)
    parser.add_argument("--equations",           type=int,   nargs="+", metavar="N", default=None)
    parser.add_argument("--domain",              type=str,   default="all_domains")
    parser.add_argument("--series",              choices=["I","II","III","crossover"], default=None)
    parser.add_argument("--benchmark",           choices=["feynman","srbench","both"], default="feynman")
    parser.add_argument("--verbose",             action="store_true")
    parser.add_argument("--quiet",               action="store_true")
    parser.add_argument("--no-llm-cache",        action="store_true", dest="no_llm_cache")
    parser.add_argument("--fail-fast",           action="store_true", dest="fail_fast")
    parser.add_argument("--log",                 type=str,   default=None, metavar="FILE")
    parser.add_argument("--runner",              type=str,   default=None)

    args = parser.parse_args()

    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _log_fh    = open(log_path, "a", buffering=1)
        sys.stdout = _TeeLogger(sys.stdout, _log_fh)
        sys.stderr = _TeeLogger(sys.stderr, _log_fh)
        print(f"  Logging to: {log_path}  (append mode)")

    runner_path = Path(args.runner) if args.runner else _RUNNER
    if not runner_path.exists():
        print(f"ERROR: Cannot find runner: {runner_path}\n"
              f"       Pass --runner /path/to/run_comparative_suite_benchmark_v2.py")
        sys.exit(1)

    # Parse --existing-results tokens
    existing_map: Dict[float, Path] = {}
    for token in args.existing_results:
        try:
            sigma_s, path_s = token.split(":", 1)
            sigma = float(sigma_s)
            p     = Path(path_s)
            if not p.exists():
                print(f"  WARNING: --existing-results file not found for sigma={sigma}: {p}")
            else:
                existing_map[sigma] = p
                print(f"  Merging existing result  sigma={sigma*100:.4g}%  -> {p.name}")
        except ValueError:
            print(f"  WARNING: Cannot parse '{token}'  (expected sigma:path)")

    noise_levels   = sorted(set(args.noise_levels))
    all_sigmas     = sorted(set(noise_levels) | set(existing_map.keys()))
    sigmas_to_run  = [s for s in noise_levels if s not in existing_map]

    print(f"\n{'='*80}")
    print("  NOISE SWEEP BENCHMARK RUNNER".center(80))
    print(f"{'='*80}")
    print(f"  Runner          : {runner_path}")
    print(f"  Results dir     : {_RESULTS_DIR}")
    print(f"  Sigma to RUN    : {[f'{s*100:.4g}%' for s in sigmas_to_run]}")
    if existing_map:
        print(f"  Sigma MERGED    : {[f'{s*100:.4g}%' for s in sorted(existing_map)]}")
    print(f"  Methods         : {args.methods}  (3=EnhancedHybridDeFi, 4=HybridLLMNN)")
    print(f"  Samples         : {args.samples}")
    print(f"  NN seeds        : {args.nn_seeds}")
    print(f"  Method timeout  : {args.method_timeout}s")
    print(f"  PySR timeout    : {args.pysr_timeout}s")
    print(f"  Fail-fast       : {'ON' if args.fail_fast else 'OFF'}")
    if args.log:
        print(f"  Log             : {args.log}")
    print(f"{'='*80}\n")

    ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_paths: Dict[float, Optional[Path]] = dict(existing_map)
    sweep_start   = time.time()

    for i, sigma in enumerate(sigmas_to_run, 1):
        print(f"\n  [{i}/{len(sigmas_to_run)}]  sigma = {sigma*100:.4g}%")
        result_paths[sigma] = _run_noise_level(sigma, args, runner_path)

    elapsed_total = time.time() - sweep_start
    print(f"\n  Total sweep time: {elapsed_total/60:.1f} min\n")

    n_available = sum(1 for v in result_paths.values() if v is not None)
    if n_available == 0:
        print("ERROR: No result JSONs available -- cannot build report.")
        sys.exit(1)

    missing = [f"sigma={s*100:.4g}%" for s in all_sigmas if result_paths.get(s) is None]
    if missing:
        print(f"  WARNING: Missing results for {missing} -- proceeding with {n_available} available.\n")

    print(f"\n{'='*80}")
    print("  GENERATING NOISE SWEEP REPORT".center(80))
    print(f"{'='*80}")

    # Build per-sigma threshold map
    args_thresholds: Dict = {
        0.0:   args.threshold_noiseless,           # 0.999999
        0.005: getattr(args, "threshold_sig0005", 0.995),
        0.01:  getattr(args, "threshold_sig001",  0.990),
        0.05:  args.threshold_noisy,               # 0.950
        0.10:  getattr(args, "threshold_sig010",  0.900),
    }
    for token in getattr(args, "threshold_per_sigma", []):
        try:
            s, v = token.split(":")
            args_thresholds[float(s)] = float(v)
        except ValueError:
            print(f"  WARNING: bad --threshold-per-sigma token: {token!r}")
    agg = _aggregate_results(all_sigmas, result_paths, args_thresholds)
    _print_noise_sweep_table(agg)
    _save_sweep_json(agg, ts)
    _save_sweep_csv(agg, ts)

    print(f"\nNoise sweep complete.\n")


if __name__ == "__main__":
    main()
