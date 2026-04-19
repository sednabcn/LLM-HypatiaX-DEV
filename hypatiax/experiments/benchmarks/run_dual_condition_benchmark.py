#!/usr/bin/env python3
"""
run_dual_condition_benchmark.py
================================

Orchestrates **both** the noisy (noise=0.05) and noiseless (noise=0.0)
benchmark passes for the HypatiaX protocol suite and produces a unified
side-by-side comparison report.

What it does
------------
1. Runs the inner benchmark pass with ``noise_level=0.05``  → *noisy pass*.
2. Runs the same inner pass with ``noise_level=0.0``  → *noiseless pass*.
3. Loads both JSON result files and produces a per-equation, per-method
   comparison table and a summary CSV.

The two passes run sequentially in the same process by importing the runner
module and calling ``main()`` twice — each call re-initialises the protocol
and suite objects so there is no state bleed between passes.

Usage
-----
    # Run all 30 Feynman equations, all 6 methods, both conditions:
    python run_dual_condition_benchmark.py

    # Skip PySR-backed methods (faster smoke-test):
    python run_dual_condition_benchmark.py --skip-pysr

    # Run only methods 1 and 2 for a quick check:
    python run_dual_condition_benchmark.py --methods 1 2

    # Single equation:
    python run_dual_condition_benchmark.py --test arrhenius

    # Resume an interrupted noisy pass before running noiseless:
    python run_dual_condition_benchmark.py --resume-noisy

    # Verbose output:
    python run_dual_condition_benchmark.py --verbose

    # Increase sample count:
    python run_dual_condition_benchmark.py --samples 500

Outputs
-------
  data/results/comparison_results/protocol_core_noisy_<TS>.json
  data/results/comparison_results/protocol_core_noiseless_<TS>.json
  data/results/comparison_results/dual_condition_comparison_<TS>.json
  data/results/comparison_results/dual_condition_summary_<TS>.csv
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
import re
# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE     = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent.parent    # both scripts live in …/experiments/benchmarks/;
                                    # .parent.parent reaches hypatiax/ — same as the runner
_RUNNER   = _HERE / "run_comparative_suite_benchmark_v2.py"  # inner benchmark runner





_RESULTS_DIR = _PKG_ROOT / "data/results/comparison_results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)



# ============================================================================
# TEE LOGGER  –  mirrors stdout/stderr to an append-mode log file
# ============================================================================

class _TeeLogger:
    """Writes every line to both the real stream and an open log file.

    Usage::
        sys.stdout = _TeeLogger(sys.stdout, open(log_path, "a"))
        sys.stderr = _TeeLogger(sys.stderr, sys.stdout)   # stderr → tee'd stdout
    """

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
# SUBPROCESS-BASED RUNNER
# ============================================================================
# We call the existing runner as a subprocess so that Julia/PySR signal
# handlers are cleanly initialised in each child process.  This also avoids
# contaminating the parent's module namespace with torch imports.

def _build_runner_cmd(
    condition: str,          # "noisy" | "noiseless"
    args: argparse.Namespace,
    runner_script: Path,
) -> List[str]:
    """Build the subprocess command for one benchmark pass.

    Design rule: parameters that have project-wide recommended values
    (nn_seeds, method_timeout, pysr_timeout, samples) are ALWAYS emitted
    so the child runner never silently falls back to its own built-in
    defaults, regardless of what those defaults happen to be.
    """
    cmd = [sys.executable, str(runner_script)]

    # ── Condition-specific flags ──────────────────────────────────────────
    if condition == "noiseless":
        cmd.append("--noiseless")
        cmd += ["--threshold", str(args.threshold_noiseless)]
    else:
        cmd += ["--threshold", str(args.threshold_noisy)]

    # ── Always-emitted parameters (safe regardless of runner defaults) ────
    cmd += ["--samples",        str(args.samples)]
    cmd += ["--nn-seeds",       str(args.nn_seeds)]
    cmd += ["--method-timeout", str(args.method_timeout)]
    cmd += ["--pysr-timeout",   str(args.pysr_timeout)]

    # ── Optional filters ──────────────────────────────────────────────────
    if args.methods:
        cmd += ["--methods"] + [str(m) for m in args.methods]
    if args.skip_pysr:
        cmd.append("--skip-pysr")
    if args.test:
        cmd += ["--test", args.test]
    if args.domain != "all_domains":
        cmd += ["--domain", args.domain]
    if args.verbose:
        cmd.append("--verbose")
    if args.quiet:
        cmd.append("--quiet")
    if args.no_llm_cache:
        cmd.append("--no-llm-cache")
    if args.equations:
        cmd += ["--equations"] + [str(e) for e in args.equations]
    if args.series:
        cmd += ["--series", args.series]
    if args.benchmark != "feynman":
        cmd += ["--benchmark", args.benchmark]

    # ── Resume logic ──────────────────────────────────────────────────────
    # --resume        : resume both passes (applies to the current condition)
    # --resume-noisy  : resume only the noisy pass
    # --resume-noiseless: resume only the noiseless pass
    should_resume = (
        args.resume
        or (condition == "noisy"     and args.resume_noisy)
        or (condition == "noiseless" and args.resume_noiseless)
    )
    if should_resume:
        cmd.append("--resume")

    # ── Per-condition checkpoint isolation ────────────────────────────────
    # Give each pass its own checkpoint file so noisy and noiseless never
    # collide. The child runner now retains the checkpoint by default —
    # no flag needed. The orchestrator owns cleanup after both passes complete.
    cmd += ["--checkpoint-name", f"protocol_core_{condition}_checkpoint"]

    return cmd


def _find_latest_result(condition: str) -> Optional[Path]:
    """Return the most-recently-created result JSON for a given condition."""
    pattern = f"protocol_core_{condition}_*.json"
    candidates = sorted(
        _RESULTS_DIR.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None



# ============================================================================
# REAL-TIME FAILURE SCANNER  -  parses child-runner stdout line by line
# ============================================================================
# The child runner prints a results table for every equation in this format:
#
#   Method                                   R2         RMSE       Time   Rank
#   -------------------------------------------------------------------------
#   PureLLM Baseline (core)                  0.9975  7.87e-12      0.0    1
#   HybridDiscoverySystem v40 (tools)           N/A       N/A    613.3    - Discovery failed
#
# _FailureScanner is fed one line at a time.  It tracks whether we are inside
# a method-results table and flags any line whose R2 column is N/A.

_TABLE_HDR_RE  = re.compile(r'^\s+Method\s+R', re.IGNORECASE)
_TABLE_SEP_RE  = re.compile(r'^\s*[=\-]{15,}\s*$')
_METHOD_ROW_RE = re.compile(
    r'^\s{2,8}'                          # leading indent (2-8 spaces)
    r'(?P<n>.+?)'                        # method name (non-greedy)
    r'\s{3,}'                            # >= 3 spaces before R2 column
    r'(?P<r2>N/A|[-\d][\d.e+\-]*)'      # R2 value or N/A
    r'(\s|$)',
    re.IGNORECASE,
)


class _FailureScanner:
    """Stateful line scanner; call .check(line) for every stdout line.

    .failure is set to a dict the moment a failing method is detected:
        {"equation": str, "method": str, "reason": str}
    """

    def __init__(self):
        self.in_table: bool = False
        self.failure: Optional[Dict] = None
        self._equation: str = "unknown"

    def check(self, line: str) -> bool:
        """Return True (and set self.failure) when a failure is found."""
        stripped = line.rstrip()

        # Track equation name from section banners
        if not self.in_table and stripped and not stripped.startswith("="):
            candidate = stripped.lstrip()
            if (len(candidate) > 8
                    and not candidate.startswith("Method")
                    and not candidate.startswith("Domain")
                    and not candidate.startswith("[")
                    and not candidate.startswith("TEST")
                    and not candidate.startswith("✗")
                    and not candidate.startswith("🎯")
                    and not candidate.startswith("⛔")
                    and not candidate.startswith("Command")):
                self._equation = candidate[:80]

        # Table entry / exit
        if _TABLE_HDR_RE.match(line):
            self.in_table = True
            return False

        if self.in_table and _TABLE_SEP_RE.match(stripped):
            if stripped.startswith("="):   # "===" closes the table
                self.in_table = False
            return False                   # "---" is just the header rule

        if not self.in_table:
            return False

        # Scan each method result row
        m = _METHOD_ROW_RE.match(line)
        if m:
            r2_str = m.group("r2")
            method = re.sub(r'\s{3,}.*$', "", m.group("n")).strip()

            if r2_str.upper() == "N/A" or "Discovery failed" in line:
                reason = "N/A R²"
                if "Discovery failed" in line:
                    reason += " — Discovery failed"
                self.failure = {
                    "equation": self._equation,
                    "method":   method,
                    "reason":   reason,
                }
                return True

        # Catch "Discovery failed" even when row regex does not match
        if "Discovery failed" in line and "N/A" in line:
            self.failure = {
                "equation": self._equation,
                "method":   stripped[:70],
                "reason":   "Discovery failed (N/A R²)",
            }
            return True

        return False


def _run_condition(
    condition: str,
    args: argparse.Namespace,
    runner_script: Path,
) -> Optional[Path]:
    """Run one benchmark pass and return the path to the saved JSON result.

    Fail-fast behaviour (enabled with --fail-fast)
    -----------------------------------------------
    Level 1 – subprocess crash:
        If the runner exits with a non-zero return code the orchestrator
        prints the failure and calls sys.exit(1) immediately.

    Level 2 – method-level failure in result JSON:
        After the pass completes, the result JSON is scanned for any method
        that has success=False, a non-OK status, an explicit error, or a
        non-finite r2.  If any are found the orchestrator prints a failure
        table and calls sys.exit(1).

    Without --fail-fast both levels fall back to the original warn-and-continue
    behaviour so existing workflows are not broken.
    """
    fail_fast = getattr(args, "fail_fast", False)
    label = "🔊 NOISY (noise=0.05)" if condition == "noisy" else "🔇 NOISELESS (noise=0.0)"
    print(f"\n{'='*80}")
    print(f"  PASS: {label}".center(80))
    if fail_fast:
        print(f"  ⛔  FAIL-FAST enabled — any method failure will abort the run")
    print(f"{'='*80}\n")

    cmd = _build_runner_cmd(condition, args, runner_script)
    print(f"  Command: {' '.join(cmd)}\n")

    t0 = time.time()

    # Stream subprocess output line-by-line for real-time fail-fast.
    # With --fail-fast: Popen + PIPE, scanner sees every line, kills on N/A.
    # Without --fail-fast: original subprocess.run, zero overhead.
    scanner      = _FailureScanner()
    returncode   = 0
    mid_run_kill = False   # True when we terminate() the child ourselves

    if fail_fast:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,          # line-buffered
            env=os.environ.copy(),
        )
        try:
            for raw_line in proc.stdout:
                print(raw_line, end="", flush=True)
                if scanner.check(raw_line):
                    f = scanner.failure
                    elapsed_now = time.time() - t0
                    print(
                        f"\n  ⛔  FAIL-FAST triggered after {elapsed_now/60:.1f} min"
                        f" — method failure detected mid-run:\n"
                        f"     Equation : {f['equation']}\n"
                        f"     Method   : {f['method']}\n"
                        f"     Reason   : {f['reason']}\n"
                        f"  Terminating subprocess …",
                        flush=True,
                    )
                    mid_run_kill = True
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
        finally:
            proc.stdout.close()
            proc.wait()
        returncode = proc.returncode
    else:
        result     = subprocess.run(cmd, env=os.environ.copy())
        returncode = result.returncode

    elapsed = time.time() - t0

    # Mid-run kill: abort immediately
    if mid_run_kill:
        print(f"  ⛔  FAIL-FAST: aborting dual-condition run.")
        sys.exit(1)

    # ── Level 1: subprocess crash ─────────────────────────────────────────
    if returncode != 0:
        print(f"\n  ❌  Runner process exited with code {returncode} "
              f"for {condition} pass  ({elapsed/60:.1f} min elapsed).")
        if fail_fast:
            print(f"  ⛔  FAIL-FAST: aborting dual-condition run.")
            sys.exit(returncode)
        else:
            print(f"      Attempting to locate partial results…")

    # Locate the JSON written by the runner
    json_path = _find_latest_result(condition)

    if json_path is None:
        print(f"\n  ❌ No result JSON found for {condition} pass.")
        if fail_fast:
            print(f"  ⛔  FAIL-FAST: no results produced — aborting.")
            sys.exit(1)
        return None

    print(f"\n  ✅ {condition.capitalize()} pass completed in {elapsed/60:.1f} min")
    print(f"     Results: {json_path}")

    # ── Level 2: method-level failures inside the JSON ────────────────────
    failures = _scan_for_failures(json_path)
    if failures:
        _print_failures(failures, condition)
        if fail_fast:
            print(f"  ⛔  FAIL-FAST: {len(failures)} method failure(s) detected — aborting.")
            sys.exit(1)
        else:
            print(f"  ⚠️  {len(failures)} method failure(s) detected — continuing "
                  f"(use --fail-fast to abort on any failure).")

    return json_path



# ============================================================================
# FAILURE SCANNER  –  inspect a result JSON for any method-level failures
# ============================================================================

# Fields the runner may use to signal a method did not succeed.
# Checked in priority order: if ANY of these indicates failure the method is
# counted as FAILED regardless of the r2 score.
_FAIL_STATUSES = frozenset({
    "failed", "error", "timeout", "exception",
    "crash", "invalid", "nan", "inf", "skipped",
})


def _scan_for_failures(json_path: Path) -> List[Dict]:
    """Parse a result JSON and return one entry per failed method.

    Returns a list of dicts::

        {
          "equation" : str,        # equation / test name
          "method"   : str,        # method name
          "reason"   : str,        # human-readable failure reason
          "r2"       : float|None, # r2 if available
        }

    An empty list means no failures were detected.
    """
    failures: List[Dict] = []
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as exc:
        # Cannot read JSON → treat as a single top-level failure
        return [{"equation": "?", "method": "?", "reason": f"Cannot read JSON: {exc}", "r2": None}]

    for test in data.get("tests", []):
        eq_name = (
            test.get("metadata", {}).get("equation_name")
            or test.get("description", "unknown")
        )[:60]

        for method_name, result in test.get("results", {}).items():
            # ── Check explicit success flag ───────────────────────────────
            if result.get("success") is False:
                reason = result.get("error") or result.get("message") or "success=False"
                failures.append({
                    "equation": eq_name,
                    "method":   method_name,
                    "reason":   str(reason)[:120],
                    "r2":       result.get("r2"),
                })
                continue

            # ── Check status string ───────────────────────────────────────
            status = str(result.get("status", "")).lower()
            if status in _FAIL_STATUSES:
                failures.append({
                    "equation": eq_name,
                    "method":   method_name,
                    "reason":   f"status={status!r}",
                    "r2":       result.get("r2"),
                })
                continue

            # ── Check for explicit error key ──────────────────────────────
            if result.get("error") and result.get("error") not in (None, "", "None"):
                failures.append({
                    "equation": eq_name,
                    "method":   method_name,
                    "reason":   str(result["error"])[:120],
                    "r2":       result.get("r2"),
                })
                continue

            # ── Check for NaN / Inf r2 ────────────────────────────────────
            r2_val = result.get("r2")
            if r2_val is not None:
                try:
                    if not np.isfinite(float(r2_val)):
                        failures.append({
                            "equation": eq_name,
                            "method":   method_name,
                            "reason":   f"r2={r2_val} (non-finite)",
                            "r2":       r2_val,
                        })
                except (TypeError, ValueError):
                    pass

    return failures


def _print_failures(failures: List[Dict], condition: str) -> None:
    """Print a formatted failure table to stdout."""
    print(f"\n{'='*80}")
    print(f"  ❌  FAILURES DETECTED IN {condition.upper()} PASS  ({len(failures)} total)".center(80))
    print(f"{'='*80}")
    print(f"  {'Equation':<35} {'Method':<28} {'R²':>8}  Reason")
    print(f"  {'-'*35} {'-'*28} {'-'*8}  {'-'*30}")
    for f in failures:
        r2_str = f"{f['r2']:.4f}" if f["r2"] is not None else "  N/A  "
        print(f"  {f['equation']:<35} {f['method']:<28} {r2_str:>8}  {f['reason']}")
    print(f"{'='*80}\n")

# ============================================================================
# COMPARISON REPORT
# ============================================================================

def _load_results(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def _extract_per_test(data: Dict) -> Dict[str, Dict]:
    """Return {equation_name: {method_name: {r2, rmse, success, ...}}}."""
    out: Dict[str, Dict] = {}
    for test in data.get("tests", []):
        eq_name = (
            test.get("description", "")
            or test.get("metadata", {}).get("equation_name", "unknown")
        )
        # Use a cleaner key — strip long descriptions
        key = test.get("metadata", {}).get("equation_name", eq_name)
        if not key or key == "unknown":
            key = eq_name[:50]
        out[key] = {
            name: res
            for name, res in test.get("results", {}).items()
        }
    return out


def _build_comparison(
    noisy_data: Dict,
    noiseless_data: Dict,
) -> Dict:
    """Build a unified comparison object."""
    noisy_tests     = _extract_per_test(noisy_data)
    noiseless_tests = _extract_per_test(noiseless_data)

    all_equations = sorted(set(noisy_tests) | set(noiseless_tests))
    all_methods   = sorted(
        set().union(*(m.keys() for m in noisy_tests.values()))
         | set().union(*(m.keys() for m in noiseless_tests.values()))
    )

    comparison = {
        "generated": datetime.now().isoformat(),
        "noise_level_noisy":     0.05,
        "noise_level_noiseless": 0.0,
        "equations": [],
        "method_summary": {},
    }

    for eq in all_equations:
        noisy_eq     = noisy_tests.get(eq, {})
        noiseless_eq = noiseless_tests.get(eq, {})
        eq_entry = {"equation": eq, "methods": {}}

        for method in all_methods:
            noisy_r     = noisy_eq.get(method, {})
            noiseless_r = noiseless_eq.get(method, {})

            n_r2   = noisy_r.get("r2",   None)
            nl_r2  = noiseless_r.get("r2", None)
            delta  = (
                (nl_r2 - n_r2)
                if (n_r2 is not None and nl_r2 is not None
                    and np.isfinite(n_r2) and np.isfinite(nl_r2))
                else None
            )
            eq_entry["methods"][method] = {
                "noisy_r2":       n_r2,
                "noiseless_r2":   nl_r2,
                "delta_r2":       delta,
                "noisy_success":  noisy_r.get("success",     False),
                "noiseless_success": noiseless_r.get("success", False),
                "noisy_rmse":     noisy_r.get("rmse",    None),
                "noiseless_rmse": noiseless_r.get("rmse", None),
            }

        comparison["equations"].append(eq_entry)

    # ── Per-method aggregated summary ─────────────────────────────────────
    for method in all_methods:
        noisy_r2s     = []
        noiseless_r2s = []
        noisy_wins    = 0
        noiseless_wins= 0

        for eq_entry in comparison["equations"]:
            m = eq_entry["methods"].get(method, {})
            n_r2  = m.get("noisy_r2")
            nl_r2 = m.get("noiseless_r2")
            if n_r2 is not None and np.isfinite(float(n_r2)):
                noisy_r2s.append(float(n_r2))
            if nl_r2 is not None and np.isfinite(float(nl_r2)):
                noiseless_r2s.append(float(nl_r2))
            if (n_r2 is not None and nl_r2 is not None
                    and np.isfinite(n_r2) and np.isfinite(nl_r2)):
                if nl_r2 > n_r2:
                    noiseless_wins += 1
                elif n_r2 >= nl_r2:
                    noisy_wins += 1

        comparison["method_summary"][method] = {
            "noisy_median_r2":     float(np.median(noisy_r2s))     if noisy_r2s     else None,
            "noiseless_median_r2": float(np.median(noiseless_r2s)) if noiseless_r2s else None,
            "noisy_mean_r2":       float(np.mean(noisy_r2s))       if noisy_r2s     else None,
            "noiseless_mean_r2":   float(np.mean(noiseless_r2s))   if noiseless_r2s else None,
            "noiseless_wins":      noiseless_wins,
            "noisy_wins":          noisy_wins,
            "n_noisy":             len(noisy_r2s),
            "n_noiseless":         len(noiseless_r2s),
        }

    return comparison


def _print_comparison_table(comparison: Dict) -> None:
    """Pretty-print a dual-condition comparison table to stdout."""
    methods = list(comparison["method_summary"].keys())
    if not methods:
        print("  (no methods to compare)")
        return

    # ── Per-method summary table ─────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  DUAL-CONDITION SUMMARY  (noise=0.05  vs  noise=0.0)".center(90))
    print(f"{'='*90}")
    hdr = (f"  {'Method':<42} {'Noisy Med R²':>12} {'NoisFree Med R²':>15}"
           f" {'ΔR² (NL−N)':>12} {'NL wins':>8}")
    print(hdr)
    print("  " + "-" * 88)

    for m in sorted(methods):
        s     = comparison["method_summary"][m]
        n_med = s.get("noisy_median_r2")
        nl_med= s.get("noiseless_median_r2")
        delta = (nl_med - n_med) if (n_med is not None and nl_med is not None) else None
        n_med_s  = f"{n_med:.4f}"  if n_med  is not None else "N/A"
        nl_med_s = f"{nl_med:.4f}" if nl_med is not None else "N/A"
        delta_s  = f"{delta:+.4f}" if delta  is not None else "N/A"
        nl_wins  = s.get("noiseless_wins", 0)
        print(f"  {m:<42} {n_med_s:>12} {nl_med_s:>15} {delta_s:>12} {nl_wins:>8}")

    print(f"{'='*90}")

    # ── Per-equation table ────────────────────────────────────────────────
    print(f"\n  Per-equation detail (best method per condition):")
    print(f"  {'Equation':<30} {'Noisy best R²':>14} {'NL best R²':>12} {'Δ':>8}")
    print("  " + "-" * 68)
    for eq_entry in comparison["equations"]:
        eq  = eq_entry["equation"][:28]
        n_best  = max(
            (v.get("noisy_r2",   -np.inf) or -np.inf
             for v in eq_entry["methods"].values()),
            default=-np.inf,
        )
        nl_best = max(
            (v.get("noiseless_r2", -np.inf) or -np.inf
             for v in eq_entry["methods"].values()),
            default=-np.inf,
        )
        nb_s  = f"{n_best:.4f}"  if np.isfinite(n_best)  else "N/A"
        nlb_s = f"{nl_best:.4f}" if np.isfinite(nl_best) else "N/A"
        if np.isfinite(n_best) and np.isfinite(nl_best):
            d_s = f"{nl_best - n_best:+.4f}"
        else:
            d_s = "N/A"
        print(f"  {eq:<30} {nb_s:>14} {nlb_s:>12} {d_s:>8}")

    print()


def _save_comparison(comparison: Dict, ts: str) -> Path:
    path = _RESULTS_DIR / f"dual_condition_comparison_{ts}.json"
    with open(path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"  💾 Comparison JSON → {path}")
    return path


def _save_csv(comparison: Dict, ts: str) -> Path:
    """Save a flat CSV with one row per (equation, method) pair."""
    path = _RESULTS_DIR / f"dual_condition_summary_{ts}.csv"
    fieldnames = [
        "equation", "method",
        "noisy_r2", "noiseless_r2", "delta_r2",
        "noisy_success", "noiseless_success",
        "noisy_rmse", "noiseless_rmse",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for eq_entry in comparison["equations"]:
            eq = eq_entry["equation"]
            for method, data in eq_entry["methods"].items():
                writer.writerow({
                    "equation":           eq,
                    "method":             method,
                    "noisy_r2":           data.get("noisy_r2"),
                    "noiseless_r2":       data.get("noiseless_r2"),
                    "delta_r2":           data.get("delta_r2"),
                    "noisy_success":      data.get("noisy_success"),
                    "noiseless_success":  data.get("noiseless_success"),
                    "noisy_rmse":         data.get("noisy_rmse"),
                    "noiseless_rmse":     data.get("noiseless_rmse"),
                })
    print(f"  📊 Summary CSV    → {path}")
    return path


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HypatiaX benchmark for BOTH noisy (σ=0.05) and noiseless conditions.\n\nUse --fail-fast to abort immediately if any method fails in any equation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Condition flags ───────────────────────────────────────────────────
    parser.add_argument(
        "--noisy-only", action="store_true", dest="noisy_only",
        help="Run only the noisy pass (noise=0.05).",
    )
    parser.add_argument(
        "--noiseless-only", action="store_true", dest="noiseless_only",
        help="Run only the noiseless pass (noise=0.0).",
    )
    parser.add_argument(
        "--compare-existing", action="store_true", dest="compare_existing",
        help=(
            "Skip both benchmark passes; load the most-recent noisy and noiseless "
            "result JSONs from the results directory and produce the comparison report."
        ),
    )

    # ── Thresholds ────────────────────────────────────────────────────────
    parser.add_argument(
        "--threshold-noisy", type=float, default=0.995, dest="threshold_noisy",
        help="R² recovery threshold for the noisy pass (default: 0.995).",
    )
    parser.add_argument(
        "--threshold-noiseless", type=float, default=0.9999, dest="threshold_noiseless",
        help="R² recovery threshold for the noiseless pass (default: 0.9999).",
    )

    # ── Standard runner flags (forwarded verbatim) ──────────────────────────
    # Defaults match the recommended production invocation:
    #   --nn-seeds 3  --samples 200  --method-timeout 900  --pysr-timeout 1100
    parser.add_argument("--methods",   type=int, nargs="+", metavar="N", default=None)
    parser.add_argument("--skip-pysr", action="store_true", dest="skip_pysr")
    parser.add_argument("--test",      type=str, default=None)
    parser.add_argument("--domain",    type=str, default="all_domains")
    parser.add_argument("--samples",   type=int, default=200,
                        help="Data points per equation (default: 200).")
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--quiet",     action="store_true")
    parser.add_argument("--no-llm-cache", action="store_true", dest="no_llm_cache")
    parser.add_argument("--nn-seeds",  type=int, default=3, dest="nn_seeds",
                        help="NN ensemble seeds per equation (default: 3).")
    parser.add_argument("--equations", type=int, nargs="+", metavar="N", default=None)
    parser.add_argument("--pysr-timeout",  type=int, default=1100, dest="pysr_timeout",
                        help="PySR wall-clock timeout in seconds (default: 1100).")
    parser.add_argument("--method-timeout", type=int, default=900, dest="method_timeout",
                        help="Per-method timeout in seconds (default: 900).")
    parser.add_argument("--series",    choices=["I","II","III","crossover"], default=None)
    parser.add_argument("--benchmark", choices=["feynman","srbench","both"], default="feynman")

    # ── Fail-fast ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--fail-fast", action="store_true", dest="fail_fast",
        help=(
            "Abort the entire dual-condition run if ANY method fails "
            "in ANY equation in either pass. "
            "Failure is detected at two levels: (1) the runner subprocess "
            "exits non-zero, and (2) the result JSON contains any method "
            "with success=False, a non-OK status, an explicit error field, "
            "or a non-finite R² score. "
            "Without this flag failures are logged as warnings and the run "
            "continues (original behaviour)."
        ),
    )

    # ── Resume flags ────────────────────────────────────────────────────────
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume BOTH passes from their last checkpoint.",
    )
    parser.add_argument(
        "--resume-noisy", action="store_true", dest="resume_noisy",
        help="Resume only the noisy pass from its last checkpoint.",
    )
    parser.add_argument(
        "--resume-noiseless", action="store_true", dest="resume_noiseless",
        help="Resume only the noiseless pass from its last checkpoint.",
    )

    # ── Log file (tee-style) ─────────────────────────────────────────────────
    parser.add_argument(
        "--log", type=str, default=None, dest="log", metavar="FILE",
        help=(
            "Append all orchestrator-level output to FILE (mirrors 2>&1 | tee -a). "
            "Pass-level subprocess output streams directly to the terminal as normal."
        ),
    )
    parser.add_argument(
        "--runner", type=str, default=None, dest="runner",
        help="Path to this script (auto-detected; only needed if running from a different directory).",
    )

    args = parser.parse_args()

    # ── Optional tee logging ──────────────────────────────────────────────
    _log_fh = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _log_fh    = open(log_path, "a", buffering=1)
        sys.stdout = _TeeLogger(sys.stdout, _log_fh)
        sys.stderr = _TeeLogger(sys.stderr, sys.stdout)   # stderr mirrors stdout
        print(f"  📝 Logging to: {log_path}  (append mode)")

    # ── Locate runner script ──────────────────────────────────────────────
    runner_path = Path(args.runner) if args.runner else _RUNNER
    if not runner_path.exists():
        print(
            f"❌  Cannot find runner script: {runner_path}\n"
            f"    Pass --runner /path/to/run_dual_condition_benchmark.py"
        )
        sys.exit(1)

    print(f"\n{'='*80}")
    print("  DUAL-CONDITION BENCHMARK RUNNER".center(80))
    print(f"{'='*80}")
    resume_note = []
    if args.resume:            resume_note.append("both")
    if args.resume_noisy:      resume_note.append("noisy")
    if args.resume_noiseless:  resume_note.append("noiseless")

    print(f"  Runner script   : {runner_path}")
    print(f"  Results dir     : {_RESULTS_DIR}")
    print(f"  Noise (noisy)   : 0.05  (R² threshold : {args.threshold_noisy})")
    print(f"  Noise (clean)   : 0.0   (R² threshold : {args.threshold_noiseless})")
    print(f"  Samples         : {args.samples}")
    print(f"  NN seeds        : {args.nn_seeds}")
    print(f"  Method timeout  : {args.method_timeout}s")
    print(f"  PySR timeout    : {args.pysr_timeout}s")
    print(f"  Resume passes   : {', '.join(resume_note) if resume_note else 'none (fresh start)'}")
    print(f"  Fail-fast       : {'ON  ⛔ — any method failure aborts the run' if args.fail_fast else 'OFF (warn and continue)'}")
    if args.log:
        print(f"  Log file        : {args.log}  (append mode)")
    print(f"{'='*80}\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    noisy_path     : Optional[Path] = None
    noiseless_path : Optional[Path] = None

    # ── --compare-existing shortcut ───────────────────────────────────────
    if args.compare_existing:
        noisy_path     = _find_latest_result("noisy")
        noiseless_path = _find_latest_result("noiseless")
        if not noisy_path:
            print("❌  No noisy result JSON found.")
            sys.exit(1)
        if not noiseless_path:
            print("❌  No noiseless result JSON found.")
            sys.exit(1)
        print(f"  Loading noisy     : {noisy_path}")
        print(f"  Loading noiseless : {noiseless_path}\n")

    else:
        # ── NOISY PASS ────────────────────────────────────────────────────
        if not args.noiseless_only:
            noisy_path = _run_condition("noisy", args, runner_path)
            if noisy_path is None:
                print("⚠️  Noisy pass produced no results. Continuing to noiseless…")

        # ── NOISELESS PASS ────────────────────────────────────────────────
        if not args.noisy_only:
            noiseless_path = _run_condition("noiseless", args, runner_path)
            if noiseless_path is None:
                print("⚠️  Noiseless pass produced no results.")

    # ── COMPARISON REPORT ─────────────────────────────────────────────────
    if noisy_path and noiseless_path:
        print(f"\n{'='*80}")
        print("  GENERATING COMPARISON REPORT".center(80))
        print(f"{'='*80}")

        noisy_data     = _load_results(noisy_path)
        noiseless_data = _load_results(noiseless_path)

        comparison = _build_comparison(noisy_data, noiseless_data)
        _print_comparison_table(comparison)
        _save_comparison(comparison, ts)
        _save_csv(comparison, ts)

        # ── Quick noise-sensitivity insight ──────────────────────────────
        total   = len(comparison["equations"])
        improved = sum(
            1 for eq in comparison["equations"]
            for v in eq["methods"].values()
            if v.get("delta_r2") is not None and v["delta_r2"] > 0.01
        )
        degraded = sum(
            1 for eq in comparison["equations"]
            for v in eq["methods"].values()
            if v.get("delta_r2") is not None and v["delta_r2"] < -0.01
        )
        print(f"\n  Noise-sensitivity summary over {total} equations:")
        print(f"  Methods that improved ≥0.01 R² in noiseless : {improved}")
        print(f"  Methods that degraded ≥0.01 R² in noiseless : {degraded}")
        print(f"  (degradation = noiseless R² < noisy R², which can happen when")
        print(f"   the noisy result is cached or the noiseless model overfits)")

    elif noisy_path or noiseless_path:
        available = noisy_path or noiseless_path
        print(
            f"\n⚠️  Only one condition result available ({available.name}).\n"
            "    Run both passes to get the comparison report."
        )
    else:
        print("\n❌  No results available for comparison.")

    # ── Orchestrator checkpoint cleanup ───────────────────────────────────
    # Child runner retains checkpoints by default. The orchestrator reads
    # had_timeouts from each condition's checkpoint; retains if True
    # (results may be unreliable due to internet drop / Julia hang).
    if not args.compare_existing:
        _ckpt_dir = _RESULTS_DIR
        for _cond in ("noisy", "noiseless"):
            _ckpt_path = _ckpt_dir / f"protocol_core_{_cond}_checkpoint.json"
            if not _ckpt_path.exists():
                continue
            try:
                with open(_ckpt_path) as _f:
                    _ckpt = json.load(_f)
                if _ckpt.get("had_timeouts"):
                    print(f"\n⚠️  {_cond} checkpoint retained (had_timeouts=True —"
                          f" internet drop or Julia hang?).")
                    print(f"   Use --resume-{_cond} on next run to continue.")
                else:
                    _ckpt_path.unlink()
                    print(f"🗑️  {_cond} checkpoint removed (clean finish).")
            except Exception as _exc:
                print(f"⚠️  Could not process {_cond} checkpoint: {_exc}")

    print(f"\n✅  Dual-condition run complete.\n")


if __name__ == "__main__":
    main()
