#!/usr/bin/env python3
"""
run_dual_sweep_benchmarks.py
============================

Orchestrates **both** the noise-level sweep and the sample-complexity sweep
for the HypatiaX protocol suite in a single command, automatically merging
any already-completed results.

What it does
------------
1. Runs ``run_noise_sweep_benchmark.py`` for the requested σ levels.
   Already-completed σ values on disk are merged via ``--existing-results``
   instead of being re-run.
2. Runs ``run_sample_complexity_benchmark.py`` for the requested n values.
   Already-completed n values are merged the same way.
3. Logs each sweep to its own file, writes a timestamped JUnit XML + HTML
   report per sweep, and isolates checkpoints so the two sweeps never collide.

Usage
-----
    # Dry-run — print resolved commands without executing (default):
    python run_dual_sweep_benchmarks.py \\
        --noise-levels 0.0 0.005 0.01 0.05 0.10 \\
        --sample-sizes 50 100 200 500 750 1000 \\
        --methods 3 4

    # Full live run with two separate log files:
    python run_dual_sweep_benchmarks.py \\
        --noise-levels 0.0 0.005 0.01 0.05 0.10 \\
        --sample-sizes 50 100 200 500 750 1000 \\
        --methods 3 4 \\
        --fail-fast \\
        --log sample_complexity.log noise_sweep.log

    # Smoke-test one equation on both sweeps first:
    python run_dual_sweep_benchmarks.py \\
        --noise-levels 0.0 0.005 0.01 0.05 0.10 \\
        --sample-sizes 50 100 200 500 750 1000 \\
        --methods 3 4 --smoke

    # Noise sweep only:
    python run_dual_sweep_benchmarks.py \\
        --noise-levels 0.0 0.005 0.01 0.05 0.10 \\
        --methods 3 4 --noise-only

    # Sample-complexity sweep only:
    python run_dual_sweep_benchmarks.py \\
        --sample-sizes 50 100 200 500 750 1000 \\
        --methods 3 4 --sc-only

    # Skip auto-detected existing results and re-run everything:
    python run_dual_sweep_benchmarks.py \\
        --noise-levels 0.0 0.005 0.01 0.05 0.10 \\
        --sample-sizes 50 100 200 500 750 1000 \\
        --methods 3 4 --no-existing

    # Resume both sweeps after a crash (skips any that finished OK):
    python run_dual_sweep_benchmarks.py \\
        --noise-levels 0.0 0.005 0.01 0.05 0.10 \\
        --sample-sizes 50 100 200 500 750 1000 \\
        --methods 3 4 --resume

    # Resume only the noise sweep (SC not touched):
    python run_dual_sweep_benchmarks.py \\
        --noise-levels 0.0 0.005 0.01 0.05 0.10 \\
        --methods 3 4 --resume-noise-only

    # Resume only the sample-complexity sweep (noise not touched):
    python run_dual_sweep_benchmarks.py \\
        --sample-sizes 50 100 200 500 750 1000 \\
        --methods 3 4 --resume-samples-only

    # Resume from a specific registry file:
    python run_dual_sweep_benchmarks.py \\
        --noise-levels 0.0 0.005 0.01 0.05 0.10 \\
        --sample-sizes 50 100 200 500 750 1000 \\
        --methods 3 4 --resume \\
        --registry data/results/comparison_results/dual_sweep_20260314_221958_registry.json

The --log flag
--------------
Accepts one or two file paths.

    --log noise_sweep.log                      # single orchestrator log
    --log sample_complexity.log noise_sweep.log  # SC log first, noise log second

When two paths are given the **first** is used for the sample-complexity sweep
and the **second** for the noise sweep — matching the order they appear in the
example command above.  When only one path is given it covers both sweeps.
The orchestrator banner is always written to both files.

Existing-result auto-detection
-------------------------------
Before launching each sweep the script scans
``data/results/comparison_results/`` for the most-recent matching JSON and
forwards it via ``--existing-results``.  Override with explicit flags::

    --noiseless-json  path/to/protocol_core_noiseless_TIMESTAMP.json
    --sig0005-json    path/to/protocol_core_sig0005_TIMESTAMP.json
    --n200-json       path/to/protocol_core_noisy_20260313_094752.json

Pass ``--no-existing`` to ignore all on-disk results and start entirely fresh.

Outputs (written by the child scripts)
----------------------------------------
  data/results/comparison_results/noise_sweep_<TS>.json
  data/results/comparison_results/noise_sweep_<TS>.csv
  data/results/comparison_results/sample_complexity_<TS>.json
  data/results/comparison_results/sample_complexity_<TS>.csv
  <save-dir>/noise_sweep_<TS>.xml          ← JUnit (CI-compatible)
  <save-dir>/noise_sweep_<TS>.html         ← self-contained HTML report
  <save-dir>/sample_complexity_<TS>.xml
  <save-dir>/sample_complexity_<TS>.html
  <log-dir>/noise_sweep.log
  <log-dir>/sample_complexity.log
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Paths — script lives at hypatiax/experiments/benchmarks/
# ---------------------------------------------------------------------------
_HERE         = Path(__file__).resolve().parent
_PKG_ROOT     = _HERE.parent.parent          # …/benchmarks → …/experiments → hypatiax/
_NOISE_SCRIPT = _HERE / "run_noise_sweep_benchmark.py"
_SC_SCRIPT    = _HERE / "run_sample_complexity_benchmark.py"
_RESULTS_DIR  = _PKG_ROOT / "data" / "results" / "comparison_results"
_LOG_DIR      = _PKG_ROOT / "logs"


# ============================================================================
# TEE LOGGER  — identical to run_dual_condition_benchmark.py
# ============================================================================

class _TeeLogger:
    """Writes every line to both the real stream and an open log file."""

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
# ORCHESTRATOR CHECKPOINT REGISTRY
# ============================================================================
# The child scripts each manage per-sigma / per-n checkpoints internally.
# The orchestrator writes its own lightweight registry so a killed or failed
# run can be inspected and --no-existing overridden on resume.

def _write_registry(save_dir: Path, ts: str, entry: dict) -> None:
    """Append *entry* to the orchestrator checkpoint registry JSON."""
    registry_path = save_dir / f"dual_sweep_{ts}_registry.json"
    try:
        registry: list = []
        if registry_path.exists():
            with open(registry_path) as f:
                registry = json.load(f)
        registry.append(entry)
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2, default=str)
    except Exception as exc:
        print(f"  ⚠️   Could not update registry: {exc}")


def _finalize_registry(save_dir: Path, ts: str, noise_ok: bool, sc_ok: bool) -> None:
    """Write final status to the registry and print its path."""
    registry_path = save_dir / f"dual_sweep_{ts}_registry.json"
    _write_registry(save_dir, ts, {
        "event":     "run_complete",
        "timestamp": datetime.now().isoformat(),
        "noise_sweep_ok":      noise_ok,
        "sc_sweep_ok":         sc_ok,
        "overall_ok":          noise_ok and sc_ok,
    })
    print(f"  📋 Checkpoint registry → {registry_path}")


# ============================================================================
# RESUME — read the last registry to decide what still needs to run
# ============================================================================

def _find_latest_registry(save_dir: Path) -> Optional[Path]:
    """Return the most-recently-modified registry JSON in save_dir."""
    candidates = sorted(
        save_dir.glob("dual_sweep_*_registry.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_resume_state(
    save_dir:          Path,
    resume_noise:      bool,
    resume_sc:         bool,
) -> tuple:
    """
    Read the most-recent registry and return:
        (noise_done, sc_done, registry_path)

    noise_done / sc_done are True when that sweep completed successfully in
    the previous run — meaning it can be skipped entirely on resume.

    If no registry is found, both are False (fresh run).
    """
    if not (resume_noise or resume_sc):
        return False, False, None

    registry_path = _find_latest_registry(save_dir)
    if registry_path is None:
        print("  ⚠️   --resume requested but no registry found — starting fresh.")
        return False, False, None

    print(f"  📋 Resuming from registry: {registry_path.name}")
    try:
        with open(registry_path) as f:
            events = json.load(f)
    except Exception as exc:
        print(f"  ⚠️   Could not read registry ({exc}) — starting fresh.")
        return False, False, registry_path

    noise_done = any(
        e.get("event") == "noise_sweep_complete" and e.get("ok") is True
        for e in events
    )
    sc_done = any(
        e.get("event") == "sc_sweep_complete" and e.get("ok") is True
        for e in events
    )

    if resume_noise:
        if noise_done:
            print("  ✅  Noise sweep already completed OK in previous run — will skip.")
        else:
            print("  🔄  Noise sweep did not complete OK — will re-run.")

    if resume_sc:
        if sc_done:
            print("  ✅  SC sweep already completed OK in previous run — will skip.")
        else:
            print("  🔄  SC sweep did not complete OK — will re-run.")

    return noise_done, sc_done, registry_path


# ============================================================================
# EXISTING-RESULT AUTO-DETECTION
# ============================================================================

def _find_latest(pattern: str) -> Optional[Path]:
    """Return the most-recently-modified JSON matching *pattern* in _RESULTS_DIR."""
    candidates = sorted(
        _RESULTS_DIR.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _resolve_existing_noise(
    noise_levels:   List[float],
    noiseless_json: Optional[str],
    sig0005_json:   Optional[str],
    no_existing:    bool,
) -> List[str]:
    """
    Return ``sigma:path`` pairs for any σ values that already have results on
    disk.  Only examines σ=0 and σ=0.005 since those are the pre-existing
    baseline runs; the remaining levels are always freshly computed.
    """
    if no_existing:
        return []

    pairs: List[str] = []

    if 0.0 in noise_levels:
        path = Path(noiseless_json) if noiseless_json else _find_latest("protocol_core_noiseless_*.json")
        if path and path.exists():
            print(f"  ✅  Merging σ=0%   : {path.name}")
            pairs.append(f"0.0:{path}")
        else:
            print("  ⚠️   σ=0%  result not found — will run from scratch")

    if 0.005 in noise_levels:
        path = Path(sig0005_json) if sig0005_json else _find_latest("protocol_core_sig0005_*.json")
        if path and path.exists():
            print(f"  ✅  Merging σ=0.5% : {path.name}")
            pairs.append(f"0.005:{path}")
        else:
            print("  ⚠️   σ=0.5% result not found — will run from scratch")

    return pairs


def _resolve_existing_sc(
    sample_sizes: List[int],
    n200_json:    Optional[str],
    no_existing:  bool,
) -> List[str]:
    """
    Return ``n:path`` pairs for any sample sizes that already have results on
    disk.  Only examines n=200 (the standard baseline run).
    """
    if no_existing:
        return []

    pairs: List[str] = []

    if 200 in sample_sizes:
        path = Path(n200_json) if n200_json else _find_latest("protocol_core_noisy_*.json")
        if path and path.exists():
            print(f"  ✅  Merging n=200  : {path.name}")
            pairs.append(f"200:{path}")
        else:
            print("  ⚠️   n=200 result not found — will run from scratch")

    return pairs


# ============================================================================
# COMMAND BUILDERS
# ============================================================================

def _build_noise_cmd(
    args:     argparse.Namespace,
    existing: List[str],
    ts:       str,
    log_file: Path,
) -> List[str]:
    # Forwards every flag the child script accepts.
    # Per-sigma threshold/noiseless injection is handled INSIDE
    # run_noise_sweep_benchmark.py — it loops over noise_levels itself.
    cmd = [
        sys.executable, str(args.noise_script),
        "--noise-levels",       *[str(s) for s in args.noise_levels],
        "--methods",            *[str(m) for m in args.methods],
        "--nn-seeds",           str(args.nn_seeds),
        "--samples",            str(args.samples),
        "--method-timeout",     str(args.method_timeout),
        "--pysr-timeout",       str(args.pysr_timeout),
        "--threshold-noisy",    str(args.threshold_noisy),
        "--threshold-noiseless",str(args.threshold_noiseless),
        "--log",                str(log_file),
    ]
    if existing:
        cmd += ["--existing-results"] + existing
    if args.fail_fast:
        cmd.append("--fail-fast")
    # --smoke always wins; otherwise forward --test / --equations / --domain
    if args.smoke:
        cmd += ["--test", "arrhenius"]
    elif getattr(args, "test", None):
        cmd += ["--test", args.test]
    if not args.smoke and getattr(args, "equations", None):
        cmd += ["--equations"] + [str(e) for e in args.equations]
    if not args.smoke and getattr(args, "domain", None):
        cmd += ["--domain", args.domain]
    if getattr(args, "skip_pysr", False):
        cmd.append("--skip-pysr")
    if getattr(args, "no_llm_cache", False):
        cmd.append("--no-llm-cache")
    if getattr(args, "threshold_per_sigma", []):
        cmd += ["--threshold-per-sigma"] + args.threshold_per_sigma
    if getattr(args, "runner", None):
        cmd += ["--runner", args.runner]
    if args.verbose:
        cmd.append("--verbose")
    return cmd


def _build_sc_cmd(
    args:     argparse.Namespace,
    existing: List[str],
    ts:       str,
    log_file: Path,
) -> List[str]:
    # Forwards every flag the child script accepts.
    # Per-n --samples injection is handled INSIDE
    # run_sample_complexity_benchmark.py — it loops over sample_sizes itself.
    cmd = [
        sys.executable, str(args.sc_script),
        "--sample-sizes",       *[str(n) for n in args.sample_sizes],
        "--methods",            *[str(m) for m in args.methods],
        "--nn-seeds",           str(args.nn_seeds),
        "--method-timeout",     str(args.method_timeout),
        "--pysr-timeout",       str(args.pysr_timeout),
        "--threshold-noisy",    str(args.threshold_noisy),
        "--threshold-noiseless",str(args.threshold_noiseless),
        "--log",                str(log_file),
    ]
    if getattr(args, "sc_noiseless", False):
        cmd.append("--noiseless")
    else:
        # Pass the fixed noise level for the SC sweep (default σ=0.05)
        cmd += ["--fixed-noise", str(getattr(args, "sc_fixed_noise", 0.05))]
    if existing:
        cmd += ["--existing-results"] + existing
    if args.fail_fast:
        cmd.append("--fail-fast")
    # --smoke always wins; otherwise forward --test / --equations / --domain
    if args.smoke:
        cmd += ["--test", "arrhenius"]
    elif getattr(args, "test", None):
        cmd += ["--test", args.test]
    if not args.smoke and getattr(args, "equations", None):
        cmd += ["--equations"] + [str(e) for e in args.equations]
    if not args.smoke and getattr(args, "domain", None):
        cmd += ["--domain", args.domain]
    if getattr(args, "skip_pysr", False):
        cmd.append("--skip-pysr")
    if getattr(args, "no_llm_cache", False):
        cmd.append("--no-llm-cache")
    if getattr(args, "threshold_per_n", []):
        cmd += ["--threshold-per-n"] + args.threshold_per_n
    if getattr(args, "runner", None):
        cmd += ["--runner", args.runner]
    if args.verbose:
        cmd.append("--verbose")
    return cmd


# ============================================================================
# SWEEP RUNNER
# ============================================================================

def _run_sweep(
    label:     str,
    cmd:       List[str],
    dry_run:   bool,
    fail_fast: bool,
) -> bool:
    """Print the command, optionally execute it, return True on success."""
    print(f"\n{'='*80}")
    print(f"  SWEEP: {label}".center(80))
    if fail_fast:
        print("  ⛔  FAIL-FAST enabled — any subprocess failure will abort".center(80))
    print(f"{'='*80}\n")
    print(f"  Command: {' '.join(cmd)}\n")

    if dry_run:
        print("  [dry-run — omit --dry-run to execute]\n")
        return True

    t0 = time.time()
    result = subprocess.run(cmd, env=os.environ.copy())
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)

    if result.returncode != 0:
        print(f"\n  ❌  {label} exited with code {result.returncode} "
              f"({mins}m {secs}s elapsed).")
        if fail_fast:
            print("  ⛔  FAIL-FAST: aborting dual-sweep run.")
            sys.exit(result.returncode)
        print("  ⚠️   Continuing despite failure (use --fail-fast to abort).")
        return False

    print(f"\n  ✅  {label} completed in {mins}m {secs}s")
    return True


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the HypatiaX noise sweep AND sample-complexity sweep in one command.\n"
            "Automatically merges any already-completed σ / n results from disk.\n\n"
            "Runs immediately — use --dry-run to preview commands without executing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Execution mode ────────────────────────────────────────────────────
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Print resolved commands without executing anything.",
    )
    parser.add_argument(
        "--noise-only", action="store_true", dest="noise_only",
        help="Run only the noise-level sweep.",
    )
    parser.add_argument(
        "--sc-only", action="store_true", dest="sc_only",
        help="Run only the sample-complexity sweep.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke-test on one equation (arrhenius) before the full run.",
    )

    # ── Equation / domain filter ──────────────────────────────────────────
    # Forwarded unchanged to both child sweep scripts so you can pin a single
    # equation or domain without editing the child scripts directly.
    # Note: --smoke takes precedence over --test when both are supplied.
    parser.add_argument(
        "--test", type=str, default=None, metavar="NAME",
        help=(
            "Run a single equation by name across all σ / n levels "
            "(forwarded to both child sweeps as --test NAME). "
            "Example: --test I.12.1"
        ),
    )
    parser.add_argument(
        "--equations", type=int, nargs="+", metavar="N", default=None,
        help=(
            "Run specific equations by 1-based index "
            "(forwarded to both child sweeps as --equations N …). "
            "Example: --equations 3 7 12"
        ),
    )
    parser.add_argument(
        "--domain", type=str, default=None, metavar="DOMAIN",
        help=(
            "Restrict to a single domain "
            "(forwarded to both child sweeps as --domain DOMAIN). "
            "Example: --domain mechanics  or  --domain feynman_mechanics"
        ),
    )

    # ── Sweep parameters ──────────────────────────────────────────────────
    parser.add_argument(
        "--noise-levels", type=float, nargs="+", metavar="σ",
        default=[0.0, 0.005, 0.01, 0.05, 0.10],
        dest="noise_levels",
        help="Noise σ values to sweep (default: 0.0 0.005 0.01 0.05 0.10).",
    )
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+", metavar="N",
        default=[50, 100, 200, 500, 750, 1000],
        dest="sample_sizes",
        help="Sample counts to sweep (default: 50 100 200 500).",
    )
    parser.add_argument(
        "--methods", type=int, nargs="+", metavar="N", default=[3, 4],
        help="Method indices to run (default: 3 4).",
    )
    parser.add_argument(
        "--nn-seeds", type=int, default=5, dest="nn_seeds",
        help="NN ensemble seeds per equation (default: 5).",
    )
    parser.add_argument(
        "--samples", type=int, default=200,
        help="Data points per equation for the noise sweep (default: 200).",
    )
    parser.add_argument(
        "--method-timeout", type=int, default=900, dest="method_timeout",
        help="Per-method wall-clock timeout in seconds (default: 900).",
    )
    parser.add_argument(
        "--pysr-timeout", type=int, default=1100, dest="pysr_timeout",
        help="PySR wall-clock timeout in seconds (default: 1100).",
    )
    parser.add_argument(
        "--threshold-noisy", type=float, default=0.950, dest="threshold_noisy",
        help="R² threshold for σ=5%% noise sweep (default: 0.950). SC sweep uses 0.995 flat.",
    )
    parser.add_argument(
        "--threshold-noiseless", type=float, default=0.999999, dest="threshold_noiseless",
        help="R² threshold for σ=0%% noiseless pass (default: 0.999999).",
    )
    parser.add_argument(
        "--threshold-per-sigma", nargs="+", default=[], dest="threshold_per_sigma",
        metavar="SIGMA:VALUE",
        help=(
            "Per-sigma threshold overrides forwarded to the noise sweep. "
            "Format: sigma:value  e.g.  0.005:0.999  0.01:0.998. "
            "Defaults: 0.0->0.9999  0.005->0.9990  0.01->0.9990  0.05->0.995  0.10->0.990"
        ),
    )
    parser.add_argument(
        "--threshold-per-n", nargs="+", default=[], dest="threshold_per_n",
        metavar="N:VALUE",
        help=(
            "Per-n threshold overrides forwarded to the SC sweep. "
            "Format: n:value  e.g.  50:0.988  100:0.992  500:0.998. "
            "Noisy defaults:     50->0.990  100->0.993  200->0.995  500->0.997. "
            "Noiseless defaults: 50->0.9990 100->0.9995 200->0.9999 500->0.9999"
        ),
    )
    parser.add_argument(
        "--sc-noiseless", action="store_true", dest="sc_noiseless",
        help="Run the sample-complexity sweep in noiseless mode (noise_level=0.0).",
    )
    parser.add_argument(
        "--sc-fixed-noise", type=float, default=0.05, dest="sc_fixed_noise",
        metavar="SIGMA",
        help=(
            "Fixed noise level (σ) for the sample-complexity sweep. "
            "Injected as HYPATIAX_NOISE_LEVEL for every n run. "
            "Ignored when --sc-noiseless is set. Default: 0.05 (σ=5%%)."
        ),
    )
    parser.add_argument(
        "--skip-pysr", action="store_true", dest="skip_pysr",
        help="Skip PySR-backed methods in both sweeps.",
    )
    parser.add_argument(
        "--no-llm-cache", action="store_true", dest="no_llm_cache",
        help="Disable LLM cache in both sweeps.",
    )
    parser.add_argument(
        "--runner", type=str, default=None,
        help="Path to run_comparative_suite_benchmark_v2.py (auto-detected).",
    )

    # ── Log files ─────────────────────────────────────────────────────────
    # Accepts 1 or 2 paths:
    #   1 path  → used for both sweeps
    #   2 paths → first = SC sweep log, second = noise sweep log
    #             (mirrors the order in the example command)
    parser.add_argument(
        "--log", type=str, nargs="+", metavar="FILE", default=None,
        help=(
            "Log file(s) for the child sweeps. "
            "Pass one file to use for both sweeps, or two files — "
            "the first is used for the sample-complexity sweep, "
            "the second for the noise sweep."
        ),
    )
    parser.add_argument(
        "--log-dir", type=str, default=str(_LOG_DIR), dest="log_dir",
        metavar="DIR",
        help=(
            "Directory for per-sweep log files when --log is not given. "
            f"Default: {_LOG_DIR}"
        ),
    )
    parser.add_argument(
        "--save-dir", type=str, default=str(_RESULTS_DIR), dest="save_dir",
        metavar="DIR",
        help=(
            "Directory for JUnit XML and HTML reports. "
            f"Default: {_RESULTS_DIR}"
        ),
    )

    # ── Existing-result overrides ─────────────────────────────────────────
    parser.add_argument(
        "--noiseless-json", type=str, default=None, dest="noiseless_json",
        metavar="FILE",
        help="Path to existing σ=0%% result JSON (auto-detected if omitted).",
    )
    parser.add_argument(
        "--sig0005-json", type=str, default=None, dest="sig0005_json",
        metavar="FILE",
        help="Path to existing σ=0.5%% result JSON (auto-detected if omitted).",
    )
    parser.add_argument(
        "--n200-json", type=str, default=None, dest="n200_json",
        metavar="FILE",
        help="Path to existing n=200 result JSON (auto-detected if omitted).",
    )
    parser.add_argument(
        "--no-existing", action="store_true", dest="no_existing",
        help="Ignore all on-disk results and re-run every σ / n from scratch.",
    )

    # ── Resume flags ──────────────────────────────────────────────────────
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume both sweeps from the last registry. "
            "Skips any sweep that completed successfully; re-runs any that failed or "
            "were interrupted."
        ),
    )
    parser.add_argument(
        "--resume-noise-only", action="store_true", dest="resume_noise_only",
        help=(
            "Resume only the noise-level sweep. "
            "Skips it if it already completed OK; otherwise re-runs it. "
            "The SC sweep is not touched."
        ),
    )
    parser.add_argument(
        "--resume-samples-only", action="store_true", dest="resume_samples_only",
        help=(
            "Resume only the sample-complexity sweep. "
            "Skips it if it already completed OK; otherwise re-runs it. "
            "The noise sweep is not touched."
        ),
    )
    parser.add_argument(
        "--registry", type=str, default=None, dest="registry",
        metavar="FILE",
        help=(
            "Explicit path to a registry JSON to resume from. "
            "Auto-detected (most-recent file in --save-dir) when omitted."
        ),
    )

    # ── Script paths (auto-detected; override if layout differs) ─────────
    parser.add_argument(
        "--noise-script", type=str, default=str(_NOISE_SCRIPT), dest="noise_script",
        metavar="FILE", help="Path to run_noise_sweep_benchmark.py.",
    )
    parser.add_argument(
        "--sc-script", type=str, default=str(_SC_SCRIPT), dest="sc_script",
        metavar="FILE", help="Path to run_sample_complexity_benchmark.py.",
    )

    # ── Behaviour ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--fail-fast", action="store_true", dest="fail_fast",
        help=(
            "Abort the entire dual-sweep run if either sweep subprocess "
            "exits non-zero."
        ),
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # ── Normalize --test aliases to canonical Feynman IDs ────────────────
    # The inner runners only accept canonical IDs (e.g. "I.12.1").  Passing a
    # human-readable name like "newton" causes an exit-code-1 crash with no
    # result JSON written.  Resolve once here so both _build_noise_cmd and
    # _build_sc_cmd inherit the corrected value automatically.
    _TEST_ALIASES: dict = {
        "newton":          "I.12.1",
        "newton_gravity":  "I.12.1",
        "gravity":         "I.12.1",
        "coulomb":         "I.12.2",
        "kinetic_energy":  "I.12.4",
        "kinetic":         "I.12.4",
        "arrhenius":       "II.11.27",
        "ideal_gas":       "II.11.28",
    }
    if getattr(args, "test", None):
        _canonical = _TEST_ALIASES.get(args.test.lower().replace(" ", "_"), args.test)
        if _canonical != args.test:
            print(f"  [INFO] --test alias '{args.test}' resolved to canonical ID '{_canonical}'")
            args.test = _canonical

    # ── Resolve and create directories ───────────────────────────────────
    args.log_dir  = Path(args.log_dir)
    args.save_dir = Path(args.save_dir)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Resolve per-sweep log file paths from --log ───────────────────────
    # Convention matching the example command:
    #   --log sample_complexity.log noise_sweep.log
    #   → [0] = SC log, [1] = noise log
    if args.log:
        if len(args.log) == 1:
            sc_log    = Path(args.log[0])
            noise_log = Path(args.log[0])
        elif len(args.log) >= 2:
            sc_log    = Path(args.log[0])
            noise_log = Path(args.log[1])
        # Ensure parent dirs exist for explicitly supplied paths
        sc_log.parent.mkdir(parents=True, exist_ok=True)
        noise_log.parent.mkdir(parents=True, exist_ok=True)
    else:
        sc_log    = args.log_dir / "sample_complexity.log"
        noise_log = args.log_dir / "noise_sweep.log"

    # ── Resolve resume state ─────────────────────────────────────────────
    # Determine which sweeps the resume flags apply to.
    # --resume              → applies to both
    # --resume-noise-only   → noise only
    # --resume-samples-only → SC only
    resume_noise = args.resume or args.resume_noise_only
    resume_sc    = args.resume or args.resume_samples_only

    # If an explicit registry path was given, override save_dir for lookup
    if args.registry:
        _reg_override = Path(args.registry)
        if not _reg_override.exists():
            print(f"❌  Registry file not found: {_reg_override}")
            sys.exit(1)
        # Patch save_dir so _load_resume_state finds it via glob
        _reg_save_dir = _reg_override.parent
    else:
        _reg_save_dir = args.save_dir

    noise_already_done = False
    sc_already_done    = False

    if resume_noise or resume_sc:
        print(f"\n{'─'*80}")
        print("  Checking resume state …")
        noise_already_done, sc_already_done, _reg_path = _load_resume_state(
            _reg_save_dir, resume_noise, resume_sc,
        )
        # If user supplied --resume-noise-only, don't skip SC even if done
        if args.resume_noise_only and not args.resume:
            sc_already_done = False
        # If user supplied --resume-samples-only, don't skip noise even if done
        if args.resume_samples_only and not args.resume:
            noise_already_done = False

    # ── Banner ────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*80}")
    print("  HYPATIA X — DUAL SWEEP BENCHMARKS".center(80))
    print(f"{'='*80}")
    print(f"  Timestamp       : {ts}")
    print(f"  Mode            : {'DRY-RUN (add --dry-run to preview)' if args.dry_run else '🚀 LIVE RUN'}")
    print(f"  Smoke test      : {'yes (arrhenius)' if args.smoke else 'no'}")
    _eq_filter = (
        f"--test {args.test}" if args.test
        else f"--equations {args.equations}" if args.equations
        else "all equations"
    )
    _dom_filter = args.domain or "all domains"
    print(f"  Equation filter : {_eq_filter}")
    print(f"  Domain filter   : {_dom_filter}")
    print(f"  Noise levels    : {args.noise_levels}")
    print(f"  Sample sizes    : {args.sample_sizes}")
    print(f"  Methods         : {args.methods}")
    print(f"  NN seeds        : {args.nn_seeds}")
    print(f"  Samples (noise) : {args.samples}")
    print(f"  Method timeout  : {args.method_timeout}s")
    print(f"  PySR timeout    : {args.pysr_timeout}s")
    print(f"  Threshold noisy : {args.threshold_noisy}")
    print(f"  Threshold clean : {args.threshold_noiseless}")
    print(f"  SC noiseless    : {getattr(args, 'sc_noiseless', False)}")
    if not getattr(args, "sc_noiseless", False):
        print(f"  SC fixed noise  : σ={getattr(args, 'sc_fixed_noise', 0.05):.4g}  "
              f"(HYPATIAX_NOISE_LEVEL injected per n run)")
    print(f"  Fail-fast       : {'ON  ⛔ — any sweep failure aborts the run' if args.fail_fast else 'OFF (warn and continue)'}")
    print(f"  Noise script    : {args.noise_script}")
    print(f"  SC script       : {args.sc_script}")
    print(f"  Log dir         : {args.log_dir}")
    print(f"  Save dir        : {args.save_dir}")
    print(f"  Noise log       : {noise_log}")
    print(f"  SC log          : {sc_log}")
    print(f"  No-existing     : {args.no_existing}")
    _resume_desc = []
    if args.resume:             _resume_desc.append("both sweeps")
    if args.resume_noise_only:  _resume_desc.append("noise only")
    if args.resume_samples_only:_resume_desc.append("samples only")
    print(f"  Resume          : {', '.join(_resume_desc) if _resume_desc else 'no (fresh start)'}")
    if noise_already_done: print("  ↳ Noise sweep   : will be SKIPPED (completed OK in previous run)")
    if sc_already_done:    print("  ↳ SC sweep      : will be SKIPPED (completed OK in previous run)")
    print(f"{'='*80}\n")

    overall_start = time.time()
    noise_ok = sc_ok = True

    # =========================================================================
    # NOISE SWEEP
    # =========================================================================
    if noise_already_done:
        print(f"\n{'─'*80}")
        print("  ⏭️   Noise sweep skipped — completed OK in previous run.")
        noise_ok = True
    elif not args.sc_only:
        print(f"\n{'─'*80}")
        print("  Resolving existing noise-sweep results …")
        noise_existing = _resolve_existing_noise(
            args.noise_levels,
            args.noiseless_json,
            args.sig0005_json,
            args.no_existing,
        )
        noise_cmd = _build_noise_cmd(args, noise_existing, ts, noise_log)
        noise_ok  = _run_sweep(
            label     = f"Noise sweep  (σ ∈ {{{', '.join(str(s) for s in args.noise_levels)}}})",
            cmd       = noise_cmd,
            dry_run   = args.dry_run,
            fail_fast = args.fail_fast,
        )
        if not args.dry_run:
            _write_registry(args.save_dir, ts, {
                "event":        "noise_sweep_complete",
                "timestamp":    datetime.now().isoformat(),
                "ok":           noise_ok,
                "noise_levels": args.noise_levels,
                "methods":      args.methods,
                "log":          str(noise_log),
                "results_dir":  str(_RESULTS_DIR),
                "checkpoint_names": [
                    f"noise_sweep_{int(s*1000):04d}_checkpoint"
                    for s in args.noise_levels
                ],
            })
            print(f"  📄 Noise sweep log  → {noise_log}")
            print(f"  📁 Outputs          → {_RESULTS_DIR}/noise_sweep_*.json / *.csv")
            print(f"  🔖 Inner checkpoints: noise_sweep_<sigma_label>_checkpoint.json")

    # =========================================================================
    # SAMPLE COMPLEXITY SWEEP
    # =========================================================================
    if sc_already_done:
        print(f"\n{'─'*80}")
        print("  ⏭️   Sample complexity sweep skipped — completed OK in previous run.")
        sc_ok = True
    elif not args.noise_only:
        print(f"\n{'─'*80}")
        print("  Resolving existing sample-complexity results …")
        sc_existing = _resolve_existing_sc(
            args.sample_sizes,
            args.n200_json,
            args.no_existing,
        )
        sc_cmd = _build_sc_cmd(args, sc_existing, ts, sc_log)
        sc_ok  = _run_sweep(
            label     = f"Sample complexity sweep  (n ∈ {{{', '.join(str(n) for n in args.sample_sizes)}}})",
            cmd       = sc_cmd,
            dry_run   = args.dry_run,
            fail_fast = args.fail_fast,
        )
        if not args.dry_run:
            _write_registry(args.save_dir, ts, {
                "event":        "sc_sweep_complete",
                "timestamp":    datetime.now().isoformat(),
                "ok":           sc_ok,
                "sample_sizes": args.sample_sizes,
                "methods":      args.methods,
                "log":          str(sc_log),
                "results_dir":  str(_RESULTS_DIR),
                "checkpoint_names": [
                    f"sample_complexity_n{n:04d}_checkpoint"
                    for n in args.sample_sizes
                ],
            })
            print(f"  📄 SC sweep log     → {sc_log}")
            print(f"  📁 Outputs          → {_RESULTS_DIR}/sample_complexity_*.json / *.csv")
            print(f"  🔖 Inner checkpoints: sample_complexity_n<NNNN>_checkpoint.json")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_elapsed = time.time() - overall_start
    mins, secs    = divmod(int(total_elapsed), 60)

    print(f"\n{'='*80}")
    print("  SUMMARY".center(80))
    print(f"{'='*80}")

    if args.dry_run:
        print("  Dry-run complete — no benchmarks were executed.")
        print("  Remove --dry-run to run for real.")
    else:
        if not args.sc_only:
            print(f"  Noise sweep            : {'✅ OK' if noise_ok else '❌ FAILED'}")
        if not args.noise_only:
            print(f"  Sample complexity sweep: {'✅ OK' if sc_ok else '❌ FAILED'}")
        print(f"  Total elapsed          : {mins}m {secs}s")
        print()
        print("  Read results in Python:")
        print("    import pandas as pd")
        print(f"    df  = pd.read_csv('{_RESULTS_DIR}/noise_sweep_{ts}.csv')")
        print(f"    df2 = pd.read_csv('{_RESULTS_DIR}/sample_complexity_{ts}.csv')")
        print("    agg = df[df['section'] == 'aggregate']")
        print("    print(agg.pivot(index='method', columns='noise_level_pct', values='median_r2'))")

    print(f"{'='*80}\n")

    if not args.dry_run:
        _finalize_registry(args.save_dir, ts, noise_ok, sc_ok)

    if not args.dry_run and not (noise_ok and sc_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
