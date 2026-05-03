#!/usr/bin/env python3
"""
run_comparative_suite_benchmark_injected.py
===========================================
Full 30-equation comparison suite that INJECTS pre-computed method-5 and
method-6 results instead of running PySR inline.

Workflow
--------
1.  Run methods 1–4 (fast: PureLLM, ImprovedNN, HybridDeFi, HybridAllDomains)
    across all 30 equations — same as the base suite but --skip-pysr.
2.  Load method-5 results from logs/exp2_symbolic_engine_checkpoint.json
    (produced by run_exp2_symbolic_engine.py).
3.  Load method-6 results from logs/exp2_hybrid_system_checkpoint.json
    (produced by run_exp2_hybrid_system.py).
4.  Merge 5+6 into every test record, recompute _compare() winner/rankings,
    then save the combined results JSON — identical in schema to the output
    of run_comparative_suite_benchmark_v2.py.

Why this exists
---------------
SymbolicEngineWithLLM and HybridDiscoverySystem v50_2 both launch Julia
subprocesses.  Running them inside the same Python process as the four fast
methods means a single OOM/SIGKILL wipes all progress.  The injected design
lets the slow PySR jobs run in separate, resumable processes while the fast
methods (which complete in <5 min total) run in a lightweight final sweep.

Usage
-----
    # 1. Run slow PySR methods first (each resumable independently):
    python3 run_exp2_symbolic_engine.py --resume
    python3 run_exp2_hybrid_system.py   --resume

    # 2. Run fast methods + inject PySR results:
    python3 run_comparative_suite_benchmark_injected.py [--resume]

    # Or from the pipeline orchestrator:
    python3 run_all_checkpoint.py --resume   # exp2_sym, exp2_hyb, exp2_inject steps

CLI flags (subset of the original benchmark script)
----------------------------------------------------
    --resume              Skip equations already in the suite checkpoint
    --checkpoint-name     Override checkpoint stem (default: exp2_all30_injected_checkpoint)
    --samples             Samples per equation (default: 200)
    --quiet               Suppress per-method verbose output
    --no-checkpoint       Disable checkpoint writes
    --sym-checkpoint      Path to method-5 checkpoint (default: logs/exp2_symbolic_engine_checkpoint.json)
    --hyb-checkpoint      Path to method-6 checkpoint (default: logs/exp2_hybrid_system_checkpoint.json)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — same as run_comparative_suite_benchmark_v2.py
# ---------------------------------------------------------------------------
_HERE     = Path(__file__).resolve().parent          # …/hypatiax/experiments/benchmarks/
_PKG_ROOT = _HERE.parent.parent                      # …/hypatiax/
_REPO_ROOT = _PKG_ROOT.parent                        # repo root
sys.path.insert(0, str(_REPO_ROOT))                  # so 'import hypatiax.*' works

_DEFAULT_SYM_CKPT = _REPO_ROOT / "logs" / "exp2_symbolic_engine_checkpoint.json"
_DEFAULT_HYB_CKPT = _REPO_ROOT / "logs" / "exp2_hybrid_system_checkpoint.json"

# ---------------------------------------------------------------------------
# juliacall guard (needed because imports inside the bench module probe Julia)
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
try:
    import juliacall as _jc  # noqa: F401
except Exception:
    pass

# ---------------------------------------------------------------------------
# Import the full benchmark module so we reuse ProtocolBenchmarkSuite,
# MethodResult, _compare, save_checkpoint etc. unchanged.
# ---------------------------------------------------------------------------
_BENCH_PATH = _HERE / "run_comparative_suite_benchmark_v2.py"
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_bench_v2", _BENCH_PATH)
_bench = _ilu.module_from_spec(_spec)          # type: ignore[arg-type]
_spec.loader.exec_module(_bench)               # type: ignore[union-attr]

ProtocolBenchmarkSuite = _bench.ProtocolBenchmarkSuite
MethodResult           = _bench.MethodResult

# Protocol
from hypatiax.protocols.experiment_protocol_all_30 import ExperimentProtocolAll

import numpy as np


# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------

def _load_pysr_checkpoint(path: Path, method_label: str) -> dict[str, dict]:
    """
    Load a method-5 or method-6 checkpoint.
    Returns dict keyed by eq_key → result dict (MethodResult.to_dict() schema).
    Missing or unreadable file → prints warning, returns empty dict so the
    suite still runs but that method column will be missing/failed.
    """
    if not path.exists():
        print(f"⚠️  {method_label} checkpoint not found: {path}", flush=True)
        print(f"    Run the standalone runner first, then retry.", flush=True)
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        results = data.get("results", {})
        n = len([v for v in results.values() if v and v.get("result")])
        print(f"✅  {method_label}: loaded {n}/{len(results)} equations from {path}", flush=True)
        return results
    except Exception as exc:
        print(f"⚠️  Could not read {method_label} checkpoint ({exc}): {path}", flush=True)
        return {}


def _eq_key(meta: dict, domain: str) -> str:
    return f"{domain}::{meta.get('equation_name', meta.get('name', str(meta)))}"


def _result_dict_to_method_result(r_dict: dict | None, method_name: str) -> MethodResult:
    """Reconstruct a MethodResult from a saved to_dict() record."""
    if r_dict is None:
        return MethodResult(
            method=method_name, success=False,
            r2=0.0, rmse=float("inf"),
            formula="N/A", error="not computed",
        )
    return MethodResult(
        method      = r_dict.get("method", method_name),
        success     = bool(r_dict.get("success", False)),
        r2          = float(r_dict.get("r2", 0.0)),
        rmse        = float(r_dict.get("rmse", float("inf"))),
        formula     = r_dict.get("formula", "N/A"),
        formula_hash= r_dict.get("formula_hash", ""),
        error       = r_dict.get("error"),
        time        = float(r_dict.get("time", 0.0)),
        metadata    = r_dict.get("metadata", {}),
    )


def _inject_pysr_into_record(
    record: dict,
    sym_by_key: dict,
    hyb_by_key: dict,
    domain: str,
    meta: dict,
    suite: ProtocolBenchmarkSuite,
    y: np.ndarray,
) -> dict:
    """
    Given a record produced by suite.run_test() (methods 1–4 only), inject
    the pre-computed method-5 and method-6 MethodResult dicts into it and
    recompute the comparison/winner fields.
    """
    key = _eq_key(meta, domain)

    sym_raw = sym_by_key.get(key, {})
    hyb_raw = hyb_by_key.get(key, {})

    sym_r = _result_dict_to_method_result(
        sym_raw.get("result") if sym_raw else None,
        "SymbolicEngineWithLLM (tools)",
    )
    hyb_r = _result_dict_to_method_result(
        hyb_raw.get("result") if hyb_raw else None,
        "HybridDiscoverySystem v50_2 (tools)",
    )

    # Merge into existing results dict
    existing = record.get("results", {})
    existing[sym_r.method] = sym_r.to_dict()
    existing[hyb_r.method] = hyb_r.to_dict()

    # Reconstruct full MethodResult objects for _compare()
    full_results: dict[str, MethodResult] = {}
    for name, rd in existing.items():
        full_results[name] = _result_dict_to_method_result(rd, name)

    # Recompute comparison with all 6 methods present
    comparison = suite._compare(full_results, y)

    record["results"]    = {n: r.to_dict() for n, r in full_results.items()}
    record["comparison"] = comparison
    record["winner"]     = comparison["winner"]
    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exp-2 injected suite: methods 1–4 live + method 5+6 from checkpoints"
    )
    parser.add_argument("--resume", action="store_true",
                        help="Skip equations already in the suite checkpoint")
    parser.add_argument("--checkpoint-name", dest="checkpoint_name",
                        default="exp2_all30_injected_checkpoint",
                        help="Checkpoint stem (default: exp2_all30_injected_checkpoint)")
    parser.add_argument("--samples", type=int, default=200,
                        help="Samples per equation (default: 200)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-method verbose output")
    parser.add_argument("--no-checkpoint", action="store_true", dest="no_checkpoint",
                        help="Disable checkpoint writes")
    parser.add_argument("--sym-checkpoint", dest="sym_checkpoint",
                        default=str(_DEFAULT_SYM_CKPT),
                        help="Path to method-5 (SymbolicEngine) checkpoint")
    parser.add_argument("--hyb-checkpoint", dest="hyb_checkpoint",
                        default=str(_DEFAULT_HYB_CKPT),
                        help="Path to method-6 (HybridSystem) checkpoint")
    args = parser.parse_args()

    # Override checkpoint name in bench module so save_checkpoint() uses correct path
    _bench._CHECKPOINT_NAME = args.checkpoint_name

    # Load PySR pre-computed results
    sym_by_key = _load_pysr_checkpoint(
        Path(args.sym_checkpoint), "SymbolicEngineWithLLM (tools)")
    hyb_by_key = _load_pysr_checkpoint(
        Path(args.hyb_checkpoint), "HybridDiscoverySystem v50_2 (tools)")

    # Build suite with ONLY methods 1–4 (no Julia/PySR)
    suite = ProtocolBenchmarkSuite(
        method_indices=[1, 2, 3, 4],
        verbose=not args.quiet,
    )
    suite._noiseless = False
    suite._threshold = 0.995

    # Protocol
    protocol = ExperimentProtocolAll()
    print(f"\n✅  ExperimentProtocolAll loaded  (protocol=all30, 30 equations)", flush=True)

    # Collect all 30 test stubs
    all_tests: list[tuple] = []
    for domain in protocol.get_all_domains():
        for (description, X, y, var_names, meta) in protocol.load_test_data(
                domain, num_samples=args.samples):
            all_tests.append((description, var_names, meta, domain))
            del X, y

    total = len(all_tests)

    # Resume state
    completed_keys: list[str] = []
    if args.resume:
        ckpt = ProtocolBenchmarkSuite.load_checkpoint()
        if ckpt:
            completed_keys = ckpt.get("completed", [])
            suite.results  = ckpt.get("tests", [])
            print(f"\n♻️  Resuming — {len(completed_keys)} done, "
                  f"{total - len(completed_keys)} remaining.", flush=True)
        else:
            print("\nℹ️  --resume: no checkpoint found, starting from scratch.", flush=True)

    use_checkpoint = not args.no_checkpoint
    if use_checkpoint:
        suite.save_checkpoint(total, completed_keys)
        print(f"📋  Checkpoint → {ProtocolBenchmarkSuite._checkpoint_path()}\n", flush=True)

    suite_start = time.time()
    test_times: list[float] = []
    global_done = len(completed_keys)

    def _eq_key_from_stub(meta, domain):
        return f"{domain}::{meta.get('equation_name', meta.get('name', str(meta)))}"

    print(f"\n🚀  Running {total} test case(s)…\n", flush=True)

    for i, (description, var_names, meta, domain) in enumerate(all_tests, 1):
        eq_key = _eq_key_from_stub(meta, domain)

        if eq_key in completed_keys:
            print(f"  ⏭️  SKIP {i}/{total}: {meta.get('equation_name', eq_key)}", flush=True)
            continue

        t_start = time.time()
        elapsed = t_start - suite_start

        print(f"\n{'='*80}", flush=True)
        print(f"  TEST {i}/{total}".center(80), flush=True)
        print(f"{'='*80}", flush=True)

        # Lazy-load X, y
        loaded = protocol.load_test_data(domain, num_samples=args.samples)
        match  = next(
            (c for c in loaded
             if c[4].get("equation_name", c[0]) == meta.get("equation_name", description)
             or c[0] == description),
            None,
        )
        if match is None:
            print(f"  ⚠️  Could not reload X/y for '{description}' — skipping", flush=True)
            global_done += 1
            completed_keys.append(eq_key)
            continue

        _, X, y, _, _ = match
        y_copy = y.copy()   # keep y for _compare() after del
        del match, loaded

        # Run methods 1–4 live
        record = suite.run_test(
            description=description,
            X=X, y=y,
            var_names=var_names,
            metadata=meta,
            domain=domain,
            verbose=not args.quiet,
        )

        del X, y
        gc.collect()

        # Inject method 5+6 from pre-computed checkpoints and recompute winner
        record = _inject_pysr_into_record(
            record, sym_by_key, hyb_by_key, domain, meta, suite, y_copy
        )
        del y_copy

        # Replace the last appended record (suite.run_test already appended it)
        if suite.results:
            suite.results[-1] = record

        t_elapsed = time.time() - t_start
        test_times.append(t_elapsed)
        global_done += 1
        completed_keys.append(eq_key)

        if use_checkpoint:
            suite.save_checkpoint(total, completed_keys)

        left = total - global_done
        avg  = sum(test_times) / len(test_times)
        print(f"\n  ✔  {global_done}/{total} done  |  this: {t_elapsed:.0f}s  |  "
              f"avg: {avg:.0f}s  |  ETA: {left * avg:.0f}s  |  {left} left", flush=True)

    # Summary + save final JSON
    try:
        suite.print_summary()
    except Exception as exc:
        print(f"⚠️  print_summary error: {exc}", flush=True)

    try:
        suite._save(args)
    except Exception as exc:
        print(f"⚠️  _save error: {exc}", flush=True)

    # Clear checkpoint on full success
    if use_checkpoint and len(completed_keys) >= total:
        ProtocolBenchmarkSuite.clear_checkpoint()

    wall = time.time() - suite_start
    h, rem = divmod(int(wall), 3600)
    m, s   = divmod(rem, 60)

    solved = sum(
        1 for rec in suite.results
        if any(
            r.get("success") and np.isfinite(r.get("r2", 0)) and r.get("r2", 0) >= 0.995
            for r in rec.get("results", {}).values()
        )
    )
    print(f"\n✅  DONE — {solved}/{total} equations solved  |  wall {h:02d}:{m:02d}:{s:02d}",
          flush=True)

    return 0 if solved >= 9 else 1


if __name__ == "__main__":
    sys.exit(main())
