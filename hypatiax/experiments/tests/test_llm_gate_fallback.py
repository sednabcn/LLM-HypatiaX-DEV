#!/usr/bin/env python3
"""
test_llm_gate_fallback.py
==========================

Purpose
-------
Exercises `exp3_nguyen12_consolidated.py` on just the 5 tasks that ever
showed a gate-related problem in the baseline data -- N5, N9, N10, N11,
N12 -- running EACH seed TWICE, back to back, in the same job:

  1. STRICT   (LLM_REQUIRE_VALID_GUESS=1, i.e. gate=TRUE, today's default)
  2. FALLBACK (LLM_REQUIRE_VALID_GUESS=0, i.e. gate=FALSE)

and prints a direct STRICT-vs-FALLBACK comparison per (seed, task), rather
than diffing against old baseline JSON files. This is the fast path: 5
tasks x 2 gate settings x N seeds, instead of a full 12-task x 2-setting
sweep.

Note: N10 and N11 were never gate failures in any baseline seed (only N5,
N9, N12 were) -- they're included here as in-run canaries, so a paired
STRICT/FALLBACK regression on a task that should be unaffected by the gate
is visible immediately, in the same table, without a separate step.

It does NOT re-implement any SR logic -- it drives the real script as a
subprocess (so it exercises the actual code path, not a mock of it).

Requirements to actually execute (this harness does not stub these out):
  - The real repo checkout with hypatiax/, PySR, juliacall/Julia, sklearn,
    sympy, and hypatia.py present (same requirements as the script itself).
  - ANTHROPIC_API_KEY set (USE_LLM needs it; without it the script forces
    USE_LLM=False and the LLM gate never engages at all, which would make
    this test meaningless).
  - Wall-clock budget: each (seed, task) pair can take up to
    PYSR_TIMEOUT seconds x 2 (H + P) per the script's own numbers
    (~5-9 min/task observed in the seed42/99/123/777/2024 result files).
    Running all 5 tasks together in ONE subprocess call per (seed, gate
    setting) -- rather than one call per task -- is what keeps this fast;
    see run_case()'s task_ids list.

Usage
-----
    # Live STRICT vs FALLBACK comparison on N5/N9/N10/N11/N12, all 5 seeds
    python3 test_llm_gate_fallback.py \\
        --script /path/to/exp3_nguyen12_consolidated.py \\
        --results-dir /tmp/gate_compare_results

    # Optionally still diff against old baseline JSONs too
    python3 test_llm_gate_fallback.py \\
        --script /path/to/exp3_nguyen12_consolidated.py \\
        --results-dir /tmp/gate_compare_results \\
        --baseline-dir /mnt/user-data/uploads

    # Narrow to specific seeds/tasks
    python3 test_llm_gate_fallback.py --script ... --seeds 99 777 \\
        --tasks N5 N12

    # Dry run: prints the exact subprocess commands/env without executing
    # PySR/Julia/the LLM -- use this to sanity-check the harness itself.
    python3 test_llm_gate_fallback.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────

RECOVERY_THRESHOLD = 0.9999  # matches the script's own h_recovered/p_recovered def

# Baseline seeds we have prior results for (seed -> filename glob in --baseline-dir)
BASELINE_FILES = {
    42:   "exp3_nguyen12_seed42.json",
    99:   "exp3_nguyen12_seed99_nshards01.json",
    123:  "exp3_nguyen12_seed123_nshards02.json",
    777:  "exp3_nguyen12_seed777_nshards03.json",
    2024: "exp3_nguyen12_seed2024_nshards04.json",
}

ALL_TASKS = [f"N{i}" for i in range(1, 13)]

# The fast-path task set: every task that ever gate-aborted in the
# baseline (N5, N9, N12), plus two same-family tasks that never did
# (N10, N11) as in-run canaries -- so a STRICT-vs-FALLBACK regression on
# an "unaffected" task shows up in the same table, same run, no separate
# step required.
DEFAULT_TASKS = ["N5", "N9", "N10", "N11", "N12"]


# ─────────────────────────────────────────────────────────────────────────
# Baseline loading
# ─────────────────────────────────────────────────────────────────────────

def load_baseline(baseline_dir: Path) -> dict:
    """Returns {seed: {task_name: {"h_r2": float|None, "p_r2": float|None}}}."""
    baseline = {}
    for seed, fname in BASELINE_FILES.items():
        path = baseline_dir / fname
        if not path.exists():
            print(f"  [baseline] WARNING: {path} not found, skipping seed {seed}")
            continue
        with open(path) as f:
            d = json.load(f)
        per_task = {}
        h_list = d["results"]["hypatiax"]
        p_list = d["results"]["pysr"]
        for h, p in zip(h_list, p_list):
            name = h["metadata"]["name"]              # e.g. "Nguyen-5"
            short = "N" + name.split("-")[1]            # -> "N5"
            per_task[short] = {
                "h_r2": _safe_r2(h["evaluation"]["r2"]),
                "p_r2": _safe_r2(p["evaluation"]["r2"]),
            }
        baseline[seed] = per_task
    return baseline


def _safe_r2(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return None
    return float(x)


def recovered(r2: Optional[float]) -> bool:
    return r2 is not None and r2 >= RECOVERY_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────
# Running the real script
# ─────────────────────────────────────────────────────────────────────────

def run_case(
    script: Path,
    seed: int,
    task_ids: list[str],
    results_dir: Path,
    run_index: int,
    require_valid_guess: bool,
    temperature: float = 0.25,
    n_candidates: Optional[int] = None,
    timeout_s: int = 3600,
    dry_run: bool = False,
) -> Optional[dict]:
    """Runs exp3_nguyen12_consolidated.py for one seed against `task_ids`
    with LLM_REQUIRE_VALID_GUESS set as requested, in an isolated
    results_dir so it can never silently return a cached result written
    under the old (strict) flag. Returns the parsed result JSON, or None
    if the run failed / dry-run.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TASK_IDS"] = " ".join(task_ids)
    env["LLM_REQUIRE_VALID_GUESS"] = "1" if require_valid_guess else "0"
    env["RESULTS_DIR"] = str(results_dir)
    # Belt-and-suspenders: RESULTS_DIR + run_index together guarantee this
    # invocation's output filename can't collide with any prior run's
    # cache, regardless of what LLM_REQUIRE_VALID_GUESS was set to before.
    env["EXPERIMENT_SEED"] = str(seed)

    cmd = [
        sys.executable, str(script),
        "--seed", str(seed),
        "--temperature", str(temperature),
        "--run-index", str(run_index),
    ]
    if n_candidates is not None:
        cmd += ["--n-candidates", str(n_candidates)]

    tag = "FALLBACK(off)" if not require_valid_guess else "STRICT(on)"
    print(f"  [{tag}] seed={seed} tasks={task_ids} run_index={run_index}")

    if dry_run:
        print(f"    would run: TASK_IDS='{env['TASK_IDS']}' "
              f"LLM_REQUIRE_VALID_GUESS={env['LLM_REQUIRE_VALID_GUESS']} "
              f"RESULTS_DIR={env['RESULTS_DIR']} {' '.join(cmd)}")
        return None

    t0 = time.time()
    try:
        # No cwd override here: `script` (and RESULTS_DIR) are relative
        # paths resolved against the caller's own working directory (repo
        # root in CI). Setting cwd=script.parent while still passing a
        # relative script path double-joins the parent directory onto
        # itself -- e.g. ".../benchmarks/hypatiax/.../benchmarks/exp3_...py"
        # -- and the subprocess fails with "No such file or directory"
        # before any actual SR search runs (this was caught by a live CI
        # run: both STRICT and FALLBACK calls failed identically at 0s
        # elapsed with that exact doubled path in stderr). If the target
        # script ever needs a specific cwd, resolve `script` to an
        # absolute path first so the two can't compound.
        proc = subprocess.run(
            cmd, env=env,
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        print(f"    ✗ TIMEOUT after {timeout_s}s")
        return None
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"    ✗ subprocess exited {proc.returncode} after {elapsed:.0f}s")
        print("    ── stderr tail ──")
        print("    " + "\n    ".join(proc.stderr.strip().splitlines()[-15:]))
        return None

    _temp_tag = str(temperature).rstrip("0").rstrip(".").replace(".", "p") or "0"
    out_path = results_dir / f"exp3_nguyen12_seed{seed}_temp{_temp_tag}_run{run_index}.json"
    if not out_path.exists():
        print(f"    ✗ expected output {out_path} not found (elapsed {elapsed:.0f}s)")
        return None

    with open(out_path) as f:
        result = json.load(f)
    print(f"    ✓ done in {elapsed:.0f}s -> {out_path}")
    return result


def extract_per_task(result: dict) -> dict:
    """Same shape as load_baseline()'s per-seed dict, from a fresh run's JSON."""
    per_task = {}
    for h, p in zip(result["results"]["hypatiax"], result["results"]["pysr"]):
        name = h["metadata"]["name"]
        short = "N" + name.split("-")[1]
        per_task[short] = {
            "h_r2": _safe_r2(h["evaluation"]["r2"]),
            "p_r2": _safe_r2(p["evaluation"]["r2"]),
            "h_gate_failed": h.get("llm_gate_failed"),
        }
    return per_task


# ─────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────

def _fmt(r2: Optional[float]) -> str:
    return "FAIL" if not recovered(r2) else f"{r2:.4f}"


def report_pair(seed: int, task: str, strict: Optional[dict], fallback: Optional[dict],
                 baseline: Optional[dict] = None) -> str:
    """One line: seed, task, STRICT(gate=TRUE) result, FALLBACK(gate=FALSE)
    result, and a verdict. Optionally cross-checks the freshly-run STRICT
    result against an old baseline JSON, in case the two disagree (e.g.
    non-determinism in the LLM sampling at "the same" seed/temperature).
    """
    s_r2 = strict.get(task, {}).get("h_r2") if strict else None
    f_r2 = fallback.get(task, {}).get("h_r2") if fallback else None
    s_str = "[not run]" if strict is None else _fmt(s_r2)
    f_str = "[not run]" if fallback is None else _fmt(f_r2)

    if strict is None or fallback is None:
        verdict = "INCOMPLETE"
    else:
        s_ok, f_ok = recovered(s_r2), recovered(f_r2)
        if not s_ok and f_ok:
            verdict = "FIXED by fallback ✓"
        elif s_ok and not f_ok:
            verdict = "REGRESSION ✗✗✗ (fallback broke a passing case)"
        elif s_ok and f_ok:
            verdict = "OK (both pass)"
        else:
            verdict = "still failing in both ✗"

    line = f"  seed {seed}  {task}:  STRICT(gate=TRUE)={s_str}   FALLBACK(gate=FALSE)={f_str}   [{verdict}]"

    if baseline is not None:
        base_r2 = baseline.get(task, {}).get("h_r2")
        if strict is not None and _fmt(base_r2) != s_str:
            line += (f"\n    ⚠ live STRICT run ({s_str}) differs from stored baseline "
                     f"({_fmt(base_r2)}) -- LLM sampling is not perfectly reproducible "
                     f"at this seed/temperature; treat the live pair above as the "
                     f"authoritative comparison for this run.")
    return line


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", type=Path, default=Path("exp3_nguyen12_consolidated.py"),
                     help="Path to exp3_nguyen12_consolidated.py")
    ap.add_argument("--tasks", type=str, nargs="+", default=DEFAULT_TASKS,
                     help=f"Task IDs to test, run together in ONE subprocess call per "
                          f"(seed, gate setting) for speed (default: {DEFAULT_TASKS})")
    ap.add_argument("--baseline-dir", type=Path, default=None,
                     help="Optional: directory containing old seed42/99/123/777/2024 result "
                          "JSONs, to sanity-check the freshly-run STRICT result against "
                          "(informational only -- the live STRICT vs FALLBACK pair from "
                          "this run is what actually gates PASS/FAIL)")
    ap.add_argument("--results-dir", type=Path, default=Path("./gate_compare_results"),
                     help="Fresh output directory for both the STRICT and FALLBACK runs "
                          "(must not reuse an old RESULTS_DIR, or a cached run could be "
                          "returned instead of a real re-run)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 99, 123, 777, 2024])
    ap.add_argument("--run-index-strict", type=int, default=10,
                     help="run-index for the gate=TRUE runs (distinct from any prior cache)")
    ap.add_argument("--run-index-fallback", type=int, default=11,
                     help="run-index for the gate=FALSE runs (distinct from the strict run "
                          "above and from any prior cache)")
    ap.add_argument("--timeout", type=int, default=3600, help="Per-subprocess timeout, seconds")
    ap.add_argument("--dry-run", action="store_true",
                     help="Print the commands/env that would run, without executing anything")
    args = ap.parse_args()

    baseline = {}
    if args.baseline_dir is not None:
        print("=" * 72)
        print("STEP 0/2 — loading baseline (for informational cross-check only)")
        print("=" * 72)
        baseline = load_baseline(args.baseline_dir)
        print()

    print("=" * 72)
    print(f"STEP 1/2 — STRICT (gate=TRUE, LLM_REQUIRE_VALID_GUESS=1) on {args.tasks}")
    print("=" * 72)
    strict_results = {}
    for seed in args.seeds:
        result = run_case(
            script=args.script, seed=seed, task_ids=args.tasks,
            results_dir=args.results_dir, run_index=args.run_index_strict,
            require_valid_guess=True, timeout_s=args.timeout, dry_run=args.dry_run,
        )
        strict_results[seed] = extract_per_task(result) if result else None

    print()
    print("=" * 72)
    print(f"STEP 2/2 — FALLBACK (gate=FALSE, LLM_REQUIRE_VALID_GUESS=0) on {args.tasks}")
    print("=" * 72)
    fallback_results = {}
    for seed in args.seeds:
        result = run_case(
            script=args.script, seed=seed, task_ids=args.tasks,
            results_dir=args.results_dir, run_index=args.run_index_fallback,
            require_valid_guess=False, timeout_s=args.timeout, dry_run=args.dry_run,
        )
        fallback_results[seed] = extract_per_task(result) if result else None

    # ── Summary ─────────────────────────────────────────────────────────
    print()
    print("#" * 72)
    print(f"# SUMMARY: STRICT(gate=TRUE) vs FALLBACK(gate=FALSE), tasks={args.tasks}")
    print("#" * 72)
    any_regression = False
    any_incomplete = False
    for seed in args.seeds:
        for task in args.tasks:
            line = report_pair(
                seed, task,
                strict_results.get(seed), fallback_results.get(seed),
                baseline=baseline.get(seed) if baseline else None,
            )
            print(line)
            if "REGRESSION" in line:
                any_regression = True
            if "INCOMPLETE" in line:
                any_incomplete = True

    print()
    if args.dry_run:
        print("(dry run — no subprocesses were actually executed)")
    elif any_regression:
        print("RESULT: ⚠ at least one regression detected — see REGRESSION ✗✗✗ lines above.")
        sys.exit(1)
    elif any_incomplete:
        print("RESULT: ⚠ at least one run did not complete — see INCOMPLETE lines above.")
        sys.exit(1)
    else:
        print("RESULT: ✓ no regressions — fallback recovers what strict couldn't, "
              "without breaking anything strict already passed.")


if __name__ == "__main__":
    main()
