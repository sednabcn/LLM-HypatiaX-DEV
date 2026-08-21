#!/usr/bin/env python3
"""
verify_proc_box_fix.py
=======================
Wall-clock timing test for the Item 10b _proc_box fix in
run_comparative_suite_benchmark_v2.py.

This does NOT touch Julia/PySR (network-blocked, can't install here).
It reuses the actual harness primitives -- _ProcBox, _kill_process_group,
_run_pysr_in_subprocess, and the same ThreadPoolExecutor + future.result()
pattern used by the real per-method outer timeout -- but swaps
_SUBPROCESS_WORKER for a tiny script that just sleeps, standing in for a
long-running PySR call.

Setup mirrors the real bug exactly:
  - inner subprocess timeout: 30s  (stand-in for the real 900-1100s)
  - outer ThreadPoolExecutor timeout: 3s  (stand-in for the real 300s)

Pre-fix behaviour (bug): the outer timeout fires at ~3s but the harness
has no reference to the real subprocess, so it keeps running until ITS
OWN 30s inner timeout -- the harness's outer-timeout promise is broken by
~27s.

Post-fix behaviour: proc_box lets the outer handler reach the live Popen
and SIGKILL its whole process group the moment the outer timeout fires,
so the subprocess is dead within ~3s, not ~30s.
"""

import concurrent.futures as _cf
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hypatiax.experiments.benchmarks.run_comparative_suite_benchmark_v2 as m

# Stand-in for the real Julia/PySR worker: just sleep, ignoring SIGTERM,
# so only a SIGKILL (via _kill_process_group) actually stops it -- same
# shape as an uninterruptible Julia call.
FAKE_WORKER = """
import signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
sys.stdin.read()  # matches real worker's stdin protocol (communicate() writes to it)
time.sleep({sleep_s})
print('{{"success": true, "note": "should never get here"}}')
"""

INNER_TIMEOUT = 30   # stand-in for real 900-1100s PySR budget
OUTER_TIMEOUT = 3    # stand-in for real 300s per-method hard timeout
SLEEP_S = 60          # worker sleeps far longer than either timeout


def run_case(use_proc_box: bool):
    m._SUBPROCESS_WORKER = FAKE_WORKER.format(sleep_s=SLEEP_S)

    proc_box = m._ProcBox() if use_proc_box else None

    pool = _cf.ThreadPoolExecutor(max_workers=1)
    t0 = time.time()
    future = pool.submit(
        m._run_pysr_in_subprocess,
        method="symbolic_engine",
        X=__import__("numpy").zeros((2, 2)),
        y=__import__("numpy").zeros(2),
        var_names=["x0", "x1"],
        description="dummy",
        metadata={},
        timeout=INNER_TIMEOUT,
        proc_box=proc_box,
    )

    timed_out = False
    try:
        future.result(timeout=OUTER_TIMEOUT)
    except _cf.TimeoutError:
        timed_out = True
        if proc_box is not None:
            live_proc = proc_box.get()
            if live_proc is not None:
                m._kill_process_group(live_proc)
    outer_elapsed = time.time() - t0

    # Wait (bounded) for the background thread/subprocess to actually die,
    # to see how long the *real* resource lived past the outer timeout.
    try:
        future.result(timeout=INNER_TIMEOUT + 5)
        thread_finished = True
    except _cf.TimeoutError:
        thread_finished = False
    total_elapsed = time.time() - t0
    pool.shutdown(wait=False)

    return {
        "timed_out_at_outer": timed_out,
        "outer_elapsed_s": round(outer_elapsed, 1),
        "total_elapsed_s": round(total_elapsed, 1),
        "thread_finished_within_bound": thread_finished,
    }


if __name__ == "__main__":
    print(f"Config: outer_timeout={OUTER_TIMEOUT}s  inner_timeout={INNER_TIMEOUT}s  worker_sleep={SLEEP_S}s\n")

    print("=== WITHOUT proc_box (reproduces the pre-fix bug) ===")
    r_broken = run_case(use_proc_box=False)
    print(r_broken)

    print("\n=== WITH proc_box (the fix as implemented in the uploaded file) ===")
    r_fixed = run_case(use_proc_box=True)
    print(r_fixed)

    print("\n=== Verdict ===")
    gap_broken = r_broken["total_elapsed_s"] - OUTER_TIMEOUT
    gap_fixed = r_fixed["total_elapsed_s"] - OUTER_TIMEOUT
    print(f"Without proc_box: real subprocess kept running ~{gap_broken:.1f}s past the outer timeout"
          f" (tracks the {INNER_TIMEOUT}s inner timeout instead).")
    print(f"With proc_box:    real subprocess died ~{gap_fixed:.1f}s after the outer timeout fired"
          f" (tracks the {OUTER_TIMEOUT}s outer timeout, as intended).")
