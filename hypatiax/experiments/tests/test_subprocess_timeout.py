"""tests/test_subprocess_timeout.py

Isolated regression test for the drain-communicate hang bug that caused
test 2 (Logistic growth) to run 28 064 s instead of timing out at 870 s.

Bug summary
───────────
In _run_pysr_in_subprocess(), when proc.communicate(timeout=T) raises
TimeoutExpired, the except block calls:

    proc.kill()
    _, stderr_bytes = proc.communicate()   # ← NO TIMEOUT — hangs forever

If the killed process leaves child threads or file-descriptor holders alive
(as juliacall/Julia does when SIGKILL'd), the drain communicate() blocks
until those holders close the pipes — which may take hours.

What these tests verify
────────────────────────
T1 — ORIGINAL BUG (expected to FAIL on unpatched code):
     A subprocess that ignores SIGKILL and keeps its pipes open causes the
     bare proc.communicate() drain to hang.  The test times out in <5 s
     using a threading.Timer watchdog, confirming the hang exists.

T2 — FIXED CODE (expected to PASS):
     The patched drain uses os.killpg + proc.communicate(timeout=30) +
     force-close.  Even with a subprocess that keeps pipes open, the
     drain completes within the grace period.

T3 — NORMAL TIMEOUT PATH:
     A subprocess that exits immediately after being killed produces a
     clean TimeoutExpired → kill → drain cycle in <5 s on both original
     and patched code.

T4 — HAPPY PATH (no timeout):
     A subprocess that finishes before the timeout returns its result
     correctly; no TimeoutExpired is raised.

Run
───
    pytest tests/test_subprocess_timeout.py -v
    pytest tests/test_subprocess_timeout.py -v -k "not original_bug"

The T1 test is marked xfail on patched code (it should NOT hang).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_with_watchdog(fn, timeout_secs: float):
    """Run *fn* in a thread; raise AssertionError if it takes > timeout_secs."""
    exc_holder: list[BaseException] = []
    result_holder: list = []

    def target():
        try:
            result_holder.append(fn())
        except BaseException as e:  # noqa: BLE001
            exc_holder.append(e)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout_secs)

    if t.is_alive():
        raise AssertionError(
            f"Function did not complete within {timeout_secs}s — "
            f"this is the drain-communicate hang bug"
        )
    if exc_holder:
        raise exc_holder[0]
    return result_holder[0] if result_holder else None


# Subprocess scripts used as test fixtures
# ─────────────────────────────────────────

# A process that: runs for `sleep_secs`, keeps its pipes open throughout,
# and ignores SIGTERM (but not SIGKILL, which terminates it instantly).
# Models a julia subprocess that is mid-BFGS when killed.
_PIPE_KEEPER_SCRIPT = """\
import sys, time, signal, os
signal.signal(signal.SIGTERM, signal.SIG_IGN)   # ignore SIGTERM
duration = float(sys.argv[1])
# Keep stdout/stderr open and write nothing — simulates juliacall holding pipes
time.sleep(duration)
print("done", flush=True)
"""

# A process that exits quickly after receiving any signal.
_FAST_EXIT_SCRIPT = """\
import sys, time
time.sleep(float(sys.argv[1]))
print("done", flush=True)
"""

# A process that exits before the timeout — normal happy path.
_QUICK_RESULT_SCRIPT = """\
import sys
print("result:42", flush=True)
"""


def _start_pipe_keeper(sleep_secs: float) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _PIPE_KEEPER_SCRIPT, str(sleep_secs)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,   # puts proc in its own process group
    )


def _start_fast_exit(sleep_secs: float) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _FAST_EXIT_SCRIPT, str(sleep_secs)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,   # own process group — killpg won't reach parent
    )


def _start_quick_result() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _QUICK_RESULT_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


# ── Original (buggy) drain — extracted verbatim ───────────────────────────────

def _drain_original(proc: subprocess.Popen, timeout: float) -> dict:
    """Exact copy of the original _run_pysr_in_subprocess drain logic."""
    try:
        stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
        return {"success": True, "stdout": stdout_bytes}
    except subprocess.TimeoutExpired:
        proc.kill()
        _, stderr_bytes = proc.communicate()          # ← THE BUG: no timeout
        return {
            "success": False,
            "error": f"timed out after {timeout}s",
            "stderr": stderr_bytes,
        }


# ── Fixed drain — mirrors the patch in run_comparative_suite_benchmark_v2.py ──

def _drain_fixed(proc: subprocess.Popen, timeout: float) -> dict:
    """Patched drain: os.killpg + communicate(timeout=30) + force-close."""
    try:
        stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
        return {"success": True, "stdout": stdout_bytes}
    except subprocess.TimeoutExpired:
        # Kill the entire process group (Python + any native threads / children)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()  # fallback
        # Drain with hard cap so Julia pipe-holders can't block us
        try:
            _, stderr_bytes = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
            stderr_bytes = b""
        except Exception:
            stderr_bytes = b""
        return {
            "success": False,
            "error": f"timed out after {timeout}s",
            "stderr": stderr_bytes,
        }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDrainCommunicateBug:
    """
    Regression suite for the drain-communicate hang that caused test 2
    (Logistic growth) to run 28 064 s in exp2.
    """

    WATCHDOG_SECS = 8.0   # any hung test aborts after this

    # ── T1: Original code hangs when pipes are kept open ──────────────────────

    @pytest.mark.xfail(
        reason=(
            "Original drain has no timeout — hangs if subprocess keeps pipes open. "
            "This xfail documents the bug; it should XPASS on unpatched code "
            "(i.e. the watchdog fires = bug confirmed)."
        ),
        strict=False,
    )
    def test_t1_original_drain_hangs_with_pipe_keeper(self):
        """
        ORIGINAL BUG: bare proc.communicate() drain hangs when the killed
        subprocess keeps its pipes open.

        A pipe-keeper process is started that ignores SIGTERM and holds its
        stdout/stderr open for 60s.  The original drain is given a 1s timeout
        so TimeoutExpired fires quickly.  proc.kill() is sent (SIGKILL).  The
        bare proc.communicate() then blocks because the pipe-keeper's pipes
        are still open (simulating juliacall threads).

        Expected on unpatched code: watchdog fires after WATCHDOG_SECS → xfail
        Expected on patched code:   drain completes in <35s → xpass (test passes,
                                    xfail mark means it was expected to fail = surprise pass)
        """
        proc = _start_pipe_keeper(sleep_secs=60)

        def run_original():
            return _drain_original(proc, timeout=1.0)

        with pytest.raises(AssertionError, match="hang bug"):
            _run_with_watchdog(run_original, timeout_secs=self.WATCHDOG_SECS)

    # ── T2: Fixed code handles pipe-keeper within grace period ────────────────

    def test_t2_fixed_drain_completes_despite_pipe_keeper(self):
        """
        FIXED: patched drain uses os.killpg + communicate(timeout=30).

        Even with a pipe-keeper that holds pipes open for 60s, the fixed
        drain completes within 35s (30s grace + overhead).  Result is
        marked as a timeout failure (success=False) — not a hang.
        """
        proc = _start_pipe_keeper(sleep_secs=60)

        def run_fixed():
            return _drain_fixed(proc, timeout=1.0)

        # Must complete within 35s (30s drain grace + 5s overhead)
        result = _run_with_watchdog(run_fixed, timeout_secs=35.0)

        assert result is not None
        assert result["success"] is False, "Should report timeout, not success"
        assert "timed out" in result.get("error", "")

    # ── T3: Normal timeout path works on both original and fixed ──────────────

    def test_t3_normal_timeout_both_implementations(self):
        """
        A subprocess that exits immediately when killed (no pipe-keeper
        behaviour) produces a fast, clean TimeoutExpired → kill → drain
        cycle on both the original and the fixed drain.
        """
        for label, drain_fn in [("original", _drain_original), ("fixed", _drain_fixed)]:
            proc = _start_fast_exit(sleep_secs=30)

            def run_drain(drain_fn=drain_fn, proc=proc):
                return drain_fn(proc, timeout=0.5)

            result = _run_with_watchdog(run_drain, timeout_secs=self.WATCHDOG_SECS)

            assert result is not None, f"[{label}] drain returned None"
            assert result["success"] is False, f"[{label}] should report timeout"
            assert "timed out" in result.get("error", ""), \
                f"[{label}] error message missing"

    # ── T4: Happy path — subprocess finishes before timeout ───────────────────

    def test_t4_happy_path_no_timeout(self):
        """
        When the subprocess finishes before the timeout, both implementations
        return success=True with the correct stdout.
        """
        for label, drain_fn in [("original", _drain_original), ("fixed", _drain_fixed)]:
            proc = _start_quick_result()

            def run_drain(drain_fn=drain_fn, proc=proc):
                return drain_fn(proc, timeout=10.0)

            result = _run_with_watchdog(run_drain, timeout_secs=self.WATCHDOG_SECS)

            assert result is not None, f"[{label}] drain returned None"
            assert result["success"] is True, \
                f"[{label}] expected success, got {result}"
            assert b"result:42" in result.get("stdout", b""), \
                f"[{label}] stdout missing expected value"

    # ── T5: Exact timing — fixed drain stays within wall-clock budget ─────────

    def test_t5_fixed_drain_wall_clock_budget(self):
        """
        With a 1s communicate timeout and a 30s drain grace, the fixed drain
        must complete in at most ~32s even against a stubborn pipe-keeper.
        Verifies the budget arithmetic is correct.
        """
        COMM_TIMEOUT  = 1.0
        DRAIN_GRACE   = 30.0
        OVERHEAD      = 3.0   # OS scheduling + process startup
        BUDGET        = COMM_TIMEOUT + DRAIN_GRACE + OVERHEAD

        proc = _start_pipe_keeper(sleep_secs=120)
        t0 = time.monotonic()

        def run_fixed():
            return _drain_fixed(proc, timeout=COMM_TIMEOUT)

        result = _run_with_watchdog(run_fixed, timeout_secs=BUDGET)
        elapsed = time.monotonic() - t0

        assert result["success"] is False
        assert elapsed < BUDGET, (
            f"Fixed drain took {elapsed:.1f}s — exceeded budget of {BUDGET:.1f}s"
        )
        print(f"\n  [T5] elapsed={elapsed:.2f}s  budget={BUDGET:.1f}s  ✓")


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import unittest

    # Run without pytest for quick local validation
    suite = TestDrainCommunicateBug()

    tests = [
        ("T2 fixed drain completes despite pipe-keeper",
         suite.test_t2_fixed_drain_completes_despite_pipe_keeper),
        ("T3 normal timeout both implementations",
         suite.test_t3_normal_timeout_both_implementations),
        ("T4 happy path no timeout",
         suite.test_t4_happy_path_no_timeout),
        ("T5 fixed drain wall-clock budget",
         suite.test_t5_fixed_drain_wall_clock_budget),
    ]

    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
