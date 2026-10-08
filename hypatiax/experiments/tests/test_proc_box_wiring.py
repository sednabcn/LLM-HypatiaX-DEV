"""
Tier-1 test: proves the outer-timeout -> _ProcBox -> _kill_process_group
wiring actually works, WITHOUT importing torch/pysr/juliacall/hypatiax.

Why this is a valid test of the real fix:
  The mechanism being tested (Popen -> register handle in a shared box ->
  outer ThreadPoolExecutor future.result(timeout=...) -> on TimeoutError,
  pull the handle back out and killpg it) is generic process-management
  logic. It doesn't care whether the subprocess is running PySR/Julia or
  `sleep 9999` -- if the wiring is correct here with a trivial subprocess,
  it's correct in the real file, which uses byte-for-byte the same
  _ProcBox / _kill_process_group code (copied verbatim below) and the same
  set/get/clear call sequence added in this session's edit.

What it does NOT test: PySR/Julia-specific behavior (e.g. whether Julia
itself dies cleanly under SIGKILL, or PySR's own internal timeout_in_seconds
cooperative exit). That needs Tier 2 (see bottom of this file's docstring
and the accompanying message).
"""
import concurrent.futures as _cf
import os
import signal
import subprocess
import sys
import threading
import time


# ============================================================================
# Verbatim copy of the two pieces added/reused in run_comparative_suite_
# benchmark_v2_FIXED.py -- _ProcBox (new) and _kill_process_group (pre-
# existing, now also called from the outer handler).
# ============================================================================

class _ProcBox:
    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None

    def set(self, proc) -> None:
        with self._lock:
            self._proc = proc

    def clear(self) -> None:
        with self._lock:
            self._proc = None

    def get(self):
        with self._lock:
            return self._proc


def _kill_process_group(proc: "subprocess.Popen") -> None:
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
        return
    except (ProcessLookupError, PermissionError, AttributeError, OSError):
        pass
    try:
        import psutil
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


# ============================================================================
# Stand-in for _run_pysr_in_subprocess: spawns a long-sleeping subprocess
# (instead of the PySR/Julia worker) with the SAME structure -- own process
# group, register into proc_box right after spawn, clear in finally.
# INNER_TIMEOUT stands in for pysr_timeout=900-1100s; here it's deliberately
# much larger than OUTER_TIMEOUT, reproducing the exact mismatch that caused
# the original bug.
# ============================================================================

def _fake_run_in_subprocess(inner_timeout: float, proc_box: "_ProcBox" = None, pidfile: str = None) -> dict:
    proc = subprocess.Popen(
        [sys.executable, "-c",
         f"import os,time; open({pidfile!r},'w').write(str(os.getpid())); time.sleep(9999)"],
        start_new_session=True,
    )
    if proc_box is not None:
        proc_box.set(proc)
    try:
        try:
            proc.wait(timeout=inner_timeout)
            return {"success": True}
        except subprocess.TimeoutExpired:
            _kill_process_group(proc)
            return {"success": False, "error": "inner timeout"}
    finally:
        if proc_box is not None:
            proc_box.clear()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ============================================================================
# Stand-in for the outer per-method timeout handler (the ThreadPoolExecutor
# + future.result(timeout=OUTER_TIMEOUT) + except TimeoutError block).
# ============================================================================

class _FakeMethod:
    def __init__(self):
        self._proc_box = None


def run_outer(outer_timeout: float, inner_timeout: float, use_fix: bool, pidfile: str):
    method = _FakeMethod()
    method._proc_box = _ProcBox() if use_fix else None

    pool = _cf.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_fake_run_in_subprocess, inner_timeout, method._proc_box, pidfile)

    t0 = time.time()
    try:
        future.result(timeout=outer_timeout)
        elapsed = time.time() - t0
        return elapsed
    except _cf.TimeoutError:
        elapsed = time.time() - t0
        # THE FIX: reach into proc_box and kill directly.
        proc_box = getattr(method, "_proc_box", None)
        if proc_box is not None:
            live_proc = proc_box.get()
            if live_proc is not None:
                _kill_process_group(live_proc)
        pool.shutdown(wait=False)
        return elapsed


def _wait_for_pidfile(pidfile: str, timeout: float = 5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(pidfile):
            try:
                return int(open(pidfile).read().strip())
            except (ValueError, OSError):
                pass
        time.sleep(0.05)
    return None


def main():
    OUTER_TIMEOUT = 3   # stands in for _METHOD_TIMEOUT_SECS (e.g. 300s)
    INNER_TIMEOUT = 60  # stands in for pysr_timeout (e.g. 900-1100s)
    import tempfile

    print("=" * 70)
    print("TEST 1: BEFORE fix (no proc_box) -- reproduces the original bug")
    print("=" * 70)
    pidfile1 = tempfile.mktemp()
    elapsed = run_outer(OUTER_TIMEOUT, INNER_TIMEOUT, use_fix=False, pidfile=pidfile1)
    pid = _wait_for_pidfile(pidfile1)
    time.sleep(0.5)  # let the OS process table settle
    alive_before = _pid_alive(pid) if pid else None
    print(f"  outer handler returned after {elapsed:.1f}s (outer_timeout={OUTER_TIMEOUT}s)")
    print(f"  spawned subprocess pid={pid}, still alive after outer return: {alive_before}")
    if alive_before:
        print("  -> CONFIRMED: subprocess orphaned, still running past the outer timeout.")
        os.kill(pid, signal.SIGKILL)  # manual cleanup, mirrors the bug report
        print("  -> (manually killed for cleanup, as the bug report describes)")
    print()

    print("=" * 70)
    print("TEST 2: AFTER fix (with proc_box) -- the wiring added this session")
    print("=" * 70)
    pidfile2 = tempfile.mktemp()
    elapsed = run_outer(OUTER_TIMEOUT, INNER_TIMEOUT, use_fix=True, pidfile=pidfile2)
    pid = _wait_for_pidfile(pidfile2)
    time.sleep(0.5)
    alive_after = _pid_alive(pid) if pid else None
    print(f"  outer handler returned after {elapsed:.1f}s (outer_timeout={OUTER_TIMEOUT}s)")
    print(f"  spawned subprocess pid={pid}, still alive after outer return: {alive_after}")
    if not alive_after:
        print("  -> CONFIRMED: subprocess killed within the outer timeout window.")
    else:
        print("  -> FAIL: subprocess still alive -- wiring did not work.")
        os.kill(pid, signal.SIGKILL)

    print()
    print("=" * 70)
    ok = (alive_before is True) and (alive_after is False)
    print("RESULT:", "PASS (bug reproduced without fix, closed with fix)" if ok else "FAIL")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
