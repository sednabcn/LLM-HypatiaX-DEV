"""
Verification harness for Issue 10b / Bug 1 (orphaned Julia process).

Since real Julia + PySR aren't available in this sandbox, this reproduces
the exact structural bug instead: a wrapper subprocess that spawns its own
OS-level child (standing in for "python worker spawns Julia"), then
compares:

  OLD behavior: subprocess.Popen(...) with no process-group isolation,
                killed via proc.kill() on timeout.
  NEW behavior: subprocess.Popen(..., start_new_session=True), killed via
                the harness's actual _kill_process_group() on timeout.

For each, we check whether the "fake julia" grandchild is still alive
after the timeout-triggered kill.
"""
import os
import signal
import subprocess
import sys
import time

PID_FILE = "/tmp/fake_julia.pid"


def _kill_process_group(proc: "subprocess.Popen") -> None:
    """Copied verbatim (logic-equivalent) from the FIXED harness file."""
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


def fake_julia_alive() -> bool:
    if not os.path.exists(PID_FILE):
        return False
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
    except (ValueError, OSError):
        return False
    # NOTE: os.kill(pid, 0) is NOT a reliable liveness check here -- a
    # SIGKILL'd process can remain a zombie (still present in the process
    # table, still answers kill(pid, 0)) until its new parent reaps it.
    # Check /proc state directly: gone entirely, or state 'Z' = dead.
    stat_path = f"/proc/{pid}/stat"
    if not os.path.exists(stat_path):
        return False
    try:
        with open(stat_path) as f:
            stat = f.read()
        state = stat.rsplit(")", 1)[-1].split()[0]
        return state not in ("Z",)
    except (OSError, IndexError):
        return False


def run_trial(label: str, isolate: bool, use_group_kill: bool, timeout: float = 2.0):
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)

    print(f"\n=== Trial: {label} ===")
    kwargs = {}
    if isolate:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        [sys.executable, "fake_julia_worker.py"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        **kwargs,
    )

    try:
        out, err = proc.communicate(timeout=timeout)
        print("(finished before timeout, unexpected)", out, err)
    except subprocess.TimeoutExpired:
        print(f"wrapper pid={proc.pid} timed out after {timeout}s as expected")
        # give the wrapper a moment to have actually spawned+registered its child
        time.sleep(0.3)
        pre_kill_alive = fake_julia_alive()
        print(f"fake-julia alive before kill: {pre_kill_alive}")

        if use_group_kill:
            _kill_process_group(proc)
        else:
            proc.kill()  # OLD behavior: only the direct child

        proc.wait(timeout=5)

    time.sleep(1.5)  # let OS process table settle / reap zombies
    post_kill_alive = fake_julia_alive()
    print(f"fake-julia alive after kill:  {post_kill_alive}")
    return post_kill_alive


if __name__ == "__main__":
    old_result = run_trial(
        "OLD (no isolation, proc.kill() only)",
        isolate=False, use_group_kill=False,
    )
    new_result = run_trial(
        "NEW (start_new_session + _kill_process_group)",
        isolate=True, use_group_kill=True,
    )

    print("\n=== Summary ===")
    print(f"OLD path leaves fake-julia orphan alive: {old_result}  (expected True = BUG reproduced)")
    print(f"NEW path leaves fake-julia orphan alive: {new_result}  (expected False = FIX confirmed)")

    if old_result and not new_result:
        print("\nRESULT: Bug 1 mechanism reproduced under OLD code; FIXED code kills the orphan. PASS.")
        sys.exit(0)
    else:
        print("\nRESULT: unexpected outcome, needs investigation.")
        sys.exit(1)
