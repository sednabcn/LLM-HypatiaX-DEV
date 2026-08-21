# Patch Report: Issue 10b — Bug 2 (`_kill_thread` Dead Code) Removed

**File patched:** `run_comparative_suite_benchmark_v2_FIXED.py`
**Related to:** Issue 10b (300s timeout enforcement). Bug 1 (orphaned Julia
process) was already fixed in the uploaded file. This report covers Bug 2
only.
**Status:** Patched and verified.

## What was checked

Verified the uploaded `run_comparative_suite_benchmark_v2.py` against both
bugs identified in `observ-10.txt`:

- **Bug 1 (orphaned Julia process) — already fixed.** `_run_pysr_in_subprocess`
  spawns with `start_new_session=True` (line 3008) and, on
  `TimeoutExpired`, calls `_kill_process_group(proc)` (line 3013), which
  `SIGKILL`s the whole process group via `os.killpg`, falling back to
  `psutil` descendant-enumeration if unavailable. No change needed here.
- **Bug 2 (`_kill_thread` dead code) — still present.** Line 3581 called
  `_kill_thread(_worker_thread.ident)`. An AST parse of the file confirmed
  no `def _kill_thread` exists anywhere. `ctypes` was still imported at
  line 63 solely to support this never-defined function. The call sat
  inside a `try` block wrapped in `except Exception: pass`, so every
  timeout silently raised `NameError` here with no visible symptom.

## Fix applied

### 1. Removed the dead `ctypes` import (line ~63)

**Before:**
```python
import concurrent.futures as _cf
import ctypes          # for _kill_thread (hard timeout enforcement)
import threading as _threading
```

**After:** `ctypes` import removed, replaced with an explanatory comment
tagged `FIX-ISSUE10B-DEAD-KILLTHREAD`.

### 2. Removed the dead `_kill_thread` call and its no-op scaffolding (line ~3568–3583)

**Before:**
```python
if verbose:
    print(f"⏱ timeout ({_METHOD_TIMEOUT_SECS}s)", end="", flush=True)
# Inject SystemExit into the background thread so it stops
# consuming API quota.  _kill_thread returns False silently
# if the thread already exited (race condition is harmless).
for _t in _threading.enumerate():
    if _t.ident and not _t.daemon and _t is not _threading.main_thread():
        pass  # only kill daemon threads spawned by our pool
# ThreadPoolExecutor worker threads ARE daemon threads —
# find them by checking the running future's thread reference
# via the pool's internal _threads set.
try:
    for _worker_thread in list(_pool._threads):
        if _worker_thread.is_alive():
            _killed = _kill_thread(_worker_thread.ident)
            if verbose:
                print(
                    f" [thread {'killed' if _killed else 'already exited'}]",
                    end="", flush=True
                )
except Exception:
    pass  # ctypes injection is best-effort; never crash the suite
```

**After:**
```python
if verbose:
    print(f"⏱ timeout ({_METHOD_TIMEOUT_SECS}s)", end="", flush=True)
# FIX-ISSUE10B-DEAD-KILLTHREAD: this block previously called
# _kill_thread(_worker_thread.ident) to inject SystemExit into
# the background thread. _kill_thread was never defined
# anywhere in this file, so every timeout silently raised
# NameError here (caught by a blanket `except Exception: pass`)
# and no kill of any kind ever happened -- the "[thread
# killed]" verbose message was never reachable in practice.
# Removed rather than implemented: the background thread is
# merely blocked on the (already-fixed) PySR subprocess call;
# it is not itself the resource leak. The thread is a daemon
# thread (ThreadPoolExecutor default), so it will not block
# process exit even if it runs to completion. The actual
# runaway resource -- an orphaned Julia OS process -- is
# handled by _kill_process_group() inside
# _run_pysr_in_subprocess(), independent of this thread.
```

This matches the original recommendation in `observ-10.txt`: *"either
implement `_kill_thread` ... or, better, delete this dead code path and
rely on Bug 1's fix — since the real problem is the orphaned OS process,
not the Python thread."* Deletion was chosen because killing the thread
would not have touched the Julia process either way — the thread is only
blocked on `communicate()`, not doing the hanging itself.

The `_timed_out` flag, the `MethodResult` timeout record, and the verbose
`⏱ timeout (...)` message are all preserved unchanged — only the
unreachable-in-practice kill attempt and its printed confirmation were
removed.

## Verification performed

- `python3 -m py_compile run_comparative_suite_benchmark_v2_FIXED.py` →
  **passes**, no syntax errors introduced.
- `grep -n "_kill_thread"` across the whole file → all remaining matches
  are inside comments explaining the removal; **zero live calls**.
- `grep -n "^import ctypes"` → **zero matches**; the dead import is gone.
- Confirmed Bug 1's fix (`_kill_process_group`, `start_new_session=True`)
  is untouched and still present (5 matches, as expected).

## Scope / what this does *not* cover

- Does not touch Bug 1 (already fixed prior to this patch).
- Does not re-run the suite. The 27,574s Arrhenius hang was already
  attributed to Bug 1 (orphaned Julia process); Bug 2 was a dead fallback
  that never fired and is not believed to have contributed to that
  specific hang. Verification of Bug 1's real-world effectiveness (test 4
  re-run, checking `ps -ef --forest` for orphaned `julia` after a forced
  short timeout) is still pending, as noted in the earlier report.

## Impact

This is a **no-behavior-change cleanup**, not a fix that alters any
result. The removed code never successfully executed (it always raised
`NameError`, always caught silently), so no run's numbers depended on it
in any way. This patch only removes misleading dead code and a stale
comment claiming behavior that never happened.
