# Bug 1 Verification Note — Process-Group Kill Mechanism

## What I could not do

The real "re-run Arrhenius (test 4) with a short timeout" step from the
report couldn't be run as literally specified: this sandbox has no Julia
and no PySR installed (`pip show pysr` → not found; `julia --version` →
not found), and PySR's own installer pulls Julia from `julialang.org`,
which isn't on the environment's allowed network list. So I couldn't
exercise the actual harness end-to-end against the actual equation.

## What I did instead

Bug 1's root cause isn't specific to Julia — it's a generic
subprocess-of-subprocess orphaning pattern: a wrapper process spawns its
own OS child, and killing only the wrapper leaves the child running. That
pattern is fully reproducible without Julia or PySR at all, so I built a
minimal stand-in and tested the exact kill logic from the fixed harness
against it.

**Files (in outputs):**
- `fake_julia_worker.py` — plays the role of the PySR subprocess worker:
  spawns its own child process (standing in for Julia), writes that
  child's PID to a file, then blocks on it — structurally identical to
  the real worker blocking on Julia.
- `verify_kill_behavior.py` — runs two trials:
  1. **OLD path**: `subprocess.Popen(...)` with no `start_new_session`,
     killed via `proc.kill()` on timeout (pre-fix behavior).
  2. **NEW path**: `subprocess.Popen(..., start_new_session=True)`,
     killed via `_kill_process_group(proc)` — the actual function copied
     from the fixed harness — on timeout (post-fix behavior).

  For each, it checks whether the "fake julia" child is still running
  after the kill, using `/proc/<pid>/stat` state rather than
  `os.kill(pid, 0)` (the latter is unreliable here — a `SIGKILL`'d
  process can sit as a zombie, still visible to a signal-0 check, until
  its new parent reaps it; I hit this as a false negative on the first
  run and switched to checking the actual process state).

## Result

```
=== Trial: OLD (no isolation, proc.kill() only) ===
fake-julia alive before kill: True
fake-julia alive after kill:  True      <- orphan survives (BUG reproduced)

=== Trial: NEW (start_new_session + _kill_process_group) ===
fake-julia alive before kill: True
fake-julia alive after kill:  False     <- orphan killed (FIX confirmed)
```

This confirms the *mechanism* the fix relies on: process-group isolation
plus a group-wide `SIGKILL` reaches a grandchild process that
`proc.kill()` alone cannot. It does not confirm that Julia specifically
behaves like this stand-in (e.g. if PySR loads `libjulia` in-process
rather than shelling out to a separate `julia` binary, Bug 1's premise
wouldn't apply at all — that assumption was never independently verified
against PySR's actual internals).

## Still outstanding

To fully close Bug 1 verification, someone with Julia/PySR available
needs to:
1. Run `verify_kill_behavior.py`-style check, or directly re-run
   Arrhenius (test 4) with a short `--method-timeout`.
2. Watch `ps -ef --forest` (or `psutil` children) for a `julia` process
   after the wrapper is killed.
3. Confirm it's gone within the timeout window, not just eventually
   reaped.
