#!/usr/bin/env python3
"""
run_exp2_symbolic_engine.py
===========================
Standalone runner for Method 5 (SymbolicEngineWithLLM) across all 30
experiment-protocol-all-30 equations.

OOM-safe design
---------------
Each equation runs in its own isolated subprocess.  The parent process never
imports juliacall or any Julia-touching code, so Julia's JIT heap (2-3 GB) is
fully released between equations instead of accumulating in the parent.

The worker also installs a SIGALRM self-destruct timer and calls os.setsid()
so the parent can kill the entire Julia process group atomically on timeout,
preventing Julia threads from lingering after the nominal PySR deadline.

Worker communication
--------------------
  parent → worker  :  JSON blob written to a temp file (X, y, metadata)
  worker → parent  :  JSON result written to a second temp file; stdout
                       is streamed live so PySR progress is visible

Writes per-equation results to:
    logs/exp2_symbolic_engine_checkpoint.json

That file is later consumed by run_comparative_suite_benchmark_injected.py
which injects the method-5 column into the full comparison table without
re-running PySR.

Usage
-----
    python3 run_exp2_symbolic_engine.py [--resume] [--pysr-timeout 1100]
    python3 run_exp2_symbolic_engine.py --resume --one-equation Kinetic
    python3 run_exp2_symbolic_engine.py --resume --pysr-timeout 1100 --ram-limit-gb 12

Checkpoint schema
-----------------
{
  "version": 1,
  "method":  "SymbolicEngineWithLLM (tools)",
  "completed": ["mechanics::Kinetic Energy", ...],
  "results": {
    "<eq_key>": {
        "description": "...",
        "domain":      "...",
        "elapsed_s":   123.4,
        "result":      { <MethodResult.to_dict()> }   // or null on skip
    },
    ...
  }
}
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap  (parent process only — no Julia / PySR imports here)
# ---------------------------------------------------------------------------
_HERE      = Path(__file__).resolve().parent   # …/hypatiax/experiments/benchmarks/
_PKG_ROOT  = _HERE.parent.parent               # …/hypatiax/
_REPO_ROOT = _PKG_ROOT.parent                  # repo root
sys.path.insert(0, str(_REPO_ROOT))

_CHECKPOINT_PATH = _REPO_ROOT / "logs" / "exp2_symbolic_engine_checkpoint.json"
_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Import ONLY the protocol + the timeout constant.
# Do NOT import _bench_v2 here — that pulls in PySR / juliacall.
from hypatiax.protocols.experiment_protocol_all_30 import ExperimentProtocolAll

# Extract the timeout constant from source without executing the bench module,
# which would trigger Julia-touching import-time side-effects.
try:
    import re as _re
    _BENCH_PATH = _HERE / "run_comparative_suite_benchmark_v2.py"
    _src = _BENCH_PATH.read_text()
    _m = _re.search(r"_METHOD_TIMEOUT_SECS\s*=\s*(\d+)", _src)
    _METHOD_TIMEOUT_SECS: int = int(_m.group(1)) if _m else 1200
    del _src, _m, _re
except Exception:
    _METHOD_TIMEOUT_SECS = 1200

_PASS_THRESHOLD = 9


# ---------------------------------------------------------------------------
# Worker script
# ---------------------------------------------------------------------------
# Written to a temp .py file and executed in a fresh subprocess per equation.
# Importing juliacall + SymbolicEngineMethod here means Julia's heap is fully
# released to the OS when the subprocess exits.

_WORKER_SCRIPT = textwrap.dedent(r"""
import gc, json, os, resource, signal, sys, time
from pathlib import Path

# NOTE: os.setsid() is intentionally absent here.  The parent spawns this
# worker with start_new_session=True, which already makes this process a
# session and process-group leader.  Calling setsid() again would raise
# PermissionError (EPERM) because a session leader cannot create a new session.

# ----- receive args ---------------------------------------------------------
if len(sys.argv) < 3:
    sys.exit("worker: expected <input_json> <output_json>")

input_path  = Path(sys.argv[1])
output_path = Path(sys.argv[2])
payload     = json.loads(input_path.read_text())

description  = payload["description"]
var_names    = payload["var_names"]
meta         = payload["meta"]
domain       = payload["domain"]
pysr_timeout = payload.get("pysr_timeout", 1100)
proc_timeout = payload.get("proc_timeout", pysr_timeout + 120)
ram_limit_gb = payload.get("ram_limit_gb", 0)   # 0 = no limit
repo_root    = payload["repo_root"]

sys.path.insert(0, repo_root)

# ----- cap address space if caller requested it -----------------------------
if ram_limit_gb > 0:
    limit_bytes = int(ram_limit_gb * 1024 ** 3)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except Exception as _e:
        print(f"[worker] RLIMIT_AS not set: {_e}", flush=True)

# ----- self-destruct timer: SIGKILL this process group after proc_timeout ---
def _self_kill(signum, frame):
    print(f"[worker] hard timeout {proc_timeout}s reached — killing process group",
          flush=True)
    os.killpg(os.getpgid(0), signal.SIGKILL)

signal.signal(signal.SIGALRM, _self_kill)
signal.alarm(proc_timeout)

# ----- cap Julia heap via env vars (must be set BEFORE juliacall import) ----
# JULIA_GC_THRESHOLD: fraction above live set before GC runs (default 2.0).
# 0.4 forces more frequent collection, reducing peak RSS.
os.environ.setdefault("JULIA_GC_THRESHOLD", "0.4")
# Disable per-thread GC arenas that pre-allocate memory at startup.
os.environ.setdefault("JULIA_NUM_THREADS",  "1")
os.environ.setdefault("JULIA_CPU_THREADS",  "1")

# ----- juliacall MUST come before torch -------------------------------------
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
try:
    import juliacall as _jc  # noqa: F401
except Exception:
    pass

# ----- load X/y from the serialised arrays ----------------------------------
import numpy as np
X = np.array(payload["X"])
y = np.array(payload["y"])

# ----- load the method ------------------------------------------------------
import importlib.util as _ilu
bench_path = payload["bench_path"]
_spec  = _ilu.spec_from_file_location("_bench_v2", bench_path)
_bench = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_bench)

_bench._PYSR_TIMEOUT = pysr_timeout
method = _bench.SymbolicEngineMethod(verbose=True)

# ----- run ------------------------------------------------------------------
t0     = time.time()
result = method.run(description, X, y, var_names, meta, verbose=True)
elapsed = time.time() - t0

signal.alarm(0)   # cancel self-destruct — finished cleanly
del X, y
gc.collect()

# ----- write result ---------------------------------------------------------
output_path.write_text(json.dumps({
    "elapsed_s": round(elapsed, 2),
    "result":    result.to_dict(),
}, default=str))
""")


# ---------------------------------------------------------------------------
# Partial-results scoreboard
# ---------------------------------------------------------------------------

def _print_partial_results(
    completed_rows: list[dict],
    total: int,
    suite_start: float,
) -> None:
    done    = len(completed_rows)
    solved  = sum(1 for r in completed_rows if r["status"] == "✅")
    failed  = sum(1 for r in completed_rows if r["status"] in ("❌", "⏱", "⚠️"))
    skipped = sum(1 for r in completed_rows if r["status"] == "↩")

    wall      = time.time() - suite_start
    remaining = total - done
    if done > 0 and remaining > 0:
        eta_s = int(wall / done * remaining)
        h, rem = divmod(eta_s, 3600)
        m, s   = divmod(rem, 60)
        eta_str = f"ETA ≈ {h:02d}:{m:02d}:{s:02d}"
    else:
        eta_str = "ETA ≈ —"

    on_track = (
        "✅ on track"
        if (done == 0 or solved / done >= _PASS_THRESHOLD / total)
        else "⚠️  behind"
    )

    BAR  = "─" * 72
    DBAR = "━" * 72
    print(f"\n{DBAR}", flush=True)
    print(
        f"  PARTIAL RESULTS  [{done}/{total} done]  "
        f"{solved} solved  {failed} failed  {skipped} resumed  "
        f"{eta_str}",
        flush=True,
    )
    print(DBAR, flush=True)
    print(
        f"  {'#':<4} {'Equation':<26} {'Domain':<12} {'R²':>8}  {'Time':>6}  Status",
        flush=True,
    )
    print(f"  {BAR}", flush=True)
    for idx, row in enumerate(completed_rows, 1):
        r2_str = f"{row['r2']:.4f}" if row.get("r2") is not None else "  —   "
        t_str  = f"{int(row.get('elapsed_s', 0))}s" if row.get("elapsed_s") else "  —"
        print(
            f"  {idx:<4} {row['eq_name'][:26]:<26} {row['domain'][:12]:<12} "
            f"{r2_str:>8}  {t_str:>6}  {row['status']}",
            flush=True,
        )
    print(f"  {BAR}", flush=True)
    pct = f"{solved / total * 100:.1f}%" if total else "—"
    print(
        f"  Threshold: {_PASS_THRESHOLD}/{total}  |  "
        f"Current: {solved}/{total} ({pct})  |  {on_track}",
        flush=True,
    )
    print(f"{DBAR}\n", flush=True)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint() -> dict:
    if _CHECKPOINT_PATH.exists():
        try:
            with open(_CHECKPOINT_PATH) as f:
                return json.load(f)
        except Exception as exc:
            print(f"⚠️  Could not read checkpoint: {exc}", flush=True)
    return {
        "version":   1,
        "method":    "SymbolicEngineWithLLM (tools)",
        "completed": [],
        "results":   {},
    }


def _save_checkpoint(state: dict) -> None:
    tmp = _CHECKPOINT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, _CHECKPOINT_PATH)


def _eq_key(meta: dict, domain: str) -> str:
    return f"{domain}::{meta.get('equation_name', meta.get('name', str(meta)))}"


# ---------------------------------------------------------------------------
# Results branch push helper
# ---------------------------------------------------------------------------
# It pushes a single file to the `results` branch after every equation save.
# Works even if the job is SIGKILL'd — push happens inline, not in a post step.

_GIT_RESULTS_BRANCH = "results"

def _push_result_to_branch(local_file_path: str, repo_relative_path: str) -> None:
    """
    Push local_file_path to repo_relative_path on the `results` branch.
    Silently skips when not running in GitHub Actions or git is unavailable.
    """
    import os, subprocess, tempfile, shutil

    if not os.environ.get("GITHUB_ACTIONS"):
        return

    token = os.environ.get("GITHUB_TOKEN", "")
    repo  = os.environ.get("GITHUB_REPOSITORY", "")
    if not token or not repo:
        print("  ⚠  GITHUB_TOKEN or GITHUB_REPOSITORY not set — skipping branch push", flush=True)
        return

    if not os.path.exists(local_file_path):
        return

    try:
        # Use a throwaway clone in /tmp so we don't dirty the working tree
        clone_dir = tempfile.mkdtemp(prefix="results_push_")
        remote    = f"https://x-access-token:{token}@github.com/{repo}.git"

        # Shallow clone of the results branch (create if absent)
        result = subprocess.run(
            ["git", "clone", "--depth=1", "--branch", _GIT_RESULTS_BRANCH,
             "--single-branch", remote, clone_dir],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            # Branch doesn't exist yet — init an orphan
            subprocess.run(["git", "init", clone_dir], capture_output=True, timeout=30)
            subprocess.run(
                ["git", "-C", clone_dir, "checkout", "--orphan", _GIT_RESULTS_BRANCH],
                capture_output=True, timeout=30
            )
            subprocess.run(
                ["git", "-C", clone_dir, "remote", "add", "origin", remote],
                capture_output=True, timeout=30
            )

        # Copy the file into the clone, preserving subdirectory structure
        dest = os.path.join(clone_dir, repo_relative_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(local_file_path, dest)

        env = {**os.environ,
               "GIT_AUTHOR_NAME":     "github-actions[bot]",
               "GIT_AUTHOR_EMAIL":    "github-actions[bot]@users.noreply.github.com",
               "GIT_COMMITTER_NAME":  "github-actions[bot]",
               "GIT_COMMITTER_EMAIL": "github-actions[bot]@users.noreply.github.com"}

        subprocess.run(["git", "-C", clone_dir, "add", repo_relative_path],
                       capture_output=True, timeout=30)
        subprocess.run(
            ["git", "-C", clone_dir, "commit", "--allow-empty",
             "-m", f"ci: checkpoint {os.path.basename(local_file_path)}"],
            capture_output=True, env=env, timeout=30
        )
        push = subprocess.run(
            ["git", "-C", clone_dir, "push", "origin", _GIT_RESULTS_BRANCH],
            capture_output=True, text=True, timeout=60
        )
        if push.returncode == 0:
            print(f"  🌿 Pushed checkpoint → branch '{_GIT_RESULTS_BRANCH}':{repo_relative_path}", flush=True)
        else:
            print(f"  ⚠  Branch push failed: {push.stderr.strip()[:120]}", flush=True)

    except Exception as exc:
        print(f"  ⚠  Branch push exception: {exc}", flush=True)
    finally:
        try:
            shutil.rmtree(clone_dir, ignore_errors=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Per-equation subprocess runner
# ---------------------------------------------------------------------------

def _run_one_equation(
    description: str,
    X,
    y,
    var_names: list[str],
    meta: dict,
    domain: str,
    pysr_timeout: int,
    ram_limit_gb: float = 0,
) -> tuple[dict | None, float]:
    """
    Spawn an isolated subprocess for a single equation.

    The worker runs in its own process group (os.setsid) and installs a
    SIGALRM self-destruct timer.  The parent also sends SIGKILL to the entire
    process group if the wall-clock deadline is exceeded, so Julia threads
    cannot linger after a timeout.

    ram_limit_gb: if > 0, sets RLIMIT_AS inside the worker to cap peak RSS.
                  Recommended: 75% of available RAM (e.g. 12 on a 16 GB machine).

    Returns (result_dict_or_None, elapsed_seconds).
    result_dict mirrors MethodResult.to_dict(); None means the worker crashed.
    """
    import signal

    # The worker's SIGALRM fires at proc_timeout; the parent kills 30 s later
    # as a backstop if SIGALRM somehow fails to terminate Julia.
    proc_timeout   = pysr_timeout + 120
    parent_timeout = proc_timeout + 30

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        inp_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        out_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",   delete=False) as f:
        worker_path = f.name
        f.write(_WORKER_SCRIPT)

    payload = {
        "description":  description,
        "var_names":    var_names,
        "meta":         meta,
        "domain":       domain,
        "pysr_timeout": pysr_timeout,
        "proc_timeout": proc_timeout,
        "ram_limit_gb": ram_limit_gb,
        "repo_root":    str(_REPO_ROOT),
        "bench_path":   str(_BENCH_PATH),
        "X":            X.tolist() if hasattr(X, "tolist") else list(X),
        "y":            y.tolist() if hasattr(y, "tolist") else list(y),
    }
    with open(inp_path, "w") as f:
        json.dump(payload, f)

    t0   = time.time()
    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, worker_path, inp_path, out_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            # start_new_session prevents SIGINT from the parent terminal
            # reaching the worker; the worker calls os.setsid() itself for
            # killpg() to work correctly.
            start_new_session=True,
        )
        for line in proc.stdout:          # type: ignore[union-attr]
            print("  │ " + line, end="", flush=True)
        proc.wait(timeout=parent_timeout)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"\n  ⚠️  Worker exited with code {proc.returncode}", flush=True)
            return None, elapsed

        out_text = Path(out_path).read_text().strip()
        if not out_text:
            print("\n  ⚠️  Worker produced no output JSON", flush=True)
            return None, elapsed

        worker_result = json.loads(out_text)
        return worker_result["result"], worker_result["elapsed_s"]

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait()
        print(
            f"\n  ⏱  Worker process group killed after {elapsed:.0f}s "
            f"(parent_timeout={parent_timeout}s)",
            flush=True,
        )
        return None, elapsed

    except Exception as exc:
        elapsed = time.time() - t0
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait()
        print(f"\n  ⚠️  Worker exception: {exc}", flush=True)
        return None, elapsed

    finally:
        for p in (inp_path, out_path, worker_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exp-2 Method-5 standalone runner (SymbolicEngineWithLLM)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip equations already in the checkpoint",
    )
    parser.add_argument(
        "--pysr-timeout",
        type=int,
        default=int(os.environ.get("PYSR_TIMEOUT", 1100)),
        dest="pysr_timeout",
        help="PySR subprocess timeout per equation (default: 1100s)",
    )
    parser.add_argument(
        "--samples", type=int, default=200,
        help="Samples per equation (default: 200)",
    )
    parser.add_argument(
        "--one-equation", dest="one_equation", default=None,
        help="Run only the equation whose description contains this string",
    )
    parser.add_argument(
        "--ram-limit-gb", dest="ram_limit_gb", type=float, default=0,
        help=(
            "Cap each worker's address space (RLIMIT_AS) to this many GB. "
            "Prevents the OOM killer from targeting a worker that overallocates. "
            "Recommended: 75%% of available RAM (e.g. 12 on a 16 GB machine). "
            "0 = no limit (default)."
        ),
    )
    args = parser.parse_args()

    protocol = ExperimentProtocolAll()
    state    = (
        _load_checkpoint()
        if args.resume
        else {
            "version":   1,
            "method":    "SymbolicEngineWithLLM (tools)",
            "completed": [],
            "results":   {},
        }
    )

    completed: set[str] = set(state.get("completed", []))
    results:   dict     = state.get("results", {})

    # Guard: if resume=True but checkpoint already covers all equations,
    # reset so this run is not a silent no-op.
    if args.resume and completed:
        protocol_check = ExperimentProtocolAll()
        total_check = sum(
            1 for dom in protocol_check.get_all_domains()
            for _ in protocol_check.load_test_data(dom, num_samples=1)
        )
        if len(completed) >= total_check:
            print(
                f"⚠️  Checkpoint marks {len(completed)}/{total_check} equations complete "
                f"— all done. If you want to re-run, set resume=false.",
                flush=True,
            )
            return 0

    # ── Collect all 30 test stubs (metadata only — no X/y yet) ───────────────
    all_tests: list[tuple] = []
    for domain in protocol.get_all_domains():
        for (description, X, y, var_names, meta) in protocol.load_test_data(
            domain, num_samples=args.samples
        ):
            all_tests.append((description, var_names, meta, domain))
            del X, y
    gc.collect()

    if args.one_equation:
        all_tests = [
            t for t in all_tests
            if args.one_equation.lower() in t[0].lower()
        ]
        if not all_tests:
            print(f"❌  No equation matching '{args.one_equation}'", flush=True)
            return 1

    total = len(all_tests)
    print(f"\n🔬  SymbolicEngineWithLLM — {total} equations", flush=True)
    if args.resume and completed:
        print(
            f"♻️  Resuming — {len(completed)} already done, "
            f"{total - len(completed)} remaining.",
            flush=True,
        )
    print(f"📋  Checkpoint → {_CHECKPOINT_PATH}\n", flush=True)
    print(
        "🧠  Memory strategy: each equation runs in an isolated subprocess.\n"
        "    Julia's JIT heap is fully released between equations.\n",
        flush=True,
    )

    suite_start     = time.time()
    solved          = 0
    completed_rows: list[dict] = []

    # Pre-populate scoreboard with already-completed equations
    for key in list(completed):
        r   = results.get(key, {})
        res = r.get("result") or {}
        r2  = res.get("r2") if res.get("success") else None
        if res.get("success"):
            solved += 1
        completed_rows.append({
            "eq_name":   r.get("description", key.split("::", 1)[-1])[:26],
            "domain":    r.get("domain", "—"),
            "r2":        r2,
            "elapsed_s": r.get("elapsed_s"),
            "status":    "↩",
        })

    for i, (description, var_names, meta, domain) in enumerate(all_tests, 1):
        key = _eq_key(meta, domain)

        # ── Skip already-completed ─────────────────────────────────────────
        if key in completed:
            print(
                f"  ⏭️  SKIP {i}/{total}: {meta.get('equation_name', description)}",
                flush=True,
            )
            continue

        print(f"\n{'='*72}", flush=True)
        print(f"  [{i}/{total}]  {description}", flush=True)
        print(f"  Domain: {domain}  |  vars: {var_names}", flush=True)
        print(f"{'='*72}", flush=True)

        # ── Reload X/y for this equation ───────────────────────────────────
        loaded = protocol.load_test_data(domain, num_samples=args.samples)
        match  = next(
            (
                c for c in loaded
                if (
                    c[4].get("equation_name", c[0])
                    == meta.get("equation_name", description)
                    or c[0] == description
                )
            ),
            None,
        )

        if match is None:
            print("  ⚠️  Could not reload X/y — skipping", flush=True)
            results[key] = {
                "description": description,
                "domain":      domain,
                "result":      None,
            }
            completed.add(key)
            state["completed"] = list(completed)
            state["results"]   = results
            _save_checkpoint(state)
            _push_result_to_branch(str(_CHECKPOINT_PATH), "logs/exp2_symbolic_engine_checkpoint.json")
            completed_rows.append({
                "eq_name":   meta.get("equation_name", description)[:26],
                "domain":    domain,
                "r2":        None,
                "elapsed_s": None,
                "status":    "⚠️",
            })
            _print_partial_results(completed_rows, total, suite_start)
            continue

        _, X, y, _, _ = match
        del match, loaded

        # ── Isolated subprocess run ────────────────────────────────────────
        r_dict, elapsed = _run_one_equation(
            description, X, y, var_names, meta, domain,
            args.pysr_timeout, ram_limit_gb=args.ram_limit_gb,
        )
        del X, y
        gc.collect()

        # ── Parse result ───────────────────────────────────────────────────
        success   = bool(r_dict and r_dict.get("success"))
        r2_val    = r_dict.get("r2") if success else None
        error_msg = (r_dict or {}).get("error", "worker crash")

        results[key] = {
            "description": description,
            "domain":      domain,
            "elapsed_s":   round(elapsed, 2),
            "result":      r_dict,
        }
        completed.add(key)
        state["completed"]  = list(completed)
        state["results"]    = results
        state["timestamp"]  = datetime.now().isoformat()
        _save_checkpoint(state)
        _push_result_to_branch(str(_CHECKPOINT_PATH), "logs/exp2_symbolic_engine_checkpoint.json")

        if success:
            solved += 1
            status_str = f"R²={r2_val:.4f}"
        else:
            status_str = f"✗ {error_msg or 'failed'}"
        print(
            f"\n  ✔  {meta.get('equation_name', description)} → "
            f"{status_str}  ({elapsed:.0f}s)",
            flush=True,
        )

        # ── Scoreboard ─────────────────────────────────────────────────────
        timed_out  = elapsed >= _METHOD_TIMEOUT_SECS - 5
        row_status = "✅" if success else ("⏱" if timed_out else "❌")
        completed_rows.append({
            "eq_name":   meta.get("equation_name", description)[:26],
            "domain":    domain,
            "r2":        r2_val,
            "elapsed_s": round(elapsed, 1),
            "status":    row_status,
        })
        _print_partial_results(completed_rows, total, suite_start)

    wall = time.time() - suite_start
    h, rem = divmod(int(wall), 3600)
    m, s   = divmod(rem, 60)
    print(f"\n{'='*72}", flush=True)
    print(
        f"  DONE — {solved}/{total} solved  |  wall time {h:02d}:{m:02d}:{s:02d}",
        flush=True,
    )
    print(f"  Checkpoint → {_CHECKPOINT_PATH}", flush=True)

    return 0 if solved >= _PASS_THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(main())
