"""
core/runners/common.py
Shared utilities for experiment runners.

Provides two layers of functionality:

  1. Low-level subprocess helpers (ensure_dir, run, python)
     Thin, reusable wrappers used by any runner that needs to launch
     sub-processes with a merged environment.

  2. High-level task runner (SCRIPT_MAP, run_task)
     Maps protocol config names to concrete benchmark scripts and
     launches them with optional FAST-mode timeout injection.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import subprocess
import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOW-LEVEL SUBPROCESS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if it does not exist; return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def run(
    cmd: list[str],
    env: dict | None = None,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """
    Thin wrapper around subprocess.run that merges *env* with os.environ.

    Parameters
    ----------
    cmd     : Command and arguments to execute.
    env     : Extra environment variables merged on top of os.environ.
    cwd     : Working directory for the subprocess (default: inherit).
    check   : Raise CalledProcessError on non-zero exit when True.
    capture : Capture stdout/stderr when True (default).
              Pass False to stream output directly to the terminal —
              useful for long-running benchmark scripts.
    """
    merged = {**os.environ, **(env or {})}
    return subprocess.run(
        cmd,
        env=merged,
        cwd=cwd,
        capture_output=capture,
        text=True,
        check=check,
    )


def python() -> str:
    """Return the path to the current Python interpreter."""
    return sys.executable


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EXPERIMENT TASK RUNNER
# ══════════════════════════════════════════════════════════════════════════════

# Map protocol config names → benchmark script paths.
SCRIPT_MAP: dict[str, str] = {
    "nguyen12_exp3":        "hypatiax/experiments/benchmarks/exp3_nguyen12_hybrid50v_02.py",
    "nguyen12_exp3b":       "hypatiax/experiments/benchmarks/exp3_nguyen12_hybrid50v_02_seed_123.py",
    "ablation_exp1":        "hypatiax/experiments/benchmarks/exp1_ablation_populations_30_updated.py",
    "defi_v3":              "hypatiax/experiments/benchmarks/hypatiax_defi_benchmark_v3c.py",
    "feynman_exp2":         "hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_v2.py",
    "noise_sweep":          "hypatiax/experiments/benchmarks/run_noise_sweep_benchmark.py",
    "hybrid_routing":       "hypatiax/experiments/benchmarks/run_hybrid_system_benchmark.py",
    "instability_rf02_04":  "hypatiax/experiments/benchmarks/run_dual_sweep_benchmarks.py",
    "extrap_comparative":   "hypatiax/experiments/benchmarks/run_dual_condition_benchmark.py",
    "provenance_audit":     "scripts/patches/provenance_audit.py",
}

# Scripts that accept --pysr-timeout / --method-timeout argparse flags.
_TIMEOUT_AWARE_SCRIPTS: frozenset[str] = frozenset({
    "run_dual_sweep",
    "run_noise_sweep",
    "run_dual_condition",
    "run_hybrid_system",
    # hypatiax_defi_benchmark_v3c does NOT accept --pysr-timeout / --method-timeout
    # (it errors with "unrecognized arguments"). Removed from this set so
    # _timeout_args() returns [] for the defi_v3 task.
})


def _timeout_args(script_name: str) -> list[str]:
    """
    Return FAST-mode timeout flags if *script_name* supports them,
    reading PYSR_TIMEOUT and METHOD_TIMEOUT from the environment.
    Returns an empty list if the script does not support these flags.
    """
    if not any(s in script_name for s in _TIMEOUT_AWARE_SCRIPTS):
        return []
    pysr_timeout   = os.environ.get("PYSR_TIMEOUT",   "1100")
    method_timeout = os.environ.get("METHOD_TIMEOUT",  "900")
    return ["--pysr-timeout", pysr_timeout, "--method-timeout", method_timeout]


def run_task(config: dict) -> dict:
    """
    Look up and execute the benchmark script named by *config["name"]*.

    Extra CLI arguments may be passed via *config["args"]* (a list of
    strings).  Scripts that support FAST-mode timeout flags receive them
    automatically from the PYSR_TIMEOUT / METHOD_TIMEOUT env vars.

    Returns a dict with keys: status, name, script, returncode (where
    applicable).
    """
    name   = config.get("name", "")
    script = SCRIPT_MAP.get(name)

    if not script:
        print(f"  ⚠  run_task: no script mapped for config name '{name}' — skipping")
        return {"status": "no_script", "name": name}

    script_path = Path(script)
    if not script_path.exists():
        print(f"  ⚠  run_task: script not found: {script}")
        return {"status": "missing_script", "name": name, "script": script}

    extra_args = config.get("args", [])
    # no_timeout_flags=True is set by universal_protocol when DEFI_V3C_NO_TIMEOUT_FLAGS
    # is in the environment — honour it as a belt-and-suspenders guard.
    fast_args  = [] if config.get("no_timeout_flags") else _timeout_args(str(script_path))
    cmd        = [python(), str(script_path)] + fast_args + extra_args

    print(f"  → running: {' '.join(cmd)}")

    # Stream output directly to the terminal (capture=False) so progress
    # from long-running benchmarks is visible in real time.
    result = run(cmd, check=False, capture=False)

    return {
        "status":     "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "name":       name,
        "script":     script,
    }
