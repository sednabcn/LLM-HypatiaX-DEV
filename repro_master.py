#!/usr/bin/env python3
"""
repro_master.py — HypatiaX Master Reproducibility Orchestrator
===============================================================
Implements the full strategy from the JMLR Reproducibility document:

  Phase 1  — Bootstrap & local --skip-slow  (Celeron)
  Phase 2  — Heavy steps on Colab/Kaggle    (exp2 · suppB · instability)
  Phase 3  — Merge checkpoint + final local --resume
  Utilities — checkpoint up/download, status report, reviewer packaging

Usage
─────
  # Full orchestration (detects what to do based on checkpoint state):
  python3 repro_master.py run

  # Individual phases:
  python3 repro_master.py phase1              # local --skip-slow
  python3 repro_master.py phase2 --env colab  # guide for Colab/Kaggle
  python3 repro_master.py phase3              # merge + final resume

  # Checkpoint utilities:
  python3 repro_master.py status              # show checkpoint state
  python3 repro_master.py upload-checkpoint   # print upload instructions
  python3 repro_master.py merge-checkpoint <path/to/downloaded.json>

  # Reviewer packaging:
  python3 repro_master.py reviewer --mode fast    # 20–60 min mode
  python3 repro_master.py reviewer --mode paper   # full paper mode

  # Generate Colab/Kaggle bash cell snippets:
  python3 repro_master.py colab-cells            # print copy-paste cells

  # Environment check:
  python3 repro_master.py doctor                 # diagnose runtime env

Env vars (override defaults):
  ANTHROPIC_API_KEY   required for all LLM steps
  PYSR_TIMEOUT        per-equation PySR wall-clock cap (default: 120 local, 1100 Colab)
  N_ITERATIONS        PySR iterations (default: 25 local, 1000 paper)
  POPULATIONS         PySR populations (default: 10 local, 30 paper)
  MAX_COMPLEXITY      expression complexity gate (default: 30)
  FAST                0=paper-quality  1=fast mode (default: auto-detected)
  REPRO_ENV           colab | kaggle | local  (default: auto-detected)
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ── Canonical paths ────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent


def _load_api_key() -> None:
    """
    Mirror the key-loading logic from run_all_checkpoint.py:
    if ANTHROPIC_API_KEY is absent from the environment, try to pull it
    from config_secrets.py (ANTHROPIC_API_KEY or API_KEY variables).
    Does nothing if the key is already set.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    secrets_path = SCRIPT_DIR / "config_secrets.py"
    if not secrets_path.exists():
        return
    try:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location("config_secrets", secrets_path)
        mod  = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        key = getattr(mod, "ANTHROPIC_API_KEY", None) or getattr(mod, "API_KEY", None)
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
    except Exception:
        pass  # silent – run_all_checkpoint.py will handle errors itself


_load_api_key()   # runs once at import time
CHECKPOINT   = SCRIPT_DIR / "logs" / "pipeline_checkpoint.json"
RESULTS_DIR  = SCRIPT_DIR / "hypatiax" / "data" / "results"
LOG_DIR      = SCRIPT_DIR / "logs"
RUN_ALL      = SCRIPT_DIR / "run_all_checkpoint.py"

# ── Slow steps (must run on Colab/Kaggle, not Celeron) ────────────────────────
SLOW_STEPS   = ["exp2", "suppB", "instability"]

# ── Paper-quality env vars ─────────────────────────────────────────────────────
PAPER_ENV = {
    "FAST":             "0",
    "N_ITERATIONS":     "1000",
    "POPULATIONS":      "30",
    "PYSR_TIMEOUT":     "1100",
    "METHOD_TIMEOUT":   "900",
    "LLM_K_RUNS":       "30",
    "N_FEYNMAN_TASKS":  "30",
    "N_NGUYEN_TASKS":   "12",
    "ENGINE_NAME":      "hybrid_system_v50_2",
    "LLM_MODEL":        "claude-sonnet-4-20250514",
    "DEFI_V3C_NO_TIMEOUT_FLAGS": "1",
    "DEFI_TASK_FILTER": "portfolio",
    "DEFI_SEEDS":       "42,99,123,777,2024",
    "SKIP_PKG_CHECK":   "1",
    "SKIP_PERF_ANALYSIS": "1",
    "HYPATIAX_CORE_OPTIONAL": "1",
    "PYTHON_JULIACALL_HANDLE_SIGNALS": "yes",
}

# ── Fast/local env vars ────────────────────────────────────────────────────────
LOCAL_ENV = {
    "FAST":             "1",
    "N_ITERATIONS":     "25",
    "POPULATIONS":      "10",
    "PYSR_TIMEOUT":     "120",
    "METHOD_TIMEOUT":   "240",
    "LLM_K_RUNS":       "1",
    "N_FEYNMAN_TASKS":  "5",
    "N_NGUYEN_TASKS":   "3",
    "ENGINE_NAME":      "hybrid_system_v50_2",
    "LLM_MODEL":        "claude-sonnet-4-20250514",
    "DEFI_V3C_NO_TIMEOUT_FLAGS": "1",
    "DEFI_TASK_FILTER": "portfolio",
    "DEFI_SEEDS":       "42",
    "SKIP_PKG_CHECK":   "1",
    "SKIP_PERF_ANALYSIS": "1",
    "HYPATIAX_CORE_OPTIONAL": "1",
    "PYTHON_JULIACALL_HANDLE_SIGNALS": "yes",
}

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def banner(text: str, char: str = "═") -> None:
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def step_banner(text: str) -> None:
    print(f"\n  ── {text}")


def ok(msg: str)   -> None: print(f"  ✓  {msg}")
def warn(msg: str) -> None: print(f"  ⚠  {msg}")
def err(msg: str)  -> None: print(f"  ✗  {msg}")
def info(msg: str) -> None: print(f"  ℹ  {msg}")


def detect_env() -> str:
    """Detect runtime environment: colab | kaggle | local."""
    env_override = os.environ.get("REPRO_ENV", "").lower()
    if env_override in ("colab", "kaggle", "local"):
        return env_override
    if os.path.exists("/content"):
        return "colab"
    if os.path.exists("/kaggle"):
        return "kaggle"
    return "local"


def is_celeron() -> bool:
    """Heuristic: detect slow local hardware."""
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read().lower()
        return "celeron" in cpuinfo or "atom" in cpuinfo
    except Exception:
        return False


def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        try:
            return json.loads(CHECKPOINT.read_text())
        except Exception:
            return {}
    return {}


def save_checkpoint(state: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(state, indent=2))


def passed_steps(state: dict) -> list:
    return [k for k, v in state.items() if v == "pass"]


def failed_steps(state: dict) -> list:
    return [k for k, v in state.items() if v == "fail"]


def slow_steps_done(state: dict) -> bool:
    return all(state.get(s) == "pass" for s in SLOW_STEPS)


def build_env(mode: str = "local") -> dict:
    """Build subprocess environment dict."""
    base = {**os.environ}
    overrides = PAPER_ENV if mode == "paper" else LOCAL_ENV
    # Env vars already set in the environment take precedence over defaults
    for k, v in overrides.items():
        if k not in os.environ:
            base[k] = v
    # Always propagate PYTHONPATH
    base.setdefault("PYTHONPATH", str(SCRIPT_DIR))
    return base


def run_pipeline_step(*extra_args, mode: str = "local", live: bool = True) -> int:
    """Run run_all_checkpoint.py with given args, streaming output."""
    if not RUN_ALL.exists():
        err(f"run_all_checkpoint.py not found at {RUN_ALL}")
        return 1

    # -u forces unbuffered stdout/stderr so output streams in real time
    cmd = [sys.executable, "-u", str(RUN_ALL)] + list(extra_args)
    info("Running: " + " ".join(cmd))

    env = build_env(mode)
    # Belt-and-suspenders: propagate to any grandchild processes too
    env["PYTHONUNBUFFERED"] = "1"

    if live:
        proc = subprocess.Popen(
            cmd, env=env, cwd=str(SCRIPT_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
        return proc.returncode
    else:
        result = subprocess.run(cmd, env=env, cwd=str(SCRIPT_DIR))
        return result.returncode


# ─────────────────────────────────────────────────────────────────────────────
# DOCTOR — environment diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def cmd_doctor(_args) -> None:
    banner("Environment Doctor")

    runtime_env = detect_env()
    info(f"Runtime environment : {runtime_env}")
    info(f"Platform            : {platform.platform()}")
    info(f"Python              : {sys.version.split()[0]}")
    info(f"Script dir          : {SCRIPT_DIR}")

    # Hardware
    if is_celeron():
        warn("Celeron/Atom detected — slow steps (exp2/suppB/instability) must run on Colab/Kaggle")
    else:
        ok("Hardware not identified as Celeron")

    # Julia
    julia_bin = shutil.which("julia")
    if julia_bin:
        try:
            out = subprocess.check_output(["julia", "--version"], text=True, timeout=10).strip()
            ok(f"Julia: {out}")
        except Exception:
            warn("Julia found but version check failed")
    else:
        warn("Julia not found — PySR steps will fail. Install via: curl -fsSL https://install.julialang.org | sh")

    # API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key.startswith("sk-ant-"):
        ok(f"ANTHROPIC_API_KEY set (sk-ant-...{api_key[-6:]})")
    elif api_key:
        warn("ANTHROPIC_API_KEY set but does not look like a valid key")
    else:
        err("ANTHROPIC_API_KEY not set — LLM steps will fail")
        info("  Fix: export ANTHROPIC_API_KEY='sk-ant-...'")

    # run_all_checkpoint.py
    if RUN_ALL.exists():
        ok("run_all_checkpoint.py found")
    else:
        err(f"run_all_checkpoint.py NOT found at {RUN_ALL}")

    # Checkpoint
    state = load_checkpoint()
    if state:
        n_passed = len(passed_steps(state))
        n_failed = len(failed_steps(state))
        slow_done = slow_steps_done(state)
        ok(f"Checkpoint: {n_passed} passed, {n_failed} failed, slow_done={slow_done}")
    else:
        info("No checkpoint found — pipeline has not started")

    # Key packages
    for pkg in ["numpy", "pysr", "sympy", "sklearn", "anthropic"]:
        try:
            __import__(pkg)
            ok(f"Python package: {pkg}")
        except ImportError:
            warn(f"Python package missing: {pkg}  (run: pip install {pkg})")

    # Env var summary
    step_banner("Active env overrides")
    for k in ["FAST", "N_ITERATIONS", "POPULATIONS", "PYSR_TIMEOUT",
              "METHOD_TIMEOUT", "LLM_K_RUNS", "MAX_COMPLEXITY"]:
        v = os.environ.get(k)
        if v:
            info(f"  {k}={v}")
        else:
            info(f"  {k}=<default>")


# ─────────────────────────────────────────────────────────────────────────────
# STATUS — checkpoint report
# ─────────────────────────────────────────────────────────────────────────────

def cmd_status(_args) -> None:
    banner("Checkpoint Status")
    state = load_checkpoint()

    if not state:
        warn("No checkpoint found — run `python3 repro_master.py phase1` to start")
        return

    # Categorise
    all_step_ids = [
        "deps", "patches-gen", "patches-apply", "fixup-init", "fixup-tex",
        "validate", "check-hypatiax-protocols",
        "exp1", "exp1b", "exp2", "exp3", "exp3b",
        "suppB", "suppA", "instability", "extrap",
        "provenance", "discover-provenance", "scan-imports", "verify", "hashlock",
        "figures", "tables",
        "audit-setup", "audit-NB-01", "audit-NB-02", "audit-NB-03", "audit-NB-04", "audit-NB-05",
    ]

    icon = {"pass": "✓", "fail": "✗", "skip": "─"}
    for sid in all_step_ids:
        status = state.get(sid, "pending")
        sym = icon.get(status, "·")
        slow_tag = " [SLOW]" if sid in SLOW_STEPS else ""
        print(f"  {sym}  {sid:32s}  {status}{slow_tag}")

    n_passed = len(passed_steps(state))
    n_failed = len(failed_steps(state))
    slow_done = slow_steps_done(state)
    total = len(all_step_ids)

    print()
    print(f"  Passed       : {n_passed}/{total}")
    print(f"  Failed       : {n_failed}")
    print(f"  Slow steps   : {'all done ✓' if slow_done else 'pending — run Phase 2 on Colab/Kaggle'}")

    if n_failed > 0:
        print()
        warn("Failed steps — rerun with:")
        for sid in failed_steps(state):
            print(f"    python3 run_all_checkpoint.py --only {sid}")

    if slow_done and n_failed == 0:
        print()
        ok("PAPER_QUALITY run complete — ready for Phase 3 merge and reviewer packaging")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Local --skip-slow
# ─────────────────────────────────────────────────────────────────────────────

def cmd_phase1(args) -> None:
    banner("Phase 1 — Local Celeron · --skip-slow")

    env = detect_env()
    if env != "local":
        warn(f"Detected env={env}. Phase 1 is designed for local hardware.")
        warn("Continuing anyway — override with REPRO_ENV=local if needed.")

    if is_celeron():
        warn("Celeron hardware detected — this run will be slow but will work.")
        warn("Ensure Julia is installed and ANTHROPIC_API_KEY is set before starting.")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        warn("ANTHROPIC_API_KEY not found in environment or config_secrets.py.")
        warn("LLM steps will fail unless run_all_checkpoint.py loads the key itself.")
        warn("To silence this: export ANTHROPIC_API_KEY='sk-ant-...'")
        warn("Continuing anyway — run_all_checkpoint.py will handle the key.")
    else:
        ok(f"ANTHROPIC_API_KEY loaded (sk-ant-...{api_key[-6:]})")

    # Check for existing checkpoint
    state = load_checkpoint()
    resume_flag = []
    if state and passed_steps(state):
        n = len(passed_steps(state))
        info(f"Existing checkpoint found ({n} steps passed). Using --resume.")
        resume_flag = ["--resume"]
    else:
        info("Fresh run — no checkpoint found.")

    info("Running: deps → patches → fixups → validate → exp1 → exp1b → exp3 → exp3b → suppA → extrap → figures → tables")
    info("Skipping: exp2 · suppB · instability  (will run on Colab/Kaggle in Phase 2)")
    print()

    rc = run_pipeline_step("--skip-slow", *resume_flag, mode="local")

    if rc == 0:
        ok("Phase 1 complete ✓")
        ok("Next step: copy logs/pipeline_checkpoint.json to Colab/Kaggle, then run Phase 2.")
        print()
        print("    python3 repro_master.py upload-checkpoint")
        print("    python3 repro_master.py colab-cells")
    else:
        err(f"Phase 1 failed (exit {rc}).")
        state = load_checkpoint()
        for sid in failed_steps(state):
            print(f"    Rerun: python3 run_all_checkpoint.py --only {sid}")
        sys.exit(rc)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Colab / Kaggle heavy steps
# ─────────────────────────────────────────────────────────────────────────────

def cmd_phase2(args) -> None:
    env = getattr(args, "env", "auto")
    if env == "auto":
        env = detect_env()
    if env == "local":
        env = "colab"   # default suggestion

    banner(f"Phase 2 — Heavy Steps · {env.capitalize()}")

    state = load_checkpoint()
    if not state:
        warn("No checkpoint found. Run Phase 1 locally first, then upload the checkpoint.")
        warn("See: python3 repro_master.py upload-checkpoint")
        sys.exit(1)

    pending_slow = [s for s in SLOW_STEPS if state.get(s) != "pass"]
    if not pending_slow:
        ok("All slow steps already passed in checkpoint — nothing to do.")
        info("Run Phase 3: python3 repro_master.py phase3")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        warn("ANTHROPIC_API_KEY not found in environment or config_secrets.py.")
        warn("Set it in the notebook with:")
        warn("    import os; os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'")
        warn("Continuing — run_all_checkpoint.py will handle the key.")
    else:
        ok(f"ANTHROPIC_API_KEY loaded (sk-ant-...{api_key[-6:]})")

    info(f"Slow steps remaining: {pending_slow}")
    info("Running each step individually (one crash does not lose others).")

    for step_id in pending_slow:
        step_banner(f"Running: {step_id}")
        rc = run_pipeline_step("--resume", "--only", step_id, mode="paper")
        if rc == 0:
            ok(f"{step_id} complete ✓")
        else:
            err(f"{step_id} failed (exit {rc}).")
            err("Checkpoint saved. Re-run this step with:")
            print("    python3 repro_master.py phase2  (it will skip already-passed steps)")
            print("    -- or --")
            print(f"    python3 run_all_checkpoint.py --resume --only {step_id}")
            sys.exit(rc)

    ok("Phase 2 complete ✓ — all slow steps passed.")
    info("Download updated logs/pipeline_checkpoint.json and run Phase 3 locally:")
    print("    python3 repro_master.py phase3")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Merge checkpoint + final local resume
# ─────────────────────────────────────────────────────────────────────────────

def cmd_phase3(args) -> None:
    banner("Phase 3 — Merge Checkpoint + Final Resume")

    state = load_checkpoint()
    if not state:
        warn("No checkpoint found. Have you merged the Colab/Kaggle checkpoint?")
        warn("See: python3 repro_master.py merge-checkpoint <path>")
        sys.exit(1)

    pending_slow = [s for s in SLOW_STEPS if state.get(s) != "pass"]
    if pending_slow:
        warn(f"Slow steps not yet in checkpoint: {pending_slow}")
        warn("Run Phase 2 on Colab/Kaggle first, then merge the checkpoint here.")
        warn("See: python3 repro_master.py merge-checkpoint <downloaded.json>")
        sys.exit(1)

    passed = passed_steps(state)
    info(f"Checkpoint has {len(passed)} steps passed (including all slow steps).")
    info("Running final --resume to complete figures / tables / audit / hashlock …")
    print()

    rc = run_pipeline_step("--resume", mode="paper")

    if rc == 0:
        ok("Phase 3 complete ✓")
        ok("PAPER_QUALITY reproducibility run is DONE.")
        print()
        ok("Next: package for reviewers:")
        print("    python3 repro_master.py reviewer --mode fast")
        print("    python3 repro_master.py reviewer --mode paper")
    else:
        err(f"Phase 3 failed (exit {rc}).")
        state = load_checkpoint()
        for sid in failed_steps(state):
            print(f"    Rerun: python3 run_all_checkpoint.py --only {sid}")
        sys.exit(rc)


# ─────────────────────────────────────────────────────────────────────────────
# RUN — auto-detect phase and proceed
# ─────────────────────────────────────────────────────────────────────────────

def cmd_run(args) -> None:
    banner("Auto-Orchestration — detecting phase")
    state = load_checkpoint()

    if not state or not passed_steps(state):
        info("No checkpoint found → starting Phase 1")
        cmd_phase1(args)
        return

    pending_slow = [s for s in SLOW_STEPS if state.get(s) != "pass"]
    env = detect_env()

    if pending_slow and env == "local":
        info(f"Phase 1 done. Slow steps pending: {pending_slow}")
        info("Cannot run slow steps on local hardware.")
        info("Next action:")
        print()
        print("  1. Copy checkpoint to Colab/Kaggle:")
        print("       python3 repro_master.py upload-checkpoint")
        print()
        print("  2. In Colab/Kaggle, run:")
        print("       python3 repro_master.py phase2")
        print()
        print("  3. Download updated checkpoint, then:")
        print("       python3 repro_master.py merge-checkpoint <downloaded.json>")
        print("       python3 repro_master.py phase3")
        return

    if pending_slow and env in ("colab", "kaggle"):
        info(f"Colab/Kaggle env detected, slow steps pending: {pending_slow}")
        cmd_phase2(args)
        return

    if not pending_slow:
        info("All slow steps done → running Phase 3 (final merge/resume)")
        cmd_phase3(args)
        return

    warn("Could not determine next phase. Run `python3 repro_master.py status` to inspect.")


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def cmd_upload_checkpoint(_args) -> None:
    banner("Checkpoint Upload Instructions")

    if not CHECKPOINT.exists():
        err("No checkpoint found. Run Phase 1 first.")
        sys.exit(1)

    state = load_checkpoint()
    n = len(passed_steps(state))
    info(f"Checkpoint at: {CHECKPOINT}  ({n} steps passed)")
    print()

    print("── Colab ──────────────────────────────────────────────────────")
    print("  Option A — Manual upload:")
    print("    1. In Colab file panel (left sidebar), navigate to /content/<repo>/logs/")
    print("    2. Upload pipeline_checkpoint.json")
    print()
    print("  Option B — Google Drive mount:")
    print("    from google.colab import drive")
    print("    drive.mount('/content/drive')")
    print("    import shutil")
    print("    shutil.copy('/content/drive/MyDrive/pipeline_checkpoint.json',")
    print("                '/content/<repo>/logs/pipeline_checkpoint.json')")
    print()
    print("── Kaggle ─────────────────────────────────────────────────────")
    print("  1. Go to kaggle.com → Datasets → New Dataset")
    print("  2. Upload pipeline_checkpoint.json")
    print("  3. In notebook, add dataset and copy:")
    print("    import shutil")
    print("    shutil.copy('/kaggle/input/<dataset>/pipeline_checkpoint.json',")
    print("                '/kaggle/working/<repo>/logs/pipeline_checkpoint.json')")
    print()
    print("── After uploading, on Colab/Kaggle run: ──────────────────────")
    print("    python3 repro_master.py phase2")


def cmd_merge_checkpoint(args) -> None:
    banner("Merge Checkpoint")

    src_path = Path(args.path)
    if not src_path.exists():
        err(f"Source checkpoint not found: {src_path}")
        sys.exit(1)

    try:
        incoming = json.loads(src_path.read_text())
    except Exception as e:
        err(f"Failed to parse {src_path}: {e}")
        sys.exit(1)

    existing = load_checkpoint()

    # Merge: take "pass" over any other status, never overwrite "pass" with "fail"
    merged = {**existing}
    upgraded = []
    for step_id, status in incoming.items():
        current = merged.get(step_id)
        if status == "pass" and current != "pass":
            merged[step_id] = "pass"
            upgraded.append(step_id)
        elif status != "pass" and current is None:
            merged[step_id] = status

    save_checkpoint(merged)

    ok(f"Merged {src_path.name} into {CHECKPOINT}")
    if upgraded:
        ok(f"Newly passed steps: {upgraded}")
    else:
        info("No new steps promoted to 'pass'")

    # Show slow step coverage
    for s in SLOW_STEPS:
        st = merged.get(s, "pending")
        sym = "✓" if st == "pass" else "·"
        print(f"  {sym}  {s}: {st}")

    pending_slow = [s for s in SLOW_STEPS if merged.get(s) != "pass"]
    if not pending_slow:
        ok("All slow steps passed → ready for Phase 3")
        print("    python3 repro_master.py phase3")
    else:
        warn(f"Slow steps still pending: {pending_slow}")
        print("    Run Phase 2 on Colab/Kaggle for remaining steps.")


# ─────────────────────────────────────────────────────────────────────────────
# REVIEWER PACKAGING
# ─────────────────────────────────────────────────────────────────────────────

def cmd_reviewer(args) -> None:
    mode = getattr(args, "mode", "fast")
    banner(f"Reviewer Packaging — {mode.upper()} mode")

    state = load_checkpoint()

    if mode == "paper":
        pending_slow = [s for s in SLOW_STEPS if state.get(s) != "pass"]
        if pending_slow:
            err("PAPER_QUALITY run not complete. Slow steps still pending:")
            for s in pending_slow: print(f"    {s}")
            err("Finish Phase 2 + Phase 3 before packaging for reviewers.")
            sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        warn("ANTHROPIC_API_KEY not found in environment or config_secrets.py.")
        warn("LLM steps will fail. To fix: export ANTHROPIC_API_KEY='sk-ant-...'")
        warn("Continuing — run_all_checkpoint.py will handle the key.")
    else:
        ok(f"ANTHROPIC_API_KEY loaded (sk-ant-...{api_key[-6:]})")

    if mode == "fast":
        info("Reviewer fast mode: ~20–60 min, validates pipeline structure")
        info("Using: --one-equation --skip-slow  (smoke-test each step)")
        rc = run_pipeline_step(
            "--one-equation", "--skip-slow", "--resume",
            "--pysr-timeout", "120",
            mode="local",
        )
    else:
        info("Reviewer paper mode: ~15–25 h, full paper numbers")
        info("Using: full pipeline with paper-quality env vars")
        rc = run_pipeline_step("--resume", mode="paper")

    if rc == 0:
        ok(f"Reviewer {mode} run complete ✓")
        if RESULTS_DIR.exists():
            figs = list((RESULTS_DIR / "figures").glob("*.pdf")) if (RESULTS_DIR / "figures").exists() else []
            tbls = list((RESULTS_DIR / "tables").glob("*.tex"))  if (RESULTS_DIR / "tables").exists() else []
            ok(f"Figures: {len(figs)}   Tables: {len(tbls)}")
    else:
        err(f"Reviewer {mode} run failed (exit {rc}).")
        sys.exit(rc)


# ─────────────────────────────────────────────────────────────────────────────
# COLAB CELLS — generate copy-pasteable notebook cells
# ─────────────────────────────────────────────────────────────────────────────

def cmd_colab_cells(_args) -> None:
    banner("Colab / Kaggle Bash Cells  (copy-paste into notebook)")

    CELLS = [
        ("Cell 1 — Clone repo & install deps", """\
# ── Cell 1: Clone & install ──────────────────────────────────────────────────
import os, subprocess, sys

REPO_URL    = "https://github.com/YOUR_ORG/hypatiaX.git"   # ← replace
BRANCH      = "main"
REPO_DIR    = "/content/hypatiaX"  # /kaggle/working/hypatiaX on Kaggle

# Set API key — use Colab Secrets or enter manually (never commit)
os.environ["ANTHROPIC_API_KEY"] = ""  # ← paste key here OR use Colab Secrets

if not os.path.exists(REPO_DIR):
    subprocess.check_call(["git", "clone", "--branch", BRANCH, REPO_URL, REPO_DIR])
os.chdir(REPO_DIR)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "-r", "requirements.txt"])
print("✓ Repo ready at", REPO_DIR)
"""),

        ("Cell 2 — Upload checkpoint from local machine", """\
# ── Cell 2: Upload checkpoint (Colab) ────────────────────────────────────────
# Option A: manual upload
from google.colab import files
import shutil, os

os.makedirs("logs", exist_ok=True)
print("Click 'Choose Files' and select your local pipeline_checkpoint.json")
uploaded = files.upload()
for fname in uploaded:
    shutil.move(fname, "logs/pipeline_checkpoint.json")
    print(f"✓ Checkpoint saved to logs/pipeline_checkpoint.json")

# Option B: from Google Drive (uncomment if preferred)
# from google.colab import drive
# drive.mount('/content/drive')
# shutil.copy('/content/drive/MyDrive/pipeline_checkpoint.json',
#             'logs/pipeline_checkpoint.json')
"""),

        ("Cell 2 — Upload checkpoint from local machine (Kaggle)", """\
# ── Cell 2: Upload checkpoint (Kaggle) ───────────────────────────────────────
# Add your checkpoint as a Kaggle Dataset first, then:
import shutil, os

DATASET_PATH = "/kaggle/input/YOUR-DATASET-NAME/pipeline_checkpoint.json"
os.makedirs("logs", exist_ok=True)
shutil.copy(DATASET_PATH, "logs/pipeline_checkpoint.json")
print("✓ Checkpoint copied to logs/pipeline_checkpoint.json")
"""),

        ("Cell 3 — Set paper-quality env vars", """\
# ── Cell 3: Paper-quality env vars ───────────────────────────────────────────
import os

PAPER_ENV = {
    "FAST":             "0",
    "N_ITERATIONS":     "1000",
    "POPULATIONS":      "30",
    "PYSR_TIMEOUT":     "1100",
    "METHOD_TIMEOUT":   "900",
    "LLM_K_RUNS":       "30",
    "N_FEYNMAN_TASKS":  "30",
    "N_NGUYEN_TASKS":   "12",
    "ENGINE_NAME":      "hybrid_system_v50_2",
    "LLM_MODEL":        "claude-sonnet-4-20250514",
    "DEFI_V3C_NO_TIMEOUT_FLAGS": "1",
    "DEFI_TASK_FILTER": "portfolio",
    "DEFI_SEEDS":       "42,99,123,777,2024",
    "SKIP_PKG_CHECK":   "1",
    "SKIP_PERF_ANALYSIS": "1",
    "HYPATIAX_CORE_OPTIONAL": "1",
    "PYTHON_JULIACALL_HANDLE_SIGNALS": "yes",
}
for k, v in PAPER_ENV.items():
    os.environ[k] = v
print("✓ Paper-quality env vars set")
"""),

        ("Cell 4a — Run exp2 (Feynman, ~4–6 h)", """\
# ── Cell 4a: exp2 — Feynman 30-equation extrapolation ────────────────────────
# Run in its own cell. A Colab disconnect only loses this step (checkpoint safe).
import subprocess, sys
result = subprocess.run(
    [sys.executable, "run_all_checkpoint.py", "--resume", "--only", "exp2"],
    cwd="/content/hypatiaX"  # adjust for Kaggle: /kaggle/working/hypatiaX
)
print("exp2 exit code:", result.returncode)
"""),

        ("Cell 4b — Run suppB (noise sweep, ~4–6 h)", """\
# ── Cell 4b: suppB — noise & sample-complexity sweep ─────────────────────────
import subprocess, sys
result = subprocess.run(
    [sys.executable, "run_all_checkpoint.py", "--resume", "--only", "suppB"],
    cwd="/content/hypatiaX"
)
print("suppB exit code:", result.returncode)
"""),

        ("Cell 4c — Run instability K=30 (~3–5 h)", """\
# ── Cell 4c: instability K=30 ─────────────────────────────────────────────────
import subprocess, sys, os
os.environ["LLM_K_RUNS"] = "30"
result = subprocess.run(
    [sys.executable, "run_all_checkpoint.py", "--resume", "--only", "instability"],
    cwd="/content/hypatiaX"
)
print("instability exit code:", result.returncode)
"""),

        ("Cell 5 — Download updated checkpoint", """\
# ── Cell 5: Download checkpoint back to local ────────────────────────────────
# Colab:
from google.colab import files
files.download("logs/pipeline_checkpoint.json")
print("✓ Download started — save as pipeline_checkpoint.json")
print("  Then on local machine:")
print("  python3 repro_master.py merge-checkpoint ~/Downloads/pipeline_checkpoint.json")
print("  python3 repro_master.py phase3")

# Kaggle: copy to output folder so it appears in notebook outputs
# import shutil
# shutil.copy("logs/pipeline_checkpoint.json",
#             "/kaggle/working/pipeline_checkpoint.json")
"""),

        ("Cell 6 — Session keep-alive (Colab only)", """\
# ── Cell 6: Colab keep-alive ──────────────────────────────────────────────────
# Run in a SEPARATE cell BEFORE launching a long step.
# Opens browser console → paste this JS to auto-click the keep-alive dialog:
#
#   setInterval(() => {
#     const btn = document.querySelector('#ok');
#     if (btn) btn.click();
#   }, 60000);
#
# Python side: no-op loop as a heartbeat signal
import time, threading

def _keepalive():
    while True:
        time.sleep(55)
        print(".", end="", flush=True)

t = threading.Thread(target=_keepalive, daemon=True)
t.start()
print("Keep-alive thread started")
"""),
    ]

    for title, code in CELLS:
        print(f"\n{'─' * 70}")
        print(f"  {title}")
        print('─' * 70)
        print(code)

    print('─' * 70)
    info("Tip: run cells 4a / 4b / 4c in SEPARATE Colab sessions")
    info("     so a disconnect on suppB does not lose exp2.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="repro_master.py",
        description="HypatiaX master reproducibility orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # run
    sub.add_parser("run", help="Auto-detect phase and proceed")

    # phase1
    sub.add_parser("phase1", help="Local --skip-slow (Celeron)")

    # phase2
    p2 = sub.add_parser("phase2", help="Heavy steps on Colab/Kaggle")
    p2.add_argument("--env", choices=["colab", "kaggle", "auto"], default="auto",
                    help="Target environment (default: auto-detect)")

    # phase3
    sub.add_parser("phase3", help="Merge checkpoint + final --resume")

    # status
    sub.add_parser("status", help="Show checkpoint state")

    # doctor
    sub.add_parser("doctor", help="Diagnose runtime environment")

    # upload-checkpoint
    sub.add_parser("upload-checkpoint", help="Print upload instructions")

    # merge-checkpoint
    p_merge = sub.add_parser("merge-checkpoint", help="Merge downloaded checkpoint")
    p_merge.add_argument("path", help="Path to downloaded pipeline_checkpoint.json")

    # reviewer
    p_rev = sub.add_parser("reviewer", help="Run reviewer packaging")
    p_rev.add_argument("--mode", choices=["fast", "paper"], default="fast",
                       help="fast (~20–60 min) or paper (~15–25 h)")

    # colab-cells
    sub.add_parser("colab-cells", help="Print Colab/Kaggle copy-paste cells")

    args = parser.parse_args()

    dispatch = {
        "run":               cmd_run,
        "phase1":            cmd_phase1,
        "phase2":            cmd_phase2,
        "phase3":            cmd_phase3,
        "status":            cmd_status,
        "doctor":            cmd_doctor,
        "upload-checkpoint": cmd_upload_checkpoint,
        "merge-checkpoint":  cmd_merge_checkpoint,
        "reviewer":          cmd_reviewer,
        "colab-cells":       cmd_colab_cells,
    }

    if args.command is None:
        parser.print_help()
        print()
        print("Quick start:")
        print("  python3 repro_master.py doctor       # check your environment")
        print("  python3 repro_master.py run          # auto-detect phase and go")
        print("  python3 repro_master.py status       # inspect checkpoint")
        print("  python3 repro_master.py colab-cells  # get Colab/Kaggle cells")
        sys.exit(0)

    fn = dispatch.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
