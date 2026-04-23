#!/usr/bin/env python3
"""
exp3_nguyen12_hybrid50v_02.py  —  Exp 3 · Nguyen-12 SR suite  (§10.8 primary)
==============================================================================
Standalone Python script version — safe to run with `python3` directly.

Origin: extracted from HypatiaX_Experiments_v6_PUBLIC.ipynb (Cell 27)
Fixes applied (v02 → v03):
  - Removed Jupyter-only magic syntax (!pip install, %env, !)
  - Added __main__ guard
  - Added sys.path setup so imports resolve from repo root
  - Stale lock cleared before run (mirrors notebook Cell 27 logic)
  - Deps checked with importlib instead of subprocess pip call
  [PATCH A] Unified seed block — random/numpy/torch/Julia all set from SEED
  [PATCH E] Google Colab import replaced with pipeline-safe API key loader
  [PATCH F] IPython download block replaced with pipeline-safe output printer
  [PATCH G] Protocol imports use try/fallback for pre/post-restructure layout

Expected result : 11/12 H (91.7 %) · 10/12 P (83.3 %) · 0/12 NN
                  MW P>NN U=113, p=0.0097
Wall time       : 30–90 min
SEED            : 42 (fixed for reproducibility; override with --seed)

Usage
-----
    python3 exp3_nguyen12_hybrid50v_02.py             # SEED=42 (default)
    python3 exp3_nguyen12_hybrid50v_02.py --seed 123  # stability check
    python3 exp3_nguyen12_hybrid50v_02.py --seed 777  # stability check
"""

import sys
import os
import pathlib
import argparse
import importlib
import random

import numpy as np

# ── 1. Resolve repo root & set sys.path ───────────────────────────────────
# Script lives at:  <repo>/hypatiax/experiments/benchmarks/exp3_nguyen12_hybrid50v_02.py
# Repo root is 3 levels up.
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parents[2]   # benchmarks/ -> experiments/ -> hypatiax/ -> repo root

# Support override via environment variable (set by pipeline or notebook)
_REPRO_ROOT = pathlib.Path(os.environ.get("REPRO_ROOT", str(_REPO_ROOT)))

for _p in [str(_REPRO_ROOT), str(_REPRO_ROOT / "hypatiax")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 2. Argument parsing (early — SEED needed before env setup) ────────────
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Exp 3 · Nguyen-12 SR suite (§10.8 primary)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for all RNG sources (default: 42)"
    )
    return parser.parse_args()

# Parse early so SEED is available for the seed block below.
# (argparse is safe to call at module level — it only reads sys.argv)
_args = _parse_args()
SEED  = int(os.environ.get("EXPERIMENT_SEED", str(_args.seed)))

# ── 3. [PATCH A] Unified seed block — ALL sources seeded from SEED ────────
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["JULIA_SEED"]     = str(SEED)   # PySR / Julia RNG
try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
except ImportError:
    pass
print(f"✅ All seeds set to {SEED}")

# ── 4. Dependency check (no !pip magic — deps managed by pipeline) ─────────
_REQUIRED = ["pysr", "anthropic", "sklearn", "scipy", "sympy", "numpy", "pandas", "matplotlib"]
_MISSING  = []
for _pkg in _REQUIRED:
    try:
        importlib.import_module(_pkg)
    except ImportError:
        _MISSING.append(_pkg)

if _MISSING:
    print(f"  ✗  Missing packages: {', '.join(_MISSING)}")
    print("     Install via:  pip install " + " ".join(_MISSING))
    print("     Or run the full pipeline first (it installs deps in Phase 0).")
    sys.exit(1)

# ── 5. Clear stale protocol cache lock (mirrors notebook Cell 27) ──────────
_results_dir = _REPRO_ROOT / "hypatiax" / "data" / "results"
_locks = list(_results_dir.glob(".lock_*")) if _results_dir.exists() else []
if _locks:
    for _l in _locks:
        _l.unlink()
    print(f"  Cleared {len(_locks)} stale lock(s) — experiment will run fresh")
else:
    print("  No stale locks found")

# ── 6. Environment variables (mirrors notebook Cell 2 / %env block) ────────
os.environ["NN_SEED"]   = str(SEED)   # always propagate resolved SEED
os.environ["PYSR_SEED"] = str(SEED)
os.environ.setdefault("LLM_MODEL",   "claude-sonnet-4-6")
os.environ.setdefault("LLM_RETRIES", "3")
os.environ.setdefault("LLM_K_RUNS",  "1")
os.environ.setdefault("ENGINE",      "hybrid_system_v50_2")
os.environ.setdefault("REPRO_ROOT",  str(_REPRO_ROOT))

# ── 7. [PATCH E] Pipeline-safe API key loader (replaces Colab userdata) ───
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

if not ANTHROPIC_API_KEY:
    # Try Colab userdata only if actually running inside Colab
    try:
        from google.colab import userdata as _colab_userdata
        ANTHROPIC_API_KEY = _colab_userdata.get("ANTHROPIC_API_KEY") or ""
    except (ImportError, Exception):
        pass

if not ANTHROPIC_API_KEY:
    # Try .env file relative to repo root
    for _env_path in [
        _REPRO_ROOT / ".env",
        _REPRO_ROOT / "hypatiax" / ".env",
    ]:
        if _env_path.exists():
            for _line in _env_path.read_text().splitlines():
                if _line.startswith("ANTHROPIC_API_KEY="):
                    ANTHROPIC_API_KEY = _line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
        if ANTHROPIC_API_KEY:
            break

USE_LLM = True
if ANTHROPIC_API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
    print("API key set ✓")
else:
    print("⚠  No API key found — LLM guidance disabled (USE_LLM forced False)")
    USE_LLM = False

# ── 8. Main experiment logic ───────────────────────────────────────────────
def run(seed: int = 42):
    """Run the Nguyen-12 benchmark via the protocol harness."""
    # ── Smoke-test env var injection ──────────────────────────────────────
    # run_all_checkpoint.py --one-equation sets these; paper-quality runs
    # leave them unset so the defaults below apply.
    _n_tasks    = int(os.environ.get("N_NGUYEN_TASKS", 12))   # default: all 12
    _niter      = int(os.environ.get("N_ITERATIONS",   1000))  # paper default
    _pops       = int(os.environ.get("POPULATIONS",    30))    # paper default
    _timeout    = int(os.environ.get("PYSR_TIMEOUT",   360))   # paper default

    print(f"\n{'='*68}")
    print(f"  Exp 3 · Nguyen-12 SR suite  (§10.8)  SEED={seed}")
    print(f"  Expected: 11/12 H (91.7%) · 10/12 P · MW U=113, p=0.0097")
    print(f"  Config  : n_tasks={_n_tasks}  niterations={_niter}  populations={_pops}  timeout={_timeout}s")
    print(f"{'='*68}\n")

    # [PATCH G] Try new restructured layout first, fall back to old layout
    try:
        from hypatiax.protocols.universal_protocol import run_protocol
    except ImportError:
        from protocols.universal_protocol import run_protocol

    try:
        from hypatiax.core.runners.common import run_task
    except ImportError:
        from core.runners.common import run_task

    config = {
        "name":        "nguyen12_exp3",
        "seed":        seed,
        "use_llm":     USE_LLM,
        "n_tasks":     _n_tasks,
        "niterations": _niter,
        "populations": _pops,
        "timeout":     _timeout,
        "args":        ["--seed", str(seed)],
    }
    result = run_protocol(config, run_task)

    print(f"\n  Protocol returned: {result}")

    # [PATCH F] Pipeline-safe output printer (replaces IPython download block)
    OUTPUT_JSON = str(_results_dir / f"exp3_nguyen12_seed{seed}.json")
    OUTPUT_TEX  = str(_results_dir / f"exp3_nguyen12_seed{seed}.tex")

    _out_files = [
        (OUTPUT_JSON, "JSON results"),
        (OUTPUT_TEX,  "LaTeX table"),
    ]
    print("\n⬇ Output files:")
    for _f, _label in _out_files:
        _path = pathlib.Path(_f)
        if _path.exists():
            print(f"  ✅ {_label}: {_path.resolve()} ({_path.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  ⚠  {_label}: NOT FOUND at {_f}")

    # IPython download links — only active when running inside a notebook/Colab
    try:
        import base64
        from IPython.display import display, HTML
        _links = []
        for _f, _label in _out_files:
            _path = pathlib.Path(_f)
            if _path.exists():
                _data = base64.b64encode(_path.read_bytes()).decode()
                _mime = "application/json" if _f.endswith(".json") else "application/x-tex"
                _links.append(
                    f'<li><a href="data:{_mime};base64,{_data}" download="{_path.name}">'
                    f'📄 {_label}</a> ({_path.stat().st_size / 1024:.1f} KB)</li>'
                )
        if _links:
            display(HTML(
                '<div style="border:1px solid #ccc;border-radius:6px;padding:12px;background:#f9f9f9">'
                '<b>⬇ Download experiment outputs</b><ul>'
                + "".join(_links)
                + "</ul></div>"
            ))
    except (ImportError, Exception):
        pass  # Not in a notebook — file paths already printed above

    return result


# ── 9. Entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    run(seed=SEED)
