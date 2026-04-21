#!/usr/bin/env python3
"""
exp3_nguyen12_hybrid50v_02.py  —  Exp 3 · Nguyen-12 SR suite  (§10.8 primary)
==============================================================================
Standalone Python script version — safe to run with `python3` directly.

Origin: extracted from HypatiaX_Experiments_v6_PUBLIC.ipynb (Cell 27)
Fixes applied:
  - Removed Jupyter-only magic syntax (!pip install, %env, !)
  - Added __main__ guard
  - Added sys.path setup so imports resolve from repo root
  - Stale lock cleared before run (mirrors notebook Cell 27 logic)
  - Deps checked with importlib instead of subprocess pip call

Expected result : 11/12 H (91.7 %) · 10/12 P (83.3 %) · 0/12 NN
                  MW P>NN U=113, p=0.0097
Wall time       : 30–90 min
SEED            : 42 (fixed for reproducibility)

Usage
-----
    python3 exp3_nguyen12_hybrid50v_02.py           # SEED=42 (default)
    python3 exp3_nguyen12_hybrid50v_02.py --seed 123  # stability check
"""

import sys
import os
import pathlib
import argparse
import importlib

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

# ── 2. Dependency check (no !pip magic — deps managed by pipeline) ─────────
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

# ── 3. Clear stale protocol cache lock (mirrors notebook Cell 27) ──────────
_results_dir = _REPRO_ROOT / "hypatiax" / "data" / "results"
_locks = list(_results_dir.glob(".lock_*")) if _results_dir.exists() else []
if _locks:
    for _l in _locks:
        _l.unlink()
    print(f"  Cleared {len(_locks)} stale lock(s) — experiment will run fresh")
else:
    print("  No stale locks found")

# ── 4. Environment variables (mirrors notebook Cell 2 / %env block) ────────
os.environ.setdefault("NN_SEED",    "42")
os.environ.setdefault("PYSR_SEED",  "42")
os.environ.setdefault("LLM_MODEL",  "claude-sonnet-4-6")
os.environ.setdefault("LLM_RETRIES","3")
os.environ.setdefault("LLM_K_RUNS", "1")
os.environ.setdefault("ENGINE",     "hybrid_system_v50_2")
os.environ.setdefault("REPRO_ROOT", str(_REPRO_ROOT))

# ── 5. Argument parsing ────────────────────────────────────────────────────
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Exp 3 · Nguyen-12 SR suite (§10.8 primary)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for LLM/NN components (default: 42)"
    )
    return parser.parse_args()

# ── 6. Main experiment logic ───────────────────────────────────────────────
def run(seed: int = 42):
    """Run the Nguyen-12 benchmark via the protocol harness."""
    # Override seeds if non-default
    if seed != 42:
        os.environ["NN_SEED"]   = str(seed)
        os.environ["PYSR_SEED"] = str(seed)

    print(f"\n{'='*68}")
    print(f"  Exp 3 · Nguyen-12 SR suite  (§10.8)  SEED={seed}")
    print(f"  Expected: 11/12 H (91.7%) · 10/12 P · MW U=113, p=0.0097")
    print(f"{'='*68}\n")

    # Import the protocol harness (resolved via sys.path above)
    from protocols.universal_protocol import run_protocol
    from core.runners.common import run_task

    config = {
        "name": "nguyen12_exp3",
        "args": (["--seed", str(seed)] if seed != 42 else []),
    }
    result = run_protocol(config, run_task)

    print(f"\n  Protocol returned: {result}")
    return result


# ── 7. Entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = _parse_args()
    run(seed=args.seed)
