#!/usr/bin/env python3
"""
run_notebook_as_py.py — HypatiaX v6 notebook runner as a plain Python script
=============================================================================
Executes every code cell in HypatiaX_Experiments_v6_PUBLIC_fast_FIXED.ipynb
in order, exactly as Jupyter would, but without requiring a kernel.

Usage
-----
    # Fast mode (default, ~30-60 min):
    python3 run_notebook_as_py.py

    # Full paper-quality run (~15-25 h):
    FAST=0 python3 run_notebook_as_py.py

    # Custom seed:
    FAST=0 PYSR_SEED=123 NN_SEED=123 python3 run_notebook_as_py.py

    # Skip slow cells (Feynman, noise sweep, instability):
    SKIP_SLOW=1 python3 run_notebook_as_py.py

    # Single cell by index (0-based):
    ONLY_CELL=15 python3 run_notebook_as_py.py

    # Dry-run — print cells without executing:
    DRY_RUN=1 python3 run_notebook_as_py.py

    # On Kaggle/Colab: set ANTHROPIC_API_KEY before running:
    ANTHROPIC_API_KEY="sk-ant-..." FAST=1 python3 run_notebook_as_py.py

Environment variables
---------------------
    FAST              0 or 1 (default 1)
    PYSR_SEED         integer (default 42)
    NN_SEED           integer (default 42)
    ANTHROPIC_API_KEY Anthropic API key
    SKIP_SLOW         1 = skip Feynman / noise / instability cells
    ONLY_CELL         integer = run only this cell index
    DRY_RUN           1 = print source, do not exec
    NOTEBOOK_PATH     path to the .ipynb file (default: same dir as this script)

Exit codes
----------
    0  All cells passed
    1  One or more cells failed (exception or non-zero subprocess)
    2  Notebook file not found
"""

from __future__ import annotations
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType

# ── locate notebook ───────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = Path(
    os.environ.get(
        "NOTEBOOK_PATH",
        str(_SCRIPT_DIR / "HypatiaX_Experiments_v6_PUBLIC_fast_FIXED.ipynb"),
    )
)

if not NOTEBOOK_PATH.exists():
    # Fallback: original name
    fallback = _SCRIPT_DIR / "HypatiaX_Experiments_v6_PUBLIC_fast.ipynb"
    if fallback.exists():
        NOTEBOOK_PATH = fallback
    else:
        print(f"ERROR: notebook not found at {NOTEBOOK_PATH}", file=sys.stderr)
        sys.exit(2)

# ── config from env ───────────────────────────────────────────────────────────
FAST       = int(os.environ.get("FAST",      "1"))
SKIP_SLOW  = int(os.environ.get("SKIP_SLOW", "0"))
DRY_RUN    = int(os.environ.get("DRY_RUN",   "0"))
ONLY_CELL  = os.environ.get("ONLY_CELL", "")

# Cells that are "slow" (Feynman, noise sweep, instability) — by label substring
_SLOW_MARKERS = [
    "Exp 2", "Feynman", "suppB", "Supp B",
    "instability", "§10.9", "noise_sweep",
]

# ── load notebook ─────────────────────────────────────────────────────────────
nb     = json.loads(NOTEBOOK_PATH.read_text())
cells  = nb["cells"]
n_code = sum(1 for c in cells if c["cell_type"] == "code")

print("=" * 70)
print(f"  HypatiaX notebook runner")
print(f"  Notebook   : {NOTEBOOK_PATH.name}")
print(f"  Total cells: {len(cells)}  ({n_code} code)")
print(f"  FAST mode  : {'ON (smoke-test)' if FAST else 'OFF (paper-quality)'}")
print(f"  Seed       : NN_SEED={os.environ.get('NN_SEED','42')}  "
      f"PYSR_SEED={os.environ.get('PYSR_SEED','42')}")
print(f"  API key    : {'SET' if os.environ.get('ANTHROPIC_API_KEY') else 'NOT SET'}")
if DRY_RUN:
    print("  DRY RUN    : cells will be printed, not executed")
print("=" * 70)

# ── shared globals for exec ───────────────────────────────────────────────────
# Each cell shares a single global namespace, mimicking Jupyter kernel state.
_globals: dict = {
    "__name__": "__main__",
    "__builtins__": __builtins__,
    # Pre-inject FAST so cell 15 reads it correctly even without the env trick
    "FAST": FAST,
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _is_slow(src: str) -> bool:
    return any(m.lower() in src.lower() for m in _SLOW_MARKERS)


def _banner(idx: int, preview: str, cell_type: str) -> None:
    tag = "CODE" if cell_type == "code" else "MD  "
    print(f"\n┌── [{idx:02d}/{len(cells)-1}] {tag}  {preview[:60]}")


def _run_cell(idx: int, src: str) -> bool:
    """Execute source in shared globals. Returns True on success."""
    t0 = time.time()
    try:
        exec(compile(src, f"<cell {idx}>", "exec"), _globals)   # noqa: S102
        elapsed = time.time() - t0
        print(f"└── ✓  ({elapsed:.1f}s)")
        return True
    except SystemExit as e:
        # SystemExit(0) is normal; SystemExit(1) is a gate failure
        elapsed = time.time() - t0
        if e.code == 0:
            print(f"└── ✓  sys.exit(0)  ({elapsed:.1f}s)")
            return True
        print(f"└── ✗  SystemExit({e.code})  ({elapsed:.1f}s)")
        return False
    except Exception:
        elapsed = time.time() - t0
        print(f"└── ✗  EXCEPTION  ({elapsed:.1f}s)")
        traceback.print_exc()
        return False


# ── main execution loop ───────────────────────────────────────────────────────
failed: list[int] = []
skipped: list[int] = []
t_total = time.time()

only_idx = int(ONLY_CELL) if ONLY_CELL.strip() else None

for idx, cell in enumerate(cells):
    ctype = cell["cell_type"]
    src   = "".join(cell.get("source", []))

    if not src.strip():
        continue

    preview = src.strip().splitlines()[0][:65]
    _banner(idx, preview, ctype)

    # Skip non-code cells
    if ctype != "code":
        print(f"└── (markdown/raw — skipped)")
        continue

    # --only filter
    if only_idx is not None and idx != only_idx:
        print(f"└── (skipped — ONLY_CELL={only_idx})")
        skipped.append(idx)
        continue

    # --skip-slow filter
    if SKIP_SLOW and _is_slow(src):
        print(f"└── (skipped — SKIP_SLOW=1, slow cell)")
        skipped.append(idx)
        continue

    if DRY_RUN:
        print(src[:300])
        print(f"└── (dry-run — not executed)")
        continue

    ok = _run_cell(idx, src)
    if not ok:
        failed.append(idx)
        # For FAST mode keep going; for FULL mode abort on failure
        if not FAST:
            print(f"\nPipeline aborted at cell {idx} (FAST=0 — strict mode).")
            break

# ── summary ───────────────────────────────────────────────────────────────────
elapsed_total = time.time() - t_total
hh, rem = divmod(int(elapsed_total), 3600)
mm, ss  = divmod(rem, 60)

print("\n" + "=" * 70)
print("  Run summary")
print("=" * 70)
print(f"  Wall time : {hh:02d}:{mm:02d}:{ss:02d}")
print(f"  Skipped   : {len(skipped)}")
print(f"  Failed    : {len(failed)}  {failed if failed else ''}")

if failed:
    print("\n  ✗ PIPELINE FAILED — check output above")
    sys.exit(1)
elif DRY_RUN:
    print("\n  (dry-run complete — no cells executed)")
else:
    print("\n  ✓ ALL CELLS PASSED")
    sys.exit(0)
