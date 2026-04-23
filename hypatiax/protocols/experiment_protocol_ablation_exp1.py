import os as _os

import pathlib as _pathlib
import sys as _sys

# ── sys.path bootstrap ────────────────────────────────────────────────────
# Ensures hypatiax.* imports resolve whether this file is run directly
# or imported by run_all_checkpoint.py.
_PROTO_DIR  = _pathlib.Path(__file__).resolve().parent
_REPO_ROOT  = _pathlib.Path( _os.environ.get("REPRO_ROOT", str(_PROTO_DIR.parent)))
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "hypatiax")]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _pathlib, _PROTO_DIR, _REPO_ROOT, _p   # keep _os and _sys — used in run()

# ── [PATCH G] Import with pre/post-restructure fallback ──────────────────
try:
    from hypatiax.protocols.universal_protocol import run_protocol
except ImportError:
    from protocols.universal_protocol import run_protocol

try:
    from hypatiax.core.runners.common import run_task
except ImportError:
    from core.runners.common import run_task

def run():
    # Read seed from env — set by run_all.py and passed to every child process.
    # Falls back to 42 so a direct `python3 experiment_protocol_ablation_exp1.py`
    # still works without any env setup.
    seed = int(_os.environ.get("PYSR_SEED", _os.environ.get("NN_SEED", 42)))

    # --one-equation smoke-test: honour N_CORE15_TASKS (set by run_all_checkpoint.py)
    # or the generic ONE_EQUATION flag.  Defaults to all 15 equations.
    _one_eq   = _os.environ.get("ONE_EQUATION", "0") == "1"
    _n_core15 = _os.environ.get("N_CORE15_TASKS")
    if _n_core15 is not None:
        n_equations = int(_n_core15)
    elif _one_eq:
        n_equations = 1
    else:
        n_equations = 15   # full Core-15 ablation

    config = {
        "name":        "ablation_exp1",
        "seed":        seed,          # propagated into run_task → SCRIPT_MAP target
        "n_equations": n_equations,   # run_protocol/run_task slices CORE_15[:n_equations]
    }
    result = run_protocol(config, run_task)

    # The orchestrator (this script) decides whether to propagate failure to
    # the shell.  run_protocol itself never calls sys.exit, so partial results
    # from other configs in the same process are never lost.
    if not result.get("success", False):
        reason = result.get("reason") or result.get("error") or "unknown"
        print(f"[FATAL] ablation_exp1 failed: {reason}", file=_sys.stderr)
        _sys.exit(1)

    return result


if __name__ == "__main__":
    run()
