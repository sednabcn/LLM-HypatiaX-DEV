import os as _os
import pathlib as _pathlib
import sys as _sys

# ── sys.path bootstrap ────────────────────────────────────────────────────
# Ensures hypatiax.* imports resolve whether this file is run directly
# or imported by run_all_checkpoint.py.
_PROTO_DIR  = _pathlib.Path(__file__).resolve().parent
_REPO_ROOT  = _pathlib.Path(_os.environ.get("REPRO_ROOT", str(_PROTO_DIR.parent)))
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "hypatiax")]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _os, _pathlib, _sys, _PROTO_DIR, _REPO_ROOT, _p

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
    config = {"name": "noise_sweep"}
    return run_protocol(config, run_task)

if __name__ == "__main__":
    run()
