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

import hashlib
import json
import os
import time

RESULTS_DIR = os.environ.get("HYPATIAX_RESULTS", "hypatiax/data/results")

def hash_config(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

def save_results(results, config):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    h = hash_config(config)
    path = os.path.join(RESULTS_DIR, f"{config['name']}_{h[:8]}.json")
    with open(path, "w") as f:
        json.dump({"config": config, "results": results}, f, indent=2)
    return path

def reproducibility_lock(config):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    h = hash_config(config)
    lock_file = os.path.join(RESULTS_DIR, f".lock_{h}")
    if os.path.exists(lock_file):
        print("Skipping (already run):", config["name"])
        return False
    with open(lock_file, "w") as f:
        f.write(str(time.time()))
    return True

if __name__ == "__main__":
    pass
