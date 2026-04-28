import argparse
import os as _os
import pathlib as _pathlib
import sys as _sys

# ── sys.path bootstrap: ensures hypatiax.* imports resolve ────────────────
_PROTO_DIR = _pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _pathlib.Path(_os.environ.get("REPRO_ROOT", str(_PROTO_DIR.parent)))
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "hypatiax")]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _pathlib, _PROTO_DIR, _REPO_ROOT, _p

# ── [PATCH G] Import with pre/post-restructure fallback ──────────────────
try:
    from hypatiax.protocols.universal_protocol import run_protocol
except ImportError:
    from protocols.universal_protocol import run_protocol

try:
    from hypatiax.core.runners.common import run_task
except ImportError:
    from core.runners.common import run_task


def parse_args():
    parser = argparse.ArgumentParser(description="Nguyen-12 Exp 3 protocol")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed", type=int, default=None, metavar="N",
        help="Run with a single seed (e.g. --seed 42)",
    )
    seed_group.add_argument(
        "--seeds", type=int, nargs="+", metavar="N",
        help="Run with multiple seeds (e.g. --seeds 99 123 777 2024)",
    )
    parser.add_argument(
        "--n-tasks", type=int, default=None, metavar="N",
        help="Limit number of Nguyen equations to run (e.g. --n-tasks 1 for smoke-test)",
    )
    return parser.parse_args()


def run():
    args = parse_args()

    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [int(_os.environ.get("EXPERIMENT_SEED", 42))]

    results = []
    for seed in seeds:
        _os.environ["EXPERIMENT_SEED"] = str(seed)
        _os.environ["NN_SEED"]         = str(seed)
        _os.environ["PYSR_SEED"]       = str(seed)

        if args.n_tasks is not None:
            _os.environ["N_NGUYEN_TASKS"] = str(args.n_tasks)

        config = {
            "name": "nguyen12_exp3",
            "seed": seed,
        }
        result = run_protocol(config, run_task)
        results.append(result)

    return results[0] if len(results) == 1 else results


if __name__ == "__main__":
    run()

    
