import argparse
import os
import pathlib
import sys

# ── sys.path bootstrap ────────────────────────────────────────────────────
# Works whether this file lives at:
#   <repo>/hypatiax/experiments/benchmarks/   (old layout)
#   <repo>/hypatiax/protocols/                (new layout)
# Repo root is resolved via REPRO_ROOT env var (set by wrapper) or inferred.
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT  = pathlib.Path(os.environ.get("REPRO_ROOT", str(_SCRIPT_DIR.parents[2])))

for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "hypatiax")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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

    # Resolve the final list of seeds to run.
    # Priority: --seeds > --seed > env var EXPERIMENT_SEED > default (42)
    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [int(os.environ.get("EXPERIMENT_SEED", 42))]

    results = []
    for seed in seeds:
        # Propagate seed into env so downstream modules pick it up
        os.environ["EXPERIMENT_SEED"] = str(seed)
        os.environ["NN_SEED"]         = str(seed)
        os.environ["PYSR_SEED"]       = str(seed)

        # Forward --n-tasks (smoke-test) and PySR params via env so the
        # benchmark script reads them with os.environ.get().
        if args.n_tasks is not None:
            os.environ["N_NGUYEN_TASKS"] = str(args.n_tasks)

        config = {
            "name": "nguyen12_exp3",
            "seed": seed,
        }
        result = run_protocol(config, run_task)
        results.append(result)

    # Return single result for single-seed runs (backwards compatible),
    # list for multi-seed runs.
    return results[0] if len(results) == 1 else results


if __name__ == "__main__":
    run()
