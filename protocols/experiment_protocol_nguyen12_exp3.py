import argparse

from protocols.universal_protocol import run_protocol
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
    return parser.parse_args()


def run():
    args = parse_args()

    # Resolve the final list of seeds to run.
    # Priority: --seeds > --seed > default (42)
    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42]

    results = []
    for seed in seeds:
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
