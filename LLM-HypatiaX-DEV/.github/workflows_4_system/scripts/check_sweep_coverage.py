#!/usr/bin/env python3
"""Report shard coverage for suppB (noise-sweep) / suppB_sc (sample-complexity).

Used by ci_postprocess.yml's figures_deploy step to decide whether the
Group-B sweep figures (fig1_r2_vs_noise ... fig_comparative_table) can be
generated as real sweeps/curves, or will be degenerate (single point) because
only one shard / one noise-level / one sample-size is present.

ci_analysis.yml does NOT merge suppB/suppB_sc shards into a single
_merged.json (MERGE_REQUIRED_EXPERIMENTS only covers exp1b/exp3b), so whatever
per-worker shard files are committed in the canonical subdir are exactly what
scripts/generate_figures.py --source auto will see.

Usage:
    python3 .github/scripts/check_sweep_coverage.py --dir <results_dir> --mode noise
    python3 .github/scripts/check_sweep_coverage.py --dir <results_dir> --mode sample_complexity

Prints three space-separated integers to stdout:
    N_FILES N_DISTINCT_AXIS_POINTS N_METHODS

- mode=noise:             globs noise_sweep_*.json, axis key = "noise_levels"
- mode=sample_complexity: globs sc_n*.json / *sample_complexity*.json,
                           axis key = "sample_sizes"

In both modes, "methods" is the union of the shard's "methods" list — used to
flag when fig_runtime_comparison / fig_comparative_table (expect 6 methods)
will be incomplete from suppB/suppB_sc data alone.
"""
import argparse
import glob
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="Canonical results directory to scan")
    parser.add_argument(
        "--mode", required=True, choices=["noise", "sample_complexity"],
        help="noise -> noise_sweep_*.json / noise_levels; "
             "sample_complexity -> sc_n*.json|*sample_complexity*.json / sample_sizes",
    )
    args = parser.parse_args()

    if args.mode == "noise":
        files = glob.glob(os.path.join(args.dir, "noise_sweep_*.json"))
        axis_key = "noise_levels"
    else:
        files = (
            glob.glob(os.path.join(args.dir, "sc_n*.json"))
            + glob.glob(os.path.join(args.dir, "*sample_complexity*.json"))
        )
        axis_key = "sample_sizes"

    files = sorted(set(files))

    axis_points: set = set()
    methods: set = set()
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:
            continue
        axis_points.update(d.get(axis_key, []))
        methods.update(d.get("methods", []))

    print(len(files), len(axis_points), len(methods))
    return 0


if __name__ == "__main__":
    sys.exit(main())
