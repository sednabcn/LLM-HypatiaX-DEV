#!/usr/bin/env python3
"""
rename_nested_v2.py

CONTEXT: the 13 "figures__figures__figures__*" files are NOT duplicates
of their root-named counterparts - visual inspection confirmed they are
a completely different figure set (HypatiaX symbolic-regression results
vs. the root set's DeFi/LLM instability analysis) that collided on
filename during a flattening export.

This script renames each figures__figures__figures__X.png to X_v2.png
(in place, same directory), so both figure sets survive under distinct,
readable names. Root-named files are left untouched.

Safety:
  - Refuses to overwrite an existing file.
  - Dry run by default; pass --apply to actually rename.

Usage:
    cd ~/Downloads/GITHUB/LLM-HypatiaX-DEV/figures
    python3 rename_nested_v2.py            # dry run, prints the plan
    python3 rename_nested_v2.py --apply    # actually renames
"""

import argparse
import sys
from pathlib import Path

PAIRS = [
    "fig_instability_3d.png",
    "fig_instability_hist.png",
    "fig_instability_phase.png",
    "fig_instability_regimes.png",
    "fig_instability_success_vs_instability.png",
    "fig_paper_complexity_vs_instability.png",
    "fig_paper_complexity_vs_success.png",
    "fig_paper_instability_hist.png",
    "fig_paper_mean_vs_instability.png",
    "fig_paper_regime_counts.png",
    "hypatiax_instability_histogram.png",
    "hypatiax_instability_scatter.png",
    "hypatiaX_three_systems.png",
]


def v2_name(base: str) -> str:
    stem = base.rsplit(".", 1)[0]
    ext = base.rsplit(".", 1)[1]
    return f"{stem}_v2.{ext}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                     help="Actually rename files. Without this flag, only prints the plan.")
    ap.add_argument("--dir", default=".", help="Directory to operate in (default: cwd)")
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    plan = []  # (src, dst)
    missing = []
    conflicts = []

    for base in PAIRS:
        src = root / f"figures__figures__figures__{base}"
        dst = root / v2_name(base)

        if not src.exists():
            missing.append(src)
            continue
        if dst.exists():
            conflicts.append((src, dst))
            continue
        plan.append((src, dst))

    print(f"Operating in {root}\n")

    print("=" * 70)
    print(f"RENAME PLAN ({len(plan)} files)")
    print("=" * 70)
    for src, dst in plan:
        print(f"  {src.name}  ->  {dst.name}")

    if missing:
        print()
        print(f"MISSING ({len(missing)}) - source file not found, skipped:")
        for m in missing:
            print(f"  {m.name}")

    if conflicts:
        print()
        print(f"CONFLICTS ({len(conflicts)}) - destination name already exists, skipped "
              f"(resolve manually):")
        for src, dst in conflicts:
            print(f"  {src.name}  ->  {dst.name}  [dst already exists]")

    print()
    if not args.apply:
        print(f"Dry run only. {len(plan)} files would be renamed. "
              "Re-run with --apply to actually rename.")
        return

    renamed = 0
    for src, dst in plan:
        try:
            src.rename(dst)
            renamed += 1
        except OSError as e:
            print(f"  ! failed to rename {src.name}: {e}", file=sys.stderr)

    print(f"Renamed {renamed}/{len(plan)} files.")
    if conflicts:
        print(f"{len(conflicts)} conflicts were left untouched - review those by hand.")


if __name__ == "__main__":
    main()
