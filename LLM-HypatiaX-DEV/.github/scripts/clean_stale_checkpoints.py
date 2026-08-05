#!/usr/bin/env python3
"""
clean_stale_checkpoints.py — quarantine stale crash-rescue checkpoint files.

Background
----------
ci_runner.yml's FIX-G5 safety net writes a crash-rescue checkpoint
(_checkpoint_shard*.json) straight into an experiment's RESULTS_DIR whenever
a worker crashes mid-run. For "instability" this directory is
"${OUT_BASE}/figures" (RESULT_SUBDIR=figures for that experiment).

Once a later run succeeds and produces the *real* final output for that
experiment (for instability: instability_analysis.csv and/or
instability_extrapolation.csv), any leftover _checkpoint_shard*.json file
in the same directory is stale dead weight — the run it was rescuing has
since completed successfully. This script finds and quarantines those
stale checkpoints.

Design choices (mirrors clean_figures_dir.py's convention):
  - Never deletes. Matched files are *moved* into a "_stale_checkpoints_removed/"
    subdirectory under the target dir, so nothing is destroyed and a human
    can always recover a file if the heuristic was wrong.
  - "Stale" is defined conservatively: a checkpoint is only quarantined if
    at least one real final-output file (instability_analysis.csv or
    instability_extrapolation.csv) already exists alongside it. If no
    final output is present yet, the checkpoint might still be needed by
    an in-flight/rescued run, so it is left alone.
  - Dry-run by default. Nothing is moved unless --apply is passed. Without
    --apply, the script only reports what it *would* do.
  - Always exits 0. This is a best-effort cleanup step, not a gate — a
    partial or no-op cleanup should never fail the CI job that calls it.

Usage
-----
  python3 clean_stale_checkpoints.py <target_dir> [--apply]

Arguments
---------
  target_dir   Directory to scan (e.g. "${OUT_BASE}/figures" for instability).
  --apply      Actually move stale checkpoints into _stale_checkpoints_removed/.
               Without this flag, the script only prints what it would do.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

CHECKPOINT_GLOB = "_checkpoint_shard*.json"
QUARANTINE_DIRNAME = "_stale_checkpoints_removed"

# Any one of these existing in target_dir is treated as proof that the real
# final output for the run has already landed, making leftover checkpoints
# in the same directory safe to quarantine.
FINAL_OUTPUT_MARKERS = (
    "instability_analysis.csv",
    "instability_extrapolation.csv",
)


def find_checkpoints(target_dir: Path) -> list[Path]:
    return sorted(target_dir.glob(CHECKPOINT_GLOB))


def has_final_output(target_dir: Path) -> bool:
    return any((target_dir / marker).exists() for marker in FINAL_OUTPUT_MARKERS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_dir", type=Path,
                         help="Directory to scan for stale checkpoint files")
    parser.add_argument("--apply", action="store_true",
                         help="Actually quarantine stale checkpoints "
                              "(default: dry-run, report only)")
    args = parser.parse_args()

    target_dir: Path = args.target_dir

    if not target_dir.exists():
        print(f"[clean_stale_checkpoints] {target_dir} does not exist — nothing to do.")
        return 0

    checkpoints = find_checkpoints(target_dir)
    if not checkpoints:
        print(f"[clean_stale_checkpoints] No {CHECKPOINT_GLOB} files found in {target_dir}.")
        return 0

    if not has_final_output(target_dir):
        print(f"[clean_stale_checkpoints] {len(checkpoints)} checkpoint file(s) found in "
              f"{target_dir}, but no final output marker "
              f"({', '.join(FINAL_OUTPUT_MARKERS)}) is present yet.")
        print("[clean_stale_checkpoints] Leaving checkpoint(s) in place — "
              "they may still be needed by an in-flight/rescued run.")
        return 0

    print(f"[clean_stale_checkpoints] {len(checkpoints)} stale checkpoint file(s) found "
          f"in {target_dir} (final output already present):")
    for ckpt in checkpoints:
        print(f"    {ckpt.name}")

    if not args.apply:
        print("[clean_stale_checkpoints] Dry-run only — pass --apply to quarantine "
              f"into {QUARANTINE_DIRNAME}/.")
        return 0

    quarantine_dir = target_dir / QUARANTINE_DIRNAME
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for ckpt in checkpoints:
        dest = quarantine_dir / ckpt.name
        try:
            shutil.move(str(ckpt), str(dest))
            moved += 1
            print(f"    moved {ckpt.name} -> {QUARANTINE_DIRNAME}/{ckpt.name}")
        except OSError as exc:
            print(f"::warning::clean_stale_checkpoints — failed to move {ckpt}: {exc}")

    print(f"[clean_stale_checkpoints] Quarantined {moved}/{len(checkpoints)} "
          f"stale checkpoint file(s) into {quarantine_dir}/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
