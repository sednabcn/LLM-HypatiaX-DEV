#!/usr/bin/env python3
"""
generate_figures.py — Invoke the HypatiaX visualisation pipeline.

Usage (called by ci_postprocess.yml):
  python scripts/generate_figures.py \
      --experiment  <id> \
      --results-dir hypatiax/data/results/<subdir> \
      --figures-dir hypatiax/data/results/<subdir>/figures \
      --source      auto

plot_results.py ignores --figures-dir and always writes its output to a
hardcoded path (hypatiax/tools/figures/results.pdf).  This wrapper:

  1. Runs plot_results.py (forwarding all args so future fixes work for free).
  2. Finds every PDF/PNG it wrote under its hardcoded output root.
  3. Moves + renames each file to:
       <figures-dir>/FIG_<subject>_<experiment>.pdf
     where <subject> is derived from the filename plot_results.py produced.
  4. Creates <figures-dir> if it does not already exist.
  5. Exits non-zero if no files were moved (silent no-op detection).

Naming convention:  FIG_<subject>_<experiment_id>.pdf
  e.g.  FIG_results_exp1b.pdf
        FIG_noise_sweep_suppB.pdf
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# plot_results.py always writes here regardless of --figures-dir.
HARDCODED_OUT_DIR = Path("hypatiax/tools/figures")

# Extensions we care about.
FIG_EXTENSIONS = {".pdf", ".png"}


def parse_our_args():
    """Extract --experiment and --figures-dir without consuming the full argv."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--experiment",  default="unknown")
    p.add_argument("--figures-dir", default=None)
    known, _ = p.parse_known_args()
    return known.experiment, known.figures_dir


def subject_from_filename(stem: str) -> str:
    """
    Derive a short subject token from the stem plot_results.py used.
    'results'        -> 'results'
    'noise_sweep'    -> 'noise_sweep'
    'results_suppB'  -> 'results'   (experiment suffix already present; strip it)
    """
    # Strip any trailing experiment-like suffix plot_results might have added.
    stem = re.sub(r"[_-](exp\w+|suppB\w*|hybrid\w*|instability|extrap)$", "", stem)
    return stem or "results"


def collect_outputs(out_dir: Path) -> list[Path]:
    """Return all PDF/PNG files directly under out_dir."""
    if not out_dir.is_dir():
        return []
    return [
        f for f in out_dir.iterdir()
        if f.is_file() and f.suffix.lower() in FIG_EXTENSIONS
    ]


def main():
    experiment, figures_dir = parse_our_args()

    if figures_dir is None:
        print("::error::--figures-dir is required", file=sys.stderr)
        sys.exit(1)

    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    # ── Snapshot what already exists under the hardcoded output dir ────────────
    before = set(collect_outputs(HARDCODED_OUT_DIR))

    # ── Run plot_results.py (forward all args verbatim) ────────────────────────
    cmd = [sys.executable, "hypatiax/tools/visualizations/plot_results.py"] + sys.argv[1:]
    print("Generating figures...")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            f"::error::plot_results.py exited with code {result.returncode}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    # ── Collect new files written by plot_results.py ───────────────────────────
    after  = set(collect_outputs(HARDCODED_OUT_DIR))
    new_files = after - before

    if not new_files:
        # Fallback: if plot_results.py overwrote an existing file in-place the
        # set-diff is empty.  Grab everything present (best effort).
        new_files = after

    if not new_files:
        print(
            f"::error::plot_results.py exited 0 but wrote no figures to {HARDCODED_OUT_DIR}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Move + rename into the canonical figures dir ───────────────────────────
    # Naming: FIG_<subject>_<experiment><ext>
    # Multiple outputs (e.g. page-split PDFs): FIG_<subject>_<n>_<experiment><ext>
    moved = []
    for i, src in enumerate(sorted(new_files), start=1):
        subject = subject_from_filename(src.stem)
        suffix  = src.suffix.lower()
        if len(new_files) == 1:
            dest_name = f"FIG_{subject}_{experiment}{suffix}"
        else:
            dest_name = f"FIG_{subject}_{i}_{experiment}{suffix}"
        dest = figures_path / dest_name
        shutil.move(str(src), str(dest))
        print(f"  ✓ {src} → {dest}")
        moved.append(dest)

    print(f"Figures OK: {len(moved)} file(s) written to {figures_path}")


if __name__ == "__main__":
    main()
