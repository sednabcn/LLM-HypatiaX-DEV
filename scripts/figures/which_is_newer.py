#!/usr/bin/env python3
"""
which_is_newer.py

For each of the 13 outlier pairs (root-level file vs the
figures__figures__figures__ triple-nested copy), reports:
  - file modification time for each (which was written to disk more recently)
  - embedded DPI metadata if present in the PNG
  - aspect ratio, to see if the triple-nested set uses a uniform
    figsize/DPI convention (suggesting a later, standardized re-render)

This is informational only - it does NOT delete or move anything.
Use the output to decide which version to keep by hand.

Usage:
    cd ~/Downloads/GITHUB/LLM-HypatiaX-DEV/figures
    python3 which_is_newer.py
"""

import sys
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Pillow not installed. Run: pip install Pillow --break-system-packages")
    sys.exit(1)

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


def info(path: Path):
    stat = path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime)
    try:
        img = Image.open(path)
        dpi = img.info.get("dpi", None)
        size = img.size
        aspect = round(size[0] / size[1], 4)
    except Exception:
        dpi, size, aspect = None, None, None
    return mtime, dpi, size, aspect


def main():
    root = Path(".").resolve()
    print(f"Comparing modification times + DPI for {len(PAIRS)} pairs in {root}\n")

    newer_count = {"root": 0, "nested": 0, "tie": 0}

    for base in PAIRS:
        a_path = root / base
        b_path = root / f"figures__figures__figures__{base}"
        if not a_path.exists() or not b_path.exists():
            print(f"[skip] missing file for {base}")
            continue

        a_mtime, a_dpi, a_size, a_aspect = info(a_path)
        b_mtime, b_dpi, b_size, b_aspect = info(b_path)

        newer = "root" if a_mtime > b_mtime else ("nested" if b_mtime > a_mtime else "tie")
        newer_count[newer] += 1

        print(f"{base}")
        print(f"   root   : mtime={a_mtime}  size={a_size}  aspect={a_aspect}  dpi={a_dpi}")
        print(f"   nested : mtime={b_mtime}  size={b_size}  aspect={b_aspect}  dpi={b_dpi}")
        print(f"   -> more recently modified: {newer}")
        print()

    print("=" * 70)
    print(f"SUMMARY: root newer in {newer_count['root']} pairs, "
          f"nested newer in {newer_count['nested']} pairs, "
          f"tied in {newer_count['tie']} pairs")
    print("=" * 70)
    print("\nNote: mtime reflects when the file was last WRITTEN TO THIS DISK "
          "(e.g. by git checkout, download, or the original export), which "
          "is not always the same as when the figure was originally "
          "generated/rendered upstream. Use this as one signal, not proof.")


if __name__ == "__main__":
    main()
