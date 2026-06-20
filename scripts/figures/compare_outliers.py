#!/usr/bin/env python3
"""
compare_outliers.py

For each of the 13 "outlier" base-name groups left over from
cleanup_figures.py, compares the kept root-level file against its
figures__figures__figures__ counterpart and reports:

  - file size difference
  - whether pixel data is identical (only metadata/compression differs)
    or whether the actual image content differs

Requires Pillow: pip install Pillow --break-system-packages

Usage:
    cd ~/Downloads/GITHUB/LLM-HypatiaX-DEV/figures
    python3 compare_outliers.py
"""

import sys
from pathlib import Path

try:
    from PIL import Image, ImageChops
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


def compare(root: Path, base: str):
    a_path = root / base
    b_path = root / f"figures__figures__figures__{base}"

    if not a_path.exists() or not b_path.exists():
        print(f"  [skip] missing file for {base}")
        return

    a_size = a_path.stat().st_size
    b_size = b_path.stat().st_size

    try:
        img_a = Image.open(a_path).convert("RGBA")
        img_b = Image.open(b_path).convert("RGBA")
    except Exception as e:
        print(f"  [error opening] {base}: {e}")
        return

    if img_a.size != img_b.size:
        print(f"{base}")
        print(f"   sizes differ: {a_size}B vs {b_size}B | "
              f"dimensions {img_a.size} vs {img_b.size}  -> DIFFERENT IMAGE (keep both, inspect)")
        return

    diff = ImageChops.difference(img_a, img_b)
    # Collapse RGBA channels into a single max-per-pixel band so bbox/extrema
    # correctly reflect "is any channel different here", regardless of mode.
    bands = diff.split()
    combined = bands[0]
    for b in bands[1:]:
        combined = ImageChops.lighter(combined, b)
    bbox = combined.getbbox()

    print(f"{base}")
    if bbox is None:
        print(f"   pixel-identical, file sizes {a_size}B vs {b_size}B "
              f"-> just re-compression/metadata. Safe to delete the figures__figures__figures__ copy.")
    else:
        max_diff = combined.getextrema()[1]
        # Fraction of pixels that actually differ, for a quick sense of scale
        hist = combined.histogram()
        diff_pixels = sum(hist[1:])  # bucket 0 = unchanged pixels
        total_pixels = combined.size[0] * combined.size[1]
        pct = 100 * diff_pixels / total_pixels
        print(f"   pixel data differs (max channel delta: {max_diff}, "
              f"{pct:.1f}% of pixels affected), file sizes {a_size}B vs {b_size}B, "
              f"diff bbox {bbox} -> ACTUAL CONTENT DIFFERENCE. "
              f"Open both and compare visually before deleting.")


def main():
    root = Path(".").resolve()
    print(f"Comparing {len(PAIRS)} outlier pairs in {root}\n")
    for base in PAIRS:
        compare(root, base)
        print()


if __name__ == "__main__":
    main()
