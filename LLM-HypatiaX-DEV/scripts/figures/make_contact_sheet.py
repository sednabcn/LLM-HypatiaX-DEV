#!/usr/bin/env python3
"""
make_contact_sheet.py

Stacks all 13 ROOT-vs-NESTED comparison pairs into ONE tall image, so
you only need to upload a single file for review instead of 13.

Reads the already-generated side-by-side images from
./_side_by_side_check/ (created by make_side_by_side.py) and stacks
them vertically, with a header row per pair showing the figure name.

If _side_by_side_check/ doesn't exist yet, run make_side_by_side.py
first.

Output: ./contact_sheet.png (single file, may be tall — that's fine,
just scroll/zoom when viewing).

Usage:
    cd ~/Downloads/GITHUB/LLM-HypatiaX-DEV/figures
    python3 make_contact_sheet.py
"""

from pathlib import Path
from PIL import Image, ImageDraw

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

SECTION_HEADER_H = 40
ROW_GAP = 16
TARGET_ROW_WIDTH = 1400  # rescale each side-by-side image to this width


def main():
    root = Path(".").resolve()
    sbs_dir = root / "_side_by_side_check"

    if not sbs_dir.exists():
        print(f"ERROR: {sbs_dir} not found. Run make_side_by_side.py first.")
        return

    rows = []  # list of (label, PIL.Image)
    for base in PAIRS:
        img_path = sbs_dir / f"compare_{base}"
        if not img_path.exists():
            print(f"[skip] missing {img_path.name}")
            continue
        img = Image.open(img_path).convert("RGB")
        scale = TARGET_ROW_WIDTH / img.width
        new_size = (TARGET_ROW_WIDTH, int(img.height * scale))
        img = img.resize(new_size)
        rows.append((base, img))

    if not rows:
        print("No comparison images found - nothing to build.")
        return

    total_h = sum(SECTION_HEADER_H + img.height + ROW_GAP for _, img in rows) + ROW_GAP
    canvas = Image.new("RGB", (TARGET_ROW_WIDTH, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y = ROW_GAP
    for i, (label, img) in enumerate(rows, start=1):
        draw.rectangle([0, y, TARGET_ROW_WIDTH, y + SECTION_HEADER_H], fill=(30, 30, 30))
        draw.text((12, y + 10), f"{i}. {label}", fill=(255, 255, 255))
        y += SECTION_HEADER_H
        canvas.paste(img, (0, y))
        y += img.height + ROW_GAP
        # separator line
        draw.line([(0, y - ROW_GAP // 2), (TARGET_ROW_WIDTH, y - ROW_GAP // 2)], fill=(200, 200, 200), width=2)

    out_path = root / "contact_sheet.png"
    canvas.save(out_path)
    print(f"Wrote {out_path}  ({canvas.width}x{canvas.height}px, {len(rows)} pairs)")
    print("Upload this single file for review.")


if __name__ == "__main__":
    main()
