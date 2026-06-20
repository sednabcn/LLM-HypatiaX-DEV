#!/usr/bin/env python3
"""
make_side_by_side.py

Builds a single PNG per pair showing root (left) vs nested (right) of
each of the 13 outlier figures, scaled to a common height so they're
easy to eyeball side by side. Saves into ./_side_by_side_check/.

Usage:
    cd ~/Downloads/GITHUB/LLM-HypatiaX-DEV/figures
    python3 make_side_by_side.py
    # then open ./_side_by_side_check/ in your file browser / image viewer
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

TARGET_H = 500
PAD = 12
LABEL_H = 30


def load_scaled(path, target_h):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    new_w = int(w * target_h / h)
    return img.resize((new_w, target_h))


def main():
    root = Path(".").resolve()
    out_dir = root / "_side_by_side_check"
    out_dir.mkdir(exist_ok=True)

    for base in PAIRS:
        a_path = root / base
        b_path = root / f"figures__figures__figures__{base}"
        if not a_path.exists() or not b_path.exists():
            print(f"[skip] {base}")
            continue

        a = load_scaled(a_path, TARGET_H)
        b = load_scaled(b_path, TARGET_H)

        canvas_w = a.width + b.width + PAD * 3
        canvas_h = TARGET_H + LABEL_H + PAD * 2
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        draw.text((PAD, PAD), "ROOT (left)", fill=(0, 0, 0))
        draw.text((PAD + a.width + PAD, PAD), "NESTED / figures^3 (right)", fill=(0, 0, 0))

        canvas.paste(a, (PAD, LABEL_H + PAD))
        canvas.paste(b, (PAD + a.width + PAD, LABEL_H + PAD))

        out_path = out_dir / f"compare_{base}"
        canvas.save(out_path)
        print(f"wrote {out_path}")

    print(f"\nDone. Open the images in {out_dir} to compare visually.")


if __name__ == "__main__":
    main()
