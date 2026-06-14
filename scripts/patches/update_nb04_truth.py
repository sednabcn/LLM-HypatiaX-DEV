"""
update_nb04_truth.py  —  Sync NB-04 TRUTH dict from live summary JSONs.

Reads exp2_pca_4060_summary.json and exp1_pca_summary.json, then patches
the TRUTH dict inside the NB-04 notebook so numerical-consistency checks
use the corrected FIX-C3 figures.

Usage:
    python scripts/patches/update_nb04_truth.py \
        --results-dir hypatiax/data/results/comparison_results \
        --nb-path     notebooks/NB-04_Numerical_Consistency_Checker.ipynb \
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def load_summary(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_truth_values(exp2_summary: dict, exp1_summary: dict) -> dict:
    """Pull the canonical figures that should appear in the TRUTH dict."""
    return {
        # Feynman PCA 40/60 corrected result
        "feynman_successes":  exp2_summary["successes"],
        "feynman_total":      exp2_summary["total"],
        "feynman_solve_rate": round(exp2_summary["solve_rate"], 4),
        "feynman_protocol":   exp2_summary.get("protocol", "pca_40_60"),
        # DeFi PCA all-74 result
        "defi_successes":     exp1_summary["successes"],
        "defi_total":         exp1_summary["total"],
        "defi_solve_rate":    round(exp1_summary["solve_rate"], 4),
    }


def patch_notebook(nb_path: Path, truth: dict, dry_run: bool) -> bool:
    """Patch the TRUTH dict literal inside NB-04's source cells."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    changed = False

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "TRUTH" not in src or "feynman_successes" not in src:
            continue

        new_src = src
        for key, value in truth.items():
            pattern = rf'(["\']){re.escape(key)}\1\s*:\s*[^\n,}}\]]+'
            replacement = f'"{key}": {json.dumps(value)}'
            new_src, n = re.subn(pattern, replacement, new_src)
            if n:
                print(f"  Updated {key} → {json.dumps(value)}")

        if new_src != src:
            cell["source"] = list(new_src)
            changed = True

    if changed and not dry_run:
        nb_path.write_text(
            json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"Written: {nb_path}")
    elif changed and dry_run:
        print(f"[dry-run] Would update: {nb_path}")
    else:
        print("No changes needed.")

    return changed


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Sync NB-04 TRUTH dict from live exp2/exp1 summary JSONs."
    )
    p.add_argument("--results-dir", required=True,
                   help="Path to comparison_results base directory.")
    p.add_argument("--nb-path", required=True,
                   help="Path to NB-04_Numerical_Consistency_Checker.ipynb.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would change without writing.")
    args = p.parse_args(argv[1:])

    results_dir = Path(args.results_dir)
    exp2_path = results_dir / "feynman-tests" / "exp2_pca_4060" / "exp2_pca_4060_summary.json"
    exp1_path = results_dir / "noise-noiseless" / "noiseless" / "defi_pca" / "exp1_pca_summary.json"

    for path in (exp2_path, exp1_path):
        if not path.exists():
            print(f"ERROR: {path} not found — run C5c experiments first.",
                  file=sys.stderr)
            return 1

    exp2 = load_summary(exp2_path)
    exp1 = load_summary(exp1_path)
    truth = extract_truth_values(exp2, exp1)

    print("Truth values from summary JSONs:")
    for k, v in truth.items():
        print(f"  {k}: {v}")
    print()

    nb_path = Path(args.nb_path)
    if not nb_path.exists():
        print(f"ERROR: {nb_path} not found.", file=sys.stderr)
        return 1

    patch_notebook(nb_path, truth, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
