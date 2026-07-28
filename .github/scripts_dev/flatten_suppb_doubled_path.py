#!/usr/bin/env python3
"""flatten_suppb_doubled_path.py — one-off cleanup for the suppB doubled-path bug.

BACKGROUND
----------
run_noise_sweep_benchmark.py did not honor the OUT_BASE env var for its
output directory: it wrote relative to its own CWD using the same
"comparison_results/feynman-tests/noise-sweep/noise-sweep" suffix that
OUT_BASE already encoded, producing a self-nested duplicate tree inside the
canonical suppB subdirectory (SUPPB_SUBDIR):

    canonical:  comparison_results/feynman-tests/noise-sweep/noise-sweep/
                    noise_sweep_*.json

    doubled:    comparison_results/feynman-tests/noise-sweep/noise-sweep/
                    comparison_results/feynman-tests/noise-sweep/
                        noise_sweep_*.json

run_all.sh STEP 10 (FIX-suppB-DOUBLED-PATH) now detects and flattens this
automatically on every new suppB run, mirroring the equivalent suppB_sc fix
in STEP 10b. This script is the one-off equivalent for cleaning up the
doubled tree that was already committed to the repo BEFORE that fix landed,
since the run_all.sh fix only prevents new doubling — it does not retroactively
clean already-committed history.

generate_figures.py's recursive glob (_latest_glob, searching `**` under
--results-dir) has been finding files inside the doubled tree all along, so
existing CI runs have NOT been silently broken — but the duplicate tree is
repo bloat, is confusing to read, and risks `generate_tables.py` or future
scripts picking up the wrong file if their own search logic is less robust.

WHAT THIS SCRIPT DOES
----------------------
1. Locates the canonical SUPPB_SUBDIR (default matches ci_postprocess.yml /
   ci_analysis.yml / run_all.sh: comparison_results/feynman-tests/noise-sweep/
   noise-sweep).
2. Searches inside it (one level of nesting expected, but the search is
   recursive in case of deeper-than-expected duplication) for a doubled
   subtree matching */noise-sweep/noise-sweep/comparison_results/feynman-tests
   /noise-sweep.
3. For each noise_sweep_*.json / noise_sweep_*.csv found inside the doubled
   tree:
     - If no file of the same name exists at the canonical location, move it
       there.
     - If a file of the same name DOES already exist canonically, the doubled
       copy is left in place and reported as a [CONFLICT] — never silently
       overwritten or deleted. Resolve conflicts by hand (compare contents;
       the canonical copy is presumed authoritative since it's the one CI has
       been reading via generate_tables.py's flat-path search).
4. After a successful (no-conflict) rescue, removes the now-empty doubled
   directory tree.
5. Defaults to DRY RUN — prints every planned action but makes no changes.
   Pass --apply to actually move files and remove the empty directory tree.

USAGE
-----
    # Dry run (default) — see what would happen, change nothing:
    python3 .github/scripts/flatten_suppb_doubled_path.py

    # Actually perform the flatten:
    python3 .github/scripts/flatten_suppb_doubled_path.py --apply

    # Custom results root / canonical subdir (defaults match the CI configs):
    python3 .github/scripts/flatten_suppb_doubled_path.py \\
        --results-root hypatiax/data/results \\
        --canonical-subdir comparison_results/feynman-tests/noise-sweep/noise-sweep \\
        --apply

Mirrors the conventions of clean_figures_dir.py (dry-run-by-default CLI,
--apply flag, [TAG]-prefixed stdout lines a caller can grep for).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

DEFAULT_RESULTS_ROOT = "hypatiax/data/results"
DEFAULT_CANONICAL_SUBDIR = "comparison_results/feynman-tests/noise-sweep/noise-sweep"
# The doubled tree is the canonical subdir's own path suffix, appended again
# one level inside itself.
DOUBLED_SUFFIX = "comparison_results/feynman-tests/noise-sweep"
RESCUE_PATTERNS = ("noise_sweep_",)
RESCUE_EXTS = (".json", ".csv")


def _is_rescue_file(name: str) -> bool:
    return name.startswith(RESCUE_PATTERNS) and name.endswith(RESCUE_EXTS)


def find_doubled_dirs(canonical_dir: str) -> list[str]:
    """Find directories under canonical_dir whose path ends with the doubled
    suffix — i.e. the self-nested duplicate tree(s)."""
    found = []
    norm_suffix = DOUBLED_SUFFIX.replace("/", os.sep)
    for root, dirs, _files in os.walk(canonical_dir):
        if root == canonical_dir:
            continue
        if root.rstrip(os.sep).endswith(norm_suffix):
            found.append(root)
            # Don't descend further into a confirmed doubled tree — its own
            # subdirs (if any) are part of the same artifact, not separate
            # doubled trees to report individually.
            dirs[:] = []
    return sorted(found)


def flatten(results_root: str, canonical_subdir: str, apply: bool) -> int:
    canonical_dir = os.path.abspath(os.path.join(results_root, canonical_subdir))

    if not os.path.isdir(canonical_dir):
        print(f"  [SKIP] canonical dir does not exist: {canonical_dir}")
        return 0

    doubled_dirs = find_doubled_dirs(canonical_dir)
    if not doubled_dirs:
        print(f"  [OK] no doubled-path directory found under {canonical_dir}")
        print("       (already flat, or run_all.sh FIX-suppB-DOUBLED-PATH already cleaned it)")
        return 0

    n_moved = 0
    n_conflict = 0
    n_skipped_other = 0

    for doubled_dir in doubled_dirs:
        print(f"  [FOUND] doubled tree: {doubled_dir}")
        had_conflict_in_this_tree = False

        for entry in sorted(os.listdir(doubled_dir)):
            src = os.path.join(doubled_dir, entry)
            if not os.path.isfile(src):
                continue
            if not _is_rescue_file(entry):
                n_skipped_other += 1
                print(f"    [SKIP-OTHER] not a noise_sweep_* file, leaving in place: {entry}")
                continue

            dst = os.path.join(canonical_dir, entry)
            if os.path.exists(dst):
                n_conflict += 1
                had_conflict_in_this_tree = True
                same_size = os.path.getsize(src) == os.path.getsize(dst)
                note = "same size" if same_size else "DIFFERENT size"
                print(f"    [CONFLICT] {entry} already exists canonically ({note}) — "
                      f"leaving doubled copy at {src}; resolve by hand.")
                continue

            print(f"    [{'MOVE' if apply else 'WOULD-MOVE'}] {entry}  "
                  f"{doubled_dir} -> {canonical_dir}")
            if apply:
                shutil.move(src, dst)
            n_moved += 1

        if had_conflict_in_this_tree:
            print(f"  [LEAVE] {doubled_dir} retained — unresolved conflict(s) above.")
            continue

        remaining = os.listdir(doubled_dir) if os.path.isdir(doubled_dir) else []
        if remaining:
            print(f"  [LEAVE] {doubled_dir} retained — non-rescue file(s) still inside: "
                  f"{remaining}")
            continue

        print(f"  [{'RMTREE' if apply else 'WOULD-RMTREE'}] {doubled_dir} (now empty)")
        if apply:
            # Walk back up removing now-empty parent dirs, stopping at
            # canonical_dir itself.
            cur = doubled_dir
            while cur != canonical_dir and os.path.isdir(cur) and not os.listdir(cur):
                parent = os.path.dirname(cur)
                os.rmdir(cur)
                cur = parent

    print()
    print(f"  Summary: {n_moved} file(s) {'moved' if apply else 'would be moved'}, "
          f"{n_conflict} conflict(s), {n_skipped_other} non-rescue file(s) left untouched.")
    if not apply and (n_moved or doubled_dirs):
        print("  (dry run — pass --apply to actually move files and remove empty dirs)")

    return 1 if n_conflict else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT,
                         help=f"Results root dir (default: {DEFAULT_RESULTS_ROOT})")
    parser.add_argument("--canonical-subdir", default=DEFAULT_CANONICAL_SUBDIR,
                         help=f"Canonical suppB subdir relative to --results-root "
                              f"(default: {DEFAULT_CANONICAL_SUBDIR})")
    parser.add_argument("--apply", action="store_true",
                         help="Actually move files and remove empty doubled dirs. "
                              "Without this flag, only prints what would happen.")
    args = parser.parse_args()

    print(f"=== flatten_suppb_doubled_path.py {'(APPLY)' if args.apply else '(DRY RUN)'} ===")
    print(f"  results-root      : {args.results_root}")
    print(f"  canonical subdir  : {args.canonical_subdir}")
    print()

    rc = flatten(args.results_root, args.canonical_subdir, args.apply)
    if rc:
        print()
        print("::warning::Unresolved conflicts found — see [CONFLICT] lines above. "
              "Doubled files were left in place; nothing was deleted.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
