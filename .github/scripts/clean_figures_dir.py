#!/usr/bin/env python3
"""
clean_figures_dir.py
=====================
One-time cleanup for a local figures/ (or tables/) directory that has
accumulated repeated "<dir>__" prefixes from merging multiple CI artifact
downloads on top of each other, e.g.:

    figures__figures__figures__fig09_r2_heatmap_regimes.png
    Figures__fig09_r2_heatmap_regimes.png
    fig09_r2_heatmap_regimes.png

All three are really the same file under different mangled names. This
script:

  1. Strips any number of leading "<word>__" prefixes (case-insensitive,
     so "figures__" and "Figures__" both count) to recover the canonical
     basename.
  2. Groups files by canonical basename.
  3. Within each group, compares file content via SHA-256:
       - If all copies are byte-identical: keep exactly one (the file with
         the fewest stripped prefixes — i.e. the cleanest name already on
         disk), rename it to the canonical basename if needed, and move
         the rest to a quarantine folder (never deletes outright).
       - If copies differ in content: nothing is auto-resolved. All
         variants are listed under "CONFLICTS" for manual review — the
         script does not guess which one is "correct".
  4. Default mode is --dry-run (prints the plan only). Pass --apply to
     actually move/rename files on disk.

Usage
-----
    python clean_figures_dir.py /path/to/figures            # dry-run
    python clean_figures_dir.py /path/to/figures --apply     # do it
    python clean_figures_dir.py /path/to/figures --apply --delete-duplicates
        # instead of quarantining duplicates, delete them outright
        # (only use after reviewing a dry-run / quarantine pass first)
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PREFIX_RE = re.compile(r'^[A-Za-z0-9_-]+__')


def canonical_name(filename: str) -> tuple[str, int]:
    """
    Strip repeated '<word>__' prefixes off a filename.
    Returns (canonical_basename, num_prefixes_stripped).
    """
    name = filename
    count = 0
    while True:
        m = PREFIX_RE.match(name)
        if not m:
            break
        name = name[m.end():]
        count += 1
    return name, count


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", help="Path to the mangled figures/ (or tables/) directory")
    p.add_argument("--apply", action="store_true",
                    help="Actually move/rename files. Without this, only prints the plan.")
    p.add_argument("--delete-duplicates", action="store_true",
                    help="Delete duplicate copies instead of quarantining them "
                         "(requires --apply; use only after reviewing a dry-run first).")
    p.add_argument("--quarantine-dir", default="_duplicates_removed",
                    help="Subfolder (relative to directory) where duplicates are "
                         "moved instead of deleted. Default: _duplicates_removed")
    args = p.parse_args()

    target = Path(args.directory).resolve()
    if not target.is_dir():
        print(f"ERROR: {target} is not a directory", file=sys.stderr)
        return 1

    quarantine = target / args.quarantine_dir

    # Only look at files directly inside target (matches the flat layout
    # shown in the original listing). Skip the quarantine dir itself if
    # this script is re-run.
    files = [f for f in target.iterdir()
             if f.is_file() and f.parent.name != args.quarantine_dir]

    groups: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        canon, _ = canonical_name(f.name)
        groups[canon].append(f)

    n_clean = 0
    n_dupe_groups = 0
    n_dupes_resolved = 0
    n_conflicts = 0
    actions: list[tuple[str, Path, Path | None]] = []  # (kind, src, dest)

    for canon, paths in sorted(groups.items()):
        if len(paths) == 1:
            f = paths[0]
            if f.name != canon:
                # Single file but still has a stray prefix — just rename it.
                actions.append(("rename", f, target / canon))
            else:
                n_clean += 1
            continue

        # Multiple files map to the same canonical name — check content.
        hashes = {f: sha256_of(f) for f in paths}
        distinct_hashes = set(hashes.values())

        if len(distinct_hashes) == 1:
            # All identical content — keep the one with fewest stripped
            # prefixes (cleanest existing name), quarantine the rest.
            paths_sorted = sorted(paths, key=lambda f: canonical_name(f.name)[1])
            keeper = paths_sorted[0]
            losers = paths_sorted[1:]

            if keeper.name != canon:
                actions.append(("rename", keeper, target / canon))
            for loser in losers:
                actions.append(("quarantine", loser, quarantine / loser.name))

            n_dupe_groups += 1
            n_dupes_resolved += len(losers)
        else:
            # Content differs — do not guess, flag for manual review.
            n_conflicts += 1
            print(f"\n[CONFLICT] '{canon}' has {len(distinct_hashes)} different "
                  f"content variants — not auto-resolved:")
            for f in paths:
                size = f.stat().st_size
                print(f"    {f.name}  ({size:,} bytes, sha256={hashes[f][:12]}…)")

    # ── Report plan ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"clean_figures_dir.py — {'APPLY' if args.apply else 'DRY-RUN'}")
    print("=" * 70)
    print(f"Directory scanned   : {target}")
    print(f"Total files found   : {len(files)}")
    print(f"Already clean       : {n_clean}")
    print(f"Duplicate groups    : {n_dupe_groups}  ({n_dupes_resolved} duplicate file(s) to remove)")
    print(f"Conflicting groups  : {n_conflicts}  (left untouched — see above)")
    print(f"Rename-only actions : {sum(1 for k, _, _ in actions if k == 'rename')}")
    print()

    if not actions:
        print("Nothing to do.")
        return 0

    if not args.apply:
        print("Planned actions (re-run with --apply to execute):")
        for kind, src, dest in actions:
            arrow = "→ rename to" if kind == "rename" else "→ quarantine to"
            print(f"  [{kind:10}] {src.name}  {arrow}  {dest}")
        return 0

    # ── Apply ───────────────────────────────────────────────────────────────
    if any(k == "quarantine" for k, _, _ in actions) and not args.delete_duplicates:
        quarantine.mkdir(exist_ok=True)

    for kind, src, dest in actions:
        if kind == "rename":
            if dest.exists():
                print(f"  [SKIP] rename target already exists: {dest}")
                continue
            src.rename(dest)
            print(f"  [RENAMED] {src.name} → {dest.name}")
        elif kind == "quarantine":
            if args.delete_duplicates:
                src.unlink()
                print(f"  [DELETED] {src.name}")
            else:
                dest_final = dest
                i = 1
                while dest_final.exists():
                    dest_final = dest.with_name(f"{dest.stem}.dup{i}{dest.suffix}")
                    i += 1
                shutil.move(str(src), str(dest_final))
                print(f"  [QUARANTINED] {src.name} → {args.quarantine_dir}/{dest_final.name}")

    print("\nDone.")
    if not args.delete_duplicates and any(k == "quarantine" for k, _, _ in actions):
        print(f"Duplicates moved to: {quarantine}")
        print("Review and delete that folder once you've confirmed nothing was lost.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
