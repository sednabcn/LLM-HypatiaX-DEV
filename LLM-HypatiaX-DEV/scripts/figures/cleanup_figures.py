#!/usr/bin/env python3
"""
cleanup_figures.py

Safely deduplicates a messy figures/ directory full of flattened
nested-export duplicates (e.g. figures__figures__figures__fig1.png),
case-variant folders (Figures__ vs figures__), and run-stamped report
PDFs (REPO_AUDIT.md_shard0_run<id>.pdf).

Strategy:
  1. Group files by their "base name" — i.e. the filename with any
     leading figures__ / Figures__ prefixes stripped, lowercased.
  2. Within each group, compute a content hash (sha256) for every file.
  3. If all files in a group share ONE hash -> true duplicates.
     Keep the file with the SHORTEST path (closest to root = canonical),
     delete the rest. Moves deleted files to a quarantine folder first
     (nothing is permanently destroyed by this script).
  4. If a group has files with DIFFERENT hashes -> NOT auto-deleted.
     Reported separately so you can review by eye (e.g. "routine" vs
     "routing" typo-named files, or genuinely different content that
     happens to share a base name).
  5. REPO_AUDIT.md_shard0_run<id>.pdf files are treated as a separate
     class: by default we keep only the most recently modified one
     (these look like one-per-CI-run artifacts) plus REPO_AUDIT.md.pdf
     and PROD__REPO_AUDIT.md.pdf (different, kept as-is).

Nothing is deleted permanently — duplicates are moved to
./_quarantine_duplicates/ so you can review and empty it yourself.

Usage:
    cd ~/Downloads/GITHUB/LLM-HypatiaX-DEV/figures
    python3 cleanup_figures.py            # dry run, prints plan only
    python3 cleanup_figures.py --apply    # actually moves duplicates
"""

import argparse
import hashlib
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

QUARANTINE = "_quarantine_duplicates"
RUN_SHARD_RE = re.compile(r"^REPO_AUDIT\.md_shard0_run(\d+)\.pdf$", re.IGNORECASE)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def base_name(fname: str) -> str:
    """Strip repeated figures__/Figures__ prefixes and lowercase."""
    name = fname
    while True:
        new = re.sub(r"^(figures__|Figures__)", "", name)
        if new == name:
            break
        name = new
    return name.lower()


def find_candidate_files(root: Path):
    """Only top-level files (this listing shows a flat dir already)."""
    return [p for p in root.iterdir() if p.is_file()]


def plan_dedup(files):
    """Return (auto_delete, leftover_singletons) given a list of Paths.

    Groups files by base_name, then within each base-name group, further
    groups by content hash. Any hash sub-group with 2+ files is a true
    duplicate set -> keep the shortest path, delete the rest. This
    correctly handles base-name groups where most copies are identical
    but one outlier (e.g. a triple-nested re-export) has slightly
    different bytes (re-compression, regeneration, etc.) - that outlier
    is simply left alone as its own singleton instead of blocking
    dedup of the rest of the group.
    """
    groups = defaultdict(list)
    for p in files:
        groups[base_name(p.name)].append(p)

    auto_delete = []     # list of (keep_path, [delete_paths...])
    singletons = []      # list of (base, hash, [single_path]) - left alone

    for base, paths in groups.items():
        if len(paths) == 1:
            continue
        hashes = defaultdict(list)
        for p in paths:
            try:
                hashes[sha256_of(p)].append(p)
            except OSError as e:
                print(f"  ! could not hash {p}: {e}", file=sys.stderr)

        for h, hpaths in hashes.items():
            if len(hpaths) >= 2:
                same = sorted(hpaths, key=lambda p: (len(str(p)), str(p)))
                keep, dupes = same[0], same[1:]
                auto_delete.append((keep, dupes))
            else:
                singletons.append((base, h, hpaths[0]))

    return auto_delete, singletons


def plan_run_shards(files):
    shard_files = []
    for p in files:
        m = RUN_SHARD_RE.match(p.name)
        if m:
            shard_files.append((int(m.group(1)), p))
    if len(shard_files) <= 1:
        return [], shard_files
    shard_files.sort(key=lambda t: t[0])  # ascending run id (proxy for time)
    *older, newest = shard_files
    return [p for _, p in older], [newest]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                     help="Actually move duplicates to quarantine. "
                          "Without this flag, only prints the plan.")
    ap.add_argument("--dir", default=".", help="Directory to clean (default: cwd)")
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    files = find_candidate_files(root)
    print(f"Scanning {root} ({len(files)} files)\n")

    auto_delete, singletons = plan_dedup(files)
    shard_delete, shard_keep = plan_run_shards(files)

    total_dupes = sum(len(d) for _, d in auto_delete) + len(shard_delete)

    print("=" * 70)
    print(f"TRUE DUPLICATES (identical content) — {sum(len(d) for _,d in auto_delete)} files to remove")
    print("=" * 70)
    for keep, dupes in auto_delete:
        print(f"\nKEEP: {keep.relative_to(root)}")
        for d in dupes:
            print(f"  DELETE (move to quarantine): {d.relative_to(root)}")
    if not auto_delete:
        print("(none)")

    print()
    print("=" * 70)
    print(f"REPO_AUDIT RUN-SHARD PDFS — keeping newest run id, removing {len(shard_delete)} older ones")
    print("=" * 70)
    for _, p in shard_keep:
        print(f"KEEP (newest): {p.relative_to(root)}")
    for p in shard_delete:
        print(f"  DELETE (move to quarantine): {p.relative_to(root)}")

    print()
    print("=" * 70)
    print(f"OUTLIERS — same base name as a duplicate group, but DIFFERENT "
          f"content from the rest ({len(singletons)} files). NOT deleted, "
          f"kept in place. Review by eye if you want to confirm these are "
          f"intentional (e.g. a regenerated/re-encoded figure).")
    print("=" * 70)
    for base, h, p in singletons:
        print(f"  {p.relative_to(root)}  (hash {h[:10]}..., base: {base})")
    if not singletons:
        print("(none)")

    print()
    print("=" * 70)
    print(f"SUMMARY: {total_dupes} files would be quarantined, "
          f"{len(singletons)} outlier files left untouched for manual review, "
          f"{len(files) - total_dupes} files untouched overall.")
    print("=" * 70)

    if not args.apply:
        print("\nDry run only. Re-run with --apply to actually move duplicates "
              f"into ./{QUARANTINE}/")
        return

    qdir = root / QUARANTINE
    qdir.mkdir(exist_ok=True)
    moved = 0
    for keep, dupes in auto_delete:
        for d in dupes:
            dest = qdir / d.name
            shutil.move(str(d), str(dest))
            moved += 1
    for p in shard_delete:
        dest = qdir / p.name
        shutil.move(str(p), str(dest))
        moved += 1

    print(f"\nMoved {moved} files into {qdir}")
    print("Review the quarantine folder, then delete it yourself when satisfied:")
    print(f"  rm -rf {qdir}")


if __name__ == "__main__":
    main()
