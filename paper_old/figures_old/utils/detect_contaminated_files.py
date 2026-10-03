#!/usr/bin/env python3
"""
detect_contaminated_files.py

Scans one or more directory trees (default: .github/ and scripts/) for
filenames matching known "contamination" patterns seen elsewhere in this
repo's figures/ dump — duplicate-run shards, doubled directory-name
prefixes, numbered copy suffixes, etc.

This script only DETECTS and REPORTS. It never deletes or moves anything.
Use the report to decide what to hand to your existing cleanup scripts
(clean_figures_dirs.sh / cleanup_figures_prefix.sh / purge_figures_contamination.sh)
or a manual `git rm`.

Usage:
    python3 detect_contaminated_files.py
    python3 detect_contaminated_files.py .github scripts
    python3 detect_contaminated_files.py --root . --json report.json
    python3 detect_contaminated_files.py --root scripts --csv report.csv

Exit code:
    0  — no contaminated files found
    1  — contaminated files found (useful as a CI gate)
    2  — usage / path error
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# Pattern definitions
#
# Each entry: (label, compiled regex, human-readable description)
# Patterns are matched against the FILENAME only (not the full path),
# case-insensitive unless noted.
# ─────────────────────────────────────────────────────────────────────────

PATTERNS: list[tuple[str, re.Pattern, str]] = [
    (
        "doubled-segment-prefix",
        re.compile(r"([A-Za-z0-9]{3,})_back__\1", re.IGNORECASE),
        "Backup-style doubled directory name, e.g. 'figures_back__figures__...' "
        "or 'figures_back__Figures__...' (case variants included)",
    ),
    (
        "doubled-segment-generic",
        re.compile(r"\b([A-Za-z0-9]{3,})__\1\b", re.IGNORECASE),
        "Any token immediately repeated with '__' glue, e.g. 'figure__figure__...'",
    ),
    (
        "shard-suffix",
        re.compile(r"_shard\d+(_run\d+)?", re.IGNORECASE),
        "Shard/run artifact suffix, e.g. '..._shard0_run26312651579'",
    ),
    (
        "numbered-copy-suffix",
        re.compile(r"__\d+(?=\.[A-Za-z0-9]+$)"),
        "Numbered duplicate-copy suffix just before the extension, "
        "e.g. '..._10.pdf' written as '...__10.pdf'",
    ),
    (
        "prod-duplicate-prefix",
        re.compile(r"^PROD__"),
        "'PROD__' prefix duplicating an existing filename, "
        "e.g. 'PROD__REPO_AUDIT.md.pdf'",
    ),
]

DEFAULT_ROOTS = [".github", "scripts"]


@dataclass
class Match:
    path: str
    filename: str
    pattern_label: str
    pattern_desc: str


def scan(roots: list[str]) -> tuple[list[Match], list[str]]:
    """Walk each root dir and return (matches, missing_roots)."""
    matches: list[Match] = []
    missing: list[str] = []

    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            missing.append(root)
            continue
        if root_path.is_file():
            # Allow passing a single file too.
            _check_file(root_path, matches)
            continue
        for dirpath, _dirnames, filenames in os.walk(root_path):
            for fname in filenames:
                _check_file(Path(dirpath) / fname, matches)

    return matches, missing


def _check_file(filepath: Path, matches: list[Match]) -> None:
    fname = filepath.name
    for label, pattern, desc in PATTERNS:
        if pattern.search(fname):
            matches.append(
                Match(
                    path=str(filepath),
                    filename=fname,
                    pattern_label=label,
                    pattern_desc=desc,
                )
            )


def print_report(matches: list[Match], missing: list[str], roots: list[str]) -> None:
    print(f"Scanned roots: {', '.join(roots)}")
    if missing:
        print(f"  (not found, skipped: {', '.join(missing)})")
    print()

    if not matches:
        print("✔ No contamination-pattern filenames found.")
        return

    by_label: dict[str, list[Match]] = {}
    for m in matches:
        by_label.setdefault(m.pattern_label, []).append(m)

    print(f"✘ Found {len(matches)} matching file(s) across {len(by_label)} pattern(s):\n")
    for label, group in by_label.items():
        desc = group[0].pattern_desc
        print(f"[{label}]  {desc}")
        for m in group:
            print(f"    {m.path}")
        print()


def write_json(matches: list[Match], out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump([asdict(m) for m in matches], f, indent=2)
    print(f"JSON report written to {out_path}")


def write_csv(matches: list[Match], out_path: str) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "filename", "pattern_label", "pattern_desc"])
        writer.writeheader()
        for m in matches:
            writer.writerow(asdict(m))
    print(f"CSV report written to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "roots",
        nargs="*",
        default=None,
        help=f"Directories (or files) to scan. Default: {' '.join(DEFAULT_ROOTS)}",
    )
    parser.add_argument("--root", dest="root_flag", action="append", default=[],
                         help="Alternate way to add a root; can be repeated.")
    parser.add_argument("--json", metavar="PATH", help="Write matches to a JSON file.")
    parser.add_argument("--csv", metavar="PATH", help="Write matches to a CSV file.")
    args = parser.parse_args()

    roots = args.roots if args.roots else []
    roots += args.root_flag
    if not roots:
        roots = DEFAULT_ROOTS

    matches, missing = scan(roots)
    print_report(matches, missing, roots)

    if args.json:
        write_json(matches, args.json)
    if args.csv:
        write_csv(matches, args.csv)

    if missing and not matches:
        # Nothing to scan and nothing found — not an error by itself,
        # but flag it clearly so it's not mistaken for "verified clean".
        print("\nNote: none of the requested roots exist in this checkout.")

    return 1 if matches else 0


if __name__ == "__main__":
    sys.exit(main())
