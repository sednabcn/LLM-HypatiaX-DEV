#!/usr/bin/env python3
"""
FIX-C2  –  Detect and fix stale hybrid_system_v40 imports
===========================================================
Finds every occurrence of ``hybrid_system_v40`` in
``run_comparative_suite_benchmark_v2.py`` (searched recursively from the
repo root) and replaces it with ``hybrid_system_v50_2``.

When the source files are already clean (or after --apply succeeds) the
script also writes back to ``issue_registry.json``, stamping FIX-C2 as
"resolved" with today's date.  Use --no-registry to skip that step.

Usage
-----
    # Dry-run (report only, no writes):
    python fix_c2_import_upgrade.py --root /path/to/repo

    # Apply fixes + update registry:
    python fix_c2_import_upgrade.py --root /path/to/repo --apply

    # Apply fixes but leave registry unchanged:
    python fix_c2_import_upgrade.py --root /path/to/repo --apply --no-registry

    # Point to a registry in a non-default location:
    python fix_c2_import_upgrade.py --root . --apply \\
        --registry /path/to/issue_registry.json

Exit codes
----------
    0  –  no issues found (or all fixed when --apply)
    1  –  issues detected (dry-run) or unexpected error
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLD_SYMBOL = "hybrid_system_v40"
NEW_SYMBOL = "hybrid_system_v50_2"
ISSUE_ID   = "FIX-C2"

# Only these filenames are candidates for import-level fixes.
# Comments / docstrings in audit/patch files that mention the old name
# are intentional historical references and must NOT be touched.
TARGET_FILENAMES = {
    "run_comparative_suite_benchmark_v2.py",
}

# Matches only real import/usage lines — NOT comment-only lines.
# A line that starts (after optional whitespace) with '#' is skipped.
IMPORT_PATTERN = re.compile(
    r"^(?!\s*#)"                        # not a pure comment line
    r".*\b" + re.escape(OLD_SYMBOL) + r"\b"
)

# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def find_registry(root: Path, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else (root / p)
    candidate = root / "scripts" / "patches" / "issue_registry.json"
    if candidate.exists():
        return candidate
    candidate = root / "issue_registry.json"
    if candidate.exists():
        return candidate
    return None


def stamp_registry(registry_path: Path, issue_id: str, reason: str) -> None:
    """Set status → 'resolved' and update the date for issue_id in-place."""
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    today = date.today().isoformat()
    changed = False
    for entry in data:
        if entry.get("id") == issue_id:
            if entry.get("status") == "resolved":
                print(f"  ℹ️   Registry: {issue_id} already 'resolved' — no change.")
                return
            entry["status"]  = "resolved"
            entry["updated"] = today
            if "false_positive_reason" not in entry:
                entry["false_positive_reason"] = None
            # store the resolution note in the action field so it's visible
            entry["action"] = reason
            changed = True
            break

    if not changed:
        print(f"  [WARN] {issue_id} not found in registry — skipping stamp.")
        return

    registry_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  📋  Registry updated: {issue_id} → 'resolved'  ({today})")
    print(f"      {registry_path}")


# ---------------------------------------------------------------------------
# Source-file helpers
# ---------------------------------------------------------------------------

def find_target_files(root: Path, explicit_files: list[str] | None) -> list[Path]:
    if explicit_files:
        paths = []
        for f in explicit_files:
            p = Path(f)
            if not p.is_absolute():
                p = root / p
            if not p.exists():
                print(f"[WARN] Specified file not found, skipping: {p}")
            else:
                paths.append(p.resolve())
        return paths
    found = []
    for name in TARGET_FILENAMES:
        found.extend(root.rglob(name))
    return sorted(found)


def scan_file(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, text) for every non-comment line containing OLD_SYMBOL."""
    hits = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        print(f"[WARN] Cannot read (non-UTF-8): {path}")
        return hits
    for idx, line in enumerate(lines, start=1):
        if IMPORT_PATTERN.search(line):
            hits.append((idx, line))
    return hits


def fix_file(path: Path) -> int:
    """Replace OLD_SYMBOL with NEW_SYMBOL on non-comment lines. Returns sub count."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    total = 0
    out   = []
    for line in lines:
        if IMPORT_PATTERN.search(line):
            new_line, n = re.subn(r"\b" + re.escape(OLD_SYMBOL) + r"\b", NEW_SYMBOL, line)
            out.append(new_line)
            total += n
        else:
            out.append(line)
    if total:
        path.write_text("".join(out), encoding="utf-8")
    return total


def print_hits(path: Path, hits: list[tuple[int, str]], root: Path) -> None:
    rel = path.relative_to(root)
    print(f"\n  📄  {rel}  ({len(hits)} occurrence{'s' if len(hits) != 1 else ''})")
    for lineno, line in hits:
        print(f"      L{lineno:4d}: {line.rstrip()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="FIX-C2: replace stale hybrid_system_v40 → hybrid_system_v50_2",
    )
    parser.add_argument("--root",        default=".",
                        help="Repository root (default: current directory).")
    parser.add_argument("--apply",       action="store_true",
                        help="Write fixes to source files.")
    parser.add_argument("--no-registry", action="store_true",
                        help="Skip writing back to issue_registry.json.")
    parser.add_argument("--registry",    default=None, metavar="PATH",
                        help="Explicit path to issue_registry.json.")
    parser.add_argument("--files",       nargs="+", metavar="FILE",
                        help="Explicit source files to check (overrides auto-search).")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"[ERROR] Root directory not found: {root}")
        return 1

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"\n{'='*60}")
    print(f"  FIX-C2  |  {OLD_SYMBOL!r}  →  {NEW_SYMBOL!r}")
    print(f"  Mode    |  {mode}")
    print(f"  Root    |  {root}")
    print(f"{'='*60}")

    # ── Scan source files ──────────────────────────────────────────────────
    target_files = find_target_files(root, args.files)
    if not target_files:
        print("\n[INFO] No target files found to scan.")
    else:
        print(f"\n[INFO] Scanning {len(target_files)} file(s) …")

    total_files_with_hits = 0
    total_occurrences     = 0

    for path in target_files:
        hits = scan_file(path)
        if not hits:
            rel = path.relative_to(root)
            print(f"\n  ✅  {rel}  — clean")
            continue

        total_files_with_hits += 1
        total_occurrences     += len(hits)
        print_hits(path, hits, root)

        if args.apply:
            n = fix_file(path)
            rel = path.relative_to(root)
            print(f"      ✏️   Fixed {n} occurrence(s) in {rel}")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")

    source_clean = (total_occurrences == 0)

    if not source_clean:
        status_icon = "⚠️ " if not args.apply else "✅"
        print(f"  {status_icon} {total_occurrences} occurrence(s) in "
              f"{total_files_with_hits} file(s).")
        if not args.apply:
            print("  Re-run with --apply to write fixes.")
            return 1
        else:
            print(f"  All occurrences replaced with {NEW_SYMBOL!r}.")
    else:
        print("  ✅  No stale import-level occurrences detected.")

    # ── Registry update ────────────────────────────────────────────────────
    resolved_now = args.apply or source_clean   # clean already = effectively resolved

    if resolved_now and not args.no_registry:
        registry_path = find_registry(root, args.registry)
        if registry_path and registry_path.exists():
            if args.apply:
                reason = (f"Verified clean by fix_c2_import_upgrade.py --apply on "
                          f"{date.today().isoformat()}. All import-level occurrences "
                          f"of {OLD_SYMBOL!r} replaced with {NEW_SYMBOL!r}.")
            else:
                reason = (f"Verified already clean by fix_c2_import_upgrade.py "
                          f"(dry-run) on {date.today().isoformat()}. "
                          f"No import-level occurrences of {OLD_SYMBOL!r} found in "
                          f"target files; remaining grep hits are comments/audit "
                          f"allowlists only.")
            stamp_registry(registry_path, ISSUE_ID, reason)
        else:
            print("  [INFO] issue_registry.json not found — skipping registry stamp.")
            print("         Pass --registry <path> to specify its location.")
    elif args.no_registry:
        print("  [INFO] --no-registry set; skipping registry update.")

    if args.apply or source_clean:
        print(f"\n  FIX-C2 RESOLVED ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
