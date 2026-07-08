#!/usr/bin/env python3
"""
clean_figures_dir.py — strip figures__* / Figures__* duplicate-prefix files
from a figures/ directory, leaving only the canonical bare-stem originals.

BACKGROUND
----------
The figures__figures__fig09_... duplication pattern is caused by local manual
merging of downloaded CI artifact zips.  When two artifact zips (each
containing a figures/ subfolder) are merged into one directory without
deduplication, the zip-extraction library prefixes the source folder name.
In practice this has been observed with an additional "_back" segment and a
trailing numbered-copy suffix from repeated merges:

    figures/fig09_r2_heatmap_regimes.png                          ← original (canonical)
    figures__fig09_r2_heatmap_regimes.png                         ← 1st re-zip round-trip
    figures__figures__fig09_r2_heatmap_regimes.png                ← 2nd re-zip round-trip
    Figures__fig09_r2_heatmap_regimes.png                         ← case variant
    figures_back__figures__fig09_r2_heatmap_regimes.png           ← "_back"-qualified variant
    figures_back__figures__fig09_r2_heatmap_regimes__10.png       ← + numbered-copy suffix
    figures_back__Figures__fig09_r2_heatmap_regimes__10.png       ← + case variant

This script groups every file that shares the same canonical stem (the name
with all leading `figures__` / `Figures__` / `figures_back__` / etc. prefixes
AND any trailing `__N` numbered-copy suffix stripped) and resolves the group:

    IDENTICAL CONTENT  → keep the bare-stem canonical file, move all
                         prefix-mangled duplicates into _duplicates_removed/
                         inside the same directory (quarantine, not delete).
    CONFLICTING CONTENT→ leave ALL files untouched and print a [CONFLICT] line
                         so the caller (ci_postprocess.yml step A17) can emit a
                         ::warning:: for manual review.  The script NEVER
                         auto-resolves genuine content conflicts.
    ONLY MANGLED FILES → if no bare-stem file exists but all mangled copies are
                         identical, rename the first one to the canonical name
                         and quarantine the rest.

OUTPUT FORMAT (stdout, one line per action)
-------------------------------------------
[DRY-RUN] <dir>/<file>  →  would quarantine
[QUARANTINE] <dir>/<file>  →  moved to _duplicates_removed/
[RENAME] <dir>/<mangled>  →  <dir>/<canonical>
[CONFLICT] <canonical_stem>: <n> files with differing content — left untouched
[OK] <canonical_stem>: only canonical file present, nothing to do
[SKIP] <dir>: not a directory or empty

The caller checks for `^[CONFLICT]` to decide whether to emit a warning.

USAGE
-----
    python3 clean_figures_dir.py <directory> [--apply]

    Without --apply: dry-run only (no files moved/renamed).
    With    --apply: perform moves/renames.

EXIT CODES
----------
    0  — completed (even if conflicts were found; caller inspects stdout)
    1  — unhandled exception
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

# Prefix patterns to strip (case-insensitive match at start of filename)
# Strip any sequence of "figures__" / "Figures__" / "figures_back__" prefixes.
#
# FIX (2026-07-07): the original regex only matched a bare "figures__"/
# "Figures__" prefix. The contamination actually observed on disk is
# "figures_back__figures__X.png" — the extra "_back" segment broke the match
# at position 0, so these files were never recognized as duplicates at all
# (each printed a false "[OK] ... nothing to do"). Added the optional
# "(?:_[Bb]ack)?" segment so both the plain and "_back"-qualified prefix
# variants strip correctly, repeated any number of times.
_PREFIX_RE = re.compile(r'^(?:[Ff]igures(?:_[Bb]ack)?__)+', re.IGNORECASE)

# FIX (2026-07-07): numbered-copy suffix ("__10", "__11", ...) immediately
# before the extension. Without stripping this too, "fig08_train_r2_bar.png"
# and "fig08_train_r2_bar__10.png" were grouped separately and never
# compared/deduped against each other. Same convention as
# detect_contaminated_files.py's "numbered-copy-suffix" pattern.
_TRAILING_NUM_RE = re.compile(r'__\d+$')

# File extensions considered "figure" files (others are ignored by this script)
_FIG_EXTS = {'.pdf', '.png', '.jpg', '.jpeg', '.eps', '.svg'}

# Also treat REPO_AUDIT / PROD__REPO_AUDIT files as noise to quarantine
_AUDIT_RE = re.compile(r'^(?:PROD__)?REPO_AUDIT', re.IGNORECASE)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def _canonical_stem(name: str) -> str:
    """Strip all leading figures__/Figures__/figures_back__ prefixes AND
    any trailing __N numbered-copy suffix from a filename, leaving the
    bare canonical name (e.g. 'fig09_r2_heatmap_regimes.png')."""
    stripped = _PREFIX_RE.sub('', name)
    p = Path(stripped)
    stem = _TRAILING_NUM_RE.sub('', p.stem)
    return stem + p.suffix


def clean_directory(directory: Path, apply: bool) -> int:
    """
    Process one directory.  Returns number of unresolved conflicts.
    """
    if not directory.is_dir():
        print(f'[SKIP] {directory}: not a directory or does not exist')
        return 0

    files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in _FIG_EXTS
    ]
    if not files:
        print(f'[SKIP] {directory}: no figure files found')
        return 0

    quarantine_dir = directory / '_duplicates_removed'
    n_conflicts = 0

    # ── Group files by canonical stem (stem without extension) ───────────────
    # Key: (canonical_stem_no_ext, ext_lower)
    # e.g. "fig09_r2_heatmap_regimes", ".png"
    groups: dict[tuple[str, str], list[Path]] = {}
    audit_files: list[Path] = []

    for f in files:
        canonical_name = _canonical_stem(f.name)
        # Audit files — quarantine unconditionally
        if _AUDIT_RE.match(canonical_name):
            audit_files.append(f)
            continue
        # Only process files that ARE prefix-mangled OR are the bare canonical
        canonical_stem_no_ext = Path(canonical_name).stem
        ext = f.suffix.lower()
        key = (canonical_stem_no_ext, ext)
        groups.setdefault(key, []).append(f)

    # ── Quarantine audit noise ────────────────────────────────────────────────
    for f in audit_files:
        _quarantine(f, quarantine_dir, apply)

    # ── Resolve each group ────────────────────────────────────────────────────
    for (stem, ext), group in sorted(groups.items()):
        canonical_name = stem + ext
        canonical_path = directory / canonical_name

        # Partition: canonical vs mangled
        canonical_files = [f for f in group if f.name == canonical_name]
        mangled_files   = [f for f in group if f.name != canonical_name]

        if not mangled_files:
            # Nothing to do — only the bare-stem file present
            print(f'[OK] {stem}{ext}: only canonical file present, nothing to do')
            continue

        # ── Hash everything in the group ─────────────────────────────────────
        try:
            hashes = {f: _sha256(f) for f in group}
        except OSError as exc:
            print(f'[ERROR] could not hash files for {stem}{ext}: {exc}', file=sys.stderr)
            continue

        unique_hashes = set(hashes.values())

        if len(unique_hashes) > 1:
            # CONFLICT — differing content; leave untouched
            print(
                f'[CONFLICT] {stem}{ext}: {len(group)} file(s) with differing '
                f'content — left untouched, needs manual review'
            )
            for f in sorted(group, key=lambda p: p.name):
                print(f'           {f.name}  sha256={hashes[f][:12]}')
            n_conflicts += 1
            continue

        # All files in the group have identical content.
        if canonical_files:
            # Canonical already exists — quarantine all mangled copies
            for f in sorted(mangled_files, key=lambda p: p.name):
                _quarantine(f, quarantine_dir, apply)
        else:
            # No canonical file — rename first mangled → canonical, quarantine rest
            first = sorted(mangled_files, key=lambda p: p.name)[0]
            rest  = sorted(mangled_files, key=lambda p: p.name)[1:]
            _rename_to_canonical(first, canonical_path, apply)
            for f in rest:
                _quarantine(f, quarantine_dir, apply)

    return n_conflicts


def _quarantine(src: Path, quarantine_dir: Path, apply: bool) -> None:
    tag = '[QUARANTINE]' if apply else '[DRY-RUN]'
    dest = quarantine_dir / src.name
    print(f'{tag} {src.name}  →  {quarantine_dir.name}/{src.name}')
    if apply:
        quarantine_dir.mkdir(exist_ok=True)
        shutil.move(str(src), str(dest))


def _rename_to_canonical(src: Path, dest: Path, apply: bool) -> None:
    tag = '[RENAME]' if apply else '[DRY-RUN RENAME]'
    print(f'{tag} {src.name}  →  {dest.name}')
    if apply:
        src.rename(dest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Strip figures__* prefix duplicates from a figures/ directory.'
    )
    parser.add_argument('directory', help='Path to the figures directory to clean')
    parser.add_argument(
        '--apply',
        action='store_true',
        default=False,
        help='Actually move/rename files (default: dry-run)',
    )
    args = parser.parse_args()

    directory = Path(args.directory).resolve()
    n_conflicts = clean_directory(directory, apply=args.apply)

    if n_conflicts:
        # Exit 0 so the caller's `|| true` doesn't mask the conflict output,
        # but the [CONFLICT] lines in stdout are what the caller grep-checks.
        sys.exit(0)


if __name__ == '__main__':
    main()
