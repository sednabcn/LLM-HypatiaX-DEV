#!/usr/bin/env python3
"""
purge_figures_dest.py — remove any figure file from the canonical LaTeX
figures/ destination directory that isn't one of the authoritative allowlist
stems, WITHOUT permanently destroying anything.

BACKGROUND
----------
ci_postprocess.yml's figures_deploy job (Step 0, "purge contamination from
DEST") used to do this purge inline with a raw bash loop calling
`git rm -f --cached` / `rm -f` directly on anything not in ALLOWLIST_STEMS.
That's exactly the kind of file this repo has seen accumulate in figures/
before: figures__*, Figures__*, figures__figures__*, REPO_AUDIT.md_shard0_run*
.pdf, stray hypatiax_* duplicates — all symptoms of someone hand-committing a
locally re-merged artifact-zip figures/ folder (see clean_figures_dir.py's own
BACKGROUND section for the full mechanism). The bash version worked, but it
broke from this repo's established convention everywhere else
(clean_figures_dir.py, flatten_suppb_doubled_path.py): dry-run by default,
--apply to act, and QUARANTINE rather than permanently delete, so a bad
allowlist edit or an unexpected legitimate file can never be silently and
irreversibly lost in CI.

This script is the same purge, restated as a proper quarantine-based tool.

WHAT THIS SCRIPT DOES
----------------------
For the given directory (non-recursive — figures/ is always flat):
  1. Finds every file whose extension is one of _FIG_EXTS.
  2. Computes its basename without extension and checks it against the
     --allow list of permitted stems.
  3. Anything not on the allowlist is moved into --quarantine-dir (default:
     a fresh tempdir OUTSIDE the repo, so it can never accidentally be
     `git add`-ed downstream and bloat a commit, while still being fully
     recoverable from the CI job's filesystem until the runner is torn down).
     Pass an explicit --quarantine-dir to keep it elsewhere.

This intentionally does NOT do clean_figures_dir.py's stem-grouping /
content-hashing — every non-allowlisted file is purged regardless of whether
it's a mangled duplicate of something legitimate or not. The allowlist IS
the source of truth for what belongs in this directory; nothing else does,
no matter how it got there.

OUTPUT FORMAT (stdout, one line per action)
-------------------------------------------
[DRY-RUN]   <dir>/<file>  →  would quarantine (not on allowlist)
[QUARANTINE] <dir>/<file>  →  moved to <quarantine_dir>/<file>
[KEEP]      <dir>/<file>  →  on allowlist
[SKIP]      <dir>: not a directory or does not exist

USAGE
-----
    python3 purge_figures_dest.py <directory> --allow STEM [STEM ...] [--apply]
        [--quarantine-dir DIR]

    Without --apply: dry-run only (no files moved).
    With    --apply: perform the quarantine moves.

EXIT CODES
----------
    0  — completed
    1  — unhandled exception
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

_FIG_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".eps", ".svg"}


def purge_directory(
    directory: Path,
    allowlist: set[str],
    quarantine_dir: Path,
    apply: bool,
) -> int:
    if not directory.is_dir():
        print(f"[SKIP] {directory}: not a directory or does not exist")
        return 0

    n_purged = 0
    for f in sorted(directory.iterdir()):
        if not f.is_file() or f.suffix.lower() not in _FIG_EXTS:
            continue

        if f.stem in allowlist:
            print(f"[KEEP] {f}  →  on allowlist")
            continue

        tag = "[QUARANTINE]" if apply else "[DRY-RUN]"
        dest = quarantine_dir / f.name
        action = f"moved to {dest}" if apply else "would quarantine (not on allowlist)"
        print(f"{tag} {f}  →  {action}")
        if apply:
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest))
        n_purged += 1

    return n_purged


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quarantine any figures/ file not on the allowlist of "
                     "authoritative stems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("directory", help="The canonical figures/ destination directory")
    parser.add_argument(
        "--allow", nargs="+", required=True, metavar="STEM",
        help="Filenames (without extension) that are allowed to remain.",
    )
    parser.add_argument(
        "--quarantine-dir", default=None,
        help="Where to move purged files. Default: a fresh tempdir OUTSIDE "
             "the repo, so quarantined files can never be accidentally "
             "git-add-ed and bloat a commit.",
    )
    parser.add_argument(
        "--apply", action="store_true", default=False,
        help="Actually move files (default: dry-run).",
    )
    args = parser.parse_args()

    directory = Path(args.directory).resolve()
    allowlist = set(args.allow)
    quarantine_dir = (
        Path(args.quarantine_dir).resolve()
        if args.quarantine_dir
        else Path(tempfile.mkdtemp(prefix="purged_figures_dest_"))
    )

    print(f"Quarantine directory: {quarantine_dir}")
    n_purged = purge_directory(directory, allowlist, quarantine_dir, apply=args.apply)
    print(f"\nSummary: {n_purged} non-allowlist file(s) "
          f"{'quarantined' if args.apply else 'would be quarantined'}.")
    if not args.apply and n_purged:
        print("(dry run — pass --apply to actually move files)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
