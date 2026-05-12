#!/usr/bin/env python3
"""
clean_checkpoint.py
===================
Removes null/failed entries from ANY HypatiaX experiment checkpoint JSON
so that --resume re-executes them on the next run.

Keeps any entry whose result.success is True.

Replaces the single-purpose clean_exp2_sym_checkpoint.py.

Usage
-----
    # Auto-detect all checkpoints under logs/ and clean them interactively
    python3 clean_checkpoint.py

    # Clean a specific checkpoint
    python3 clean_checkpoint.py --checkpoint logs/exp2_symbolic_engine_checkpoint.json

    # Clean all checkpoints under a directory (non-interactive)
    python3 clean_checkpoint.py --dir logs/ --all

    # Match by experiment name fragment
    python3 clean_checkpoint.py --exp exp3

    # Preview without writing
    python3 clean_checkpoint.py --dry-run

    # Require a minimum r2 to keep (float, default 0 — any success=True is kept)
    python3 clean_checkpoint.py --min-r2 0.95

    # Remove ALL entries regardless of success (full reset)
    python3 clean_checkpoint.py --checkpoint logs/exp1_checkpoint.json --reset

Exit codes
----------
    0   Clean (or nothing to do)
    1   File not found / JSON parse error
    2   No checkpoints matched the filter
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ── Default search root ───────────────────────────────────────────────────────
# Walks up from this script's location to find the repo root, then looks for
# a logs/ directory.  The CHECKPOINT_DIR env-var overrides everything.

def _find_default_dir() -> Path:
    env = os.environ.get("CHECKPOINT_DIR")
    if env:
        return Path(env)

    # Try repo-relative logs/ next to this script
    here = Path(__file__).resolve().parent
    for candidate in [here / "logs", here.parent / "logs"]:
        if candidate.is_dir():
            return candidate

    # Fallback: the well-known absolute path from the original script
    fallback = (
        Path.home()
        / "Downloads/GITHUB/LLM-HypatiaX-PAPERS-Public/logs"
    )
    return fallback


# ── Entry-level success test ──────────────────────────────────────────────────

def _is_success(entry: dict, min_r2: float) -> bool:
    """Return True if this checkpoint entry should be kept."""
    res = entry.get("result")
    if not res:
        return False
    if not res.get("success"):
        return False
    if min_r2 > 0:
        r2 = res.get("r2")
        if r2 is None:
            return False
        try:
            return float(r2) >= min_r2
        except (TypeError, ValueError):
            return False
    return True


# ── Single-file cleaner ───────────────────────────────────────────────────────

def clean_one(path: Path, dry_run: bool, min_r2: float, reset: bool) -> int:
    """
    Clean a single checkpoint file.

    Returns
    -------
    0  nothing changed (already clean, or dry-run)
    1  file was cleaned (or would be in dry-run)
    -1 error
    """
    if not path.exists():
        print(f"  ❌  Not found: {path}", file=sys.stderr)
        return -1

    try:
        with open(path) as fh:
            ckpt = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"  ❌  JSON parse error in {path}: {exc}", file=sys.stderr)
        return -1

    results   = ckpt.get("results",   {})
    completed = ckpt.get("completed", [])

    if not results:
        print(f"  ℹ   {path.name}: no 'results' key — skipping")
        return 0

    if reset:
        keep   = {}
        remove = dict(results)
    else:
        keep   = {k: v for k, v in results.items() if _is_success(v, min_r2)}
        remove = {k: v for k, v in results.items() if k not in keep}

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  File   : {path}")
    print(f"  Total  : {len(results):>4}")
    print(f"  Keep ✅: {len(keep):>4}"
          + (f"  (r2 ≥ {min_r2})" if min_r2 > 0 else ""))
    print(f"  Remove ❌: {len(remove):>4}"
          + ("  [FULL RESET]" if reset else ""))

    if keep and not reset:
        print("\n  Keeping:")
        for k, v in keep.items():
            r2 = v.get("result", {}).get("r2", "?")
            expr = v.get("result", {}).get("expression", "")
            expr_str = f"  expr={expr[:40]}" if expr else ""
            print(f"    ✅  {k:<40}  r2={r2}{expr_str}")

    if remove:
        print("\n  Removing:")
        for k, v in remove.items():
            reason = "null result" if not v.get("result") else (
                "success=False" if not v.get("result", {}).get("success") else
                f"r2 < {min_r2}"
            )
            print(f"    ❌  {k:<40}  ({reason})")

    if not remove:
        print("\n  ✅  Already clean — nothing to do.")
        return 0

    if dry_run:
        print("\n  --dry-run: no changes written.")
        return 1   # "would have changed"

    # ── Write ─────────────────────────────────────────────────────────────────
    ckpt["results"]   = keep
    ckpt["completed"] = [k for k in completed if k in keep]

    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w") as fh:
            json.dump(ckpt, fh, indent=2, default=str)
        os.replace(tmp, path)
    except OSError as exc:
        print(f"  ❌  Write failed: {exc}", file=sys.stderr)
        if tmp.exists():
            tmp.unlink()
        return -1

    print(f"\n  ✅  Cleaned: {len(remove)} removed, {len(keep)} kept.")
    print(f"      Re-run your experiment with --resume to fill the gaps.")
    return 1


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover(directory: Path, exp_filter: Optional[str]) -> list[Path]:
    """Return all *_checkpoint.json files under directory, optionally filtered."""
    if not directory.exists():
        return []
    pattern = "**/*_checkpoint.json"
    paths = sorted(directory.glob(pattern))
    if exp_filter:
        paths = [p for p in paths if exp_filter.lower() in p.name.lower()]
    return paths


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    default_dir = _find_default_dir()

    p = argparse.ArgumentParser(
        description="Clean null/failed entries from any HypatiaX checkpoint JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Exit codes")[0].strip(),
    )

    target = p.add_mutually_exclusive_group()
    target.add_argument(
        "--checkpoint", type=Path, default=None, metavar="PATH",
        help="Path to a specific checkpoint JSON file.",
    )
    target.add_argument(
        "--dir", type=Path, default=None, metavar="DIR",
        help=f"Directory to search for checkpoints "
             f"(default auto-detected: {default_dir}).",
    )

    p.add_argument(
        "--exp", default=None, metavar="FRAGMENT",
        help="Filter checkpoints whose filename contains FRAGMENT "
             "(e.g. 'exp1', 'feynman', 'suppB'). Only applies to --dir mode.",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Clean all matched checkpoints without prompting.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be removed without writing any changes.",
    )
    p.add_argument(
        "--min-r2", type=float, default=0.0, metavar="FLOAT",
        help="Minimum r2 threshold to keep an entry (default: 0 — keep any success=True).",
    )
    p.add_argument(
        "--reset", action="store_true",
        help="Remove ALL entries regardless of success (full checkpoint reset).",
    )
    p.add_argument(
        "--list", action="store_true",
        help="List discovered checkpoints and exit.",
    )

    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # ── Single file mode ──────────────────────────────────────────────────────
    if args.checkpoint:
        rc = clean_one(args.checkpoint, args.dry_run, args.min_r2, args.reset)
        return 0 if rc >= 0 else 1

    # ── Directory / discovery mode ────────────────────────────────────────────
    search_dir = args.dir or _find_default_dir()
    checkpoints = discover(search_dir, args.exp)

    if not checkpoints:
        filter_note = f" matching '{args.exp}'" if args.exp else ""
        print(f"⚠   No checkpoints found{filter_note} under: {search_dir}",
              file=sys.stderr)
        print("    Use --checkpoint PATH to specify a file directly.",
              file=sys.stderr)
        return 2

    print(f"Found {len(checkpoints)} checkpoint(s) under {search_dir}"
          + (f" [filter: '{args.exp}']" if args.exp else "") + ":")
    for i, cp in enumerate(checkpoints):
        print(f"  [{i+1:02d}] {cp.relative_to(search_dir)}")

    if args.list:
        return 0

    # ── Confirm if not --all ──────────────────────────────────────────────────
    targets: list[Path]
    if args.all or args.dry_run:
        targets = checkpoints
    else:
        print()
        ans = input(
            f"Clean all {len(checkpoints)} checkpoint(s)? "
            "[y=all / n=abort / number to pick one] > "
        ).strip().lower()

        if ans == "y":
            targets = checkpoints
        elif ans.isdigit():
            idx = int(ans) - 1
            if not (0 <= idx < len(checkpoints)):
                print("❌  Index out of range.", file=sys.stderr)
                return 1
            targets = [checkpoints[idx]]
        else:
            print("Aborted.")
            return 0

    # ── Run ───────────────────────────────────────────────────────────────────
    total_changed = 0
    total_errors  = 0
    for cp in targets:
        rc = clean_one(cp, args.dry_run, args.min_r2, args.reset)
        if rc > 0:
            total_changed += 1
        elif rc < 0:
            total_errors += 1

    print(f"\n{'═'*60}")
    print(f"  Summary: {total_changed} cleaned, "
          f"{len(targets) - total_changed - total_errors} already clean, "
          f"{total_errors} error(s).")
    if args.dry_run:
        print("  (dry-run — no files were modified)")
    print()

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
