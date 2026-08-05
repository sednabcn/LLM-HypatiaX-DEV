#!/usr/bin/env python3
"""
clean_exp2_sym_checkpoint.py
============================
Removes null/failed entries from exp2_symbolic_engine_checkpoint.json
that were written by the setsid-crash run, so --resume re-executes them.

Keeps any entry whose result.success is True (e.g. kinetic_energy).

Usage
-----
    python3 clean_exp2_sym_checkpoint.py [--dry-run]

Options
-------
    --dry-run   Print what would be removed without modifying the file.
    --checkpoint PATH   Override the default checkpoint path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Default path — matches the runner's _CHECKPOINT_PATH
_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE  # adjust if running from a different location
_DEFAULT   = Path(os.environ.get(
    "CHECKPOINT",
    Path.home() / "Downloads/GITHUB/LLM-HypatiaX-PAPERS-Public"
    / "logs" / "exp2_symbolic_engine_checkpoint.json"
))


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean null/failed checkpoint entries")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be removed without writing")
    parser.add_argument("--checkpoint", default=str(_DEFAULT),
                        help=f"Path to checkpoint JSON (default: {_DEFAULT})")
    args = parser.parse_args()

    path = Path(args.checkpoint)
    if not path.exists():
        print(f"❌  Checkpoint not found: {path}", file=sys.stderr)
        return 1

    with open(path) as f:
        ckpt = json.load(f)

    results   = ckpt.get("results",   {})
    completed = ckpt.get("completed", [])

    keep   = {}
    remove = {}
    for key, entry in results.items():
        res = entry.get("result")
        if res and res.get("success"):
            keep[key] = entry
        else:
            remove[key] = entry

    print(f"Checkpoint : {path}")
    print(f"Total      : {len(results)}")
    print(f"Keep (✅)  : {len(keep)}")
    print(f"Remove (❌): {len(remove)}")
    print()

    if keep:
        print("Keeping:")
        for k, v in keep.items():
            r2 = v["result"].get("r2", "?")
            print(f"  ✅  {k}  r2={r2}")
        print()

    if remove:
        print("Removing:")
        for k in remove:
            print(f"  ❌  {k}")
        print()

    if not remove:
        print("Nothing to clean — checkpoint is already clean.")
        return 0

    if args.dry_run:
        print("--dry-run: no changes written.")
        return 0

    # Rebuild checkpoint with only the kept entries
    ckpt["results"]   = keep
    ckpt["completed"] = [k for k in completed if k in keep]

    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(ckpt, f, indent=2, default=str)
    os.replace(tmp, path)

    print(f"✅  Checkpoint cleaned. {len(remove)} entries removed, "
          f"{len(keep)} kept.")
    print(f"    Re-run with --resume to execute the removed equations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
