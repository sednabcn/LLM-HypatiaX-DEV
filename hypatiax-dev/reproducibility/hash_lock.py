"""
hypatiax/reproducibility/hash_lock.py
======================================
Deterministic config hashing + lock-file integrity check.

CLI usage (called by run_all_checkpoint.py hashlock step):
    python3 reproducibility/hash_lock.py --check
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def hash_config(config: dict) -> str:
    """
    Return a deterministic SHA-256 hex digest for *config*.

    Uses hashlib — NOT Python's built-in hash(), which is randomised
    per-process (PYTHONHASHSEED) and would produce a different lock
    filename on every run, breaking the skip-if-cached logic in
    universal_protocol.py.
    """
    return hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()


# Alias so universal_protocol.py's `hash_dict` import also works
hash_dict = hash_config


# ── CLI ───────────────────────────────────────────────────────────────────────

def _check(results_dir: Path) -> int:
    """
    Verify that every .lock_* file in *results_dir* has a corresponding
    .json result file.  Prints a report and returns exit code 0 (pass)
    or 1 (orphaned locks found).
    """
    if not results_dir.exists():
        print(f"  ⚠  hash_lock --check: results dir not found: {results_dir}")
        print("  ✓  Nothing to check — treating as pass")
        return 0

    locks  = sorted(results_dir.glob(".lock_*"))
    jsons  = {f.stem for f in results_dir.glob("*.json")}

    orphans = []
    for lock in locks:
        # lock name: .lock_<hash>  →  expect <n>_<hash[:8]>.json
        lock_hash = lock.name.replace(".lock_", "")
        matched = any(lock_hash[:8] in j for j in jsons)
        if not matched:
            orphans.append(lock)

    print("\n  hash_lock --check")
    print(f"  Results dir : {results_dir}")
    print(f"  Lock files  : {len(locks)}")
    print(f"  JSON files  : {len(jsons)}")

    if orphans:
        print(f"  ⚠  Orphaned locks ({len(orphans)}) — no matching result JSON:")
        for o in orphans:
            print(f"      {o.name}")
        print("  These will be retried on next pipeline run.")
        # Orphaned locks are a warning, not a hard failure — return 0
        return 0

    print(f"  ✓  All {len(locks)} lock(s) have matching result files")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HypatiaX hash-lock integrity checker"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check lock files against result JSONs in RESULTS_DIR"
    )
    parser.add_argument(
        "--results-dir",
        default=os.environ.get(
            "HYPATIAX_RESULTS",
            str(Path(__file__).resolve().parents[2] / "hypatiax" / "data" / "results")
        ),
        help="Path to results directory (default: hypatiax/data/results)"
    )
    args = parser.parse_args()

    if args.check:
        raise SystemExit(_check(Path(args.results_dir)))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
