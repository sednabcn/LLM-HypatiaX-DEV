#!/usr/bin/env python3
"""
merge_exp2_shards.py
────────────────────
Combines exp2 per-shard checkpoint JSONs into a single exp2_all30_checkpoint.json.
Called by the exp2-merge CI job after all 5 matrix shards complete.

Usage:
    python3 .github/scripts/merge_exp2_shards.py \
        --shard-dir hypatiax/data/results/comparison_results/ \
        --output    hypatiax/data/results/comparison_results/exp2_all30_checkpoint.json \
        --pass-threshold 9

Exit codes:
    0  — merge succeeded and ≥ pass_threshold equations passed
    1  — merge succeeded but fewer than pass_threshold equations passed
    2  — no shard files found (fatal)
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge exp2 shard checkpoints")
    p.add_argument("--shard-dir", required=True,
                   help="Directory containing exp2_shard_*_checkpoint.json files")
    p.add_argument("--output", required=True,
                   help="Path to write the merged checkpoint JSON")
    p.add_argument("--pass-threshold", type=int, default=9,
                   help="Minimum number of passing equations for exit 0 (default: 9)")
    return p.parse_args()


def load_shard(path: Path) -> dict:
    """Load a shard checkpoint, returning {} on parse error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  ⚠  Could not read {path.name}: {exc}", file=sys.stderr)
        return {}


def merge(args: argparse.Namespace) -> int:
    shard_dir = Path(args.shard_dir)
    output_path = Path(args.output)

    shard_files = sorted(shard_dir.glob("exp2_shard_*_checkpoint.json"))
    if not shard_files:
        print(
            f"FATAL: no shard checkpoint files found in {shard_dir}",
            file=sys.stderr,
        )
        return 2

    print(f"Found {len(shard_files)} shard file(s):")
    for f in shard_files:
        print(f"  {f.name}")

    merged: dict = {}
    for shard_file in shard_files:
        data = load_shard(shard_file)
        overlap = set(merged) & set(data)
        if overlap:
            print(
                f"  ⚠  Overlapping equation keys from {shard_file.name}: {overlap}",
                file=sys.stderr,
            )
        merged.update(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"\nMerged checkpoint written → {output_path}")
    print(f"Total equation entries   : {len(merged)}")

    # Count passes — support both {"status": "passed"} and {"passed": true} shapes
    n_passed = sum(
        1 for v in merged.values()
        if (isinstance(v, dict) and (
            v.get("status") == "passed" or v.get("passed") is True
        ))
    )
    print(f"Equations passed         : {n_passed} / {len(merged)}")
    print(f"Pass threshold           : {args.pass_threshold}")

    if n_passed >= args.pass_threshold:
        print("✓ PASS — threshold met")
        return 0
    else:
        print(
            f"✗ FAIL — only {n_passed}/{len(merged)} equations passed "
            f"(threshold {args.pass_threshold})",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(merge(parse_args()))
