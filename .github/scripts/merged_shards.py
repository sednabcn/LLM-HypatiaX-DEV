#!/usr/bin/env python3
"""
HypatiaX Clean Merge Engine (Production Grade)

Fixes:
- Empty merge failures due to schema mismatch
- Nested result structures (v1 + v2 compatibility)
- Silent drops of valid domains (amm, risk_var)
"""

from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable, Tuple
import logging


# ----------------------------
# Logging
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("hypatiax.merge")


# ----------------------------
# Config
# ----------------------------

@dataclass
class MergeConfig:
    exp: str
    result_dir: Path
    pending: List[str]


# ----------------------------
# Utilities
# ----------------------------

def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def safe_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def is_valid_record(v: Any) -> bool:
    return isinstance(v, dict) and len(v) > 0


# ----------------------------
# Core extraction logic
# ----------------------------

def extract_domain_records(obj: Any, domains: List[str]) -> Dict[str, List[Any]]:
    """
    Recursively extract domain records from arbitrary nested JSON.
    Works with:
    - dict of dicts
    - list of dicts
    - nested experiment outputs
    """
    collected: Dict[str, List[Any]] = {d: [] for d in domains}

    def walk(x: Any):
        if isinstance(x, dict):
            domain = x.get("domain")
            if domain in domains:
                collected[domain].append(x)

            for v in x.values():
                walk(v)

        elif isinstance(x, list):
            for i in x:
                walk(i)

    walk(obj)
    return collected


# ----------------------------
# Merge logic
# ----------------------------

def merge_records(records: Dict[str, List[Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}

    for domain, items in records.items():
        cleaned = [r for r in items if is_valid_record(r)]

        if cleaned:
            merged[domain] = {
                "count": len(cleaned),
                "data": cleaned
            }

    return merged


# ----------------------------
# Diagnostics
# ----------------------------

def print_diagnostics(extracted: Dict[str, List[Any]], pending: List[str]) -> None:
    logger.info("=" * 30)
    logger.info("MERGE DIAGNOSTICS")
    logger.info("=" * 30)

    total = 0
    for k, v in extracted.items():
        logger.info(f"[FOUND] {k}: {len(v)} records")
        total += len(v)

    missing = [k for k in pending if len(extracted.get(k, [])) == 0]

    logger.info("")
    logger.info(f"TOTAL FOUND: {total}")
    logger.info(f"MISSING: {missing}")
    logger.info("=" * 30)


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--result-subdir", required=True)
    parser.add_argument(
        "--pending",
        default='["amm","risk_var","liquidity","expected_shortfall","liquidation","risk","lending","staking","trading","derivatives"]'
    )

    args = parser.parse_args()

    config = MergeConfig(
        exp=args.exp,
        result_dir=Path("hypatiax/data/results") / args.result_subdir,
        pending=json.loads(args.pending)
    )

    merged_file = config.result_dir / f"{config.exp}_merged.json"
    stats_file = config.result_dir / f"{config.exp}_stats.json"

    logger.info("=" * 30)
    logger.info("HypatiaX Clean Merge Engine (Production)")
    logger.info("=" * 30)
    logger.info(f"EXP: {config.exp}")
    logger.info(f"RESULT_DIR: {config.result_dir}")

    logger.info(f"[INFO] Loading: {merged_file}")

    raw = load_json(merged_file)

    if not raw:
        raise RuntimeError(
            "EMPTY INPUT FILE: merged JSON is empty. "
            "Upstream pipeline did not generate results."
        )

    extracted = extract_domain_records(raw, config.pending)

    print_diagnostics(extracted, config.pending)

    merged = merge_records(extracted)

    if len(merged) == 0:
        raise RuntimeError(
            "Merge produced 0 rows after extraction. "
            "This confirms schema mismatch (likely v1 nested structure)."
        )

    stats = {
        "exp": config.exp,
        "domains": {
            k: len(v["data"]) for k, v in merged.items()
        }
    }

    safe_write_json(merged_file, merged)
    safe_write_json(stats_file, stats)

    logger.info("=" * 30)
    logger.info(f"WRITE OK: {merged_file}")
    logger.info(f"WRITE OK: {stats_file}")
    logger.info("Pipeline complete (production-safe)")
    logger.info("=" * 30)


if __name__ == "__main__":
    main()
