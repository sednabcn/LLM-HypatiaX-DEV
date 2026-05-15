#!/usr/bin/env python3
"""
HypatiaX Unified Consolidation Engine
=====================================

Production-safe shard merger for:

- exp1
- exp1b
- exp2
- exp2_feynman
- exp3
- exp3b
- suppA
- suppB
- suppB_sc
- instability
- extrap
- hybrid_all_domains

Design goals
------------

1. Canonical normalization layer
2. Deterministic task identity
3. Recursive extraction
4. Duplicate-safe merge policy
5. Stable metrics/statistics
6. Explicit diagnostics
7. Schema-forward compatibility

This implementation REPLACES all legacy merge heuristics.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from scipy import stats as scipy_stats


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("hypatiax.merge")


# ============================================================
# CONSTANTS
# ============================================================

DEFI_IDS = {
    "amm",
    "risk_var",
    "liquidity",
    "expected_shortfall",
    "liquidation",
    "risk",
    "lending",
    "staking",
    "trading",
    "derivatives",
}

EQ_ID_TO_DEFI = {
    "Annualised Portfolio tracking error": "amm",
    "Correlated Portfolio VaR": "risk_var",
    "Portfolio VaR for two correlated": "liquidity",
    "Portfolio Expected Shortfall for correlated": "expected_shortfall",
    "Portfolio Sharpe Ratio": "risk",
    "Portfolio Sortino Ratio": "staking",
    "Portfolio Beta": "lending",
    "Portfolio Information Ratio": "trading",
    "Portfolio Maximum Drawdown": "derivatives",
    "Portfolio Omega Ratio": "liquidation",
}

META_KEYS = {
    "summary",
    "metadata",
    "generated_at",
    "config",
    "run_info",
    "experiment",
    "source_run_id",
    "methods",
    "timestamp",
    "script",
    "purelm_truncation_audit",
}


# ============================================================
# CONFIG
# ============================================================

@dataclass
class MergeConfig:
    exp: str
    result_dir: Path
    artifact_dir: Path
    pending: List[str]


# ============================================================
# UTILS
# ============================================================


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)



def safe_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)



def is_nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


# ============================================================
# NORMALIZATION
# ============================================================


def canonical_task_id(obj: Dict[str, Any]) -> Optional[str]:
    """
    Resolve ONE deterministic task identity.
    """

    candidates = [
        obj.get("task_id"),
        obj.get("equation_id"),
        obj.get("protocol"),
        obj.get("domain"),
        obj.get("id"),
        obj.get("name"),
    ]

    for c in candidates:
        if c:
            return EQ_ID_TO_DEFI.get(str(c), str(c))

    return None



def normalize_model_dict(d: Any) -> Dict[str, Any]:

    if not isinstance(d, dict):
        return {}

    out = dict(d)

    if "test_r2" in out and "extrap_r2" not in out:
        out["extrap_r2"] = out["test_r2"]

    return out



def normalize_row(raw: Any) -> Optional[Dict[str, Any]]:

    if not isinstance(raw, dict):
        return None

    row = dict(raw)

    inner = row.get("results")
    if not isinstance(inner, dict):
        inner = row

    hyp = (
        inner.get("hypatia")
        or inner.get("pure_llm")
        or {}
    )

    nn = (
        inner.get("nn")
        or inner.get("neural_network")
        or {}
    )

    task_id = canonical_task_id(row)

    if not task_id:
        return None

    return {
        "task_id": task_id,
        "name": row.get("name") or row.get("equation_id") or task_id,
        "domain": row.get("domain") or task_id,
        "hypatia": normalize_model_dict(hyp),
        "nn": normalize_model_dict(nn),
    }


# ============================================================
# EXTRACTION
# ============================================================


def extract_rows(obj: Any) -> Iterable[Dict[str, Any]]:

    found = []

    def walk(x: Any):

        if isinstance(x, list):
            for i in x:
                walk(i)
            return

        if not isinstance(x, dict):
            return

        normalized = normalize_row(x)

        if normalized:
            found.append(normalized)

        for k, v in x.items():
            if k not in META_KEYS:
                walk(v)

    walk(obj)

    return found


# ============================================================
# MERGE POLICY
# ============================================================


def score_row(row: Dict[str, Any]) -> int:

    score = 0

    h = row.get("hypatia", {})
    n = row.get("nn", {})

    if h.get("extrap_r2") is not None:
        score += 10

    if h.get("train_r2") is not None:
        score += 5

    if h.get("best_expression"):
        score += 3

    if n.get("extrap_r2") is not None:
        score += 2

    return score



def merge_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:

    merged: Dict[str, Dict[str, Any]] = {}

    for row in rows:

        tid = row["task_id"]

        if tid not in merged:
            merged[tid] = row
            continue

        old_score = score_row(merged[tid])
        new_score = score_row(row)

        if new_score > old_score:
            merged[tid] = row

    return merged


# ============================================================
# STATS
# ============================================================


def build_stats(exp: str, merged: Dict[str, Any], pending: List[str]):

    hyp = []
    nn = []
    successes = 0

    for row in merged.values():

        hr2 = (row.get("hypatia") or {}).get("extrap_r2")
        nr2 = (row.get("nn") or {}).get("extrap_r2")

        if hr2 is not None and not is_nan(hr2):
            hyp.append(float(hr2))

            if hr2 > 0.99:
                successes += 1

        if nr2 is not None and not is_nan(nr2):
            nn.append(float(nr2))

    stats = {
        "experiment": exp,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_total": len(pending),
        "n_merged": len(merged),
        "n_successes": successes,
        "success_rate": (
            successes / len(merged)
            if merged else None
        ),
        "hyp_extrap_mean": (
            float(np.mean(hyp))
            if hyp else None
        ),
        "hyp_extrap_median": (
            float(np.median(hyp))
            if hyp else None
        ),
        "nn_extrap_mean": (
            float(np.mean(nn))
            if nn else None
        ),
        "nn_extrap_median": (
            float(np.median(nn))
            if nn else None
        ),
    }

    if len(hyp) >= 5 and len(nn) >= 5:

        u, p = scipy_stats.mannwhitneyu(
            hyp,
            nn,
            alternative="greater",
        )

        stats["mw_U"] = float(u)
        stats["mw_p"] = float(p)
        stats["mw_significant"] = bool(p < 0.05)

    return stats


# ============================================================
# CSV
# ============================================================


def write_csv(path: Path, merged: Dict[str, Any]):

    rows = [
        "task_id,name,domain,hyp_train_r2,hyp_extrap_r2,nn_extrap_r2,success,best_expression"
    ]

    for tid, row in sorted(merged.items()):

        h = row.get("hypatia") or {}
        n = row.get("nn") or {}

        he = h.get("extrap_r2")
        ok = isinstance(he, float) and he > 0.99

        expr = str(
            h.get("best_expression", "")
        ).replace(",", ";")

        rows.append(
            f'{tid},'
            f'{row.get("name","")},'
            f'{row.get("domain","")},'
            f'{h.get("train_r2","")},'
            f'{h.get("extrap_r2","")},'
            f'{n.get("extrap_r2","")},'
            f'{ok},'
            f'{expr}'
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write("\n".join(rows))


# ============================================================
# MAIN
# ============================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--exp", required=True)
    parser.add_argument("--result-subdir", required=True)
    parser.add_argument("--artifact-dir", default="downloaded_artifacts")
    parser.add_argument("--pending", default="[]")

    args = parser.parse_args()

    config = MergeConfig(
        exp=args.exp,
        result_dir=Path("hypatiax/data/results") / args.result_subdir,
        artifact_dir=Path(args.artifact_dir),
        pending=json.loads(args.pending),
    )

    logger.info("=" * 70)
    logger.info("HypatiaX Unified Consolidation Engine")
    logger.info("=" * 70)
    logger.info(f"EXP: {config.exp}")
    logger.info(f"RESULT_DIR: {config.result_dir}")
    logger.info(f"ARTIFACT_DIR: {config.artifact_dir}")

    files = sorted(
        glob.glob(
            f"{config.artifact_dir}/**/*.json",
            recursive=True,
        )
    )

    logger.info(f"JSON FILES FOUND: {len(files)}")

    all_rows = []

    for path in files:

        logger.info("-" * 70)
        logger.info(f"READ: {path}")

        try:
            data = load_json(Path(path))
            rows = list(extract_rows(data))

            logger.info(f"ROWS EXTRACTED: {len(rows)}")

            all_rows.extend(rows)

        except Exception as e:
            logger.exception(f"FAILED TO READ: {path} :: {e}")

    merged = merge_rows(all_rows)

    logger.info("=" * 70)
    logger.info("MERGED TASKS")
    logger.info("=" * 70)

    for k in sorted(merged.keys()):
        logger.info(f"  - {k}")

    missing = [
        t for t in config.pending
        if t not in merged
    ]

    coverage = (
        100 * len(merged) / len(config.pending)
        if config.pending else 100
    )

    logger.info(
        f"COVERAGE: {len(merged)}/{len(config.pending)} ({coverage:.1f}%)"
    )

    if missing:
        logger.warning(f"MISSING TASKS: {missing}")

    if not merged:
        raise RuntimeError(
            "FATAL: merge produced zero rows"
        )

    stats = build_stats(
        config.exp,
        merged,
        config.pending,
    )

    merged_path = config.result_dir / f"{config.exp}_merged.json"
    stats_path = config.result_dir / f"{config.exp}_stats.json"
    csv_path = config.result_dir / f"{config.exp}_merged.csv"

    safe_write_json(merged_path, merged)
    safe_write_json(stats_path, stats)
    write_csv(csv_path, merged)

    logger.info("=" * 70)
    logger.info(f"WRITE OK: {merged_path}")
    logger.info(f"WRITE OK: {stats_path}")
    logger.info(f"WRITE OK: {csv_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
