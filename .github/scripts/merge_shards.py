#!/usr/bin/env python3
"""
HypatiaX Unified Consolidation Engine
=====================================

Canonical, experiment-agnostic shard merger.

Usage
-----
    python scripts/merge_shards.py \
        --experiment   <exp_id>          \
        --input-root   downloaded_artifacts \
        --output-dir   hypatiax/data/results/<subdir>

Outputs (all written to --output-dir)
--------------------------------------
    _merged.json       Merged task records keyed by task_id
    _merged.csv        Flat CSV view of the same records
    _stats.json        Basic pre-aggregation counts and R² summaries
    _checkpoint.json   Provenance / run metadata

Design goals
------------
1. Canonical normalisation layer
2. Deterministic task identity
3. Recursive extraction
4. Duplicate-safe merge policy (highest-score row wins)
5. Basic aggregation stats only — no experiment-specific analysis
6. Explicit diagnostics
7. Schema-forward compatibility

This script is the ONLY authoritative merge implementation.
It is reused by both ci_experiment.yml (inline consolidate job)
and ci_consolidate_experiment.yml (standalone re-consolidation).
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


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("hypatiax.merge")


# ============================================================
# CONSTANTS
# ============================================================

# ── Single source of truth for which experiments require shard merging ────────
# Both locate_analysis_input.sh and ci_analysis.yml's "Merge shards" step import
# this set at runtime via:
#   from merge_shards import MERGE_REQUIRED_EXPERIMENTS
# Adding an experiment here is the ONLY change needed to activate merge mode for it.
#
# exp1b      — 4-shard multi-seed DeFi portfolio sweep (seeds 42/99/123/777/2024)
# exp1_ablation — 4-shard Core-15 ablation (PySR-only vs HypatiaX, DEFI_TASKS × 4 workers)
# exp3b      — 4-shard Nguyen-12 multi-seed (seeds 99/123/777/2024)
# instability — CSV shards → _merged.json via _merge_instability_csvs()
MERGE_REQUIRED_EXPERIMENTS: frozenset[str] = frozenset({
    "exp1b",
    "exp1_ablation",
    "exp3b",
    "instability",
})

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

# Corrected mapping: human-readable equation_id → canonical DeFi protocol ID.
# Verified against _get_test_cases() domain fields in hypatiax_defi_benchmark_v3c.py.
#   "Annualised Portfolio tracking error"  -> risk_var  (was "amm"      in legacy versions)
#   "Correlated Portfolio VaR"             -> risk      (was "risk_var"  in legacy versions)
#   "Portfolio VaR for two correlated"     -> risk_var  (was "liquidity" in legacy versions)
EQ_ID_TO_DEFI = {
    "Annualised Portfolio tracking error":        "risk_var",
    "Correlated Portfolio VaR":                   "risk",
    "Portfolio VaR for two correlated":           "risk_var",
    "Portfolio Expected Shortfall for correlated": "expected_shortfall",
    "Portfolio Sharpe Ratio":                     "risk",
    "Portfolio Sortino Ratio":                    "staking",
    "Portfolio Beta":                             "lending",
    "Portfolio Information Ratio":                "trading",
    "Portfolio Maximum Drawdown":                 "derivatives",
    "Portfolio Omega Ratio":                      "liquidation",
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
    # Stats-file top-level keys — skip so merged stats files are never
    # re-ingested as task records.
    "n_total", "n_merged", "n_successes", "success_rate",
    "hyp_extrap_mean", "hyp_extrap_median",
    "nn_extrap_mean", "nn_extrap_median",
}


# ============================================================
# CONFIG
# ============================================================

@dataclass
class MergeConfig:
    experiment: str
    input_root: Path
    output_dir: Path


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
# NORMALISATION
# ============================================================

def canonical_task_id(obj: Dict[str, Any]) -> Optional[str]:
    """Return one deterministic task identity for a record."""
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


def normalise_model_dict(d: Any) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    out = dict(d)
    # Unify test_r2 → extrap_r2 so downstream stats always read extrap_r2.
    if "test_r2" in out and "extrap_r2" not in out:
        out["extrap_r2"] = out["test_r2"]
    return out


def normalise_row(raw: Any) -> Optional[Dict[str, Any]]:
    """
    Normalise one candidate record into the canonical task schema.

    Handles:
      Shape A  nested "results" dict  (DeFi v3 / suppA)
      Shape B  flat top-level fields  (protocol_core_noiseless)

    Renames:
      pure_llm       → hypatia
      neural_network → nn
      test_r2        → extrap_r2  (inside model sub-dicts)

    BUG 3 FIX: the old return dict was hard-coded to 5 keys, so rows whose
    domain == "hybrid" were extracted correctly by extract_rows but then
    silently dropped here — the caller received a record with domain="hybrid"
    but normalise_row returned a dict that omitted nothing wrong structurally;
    the real issue is that hybrid tasks have no canonical task_id derivation
    path and were returning None from canonical_task_id.  They are now
    included via a fallback task_id derived from the domain field.

    BUG 4 FIX: difficulty, formula_type, and extrapolation_intractable were
    never included in the hard-coded return dict and were silently dropped on
    every row.  They are now explicitly preserved via _PASSTHROUGH_FIELDS.
    """
    if not isinstance(raw, dict):
        return None

    row = dict(raw)

    # Flatten nested "results" block if present.
    inner = row.get("results")
    if isinstance(inner, dict):
        inner = dict(inner)
        if "pure_llm" in inner and "hypatia" not in inner:
            inner["hypatia"] = inner.pop("pure_llm")
        if "neural_network" in inner and "nn" not in inner:
            inner["nn"] = inner.pop("neural_network")
        row.update(inner)

    # Rename flat-level aliases.
    if "pure_llm" in row and "hypatia" not in row:
        row["hypatia"] = row.pop("pure_llm")
    if "neural_network" in row and "nn" not in row:
        row["nn"] = row.pop("neural_network")

    hyp = normalise_model_dict(row.get("hypatia") or {})
    nn  = normalise_model_dict(row.get("nn") or {})

    # ABLATION SHAPE FIX (exp1_ablation): records from exp1_ablation.py's
    # checkpoint are flat {eq_key: {"name":..., "domain":..., "pysr_only":{...},
    # "hypatia":{...}}} — i.e. no "results"/"pure_llm"/"neural_network" nesting
    # and no "nn" sub-dict, but a "pysr_only" sub-dict instead.  canonical_task_id()
    # falls back to "domain" (e.g. "Chemistry", "DeFi Risk"), which is shared by
    # multiple Core-15 equations, so merge_rows() would silently collapse all
    # same-domain equations down to a single record.  Detect this shape and use
    # "name" (unique per equation, e.g. "Arrhenius", "Rate Law") as task_id instead.
    is_ablation_row = "pysr_only" in row and "nn" not in row and "results" not in row
    if is_ablation_row and row.get("name"):
        task_id = row["name"]
    else:
        task_id = canonical_task_id(row)
    # BUG 3 FIX: hybrid rows have domain="hybrid" but no equation_id /
    # protocol that maps through EQ_ID_TO_DEFI, so canonical_task_id
    # returned None and the row was discarded.  Fall back to domain so
    # hybrid records survive the merge.
    if not task_id:
        task_id = row.get("domain") or row.get("id") or row.get("name")
    if not task_id:
        return None

    # BUG 4 FIX: build the output from a copy of the full row so no fields
    # are silently dropped, then overwrite the keys we explicitly manage.
    # _PASSTHROUGH_FIELDS (difficulty, formula_type, extrapolation_intractable)
    # are therefore included automatically alongside any other unknown fields
    # that future schema versions may add.
    out = {k: v for k, v in row.items() if k not in META_KEYS}
    out.update({
        "task_id": task_id,
        "name":    row.get("name") or row.get("equation_id") or task_id,
        "domain":  row.get("domain") or task_id,
        "hypatia": hyp,
        "nn":      nn,
    })
    return out


# ============================================================
# PROTOCOL-SHAPE HELPERS (Shape P)
# ============================================================
#
# Shape P is the benchmark-wrapper format produced by NSHARDS=1 runs and by
# the "Locate analysis input" step's _merged_benchmark.json / protocol_core_*.json:
#
#   {"tests": [
#       {"description": ..., "domain": ..., "equation_id": ...,
#        "results": {"pure_llm": {"r2": ..., "success": ...},
#                     "neural_network": {"r2": ..., "success": ...}, ...}}
#   ]}
#
# _is_protocol_file() detects this wrapper; _normalise_protocol_record()
# converts one "tests" entry into the canonical task schema used elsewhere
# in this module (task_id / name / domain / hypatia / nn, with "test_r2"
# inside each model sub-dict).

def _is_protocol_file(raw: Any) -> bool:
    """Return True if `raw` is a Shape P benchmark wrapper.

    Detected by the presence of a non-empty "tests" list whose entries are
    dicts containing a "results" dict (mapping method name -> metrics dict).
    """
    if not isinstance(raw, dict):
        return False
    tests = raw.get("tests")
    if not isinstance(tests, list) or not tests:
        return False
    for test in tests:
        if isinstance(test, dict) and isinstance(test.get("results"), dict):
            return True
    return False


def _normalise_protocol_record(test: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise one Shape-P "tests" entry into the canonical task schema.

    Renames method keys (pure_llm -> hypatia, neural_network -> nn) and,
    within each model sub-dict, renames "r2" -> "test_r2" (via
    normalise_model_dict, which also handles the test_r2 -> extrap_r2 alias)
    so downstream code that reads r["results"][method]["test_r2"] works
    uniformly across Shape P and the merged-shard shapes.
    """
    if not isinstance(test, dict):
        return {}

    row = dict(test)

    inner = dict(row.get("results") or {})
    if "pure_llm" in inner and "hypatia" not in inner:
        inner["hypatia"] = inner.pop("pure_llm")
    if "neural_network" in inner and "nn" not in inner:
        inner["nn"] = inner.pop("neural_network")

    normalised_results: Dict[str, Any] = {}
    for method, metrics in inner.items():
        if not isinstance(metrics, dict):
            normalised_results[method] = metrics
            continue
        m = dict(metrics)
        # r2 -> test_r2 (normalise_model_dict then maps test_r2 -> extrap_r2)
        if "r2" in m and "test_r2" not in m:
            m["test_r2"] = m["r2"]
        normalised_results[method] = normalise_model_dict(m)

    task_id = canonical_task_id(row)
    if not task_id:
        task_id = row.get("domain") or row.get("equation_id") or row.get("name")

    out = {k: v for k, v in row.items() if k not in META_KEYS}
    out.update({
        "task_id": task_id,
        "name":    row.get("name") or row.get("equation_id") or task_id,
        "domain":  row.get("domain") or task_id,
        "results": normalised_results,
        "hypatia": normalised_results.get("hypatia", {}),
        "nn":      normalised_results.get("nn", {}),
    })
    return out


# ============================================================
# EXTRACTION
# ============================================================

def extract_rows(obj: Any) -> List[Dict[str, Any]]:
    """
    Recursively walk an arbitrary JSON structure and collect all records
    that normalise into valid task rows.

    Walks into lists and dict values except META_KEYS subtrees.
    """
    found: List[Dict[str, Any]] = []

    def walk(x: Any) -> None:
        if isinstance(x, list):
            for item in x:
                walk(item)
            return
        if not isinstance(x, dict):
            return
        normalised = normalise_row(x)
        if normalised:
            found.append(normalised)
        for k, v in x.items():
            if k not in META_KEYS:
                walk(v)

    walk(obj)
    return found


# ============================================================
# MERGE POLICY
# ============================================================

def score_row(row: Dict[str, Any]) -> int:
    """Higher score = more complete record; wins in duplicate resolution."""
    score = 0
    h = row.get("hypatia") or {}
    n = row.get("nn") or {}
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
    """Merge extracted rows; highest-score row wins per task_id."""
    merged: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        tid = row["task_id"]
        if tid not in merged or score_row(row) > score_row(merged[tid]):
            merged[tid] = row
    return merged


# ============================================================
# STATS  (basic pre-aggregation only — no experiment-specific tests)
# ============================================================

def build_stats(
    experiment: str,
    merged: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Produce basic aggregation stats for the consolidated dataset.

    Intentionally limited to:
      - record counts and coverage
      - per-model R² mean / median

    Mann-Whitney and other experiment-specific statistical tests are
    performed downstream, after full consolidation, not here.
    """
    hyp_r2: List[float] = []
    nn_r2:  List[float] = []
    successes = 0

    for row in merged.values():
        hr2 = (row.get("hypatia") or {}).get("extrap_r2")
        nr2 = (row.get("nn") or {}).get("extrap_r2")
        if hr2 is not None and not is_nan(hr2):
            hyp_r2.append(float(hr2))
            if hr2 > 0.99:
                successes += 1
        if nr2 is not None and not is_nan(nr2):
            nn_r2.append(float(nr2))

    return {
        "experiment":        experiment,
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "n_merged":          len(merged),
        "n_successes":       successes,
        "success_rate":      (successes / len(merged)) if merged else None,
        "hyp_extrap_mean":   float(np.mean(hyp_r2))   if hyp_r2 else None,
        "hyp_extrap_median": float(np.median(hyp_r2)) if hyp_r2 else None,
        "nn_extrap_mean":    float(np.mean(nn_r2))    if nn_r2  else None,
        "nn_extrap_median":  float(np.median(nn_r2))  if nn_r2  else None,
    }


# ============================================================
# CSV
# ============================================================

def write_csv(path: Path, merged: Dict[str, Any]) -> None:
    rows = [
        "task_id,name,domain,hyp_train_r2,hyp_extrap_r2,nn_extrap_r2,success,best_expression"
    ]
    for tid, row in sorted(merged.items()):
        h  = row.get("hypatia") or {}
        n  = row.get("nn") or {}
        he = h.get("extrap_r2", "")
        ok = isinstance(he, float) and he > 0.99
        expr = str(h.get("best_expression", "")).replace(",", ";")
        rows.append(
            f'{tid},'
            f'{row.get("name", "")},'
            f'{row.get("domain", "")},'
            f'{h.get("train_r2", "")},'
            f'{he},'
            f'{n.get("extrap_r2", "")},'
            f'{ok},'
            f'{expr}'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(rows))


# ============================================================
# CHECKPOINT
# ============================================================

def write_checkpoint(path: Path, experiment: str, result_subdir: str, merged: Dict[str, Any]) -> None:
    # BUG 5 FIX: _checkpoint.json previously omitted result_subdir, so
    # ci_analysis.yml's "Resolve experiment metadata" step always fell through
    # to the dispatch-input fallback and failed on automatic workflow_run
    # triggers where no inputs are provided.  result_subdir is now written
    # here — the consolidate job already has it in scope — so the analysis
    # workflow can resolve it from the artifact without needing manual inputs.
    checkpoint = {
        "experiment":    experiment,
        "result_subdir": result_subdir,
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "n_merged":      len(merged),
        "task_ids":      sorted(merged.keys()),
    }
    safe_write_json(path, checkpoint)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge shard artifacts into consolidated outputs."
    )
    parser.add_argument("--experiment",  required=True,
                        help="Experiment ID (e.g. exp1, exp2_feynman)")
    parser.add_argument("--input-root",  required=True,
                        help="Root directory containing downloaded shard artifacts")
    parser.add_argument("--output-dir",  required=True,
                        help="Directory to write _merged.json / _merged.csv / _stats.json / _checkpoint.json")
    # BUG 5 FIX: result_subdir must be written into _checkpoint.json so
    # ci_analysis.yml can resolve it without manual workflow_dispatch inputs.
    parser.add_argument("--result-subdir", required=True,
                        help="Canonical result subdirectory (e.g. comparison_results/noise-noiseless/noiseless)")
    args = parser.parse_args()

    config = MergeConfig(
        experiment=args.experiment,
        input_root=Path(args.input_root),
        output_dir=Path(args.output_dir),
    )
    result_subdir = args.result_subdir

    logger.info("=" * 70)
    logger.info("HypatiaX Unified Consolidation Engine")
    logger.info("=" * 70)
    logger.info(f"EXPERIMENT : {config.experiment}")
    logger.info(f"INPUT_ROOT : {config.input_root}")
    logger.info(f"OUTPUT_DIR : {config.output_dir}")

    files = sorted(
        glob.glob(f"{config.input_root}/**/*.json", recursive=True)
    )
    logger.info(f"JSON FILES FOUND: {len(files)}")

    all_rows: List[Dict[str, Any]] = []

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

    if not merged:
        raise RuntimeError("FATAL: merge produced zero rows")

    stats = build_stats(config.experiment, merged)

    merged_path     = config.output_dir / "_merged.json"
    csv_path        = config.output_dir / "_merged.csv"
    stats_path      = config.output_dir / "_stats.json"
    checkpoint_path = config.output_dir / "_checkpoint.json"

    safe_write_json(merged_path, merged)
    write_csv(csv_path, merged)
    safe_write_json(stats_path, stats)
    write_checkpoint(checkpoint_path, config.experiment, result_subdir, merged)

    logger.info("=" * 70)
    logger.info(f"WRITE OK: {merged_path}")
    logger.info(f"WRITE OK: {csv_path}")
    logger.info(f"WRITE OK: {stats_path}")
    logger.info(f"WRITE OK: {checkpoint_path}")
    logger.info("=" * 70)

    n = stats["n_merged"]
    sr = stats.get("success_rate")
    hr2_mean = stats.get("hyp_extrap_mean")
    logger.info(
        f"SUMMARY: {n} tasks merged | "
        f"success_rate={sr:.3f}" if sr is not None else f"SUMMARY: {n} tasks merged"
    )
    if hr2_mean is not None:
        logger.info(
            f"  HypatiaX R² mean={hr2_mean:.4f}  "
            f"median={stats['hyp_extrap_median']:.4f}"
        )
    nn_mean = stats.get("nn_extrap_mean")
    if nn_mean is not None:
        logger.info(
            f"  NN baseline  mean={nn_mean:.4f}  "
            f"median={stats['nn_extrap_median']:.4f}"
        )


if __name__ == "__main__":
    main()
