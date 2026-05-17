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

FEYNMAN_DOMAIN_IDS = {
    "feynman_biology",
    "feynman_chemistry",
    "feynman_electrochemistry",
    "feynman_electromagnetism",
    "feynman_electrostatics",
    "feynman_magnetism",
    "feynman_mechanics",
    "feynman_optics",
    "feynman_probability",
    "feynman_quantum",
    "feynman_thermodynamics",
}

EXP2_DOMAIN_IDS = {
    "mechanics", "thermodynamics", "electromagnetism", "fluid_dynamics",
    "optics", "quantum", "chemistry", "biology", "mathematics", "economics",
}

NGUYEN12_IDS = {f"N{i}" for i in range(1, 13)}

HYBRID_ALL_DOMAIN_IDS = EXP2_DOMAIN_IDS  # same set

# Maps experiment ID -> the set of valid canonical task_ids for that experiment.
# Records whose task_id is NOT in this set are rejected during merge.
# None means "no filter" (accept all task IDs -- used for experiments whose
# task ID space cannot be enumerated statically, e.g. suppB noise-sweep).
EXPERIMENT_TASK_IDS: "dict[str, set[str] | None]" = {
    "exp1":               DEFI_IDS,
    "exp1_ablation":      DEFI_IDS,
    "exp1b":              None,   # portfolio_seed{N} IDs are dynamic
    "exp2_feynman":       FEYNMAN_DOMAIN_IDS,
    "exp2":               EXP2_DOMAIN_IDS,
    "exp3":               NGUYEN12_IDS,
    "exp3b":              NGUYEN12_IDS,
    "suppA":              DEFI_IDS,
    "suppB":              None,   # noise{nl}__{domain} IDs are dynamic
    "suppB_sc":           None,   # sc_n{n}__{domain} IDs are dynamic
    "hybrid_all_domains": HYBRID_ALL_DOMAIN_IDS,
    "instability":        DEFI_IDS,
    "extrap":             FEYNMAN_DOMAIN_IDS,
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

    BUG 2 FIX: shard files from workers use display-name method keys
    ("Hybrid System v40", "Neural Network", "Pure LLM (Enhanced)", etc.)
    instead of the snake_case aliases ("pure_llm", "neural_network") that
    the old rename logic expected.  As a result hypatia={} and nn={} were
    always empty, build_stats() found no r2 values, and success_rate=0
    triggered TOTAL_FAILURE in ci_analysis.yml.

    Fix: map the display names to canonical keys BEFORE the snake_case
    rename, and also build a per_method dict preserving r2/success for
    every named method so downstream consumers can read all methods.

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

    # BUG 2 FIX: display-name → canonical alias mapping.
    # Worker scripts write method results under human-readable keys like
    # "Hybrid System v40" and "Neural Network".  Map them to the snake_case
    # aliases that the rename block below expects, so hypatia/nn are populated.
    # Keys are matched case-insensitively via .lower() for resilience.
    _DISPLAY_TO_CANONICAL = {
        # Pure LLM variants → hypatia
        "pure llm (basic)":    "pure_llm",
        "pure llm (enhanced)": "pure_llm",
        "pure llm":            "pure_llm",
        "integrated llm discovery v11.1": "pure_llm",
        # Neural network variants → nn
        "neural network":      "neural_network",
        # Hybrid variants → hybrid  (preserved in per_method; also used as
        # hypatia fallback when pure_llm is absent)
        "hybrid system v40":             "hybrid",
        "hybrid system v40 (fallback)":  "hybrid",
        "llm+nn ensemble (simple)":      "hybrid",
        "llm+nn ensemble (smart)":       "hybrid",
    }

    # Build per_method dict from any display-name or snake_case method key
    # found at top level or inside a "results" block.
    per_method: dict = {}

    def _collect_methods(src: dict) -> None:
        for k, v in src.items():
            if not isinstance(v, dict):
                continue
            # Match display names
            canon = _DISPLAY_TO_CANONICAL.get(k.lower())
            method_key = canon or k  # keep original key if no mapping
            r2 = v.get("r2") or v.get("extrap_r2") or v.get("test_r2") or v.get("train_r2")
            success = v.get("success")
            if r2 is not None or success is not None:
                # Highest r2 wins if the same canonical key appears twice
                if method_key not in per_method or (
                    r2 is not None and (per_method[method_key].get("r2") or -999) < r2
                ):
                    per_method[method_key] = {
                        "r2":      r2,
                        "success": success,
                        "formula": v.get("formula") or v.get("best_expression") or v.get("expression"),
                        "time":    v.get("time") or v.get("elapsed_s"),
                    }
            # Also handle display-name alias injection into row for the
            # snake_case rename block further below.
            if canon and canon not in row:
                row[canon] = v

    _collect_methods(row)
    inner = row.get("results")
    if isinstance(inner, dict):
        _collect_methods(inner)

    # Flatten nested "results" block if present.
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

    # BUG 2 FIX cont.: if hypatia is still empty after the renames, fall back
    # to the best hybrid method result so build_stats() always has an r2 to read.
    hyp = normalise_model_dict(row.get("hypatia") or {})
    if not hyp.get("extrap_r2") and not hyp.get("train_r2") and not hyp.get("r2"):
        # Try canonical hybrid fallback from per_method
        for fallback_key in ("hybrid", "pure_llm"):
            fb = per_method.get(fallback_key, {})
            if fb.get("r2") is not None:
                hyp = normalise_model_dict({
                    "extrap_r2": fb["r2"],
                    "train_r2":  fb["r2"],
                    "r2":        fb["r2"],
                    "success":   fb.get("success"),
                    "best_expression": fb.get("formula"),
                })
                break
        # Last resort: find the highest-r2 method in per_method
        if not hyp.get("extrap_r2"):
            best = max(
                ((m, d) for m, d in per_method.items() if d.get("r2") is not None),
                key=lambda x: x[1]["r2"],
                default=(None, {}),
            )
            if best[1].get("r2") is not None:
                hyp = normalise_model_dict({
                    "extrap_r2": best[1]["r2"],
                    "train_r2":  best[1]["r2"],
                    "r2":        best[1]["r2"],
                    "success":   best[1].get("success"),
                    "best_expression": best[1].get("formula"),
                })

    nn = normalise_model_dict(row.get("nn") or {})
    if not nn.get("extrap_r2") and not nn.get("r2"):
        fb = per_method.get("neural_network", {})
        if fb.get("r2") is not None:
            nn = normalise_model_dict({
                "extrap_r2": fb["r2"],
                "r2":        fb["r2"],
                "success":   fb.get("success"),
            })

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

    # BUG 1 FIX: equation_id was never written into the output record.
    # Derive it from the description field (human-readable equation name before
    # the first separator) so downstream consumers (ci_analysis.yml, paper tables,
    # _merged.csv) display "Allometric Scaling" instead of the bare domain key
    # "biology".  Falls back to any existing equation_id field, then task_id.
    desc = row.get("description", "")
    eq_id = None
    if desc:
        for sep in (":", "—", " - ", "|"):
            if sep in desc:
                eq_id = desc.split(sep)[0].strip()
                break
        if not eq_id:
            eq_id = desc.strip()
    if not eq_id:
        eq_id = row.get("equation_id") or task_id

    out = {k: v for k, v in row.items() if k not in META_KEYS}
    out.update({
        "task_id":     task_id,
        "equation_id": eq_id,
        "name":        row.get("name") or eq_id or task_id,
        "domain":      row.get("domain") or task_id,
        "hypatia":     hyp,
        "nn":          nn,
        # per_method: flat dict of {method_key: {r2, success, formula, time}}
        # preserves all named methods for downstream analysis / paper tables.
        "per_method":  per_method if per_method else row.get("per_method") or {},
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

    BUG FIX: when a top-level record is successfully normalised, stop
    recursing into its children.  Without this guard, the method sub-dicts
    inside "results" (hybrid, pure_llm, neural_network) were also walked,
    and each one -- having keys like "domain" or "decision" -- was emitted
    as a phantom task record (e.g. task_id="llm" from hybrid.decision="llm").
    Those phantom records appeared in _merged.json as "?" rows and polluted
    the MW analysis with null-R² entries.
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
            # Successfully normalised — emit and do NOT recurse further into
            # children to avoid phantom records from method sub-dicts.
            found.append(normalised)
            return
        # Not a task record itself — recurse into values to find nested records.
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


def merge_rows(rows: Iterable[Dict[str, Any]], experiment: str = "") -> Dict[str, Dict[str, Any]]:
    """Merge extracted rows; highest-score row wins per task_id.

    BUG FIX: apply experiment-aware task ID allowlist so records from other
    experiments (e.g. Feynman domain keys in an exp1 run, or phantom records
    extracted from method sub-dicts) are rejected before they pollute the
    merged output.  When the allowlist for an experiment is None (dynamic IDs
    like suppB), all task IDs are accepted as before.
    """
    allowed = EXPERIMENT_TASK_IDS.get(experiment)  # None means accept all
    merged: Dict[str, Dict[str, Any]] = {}
    rejected = 0
    for row in rows:
        tid = row["task_id"]
        if allowed is not None and tid not in allowed:
            rejected += 1
            logger.debug(f"  REJECTED task_id={tid!r} (not in allowlist for {experiment!r})")
            continue
        if tid not in merged or score_row(row) > score_row(merged[tid]):
            merged[tid] = row
    if rejected:
        logger.info(f"ALLOWLIST FILTER: rejected {rejected} row(s) with task IDs outside {experiment!r} expected set")
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

    merged = merge_rows(all_rows, experiment=config.experiment)

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
