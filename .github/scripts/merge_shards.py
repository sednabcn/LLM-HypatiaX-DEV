#!/usr/bin/env python3
"""
.github/scripts/merge_shards.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HypatiaX  ·  Consolidate per-shard partial JSONs → final experiment result

MERGE_REQUIRED experiments (called by ci_runner.yml consolidate job OR
ci_analysis.yml inline): exp1b, exp3b, exp2_feynman.
All other experiments are single-worker and do NOT use this script in CI
(exp3 uses a direct shard-file read path in ci_analysis.yml).

Called by ci_experiment.yml consolidate job (Job 3):

    python .github/scripts/merge_shards.py \\
        --experiment    "${EXP}" \\
        --input-root    downloaded_artifacts \\
        --output-dir    "${OUT_BASE}/${RESULT_SUBDIR}" \\
        --result-subdir "${RESULT_SUBDIR}"

Source of truth: run_comparative_suite_benchmark_v2.py
  · EXP_CONFIG shard_globs   ← actual output filenames produced by the worker
  · array_key per experiment  ← "tests" for protocol_core_*.json files,
                                 "results" for legacy shapes
  · result_subdir             ← canonical per-experiment output path

Writes four canonical output files into --output-dir:
    _merged.json        all task records merged by task_id
    _merged.csv         flat CSV view
    _stats.json         pre-aggregated counts + R² summaries
    _checkpoint.json    provenance / run metadata (consumed by ci_analysis.yml)

Changes in current revision:
  · FIX-GLOBS: EXP_CONFIG shard_globs updated for every experiment to match
    the actual filenames written by run_comparative_suite_benchmark_v2.py:
      - exp1/exp1b/exp2/exp2_feynman/extrap: primary output is
        protocol_core_{noiseless|noisy}_<timestamp>.json (Shape {tests:[...]}).
        Previous globs matched zero files for extrap, exp2, exp2_feynman.
      - exp2/exp2_feynman checkpoints: named exp2_checkpoint_<domain>.json and
        feynman_exp2_checkpoint_feynman_<domain>.json respectively — not the
        _shard* suffix that was previously listed.
      - suppA: actual file is extrapolation_*enhanced*.json; previous globs
        (consolidated_hybrid_*.json etc.) matched nothing.
      - suppB/suppB_sc: added protocol_core_*.json as fallback.
      - exp1 result_subdir corrected: noiseless/defi (was noiseless/).
      - suppB result_subdir corrected: noise-sweep (matches ci_runner.yml RESULT_SUBDIR;
        the /noise-sweep nested-dir assumption was wrong — runner writes directly to
        comparison_results/feynman-tests/noise-sweep).
  · FIX-ARRAY-KEY: array_key changed from "results" to "tests" for all
    experiments that use the protocol_core_*.json wrapper shape
    (exp1, exp1b, exp2, exp2_feynman, extrap, hybrid_all_domains, suppB, suppB_sc).
    Previous array_key="results" caused _extract_records to find no list,
    fall through to Shape C/D, and misread the structure entirely.
  · FIX-METHOD-NAMES: _normalise_protocol_record() added. Translates raw method
    strings from the worker ("PureLLM Baseline (core)" etc.) to the canonical
    keys run_analysis.py reads ("pure_llm", "neural_network", "hybrid").
    Also maps field names: r2→test_r2, time→time_s.
    Applied automatically when _is_protocol_file() detects Shape P.
    Skipped only for exp1_ablation (hypatia/pysr_only schema). exp2_feynman
    is normalised like exp2 — it uses the same v2 worker method schema.
  · FIX-EXP-ID: _exp_id injected into cfg in merge_experiment() so
    _extract_records() can gate ablation-vs-standard normalisation per-file.
    Only exp1_ablation is in _ABLATION_EXPERIMENTS; exp2_feynman is standard.
  · _extract_records() gains Shape P (protocol wrapper) before existing
    Shape A/B/C/D, with explicit _is_protocol_file() guard.
  · instability added to EXP_CONFIG (array_key=None triggers CSV-only path).
  · _is_solved() replaces inline r.get("status") == "ok" in _compute_stats().
  · _compute_stats() iterates r2/r2_score/r2_noiseless/best_r2; also checks
    normalised test_r2 field produced by _normalise_protocol_record().
  · Zero-solve warning emitted to stderr when n_solved=0 and n_tasks>0.
  · FIX-EMPTY-FAIL: merge_experiment() now hard-fails (returns 1) with a
    detailed diagnostic when _find_shard_files() returns 0 files OR when
    0 records are extracted from the files found.  Both conditions previously
    silently wrote an empty _merged.json and returned 0, causing EMPTY_DATASET
    to fire in ci_analysis.yml instead of at the merge step where the real
    fault lies.  The new diagnostics include:
      - A recursive listing of everything present under --input-root (so glob
        mismatches are immediately visible in the CI log).
      - Per-file top-key summary when files are found but yield 0 records
        (so array_key mismatches are immediately visible).
  · FIX-CSV: _find_shard_files() allowed suffixes derived from shard_globs.
  · FIX-SKIP-ANALYSIS: _pick_shard_file() _SKIP_NAMES now includes
    _analysis.json so a pipeline output from a prior run can never be
    returned as a valid shard input.
  · FIX-SHAPE-A-DICT: _extract_records() now handles both dict-under-array_key
    variants produced by the exp3/exp3b runner:
      A-dict-of-lists: {"results": {"hypatiax": [...], "pysr": [...]}}
        Actual runner output — system names as keys, lists of records as values.
        Previously fell through all Shape A guards (which expected list or
        dict-of-dict values) and extracted 0 records, causing EMPTY DATASET
        in both merge_shards.py and validate_analysis_input.py.
        New branch checks all(isinstance(v, list)) and flattens all system
        lists into one flat record list.
      A-dict-of-tasks: {"results": {"N1": {...}, "N2": {...}}}
        Documented Nguyen schema — equation IDs as keys, record dicts as values.
        Preserved unchanged; evaluated after A-dict-of-lists so list-valued
        inner dicts never fall into this branch."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  Per-experiment configuration
#  Mirrors run_all_checkpoint.py: Step.result_glob, Step.post_move, and
#  EXP_RESULT_SUBDIR; plus the merge_key each benchmark script uses.
# ─────────────────────────────────────────────────────────────────────────────
# Experiments that REQUIRE shard merging
MERGE_REQUIRED_EXPERIMENTS = {
    "exp1b",
    "exp3b",
    "instability",   # CSV-only; merge_shards.py converts CSVs → _merged.json
}

EXP_CONFIG: dict[str, dict] = {

    # ── exp1: Core DeFi extrapolation (noiseless) ─────────────────────────────
    # Worker (run_comparative_suite_benchmark_v2.py) writes to --output-dir:
    #   protocol_core_noiseless_<timestamp>.json  ← Shape: {tests:[...]}  ← PRIMARY
    #   benchmark_results.json                    ← flat list (FIX-7 convenience export)
    #   hypatiax_defi_benchmark_v3_results*.json  ← if present from legacy run
    # array_key="tests" matches the {tests:[...]} wrapper shape.
    # _normalise_protocol_record() pivots each tests[] entry into the canonical
    # per-equation schema run_analysis.py expects.
    "exp1": dict(
        result_subdir="comparison_results/noise-noiseless/noiseless/defi",
        shard_globs=[
            "protocol_core_noiseless_*.json",    # primary output from v2 worker
            "protocol_core_noisy_*.json",        # noisy variant fallback
            "hypatiax_defi_benchmark_v3_results.json",   # exact legacy name
            "hypatiax_defi_benchmark_v3*results*.json",  # any legacy variant
            "hypatiax_defi_benchmark_v3*.json",
            "defi_v3_*.json",
        ],
        merge_key="equation_id",
        fallback_keys=["task_id", "equation", "name", "description"],
        array_key="tests",   # {tests:[{description, domain, results:{method:{r2,...}}}]}
    ),

    # ── exp1b: DeFi seed sweep + portfolio variance (noise=15) ───────────────
    # 4-worker experiment → merge_shards.py is called → _merged.json produced.
    # Each shard writes protocol_core_noiseless_<timestamp>.json (Shape {tests:[...]})
    # plus comparison_FIXED_<timestamp>.json from the portfolio-variance path.
    "exp1b": dict(
        result_subdir="comparison_results/noise-noiseless/15",
        shard_globs=[
            "protocol_core_noiseless_*.json",    # primary shard output
            "protocol_core_noisy_*.json",
            "comparison_FIXED_*.json",           # portfolio-variance output
            "hypatiax_defi_benchmark_v3*.json",
            "*portfolio*variance*.json",
            "defi_v3_*.json",
        ],
        merge_key="equation_id",
        fallback_keys=["task_id", "equation", "name", "description"],
        array_key="tests",
    ),

    # ── exp2_feynman: Feynman SR noisy benchmark — standard multi-method runner ──
    # Single-worker experiment using run_comparative_suite_benchmark_v2.py
    # (same runner as exp1, exp2, extrap). Worker commits directly to the repo;
    # ci_analysis.yml runs merge_shards.py inline to produce _merged.json.
    #
    # Actual files committed by the worker:
    #   protocol_core_noiseless_<timestamp>.json  ← Shape P {tests:[...]}  PRIMARY
    #   benchmark_results.json                    ← flat list (Shape B) fallback
    #   feynman_exp2_checkpoint_feynman_<domain>.json  ← per-domain checkpoint
    #
    # OPTION A FIX: exp2_feynman is NOT ablation. The worker never produces
    # hypatia/pysr_only/extrap_r2_far keys. It outputs PureLLM Baseline (core),
    # ImprovedNN (core), EnhancedHybridSystemDeFi (core), SymbolicEngineWithLLM
    # (tools) — the same method schema as exp2.
    # _normalise_protocol_record() maps these to canonical slugs (pure_llm,
    # neural_network, hybrid, symbolic_engine); run_analysis.py routes to
    # analyse() in multi_method mode. _extract_records is_ablation=False.
    "exp2_feynman": dict(
        result_subdir="comparison_results/feynman-tests/exp2",
        shard_globs=[
            "protocol_core_noiseless_*.json",    # PRIMARY — Shape P {tests:[...]} from v2 worker
            "protocol_core_noisy_*.json",        # noisy variant
            "benchmark_results.json",            # flat list fallback (Shape B)
            "feynman_exp2_checkpoint_feynman_*.json",  # actual checkpoint names in tree
            "exp2_feynman_checkpoint_*.json",    # legacy naming
            "exp2_feynman_merged*.json",
            "exp2_results.json",
            "I_*.json",
            "II_*.json",
            "III_*.json",
        ],
        merge_key="equation_id",   # set by _normalise_protocol_record() on Shape P
        fallback_keys=["equation", "description", "task_id", "name"],
        array_key="tests",   # Shape P: {tests:[{description, domain, results:{...}}]}
    ),

    # ── exp2: Combined five-system comparison — all methods ───────────────────
    # Worker writes protocol_core_noiseless_<timestamp>.json (Shape {tests:[...]})
    # and per-domain checkpoints: exp2_checkpoint_<domain>.json (NOT _shard suffix).
    # multi_method mode: run_analysis.py expects canonical method names.
    # _normalise_protocol_record() maps raw method strings → canonical keys.
    "exp2": dict(
        result_subdir="comparison_results/feynman-tests/exp2_multi",
        shard_globs=[
            "protocol_core_noiseless_*.json",    # primary per-shard output
            "protocol_core_noisy_*.json",
            "exp2_checkpoint_*.json",            # actual checkpoint names in tree (no _shard)
            "exp2_checkpoint_shard*.json",       # legacy naming fallback
            "exp2_merged*.json",
            "exp2_stats.json",
        ],
        merge_key="equation_id",
        fallback_keys=["equation", "task_id", "name", "description"],
        array_key="tests",
    ),

    # ── exp3: Nguyen-12 SEED=42 ───────────────────────────────────────────────
    # Single-worker experiment — merge_shards.py is NOT called by ci_analysis.yml.
    # ci_runner.yml commits exp3_nguyen12_seed42.json directly; ci_analysis.yml
    # reads it via the shard-file fallback path (no _merged.json produced).
    # This EXP_CONFIG entry supports manual standalone invocations only.
    # exp3_nguyen12_hybrid50v_02.py writes its own JSON schema under array_key
    # "results" (not the protocol_core wrapper). pysr mode in run_analysis.py
    # means no method-name normalisation is applied.
    "exp3": dict(
        result_subdir="extrapolation",
        shard_globs=[
            "*nguyen*seed42*.json",
            "*nguyen12*42*.json",
            "full_run_*.json",
            "report_hybrid_*.json",
            "hybrid_defi_*.json",
        ],
        merge_key="equation",
        fallback_keys=["task_id", "name", "equation_id"],
        array_key="results",
    ),

    # ── exp3b: Nguyen-12 seeds 99/123/777/2024 ────────────────────────────────
    # 4-worker experiment → merge_shards.py IS called by ci_runner.yml consolidate
    # job → _merged.json committed to repo → ci_analysis.yml reads it directly.
    # Each shard writes *nguyen*seed<N>*.json via exp3_nguyen12_hybrid50v_02.py.
    "exp3b": dict(
        result_subdir="extrapolation/multi_seed",
        shard_globs=[
            "*nguyen*.json",
            "full_run_*.json",
            "report_hybrid_*.json",
            "hybrid_defi_*.json",
        ],
        merge_key="equation",
        fallback_keys=["task_id", "name", "equation_id"],
        array_key="results",
    ),

    # ── suppA: Hybrid-PySR DeFi benchmark ────────────────────────────────────
    # Tree shows: extrapolation_73cases_enhanced.json in hybrid_pysr/defi/
    # Previous globs (consolidated_hybrid_*.json, hybrid_system*.json) did not
    # match. Added extrapolation_*enhanced*.json and *extrapolation*.json as
    # primary globs alongside the legacy names.
    "suppA": dict(
        result_subdir="hybrid_pysr/defi",
        shard_globs=[
            "extrapolation_*enhanced*.json",     # actual file: extrapolation_73cases_enhanced.json
            "extrapolation_*.json",              # any extrapolation result
            "consolidated_hybrid_*.json",
            "hybrid_system*.json",
            "hybrid_llm_nn_all_domains_*.json",
            "ablation_exp1_*.json",
        ],
        merge_key="equation_id",
        fallback_keys=["domain", "task_id", "equation", "name"],
        array_key="results",
    ),

    # ── suppB: Noise sweep σ ∈ {0, 0.5, 1, 5, 10}% × 30 equations ───────────
    # ci_runner.yml plan sets RESULT_SUBDIR="comparison_results/feynman-tests/noise-sweep"
    # and the move_matching step writes all top-level outputs to TARGET (that path).
    # Per-equation sub-dirs are rescued to noise-sweep/<eq-dir>/ by the inline
    # find loop in ci_runner.yml, but top-level noise_sweep_*.json files land at
    # the flat noise-sweep/ level — there is NO nested noise-sweep/noise-sweep dir.
    # run_noise_sweep_benchmark.py writes:
    #   noise_sweep_<timestamp>.json  ← consolidated result (Shape: list or {results:[...]})
    #   noise_sweep_sig<N>_checkpoint.json  ← per-sigma checkpoint
    #   protocol_core_noisy_<timestamp>.json  ← if using v2 worker
    # task_id format: "noise{σ}__{feynman_domain}"  e.g. "noise5.0__feynman_mechanics"
    "suppB": dict(
        result_subdir="comparison_results/feynman-tests/noise-sweep",
        shard_globs=[
            "noise_sweep_*.json",
            "protocol_core_noisy_*.json",
            "protocol_core_noiseless_*.json",
            "suppB_*.json",
        ],
        merge_key="task_id",
        fallback_keys=["equation", "name", "equation_id"],
        array_key="results",
    ),

    # ── suppB_sc: Sample-complexity sweep n ∈ {50…1000} × 30 equations ───────
    # run_sample_complexity_benchmark.py writes:
    #   sample_complexity_<timestamp>.json  ← consolidated result
    #   sample_complexity_n<N>_checkpoint.json  ← per-n checkpoint
    #   protocol_core_noisy_<timestamp>.json  ← if using v2 worker
    # task_id format: "sc_n{n}__{feynman_id}"  e.g. "sc_n200__I.6.20"
    "suppB_sc": dict(
        result_subdir="comparison_results/feynman-tests/sample-complexity",
        shard_globs=[
            "sample_complexity_*.json",
            "sample_complexity_*.csv",
            "protocol_core_noisy_*.json",
            "protocol_core_noiseless_*.json",
        ],
        merge_key="task_id",
        fallback_keys=["equation", "name", "equation_id"],
        array_key="results",
    ),

    # ── hybrid_all_domains: LLM+NN all-domains one-shot ──────────────────────
    # hybrid_system_llm_nn_all_domains.py writes hybrid_llm_nn_all_domains_*.json
    # Glob already matches actual file (hybrid_llm_nn_all_domains_20260522_200646.json).
    # multi_method mode: _normalise_protocol_record() maps method names.
    "hybrid_all_domains": dict(
        result_subdir="hybrid_llm_nn/all_domains",
        shard_globs=[
            "hybrid_llm_nn_all_domains_*.json",
            "protocol_core_noiseless_*.json",
            "protocol_core_noisy_*.json",
        ],
        merge_key="equation_id",
        fallback_keys=["domain", "task_id", "equation", "name"],
        array_key="results",
    ),

    # ── exp2_feynman_extrap: OOD extrapolation step for exp2_feynman ──────────
    # Runs AFTER exp2_feynman completes.  merge_extrap_into_benchmark.py reads
    # the exp2_feynman benchmark output and generates protocol_core_extrap_*.json
    # files containing extrap_r2_far values, then writes ablation_paired.json
    # into the same result_subdir (comparison_results/feynman-tests/extrap).
    # NSHARDS=1: single-worker step — merge_shards.py is NOT called by CI.
    # ci_analysis.yml routes this experiment via the DIRECT input mode, reading
    # protocol_core_extrap_*.json directly from RESULT_DIR.
    # run_analysis.py reads ablation_paired.json (ablation mode) when present.
    "exp2_feynman_extrap": dict(
        result_subdir="comparison_results/feynman-tests/extrap",
        shard_globs=[
            "protocol_core_extrap_*.json",       # PRIMARY — written by merge_extrap_into_benchmark.py
            "ablation_paired.json",              # paired {hypatia,pysr_only}.extrap_r2_far schema
            "protocol_core_noiseless_*.json",    # fallback from v2 worker
            "protocol_core_noisy_*.json",
        ],
        merge_key="equation_id",
        fallback_keys=["equation", "task_id", "name", "description"],
        array_key="tests",
    ),

    # ── exp1_ablation: paired pysr_only vs hypatia ablation (manual-only) ────
    # NOT dispatched by ci_experiment.yml or ci_schedule_all.yml.
    # Kept here to support manual standalone runs alongside exp2_feynman.
    # Uses the same ablation schema (hypatia/pysr_only keys, extrap_r2_far);
    # routes to analyse_ablation() in run_analysis.py.
    # If promoted to CI, entries are also needed in ci_experiment.yml and
    # ci_analysis.yml (result_subdir mapping + dispatch menu).
    "exp1_ablation": dict(
        result_subdir="comparison_results/feynman-tests/exp1_ablation",
        shard_globs=[
            "exp1_ablation_checkpoint_shard*.json",
            "exp1_ablation_merged*.json",
            "exp1_ablation_results.json",
            "exp1_ablation_stats.json",
        ],
        merge_key="equation",
        fallback_keys=["equation_name", "equation_id", "task_id", "name"],
        array_key="results",
    ),

    # ── instability: Instability Index S10.9 (CSV-only path) ─────────────────
    # run_instability_suite.py writes:
    #   instability_analysis.csv      → figures/
    #   instability_extrapolation.csv → figures/ (Stage 2, if benchmark JSON present)
    #   fig_paper_*.{png,pdf}         → figures/ (12 figure stems)
    # array_key=None signals merge_experiment() to take the CSV-only branch
    # (_merge_instability_csvs) instead of the JSON record path.
    "instability": dict(
        result_subdir="figures",
        shard_globs=[
            "instability_analysis.csv",
            "instability_extrapolation.csv",
            "instability_*.csv",
        ],
        merge_key="case_id",
        fallback_keys=["equation", "name", "task_id"],
        array_key=None,   # CSV-only: triggers _merge_instability_csvs()
    ),

    # ── extrap: OOD extrapolation comparative suite ───────────────────────────
    # run_comparative_suite_benchmark_v2.py --extrap --output-dir <RESULT_SUBDIR>
    # writes to the output dir:
    #   protocol_core_noisy_<timestamp>.json     ← Shape {tests:[...]}  PRIMARY
    #   extrap_checkpoint_feynman_<domain>.json  ← per-domain checkpoint (actual in tree)
    #   benchmark_results.json                   ← flat convenience export (FIX-7)
    # Previous globs (all_domains_extrap_v4_*.json etc.) matched NO files in the tree.
    # ood mode: _normalise_protocol_record() maps raw method names → canonical.
    "extrap": dict(
        result_subdir="comparison_results/extrapolation",
        shard_globs=[
            "protocol_core_noisy_*.json",        # primary shard output
            "protocol_core_noiseless_*.json",    # noiseless variant
            "extrap_checkpoint_feynman_*.json",  # actual per-domain checkpoint names in tree
            "extrap_checkpoint_*.json",          # any extrap checkpoint
            "all_domains_extrap_v4_*.json",      # legacy naming
            "standalone_llm_nn_*.json",
            "standalone_real_methods_*.json",
        ],
        merge_key="equation_id",
        fallback_keys=["equation", "task_id", "name", "description"],
        array_key="tests",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_shard_files(root: Path, globs: list[str]) -> list[Path]:
    """
    Collect all JSON/CSV files under `root` (recursively) matching any glob.
    Skips:
      · _report.md and any non-.json/.csv files (they are never result data)
      · our own output files (_merged.json, _merged.csv etc.)
      · worker checkpoint/stub files
    Each unique Path is returned once regardless of how many globs match it.

    NOTE: .csv files are allowed to support experiments whose shard_globs
    explicitly include CSV patterns (instability, suppB_sc).  Previously the
    hard .json-only filter silently discarded every CSV match, so those
    experiments always returned 0 shard files.
    """
    found: list[Path] = []
    seen:  set[Path]  = set()

    # Always skip these by name — they appear in every shard artifact dir
    _SKIP_NAMES = frozenset({
        "_report.md", "_merged.json", "_merged.csv",
        "_stats.json", "_checkpoint.json",
        "_analysis.json",   # pipeline output from run_analysis.py — never a valid shard
    })

    # Derive allowed suffixes from the glob patterns; always include .json.
    # Any glob that ends in ".csv" (case-insensitive) also admits .csv files.
    _allowed_suffixes: frozenset[str] = frozenset({".json"} | {
        Path(g).suffix.lower()
        for g in globs
        if Path(g).suffix.lower() in (".json", ".csv")
    })

    for pattern in globs:
        for match in sorted(root.rglob(pattern)):
            if not match.is_file():
                continue
            if match in seen:
                continue
            if match.name in _SKIP_NAMES:
                continue
            if match.suffix.lower() not in _allowed_suffixes:
                continue
            if "_assembled" in match.name:
                continue
            seen.add(match)
            found.append(match)

    return found


def _is_stub(raw: object) -> bool:
    """True for {"_meta": {"stub": true}} written by FIX-G6."""
    return (
        isinstance(raw, dict)
        and isinstance(raw.get("_meta"), dict)
        and raw["_meta"].get("stub") is True
    )


def _is_worker_checkpoint(raw: object) -> bool:
    """True for checkpoint_worker_shard*.json files (task tracking, not results)."""
    return isinstance(raw, dict) and "completed" in raw and "run_id_map" in raw


# ---------------------------------------------------------------------------
# Protocol-file normalisation
# ---------------------------------------------------------------------------
# run_comparative_suite_benchmark_v2.py writes:
#   { "tests": [ { "description": str, "domain": str,
#                  "results": { "<RawMethodName>": { "r2", "success", "time", ... } },
#                  "winner": str, "timestamp": str } ] }
#
# run_analysis.py expects each record to look like:
#   { "equation_id": str, "difficulty": str, "formula_type": str,
#     "extrapolation_intractable": bool,
#     "results": { "pure_llm":       { "test_r2", "success", "time_s", ... },
#                  "neural_network": { ..., "timed_out": bool },
#                  "hybrid":         { ..., "decision": str } } }
#
# This mapping translates raw method strings → canonical keys.
# Extra methods (symbolic_engine, hybrid_v50_2, hybrid_all_domains) are kept
# under their canonical slugs so multi_method mode can see them; standard/ood
# analysis only reads pure_llm/neural_network/hybrid.
_RAW_METHOD_TO_CANONICAL: dict[str, str] = {
    "PureLLM Baseline (core)":              "pure_llm",
    "ImprovedNN (core)":                    "neural_network",
    "EnhancedHybridSystemDeFi (core)":      "hybrid",
    "HybridSystemLLMNN all-domains (core)": "hybrid_all_domains",
    "SymbolicEngineWithLLM (tools)":        "symbolic_engine",
    "HybridDiscoverySystem v50_2 (tools)":  "hybrid_v50_2",
}

# Experiments whose records must NOT be normalised: the ablation schema uses
# hypatia/pysr_only keys and extrap_r2_far — run_analysis.py reads them
# directly via analyse_ablation(), so re-shaping would break it.
# exp2_feynman is NOT ablation: it uses run_comparative_suite_benchmark_v2.py
# and produces the standard PureLLM/ImprovedNN/Hybrid/SymbolicEngine schema.
# merge_shards.py normalises it to canonical slugs; run_analysis.py routes it
# to multi_method mode (same as exp2).
_ABLATION_EXPERIMENTS = {"exp1_ablation"}


def _is_protocol_file(raw: object) -> bool:
    """True when raw is a protocol_core_*.json top-level wrapper {tests:[...]}."""
    return (
        isinstance(raw, dict)
        and isinstance(raw.get("tests"), list)
        and raw.get("tests")
        and isinstance(raw["tests"][0], dict)
        and "results" in raw["tests"][0]
        and isinstance(raw["tests"][0]["results"], dict)
        # Distinguish from other {tests:[...]} shapes by checking a method value
        # is itself a dict with r2/success (not a list of equation records).
        and any(
            isinstance(v, dict) and ("r2" in v or "success" in v)
            for v in raw["tests"][0]["results"].values()
        )
    )


def _normalise_protocol_record(test: dict) -> dict:
    """
    Convert one entry from protocol_core_*.json tests[] into the canonical
    per-equation record shape that run_analysis.py expects.

    Input:  { description, domain, results: {RawMethodName: {r2, success, time, ...}} }
    Output: { equation_id, equation, domain, difficulty, formula_type,
              extrapolation_intractable,
              results: { canonical_method: {test_r2, train_r2, success, time_s,
                                            extrapolation_gap, stability_score} } }
    """
    desc   = test.get("description", "")
    domain = test.get("domain", "")

    # Derive equation_id from description: take text before first separator.
    eq_id = desc
    for sep in (" — ", " - ", ": ", " | "):
        if sep in desc:
            eq_id = desc.split(sep)[0].strip()
            break

    canonical_results: dict = {}
    for raw_name, res in test.get("results", {}).items():
        if not isinstance(res, dict):
            continue
        canonical = _RAW_METHOD_TO_CANONICAL.get(raw_name)
        if canonical is None:
            # Unknown method: keep under a slugified name so it's not lost.
            canonical = raw_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        canonical_results[canonical] = {
            "train_r2":          None,               # not produced by v2 worker
            "test_r2":           res.get("r2"),
            "success":           res.get("success", False),
            "time_s":            res.get("time"),
            "extrapolation_gap": res.get("extrap_r2"),  # present in --extrap runs
            "stability_score":   None,
            # Preserve extra fields that run_analysis.py may read conditionally.
            "timed_out":         res.get("metadata", {}).get("timed_out", False)
                                 if isinstance(res.get("metadata"), dict) else False,
            "decision":          res.get("decision"),
        }

    return {
        "equation_id":              eq_id,
        "equation":                 eq_id,
        "description":              desc,
        "domain":                   domain,
        "difficulty":               test.get("difficulty"),
        "formula_type":             test.get("formula_type"),
        "extrapolation_intractable": test.get("extrapolation_intractable", False),
        "winner":                   test.get("winner"),
        "results":                  canonical_results,
    }


def _collect_run_id_map(shard_files: list[Path]) -> dict[str, str]:
    """
    Scan all shard files and collect the merged run_id_map from any worker
    checkpoint files (which are otherwise skipped by _extract_records).

    run_id_map is written by the worker step 'Set consolidate outputs' as:
        {"task_id": "stable_run_id", ...}
    and lets _enrich_equation_id patch task_id from the stable run identifier
    so records are consistently keyed even when a shard re-runs under a new
    GitHub run_id.
    """
    merged_map: dict[str, str] = {}
    for fpath in shard_files:
        try:
            raw = json.loads(fpath.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if _is_worker_checkpoint(raw):
            run_id_map = raw.get("run_id_map", {})
            if isinstance(run_id_map, dict):
                merged_map.update(run_id_map)
    return merged_map


def _extract_records(filepath: Path, cfg: dict) -> list[dict]:
    """
    Load one partial result file and return a flat list of task record dicts.

    Handles the shapes the HypatiaX benchmark scripts produce:

    P.               {"tests": [{description, domain, results:{RawMethod:{r2,...}}}]}
                     protocol_core_*.json from run_comparative_suite_benchmark_v2.py.
                     Normalised to canonical schema unless experiment is ablation.
    A-list.          {"results": [{...}, ...]}
                     Wrapper dict with list under array_key.
    A-dict-of-lists. {"results": {"hypatiax": [{...},...], "pysr": [{...},...]}}
                     Wrapper dict; array_key maps system names to lists of records.
                     Actual exp3/exp3b runner output (seed42, seed2024, etc.).
    A-dict-of-tasks. {"results": {"N1": {...}, "N2": {...}}}
                     Wrapper dict; array_key maps equation IDs to single record dicts.
                     Documented Nguyen schema (dict-of-tasks variant).
    B.               [{...}, ...]  Top-level list.
    C.               {"task_id": {...}, "task_id2": {...}}
                     Top-level dict keyed by task identifiers (no array_key wrapper).
    D.               {single record}  One task record as a bare dict.
    """
    try:
        raw = json.loads(filepath.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        print(f"  ⚠  JSON error in {filepath.name}: {exc}", file=sys.stderr)
        return []

    if _is_stub(raw) or _is_worker_checkpoint(raw):
        return []

    array_key     = cfg.get("array_key")
    merge_key     = cfg["merge_key"]
    fallback_keys = cfg.get("fallback_keys", [])
    exp_id        = cfg.get("_exp_id", "")   # injected by merge_experiment()
    is_ablation   = exp_id in _ABLATION_EXPERIMENTS

    # Shape P — protocol_core_*.json: {tests:[{description, domain, results:{method:{...}}}]}
    # Apply only when array_key is "tests" AND the file is actually a protocol wrapper.
    # Ablation experiments skip normalisation (their records use hypatia/pysr_only keys).
    if array_key == "tests" and _is_protocol_file(raw):
        records = []
        for test in raw["tests"]:
            if not isinstance(test, dict):
                continue
            if is_ablation:
                # Ablation: pass through raw test record; run_analysis.py reads it directly.
                rec = dict(test)
                if not rec.get(merge_key):
                    for fb in fallback_keys:
                        if rec.get(fb):
                            rec[merge_key] = str(rec[fb])
                            break
                records.append(rec)
            else:
                records.append(_normalise_protocol_record(test))
        return records

    # Shape A-list — wrapper dict with list under array_key
    if array_key and isinstance(raw, dict) and isinstance(raw.get(array_key), list):
        return [r for r in raw[array_key] if isinstance(r, dict)]

    # Shape A-dict-of-lists — wrapper dict with dict-of-lists under array_key
    # (exp3/exp3b actual runner output: {"results": {"hypatiax": [...], "pysr": [...]}})
    # Keys are system/method names; values are lists of per-equation records.
    # Must be checked BEFORE A-dict-of-tasks: both have a dict under array_key,
    # but this shape has list values while A-dict-of-tasks has dict values.
    if array_key and isinstance(raw, dict) and isinstance(raw.get(array_key), dict):
        inner = raw[array_key]
        non_meta = {k: v for k, v in inner.items() if not k.startswith("_")}
        if non_meta and all(isinstance(v, list) for v in non_meta.values()):
            return [r for records in non_meta.values() for r in records if isinstance(r, dict)]

    # Shape A-dict-of-tasks — wrapper dict with dict-of-tasks under array_key
    # (Nguyen schema as documented: {"results": {"N1": {...}, "N2": {...}}})
    # Distinct from Shape C: array_key is explicitly declared for this experiment.
    if array_key and isinstance(raw, dict) and isinstance(raw.get(array_key), dict):
        inner = raw[array_key]
        records: list[dict] = []
        for k, v in inner.items():
            if k.startswith("_") or not isinstance(v, dict):
                continue
            rec = dict(v)
            if not rec.get(merge_key):
                for fb in fallback_keys:
                    if rec.get(fb):
                        rec[merge_key] = str(rec[fb])
                        break
                else:
                    rec[merge_key] = k
            records.append(rec)
        return records

    # Shape B — top-level list
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]

    if isinstance(raw, dict):
        values = [v for k, v in raw.items() if not k.startswith("_")]

        # Shape C — every non-meta value is a dict  → dict-of-tasks
        if values and all(isinstance(v, dict) for v in values):
            records: list[dict] = []
            for k, v in raw.items():
                if k.startswith("_"):
                    continue
                rec = dict(v)
                # Ensure merge_key is populated
                if not rec.get(merge_key):
                    for fb in fallback_keys:
                        if rec.get(fb):
                            rec[merge_key] = str(rec[fb])
                            break
                    else:
                        rec[merge_key] = k
                records.append(rec)
            return records

        # Shape D — single task record
        has_id = raw.get(merge_key) or any(raw.get(k) for k in fallback_keys)
        if has_id:
            rec = dict(raw)
            if not rec.get(merge_key):
                for fb in fallback_keys:
                    if rec.get(fb):
                        rec[merge_key] = str(rec[fb])
                        break
            return [rec]

    return []


def _task_id(record: dict, cfg: dict) -> str:
    """Return the stable unique string for a task record."""
    for key in [cfg["merge_key"]] + cfg.get("fallback_keys", []) + ["task_id", "name"]:
        val = record.get(key)
        if val and str(val) not in ("", "?"):
            return str(val)
    return "unknown"


def _is_solved(record: dict) -> bool:
    """
    Return True if a task record counts as solved.

    Handles the status-field variations across HypatiaX benchmark scripts:
      · "status"  : "ok" | "solved" | "success" | "passed"  (string)
      · "solved"  : True | 1  (boolean / int flag)
      · "success" : True | 1
      · R²-threshold fallback: if none of the above fields exist, treat the
        record as solved when its best r2/r2_score ≥ 0.9999 (paper threshold).
    First-write-wins field priority matches the merge logic.
    """
    _OK_STRINGS = {"ok", "solved", "success", "passed"}

    # String status field
    status = record.get("status")
    if status is not None:
        return str(status).lower() in _OK_STRINGS

    # Boolean / int flags written by some benchmark scripts
    for flag_key in ("solved", "success", "passed"):
        val = record.get(flag_key)
        if val is not None:
            if isinstance(val, bool):
                return val
            try:
                return bool(int(val))
            except (TypeError, ValueError):
                pass

    # Fallback: infer from R² if no explicit status field present.
    # Check both top-level r2 fields (legacy) and normalised test_r2 in results
    # sub-dicts (produced by _normalise_protocol_record).
    for r2_key in ("r2", "r2_score", "r2_noiseless", "best_r2"):
        raw = record.get(r2_key)
        if raw is not None:
            try:
                return float(raw) >= 0.9999
            except (TypeError, ValueError):
                pass

    # Normalised records: check any canonical method's test_r2.
    for method_res in record.get("results", {}).values():
        if not isinstance(method_res, dict):
            continue
        # success flag in sub-dict takes priority over R² threshold.
        sub_success = method_res.get("success")
        if sub_success is not None:
            if isinstance(sub_success, bool):
                return sub_success
            try:
                return bool(int(sub_success))
            except (TypeError, ValueError):
                pass
        raw = method_res.get("test_r2")
        if raw is not None:
            try:
                return float(raw) >= 0.9999
            except (TypeError, ValueError):
                pass

    return False


def _compute_stats(merged: dict[str, dict]) -> dict:
    records = list(merged.values())
    n       = len(records)
    n_ok    = sum(1 for r in records if _is_solved(r))
    r2_vals = []
    for r in records:
        # Check top-level r2 fields (legacy / Nguyen scripts).
        found = False
        for r2_key in ("r2", "r2_score", "r2_noiseless", "best_r2"):
            raw = r.get(r2_key)
            if raw is None:
                continue
            try:
                v = float(raw)
                if not (v != v):  # NaN check
                    r2_vals.append(v)
                    found = True
                    break
            except (TypeError, ValueError):
                pass
        # Check normalised results sub-dicts (protocol_core_*.json after normalisation).
        if not found:
            for method_res in r.get("results", {}).values():
                if not isinstance(method_res, dict):
                    continue
                raw = method_res.get("test_r2")
                if raw is None:
                    continue
                try:
                    v = float(raw)
                    if not (v != v):
                        r2_vals.append(v)
                        break
                except (TypeError, ValueError):
                    pass

    return {
        "n_tasks":         n,
        "n_solved":        n_ok,
        "solve_rate":      round(n_ok / n, 4) if n else 0.0,
        "r2_mean":         round(sum(r2_vals) / len(r2_vals), 4) if r2_vals else None,
        "r2_median":       round(sorted(r2_vals)[len(r2_vals) // 2], 4) if r2_vals else None,
        "r2_ge_0_99":      sum(1 for v in r2_vals if v >= 0.99),
        "r2_ge_0_9999":    sum(1 for v in r2_vals if v >= 0.9999),
        "n_with_r2":       len(r2_vals),
    }


def _write_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in records:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


# ─────────────────────────────────────────────────────────────────────────────
#  instability: CSV-only path
# ─────────────────────────────────────────────────────────────────────────────

def _merge_instability_csvs(shard_files: list[Path], out_dir: Path) -> dict:
    """
    Concatenate instability_analysis.csv shards, deduplicate by case_id.
    Writes _merged.csv, _merged.json, _stats.json, _checkpoint.json.

    _merged.json is written (as a JSON array of the same rows) so that
    locate_analysis_input.sh can find it via its standard _merged.json
    fast-path and route INPUT_MODE=merged to run_analysis.py --input-json.
    Without it the locate script falls into shard mode, finds no *.json
    files (the directory only contains CSVs/PDFs/PNGs), and exits with
    "No shard JSON files found".
    """
    all_rows: list[dict] = []
    seen_ids: set[str] = set()
    fieldnames: list[str] = []

    for csv_path in shard_files:
        if csv_path.suffix.lower() != ".csv":
            continue
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not fieldnames and reader.fieldnames:
                    fieldnames = list(reader.fieldnames)
                for row in reader:
                    uid = row.get("case_id") or row.get("equation") or str(row)
                    if uid not in seen_ids:
                        seen_ids.add(uid)
                        all_rows.append(row)
        except Exception as exc:
            print(f"  ⚠  CSV error {csv_path.name}: {exc}", file=sys.stderr)

    if fieldnames and all_rows:
        with open(out_dir / "_merged.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    # Write _merged.json — required by locate_analysis_input.sh fast-path.
    # validate_analysis_input.py will match this as format=flat_list (Tier 1).
    (out_dir / "_merged.json").write_text(
        json.dumps(all_rows, indent=2, ensure_ascii=False)
    )

    stats = {"n_tasks": len(all_rows), "n_shard_files": len(shard_files)}
    (out_dir / "_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  Core merge
# ─────────────────────────────────────────────────────────────────────────────

def merge_experiment(
    exp_id: str,
    input_root: Path,
    output_dir: Path,
    result_subdir: str,
    run_id: str = "",
    verbose: bool = True,
) -> int:
    """
    Merge all per-shard partial JSONs for `exp_id` found under `input_root`
    into the four canonical output files in `output_dir`.

    Returns 0 on success, 1 on error.
    """
    cfg = EXP_CONFIG.get(exp_id)
    if cfg is None:
        print(f"ERROR: unknown experiment '{exp_id}'", file=sys.stderr)
        print(f"  Known: {', '.join(sorted(EXP_CONFIG))}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'═'*68}")
        print(f"  merge_shards · [{exp_id}]")
        print(f"  input_root  : {input_root}")
        print(f"  output_dir  : {output_dir}")
        print(f"  subdir      : {result_subdir}")
        print(f"{'═'*68}")

    # ── 1. Find all partial shard files ────────────────────────────────────
    shard_files = _find_shard_files(input_root, cfg["shard_globs"])

    if verbose:
        print(f"\n  Shard files found: {len(shard_files)}")
        for f in shard_files:
            try:
                rel = f.relative_to(input_root)
            except ValueError:
                rel = f
            print(f"    · {rel}")

    if not shard_files:
        print(f"  ERROR: no partial result files found under {input_root}", file=sys.stderr)
        print(f"         globs tried:", file=sys.stderr)
        for g in cfg["shard_globs"]:
            print(f"           · {g}", file=sys.stderr)
        # Emit a recursive listing of what IS present so CI logs show the
        # actual filenames — the most common root cause is a glob mismatch
        # between the pattern and the real filename the worker produced.
        print(f"\n  Actual contents of {input_root} (recursive):", file=sys.stderr)
        all_files = sorted(input_root.rglob("*"))
        if not all_files:
            print("    (directory is empty or does not exist)", file=sys.stderr)
        else:
            for fp in all_files:
                if fp.is_file():
                    try:
                        rel = fp.relative_to(input_root)
                    except ValueError:
                        rel = fp
                    print(f"    {rel}", file=sys.stderr)
        print(
            "\n  ACTION REQUIRED: update EXP_CONFIG shard_globs for "
            f"'{exp_id}' in merge_shards.py to match the filenames listed above.",
            file=sys.stderr,
        )
        _write_stub_checkpoint(output_dir, exp_id, result_subdir, run_id,
                               error="no_shard_files")
        return 1

    # ── 2. instability is CSV-only ──────────────────────────────────────────
    if cfg.get("array_key") is None:
        stats = _merge_instability_csvs(shard_files, output_dir)
        _write_checkpoint(output_dir, exp_id, result_subdir, run_id,
                          shard_files, stats, n_merged=stats["n_tasks"])
        if verbose:
            print(f"\n  ✅  instability: {stats['n_tasks']} rows assembled")
        return 0

    # ── 3. Extract + merge records ──────────────────────────────────────────
    merged: dict[str, dict] = {}
    total_raw = 0

    # Inject exp_id into cfg so _extract_records can gate ablation logic.
    cfg_with_id = {**cfg, "_exp_id": exp_id}

    for fpath in shard_files:
        records = _extract_records(fpath, cfg_with_id)
        total_raw += len(records)
        for rec in records:
            tid = _task_id(rec, cfg)
            if tid == "unknown":
                tid = f"{fpath.stem}_{len(merged)}"
            # Set the merge key on the record if absent
            if not rec.get(cfg["merge_key"]):
                rec[cfg["merge_key"]] = tid
            existing = merged.get(tid)
            if existing is None:
                merged[tid] = rec
            else:
                # First-write-wins for each field; fill blanks from later shards
                for k, v in rec.items():
                    if k not in existing or existing[k] in (None, "", "?"):
                        existing[k] = v

    if verbose:
        print(f"\n  Raw records extracted : {total_raw}")
        print(f"  Unique task IDs       : {len(merged)}")

    if not merged:
        print("  ERROR: no task records could be extracted from any shard file.",
              file=sys.stderr)
        print(f"         {len(shard_files)} shard file(s) were found but all "
              "produced 0 records.", file=sys.stderr)
        print(f"         array_key={cfg.get('array_key')!r}  "
              f"merge_key={cfg.get('merge_key')!r}", file=sys.stderr)
        print("\n  Per-file summary:", file=sys.stderr)
        for fpath in shard_files:
            try:
                raw = json.loads(fpath.read_text(encoding="utf-8", errors="replace"))
                if isinstance(raw, dict):
                    top_keys = list(raw.keys())[:8]
                    ttype = "dict"
                elif isinstance(raw, list):
                    top_keys = [f"(list, len={len(raw)})"]
                    ttype = "list"
                else:
                    top_keys = [str(type(raw))]
                    ttype = "other"
                print(f"    · {fpath.name}: type={ttype}  top-keys={top_keys}",
                      file=sys.stderr)
            except Exception as exc:
                print(f"    · {fpath.name}: JSON error — {exc}", file=sys.stderr)
        print(
            "\n  ACTION REQUIRED: confirm array_key in EXP_CONFIG matches the "
            "top-level key that wraps the results list in the files above.",
            file=sys.stderr,
        )
        _write_stub_checkpoint(output_dir, exp_id, result_subdir, run_id,
                               error="no_records_extracted")
        return 1

    # ── 4. Enrich equation_id (and task_id via run_id_map) ─────────────────
    run_id_map = _collect_run_id_map(shard_files)
    _enrich_equation_id(merged, run_id_map=run_id_map)

    # ── 5. Compute stats ────────────────────────────────────────────────────
    stats = _compute_stats(merged)
    if verbose:
        print(f"\n  Stats:")
        print(f"    n_tasks     : {stats['n_tasks']}")
        print(f"    n_solved    : {stats['n_solved']}  "
              f"({stats['solve_rate']*100:.1f}%)")
        if stats['n_solved'] == 0 and stats['n_tasks'] > 0:
            print(
                f"\n  ⚠  WARNING: n_solved=0.  This usually means the benchmark script\n"
                f"     uses a status field name other than 'status'/'solved'/'success'.\n"
                f"     Inspect a sample record from _merged.json and compare against\n"
                f"     _is_solved() in merge_shards.py.  Also check that r2/r2_score\n"
                f"     fields are present if the R²-threshold fallback is intended.",
                file=sys.stderr,
            )
        if stats["r2_mean"] is not None:
            print(f"    R² mean     : {stats['r2_mean']:.4f}")
            print(f"    R² ≥ 0.9999 : {stats['r2_ge_0_9999']}  "
                  f"(strict §10.8 threshold)")

    # ── 6. Write _merged.json ───────────────────────────────────────────────
    merged_json_path = output_dir / "_merged.json"
    merged_json_path.write_text(
        json.dumps(list(merged.values()), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── 7. Write _merged.csv ────────────────────────────────────────────────
    merged_csv_path = output_dir / "_merged.csv"
    _write_csv(list(merged.values()), merged_csv_path)

    # ── 8. Write _stats.json ────────────────────────────────────────────────
    stats_path = output_dir / "_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ── 9. Write _checkpoint.json ───────────────────────────────────────────
    _write_checkpoint(output_dir, exp_id, result_subdir, run_id,
                      shard_files, stats, n_merged=len(merged))

    if verbose:
        print(f"\n  ✅  Written to {output_dir}:")
        for name in ("_merged.json", "_merged.csv", "_stats.json", "_checkpoint.json"):
            size = (output_dir / name).stat().st_size
            print(f"    {name:<22}  {size:>8,} bytes")

    return 0


def _enrich_equation_id(merged: dict[str, dict], run_id_map: dict[str, str] | None = None) -> None:
    """
    Patch every record so equation_id is set, and backfill task_id from
    run_id_map when present.

    Priority (mirrors ci_experiment.yml consolidate 'Enrich _merged.json' step):
      1. Already set and not "?"
      2. equation_id / eq_id / equation inside any per-method sub-record
      3. Parse 'description' field before first separator (: — - |)
      4. Top-level dict key (= task_id)

    task_id patching (mirrors the Enrich step's run_id_map cross-reference):
      · If run_id_map is provided and the top-level key is present in it,
        set task_id to the stable run identifier from the map so records
        remain consistently keyed across re-runs with different GitHub run_ids.
    """
    if run_id_map is None:
        run_id_map = {}
    for top_key, record in merged.items():
        if not isinstance(record, dict):
            continue
        if record.get("equation_id") and record["equation_id"] != "?":
            continue

        eq_id = None

        # Priority 2
        for v in record.values():
            if isinstance(v, dict):
                candidate = (v.get("equation_id") or v.get("eq_id")
                             or v.get("equation"))
                if candidate and candidate != "?":
                    eq_id = str(candidate)
                    break

        # Priority 3
        if not eq_id:
            desc = record.get("description", "")
            if desc:
                for sep in (":", "—", " - ", "|"):
                    if sep in desc:
                        eq_id = desc.split(sep)[0].strip()
                        break
                if not eq_id:
                    eq_id = desc.strip()

        # Priority 4
        if not eq_id:
            eq_id = top_key if top_key not in ("", "?") else None

        if eq_id:
            record["equation_id"] = eq_id
            if not record.get("task_id") or record["task_id"] == "?":
                # Prefer the stable run_id_map identifier over the raw top-level key.
                record["task_id"] = run_id_map.get(top_key, top_key)


def _write_checkpoint(
    out_dir: Path,
    exp_id: str,
    result_subdir: str,
    run_id: str,
    shard_files: list[Path],
    stats: dict,
    n_merged: int,
) -> None:
    """
    Write _checkpoint.json consumed by ci_analysis.yml via workflow_run event.
    Fields match what ci_experiment.yml's 'Set consolidate outputs' step emits.
    """
    checkpoint = {
        "exp_id":         exp_id,
        "result_subdir":  result_subdir,
        "run_id":         run_id,
        "merged_at":      datetime.now(timezone.utc).isoformat(),
        "n_merged":       n_merged,
        "stats":          stats,
        "shard_files":    [str(f) for f in shard_files],
        "n_shard_files":  len(shard_files),
    }
    (out_dir / "_checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2), encoding="utf-8"
    )


def _write_stub_checkpoint(
    out_dir: Path,
    exp_id: str,
    result_subdir: str,
    run_id: str,
    error: str,
) -> None:
    """
    Write a minimal _checkpoint.json even on failure (mirrors FIX-G6 stub pattern)
    so actions/cache/save never fails on a missing path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "_meta":         {"stub": True},
        "exp_id":        exp_id,
        "result_subdir": result_subdir,
        "run_id":        run_id,
        "merged_at":     datetime.now(timezone.utc).isoformat(),
        "n_merged":      0,
        "error":         error,
    }
    (out_dir / "_checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2), encoding="utf-8"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CLI  — matches the exact invocation in ci_experiment.yml consolidate job:
#
#    python .github/scripts/merge_shards.py \
#        --experiment    "${EXP}" \
#        --input-root    downloaded_artifacts \
#        --output-dir    "${OUT_BASE}/${RESULT_SUBDIR}" \
#        --result-subdir "${RESULT_SUBDIR}"
# ─────────────────────────────────────────────────────────────────────────────

def _pick_shard_file(exp_id: str, search_dir: Path) -> int:
    """
    CLI helper: print the best matching non-empty shard file for `exp_id`
    in `search_dir` and exit 0, or exit 1 if nothing suitable is found.

    Called by ci_analysis.yml "Locate input JSON" step for single-worker
    experiments to avoid duplicating EXP_CONFIG shard_globs in the YAML.

    Selection rules (same logic as _find_shard_files but for a single dir):
      1. Try each glob from EXP_CONFIG[exp_id]['shard_globs'] in order.
         First non-empty, non-stub, non-metadata match wins.
      2. Fallback: any *.json in search_dir that isn't in SKIP_NAMES and
         isn't empty / a stub.
    """
    cfg = EXP_CONFIG.get(exp_id)
    if cfg is None:
        print(f"ERROR: unknown experiment '{exp_id}'", file=sys.stderr)
        return 1

    _SKIP_NAMES = frozenset({
        "_report.md", "_merged.json", "_merged.csv",
        "_stats.json", "_checkpoint.json",
        "_analysis.json",       # pipeline output written by run_analysis.py — never a valid shard
        "benchmark_results.json",  # empty flat-list convenience export from v2 worker
    })

    def _is_non_empty(path: Path) -> bool:
        try:
            raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return False
        if isinstance(raw, list):
            return len(raw) > 0
        if isinstance(raw, dict):
            # Reject stub checkpoints written by _write_stub_checkpoint().
            if isinstance(raw.get("_meta"), dict) and raw["_meta"].get("stub"):
                return False
            return bool(raw)
        return False

    # Priority 1: globs in declared order — flat search (original behaviour).
    for pattern in cfg["shard_globs"]:
        for m in sorted(search_dir.glob(pattern)):
            if (m.is_file()
                    and m.name not in _SKIP_NAMES
                    and not m.name.startswith("_")
                    and _is_non_empty(m)):
                print(str(m))
                return 0

    # Priority 2: recursive rglob fallback — same globs, searched into subdirs.
    # Needed when the runner commits outputs into per-equation subdirectories
    # (e.g. suppB writes noise_sweep_*.json under noise-sweep/<eq-dir>/).
    # We prefer the shallowest hit so top-level files still win over nested ones.
    for pattern in cfg["shard_globs"]:
        candidates = [
            m for m in sorted(search_dir.rglob(pattern))
            if (m.is_file()
                and m.name not in _SKIP_NAMES
                and not m.name.startswith("_")
                and _is_non_empty(m))
        ]
        if candidates:
            # Pick the candidate closest to search_dir (fewest path parts).
            best = min(candidates, key=lambda p: len(p.parts))
            print(str(best))
            return 0

    # Priority 3: any *.json fallback — flat then recursive.
    for glob_fn, scope in [(search_dir.glob, "*.json"), (search_dir.rglob, "*.json")]:
        for m in sorted(glob_fn(scope)):
            if (m.is_file()
                    and m.name not in _SKIP_NAMES
                    and not m.name.startswith("_")
                    and _is_non_empty(m)):
                print(str(m))
                return 0

    print(
        f"ERROR: no suitable shard file found in {search_dir} for experiment '{exp_id}'.",
        file=sys.stderr,
    )
    print(f"  Globs tried: {cfg['shard_globs']}", file=sys.stderr)
    print("  Files present (recursive):", file=sys.stderr)
    for f in sorted(search_dir.rglob("*.json")):
        print(f"    {f.relative_to(search_dir)}", file=sys.stderr)
    return 1


def main() -> None:
    # ── Subcommand: --pick-shard-file ─────────────────────────────────────────
    # Lightweight helper used by ci_analysis.yml "Locate input JSON" step.
    # Must be handled BEFORE the main argparse block to avoid conflicts.
    if len(sys.argv) >= 2 and sys.argv[1] == "--pick-shard-file":
        if len(sys.argv) < 4:
            print(
                "Usage: merge_shards.py --pick-shard-file <exp_id> <search_dir>",
                file=sys.stderr,
            )
            sys.exit(1)
        rc = _pick_shard_file(
            exp_id     = sys.argv[2],
            search_dir = Path(sys.argv[3]).expanduser().resolve(),
        )
        sys.exit(rc)

    parser = argparse.ArgumentParser(
        description="Merge HypatiaX per-shard partial JSONs → 4 canonical output files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiment", "-e",
        required=True,
        metavar="EXP_ID",
        choices=sorted(EXP_CONFIG),
        help=f"Experiment ID. One of: {', '.join(sorted(EXP_CONFIG))}",
    )
    parser.add_argument(
        "--input-root", "-i",
        required=True,
        metavar="DIR",
        help=(
            "Root directory that contains the downloaded shard artifact folders. "
            "Searched recursively for files matching each experiment's shard globs."
        ),
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        metavar="DIR",
        help=(
            "Directory where _merged.json, _merged.csv, _stats.json and "
            "_checkpoint.json are written. Created if absent."
        ),
    )
    parser.add_argument(
        "--result-subdir",
        required=True,
        metavar="SUBDIR",
        help=(
            "Relative result subdir (e.g. 'comparison_results/feynman-tests/exp2'). "
            "Embedded in _checkpoint.json for ci_analysis.yml."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("GITHUB_RUN_ID", ""),
        metavar="ID",
        help="GitHub Actions run_id (written to _checkpoint.json). "
             "Defaults to $GITHUB_RUN_ID.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )

    args = parser.parse_args()

    rc = merge_experiment(
        exp_id        = args.experiment,
        input_root    = Path(args.input_root).expanduser().resolve(),
        output_dir    = Path(args.output_dir).expanduser().resolve(),
        result_subdir = args.result_subdir,
        run_id        = args.run_id,
        verbose       = not args.quiet,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
