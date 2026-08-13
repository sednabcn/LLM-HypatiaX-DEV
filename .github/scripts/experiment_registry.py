#!/usr/bin/env python3
"""
experiment_registry.py
-----------------------
Single source of truth for "where does experiment X's result JSON live, and
which glob pattern finds it" — previously a comment-only table duplicated
(and drifting) across generate_tables.py, generate_figures.py, and
run_analysis.py.

Import this instead of re-deriving subdir/glob logic per-script. Consumers
that need path resolution beyond a lookup (e.g. picking the *newest*
non-checkpoint file) should use find_best() / find_sweep() below, which
apply the same exclude-then-sort discipline in one place.

Why this exists (see generate_tables.py's SC-CHECKPOINT-POLLUTION comment):
mtime-based "latest file wins" silently picked an in-progress checkpoint
shard over the real consolidated result in every CI run, because the
exclusion list was applied inconsistently. Centralising both the table AND
the selection logic closes that class of bug rather than just this one
instance of it.

Usage
-----
    from experiment_registry import REGISTRY, find_best

    entry = REGISTRY["exp2_feynman"]
    data, path = find_best(results_dir, entry, patched_dir=patched_dir)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Substrings that mark a file as NOT a canonical/final result, even though it
# matches the experiment's glob pattern. Applied BEFORE mtime sort, always.
#   - "checkpoint" : mid-run shard written by a benchmark runner, later
#                     superseded by the consolidated file (same basename
#                     family, so a naive glob can't tell them apart).
#   - "_sig"        : per-sigma shard from a noise sweep, not the merged sweep.
#   - "MISSING"     : placeholder written by a runner on a failed task ID.
# Extend this list, don't add a new one — a second exclude list for a second
# script is exactly how the checkpoint-shadowing bug happened the first time.
# ---------------------------------------------------------------------------
EXCLUDE_SUBSTRINGS: tuple[str, ...] = ("checkpoint", "_sig", "MISSING")


@dataclass(frozen=True)
class ExperimentEntry:
    """Where one experiment's result JSON(s) live and how to find them.

    subdir       : path relative to RESULTS/PATCHED root (empty string = root)
    glob_pattern : filename glob within subdir, e.g. "protocol_core_*.json"
    shards       : True if this experiment writes one file per task ID
                   (needs consolidation before analysis); False if a single
                   worker writes one canonical file directly.
    extra_subdirs: additional RESULTS-relative subdirs to check, in order,
                   after (PATCHED/subdir, RESULTS/subdir) — mirrors
                   load_best()'s extra_subdirs parameter in generate_tables.py.
    notes        : free text, carried from the original table comment.
    """
    subdir: str
    glob_pattern: str
    shards: bool = False
    extra_subdirs: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""


# ---------------------------------------------------------------------------
# The registry itself — ported from the comment table at the top of
# generate_tables.py ("JSON location map (run_all.sh -> tables-generator)").
# If you add a new experiment step to run_all.sh, add its row here FIRST;
# generate_tables.py / generate_figures.py / merge_shards.py should all read
# this dict rather than hardcoding a new subdir/glob pair.
# ---------------------------------------------------------------------------
REGISTRY: dict[str, ExperimentEntry] = {
    "exp1": ExperimentEntry(
        subdir="",
        glob_pattern="benchmark_results*.json",
        notes="Also checked: hypatiax_defi_benchmark_v3*results*.json (defi fallback).",
    ),
    "exp1b": ExperimentEntry(
        subdir="",
        glob_pattern="portfolio_variance*.json",
        notes="Includes portfolio_variance_seed_sweep.json.",
    ),
    "exp1_ablation": ExperimentEntry(
        subdir="ablation/exp1_ablation",
        glob_pattern="*.json",
        shards=True,
        notes="NSHARDS=4; shards merged to _merged.json by merge_shards.py "
              "before this subdir is read.",
    ),
    "exp1_five": ExperimentEntry(
        subdir="five_systems/exp1_five",
        glob_pattern="exp1_five_results*.json",
        notes="No fallback — only the launched experiment's results are used.",
    ),
    "exp2_five": ExperimentEntry(
        subdir="five_systems/exp2_five",
        glob_pattern="*.json",
    ),
    "exp2_feynman": ExperimentEntry(
        subdir="comparison_results/feynman-tests/exp2",
        glob_pattern="exp2_results*.json",
        shards=True,
        notes="Single CI worker, but internally loops over 11 Feynman domains "
              "and writes one protocol_core_*.json per domain.",
    ),
    "exp2_feynman_pca": ExperimentEntry(
        subdir="comparison_results/feynman-tests/exp2",
        glob_pattern="protocol_core_noiseless_pca_*.json",
        shards=True,
        notes="Mirrors exp2_feynman: same 11-domain registry, single shard table entry.",
    ),
    "exp2_feynman_extrap": ExperimentEntry(
        subdir="",
        glob_pattern="protocol_core_extrap_*.json",
        shards=True,
        notes="Consolidated today by merge_extrap_into_benchmark.py into "
              "ablation_paired.json (NOT _merged.json — different schema, "
              "different consumer: run_analysis.py's analyse_ablation()).",
    ),
    "exp2": ExperimentEntry(
        subdir="comparison_results",
        glob_pattern="all_systems_merged.json",
        notes="exp2_run.log has no dedicated per-run JSON of its own.",
    ),
    "extrap": ExperimentEntry(
        subdir="comparison_results/extrapolation",
        glob_pattern="all_domains_extrap_v4_*.json",
        shards=True,
        notes="11-domain registry (FEYNMAN_DOMAINS), single-shard CI table entry.",
    ),
    "hybrid_all_domains": ExperimentEntry(
        subdir="hybrid_llm_nn/all_domains",
        glob_pattern="hybrid_llm_nn_all_domains_*.json",
        shards=True,
        notes="10-domain registry, single-shard CI table entry.",
    ),
    "exp3": ExperimentEntry(
        subdir="nguyen12",
        glob_pattern="exp3_nguyen12_seed*.json",
        shards=True,
        notes="12 equations x 5 seeds = 60 combos; nguyen12 script writes to "
              "cwd, may need explicit --results-dir.",
    ),
    "instability": ExperimentEntry(
        subdir="figures",
        glob_pattern="instability*.json",
        notes="Also reads instability_analysis.csv from the same subdir. "
              "Produces no _merged.json — not a shard-consolidation case.",
    ),
    "suppB": ExperimentEntry(
        subdir="comparison_results/feynman-tests/noise-sweep",
        glob_pattern="noise_sweep_*.json",
    ),
    "suppB_sc": ExperimentEntry(
        subdir="comparison_results/feynman-tests/sample-complexity",
        glob_pattern="sample_complexity_*.json",
        notes="See EXCLUDE_SUBSTRINGS docstring: checkpoint shards from this "
              "dir previously shadowed the canonical consolidated file.",
    ),
    "noiseless": ExperimentEntry(
        subdir="comparison_results/noise-noiseless/noiseless",
        glob_pattern="protocol_core_noiseless_*.json",
    ),
}


# ---------------------------------------------------------------------------
# Shape-detection: complement to the location registry above. Location tells
# you WHERE a file is; this tells you WHAT'S IN IT, mirroring run_analysis.py's
# _load_records_from_json Shape P/S/N/A sniffing. Keyed off structural
# fingerprints in the loaded JSON, not the experiment name or filename — the
# actual answer to "different structure in each experiment": detect per-file,
# don't hardcode per-experiment.
# ---------------------------------------------------------------------------
def detect_shape(data: dict | list) -> str:
    """Return one of 'flat_list', 'protocol_wrapper', 'noise_sweep',
    'sample_complexity_sweep', or 'unknown'."""
    if isinstance(data, list):
        return "flat_list"
    if isinstance(data, dict):
        if "per_n" in data:
            return "sample_complexity_sweep"
        if "results" in data and isinstance(data.get("results"), (list, dict)):
            return "protocol_wrapper"
        if "sigma" in data or "noise_level" in data:
            return "noise_sweep"
    return "unknown"


# ---------------------------------------------------------------------------
# Selection helpers — exclude-then-sort, applied once, used everywhere.
# ---------------------------------------------------------------------------
def _filtered_glob(d: Path, glob_pattern: str) -> list[Path]:
    if not d.exists():
        return []
    return [p for p in d.glob(glob_pattern)
            if not any(s in p.name for s in EXCLUDE_SUBSTRINGS)]


def find_candidates(
    results_dir: Path,
    entry: ExperimentEntry,
    patched_dir: Path | None = None,
) -> list[Path]:
    """All non-excluded matches for `entry`, across patched/results/extra
    subdirs, newest first. Does not read file contents."""
    search_dirs: list[Path] = []
    if patched_dir is not None:
        search_dirs.append(patched_dir / entry.subdir if entry.subdir else patched_dir)
    search_dirs.append(results_dir / entry.subdir if entry.subdir else results_dir)
    for extra in entry.extra_subdirs:
        search_dirs.append(results_dir / extra if extra else results_dir)

    candidates: list[Path] = []
    for d in search_dirs:
        candidates.extend(_filtered_glob(d, entry.glob_pattern))
    return sorted(candidates, key=os.path.getmtime, reverse=True)


def find_best(
    results_dir: Path,
    entry: ExperimentEntry,
    patched_dir: Path | None = None,
) -> tuple[dict | list | None, Path | None]:
    """Load the newest non-excluded matching JSON. Returns (data, path),
    (None, None) if nothing usable was found. Mirrors generate_tables.py's
    load_best(), generalised to any registry entry."""
    for path in find_candidates(results_dir, entry, patched_dir):
        try:
            return json.loads(path.read_text()), path
        except (json.JSONDecodeError, OSError):
            continue
    return None, None


def find_shards(
    results_dir: Path,
    entry: ExperimentEntry,
    patched_dir: Path | None = None,
) -> list[Path]:
    """All non-excluded matching files, unsorted-by-recency (shard sets are
    consumed as a whole, not "pick the newest one"). Use for entry.shards=True
    experiments feeding a consolidation step."""
    search_dirs: list[Path] = []
    if patched_dir is not None:
        search_dirs.append(patched_dir / entry.subdir if entry.subdir else patched_dir)
    search_dirs.append(results_dir / entry.subdir if entry.subdir else results_dir)
    for extra in entry.extra_subdirs:
        search_dirs.append(results_dir / extra if extra else results_dir)

    seen: dict[str, Path] = {}
    for d in search_dirs:
        for p in _filtered_glob(d, entry.glob_pattern):
            seen.setdefault(p.name, p)  # first dir in priority order wins
    return sorted(seen.values())


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect the experiment registry / test file discovery.")
    ap.add_argument("experiment", choices=sorted(REGISTRY), help="Experiment key to look up.")
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--patched-dir", type=Path, default=None)
    args = ap.parse_args()

    entry = REGISTRY[args.experiment]
    print(f"{args.experiment}: subdir={entry.subdir!r} glob={entry.glob_pattern!r} "
          f"shards={entry.shards}")
    if entry.shards:
        shards = find_shards(args.results_dir, entry, args.patched_dir)
        print(f"  {len(shards)} shard file(s):")
        for p in shards:
            print(f"    {p}")
    else:
        data, path = find_best(args.results_dir, entry, args.patched_dir)
        print(f"  best match: {path}")
        if data is not None:
            print(f"  shape: {detect_shape(data)}")
