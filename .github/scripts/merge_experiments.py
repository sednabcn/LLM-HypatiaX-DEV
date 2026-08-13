#!/usr/bin/env python3
"""
merge_experiment_json.py

Merge experiment JSON outputs into one normalized JSON file.

The merger is intentionally schema-tolerant:
  - accepts JSON objects or lists
  - understands common result containers
  - extracts equation/domain/seed/parameter metadata
  - preserves unknown fields
  - does not overwrite results from different experimental conditions
  - detects duplicate experimental configurations
  - optionally derives metadata from filenames

Example:

    python merge_experiment_json.py \
        --input-dir hypatiax/data/results \
        --output hypatiax/data/results/_merged.json

Or:

    python merge_experiment_json.py \
        --input-dir results \
        --output results/_merged.json \
        --experiment exp1

Filename metadata examples supported:

    equation_001_domain_biology_seed42.json
    eq001_biology_seed42.json
    feynman_biology_seed123.json

Metadata can also be present inside the JSON itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

try:
    # Optional: lets --experiment default --input-dir from the registry's
    # subdir/glob_pattern, the same way merge_shards.py resolves its
    # experiment's location. Not a hard dependency -- --input-dir can still
    # be passed explicitly for standalone / local runs.
    from experiment_registry import REGISTRY
except ImportError:  # pragma: no cover - registry not on path in isolation
    REGISTRY = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_RESULT_KEYS = (
    "results",
    "equation_results",
    "domain_results",
    "equations",
    "records",
    "data",
    "rows",
    "items",
    "entries",
    "output",
    "outputs",
    "benchmark_results",
    "eval_results",
    "test_results",
    "experiments",
    "cases",
    "metrics",
    "details",
    # Shape P ("tests": [{"equation_id":..., "domain":..., "seed":...,
    # "results": {"pure_llm": {...}, "neural_network": {...}}}, ...]) --
    # the wrapper written by extrap / exp2_feynman / exp2_feynman_pca's
    # protocol_core_*.json files. Same shape merge_shards.py detects via
    # its own _is_protocol_file(). Without this key present, "tests" falls
    # through to the generic per-key fallback loop below, which still finds
    # rows eventually but (before the IDENTITY_KEYS fix a few lines down)
    # used to lose equation_id/domain/seed context on the way there.
    "tests",
)


# ---------------------------------------------------------------------------
# Identity fields that must survive recursion.
#
# extract_result_rows() unwraps nested result containers -- e.g. Shape P's
# {"equation_id": "eq1", "domain": "biology", "seed": 42,
#  "results": {"pure_llm": {"r2": 0.95}}} -- and by the time it reaches the
# leaf metric dict ({"r2": 0.95}, several levels down inside "results" ->
# "pure_llm"), the equation_id/domain/seed that were sitting on the
# *ancestor* dict are gone; the leaf itself never had them. merge_metadata()
# then can't find them on the row and falls back to filename regexes, which
# don't match every naming convention -- so rows silently end up with
# equation_id=domain=seed=None, and worse, unrelated rows can collide on the
# same run_id (false "duplicate") once their identity is gone. See
# _propagate_identity() below, called at every recursion step, so identity
# backfills level-by-level as the recursion unwinds instead of being lost.
# ---------------------------------------------------------------------------
IDENTITY_KEYS = (
    "equation_id", "eq_id", "equation", "equation_name", "problem_id", "case_id", "id",
    "domain", "domain_id", "physics_domain", "dataset_domain",
    "seed", "random_seed", "rng_seed", "pysr_seed", "nn_seed", "random_state",
)


def _propagate_identity(parent: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Backfill IDENTITY_KEYS from `parent` onto each row missing them.

    In-place. Never overwrites a value a row already carries -- a row's own
    identity fields always win over an ancestor's.
    """
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in IDENTITY_KEYS:
            if key in parent and key not in row and parent[key] is not None:
                row[key] = parent[key]


# ---------------------------------------------------------------------------
# Single source of truth for which experiments require this merger.
#
# This is the parameter/domain-loop counterpart to merge_shards.py's
# MERGE_REQUIRED_EXPERIMENTS. merge_shards.py consolidates CI *shard*
# fan-out (N parallel workers, each running a subset of seeds/tasks,
# recombined post-hoc with a highest-score-wins policy). This module instead
# consolidates the per-domain / per-parameter files that a SINGLE worker
# writes when run_all.sh loops it locally in bash, e.g.:
#
#   for DOMAIN_ID in ${FEYNMAN_DOMAINS}; do ... done
#
# That distinction matters for the merge policy: because there is only one
# worker and each iteration targets a disjoint domain/parameter value (not a
# retry of the same task), silently keeping "highest score wins" per
# equation_id (merge_shards.py's policy) would be wrong here -- two
# iterations legitimately produce different, non-duplicate rows for the same
# equation_id under different domains/parameters. That's why merge_results()
# below keys duplicate-detection on the full (experiment, equation_id,
# domain, seed, method, parameters) tuple via run_id, and keeps every row
# rather than collapsing on equation_id alone.
#
# DO NOT wire this set into ci_analysis.yml. A step doing exactly that was
# tried and reverted: this module's merged output (one row per
# equation/domain/method, metrics nested under a "result" key, top-level
# "results" as a *list*) matches none of the record shapes
# run_analysis.py's _load_records_from_json() understands (Shape A/B/C/P/S/N
# all expect "results" as a *dict* keyed by method, or a paired
# hypatia/pysr_only record for ablation mode). Concretely, running this
# against exp2_feynman would flatten its paired ablation record into
# separate per-method rows, which analyse_ablation()'s schema guard rejects
# as WRONG_SCHEMA_FOR_ABLATION (hard CI fatal). extrap and exp2_feynman_pca
# already work correctly today via ci_analysis.yml's existing SHARDS mode
# (shard_manifest.txt listing the per-domain protocol_core_*.json files
# directly, read by run_analysis.py's own Shape-P detection) -- they need no
# merge step at all.
#
# This module and PARAM_LOOP_MERGE_EXPERIMENTS remain available for
# standalone/local use (e.g. producing a human-inspectable consolidated
# view of a domain-loop experiment's results outside of CI), but should
# not be imported by ci_analysis.yml. Do NOT add an experiment here that is
# already in merge_shards.MERGE_REQUIRED_EXPERIMENTS -- the two mergers
# solve different problems (shard fan-out vs. local domain loop) and an
# experiment should never need both.
#
# extrap              -- for DOMAIN_ID in ${FEYNMAN_DOMAINS} (run_all.sh STEP 3).
#                         Registry: extrap, shards=True, 11-domain registry,
#                         single-shard CI table entry (experiment_registry.py).
# exp2_feynman        -- for DOMAIN_ID in ${FEYNMAN_DOMAINS} (STEP 5).
#                         Registry note: "Single CI worker, but internally
#                         loops over 11 Feynman domains and writes one
#                         protocol_core_*.json per domain."
# exp2_feynman_pca    -- for DOMAIN_ID in ${FEYNMAN_DOMAINS}, invoked from the
#                         exp2_feynman_pca_4060 step (STEP 5b). Registry:
#                         "Mirrors exp2_feynman: same 11-domain registry,
#                         single shard table entry."
#
# NOTE on "exp2_feynman_extrap": it is intentionally NOT in this set, even
# though its STEP 5b sub-block also loops `for DOMAIN_ID in ${ACTIVE_DOMAINS}`.
# It already has a bespoke, schema-specific consolidator --
# merge_extrap_into_benchmark.py -- called inline inside that step, producing
# ablation_paired.json (not this module's output shape). Routing it through
# this generic merger as well would give run_analysis.py two conflicting
# consolidated files for the same experiment.
#
# NOTE on "exp2": its STEP 6 sub-block also loops `for DOMAIN_ID in
# ${EXP2_DOMAINS}`, but experiment_registry.py marks it shards=False --
# "exp2_run.log has no dedicated per-run JSON of its own" -- i.e. the
# five-system comparison script already consolidates internally across its
# domain loop and writes a single all_systems_merged.json directly. Adding it
# here would merge an already-merged file back into itself.
#
# NOTE on "hybrid_all_domains": it covers 10 domains but does so inside one
# Python invocation (run_all.sh's comment calls it "one-shot"), not a bash
# `for DOMAIN_ID` loop -- there is nothing for this module to consolidate.
# ---------------------------------------------------------------------------
PARAM_LOOP_MERGE_EXPERIMENTS: frozenset[str] = frozenset({
    "extrap",
    "exp2_feynman",
    "exp2_feynman_pca",
})


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_hash(value: Any) -> str:
    """Stable short hash used for duplicate detection."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def first_value(obj: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return None


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_seed(obj: dict[str, Any], filename: str) -> Any:
    value = first_value(
        obj,
        (
            "seed",
            "random_seed",
            "rng_seed",
            "pysr_seed",
            "nn_seed",
            "random_state",
        ),
    )

    if value is not None:
        return value

    match = re.search(r"(?:^|[_-])seed[_-]?(\d+)(?:[_\-.]|$)", filename)
    if match:
        return int(match.group(1))

    return None


def extract_equation_id(obj: dict[str, Any], filename: str) -> Any:
    value = first_value(
        obj,
        (
            "equation_id",
            "eq_id",
            "equation",
            "equation_name",
            "problem_id",
            "case_id",
            "id",
        ),
    )

    if value is not None:
        return value

    patterns = (
        r"(?:equation|eq)[_-]?([A-Za-z0-9.-]+)",
    )

    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def extract_domain(obj: dict[str, Any], filename: str) -> Any:
    value = first_value(
        obj,
        (
            "domain",
            "domain_id",
            "physics_domain",
            "dataset_domain",
        ),
    )

    if value is not None:
        return value

    # Common naming convention:
    #   feynman_biology_seed42.json
    #   equation_001_domain_biology_seed42.json
    match = re.search(
        r"(?:domain[_-])([A-Za-z0-9.-]+)",
        filename,
        re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return None


def extract_parameters(obj: dict[str, Any]) -> dict[str, Any]:
    """
    Collect experiment parameters without assuming a single schema.
    """

    parameters: dict[str, Any] = {}

    explicit = obj.get("parameters")

    if isinstance(explicit, dict):
        parameters.update(explicit)

    config = obj.get("config")

    if isinstance(config, dict):
        parameters.update(config)

    # Common experiment-level parameters.
    parameter_keys = (
        "noise",
        "noise_level",
        "sigma",
        "samples",
        "n_samples",
        "populations",
        "population",
        "generations",
        "tournament_size",
        "parsimony",
        "timeout",
        "method_timeout",
        "pysr_timeout",
        "extrap_multiplier",
        "extrap_train_frac",
        "train_frac",
        "split",
        "split_protocol",
        "test_size",
        "train_size",
    )

    for key in parameter_keys:
        if key in obj and key not in parameters:
            parameters[key] = obj[key]

    return parameters


def extract_method(obj: dict[str, Any]) -> Any:
    return first_value(
        obj,
        (
            "method",
            "model",
            "system",
            "algorithm",
            "approach",
            "arm",
        ),
    )


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def looks_like_metric_record(obj: dict[str, Any]) -> bool:
    """
    Decide whether a dictionary is already a result row.

    This deliberately accepts many common metric names because the
    experiments in run_all.sh use several different schemas.
    """

    metric_keys = {
        "r2",
        "r2_test",
        "r2_train",
        "rmse",
        "rmse_test",
        "rmse_train",
        "mae",
        "mse",
        "error",
        "success",
        "status",
        "formula",
        "prediction",
        "predictions",
        "runtime",
        "time",
    }

    return bool(metric_keys.intersection(obj.keys()))


def extract_result_rows(data: Any) -> list[dict[str, Any]]:
    """
    Recursively normalize common JSON structures into a list of dictionaries.

    Example:

        {"results": [{"equation": "001", "r2": 0.99}]}

    becomes:

        [{"equation": "001", "r2": 0.99}]
    """

    rows: list[dict[str, Any]] = []

    if isinstance(data, list):
        for item in data:
            rows.extend(extract_result_rows(item))
        return rows

    if not isinstance(data, dict):
        return rows

    # Direct result row.
    if looks_like_metric_record(data):
        rows.append(dict(data))
        return rows

    # Known result containers.
    for key in DEFAULT_RESULT_KEYS:
        value = data.get(key)

        if value is None:
            continue

        if isinstance(value, list):
            for item in value:
                # Each list item (e.g. one "tests" entry) may itself carry
                # equation_id/domain/seed that its own leaf rows need --
                # propagate from `item` first, then from the outer `data`
                # for anything `item` didn't have (e.g. a file-level seed
                # shared by every test).
                item_rows = extract_result_rows(item)
                if isinstance(item, dict):
                    _propagate_identity(item, item_rows)
                _propagate_identity(data, item_rows)
                rows.extend(item_rows)

        elif isinstance(value, dict):
            # Example:
            # per_noise -> noise -> equation -> result
            child_rows = extract_result_rows(value)
            _propagate_identity(data, child_rows)
            rows.extend(child_rows)

    # Handle nested method/equation dictionaries that do not use
    # one of the standard container names.
    if not rows:
        for key, value in data.items():

            if isinstance(value, dict):

                # Equation-keyed or method-keyed structure.
                if looks_like_metric_record(value):
                    row = dict(value)
                    row.setdefault("_key", key)
                    _propagate_identity(data, [row])
                    rows.append(row)
                else:
                    children = extract_result_rows(value)

                    for child in children:
                        child.setdefault("_key", key)

                    _propagate_identity(data, children)
                    rows.extend(children)

            elif isinstance(value, list):
                child_rows = extract_result_rows(value)
                _propagate_identity(data, child_rows)
                rows.extend(child_rows)

    return rows


# ---------------------------------------------------------------------------
# Metadata propagation
# ---------------------------------------------------------------------------

def merge_metadata(
    file_data: dict[str, Any],
    row: dict[str, Any],
    filename: str,
) -> dict[str, Any]:

    equation_id = (
        extract_equation_id(row, filename)
        or extract_equation_id(file_data, filename)
    )

    domain = (
        extract_domain(row, filename)
        or extract_domain(file_data, filename)
    )

    seed = (
        extract_seed(row, filename)
        if isinstance(row, dict)
        else None
    )

    if seed is None:
        seed = extract_seed(file_data, filename)

    parameters = extract_parameters(file_data)

    row_parameters = row.get("parameters")
    if isinstance(row_parameters, dict):
        parameters.update(row_parameters)

    # "_key" is set by extract_result_rows()'s method-keyed-dict fallback
    # (e.g. {"pure_llm": {"r2": ...}} -> row={"r2": ..., "_key": "pure_llm"}).
    # It was already being popped from the final row below but never
    # actually consulted for `method` -- so pure_llm/neural_network rows
    # from exp2_feynman/extrap/exp2_feynman_pca's Shape P files always
    # resolved method=None even though the method name was sitting right
    # there. row.get("method") (an explicit field) still wins if present.
    method = row.get("method") or row.get("_key") or extract_method(file_data)

    result = dict(row)

    # Remove internal helper field from final output.
    result.pop("_key", None)

    return {
        "equation_id": equation_id,
        "domain": domain,
        "seed": seed,
        "method": method,
        "parameters": parameters,
        "result": result,
        "source_file": filename,
    }


# ---------------------------------------------------------------------------
# Main merge operation
# ---------------------------------------------------------------------------

def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None so JSON serialization
    doesn't fail when allow_nan=False."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def merge_results(
    input_dir: Path,
    output: Path,
    experiment: str | None = None,
    recursive: bool = True,
) -> dict[str, Any]:

    pattern = "**/*.json" if recursive else "*.json"

    files = sorted(input_dir.glob(pattern))

    # Never merge the output into itself.
    files = [
        p for p in files
        if p.resolve() != output.resolve()
        and not p.name.startswith("_merged")
    ]

    merged: list[dict[str, Any]] = []

    stats = {
        "files_found": len(files),
        "files_loaded": 0,
        "files_failed": 0,
        "rows": 0,
        "duplicates": 0,
    }

    errors: list[dict[str, str]] = []
    seen: dict[str, str] = {}

    for path in files:

        try:
            data = load_json(path)
            stats["files_loaded"] += 1

        except Exception as exc:
            stats["files_failed"] += 1
            errors.append(
                {
                    "file": str(path),
                    "error": str(exc),
                }
            )
            continue

        # Skip obvious summary/metadata files.
        if path.name in {
            "_merged.json",
            "manifest.json",
            "checkpoint.json",
            "split_protocol_disclosure.json",
        }:
            continue

        rows = extract_result_rows(data)

        # Some files contain one result object directly but do not contain
        # standard metric fields. Preserve them instead of losing the file.
        if not rows:
            if isinstance(data, dict):
                rows = [data]
            else:
                rows = [{"value": data}]

        for row in rows:

            if not isinstance(row, dict):
                row = {"value": row}

            record = merge_metadata(
                file_data=data if isinstance(data, dict) else {},
                row=row,
                filename=path.name,
            )

            if experiment is not None:
                record["experiment"] = experiment

            # Create a configuration identity.
            identity = {
                "experiment": record.get("experiment"),
                "equation_id": record.get("equation_id"),
                "domain": record.get("domain"),
                "seed": record.get("seed"),
                "method": record.get("method"),
                "parameters": record.get("parameters"),
            }

            record["run_id"] = stable_hash(identity)

            # Detect exact duplicate configurations.
            run_id = record["run_id"]

            if run_id in seen:
                stats["duplicates"] += 1

                # Keep both results rather than silently overwriting.
                record["duplicate_of"] = seen[run_id]

            else:
                seen[run_id] = record["source_file"]

            merged.append(record)
            stats["rows"] += 1

    result = {
        "schema_version": "1.0",
        "experiment": experiment,
        "statistics": stats,
        "errors": errors,
        "results": merged,
    }

    output.parent.mkdir(parents=True, exist_ok=True)

    # Sanitize the result to remove NaN/Inf values before serializing,
    # since allow_nan=False would otherwise raise ValueError.
    sanitized_result = _sanitize_for_json(result)

    with output.open("w", encoding="utf-8") as f:
        json.dump(
            sanitized_result,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        f.write("\n")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:

    parser = argparse.ArgumentParser(
        description="Merge experiment JSON outputs into one normalized JSON file."
    )

    parser.add_argument(
        "--input-dir",
        required=False,
        type=Path,
        default=None,
        help=(
            "Directory containing experiment JSON files. Optional when "
            "--experiment is given and resolvable via experiment_registry.REGISTRY "
            "(uses --results-root/<entry.subdir>); otherwise required."
        ),
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=(
            "Root results directory, used with --experiment to resolve "
            "--input-dir from experiment_registry.REGISTRY when --input-dir "
            "is not given explicitly."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output merged JSON file.",
    )

    parser.add_argument(
        "--experiment",
        default=None,
        help="Optional experiment name, e.g. exp1, exp3, suppB.",
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search subdirectories.",
    )

    args = parser.parse_args()

    # FATAL-guard, mirroring merge_shards.py's philosophy of failing loudly
    # at merge time rather than producing a quietly-wrong _merged.json:
    # this module's duplicate-preserving policy is only correct for the
    # local domain/parameter-loop experiments in PARAM_LOOP_MERGE_EXPERIMENTS.
    # Calling it for a shard-fan-out experiment (exp1b, exp1_ablation, exp3b,
    # suppB, suppB_sc -- merge_shards.MERGE_REQUIRED_EXPERIMENTS) would keep
    # every shard's duplicate/retry rows as if they were distinct domain
    # results, which is wrong for that merge policy.
    if args.experiment is not None and args.experiment not in PARAM_LOOP_MERGE_EXPERIMENTS:
        print(
            f"FATAL: --experiment {args.experiment!r} is not in "
            f"PARAM_LOOP_MERGE_EXPERIMENTS "
            f"({sorted(PARAM_LOOP_MERGE_EXPERIMENTS)}). "
            "If this experiment uses CI shard fan-out (seeds/tasks split "
            "across parallel workers), use merge_shards.py instead. If it "
            "already writes one consolidated file per run, it needs neither "
            "merger -- see the NOTE comments above PARAM_LOOP_MERGE_EXPERIMENTS.",
            file=sys.stderr,
        )
        return 2

    input_dir = args.input_dir
    if input_dir is None:
        if REGISTRY is not None and args.experiment in REGISTRY and args.results_root is not None:
            entry_subdir = REGISTRY[args.experiment].subdir
            input_dir = args.results_root / entry_subdir if entry_subdir else args.results_root
            print(f"[merge_experiments] --input-dir not given; resolved from registry: {input_dir}")
        else:
            print(
                "FATAL: --input-dir was not given and could not be resolved "
                "from the registry (need --experiment present in "
                "experiment_registry.REGISTRY AND --results-root set).",
                file=sys.stderr,
            )
            return 2

    result = merge_results(
        input_dir=input_dir,
        output=args.output,
        experiment=args.experiment,
        recursive=not args.no_recursive,
    )

    stats = result["statistics"]

    print()
    print("=== JSON MERGE SUMMARY ===")
    print(f"Input directory : {input_dir}")
    print(f"Output          : {args.output}")
    print(f"Files found     : {stats['files_found']}")
    print(f"Files loaded    : {stats['files_loaded']}")
    print(f"Files failed    : {stats['files_failed']}")
    print(f"Result rows     : {stats['rows']}")
    print(f"Duplicates      : {stats['duplicates']}")

    if result["errors"]:
        print()
        print("Errors:")
        for error in result["errors"]:
            print(f"  {error['file']}: {error['error']}")

    print()
    print("Merge complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

