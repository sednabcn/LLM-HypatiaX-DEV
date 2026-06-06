#!/usr/bin/env python3
"""
validate_analysis_input.py
--------------------------
Reads INPUT_MODE, INPUT_JSON, SHARD_MANIFEST from the environment and counts
the total number of result records across all input files.  Exits 1 on an
empty dataset (FATAL: EMPTY DATASET).

Called by ci_analysis.yml "Validate input data" step.

Schema support
--------------
Two tiers of schema handling are provided:

  TIER 1 — Canonical (preferred)
    All runners should emit one of these.  Detection is unambiguous; no
    field-name heuristics required.

    flat_list          Top-level JSON array.  Each element is one record.
                       [ {record}, {record}, ... ]

    results_list       Wrapper dict with a "results" key whose value is a list.
                       { "results": [{record}, ...], ... }

    tests_list         Wrapper dict with a "tests" key whose value is a list.
                       { "tests": [{record}, ...], ... }

    results_dict_of_lists
                       Wrapper dict with a "results" key whose value is a
                       dict mapping system/method names to lists of records.
                       { "results": { "sysA": [{record}, ...], ... }, ... }
                       Emitted by the exp3 runner (seed42 / multi-seed shards).

  TIER 2 — Legacy / third-party (tolerated, not recommended)
    Formats produced by older runners or external tools.  Detection relies on
    structural heuristics and may misfire on unusual inputs.  Migrate runners
    to a Tier 1 format when possible.

    results_dict_of_dicts_methods
                       "results" maps equation IDs to per-method dicts.
                       { "results": { "N1": { "hypatiax": {r2,...}, ... } } }

    results_dict_of_dicts_flat
                       "results" maps equation IDs to flat record dicts.
                       { "results": { "N1": { "r2": 0.9, ... }, ... } }

    toplevel_method_nested
                       Top-level dict maps equation IDs to per-method dicts.
                       No "results" wrapper.  Old nested method shape.
                       { "N1": { "hypatiax": {r2,...}, "pysr": {r2,...} } }

    toplevel_flat_records
                       Top-level dict maps equation IDs to flat record dicts.
                       { "N1": { "r2": 0.9, "method": "hypatiax", ... } }

    toplevel_generic_dicts
                       Top-level dict maps arbitrary keys to dicts with no
                       recognisable result fields.  Catch-all for
                       merge_shards.py output keyed by equation_id.
                       { "N1": { "equation_id": "N1", "test_r2": 0.9 } }

    noise_sweep_per_noise
                       suppB noise-sweep runner output.  Top-level dict with
                       a "per_noise" key mapping noise levels to method dicts,
                       each containing an "equations" sub-dict of per-equation
                       records.  Aggregate stats (median_r2, recovery_rate,
                       etc.) are hoisted onto every extracted record.
                       { "per_noise": { "0.0": { "MethodA": {
                           "equations": { "Eq1": {r2, rmse, ...} } } } } }

    sample_complexity_per_n
                       suppC sample-complexity runner output.  Top-level dict
                       with a "per_n" key mapping sample sizes to dicts
                       containing "method_summary" (aggregate stats) and
                       "per_equation" (equation → method → {r2, rmse, success}).
                       The meaningful unit is one (sample_size x equation x
                       method) triple; method_summary stats are hoisted onto
                       each record.
                       { "per_n": { "50": { "per_equation": {
                           "Eq1": { "MethodA": {r2, ...} } } } } }

Adding a new format
-------------------
  Tier 1: add one entry to _TIER1_EXTRACTORS (guard + extract lambda).
  Tier 2: add one entry to _TIER2_EXTRACTORS with an explanatory comment.
  Either way: add a corresponding test case to _SELF_TEST_CASES at the bottom
  of this file and run:  python3 validate_analysis_input.py --self-test
"""

import json
import os
import pathlib
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_result_fields(d):
    """True if dict d looks like a result record (contains a known metric key)."""
    return isinstance(d, dict) and bool({"r2", "success", "r_squared", "method"} & d.keys())


def _flatten_method_dict(mapping):
    """
    Flatten {eq_id: {method: {r2, ...}}} into a list of records,
    injecting 'equation' and 'method' keys.
    """
    records = []
    for eq_id, methods in mapping.items():
        if eq_id.startswith("_"):
            continue
        if isinstance(methods, dict):
            for method, mval in methods.items():
                if isinstance(mval, dict):
                    rec = dict(mval)
                    rec.setdefault("equation", eq_id)
                    rec.setdefault("method", method)
                    records.append(rec)
    return records


def _non_meta(data):
    """Strip well-known metadata keys from a top-level dict."""
    return {
        k: v for k, v in data.items()
        if not k.startswith("_") and k not in ("stats", "summary", "metadata", "config")
    }


# ---------------------------------------------------------------------------
# Tier 1 — Canonical extractors
# Each entry: (format_name, guard(data) -> bool, extract(data) -> list)
# Guards are evaluated in order; first match wins.
# ---------------------------------------------------------------------------

_TIER1_EXTRACTORS = [
    (
        "flat_list",
        lambda d: isinstance(d, list),
        lambda d: d,
    ),
    (
        "results_list",
        lambda d: isinstance(d, dict) and isinstance(d.get("results"), list),
        lambda d: d["results"],
    ),
    (
        "tests_list",
        lambda d: isinstance(d, dict) and isinstance(d.get("tests"), list),
        lambda d: d["tests"],
    ),
    (
        "results_dict_of_lists",
        lambda d: (
            isinstance(d, dict)
            and isinstance(d.get("results"), dict)
            and bool(d["results"])
            and all(isinstance(v, list) for v in d["results"].values())
        ),
        lambda d: [r for rs in d["results"].values() for r in rs],
    ),
]

# ---------------------------------------------------------------------------
# Tier 2 — Legacy / third-party extractors
# Same tuple shape as Tier 1.  Heuristic guards; document the source runner.
# ---------------------------------------------------------------------------

_TIER2_EXTRACTORS = [
    # Old runner: results keyed by equation, each value a dict of method->metrics
    (
        "results_dict_of_dicts_methods",
        lambda d: (
            isinstance(d, dict)
            and isinstance(d.get("results"), dict)
            and any(
                isinstance(v, dict)
                and any(isinstance(mv, dict) and _has_result_fields(mv) for mv in v.values())
                for v in d["results"].values()
                if isinstance(v, dict)
            )
        ),
        lambda d: _flatten_method_dict(d["results"]),
    ),
    # Old runner: results keyed by equation, each value a flat record dict
    (
        "results_dict_of_dicts_flat",
        lambda d: (
            isinstance(d, dict)
            and isinstance(d.get("results"), dict)
            and all(
                not k.startswith("_") and _has_result_fields(v)
                for k, v in d["results"].items()
                if isinstance(v, dict)
            )
            and bool(d["results"])
        ),
        lambda d: [
            {**v, "equation": v.get("equation", k)}
            for k, v in d["results"].items()
            if isinstance(v, dict) and not k.startswith("_")
        ],
    ),
    # Top-level dict, no "results" wrapper, equation->method->metrics (old nested shape)
    (
        "toplevel_method_nested",
        lambda d: (
            isinstance(d, dict)
            and bool(_non_meta(d))
            and (lambda nm: (
                bool(nm)
                and isinstance(next(iter(nm.values())), dict)
                and (lambda iv: isinstance(iv, dict) and _has_result_fields(iv))(
                    next(iter(next(iter(nm.values())).values()), None)
                )
            ))(_non_meta(d))
        ),
        lambda d: _flatten_method_dict(_non_meta(d)),
    ),
    # Top-level dict, no "results" wrapper, equation->flat record with result fields
    (
        "toplevel_flat_records",
        lambda d: (
            isinstance(d, dict)
            and bool(_non_meta(d))
            and all(_has_result_fields(v) for v in _non_meta(d).values() if isinstance(v, dict))
            and any(isinstance(v, dict) for v in _non_meta(d).values())
        ),
        lambda d: [
            {**v, "equation": v.get("equation", k)}
            for k, v in _non_meta(d).items()
            if isinstance(v, dict)
        ],
    ),
    # suppB noise-sweep runner output:
    #   { "generated": ..., "noise_levels": [...], "methods": [...],
    #     "per_noise": { "0.0": { "MethodA": { "equations": { "EqName": {r2, rmse, ...} } } } },
    #     "cross_noise_summary": { ... } }
    # The meaningful unit for analysis is one (noise_level x method x equation) triple.
    # Aggregate stats under per_noise[nl][method] (median_r2, recovery_rate, etc.) are
    # hoisted onto every record so run_analysis.py has full context without a join.
    (
        "noise_sweep_per_noise",
        lambda d: (
            isinstance(d, dict)
            and isinstance(d.get("per_noise"), dict)
            and bool(d["per_noise"])
        ),
        lambda d: [
            {
                "noise_level":          nl,
                "method":               method,
                "equation":             eq,
                "median_r2":            m_val.get("median_r2"),
                "mean_r2":              m_val.get("mean_r2"),
                "std_r2":               m_val.get("std_r2"),
                "recovery_rate":        m_val.get("recovery_rate"),
                "n_success":            m_val.get("n_success"),
                "n_total":              m_val.get("n_total"),
                "threshold_used":       m_val.get("threshold_used"),
                "n_catastrophic":       m_val.get("n_catastrophic"),
                **({k: v for k, v in eq_val.items()} if isinstance(eq_val, dict) else {}),
            }
            for nl, nl_val in d["per_noise"].items()
            if isinstance(nl_val, dict)
            for method, m_val in nl_val.items()
            if isinstance(m_val, dict)
            for eq, eq_val in m_val.get("equations", {}).items()
            if isinstance(eq_val, dict)
        ],
    ),
    # suppC sample-complexity runner output:
    #   { "generated": ..., "sample_sizes": [...], "mode": ..., "threshold": {...},
    #     "methods": [...],
    #     "per_n": {
    #       "50": {
    #         "method_summary": { "MethodA": {median_r2, recovery_rate, ...} },
    #         "per_equation":   { "EqName":  { "MethodA": {r2, rmse, success}, ... } }
    #       }, ...
    #     },
    #     "data_efficiency": { "MethodA": {min_n_above_threshold, recovery_curve, ...} }
    #   }
    # The meaningful unit is one (sample_size x equation x method) triple.
    # method_summary stats are hoisted onto each record for full context.
    (
        "sample_complexity_per_n",
        lambda d: (
            isinstance(d, dict)
            and isinstance(d.get("per_n"), dict)
            and bool(d["per_n"])
        ),
        lambda d: [
            {
                "sample_size":    int(n),
                "equation":       eq,
                "method":         method,
                "mode":           d.get("mode"),
                "threshold":      d.get("threshold", {}).get(str(n)),
                # method_summary stats hoisted for full context
                "median_r2":      d["per_n"][n].get("method_summary", {}).get(method, {}).get("median_r2"),
                "mean_r2":        d["per_n"][n].get("method_summary", {}).get(method, {}).get("mean_r2"),
                "std_r2":         d["per_n"][n].get("method_summary", {}).get(method, {}).get("std_r2"),
                "recovery_rate":  d["per_n"][n].get("method_summary", {}).get(method, {}).get("recovery_rate"),
                "n_success":      d["per_n"][n].get("method_summary", {}).get(method, {}).get("n_success"),
                "n_total":        d["per_n"][n].get("method_summary", {}).get(method, {}).get("n_total"),
                # per-equation metrics
                **({k: v for k, v in eq_val.items()} if isinstance(eq_val, dict) else {}),
            }
            for n, n_val in d["per_n"].items()
            if isinstance(n_val, dict)
            for eq, eq_methods in n_val.get("per_equation", {}).items()
            if isinstance(eq_methods, dict)
            for method, eq_val in eq_methods.items()
            if isinstance(eq_val, dict)
        ],
    ),
    # Experiment summary / metadata dict (e.g. exp2_pca_4060_summary.json).
    # These files hold aggregate stats (n_pass, n_total, solve_rate, …) for
    # human inspection and are not record files.  We recognise them explicitly
    # so load_records returns [] rather than raising "no extractor matched".
    # Guard: top-level dict that has ALL of the canonical summary keys and
    # contains NO list-of-records or results-keyed children.
    (
        "experiment_summary_dict",
        lambda d: (
            isinstance(d, dict)
            and {"n_pass", "n_total", "solve_rate"}.issubset(d.keys())
            and not isinstance(d.get("results"), (list, dict))
            and not isinstance(d.get("tests"), list)
            and not isinstance(d.get("per_noise"), dict)
            and not isinstance(d.get("per_n"), dict)
        ),
        lambda d: [],   # no per-record data to validate
    ),
    # Top-level dict, no "results" wrapper, equation->generic dict (merge_shards.py output)
    (
        "toplevel_generic_dicts",
        lambda d: (
            isinstance(d, dict)
            and bool(_non_meta(d))
            and all(isinstance(v, dict) for v in _non_meta(d).values())
        ),
        lambda d: [
            {**v, "equation_id": v.get("equation_id", k), "equation": v.get("equation", k)}
            for k, v in _non_meta(d).items()
        ],
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_records(path):
    """
    Load result records from *path*, auto-detecting the JSON schema.

    Returns a list of record dicts.  Raises ValueError if no extractor
    matches (rather than silently returning []).

    Prints a single diagnostic line to stdout:
        format=<name>  tier=<1|2>  records=<n>
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for tier, extractors in ((1, _TIER1_EXTRACTORS), (2, _TIER2_EXTRACTORS)):
        for name, guard, extract in extractors:
            try:
                matched = guard(data)
            except Exception:
                matched = False
            if matched:
                try:
                    records = extract(data)
                except Exception as exc:
                    raise ValueError(
                        f"{path}: extractor '{name}' (tier {tier}) raised: {exc}"
                    ) from exc
                print(f"  format={name}  tier={tier}  records={len(records)}")
                return records

    top = list(data.keys()) if isinstance(data, dict) else type(data).__name__
    raise ValueError(
        f"{path}: no extractor matched.\n"
        f"  Top-level type : {type(data).__name__}\n"
        f"  Top-level keys : {top}\n"
        f"  Add a new extractor to _TIER1_EXTRACTORS or _TIER2_EXTRACTORS\n"
        f"  and a test case to _SELF_TEST_CASES."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _run_self_tests()
        return

    mode = os.environ["INPUT_MODE"]
    total = 0

    if mode in ("merged", "direct"):
        path = os.environ["INPUT_JSON"]
        if not path or not pathlib.Path(path).is_file():
            print(f"::error::INPUT_JSON='{path}' does not exist or is not a file.")
            sys.exit(1)
        label = "Merged" if mode == "merged" else "Direct"
        print(f"{label} file: {path}")
        records = load_records(path)
        print(f"Records: {len(records)}")
        total += len(records)

    elif mode == "shards":
        manifest_path = os.environ.get("SHARD_MANIFEST", "")
        if not manifest_path or not pathlib.Path(manifest_path).is_file():
            print(f"::error::SHARD_MANIFEST='{manifest_path}' is not a file.")
            sys.exit(1)
        manifest = pathlib.Path(manifest_path)
        n_summary_shards = 0
        for line in manifest.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            print(f"Shard: {line}")
            records = load_records(line)
            print(f"Records: {len(records)}")
            total += len(records)
            # experiment_summary_dict shards legitimately return 0 records;
            # track them so we don't misfire FATAL: EMPTY DATASET.
            if len(records) == 0:
                n_summary_shards += 1

    else:
        print(f"::error::Unknown INPUT_MODE='{mode}'. Expected: merged | direct | shards")
        sys.exit(1)

    print(f"TOTAL_RECORDS={total}")

    # Fail only when there are zero records AND no shard was a recognised
    # summary/metadata file (which legitimately contributes 0 records).
    _n_summary = n_summary_shards if mode == "shards" else 0
    if total == 0 and _n_summary == 0:
        print()
        print("FATAL: EMPTY DATASET")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Self-tests  (python3 validate_analysis_input.py --self-test)
# ---------------------------------------------------------------------------
#
# Each entry:
#   name          Human-readable label shown in test output
#   payload       The Python object that will be JSON-serialised and fed to load_records
#   expected_n    Expected number of records returned (None = expect ValueError)
#   expected_fmt  Expected format name reported by load_records (None = don't check)
#   tier          Expected tier (None = don't check)
#
_SELF_TEST_CASES = [
    # ------------------------------------------------------------------
    # Tier 1
    # ------------------------------------------------------------------
    dict(
        name="tier1 / flat_list",
        payload=[{"r2": 0.9, "equation": "N1"}, {"r2": 0.8, "equation": "N2"}],
        expected_n=2, expected_fmt="flat_list", tier=1,
    ),
    dict(
        name="tier1 / results_list",
        payload={"results": [{"r2": 0.9}, {"r2": 0.8}], "config": {}},
        expected_n=2, expected_fmt="results_list", tier=1,
    ),
    dict(
        name="tier1 / tests_list",
        payload={"tests": [{"r2": 0.9}, {"r2": 0.7}]},
        expected_n=2, expected_fmt="tests_list", tier=1,
    ),
    dict(
        name="tier1 / results_dict_of_lists  (exp3 runner / seed42)",
        payload={
            "config": {"seed": 42},
            "results": {
                "hypatiax": [{"system": "hypatiax", "evaluation": {"r2": 1.0}, "equation_name": "N1"}],
                "pysr":     [{"system": "pysr",     "evaluation": {"r2": 0.9}, "equation_name": "N1"}],
            },
            "summary": {"n_total": 1},
        },
        expected_n=2, expected_fmt="results_dict_of_lists", tier=1,
    ),
    dict(
        name="tier1 / results_dict_of_lists  multi-system 12-eq",
        payload={
            "results": {
                "sysA": [{"r2": i * 0.1} for i in range(12)],
                "sysB": [{"r2": i * 0.1} for i in range(12)],
            }
        },
        expected_n=24, expected_fmt="results_dict_of_lists", tier=1,
    ),
    # ------------------------------------------------------------------
    # Tier 2
    # ------------------------------------------------------------------
    dict(
        name="tier2 / results_dict_of_dicts_methods",
        payload={
            "results": {
                "N1": {"hypatiax": {"r2": 0.99, "success": True}, "pysr": {"r2": 0.95}},
                "N2": {"hypatiax": {"r2": 0.80}, "pysr": {"r2": 0.75}},
            }
        },
        expected_n=4, expected_fmt="results_dict_of_dicts_methods", tier=2,
    ),
    dict(
        name="tier2 / results_dict_of_dicts_flat",
        payload={
            "results": {
                "N1": {"r2": 0.99, "method": "hypatiax"},
                "N2": {"r2": 0.80, "method": "hypatiax"},
            }
        },
        expected_n=2, expected_fmt="results_dict_of_dicts_flat", tier=2,
    ),
    dict(
        name="tier2 / toplevel_method_nested",
        payload={
            "N1": {"hypatiax": {"r2": 0.99, "success": True}, "pysr": {"r2": 0.95}},
            "N2": {"hypatiax": {"r2": 0.80}, "pysr": {"r2": 0.75}},
        },
        expected_n=4, expected_fmt="toplevel_method_nested", tier=2,
    ),
    dict(
        name="tier2 / toplevel_flat_records",
        payload={
            "N1": {"r2": 0.99, "method": "hypatiax"},
            "N2": {"r2": 0.80, "method": "pysr"},
            "N3": {"success": True, "r2": 0.70, "method": "hypatiax"},
        },
        expected_n=3, expected_fmt="toplevel_flat_records", tier=2,
    ),
    dict(
        name="tier2 / toplevel_generic_dicts  (merge_shards.py output)",
        payload={
            "N1": {"equation_id": "N1", "test_r2": 0.9, "expression": "x**2"},
            "N2": {"equation_id": "N2", "test_r2": 0.8, "expression": "x+1"},
        },
        expected_n=2, expected_fmt="toplevel_generic_dicts", tier=2,
    ),
    dict(
        name="tier2 / noise_sweep_per_noise  (suppB runner)",
        payload={
            "generated": "2026-05-22T18:22:26",
            "noise_levels": [0.0, 0.05],
            "methods": ["MethodA", "MethodB"],
            "per_noise": {
                "0.0": {
                    "MethodA": {
                        "median_r2": 0.999, "mean_r2": 0.998, "std_r2": 0.001,
                        "recovery_rate": 0.9, "n_success": 9, "n_total": 10,
                        "threshold_used": 0.95, "n_catastrophic": 0,
                        "equations": {
                            "Eq1": {"r2": 1.0, "rmse": 0.0, "success": True, "catastrophic": False},
                            "Eq2": {"r2": 0.99, "rmse": 0.01, "success": True, "catastrophic": False},
                        },
                    },
                    "MethodB": {
                        "median_r2": 0.95, "mean_r2": 0.94, "std_r2": 0.02,
                        "recovery_rate": 0.8, "n_success": 8, "n_total": 10,
                        "threshold_used": 0.95, "n_catastrophic": 1,
                        "equations": {
                            "Eq1": {"r2": 0.95, "rmse": 0.1, "success": True, "catastrophic": False},
                        },
                    },
                },
                "0.05": {
                    "MethodA": {
                        "median_r2": 0.97, "mean_r2": 0.96, "std_r2": 0.02,
                        "recovery_rate": 0.7, "n_success": 7, "n_total": 10,
                        "threshold_used": 0.95, "n_catastrophic": 1,
                        "equations": {
                            "Eq1": {"r2": 0.97, "rmse": 0.05, "success": True, "catastrophic": False},
                        },
                    },
                },
            },
            "cross_noise_summary": {},
        },
        expected_n=4, expected_fmt="noise_sweep_per_noise", tier=2,
    ),
    dict(
        name="tier2 / sample_complexity_per_n  (suppC runner)",
        payload={
            "generated": "2026-05-22T19:11:12",
            "sample_sizes": [50, 100],
            "mode": "noisy",
            "threshold": {"50": 0.995, "100": 0.995},
            "methods": ["MethodA", "MethodB"],
            "per_n": {
                "50": {
                    "method_summary": {
                        "MethodA": {"median_r2": 0.999, "mean_r2": 0.998, "std_r2": 0.001,
                                    "recovery_rate": 1.0, "n_success": 2, "n_total": 2,
                                    "threshold_used": 0.995},
                        "MethodB": {"median_r2": 0.95, "mean_r2": 0.94, "std_r2": 0.02,
                                    "recovery_rate": 0.5, "n_success": 1, "n_total": 2,
                                    "threshold_used": 0.995},
                    },
                    "per_equation": {
                        "Eq1": {
                            "MethodA": {"r2": 1.0,  "rmse": 0.0,  "success": True},
                            "MethodB": {"r2": 0.95, "rmse": 0.1,  "success": True},
                        },
                        "Eq2": {
                            "MethodA": {"r2": 0.99, "rmse": 0.01, "success": True},
                            "MethodB": {"r2": 0.60, "rmse": 0.5,  "success": False},
                        },
                    },
                },
                "100": {
                    "method_summary": {
                        "MethodA": {"median_r2": 1.0, "mean_r2": 1.0, "std_r2": 0.0,
                                    "recovery_rate": 1.0, "n_success": 2, "n_total": 2,
                                    "threshold_used": 0.995},
                    },
                    "per_equation": {
                        "Eq1": {
                            "MethodA": {"r2": 1.0, "rmse": 0.0, "success": True},
                        },
                    },
                },
            },
            "data_efficiency": {
                "MethodA": {"min_n_above_threshold": 50, "recovery_curve": {"50": 1.0, "100": 1.0}},
            },
        },
        # 50: 2 eq x 2 methods = 4, 100: 1 eq x 1 method = 1  → total 5
        expected_n=5, expected_fmt="sample_complexity_per_n", tier=2,
    ),
    dict(
        name="tier2 / experiment_summary_dict  (exp2_pca_4060_summary.json)",
        payload={
            "fixc3_step": "pca_4060",
            "description": "PCA 40/60 split summary",
            "split_protocol": "pca",
            "extrap_train_frac": 0.4,
            "extrap_multiplier": 1.5,
            "n_pass": 162,
            "n_total": 180,
            "solve_rate": 0.9,
            "paper_legacy_claim": 0.88,
            "source_files": ["benchmark_results_extrap.json"],
        },
        expected_n=0, expected_fmt="experiment_summary_dict", tier=2,
    ),
    # ------------------------------------------------------------------
    # Error / no-match
    # ------------------------------------------------------------------
    dict(
        name="no-match / scalar string",
        payload="just a string",
        expected_n=None, expected_fmt=None, tier=None,
    ),
    dict(
        name="no-match / empty dict",
        payload={},
        expected_n=None, expected_fmt=None, tier=None,
    ),
    dict(
        name="no-match / dict with only meta keys",
        payload={"_meta": {}, "summary": {}, "stats": {}},
        expected_n=None, expected_fmt=None, tier=None,
    ),
]


def _run_self_tests():
    import tempfile, traceback

    passed = failed = 0
    tier_counts = {1: {"pass": 0, "fail": 0}, 2: {"pass": 0, "fail": 0}, None: {"pass": 0, "fail": 0}}

    # Group for display
    current_group = None

    for case in _SELF_TEST_CASES:
        name        = case["name"]
        payload     = case["payload"]
        expected_n  = case["expected_n"]
        expected_fmt = case.get("expected_fmt")
        tier        = case.get("tier")

        # Print group header
        group = name.split("/")[0].strip()
        if group != current_group:
            print(f"\n  {'─' * 56}")
            print(f"  {group.upper()}")
            print(f"  {'─' * 56}")
            current_group = group

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(payload, tf)
            tmp_path = tf.name

        captured_fmt = None
        captured_n   = None
        error        = None

        # Monkey-patch print to capture the format= line
        original_print = __builtins__["print"] if isinstance(__builtins__, dict) else print
        import builtins
        _captured = []
        _orig = builtins.print
        def _capturing_print(*args, **kwargs):
            _captured.append(" ".join(str(a) for a in args))
            _orig(*args, **kwargs)
        builtins.print = _capturing_print

        try:
            records = load_records(tmp_path)
            captured_n = len(records)
            for line in _captured:
                if line.strip().startswith("format="):
                    parts = dict(p.split("=", 1) for p in line.strip().split() if "=" in p)
                    captured_fmt = parts.get("format")
        except ValueError as exc:
            error = exc
        finally:
            builtins.print = _orig
            pathlib.Path(tmp_path).unlink(missing_ok=True)

        # Evaluate
        if expected_n is None:
            # Expect a ValueError
            ok = error is not None
            status = "PASS" if ok else "FAIL (expected ValueError, got records)"
        else:
            fmt_ok = (expected_fmt is None) or (captured_fmt == expected_fmt)
            ok = (error is None) and (captured_n == expected_n) and fmt_ok
            if error:
                status = f"FAIL (unexpected error: {error})"
            elif captured_n != expected_n:
                status = f"FAIL (got {captured_n} records, expected {expected_n})"
            elif not fmt_ok:
                status = f"FAIL (format='{captured_fmt}', expected='{expected_fmt}')"
            else:
                status = "PASS"

        label = name.split("/", 1)[-1].strip()
        print(f"  {'✓' if ok else '✗'}  {label:<48}  {status}")

        if ok:
            passed += 1
            tier_counts[tier]["pass"] += 1
        else:
            failed += 1
            tier_counts[tier]["fail"] += 1

    print(f"\n  {'═' * 56}")
    print(f"  Results: {passed} passed, {failed} failed")
    for t, counts in sorted((k, v) for k, v in tier_counts.items() if k is not None):
        total = counts["pass"] + counts["fail"]
        if total:
            print(f"    Tier {t}: {counts['pass']}/{total} passed")
    no_t = tier_counts[None]
    if no_t["pass"] + no_t["fail"]:
        print(f"    Error cases: {no_t['pass']}/{no_t['pass'] + no_t['fail']} passed")
    print(f"  {'═' * 56}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
