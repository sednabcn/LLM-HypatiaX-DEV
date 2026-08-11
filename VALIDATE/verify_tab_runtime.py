#!/usr/bin/env python3
"""
verify_tab_runtime.py — recompute Table 5 (runtime) directly from raw
results, to settle whether HypatiaX is "1.73x faster" (original tab:timing)
or "4.09x slower" (tab_runtime_corrected.tex, tab:runtime) on the
LLM-routed subset.

CONTRACT (matches scripts/investigate_item3_runtime.py in the patch
tooling): read raw data only, never the .tex files, never guess. Print one
JSON result to stdout; diagnostics go to stderr. Exit nonzero if the file
can't be found/parsed, or if the schema doesn't match what's expected —
never silently fall back to a placeholder number.

USAGE
    python3 verify_tab_runtime.py --file hypatiax_defi_benchmark_v3_results_seed42.json

WHAT THIS ASSUMES ABOUT THE JSON SCHEMA
    A top-level list (or a dict with a list under a "results"/"tasks" key)
    of per-task records, each with (name variants are auto-detected):
      - a per-method wall-clock time in seconds, one field per method
        OR one record per (task, method) pair with a "method" field and a
        single time field
      - a boolean / string flag indicating whether the HypatiaX run for
        that task was LLM-routed vs. NN-fallback-routed
    If your file's schema doesn't match, this script will print exactly
    what it found and exit nonzero rather than guess field mappings.
"""
import argparse
import json
import statistics
import sys

# Candidate field names, in priority order, for each thing we need.
METHOD_FIELD_CANDIDATES = ["method", "system", "model_name", "config"]
TIME_FIELD_CANDIDATES = ["wall_time_s", "runtime_s", "time_s", "duration_s", "wall_clock_s"]
ROUTED_FIELD_CANDIDATES = ["llm_routed", "routed_to_llm", "routing_decision", "used_llm_formula", "stage3_routed"]

METHOD_NAME_MAP = {
    # normalize whatever labels the JSON uses to the four rows in the table
    "pure_llm": "Pure LLM", "purellm": "Pure LLM", "llm_only": "Pure LLM",
    "neural_mlp": "Neural MLP", "nn": "Neural MLP", "mlp": "Neural MLP",
    "hypatiax": "HypatiaX", "hybrid": "HypatiaX", "hybrid_v50_2": "HypatiaX",
}


def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


def find_field(record, candidates):
    for c in candidates:
        if c in record:
            return c
    return None


def load_records(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "tasks", "records", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(
        "expected a JSON list of per-task records, or a dict with a "
        "'results'/'tasks'/'records'/'data' list — got: "
        f"{type(data).__name__} with keys {list(data)[:10] if isinstance(data, dict) else 'n/a'}"
    )


def normalize_method(raw):
    key = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    return METHOD_NAME_MAP.get(key, str(raw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True,
                     help="path to hypatiax_defi_benchmark_v3_results_seed42.json (or equivalent raw results file)")
    args = ap.parse_args()

    try:
        records = load_records(args.file)
    except FileNotFoundError:
        eprint(f"FATAL: file not found: {args.file}")
        sys.exit(2)
    except (json.JSONDecodeError, ValueError) as e:
        eprint(f"FATAL: could not parse {args.file} as expected schema: {e}")
        sys.exit(2)

    if not records:
        eprint("FATAL: file parsed but contained zero records")
        sys.exit(2)

    sample = records[0]
    method_field = find_field(sample, METHOD_FIELD_CANDIDATES)
    time_field = find_field(sample, TIME_FIELD_CANDIDATES)
    routed_field = find_field(sample, ROUTED_FIELD_CANDIDATES)

    if not method_field or not time_field:
        eprint("FATAL: could not identify method/time fields in the first record.")
        eprint(f"  Record keys found: {sorted(sample.keys())}")
        eprint(f"  Tried method field candidates: {METHOD_FIELD_CANDIDATES}")
        eprint(f"  Tried time field candidates:   {TIME_FIELD_CANDIDATES}")
        eprint("  Edit METHOD_FIELD_CANDIDATES / TIME_FIELD_CANDIDATES above to match your schema, then rerun.")
        sys.exit(2)

    eprint(f"[schema] method_field={method_field!r} time_field={time_field!r} routed_field={routed_field!r}")

    by_method = {}
    hypatiax_llm_routed_times = []
    unrouted_flag_missing = 0

    for rec in records:
        m = normalize_method(rec.get(method_field))
        t = rec.get(time_field)
        if t is None:
            continue
        by_method.setdefault(m, []).append(float(t))

        if m == "HypatiaX":
            if routed_field and routed_field in rec:
                val = rec[routed_field]
                is_llm_routed = val in (True, "llm", "llm_routed", "stage3", "yes")
                if is_llm_routed:
                    hypatiax_llm_routed_times.append(float(t))
            else:
                unrouted_flag_missing += 1

    if unrouted_flag_missing and not hypatiax_llm_routed_times:
        eprint(
            f"WARNING: routing flag not found on {unrouted_flag_missing} HypatiaX records "
            f"(tried {ROUTED_FIELD_CANDIDATES}) — cannot compute the LLM-routed-only (n=68) row. "
            "Fix ROUTED_FIELD_CANDIDATES or the normalize logic above, then rerun."
        )

    def stats(times):
        if not times:
            return None
        return {
            "mean": round(statistics.mean(times), 4),
            "median": round(statistics.median(times), 4),
            "n": len(times),
        }

    computed = {m: stats(times) for m, times in by_method.items()}
    nn_stats = computed.get("Neural MLP")

    result = {"per_method": computed}

    if nn_stats:
        for m, s in computed.items():
            if m == "Neural MLP" or s is None:
                continue
            s["vs_nn_mean_ratio"] = round(s["mean"] / nn_stats["mean"], 3)
            s["vs_nn_median_ratio"] = round(s["median"] / nn_stats["median"], 3)

        if hypatiax_llm_routed_times:
            routed_stats = stats(hypatiax_llm_routed_times)
            routed_stats["vs_nn_mean_ratio"] = round(routed_stats["mean"] / nn_stats["mean"], 3)
            routed_stats["vs_nn_median_ratio"] = round(routed_stats["median"] / nn_stats["median"], 3)
            result["hypatiax_llm_routed_only"] = routed_stats

    # --- Compare against BOTH the original table's and the corrected
    #     table's claims. Neither is assumed correct going in.
    claims = {
        "original_tab_timing": {
            "HypatiaX_mean_vs_nn": 2.30,       # "2.30x slower (mean)"
            "HypatiaX_median_vs_nn": 1 / 1.64,  # "1.64x faster (median)" -> ratio < 1
            "llm_routed_n": 68,
            "llm_routed_vs_nn": 1 / 1.73,       # "1.73x faster" -> ratio < 1
        },
        "tab_runtime_corrected": {
            "HypatiaX_mean_vs_nn": 4.82,
            "HypatiaX_median_vs_nn": 3.71,
            "llm_routed_n": 68,
            "llm_routed_vs_nn": 4.09,
        },
    }
    result["claims_being_checked"] = claims

    checks = {}
    hx = computed.get("HypatiaX")
    if hx:
        for label, c in claims.items():
            checks[label] = {
                "mean_ratio_matches": abs(hx["vs_nn_mean_ratio"] - c["HypatiaX_mean_vs_nn"]) < 0.05,
                "median_ratio_matches": abs(hx["vs_nn_median_ratio"] - c["HypatiaX_median_vs_nn"]) < 0.05,
            }
        if "hypatiax_llm_routed_only" in result:
            r = result["hypatiax_llm_routed_only"]
            for label, c in claims.items():
                checks[label]["llm_routed_n_matches"] = (r["n"] == c["llm_routed_n"])
                checks[label]["llm_routed_ratio_matches"] = abs(r["vs_nn_mean_ratio"] - c["llm_routed_vs_nn"]) < 0.05
    result["which_table_matches_raw_data"] = checks

    eprint("\n[computed from raw data]")
    eprint(json.dumps(result, indent=2))

    print(json.dumps(result))
    sys.exit(0)


if __name__ == "__main__":
    main()
