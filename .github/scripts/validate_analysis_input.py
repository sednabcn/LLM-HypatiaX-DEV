#!/usr/bin/env python3
"""
validate_analysis_input.py
--------------------------
Reads INPUT_MODE, INPUT_JSON, SHARD_MANIFEST from the environment and counts
the total number of result records across all input files.  Exits 1 on an
empty dataset (FATAL: EMPTY DATASET).

Called by ci_analysis.yml "Validate input data" step.
"""
import json
import os
import pathlib
import sys


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: flat list of records
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        return []

    # Case 2: {"results": [...]} or {"tests": [...]}  — standard shard/benchmark wrapper
    for wrapper_key in ("results", "tests"):
        if wrapper_key in data and isinstance(data[wrapper_key], list):
            return data[wrapper_key]

    # Case 3: {"results": {"eq_id": {...}, ...}}  — dict-of-dicts under "results"
    if "results" in data and isinstance(data["results"], dict):
        records = []
        for eq_id, eq_val in data["results"].items():
            if eq_id.startswith("_"):
                continue
            if isinstance(eq_val, dict):
                has_method_keys = any(
                    isinstance(v, dict) and ("r2" in v or "success" in v or "r_squared" in v)
                    for v in eq_val.values()
                )
                if has_method_keys:
                    for method, mval in eq_val.items():
                        if isinstance(mval, dict):
                            rec = dict(mval)
                            rec.setdefault("equation", eq_id)
                            rec.setdefault("method", method)
                            records.append(rec)
                else:
                    rec = dict(eq_val)
                    rec.setdefault("equation", eq_id)
                    records.append(rec)
        return records

    # Case 4: top-level dict keyed by equation/task (no "results" wrapper).
    # This is the canonical shape written by merge_shards.py:
    #   { "equation_id_1": { ...record... }, "equation_id_2": { ... }, ... }
    # Records may use "test_r2", "results", "equation_id", etc. — we do NOT
    # require specific field names; any dict value that isn't a meta key counts.
    non_meta = {
        k: v for k, v in data.items()
        if not k.startswith("_") and k not in ("stats", "summary", "metadata")
    }
    if not non_meta:
        return []

    first_val = next(iter(non_meta.values()))

    # Sub-case 4a: {eq_id: {method: {r2, success, ...}}}  — old nested method shape
    if isinstance(first_val, dict):
        inner_first = next(iter(first_val.values()), None) if first_val else None
        if isinstance(inner_first, dict) and (
            "r2" in inner_first or "success" in inner_first or "r_squared" in inner_first
        ):
            records = []
            for eq_id, methods in non_meta.items():
                if isinstance(methods, dict):
                    for method, mval in methods.items():
                        if isinstance(mval, dict):
                            rec = dict(mval)
                            rec.setdefault("equation", eq_id)
                            rec.setdefault("method", method)
                            records.append(rec)
            return records

        # Sub-case 4b: {eq_id: {r2/success/method, ...}}  — flat per-equation record
        if (
            "r2" in first_val or "success" in first_val
            or "r_squared" in first_val or "method" in first_val
        ):
            records = []
            for eq_id, rec_val in non_meta.items():
                if isinstance(rec_val, dict):
                    rec = dict(rec_val)
                    rec.setdefault("equation", eq_id)
                    records.append(rec)
            return records

        # Sub-case 4c: generic dict-of-dicts (merge_shards.py output).
        # Covers normalised protocol records keyed by equation_id:
        #   { "eq_id": { "equation_id": ..., "results": {...}, "test_r2": ..., ... } }
        # Accept any non-empty dict value without requiring specific field names.
        if all(isinstance(v, dict) for v in non_meta.values()):
            records = []
            for eq_id, rec_val in non_meta.items():
                rec = dict(rec_val)
                rec.setdefault("equation_id", eq_id)
                rec.setdefault("equation", eq_id)
                records.append(rec)
            return records

    return []


def main():
    mode = os.environ["INPUT_MODE"]
    total = 0

    if mode in ("merged", "direct"):
        path = os.environ["INPUT_JSON"]
        if not path or not pathlib.Path(path).is_file():
            print(f"::error::INPUT_JSON='{path}' does not exist or is not a file.")
            sys.exit(1)
        records = load_records(path)
        label = "Merged" if mode == "merged" else "Direct"
        print(f"{label} file: {path}")
        print(f"Records: {len(records)}")
        total += len(records)
    elif mode == "shards":
        manifest_path = os.environ.get("SHARD_MANIFEST", "")
        if not manifest_path or not pathlib.Path(manifest_path).is_file():
            print(f"::error::SHARD_MANIFEST='{manifest_path}' is not a file.")
            sys.exit(1)
        manifest = pathlib.Path(manifest_path)
        for line in manifest.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            records = load_records(line)
            print(f"Shard: {line}")
            print(f"Records: {len(records)}")
            total += len(records)
    else:
        print(f"::error::Unknown INPUT_MODE='{mode}'. Expected: merged | direct | shards")
        sys.exit(1)

    print(f"TOTAL_RECORDS={total}")

    if total == 0:
        print()
        print("FATAL: EMPTY DATASET")
        sys.exit(1)


if __name__ == "__main__":
    main()
