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

    # Case 2: {"results": [...]}  — standard shard wrapper
    if "results" in data and isinstance(data["results"], list):
        return data["results"]

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

    # Case 4: top-level dict keyed by equation/task (no "results" wrapper)
    non_meta = {
        k: v for k, v in data.items()
        if not k.startswith("_") and k not in ("stats", "summary", "metadata")
    }
    if not non_meta:
        return []

    first_val = next(iter(non_meta.values()))

    # Sub-case 4a: {eq_id: {method: {r2, success, ...}}}
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

        # Sub-case 4b: {eq_id: {r2, success, method, ...}}
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

    return []


def main():
    mode = os.environ.get("INPUT_MODE", "").strip()
    if not mode:
        print("FATAL: INPUT_MODE is not set or empty", file=sys.stderr)
        sys.exit(1)

    total = 0

    if mode in ("merged", "direct"):
        # Both merged and direct use a single JSON file via INPUT_JSON.
        path = os.environ.get("INPUT_JSON", "").strip()
        if not path:
            print(f"FATAL: INPUT_MODE={mode} but INPUT_JSON is not set or empty",
                  file=sys.stderr)
            sys.exit(1)
        records = load_records(path)
        label = "Merged" if mode == "merged" else "Direct"
        print(f"{label} file: {path}")
        print(f"Records: {len(records)}")
        total += len(records)

    elif mode == "shards":
        # ci_analysis.yml writes  SHARD_MANIFEST=  (empty) for merged/direct modes
        # so that the env var is always defined.  An empty value here means the
        # "Locate analysis input" step never reached the SHARDS branch — config error.
        raw = os.environ.get("SHARD_MANIFEST", "").strip()
        if not raw:
            print(
                "FATAL: INPUT_MODE=shards but SHARD_MANIFEST is not set or empty.\n"
                "       Check the 'Locate analysis input' step in ci_analysis.yml.",
                file=sys.stderr,
            )
            sys.exit(1)

        manifest = pathlib.Path(raw).resolve()
        print(f"Manifest: {manifest}")

        if not manifest.exists():
            print(f"FATAL: manifest does not exist: {manifest}", file=sys.stderr)
            sys.exit(1)
        if manifest.is_dir():
            print(
                f"FATAL: manifest path is a directory, expected a file: {manifest}",
                file=sys.stderr,
            )
            sys.exit(1)

        for line in manifest.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            records = load_records(line)
            print(f"Shard: {line}")
            print(f"Records: {len(records)}")
            total += len(records)

    else:
        print(f"FATAL: unknown INPUT_MODE={mode!r}. Expected: merged | direct | shards",
              file=sys.stderr)
        sys.exit(1)

    print(f"TOTAL_RECORDS={total}")

    if total == 0:
        print()
        print("FATAL: EMPTY DATASET")
        sys.exit(1)


if __name__ == "__main__":
    main()
