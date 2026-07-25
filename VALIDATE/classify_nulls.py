#!/usr/bin/env python3
"""
classify_nulls.py

For a target method, classifies every null extrap_r2_far value across a set
of shard files into one of the four known branches of
compute_extrap_r2_far() (run_comparative_suite_benchmark_v2.py):

  A - test-level: far-region split itself missing/degenerate
      (cannot be distinguished from the JSON alone unless extrap metadata
      like extrap_n_test is present on the record; flagged as "A?" if the
      record has no per-method entry at all for ANY method, which is the
      only externally observable symptom of this branch)
  B - method fit did not succeed (results[method].success == False)
  C - method returned an NN-tag / fallback / empty formula (no expression
      to evaluate)
  D - formula existed but evaluation still produced no value (most likely
      branch D, an exception during far-region evaluation)

Usage:
    python classify_nulls.py DIR --method "HybridDiscoverySystem v50_2 (tools)" \
        --pattern "protocol_core_extrap_20260722_*.json"
"""
import argparse
import glob
import json
import os


def classify(rec, method):
    all_methods_null = all(
        v is None for v in (rec.get("extrap_r2_far") or {}).values()
    ) and len(rec.get("extrap_r2_far") or {}) > 0

    block = (rec.get("results") or {}).get(method, {})
    success = block.get("success")
    formula = (block.get("formula") or "").strip()

    is_nn_tag = (
        formula.startswith("ImprovedNN(")
        or formula.startswith("[NN fallback")
        or formula in ("N/A", "")
    )

    if not rec.get("extrap_r2_far"):
        return "A? (no extrap_r2_far dict on this record at all -- likely far-split guard)"
    if success is False:
        return "B (method fit did not succeed)"
    if is_nn_tag:
        return f"C (no formula to evaluate -- formula={formula!r})"
    if formula:
        return "D? (formula present but still null -- likely an evaluation exception)"
    return "UNRESOLVED -- inspect this record manually"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--method", required=True)
    ap.add_argument("--pattern", default="protocol_core_*.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.directory, args.pattern)))
    null_count = 0
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        records = data.get("tests", data if isinstance(data, list) else [data])
        for rec in records:
            far_map = rec.get("extrap_r2_far") or {}
            val = far_map.get(args.method)
            if val is None:
                null_count += 1
                domain = rec.get("domain", "")
                eq = rec.get("description", rec.get("equation_name", ""))
                reason = classify(rec, args.method)
                print(f"[{domain}] {eq}")
                print(f"    -> {reason}")

    print(f"\nTotal nulls classified: {null_count}")


if __name__ == "__main__":
    main()
