#!/usr/bin/env python3
"""
Build exp2_random8020_summary.json from raw protocol_core_noiseless_*.json
run files, in the same schema as exp2_pca_4060_summary.json.

Usage:
    python3 build_exp2_random8020_summary.py /path/to/raw_run_dir \
        --pattern "protocol_core_noiseless_2026*.json" \
        --out exp2_random8020_summary.json \
        --threshold 0.999999 \
        --paper-legacy-claim "9/30 = 0.300 (random_80_20)"

Assumptions (mirrors exp2_pca_4060_summary.json's actual construction,
verified against the uploaded protocol_core_noiseless_2026*.json files):
  - Each raw file has the shape:
    {"tests": [{"domain": ..., "results": {method: {"r2": ..., "success": ...,
                                                      "metadata": {"decision": ...}}}}]}
  - "Pass" for a test/method = success is True AND r2 >= threshold. If a raw
    file stores the held-out/extrapolation score under a different key
    (e.g. "extrap_r2_far" or "extrap_r2"), set --score-field accordingly.
  - Only these three methods count toward per_method / overall n_pass,n_total:
        EnhancedHybridSystemDeFi (core)
        HybridDiscoverySystem v50_2 (tools)
        HybridSystemLLMNN all-domains (core)
  - Top-level "n_pass"/"n_total" are the SUM across the three per_method
    counts (not a single primary-method count). Confirmed against
    exp2_pca_4060_summary.json: 23+18+30=71 pass, 30*3=90 total.
  - "forced_domains" is the fixed list of domains where HybridAllDomainsMethod
    forces LLM routing. Edit FORCED_DOMAINS below if a run uses a different
    guard list.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

METHODS_OF_INTEREST = [
    "EnhancedHybridSystemDeFi (core)",
    "HybridDiscoverySystem v50_2 (tools)",
    "HybridSystemLLMNN all-domains (core)",
]

FORCED_DOMAINS = [
    "feynman_electromagnetism",
    "feynman_mechanics",
    "feynman_optics",
    "feynman_quantum",
    "feynman_thermodynamics",
]


def load_raw_files(run_dir, pattern):
    paths = sorted(glob.glob(os.path.join(run_dir, pattern)))
    if not paths:
        sys.exit(f"No files matched {pattern!r} in {run_dir}")
    files = []
    for p in paths:
        with open(p) as f:
            files.append((os.path.basename(p), json.load(f)))
    return files


def passes(result, threshold, score_field):
    if result is None:
        return False
    if not result.get("success", False):
        return False
    score = result.get(score_field, result.get("r2"))
    if score is None:
        return False
    try:
        return score >= threshold
    except TypeError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--pattern", default="protocol_core_noiseless_2026*.json")
    ap.add_argument("--out", default="exp2_random8020_summary.json")
    ap.add_argument("--threshold", type=float, default=0.999999)
    ap.add_argument("--score-field", default="r2",
                     help="Field to compare against threshold, e.g. 'r2' or 'extrap_r2_far'")
    ap.add_argument("--paper-legacy-claim", default="9/30 = 0.300 (random_80_20)")
    args = ap.parse_args()

    raw_files = load_raw_files(args.run_dir, args.pattern)

    per_method = {m: {"n_pass": 0, "n_total": 0} for m in METHODS_OF_INTEREST}
    by_domain = defaultdict(lambda: {"llm": 0, "other": 0, "forced": False})

    for _, data in raw_files:
        for test in data.get("tests", []):
            domain = test.get("domain", "unknown")
            results = test.get("results", {})

            for method in METHODS_OF_INTEREST:
                res = results.get(method)
                per_method[method]["n_total"] += 1
                if passes(res, args.threshold, args.score_field):
                    per_method[method]["n_pass"] += 1

            hybrid_res = results.get("HybridSystemLLMNN all-domains (core)")
            if hybrid_res is not None:
                decision = (hybrid_res.get("metadata") or {}).get("decision", "other")
                forced = domain in FORCED_DOMAINS
                by_domain[domain]["forced"] = forced
                if decision == "llm":
                    by_domain[domain]["llm"] += 1
                else:
                    by_domain[domain]["other"] += 1

    forced_llm = sum(v["llm"] for v in by_domain.values() if v["forced"])
    forced_total = sum(v["llm"] + v["other"] for v in by_domain.values() if v["forced"])
    natural_llm = sum(v["llm"] for v in by_domain.values() if not v["forced"])
    natural_total = sum(v["llm"] + v["other"] for v in by_domain.values() if not v["forced"])

    per_method_out = {}
    for method, counts in per_method.items():
        n_pass, n_total = counts["n_pass"], counts["n_total"]
        per_method_out[method] = {
            "n_pass": n_pass,
            "n_total": n_total,
            "solve_rate": (n_pass / n_total) if n_total else 0.0,
        }

    # Overall n_pass/n_total = sum across the three per_method counts
    # (matches exp2_pca_4060_summary.json's convention: 71/90).
    n_pass_overall = sum(v["n_pass"] for v in per_method.values())
    n_total_overall = sum(v["n_total"] for v in per_method.values())

    summary = {
        "fixc3_step": "exp2_feynman_random8020",
        "description": "Corrected Feynman result \u2014 random 80/20 split (legacy protocol)",
        "split_protocol": "random_80_20",
        "n_pass": n_pass_overall,
        "n_total": n_total_overall,
        "solve_rate": (n_pass_overall / n_total_overall) if n_total_overall else 0.0,
        "per_method": per_method_out,
        "hybrid_llm_routing": {
            "note": ("'HybridSystemLLMNN all-domains (core)' decision routing, broken out "
                     "by whether the domain is covered by the explicit force_llm guard in "
                     "HybridAllDomainsMethod.run() (run_protocol_benchmark_core.py) \u2014 "
                     "see comment above this block."),
            "forced_domains": FORCED_DOMAINS,
            "forced_llm_count": f"{forced_llm}/{forced_total}",
            "natural_llm_count": f"{natural_llm}/{natural_total}",
            "by_domain": dict(by_domain),
        },
        "paper_legacy_claim": args.paper_legacy_claim,
        "source_files": [name for name, _ in raw_files],
    }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote {args.out}: {n_pass_overall}/{n_total_overall} "
          f"({summary['solve_rate']:.3f}) from {len(raw_files)} source files")


if __name__ == "__main__":
    main()
