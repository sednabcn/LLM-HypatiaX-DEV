#!/usr/bin/env python3
"""
Looks at every PureLLM Baseline (core) result across the exp2_pca_4060 shards
and splits them into:
  - explicitly hardcoded (metadata.is_hardcoded == True)
  - perfect/near-perfect but NOT flagged hardcoded (r2 >= 0.9999, is_hardcoded
    False/missing) -- these are the suspicious "possible unflagged memorization"
    cases worth manual review
  - genuinely imperfect (r2 < 0.9999) -- presumably real solve attempts

For each of the middle bucket, prints domain/description + a formula snippet
so a human can eyeball whether it looks like a memorized textbook formula.
"""
import json
import glob

SHARD_DIR = "hypatiax/data/results/comparison_results/feynman-tests/exp2_pca_4060"
METHOD = "PureLLM Baseline (core)"
THRESHOLD = 0.9999

def main():
    files = sorted(glob.glob(f"{SHARD_DIR}/protocol_core_noiseless_pca_*.json"))
    hardcoded, unflagged_perfect, imperfect = [], [], []

    for fpath in files:
        d = json.load(open(fpath))
        for test in d.get("tests", []):
            res = test.get("results", {}).get(METHOD)
            if not res:
                continue
            entry = {
                "domain": test.get("domain"),
                "description": test.get("description"),
                "r2": res.get("r2"),
                "success": res.get("success"),
                "formula": (res.get("formula") or "")[:200],
                "is_hardcoded": res.get("metadata", {}).get("is_hardcoded"),
                "file": fpath,
            }
            if entry["is_hardcoded"]:
                hardcoded.append(entry)
            elif entry["r2"] is not None and entry["r2"] >= THRESHOLD:
                unflagged_perfect.append(entry)
            else:
                imperfect.append(entry)

    print(f"Hardcoded (flagged):        {len(hardcoded)}")
    print(f"Unflagged but perfect:      {len(unflagged_perfect)}  <-- review these")
    print(f"Genuinely imperfect (real): {len(imperfect)}")
    print()
    print("=== Unflagged-but-perfect cases (possible unflagged memorization) ===")
    for e in unflagged_perfect:
        print(f"- [{e['domain']}] {e['description']}  (r2={e['r2']})")
        print(f"    formula: {e['formula']}")
    print()
    print("=== Genuinely imperfect cases (for contrast) ===")
    for e in imperfect:
        print(f"- [{e['domain']}] {e['description']}  (r2={e['r2']})")

if __name__ == "__main__":
    main()
