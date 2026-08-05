#!/usr/bin/env python3
"""Cheap pre-check: does the augmentation-mismatch bug show up anywhere in an
existing hypatiax_defi_benchmark_pca_results.json (or v3c) file?

Usage: python3 check_nn_nan_fingerprint.py path/to/results.json
"""
import json, sys, math

path = sys.argv[1] if len(sys.argv) > 1 else "hypatiax_defi_benchmark_pca_results.json"
data = json.loads(open(path).read())

hits = []
for c in data:
    nn = c.get("results", {}).get("neural_network", {})
    err = nn.get("error", "") or ""
    tr2 = nn.get("test_r2")
    is_nan = tr2 is None or (isinstance(tr2, float) and math.isnan(tr2))
    if "StandardScaler is expecting" in err or ("features, but" in err) or is_nan:
        hits.append((c.get("equation_id"), c.get("seed"), err or "(nan, no error string)"))

print(f"Scanned {len(data)} records in {path}")
print(f"Records matching the feature-mismatch fingerprint: {len(hits)}")
for h in hits:
    print(" ", h)
