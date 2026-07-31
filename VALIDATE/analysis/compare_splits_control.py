#!/usr/bin/env python3
"""
compare_splits_control.py
==========================
Decisive falsification test for the "same NN result across scripts" finding.

Background
----------
On "Portfolio Sharpe Ratio" (features: portfolio_return, volatility — both plain
np.linspace arrays, monotonically increasing together), v3c.py's _aggressive_split
(percentile on raw feature 1) and pca.py's pca_directed_split (PC1 projection)
produced near-identical NN predictions (max diff ~1.19e-07, i.e. float32 epsilon)
on both train and test sets, with train_r2/test_r2 matching to ~8-10 significant
figures. That match is fully explained by the two input features being collinear
by construction — sorting by either raw feature or by PC1 gives the same row
ordering, so both scripts end up training on the same partition.

This script re-runs the identical comparison on "Constant product formula
(multivariate)", where the three features (reserve_x, reserve_y, invariant_k)
are independent uniform draws from the same rng stream — NOT collinear. If the
two split functions are genuinely different (as their code suggests), this case
should show real divergence in train_r2/test_r2 and in the prediction arrays —
NOT the float32-epsilon-level agreement seen on the Sharpe Ratio case.

Interpretation
--------------
  DIVERGES  (r2 differs beyond ~1e-3, or lengths/values clearly different)
      -> Confirms the Sharpe Ratio match was a collinearity artifact, not
         fabrication. The two scripts really do apply different splits in
         general. Close the provenance/relabeling theory for this JSON.

  STILL EPSILON-CLOSE (~1e-7 level agreement, matching r2 to many decimals)
      -> Two structurally different split algorithms cannot coincidentally
         agree on independently-sampled data. This would be the actual
         smoking gun — escalate the provenance question and check file
         metadata (mtimes / git history) on both JSONs next.

Usage
-----
    python compare_splits_control.py results_v3c.json results_pca.json

Defaults to results_v3c.json / results_pca.json in the current directory if
no arguments are given (matching the filenames used in earlier steps).
"""

import json
import sys
from pathlib import Path

CASE_NAME = "Constant product formula (multivariate)"

# Machine epsilon for float32 - the specific signature we're checking for.
FLOAT32_EPS = 1.1920928955078125e-07
# Threshold above which we call it "real divergence" rather than
# floating-point noise from independent training runs on identical data.
DIVERGENCE_THRESHOLD = 1e-3


def load(path: str) -> list:
    p = Path(path)
    if not p.exists():
        sys.exit(f"❌ File not found: {p}")
    return json.loads(p.read_text())


def get_case(results: list, name: str = CASE_NAME) -> dict:
    for row in results:
        if row.get("equation_id") == name:
            return row
    available = sorted({row.get("equation_id", "?") for row in results})
    sys.exit(
        f"❌ Case '{name}' not found in this JSON.\n"
        f"   Available equation_id values include:\n   - "
        + "\n   - ".join(available[:20])
        + ("\n   ... (truncated)" if len(available) > 20 else "")
    )


def compare_vectors(a: list, b: list, label: str) -> None:
    print(f"  {label} lengths: {len(a)} vs {len(b)}")
    if len(a) != len(b):
        print(f"  {label}: ⚠️  DIFFERENT LENGTHS — different partitions, cannot diff pointwise")
        return
    diffs = [abs(x - y) for x, y in zip(a, b)]
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    print(f"  {label} max diff:  {max_diff:.6e}")
    print(f"  {label} mean diff: {mean_diff:.6e}")
    if max_diff <= FLOAT32_EPS * 5:
        print(f"  {label}: 🚩 epsilon-close (same signature as the Sharpe Ratio case)")
    elif max_diff <= DIVERGENCE_THRESHOLD:
        print(f"  {label}: 🤔 small but above float32-epsilon — borderline, inspect manually")
    else:
        print(f"  {label}: ✅ clearly diverges — different partitions/training, as expected")


def main():
    v3c_path = sys.argv[1] if len(sys.argv) > 1 else "results_v3c.json"
    pca_path = sys.argv[2] if len(sys.argv) > 2 else "results_pca.json"

    r_v3c = load(v3c_path)
    r_pca = load(pca_path)

    c1 = get_case(r_v3c)["results"]["neural_network"]
    c2 = get_case(r_pca)["results"]["neural_network"]

    print(f"=== Control case: {CASE_NAME!r} ===")
    print(f"    (features reserve_x, reserve_y, invariant_k are independent")
    print(f"     rng.uniform draws — NOT collinear like the Sharpe Ratio case)\n")

    print(f"train_r2: {c1['train_r2']!r}  vs  {c2['train_r2']!r}")
    r2_train_diff = abs(c1["train_r2"] - c2["train_r2"])
    print(f"  -> abs diff: {r2_train_diff:.6e}")

    print(f"test_r2:  {c1['test_r2']!r}  vs  {c2['test_r2']!r}")
    r2_test_diff = abs(c1["test_r2"] - c2["test_r2"])
    print(f"  -> abs diff: {r2_test_diff:.6e}")

    print(f"time_s:   {c1.get('time_s')}  vs  {c2.get('time_s')}\n")

    compare_vectors(c1["y_pred_train"], c2["y_pred_train"], "y_pred_train")
    compare_vectors(c1["y_pred_test"],  c2["y_pred_test"],  "y_pred_test")

    print("\n=== Verdict ===")
    if r2_train_diff <= FLOAT32_EPS * 10 and r2_test_diff <= FLOAT32_EPS * 10:
        print(
            "🚩 STILL epsilon-close on a non-collinear case.\n"
            "   Two structurally different split functions should not agree this\n"
            "   tightly on independently-sampled features. This is NOT explained\n"
            "   by the collinearity argument that resolved the Sharpe Ratio case.\n"
            "   -> Escalate: re-open the provenance/relabeling question, and check\n"
            "      file mtimes / git history on both result JSONs next."
        )
    elif r2_train_diff <= DIVERGENCE_THRESHOLD and r2_test_diff <= DIVERGENCE_THRESHOLD:
        print(
            "🤔 Small divergence, but not clearly beyond noise. Inspect the raw\n"
            "   numbers above manually before concluding either way."
        )
    else:
        print(
            "✅ Clear divergence, as expected for genuinely different splits.\n"
            "   This confirms the Sharpe Ratio match was a collinearity artifact,\n"
            "   not evidence of a shared/relabeled JSON. Close that theory out and\n"
            "   keep the Gate B naming-workaround as the standing finding."
        )


if __name__ == "__main__":
    main()
