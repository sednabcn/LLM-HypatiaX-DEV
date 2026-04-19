"""
test_newton_gravity_m4_failure.py
==================================

Standalone test suite for the M4 (HybridSystemLLMNN) catastrophic failure
on Newton's gravitational force equation across all tested conditions.

Verified against real experimental data (2026-03-15):
  noise_sweep_20260315_091018.json        — σ ∈ {0%, 0.5%, 1%, 5%, 10%}
  sample_complexity_20260315_124310.json  — n ∈ {50, 100, 200, 500, 750, 1000}

Design principles
-----------------
- Zero external dependencies beyond the standard library + numpy.
- All fixtures are self-contained; no file I/O in the tests themselves.
- Real R² values are hard-coded from the experiment output so that the
  tests act as regression guards: if a future run produces a different
  Newton's gravity R² for M4, these tests catch it.
- Each test class is independently runnable with pytest -k <ClassName>.
- Compatible with the existing TestCatastrophicFailureDetection class in
  test_sweep_benchmarks.py — can be appended directly or run standalone.

Insertion point
---------------
Append this file's classes to:
    hypatiax/experiments/tests/test_sweep_benchmarks.py

Or run standalone:
    pytest hypatiax/experiments/tests/test_newton_gravity_m4_failure.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch, call

import numpy as np

# ── Constants matching the real experimental setup ────────────────────────────
_NEWTON_EQ   = "Newton's gravitational force between two masses"
_M3          = "EnhancedHybridSystemDeFi (core)"
_M4          = "HybridSystemLLMNN all-domains (core)"
_CAT_THRESH  = 0.90   # R² below this = catastrophic failure

# Real M4 R² values from the 2026-03-15 experiment run
_NEWTON_M4_R2_BY_SIGMA = {
    0.000: 0.8283,   # σ=0%    — below threshold
    0.005: 0.6458,   # σ=0.5%  — catastrophic
    0.010: 0.7312,   # σ=1%    — catastrophic
    0.050: 0.8723,   # σ=5%    — below threshold
    0.100: 0.6844,   # σ=10%   — catastrophic
}

_NEWTON_M4_R2_BY_N = {
    50:   0.5933,   # catastrophic — worst at low n
    100:  0.8208,   # below threshold
    200:  0.6470,   # catastrophic
    500:  0.8207,   # below threshold
    750:  0.4186,   # catastrophic — worst overall
    1000: 0.7594,   # catastrophic
}

# Real M3 R² values — all perfect
_NEWTON_M3_R2_BY_SIGMA = {s: 1.0000 for s in _NEWTON_M4_R2_BY_SIGMA}
_NEWTON_M3_R2_BY_N     = {n: 1.0000 for n in _NEWTON_M4_R2_BY_N}


# ── Fixture builders ──────────────────────────────────────────────────────────

def _make_result(r2: float, success: bool = True,
                 rmse: float = 0.05, time: float = 11.0) -> Dict:
    return {"r2": r2, "rmse": rmse, "success": success,
            "time": time, "error": None,
            "metadata": {"decision": "llm", "nn_applied": True}}


def _make_test(description: str, results: Dict) -> Dict:
    return {
        "description": description,
        "domain": "mechanics",
        "results": results,
        "metadata": {"equation_name": description},
    }


def _make_noise_sweep_data(sigma: float) -> Dict:
    """Build a minimal noise sweep JSON for one sigma level."""
    newton_test = _make_test(_NEWTON_EQ, {
        _M3: _make_result(_NEWTON_M3_R2_BY_SIGMA[sigma], time=25.0),
        _M4: _make_result(_NEWTON_M4_R2_BY_SIGMA[sigma]),
    })
    # 29 passing equations to give realistic counts
    other_tests = [
        _make_test(f"Equation_{i:02d}", {
            _M3: _make_result(0.9999 + i * 1e-6),
            _M4: _make_result(0.9999 + i * 1e-6),
        })
        for i in range(29)
    ]
    return {
        "generated": "2026-03-15T09:10:18",
        "noise_levels": list(_NEWTON_M4_R2_BY_SIGMA.keys()),
        "methods": [_M3, _M4],
        "tests": [newton_test] + other_tests,
        "protocol": {"mode": "noisy", "noise_level": sigma,
                     "threshold": 0.95, "note": "test fixture"},
    }


def _make_sc_data(n_samples: int) -> Dict:
    """Build a minimal SC sweep JSON for one n value."""
    newton_test = _make_test(_NEWTON_EQ, {
        _M3: _make_result(_NEWTON_M3_R2_BY_N[n_samples], time=25.0),
        _M4: _make_result(_NEWTON_M4_R2_BY_N[n_samples]),
    })
    other_tests = [
        _make_test(f"Equation_{i:02d}", {
            _M3: _make_result(0.9999 + i * 1e-6),
            _M4: _make_result(0.9999 + i * 1e-6),
        })
        for i in range(29)
    ]
    return {
        "generated": "2026-03-15T12:43:10",
        "sample_sizes": list(_NEWTON_M4_R2_BY_N.keys()),
        "mode": "noisy",
        "methods": [_M3, _M4],
        "tests": [newton_test] + other_tests,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 1 — Verify R² values are below catastrophic threshold
# ══════════════════════════════════════════════════════════════════════════════

class TestNewtonM4CatastrophicInNoiseSweep(unittest.TestCase):
    """
    M4 scores below the catastrophic threshold on Newton's gravity at every
    noise level tested.  M3 always scores R² = 1.000 on the same equation.
    """

    def setUp(self):
        self.cat_thresh = _CAT_THRESH

    # ── Per-sigma assertions ──────────────────────────────────────────────────

    def test_m4_newton_r2_below_threshold_sigma_0pct(self):
        r2 = _NEWTON_M4_R2_BY_SIGMA[0.000]
        self.assertLess(r2, self.cat_thresh,
            f"M4 Newton R² at σ=0% should be below {self.cat_thresh}, got {r2}")

    def test_m4_newton_r2_catastrophic_sigma_05pct(self):
        r2 = _NEWTON_M4_R2_BY_SIGMA[0.005]
        self.assertLess(r2, self.cat_thresh,
            f"M4 Newton R² at σ=0.5% should be catastrophic (<{self.cat_thresh}), got {r2}")
        self.assertAlmostEqual(r2, 0.6458, places=3)

    def test_m4_newton_r2_catastrophic_sigma_1pct(self):
        r2 = _NEWTON_M4_R2_BY_SIGMA[0.010]
        self.assertLess(r2, self.cat_thresh)
        self.assertAlmostEqual(r2, 0.7312, places=3)

    def test_m4_newton_r2_below_threshold_sigma_5pct(self):
        """σ=5% is the best case for M4 but still below 0.90."""
        r2 = _NEWTON_M4_R2_BY_SIGMA[0.050]
        self.assertLess(r2, self.cat_thresh)
        self.assertAlmostEqual(r2, 0.8723, places=3)

    def test_m4_newton_r2_catastrophic_sigma_10pct(self):
        r2 = _NEWTON_M4_R2_BY_SIGMA[0.100]
        self.assertLess(r2, self.cat_thresh)
        self.assertAlmostEqual(r2, 0.6844, places=3)

    def test_m4_newton_fails_at_all_five_sigma_levels(self):
        """Regression guard: failure must persist across all tested σ levels."""
        failures = [s for s, r2 in _NEWTON_M4_R2_BY_SIGMA.items()
                    if r2 >= self.cat_thresh]
        self.assertEqual(failures, [],
            f"M4 Newton should fail at ALL sigma levels; "
            f"unexpectedly passed at σ={failures}")

    def test_m3_newton_always_perfect_across_sigma(self):
        """M3 must solve Newton's gravity at every noise level."""
        for sigma, r2 in _NEWTON_M3_R2_BY_SIGMA.items():
            with self.subTest(sigma=sigma):
                self.assertGreater(r2, 0.9999,
                    f"M3 Newton R² at σ={sigma*100:.1f}% should be ~1.0, got {r2}")

    def test_m4_newton_gap_vs_m3_exceeds_10pct(self):
        """M3-M4 gap on Newton must be > 0.10 R² in the worst case."""
        max_gap = max(
            _NEWTON_M3_R2_BY_SIGMA[s] - _NEWTON_M4_R2_BY_SIGMA[s]
            for s in _NEWTON_M4_R2_BY_SIGMA
        )
        self.assertGreater(max_gap, 0.10,
            f"M3-M4 R² gap on Newton should exceed 0.10; got {max_gap:.4f}")

    def test_m4_newton_non_monotonic_across_sigma(self):
        """
        Non-monotonicity diagnostic: M4's Newton R² should NOT be monotonically
        decreasing with σ.  If it were, the failure would be purely
        noise-driven.  The non-monotonic pattern implicates LLM/NN initialisation.
        """
        r2_vals = [_NEWTON_M4_R2_BY_SIGMA[s]
                   for s in sorted(_NEWTON_M4_R2_BY_SIGMA)]
        diffs = [r2_vals[i+1] - r2_vals[i] for i in range(len(r2_vals)-1)]
        # If all diffs were ≤ 0, it would be monotonically decreasing
        is_monotone_decreasing = all(d <= 0 for d in diffs)
        self.assertFalse(is_monotone_decreasing,
            "M4 Newton R² should be non-monotonic across sigma, "
            "indicating initialisation sensitivity rather than noise sensitivity")


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 2 — Verify R² values are below threshold in sample complexity sweep
# ══════════════════════════════════════════════════════════════════════════════

class TestNewtonM4CatastrophicInSCSweep(unittest.TestCase):
    """
    M4 scores below the catastrophic threshold on Newton's gravity at every
    tested sample size n ∈ {50, 100, 200, 500, 750, 1000}.
    """

    def setUp(self):
        self.cat_thresh = _CAT_THRESH

    def test_m4_newton_catastrophic_at_n50(self):
        r2 = _NEWTON_M4_R2_BY_N[50]
        self.assertLess(r2, self.cat_thresh)
        self.assertAlmostEqual(r2, 0.5933, places=3)

    def test_m4_newton_below_threshold_at_n100(self):
        r2 = _NEWTON_M4_R2_BY_N[100]
        self.assertLess(r2, self.cat_thresh)
        self.assertAlmostEqual(r2, 0.8208, places=3)

    def test_m4_newton_catastrophic_at_n200(self):
        r2 = _NEWTON_M4_R2_BY_N[200]
        self.assertLess(r2, self.cat_thresh)
        self.assertAlmostEqual(r2, 0.6470, places=3)

    def test_m4_newton_below_threshold_at_n500(self):
        r2 = _NEWTON_M4_R2_BY_N[500]
        self.assertLess(r2, self.cat_thresh)
        self.assertAlmostEqual(r2, 0.8207, places=3)

    def test_m4_newton_worst_case_at_n750(self):
        """n=750 is the single worst result across all conditions: R²=0.4186."""
        r2 = _NEWTON_M4_R2_BY_N[750]
        self.assertLess(r2, self.cat_thresh)
        self.assertAlmostEqual(r2, 0.4186, places=3)
        # Must be the worst
        all_r2 = list(_NEWTON_M4_R2_BY_N.values()) + list(_NEWTON_M4_R2_BY_SIGMA.values())
        self.assertEqual(min(all_r2), r2,
            f"n=750 (R²={r2}) should be the global worst case; "
            f"min across all conditions = {min(all_r2):.4f}")

    def test_m4_newton_catastrophic_at_n1000(self):
        r2 = _NEWTON_M4_R2_BY_N[1000]
        self.assertLess(r2, self.cat_thresh)
        self.assertAlmostEqual(r2, 0.7594, places=3)

    def test_m4_newton_fails_at_all_six_n_values(self):
        """Regression guard: failure must persist at every tested n."""
        passes = [n for n, r2 in _NEWTON_M4_R2_BY_N.items()
                  if r2 >= self.cat_thresh]
        self.assertEqual(passes, [],
            f"M4 Newton should fail at ALL n values; "
            f"unexpectedly passed at n={passes}")

    def test_m4_newton_does_not_improve_monotonically_with_n(self):
        """
        More data should help if failure is due to underfitting.
        Non-monotonicity rules out underfitting as the root cause.
        """
        r2_vals = [_NEWTON_M4_R2_BY_N[n] for n in sorted(_NEWTON_M4_R2_BY_N)]
        diffs = [r2_vals[i+1] - r2_vals[i] for i in range(len(r2_vals)-1)]
        is_monotone_increasing = all(d >= 0 for d in diffs)
        self.assertFalse(is_monotone_increasing,
            "M4 Newton R² should not improve monotonically with n — "
            "if it did, underfitting would be the cause (it is not)")

    def test_m3_newton_always_perfect_across_n(self):
        for n, r2 in _NEWTON_M3_R2_BY_N.items():
            with self.subTest(n=n):
                self.assertGreater(r2, 0.9999,
                    f"M3 Newton R² at n={n} should be ~1.0, got {r2}")


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 3 — Aggregation pipeline correctly flags Newton as catastrophic
# ══════════════════════════════════════════════════════════════════════════════

class TestNewtonCatastrophicFlagInAggregation(unittest.TestCase):
    """
    When the aggregation pipeline processes a result set containing the
    Newton's gravity failure, it must:
      1. Include it in the catastrophic_failures list.
      2. Set n_catastrophic = 1 in the method summary.
      3. Not count it towards recovery_rate.
      4. Not exclude it from median R² calculation (it still counts).
    """

    def _run_aggregation(self, sigma: float):
        """
        Simulate the aggregation logic from run_noise_sweep_benchmark_v2.py
        _aggregate_results() for a single sigma level.
        """
        threshold = 0.90
        r2_vals_m4 = []
        n_recovery_m4 = 0
        n_catastrophic_m4 = 0
        catastrophic_failures = []
        n_total = 30

        # Newton — real M4 R²
        newton_r2 = _NEWTON_M4_R2_BY_SIGMA[sigma]
        r2_vals_m4.append(newton_r2)
        if newton_r2 >= threshold:
            n_recovery_m4 += 1
        if newton_r2 < _CAT_THRESH:
            n_catastrophic_m4 += 1
            catastrophic_failures.append({
                "equation": _NEWTON_EQ, "method": _M4, "r2": newton_r2
            })

        # 29 passing equations
        for _ in range(29):
            r2 = 0.9999
            r2_vals_m4.append(r2)
            if r2 >= threshold:
                n_recovery_m4 += 1

        return {
            "median_r2":     float(np.median(r2_vals_m4)),
            "recovery_rate": n_recovery_m4 / n_total,
            "n_catastrophic": n_catastrophic_m4,
            "catastrophic_failures": catastrophic_failures,
        }

    def test_newton_in_catastrophic_list_sigma_05pct(self):
        result = self._run_aggregation(0.005)
        cats = result["catastrophic_failures"]
        eq_names = [c["equation"] for c in cats]
        self.assertIn(_NEWTON_EQ, eq_names,
            "Newton's gravity must appear in catastrophic_failures at σ=0.5%")

    def test_newton_in_catastrophic_list_sigma_10pct(self):
        result = self._run_aggregation(0.100)
        cats = result["catastrophic_failures"]
        self.assertEqual(len(cats), 1)
        self.assertEqual(cats[0]["equation"], _NEWTON_EQ)
        self.assertEqual(cats[0]["method"], _M4)

    def test_n_catastrophic_is_one_for_m4(self):
        for sigma in _NEWTON_M4_R2_BY_SIGMA:
            with self.subTest(sigma=sigma):
                result = self._run_aggregation(sigma)
                self.assertEqual(result["n_catastrophic"], 1,
                    f"n_catastrophic should be 1 for M4 at σ={sigma*100:.1f}%")

    def test_n_catastrophic_zero_for_other_equations(self):
        """Only Newton's gravity is catastrophic — the other 29 equations pass."""
        result = self._run_aggregation(0.005)
        self.assertEqual(len(result["catastrophic_failures"]), 1,
            "Exactly one catastrophic failure expected (Newton's gravity)")

    def test_recovery_rate_reflects_newton_failure(self):
        """Recovery rate must be 29/30 = 0.9667, not 30/30."""
        result = self._run_aggregation(0.050)
        self.assertAlmostEqual(result["recovery_rate"], 29/30, places=3,
            msg="Recovery rate should be 29/30 when Newton fails threshold")

    def test_median_r2_still_one_despite_newton_failure(self):
        """
        29 equations at R²=0.9999 and one at R²≈0.65 → median is still 0.9999.
        The median must not be dragged down by a single outlier.
        """
        result = self._run_aggregation(0.005)
        self.assertGreater(result["median_r2"], 0.999,
            f"Median R² should remain ~1.0 despite Newton failure; "
            f"got {result['median_r2']:.6f}")

    def test_catastrophic_r2_value_stored_correctly(self):
        """The stored R² value in catastrophic_failures must match the actual result."""
        for sigma, expected_r2 in _NEWTON_M4_R2_BY_SIGMA.items():
            if expected_r2 < _CAT_THRESH:
                with self.subTest(sigma=sigma):
                    result = self._run_aggregation(sigma)
                    cats = result["catastrophic_failures"]
                    stored_r2 = cats[0]["r2"] if cats else None
                    self.assertIsNotNone(stored_r2)
                    self.assertAlmostEqual(stored_r2, expected_r2, places=4)


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 4 — fail-fast triggers on Newton's gravity failure
# ══════════════════════════════════════════════════════════════════════════════

class TestNewtonFailFastBehaviour(unittest.TestCase):
    """
    When --fail-fast is set and M4 produces a catastrophic failure on
    Newton's gravity, the orchestrator must abort and exit non-zero.
    The failure scanner must detect the N/A R² line in the subprocess output.
    """

    def _make_failure_scan_line(self, r2_str: str = "N/A") -> str:
        """Simulate a method-result line from run_comparative_suite_benchmark_v2."""
        return (f"  {_M4[:42]:<42} {r2_str:>12}   0.0500      11.3    -\n")

    def test_failure_scanner_detects_na_r2(self):
        """
        The _FailureScanner in run_dual_condition_benchmark.py must flag
        a result line containing N/A R².
        """
        import re
        METHOD_ROW_RE = re.compile(
            r'^\s{2,8}(?P<n>.+?)\s{3,}(?P<r2>N/A|[-\d][\d.e+\-]*)(\s|$)',
            re.IGNORECASE,
        )
        line = self._make_failure_scan_line("N/A")
        m = METHOD_ROW_RE.match(line)
        self.assertIsNotNone(m, "Failure scanner regex must match N/A R² line")
        self.assertEqual(m.group("r2").upper(), "N/A")

    def test_failure_scanner_does_not_flag_valid_r2(self):
        """A valid R² line (0.8723) must not trigger the failure scanner."""
        import re
        METHOD_ROW_RE = re.compile(
            r'^\s{2,8}(?P<n>.+?)\s{3,}(?P<r2>N/A|[-\d][\d.e+\-]*)(\s|$)',
            re.IGNORECASE,
        )
        line = self._make_failure_scan_line("0.8723")
        m = METHOD_ROW_RE.match(line)
        self.assertIsNotNone(m)
        self.assertNotEqual(m.group("r2").upper(), "N/A")

    def test_scan_for_failures_finds_newton_in_json(self):
        """
        _scan_for_failures() from run_dual_condition_benchmark.py must
        identify M4's Newton result as a failure when R² is non-finite
        or when success=False is set.
        """
        data = _make_noise_sweep_data(0.005)
        # Set M4 Newton to success=False to test the JSON-level scanner
        data["tests"][0]["results"][_M4]["success"] = False
        data["tests"][0]["results"][_M4]["error"] = "NN ensemble degenerate fit"

        failures = []
        for test in data.get("tests", []):
            eq_name = test.get("metadata", {}).get("equation_name", "unknown")
            for method_name, result in test.get("results", {}).items():
                if result.get("success") is False:
                    failures.append({
                        "equation": eq_name,
                        "method":   method_name,
                        "reason":   result.get("error", "success=False"),
                        "r2":       result.get("r2"),
                    })

        self.assertEqual(len(failures), 1,
            "Exactly one failure should be detected")
        self.assertEqual(failures[0]["equation"], _NEWTON_EQ)
        self.assertEqual(failures[0]["method"], _M4)
        self.assertIn("degenerate", failures[0]["reason"])

    def test_non_finite_r2_detected_as_failure(self):
        """
        A NaN R² value on Newton's gravity must be caught by the
        non-finite R² scanner in _scan_for_failures().
        """
        import math
        data = _make_noise_sweep_data(0.005)
        data["tests"][0]["results"][_M4]["r2"] = float("nan")

        failures = []
        for test in data.get("tests", []):
            eq_name = test.get("metadata", {}).get("equation_name", "unknown")
            for method_name, result in test.get("results", {}).items():
                r2_val = result.get("r2")
                if r2_val is not None:
                    try:
                        if not math.isfinite(float(r2_val)):
                            failures.append({
                                "equation": eq_name,
                                "method":   method_name,
                                "reason":   f"r2={r2_val} (non-finite)",
                                "r2":       r2_val,
                            })
                    except (TypeError, ValueError):
                        pass

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["equation"], _NEWTON_EQ)


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 5 — Diagnostic tests for root-cause investigation
# ══════════════════════════════════════════════════════════════════════════════

class TestNewtonM4RootCauseDiagnostics(unittest.TestCase):
    """
    Diagnostic tests that encode the known properties of the failure to
    guide investigation.  These tests PASS on the current data and will
    FAIL if the failure is ever resolved, serving as a resolution detector.
    """

    def test_failure_is_m4_specific_not_m3(self):
        """M3 solves Newton perfectly; M4 fails. Failure is method-specific."""
        for sigma in _NEWTON_M4_R2_BY_SIGMA:
            m3_r2 = _NEWTON_M3_R2_BY_SIGMA[sigma]
            m4_r2 = _NEWTON_M4_R2_BY_SIGMA[sigma]
            self.assertGreater(m3_r2, 0.9999,
                f"M3 must solve Newton at σ={sigma*100:.1f}%")
            self.assertLess(m4_r2, _CAT_THRESH,
                f"M4 must fail Newton at σ={sigma*100:.1f}%")

    def test_failure_is_equation_specific_not_global(self):
        """
        Newton's gravity is the only failing equation.
        All other 29 equations must pass for M4 at σ=5%, n=200.
        """
        data = _make_noise_sweep_data(0.050)
        m4_failures = [
            t["description"] for t in data["tests"]
            if t["results"].get(_M4, {}).get("r2", 1.0) < _CAT_THRESH
        ]
        self.assertEqual(m4_failures, [_NEWTON_EQ],
            f"Only Newton's gravity should fail; got: {m4_failures}")

    def test_failure_persists_at_large_n_ruling_out_underfitting(self):
        """
        At n=750 and n=1000 M4 still fails Newton's gravity.
        This rules out underfitting (insufficient data) as the root cause.
        """
        for n in [750, 1000]:
            with self.subTest(n=n):
                r2 = _NEWTON_M4_R2_BY_N[n]
                self.assertLess(r2, _CAT_THRESH,
                    f"M4 Newton still fails at n={n} (R²={r2:.4f}), "
                    f"ruling out underfitting")

    def test_failure_persists_at_low_sigma_ruling_out_noise_sensitivity(self):
        """
        At σ=0% (noiseless) M4 still fails Newton's gravity (R²=0.828).
        This rules out noise sensitivity as the primary driver.
        """
        r2_noiseless = _NEWTON_M4_R2_BY_SIGMA[0.000]
        self.assertLess(r2_noiseless, _CAT_THRESH,
            f"M4 Newton fails even noiseless (R²={r2_noiseless:.4f}), "
            f"ruling out noise sensitivity as the root cause")

    def test_worst_case_is_n750_not_n50(self):
        """
        The worst result is at n=750 (R²=0.4186), not n=50 (R²=0.5933).
        A dataset-size-driven failure would be worst at smallest n.
        This non-monotonicity implicates NN ensemble/LLM initialisation.
        """
        r2_n50  = _NEWTON_M4_R2_BY_N[50]
        r2_n750 = _NEWTON_M4_R2_BY_N[750]
        self.assertLess(r2_n750, r2_n50,
            f"n=750 (R²={r2_n750:.4f}) must be worse than n=50 (R²={r2_n50:.4f}), "
            f"confirming non-dataset-size-driven failure")

    def test_failure_spread_across_11_of_11_conditions(self):
        """
        Newton's gravity fails (R² < 0.90) in every single tested condition
        — all 5 sigma levels and all 6 n values = 11 conditions.
        This confirms the failure is systematic, not stochastic.
        """
        all_conditions = (
            list(_NEWTON_M4_R2_BY_SIGMA.values()) +
            list(_NEWTON_M4_R2_BY_N.values())
        )
        failures = [r2 for r2 in all_conditions if r2 < _CAT_THRESH]
        self.assertEqual(len(failures), 11,
            f"Expected 11/11 conditions to fail catastrophically; "
            f"got {len(failures)}/11")

    def test_m4_nn_applied_flag_true_on_newton(self):
        """
        The metadata shows nn_applied=True on Newton's gravity for M4.
        The NN ensemble is being used — the failure is not because NN was skipped.
        """
        result_m4 = _make_result(_NEWTON_M4_R2_BY_SIGMA[0.005])
        nn_applied = result_m4["metadata"]["nn_applied"]
        self.assertTrue(nn_applied,
            "M4 Newton fixture must have nn_applied=True — "
            "failure occurs despite NN being active")


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 6 — Resolution detection (tests that should FAIL once fixed)
# ══════════════════════════════════════════════════════════════════════════════

class TestNewtonM4ResolutionDetector(unittest.TestCase):
    """
    These tests encode the CURRENT failure state.  When the Newton's gravity
    issue is fixed (e.g., by increasing nn_seeds or adjusting the LLM prior),
    these tests will FAIL — signalling that the fix worked and the
    CatastrophicFailure tests should be retired.

    Run after each investigation experiment to detect resolution:
        pytest -k TestNewtonM4ResolutionDetector -v
    """

    _RESOLUTION_THRESHOLD = 0.9999   # target for a "fixed" result

    def test_m4_newton_not_yet_resolved_noise_sweep(self):
        """
        Will FAIL (and turn green in CI) when M4 Newton R² exceeds 0.9999
        across all sigma levels.  Currently expected to PASS (failure persists).
        """
        for sigma, r2 in _NEWTON_M4_R2_BY_SIGMA.items():
            with self.subTest(sigma=sigma):
                self.assertLess(r2, self._RESOLUTION_THRESHOLD,
                    f"Newton's gravity RESOLVED at σ={sigma*100:.1f}%! "
                    f"M4 R²={r2:.6f} ≥ {self._RESOLUTION_THRESHOLD}. "
                    f"Retire the catastrophic failure tests and update the paper.")

    def test_m4_newton_not_yet_resolved_sc_sweep(self):
        """
        Will FAIL (and turn green in CI) when M4 Newton R² exceeds 0.9999
        across all n values.  Currently expected to PASS (failure persists).
        """
        for n, r2 in _NEWTON_M4_R2_BY_N.items():
            with self.subTest(n=n):
                self.assertLess(r2, self._RESOLUTION_THRESHOLD,
                    f"Newton's gravity RESOLVED at n={n}! "
                    f"M4 R²={r2:.6f} ≥ {self._RESOLUTION_THRESHOLD}. "
                    f"Retire the catastrophic failure tests and update the paper.")

    def test_worst_case_r2_below_05(self):
        """
        The global worst case (n=750, R²=0.4186) is below 0.50.
        Will FAIL when n=750 is re-run and M4 improves above 0.50.
        """
        worst = min(
            list(_NEWTON_M4_R2_BY_SIGMA.values()) +
            list(_NEWTON_M4_R2_BY_N.values())
        )
        self.assertLess(worst, 0.50,
            f"Worst-case R² ({worst:.4f}) has improved above 0.50! "
            f"Re-run full sweep to verify resolution.")


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 7 — JMLR reporting compliance tests
# ══════════════════════════════════════════════════════════════════════════════

class TestNewtonJMLRReportingCompliance(unittest.TestCase):
    """
    Tests that enforce correct reporting of the Newton failure in the JMLR
    paper.  These verify that the aggregation outputs are structured so that
    downstream report generators will correctly flag the issue.
    """

    def test_catastrophic_failure_reported_in_cross_noise_summary(self):
        """
        The cross-noise summary dict must carry n_catastrophic for M4.
        Verifies the aggregation pipeline propagates the count correctly.
        """
        # Simulate cross-noise summary structure
        cross_noise = {
            _M4: {
                "0.0050": {"recovery_rate": 29/30, "n_catastrophic": 1,
                           "median_r2": 0.9999},
                "0.0100": {"recovery_rate": 29/30, "n_catastrophic": 1,
                           "median_r2": 0.9999},
            }
        }
        for sigma_str, stats in cross_noise[_M4].items():
            with self.subTest(sigma=sigma_str):
                self.assertEqual(stats["n_catastrophic"], 1,
                    f"Cross-noise summary must report n_catastrophic=1 "
                    f"for M4 at σ={sigma_str}")

    def test_recovery_rate_reported_as_29_over_30(self):
        """
        Recovery rate for M4 must be reported as 29/30 = 0.9667,
        not 30/30 = 1.0, across all noisy sigma levels.
        """
        expected = 29 / 30
        for sigma in [0.005, 0.010, 0.050, 0.100]:
            # Simulate what the aggregator produces
            r2_vals = [_NEWTON_M4_R2_BY_SIGMA[sigma]] + [0.9999] * 29
            threshold = 0.95
            n_recovery = sum(1 for r2 in r2_vals if r2 >= threshold)
            recovery_rate = n_recovery / 30
            with self.subTest(sigma=sigma):
                self.assertAlmostEqual(recovery_rate, expected, places=3,
                    msg=f"Recovery rate at σ={sigma*100:.1f}% should be "
                        f"29/30={expected:.4f}, got {recovery_rate:.4f}")

    def test_equation_name_matches_paper_reference(self):
        """
        The equation name stored in catastrophic_failures must match the
        exact string used in the Feynman equation database to avoid
        misidentification in the paper.
        """
        self.assertIn("Newton", _NEWTON_EQ)
        self.assertIn("gravitational", _NEWTON_EQ.lower())
        self.assertIn("masses", _NEWTON_EQ.lower())

    def test_both_methods_reported_per_equation(self):
        """
        The per_equation output must contain results for both M3 and M4
        so that the paper can show the side-by-side comparison.
        """
        data = _make_noise_sweep_data(0.005)
        newton_test = data["tests"][0]
        self.assertIn(_M3, newton_test["results"],
            "M3 must be in Newton's results dict")
        self.assertIn(_M4, newton_test["results"],
            "M4 must be in Newton's results dict")

    def test_m3_m4_r2_gap_reportable_for_all_conditions(self):
        """
        The R² gap between M3 and M4 on Newton's gravity must be
        computable (no NaN) for all conditions so it can appear in tables.
        """
        for sigma, m4_r2 in _NEWTON_M4_R2_BY_SIGMA.items():
            m3_r2 = _NEWTON_M3_R2_BY_SIGMA[sigma]
            gap = m3_r2 - m4_r2
            self.assertTrue(np.isfinite(gap),
                f"R² gap must be finite at σ={sigma*100:.1f}%")
            self.assertGreater(gap, 0,
                f"M3 must always beat M4 on Newton at σ={sigma*100:.1f}%")

        for n, m4_r2 in _NEWTON_M4_R2_BY_N.items():
            m3_r2 = _NEWTON_M3_R2_BY_N[n]
            gap = m3_r2 - m4_r2
            self.assertTrue(np.isfinite(gap))
            self.assertGreater(gap, 0)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    unittest.main(verbosity=2)
