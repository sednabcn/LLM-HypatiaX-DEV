"""
test_extrapolation_helpers.py — real pytest unit tests for the deterministic
helper logic inside test_enhanced_defi_extrapolation.py.

Why this file exists
---------------------
`make test` / CI point pytest at hypatiax/experiments/tests/. The only file
there, test_enhanced_defi_extrapolation.py, is a 73-case benchmark *runner*
(needs PySR/Julia/torch/an Anthropic API key and takes hours) — it defines no
`def test_*` functions or `Test*` classes, so pytest silently collects 0
items and `make test` reports success without checking anything.

This file adds fast, deterministic unit tests for the pure helper methods
that don't require the heavy symbolic/NN backends: data-splitting, formula
routing heuristics, and the test-case catalogue itself. It also covers
`_eval_formula_r2`, whose import path was broken (see fix below) and which
always silently returned (nan, False) as a result.
"""

import numpy as np
import pytest

from hypatiax.experiments.tests.test_enhanced_defi_extrapolation import (
    EnhancedExtrapolationTest,
)


@pytest.fixture
def tester():
    return EnhancedExtrapolationTest()


# ─────────────────────────────────────────────────────────────────────────
# create_aggressive_split
# ─────────────────────────────────────────────────────────────────────────

class TestCreateAggressiveSplit:
    def test_high_split_trains_on_low_values(self, tester):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 100, size=(200, 1))
        y = X.flatten() * 2

        X_train, y_train, X_test, y_test = tester.create_aggressive_split(
            X, y, {"split_var_idx": 0, "split_type": "high"}
        )

        assert len(X_train) > 0 and len(X_test) > 0
        assert len(X_train) + len(X_test) <= len(X)
        # Train partition should skew toward lower values than test partition
        assert X_train.flatten().mean() < X_test.flatten().mean()

    def test_low_split_trains_on_high_values(self, tester):
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 100, size=(200, 1))
        y = X.flatten() * 2

        X_train, y_train, X_test, y_test = tester.create_aggressive_split(
            X, y, {"split_var_idx": 0, "split_type": "low"}
        )

        assert len(X_train) > 0 and len(X_test) > 0
        assert X_train.flatten().mean() > X_test.flatten().mean()

    def test_handles_1d_input(self, tester):
        rng = np.random.default_rng(2)
        X = rng.uniform(0, 100, size=200)  # 1-D, not (200, 1)
        y = X * 3

        X_train, y_train, X_test, y_test = tester.create_aggressive_split(
            X, y, {"split_var_idx": 0, "split_type": "high"}
        )
        assert len(X_train) > 0 and len(X_test) > 0

    def test_falls_back_when_partitions_too_small(self, tester):
        # Tiny dataset: percentile split would leave < 20 rows in a partition,
        # so the method must fall back to the index-based 40/60 split.
        X = np.arange(10, dtype=float).reshape(-1, 1)
        y = X.flatten() * 2

        X_train, y_train, X_test, y_test = tester.create_aggressive_split(
            X, y, {"split_var_idx": 0, "split_type": "high"}
        )
        assert len(X_train) == 4          # int(0.4 * 10)
        assert len(X_test) == 6
        np.testing.assert_array_equal(X_train.flatten(), [0, 1, 2, 3])

    def test_multi_column_uses_correct_variable(self, tester):
        rng = np.random.default_rng(3)
        n = 200
        X = np.column_stack([
            rng.uniform(0, 100, n),   # col 0
            rng.uniform(0, 1, n),     # col 1 — irrelevant to the split
        ])
        y = X[:, 0] * 2

        X_train, y_train, X_test, y_test = tester.create_aggressive_split(
            X, y, {"split_var_idx": 0, "split_type": "high"}
        )
        assert X_train[:, 0].mean() < X_test[:, 0].mean()


# ─────────────────────────────────────────────────────────────────────────
# _formula_has_transcendental
# ─────────────────────────────────────────────────────────────────────────

class TestFormulaHasTranscendental:
    @pytest.mark.parametrize("code", [
        "def formula(x0): return np.exp(x0)",
        "def formula(x0): return math.log(x0)",
        "def formula(x0): return np.sqrt(x0)",
        "def formula(x0): return norm.cdf(x0)",
        "def formula(x0): return np.maximum(x0, 0)",
        "def formula(x0): return x0**0.5",
        "def formula(x0): return np.sin(x0) + np.cos(x0)",
    ])
    def test_detects_transcendental_tokens(self, code):
        assert EnhancedExtrapolationTest._formula_has_transcendental(code) is True

    @pytest.mark.parametrize("code", [
        "def formula(x0): return 2 * x0",
        "def formula(x0, x1): return x0 + x1 - 3",
        "def formula(x0): return x0 * x0",
    ])
    def test_plain_polynomial_is_not_transcendental(self, code):
        assert EnhancedExtrapolationTest._formula_has_transcendental(code) is False

    def test_is_case_insensitive(self):
        assert EnhancedExtrapolationTest._formula_has_transcendental(
            "def formula(x0): return NP.EXP(x0)"
        ) is True


# ─────────────────────────────────────────────────────────────────────────
# _distance_llm_weight
# ─────────────────────────────────────────────────────────────────────────

class TestDistanceLlmWeight:
    def test_in_distribution_gives_base_weight(self):
        X_train = np.array([[0.0], [10.0]])
        X_test = np.array([[5.0]])   # fully inside [0, 10]
        w = EnhancedExtrapolationTest._distance_llm_weight(
            X_test, X_train, base_weight=0.3
        )
        assert w == pytest.approx(0.3)

    def test_fully_out_of_distribution_gives_weight_one(self):
        X_train = np.array([[0.0], [10.0]])
        X_test = np.array([[100.0]])  # outside range
        w = EnhancedExtrapolationTest._distance_llm_weight(
            X_test, X_train, base_weight=0.3
        )
        assert w == pytest.approx(1.0)

    def test_partial_out_of_distribution_is_between_bounds(self):
        X_train = np.array([[0.0], [10.0]])
        X_test = np.array([[5.0], [20.0]])  # 1 of 2 points out of range
        w = EnhancedExtrapolationTest._distance_llm_weight(
            X_test, X_train, base_weight=0.3
        )
        assert 0.3 < w < 1.0

    def test_handles_1d_arrays(self):
        X_train = np.array([0.0, 10.0])
        X_test = np.array([50.0])
        w = EnhancedExtrapolationTest._distance_llm_weight(X_test, X_train)
        assert w == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────
# _eval_formula_r2
# regression test for the broken-import bug (see git history / fix notes):
# the method's `from hypatiax.experiments.tests.hybrid_ensemble_system_defi_domain
# import execute_python_code_get_predictions` pointed at a module that doesn't
# exist, so the surrounding bare `except Exception` swallowed the
# ModuleNotFoundError and every call silently returned (nan, False).
# ─────────────────────────────────────────────────────────────────────────

class TestEvalFormulaR2:
    def test_perfect_formula_returns_r2_one(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0])
        r2, success = EnhancedExtrapolationTest._eval_formula_r2(
            "def formula(x0): return 2 * x0", X, y, ["x0"]
        )
        assert success is True
        assert r2 == pytest.approx(1.0, abs=1e-9)

    def test_import_path_resolves(self):
        # Directly exercises the fixed import path so a future regression
        # (e.g. someone reverting the path) fails loudly here instead of
        # silently degrading to (nan, False) inside a bare except.
        from hypatiax.core.generation.hybrid_defi_llm_nn.hybrid_ensemble_system_defi_domain import (
            execute_python_code_get_predictions,
        )
        X = np.array([[1.0], [2.0]])
        preds = execute_python_code_get_predictions(
            "def formula(x0): return x0 + 1", X
        )
        np.testing.assert_allclose(preds, [2.0, 3.0])


# ─────────────────────────────────────────────────────────────────────────
# get_test_cases — catalogue sanity checks
# ─────────────────────────────────────────────────────────────────────────

class TestGetTestCases:
    REQUIRED_KEYS = {"name", "domain", "difficulty", "formula_type",
                      "num_samples", "config"}

    def test_docstring_case_count_matches_reality(self, tester):
        cases = tester.get_test_cases()
        # The module docstring and class docstring both claim 73 cases.
        assert len(cases) == 73, (
            f"get_test_cases() returned {len(cases)} cases; module docs claim 73. "
            "Update the docstrings or the catalogue so they agree."
        )

    def test_every_case_has_required_fields(self, tester):
        for case in tester.get_test_cases():
            missing = self.REQUIRED_KEYS - case.keys()
            assert not missing, f"case {case.get('name')!r} missing keys: {missing}"

    def test_case_names_are_unique(self, tester):
        names = [c["name"] for c in tester.get_test_cases()]
        dupes = {n for n in names if names.count(n) > 1}
        assert not dupes, f"duplicate case names: {dupes}"

    def test_split_config_has_valid_type(self, tester):
        for case in tester.get_test_cases():
            split_type = case["config"].get("split_type")
            assert split_type in ("high", "low"), (
                f"case {case['name']!r} has invalid split_type={split_type!r}"
            )
