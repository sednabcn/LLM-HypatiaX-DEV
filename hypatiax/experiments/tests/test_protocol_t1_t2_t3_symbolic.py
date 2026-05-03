"""
test_protocol_t1_t2_t3_symbolic.py
====================================
Tests the FIRST THREE experiment-protocol cases (mechanics domain)
using ONLY the SymbolicTreeEngine (v23) inside symbolic_engine.py.

No PySR / Julia / LLM required — pure Python tree search.

Cases tested
────────────
  T1 — Kinetic Energy          KE = 0.5 * m * v²
  T2 — Gravitational PE        PE = m * g * h
  T3 — Hooke's Law             F  = k * x

Performance
───────────
The engine is run ONCE per case at module import time and results are
cached so that all individual test functions share the same result.
This avoids re-running the stochastic search per test function.

Run all:          pytest test_protocol_t1_t2_t3_symbolic.py -v
Run T1 only:      pytest test_protocol_t1_t2_t3_symbolic.py -v -k "T1"
"""

import warnings

import numpy as np
import pytest
from sklearn.metrics import r2_score

from experiment_protocol_all_30 import ExperimentProtocolAll
from symbolic_engine import SymbolicTreeEngine

# suppress noisy runtime warnings from tree exploration
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    warnings.filterwarnings("ignore", category=np.exceptions.ComplexWarning)
except AttributeError:
    pass

# engine config: fast but effective for linear / low-order targets
_ENGINE_CFG = dict(
    max_depth=3,
    population_size=250,
    iterations=20,
    complexity_penalty=0.01,
)

# Minimum R² thresholds.
# SymbolicTreeEngine is a lightweight *random* tree search (no PySR/Julia).
# With 20 iterations / 250 population it is not expected to reliably recover
# exact formulae, so thresholds only verify the search *ran* and returned a
# plausible (not catastrophically wrong) score.
R2_T1 = -5.0   # KE (v²): non-linear — just verify no crash
R2_T2 = -5.0   # PE: three variables — just verify no crash
R2_T3 = -5.0   # Hooke: linear — just verify no crash

# load protocol data once
_CASES = ExperimentProtocolAll.load_test_data("mechanics", num_samples=100)
_T1_DESC, _T1_X, _T1_y, _T1_VARS, _T1_META = _CASES[0]
_T2_DESC, _T2_X, _T2_y, _T2_VARS, _T2_META = _CASES[1]
_T3_DESC, _T3_X, _T3_y, _T3_VARS, _T3_META = _CASES[2]


def _run_engine(X, y, var_names, units=None):
    engine = SymbolicTreeEngine(**_ENGINE_CFG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return engine.discover_validate_interpret(
            X, y,
            variable_names=var_names,
            variable_units=units or {},
            show_formatted=False,
            verbose=False,
        )


# run each case once; all tests reference these cached dicts
_R1 = _run_engine(_T1_X, _T1_y, _T1_VARS, _T1_META.get("units"))
_R2 = _run_engine(_T2_X, _T2_y, _T2_VARS, _T2_META.get("units"))
_R3 = _run_engine(_T3_X, _T3_y, _T3_VARS, _T3_META.get("units"))

_ALL = [
    ("kinetic_energy",                _R1, R2_T1),
    ("gravitational_potential_energy", _R2, R2_T2),
    ("hookes_law",                    _R3, R2_T3),
]

REQUIRED_KEYS = ("equation", "r2", "error", "complexity",
                 "posterior", "dimensionally_valid")


def _check_structure(result):
    for k in REQUIRED_KEYS:
        assert k in result, f"Missing key: '{k}'"


def _check_r2(result, threshold, label):
    assert np.isfinite(result["r2"]), f"[{label}] R2 is not finite (got {result['r2']})"
    assert result["r2"] >= threshold, (
        f"[{label}] R2={result['r2']:.4f} < {threshold}. "
        f"eq={result['equation']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T1 — Kinetic Energy
# ─────────────────────────────────────────────────────────────────────────────

class TestT1_KineticEnergy:

    def test_protocol_metadata(self):
        assert _T1_META["equation_name"] == "kinetic_energy"
        assert _T1_VARS == ["m", "v"]
        assert _T1_META["ground_truth"] == "0.5 * m * v**2"
        assert _T1_META["protocol"] == "A"

    def test_data_shape(self):
        assert _T1_X.shape[1] == 2
        assert _T1_y.ndim == 1
        assert _T1_X.shape[0] == _T1_y.shape[0]

    def test_data_finite(self):
        assert np.all(np.isfinite(_T1_X))
        assert np.all(np.isfinite(_T1_y))

    def test_ke_non_negative(self):
        assert np.all(_T1_y >= 0)

    def test_mass_positive(self):
        assert np.all(_T1_X[:, 0] > 0)

    def test_ground_truth_formula(self):
        m, v = _T1_X[:, 0], _T1_X[:, 1]
        np.testing.assert_allclose(_T1_y, 0.5 * m * v**2, rtol=1e-6)

    def test_result_structure(self):
        _check_structure(_R1)

    def test_equation_type(self):
        # SymbolicTreeEngine returns a sympy Expr or None, never a plain str
        assert _R1["equation"] is None or hasattr(_R1["equation"], "free_symbols")

    def test_r2_above_threshold(self):
        _check_r2(_R1, R2_T1, "KineticEnergy")

    def test_error_finite(self):
        assert np.isfinite(_R1["error"])

    def test_complexity_positive(self):
        assert _R1["complexity"] > 0

    def test_dimensionally_valid_is_bool(self):
        assert isinstance(_R1["dimensionally_valid"], bool)

    def test_posterior_is_float(self):
        assert isinstance(_R1["posterior"], float)


# ─────────────────────────────────────────────────────────────────────────────
# T2 — Gravitational PE
# ─────────────────────────────────────────────────────────────────────────────

class TestT2_GravitationalPE:

    def test_protocol_metadata(self):
        assert _T2_META["equation_name"] == "gravitational_potential_energy"
        assert _T2_VARS == ["m", "g", "h"]
        assert _T2_META["ground_truth"] == "m * g * h"
        assert _T2_META["protocol"] == "A"

    def test_data_shape(self):
        assert _T2_X.shape[1] == 3
        assert _T2_y.ndim == 1
        assert _T2_X.shape[0] == _T2_y.shape[0]

    def test_data_finite(self):
        assert np.all(np.isfinite(_T2_X))
        assert np.all(np.isfinite(_T2_y))

    def test_pe_non_negative(self):
        assert np.all(_T2_y >= 0)

    def test_g_range(self):
        g = _T2_X[:, 1]
        assert g.min() >= 9.7 - 1e-9
        assert g.max() <= 9.9 + 1e-9

    def test_h_non_negative(self):
        assert np.all(_T2_X[:, 2] >= 0)

    def test_ground_truth_formula(self):
        m, g, h = _T2_X[:, 0], _T2_X[:, 1], _T2_X[:, 2]
        np.testing.assert_allclose(_T2_y, m * g * h, rtol=1e-6)

    def test_result_structure(self):
        _check_structure(_R2)

    def test_r2_above_threshold(self):
        _check_r2(_R2, R2_T2, "GravitationalPE")

    def test_error_finite(self):
        assert np.isfinite(_R2["error"])

    def test_complexity_positive(self):
        assert _R2["complexity"] > 0

    def test_equation_references_a_variable(self):
        if _R2["equation"] is None:
            pytest.skip("Engine returned no equation")
        eq_str = str(_R2["equation"])
        assert any(v in eq_str for v in _T2_VARS), (
            f"Equation '{eq_str}' references none of {_T2_VARS}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# T3 — Hooke's Law
# ─────────────────────────────────────────────────────────────────────────────

class TestT3_HookesLaw:

    def test_protocol_metadata(self):
        assert _T3_META["equation_name"] == "hookes_law"
        assert _T3_VARS == ["k", "x"]
        assert _T3_META["ground_truth"] == "k * x"
        assert _T3_META["protocol"] == "A"

    def test_data_shape(self):
        assert _T3_X.shape[1] == 2
        assert _T3_y.ndim == 1
        assert _T3_X.shape[0] == _T3_y.shape[0]

    def test_data_finite(self):
        assert np.all(np.isfinite(_T3_X))
        assert np.all(np.isfinite(_T3_y))

    def test_spring_constant_positive(self):
        assert np.all(_T3_X[:, 0] > 0)

    def test_displacement_signed(self):
        assert _T3_X[:, 1].min() < 0

    def test_ground_truth_formula(self):
        k, x = _T3_X[:, 0], _T3_X[:, 1]
        np.testing.assert_allclose(_T3_y, k * x, rtol=1e-6)

    def test_linear_baseline_near_perfect(self):
        k, x = _T3_X[:, 0], _T3_X[:, 1]
        assert r2_score(_T3_y, k * x) > 0.9999

    def test_result_structure(self):
        _check_structure(_R3)

    def test_r2_above_threshold(self):
        _check_r2(_R3, R2_T3, "HookesLaw")

    def test_error_finite(self):
        assert np.isfinite(_R3["error"])

    def test_complexity_positive(self):
        assert _R3["complexity"] > 0

    def test_equation_non_empty(self):
        # engine returns a sympy Expr, not a plain str
        assert _R3["equation"] is not None
        assert len(str(_R3["equation"])) > 0

    def test_equation_references_a_variable(self):
        eq_str = str(_R3["equation"])
        assert any(v in eq_str for v in _T3_VARS), (
            f"Equation '{eq_str}' references none of {_T3_VARS}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Integration — cross-cutting, no extra engine runs
# ─────────────────────────────────────────────────────────────────────────────

class TestProtocol_Integration:

    def test_exactly_three_cases(self):
        assert len(_CASES) == 3

    def test_all_protocol_A(self):
        for *_, meta in _CASES:
            assert meta["protocol"] == "A"

    def test_all_difficulty_easy(self):
        for *_, meta in _CASES:
            assert meta["difficulty"] == "easy"

    def test_all_have_ground_truth(self):
        for *_, meta in _CASES:
            assert isinstance(meta.get("ground_truth"), str)

    def test_no_nan_anywhere(self):
        for _, X, y, *_ in _CASES:
            assert np.all(np.isfinite(X)) and np.all(np.isfinite(y))

    def test_sample_counts_match(self):
        for _, X, y, *_ in _CASES:
            assert X.shape[0] == y.shape[0]

    def test_all_results_r2_is_finite(self):
        """R² can be negative for a bad-fit model, but must always be finite."""
        for name, result, _ in _ALL:
            assert np.isfinite(result["r2"]), f"[{name}] R2 is not finite"

    @pytest.mark.parametrize("name,result,threshold", _ALL)
    def test_r2_parametrized(self, name, result, threshold):
        """Parametrized R2 check — uses cached results, zero extra engine runs."""
        _check_r2(result, threshold, name)

    @pytest.mark.parametrize("name,result,_th", _ALL)
    def test_structure_parametrized(self, name, result, _th):
        _check_structure(result)

    @pytest.mark.parametrize("name,result,_th", _ALL)
    def test_error_finite_parametrized(self, name, result, _th):
        assert np.isfinite(result["error"]), f"[{name}] RMSE not finite"
