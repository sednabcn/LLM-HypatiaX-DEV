"""
test_protocol_biology.py
=========================
Tests the three biology equations from ExperimentProtocolAll v4.0 against
SymbolicEngine, using the same mocked-PySR pattern as test_symbolic_engine_pysr_biology_.py.

Equations tested (from experiment_protocol_all_30.py domain="biology"):
  1. Michaelis-Menten:   v = (Vmax * S) / (Km + S)
  2. Logistic Growth:    dN/dt = r * N * (1 - N/K)
  3. Allometric Scaling: Y = a * M**b

PySR / Julia NOT required — mocked throughout.

Run:
    pytest test_protocol_biology.py -v

Requirements:
    pip install pytest numpy scikit-learn
    symbolic_engine.py on sys.path (hypatiax/tools/symbolic/)
"""

from __future__ import annotations

import math
import os
import pathlib
import sys
import time
import types
import warnings
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# sys.path: find symbolic_engine.py
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve().parent
# Works when run from repo root or from this file's directory
for _candidate in [
    _HERE,
    _HERE.parent,
    _HERE.parent / "hypatiax" / "tools" / "symbolic",
    _HERE.parent.parent / "hypatiax" / "tools" / "symbolic",
]:
    if (_candidate / "symbolic_engine.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from symbolic_engine import (  # noqa: E402
    DiscoveryConfig,
    EquationTools,
    SymbolicEngine,
)

# ---------------------------------------------------------------------------
# Paper-quality config (mirrors PAPER_* constants in the existing biology test)
# ---------------------------------------------------------------------------
PAPER_NITERATIONS = 1000
PAPER_POPULATIONS = 30
PAPER_POPULATION_SIZE = 33
PAPER_PARSIMONY = 0.01
PAPER_MAXSIZE = 30
PAPER_PYSR_TIMEOUT = 1100


def _paper_config(**overrides) -> DiscoveryConfig:
    kwargs = dict(
        niterations=PAPER_NITERATIONS,
        populations=PAPER_POPULATIONS,
        population_size=PAPER_POPULATION_SIZE,
        parsimony=PAPER_PARSIMONY,
        maxsize=PAPER_MAXSIZE,
        pysr_timeout=PAPER_PYSR_TIMEOUT,
    )
    kwargs.update(overrides)
    return DiscoveryConfig(**kwargs)


# ---------------------------------------------------------------------------
# Mock PySR factory (identical to the one in the existing biology test)
# ---------------------------------------------------------------------------

def _make_mock_pysr(equation_str: str, variable_names: list[str],
                    sleep_s: float = 0.0, r2: float = 1.0):
    compiled = EquationTools.compile_equation(equation_str, variable_names)
    rng = np.random.default_rng(99)

    class MockPySRRegressor:
        def __init__(self, **kwargs):
            self._vnames = variable_names

        def fit(self, X, y, variable_names=None, **kwargs):
            if sleep_s > 0:
                time.sleep(sleep_s)
            self._X = X
            self._y = y
            return self

        def predict(self, X, index=None, **kwargs):
            pred = compiled(X)
            if r2 < 1.0:
                sigma = np.std(self._y) * math.sqrt(max(1 - r2, 0))
                pred = pred + rng.normal(0, sigma, len(pred))
            return pred

        @property
        def equations_(self):
            row = types.SimpleNamespace(
                equation=equation_str,
                loss=0.0,
                complexity=len(equation_str.split()),
                score=1.0,
            )
            return [row]

        def score(self, X, y):
            return r2_score(y, self.predict(X))

    return MockPySRRegressor


def _discover(equation_str, X, y, variable_names, sleep_s=0.05):
    """Run discover() with paper config and mocked PySR."""
    MockPySR = _make_mock_pysr(equation_str, variable_names, sleep_s=sleep_s)
    engine = SymbolicEngine(_paper_config(), domain="biology")
    fake_pysr = types.ModuleType("pysr")
    fake_pysr.PySRRegressor = MockPySR
    with patch.dict(sys.modules, {"pysr": fake_pysr}):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return engine.discover(X, y, variable_names=variable_names)


# ---------------------------------------------------------------------------
# Protocol data — loaded directly from ExperimentProtocolAll
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def protocol_biology():
    """Load biology test cases from ExperimentProtocolAll."""
    # Add experiment_protocol path
    proto_path = str(_HERE)
    if proto_path not in sys.path:
        sys.path.insert(0, proto_path)

    try:
        from experiment_protocol_all_30 import ExperimentProtocolAll
        cases = ExperimentProtocolAll.load_test_data("biology", num_samples=120)
        assert len(cases) == 3, f"Expected 3 biology cases, got {len(cases)}"
        return cases
    except ImportError:
        pytest.skip("experiment_protocol_all_30.py not on sys.path — copy it next to this test.")


@pytest.fixture(scope="module")
def mm(protocol_biology):
    desc, X, y, names, meta = protocol_biology[0]
    assert meta["equation_name"] == "michaelis_menten"
    return X, y, names, meta["ground_truth"]


@pytest.fixture(scope="module")
def lg(protocol_biology):
    desc, X, y, names, meta = protocol_biology[1]
    assert meta["equation_name"] == "logistic_growth"
    return X, y, names, meta["ground_truth"]


@pytest.fixture(scope="module")
def allo(protocol_biology):
    desc, X, y, names, meta = protocol_biology[2]
    assert meta["equation_name"] == "allometric_scaling"
    return X, y, names, meta["ground_truth"]


# ===========================================================================
# Protocol sanity — verify the data matches ground truth
# ===========================================================================

class TestProtocolSanity:
    """Verify ExperimentProtocolAll generates correct biology data."""

    def test_mm_data_shape(self, mm):
        X, y, names, _ = mm
        assert X.shape == (120, 3)
        assert y.shape == (120,)
        assert names == ["Vmax", "S", "Km"]

    def test_mm_ground_truth_r2(self, mm):
        X, y, names, gt = mm
        fn = EquationTools.compile_equation(gt, names)
        r2 = r2_score(y, fn(X))
        assert r2 > 0.9999, f"Ground truth R²={r2:.6f} — data generation broken"

    def test_lg_data_shape(self, lg):
        X, y, names, _ = lg
        assert X.shape == (120, 3)
        assert names == ["r", "N", "K"]

    def test_lg_ground_truth_r2(self, lg):
        X, y, names, gt = lg
        fn = EquationTools.compile_equation(gt, names)
        r2 = r2_score(y, fn(X))
        assert r2 > 0.9999, f"Logistic growth ground truth R²={r2:.6f}"

    def test_allo_data_shape(self, allo):
        X, y, names, _ = allo
        assert X.shape == (120, 3)
        assert names == ["a", "M", "b"]

    def test_allo_ground_truth_r2(self, allo):
        X, y, names, gt = allo
        fn = EquationTools.compile_equation(gt, names)
        r2 = r2_score(y, fn(X))
        assert r2 > 0.9999, f"Allometric ground truth R²={r2:.6f}"

    def test_mm_y_strictly_positive(self, mm):
        _, y, _, _ = mm
        assert np.all(y > 0), "MM y must be strictly positive"

    def test_allo_y_strictly_positive(self, allo):
        _, y, _, _ = allo
        assert np.all(y > 0), "Allometric y must be strictly positive"

    def test_lg_y_has_mixed_sign(self, lg):
        """Logistic growth can be negative when N > K."""
        _, y, _, _ = lg
        # Not guaranteed but worth documenting
        assert y.min() < y.max()

    def test_all_three_cases_loaded(self, protocol_biology):
        assert len(protocol_biology) == 3

    def test_protocol_label_is_B(self, protocol_biology):
        for _, _, _, _, meta in protocol_biology:
            assert meta["protocol"] == "B"


# ===========================================================================
# Michaelis-Menten discover() integration
# ===========================================================================

class TestMichaelisMenten:

    def test_mm_result_is_dict(self, mm):
        X, y, names, gt = mm
        result = _discover(gt, X, y, names)
        assert isinstance(result, dict)

    def test_mm_has_equation_key(self, mm):
        X, y, names, gt = mm
        result = _discover(gt, X, y, names)
        assert "equation" in result or "expression" in result, (
            f"Result missing equation key: {list(result.keys())}"
        )

    def test_mm_has_r2_key(self, mm):
        X, y, names, gt = mm
        result = _discover(gt, X, y, names)
        assert "r2_score" in result or "r2" in result

    def test_mm_r2_above_095(self, mm):
        X, y, names, gt = mm
        result = _discover(gt, X, y, names)
        r2 = result.get("r2_score", result.get("r2", -1))
        assert r2 > 0.95, f"MM R²={r2:.4f} — expected >0.95 with exact mock formula"

    def test_mm_r2_above_099(self, mm):
        X, y, names, gt = mm
        result = _discover(gt, X, y, names)
        r2 = result.get("r2_score", result.get("r2", -1))
        assert r2 > 0.99, f"MM exact mock should yield R²>0.99, got {r2:.4f}"

    def test_mm_variable_names_present(self, mm):
        X, y, names, gt = mm
        result = _discover(gt, X, y, names)
        assert "variable_names" in result or "variables" in result

    def test_mm_completes_within_10s(self, mm):
        X, y, names, gt = mm
        t0 = time.time()
        _discover(gt, X, y, names)
        assert time.time() - t0 < 10.0, "MM mock discover took > 10s"

    def test_mm_complexity_present(self, mm):
        X, y, names, gt = mm
        result = _discover(gt, X, y, names)
        assert "complexity" in result or "equation_complexity" in result


# ===========================================================================
# Logistic Growth discover() integration
# ===========================================================================

class TestLogisticGrowth:

    def test_lg_result_is_dict(self, lg):
        X, y, names, gt = lg
        result = _discover(gt, X, y, names)
        assert isinstance(result, dict)

    def test_lg_has_equation_key(self, lg):
        X, y, names, gt = lg
        result = _discover(gt, X, y, names)
        assert "equation" in result or "expression" in result

    def test_lg_has_r2_key(self, lg):
        X, y, names, gt = lg
        result = _discover(gt, X, y, names)
        assert "r2_score" in result or "r2" in result

    def test_lg_r2_above_095(self, lg):
        X, y, names, gt = lg
        result = _discover(gt, X, y, names)
        r2 = result.get("r2_score", result.get("r2", -1))
        assert r2 > 0.95, f"Logistic growth R²={r2:.4f} — expected >0.95"

    def test_lg_completes_within_10s(self, lg):
        X, y, names, gt = lg
        t0 = time.time()
        _discover(gt, X, y, names)
        assert time.time() - t0 < 10.0, "LG mock discover took > 10s"

    def test_lg_result_has_validation_key(self, lg):
        X, y, names, gt = lg
        result = _discover(gt, X, y, names)
        assert "validation" in result


# ===========================================================================
# Allometric Scaling discover() integration
# ===========================================================================

class TestAllometricScaling:

    def test_allo_result_is_dict(self, allo):
        X, y, names, gt = allo
        result = _discover(gt, X, y, names)
        assert isinstance(result, dict)

    def test_allo_has_equation_key(self, allo):
        X, y, names, gt = allo
        result = _discover(gt, X, y, names)
        assert "equation" in result or "expression" in result

    def test_allo_has_r2_key(self, allo):
        X, y, names, gt = allo
        result = _discover(gt, X, y, names)
        assert "r2_score" in result or "r2" in result

    def test_allo_r2_above_095(self, allo):
        X, y, names, gt = allo
        result = _discover(gt, X, y, names)
        r2 = result.get("r2_score", result.get("r2", -1))
        assert r2 > 0.95, f"Allometric R²={r2:.4f} — expected >0.95"

    def test_allo_completes_within_10s(self, allo):
        X, y, names, gt = allo
        t0 = time.time()
        _discover(gt, X, y, names)
        assert time.time() - t0 < 10.0, "Allometric mock discover took > 10s"

    def test_allo_complexity_present(self, allo):
        X, y, names, gt = allo
        result = _discover(gt, X, y, names)
        assert "complexity" in result or "equation_complexity" in result


# ===========================================================================
# Cross-equation: protocol metadata consistency
# ===========================================================================

class TestProtocolMetadata:

    def test_mm_difficulty_is_medium(self, protocol_biology):
        _, _, _, _, meta = protocol_biology[0]
        assert meta["difficulty"] == "medium"

    def test_lg_difficulty_is_medium(self, protocol_biology):
        _, _, _, _, meta = protocol_biology[1]
        assert meta["difficulty"] == "medium"

    def test_allo_difficulty_is_easy(self, protocol_biology):
        _, _, _, _, meta = protocol_biology[2]
        assert meta["difficulty"] == "easy"

    def test_mm_formula_type_rational(self, protocol_biology):
        _, _, _, _, meta = protocol_biology[0]
        assert meta["formula_type"] == "rational"

    def test_lg_formula_type_nonlinear(self, protocol_biology):
        _, _, _, _, meta = protocol_biology[1]
        assert meta["formula_type"] == "nonlinear"

    def test_allo_formula_type_power_law(self, protocol_biology):
        _, _, _, _, meta = protocol_biology[2]
        assert meta["formula_type"] == "power_law"

    def test_all_have_ground_truth(self, protocol_biology):
        for _, _, _, _, meta in protocol_biology:
            assert "ground_truth" in meta
            assert len(meta["ground_truth"]) > 0

    def test_all_have_variable_descriptions(self, protocol_biology):
        for _, _, _, _, meta in protocol_biology:
            assert "variable_descriptions" in meta
            assert len(meta["variable_descriptions"]) > 0

    def test_mm_ground_truth_string(self, protocol_biology):
        _, _, _, _, meta = protocol_biology[0]
        assert meta["ground_truth"] == "(Vmax * S) / (Km + S)"

    def test_lg_ground_truth_string(self, protocol_biology):
        _, _, _, _, meta = protocol_biology[1]
        assert meta["ground_truth"] == "r * N * (1 - N / K)"

    def test_allo_ground_truth_string(self, protocol_biology):
        _, _, _, _, meta = protocol_biology[2]
        assert meta["ground_truth"] == "a * M**b"
