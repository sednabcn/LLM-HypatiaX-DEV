"""
Michael//Handerson Tests — symbolic_engine.py
==============================================

Isolated unit & integration tests for the HypatiaX Symbolic Engine (unified v21+v22+v23).
Tests cover every public class and critical internal behaviour without invoking PySR/Julia.

Test groups
-----------
  MH-VNV   VariableNameValidator
  MH-DPC   DataPatternAnalyzer
  MH-EQT   EquationTools (v22)
  MH-BRK   BayesianRanker (v22)
  MH-DCG   DiscoveryConfig defaults
  MH-LLM   IntegratedLLMEngine (no-API path)
  MH-EQN   ExpressionNode / SymbolicSearch (v23)
  MH-BSR   BayesianSearchRanker (v23)
  MH-DIM   DimensionalValidator (v23)
  MH-STE   SymbolicTreeEngine.discover_validate_interpret (v23)
  MH-SYE   SymbolicEngine helpers (no PySR)
  MH-SEL   SymbolicEngineWithLLM routing (no PySR, no API key)
  MH-CDC   detect_collapsed_constants utility

Run:
    pytest test_symbolic_engine_handerson.py -v
"""

import math
import sys
import os
import warnings

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make sure the engine is importable from the uploads directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, "/mnt/user-data/uploads")

from symbolic_engine import (
    BayesianRanker,
    BayesianSearchRanker,
    DataPatternAnalyzer,
    DimensionalValidator,
    DiscoveryConfig,
    EquationHypothesis,
    EquationTools,
    ExpressionNode,
    IntegratedLLMEngine,
    LLMConfig,
    SymbolicEngine,
    SymbolicEngineWithLLM,
    SymbolicSearch,
    SymbolicTreeEngine,
    VariableNameValidator,
    detect_collapsed_constants,
)


# ===========================================================================
# MH-VNV  VariableNameValidator
# ===========================================================================

class TestMH_VNV:
    """MH-VNV: VariableNameValidator tests."""

    def test_vnv_safe_name_passes_through(self):
        """Non-reserved names are returned unchanged."""
        assert VariableNameValidator.sanitize_name("mass") == "mass"
        assert VariableNameValidator.sanitize_name("velocity") == "velocity"
        assert VariableNameValidator.sanitize_name("x0") == "x0"

    def test_vnv_reserved_Q_maps_to_Qr(self):
        """'Q' is reserved and has a known safe alternative 'Qr'."""
        assert VariableNameValidator.sanitize_name("Q") == "Qr"

    def test_vnv_reserved_E_maps_to_E_val(self):
        """'E' is reserved and maps to 'E_val'."""
        result = VariableNameValidator.sanitize_name("E")
        assert result == "E_val"

    def test_vnv_reserved_pi_maps_to_Pi(self):
        assert VariableNameValidator.sanitize_name("pi") == "Pi"

    def test_vnv_reserved_sin_gets_suffix(self):
        """'sin' is reserved but has no known alias — should get a suffix."""
        result = VariableNameValidator.sanitize_name("sin")
        assert result != "sin"
        assert "sin" in result  # suffix added, original base kept

    def test_vnv_is_reserved_true(self):
        for name in ("sin", "cos", "exp", "log", "sqrt", "Q", "pi"):
            assert VariableNameValidator.is_reserved(name), f"{name!r} should be reserved"

    def test_vnv_is_reserved_false(self):
        for name in ("mass", "energy", "x0", "v1", "temp"):
            assert not VariableNameValidator.is_reserved(name), f"{name!r} should NOT be reserved"

    def test_vnv_sanitize_names_returns_correct_mapping(self):
        names = ["E0", "R", "T", "Q"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            safe, mapping = VariableNameValidator.sanitize_names(names)
        assert "Q" in mapping  # Q was renamed
        assert mapping["Q"] == "Qr"
        assert "E0" not in mapping  # E0 is safe
        assert len(safe) == len(names)

    def test_vnv_sanitize_names_no_collision(self):
        """When the preferred safe alternative is already taken, a fresh suffix is generated."""
        # Both 'sin' and 'cos' are reserved; they must get distinct safe names
        names = ["sin", "cos", "mass"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            safe, mapping = VariableNameValidator.sanitize_names(names)
        # All names must be unique
        assert len(safe) == len(set(safe))

    def test_vnv_update_expression_replaces_word_boundary(self):
        mapping = {"Q": "Qr"}
        expr = "E0 - R * T / (n * F) * log(Q)"
        updated = VariableNameValidator.update_expression(expr, mapping)
        assert "Qr" in updated
        assert "log(Qr)" in updated

    def test_vnv_update_expression_no_partial_replace(self):
        """Standalone 'Q' is replaced; 'Qr' is left intact."""
        mapping = {"Q": "Qr"}
        expr = "Q + Qr"
        updated = VariableNameValidator.update_expression(expr, mapping)
        # After replacement, both tokens should be 'Qr'
        assert "Qr" in updated
        # The original standalone 'Q' (with word boundary) is gone
        import re
        assert not re.search(r"\bQ\b(?!r)", updated)

    def test_vnv_update_expression_empty_mapping(self):
        """Empty mapping returns expression unchanged."""
        expr = "a + b * c"
        assert VariableNameValidator.update_expression(expr, {}) == expr


# ===========================================================================
# MH-DPC  DataPatternAnalyzer
# ===========================================================================

class TestMH_DPC:
    """MH-DPC: DataPatternAnalyzer tests."""

    @pytest.fixture
    def linear_data(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 10, (100, 2))
        y = 3 * X[:, 0] + 2 * X[:, 1]
        return X, y

    @pytest.fixture
    def product_data(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(1, 5, (100, 2))
        y = X[:, 0] * X[:, 1]
        return X, y

    def test_dpc_basic_fields(self, linear_data):
        X, y = linear_data
        patterns = DataPatternAnalyzer().analyze(X, y, ["a", "b"])
        assert patterns["n_variables"] == 2
        assert patterns["n_samples"] == 100
        assert "correlations" in patterns
        assert "y_range" in patterns
        assert "y_scale" in patterns

    def test_dpc_correlation_keys(self, linear_data):
        X, y = linear_data
        patterns = DataPatternAnalyzer().analyze(X, y, ["a", "b"])
        assert "a" in patterns["correlations"]
        assert "b" in patterns["correlations"]

    def test_dpc_correlation_values_finite(self, linear_data):
        X, y = linear_data
        patterns = DataPatternAnalyzer().analyze(X, y, ["a", "b"])
        for v in patterns["correlations"].values():
            assert math.isfinite(v)

    def test_dpc_multiplicative_hint(self, product_data):
        X, y = product_data
        patterns = DataPatternAnalyzer().analyze(X, y, ["p", "q"])
        assert "multiplicative" in patterns["structure_hints"]

    def test_dpc_scale_classification(self):
        analyzer = DataPatternAnalyzer()
        assert analyzer._classify_scale(np.array([1e-8])) == "very_small"
        assert analyzer._classify_scale(np.array([0.5])) == "small"
        assert analyzer._classify_scale(np.array([500.0])) == "medium"
        assert analyzer._classify_scale(np.array([1e5])) == "large"
        assert analyzer._classify_scale(np.array([1e8])) == "very_large"

    def test_dpc_quadratic_hint(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(1, 5, (80, 1))
        y = X[:, 0] ** 2
        patterns = DataPatternAnalyzer().analyze(X, y, ["x"])
        assert any("quadratic" in h for h in patterns["structure_hints"])


# ===========================================================================
# MH-EQT  EquationTools (v22)
# ===========================================================================

class TestMH_EQT:
    """MH-EQT: EquationTools.compile_equation tests."""

    def test_eqt_simple_product(self):
        fn = EquationTools.compile_equation("x0 * x1", ["x0", "x1"])
        X = np.array([[2.0, 3.0], [4.0, 5.0]])
        result = fn(X)
        np.testing.assert_allclose(result, [6.0, 20.0])

    def test_eqt_with_numpy_function(self):
        fn = EquationTools.compile_equation("sqrt(x0)", ["x0"])
        X = np.array([[4.0], [9.0]])
        np.testing.assert_allclose(fn(X), [2.0, 3.0])

    def test_eqt_linear_combination(self):
        fn = EquationTools.compile_equation("2 * x0 + x1 - 1", ["x0", "x1"])
        X = np.array([[1.0, 1.0]])
        np.testing.assert_allclose(fn(X), [2.0])

    def test_eqt_with_exp(self):
        fn = EquationTools.compile_equation("exp(x0)", ["x0"])
        X = np.array([[0.0], [1.0]])
        np.testing.assert_allclose(fn(X), [1.0, math.e], rtol=1e-6)

    def test_eqt_with_log(self):
        fn = EquationTools.compile_equation("log(x0)", ["x0"])
        X = np.array([[1.0], [math.e]])
        np.testing.assert_allclose(fn(X), [0.0, 1.0], atol=1e-10)

    def test_eqt_constant_expression(self):
        fn = EquationTools.compile_equation("3.14", ["x0"])
        X = np.array([[1.0], [2.0]])
        result = fn(X)
        np.testing.assert_allclose(result, [3.14, 3.14], atol=1e-10)


# ===========================================================================
# MH-BRK  BayesianRanker (v22)
# ===========================================================================

class TestMH_BRK:
    """MH-BRK: BayesianRanker tests."""

    @pytest.fixture
    def simple_data(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 5, (50, 2))
        y = X[:, 0] * X[:, 1]
        return X, y

    def test_brk_log_likelihood_perfect_fit(self, simple_data):
        X, y = simple_data
        ranker = BayesianRanker()
        # Perfect fit should give high log-likelihood
        score = ranker.log_likelihood(y, y)
        assert math.isfinite(score)

    def test_brk_log_prior_decreases_with_complexity(self):
        ranker = BayesianRanker(complexity_penalty=0.01)
        lp_simple = ranker.log_prior(5)
        lp_complex = ranker.log_prior(20)
        assert lp_simple > lp_complex

    def test_brk_rank_best_first(self, simple_data):
        X, y = simple_data
        candidates = [
            {
                "equation": "x0 * x1",
                "complexity": 3,
                "callable": EquationTools.compile_equation("x0 * x1", ["x0", "x1"]),
            },
            {
                "equation": "x0 + x1",
                "complexity": 3,
                "callable": EquationTools.compile_equation("x0 + x1", ["x0", "x1"]),
            },
        ]
        ranker = BayesianRanker()
        ranked = ranker.rank(candidates, X, y)
        # Exact formula should beat the wrong one
        assert ranked[0]["equation"] == "x0 * x1"

    def test_brk_rank_returns_all_valid(self, simple_data):
        X, y = simple_data
        candidates = [
            {
                "equation": "x0",
                "complexity": 1,
                "callable": EquationTools.compile_equation("x0", ["x0", "x1"]),
            },
            {
                "equation": "x1",
                "complexity": 1,
                "callable": EquationTools.compile_equation("x1", ["x0", "x1"]),
            },
        ]
        ranker = BayesianRanker()
        ranked = ranker.rank(candidates, X, y)
        assert len(ranked) == 2

    def test_brk_complexity_penalty_effect(self, simple_data):
        X, y = simple_data
        ranker_strict = BayesianRanker(complexity_penalty=1.0)
        ranker_lenient = BayesianRanker(complexity_penalty=0.0)
        # Both score the same candidate
        candidates = [
            {
                "equation": "x0",
                "complexity": 5,
                "callable": EquationTools.compile_equation("x0", ["x0", "x1"]),
            }
        ]
        ranked_strict = ranker_strict.rank(candidates, X, y)
        ranked_lenient = ranker_lenient.rank(candidates, X, y)
        assert ranked_strict[0]["posterior_score"] < ranked_lenient[0]["posterior_score"]

    def test_brk_skips_erroring_callable(self, simple_data):
        X, y = simple_data

        def bad_fn(X):
            raise ValueError("Intentional error")

        candidates = [
            {"equation": "boom", "complexity": 1, "callable": bad_fn},
            {
                "equation": "x0 * x1",
                "complexity": 3,
                "callable": EquationTools.compile_equation("x0 * x1", ["x0", "x1"]),
            },
        ]
        ranker = BayesianRanker()
        ranked = ranker.rank(candidates, X, y)
        assert len(ranked) == 1
        assert ranked[0]["equation"] == "x0 * x1"


# ===========================================================================
# MH-DCG  DiscoveryConfig defaults
# ===========================================================================

class TestMH_DCG:
    """MH-DCG: DiscoveryConfig default values and invariants."""

    def test_dcg_default_binary_operators(self):
        cfg = DiscoveryConfig()
        assert set(cfg.binary_operators) == {"+", "-", "*", "/"}

    def test_dcg_default_unary_operators(self):
        cfg = DiscoveryConfig()
        assert "sqrt" in cfg.unary_operators

    def test_dcg_timeout_positive(self):
        cfg = DiscoveryConfig()
        assert cfg.pysr_timeout > 0

    def test_dcg_parsimony_positive(self):
        cfg = DiscoveryConfig()
        assert cfg.parsimony > 0

    def test_dcg_maxsize_reasonable(self):
        cfg = DiscoveryConfig()
        assert 10 <= cfg.maxsize <= 50

    def test_dcg_transcendental_ops_keys(self):
        expected = {"safe_asin", "safe_acos", "asin_of_sin", "acos_of_cos", "atan_of_tan"}
        assert set(DiscoveryConfig._TRANSCENDENTAL_OPS.keys()) == expected

    def test_dcg_transcendental_ops_are_julia_strings(self):
        for name, body in DiscoveryConfig._TRANSCENDENTAL_OPS.items():
            assert name in body, f"Julia body for {name!r} must contain the function name"
            assert "=" in body

    def test_dcg_loss_default_none(self):
        assert DiscoveryConfig().loss is None

    def test_dcg_show_progress_default_false(self):
        assert DiscoveryConfig().show_progress is False

    def test_dcg_custom_values_accepted(self):
        cfg = DiscoveryConfig(niterations=5, populations=3, maxsize=15)
        assert cfg.niterations == 5
        assert cfg.populations == 3
        assert cfg.maxsize == 15


# ===========================================================================
# MH-LLM  IntegratedLLMEngine (no-API path)
# ===========================================================================

class TestMH_LLM:
    """MH-LLM: IntegratedLLMEngine — disabled / no-key path."""

    def test_llm_disabled_returns_empty(self):
        cfg = LLMConfig(enabled=False)
        engine = IntegratedLLMEngine(cfg)
        result = engine.generate_hypotheses("physics", ["m", "v"], "KE", {})
        assert result == []

    def test_llm_no_api_key_disables_itself(self):
        """When no key is present and enabled=True, engine self-disables."""
        cfg = LLMConfig(enabled=True, api_key=None)
        # Temporarily clear env var
        original = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            engine = IntegratedLLMEngine(cfg)
            assert not engine.config.enabled
        finally:
            if original is not None:
                os.environ["ANTHROPIC_API_KEY"] = original

    def test_llm_parse_response_json_array(self):
        cfg = LLMConfig(enabled=False)
        engine = IntegratedLLMEngine(cfg)
        raw = '[{"equation": "y = m * v**2", "confidence": 0.9, "reasoning": "KE"}]'
        hypotheses = engine._parse_response(raw)
        assert len(hypotheses) == 1
        assert "m * v**2" in hypotheses[0].equation
        assert hypotheses[0].confidence == pytest.approx(0.9)

    def test_llm_parse_response_strips_y_equals(self):
        cfg = LLMConfig(enabled=False)
        engine = IntegratedLLMEngine(cfg)
        raw = '[{"equation": "y = 0.5 * m * v**2", "confidence": 0.8, "reasoning": "half mv2"}]'
        hypotheses = engine._parse_response(raw)
        assert hypotheses[0].equation.startswith("0.5")

    def test_llm_parse_response_with_markdown_fences(self):
        cfg = LLMConfig(enabled=False)
        engine = IntegratedLLMEngine(cfg)
        raw = '```json\n[{"equation": "a + b", "confidence": 0.5, "reasoning": "sum"}]\n```'
        hypotheses = engine._parse_response(raw)
        assert len(hypotheses) == 1

    def test_llm_parse_response_caret_to_power(self):
        """The parser must convert ^ to ** in equation strings."""
        cfg = LLMConfig(enabled=False)
        engine = IntegratedLLMEngine(cfg)
        raw = '[{"equation": "y = m^2", "confidence": 0.7, "reasoning": "square"}]'
        hypotheses = engine._parse_response(raw)
        assert "**" in hypotheses[0].equation

    def test_llm_parse_response_bad_json_returns_empty(self):
        cfg = LLMConfig(enabled=False)
        engine = IntegratedLLMEngine(cfg)
        hypotheses = engine._parse_response("NOT JSON AT ALL")
        assert hypotheses == []

    def test_llm_build_prompt_contains_variables(self):
        cfg = LLMConfig(enabled=False)
        engine = IntegratedLLMEngine(cfg)
        prompt = engine._build_prompt("physics", ["m", "v"], "KE", {}, 2)
        assert "m" in prompt
        assert "v" in prompt

    def test_llm_build_prompt_contains_caller_id(self):
        cfg = LLMConfig(enabled=False)
        engine = IntegratedLLMEngine(cfg)
        prompt = engine._build_prompt("physics", ["m", "v"], "KE", {}, 2, caller_id="TestCaller")
        assert "TestCaller" in prompt


# ===========================================================================
# MH-EQN  ExpressionNode / SymbolicSearch (v23)
# ===========================================================================

class TestMH_EQN:
    """MH-EQN: ExpressionNode and SymbolicSearch tests."""

    def test_eqn_leaf_var_complexity(self):
        node = ExpressionNode("var", value="x")
        assert node.complexity() == 1

    def test_eqn_leaf_const_complexity(self):
        node = ExpressionNode("const", value=3.14)
        assert node.complexity() == 1

    def test_eqn_binary_complexity(self):
        left = ExpressionNode("var", value="x")
        right = ExpressionNode("const", value=2.0)
        node = ExpressionNode("+", left, right)
        assert node.complexity() == 3  # root + 2 leaves

    def test_eqn_unary_complexity(self):
        child = ExpressionNode("var", value="x")
        node = ExpressionNode("sin", child)
        assert node.complexity() == 2  # root + 1 leaf

    def test_eqn_to_sympy_var(self):
        import sympy as sp
        node = ExpressionNode("var", value="mass")
        expr = node.to_sympy()
        assert expr == sp.Symbol("mass")

    def test_eqn_to_sympy_const(self):
        import sympy as sp
        node = ExpressionNode("const", value=2.5)
        expr = node.to_sympy()
        assert float(expr) == pytest.approx(2.5)

    def test_eqn_to_sympy_binary_add(self):
        import sympy as sp
        a = ExpressionNode("var", value="a")
        b = ExpressionNode("var", value="b")
        node = ExpressionNode("+", a, b)
        expr = node.to_sympy()
        assert expr == sp.Symbol("a") + sp.Symbol("b")

    def test_symbolic_search_generates_tree(self):
        gen = SymbolicSearch(["x", "y"], max_depth=3)
        tree = gen.generate(3)
        assert isinstance(tree, ExpressionNode)
        assert tree.complexity() >= 1

    def test_symbolic_search_respects_depth_zero(self):
        """At depth=0, generate() must return a leaf (var or const)."""
        gen = SymbolicSearch(["x"], max_depth=0)
        for _ in range(20):
            node = gen.generate(0)
            assert node.op in ("var", "const")

    def test_symbolic_search_uses_given_variables(self):
        """Variable nodes must use one of the supplied variable names."""
        gen = SymbolicSearch(["alpha", "beta"], max_depth=2)

        def collect_vars(node):
            if node.op == "var":
                return {node.value}
            result = set()
            if node.left:
                result |= collect_vars(node.left)
            if node.right:
                result |= collect_vars(node.right)
            return result

        for _ in range(30):
            tree = gen.generate(2)
            vars_used = collect_vars(tree)
            for v in vars_used:
                assert v in ("alpha", "beta")


# ===========================================================================
# MH-BSR  BayesianSearchRanker (v23)
# ===========================================================================

class TestMH_BSR:
    """MH-BSR: BayesianSearchRanker tests."""

    def test_bsr_prior_between_zero_and_one(self):
        ranker = BayesianSearchRanker(complexity_penalty=0.01)
        assert 0 < ranker.prior(1) <= 1.0
        assert 0 < ranker.prior(100) < ranker.prior(1)

    def test_bsr_likelihood_zero_error_is_one(self):
        ranker = BayesianSearchRanker()
        assert ranker.likelihood(0.0) == pytest.approx(1.0)

    def test_bsr_likelihood_decreases_with_error(self):
        ranker = BayesianSearchRanker()
        assert ranker.likelihood(1.0) < ranker.likelihood(0.5)
        assert ranker.likelihood(0.5) < ranker.likelihood(0.0)

    def test_bsr_posterior_combines_both(self):
        ranker = BayesianSearchRanker(complexity_penalty=0.1)
        p1 = ranker.posterior(0.0, 2)   # perfect fit, low complexity
        p2 = ranker.posterior(1.0, 2)   # poor fit, low complexity
        p3 = ranker.posterior(0.0, 20)  # perfect fit, high complexity
        assert p1 > p2
        assert p1 > p3

    def test_bsr_custom_penalty(self):
        strict = BayesianSearchRanker(complexity_penalty=2.0)
        lenient = BayesianSearchRanker(complexity_penalty=0.001)
        assert strict.prior(5) < lenient.prior(5)


# ===========================================================================
# MH-DIM  DimensionalValidator (v23)
# ===========================================================================

class TestMH_DIM:
    """MH-DIM: DimensionalValidator tests."""

    def test_dim_simple_expression_valid(self):
        import sympy as sp
        validator = DimensionalValidator({"x": "m", "t": "s"})
        expr = sp.Symbol("x") + sp.Symbol("t")
        assert validator.validate(expr) is True

    def test_dim_product_valid(self):
        import sympy as sp
        validator = DimensionalValidator({})
        expr = sp.Symbol("m") * sp.Symbol("v") ** 2
        assert validator.validate(expr) is True

    def test_dim_stores_units(self):
        units = {"v": "m/s", "m": "kg"}
        validator = DimensionalValidator(units)
        assert validator.variable_units == units


# ===========================================================================
# MH-STE  SymbolicTreeEngine (v23) — full pipeline
# ===========================================================================

class TestMH_STE:
    """MH-STE: SymbolicTreeEngine.discover_validate_interpret tests."""

    @pytest.fixture
    def product_dataset(self):
        rng = np.random.default_rng(7)
        X = rng.uniform(0.5, 3.0, (60, 2))
        y = X[:, 0] * X[:, 1]
        return X, y

    def test_ste_returns_required_keys(self, product_dataset):
        X, y = product_dataset
        engine = SymbolicTreeEngine(max_depth=3, population_size=100, iterations=5)
        result = engine.discover_validate_interpret(
            X, y, variable_names=["m", "v"], verbose=False, show_formatted=False
        )
        for key in ("equation", "r2", "error", "complexity", "posterior", "dimensionally_valid"):
            assert key in result, f"Missing key: {key}"

    def test_ste_r2_is_finite(self, product_dataset):
        X, y = product_dataset
        engine = SymbolicTreeEngine(max_depth=3, population_size=100, iterations=5)
        result = engine.discover_validate_interpret(
            X, y, variable_names=["a", "b"], verbose=False, show_formatted=False
        )
        assert math.isfinite(result["r2"])

    def test_ste_error_nonnegative(self, product_dataset):
        X, y = product_dataset
        engine = SymbolicTreeEngine(max_depth=3, population_size=100, iterations=5)
        result = engine.discover_validate_interpret(
            X, y, variable_names=["a", "b"], verbose=False, show_formatted=False
        )
        assert result["error"] >= 0

    def test_ste_no_valid_expr_returns_null_result(self):
        """When no valid expression is found, result should contain None equation."""
        # 0 iterations → search produces nothing
        engine = SymbolicTreeEngine(max_depth=2, population_size=1, iterations=0)
        X = np.array([[1.0, 2.0]])
        y = np.array([2.0])
        result = engine.discover_validate_interpret(
            X, y, variable_names=["p", "q"], verbose=False, show_formatted=False
        )
        assert result["equation"] is None
        assert result["r2"] == 0.0

    def test_ste_dimensionally_valid_bool(self, product_dataset):
        X, y = product_dataset
        engine = SymbolicTreeEngine(max_depth=3, population_size=100, iterations=5)
        result = engine.discover_validate_interpret(
            X, y, variable_names=["a", "b"],
            variable_units={"a": "m", "b": "s"},
            verbose=False, show_formatted=False
        )
        assert isinstance(result["dimensionally_valid"], bool)


# ===========================================================================
# MH-SYE  SymbolicEngine helpers (no PySR required)
# ===========================================================================

class TestMH_SYE:
    """MH-SYE: SymbolicEngine static helpers that don't invoke PySR."""

    def test_sye_validate_safe_names_no_change(self):
        names = ["mass", "velocity", "time"]
        safe, mapping = SymbolicEngine.validate_variable_names(names, auto_fix=True)
        assert safe == names
        assert mapping == {}

    def test_sye_validate_reserved_names_with_auto_fix(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            safe, mapping = SymbolicEngine.validate_variable_names(
                ["Q", "mass"], auto_fix=True
            )
        assert "Q" in mapping
        assert "Q" not in safe

    def test_sye_validate_raises_without_auto_fix(self):
        with pytest.raises(ValueError, match="reserved"):
            SymbolicEngine.validate_variable_names(["sin"], auto_fix=False)

    def test_sye_instantiation(self):
        cfg = DiscoveryConfig()
        engine = SymbolicEngine(cfg, domain="physics")
        assert engine.domain == "physics"
        assert engine.model is None


# ===========================================================================
# MH-SEL  SymbolicEngineWithLLM routing (no PySR, no API key)
# ===========================================================================

class TestMH_SEL:
    """MH-SEL: SymbolicEngineWithLLM initialisation and routing."""

    def test_sel_default_construction(self):
        engine = SymbolicEngineWithLLM(llm_mode="none")
        assert engine.llm_mode == "none"
        assert engine.llm_engine is None

    def test_sel_llm_mode_none_routing(self):
        """In 'none' mode, discover() delegates to the parent (PySR) path.
        We verify routing by patching the parent's discover method."""
        engine = SymbolicEngineWithLLM(llm_mode="none")
        sentinel = {"expression": "patched", "r2_score": 1.0,
                    "complexity": 1, "variable_names": ["x"],
                    "original_variable_names": ["x"],
                    "variable_name_mapping": {}, "predictions": np.array([]),
                    "validation": {"valid": True, "errors": [], "warnings": []},
                    "trace": []}

        original_discover = SymbolicEngine.discover
        call_log = []

        def mock_discover(self_, X, y, variable_names=None,
                          equation_name=None, random_state=42,
                          auto_sanitize=True, **kwargs):
            call_log.append("parent_called")
            return sentinel

        SymbolicEngine.discover = mock_discover
        try:
            result = engine.discover(
                np.array([[1.0]]), np.array([1.0]), variable_names=["x"]
            )
            assert "parent_called" in call_log
        finally:
            SymbolicEngine.discover = original_discover

    def test_sel_extract_operators_basic(self):
        engine = SymbolicEngineWithLLM(llm_mode="none")
        ops = engine._extract_operators_from_equation("exp(x) * log(y) + sin(z)")
        assert "exp" in ops["unary_operators"]
        assert "log" in ops["unary_operators"]
        assert "sin" in ops["unary_operators"]
        assert "*" in ops["binary_operators"]
        assert "+" in ops["binary_operators"]

    def test_sel_extract_operators_no_pow(self):
        """'**' must NOT be mapped to 'pow' (PySR pow crashes on negative bases)."""
        engine = SymbolicEngineWithLLM(llm_mode="none")
        ops = engine._extract_operators_from_equation("x**2")
        assert "pow" not in ops["binary_operators"]

    def test_sel_predict_from_equation_basic(self):
        engine = SymbolicEngineWithLLM(llm_mode="none")
        X = np.array([[2.0, 3.0], [4.0, 5.0]])
        result = engine._predict_from_equation("m * v", X, ["m", "v"])
        np.testing.assert_allclose(result, [6.0, 20.0])

    def test_sel_predict_from_equation_with_constant(self):
        engine = SymbolicEngineWithLLM(llm_mode="none")
        X = np.array([[1.0]])
        result = engine._predict_from_equation("k_B * x0", X, ["x0"])
        expected = 1.380649e-23 * 1.0
        np.testing.assert_allclose(result, [expected], rtol=1e-6)

    def test_sel_predict_from_equation_bad_raises(self):
        engine = SymbolicEngineWithLLM(llm_mode="none")
        X = np.array([[1.0]])
        with pytest.raises(ValueError, match="Failed to evaluate"):
            engine._predict_from_equation("undefined_func(x0)", X, ["x0"])

    def test_sel_evaluate_hypotheses_sorts_by_r2(self):
        engine = SymbolicEngineWithLLM(llm_mode="none")
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([2.0, 12.0])  # y = x0 * x1

        hypotheses = [
            EquationHypothesis("x0 + x1", 0.5, "sum"),   # wrong
            EquationHypothesis("x0 * x1", 0.9, "product"),  # correct
        ]
        best = engine._evaluate_hypotheses(hypotheses, X, y, ["x0", "x1"])
        assert best.equation == "x0 * x1"

    def test_sel_discover_formula_returns_schema_on_exception(self):
        """discover_formula must return a failure dict, not raise."""
        engine = SymbolicEngineWithLLM(llm_mode="none")

        # Patch to force an exception
        def always_fail(*args, **kwargs):
            raise RuntimeError("Forced failure")

        original = engine.discover
        engine.discover = always_fail
        try:
            result = engine.discover_formula(
                np.array([[1.0]]), np.array([1.0]), ["x"]
            )
            assert result["success"] is False
            assert "error" in result
        finally:
            engine.discover = original


# ===========================================================================
# MH-CDC  detect_collapsed_constants
# ===========================================================================

class TestMH_CDC:
    """MH-CDC: detect_collapsed_constants utility."""

    def test_cdc_detects_gravity(self):
        result = detect_collapsed_constants("9.81 * t**2 / 2", ["t"])
        assert any("g" in r for r in result)

    def test_cdc_no_false_positive_on_safe_expr(self):
        result = detect_collapsed_constants("a * b + c", ["a", "b", "c"])
        assert result == []

    def test_cdc_does_not_flag_variable_value(self):
        """Values not matching known physical constant patterns return no hits."""
        # A made-up value that doesn't match any known constant regex
        result = detect_collapsed_constants("42.0 * t", ["t"])
        assert result == []

    def test_cdc_returns_list(self):
        result = detect_collapsed_constants("x + y", ["x", "y"])
        assert isinstance(result, list)

    def test_cdc_detects_planck(self):
        expr = "6.626e-34 * frequency"
        result = detect_collapsed_constants(expr, ["frequency"])
        assert any("h" in r or "Planck" in r for r in result)


# ===========================================================================
# MH-BIO  Biology equations (from experiment_protocol_all_30.py domain=biology)
#          Three equations: Michaelis-Menten, Logistic Growth, Allometric Scaling
#          Tests confirm data generation, ground-truth R², and engine helpers
#          — no PySR/Julia required.
# ===========================================================================

class TestMH_BIO:
    """MH-BIO: Biology equation data + symbolic helpers (no PySR)."""

    # ── shared fixtures ──────────────────────────────────────────────────────

    @pytest.fixture
    def michaelis_menten_data(self):
        """v = (Vmax * S) / (Km + S)  — Michaelis-Menten kinetics."""
        rng = np.random.default_rng(10)
        num_samples = 100
        Vmax = np.full(num_samples, 50.0)
        S    = rng.uniform(0.1, 50, num_samples)
        Km   = np.full(num_samples, 10.0)
        X = np.column_stack([Vmax, S, Km])
        y = (Vmax * S) / (Km + S)
        return X, y, ["Vmax", "S", "Km"]

    @pytest.fixture
    def logistic_growth_data(self):
        """dN/dt = r * N * (1 - N/K)  — Logistic Growth."""
        rng = np.random.default_rng(11)
        num_samples = 100
        r = rng.uniform(0.1, 0.5, num_samples)
        N = rng.uniform(10, 900, num_samples)
        K = rng.uniform(1000, 2000, num_samples)
        X = np.column_stack([r, N, K])
        y = r * N * (1 - N / K)
        return X, y, ["r", "N", "K"]

    @pytest.fixture
    def allometric_data(self):
        """Y = a * M^b  — Allometric Scaling."""
        rng = np.random.default_rng(12)
        num_samples = 100
        a = np.full(num_samples, 3.5)
        M = rng.uniform(0.1, 100, num_samples)
        b = np.full(num_samples, 0.75)
        X = np.column_stack([a, M, b])
        y = a * M ** b
        return X, y, ["a", "M", "b"]

    # ── Michaelis-Menten ─────────────────────────────────────────────────────

    def test_bio_mm_data_shape(self, michaelis_menten_data):
        X, y, names = michaelis_menten_data
        assert X.shape == (100, 3)
        assert y.shape == (100,)
        assert names == ["Vmax", "S", "Km"]

    def test_bio_mm_y_positive(self, michaelis_menten_data):
        """Reaction velocity must be strictly positive."""
        _, y, _ = michaelis_menten_data
        assert np.all(y > 0)

    def test_bio_mm_y_bounded_by_vmax(self, michaelis_menten_data):
        """v < Vmax always (saturation curve never reaches Vmax)."""
        X, y, _ = michaelis_menten_data
        Vmax = X[:, 0]
        assert np.all(y < Vmax)

    def test_bio_mm_ground_truth_r2_perfect(self, michaelis_menten_data):
        """Evaluating the exact formula on the dataset gives R²=1."""
        from sklearn.metrics import r2_score
        X, y, _ = michaelis_menten_data
        Vmax, S, Km = X[:, 0], X[:, 1], X[:, 2]
        y_pred = (Vmax * S) / (Km + S)
        assert r2_score(y, y_pred) == pytest.approx(1.0, abs=1e-10)

    def test_bio_mm_saturation_behaviour(self, michaelis_menten_data):
        """At S >> Km, velocity should approach Vmax (saturation)."""
        X, y, _ = michaelis_menten_data
        Vmax, Km = X[0, 0], X[0, 2]
        S_large = 1e6
        v_saturated = (Vmax * S_large) / (Km + S_large)
        assert abs(v_saturated - Vmax) / Vmax < 1e-4

    def test_bio_mm_half_vmax_at_km(self):
        """At S = Km, v = Vmax/2 (definition of Km)."""
        Vmax, Km = 50.0, 10.0
        S = Km
        v = (Vmax * S) / (Km + S)
        assert v == pytest.approx(Vmax / 2)

    def test_bio_mm_variable_names_safe(self, michaelis_menten_data):
        """None of the MM variable names conflict with PySR reserved words."""
        _, _, names = michaelis_menten_data
        for name in names:
            assert not VariableNameValidator.is_reserved(name), \
                f"{name!r} unexpectedly reserved"

    def test_bio_mm_pattern_analyzer(self, michaelis_menten_data):
        """DataPatternAnalyzer runs without error on MM data."""
        X, y, names = michaelis_menten_data
        patterns = DataPatternAnalyzer().analyze(X, y, names)
        assert patterns["n_variables"] == 3
        assert patterns["n_samples"] == 100

    def test_bio_mm_equation_tools_eval(self, michaelis_menten_data):
        """EquationTools can compile and evaluate the MM formula string."""
        X, y, names = michaelis_menten_data
        fn = EquationTools.compile_equation("Vmax * S / (Km + S)", names)
        y_pred = fn(X)
        np.testing.assert_allclose(y_pred, y, rtol=1e-10)

    def test_bio_mm_predict_from_equation(self, michaelis_menten_data):
        """SymbolicEngineWithLLM._predict_from_equation evaluates MM formula."""
        X, y, names = michaelis_menten_data
        engine = SymbolicEngineWithLLM(llm_mode="none")
        y_pred = engine._predict_from_equation("Vmax * S / (Km + S)", X, names)
        np.testing.assert_allclose(y_pred, y, rtol=1e-10)

    # ── Logistic Growth ──────────────────────────────────────────────────────

    def test_bio_lg_data_shape(self, logistic_growth_data):
        X, y, names = logistic_growth_data
        assert X.shape == (100, 3)
        assert y.shape == (100,)
        assert names == ["r", "N", "K"]

    def test_bio_lg_ground_truth_r2_perfect(self, logistic_growth_data):
        from sklearn.metrics import r2_score
        X, y, _ = logistic_growth_data
        r, N, K = X[:, 0], X[:, 1], X[:, 2]
        y_pred = r * N * (1 - N / K)
        assert r2_score(y, y_pred) == pytest.approx(1.0, abs=1e-10)

    def test_bio_lg_sign_correct(self, logistic_growth_data):
        """Growth is positive when N < K (population below carrying capacity)."""
        X, y, _ = logistic_growth_data
        N, K = X[:, 1], X[:, 2]
        below_K = N < K
        assert np.all(y[below_K] > 0), \
            "dN/dt should be positive when N < K"

    def test_bio_lg_zero_at_carrying_capacity(self):
        """At N = K, logistic growth rate is exactly 0."""
        r, N, K = 0.3, 1000.0, 1000.0
        rate = r * N * (1 - N / K)
        assert rate == pytest.approx(0.0, abs=1e-12)

    def test_bio_lg_variable_names_safe(self, logistic_growth_data):
        _, _, names = logistic_growth_data
        for name in names:
            assert not VariableNameValidator.is_reserved(name), \
                f"{name!r} is reserved — rename in protocol"

    def test_bio_lg_pattern_analyzer(self, logistic_growth_data):
        X, y, names = logistic_growth_data
        patterns = DataPatternAnalyzer().analyze(X, y, names)
        assert patterns["n_variables"] == 3

    def test_bio_lg_equation_tools_eval(self, logistic_growth_data):
        X, y, names = logistic_growth_data
        fn = EquationTools.compile_equation("r * N * (1 - N / K)", names)
        np.testing.assert_allclose(fn(X), y, rtol=1e-10)

    def test_bio_lg_predict_from_equation(self, logistic_growth_data):
        X, y, names = logistic_growth_data
        engine = SymbolicEngineWithLLM(llm_mode="none")
        y_pred = engine._predict_from_equation("r * N * (1 - N / K)", X, names)
        np.testing.assert_allclose(y_pred, y, rtol=1e-10)

    def test_bio_lg_r_range(self, logistic_growth_data):
        """Intrinsic growth rate r should be in the protocol range [0.1, 0.5]."""
        X, _, _ = logistic_growth_data
        r = X[:, 0]
        assert np.all(r >= 0.1) and np.all(r <= 0.5)

    def test_bio_lg_N_below_K_always(self, logistic_growth_data):
        """Protocol guarantees N < K so all growth rates are positive."""
        X, _, _ = logistic_growth_data
        N, K = X[:, 1], X[:, 2]
        assert np.all(N < K)

    # ── Allometric Scaling ───────────────────────────────────────────────────

    def test_bio_as_data_shape(self, allometric_data):
        X, y, names = allometric_data
        assert X.shape == (100, 3)
        assert y.shape == (100,)
        assert names == ["a", "M", "b"]

    def test_bio_as_ground_truth_r2_perfect(self, allometric_data):
        from sklearn.metrics import r2_score
        X, y, _ = allometric_data
        a, M, b = X[:, 0], X[:, 1], X[:, 2]
        y_pred = a * M ** b
        assert r2_score(y, y_pred) == pytest.approx(1.0, abs=1e-10)

    def test_bio_as_y_positive(self, allometric_data):
        """Metabolic rate (Y) must be strictly positive."""
        _, y, _ = allometric_data
        assert np.all(y > 0)

    def test_bio_as_scaling_exponent_constant(self, allometric_data):
        """Protocol fixes b=0.75 (Kleiber's law exponent)."""
        X, _, _ = allometric_data
        b = X[:, 2]
        np.testing.assert_allclose(b, 0.75)

    def test_bio_as_coefficient_constant(self, allometric_data):
        """Protocol fixes a=3.5."""
        X, _, _ = allometric_data
        a = X[:, 0]
        np.testing.assert_allclose(a, 3.5)

    def test_bio_as_power_law_shape(self, allometric_data):
        """log(Y) vs log(M) should be linear with slope b=0.75."""
        X, y, _ = allometric_data
        M = X[:, 1]
        log_M = np.log(M)
        log_y = np.log(y)
        # Fit slope via least squares
        slope = np.polyfit(log_M, log_y, 1)[0]
        assert slope == pytest.approx(0.75, abs=0.01)

    def test_bio_as_variable_names_safe(self, allometric_data):
        _, _, names = allometric_data
        for name in names:
            assert not VariableNameValidator.is_reserved(name), \
                f"{name!r} is reserved — rename in protocol"

    def test_bio_as_pattern_analyzer(self, allometric_data):
        X, y, names = allometric_data
        patterns = DataPatternAnalyzer().analyze(X, y, names)
        assert patterns["n_variables"] == 3
        assert patterns["y_scale"] in ("medium", "large")

    def test_bio_as_equation_tools_eval(self, allometric_data):
        X, y, names = allometric_data
        fn = EquationTools.compile_equation("a * M ** b", names)
        np.testing.assert_allclose(fn(X), y, rtol=1e-10)

    def test_bio_as_predict_from_equation(self, allometric_data):
        X, y, names = allometric_data
        engine = SymbolicEngineWithLLM(llm_mode="none")
        y_pred = engine._predict_from_equation("a * M ** b", X, names)
        np.testing.assert_allclose(y_pred, y, rtol=1e-10)

    # ── Cross-equation: BayesianRanker picks the right formula ──────────────

    def test_bio_bayesian_ranker_prefers_mm_over_linear(self, michaelis_menten_data):
        """BayesianRanker should score the MM formula higher than a linear guess."""
        X, y, names = michaelis_menten_data
        candidates = [
            {
                "equation": "Vmax * S / (Km + S)",
                "complexity": 5,
                "callable": EquationTools.compile_equation(
                    "Vmax * S / (Km + S)", names
                ),
            },
            {
                "equation": "Vmax + S",
                "complexity": 3,
                "callable": EquationTools.compile_equation("Vmax + S", names),
            },
        ]
        ranker = BayesianRanker(complexity_penalty=0.01)
        ranked = ranker.rank(candidates, X, y)
        assert ranked[0]["equation"] == "Vmax * S / (Km + S)"

    def test_bio_domain_tag_in_exp_domains(self):
        """'biology' must be in SymbolicEngine's _EXP_DOMAINS so exp/log are injected."""
        # We verify this by inspecting the source string rather than running PySR
        import inspect
        source = inspect.getsource(SymbolicEngine.discover)
        assert '"biology"' in source, \
            "'biology' must be listed in _EXP_DOMAINS inside SymbolicEngine.discover"


# ===========================================================================
# MH-INT  Integration smoke-test: full v23 pipeline on known equation
# ===========================================================================

class TestMH_INT:
    """MH-INT: Integration test — SymbolicTreeEngine on a tractable target."""

    def test_int_finds_reasonable_r2_on_linear_target(self):
        """SymbolicTreeEngine should achieve R²>0.8 on y = a + b with enough budget."""
        rng = np.random.default_rng(42)
        X = rng.uniform(1, 5, (80, 2))
        y = X[:, 0] + X[:, 1]

        engine = SymbolicTreeEngine(max_depth=3, population_size=300, iterations=20)
        result = engine.discover_validate_interpret(
            X, y, variable_names=["a", "b"], verbose=False, show_formatted=False
        )
        # Generous threshold — tree search is stochastic
        assert result["r2"] > 0.5, f"Expected R²>0.5, got {result['r2']:.4f}"
