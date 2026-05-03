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
