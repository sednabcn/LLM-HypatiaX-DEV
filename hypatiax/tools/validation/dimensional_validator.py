#!/usr/bin/env python3
"""
Layer 1 — Dimensional analysis validator for HypatiaX symbolic regression.

Validates that candidate mathematical expressions are dimensionally consistent
using Pint for unit tracking and SymPy for expression tree traversal.

Rejects expressions where physically incompatible units are added or subtracted,
where function arguments (log, exp, trig) receive quantities with physical units,
or where numerical stability limits are exceeded (exponent overflow, division by
a symbolic zero).

Key design decisions
--------------------
- Unit parsing failures for individual variables degrade the score but do not
  abort validation, because PySR may produce expressions with partially-known
  variable sets.
- Simplification failures fall back to the unsimplified tree silently; this is
  intentional because SymPy's simplify() can time-out on large expressions.
- The outer try/except in validate() is a last-resort safety net; all expected
  error paths are handled explicitly above it.

Designed to be called by EnsembleValidator as the first validation layer.
Can also be used standalone via the validate_expression() convenience function.

Dependencies
------------
    pint >= 0.20
    sympy >= 1.12
    numpy >= 1.24
"""

import logging
import random
import re
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import sympy as sp
from pint import UnitRegistry

# ---------------------------------------------------------------------------
# Module-level reproducibility seeds.
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def safe_sympify(
    expression_str: str,
    variable_names: Optional[List[str]] = None,
) -> sp.Expr:
    """Parse *expression_str* into a SymPy expression with Pint isolation.

    Pint registers its own unit symbols in the global SymPy namespace, which
    can corrupt parsing.  This helper builds an isolated local dictionary of
    plain SymPy symbols so that Pint units never leak into the expression tree.

    Args:
        expression_str: Mathematical expression as a Python-syntax string.
        variable_names: Variable names to pre-declare as real SymPy symbols.
            If omitted, SymPy infers symbols from the expression.

    Returns:
        Parsed SymPy expression.

    Raises:
        ValueError: If the expression cannot be parsed by any strategy.
    """
    if not isinstance(expression_str, str):
        expression_str = str(expression_str)

    local_dict: Dict[str, Any] = {}
    if variable_names:
        for var in variable_names:
            local_dict[var] = sp.Symbol(var, real=True)

    local_dict.update(
        {
            "exp": sp.exp,
            "log": sp.log,
            "ln": sp.log,
            "log10": lambda x: sp.log(x, 10),
            "sqrt": sp.sqrt,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "abs": sp.Abs,
            # PySR custom operator aliases — map Julia operator names to their
            # SymPy equivalents so expressions like safe_asin((n1/n2)*sin(theta1))
            # parse correctly without NameError in the validator layers.
            "safe_asin":   sp.asin,
            "safe_acos":   sp.acos,
            "asin_of_sin": sp.asin,
            "acos_of_cos": sp.acos,
            "atan_of_tan": sp.atan,
            # Standard inverse trig (sometimes used directly in expressions)
            "asin":  sp.asin,
            "acos":  sp.acos,
            "atan":  sp.atan,
            "arcsin": sp.asin,
            "arccos": sp.acos,
            "arctan": sp.atan,
        }
    )

    # First attempt: evaluate=False preserves structure (preferred).
    try:
        return sp.sympify(expression_str, locals=local_dict, evaluate=False)
    except Exception:
        pass

    # Second attempt: evaluate=True for expressions SymPy auto-simplifies.
    try:
        return sp.sympify(expression_str, locals=local_dict, evaluate=True)
    except Exception as exc:
        raise ValueError(
            f"Could not parse expression '{expression_str}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------

class DimensionalValidator:
    """Layer 1 dimensional validator — rejects expressions with unit errors.

    Uses Pint for unit arithmetic and SymPy for expression tree traversal.
    Walks the expression tree recursively to infer the output unit, checking
    that:

    - Addition/subtraction operands share the same physical dimension.
    - log/exp/trig arguments are dimensionless (or dimensionless ratios).
    - Exponents are dimensionless numbers.
    - Numerical stability limits are respected (exponent magnitude, division
      by symbolic variables).

    Scoring
    -------
    Starts at 100.0.  Each detected error or warning reduces the score by a
    fixed penalty.  The final score is clamped to [0, 100].

    History
    -------
    The last *max_history* results are stored in ``validation_history`` as a
    bounded deque so memory use is predictable in long-running campaigns.

    Constants
    ---------
    MAX_SAFE_EXPONENT : float
        Exponents beyond this magnitude raise a hard error.
    """

    MAX_SAFE_VALUE = 1e308
    MIN_SAFE_VALUE = 1e-308
    MAX_SAFE_EXPONENT = 100
    EPSILON = 1e-10

    def __init__(self, max_history: Optional[int] = 1000) -> None:
        """Initialise the validator with a Pint unit registry.

        Args:
            max_history: Maximum number of results to retain in
                ``validation_history``.  Pass ``None`` for an unbounded list
                (not recommended for long campaigns).
        """
        self.ureg = UnitRegistry()

        # Register finance units that Pint does not include by default.
        # This is expected to succeed on a fresh registry; log a warning if it
        # fails so the caller knows custom units are unavailable.
        try:
            self.ureg.define("USD = [currency]")
        except Exception as exc:
            logger.warning(
                "Could not register custom unit 'USD' in Pint registry: %s", exc
            )

        if max_history is not None:
            self.validation_history: Any = deque(maxlen=max_history)
        else:
            self.validation_history = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        expression_str: str,
        variable_units: Dict[str, str],
        variable_bounds: Optional[Dict[str, tuple]] = None,
        constant_info: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Validate dimensional consistency of *expression_str*.

        Args:
            expression_str: Mathematical expression using Python syntax.
            variable_units: Mapping of variable name → Pint unit string, e.g.
                ``{"v": "m/s", "m": "kg"}``.  Use ``"dimensionless"`` or ``""``
                for pure numbers.
            variable_bounds: Optional mapping of variable name → ``(low, high)``
                tuple.  Used by the numerical stability checker to tighten
                overflow analysis.  Not required for unit checking.
            constant_info: Optional mapping of known physical constants and
                their values.  Reserved for future absorbed-constant detection.

        Returns:
            Result dictionary with keys:

            - ``valid`` (bool): True if no hard errors were found.
            - ``score`` (float): Quality score in [0, 100].
            - ``errors`` (list[str]): Fatal issues.
            - ``warnings`` (list[str]): Non-fatal advisories.
            - ``dimensionally_consistent`` (bool)
            - ``variable_dimensions`` (dict): Inferred unit per variable.
            - ``inferred_output_unit`` (str | None): Unit of the expression result.
            - ``numerical_stability`` (dict): Stability sub-report.
            - ``overflow_risks`` (list)
            - ``simplified_expression`` (str | None)
        """
        result: Dict[str, Any] = {
            "valid": True,
            "score": 100.0,
            "errors": [],
            "warnings": [],
            "dimensionally_consistent": True,
            "variable_dimensions": {},
            "inferred_output_unit": None,
            "numerical_stability": {"stable": True, "issues": []},
            "overflow_risks": [],
            "simplified_expression": None,
        }

        if not expression_str or not expression_str.strip():
            result["valid"] = False
            result["score"] = 0.0
            result["errors"].append("Empty expression")
            logger.debug("Validation rejected: empty expression")
            self._add_to_history(result)
            return result

        try:
            # --- 1. Parse variable units --------------------------------
            var_units_map: Dict[str, Any] = {}
            for var_name, unit_str in variable_units.items():
                try:
                    normalised = str(unit_str).strip().lower()
                    if not unit_str or normalised in ("dimensionless", "none", ""):
                        unit = self.ureg.dimensionless
                    else:
                        unit = self.ureg.parse_units(unit_str)
                    var_units_map[var_name] = unit
                    result["variable_dimensions"][var_name] = (
                        "dimensionless"
                        if unit == self.ureg.dimensionless
                        else str(unit)
                    )
                except Exception as exc:
                    # A bad unit string is a warning, not a hard failure.
                    # The variable is treated as dimensionless so parsing can
                    # continue; this avoids blocking the entire validation when
                    # one variable has an unrecognised unit.
                    msg = f"Unit parse warning for '{var_name}' ('{unit_str}'): {exc}"
                    result["warnings"].append(msg)
                    logger.warning(msg)
                    var_units_map[var_name] = self.ureg.dimensionless
                    result["score"] -= 5

            # --- 2. Parse expression ------------------------------------
            try:
                expr = safe_sympify(expression_str, list(variable_units.keys()))
            except ValueError as exc:
                result["errors"].append(f"Parse error: {exc}")
                result["valid"] = False
                result["score"] = 0.0
                logger.debug("Validation rejected: parse error — %s", exc)
                self._add_to_history(result)
                return result

            # --- 3. Simplify (best-effort; timeout-safe fallback) -------
            try:
                simplified = sp.simplify(expr)
                result["simplified_expression"] = str(simplified)
            except Exception as exc:
                # SymPy simplification can time-out or raise on exotic
                # expressions.  Fall back to the unsimplified tree.
                logger.debug(
                    "Simplification failed for '%s', proceeding with raw tree: %s",
                    expression_str, exc,
                )
                simplified = expr
                result["simplified_expression"] = str(expr)

            # --- 4. Unit inference --------------------------------------
            unit_result = self._infer_units_correctly(simplified, var_units_map)

            result["inferred_output_unit"] = unit_result["unit_str"]
            result["dimensionally_consistent"] = unit_result["consistent"]
            result["errors"].extend(unit_result["errors"])
            result["warnings"].extend(unit_result["warnings"])
            result["score"] -= unit_result["penalty"]

            if not unit_result["consistent"]:
                result["valid"] = False

            # --- 5. Numerical stability ---------------------------------
            stability = self._check_numerical_stability(
                simplified, var_units_map, variable_bounds
            )
            result["numerical_stability"] = stability
            result["warnings"].extend(stability["warnings"])
            result["errors"].extend(stability["errors"])
            result["score"] -= stability["penalty"]

            if not stability["stable"]:
                result["valid"] = False

        except Exception as exc:
            # Last-resort catch.  All expected error paths are handled above;
            # reaching here indicates an unexpected SymPy or Pint internal error.
            logger.exception(
                "Unexpected error during validation of '%s'", expression_str
            )
            result["valid"] = False
            result["score"] = 0.0
            result["errors"].append(f"Unexpected validation error: {exc}")

        result["score"] = max(0.0, min(100.0, result["score"]))
        logger.debug(
            "Validation complete for '%s': valid=%s score=%.1f",
            expression_str, result["valid"], result["score"],
        )
        self._add_to_history(result)
        return result

    # ------------------------------------------------------------------
    # Internal — unit inference
    # ------------------------------------------------------------------

    def _infer_units_correctly(
        self,
        expr: sp.Expr,
        var_units_map: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Walk the SymPy expression tree and infer the output unit.

        This is the core of the dimensional validator.  It handles addition,
        multiplication, division (encoded as ``Pow(base, -1)``), powers,
        logarithms, exponentials, square roots, and trigonometric functions.

        Returns:
            Dict with keys ``unit_str``, ``consistent`` (bool), ``errors``,
            ``warnings``, and ``penalty`` (float).
        """
        infer_result: Dict[str, Any] = {
            "unit_str": None,
            "consistent": True,
            "errors": [],
            "warnings": [],
            "penalty": 0,
        }

        def get_unit(node: sp.Expr) -> Any:
            """Recursively compute the Pint unit of *node*."""

            if node.is_Number:
                return self.ureg.dimensionless

            if isinstance(node, sp.Symbol):
                return var_units_map.get(str(node), self.ureg.dimensionless)

            # Addition: all non-dimensionless terms must share the same unit.
            if isinstance(node, sp.Add):
                term_units = []
                for term in node.args:
                    unit = get_unit(term)
                    if unit != self.ureg.dimensionless:
                        term_units.append((term, unit))

                if not term_units:
                    return self.ureg.dimensionless

                base_term, base_unit = term_units[0]
                for term, unit in term_units[1:]:
                    if not self._units_equivalent(base_unit, unit):
                        infer_result["errors"].append(
                            f"Incompatible units in addition: {base_unit} vs {unit}"
                        )
                        infer_result["consistent"] = False
                        infer_result["penalty"] += 20

                return base_unit

            # Multiplication / division (division = Pow(base, -1)).
            if isinstance(node, sp.Mul):
                result_unit = self.ureg.dimensionless
                for factor in node.args:
                    if isinstance(factor, sp.Pow) and factor.exp == -1:
                        divisor_unit = get_unit(factor.base)
                        try:
                            result_unit = result_unit / divisor_unit
                        except Exception as exc:
                            infer_result["warnings"].append(
                                f"Division unit issue: {exc}"
                            )
                    else:
                        factor_unit = get_unit(factor)
                        try:
                            result_unit = result_unit * factor_unit
                        except Exception as exc:
                            infer_result["warnings"].append(
                                f"Multiplication unit issue: {exc}"
                            )
                return result_unit

            # Power: base^exponent.
            if isinstance(node, sp.Pow):
                base_unit = get_unit(node.base)

                if not node.exp.is_Number:
                    exp_unit = get_unit(node.exp)
                    if exp_unit != self.ureg.dimensionless:
                        infer_result["errors"].append(
                            f"Exponent must be dimensionless, got: {node.exp}"
                        )
                        infer_result["consistent"] = False
                        infer_result["penalty"] += 15
                        return self.ureg.dimensionless

                if base_unit == self.ureg.dimensionless:
                    return self.ureg.dimensionless

                try:
                    return base_unit ** float(node.exp)
                except Exception as exc:
                    infer_result["warnings"].append(f"Power unit issue: {exc}")
                    return self.ureg.dimensionless

            # Functions.
            if isinstance(node, sp.Function):
                fname = node.func.__name__.lower()
                arg_unit = get_unit(node.args[0])

                if fname in ("log", "ln", "log10"):
                    # Allow dimensionless ratios: log(A/B) where units cancel.
                    if self._is_ratio_with_same_units(node.args[0], var_units_map):
                        return self.ureg.dimensionless
                    if arg_unit != self.ureg.dimensionless:
                        infer_result["errors"].append(
                            f"log() requires a dimensionless argument, got {arg_unit}"
                        )
                        infer_result["consistent"] = False
                        infer_result["penalty"] += 15
                    return self.ureg.dimensionless

                if fname == "exp":
                    if arg_unit != self.ureg.dimensionless:
                        infer_result["errors"].append(
                            f"exp() requires a dimensionless argument, got {arg_unit}"
                        )
                        infer_result["consistent"] = False
                        infer_result["penalty"] += 15
                    return self.ureg.dimensionless

                if fname == "sqrt":
                    try:
                        return arg_unit ** 0.5
                    except Exception as exc:
                        infer_result["warnings"].append(f"sqrt unit issue: {exc}")
                        return self.ureg.dimensionless

                if fname in ("sin", "cos", "tan"):
                    if arg_unit != self.ureg.dimensionless:
                        infer_result["warnings"].append(
                            f"{fname}() expects dimensionless (radians), got {arg_unit}"
                        )
                        infer_result["penalty"] += 5
                    return self.ureg.dimensionless

            # Unknown node type: treat as dimensionless (conservative).
            return self.ureg.dimensionless

        try:
            output_unit = get_unit(expr)
            infer_result["unit_str"] = (
                "dimensionless"
                if output_unit == self.ureg.dimensionless
                else str(output_unit)
            )
        except Exception as exc:
            infer_result["warnings"].append(f"Unit inference failed: {exc}")
            infer_result["unit_str"] = "unknown"
            infer_result["penalty"] += 10
            logger.debug("Unit inference error for expression: %s", exc)

        return infer_result

    # ------------------------------------------------------------------
    # Internal — unit utilities
    # ------------------------------------------------------------------

    def _units_equivalent(self, u1: Any, u2: Any) -> bool:
        """Return True if *u1* and *u2* share the same physical dimensionality.

        Compares Pint dimensionality dicts.  Falls back to string comparison
        if Pint dimensionality attributes are unavailable.
        """
        try:
            d1 = getattr(u1, "dimensionality", None)
            d2 = getattr(u2, "dimensionality", None)
            if d1 is None or d2 is None:
                return str(u1) == str(u2)
            return d1 == d2
        except Exception as exc:
            logger.debug("Unit equivalence check failed: %s", exc)
            return False

    def _is_ratio_with_same_units(
        self,
        expr: sp.Expr,
        var_units_map: Dict[str, Any],
    ) -> bool:
        """Return True if *expr* is a ratio A/B where A and B share units.

        Handles simple patterns: ``A/B``, ``(A*C)/(B*D)``.  Returns False for
        complex sub-expressions to avoid false negatives.
        """
        try:
            if not isinstance(expr, sp.Mul):
                return False

            numer_factors = []
            denom_factors = []
            for factor in expr.args:
                if isinstance(factor, sp.Pow) and factor.exp == -1:
                    denom_factors.append(factor.base)
                else:
                    numer_factors.append(factor)

            if not numer_factors or not denom_factors:
                return False

            def combined_unit(factors: list) -> Any:
                unit = self.ureg.dimensionless
                for f in factors:
                    if isinstance(f, sp.Symbol):
                        unit = unit * var_units_map.get(str(f), self.ureg.dimensionless)
                    elif f.is_Number:
                        pass  # numbers are dimensionless
                    else:
                        return None  # complex sub-expression — give up
                return unit

            nu = combined_unit(numer_factors)
            du = combined_unit(denom_factors)
            if nu is not None and du is not None:
                return self._units_equivalent(nu, du)
            return False

        except Exception as exc:
            logger.debug("Ratio unit check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal — numerical stability
    # ------------------------------------------------------------------

    def _check_numerical_stability(
        self,
        expr: sp.Expr,
        var_units_map: Dict[str, Any],
        variable_bounds: Optional[Dict[str, tuple]],
    ) -> Dict[str, Any]:
        """Scan the expression tree for numerical stability risks.

        Checks performed:

        1. Exponents beyond MAX_SAFE_EXPONENT (hard error).
        2. Large-but-safe exponents > 10 (warning).
        3. Exponential functions without bounded arguments (warning).
        4. Division by a bare symbol (warning).
        5. Products of more than four symbols (overflow warning).

        Returns:
            Dict with keys ``stable`` (bool), ``issues``, ``warnings``,
            ``errors``, and ``penalty`` (float).
        """
        result: Dict[str, Any] = {
            "stable": True,
            "issues": [],
            "warnings": [],
            "errors": [],
            "penalty": 0,
        }

        # FIX: merged 4 separate preorder_traversal passes into one (was O(4N)).
        # Also added exp_warning_issued flag to suppress duplicate exp() warnings
        # when the same expression contains multiple exp() nodes (the original
        # emitted one warning *per node* on every traversal pass).
        exp_warning_issued = False
        long_mul_checked = False

        for node in sp.preorder_traversal(expr):

            # Check 1 & 2: Exponent magnitude
            if isinstance(node, sp.Pow):
                base, exp = node.args
                if exp.is_Number:
                    exp_val = float(exp)
                    if abs(exp_val) > self.MAX_SAFE_EXPONENT:
                        result["errors"].append(
                            f"Exponent {exp_val} exceeds safe limit "
                            f"({self.MAX_SAFE_EXPONENT})"
                        )
                        result["stable"] = False
                        result["penalty"] += 30
                    elif abs(exp_val) > 10:
                        result["warnings"].append(
                            f"Large exponent {exp_val} — verify variable bounds"
                        )
                        result["penalty"] += 5

            # Check 3: exp() argument boundedness (one warning per expression)
            if (
                not exp_warning_issued
                and isinstance(node, sp.Function)
                and node.func.__name__ == "exp"
            ):
                result["warnings"].append(
                    "exp() detected — verify argument remains bounded"
                )
                result["penalty"] += 3
                exp_warning_issued = True

            # Checks 4 & 5: Mul-level division and long product chain
            if isinstance(node, sp.Mul):
                for factor in node.args:
                    if isinstance(factor, sp.Pow) and factor.exp == -1:
                        divisor = factor.base
                        if isinstance(divisor, sp.Symbol):
                            result["warnings"].append(
                                f"Division by symbol '{divisor}' — ensure {divisor} ≠ 0"
                            )
                            result["penalty"] += 3

                if not long_mul_checked:
                    factors = [f for f in node.args if isinstance(f, sp.Symbol)]
                    if len(factors) > 4:
                        result["warnings"].append(
                            f"Product of {len(factors)} symbols — check for overflow"
                        )
                        result["penalty"] += 5
                    long_mul_checked = True  # only flag the first long Mul found

        return result

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def _add_to_history(self, result: Dict[str, Any]) -> None:
        """Append *result* to the bounded validation history."""
        self.validation_history.append(result)

    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Return a snapshot of the validation history as a plain list."""
        return list(self.validation_history)

    def clear_history(self) -> None:
        """Clear all stored validation results."""
        if isinstance(self.validation_history, deque):
            self.validation_history.clear()
        else:
            self.validation_history = []


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def validate_expression(
    expression_str: str,
    variable_units: Dict[str, str],
    variable_bounds: Optional[Dict[str, tuple]] = None,
    constant_info: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper: create a one-shot DimensionalValidator and validate.

    Args:
        expression_str: Mathematical expression string.
        variable_units: Mapping of variable name → Pint unit string.
        variable_bounds: Optional bounds dict forwarded to the stability checker.
        constant_info: Optional known-constants dict (reserved for future use).

    Returns:
        Validation result dict — see :meth:`DimensionalValidator.validate`.
    """
    validator = DimensionalValidator()
    return validator.validate(
        expression_str, variable_units, variable_bounds, constant_info
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 80)
    print("DIMENSIONAL VALIDATOR v9.0 — SELF-TEST SUITE")
    print("=" * 80)

    test_cases = [
        {
            "name": "Kinetic Energy",
            "expr": "v*m*v*0.5",
            "units": {"m": "kg", "v": "m/s"},
            "should_pass": True,
        },
        {
            "name": "Ohm's Law",
            "expr": "I*R",
            "units": {"I": "A", "R": "ohm"},
            "should_pass": True,
        },
        {
            "name": "Bernoulli's Equation",
            "expr": "P + g*h*rho + v*v*rho*0.5",
            "units": {
                "P": "Pa", "g": "m/s**2", "h": "m",
                "rho": "kg/m**3", "v": "m/s",
            },
            "should_pass": True,
        },
        {
            "name": "Logistic Growth",
            "expr": "N*(r - N*r/K)",
            "units": {"N": "dimensionless", "r": "1/s", "K": "dimensionless"},
            "should_pass": True,
        },
        {
            "name": "Price Elasticity",
            "expr": "delta_Q/Q / (delta_P/P)",
            "units": {
                "delta_Q": "dimensionless", "Q": "dimensionless",
                "delta_P": "dimensionless", "P": "dimensionless",
            },
            "should_pass": True,
        },
        {
            "name": "Henderson-Hasselbalch",
            "expr": "pKa + log(A_minus/HA)",
            "units": {"pKa": "dimensionless", "A_minus": "mol/L", "HA": "mol/L"},
            "should_pass": True,
        },
        {
            "name": "Invalid: pressure + velocity",
            "expr": "P + v",
            "units": {"P": "Pa", "v": "m/s"},
            "should_pass": False,
        },
        {
            "name": "Invalid: log of dimensioned quantity",
            "expr": "log(P)",
            "units": {"P": "Pa"},
            "should_pass": False,
        },
    ]

    passed = failed = 0
    for i, tc in enumerate(test_cases, 1):
        result = validate_expression(tc["expr"], tc["units"])
        is_valid = result["valid"] and result["score"] >= 70
        ok = is_valid == tc["should_pass"]
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"\nTest {i}: {tc['name']}")
        print(f"  Expression : {tc['expr']}")
        print(f"  Valid={result['valid']}  Score={result['score']:.1f}  {status}")
        if not ok:
            print(f"  Errors   : {result['errors']}")
            print(f"  Warnings : {result['warnings'][:2]}")
            failed += 1
        else:
            passed += 1

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed}/{len(test_cases)} passed")
    print("✅ ALL TESTS PASSED" if failed == 0 else f"❌ {failed} test(s) FAILED")
    print("=" * 80)
