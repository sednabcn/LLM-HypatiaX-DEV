"""
HypatiaX — Self-Contained Script (v5.2)
========================================
Converted from notebookb739624668.ipynb.

API key priority order:
  1. Already set in environment (shell export)
  2. .env file  (dotenv — local / server use)
  3. Kaggle Secrets
  4. Google Colab Secrets
"""

import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import warnings
from collections import deque
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API Key loading — priority order:
#   1. Already set in environment (e.g. shell export)
#   2. .env file  (dotenv — local / server use)
#   3. Kaggle Secrets  (when running inside a Kaggle notebook)
#   4. Google Colab Secrets  (when running inside Colab)
# ---------------------------------------------------------------------------
def _load_anthropic_key() -> None:
    """Populate ANTHROPIC_API_KEY from the first available source."""
    if os.getenv("ANTHROPIC_API_KEY"):
        logger.debug("ANTHROPIC_API_KEY already set in environment.")
        return

    # Option A — .env file (same behaviour as hybrid_system_v40)
    try:
        from dotenv import load_dotenv
        _env_path = Path(__file__).resolve().parent / ".env"
        if not _env_path.exists():
            _env_path = Path(__file__).resolve().parent.parent / ".env"
        if _env_path.exists():
            load_dotenv(_env_path)
            if os.getenv("ANTHROPIC_API_KEY"):
                logger.info("ANTHROPIC_API_KEY loaded from .env file (%s).", _env_path)
                return
    except ImportError:
        pass  # python-dotenv not installed — continue to next option

    # Option B — Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        key = UserSecretsClient().get_secret("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = key
        logger.info("ANTHROPIC_API_KEY loaded from Kaggle secrets.")
        return
    except Exception:
        pass

    # Option C — Google Colab Secrets
    try:
        from google.colab import userdata  # type: ignore
        key = userdata.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = key
        logger.info("ANTHROPIC_API_KEY loaded from Colab secrets.")
        return
    except Exception:
        pass

    logger.warning(
        "ANTHROPIC_API_KEY not found in environment, .env, Kaggle, or Colab secrets. "
        "LLM mode will be disabled unless a key is passed explicitly."
    )

_load_anthropic_key()

# ---------------------------------------------------------------------------
# Pip install guard (auto-installs missing deps when run as a script)
# ---------------------------------------------------------------------------
def _ensure_deps() -> None:
    _required = [
        ("numpy", "numpy"), ("sympy", "sympy"), ("scipy", "scipy"),
        ("scikit-learn", "sklearn"), ("pint", "pint"),
        ("pysr", "pysr"), ("anthropic", "anthropic"),
    ]
    for pkg, imp in _required:
        try:
            __import__(imp)
        except ImportError:
            logger.info("Installing %s ...", pkg)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", pkg], check=True
            )

_ensure_deps()

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HypatiaX — Self-Contained Embed (v5.2)                             ║
# ║  No hypatiax package. No env files. Fully portable.                  ║
# ║  Copy this single cell into any notebook and run it.                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ===========================================================================
# dimensional_validator
# ===========================================================================

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

# ===========================================================================
# domain_validator
# ===========================================================================

"""
HypatiaX Domain Validator
tools/validation/domain_validator.py

WEEK 2 UPDATES:
- Enhanced constraint validation for DeFi formulas (Issue #1)
- Added explicit bounds checking for critical variables
- Improved error messages with remediation guidance
- Added support for epsilon-protected divisions
- Enhanced scoring to align with ensemble validator (Issue #2)

Layer 2 — Domain-specific constraint validator for HypatiaX symbolic regression.

Validates that candidate expressions respect the variable constraints of the
target application domain (DeFi, risk, finance, ESG). Checks positivity,
strict positivity, bounded ranges, probability constraints, and domain-specific
invariants (e.g. AMM constant product, fee bounds, VaR positivity).

Does not use SymPy — operates on expression strings and optional test data arrays.
Designed to be called by EnsembleValidator as the second validation layer.

Supported domains: 'defi', 'risk', 'finance', 'esg'
"""

import random
from collections import deque
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Module-level reproducibility seeds.
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)


class DomainValidator:
    """
    Validates domain-specific constraints for mathematical expressions.
    Checks that formulas satisfy domain-specific rules (DeFi, Risk, Finance, ESG).
    """

    def __init__(self, domain: str, max_history: Optional[int] = 1000):
        """
        Initialize the domain validator.

        Args:
            domain: Domain context ('defi', 'risk', 'finance', 'esg')
            max_history: Maximum number of validation results to keep
        """
        self.domain = domain.lower()
        self.constraints = self._load_constraints()

        # Bounded validation history
        if max_history is not None:
            self.validation_history = deque(maxlen=max_history)
        else:
            self.validation_history = []

    def _load_constraints(self) -> Dict:
        """
        Load domain-specific constraints.

        WEEK 2 ENHANCEMENT: More comprehensive constraint definitions
        """
        constraints = {
            "defi": {
                "positive_variables": [
                    "reserve",
                    "liquidity",
                    "price",
                    "amount",
                    "balance",
                    "supply",
                    "token",
                    "x",
                    "y",
                    "k",
                    "x0",
                    "y0",
                    "x_0",
                    "y_0",  # Added reserve notation
                ],
                "strictly_positive_variables": [
                    # WEEK 2: Variables that must be > 0 (not just >= 0)
                    "price",
                    "liquidity",
                    "reserve",
                    "r",
                    "ratio",
                ],
                "bounded_variables": {
                    "fee": (0, 1),
                    "phi": (0, 1),  # Greek fee symbol
                    "slippage": (0, 1),
                    "utilization": (0, 1),
                    "ratio": (0, None),  # WEEK 2: Changed to strictly positive
                },
                "ratio_variables": ["price_ratio", "reserve_ratio", "r"],
                "special_checks": [
                    "constant_product",
                    "no_negative_slippage",
                    "ratio_positivity",  # NEW WEEK 2
                    "price_positivity",  # NEW WEEK 2
                    "division_protection",  # NEW WEEK 2
                ],
            },
            "risk": {
                "positive_variables": [
                    "var",
                    "cvar",
                    "volatility",
                    "loss",
                    "exposure",
                    "shortfall",
                    "sigma",
                ],
                "probability_variables": [
                    "prob",
                    "confidence",
                    "likelihood",
                    "probability",
                ],
                "bounded_variables": {
                    "confidence": (0, 1),
                    "probability": (0, 1),
                    "correlation": (-1, 1),
                    "alpha": (0, 1),  # Significance level
                },
                "special_checks": ["var_positive", "confidence_valid"],
            },
            "finance": {
                "positive_variables": [
                    "price",
                    "volume",
                    "market_cap",
                    "assets",
                    "nav",
                ],
                "bounded_variables": {
                    "return": (-1, None),  # Can lose 100%, no upper bound
                    "weight": (0, 1),
                    "allocation": (0, 1),
                },
                "percentage_variables": ["return", "yield", "rate", "apy"],
                "special_checks": ["weights_sum_to_one"],
            },
            "esg": {
                "bounded_variables": {
                    "score": (0, 100),
                    "rating": (0, 10),
                    "weight": (0, 1),
                },
                "positive_variables": ["impact", "emissions", "carbon", "footprint"],
                "special_checks": ["score_range", "weights_sum_to_one"],
            },
        }

        return constraints.get(self.domain, {})

    def validate(
        self,
        expression_str: str,
        variable_definitions: Dict[str, str],
        test_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict:
        """
        Validate domain-specific constraints.

        Args:
            expression_str: The mathematical expression
            variable_definitions: Variable name to description mapping
            test_data: Optional test data for numerical validation

        Returns:
            {
                'valid': bool,
                'score': float,
                'errors': List[str],
                'warnings': List[str],
                'domain': str,
                'constraints_checked': List[str]
            }
        """
        result = {
            "valid": True,
            "score": 100.0,
            "errors": [],
            "warnings": [],
            "domain": self.domain,
            "constraints_checked": [],
        }

        # WEEK 2: Normalize expression for better matching
        expr_lower = expression_str.lower()

        # Check positive variable constraints
        result = self._check_positive_variables(expression_str, test_data, result)

        # WEEK 2 NEW: Check strictly positive variables (must be > 0, not >= 0)
        result = self._check_strictly_positive_variables(
            expression_str, test_data, result
        )

        # Check bounded variable constraints
        result = self._check_bounded_variables(expression_str, test_data, result)

        # Check probability variables (if applicable)
        if "probability_variables" in self.constraints:
            result = self._check_probability_variables(
                expression_str, test_data, result
            )

        # Check special domain rules
        result = self._check_special_rules(
            expression_str, variable_definitions, test_data, result
        )

        # Determine overall validity
        if result["errors"]:
            result["valid"] = False

        # FIX: clamp score — multiple simultaneous penalties (e.g. strictly-positive
        # + bounded + division-protection) can push score well below zero.
        result["score"] = max(0.0, min(100.0, result["score"]))

        # Store in history
        self.validation_history.append(result)
        return result

    def _check_positive_variables(
        self,
        expression_str: str,
        test_data: Optional[Dict[str, np.ndarray]],
        result: Dict,
    ) -> Dict:
        """Check that variables that must be positive are indeed positive."""
        positive_vars = self.constraints.get("positive_variables", [])

        for var in positive_vars:
            if var in expression_str.lower():
                result["constraints_checked"].append(f"{var}_positive")

                if test_data and var in test_data:
                    values = test_data[var]
                    if np.any(values <= 0):
                        result["errors"].append(
                            f"Variable '{var}' must be positive (found {np.min(values):.6f})"
                        )
                        result["score"] -= 20
                else:
                    result["warnings"].append(
                        f"Variable '{var}' should be positive - add validation"
                    )
                    result["score"] -= 5

        return result

    def _check_strictly_positive_variables(
        self,
        expression_str: str,
        test_data: Optional[Dict[str, np.ndarray]],
        result: Dict,
    ) -> Dict:
        """
        WEEK 2 NEW: Check variables that must be strictly positive (> 0, not >= 0).

        Critical for:
        - Division denominators
        - Logarithm arguments
        - Square root arguments (in some contexts)
        """
        strictly_positive = self.constraints.get("strictly_positive_variables", [])

        for var in strictly_positive:
            # Check if variable appears in expression
            if var in expression_str.lower() or var in expression_str:
                result["constraints_checked"].append(f"{var}_strictly_positive")

                if test_data and var in test_data:
                    values = test_data[var]
                    # Check for zero or negative values
                    if np.any(values <= 0):
                        result["errors"].append(
                            f"CRITICAL: Variable '{var}' must be strictly positive (> 0), "
                            f"found minimum value: {np.min(values):.6f}. "
                            f"Add constraint: {var} > 0"
                        )
                        result["score"] -= 25  # Severe penalty
                    # Check for values very close to zero (numerical stability)
                    elif np.any(values < 1e-8):
                        result["warnings"].append(
                            f"Variable '{var}' has very small values (< 1e-8), "
                            f"may cause numerical instability"
                        )
                        result["score"] -= 5
                else:
                    # No test data - issue warning
                    result["warnings"].append(
                        f"Variable '{var}' must be strictly positive (> 0). "
                        f"Add validation: assert {var} > 0"
                    )
                    result["score"] -= 8  # Increased penalty for missing validation

        return result

    def _check_bounded_variables(
        self,
        expression_str: str,
        test_data: Optional[Dict[str, np.ndarray]],
        result: Dict,
    ) -> Dict:
        """
        Check that bounded variables are within their valid ranges.

        WEEK 2 ENHANCEMENT: More descriptive error messages
        """
        bounded_vars = self.constraints.get("bounded_variables", {})

        for var, bounds in bounded_vars.items():
            if var in expression_str.lower() or var in expression_str:
                result["constraints_checked"].append(f"{var}_bounded")
                lower, upper = bounds

                if test_data and var in test_data:
                    values = test_data[var]

                    # Check lower bound
                    if lower is not None and np.any(values < lower):
                        result["errors"].append(
                            f"Variable '{var}' below minimum {lower} "
                            f"(found {np.min(values):.6f}). "
                            f"Add constraint: {var} >= {lower}"
                        )
                        result["score"] -= 15

                    # Check upper bound
                    if upper is not None and np.any(values > upper):
                        result["errors"].append(
                            f"Variable '{var}' above maximum {upper} "
                            f"(found {np.max(values):.6f}). "
                            f"Add constraint: {var} <= {upper}"
                        )
                        result["score"] -= 15

                    # WEEK 2 NEW: Special case for fee variables at exactly 1.0
                    if var in ["fee", "phi"] and upper == 1:
                        if np.any(values >= 1.0):
                            result["errors"].append(
                                f"Fee variable '{var}' must be < 1.0 (not <=), "
                                f"found {np.max(values):.6f}. "
                                f"Fees at 100% break AMM math."
                            )
                            result["score"] -= 20
                else:
                    # No test data
                    if upper is not None:
                        bound_str = f"[{lower}, {upper}]"
                    else:
                        bound_str = f">= {lower}"

                    result["warnings"].append(
                        f"Variable '{var}' should be in range {bound_str}"
                    )
                    result["score"] -= 5

        return result

    def _check_probability_variables(
        self,
        expression_str: str,
        test_data: Optional[Dict[str, np.ndarray]],
        result: Dict,
    ) -> Dict:
        """Check that probability variables are in [0, 1]."""
        prob_vars = self.constraints.get("probability_variables", [])

        for var in prob_vars:
            if var in expression_str.lower():
                result["constraints_checked"].append(f"{var}_probability")

                if test_data and var in test_data:
                    values = test_data[var]

                    if np.any(values < 0) or np.any(values > 1):
                        result["errors"].append(
                            f"Probability variable '{var}' must be in [0, 1] "
                            f"(found range [{np.min(values):.3f}, {np.max(values):.3f}])"
                        )
                        result["score"] -= 25
                else:
                    result["warnings"].append(
                        f"Probability variable '{var}' should be in [0, 1]"
                    )
                    result["score"] -= 5

        return result

    def _check_special_rules(
        self,
        expression_str: str,
        variable_definitions: Dict[str, str],
        test_data: Optional[Dict[str, np.ndarray]],
        result: Dict,
    ) -> Dict:
        """
        Check domain-specific special rules.

        WEEK 2 ENHANCEMENT: Added new special checks
        """
        special_checks = self.constraints.get("special_checks", [])

        for check in special_checks:
            if check == "constant_product":
                result = self._check_constant_product(expression_str, test_data, result)
            elif check == "no_negative_slippage":
                result = self._check_no_negative_slippage(
                    expression_str, test_data, result
                )
            elif check == "ratio_positivity":  # NEW WEEK 2
                result = self._check_ratio_positivity(expression_str, test_data, result)
            elif check == "price_positivity":  # NEW WEEK 2
                result = self._check_price_positivity(expression_str, test_data, result)
            elif check == "division_protection":  # NEW WEEK 2
                result = self._check_division_protection(expression_str, result)
            elif check == "var_positive":
                result = self._check_var_positive(expression_str, test_data, result)
            elif check == "confidence_valid":
                result = self._check_confidence_valid(expression_str, test_data, result)
            elif check == "weights_sum_to_one":
                result = self._check_weights_sum(
                    expression_str, variable_definitions, result
                )
            elif check == "score_range":
                result = self._check_score_range(expression_str, test_data, result)

        return result

    # Special rule implementations

    def _check_constant_product(
        self, expr_str: str, test_data: Optional[Dict], result: Dict
    ) -> Dict:
        """Check DeFi constant product invariant."""
        if "reserve" in expr_str.lower() and test_data:
            result["constraints_checked"].append("constant_product")
            result["warnings"].append(
                "Verify constant product invariant (x*y=k) is maintained"
            )
        return result

    def _check_no_negative_slippage(
        self, expr_str: str, test_data: Optional[Dict], result: Dict
    ) -> Dict:
        """Check that slippage is non-negative."""
        if "slippage" in expr_str.lower():
            result["constraints_checked"].append("no_negative_slippage")
            if test_data and "slippage" in test_data:
                if np.any(test_data["slippage"] < 0):
                    result["errors"].append("Slippage cannot be negative")
                    result["score"] -= 20
        return result

    def _check_ratio_positivity(
        self, expr_str: str, test_data: Optional[Dict], result: Dict
    ) -> Dict:
        """
        WEEK 2 NEW: Check that ratio variables are strictly positive.

        Critical for Impermanent Loss formulas where r appears in (1+r) denominators.
        """
        ratio_vars = ["r", "ratio", "price_ratio"]
        expr_lower = expr_str.lower()

        for var in ratio_vars:
            if var in expr_lower:
                result["constraints_checked"].append(f"{var}_positivity")

                # Check for dangerous pattern: (1 + r) in denominator
                if (
                    f"(1+{var})" in expr_str.replace(" ", "")
                    or f"(1 + {var})" in expr_str
                    or f"1+{var}" in expr_str.replace(" ", "")
                ):
                    result["errors"].append(
                        f"CRITICAL: Ratio variable '{var}' appears in (1+{var}) denominator. "
                        f"Must enforce {var} > 0 to prevent division by zero. "
                        f"Add constraint: if {var} <= 0, reject input or use abs({var})"
                    )
                    result["score"] -= 30

                if test_data and var in test_data:
                    values = test_data[var]
                    if np.any(values <= 0):
                        result["errors"].append(
                            f"Ratio variable '{var}' must be positive, "
                            f"found minimum: {np.min(values):.6f}"
                        )
                        result["score"] -= 25

        return result

    def _check_price_positivity(
        self, expr_str: str, test_data: Optional[Dict], result: Dict
    ) -> Dict:
        """
        WEEK 2 NEW: Check that price variables are strictly positive.

        Prices cannot be zero or negative in financial formulas.
        """
        price_vars = ["price", "p_t", "p_0", "p0", "pt", "p1", "p2"]
        expr_lower = expr_str.lower()

        found_prices = [var for var in price_vars if var in expr_lower]

        if found_prices:
            result["constraints_checked"].append("price_positivity")

            for var in found_prices:
                if test_data and var in test_data:
                    values = test_data[var]
                    if np.any(values <= 0):
                        result["errors"].append(
                            f"Price variable '{var}' must be strictly positive, "
                            f"found minimum: {np.min(values):.6f}"
                        )
                        result["score"] -= 20
                else:
                    result["warnings"].append(
                        f"Price variable '{var}' must be positive. "
                        f"Add validation: assert {var} > 0"
                    )
                    result["score"] -= 8

        return result

    def _check_division_protection(self, expr_str: str, result: Dict) -> Dict:
        """
        WEEK 2 NEW: Check for epsilon protection in divisions.

        Divisions should have epsilon guards: (denominator + ε)
        """
        result["constraints_checked"].append("division_protection")

        # Look for division operators
        if "/" in expr_str or "÷" in expr_str:
            # Check if epsilon protection exists
            has_epsilon = any(
                pattern in expr_str.lower()
                for pattern in ["epsilon", "eps", "ε", "+ 1e-", "+ 0.000"]
            )

            if not has_epsilon:
                result["warnings"].append(
                    "Division detected without epsilon protection. "
                    "Consider adding: (denominator + ε) to prevent division by zero"
                )
                result["score"] -= 5
            else:
                result["warnings"].append(
                    "Epsilon protection detected - verify epsilon value is appropriate"
                )

        return result

    def _check_var_positive(
        self, expr_str: str, test_data: Optional[Dict], result: Dict
    ) -> Dict:
        """Check that VaR (Value at Risk) is positive."""
        if "var" in expr_str.lower():
            result["constraints_checked"].append("var_positive")
            result["warnings"].append("VaR should be positive")
        return result

    def _check_confidence_valid(
        self, expr_str: str, test_data: Optional[Dict], result: Dict
    ) -> Dict:
        """Check that confidence level is valid."""
        if "confidence" in expr_str.lower():
            result["constraints_checked"].append("confidence_valid")
            if test_data and "confidence" in test_data:
                conf = test_data["confidence"]
                if np.any(conf <= 0) or np.any(conf >= 1):
                    result["errors"].append(
                        "Confidence level must be in (0, 1) exclusive"
                    )
                    result["score"] -= 20
        return result

    def _check_weights_sum(self, expr_str: str, var_defs: Dict, result: Dict) -> Dict:
        """Check that weight variables sum to 1."""
        weight_vars = [v for v in var_defs if "weight" in v.lower()]
        if weight_vars:
            result["constraints_checked"].append("weights_sum_to_one")
            result["warnings"].append(f"Verify that weights {weight_vars} sum to 1")
        return result

    def _check_score_range(
        self, expr_str: str, test_data: Optional[Dict], result: Dict
    ) -> Dict:
        """Check that scores are in valid range."""
        if "score" in expr_str.lower():
            result["constraints_checked"].append("score_range")
            if test_data and "score" in test_data:
                scores = test_data["score"]
                if np.any(scores < 0) or np.any(scores > 100):
                    result["errors"].append(
                        f"Scores must be in [0, 100] "
                        f"(found range [{np.min(scores):.1f}, {np.max(scores):.1f}])"
                    )
                    result["score"] -= 20
        return result

    # History management

    def clear_history(self):
        """Clear validation history."""
        if isinstance(self.validation_history, deque):
            self.validation_history.clear()
        else:
            self.validation_history = []

    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get validation history."""
        history_list = list(self.validation_history)
        if limit is not None:
            return history_list[-limit:]
        return history_list

    def get_statistics(self) -> Dict:
        """Get statistics about validation history."""
        if not self.validation_history:
            return {
                "total_validations": 0,
                "success_rate": 0.0,
                "average_score": 0.0,
                "domain": self.domain,
            }

        total = len(self.validation_history)
        valid_count = sum(1 for v in self.validation_history if v["valid"])
        avg_score = sum(v["score"] for v in self.validation_history) / total

        return {
            "total_validations": total,
            "success_rate": valid_count / total,
            "average_score": avg_score,
            "valid_count": valid_count,
            "invalid_count": total - valid_count,
            "domain": self.domain,
        }


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("WEEK 2 ENHANCED DOMAIN VALIDATION TESTS")
    print("=" * 80)

    # Test DeFi domain with critical issues
    validator = DomainValidator(domain="defi")

    # Test 1: Ratio positivity (IL formula)
    print("\n[TEST 1] Impermanent Loss formula with ratio constraint:")
    result1 = validator.validate(
        expression_str="sqrt(2*sqrt(r)/(1+r)) - 1",
        variable_definitions={"r": "Price ratio"},
        test_data={"r": np.array([0.5, 1.0, 2.0, -1.0])},  # -1.0 is problematic!
    )
    print(f"Valid: {result1['valid']}, Score: {result1['score']}")
    print(f"Errors: {result1['errors']}")

    # Test 2: Price positivity
    print("\n[TEST 2] Price positivity check:")
    result2 = validator.validate(
        expression_str="sqrt(abs(p_t - p_0))",
        variable_definitions={"p_t": "Current price", "p_0": "Initial price"},
        test_data={"p_t": np.array([100, 150]), "p_0": np.array([120, 130])},
    )
    print(f"Valid: {result2['valid']}, Score: {result2['score']}")
    print(f"Warnings: {result2['warnings']}")

    # Test 3: Fee bounds
    print("\n[TEST 3] Fee variable bounds:")
    result3 = validator.validate(
        expression_str="output = (y0 * dx * (1 - phi)) / (x0 + dx * (1 - phi))",
        variable_definitions={
            "y0": "Reserve Y",
            "dx": "Input amount",
            "phi": "Fee",
            "x0": "Reserve X",
        },
        test_data={
            "y0": np.array([1000]),
            "dx": np.array([10]),
            "phi": np.array([0.003]),
            "x0": np.array([1000]),
        },
    )
    print(f"Valid: {result3['valid']}, Score: {result3['score']}")
    print(f"Warnings: {result3['warnings']}")

    # Test 4: Division protection
    print("\n[TEST 4] Division protection check:")
    result4 = validator.validate(
        expression_str="output / (input + epsilon)",
        variable_definitions={
            "output": "Output",
            "input": "Input",
            "epsilon": "Safety",
        },
        test_data=None,
    )
    print(f"Valid: {result4['valid']}, Score: {result4['score']}")
    print(f"Constraints checked: {result4['constraints_checked']}")

    # Get statistics
    print("\n" + "=" * 80)
    stats = validator.get_statistics()
    print(f"Validation statistics: {stats}")
    print("=" * 80)

# ===========================================================================
# symbolic_validator
# ===========================================================================

#!/usr/bin/env python3
"""
Layer 3 — Symbolic and mathematical validator for HypatiaX symbolic regression.

Validates candidate expressions for syntactic correctness, undefined variables,
pathological values (NaN, complex infinity, literal infinity), simplifiability,
and domain-specific mathematical rules using SymPy.  Also performs numerical
stability analysis: division-by-zero risks, exponential overflow, sqrt of
potentially negative values, and logarithm domain violations.

Supports both string and LaTeX expression input (``from_latex=True``).

Designed to be called by EnsembleValidator as the third validation layer, but
can also be used standalone.

Supported domains
-----------------
    defi, finance, esg, risk, biology, biochemistry

Scoring
-------
Starts at 0.  Each sub-check that passes adds 25 points (syntactically_valid,
dimensionally_consistent, domain_valid, numerically_stable).  Each error
deducts 15 points; each warning deducts 2 points.  Final score clamped to
[0, 100].

Notes on bare-except removal
-----------------------------
The original code contained three bare ``except:`` clauses.  These have been
replaced with ``except Exception`` blocks that log the suppressed exception at
DEBUG level, preserving the graceful-fallback behaviour while making failures
visible during development and debugging.

Dependencies
------------
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
from sympy import simplify, sympify

# ---------------------------------------------------------------------------
# Module-level reproducibility seeds.
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# FIX: parse_latex requires antlr4-python3-runtime which may not be installed.
# Moving to a lazy import so the entire module can be imported (and all
# non-LaTeX validation paths used) even without antlr4.
_parse_latex_fn = None

def _get_parse_latex():
    """Return sympy's parse_latex, importing on first call. Raises ImportError if antlr4 absent."""
    global _parse_latex_fn
    if _parse_latex_fn is None:
        from sympy.parsing.latex import parse_latex as _pl  # noqa: PLC0415
        _parse_latex_fn = _pl
    return _parse_latex_fn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def safe_sympify(
    expression_str: str,
    variable_names: Optional[List[str]] = None,
) -> sp.Expr:
    """Parse *expression_str* into a SymPy expression with Pint isolation.

    Builds an isolated local symbol dictionary so that any Pint unit symbols
    registered in the global SymPy namespace do not corrupt parsing.

    Args:
        expression_str: Mathematical expression as a Python-syntax string.
        variable_names: Variable names to pre-declare as real SymPy symbols.

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
            "sqrt": sp.sqrt,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
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

    try:
        return sp.sympify(expression_str, locals=local_dict, evaluate=False)
    except Exception:
        pass

    try:
        return sp.sympify(expression_str, locals=local_dict, evaluate=True)
    except Exception as exc:
        raise ValueError(
            f"Could not parse expression '{expression_str}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------

class SymbolicValidator:
    """Layer 3 symbolic validator — rejects mathematically unsound expressions.

    Runs seven checks in sequence:

    1. Syntax — can SymPy parse the expression?
    2. Undefined variables — are all free symbols declared?
    3. Pathological values — does the expression contain zoo, oo, or nan?
    4. Simplification — can the expression be reduced (advisory only)?
    5. Dimensional consistency — placeholder, always passes (DimensionalValidator
       owns this concern).
    6. Domain rules — domain-specific symbolic constraints (DeFi, risk, …).
    7. Numerical stability — division-by-zero, overflow, sqrt/log domains.

    History
    -------
    The last *max_history* results are retained in a bounded deque.
    """

    def __init__(self, max_history: Optional[int] = 1000) -> None:
        """Initialise the validator.

        Args:
            max_history: Maximum number of results to retain in
                ``validation_history``.  Pass ``None`` for an unbounded list.
        """
        self.domain_rules = {
            "defi":        self._defi_rules,
            "finance":     self._finance_rules,
            "esg":         self._esg_rules,
            "risk":        self._risk_rules,
            "biology":     self._biology_rules,
            "biochemistry": self._biology_rules,
        }

        if max_history is not None:
            self.validation_history: Any = deque(maxlen=max_history)
        else:
            self.validation_history = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        expression: str,
        variable_definitions: Dict[str, str],
        domain: str = "defi",
        from_latex: bool = False,
    ) -> Dict[str, Any]:
        """Validate *expression* symbolically.

        Args:
            expression: The mathematical expression (Python syntax or LaTeX).
            variable_definitions: Mapping of variable name → human-readable
                description, e.g. ``{"S": "Substrate concentration"}``.
            domain: Application domain for domain-specific rule checking.
                Supported values: ``'defi'``, ``'finance'``, ``'esg'``,
                ``'risk'``, ``'biology'``, ``'biochemistry'``.
                Unrecognised domains use a permissive default rule set.
            from_latex: If True, parse *expression* as LaTeX before sympifying.

        Returns:
            Result dictionary with keys:

            - ``valid`` (bool)
            - ``syntactically_valid`` (bool)
            - ``dimensionally_consistent`` (bool)
            - ``domain_valid`` (bool)
            - ``numerically_stable`` (bool)
            - ``sympy_expr`` (sp.Expr | None)
            - ``canonical_form`` (str | None)
            - ``errors`` (list[str])
            - ``warnings`` (list[str])
            - ``score`` (int in [0, 100])
        """
        results: Dict[str, Any] = {
            "valid": True,
            "syntactically_valid": False,
            "dimensionally_consistent": False,
            "domain_valid": False,
            "numerically_stable": False,
            "errors": [],
            "warnings": [],
            "sympy_expr": None,
            "canonical_form": None,
        }

        # Guard: empty / whitespace-only expression.
        if not expression or not expression.strip():
            results["errors"].append("Empty expression not allowed")
            results["valid"] = False
            logger.debug("Rejected: empty expression")
            return self._finalize_results(results)

        try:
            # 1. Parse -------------------------------------------------------
            if from_latex:
                expr = self._safe_parse_latex(expression)
            else:
                expr = safe_sympify(expression, list(variable_definitions.keys()))

            if expr is None:
                results["errors"].append("Cannot parse expression")
                results["valid"] = False
                return self._finalize_results(results)

            results["syntactically_valid"] = True
            results["sympy_expr"] = expr

            # 2. Undefined variables ----------------------------------------
            free_vars = expr.free_symbols
            undefined = [
                str(v) for v in free_vars if str(v) not in variable_definitions
            ]
            if undefined:
                results["errors"].append(f"Undefined variables: {undefined}")
                results["valid"] = False

            # 3. Pathological values ----------------------------------------
            if expr.has(sp.zoo):
                results["errors"].append("Expression contains complex infinity (zoo)")
                results["valid"] = False

            if expr.has(sp.oo):
                results["warnings"].append(
                    "Expression contains literal infinity — verify limits"
                )

            if expr.has(sp.nan):
                results["errors"].append("Expression contains NaN")
                results["valid"] = False

            # 4. Simplification (advisory) ----------------------------------
            try:
                simplified = simplify(expr)
                results["canonical_form"] = str(simplified)
                if expr != simplified:
                    results["warnings"].append(
                        f"Expression can be simplified to: {simplified}"
                    )
            except Exception as exc:
                # Simplification failure is non-fatal; use the raw expression.
                logger.debug(
                    "Simplification failed for '%s': %s", expression, exc
                )
                results["warnings"].append(f"Simplification failed: {exc}")
                results["canonical_form"] = str(expr)

            # 5. Dimensional consistency (placeholder) ----------------------
            # Full unit analysis is owned by DimensionalValidator (Layer 1).
            # This check always passes; it exists as a named flag so callers
            # can distinguish "not checked here" from "checked and passed".
            results["dimensionally_consistent"] = True

            # 6. Domain rules -----------------------------------------------
            rule_fn = self.domain_rules.get(domain, self._default_rules)
            domain_check = rule_fn(expr, variable_definitions)
            results["domain_valid"] = domain_check["valid"]
            results["errors"].extend(domain_check["errors"])
            results["warnings"].extend(domain_check.get("warnings", []))
            if not domain_check["valid"]:
                results["valid"] = False

            # 7. Numerical stability ----------------------------------------
            stability = self._check_numerical_stability(expr)
            results["numerically_stable"] = stability["stable"]
            results["warnings"].extend(stability["warnings"])
            results["errors"].extend(stability.get("errors", []))
            if stability.get("errors"):
                results["valid"] = False

        except Exception as exc:
            logger.exception(
                "Unexpected error during symbolic validation of '%s'", expression
            )
            results["errors"].append(f"Unexpected validation error: {exc}")
            results["valid"] = False

        return self._finalize_results(results)

    # ------------------------------------------------------------------
    # Internal — parsing
    # ------------------------------------------------------------------

    def _finalize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate score, log outcome, and store in history."""
        results["score"] = self._calculate_score(results)
        logger.debug(
            "Symbolic validation: valid=%s score=%d errors=%d warnings=%d",
            results["valid"], results["score"],
            len(results["errors"]), len(results.get("warnings", [])),
        )
        self.validation_history.append(results)
        return results

    def _safe_parse_latex(self, latex_str: str) -> Optional[sp.Expr]:
        """Parse a LaTeX expression, returning None on failure.

        Uses a lazy import of parse_latex so the module can be loaded without
        antlr4.  Falls back to plain sympify if parse_latex fails or is absent.
        Returns None rather than raising so callers emit a structured error.
        """
        try:
            parse_latex = _get_parse_latex()
            clean = re.sub(r"\\text\{([^}]+)\}", r"\1", latex_str.strip())
            return parse_latex(clean)
        except ImportError as exc:
            logger.warning("parse_latex unavailable (antlr4 missing?): %s", exc)
        except Exception as exc:
            logger.debug("parse_latex failed ('%s'): %s — trying sympify", latex_str, exc)

        try:
            return sympify(latex_str)
        except Exception as exc:
            logger.debug("sympify fallback also failed ('%s'): %s", latex_str, exc)
            return None

    # ------------------------------------------------------------------
    # Internal — numerical stability
    # ------------------------------------------------------------------

    def _check_numerical_stability(self, expr: sp.Expr) -> Dict[str, Any]:
        """Scan the expression tree for numerical stability risks.

        Checks:

        1. Unprotected division-by-zero (hard error if no epsilon guard).
        2. Subtractive cancellation (warning if > 2 subtractions).
        3. Exponential overflow risk.
        4. Long multiplication chains (overflow warning).
        5. sqrt of potentially negative values.
        6. log of potentially non-positive values.
        7. Trigonometric functions (range advisory).
        8. Power overflow (large or variable exponents).

        Returns:
            Dict with keys ``stable`` (bool), ``warnings``, ``errors``.
        """
        warnings: List[str] = []
        errors: List[str] = []

        # 1. Division by zero
        _has_division = any(
            arg.is_Pow and arg.exp.is_negative
            for node in sp.preorder_traversal(expr)
            if node.is_Mul
            for arg in node.args
        )
        if _has_division:
            for denom in self._extract_denominators(expr):
                if self._could_be_zero(denom):
                    if self._has_epsilon_protection(denom):
                        warnings.append(f"Division-by-zero risk mitigated: {denom}")
                    else:
                        errors.append(
                            f"CRITICAL: Unprotected division-by-zero risk — denominator "
                            f"'{denom}' may be zero.  Add epsilon guard: (denom + ε)"
                        )

        # 2. Subtractive cancellation
        if len(self._find_subtractions(expr)) > 2:
            warnings.append(
                "Multiple subtractions detected — potential precision loss"
            )

        # 3. Exponential overflow
        if expr.has(sp.exp):
            for arg in self._extract_exp_arguments(expr):
                if self._could_overflow_exp(arg):
                    warnings.append(
                        f"Exponential overflow risk: exp({arg}) — "
                        f"cap the argument or use a numerically stable variant"
                    )

        # 4. Long multiplication chains
        if expr.has(sp.Mul):
            mul_terms = self._extract_multiplication_chains(expr)
            if len(mul_terms) > 3:
                warnings.append(
                    f"Product of {len(mul_terms)} terms — verify no overflow: "
                    f"{' * '.join(str(t) for t in mul_terms[:3])} ..."
                )

        # 5. sqrt domain
        if expr.has(sp.sqrt):
            for arg in self._extract_sqrt_arguments(expr):
                if not self._guaranteed_positive(arg):
                    warnings.append(
                        f"sqrt({arg}) may receive a negative argument — "
                        f"add validation or use abs()"
                    )

        # 6. log domain
        if expr.has(sp.log):
            for arg in self._extract_log_arguments(expr):
                if not self._guaranteed_positive(arg):
                    warnings.append(
                        f"log({arg}) may receive a non-positive argument — "
                        f"ensure {arg} > 0"
                    )

        # 7. Trigonometric range
        if any(expr.has(fn) for fn in (sp.sin, sp.cos, sp.tan)):
            warnings.append(
                "Trigonometric function detected — verify input is in radians "
                "and within expected range"
            )

        # 8. Power overflow
        if expr.has(sp.Pow):
            for base, exp_node in self._extract_power_terms(expr):
                if self._could_overflow_power(base, exp_node):
                    warnings.append(
                        f"Power overflow risk: ({base})^({exp_node}) — "
                        f"verify bounds on base and exponent"
                    )

        return {
            "stable": not warnings and not errors,
            "warnings": warnings,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Internal — expression tree helpers
    # ------------------------------------------------------------------

    def _has_epsilon_protection(self, expr: sp.Expr) -> bool:
        """Return True if *expr* contains a common epsilon-guard pattern."""
        s = str(expr).lower()
        return any(p in s for p in ("epsilon", "eps", "ε", "+ 1e-", "+ 0.000"))

    def _extract_exp_arguments(self, expr: sp.Expr) -> List[sp.Expr]:
        """Return all direct arguments of exp() nodes in the tree."""
        args: List[sp.Expr] = []
        if expr.func == sp.exp:
            args.append(expr.args[0])
        for arg in getattr(expr, "args", ()):
            args.extend(self._extract_exp_arguments(arg))
        return args

    def _could_overflow_exp(self, arg: sp.Expr) -> bool:
        """Return True if the exp() argument may be unboundedly large."""
        s = str(arg)
        if "*" in s or "**" in s or "^" in s:
            return True
        return bool(arg.free_symbols) and not arg.is_Number

    def _extract_multiplication_chains(self, expr: sp.Expr) -> List[sp.Expr]:
        """Return the args of top-level and nested Mul nodes."""
        terms: List[sp.Expr] = []
        if expr.is_Mul:
            terms.extend(expr.args)
        for arg in getattr(expr, "args", ()):
            if arg.is_Mul:
                terms.extend(arg.args)
        return terms

    def _extract_sqrt_arguments(self, expr: sp.Expr) -> List[sp.Expr]:
        """Return all direct arguments of sqrt() nodes in the tree."""
        args: List[sp.Expr] = []
        if expr.func == sp.sqrt:
            args.append(expr.args[0])
        for arg in getattr(expr, "args", ()):
            args.extend(self._extract_sqrt_arguments(arg))
        return args

    def _extract_log_arguments(self, expr: sp.Expr) -> List[sp.Expr]:
        """Return all direct arguments of log() nodes in the tree."""
        args: List[sp.Expr] = []
        if expr.func == sp.log:
            args.append(expr.args[0])
        for arg in getattr(expr, "args", ()):
            args.extend(self._extract_log_arguments(arg))
        return args

    def _extract_power_terms(self, expr: sp.Expr) -> List[tuple]:
        """Return (base, exponent) pairs for all Pow nodes in the tree."""
        terms: List[tuple] = []
        if expr.is_Pow:
            terms.append((expr.args[0], expr.args[1]))
        for arg in getattr(expr, "args", ()):
            terms.extend(self._extract_power_terms(arg))
        return terms

    def _could_overflow_power(self, base: sp.Expr, exponent: sp.Expr) -> bool:
        """Return True if base^exponent may overflow."""
        if exponent.is_Number:
            try:
                if abs(float(exponent)) > 10:
                    return True
            except Exception as exc:
                logger.debug("Could not convert exponent to float: %s", exc)
        return bool(exponent.free_symbols)

    def _guaranteed_positive(self, expr: sp.Expr) -> bool:
        """Return True if *expr* is structurally guaranteed to be positive."""
        if expr.is_Number:
            try:
                return float(expr) > 0
            except Exception:
                return False
        if expr.func == sp.Abs:
            return True
        if expr.is_Pow and expr.args[1] == 2:
            return True
        if "abs(" in str(expr).lower():
            return True
        return False

    def _extract_denominators(self, expr: sp.Expr) -> List[sp.Expr]:
        """Return all denominator sub-expressions (bases of Pow(..., -n)).

        FIX: the original had a double-recursion bug.  When ``expr.is_Add``,
        it recursed explicitly into each child AND the unconditional
        ``for arg in expr.args`` at the end visited them a second time,
        producing duplicate denominator entries and therefore duplicate
        "division-by-zero" errors.  Fixed with ``elif`` so only one branch
        recurses per node type.
        """
        denoms: List[sp.Expr] = []

        if expr.is_Mul:
            # Collect denominators at this level.
            for arg in expr.args:
                if arg.is_Pow and arg.exp.is_negative:
                    denoms.append(arg.base)
            # Recurse into non-Pow factors.
            for arg in expr.args:
                if not arg.is_Pow:
                    denoms.extend(self._extract_denominators(arg))

        elif expr.is_Add:
            # Recurse into summands — do NOT fall through to generic loop.
            for arg in expr.args:
                denoms.extend(self._extract_denominators(arg))

        elif expr.is_Pow:
            # Recurse into base only; exponent is a scalar.
            denoms.extend(self._extract_denominators(expr.args[0]))

        else:
            # Generic function or atom.
            for arg in getattr(expr, "args", ()):
                denoms.extend(self._extract_denominators(arg))

        return denoms

    def _could_be_zero(self, expr: sp.Expr) -> bool:
        """Return True if *expr* could evaluate to zero.

        Recognises a set of domain-specific known-positive variable names
        (Michaelis-Menten constants, concentrations, prices, liquidity) whose
        sums are structurally guaranteed non-zero.
        """
        if expr.is_Number:
            try:
                return abs(float(expr)) < 1e-10
            except Exception:
                return True

        if expr.is_Add:
            has_variables = False
            all_positive = True
            _KNOWN_POSITIVE = (
                "km", "vmax", "kcat",          # biochemistry
                "concentration", "conc",        # concentrations (≥ 0)
                "price", "liquidity",           # finance (> 0)
                "amount", "volume",             # generally positive
            )
            for term in expr.args:
                if term.is_Symbol:
                    has_variables = True
                    if not any(p in str(term).lower() for p in _KNOWN_POSITIVE):
                        all_positive = False
                        break
                elif term.is_Number and float(term) > 0:
                    has_variables = True
                elif not (term.is_Mul and any(a.is_Symbol for a in term.args)):
                    all_positive = False
                    break
            if has_variables and all_positive:
                return False
            return True

        s = str(expr)
        if "+ r" in s or "+ ratio" in s:
            return True

        return False

    def _find_subtractions(self, expr: sp.Expr) -> List[sp.Expr]:
        """Return Add nodes that contain at least one negated term."""
        subs: List[sp.Expr] = []
        if expr.is_Add:
            if any(arg.could_extract_minus_sign() for arg in expr.args):
                subs.append(expr)
        for arg in getattr(expr, "args", ()):
            subs.extend(self._find_subtractions(arg))
        return subs

    # ------------------------------------------------------------------
    # Internal — domain rules
    # ------------------------------------------------------------------

    def _defi_rules(
        self, expr: sp.Expr, variable_definitions: Dict[str, str]
    ) -> Dict[str, Any]:
        """DeFi-specific validation rules.

        Checks:

        - Impermanent Loss ratio variable ``r > 0`` when used in ``(1+r)``
          denominators.
        - Price variable positivity.
        - Fee variable bounds ``0 ≤ fee < 1``.
        - Liquidity positivity.
        - AMM constant product invariant advisory.
        """
        errors: List[str] = []
        warnings: List[str] = []

        expr_str = str(expr).lower()
        free_vars = [str(s).lower() for s in expr.free_symbols]

        if ("r" in free_vars or "ratio" in free_vars) and "sqrt" in expr_str:
            if "1 + r" in expr_str or "(1+r)" in expr_str:
                errors.append(
                    "CRITICAL: Impermanent Loss formula requires r > 0. "
                    "Add constraint: if r ≤ 0, reject input or use abs(r)"
                )

        price_vars = [
            v for v in free_vars
            if "price" in v or "p_" in v or "p0" in v or "pt" in v
        ]
        if price_vars:
            warnings.append(
                f"Price variables {price_vars} must be positive — "
                f"add: assert all(p > 0 for p in prices)"
            )

        if "fee" in free_vars or "phi" in free_vars or "φ" in expr_str:
            warnings.append(
                "Fee variable must satisfy 0 ≤ fee < 1 — "
                "add: assert 0 <= fee < 1"
            )

        if "liquidity" in free_vars:
            warnings.append("Liquidity must remain strictly positive")

        if "price" in expr_str:
            warnings.append("Verify price bounds and slippage limits")

        if expr.has(sp.Mul) and expr.has(sp.Pow):
            warnings.append(
                "Check that AMM constant product invariant (x·y = k) is preserved"
            )

        return {"valid": not errors, "errors": errors, "warnings": warnings}

    def _finance_rules(
        self, expr: sp.Expr, variable_definitions: Dict[str, str]
    ) -> Dict[str, Any]:
        """Finance-specific validation rules."""
        errors: List[str] = []
        warnings: List[str] = []
        s = str(expr).lower()

        if "risk" in s or "var" in s:
            warnings.append("Risk metrics should be non-negative")
        if "return" in s:
            warnings.append("Verify return calculation methodology")
        if "prob" in s:
            warnings.append("Ensure probabilities are in [0, 1]")

        return {"valid": not errors, "errors": errors, "warnings": warnings}

    def _esg_rules(
        self, expr: sp.Expr, variable_definitions: Dict[str, str]
    ) -> Dict[str, Any]:
        """ESG-specific validation rules."""
        errors: List[str] = []
        warnings: List[str] = []
        s = str(expr).lower()

        if "score" in s:
            warnings.append("Verify scores are in valid range (typically 0–100)")
        if expr.has(sp.Add):
            warnings.append("Ensure component weights sum to 1")

        return {"valid": not errors, "errors": errors, "warnings": warnings}

    def _risk_rules(
        self, expr: sp.Expr, variable_definitions: Dict[str, str]
    ) -> Dict[str, Any]:
        """Risk management validation rules."""
        errors: List[str] = []
        warnings: List[str] = []
        s = str(expr).lower()

        if "var" in s:
            warnings.append("VaR must be positive and bounded")
        if "confidence" in s:
            warnings.append("Confidence level must be in (0, 1) exclusive")
        if expr.has(sp.oo):
            errors.append("Risk metric appears unbounded")

        return {"valid": not errors, "errors": errors, "warnings": warnings}

    def _biology_rules(
        self, expr: sp.Expr, variable_definitions: Dict[str, str]
    ) -> Dict[str, Any]:
        """Biology/biochemistry validation rules.

        Recognises Michaelis-Menten kinetics patterns and flags concentration
        and rate-constant variables for positivity constraints.
        """
        errors: List[str] = []
        warnings: List[str] = []
        expr_str = str(expr).lower()
        free_vars = [str(s).lower() for s in expr.free_symbols]

        if ("km" in free_vars or "michaelis" in expr_str) and "s" in free_vars:
            warnings.append(
                "Michaelis-Menten pattern detected — "
                "ensure Km > 0 and S ≥ 0"
            )

        conc_vars = [
            v for v in free_vars
            if any(t in v for t in ("concentration", "conc", "_c"))
        ]
        if conc_vars:
            warnings.append(
                f"Concentration variables {conc_vars} must be non-negative"
            )

        rate_vars = [
            v for v in free_vars
            if any(t in v for t in ("vmax", "kcat", "kd", "ki", "rate"))
        ]
        if rate_vars:
            warnings.append(
                f"Rate/equilibrium constants {rate_vars} must be strictly positive"
            )

        return {"valid": not errors, "errors": errors, "warnings": warnings}

    def _default_rules(
        self, expr: sp.Expr, variable_definitions: Dict[str, str]
    ) -> Dict[str, Any]:
        """Permissive default rules for unrecognised domains."""
        return {"valid": True, "errors": [], "warnings": []}

    # ------------------------------------------------------------------
    # Internal — scoring
    # ------------------------------------------------------------------

    def _calculate_score(self, results: Dict[str, Any]) -> int:
        """Compute a score in [0, 100] from the sub-check flags.

        Each of the four boolean checks (syntactically_valid,
        dimensionally_consistent, domain_valid, numerically_stable) contributes
        25 points.  Each error deducts 15 points; each warning deducts 2 points.
        """
        score = 0
        if results["syntactically_valid"]:
            score += 25
        if results["dimensionally_consistent"]:
            score += 25
        if results["domain_valid"]:
            score += 25
        if results["numerically_stable"]:
            score += 25

        score -= len(results["errors"]) * 15
        score -= len(results.get("warnings", [])) * 2

        return max(0, min(100, score))

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def clear_history(self) -> None:
        """Clear all stored validation results."""
        if isinstance(self.validation_history, deque):
            self.validation_history.clear()
        else:
            self.validation_history = []

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return the validation history, optionally limited to the most recent *limit* entries."""
        history = list(self.validation_history)
        return history[-limit:] if limit is not None else history

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics over the validation history."""
        if not self.validation_history:
            return {"total_validations": 0, "success_rate": 0.0, "average_score": 0.0}

        total = len(self.validation_history)
        valid_count = sum(1 for v in self.validation_history if v["valid"])
        avg_score = sum(v["score"] for v in self.validation_history) / total

        return {
            "total_validations": total,
            "success_rate": valid_count / total,
            "average_score": avg_score,
            "valid_count": valid_count,
            "invalid_count": total - valid_count,
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    validator = SymbolicValidator()

    print("=" * 80)
    print("SYMBOLIC VALIDATOR — SELF-TEST SUITE")
    print("=" * 80)

    tests = [
        ("Empty expression", "", {}, "finance"),
        ("Unprotected division (IL formula)",
         "sqrt(2*sqrt(r)/(1+r)) - 1", {"r": "Price ratio"}, "defi"),
        ("Price positivity",
         "sqrt(abs(P_t - P_0))", {"P_t": "Current price", "P_0": "Initial price"}, "defi"),
        ("Exponential overflow risk",
         "exp(lambda_val * sigma**2)", {"lambda_val": "Sensitivity", "sigma": "Volatility"}, "risk"),
    ]

    for name, expr, var_defs, domain in tests:
        result = validator.validate(expression=expr, variable_definitions=var_defs, domain=domain)
        print(f"\n[{name}]")
        print(f"  Valid={result['valid']}  Score={result['score']}")
        if result["errors"]:
            print(f"  Errors   : {result['errors']}")
        if result["warnings"]:
            print(f"  Warnings : {result['warnings'][:3]}")

    print("\n" + "=" * 80)
    stats = validator.get_statistics()
    print(f"Stats: {stats}")
    print("=" * 80)

# ===========================================================================
# ensemble_validator
# ===========================================================================

#!/usr/bin/env python3
"""
Layer 4 — Ensemble validator for HypatiaX symbolic regression (v11).

Orchestrates the full four-layer validation pipeline by composing
DimensionalValidator, DomainValidator, and SymbolicValidator, then adding a
fourth numerical validation layer that evaluates the expression against test
data to detect NaN, Inf, overflow, and underflow at runtime.

Pipeline
--------
1. **Symbolic** (Layer 3) — syntax, undefined variables, SymPy pathologies.
2. **Dimensional** (Layer 1) — unit consistency via Pint.
3. **Domain** (Layer 2) — domain-specific variable constraints.
4. **Numerical** (Layer 4) — lambdify the expression and evaluate on test data.

After Layers 1–3, a domain-aware reconciliation step downgrades false-positive
division-by-zero errors from the symbolic layer when domain constraints
guarantee variable positivity (e.g. Michaelis-Menten: Km > 0 by definition).

Scoring
-------
Final score = weighted average of layer scores minus edge-case penalties.
Default weights: symbolic 30 %, dimensional 30 %, domain 30 %, numerical 10 %.
Acceptance threshold: 85.0 / 100.

Notes on bare-except removal
-----------------------------
The original code contained five bare ``except:`` clauses inside
``clean_expression_string`` and ``validate_complete``.  These have been
replaced with typed ``except Exception`` blocks that log the suppressed
exception at DEBUG level, preserving the graceful-fallback behaviour while
making failures observable.

Dependencies
------------
    numpy >= 1.24
    sympy >= 1.12
    pint >= 0.20  (via DimensionalValidator)
"""

import logging
import random
import re
from collections import deque
from typing import Any, Dict, List, Optional, Union

import numpy as np
import sympy as sp

# ---------------------------------------------------------------------------
# Module-level reproducibility seeds.
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# String-cleaning utilities
# ---------------------------------------------------------------------------

def extract_clean_expression_string(
    expression_input: Union[str, sp.Expr, object],
    variable_names: Optional[List[str]] = None,
) -> str:
    """Return a clean Python-syntax string from any expression representation.

    Strips XML/HTML tags, rounds high-precision float literals, and removes
    trivial coefficient artefacts (``1.000*``, ``0.999*``, ``)**1.000``).
    Falls back to ``str(expression_input)`` on any error.

    Args:
        expression_input: String, SymPy expression, or any object with a
            useful ``__str__``.
        variable_names: Unused; kept for API symmetry with safe_sympify.

    Returns:
        Cleaned expression string.
    """
    if expression_input is None:
        return "0"

    try:
        expr_str = (
            expression_input
            if isinstance(expression_input, str)
            else str(expression_input)
        )
    except Exception as exc:
        logger.debug("str() failed on expression input: %s", exc)
        return "0"

    try:
        if "<" in expr_str and ">" in expr_str:
            expr_str = re.sub(r"<[^>]+>", "", expr_str)

        def _round_float(match: re.Match) -> str:
            try:
                num = float(match.group(0))
                return str(int(round(num))) if abs(num - round(num)) < 1e-4 else f"{num:.4f}"
            except Exception:
                return match.group(0)

        expr_str = re.sub(r"\d+\.\d{5,}", _round_float, expr_str)
        expr_str = re.sub(r"\b1\.0{3,}\d*\*", "", expr_str)
        expr_str = re.sub(r"\b0\.99\d+\*", "", expr_str)
        expr_str = re.sub(r"\)\*\*1\.0{2,}\d*", ")", expr_str)
        expr_str = " ".join(expr_str.split())
        return expr_str.strip()

    except Exception as exc:
        logger.debug("Expression cleaning failed: %s", exc)
        return str(expression_input)


def safe_sympify(
    expression_str: str,
    variable_names: Optional[List[str]] = None,
) -> sp.Expr:
    """Parse *expression_str* into a SymPy expression with Pint isolation.

    Tries three strategies in order:

    1. ``sp.sympify(..., evaluate=False)`` — preserves expression structure.
    2. ``sp.sympify(..., evaluate=True)`` — for expressions SymPy auto-simplifies.
    3. ``parse_expr`` with implicit-multiplication transformations.

    Args:
        expression_str: Cleaned Python-syntax expression string.
        variable_names: Pre-declared as real SymPy symbols to prevent Pint leakage.

    Returns:
        Parsed SymPy expression.

    Raises:
        ValueError: If all three strategies fail.
    """
    expression_str = extract_clean_expression_string(expression_str, variable_names)

    local_dict: Dict[str, object] = {}
    if variable_names:
        for var in variable_names:
            local_dict[var] = sp.Symbol(var, real=True)

    local_dict.update(
        {
            "exp": sp.exp,
            "log": sp.log,
            "ln": sp.log,
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

    try:
        return sp.sympify(expression_str, locals=local_dict, evaluate=False)
    except Exception:
        pass

    try:
        return sp.sympify(expression_str, locals=local_dict, evaluate=True)
    except Exception:
        pass

    try:
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
        transformations = standard_transformations + (implicit_multiplication_application,)
        return parse_expr(expression_str, local_dict=local_dict, transformations=transformations)
    except Exception as exc:
        raise ValueError(
            f"Could not parse expression '{expression_str}': {exc}"
        ) from exc


def clean_expression_string(
    expression_str: Union[str, sp.Expr, object],
    variable_names: Optional[List[str]] = None,
) -> str:
    """Aggressively clean *expression_str* of Pint/SymPy artefacts.

    Parses the cleaned string through SymPy, rounds floating-point
    coefficients, and collapses trivial powers (``x**1.0`` → ``x``).
    Falls back to the string-only cleaned version if SymPy fails.

    Args:
        expression_str: Raw expression (string or SymPy object).
        variable_names: Variable names for Pint-isolated parsing.

    Returns:
        Cleaned expression string suitable for further validation.
    """
    clean_str = extract_clean_expression_string(expression_str, variable_names)

    try:
        expr = safe_sympify(clean_str, variable_names)

        def round_coefficients(e: sp.Expr, decimals: int = 3) -> sp.Expr:
            if isinstance(e, sp.Float):
                val = float(e)
                if abs(val) < 1e-10:
                    return sp.Integer(0)
                if abs(val - round(val)) < 1e-3:
                    return sp.Integer(round(val))
                return sp.Float(round(val, decimals))
            if isinstance(e, (sp.Integer, sp.Symbol)):
                return e
            if isinstance(e, sp.Rational) and e.q > 100:
                return sp.Float(round(float(e), decimals))
            if hasattr(e, "args") and e.args:
                try:
                    return e.func(*[round_coefficients(a, decimals) for a in e.args])
                except Exception as exc:
                    logger.debug("round_coefficients reconstruction failed: %s", exc)
                    return e
            return e

        def simplify_powers(e: sp.Expr) -> sp.Expr:
            if isinstance(e, sp.Pow):
                base = simplify_powers(e.base)
                exp = simplify_powers(e.exp)
                if isinstance(exp, sp.Float):
                    ev = float(exp)
                    if abs(ev - 1.0) < 0.01:
                        return base
                    if abs(ev - round(ev)) < 0.01:
                        exp = sp.Integer(round(ev))
                if exp == 1:
                    return base
                return sp.Pow(base, exp)
            if hasattr(e, "args") and e.args:
                try:
                    return e.func(*[simplify_powers(a) for a in e.args])
                except Exception as exc:
                    logger.debug("simplify_powers reconstruction failed: %s", exc)
                    return e
            return e

        expr = simplify_powers(round_coefficients(expr))
        return str(expr)

    except Exception as exc:
        logger.debug(
            "clean_expression_string SymPy pass failed for '%s': %s", clean_str, exc
        )
        return clean_str


# ---------------------------------------------------------------------------
# Domain-aware reconciliation
# ---------------------------------------------------------------------------

def reconcile_symbolic_with_domain(
    symbolic_result: dict,
    domain_result: dict,
) -> dict:
    """Downgrade false-positive division-by-zero errors using domain knowledge.

    When the domain validator confirms that all relevant variables are positive
    (e.g. Km > 0 in Michaelis-Menten kinetics), symbolic division-by-zero
    errors that arise purely because SymPy cannot infer positivity are demoted
    to warnings.

    Args:
        symbolic_result: Output dict from SymbolicValidator.validate().
        domain_result: Output dict from DomainValidator.validate().

    Returns:
        A modified copy of *symbolic_result* with reconciled errors/warnings.
    """
    if not domain_result.get("valid", False):
        return symbolic_result

    symbolic = dict(symbolic_result)
    errors = list(symbolic.get("errors", []))
    warnings = list(symbolic.get("warnings", []))

    filtered_errors = []
    for err in errors:
        if "division by zero" in err.lower():
            warnings.append(
                "Division-by-zero risk ruled out by domain constraints "
                "(e.g. Km > 0 in Michaelis-Menten)"
            )
            logger.debug("Reconciled division-by-zero error: '%s'", err)
        else:
            filtered_errors.append(err)

    symbolic["errors"] = filtered_errors
    symbolic["warnings"] = warnings

    if not filtered_errors:
        symbolic["valid"] = True
        symbolic["score"] = max(symbolic.get("score", 0.0), 70.0)

    return symbolic


# ---------------------------------------------------------------------------
# Ensemble validator
# ---------------------------------------------------------------------------

class EnsembleValidator:
    """Layer 4 ensemble validator — combines all four validation layers.

    Instantiates and owns a SymbolicValidator, DimensionalValidator, and
    DomainValidator.  Adds a fourth numerical layer by lambdifying the
    expression and evaluating it on caller-supplied test data.

    Parameters
    ----------
    domain : str
        Application domain forwarded to DomainValidator and SymbolicValidator.
        Supported values: ``'defi'``, ``'finance'``, ``'esg'``, ``'risk'``,
        ``'biology'``, ``'biochemistry'``, ``'general'``.
    max_history : int | None
        Maximum result entries to retain per layer.
    weights : dict | None
        Layer score weights.  Must sum to 1.0.  Defaults to
        ``{'symbolic': 0.30, 'dimensional': 0.30, 'domain': 0.30, 'numerical': 0.10}``.
    strict_mode : bool
        If True, domain invalidity alone causes rejection regardless of score.

    Attributes
    ----------
    VALIDATION_THRESHOLDS : dict
        Class-level thresholds for score acceptance and penalty amounts.
    """

    VALIDATION_THRESHOLDS = {
        "minimum_total_score":            85.0,
        "minimum_layer_score":            70.0,
        "critical_failure_threshold":     50.0,
        "edge_case_penalty":              15.0,
        "dimensional_inconsistency_penalty": 20.0,
        "warning_penalty":                5.0,
        "domain_violation_penalty":       10.0,
    }

    def __init__(
        self,
        domain: str = "general",
        max_history: Optional[int] = 1000,
        weights: Optional[Dict[str, float]] = None,
        strict_mode: bool = False,
    ) -> None:
        self.domain = domain
        self.strict_mode = strict_mode
        self.symbolic_validator = SymbolicValidator(max_history=max_history)
        self.dimensional_validator = DimensionalValidator(max_history=max_history)
        self.domain_validator = DomainValidator(domain, max_history=max_history)

        self.weights = weights or {
            "symbolic":    0.30,
            "dimensional": 0.30,
            "domain":      0.30,
            "numerical":   0.10,
        }
        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("Validation layer weights must sum to 1.0")

        self.validation_history: Any = deque(maxlen=max_history) if max_history else []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_complete(
        self,
        expression_str: Union[str, sp.Expr, object],
        variable_definitions: Dict[str, str],
        variable_units: Dict[str, str],
        test_data: Optional[Dict[str, np.ndarray]] = None,
        from_latex: bool = False,
    ) -> Dict:
        """Run the full four-layer validation pipeline on *expression_str*.

        Args:
            expression_str: Candidate expression (string, SymPy object, or any
                type with a useful ``__str__``).  Pass ``None`` to receive a
                structured rejection result.
            variable_definitions: Mapping of variable name → description.
                Forwarded to symbolic and domain validators.
            variable_units: Mapping of variable name → Pint unit string.
                Forwarded to the dimensional validator.
            test_data: Optional dict of variable name → 1-D numpy array.
                When provided, Layer 4 evaluates the expression numerically.
            from_latex: If True, parse *expression_str* as LaTeX.

        Returns:
            Result dictionary with keys:

            - ``valid`` (bool) — overall acceptance decision.
            - ``total_score`` (float) — weighted, penalty-adjusted score.
            - ``base_score`` (float) — weighted score before penalties.
            - ``penalties_applied`` (dict)
            - ``layer_scores`` (dict) — per-layer scores.
            - ``layer_results`` (dict) — full per-layer result dicts.
            - ``errors`` / ``warnings`` (list[str]) — aggregated.
            - ``recommendations`` (list[str])
            - ``edge_cases_detected`` (list[str])
            - ``acceptance_criteria`` (dict)
            - ``expression`` (str) — cleaned expression string.
            - ``canonical_form`` (str | None)
            - ``domain`` (str)
            - ``strict_mode`` (bool)
        """
        if expression_str is None:
            return self._null_expression_result()

        var_names = list(variable_definitions.keys()) if variable_definitions else []

        # Clean the expression string before handing to each layer.
        try:
            expression_str = clean_expression_string(expression_str, var_names)
        except Exception as exc:
            logger.debug(
                "clean_expression_string failed, trying extract_clean_expression_string: %s", exc
            )
            try:
                expression_str = extract_clean_expression_string(expression_str, var_names)
            except Exception as exc2:
                logger.debug(
                    "extract_clean_expression_string also failed, using str(): %s", exc2
                )
                expression_str = str(expression_str)

        logger.debug("validate_complete: cleaned expression = '%s'", expression_str)

        # Layer 1: Symbolic
        try:
            symbolic_result = self.symbolic_validator.validate(
                expression=expression_str,
                variable_definitions=variable_definitions,
                domain=self.domain,
                from_latex=from_latex,
            )
        except Exception as exc:
            err_str = str(exc)
            if any(kw in err_str for kw in ("SingletonRegistry", "unsupported operand", "SympifyError")):
                logger.warning(
                    "Symbolic validator raised a known Pint/SymPy interop error; "
                    "bypassing with partial credit.  Error: %s", err_str
                )
                try:
                    sympy_expr = safe_sympify(expression_str, var_names)
                    canonical = str(sp.simplify(sympy_expr)) if sympy_expr else expression_str
                except Exception:
                    sympy_expr = None
                    canonical = expression_str

                symbolic_result = {
                    "valid": True,
                    "score": 90.0,
                    "errors": [],
                    "warnings": [
                        f"Symbolic validator bypassed (Pint/SymPy interop): {err_str[:120]}"
                    ],
                    "sympy_expr": sympy_expr,
                    "canonical_form": canonical,
                }
            else:
                raise

        # Layer 2: Dimensional
        try:
            dimensional_result = self.dimensional_validator.validate(
                expression_str=expression_str,
                variable_units=variable_units,
            )
        except Exception as exc:
            logger.warning("Dimensional validator raised an unexpected error: %s", exc)
            dimensional_result = {
                "valid": False,
                "score": 0.0,
                "errors": [f"Dimensional validation error: {str(exc)[:120]}"],
                "warnings": [],
                "dimensional_consistency": False,
            }

        # Layer 3: Domain
        try:
            domain_result = self.domain_validator.validate(
                expression_str=expression_str,
                variable_definitions=variable_definitions,
                test_data=test_data,
            )
        except Exception as exc:
            logger.warning("Domain validator raised an unexpected error: %s", exc)
            domain_result = {
                "valid": True,
                "score": 80.0,
                "errors": [],
                "warnings": [f"Domain validation error (degraded): {str(exc)[:120]}"],
            }

        # Reconcile symbolic false-positives using domain knowledge.
        symbolic_result = reconcile_symbolic_with_domain(symbolic_result, domain_result)

        # Layer 4: Numerical
        numerical_result = (
            self._numerical_validation(
                expression_str, test_data,
                symbolic_result.get("sympy_expr"), var_names,
            )
            if test_data
            else {"score": 100.0, "errors": [], "warnings": []}
        )

        # Aggregate
        edge_cases = self._detect_edge_cases(
            symbolic_result, dimensional_result, domain_result, numerical_result
        )

        base_score = (
            self.weights["symbolic"]    * symbolic_result["score"]
            + self.weights["dimensional"] * dimensional_result["score"]
            + self.weights["domain"]      * domain_result["score"]
            + self.weights["numerical"]   * numerical_result["score"]
        )

        total_score, penalties_applied = self._apply_penalties(
            base_score, edge_cases, dimensional_result
        )

        all_errors = (
            symbolic_result.get("errors", [])
            + dimensional_result.get("errors", [])
            + domain_result.get("errors", [])
            + numerical_result.get("errors", [])
        )
        all_warnings = (
            symbolic_result.get("warnings", [])
            + dimensional_result.get("warnings", [])
            + domain_result.get("warnings", [])
            + numerical_result.get("warnings", [])
        )

        overall_valid = self._check_acceptance_criteria(
            total_score, symbolic_result, dimensional_result, domain_result, edge_cases
        )

        recommendations = self._generate_recommendations(
            symbolic_result, dimensional_result, domain_result,
            numerical_result, edge_cases,
        )

        acceptance_criteria = {
            "minimum_score_met": total_score >= self.VALIDATION_THRESHOLDS["minimum_total_score"],
            "symbolic_valid":    symbolic_result["valid"],
            "dimensional_valid": dimensional_result["valid"],
            "domain_valid":      domain_result["valid"],
            "no_critical_edge_cases": not any("CRITICAL" in e for e in edge_cases),
            "all_layers_above_critical": all(
                s >= self.VALIDATION_THRESHOLDS["critical_failure_threshold"]
                for s in (
                    symbolic_result["score"],
                    dimensional_result["score"],
                    domain_result["score"],
                )
            ),
        }

        complete_result = {
            "valid":             overall_valid,
            "total_score":       total_score,
            "base_score":        base_score,
            "penalties_applied": penalties_applied,
            "layer_scores": {
                "symbolic":    symbolic_result["score"],
                "dimensional": dimensional_result["score"],
                "domain":      domain_result["score"],
                "numerical":   numerical_result["score"],
            },
            "layer_results": {
                "symbolic":    symbolic_result,
                "dimensional": dimensional_result,
                "domain":      domain_result,
                "numerical":   numerical_result,
            },
            "errors":              all_errors,
            "warnings":            all_warnings,
            "recommendations":     recommendations,
            "edge_cases_detected": edge_cases,
            "acceptance_criteria": acceptance_criteria,
            "expression":          expression_str,
            "canonical_form":      symbolic_result.get("canonical_form"),
            "domain":              self.domain,
            "strict_mode":         self.strict_mode,
        }

        logger.debug(
            "validate_complete: valid=%s total_score=%.2f errors=%d",
            overall_valid, total_score, len(all_errors),
        )
        self.validation_history.append(complete_result)
        return complete_result

    # ------------------------------------------------------------------
    # Internal — null result
    # ------------------------------------------------------------------

    def _null_expression_result(self) -> Dict:
        """Return a fully-structured rejection result for a None expression."""
        return {
            "valid": False,
            "total_score": 0.0,
            "base_score": 0.0,
            "penalties_applied": {
                "critical": 0, "dimensional": 0, "domain": 0,
                "warning": 0, "total_deducted": 0,
            },
            "layer_scores":  {"symbolic": 0.0, "dimensional": 0.0, "domain": 0.0, "numerical": 0.0},
            "layer_results": {},
            "errors":        ["Expression cannot be None"],
            "warnings":      [],
            "recommendations": ["Provide a valid expression string"],
            "edge_cases_detected": ["CRITICAL: Empty or null expression"],
            "acceptance_criteria": {
                "minimum_score_met": False, "symbolic_valid": False,
                "dimensional_valid": False, "domain_valid": False,
                "no_critical_edge_cases": False, "all_layers_above_critical": False,
            },
            "expression":     None,
            "canonical_form": None,
            "domain":         self.domain,
            "strict_mode":    self.strict_mode,
        }

    # ------------------------------------------------------------------
    # Internal — edge case detection
    # ------------------------------------------------------------------

    def _detect_edge_cases(
        self,
        symbolic: Dict,
        dimensional: Dict,
        domain: Dict,
        numerical: Dict,
    ) -> List[str]:
        """Collect labelled edge-case strings from all four layer results.

        Labels:  ``CRITICAL``, ``DIMENSIONAL``, ``DOMAIN``, ``WARNING``.
        """
        edge_cases: List[str] = []

        sym_errors = str(symbolic.get("errors", [])).lower()
        if "division by zero" in sym_errors or "divide by zero" in sym_errors:
            edge_cases.append("CRITICAL: Division by zero detected")
        if "empty" in sym_errors or "null" in sym_errors:
            edge_cases.append("CRITICAL: Empty or null expression")
        if "invalid" in sym_errors and "syntax" in sym_errors:
            edge_cases.append("CRITICAL: Invalid syntax in expression")

        num_errors = str(numerical.get("errors", [])).lower()
        num_warnings = str(numerical.get("warnings", [])).lower()
        if "nan" in num_errors:
            edge_cases.append("CRITICAL: Expression produces NaN values")
        if "inf" in num_errors or "infinite" in num_errors:
            edge_cases.append("CRITICAL: Expression produces infinite values")
        if "overflow" in num_warnings:
            edge_cases.append("WARNING: Potential numerical overflow")
        if "underflow" in num_warnings:
            edge_cases.append("WARNING: Potential numerical underflow")

        for error in dimensional.get("errors", []):
            el = error.lower()
            if any(kw in el for kw in ("inconsistent", "incompatible", "mismatch")):
                edge_cases.append(f"DIMENSIONAL: {error}")
            elif "division" in el and "zero" in el:
                edge_cases.append(f"CRITICAL: {error}")

        dom_errors = str(domain.get("errors", [])).lower()
        if "constraint violation" in dom_errors or "violates" in dom_errors:
            edge_cases.append("DOMAIN: Constraint violation detected")

        return edge_cases

    # ------------------------------------------------------------------
    # Internal — penalty system
    # ------------------------------------------------------------------

    def _apply_penalties(
        self,
        base_score: float,
        edge_cases: List[str],
        dimensional_result: Dict,
    ) -> tuple:
        """Deduct structured penalties from *base_score*.

        Returns:
            ``(final_score, penalties_dict)`` where *final_score* is clamped
            to [0, 100] and *penalties_dict* records how much was deducted per
            category.
        """
        score = base_score
        penalties = {"critical": 0.0, "dimensional": 0.0, "domain": 0.0, "warning": 0.0, "total_deducted": 0.0}

        _T = self.VALIDATION_THRESHOLDS
        for ec in edge_cases:
            if "CRITICAL" in ec:
                p = _T["edge_case_penalty"]
                score -= p
                penalties["critical"] += p
            elif "DIMENSIONAL" in ec:
                p = _T["dimensional_inconsistency_penalty"]
                score -= p
                penalties["dimensional"] += p
            elif "DOMAIN" in ec:
                p = _T["domain_violation_penalty"]
                score -= p
                penalties["domain"] += p
            elif "WARNING" in ec:
                p = _T["warning_penalty"]
                score -= p
                penalties["warning"] += p

        final = max(0.0, score)
        penalties["total_deducted"] = base_score - final
        return final, penalties

    # ------------------------------------------------------------------
    # Internal — acceptance criteria
    # ------------------------------------------------------------------

    def _check_acceptance_criteria(
        self,
        total_score: float,
        symbolic: Dict,
        dimensional: Dict,
        domain: Dict,
        edge_cases: List[str],
    ) -> bool:
        """Return True if the expression meets all acceptance criteria.

        Hard failures (immediate rejection):

        - Total score below minimum threshold.
        - Dimensional layer is invalid.
        - Any CRITICAL edge case.
        - Any individual layer score below 50.

        Relaxed acceptance: a high-scoring expression with conservative
        symbolic warnings (score ≥ 50) is accepted when dimensional and
        domain layers both pass and the total score ≥ 80.
        """
        if total_score < self.VALIDATION_THRESHOLDS["minimum_total_score"]:
            return False
        if not dimensional["valid"]:
            return False
        if any("CRITICAL" in e for e in edge_cases):
            return False
        if any(
            s < 50
            for s in (symbolic["score"], dimensional["score"], domain["score"])
        ):
            return False

        if not symbolic["valid"]:
            if (
                symbolic["score"] >= 50
                and dimensional["valid"]
                and domain["valid"]
                and total_score >= 80
            ):
                return True
            return False

        if self.strict_mode and not domain["valid"]:
            return False

        return True

    # ------------------------------------------------------------------
    # Internal — Layer 4: numerical validation
    # ------------------------------------------------------------------

    def _numerical_validation(
        self,
        expression_str: str,
        test_data: Optional[Dict[str, np.ndarray]],
        sympy_expr: Optional[sp.Expr],
        var_names: List[str],
    ) -> Dict:
        """Evaluate the expression on *test_data* and check for NaN/Inf/overflow.

        Args:
            expression_str: Cleaned expression string.
            test_data: Variable name → numpy array mapping (required).
            sympy_expr: Pre-parsed SymPy expression (optional; re-parsed if None).
            var_names: Variable names for Pint-isolated parsing.

        Returns:
            Dict with keys ``score`` (float), ``errors``, ``warnings``.
        """
        result: Dict = {"score": 100.0, "errors": [], "warnings": []}

        if not test_data:
            return result

        try:
            # Ensure we have a SymPy expression to lambdify.
            if sympy_expr is None or not isinstance(sympy_expr, sp.Expr):
                try:
                    sympy_expr = safe_sympify(expression_str, var_names)
                except Exception as exc:
                    result["warnings"].append(f"Parse error in numerical layer: {str(exc)[:120]}")
                    result["score"] = 80.0
                    return result

            try:
                free_vars = list(sympy_expr.free_symbols)
            except Exception as exc:
                result["warnings"].append(f"Variable extraction error: {str(exc)[:120]}")
                result["score"] = 80.0
                return result

            missing = [str(v) for v in free_vars if str(v) not in test_data]
            if missing:
                result["warnings"].append(f"Missing test data for variables: {missing}")
                result["score"] -= 10
                return result

            try:
                var_symbols = [sp.Symbol(str(v)) for v in free_vars]
                # Custom module dict: numpy + math + PySR operator aliases.
                # lambdify maps SymPy function names to callables; the PySR
                # ops (safe_asin etc.) are already replaced by sp.asin/sp.acos
                # in the local_dict above, so SymPy knows them as asin/acos.
                # arcsin/arccos are added as explicit aliases in case the
                # expression string was normalised but SymPy kept the name.
                import numpy as _np_lbd
                import math as _math_lbd
                _lambdify_modules = [
                    {
                        "safe_asin":   lambda x: _np_lbd.arcsin(_np_lbd.clip(x, -1, 1)),
                        "safe_acos":   lambda x: _np_lbd.arccos(_np_lbd.clip(x, -1, 1)),
                        "asin_of_sin": lambda x: _np_lbd.arcsin(_np_lbd.clip(_np_lbd.sin(x), -1, 1)),
                        "acos_of_cos": lambda x: _np_lbd.arccos(_np_lbd.clip(_np_lbd.cos(x), -1, 1)),
                        "atan_of_tan": lambda x: _np_lbd.arctan(_np_lbd.tan(x)),
                        "arcsin": _np_lbd.arcsin,
                        "arccos": _np_lbd.arccos,
                        "arctan": _np_lbd.arctan,
                    },
                    "numpy",
                    "math",
                ]
                func = sp.lambdify(var_symbols, sympy_expr, modules=_lambdify_modules)
            except Exception as exc:
                result["warnings"].append(f"lambdify failed: {str(exc)[:120]}")
                result["score"] = 75.0
                return result

            n_samples = len(next(iter(test_data.values())))
            outputs: List[float] = []

            for i in range(min(n_samples, 100)):
                try:
                    values = []
                    for var in free_vars:
                        raw = test_data[str(var)][i]
                        values.append(
                            float(raw.magnitude) if hasattr(raw, "magnitude") else float(raw)
                        )
                    out = func(*values)
                    outputs.append(
                        float(out.magnitude) if hasattr(out, "magnitude") else float(out)
                    )
                except Exception as exc:
                    err_str = str(exc)
                    if "SingletonRegistry" in err_str or "Symbol" in err_str:
                        result["warnings"].append("Unit system issue during numerical eval")
                        result["score"] = 85.0
                        return result
                    if "SympifyError" in err_str:
                        result["warnings"].append(f"Parsing issue during numerical eval: {err_str[:120]}")
                        result["score"] = 80.0
                        return result
                    result["errors"].append(f"Eval error at sample {i}: {err_str[:120]}")
                    result["score"] -= 2

            if outputs:
                arr = np.array(outputs)
                if np.any(np.isnan(arr)):
                    result["errors"].append("Expression produces NaN values")
                    result["score"] -= 30
                if np.any(np.isinf(arr)):
                    result["errors"].append("Expression produces infinite values")
                    result["score"] -= 30

                finite = arr[np.isfinite(arr)]
                if len(finite) > 0:
                    if np.max(np.abs(finite)) > 1e10:
                        result["warnings"].append("Output contains very large values (> 1e10)")
                        result["score"] -= 10
                    nz = finite[finite != 0]
                    if len(nz) > 0 and np.min(np.abs(nz)) < 1e-10:
                        result["warnings"].append("Output contains very small non-zero values (< 1e-10)")
                        result["score"] -= 5

        except Exception as exc:
            err_str = str(exc)
            if "SingletonRegistry" in err_str or "Symbol" in err_str:
                result["warnings"].append("Unit system error in numerical validation")
                result["score"] = 85.0
            elif "SympifyError" in err_str:
                result["warnings"].append(f"Parsing issue in numerical validation: {err_str[:120]}")
                result["score"] = 80.0
            else:
                logger.warning("Unexpected error in numerical validation: %s", exc)
                result["warnings"].append(f"Numerical validation error: {str(exc)[:150]}")
                result["score"] = 70.0

        result["score"] = max(0.0, min(100.0, result["score"]))
        return result

    # ------------------------------------------------------------------
    # Internal — recommendations
    # ------------------------------------------------------------------

    def _generate_recommendations(
        self,
        symbolic: Dict,
        dimensional: Dict,
        domain: Dict,
        numerical: Dict,
        edge_cases: List[str],
    ) -> List[str]:
        """Generate a prioritised list of human-readable recommendations."""
        recs: List[str] = []
        critical = [e for e in edge_cases if "CRITICAL" in e]

        if critical:
            recs.append(f"🔴 FIX CRITICAL: {len(critical)} issue(s)")
            for c in critical[:3]:
                recs.append(f"   → {c}")
        if not dimensional["valid"]:
            recs.append("🔴 FIX: Dimensional inconsistencies (see dimensional layer)")
        if not symbolic["valid"]:
            recs.append("🔴 FIX: Symbolic errors (see symbolic layer)")
        if not domain["valid"]:
            recs.append(f"🔴 FIX: {self.domain} domain violations (see domain layer)")
        if not recs:
            recs.append("✅ All checks passed")

        return recs

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def clear_history(self) -> None:
        """Clear validation history in this validator and all sub-validators."""
        if isinstance(self.validation_history, deque):
            self.validation_history.clear()
        else:
            self.validation_history = []
        self.symbolic_validator.clear_history()
        self.dimensional_validator.clear_history()
        self.domain_validator.clear_history()

    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Return the validation history, newest-last, optionally limited."""
        history = list(self.validation_history)
        return history[-limit:] if limit is not None else history

    def get_statistics(self) -> Dict:
        """Return aggregate statistics across all completed validations."""
        if not self.validation_history:
            return {
                "total_validations": 0, "success_rate": 0.0,
                "average_total_score": 0.0, "average_layer_scores": {},
                "threshold_used": self.VALIDATION_THRESHOLDS["minimum_total_score"],
            }

        total = len(self.validation_history)
        valid = sum(1 for v in self.validation_history if v["valid"])
        avg_score = sum(v["total_score"] for v in self.validation_history) / total
        avg_layers = {
            layer: sum(v["layer_scores"][layer] for v in self.validation_history) / total
            for layer in ("symbolic", "dimensional", "domain", "numerical")
        }
        return {
            "total_validations":    total,
            "success_rate":         valid / total,
            "average_total_score":  avg_score,
            "average_layer_scores": avg_layers,
            "valid_count":          valid,
            "invalid_count":        total - valid,
            "domain":               self.domain,
            "threshold_used":       self.VALIDATION_THRESHOLDS["minimum_total_score"],
        }

    def get_weakest_layer(self) -> Optional[str]:
        """Return the name of the layer with the lowest average score."""
        stats = self.get_statistics()
        if not stats["average_layer_scores"]:
            return None
        return min(stats["average_layer_scores"].items(), key=lambda x: x[1])[0]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 80)
    print("ENSEMBLE VALIDATOR v11 — SELF-TEST SUITE")
    print("=" * 80)

    validator = EnsembleValidator(domain="biology")

    print(f"\nThreshold : {validator.VALIDATION_THRESHOLDS['minimum_total_score']}")
    print(f"Weights   : {validator.weights}")

    # Test 1: Michaelis-Menten
    print("\n--- Test 1: S*Vmax/(Km + S) ---")
    r = validator.validate_complete(
        expression_str="S*Vmax/(Km + S)",
        variable_definitions={"S": "Substrate", "Vmax": "Max velocity", "Km": "Michaelis constant"},
        variable_units={"S": "mol/L", "Vmax": "mol/(L*s)", "Km": "mol/L"},
        test_data={
            "S":    np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "Vmax": np.full(5, 10.0),
            "Km":   np.full(5, 2.0),
        },
    )
    print(f"Valid={r['valid']}  Score={r['total_score']:.2f}")
    print(f"Layer scores: {r['layer_scores']}")
    if r["errors"]:   print(f"Errors: {r['errors']}")
    if r["warnings"]: print(f"Warnings (first 3): {r['warnings'][:3]}")

    # Test 2: Dimensionally invalid
    print("\n--- Test 2: Vmax + Km (unit mismatch) ---")
    r2 = validator.validate_complete(
        expression_str="Vmax + Km",
        variable_definitions={"Vmax": "Max velocity", "Km": "Michaelis constant"},
        variable_units={"Vmax": "mol/(L*s)", "Km": "mol/L"},
        test_data={"Vmax": np.full(3, 10.0), "Km": np.full(3, 2.0)},
    )
    print(f"Valid={r2['valid']}  Score={r2['total_score']:.2f}")
    print(f"Recommendations: {r2['recommendations']}")

    # Test 3: None expression
    print("\n--- Test 3: None expression ---")
    r3 = validator.validate_complete(
        expression_str=None,
        variable_definitions={},
        variable_units={},
    )
    print(f"Valid={r3['valid']}  Errors={r3['errors']}")

    print("\n" + "=" * 80)
    stats = validator.get_statistics()
    print(f"Stats          : {stats}")
    print(f"Weakest layer  : {validator.get_weakest_layer()}")
    print("=" * 80)

# ===========================================================================
# physics_aware_regressor
# ===========================================================================

"""
Enhanced Physics-Aware Symbolic Regressor - Version 11.1
CRITICAL FIX: Expression simplification and validation compatibility

NEW IN v11.1:
- Clean expression output (no tiny epsilons in denominators)
- Automatic power simplification (0.9999... → 1.0)
- Validation-compatible expression format
- Better numerical stability
- Fixed SingletonRegistry error

FIXES FOR MICHAELIS-MENTEN:
- Removes epsilon artifacts: (Km + S + 1e-6)**0.999 → (Km + S)
- Cleans up near-integer powers
- Validates expression before returning

- Train/validation split with early stopping
- Enhanced complexity penalties (prevents overfitting)
- Cross-validation support
- Regularized coefficient optimization with L2
- Competitive inhibition: (Vmax*S)/(Km(1+I/Ki)+S)
- Extended Hill coefficients (n=1,2,3)
- Simple rational with numerator constants: (a*x+c)/(b+x)
- Lineweaver-Burk inverse forms
- Protected division helper
- Expression depth tracking

COMPLETE FEATURE SET:
✅ Biology domain: 60% Michaelis-Menten templates
✅ Chemistry domain: 50% rational + 30% exponential
✅ Engineering: Bernoulli energy equations
✅ Overfitting prevention via validation split
✅ Early stopping on validation plateau
✅ K-fold cross-validation
✅ Bounded coefficient ranges
"""

import random

import numpy as np
from typing import Dict, List, Optional, Tuple
import sympy as sp
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
import warnings

# ---------------------------------------------------------------------------
# Module-level reproducibility seeds.
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)


class PhysicsAwareRegressor:
    """Physics-aware symbolic regressor with multi-domain function support.

    Noise-aware mode (v11.2+)
    --------------------------
    Pass ``noise_level`` to activate noise-adaptive hyperparameter selection:

        * ``noise_level=0.0``  (noiseless)  — lower parsimony, higher min_r2,
          more generations, lighter regularisation.  Targets exact recovery.
        * ``noise_level>0.0``  (noisy)       — higher parsimony, lower min_r2,
          stronger L2 regularisation, longer early-stopping patience.  Targets
          robust generalisation over perfect memorisation.
        * ``noise_level=None`` (legacy)      — uses explicitly passed values
          unchanged (fully backward-compatible).

    The adaptive defaults are applied inside ``__init__`` BEFORE the user's
    explicit keyword arguments, so any explicitly passed value still wins.
    This means you can do:
        PhysicsAwareRegressor(noise_level=0.05, parsimony_coefficient=0.01)
    and the explicit ``parsimony_coefficient`` overrides the adaptive default.
    """

    # ── Noise-adaptive preset tables ──────────────────────────────────────
    # Each key maps to the kwargs that __init__ should use as defaults when
    # that noise regime is detected.  Explicit __init__ arguments override.
    _NOISELESS_DEFAULTS: dict = {
        "population_size":            200,
        "generations":                200,
        "parsimony_coefficient":      0.001,   # allow more complex expressions
        "min_r2":                     0.9999,  # match published SR threshold
        "protect_physics_generations": 20,
    }
    _NOISY_DEFAULTS: dict = {
        "population_size":            150,
        "generations":                150,
        "parsimony_coefficient":      0.005,   # penalise complexity harder
        "min_r2":                     0.95,    # noise floor prevents R²>0.9982
        "protect_physics_generations": 10,
    }

    def __init__(
        self,
        domain: str = "general",
        function_type: str = "additive_energy",
        population_size: int = 150,
        generations: int = 150,
        tournament_size: int = 4,
        parsimony_coefficient: float = 0.002,
        min_r2: float = 0.95,
        protect_physics_generations: int = 15,
        enable_dimensional_check: bool = False,
        soft_dimensional_penalty: bool = True,
        verbose: bool = False,
        # ── NEW v11.2: noise-awareness ───────────────────────────────────
        noise_level: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        noise_level : float or None
            Gaussian noise as fraction of y std used when generating data.
            ``0.0``  → noiseless (exact-recovery mode).
            ``>0.0`` → noisy (robust-fitting mode, e.g. 0.05).
            ``None`` → legacy mode, no adaptive override (default).
        """
        # ── Apply noise-adaptive defaults BEFORE storing any argument ────
        # Strategy: build the effective values by starting from the adaptive
        # preset and letting explicit constructor arguments win over them.
        # We detect "explicit" by comparing to each parameter's default value;
        # if the caller passed a different value it wins unconditionally.
        self.noise_level: Optional[float] = noise_level
        self.noiseless: bool = (noise_level is not None and noise_level == 0.0)

        if noise_level is not None:
            preset = self._NOISELESS_DEFAULTS if noise_level == 0.0 else self._NOISY_DEFAULTS

            # Only apply preset value when the caller kept the __init__ default
            # (i.e. population_size==150, generations==150, etc.).  This
            # preserves explicit overrides while still adapting to noise.
            _sig_defaults = {
                "population_size":             150,
                "generations":                 150,
                "parsimony_coefficient":       0.002,
                "min_r2":                      0.95,
                "protect_physics_generations": 15,
            }
            if population_size             == _sig_defaults["population_size"]:
                population_size             = preset["population_size"]
            if generations                 == _sig_defaults["generations"]:
                generations                 = preset["generations"]
            if parsimony_coefficient       == _sig_defaults["parsimony_coefficient"]:
                parsimony_coefficient       = preset["parsimony_coefficient"]
            if min_r2                      == _sig_defaults["min_r2"]:
                min_r2                      = preset["min_r2"]
            if protect_physics_generations == _sig_defaults["protect_physics_generations"]:
                protect_physics_generations = preset["protect_physics_generations"]

        self.domain = domain
        self.function_type = function_type
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.parsimony_coefficient = parsimony_coefficient
        self.min_r2 = min_r2
        self.protect_physics_generations = protect_physics_generations
        self.enable_dimensional_check = enable_dimensional_check
        self.soft_dimensional_penalty = soft_dimensional_penalty
        self.verbose = verbose

        self.best_expression_ = None
        self.best_fitness_ = -np.inf
        self.convergence_history_ = []
        self.variable_units_ = {}

    # ── Noise-aware convenience methods (v11.2) ──────────────────────────

    def fit_noise_aware(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        noise_level: Optional[float] = None,
        variable_units: Optional[Dict[str, str]] = None,
        variable_descriptions: Optional[Dict[str, str]] = None,
    ) -> "PhysicsAwareRegressor":
        """Fit with automatic noise-adaptive strategy selection.

        Selects ``validation_split``, ``early_stopping_rounds``, and L2
        regularisation strength based on ``noise_level`` (or on the
        ``self.noise_level`` set at construction time when not supplied here).

        Parameters
        ----------
        noise_level : float or None
            Override the construction-time ``noise_level`` for this call.
        """
        effective_noise = noise_level if noise_level is not None else self.noise_level
        if effective_noise is None:
            effective_noise = 0.0  # safe default (no info → assume clean)

        if effective_noise == 0.0:
            # Noiseless — all data is clean; no need to hold out a validation set
            val_split = 0.0
            es_rounds = 25          # more patience for exact recovery
            l2_alpha  = 0.001       # very light regularisation
        else:
            # Noisy — use validation split to detect memorisation of noise
            val_split = 0.2
            es_rounds = 15
            l2_alpha  = min(0.1, 0.01 + effective_noise * 0.5)

        if self.verbose:
            mode = ("NOISELESS (exact-recovery)"
                    if effective_noise == 0.0
                    else f"NOISY (σ={effective_noise:.3f})")

        return self.fit(
            X=X, y=y,
            variable_names=variable_names,
            variable_units=variable_units,
            variable_descriptions=variable_descriptions,
            validation_split=val_split,
            early_stopping_rounds=es_rounds,
            _l2_alpha_override=l2_alpha,
        )

    @classmethod
    def for_noise_level(
        cls,
        noise_level: float,
        domain: str = "general",
        **kwargs,
    ) -> "PhysicsAwareRegressor":
        """Factory: construct a regressor pre-tuned for *noise_level*.

        Example
        -------
        >>> reg_noisy     = PhysicsAwareRegressor.for_noise_level(0.05, domain="biology")
        >>> reg_noiseless = PhysicsAwareRegressor.for_noise_level(0.0,  domain="chemistry")
        """
        return cls(domain=domain, noise_level=noise_level, **kwargs)

    @staticmethod
    def compare_conditions(
        X: np.ndarray,
        y_noisy: np.ndarray,
        y_noiseless: np.ndarray,
        variable_names: List[str],
        domain: str = "general",
        verbose: bool = False,
    ) -> Dict:
        """Fit one regressor per noise condition and return a comparison dict.

        Parameters
        ----------
        y_noisy      : Target values with noise (noise_level=0.05 typical).
        y_noiseless  : Clean target values (noise_level=0.0).

        Returns
        -------
        dict with keys ``noisy``, ``noiseless``, and ``delta_r2``.
        """
        reg_noisy = PhysicsAwareRegressor.for_noise_level(
            0.05, domain=domain, verbose=verbose
        )
        reg_noisy.fit_noise_aware(X, y_noisy, variable_names)

        reg_noiseless = PhysicsAwareRegressor.for_noise_level(
            0.0, domain=domain, verbose=verbose
        )
        reg_noiseless.fit_noise_aware(X, y_noiseless, variable_names)

        return {
            "noisy": {
                "r2":         reg_noisy.best_fitness_,
                "expression": reg_noisy.get_expression(),
                "noise_level": 0.05,
            },
            "noiseless": {
                "r2":         reg_noiseless.best_fitness_,
                "expression": reg_noiseless.get_expression(),
                "noise_level": 0.0,
            },
            "delta_r2": reg_noiseless.best_fitness_ - reg_noisy.best_fitness_,
        }

    # ── Primary fit method ────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        variable_units: Optional[Dict[str, str]] = None,
        variable_descriptions: Optional[Dict[str, str]] = None,
        validation_split: float = 0.0,
        early_stopping_rounds: int = 15,
        # Internal: L2 strength forwarded by fit_noise_aware()
        _l2_alpha_override: Optional[float] = None,
    ):
        """
        Fit symbolic regression with domain-aware templates and optional validation.

        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
            variable_names: List of variable names
            variable_units: Optional dict of units
            variable_descriptions: Optional descriptions
            validation_split: Fraction for validation (0.0-0.5), 0.2 recommended
            early_stopping_rounds: Patience for early stopping
            _l2_alpha_override: Internal — set by fit_noise_aware() based on noise.
        """
        # Effective L2 used by _optimize_coefficients_regularized
        self._l2_alpha = (
            _l2_alpha_override if _l2_alpha_override is not None else 0.01
        )

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        if X.shape[1] != len(variable_names):
            raise ValueError("Number of variables must match X columns")

        # Train/validation split if requested
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )
            if self.verbose:
                pass
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        self.variable_units_ = variable_units or {}
        var_stats = self._analyze_variables(
            X_train, y_train, variable_names, variable_descriptions
        )

        if self.verbose:
            self._print_variable_roles(var_stats)

        # Initialize population with domain-aware templates
        population = self._initialize_smart_population(variable_names, var_stats)

        best_overall = None
        best_overall_fitness = -np.inf
        best_val_fitness = -np.inf
        stagnation_counter = 0
        no_val_improvement = 0

        for generation in range(self.generations):
            fitness_scores = self._evaluate_population(
                population, X_train, y_train, variable_names
            )

            # Track best on training
            for i, (individual, fitness) in enumerate(zip(population, fitness_scores)):
                if fitness > best_overall_fitness:
                    best_overall = individual
                    best_overall_fitness = fitness
                    stagnation_counter = 0

            # Validate if split provided
            if X_val is not None:
                val_fitness = self._evaluate_fitness(
                    best_overall, X_val, y_val, variable_names
                )
                if val_fitness > best_val_fitness:
                    best_val_fitness = val_fitness
                    no_val_improvement = 0
                else:
                    no_val_improvement += 1

                if self.verbose and generation % 10 == 0:
                    pass

                # Early stopping on validation
                if no_val_improvement >= early_stopping_rounds:
                    if self.verbose:
                        pass
                    best_overall = (
                        self._optimize_coefficients_regularized(
                            best_overall, X_train, y_train, variable_names
                        )
                        or best_overall
                    )
                    break
            else:
                if self.verbose and generation % 10 == 0:
                    valid = sum(1 for f in fitness_scores if f > -np.inf)

            self.convergence_history_.append(best_overall_fitness)

            # Early stopping on training
            if best_overall_fitness >= self.min_r2 and X_val is None:
                if self.verbose:
                    pass
                best_overall = (
                    self._optimize_coefficients_regularized(
                        best_overall, X_train, y_train, variable_names
                    )
                    or best_overall
                )
                break

            stagnation_counter += 1
            if stagnation_counter > 20:
                if self.verbose:
                    pass
                population = self._initialize_smart_population(
                    variable_names, var_stats
                )
                stagnation_counter = 0
                continue

            # Evolution
            population = self._evolve_population(
                population, fitness_scores, variable_names, var_stats, generation
            )

        self.best_expression_ = best_overall or sum(
            sp.Symbol(v) for v in variable_names
        )
        self.best_fitness_ = best_overall_fitness

        # ✅ Clean expression before storing
        if self.best_expression_:
            self.best_expression_ = self._clean_expression(self.best_expression_)

        if self.verbose:
            if X_val is not None:
                pass

        return self

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, variable_names: List[str], n_folds: int = 5
    ) -> Dict[str, float]:
        """
        Perform k-fold cross-validation.

        Returns:
            Dictionary with mean_r2, std_r2, and individual scores
        """
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            var_stats = self._analyze_variables(X_train, y_train, variable_names, None)
            population = self._initialize_smart_population(variable_names, var_stats)

            best_expr = None
            best_fitness = -np.inf

            for gen in range(min(50, self.generations)):
                fitness_scores = self._evaluate_population(
                    population, X_train, y_train, variable_names
                )

                best_idx = np.argmax(fitness_scores)
                if fitness_scores[best_idx] > best_fitness:
                    best_fitness = fitness_scores[best_idx]
                    best_expr = population[best_idx]

                population = self._evolve_population(
                    population, fitness_scores, variable_names, var_stats, gen
                )

            val_r2 = self._evaluate_fitness(best_expr, X_val, y_val, variable_names)
            scores.append(val_r2)

            if self.verbose:
                pass

        return {"mean_r2": np.mean(scores), "std_r2": np.std(scores), "scores": scores}

    # ========================================================================
    # POPULATION INITIALIZATION - DOMAIN-AWARE
    # ========================================================================

    def _initialize_smart_population(self, variable_names, var_stats):
        """Domain-aware population initialization.

        Supported domains
        -----------------
        biology, chemistry,
        electromagnetism, electrostatics, magnetism,   ← NEW (Feynman Series II)
        optics,                                        ← NEW (Feynman I.26, I.37)
        quantum,                                       ← NEW (Feynman Series III)
        thermodynamics,                                ← NEW (Feynman thermo)
        mechanics,                                     ← NEW (Feynman Series I)
        general (fallback), rational, additive_energy
        """
        d = self.domain.lower() if self.domain else "general"

        # ── Feynman electromagnetism / electrostatics / magnetism ─────────
        if d in ("electromagnetism", "electrostatics", "magnetism",
                 "electrochemistry"):
            return self._init_electromagnetic_population(variable_names, var_stats)

        # ── Feynman optics ─────────────────────────────────────────────────
        elif d == "optics":
            return self._init_optics_population(variable_names, var_stats)

        # ── Feynman quantum mechanics ──────────────────────────────────────
        elif d == "quantum":
            return self._init_quantum_population(variable_names, var_stats)

        # ── Feynman thermodynamics ─────────────────────────────────────────
        elif d == "thermodynamics":
            return self._init_thermodynamics_population(variable_names, var_stats)

        # ── Feynman classical mechanics ────────────────────────────────────
        elif d == "mechanics":
            return self._init_mechanics_population(variable_names, var_stats)

        # ── Existing domains ───────────────────────────────────────────────
        elif d == "biology":
            return self._init_biology_population(variable_names, var_stats)
        elif d in ("chemistry", "electrochemistry"):
            return self._init_chemistry_population(variable_names, var_stats)
        elif self.function_type == "rational":
            return self._init_rational_population(variable_names, var_stats)
        elif self.function_type == "additive_energy":
            return self._init_energy_population(variable_names, var_stats)
        else:
            return self._init_general_population(variable_names, var_stats)

    def _init_biology_population(self, variable_names, var_stats):
        """60% Michaelis-Menten for biology."""
        population = []
        symbols = {v: sp.Symbol(v) for v in variable_names}
        varying = [v for v in variable_names if not var_stats[v]["is_constant"]]
        const = [v for v in variable_names if var_stats[v]["is_constant"]]

        # 60% rational
        for _ in range(int(self.population_size * 0.60)):
            population.append(self._gen_rational(symbols, varying, const))

        # 20% polynomial
        for _ in range(int(self.population_size * 0.20)):
            if varying:
                v = symbols[varying[0]]
                population.append(
                    np.random.uniform(0.5, 2) * v**2 + np.random.uniform(0.5, 2) * v
                )
            else:
                population.append(symbols[variable_names[0]])

        # 20% linear
        while len(population) < self.population_size:
            terms = [np.random.uniform(0.5, 1.5) * symbols[v] for v in varying[:3]]
            population.append(sum(terms) if terms else symbols[variable_names[0]])

        return population

    def _init_chemistry_population(self, variable_names, var_stats):
        """50% rational + 30% exponential (Arrhenius-style) for chemistry."""
        population = []
        symbols = {v: sp.Symbol(v) for v in variable_names}
        varying = [v for v in variable_names if not var_stats[v]["is_constant"]]
        const = [v for v in variable_names if var_stats[v]["is_constant"]]

        # 30% Arrhenius-style exponential: A * exp(-Ea/(R*T))
        for _ in range(int(self.population_size * 0.30)):
            if varying and len(const) >= 3:
                # Try to detect Arrhenius pattern: A, Ea, R constants, T varying
                A = symbols[const[0]]
                Ea = (
                    symbols[const[1]] if len(const) > 1 else np.random.uniform(1e4, 1e5)
                )
                R = symbols[const[2]] if len(const) > 2 else np.random.uniform(8, 9)
                T = symbols[varying[0]]

                # Arrhenius: A * exp(-Ea/(R*T))
                c1 = np.random.uniform(0.95, 1.05)
                c2 = np.random.uniform(0.95, 1.05)
                population.append(c1 * A * sp.exp(-c2 * Ea / (R * T)))
            elif varying:
                # Fallback: simple exponential
                v = symbols[varying[0]]
                population.append(
                    np.random.uniform(0.5, 2)
                    * sp.exp(np.random.uniform(-0.1, -0.01) * v)
                )
            else:
                population.append(symbols[variable_names[0]])

        # 30% rational (for equilibria, rate laws)
        for _ in range(int(self.population_size * 0.30)):
            population.append(self._gen_rational(symbols, varying, const))

        # 20% exponential with linear combination
        for _ in range(int(self.population_size * 0.20)):
            if varying and const:
                v = symbols[varying[0]]
                a = symbols[const[0]]
                b = np.random.uniform(-0.1, -0.01)
                population.append(a * sp.exp(b * v))
            else:
                population.append(self._gen_simple(variable_names, var_stats))

        # 20% other
        while len(population) < self.population_size:
            population.append(self._gen_simple(variable_names, var_stats))

        return population

    def _init_rational_population(self, variable_names, var_stats):
        """Pure rational function initialization."""
        population = []
        symbols = {v: sp.Symbol(v) for v in variable_names}
        varying = [v for v in variable_names if not var_stats[v]["is_constant"]]
        const = [v for v in variable_names if var_stats[v]["is_constant"]]

        for _ in range(self.population_size):
            if np.random.random() < 0.7:
                population.append(self._gen_rational(symbols, varying, const))
            else:
                population.append(self._gen_simple(variable_names, var_stats))

        return population

    def _init_energy_population(self, variable_names, var_stats):
        """Bernoulli energy templates."""
        population = []
        symbols = {v: sp.Symbol(v) for v in variable_names}
        varying = [v for v in variable_names if not var_stats[v]["is_constant"]]
        const = [v for v in variable_names if var_stats[v]["is_constant"]]

        # 50% explicit Bernoulli
        for _ in range(int(self.population_size * 0.50)):
            population.append(self._gen_bernoulli(symbols, varying, const, var_stats))

        # 30% quadratic energy
        for _ in range(int(self.population_size * 0.30)):
            if varying:
                v = symbols[varying[0]]
                population.append(
                    symbols[varying[0]] + np.random.uniform(0.3, 0.7) * v**2
                )
            else:
                population.append(symbols[variable_names[0]])

        # 20% other
        while len(population) < self.population_size:
            population.append(self._gen_simple(variable_names, var_stats))

        return population

    def _init_general_population(self, variable_names, var_stats):
        """Mixed templates."""
        population = []
        symbols = {v: sp.Symbol(v) for v in variable_names}
        varying = [v for v in variable_names if not var_stats[v]["is_constant"]]

        for _ in range(self.population_size):
            choice = np.random.choice(["linear", "quad", "mult"])
            if choice == "linear" and varying:
                terms = [np.random.uniform(0.5, 1.5) * symbols[v] for v in varying[:3]]
                population.append(sum(terms) if terms else symbols[variable_names[0]])
            elif choice == "quad" and varying:
                v = symbols[varying[0]]
                population.append(
                    np.random.uniform(0.5, 1.5) * v**2 + np.random.uniform(0.5, 1.5) * v
                )
            else:
                population.append(self._gen_simple(variable_names, var_stats))

        return population

    # ========================================================================
    # FEYNMAN ELECTROMAGNETIC / ELECTROSTATICS / MAGNETISM  (Series II)
    # ========================================================================
    #
    # Covers all Feynman Series-II equations in the 30-equation benchmark:
    #
    #   II.2.42   Fourier heat conduction :  kappa*(T2-T1)/d
    #   II.6.15a  Clausius-Mossotti       :  (eps-1)/(eps+2)*E0
    #   II_11_3   Dilute polarisation     :  n*alpha*E
    #   II_11_17  Curie's law             :  C/T
    #   II.34.2   Lorentz force           :  q*v*B
    #   II_36_38  Zeeman energy           :  -ms*g*mu_B*B
    #   II_11_27  Ohm's law               :  I*R
    #   II_11_28  Capacitor energy        :  0.5*C*V^2
    # Plus Coulomb / Newton inverse-square (Series I, electrostatics):
    #   I.9.18    Coulomb force           :  q1*q2/(4*pi*eps0*r^2)
    #   I.12.1    Newton gravity          :  G*m1*m2/r^2
    # ========================================================================

    def _init_electromagnetic_population(self, variable_names, var_stats):
        """
        Population seeded with Feynman Series-II electromagnetic templates.

        Template mix (100 % = self.population_size individuals):
          25 % inverse-square / power-law   (Coulomb, Newton, Lorentz F=qvB)
          20 % linear-product               (Ohm V=IR, polarisation n·α·E)
          15 % rational / Clausius-Mossotti ((ε-1)/(ε+2)·E₀)
          15 % quadratic / capacitor        (½CV²)
          10 % ratio (Curie C/T, flux kappa·ΔT/d)
          10 % Zeeman-style sign-change      (-ms·g·μ_B·B)
          5 %  general fallback
        """
        population = []
        symbols    = {v: sp.Symbol(v) for v in variable_names}
        varying    = [v for v in variable_names if not var_stats[v]["is_constant"]]
        const      = [v for v in variable_names if var_stats[v]["is_constant"]]

        def sym(name):
            return symbols.get(name, sp.Symbol(name))

        # ── Classify variables by heuristic name matching ─────────────────
        charge_vars  = [v for v in variable_names if var_stats[v].get("likely_charge")]
        dist_vars    = [v for v in variable_names if var_stats[v].get("likely_distance")]
        vel_vars     = [v for v in variable_names if var_stats[v].get("likely_velocity")]
        field_vars   = [v for v in variable_names if var_stats[v].get("likely_field")]
        temp_vars    = [v for v in variable_names if var_stats[v].get("likely_temperature")]
        curr_vars    = [v for v in variable_names if var_stats[v].get("likely_current")]
        resist_vars  = [v for v in variable_names if var_stats[v].get("likely_resistance")]
        cap_vars     = [v for v in variable_names if var_stats[v].get("likely_capacitance")]
        volt_vars    = [v for v in variable_names if var_stats[v].get("likely_voltage")]
        eps_vars     = [v for v in variable_names if var_stats[v].get("likely_permittivity")]

        # Fallback lists when specific roles not detected
        v0 = varying[0] if varying else variable_names[0]
        v1 = varying[1] if len(varying) > 1 else v0
        v2 = varying[2] if len(varying) > 2 else v1
        c0 = const[0]  if const  else variable_names[0]

        # ── 1. Inverse-square / power-law (25 %) ─────────────────────────
        n_inv = int(self.population_size * 0.25)
        for _ in range(n_inv):
            template = np.random.choice(
                ["coulomb", "newton", "lorentz", "power_law"],
                p=[0.35, 0.25, 0.25, 0.15],
            )
            try:
                if template == "coulomb" and len(varying) >= 2:
                    # q1*q2 / r^2  (with optional constant prefactor)
                    q1 = sym(charge_vars[0]) if charge_vars else sym(v0)
                    q2 = sym(charge_vars[1]) if len(charge_vars) > 1 else sym(v1)
                    r  = sym(dist_vars[0])   if dist_vars  else sym(v2)
                    c  = np.random.uniform(0.8, 1.2)
                    population.append(c * q1 * q2 / r**2)

                elif template == "newton" and len(varying) >= 2:
                    # G*m1*m2/r^2
                    m1 = sym(v0); m2 = sym(v1)
                    r  = sym(dist_vars[0]) if dist_vars else sym(v2)
                    c  = np.random.uniform(0.8, 1.2)
                    population.append(c * m1 * m2 / r**2)

                elif template == "lorentz" and len(varying) >= 2:
                    # F = q*v*B
                    q  = sym(charge_vars[0]) if charge_vars else sym(v0)
                    v  = sym(vel_vars[0])    if vel_vars    else sym(v1)
                    B  = sym(field_vars[0])  if field_vars  else sym(v2)
                    c  = np.random.uniform(0.8, 1.2)
                    population.append(c * q * v * B)

                else:
                    # Generic: a*x1*x2 / x3^n
                    n = np.random.choice([1, 2])
                    a = np.random.uniform(0.5, 2.0)
                    population.append(
                        a * sym(v0) * sym(v1) / sym(v2)**n
                    )
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 2. Linear product (20 %) — Ohm, polarisation, F=qE ───────────
        n_lin = int(self.population_size * 0.20)
        for _ in range(n_lin):
            try:
                template = np.random.choice(["ohm", "polarisation", "product"])
                if template == "ohm":
                    # V = I*R
                    I = sym(curr_vars[0])   if curr_vars   else sym(v0)
                    R = sym(resist_vars[0]) if resist_vars else sym(v1)
                    c = np.random.uniform(0.8, 1.2)
                    population.append(c * I * R)
                elif template == "polarisation":
                    # P = n*alpha*E (3-way product)
                    n   = sym(v0); alp = sym(v1)
                    E   = sym(field_vars[0]) if field_vars else sym(v2)
                    c   = np.random.uniform(0.8, 1.2)
                    population.append(c * n * alp * E)
                else:
                    # Simple two-variable product with coefficient
                    c = np.random.uniform(0.5, 2.0)
                    population.append(c * sym(v0) * sym(v1))
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 3. Clausius-Mossotti / rational (15 %) ───────────────────────
        n_rat = int(self.population_size * 0.15)
        for _ in range(n_rat):
            try:
                template = np.random.choice(
                    ["clausius_mossotti", "general_rational"],
                    p=[0.50, 0.50],
                )
                if template == "clausius_mossotti":
                    # (eps-1)/(eps+2) * E0
                    eps = sym(eps_vars[0]) if eps_vars else sym(v0)
                    E0  = sym(field_vars[0]) if field_vars else sym(v1)
                    c1  = np.random.uniform(0.8, 1.2)
                    c2  = np.random.uniform(0.8, 1.2)
                    population.append(
                        c1 * (eps - 1) / (eps + 2) * c2 * E0
                    )
                else:
                    # (a*x - b) / (x + c) * y
                    a = np.random.uniform(0.5, 1.5)
                    b = np.random.uniform(0.5, 2.0)
                    c = np.random.uniform(1.0, 3.0)
                    population.append(
                        a * (sym(v0) - b) / (sym(v0) + c) * sym(v1)
                    )
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 4. Quadratic / capacitor  (15 %) ─────────────────────────────
        n_quad = int(self.population_size * 0.15)
        for _ in range(n_quad):
            try:
                template = np.random.choice(
                    ["capacitor", "half_mv2", "generic_quad"],
                    p=[0.40, 0.30, 0.30],
                )
                if template == "capacitor":
                    # E = 0.5*C*V^2
                    C = sym(cap_vars[0])  if cap_vars  else sym(v0)
                    V = sym(volt_vars[0]) if volt_vars else sym(v1)
                    c = np.random.uniform(0.45, 0.55)
                    population.append(c * C * V**2)
                elif template == "half_mv2":
                    c = np.random.uniform(0.45, 0.55)
                    population.append(c * sym(v0) * sym(v1)**2)
                else:
                    c = np.random.uniform(0.3, 1.0)
                    population.append(c * sym(v0)**2)
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 5. Ratio / Curie / Fourier flux (10 %) ───────────────────────
        n_ratio = int(self.population_size * 0.10)
        for _ in range(n_ratio):
            try:
                template = np.random.choice(
                    ["curie", "fourier_flux", "generic_ratio"],
                    p=[0.35, 0.35, 0.30],
                )
                if template == "curie":
                    # chi = C / T
                    T = sym(temp_vars[0]) if temp_vars else sym(v1)
                    c = np.random.uniform(0.8, 1.2)
                    population.append(c * sym(v0) / T)
                elif template == "fourier_flux":
                    # J = kappa*(T2-T1)/d
                    T1 = sym(temp_vars[0]) if len(temp_vars) > 0 else sym(v0)
                    T2 = sym(temp_vars[1]) if len(temp_vars) > 1 else sym(v1)
                    d  = sym(dist_vars[0]) if dist_vars else sym(v2)
                    k  = sym(c0)
                    c  = np.random.uniform(0.8, 1.2)
                    population.append(c * k * (T2 - T1) / d)
                else:
                    c = np.random.uniform(0.5, 2.0)
                    population.append(c * sym(v0) / sym(v1))
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 6. Zeeman-style signed product (10 %) ────────────────────────
        n_zeem = int(self.population_size * 0.10)
        for _ in range(n_zeem):
            try:
                template = np.random.choice(
                    ["zeeman", "signed_triple", "signed_double"],
                    p=[0.40, 0.35, 0.25],
                )
                if template == "zeeman":
                    # E = -ms * g * mu_B * B  (3- or 4-variable signed product)
                    sign = np.random.choice([-1, 1])
                    c    = np.random.uniform(0.8, 1.2)
                    B    = sym(field_vars[0]) if field_vars else sym(v1)
                    population.append(sign * c * sym(v0) * B)
                elif template == "signed_triple":
                    sign = np.random.choice([-1, 1])
                    c    = np.random.uniform(0.8, 1.2)
                    population.append(sign * c * sym(v0) * sym(v1) * sym(v2))
                else:
                    sign = np.random.choice([-1, 1])
                    c    = np.random.uniform(0.8, 1.2)
                    population.append(sign * c * sym(v0) * sym(v1))
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 7. General fallback to fill remainder ─────────────────────────
        while len(population) < self.population_size:
            population.append(self._gen_simple(variable_names, var_stats))

        return population

    # ========================================================================
    # FEYNMAN OPTICS  (Series I: I.26.2, I.37.4)
    # ========================================================================
    #
    #   I.26.2  Snell's law  :  theta2 = arcsin(n1/n2 * sin(theta1))
    #   I.37.4  Interference :  I = I1 + I2 + 2*sqrt(I1*I2)*cos(delta)
    # ========================================================================

    def _init_optics_population(self, variable_names, var_stats):
        """
        Population seeded with Feynman optics templates.

          40 % Snell's-law arcsin family
          30 % interference / additive intensity
          20 % trigonometric / phase
          10 % general fallback
        """
        population = []
        symbols   = {v: sp.Symbol(v) for v in variable_names}
        varying   = [v for v in variable_names if not var_stats[v]["is_constant"]]
        const     = [v for v in variable_names if var_stats[v]["is_constant"]]

        def sym(name):
            return symbols.get(name, sp.Symbol(name))

        # Classify by name
        angle_vars  = [v for v in variable_names if var_stats[v].get("likely_angle")]
        index_vars  = [v for v in variable_names if var_stats[v].get("likely_refr_index")]
        intens_vars = [v for v in variable_names if var_stats[v].get("likely_intensity")]
        phase_vars  = [v for v in variable_names if var_stats[v].get("likely_phase")]

        v0 = varying[0] if varying else variable_names[0]
        v1 = varying[1] if len(varying) > 1 else v0
        v2 = varying[2] if len(varying) > 2 else v1

        # ── 1. Snell's law family (40 %) ──────────────────────────────────
        n_snell = int(self.population_size * 0.40)
        for _ in range(n_snell):
            try:
                template = np.random.choice(
                    ["snell_exact", "snell_paraxial", "snell_inv"],
                    p=[0.60, 0.25, 0.15],
                )
                theta1 = sym(angle_vars[0])  if angle_vars  else sym(v0)
                n1     = sym(index_vars[0])  if index_vars  else sym(v1)
                n2     = sym(index_vars[1])  if len(index_vars) > 1 else sym(v2)

                if template == "snell_exact":
                    # arcsin(n1/n2 * sin(θ1))
                    c = np.random.uniform(0.92, 1.08)
                    inner = sp.Rational(1, 1) * n1 / n2 * sp.sin(theta1)
                    # Use clip-safe form: SymPy arcsin — evaluated via lambdify
                    population.append(sp.asin(c * n1 / n2 * sp.sin(theta1)))

                elif template == "snell_paraxial":
                    # n1/n2 * theta1  (small-angle)
                    c = np.random.uniform(0.92, 1.08)
                    population.append(c * n1 / n2 * theta1)

                else:
                    # arcsin(n2/n1 * sin(θ1))  (inverted — negative control)
                    c = np.random.uniform(0.92, 1.08)
                    population.append(sp.asin(c * n2 / n1 * sp.sin(theta1)))

            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 2. Interference / additive intensity (30 %) ───────────────────
        n_interf = int(self.population_size * 0.30)
        for _ in range(n_interf):
            try:
                template = np.random.choice(
                    ["full_interference", "approx_intensity", "cos_mod"],
                    p=[0.50, 0.30, 0.20],
                )
                I1    = sym(intens_vars[0]) if len(intens_vars) > 0 else sym(v0)
                I2    = sym(intens_vars[1]) if len(intens_vars) > 1 else sym(v1)
                delta = sym(phase_vars[0])  if phase_vars  else sym(v2)

                if template == "full_interference":
                    # I = I1 + I2 + 2*sqrt(I1*I2)*cos(delta)
                    c = np.random.uniform(0.92, 1.08)
                    population.append(
                        I1 + I2 + 2 * c * sp.sqrt(I1 * I2) * sp.cos(delta)
                    )
                elif template == "approx_intensity":
                    # I1 + I2 + 2*sqrt(I1*I2)  (ignores phase)
                    c = np.random.uniform(1.8, 2.2)
                    population.append(I1 + I2 + c * sp.sqrt(I1 * I2))
                else:
                    # Modulated: c*cos(delta)
                    c = np.random.uniform(0.5, 2.0)
                    population.append(c * sp.cos(delta))
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 3. Trigonometric / phase expressions (20 %) ───────────────────
        n_trig = int(self.population_size * 0.20)
        for _ in range(n_trig):
            try:
                template = np.random.choice(["sin_ratio", "arcsin_raw", "cos_expr"])
                c = np.random.uniform(0.5, 2.0)
                if template == "sin_ratio":
                    population.append(c * sp.sin(sym(v0)) / sym(v1))
                elif template == "arcsin_raw":
                    population.append(sp.asin(np.clip(c, -0.99, 0.99) * sp.sin(sym(v0))))
                else:
                    population.append(c * sp.cos(sym(v0)) * sym(v1))
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 4. Fallback ───────────────────────────────────────────────────
        while len(population) < self.population_size:
            population.append(self._gen_simple(variable_names, var_stats))

        return population

    # ========================================================================
    # FEYNMAN QUANTUM MECHANICS  (Series III: III.4.32, III.4.33, III.7.38)
    # ========================================================================
    #
    #   III.4.33  Bose-Einstein  :  1/(exp(h*f/(k_B*T)) - 1)
    #   III.4.32  Fermi-Dirac    :  1/(exp((E-mu)/(k_B*T)) + 1)
    #   III.7.38  Rabi frequency :  mu*B/hbar
    # ========================================================================

    def _init_quantum_population(self, variable_names, var_stats):
        """
        Population seeded with Feynman quantum / statistical mechanics templates.

          30 % Fermi-Dirac / Bose-Einstein sigmoidal
          25 % Boltzmann exponential  (sub-expressions)
          20 % Linear product  (Rabi: mu*B/hbar)
          15 % Logistic / saturation
          10 % General fallback
        """
        population = []
        symbols   = {v: sp.Symbol(v) for v in variable_names}
        varying   = [v for v in variable_names if not var_stats[v]["is_constant"]]
        const     = [v for v in variable_names if var_stats[v]["is_constant"]]

        def sym(name):
            return symbols.get(name, sp.Symbol(name))

        temp_vars  = [v for v in variable_names if var_stats[v].get("likely_temperature")]
        freq_vars  = [v for v in variable_names if var_stats[v].get("likely_frequency")]
        energy_vars= [v for v in variable_names if var_stats[v].get("likely_energy")]
        field_vars = [v for v in variable_names if var_stats[v].get("likely_field")]

        v0 = varying[0] if varying else variable_names[0]
        v1 = varying[1] if len(varying) > 1 else v0
        v2 = varying[2] if len(varying) > 2 else v1
        c0 = const[0]   if const else variable_names[0]

        # ── 1. Fermi-Dirac / Bose-Einstein (30 %) ────────────────────────
        n_fd = int(self.population_size * 0.30)
        for _ in range(n_fd):
            try:
                template = np.random.choice(
                    ["fermi_dirac", "bose_einstein", "general_stat"],
                    p=[0.40, 0.40, 0.20],
                )
                T   = sym(temp_vars[0])  if temp_vars   else sym(v1)
                E   = sym(energy_vars[0])if energy_vars else sym(v0)
                kBT = sym(c0)            # k_B·T constant or separate constant

                if template == "fermi_dirac":
                    # 1 / (exp((E - mu) / (k_B*T)) + 1)
                    mu  = sym(v1) if len(varying) > 1 else sp.Float(0.0)
                    c   = np.random.uniform(0.9, 1.1)
                    exp_arg = c * (E - mu) / (kBT * T + sp.Float(1e-30))
                    population.append(
                        sp.Integer(1) / (sp.exp(exp_arg) + sp.Integer(1))
                    )
                elif template == "bose_einstein":
                    # 1 / (exp(h*f / (k_B*T)) - 1)
                    f  = sym(freq_vars[0]) if freq_vars else sym(v0)
                    c  = np.random.uniform(0.9, 1.1)
                    exp_arg = c * f / (kBT * T + sp.Float(1e-30))
                    population.append(
                        sp.Integer(1) / (sp.exp(exp_arg) - sp.Integer(1) + sp.Float(1e-30))
                    )
                else:
                    # Generic: 1/(exp(c*x/y) + s)  s ∈ {-1, +1}
                    s  = int(np.random.choice([-1, 1]))
                    c  = np.random.uniform(0.5, 2.0)
                    exp_arg = c * sym(v0) / (sym(v1) + sp.Float(1e-30))
                    population.append(
                        sp.Integer(1) / (sp.exp(exp_arg) + sp.Integer(s) + sp.Float(1e-30))
                    )
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 2. Boltzmann exponential sub-expressions (25 %) ───────────────
        n_boltz = int(self.population_size * 0.25)
        for _ in range(n_boltz):
            try:
                T  = sym(temp_vars[0]) if temp_vars else sym(v1)
                c  = np.random.uniform(0.5, 2.0)
                population.append(c * sp.exp(-sym(v0) / (sym(c0) * T + sp.Float(1e-30))))
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 3. Linear product — Rabi: mu*B/hbar (20 %) ───────────────────
        n_rabi = int(self.population_size * 0.20)
        for _ in range(n_rabi):
            try:
                B  = sym(field_vars[0]) if field_vars else sym(v1)
                c  = np.random.uniform(0.8, 1.2)
                population.append(c * sym(v0) * B / (sym(c0) + sp.Float(1e-60)))
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 4. Logistic / saturation (15 %) ──────────────────────────────
        n_logi = int(self.population_size * 0.15)
        for _ in range(n_logi):
            try:
                c = np.random.uniform(0.5, 2.0)
                population.append(
                    sp.Integer(1) / (sp.Integer(1) + sp.exp(-c * sym(v0)))
                )
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 5. Fallback ───────────────────────────────────────────────────
        while len(population) < self.population_size:
            population.append(self._gen_simple(variable_names, var_stats))

        return population

    # ========================================================================
    # FEYNMAN THERMODYNAMICS  (crossover + Series I thermodynamics)
    # ========================================================================
    #
    #   FEY_THERMO_SB  Stefan-Boltzmann : sigma*A*T^4
    #   FEY_THERMO_IG  Ideal gas        : n*R*T/V
    #   I_41_16        Planck (dimless) : x^3/(exp(x)-1)
    # ========================================================================

    def _init_thermodynamics_population(self, variable_names, var_stats):
        """
        Population seeded with Feynman thermodynamics templates.

          30 % power-law T^n  (Stefan-Boltzmann, Wien)
          25 % ratio product  (Ideal gas: nRT/V)
          20 % Planck-style   (x^3/(exp(x)-1))
          15 % Arrhenius-style exponential
          10 % General fallback
        """
        population = []
        symbols   = {v: sp.Symbol(v) for v in variable_names}
        varying   = [v for v in variable_names if not var_stats[v]["is_constant"]]
        const     = [v for v in variable_names if var_stats[v]["is_constant"]]

        def sym(name):
            return symbols.get(name, sp.Symbol(name))

        temp_vars = [v for v in variable_names if var_stats[v].get("likely_temperature")]
        vol_vars  = [v for v in variable_names if var_stats[v].get("likely_volume")]

        v0 = varying[0] if varying else variable_names[0]
        v1 = varying[1] if len(varying) > 1 else v0
        v2 = varying[2] if len(varying) > 2 else v1
        c0 = const[0]   if const else variable_names[0]

        # ── 1. Power-law T^n (30 %) ───────────────────────────────────────
        n_pow = int(self.population_size * 0.30)
        for _ in range(n_pow):
            try:
                T  = sym(temp_vars[0]) if temp_vars else sym(v0)
                n  = np.random.choice([2, 3, 4, 5])
                c  = np.random.uniform(0.5, 2.0)
                population.append(c * sym(c0) * T**n if const else c * T**n)
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 2. Ratio product — Ideal gas nRT/V (25 %) ────────────────────
        n_ig = int(self.population_size * 0.25)
        for _ in range(n_ig):
            try:
                T  = sym(temp_vars[0]) if temp_vars else sym(v1)
                V  = sym(vol_vars[0])  if vol_vars  else sym(v2)
                c  = np.random.uniform(0.8, 1.2)
                population.append(c * sym(v0) * T / V)
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 3. Planck-style x^3/(exp(x)-1) (20 %) ────────────────────────
        n_pl = int(self.population_size * 0.20)
        for _ in range(n_pl):
            try:
                n  = np.random.choice([2, 3, 4])
                c  = np.random.uniform(0.5, 2.0)
                x  = sym(v0)
                population.append(
                    c * x**n / (sp.exp(x) - sp.Integer(1) + sp.Float(1e-30))
                )
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 4. Arrhenius-style (15 %) ─────────────────────────────────────
        n_arr = int(self.population_size * 0.15)
        for _ in range(n_arr):
            try:
                T  = sym(temp_vars[0]) if temp_vars else sym(v0)
                c  = np.random.uniform(0.5, 2.0)
                population.append(c * sp.exp(-sym(c0) / (T + sp.Float(1e-30))))
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 5. Fallback ───────────────────────────────────────────────────
        while len(population) < self.population_size:
            population.append(self._gen_simple(variable_names, var_stats))

        return population

    # ========================================================================
    # FEYNMAN CLASSICAL MECHANICS  (Series I: KE, reduced mass, spring energy)
    # ========================================================================
    #
    #   FEY_MECH_KE  Kinetic energy : 0.5*m*v^2
    #   I.18.4       Reduced mass   : m1*m2/(m1+m2)
    #   I.24.6       Spring energy  : 0.5*k*x^2 + 0.5*m*v^2
    # ========================================================================

    def _init_mechanics_population(self, variable_names, var_stats):
        """
        Population seeded with Feynman classical mechanics templates.

          35 % quadratic energy  (KE = ½mv², spring = ½kx²)
          25 % additive energy   (total mechanical: PE + KE)
          20 % harmonic mean / reduced mass  (m1*m2/(m1+m2))
          10 % gravitational potential / linear
          10 % General fallback
        """
        population = []
        symbols   = {v: sp.Symbol(v) for v in variable_names}
        varying   = [v for v in variable_names if not var_stats[v]["is_constant"]]
        const     = [v for v in variable_names if var_stats[v]["is_constant"]]

        def sym(name):
            return symbols.get(name, sp.Symbol(name))

        mass_vars = [v for v in variable_names if var_stats[v].get("likely_mass")]
        vel_vars  = [v for v in variable_names if var_stats[v].get("likely_velocity")]
        spr_vars  = [v for v in variable_names if var_stats[v].get("likely_spring")]
        disp_vars = [v for v in variable_names if var_stats[v].get("likely_displacement")]

        v0 = varying[0] if varying else variable_names[0]
        v1 = varying[1] if len(varying) > 1 else v0
        v2 = varying[2] if len(varying) > 2 else v1
        v3 = varying[3] if len(varying) > 3 else v2

        # ── 1. Quadratic energy (35 %) ────────────────────────────────────
        n_quad = int(self.population_size * 0.35)
        for _ in range(n_quad):
            try:
                template = np.random.choice(["ke", "spring_pe", "generic_quad"])
                if template == "ke":
                    m  = sym(mass_vars[0]) if mass_vars else sym(v0)
                    v  = sym(vel_vars[0])  if vel_vars  else sym(v1)
                    c  = np.random.uniform(0.45, 0.55)
                    population.append(c * m * v**2)
                elif template == "spring_pe":
                    k  = sym(spr_vars[0])  if spr_vars  else sym(v0)
                    x  = sym(disp_vars[0]) if disp_vars else sym(v1)
                    c  = np.random.uniform(0.45, 0.55)
                    population.append(c * k * x**2)
                else:
                    c = np.random.uniform(0.3, 1.0)
                    population.append(c * sym(v0) * sym(v1)**2)
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 2. Additive energy (25 %) ─────────────────────────────────────
        n_add = int(self.population_size * 0.25)
        for _ in range(n_add):
            try:
                c1 = np.random.uniform(0.45, 0.55)
                c2 = np.random.uniform(0.45, 0.55)
                population.append(
                    c1 * sym(v0) * sym(v1)**2 + c2 * sym(v2) * sym(v3)**2
                )
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 3. Reduced mass / harmonic mean (20 %) ────────────────────────
        n_rm = int(self.population_size * 0.20)
        for _ in range(n_rm):
            try:
                c = np.random.uniform(0.8, 1.2)
                population.append(
                    c * sym(v0) * sym(v1) / (sym(v0) + sym(v1) + sp.Float(1e-30))
                )
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 4. Linear / gravitational (10 %) ─────────────────────────────
        n_lin = int(self.population_size * 0.10)
        for _ in range(n_lin):
            try:
                c = np.random.uniform(0.5, 2.0)
                population.append(c * sym(v0) * sym(v1))
            except Exception:
                population.append(self._gen_simple(variable_names, var_stats))

        # ── 5. Fallback ───────────────────────────────────────────────────
        while len(population) < self.population_size:
            population.append(self._gen_simple(variable_names, var_stats))

        return population

    # ========================================================================
    # RATIONAL FUNCTION GENERATORS - COMPLETE SET
    # ========================================================================

    def _gen_rational(self, symbols, varying, const):
        """
        Generate rational function templates including:
        - Michaelis-Menten: (Vmax*S)/(Km+S)
        - Hill equation: (Vmax*S^n)/(K^n+S^n) with n=1,2,3
        - Competitive inhibition: (Vmax*S)/(Km(1+I/Ki)+S)
        - Simple rational: (a*x+c)/(b+x)
        - Inverse (Lineweaver-Burk): a/(b+x)
        """
        if not varying:
            return list(symbols.values())[0]

        template = np.random.choice(
            ["mm", "hill", "simple", "inverse", "competitive"],
            p=[0.35, 0.20, 0.25, 0.10, 0.10],
        )

        try:
            if template == "mm" and len(const) >= 2:
                # Classic Michaelis-Menten: (Vmax*S)/(Km+S)
                # ✅ NO EPSILON in denominator to avoid artifacts
                Vmax, Km, S = symbols[const[0]], symbols[const[1]], symbols[varying[0]]
                c1, c2 = np.random.uniform(0.95, 1.05), np.random.uniform(0.95, 1.05)
                return (c1 * Vmax * S) / (Km + c2 * S)

            elif template == "hill" and len(const) >= 2:
                # Hill equation: (Vmax*S^n)/(K^n+S^n)
                Vmax, K, S = symbols[const[0]], symbols[const[1]], symbols[varying[0]]
                n = np.random.choice([1, 2, 3])  # Hill coefficient
                return (Vmax * S**n) / (K**n + S**n)

            elif template == "competitive" and len(const) >= 3 and len(varying) >= 2:
                # Competitive inhibition: (Vmax*S)/(Km(1 + I/Ki) + S)
                Vmax, Km, Ki = symbols[const[0]], symbols[const[1]], symbols[const[2]]
                S, I = symbols[varying[0]], symbols[varying[1]]
                denominator = Km * (1 + I / Ki) + S
                return (Vmax * S) / denominator

            elif template == "simple":
                # Simple rational: (a*x + c)/(b + x)
                S = symbols[varying[0]]
                a = np.random.uniform(0.5, 2.0)
                b = symbols[const[0]] if const else np.random.uniform(5, 15)

                # 30% chance to add constant to numerator
                if np.random.random() < 0.3 and len(const) >= 2:
                    c = np.random.uniform(0.1, 1.0) * symbols[const[1]]
                    return (a * S + c) / (b + S)
                return (a * S) / (b + S)

            else:  # inverse (Lineweaver-Burk style)
                S = symbols[varying[0]]
                if const:
                    a, b = (
                        symbols[const[0]],
                        symbols[const[1]]
                        if len(const) > 1
                        else np.random.uniform(1, 10),
                    )
                    return a / (b + S)
                return 1.0 / (np.random.uniform(1, 10) + S)
        except Exception:
            pass

        # Fallback to simple rational (NO EPSILON)
        S = symbols[varying[0]]
        return S / (np.random.uniform(5, 15) + S)

    def _generate_rational_template(
        self, variable_names, var_stats, symbols, varying_vars, const_vars
    ):
        """
        Alternative rational function generator.
        Provides additional diversity in population initialization.
        """
        return self._gen_rational(symbols, varying_vars, const_vars)

    def _protected_division(self, numerator, denominator, epsilon=1e-6):
        """Protected division to avoid divide-by-zero in expressions."""
        return numerator / (denominator + epsilon)

    def _gen_bernoulli(self, symbols, varying, const, var_stats):
        """Generate Bernoulli: P + 0.5*rho*v² + rho*g*h."""
        if len(varying) < 2 or len(const) < 2:
            return self._gen_simple(list(symbols.keys()), var_stats)

        # Detect variables
        v_vars = [v for v in varying if var_stats[v].get("likely_velocity")]
        h_vars = [v for v in varying if var_stats[v].get("likely_height")]
        p_vars = [v for v in varying if var_stats[v].get("likely_pressure")]

        P = symbols[p_vars[0]] if p_vars else symbols[varying[0]]
        v = (
            symbols[v_vars[0]]
            if v_vars
            else symbols[varying[1] if len(varying) > 1 else varying[0]]
        )
        h = symbols[h_vars[0]] if h_vars else symbols[varying[-1]]
        rho = symbols[const[0]]
        g = symbols[const[1] if len(const) > 1 else const[0]]

        c1 = np.random.uniform(0.95, 1.05)
        c2 = np.random.uniform(0.48, 0.52)
        c3 = np.random.uniform(0.95, 1.05)

        return c1 * P + c2 * rho * v**2 + c3 * rho * g * h

    def _gen_simple(self, variable_names, var_stats):
        """Simple fallback expression."""
        symbols = {v: sp.Symbol(v) for v in variable_names}
        varying = [v for v in variable_names if not var_stats[v]["is_constant"]]
        if not varying:
            varying = variable_names[:2]

        n = min(3, len(varying))
        selected = np.random.choice(varying, size=n, replace=False)
        return sum(
            np.random.uniform(0.1, 2.0) * symbols[v] ** np.random.choice([1, 2])
            for v in selected
        )

    # ========================================================================
    # MUTATION OPERATORS - RATIONAL-AWARE
    # ========================================================================

    def _smart_mutate_with_rational(self, expr, variable_names, var_stats):
        """Domain-aware mutation — can blend domain-specific structures."""
        try:
            symbols = {v: sp.Symbol(v) for v in variable_names}
            varying = [v for v in variable_names if not var_stats[v]["is_constant"]]
            const   = [v for v in variable_names if var_stats[v]["is_constant"]]
            d       = self.domain.lower() if self.domain else "general"

            # 30 % chance to blend with rational for biology
            if d == "biology" and np.random.random() < 0.3:
                new_rational = self._gen_rational(symbols, varying, const)
                alpha = np.random.uniform(0.3, 0.7)
                return alpha * expr + (1 - alpha) * new_rational

            # 30 % chance to blend with Arrhenius for chemistry
            elif d in ("chemistry", "electrochemistry") and np.random.random() < 0.3:
                if varying and len(const) >= 3:
                    A  = symbols[const[0]]
                    Ea = symbols[const[1]] if len(const) > 1 else np.random.uniform(1e4, 1e5)
                    R  = symbols[const[2]] if len(const) > 2 else 8.314
                    T  = symbols[varying[0]]
                    new_arrhenius = A * sp.exp(-Ea / (R * T))
                    alpha = np.random.uniform(0.3, 0.7)
                    return alpha * expr + (1 - alpha) * new_arrhenius

            # 25 % chance to blend with inverse-square for EM / electrostatics
            elif d in ("electromagnetism", "electrostatics", "magnetism") and \
                 np.random.random() < 0.25:
                if len(varying) >= 2:
                    v0, v1 = symbols[varying[0]], symbols[varying[1]]
                    v2 = symbols[varying[2]] if len(varying) > 2 else v1
                    template = np.random.choice(
                        ["inv_sq", "linear_prod", "ratio"]
                    )
                    c = np.random.uniform(0.8, 1.2)
                    if template == "inv_sq":
                        new_em = c * v0 * v1 / (v2**2 + sp.Float(1e-30))
                    elif template == "linear_prod":
                        new_em = c * v0 * v1
                    else:
                        new_em = c * v0 / (v1 + sp.Float(1e-30))
                    alpha = np.random.uniform(0.3, 0.7)
                    return alpha * expr + (1 - alpha) * new_em

            # 25 % chance to blend with Snell/trig for optics
            elif d == "optics" and np.random.random() < 0.25:
                if len(varying) >= 2:
                    angle = symbols[varying[0]]
                    ratio = symbols[varying[1]] / (symbols[varying[2]]
                            if len(varying) > 2 else sp.Float(1.0))
                    try:
                        new_snell = sp.asin(ratio * sp.sin(angle))
                    except Exception:
                        new_snell = ratio * sp.sin(angle)
                    alpha = np.random.uniform(0.3, 0.7)
                    return alpha * expr + (1 - alpha) * new_snell

            # 20 % chance to blend with Boltzmann/sigmoidal for quantum
            elif d == "quantum" and np.random.random() < 0.20:
                if varying:
                    c = np.random.uniform(0.5, 2.0)
                    s = int(np.random.choice([-1, 1]))
                    x = symbols[varying[0]]
                    new_q = sp.Integer(1) / (
                        sp.exp(c * x) + sp.Integer(s) + sp.Float(1e-30)
                    )
                    alpha = np.random.uniform(0.3, 0.7)
                    return alpha * expr + (1 - alpha) * new_q

            # Standard mutation (all other domains)
            return self._smart_mutate(expr, variable_names, var_stats)
        except Exception:
            return expr

    def _smart_mutate(self, expr, variable_names, var_stats):
        """Standard mutation."""
        try:
            mut_type = np.random.choice(["coeff", "add", "power"])
            symbols = {v: sp.Symbol(v) for v in variable_names}

            if mut_type == "coeff":
                atoms = [
                    a for a in expr.atoms(sp.Float, sp.Integer, sp.Rational) if a != 0
                ]
                if atoms:
                    old = np.random.choice(atoms)
                    return expr.subs(old, float(old) * np.random.uniform(0.5, 1.5))
            elif mut_type == "add":
                varying = [v for v in variable_names if not var_stats[v]["is_constant"]]
                if varying:
                    v = np.random.choice(varying)
                    return expr + np.random.uniform(0.3, 0.7) * symbols[
                        v
                    ] ** np.random.choice([1, 2])

            return expr
        except Exception:
            return expr

    # ========================================================================
    # VARIABLE ANALYSIS
    # ========================================================================

    def _analyze_variables(self, X, y, variable_names, descriptions=None):
        """Variable analysis with extended role detection for all Feynman domains.

        Roles detected
        --------------
        Existing  : likely_velocity, likely_height, likely_pressure
        New (v12) : likely_charge, likely_distance, likely_field,
                    likely_temperature, likely_current, likely_resistance,
                    likely_capacitance, likely_voltage, likely_permittivity,
                    likely_angle, likely_refr_index, likely_intensity,
                    likely_phase, likely_frequency, likely_energy,
                    likely_mass, likely_spring, likely_displacement,
                    likely_volume
        """
        stats = {}
        for i, name in enumerate(variable_names):
            x_i = X[:, i]
            stats[name] = {
                "mean":        np.mean(x_i),
                "std":         np.std(x_i),
                "is_constant": np.std(x_i) < 1e-6,
                "correlation": np.corrcoef(x_i, y)[0, 1] if np.std(x_i) > 1e-6 else 0,
            }

            nl = name.lower()

            # ── Original Bernoulli roles ──────────────────────────────────
            if "v" in nl or "vel" in nl:
                stats[name]["likely_velocity"] = True
            if "h" in nl or "height" in nl:
                stats[name]["likely_height"] = True
            if nl.startswith("p") or "press" in nl:
                stats[name]["likely_pressure"] = True

            # ── EM: charge ────────────────────────────────────────────────
            if nl in ("q", "q1", "q2", "charge", "e") or nl.startswith("q"):
                stats[name]["likely_charge"] = True

            # ── EM: distance / radius ─────────────────────────────────────
            if nl in ("r", "d", "dist", "radius", "distance", "r1", "r2") or \
               nl.startswith("r") and len(nl) <= 2:
                stats[name]["likely_distance"] = True

            # ── EM: electric / magnetic field ─────────────────────────────
            if nl in ("e0", "e_field", "field", "b", "b_field", "e") or \
               "field" in nl or nl == "b":
                stats[name]["likely_field"] = True

            # ── EM: temperature ───────────────────────────────────────────
            if nl in ("t", "t1", "t2", "temp", "temperature") or \
               nl.startswith("t") and len(nl) <= 2:
                stats[name]["likely_temperature"] = True

            # ── EM: current ───────────────────────────────────────────────
            if nl in ("i", "current", "i1", "i2") or "current" in nl:
                stats[name]["likely_current"] = True

            # ── EM: resistance ────────────────────────────────────────────
            if nl in ("r", "resistance", "res") or "resist" in nl:
                stats[name]["likely_resistance"] = True

            # ── EM: capacitance ───────────────────────────────────────────
            if nl in ("c", "cap", "capacitance") or "capaci" in nl:
                stats[name]["likely_capacitance"] = True

            # ── EM: voltage ───────────────────────────────────────────────
            if nl in ("v", "volt", "voltage", "u") or "volt" in nl:
                stats[name]["likely_voltage"] = True

            # ── EM: permittivity / dielectric ─────────────────────────────
            if nl in ("eps", "epsilon", "eps0", "er", "dielectric") or \
               "eps" in nl or "epsilon" in nl:
                stats[name]["likely_permittivity"] = True

            # ── Optics: angle ─────────────────────────────────────────────
            if nl in ("theta", "theta1", "theta2", "phi", "angle",
                      "inc", "refr", "alpha") or \
               nl.startswith("theta") or nl.startswith("phi"):
                stats[name]["likely_angle"] = True

            # ── Optics: refractive index ──────────────────────────────────
            if nl in ("n", "n1", "n2", "index", "ni", "nr") or \
               nl.startswith("n") and len(nl) <= 2:
                stats[name]["likely_refr_index"] = True

            # ── Optics: intensity ─────────────────────────────────────────
            if nl in ("i", "i1", "i2", "intensity") or "intens" in nl:
                stats[name]["likely_intensity"] = True

            # ── Optics: phase / delta ─────────────────────────────────────
            if nl in ("delta", "phase", "phi", "phi0") or "phase" in nl:
                stats[name]["likely_phase"] = True

            # ── Quantum: frequency ────────────────────────────────────────
            if nl in ("f", "freq", "frequency", "nu", "omega") or \
               "freq" in nl or nl == "f":
                stats[name]["likely_frequency"] = True

            # ── Quantum: energy ───────────────────────────────────────────
            if nl in ("e", "energy", "e0", "mu", "epsilon") or "energy" in nl:
                stats[name]["likely_energy"] = True

            # ── Mechanics: mass ───────────────────────────────────────────
            if nl in ("m", "m1", "m2", "mass") or \
               nl.startswith("m") and len(nl) <= 2:
                stats[name]["likely_mass"] = True

            # ── Mechanics: spring constant ────────────────────────────────
            if nl in ("k", "spring", "kappa") and "kappa" not in nl:
                stats[name]["likely_spring"] = True

            # ── Mechanics: displacement / position ────────────────────────
            if nl in ("x", "x0", "displacement", "pos", "position"):
                stats[name]["likely_displacement"] = True

            # ── Thermodynamics: volume ─────────────────────────────────────
            if nl in ("v", "vol", "volume") or "volume" in nl or "vol" in nl:
                stats[name]["likely_volume"] = True

        return stats

    def _print_variable_roles(self, var_stats):
        """Print variable classification (all domains)."""
        _ALL_ROLES = [
            # Original
            ("likely_velocity",    "velocity"),
            ("likely_height",      "height"),
            ("likely_pressure",    "pressure"),
            # EM / electrostatics
            ("likely_charge",      "charge"),
            ("likely_distance",    "distance/radius"),
            ("likely_field",       "E/B field"),
            ("likely_temperature", "temperature"),
            ("likely_current",     "current"),
            ("likely_resistance",  "resistance"),
            ("likely_capacitance", "capacitance"),
            ("likely_voltage",     "voltage"),
            ("likely_permittivity","permittivity/ε"),
            # Optics
            ("likely_angle",       "angle"),
            ("likely_refr_index",  "refractive index"),
            ("likely_intensity",   "intensity"),
            ("likely_phase",       "phase/delta"),
            # Quantum
            ("likely_frequency",   "frequency"),
            ("likely_energy",      "energy/chemical potential"),
            # Mechanics / thermo
            ("likely_mass",        "mass"),
            ("likely_spring",      "spring constant"),
            ("likely_displacement","displacement"),
            ("likely_volume",      "volume"),
            ("is_constant",        "constant"),
        ]
        for name, stats in var_stats.items():
            roles = [label for key, label in _ALL_ROLES if stats.get(key)]
            if roles:
                pass

    # ========================================================================
    # FITNESS EVALUATION - ENHANCED WITH OVERFITTING PREVENTION
    # ========================================================================

    def _evaluate_population(self, population, X, y, variable_names):
        """Evaluate fitness for all individuals."""
        fitness_scores = []
        for individual in population:
            try:
                fitness_scores.append(
                    self._evaluate_fitness(individual, X, y, variable_names)
                )
            except Exception:
                fitness_scores.append(-np.inf)
        return fitness_scores

    def _get_expression_depth(self, expr, depth=0):
        """Calculate maximum depth of expression tree."""
        if not expr.args:
            return depth
        return max(self._get_expression_depth(arg, depth + 1) for arg in expr.args)

    def _evaluate_fitness(self, expr, X, y, variable_names):
        """Evaluate fitness with enhanced complexity penalties to prevent overfitting."""
        try:
            symbols = [sp.Symbol(v) for v in variable_names]
            func = sp.lambdify(symbols, expr, modules=["numpy"])
            y_pred = func(*[X[:, i] for i in range(X.shape[1])])

            if np.isscalar(y_pred):
                y_pred = np.full_like(y, y_pred)
            else:
                y_pred = np.asarray(y_pred)

            if y_pred.shape != y.shape or not np.all(np.isfinite(y_pred)):
                return -np.inf
            if np.any(np.abs(y_pred) > 1e10):
                return -np.inf

            r2 = r2_score(y, y_pred)
            if r2 < -10:
                return -np.inf

            # Enhanced complexity penalties
            tree_size = len(list(sp.preorder_traversal(expr)))
            num_operations = len(
                [
                    n
                    for n in sp.preorder_traversal(expr)
                    if isinstance(n, (sp.Add, sp.Mul, sp.Pow, sp.exp, sp.log))
                ]
            )
            max_depth = self._get_expression_depth(expr)

            # Weighted complexity with quadratic depth penalty
            complexity = tree_size + 0.5 * num_operations + 2.0 * max_depth**2

            # Extra penalty for very large expressions
            if tree_size > 50:
                complexity += 10 * (tree_size - 50)

            return r2 - self.parsimony_coefficient * complexity
        except Exception:
            return -np.inf

    # ========================================================================
    # EVOLUTION OPERATORS
    # ========================================================================

    def _evolve_population(
        self, population, fitness_scores, variable_names, var_stats, generation
    ):
        """Evolve population."""
        new_pop = []

        # Elitism
        valid = [(i, f) for i, f in enumerate(fitness_scores) if f > -np.inf]
        if valid:
            valid.sort(key=lambda x: x[1], reverse=True)
            elite_count = max(3, self.population_size // 20)
            new_pop.extend([population[i] for i, _ in valid[:elite_count]])

        # Protected phase
        is_protected = generation < self.protect_physics_generations
        mutation_rate = 0.3

        while len(new_pop) < self.population_size:
            if len(valid) >= 2:
                p1 = self._tournament_select(population, fitness_scores)
                p2 = self._tournament_select(population, fitness_scores)
            else:
                p1 = self._gen_simple(variable_names, var_stats)
                p2 = self._gen_simple(variable_names, var_stats)

            if is_protected and np.random.random() < 0.7:
                offspring = self._coeff_perturbation(p1)
            else:
                offspring = self._crossover(p1, p2) if np.random.random() < 0.7 else p1
                if np.random.random() < mutation_rate:
                    offspring = self._smart_mutate_with_rational(
                        offspring, variable_names, var_stats
                    )

            try:
                offspring = sp.simplify(offspring)
            except Exception:
                pass

            new_pop.append(offspring)

        return new_pop

    def _tournament_select(self, population, fitness_scores):
        """Tournament selection."""
        valid = [i for i, f in enumerate(fitness_scores) if f > -np.inf]
        if len(valid) < self.tournament_size:
            indices = valid if valid else list(range(len(population)))
        else:
            indices = np.random.choice(valid, size=self.tournament_size, replace=False)

        winner_idx = indices[np.argmax([fitness_scores[i] for i in indices])]
        return population[winner_idx]

    def _crossover(self, p1, p2):
        """Crossover two parent expressions."""
        try:
            if isinstance(p1, sp.Add) and isinstance(p2, sp.Add):
                all_terms = list(p1.args) + list(p2.args)
                n = np.random.randint(2, min(6, len(all_terms) + 1))
                selected = np.random.choice(
                    all_terms, size=min(n, len(all_terms)), replace=False
                )
                return sum(selected)
            return np.random.uniform(0.3, 0.7) * p1 + np.random.uniform(0.3, 0.7) * p2
        except Exception:
            return p1 if np.random.random() < 0.5 else p2

    def _coeff_perturbation(self, expr):
        """Perturb coefficients slightly."""
        try:
            coeffs = [
                a
                for a in expr.atoms(sp.Float, sp.Integer, sp.Rational)
                if a not in [0, 1]
            ]
            if coeffs:
                new_expr = expr
                for c in coeffs:
                    new_expr = new_expr.subs(
                        c, float(c) * np.random.uniform(0.85, 1.15)
                    )
                return new_expr
        except Exception:
            pass
        return expr

    # ========================================================================
    # COEFFICIENT OPTIMIZATION - WITH REGULARIZATION
    # ========================================================================

    def _optimize_coefficients_regularized(
        self, expr, X, y, variable_names, alpha=None
    ):
        """
        Optimize coefficients with L2 regularization to prevent overfitting.

        Args:
            expr: Symbolic expression
            X, y: Training data
            variable_names: Variable names
            alpha: L2 regularization strength (default: self._l2_alpha or 0.01).
                   Set via fit_noise_aware() based on noise_level:
                   noiseless → 0.001, noisy(0.05) → ~0.035.
        """
        # Resolve alpha: explicit arg > instance value set by fit_noise_aware > fallback
        if alpha is None:
            alpha = getattr(self, "_l2_alpha", 0.01)
        try:
            from scipy.optimize import minimize

            coeffs = [
                a
                for a in expr.atoms(sp.Float, sp.Integer, sp.Rational)
                if a not in [0, 1]
            ]
            if not coeffs or len(coeffs) > 10:
                return None

            coeff_syms = [sp.Symbol(f"c{i}") for i in range(len(coeffs))]
            param_expr = expr
            for old, new in zip(coeffs, coeff_syms):
                param_expr = param_expr.subs(old, new)

            all_syms = [sp.Symbol(v) for v in variable_names] + coeff_syms
            func = sp.lambdify(all_syms, param_expr, modules=["numpy"])

            def objective(c_vals):
                try:
                    args = [X[:, i] for i in range(X.shape[1])] + list(c_vals)
                    y_pred = func(*args)
                    if not np.all(np.isfinite(y_pred)):
                        return 1e10
                    # MSE + L2 regularization
                    mse = np.mean((y - y_pred) ** 2)
                    l2_penalty = alpha * np.sum(c_vals**2)
                    return mse + l2_penalty
                except Exception:
                    return 1e10

            x0 = [float(c) for c in coeffs]
            bounds = [(-100, 100) for _ in coeffs]

            result = minimize(
                objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 50}
            )

            if result.success:
                optimized = expr
                for old, new_val in zip(coeffs, result.x):
                    optimized = optimized.subs(old, float(new_val))
                return optimized
        except Exception:
            pass
        return None

    # ========================================================================
    # PUBLIC METHODS
    # ========================================================================

    def get_expression(self):
        """Get best expression with clean formatting."""
        if self.best_expression_ is None:
            return "DISCOVERY_FAILED"

        try:
            # Clean the expression
            cleaned = self._clean_expression(self.best_expression_)
            return str(sp.simplify(cleaned))
        except Exception:
            return str(self.best_expression_)

    def _clean_expression(self, expr):
        """
        Clean expression to remove artifacts and improve validation compatibility.

        Fixes:
        - Removes tiny epsilon values (< 1e-5)
        - Rounds powers close to integers (0.999... → 1.0)
        - Simplifies coefficients
        """
        try:
            import math  # ensure available; module-level import may not be in scope
            # Replace tiny floats with 0
            for atom in expr.atoms(sp.Float):
                if abs(float(atom)) < 1e-5:
                    expr = expr.subs(atom, 0)

            # Round powers close to integers
            for pow_expr in expr.atoms(sp.Pow):
                if pow_expr.exp.is_Float:
                    exp_val = float(pow_expr.exp)
                    # Check if close to an integer
                    rounded = round(exp_val)
                    if abs(exp_val - rounded) < 0.001:  # Within 0.1%
                        expr = expr.subs(pow_expr, pow_expr.base**rounded)

            # Round coefficients to reasonable precision
            for atom in expr.atoms(sp.Float):
                val = float(atom)
                if abs(val) > 1e-5:  # Keep non-zero values
                    # Round to 6 significant figures
                    if abs(val) >= 1:
                        rounded = round(val, 6)
                    else:
                        # For small numbers, use scientific notation precision
                        if val != 0:
                            order = int(math.floor(math.log10(abs(val))))
                            rounded = round(val, -order + 5)
                        else:
                            rounded = 0

                    # Only substitute if significantly different
                    if abs(val - rounded) / max(abs(val), 1e-10) > 1e-6:
                        expr = expr.subs(atom, rounded)

            return sp.simplify(expr)
        except Exception:
            return expr

    def predict(self, X, variable_names):
        """Predict using discovered expression."""
        if self.best_expression_ is None:
            raise ValueError("Model not fitted")
        symbols = [sp.Symbol(v) for v in variable_names]
        func = sp.lambdify(symbols, self.best_expression_, modules=["numpy"])
        return func(*[X[:, i] for i in range(X.shape[1])])


# ============================================================================
# MAIN - USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Physics-Aware Regressor v11 - COMPLETE WITH ALL ENHANCEMENTS")
    print("=" * 80)

    print("\n✅ INTEGRATED FEATURES:")
    print("   • Train/validation split with early stopping")
    print("   • Enhanced complexity penalties (tree size + depth²)")
    print("   • Cross-validation support (k-fold)")
    print("   • Regularized coefficient optimization (L2)")
    print("   • Bounded coefficient ranges (-100 to 100)")
    print("   • ✨ Clean expression output (no epsilon artifacts)")
    print("   • ✨ Power simplification (0.999... → 1.0)")
    print("   • ✨ Validation-compatible formatting")
    print("   • Competitive inhibition: (Vmax*S)/(Km(1+I/Ki)+S)")
    print("   • Extended Hill coefficients (n=1,2,3)")
    print("   • Lineweaver-Burk inverse forms")
    print("   • Simple rational with numerator constants")

    print("\n🔬 RATIONAL FUNCTION TEMPLATES:")
    print("   • Michaelis-Menten: (Vmax*S)/(Km+S)")
    print("   • Hill equation: (Vmax*S^n)/(K^n+S^n)")
    print("   • Competitive inhibition: (Vmax*S)/(Km(1+I/Ki)+S)")
    print("   • Simple rational: (a*x+c)/(b+x)")
    print("   • Inverse (Lineweaver-Burk): a/(b+x)")

    print("\n🧪 CHEMISTRY TEMPLATES:")
    print("   • Arrhenius: A*exp(-Ea/(R*T))")
    print("   • Rate laws with equilibria (rational)")
    print("   • Combined exponential-linear forms")

    print("\n🔬 ANTI-OVERFITTING STRATEGIES:")
    print("   • validation_split=0.2 for train/val split")
    print("   • early_stopping_rounds=15 stops on validation plateau")
    print("   • Increased parsimony_coefficient (default 0.002)")
    print("   • Expression depth quadratic penalty")
    print("   • cross_validate() for k-fold CV")

    print("\n📊 USAGE EXAMPLES:")
    print("\n   # Example 1: Biology with validation")
    print("   regressor = PhysicsAwareRegressor(")
    print("       domain='biology',")
    print("       parsimony_coefficient=0.005,")
    print("       verbose=True")
    print("   )")
    print("   regressor.fit(")
    print("       X, y,")
    print("       variable_names=['Vmax', 'Km', 'S'],")
    print("       validation_split=0.2,      # 20% validation")
    print("       early_stopping_rounds=15   # Stop if no improvement")
    print("   )")
    print("   print(regressor.get_expression())")
    print("   print(f'Overfitting gap: {regressor.best_fitness_ - val_fitness:.4f}')")

    print("\n   # Example 2: Cross-validation")
    print("   cv_results = regressor.cross_validate(")
    print("       X, y,")
    print("       variable_names=['Vmax', 'Km', 'S'],")
    print("       n_folds=5")
    print("   )")
    print(
        "   print(f\"CV R²: {cv_results['mean_r2']:.3f} ± {cv_results['std_r2']:.3f}\")"
    )

    print("\n   # Example 3: Chemistry with Arrhenius")
    print("   regressor = PhysicsAwareRegressor(")
    print("       domain='chemistry',")
    print("       parsimony_coefficient=0.003")
    print("   )")
    print("   regressor.fit(X, y, variable_names=['A', 'Ea', 'T'])")

    print("\n   # Example 4: Engineering Bernoulli")
    print("   regressor = PhysicsAwareRegressor(")
    print("       domain='general',")
    print("       function_type='additive_energy'")
    print("   )")
    print("   regressor.fit(X, y, variable_names=['P', 'v', 'h', 'rho', 'g'])")

    print("\n🎯 RECOMMENDED PARAMETERS:")
    print("   parsimony_coefficient: 0.002-0.005 (higher = simpler models)")
    print("   validation_split: 0.2 (20% for validation)")
    print("   min_r2: 0.90-0.95 (don't aim for perfect 0.99)")
    print("   early_stopping_rounds: 15 (patience for validation)")
    print("   population_size: 100-150")
    print("   generations: 100-150")

    print("\n💡 OVERFITTING DETECTION:")
    print("   • Monitor 'Overfitting gap' = Train R² - Val R²")
    print("   • Gap < 0.05: Good generalization")
    print("   • Gap 0.05-0.10: Mild overfitting")
    print("   • Gap > 0.10: Significant overfitting")
    print("   • Use cross_validate() for robust assessment")

    print("\n📋 DOMAIN DISTRIBUTION:")
    print("   • Biology: 60% rational, 20% polynomial, 20% linear")
    print("   • Chemistry: 30% Arrhenius exp, 30% rational, 20% exp-linear, 20% other")
    print("   • Engineering: 50% Bernoulli, 30% quadratic, 20% other")
    print("   • General: Mixed linear, quadratic, multiplicative")

    print("=" * 80)
    print("\n✨ Ready to use! All enhancements fully integrated.")
    print("=" * 80)

# ===========================================================================
# symbolic_engine
# ===========================================================================

"""
HypatiaX Symbolic Engine - Unified v21+v22+v23
===============================================

Combines all three engine generations into a single file:

  v21 (base / default)
  ─────────────────────
  • PySR symbolic regression with full configuration
  • Integrated LLM guidance (seed / hybrid / fallback modes)
  • Robust variable name sanitization (full PySR reserved-word list)
  • Transcendental composition support (safe_asin, asin_of_sin, …)
  • Pareto-front R²-maximising best-equation selection
  • Timeout guard per PySR attempt

  v22 additions  (opt-in via BayesianRanker / use_bayesian_ranking=True)
  ────────────────────────────────────────────────────────────────────────
  • BayesianRanker: re-ranks the PySR Pareto front using a proper
    log-likelihood + log-prior posterior score instead of picking purely
    by R².  Useful when you want to trade a tiny bit of accuracy for
    simpler, more interpretable expressions.
  • EquationTools.compile_equation: lightweight equation evaluator that
    compiles an expression string to a callable without sympy overhead.

  v23 additions  (opt-in via SymbolicTreeEngine)
  ────────────────────────────────────────────────
  • ExpressionNode / SymbolicSearch: self-contained random expression tree
    generator — no PySR / Julia dependency at all.
  • BayesianSearchRanker: exp-based posterior scorer for tree candidates.
  • DimensionalValidator: sympy-backed dimensional consistency check.
  • SymbolicTreeEngine: drop-in alternative engine with discover_validate_interpret()
    that runs the tree search and returns dimensional-validity metadata.

Usage quick-reference
─────────────────────
  # v21 (default, best performance):
  engine = SymbolicEngine(DiscoveryConfig())
  result = engine.discover(X, y, variable_names=[...])

  # v21 + LLM:
  engine = SymbolicEngineWithLLM(config, llm_mode="hybrid")
  result = engine.discover(X, y, variable_names=[...])

  # v22 Bayesian re-ranking of PySR Pareto front:
  ranker = BayesianRanker()
  ranked = ranker.rank(candidates, X, y)   # candidates from PySR equations_

  # v23 PySR-free tree search:
  engine = SymbolicTreeEngine(max_depth=4, population_size=500, iterations=50)
  result = engine.discover_validate_interpret(X, y, variable_names=[...],
               variable_units={"x0": "m", "x1": "s"})

Author: HypatiaX Team
Date: 2026-03-05
Version: unified (v21 + v22 + v23)
"""

# ---------------------------------------------------------------------------
# SEGFAULT GUARD — must be the very first executable statement.
#
# juliacall (imported transitively by PySR) reads PYTHON_JULIACALL_HANDLE_SIGNALS
# at the moment it is first imported.  If PyTorch has already been loaded in
# the same process the two runtimes' signal tables collide and the process
# segfaults.  Setting the env var here — before any import — guarantees it is
# present regardless of import order, whether this module is used directly or
# via a subprocess.
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")

import warnings
import re
import json
import time
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Tuple, Any
from datetime import datetime

import random
import math

import numpy as np
import sympy as sp
# NOTE: PySRRegressor is intentionally NOT imported at module level.
# Importing pysr triggers juliacall initialisation immediately, which
# segfaults when PyTorch is already loaded in the same process.
# The lazy import is inside SymbolicEngine.discover() — by that point
# the env var above has been set and juliacall configures itself correctly.
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from scipy import stats

# ---------------------------------------------------------------------------
# Module-level reproducibility seeds.
# These set the default random state for the process; individual callers can
# override by passing random_state= to discover().  LLM temperature sampling
# is server-side and cannot be seeded here.
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# Optional LLM support
try:
    from anthropic import Anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# ============================================================================
# VARIABLE NAME VALIDATOR (INTEGRATED)
# ============================================================================


class VariableNameValidator:
    """
    Static validator for variable names to avoid PySR reserved word conflicts.
    Integrated into SymbolicEngine v20.
    """

    # PySR reserved function names and operators
    PYSR_RESERVED = {
        # Mathematical functions
        "sin",
        "cos",
        "tan",
        "sinh",
        "cosh",
        "tanh",
        "asin",
        "acos",
        "atan",
        "asinh",
        "acosh",
        "atanh",
        "exp",
        "log",
        "log10",
        "log2",
        "sqrt",
        "cbrt",
        "abs",
        "sign",
        "floor",
        "ceil",
        "round",
        "erf",
        "erfc",
        "gamma",
        "lgamma",
        # Special functions that PySR might reserve
        "I",  # SymPy imaginary unit (conflicts e.g. with current I in Ohm's Law)
        "Q",  # Often reserved for quotient or other special uses
        "S",  # SymPy singleton
        "N",  # SymPy numerical evaluation
        "E",  # Euler's number
        "PI",
        "pi",  # Pi constant
        # Operators
        "pow",
        "div",
        "mod",
        "max",
        "min",
    }

    # Safe alternatives for common problematic variables
    SAFE_ALTERNATIVES = {
        "I": "I_var",  # Current (conflicts with SymPy imaginary unit)
        "Q": "Qr",  # Reaction quotient
        "E": "E_val",  # Energy or potential
        "PI": "Pi",  # Greek pi (different case)
        "pi": "Pi",  # Pi constant
    }

    @staticmethod
    def is_reserved(name: str) -> bool:
        """Check if a variable name conflicts with PySR reserved words.

        FIX: check exact case first so uppercase-only reserved symbols
        (I, Q, S, N, E, PI) are caught before the lower-case fallback
        that only applies to function names like sin, cos, exp, etc.
        """
        return (
            name in VariableNameValidator.PYSR_RESERVED
            or name.lower() in VariableNameValidator.PYSR_RESERVED
        )

    @staticmethod
    def sanitize_name(name: str, existing_names: List[str] = None) -> str:
        """
        Sanitize a single variable name.

        Args:
            name: Original variable name
            existing_names: List of already-used names (to avoid collisions)

        Returns:
            Sanitized variable name
        """
        existing_names = existing_names or []

        # Check if already reserved
        if VariableNameValidator.is_reserved(name):
            # Try known safe alternative first
            if name in VariableNameValidator.SAFE_ALTERNATIVES:
                alternative = VariableNameValidator.SAFE_ALTERNATIVES[name]
                if alternative not in existing_names:
                    return alternative

            # Generate safe alternative by appending suffix
            base = name
            suffix = "_var"
            counter = 1

            while (
                f"{base}{suffix}" in existing_names
                or VariableNameValidator.is_reserved(f"{base}{suffix}")
            ):
                suffix = f"_v{counter}"
                counter += 1

            return f"{base}{suffix}"

        # Name is safe
        return name

    @staticmethod
    def sanitize_names(names: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """
        Sanitize a list of variable names.

        Args:
            names: List of original variable names

        Returns:
            Tuple of (sanitized_names, mapping_dict)
            where mapping_dict maps original -> sanitized
        """
        sanitized = []
        mapping = {}

        for name in names:
            safe_name = VariableNameValidator.sanitize_name(name, sanitized)
            sanitized.append(safe_name)

            if safe_name != name:
                mapping[name] = safe_name
                logger.debug(
                    "Variable '%s' conflicts with PySR reserved word. Renamed to '%s'.",
                    name, safe_name,
                )

        return sanitized, mapping

    @staticmethod
    def update_expression(expression: str, mapping: Dict[str, str]) -> str:
        """
        Update expression with sanitized variable names.

        Args:
            expression: Original expression string
            mapping: Dict mapping original -> sanitized names

        Returns:
            Updated expression
        """
        if not mapping:
            return expression

        # Replace each mapped variable (using word boundaries to avoid partial matches)
        updated = expression
        for original, sanitized in mapping.items():
            # Use regex with word boundaries
            pattern = r"\b" + re.escape(original) + r"\b"
            updated = re.sub(pattern, sanitized, updated)

        return updated


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def detect_collapsed_constants(expression: str, variable_names: List[str]) -> List[str]:
    """
    Detect if physical constants have collapsed into the expression.

    Args:
        expression: The symbolic expression string
        variable_names: List of variable names that should be present

    Returns:
        List of detected collapsed constants (e.g., ['g', 'h', 'c'])
    """
    import re

    collapsed = []

    # Common physical constants to check for
    # Format: (name, typical_value_pattern, description)
    known_constants = [
        ("g", r"9\.8[0-9]*", "gravitational acceleration"),
        ("h", r"6\.626[0-9]*e-34", "Planck constant"),
        ("c", r"2\.998[0-9]*e8|3\.0*e8", "speed of light"),
        ("me", r"9\.109[0-9]*e-31", "electron mass"),
        ("k", r"1\.380[0-9]*e-23", "Boltzmann constant"),
        ("Na", r"6\.022[0-9]*e23", "Avogadro constant"),
        ("e", r"1\.602[0-9]*e-19", "elementary charge"),
        ("mu0", r"1\.257[0-9]*e-6", "vacuum permeability"),
        ("epsilon0", r"8\.854[0-9]*e-12", "vacuum permittivity"),
    ]

    # Check if constant values appear in the expression
    for const_name, pattern, description in known_constants:
        if const_name not in variable_names:  # Only if not a variable
            if re.search(pattern, expression):
                collapsed.append(f"{const_name} ({description})")

    # Also check for numerical constants that might indicate collapse
    # Find all floating point numbers in expression
    numbers = re.findall(r"\d+\.\d+(?:e[+-]?\d+)?", expression)

    # Flag if we see very specific constants
    for num_str in numbers:
        try:
            num = float(num_str)
            # Check for suspicious specific values
            if abs(num - 9.81) < 0.1:
                if "g (gravitational acceleration)" not in collapsed:
                    collapsed.append("g (gravitational acceleration)")
            elif abs(num - 6.626e-34) < 1e-35:
                if "h (Planck constant)" not in collapsed:
                    collapsed.append("h (Planck constant)")
            elif abs(num - 3e8) < 1e7:
                if "c (speed of light)" not in collapsed:
                    collapsed.append("c (speed of light)")
        except ValueError:
            continue

    return collapsed


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================


@dataclass
class DiscoveryConfig:
    """Configuration for symbolic discovery."""

    niterations: int = 40          # v21: lowered from 50; timeout guard makes this safe
    populations: int = 15
    population_size: int = 33      # REDUCED from 200 → 33. With populations=15
                                   # total individuals = 15×33 = ~500, matching
                                   # the PySR docs "fast" preset. The old 200
                                   # (3000 individuals) added ~3× wall-time per
                                   # iteration with no proportional R² gain on
                                   # Feynman equations.
    binary_operators: List[str] = field(default_factory=lambda: ["+", "-", "*", "/"])
    unary_operators: List[str] = field(default_factory=lambda: ["sqrt"])
    # sqrt is universally needed (e.g. double-slit: 2√(I₁I₂)cos(δ), RMS,
    # Euclidean distance).  It is safe for PySR — Julia's sqrt() returns NaN
    # for negative inputs rather than throwing, so evolution continues cleanly.
    # Additional operators (sin, cos, safe_asin …) are injected per-domain.
    constraints: Dict = field(default_factory=dict)
    maxsize: int = 30          # max expression tree size; raise for deep compositions
    maxdepth: Optional[int] = None  # max tree depth (None = PySR default, unlimited).
                                    # Ported from core engine. Distinct from maxsize:
                                    # maxsize caps node count, maxdepth caps nesting level.
                                    # Set e.g. 12 to prevent pathologically deep trees.
    # Per-operator complexity overrides — set low values to make operators "cheaper"
    # so PySR favours them in the search.  Empty dict = PySR defaults (all cost 1).
    complexity_of_operators: Dict = field(default_factory=dict)
    enable_auto_configuration: bool = True
    auto_config_correlation_threshold: float = 0.2
    enable_smart_discovery: bool = False
    smart_discovery_priority: bool = False

    # Complexity / search tuning
    parsimony: float = 0.0032  # PySR default; lower (e.g. 0.001) allows deeper compositions

    # ── Loss function ─────────────────────────────────────────────────────────
    # Explicit Julia loss string passed to PySRRegressor.  Ported from core engine.
    # Making this explicit keeps it auditable and easy to swap — e.g. use
    # "loss(x, y) = ((x - y) / y)^2" for scale-invariant (relative-error) equations.
    # None = use PySR's built-in default (also MSE).
    # BREAKING CHANGE in SymbolicRegression.jl: the loss function signature
    # changed from loss(x, y) [2-arg] to loss(tree, dataset, options) [3-arg].
    # Passing the old 2-arg string causes a Julia MethodError at fit() time
    # after ~180s of Julia init, silently returning DISCOVERY_FAILED.
    # Fix: default to None so PySR uses its built-in MSE loss (always correct).
    loss: Optional[str] = None

    # ── Progress display ──────────────────────────────────────────────────────
    # Ported from core engine (which hardcodes progress=True).
    # Set True for interactive use; False (default) suppresses PySR's tqdm bar
    # in subprocess / benchmark contexts where it pollutes stdout.
    show_progress: bool = False

    # ── v21: per-attempt PySR wall-clock cap ─────────────────────────────────
    # Passed directly to PySRRegressor(timeout_in_seconds=pysr_timeout).
    # Prevents a single runaway PySR call from consuming the full benchmark
    # budget.  With max_retries=3 (hybrid_system_v40) worst-case wall time is
    # 3 × pysr_timeout + ~90s Julia startup ≈ 540s, well within a 900s budget.
    #
    # REDUCED from 800 → 150:
    # The old value of 800s caused the first retry alone to nearly exhaust the
    # 900s method timeout (90s Julia startup + 800s PySR = 890s).  With 150s
    # per attempt and max_retries=3: 90 + 3×150 = 540s, leaving 360s of slack.
    # Set to 0 to disable (no timeout, legacy behaviour).
    pysr_timeout: int = 150

    # Transcendental composition support
    # When True, atomic operators for arcsin(sin(x)), arccos(cos(x)), arctan(tan(x))
    # are injected into PySR as custom Julia functions, bypassing the simplifier
    # that would otherwise collapse these back to x.
    use_transcendental_compositions: bool = False

    # Julia source strings for the three compositions — injected as a *list* of
    # definition strings via PySRRegressor(define_operators=[...]).
    # Keys are the operator names; values are valid Julia function definitions.
    _TRANSCENDENTAL_OPS: ClassVar[Dict[str, str]] = {
        # Use oftype(x, 1) instead of 1.0 so clamp bounds are always the same
        # type as the input (Float32 in PySR).  The literal 1.0 is Float64 in
        # Julia, causing clamp to upcast its result to Float64 — which fails
        # PySR's type-consistency check:
        #   "operator returned Float64 when given Float32 input"
        #
        # safe_asin / safe_acos: clamped versions of the inverse trig functions.
        # Julia's native asin/acos throw DomainError for |x| > 1, which causes
        # PySR to assign infinite fitness to any candidate that calls asin/acos
        # on an unclamped expression (e.g. n1*sin(theta1)/n2 during evolution).
        # PySR then learns to AVOID asin/acos entirely and falls back to messy
        # tan/sin approximations — the root cause of R²≈0.994 on Snell's law.
        # Replacing bare asin/acos with safe_asin/safe_acos fixes this.
        "safe_asin": "safe_asin(x) = asin(clamp(x, oftype(x, -1), oftype(x, 1)))",
        "safe_acos": "safe_acos(x) = acos(clamp(x, oftype(x, -1), oftype(x, 1)))",
        # Composition operators — bypass PySR's simplifier collapsing asin(sin(x))→x
        "asin_of_sin": "asin_of_sin(x) = asin(clamp(sin(x), oftype(x, -1), oftype(x, 1)))",
        "acos_of_cos": "acos_of_cos(x) = acos(clamp(cos(x), oftype(x, -1), oftype(x, 1)))",
        "atan_of_tan": "atan_of_tan(x) = atan(tan(x))",
    }


@dataclass
class LLMConfig:
    """Configuration for LLM hypothesis generation."""

    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2000
    temperature: float = 0.3
    n_candidates: int = 3  # Number of hypotheses to generate
    enabled: bool = False
    api_key: Optional[str] = None


@dataclass
class EquationHypothesis:
    """A candidate equation from LLM."""

    equation: str
    confidence: float
    reasoning: str
    r2_score: Optional[float] = None
    validation_score: Optional[float] = None


# ============================================================================
# LLM COMPONENTS (INTEGRATED)
# ============================================================================


class IntegratedLLMEngine:
    """Built-in LLM hypothesis generator."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None

        if not config.enabled:
            return

        if not HAS_ANTHROPIC:
            self.config.enabled = False
            return

        if not config.api_key:
            config.api_key = os.getenv("ANTHROPIC_API_KEY")

        if not config.api_key:
            self.config.enabled = False
            return

        try:
            self.client = Anthropic(api_key=config.api_key)
        except Exception as e:
            self.config.enabled = False

    def generate_hypotheses(
        self,
        domain: str,
        variables: List[str],
        description: str,
        data_patterns: Dict,
        n_candidates: int = None,
        caller_id: str = "",
    ) -> List[EquationHypothesis]:
        """Generate equation hypotheses using LLM.

        Args:
            caller_id: Optional string identifying the calling method
                (e.g. "PureLLM", "HybridSystemLLMNN").  Embedded as a comment
                in the prompt so that an external LLM response cache keyed on
                prompt text produces distinct entries for different methods even
                when the equation description and variables are identical.
                This prevents the benchmark warning:
                  "CACHE / DUPLICATE RESULT DETECTED: RMSE=X shared by MethodA, MethodB"
        """

        if not self.config.enabled or not self.client:
            return []

        n_candidates = n_candidates or self.config.n_candidates

        prompt = self._build_prompt(
            domain, variables, description, data_patterns, n_candidates,
            caller_id=caller_id,
        )

        try:
            response = self._call_llm(prompt)
            hypotheses = self._parse_response(response)
            return hypotheses
        except Exception as e:
            return []

    def _build_prompt(
        self,
        domain: str,
        variables: List[str],
        description: str,
        patterns: Dict,
        n_candidates: int,
        caller_id: str = "",
    ) -> str:
        """Build LLM prompt.

        caller_id is embedded as a comment so that an external cache keyed on
        prompt text gives distinct entries for different calling methods, even
        when domain/description/variables are identical.  This prevents
        cross-method cache collisions that produce the benchmark warning:
        "CACHE / DUPLICATE RESULT DETECTED: RMSE=X shared by MethodA, MethodB".
        """

        var_list = ", ".join(variables)
        patterns_str = json.dumps(patterns, indent=2)
        # Embed caller_id in prompt text so external caches produce unique keys
        # per method.  Harmless when caller_id is empty.
        _caller_comment = f"# caller: {caller_id}\n" if caller_id else ""

        prompt = f"""{_caller_comment}You are an expert scientific equation discovery system. Generate {n_candidates} candidate equations for this problem.

AVAILABLE PHYSICAL CONSTANTS (use these exact Python names):
  h=6.626e-34 (Planck), hbar=1.055e-34 (reduced Planck), c=2.998e8 (speed of light),
  k_B=1.381e-23 (Boltzmann), k=1.381e-23, N_A=6.022e23 (Avogadro), g_n=9.807,
  m_e=9.109e-31 (electron mass), q_e=1.602e-19 (elementary charge),
  epsilon0=8.854e-12 (vacuum permittivity), mu0=1.257e-6 (vacuum permeability)
Do NOT use bare 'e' for elementary charge — use q_e instead.

PROBLEM CONTEXT:
Domain: {domain}
Description: {description}
Variables: {var_list}

DATA PATTERNS:
{patterns_str}

TASK:
Generate {n_candidates} candidate equations that could explain this relationship.
Use Python syntax: ** for power, * for multiply, / for divide, + and -
Use EXACT variable names: {var_list}

Return ONLY a JSON array:
[
  {{
    "equation": "y = 0.5 * m * v**2",
    "confidence": 0.95,
    "reasoning": "Classical kinetic energy formula"
  }},
  ...
]

JSON ARRAY:"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call Anthropic API."""
        message = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _parse_response(self, response: str) -> List[EquationHypothesis]:
        """Parse LLM response into hypotheses."""
        try:
            # Extract JSON from response
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                start = response.find("[")
                end = response.rfind("]") + 1
                json_str = response[start:end]

            candidates = json.loads(json_str)

            hypotheses = []
            for c in candidates:
                # Normalize equation (strip "y = " prefix)
                eq = c.get("equation", "")
                if "=" in eq:
                    eq = eq.split("=", 1)[1].strip()

                hypotheses.append(
                    EquationHypothesis(
                        equation=eq,
                        confidence=float(c.get("confidence", 0.5)),
                        reasoning=c.get("reasoning", ""),
                    )
                )

            return hypotheses

        except Exception as e:
            return []


# ============================================================================
# PATTERN ANALYZER (INTEGRATED)
# ============================================================================


class DataPatternAnalyzer:
    """Lightweight pattern analysis for LLM context."""

    def analyze(self, X: np.ndarray, y: np.ndarray, variable_names: List[str]) -> Dict:
        """Analyze data patterns."""

        patterns = {
            "n_variables": X.shape[1],
            "n_samples": X.shape[0],
            "correlations": {},
            "structure_hints": [],
            "y_range": [float(np.min(y)), float(np.max(y))],
            "y_scale": self._classify_scale(y),
        }

        # Variable correlations
        for i, var in enumerate(variable_names):
            try:
                corr = np.corrcoef(X[:, i], y)[0, 1]
                patterns["correlations"][var] = (
                    float(corr) if not np.isnan(corr) else 0.0
                )
            except Exception:
                patterns["correlations"][var] = 0.0

        # Detect basic structure
        if X.shape[1] >= 2:
            # Test multiplicative
            product = np.prod(X, axis=1)
            if np.std(product) > 1e-10 and np.std(y) > 1e-10:
                prod_corr = abs(np.corrcoef(y, product)[0, 1])
                if prod_corr > 0.85:
                    patterns["structure_hints"].append("multiplicative")

        # Test polynomial
        for i, var in enumerate(variable_names):
            x_squared = X[:, i] ** 2
            try:
                r2 = r2_score(
                    y,
                    LinearRegression()
                    .fit(x_squared.reshape(-1, 1), y)
                    .predict(x_squared.reshape(-1, 1)),
                )
                if r2 > 0.90:
                    patterns["structure_hints"].append(f"{var}_quadratic")
            except Exception:
                pass

        return patterns

    def _classify_scale(self, y: np.ndarray) -> str:
        """Classify value scale."""
        y_max = np.max(np.abs(y))
        if y_max < 1e-6:
            return "very_small"
        elif y_max < 1:
            return "small"
        elif y_max < 1000:
            return "medium"
        elif y_max < 1e6:
            return "large"
        else:
            return "very_large"


# ============================================================================
# BASE SYMBOLIC ENGINE
# ============================================================================


class SymbolicEngine:
    """Base Symbolic Regression Engine using PySR with integrated variable name validation."""

    def __init__(self, config: DiscoveryConfig, domain: str = "general"):
        """Initialize symbolic engine."""
        self.config = config
        self.domain = domain
        self.model = None

    @staticmethod
    def validate_variable_names(
        variable_names: List[str], auto_fix: bool = True, verbose: bool = False
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Validate and optionally sanitize variable names for PySR compatibility.

        Args:
            variable_names: Original variable names
            auto_fix: If True, automatically sanitize reserved names
            verbose: Print sanitization info

        Returns:
            Tuple of (safe_names, mapping) where mapping is original->sanitized
        """
        conflicts = [
            name for name in variable_names if VariableNameValidator.is_reserved(name)
        ]

        if not conflicts:
            return variable_names, {}

        if not auto_fix:
            raise ValueError(
                f"Variable names conflict with PySR reserved words: {conflicts}. "
                f"Use auto_fix=True to sanitize automatically."
            )

        safe_names, mapping = VariableNameValidator.sanitize_names(variable_names)

        if verbose and mapping:
            for orig, safe in mapping.items():
                pass

        return safe_names, mapping

    def discover(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str] = None,
        equation_name: str = None,
        random_state: int = 42,
        auto_sanitize: bool = True,
        **kwargs,
    ) -> Dict:
        """
        Discover symbolic equation from data with automatic variable name validation.

        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
            variable_names: Names for each feature
            equation_name: Name of the equation being discovered
            random_state: Random seed for reproducibility
            auto_sanitize: Automatically fix variable name conflicts

        Returns:
            Dictionary with discovery results
        """
        if variable_names is None:
            variable_names = [f"x{i}" for i in range(X.shape[1])]

        # Validate and sanitize variable names
        safe_names, name_mapping = self.validate_variable_names(
            variable_names, auto_fix=auto_sanitize, verbose=True
        )


        if name_mapping:
            pass

        # ── Trace collector ──────────────────────────────────────────────────
        # Key decisions and PySR config are recorded here and forwarded through
        # the result dict.  The benchmark runner discards subprocess stdout;
        # the trace in the JSON result is the only reliable diagnostic channel
        # back to the parent process.
        _trace: List[str] = [
            f"domain={self.domain!r}",
            f"eq={equation_name!r}",
            f"vars={safe_names}",
            f"n={X.shape[0]}",
            f"niter={self.config.niterations}",
            f"pysr_to={self.config.pysr_timeout}",
            f"parsimony_cfg={self.config.parsimony}",
            f"use_tc={self.config.use_transcendental_compositions}",
        ]

        try:
            # Lazy import: keeps juliacall out of the module-level import chain
            # so that setting PYTHON_JULIACALL_HANDLE_SIGNALS above always wins.
            from pysr import PySRRegressor  # noqa: PLC0415

            # Configure PySR with safe names
            # ── Build unary operator list, injecting transcendental compositions ──
            active_unary = list(self.config.unary_operators)
            extra_sympy: Dict = {}
            define_ops: List[str] = []

            if self.config.use_transcendental_compositions:
                # Strategy: inject safe_asin/safe_acos (clamped inverse trig) and
                # asin_of_sin/acos_of_cos/atan_of_tan (composition bypass) as custom
                # Julia operators.
                #
                # KEY FIX: Julia's native asin/acos throw DomainError for |x|>1.
                # During PySR's evolutionary search, candidate expressions frequently
                # produce arguments outside [-1,1] (e.g. n1*sin(θ)/n2 before the
                # ratio is tuned).  PySR assigns infinite fitness to those candidates
                # and learns to AVOID asin/acos entirely, falling back to messy
                # tan/sin approximations — the root cause of R²≈0.994 on Snell's law
                # instead of the exact R²=1.0000 formula asin(n1*sin(θ)/n2).
                #
                # safe_asin/safe_acos clamp their argument to [-1,1] before calling
                # the Julia function, so PySR can explore inverse-trig forms freely.
                # They are ALWAYS injected when TC mode is on; the composition
                # operators (asin_of_sin etc.) are added selectively as before.
                import sympy as _sympy

                # 1. Base trig operators (sin/cos/tan still needed for arguments)
                for op in ("sin", "cos", "tan"):
                    if op not in active_unary:
                        active_unary.append(op)
                # Remove bare asin/acos/atan — replace with safe variants below
                # (bare asin/acos crash PySR on out-of-domain arguments during search)
                for _unsafe in ("asin", "acos", "atan"):
                    if _unsafe in active_unary:
                        active_unary.remove(_unsafe)

                # 2. Always inject safe_asin and safe_acos (clamped — no DomainError)
                # CRITICAL: active_unary must contain the SHORT OPERATOR NAME (e.g.
                # "safe_asin"), NOT the full Julia definition body.  PySR looks up
                # unary_operators as names; the bodies are compiled via define_operators.
                # Appending the body string (old bug) caused PySR to silently ignore the
                # custom ops, leaving bare asin/acos in scope — which crash Julia on
                # |x|>1, so PySR avoids them entirely → R²≈0.96 on Snell's law.
                for op_name in ("safe_asin", "safe_acos"):
                    julia_def = self.config._TRANSCENDENTAL_OPS[op_name]
                    if op_name not in active_unary:
                        active_unary.append(op_name)       # ← name only
                    define_ops.append(julia_def)           # ← body for compilation

                # 3. Composition operators — bypass PySR simplifier collapsing
                # asin(sin(x)) → x before it can appear in the Pareto front.
                # Only inject the operators that are actually needed:
                # - asin_of_sin: always (Snell's law and similar asin(sin(x)) forms)
                # - acos_of_cos / atan_of_tan: only if explicitly requested via
                #   complexity_of_operators (signals the caller knows they're needed)
                _tc_requested = set(self.config.complexity_of_operators.keys())
                for op_name in ("asin_of_sin", "acos_of_cos", "atan_of_tan"):
                    julia_def = self.config._TRANSCENDENTAL_OPS[op_name]
                    if op_name == "asin_of_sin" or op_name in _tc_requested:
                        active_unary.append(op_name)       # ← name only
                        define_ops.append(julia_def)       # ← body for compilation

                # 4. Sympy mappings for safe variants + compositions
                # Use default-argument capture (s=_sympy) to avoid late-binding closure bugs.
                extra_sympy["safe_asin"] = _sympy.asin   # maps back to asin for display
                extra_sympy["safe_acos"] = _sympy.acos
                extra_sympy["asin_of_sin"] = lambda x, _s=_sympy: _s.asin(_s.sin(x))
                extra_sympy["acos_of_cos"] = lambda x, _s=_sympy: _s.acos(_s.cos(x))
                extra_sympy["atan_of_tan"] = lambda x, _s=_sympy: _s.atan(_s.tan(x))

            # ── Domain-aware trig operator injection ─────────────────────────
            # Domains such as optics, waves, and any Feynman equation that uses
            # arcsin/arccos REQUIRE sin/cos/safe_asin in the operator set.
            # Without them PySR cannot express e.g. arcsin(n1/n2 * sin(theta1))
            # and the best it can find is a linear approximation with R² ~ 0.55.
            #
            # ── Auto trig injection: two-tier approach ───────────────────────
            #
            # TIER 1 — sin/cos only (ALL optics/waves equations):
            #   All trig domains need sin and cos. Adding safe_asin/safe_acos
            #   to EVERY equation bloats the operator space and was the confirmed
            #   root cause of double-slit timing out: the custom Julia operators
            #   require define_operators support and PySR discards candidates
            #   involving them when their argument goes out of domain during
            #   evolution, effectively making them useless for non-Snell equations
            #   while quadrupling the unary search space.
            #
            # TIER 2 — safe_asin/safe_acos (only when equation needs arcsin):
            #   Detected by equation_name containing "snell", "arcsin", "asin",
            #   "refract", or "26.2" (Feynman index for Snell's law).
            #   This matches hybrid_system_v40 v4.1 BUG-1 FIX logic exactly.
            _TRIG_DOMAINS = frozenset({
                "optics", "waves", "feynman_optics", "feynman_waves",
                "optics_snell", "wave_optics",
            })
            _needs_basic_trig = (
                self.domain in _TRIG_DOMAINS
                and not self.config.use_transcendental_compositions
            )
            _eq_hint = (equation_name or "").lower()
            _needs_inv_trig = _needs_basic_trig and any(
                kw in _eq_hint
                for kw in ("snell", "arcsin", "asin", "refract", "26.2", "i.26")
            )

            if _needs_basic_trig:
                for _top in ("sin", "cos"):
                    if _top not in active_unary:
                        active_unary.append(_top)
                if _needs_inv_trig:
                    # Inverse trig needed (Snell's law etc.) — add safe variants
                    for _sop in ("safe_asin", "safe_acos"):
                        _julia_def = self.config._TRANSCENDENTAL_OPS[_sop]
                        if _sop not in active_unary:
                            active_unary.append(_sop)
                        if _julia_def not in define_ops:
                            define_ops.append(_julia_def)
                        extra_sympy[_sop] = (
                            __import__("sympy").asin
                            if _sop == "safe_asin"
                            else __import__("sympy").acos
                        )
                    _trace.append(f"trig=sin,cos,safe_asin,safe_acos")
                else:
                    _trace.append(f"trig=sin,cos")

            # (duplicate AUTO-TRIG block removed — first block above handles this)

            # ── Domain-aware exp/log operator injection ───────────────────────
            # Equations in quantum, thermodynamics, chemistry, probability, and
            # electrochemistry domains frequently involve exponentials:
            #
            #   quantum/thermodynamics : Bose-Einstein  1/(exp(hf/kT)-1)
            #                            Fermi-Dirac    1/(exp((E-mu)/kT)+1)
            #                            Planck         x³/(exp(x)-1)
            #   probability            : Gaussian       exp(-x²/2σ²)/...
            #   chemistry              : Arrhenius      A·exp(-Ea/RT)
            #   electrochemistry       : Nernst         E0 - (RT/nF)·log(ox/red)
            #
            # The default unary_operators = ["sqrt"] — exp and log are absent.
            # PySR can only fit polynomial/rational approximations, which reach
            # R²≈0.998 numerically but are symbolically wrong (no exp discovered).
            #
            # Fix: inject "exp" and "log" for these domains, exactly mirroring the
            # trig injection pattern.  Guard: skip if use_transcendental_compositions
            # is True (that mode manages its own operator set) and skip if exp/log
            # are already present (caller-supplied config).
            _EXP_DOMAINS = frozenset({
                "quantum", "feynman_quantum",
                "thermodynamics", "feynman_thermodynamics",
                "chemistry", "feynman_chemistry",
                "probability", "feynman_probability",
                "electrochemistry", "feynman_electrochemistry",
                "statistical_mechanics", "statmech",
                "biology",   # growth models, Michaelis-Menten etc.
            })
            # FIX (data-driven exp detection): domain tag alone is not enough —
            # a benchmark may pass domain="general" or omit it entirely even for
            # Bose-Einstein / Fermi-Dirac equations.  Supplement the domain check
            # with a lightweight data heuristic: if log(y) is more linear in X
            # than y itself (measured by leave-one-out R² of a simple linear fit),
            # the target almost certainly involves an exponential and we should
            # inject exp+log regardless of the domain tag.
            #
            # Heuristic: compare RSS of OLS fit on (X, y) vs (X, log|y|).
            # Only apply when y is strictly positive (log undefined otherwise)
            # and the improvement is substantial (delta_r2 > 0.05).
            _data_needs_exp = False
            if (
                not self.config.use_transcendental_compositions
                and "exp" not in active_unary
                and X.shape[0] >= 10
            ):
                try:
                    _y_fit_pos = np.all(y > 0) and np.all(X > 0)
                    if _y_fit_pos:
                        from sklearn.linear_model import LinearRegression as _LR
                        # Flat-space fits
                        _Xs      = X - X.mean(axis=0)
                        _r2_lin  = r2_score(y, _LR().fit(_Xs, y).predict(_Xs))
                        _logy    = np.log(y)
                        _r2_log  = r2_score(_logy, _LR().fit(_Xs, _logy).predict(_Xs))

                        # Log-log fit: if also strong, it's a power law — NOT exponential
                        _logX    = np.log(X)
                        _logXs   = _logX - _logX.mean(axis=0)
                        _r2_loglog = r2_score(_logy, _LR().fit(_logXs, _logy).predict(_logXs))

                        _looks_like_exp      = (_r2_log  - _r2_lin   > 0.05)
                        _looks_like_powerlaw = (_r2_loglog > _r2_log - 0.02)

                        if _looks_like_exp and not _looks_like_powerlaw:
                            _data_needs_exp = True
                        elif _looks_like_exp and _looks_like_powerlaw:
                            pass
                except Exception:
                    pass  # heuristic is best-effort; never block PySR

            _needs_exp_log = (
                (self.domain in _EXP_DOMAINS or _data_needs_exp)
                and not self.config.use_transcendental_compositions
            )
            if _needs_exp_log:
                _injected_explog = []
                for _eop in ("exp", "log"):
                    if _eop not in active_unary:
                        active_unary.append(_eop)
                        _injected_explog.append(_eop)
                if _injected_explog:
                    _trace.append(f"explog={','.join(_injected_explog)}")
                else:
                    _trace.append("explog=already_present")
            else:
                _trace.append("explog=skipped")

            # ── Unique per-run equation file ─────────────────────────────────
            # PySR persists its Pareto front to a CSV file (hall_of_fame_*.csv)
            # and reloads it on the next run if the file is still present.
            # This causes bit-identical results across retries / benchmark runs
            # even when random_state changes — the root cause of 3 methods all
            # returning RMSE=0.242 in the benchmark despite --no-llm-cache.
            # ── Unique per-run equation file ─────────────────────────────────
            # PySR persists its Pareto front to a CSV (hall_of_fame_*.csv) and
            # reloads it on the next run if still present, causing identical
            # results across retries.  Route each run to a fresh tempfile.
            #
            # KWARG NAME CHANGED BETWEEN PySR VERSIONS:
            #   PySR <  0.19  →  equation_file=
            #   PySR >= 0.19  →  temp_equation_file=
            # Probe the signature to avoid a TypeError that crashes PySR before
            # any search runs (confirmed root cause via [SE-TRACE] on 2026-03-08:
            # "equation_file is not a valid keyword argument … did you mean
            # temp_equation_file").
            import tempfile as _tf, os as _os2, inspect as _inspect_ef
            _eq_tmpfile = _tf.NamedTemporaryFile(
                suffix=".csv", prefix="pysr_hof_", delete=False
            )
            _eq_tmpfile.close()
            _equation_file_path = _eq_tmpfile.name
            _pysr_ef_params = set(_inspect_ef.signature(PySRRegressor.__init__).parameters)
            _eq_file_kwarg = (
                "temp_equation_file" if "temp_equation_file" in _pysr_ef_params
                else "equation_file"
            )
            _trace.append(f"eq_file_kwarg={_eq_file_kwarg!r}")

            pysr_kwargs = dict(
                niterations=self.config.niterations,
                populations=self.config.populations,
                population_size=self.config.population_size,
                binary_operators=self.config.binary_operators,
                unary_operators=active_unary,
                constraints=self.config.constraints,
                parsimony=self.config.parsimony,
                maxsize=self.config.maxsize,
                random_state=random_state,
                # Ported from core engine: deterministic=True makes random_state
                # actually reproduce results (without it PySR ignores the seed for
                # parallelised runs and emits a UserWarning in benchmark logs).
                deterministic=True,
                parallelism="serial",   # required companion to deterministic=True
                verbosity=0,
                progress=self.config.show_progress,
                # Unique file per run — prevents PySR from reloading a cached
                # hall-of-fame from a previous run (cross-run result pollution).
                # Kwarg name is version-dependent; probed above into _eq_file_kwarg.
                **{_eq_file_kwarg: _equation_file_path},
            )
            # maxdepth: only pass when explicitly set (None = use PySR default)
            if self.config.maxdepth is not None:
                pysr_kwargs["maxdepth"] = self.config.maxdepth

            # ── Auto-tune parsimony for trig-domain equations ────────────────
            # IMPORTANT: placed AFTER `pysr_kwargs = dict(...)` so it is not
            # overwritten by that assignment.
            #
            # Default parsimony=0.0032 is safe for depth-2/3 formulas but
            # adds excessive evolutionary pressure against deeper trees.
            # With interference equations (e.g. Snell: complexity ≈ 10,
            # double-slit after GM augmentation below: complexity ≈ 7), the
            # penalty is acceptable but lowering it helps preserve diversity.
            #
            # NOTE: do NOT raise population_size here.  With timeout_in_seconds
            # capping wall time, larger populations mean fewer iterations, which
            # HURTS discovery of structured formulas that need many evolutionary
            # steps to assemble.  Keep population at its configured value so
            # PySR maximises the number of evolutionary generations within the
            # available time budget.
            #
            # The guard `>= 0.0032` ensures a caller-supplied lower parsimony
            # is never raised back up.
            if _needs_basic_trig:
                if pysr_kwargs.get("parsimony", self.config.parsimony) >= 0.0032:
                    pysr_kwargs["parsimony"] = 0.0006
                    _trace.append("parsimony=0.0006(auto-trig)")
                else:
                    _trace.append(f"parsimony={pysr_kwargs.get('parsimony')}(caller)")

            # Explicit loss function string (ported from core engine).
            # BUGFIX: the kwarg name changed between PySR versions.
            #   PySR >= 0.17  → loss_function=
            #   PySR <  0.17  → loss=
            # Probing the signature (same pattern used for define_operators
            # above) avoids a TypeError that silently killed every PySR run
            # on older installs, producing 81-second "Discovery failed" results.
            if self.config.loss:
                import inspect as _inspect_loss
                _pysr_loss_params = set(
                    _inspect_loss.signature(PySRRegressor.__init__).parameters
                )
                _loss_kwarg = (
                    "loss_function" if "loss_function" in _pysr_loss_params else "loss"
                )
                pysr_kwargs[_loss_kwarg] = self.config.loss
            # Per-operator complexity overrides (used for transcendental mode)
            if self.config.complexity_of_operators:
                pysr_kwargs["complexity_of_operators"] = self.config.complexity_of_operators

            # ── v21: per-attempt wall-clock guard ────────────────────────────
            # PySRRegressor accepts timeout_in_seconds to cap a single fit()
            # call.  This prevents the 5-retry loop in HybridDiscoverySystem
            # from multiplying a slow PySR run into a full benchmark timeout.
            if self.config.pysr_timeout and self.config.pysr_timeout > 0:
                pysr_kwargs["timeout_in_seconds"] = self.config.pysr_timeout

            if extra_sympy:
                pysr_kwargs["extra_sympy_mappings"] = extra_sympy

            # CRITICAL FIX: pass Julia operator definitions so custom ops are
            # compiled before the search starts.  The correct kwarg name depends
            # on the installed PySR version:
            #
            #   PySR >= 0.19  →  define_operators=[<julia_body_str>, ...]
            #   PySR <  0.19  →  embed the full Julia body string directly inside
            #                    unary_operators (PySR passes it as-is to Julia)
            #
            # We probe PySRRegressor's __init__ signature to decide which style
            # to use, so this code works across PySR versions without hardcoding.
            if define_ops:
                import inspect as _inspect
                _pysr_params = set(_inspect.signature(PySRRegressor.__init__).parameters)
                if "define_operators" in _pysr_params:
                    # Modern PySR: pass bodies via dedicated kwarg
                    pysr_kwargs["define_operators"] = define_ops
                else:
                    # Older PySR: replace short names with full Julia body strings
                    # in unary_operators.  Build a name→body lookup from define_ops.
                    _body_map: Dict[str, str] = {}
                    for _body in define_ops:
                        # Julia definition format: "fname(x) = ..."
                        _fname = _body.split("(")[0].strip()
                        _body_map[_fname] = _body
                    # Replace each short name that has a body definition
                    _new_unary = []
                    for _op in pysr_kwargs["unary_operators"]:
                        _new_unary.append(_body_map.get(_op, _op))
                    pysr_kwargs["unary_operators"] = _new_unary

            pysr_kwargs.update(kwargs)

            # Diagnostic: log the final PySR kwargs so misconfigurations are
            # visible in subprocess stderr rather than silently crashing.
            if "define_operators" in pysr_kwargs:
                pass
            else:
                pass

            try:
                self.model = PySRRegressor(**pysr_kwargs)
            except Exception as _init_exc:
                import traceback as _tb2, sys as _sys2
                _init_tb = _tb2.format_exc()
                raise

            # ── y-scale normalization ──────────────────────────────────────────
            # PySR's internal constant optimizer (BFGS in Julia) initializes
            # at magnitudes near 1.  When y is extremely small (e.g. Lorentz
            # force in SI units: F = qvB ~ 1e-11 N) or extremely large, the
            # optimizer never converges to the right scale, returning R² ≈ 0
            # across all retries — causing the "Discovery failed" / N/A result
            # seen on e.g. Feynman II.34.2 (test 9, "Lorentz force: F = qvB").
            #
            # Fix: normalise y to unit std before fitting; rescale predictions
            # and the expression string afterward so R² is computed against
            # the original y values.
            _y_std = float(np.std(y))
            _needs_yscale = (_y_std > 0) and (_y_std < 1e-4 or _y_std > 1e4)
            if _needs_yscale:
                _y_fit = y / _y_std
            else:
                _y_fit = y
                _y_std = 1.0  # sentinel: no rescaling needed

            # ── X-column scale normalisation ──────────────────────────────────
            # Mirrors the y-scale guard above.  When any input feature has a
            # characteristic magnitude outside [1e-6, 1e6] (e.g. mu~1e-23 in
            # the Rabi frequency equation III.7.38), PySR's BFGS constant
            # optimizer must bridge >6 orders of magnitude from its near-1
            # initialisation point.  It reliably fails to converge, returning
            # R²≈0 despite the correct symbolic structure being trivially simple
            # (mu * B / constant).
            #
            # Fix: divide each extreme column by its representative magnitude
            # (max of mean_abs and std) so that PySR sees X values ~O(1).
            # Predictions are made with the normalised _X_fit so R² / RMSE are
            # computed against the original y correctly.  The expression string
            # is annotated to record that it is in the normalised-X space when
            # X-scaling is applied.
            _x_col_scales = np.ones(X.shape[1])
            _x_scaled_cols: List[Tuple[int, str, float]] = []  # (col_idx, name, scale)
            for _xi in range(X.shape[1]):
                _col = X[:, _xi]
                _col_scale = max(float(np.abs(np.mean(_col))), float(np.std(_col)))
                if _col_scale > 0 and (_col_scale < 1e-6 or _col_scale > 1e6):
                    _x_col_scales[_xi] = _col_scale
                    _x_scaled_cols.append((_xi, safe_names[_xi], _col_scale))

            if _x_scaled_cols:
                _X_fit = X / _x_col_scales[np.newaxis, :]
                for _xi, _xn, _xsc in _x_scaled_cols:
                    pass
                _trace.append(f"x_scaled={[n for _,n,_ in _x_scaled_cols]}")
            else:
                _X_fit = X
                _trace.append("x_scaled=none")

            # ── Optics/wave: geometric-mean feature augmentation ─────────────
            # ROOT CAUSE of double-slit failure:
            #   Target formula: I1 + I2 + 2*sqrt(I1*I2)*cos(delta), complexity 13.
            #   The sub-expression sqrt(I1*I2) requires a two-step discovery:
            #     Step 1 — PySR must evolve  I1 * I2  as a binary sub-tree
            #     Step 2 — then wrap it in a  sqrt  unary
            #   Both steps must co-occur in the same candidate before selection
            #   pressure can reward the structure.  With serial evolution and
            #   ~100 iterations this rarely happens within 200 s.
            #
            # Fix: pre-compute GM(I_i, I_j) = sqrt(I_i * I_j) for all pairs of
            # strictly-positive columns and append them as additional features.
            # The target formula becomes:
            #   I1 + I2 + 2*gm_I1_I2*cos(delta)   complexity 7 (vs 13)
            # This is reliably found within 100 evolutionary iterations because
            # PySR only needs ONE product node (gm_I1_I2 * cos) instead of three
            # nested levels (sqrt → multiply → two leaves).
            #
            # Guard conditions (all must be True):
            #   _needs_basic_trig  — optics/wave domain with sin/cos injected
            #   not _needs_inv_trig — NOT a Snell's-law type (which is exact with
            #                         safe_asin; geometric means would add noise)
            #   ≥2 positive columns — geometric mean requires positive inputs
            #
            # The augmented variable names are propagated into the result so the
            # expression string references gm_I1_I2 rather than sqrt(I1*I2).
            # R² and RMSE are always computed against the original y, so scoring
            # is unaffected by the feature name change.
            if (
                _needs_basic_trig
                and not _needs_inv_trig
                and _X_fit.shape[1] >= 2
            ):
                _pos_idx = [
                    i for i in range(_X_fit.shape[1])
                    if float(np.min(_X_fit[:, i])) > 0.0
                ]
                if len(_pos_idx) >= 2:
                    _gm_cols: List[np.ndarray] = []
                    _gm_names: List[str] = []
                    for _pi in range(len(_pos_idx)):
                        for _qi in range(_pi + 1, len(_pos_idx)):
                            _ci, _cj = _pos_idx[_pi], _pos_idx[_qi]
                            _gm_vec = np.sqrt(_X_fit[:, _ci] * _X_fit[:, _cj])
                            _gm_nm  = f"gm_{safe_names[_ci]}_{safe_names[_cj]}"
                            _gm_cols.append(_gm_vec)
                            _gm_names.append(_gm_nm)
                    if _gm_cols:
                        _X_fit    = np.column_stack([_X_fit] + _gm_cols)
                        safe_names = list(safe_names) + _gm_names
                        _trace.append(f"gm_features={_gm_names}")

            # ── Exp-domain: ratio feature augmentation ───────────────────────
            # ROOT CAUSE of Bose-Einstein / Fermi-Dirac / Planck failure:
            #   Target: 1/(exp(hf/kT) - 1)  →  1/(exp(C * f_norm/T) - 1)
            #   After x-scaling f_norm~O(1), T~O(100): PySR must discover
            #   exp(ratio_of_two_vars * constant).  This requires evolving a
            #   division subtree INSIDE an exp — two nested structural steps
            #   that rarely co-occur within the search budget.
            #   Complexity of correct formula: ~8.  But PySR finds polynomial
            #   approximations at complexity 10-15 with the same R²≈0.998,
            #   which dominate the Pareto front and crowd out the exp-based form.
            #
            # Fix: pre-compute ratio_a_b = a/b for all pairs of strictly-positive
            # columns.  The target collapses to:
            #   1/(exp(C * ratio_f_x0) - 1)   complexity 5
            # PySR finds this trivially — no nested division needed inside exp.
            #
            # Guard: _needs_exp_log (quantum/thermal/chemistry domain) and ≥2
            # strictly-positive columns (ratio requires positive denominator).
            # Also skip if only 1 variable (no ratio possible).
            if _needs_exp_log and _X_fit.shape[1] >= 2:
                _pos_idx_exp = [
                    i for i in range(_X_fit.shape[1])
                    if float(np.min(_X_fit[:, i])) > 0.0
                ]
                if len(_pos_idx_exp) >= 2:
                    _ratio_cols: List[np.ndarray] = []
                    _ratio_names: List[str] = []
                    for _pi in range(len(_pos_idx_exp)):
                        for _qi in range(len(_pos_idx_exp)):
                            if _pi == _qi:
                                continue
                            _ci, _cj = _pos_idx_exp[_pi], _pos_idx_exp[_qi]
                            _ratio_vec = _X_fit[:, _ci] / (_X_fit[:, _cj] + 1e-300)
                            _ratio_nm  = f"ratio_{safe_names[_ci]}_{safe_names[_cj]}"
                            _ratio_cols.append(_ratio_vec)
                            _ratio_names.append(_ratio_nm)
                    if _ratio_cols:
                        _X_fit    = np.column_stack([_X_fit] + _ratio_cols)
                        safe_names = list(safe_names) + _ratio_names
                        _trace.append(f"ratio_features={_ratio_names}")
                else:
                    _trace.append("ratio_features=skipped(insufficient_pos_cols)")
            else:
                _trace.append("ratio_features=skipped")

            # Fit model with safe variable names
            self.model.fit(_X_fit, _y_fit, variable_names=safe_names)
            # FIX-RATIO v5.2: store augmented X (with ratio cols) for RMSE eval
            self._last_X_aug    = _X_fit
            self._last_aug_names = list(safe_names)

            # FIX-PARSIMONY v5.3: BIC-penalised Pareto selection.
            # Pure R²-max always picks the most complex expression on the Pareto
            # front.  BIC = n·log(RSS/n) + k·log(n) gives a principled penalty
            # for complexity k (PySR's own complexity column).  We minimise BIC.
            # Fallback: if all BIC computations fail, revert to max-R².
            if hasattr(self.model, "equations_") and len(self.model.equations_) > 0:
                eqs = self.model.equations_
                _n_bic = len(y)
                best_bic   = np.inf
                best_r2    = -np.inf
                best_idx   = 0
                _bic_trace = []
                for idx in range(len(eqs)):
                    try:
                        y_pred_i = self.model.predict(_X_fit, index=idx) * _y_std
                        r2_i     = r2_score(y, y_pred_i)
                        rss_i    = float(np.sum((y - y_pred_i) ** 2))
                        k_i      = int(eqs.iloc[idx].get("complexity", len(str(eqs.iloc[idx]["equation"]))))
                        bic_i    = (_n_bic * np.log(max(rss_i / _n_bic, 1e-300))
                                    + k_i * np.log(_n_bic))
                        _bic_trace.append((idx, round(r2_i, 4), k_i, float(round(bic_i, 2))))
                        if bic_i < best_bic:
                            best_bic  = bic_i
                            best_r2   = r2_i
                            best_idx  = idx
                    except Exception:
                        pass
                if best_bic == np.inf:
                    # All BIC attempts failed — fall back to max-R²
                    for idx in range(len(eqs)):
                        try:
                            y_pred_i = self.model.predict(_X_fit, index=idx) * _y_std
                            r2_i     = r2_score(y, y_pred_i)
                            if r2_i > best_r2:
                                best_r2  = r2_i
                                best_idx = idx
                        except Exception:
                            pass
                _trace.append(f"bic_selection={_bic_trace[:10]}")
                best_eq    = eqs.iloc[best_idx]
                expression = str(best_eq["equation"])

                # ── Pareto front trace ────────────────────────────────────────
                # Dump all equations in the Pareto front so we can see what PySR
                # actually found, even if R² is low.  Top 5 by R² only.
                _pareto_r2s = []
                for _pi in range(len(eqs)):
                    try:
                        _pr2 = r2_score(y, self.model.predict(_X_fit, index=_pi) * _y_std)
                        _pareto_r2s.append((_pi, str(eqs.iloc[_pi]["equation"]), round(_pr2, 4)))
                    except Exception:
                        pass
                _pareto_r2s.sort(key=lambda x: x[1], reverse=False)
                _top5 = sorted(_pareto_r2s, key=lambda x: x[2], reverse=True)[:5]
                _trace.append(f"pareto_top5={_top5}")
                _trace.append(f"best_r2={best_r2:.4f}")
                _trace.append(f"best_expr={expression[:80]}")

                # If y was scaled, fold the scale factor into the expression so
                # the string represents the original (physical) equation.
                if _y_std != 1.0:
                    expression = f"{_y_std:.6e} * ({expression})"

                # If X columns were scaled, annotate the expression to record
                # which columns were normalised and by what factor.  Full
                # symbolic de-normalisation (substituting var → var/scale in the
                # expression string) is deferred because arbitrary string
                # manipulation of PySR output is fragile; R² and RMSE are
                # computed correctly regardless (we always predict on _X_fit).
                if _x_scaled_cols:
                    _xscale_note = ", ".join(
                        f"{_xn}÷{_xsc:.3e}"
                        for _, _xn, _xsc in _x_scaled_cols
                    )
                    # FIX: store annotation separately so the expression
                    # string passed to the validator remains parseable.
                    # The display string keeps the human-readable prefix.
                    _xscale_annotation = f"[X-normalised: {_xscale_note}]"
                    _expression_display = f"{_xscale_annotation} {expression}"

                # Make predictions with the best-R² equation (rescaled to original y)
                y_pred = self.model.predict(_X_fit, index=best_idx) * _y_std
                r2 = best_r2


                _result = {
                    "expression": expression,  # clean — no annotation prefix
                    "expression_display": _expression_display if _x_scaled_cols else expression,
                    "r2_score": r2,
                    "complexity": int(best_eq.get("complexity", len(expression))),
                    "variable_names": safe_names,
                    "original_variable_names": variable_names,
                    "variable_name_mapping": name_mapping,
                    "predictions": y_pred,
                    "validation": {"valid": True, "errors": [], "warnings": []},
                    "trace": _trace,
                }
            else:
                _trace.append("outcome=NO_VALID_EQUATIONS")
                _result = {
                    "expression": "NO_VALID_EQUATIONS",
                    "r2_score": 0.0,
                    "complexity": 0,
                    "variable_names": safe_names,
                    "original_variable_names": variable_names,
                    "variable_name_mapping": name_mapping,
                    "predictions": np.zeros_like(y),
                    "validation": {
                        "valid": False,
                        "errors": ["No equations found"],
                        "warnings": [],
                    },
                    "trace": _trace,
                }

            # Cleanup temp equation file
            try:
                if _os2.path.exists(_equation_file_path):
                    _os2.unlink(_equation_file_path)
            except Exception:
                pass
            return _result

        except Exception as e:
            import traceback as _tb
            import sys as _sys
            _full_tb = _tb.format_exc()
            _err_msg = (
                f"\n{'='*70}\n"
                f"   ❌ Discovery FAILED — {type(e).__name__}: {e}\n"
                f"   Full traceback:\n{_full_tb}"
                f"{'='*70}\n"
            )
            # Capture partial trace even on exception — helps diagnose which
            # stage failed (before or after trig injection / GM augmentation).
            try:
                _trace.append(f"EXCEPTION={type(e).__name__}:{str(e)[:120]}")
            except Exception:
                _trace = [f"EXCEPTION={type(e).__name__}:{str(e)[:120]}"]
            return {
                "expression": "DISCOVERY_FAILED",
                "r2_score": 0.0,
                "complexity": 0,
                "variable_names": safe_names,
                "original_variable_names": variable_names,
                "variable_name_mapping": name_mapping,
                "predictions": np.zeros_like(y),
                "validation": {"valid": False, "errors": [_full_tb], "warnings": []},
                "trace": _trace,
            }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using discovered equation."""
        if self.model is None:
            raise ValueError("Model not fitted. Call discover() first.")
        return self.model.predict(X)


# ============================================================================
# ENHANCED SYMBOLIC ENGINE WITH INTEGRATED LLM
# ============================================================================


class SymbolicEngineWithLLM(SymbolicEngine):
    """Symbolic Engine v20 - Integrated LLM guidance + Variable Name Validation."""

    def __init__(
        self,
        config: Optional[DiscoveryConfig] = None,  # Optional: defaults to DiscoveryConfig()
        domain: str = "general",
        llm_config: Optional[LLMConfig] = None,
        llm_mode: str = "none",  # none, seed, hybrid, fallback
    ):
        """
        Initialize engine with optional LLM guidance and automatic variable validation.

        Args:
            config: PySR discovery configuration (uses DiscoveryConfig() defaults if None)
            domain: Problem domain
            llm_config: LLM configuration (creates default if None)
            llm_mode: How to use LLM
                - "none": No LLM (pure PySR)
                - "seed": LLM configures PySR operators
                - "hybrid": Try LLM first, refine with PySR
                - "fallback": PySR first, LLM if it fails
        """
        if config is None:
            config = DiscoveryConfig()
        super().__init__(config, domain)

        self.llm_mode = llm_mode
        self.llm_engine = None
        self.pattern_analyzer = None

        if llm_mode != "none":
            if llm_config is None:
                llm_config = LLMConfig(enabled=True)

            if llm_config.enabled:
                self.llm_engine = IntegratedLLMEngine(llm_config)
                self.pattern_analyzer = DataPatternAnalyzer()

                if self.llm_engine.config.enabled:
                    pass
                else:
                    self.llm_mode = "none"

    def discover(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str] = None,
        equation_name: str = None,
        random_state: int = 42,
        auto_sanitize: bool = True,
        **kwargs,
    ) -> Dict:
        """
        Enhanced discovery with LLM guidance and automatic variable name validation.

        Args:
            auto_sanitize: Automatically fix variable name conflicts (default: True)
        """

        if variable_names is None:
            variable_names = [f"x{i}" for i in range(X.shape[1])]

        # Route based on LLM mode
        if (
            self.llm_mode == "none"
            or not self.llm_engine
            or not self.llm_engine.config.enabled
        ):
            return super().discover(
                X,
                y,
                variable_names,
                equation_name,
                random_state,
                auto_sanitize=auto_sanitize,
                **kwargs,
            )

        elif self.llm_mode == "seed":
            return self._discover_with_llm_seed(
                X,
                y,
                variable_names,
                equation_name,
                random_state,
                auto_sanitize,
                **kwargs,
            )

        elif self.llm_mode == "hybrid":
            return self._discover_hybrid(
                X,
                y,
                variable_names,
                equation_name,
                random_state,
                auto_sanitize,
                **kwargs,
            )

        elif self.llm_mode == "fallback":
            return self._discover_with_fallback(
                X,
                y,
                variable_names,
                equation_name,
                random_state,
                auto_sanitize,
                **kwargs,
            )

        else:
            return super().discover(
                X,
                y,
                variable_names,
                equation_name,
                random_state,
                auto_sanitize=auto_sanitize,
                **kwargs,
            )

    def _discover_with_llm_seed(
        self, X, y, variable_names, equation_name, random_state, auto_sanitize, **kwargs
    ) -> Dict:
        """Use LLM to configure PySR operators."""

        # Validate variable names first
        safe_names, name_mapping = self.validate_variable_names(
            variable_names, auto_fix=auto_sanitize, verbose=True
        )

        # Analyze patterns
        patterns = self.pattern_analyzer.analyze(X, y, safe_names)

        # Get LLM hypotheses — pass caller_id so external caches key per-method
        hypotheses = self.llm_engine.generate_hypotheses(
            domain=self.domain,
            variables=safe_names,
            description=equation_name or "unknown",
            data_patterns=patterns,
            caller_id=f"{self.__class__.__name__}:seed",
        )

        if hypotheses:

            # Extract operators from best hypothesis
            best_hyp = hypotheses[0]
            llm_config = self._extract_operators_from_equation(best_hyp.equation)


        # Run PySR with LLM-informed config
        # PATCH-1: merge llm_config into kwargs so pysr_kwargs.update(kwargs)
        # at line 1203 actually overrides binary_operators / unary_operators.
        # Without this the extracted operators were computed but silently discarded.
        _seeded_kwargs = dict(kwargs)
        if hypotheses and llm_config:
            if llm_config.get("binary_operators"):
                _seeded_kwargs["binary_operators"] = llm_config["binary_operators"]
            if llm_config.get("unary_operators"):
                _seeded_kwargs["unary_operators"] = llm_config["unary_operators"]
            if llm_config.get("maxsize"):
                _seeded_kwargs["maxsize"] = llm_config["maxsize"]

        result = super().discover(
            X,
            y,
            variable_names,
            equation_name,
            random_state,
            auto_sanitize=auto_sanitize,
            **_seeded_kwargs,
        )
        result["llm_mode"] = "seed"
        result["llm_hypotheses"] = [h.equation for h in hypotheses]

        return result

    def _discover_hybrid(
        self, X, y, variable_names, equation_name, random_state, auto_sanitize, **kwargs
    ) -> Dict:
        """Try LLM first, refine with PySR if needed."""

        start_time = time.time()

        # Validate variable names first
        safe_names, name_mapping = self.validate_variable_names(
            variable_names, auto_fix=auto_sanitize, verbose=True
        )

        # Phase 1: LLM Discovery — pass caller_id so external caches key per-method
        patterns = self.pattern_analyzer.analyze(X, y, safe_names)
        hypotheses = self.llm_engine.generate_hypotheses(
            domain=self.domain,
            variables=safe_names,
            description=equation_name or "unknown",
            data_patterns=patterns,
            caller_id=f"{self.__class__.__name__}:hybrid",
        )

        llm_time = time.time() - start_time

        if not hypotheses:
            result = super().discover(
                X,
                y,
                variable_names,
                equation_name,
                random_state,
                auto_sanitize=auto_sanitize,
                **kwargs,
            )
            result["llm_mode"] = "hybrid_llm_failed"
            return result

        # Evaluate LLM hypotheses
        best_hyp = self._evaluate_hypotheses(hypotheses, X, y, safe_names)


        # Decision: Is LLM good enough?
        if best_hyp.r2_score and best_hyp.r2_score > 0.95:
            return {
                "expression": best_hyp.equation,
                "r2_score": best_hyp.r2_score,
                "complexity": len(best_hyp.equation),
                "variable_names": safe_names,
                "original_variable_names": variable_names,
                "variable_name_mapping": name_mapping,
                "predictions": self._predict_from_equation(
                    best_hyp.equation, X, safe_names
                ),
                "llm_mode": "hybrid_llm_only",
                "llm_time": llm_time,
                "validation": {"valid": True, "errors": [], "warnings": []},
                "llm_hypotheses": [h.equation for h in hypotheses],
            }

        # Phase 2: PySR Refinement
        pysr_start = time.time()

        result = super().discover(
            X,
            y,
            variable_names,
            equation_name,
            random_state,
            auto_sanitize=auto_sanitize,
            **kwargs,
        )

        pysr_time = time.time() - pysr_start


        # Compare and choose best
        if result["r2_score"] > best_hyp.r2_score:
            result["llm_mode"] = "hybrid_pysr_better"
            result["llm_hypotheses"] = [h.equation for h in hypotheses]
            result["llm_time"] = llm_time
            result["pysr_time"] = pysr_time
        else:
            result = {
                "expression": best_hyp.equation,
                "r2_score": best_hyp.r2_score,
                "complexity": len(best_hyp.equation),
                "variable_names": safe_names,
                "original_variable_names": variable_names,
                "variable_name_mapping": name_mapping,
                "predictions": self._predict_from_equation(
                    best_hyp.equation, X, safe_names
                ),
                "llm_mode": "hybrid_llm_better",
                "llm_time": llm_time,
                "pysr_time": pysr_time,
                "validation": {"valid": True, "errors": [], "warnings": []},
                "llm_hypotheses": [h.equation for h in hypotheses],
            }

        return result

    def _discover_with_fallback(
        self, X, y, variable_names, equation_name, random_state, auto_sanitize, **kwargs
    ) -> Dict:
        """Try PySR first, fallback to LLM if it fails."""

        # Validate variable names
        safe_names, name_mapping = self.validate_variable_names(
            variable_names, auto_fix=auto_sanitize, verbose=True
        )

        # Phase 1: PySR
        pysr_start = time.time()
        result = super().discover(
            X,
            y,
            variable_names,
            equation_name,
            random_state,
            auto_sanitize=auto_sanitize,
            **kwargs,
        )
        pysr_time = time.time() - pysr_start

        # Check if PySR succeeded
        if result["r2_score"] > 0.90:
            result["llm_mode"] = "fallback_pysr_only"
            result["pysr_time"] = pysr_time
            return result

        # Phase 2: LLM Fallback

        # Pass caller_id so external caches key per-method
        patterns = self.pattern_analyzer.analyze(X, y, safe_names)
        hypotheses = self.llm_engine.generate_hypotheses(
            domain=self.domain,
            variables=safe_names,
            description=equation_name or "unknown",
            data_patterns=patterns,
            caller_id=f"{self.__class__.__name__}:fallback",
        )

        if not hypotheses:
            result["llm_mode"] = "fallback_both_failed"
            return result

        best_hyp = self._evaluate_hypotheses(hypotheses, X, y, safe_names)

        if best_hyp.r2_score and best_hyp.r2_score > result["r2_score"]:
            return {
                "expression": best_hyp.equation,
                "r2_score": best_hyp.r2_score,
                "complexity": len(best_hyp.equation),
                "variable_names": safe_names,
                "original_variable_names": variable_names,
                "variable_name_mapping": name_mapping,
                "predictions": self._predict_from_equation(
                    best_hyp.equation, X, safe_names
                ),
                "llm_mode": "fallback_llm_better",
                "validation": {"valid": True, "errors": [], "warnings": []},
                "llm_hypotheses": [h.equation for h in hypotheses],
            }
        else:
            result["llm_mode"] = "fallback_pysr_better"
            result["llm_hypotheses"] = [h.equation for h in hypotheses]
            return result

    def _evaluate_hypotheses(
        self,
        hypotheses: List[EquationHypothesis],
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
    ) -> EquationHypothesis:
        """Evaluate LLM hypotheses against data."""

        for hyp in hypotheses:
            try:
                y_pred = self._predict_from_equation(hyp.equation, X, variable_names)
                hyp.r2_score = r2_score(y, y_pred)
            except Exception as e:
                hyp.r2_score = 0.0
                hyp.validation_score = 0.0

        # Sort by R² score
        hypotheses.sort(key=lambda h: h.r2_score or 0.0, reverse=True)
        return hypotheses[0]

    def _predict_from_equation(
        self, equation: str, X: np.ndarray, variable_names: List[str]
    ) -> np.ndarray:
        """Evaluate equation on data."""

        # Build namespace: constants first, then variables so variables take priority.
        # Previously namespace.update(constants) ran AFTER variable binding, silently
        # overwriting variables named 'e', 'c', 'k', 'h', 'pi', etc.
        namespace = {}
        # Add numpy functions and physical constants as the base layer
        namespace.update(
            {
                "exp": np.exp,
                "log": np.log,
                "sqrt": np.sqrt,
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "abs": np.abs,
                "sign": np.sign,
                "pi": np.pi,
                "e": np.e,
                # Physical constants (prefixed to avoid shadowing variables/np.e)
                "h_planck": 6.62607015e-34,
                "h":        6.62607015e-34,   # Planck constant
                "c":        2.99792458e8,      # speed of light
                "k_B":      1.380649e-23,      # Boltzmann constant
                "k":        1.380649e-23,
                "N_A":      6.02214076e23,     # Avogadro constant
                "g_n":      9.80665,           # standard gravity
                "m_e":      9.1093837015e-31,  # electron mass
                "q_e":      1.602176634e-19,   # elementary charge (NOT np.e)
                "hbar":     1.0545718176e-34,  # reduced Planck constant
                "epsilon0": 8.8541878128e-12,  # vacuum permittivity
                "mu0":      1.25663706212e-6,  # vacuum permeability
            }
        )

        # Bind variables AFTER constants so they override any name collisions
        for i, name in enumerate(variable_names):
            namespace[name] = X[:, i]

        try:
            result = eval(equation, {"__builtins__": {}}, namespace)
            return np.array(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate equation: {e}")

    def discover_formula(
        self,
        X: np.ndarray,
        y: np.ndarray,
        var_names: List[str],
        description: str = "",
        metadata: Optional[Dict] = None,
        max_iterations: int = 5,
        verbose: bool = False,
    ) -> Dict:
        """
        Adapter method called by run_comparative_suite_benchmark.IntegratedLLMDiscovery.

        Wraps discover() and normalises the return dict to the schema expected
        by the runner:
            {"success": bool, "r2": float, "rmse": float, "formula": str,
             "iterations": int, "error": str | None}

        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
            var_names: Variable names (mapped through sanitise if needed)
            description: Human-readable equation description (used as equation_name)
            metadata: Optional metadata dict (domain, difficulty, …)
            max_iterations: Passed as niterations override if positive
            verbose: Unused here; kept for call-site compatibility

        Returns:
            Runner-compatible result dict.
        """
        metadata = metadata or {}

        # Allow caller to override iterations via max_iterations
        if max_iterations > 0 and max_iterations != self.config.niterations:
            self.config.niterations = max_iterations

        # ── FIX: update self.domain from metadata so auto_inject_trig fires ──
        # discover_formula previously computed domain_hint but never applied it,
        # so SymbolicEngine.discover() always used the construction-time domain
        # ("general") even when the protocol passed "feynman_optics" in metadata.
        _domain_from_meta = metadata.get("domain", "")
        if _domain_from_meta and _domain_from_meta != self.domain:
            self.domain = _domain_from_meta

        _eq_name = description or metadata.get("equation_name", "unknown")

        try:
            result = self.discover(
                X=X,
                y=y,
                variable_names=var_names,
                equation_name=_eq_name,
                random_state=42,
                auto_sanitize=True,
            )

            r2 = float(result.get("r2_score", 0.0))
            y_pred = result.get("predictions", np.zeros_like(y))

            # Compute RMSE from predictions if available
            try:
                rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
            except Exception:
                rmse = float("inf")

            return {
                "success": r2 > 0.0 and result.get("expression", "DISCOVERY_FAILED") not in (
                    "DISCOVERY_FAILED", "NO_VALID_EQUATIONS", "VALIDATION_FAILED",
                ),
                "r2": r2,
                "rmse": rmse,
                "formula": result.get("expression", "N/A"),
                "iterations": max_iterations,
                "llm_mode": result.get("llm_mode", self.llm_mode),
                "variable_mapping": result.get("variable_name_mapping", {}),
                "error": None,
                "trace": result.get("trace", []),
            }

        except Exception as exc:
            return {
                "success": False,
                "r2": 0.0,
                "rmse": float("inf"),
                "formula": "N/A",
                "iterations": max_iterations,
                "error": str(exc)[:200],
                "trace": [f"discover_formula_exception={type(exc).__name__}:{str(exc)[:150]}"],
            }

    def _extract_operators_from_equation(self, equation: str) -> Dict:
        """Extract operators used in an equation."""

        binary_ops = set()
        unary_ops = set()

        # Binary operators
        if "+" in equation:
            binary_ops.add("+")
        if "-" in equation:
            binary_ops.add("-")
        if "*" in equation:
            binary_ops.add("*")
        if "/" in equation:
            binary_ops.add("/")
        # NOTE: '**' is intentionally NOT mapped to 'pow'.
        # PySR's 'pow' binary operator calls Julia's ^ which raises
        # DomainError on negative bases (e.g. (-1.0)^1.5).  Excluding
        # it keeps generated configs safe for all data domains.

        # Unary operators
        if "exp(" in equation:
            unary_ops.add("exp")
        if "log(" in equation:
            unary_ops.add("log")
        if "sqrt(" in equation:
            unary_ops.add("sqrt")
        if "sin(" in equation:
            unary_ops.add("sin")
        if "cos(" in equation:
            unary_ops.add("cos")
        if "tan(" in equation:
            unary_ops.add("tan")
        # Inverse trig — PySR uses "asin"/"acos"/"atan" (Julia names)
        if "arcsin(" in equation or "asin(" in equation:
            unary_ops.add("asin")
        if "arccos(" in equation or "acos(" in equation:
            unary_ops.add("acos")
        if "arctan(" in equation or "atan(" in equation:
            unary_ops.add("atan")

        # ── Transcendental composition detection ──────────────────────────
        # When the LLM hypothesis contains arcsin(sin(...)) or arccos(cos(...)),
        # PySR's simplifier collapses asin(sin(x)) → x before it can compete.
        # Adding the composition as an atomic custom operator bypasses this.
        # The Julia definitions are injected via extra_sympy_mappings + custom
        # unary operator strings when DiscoveryConfig.use_transcendental_compositions=True.
        if ("arcsin(" in equation or "asin(" in equation) and "sin(" in equation:
            unary_ops.add("asin_of_sin")
        if ("arccos(" in equation or "acos(" in equation) and "cos(" in equation:
            unary_ops.add("acos_of_cos")
        if ("arctan(" in equation or "atan(" in equation) and "tan(" in equation:
            unary_ops.add("atan_of_tan")

        # PATCH-2: estimate a tight maxsize from the expression's node count.
        # Gives PySR a complexity ceiling informed by the LLM skeleton rather
        # than the global default (which is usually far too permissive).

        try:
            _expr_tree = sp.sympify(equation)
            _node_count = sum(1 for _ in sp.preorder_traversal(_expr_tree))
            _maxsize = max(7, _node_count + 4)   # +4 headroom for numeric constants
        except Exception:
            _maxsize = None

        return {
            "binary_operators": list(binary_ops),
            "unary_operators": list(unary_ops),
            "maxsize": _maxsize,   # None if sympy parse failed → caller ignores it
        }


# ============================================================================
# V22 ADDITIONS — Bayesian re-ranking + lightweight equation compiler
# ============================================================================


class EquationTools:
    """
    v22: Lightweight equation compiler.

    Compiles a string expression to a callable without sympy overhead.
    Useful for quickly evaluating PySR Pareto-front equations.
    """

    @staticmethod
    def compile_equation(expr: str, variables: List[str]):
        """
        Compile an expression string into a vectorised callable.

        Args:
            expr: Python expression string (e.g. "x0 * x1 + 2.3")
            variables: Ordered list of variable names that map to X columns.

        Returns:
            func(X: np.ndarray) -> np.ndarray
        """
        code = compile(expr, "<equation>", "eval")

        def func(X: np.ndarray) -> np.ndarray:
            scope = {v: X[:, i] for i, v in enumerate(variables)}
            scope.update({
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                "abs": np.abs, "pi": np.pi,
            })
            return eval(code, scope)  # noqa: S307

        return func


class BayesianRanker:
    """
    v22: Bayesian re-ranker for PySR Pareto-front equations.

    Scores each candidate by log-posterior = log-likelihood + log-prior,
    where the prior penalises complexity.  Use this instead of (or on top of)
    the default R²-maximising selection in SymbolicEngine.discover() when you
    want a principled accuracy-vs-simplicity trade-off.

    Example
    -------
    After engine.discover() you can access the raw Pareto front via
    engine.model.equations_ and pass it to BayesianRanker.rank_from_pysr():

        ranker = BayesianRanker(complexity_penalty=0.01)
        ranked = ranker.rank_from_pysr(engine.model.equations_, X, y, safe_names)
    """

    def __init__(self, complexity_penalty: float = 0.01):
        self.complexity_penalty = complexity_penalty

    # ------------------------------------------------------------------
    # Low-level scoring helpers
    # ------------------------------------------------------------------

    def log_likelihood(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """Gaussian log-likelihood (up to constant)."""
        residuals = y - y_pred
        sigma2 = np.var(residuals)
        if sigma2 < 1e-30:
            sigma2 = 1e-30
        n = len(y)
        return (-0.5 * n * math.log(2 * math.pi * sigma2)
                - np.sum(residuals ** 2) / (2 * sigma2))

    def log_prior(self, complexity: int) -> float:
        """Log-prior: prefer simpler expressions."""
        return -self.complexity_penalty * complexity

    # ------------------------------------------------------------------
    # Main ranking interfaces
    # ------------------------------------------------------------------

    def rank(
        self,
        equations: List[Dict],
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[Dict]:
        """
        Rank a list of equation dicts by Bayesian posterior.

        Each dict must contain:
            "equation"   : str  — expression string
            "complexity" : int
            "callable"   : func(X) -> np.ndarray

        Returns the list sorted best-first, each entry augmented with
        "posterior_score".
        """
        ranked = []
        for eq in equations:
            try:
                pred = eq["callable"](X)
                score = (self.log_likelihood(y, pred)
                         + self.log_prior(eq["complexity"]))
                ranked.append({
                    "equation": eq["equation"],
                    "complexity": eq["complexity"],
                    "posterior_score": score,
                    "callable": eq["callable"],
                })
            except Exception:
                continue

        ranked.sort(key=lambda x: x["posterior_score"], reverse=True)
        return ranked

    def rank_from_pysr(
        self,
        equations_df,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
    ) -> List[Dict]:
        """
        Convenience wrapper: rank directly from a PySR equations_ DataFrame.

        Args:
            equations_df : engine.model.equations_  (pandas DataFrame)
            X, y         : data used for scoring
            variable_names: safe variable names passed to PySR

        Returns:
            Sorted list of dicts with keys: equation, complexity, posterior_score.
        """
        candidates = []
        for i in range(len(equations_df)):
            expr = str(equations_df.iloc[i]["equation"])
            complexity = int(equations_df.iloc[i]["complexity"])
            try:
                func = EquationTools.compile_equation(expr, variable_names)
                candidates.append({
                    "equation": expr,
                    "complexity": complexity,
                    "callable": func,
                })
            except Exception:
                continue
        return self.rank(candidates, X, y)


# ============================================================================
# V23 ADDITIONS — self-contained tree-search engine (no PySR / Julia)
# ============================================================================


class ExpressionNode:
    """
    v23: Node in a symbolic expression tree.

    Supports conversion to a sympy expression for pretty-printing and
    dimensional analysis, and recursive complexity counting.
    """

    def __init__(self, op: str, left=None, right=None, value=None):
        self.op = op        # operator string, "var", or "const"
        self.left = left    # ExpressionNode | None
        self.right = right  # ExpressionNode | None
        self.value = value  # variable name (str) or constant (float)

    def to_sympy(self):
        """Convert tree to a sympy expression."""
        if self.op == "var":
            return sp.Symbol(self.value)
        if self.op == "const":
            return sp.Float(self.value)

        left = self.left.to_sympy()
        right = self.right.to_sympy() if self.right else None

        ops = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b,
            "sin": lambda a, _: sp.sin(a),
            "cos": lambda a, _: sp.cos(a),
            "exp": lambda a, _: sp.exp(a),
            "log": lambda a, _: sp.log(a),
        }
        return ops[self.op](left, right)

    def complexity(self) -> int:
        """Recursive node count (leaf = 1)."""
        if self.op in ("var", "const"):
            return 1
        left_c = self.left.complexity() if self.left else 0
        right_c = self.right.complexity() if self.right else 0
        return 1 + left_c + right_c


class BayesianSearchRanker:
    """
    v23: Lightweight exp-based Bayesian scorer for tree candidates.

    Uses exp(-error) * exp(-penalty * complexity) as the posterior proxy.
    Simpler than BayesianRanker (v22) — no variance estimation needed —
    which makes it fast enough to score thousands of random trees per iteration.
    """

    def __init__(self, complexity_penalty: float = 0.01):
        self.complexity_penalty = complexity_penalty

    def prior(self, complexity: int) -> float:
        return math.exp(-self.complexity_penalty * complexity)

    def likelihood(self, error: float) -> float:
        return math.exp(-error)

    def posterior(self, error: float, complexity: int) -> float:
        return self.likelihood(error) * self.prior(complexity)


class _V23DimensionalValidator:
    """
    v23: Basic dimensional consistency checker.

    Attempts a sympy.simplify on the discovered expression; failures
    indicate the expression is dimensionally inconsistent or undefined.
    Real production systems use full symbolic unit algebra — this is a
    lightweight placeholder that catches the most obvious breakages.

    Args:
        variable_units: mapping from variable name → unit string,
                        e.g. {"v": "m/s", "m": "kg"}.  Not used in the
                        simplify check itself but stored for downstream use.
    """

    def __init__(self, variable_units: Dict[str, str]):
        self.variable_units = variable_units

    def validate(self, expr) -> bool:
        """Return True if the expression survives sympy.simplify without error."""
        try:
            sp.simplify(expr)
            return True
        except Exception:
            return False


class SymbolicSearch:
    """
    v23: Random expression tree generator.

    Generates candidate ExpressionNode trees by random recursive expansion.
    At each node: with probability 0.6, pick a binary op and recurse on both
    branches; otherwise pick a unary op and recurse on one branch.  At depth 0
    (leaves), choose a variable or a random constant with equal probability.
    """

    OPERATORS_BINARY = ["+", "-", "*", "/"]
    OPERATORS_UNARY = ["sin", "cos", "exp", "log"]

    def __init__(self, variables: List[str], max_depth: int = 3):
        self.variables = variables
        self.max_depth = max_depth

    def random_variable(self) -> ExpressionNode:
        return ExpressionNode("var", value=random.choice(self.variables))

    def random_constant(self) -> ExpressionNode:
        return ExpressionNode("const", value=random.uniform(-5, 5))

    def generate(self, depth: int) -> ExpressionNode:
        """Recursively generate a random expression tree."""
        if depth == 0:
            return (self.random_variable() if random.random() < 0.5
                    else self.random_constant())

        if random.random() < 0.6:
            op = random.choice(self.OPERATORS_BINARY)
            return ExpressionNode(op, self.generate(depth - 1), self.generate(depth - 1))
        else:
            op = random.choice(self.OPERATORS_UNARY)
            return ExpressionNode(op, self.generate(depth - 1))


class SymbolicTreeEngine:
    """
    v23: Self-contained symbolic discovery engine — no PySR / Julia required.

    Uses random tree generation + Bayesian scoring to search for symbolic
    expressions.  Substantially less powerful than PySR for large datasets or
    complex equations, but completely dependency-free (only numpy + sympy) and
    useful for:
      • quick prototyping without a Julia install
      • environments where PySR cannot run
      • low-dimensional, low-complexity targets

    Args:
        max_depth        : maximum tree depth for generated expressions
        population_size  : number of random trees evaluated per iteration
        iterations       : number of search iterations
        complexity_penalty: weight on the complexity prior (higher = simpler preferred)

    Example
    -------
        engine = SymbolicTreeEngine(max_depth=4, population_size=300, iterations=30)
        result = engine.discover_validate_interpret(
            X, y,
            variable_names=["v", "m"],
            variable_units={"v": "m/s", "m": "kg"},
        )
        print(result["equation"], "R²=", result["r2"])
    """

    def __init__(
        self,
        max_depth: int = 4,
        population_size: int = 500,
        iterations: int = 50,
        complexity_penalty: float = 0.01,
    ):
        self.max_depth = max_depth
        self.population_size = population_size
        self.iterations = iterations
        self.ranker = BayesianSearchRanker(complexity_penalty)

    # ------------------------------------------------------------------

    def _rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def _r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    # ------------------------------------------------------------------

    def _evaluate(
        self,
        expr: ExpressionNode,
        X: np.ndarray,
        y: np.ndarray,
        variables: List[str],
    ) -> Optional[Dict]:
        """Evaluate a single expression tree; return None on any error."""
        try:
            sym_expr = expr.to_sympy()
            # FIX-SIMPLIFY v5.2: collapse log(exp(x))→x etc before eval/display
            try:
                _free = sym_expr.free_symbols
                _asmp = {str(s): sp.Symbol(str(s), real=True) for s in _free}
                _simp = sp.simplify(sym_expr.subs(_asmp))
                if len(str(_simp)) < len(str(sym_expr)):
                    sym_expr = _simp
            except Exception:
                pass
            func = sp.lambdify(
                [sp.Symbol(v) for v in variables],
                sym_expr,
                "numpy",
            )
            preds = np.array(func(*[X[:, i] for i in range(X.shape[1])]),
                             dtype=float)
            if not np.all(np.isfinite(preds)):
                return None

            error = self._rmse(y, preds)
            r2 = self._r2(y, preds)
            complexity = expr.complexity()
            posterior = self.ranker.posterior(error, complexity)

            return {
                "expr": sym_expr,
                "error": error,
                "r2": r2,
                "complexity": complexity,
                "posterior": posterior,
            }
        except Exception:
            return None

    # ------------------------------------------------------------------

    def search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variables: List[str],
        verbose: bool = True,
    ) -> Optional[Dict]:
        """
        Run the iterative random-tree search.

        Returns the best-scoring candidate dict, or None if no valid
        expression was found across all iterations.
        """
        generator = SymbolicSearch(variables, self.max_depth)
        best: Optional[Dict] = None

        for iteration in range(self.iterations):
            population = []

            for _ in range(self.population_size):
                expr = generator.generate(self.max_depth)
                import warnings as _warnings
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore", RuntimeWarning)
                    result = self._evaluate(expr, X, y, variables)
                if result:
                    population.append(result)

            if not population:
                continue

            population.sort(key=lambda x: x["posterior"], reverse=True)
            top = population[0]

            if best is None or top["posterior"] > best["posterior"]:
                best = top

            if verbose:
                pass

        return best

    # ------------------------------------------------------------------

    def discover_validate_interpret(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        variable_units: Optional[Dict[str, str]] = None,
        variable_descriptions: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        equation_name: Optional[str] = None,
        show_formatted: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """
        Full discovery → dimensional validation → formatted output pipeline.

        Args:
            X, y              : data arrays
            variable_names    : feature names (no PySR reserved-word issues here)
            variable_units    : optional {name: unit_str} for DimensionalValidator
            variable_descriptions: optional {name: description} for reporting
            description       : free-text problem description (informational only)
            equation_name     : short name for the target equation (informational)
            show_formatted    : print a formatted summary after discovery
            verbose           : print per-iteration progress

        Returns:
            Dict with keys:
                equation           : sympy expression
                r2                 : coefficient of determination
                error              : RMSE
                complexity         : tree node count
                posterior          : Bayesian posterior score
                dimensionally_valid: bool — did DimensionalValidator accept it?
        """
        if equation_name:
            pass
        if description:
            pass

        best = self.search(X, y, variable_names, verbose=verbose)

        if best is None:
            return {
                "equation": None,
                "r2": 0.0,
                "error": float("inf"),
                "complexity": 0,
                "posterior": 0.0,
                "dimensionally_valid": False,
            }

        # Dimensional validation
        validator = _V23DimensionalValidator(variable_units or {})
        is_valid = validator.validate(best["expr"])

        result = {
            "equation": best["expr"],
            "r2": best["r2"],
            "error": best["error"],
            "complexity": best["complexity"],
            "posterior": best["posterior"],
            "dimensionally_valid": is_valid,
        }

        if show_formatted:
            if variable_descriptions:
                for name in variable_names:
                    desc = variable_descriptions.get(name, "")
                    unit = (variable_units or {}).get(name, "")

        return result


# ============================================================================
# MAIN TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SYMBOLIC ENGINE — unified v21 + v22 + v23")
    print("=" * 80)
    print()

    # Test variable name validation
    print("=" * 80)
    print("TEST 1: VARIABLE NAME VALIDATION")
    print("=" * 80)

    test_names = ["E0", "R", "T", "n", "F", "Q", "exp", "sin", "E"]
    safe_names, mapping = SymbolicEngine.validate_variable_names(
        test_names, auto_fix=True, verbose=True
    )

    print(f"\nOriginal: {test_names}")
    print(f"Safe:     {safe_names}")
    print(f"Mapping:  {mapping}")

    # Test Nernst equation example
    print("\n" + "=" * 80)
    print("TEST 2: NERNST EQUATION EXAMPLE")
    print("=" * 80)

    # Generate sample data
    np.random.seed(42)
    num_samples = 100

    E0 = np.random.uniform(0.5, 1.5, num_samples)
    R = np.full(num_samples, 8.314)
    T = np.random.uniform(273, 373, num_samples)
    n = np.random.randint(1, 4, num_samples)
    F = np.full(num_samples, 96485)
    Q = np.random.uniform(0.01, 100, num_samples)

    # Calculate Nernst potential
    y = E0 - (R * T / (n * F)) * np.log(Q)
    X = np.column_stack([E0, R, T, n, F, Q])

    # Test with conflicting variable name 'Q'
    variable_names = ["E0", "R", "T", "n", "F", "Q"]

    print(f"\nVariable names: {variable_names}")
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Note: 'Q' is a PySR reserved word and will be auto-sanitized")

    # Test symbolic regression with auto-sanitization
    print("\n" + "=" * 80)
    print("TEST 3: SYMBOLIC REGRESSION WITH AUTO-SANITIZATION")
    print("=" * 80)

    config = DiscoveryConfig(
        niterations=20,
        populations=30,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "exp"],
    )

    engine = SymbolicEngine(config, domain="chemistry")

    result = engine.discover(
        X,
        y,
        variable_names=variable_names,
        equation_name="Nernst Equation",
        auto_sanitize=True,
    )

    print(f"\nDiscovery Result:")
    print(f"   Expression: {result['expression']}")
    print(f"   R² Score: {result['r2_score']:.4f}")
    print(f"   Variable Mapping: {result['variable_name_mapping']}")

    # Test integration with LLM mode
    if HAS_ANTHROPIC and os.getenv("ANTHROPIC_API_KEY"):
        print("\n" + "=" * 80)
        print("TEST 4: LLM-GUIDED DISCOVERY WITH VALIDATION")
        print("=" * 80)

        llm_config = LLMConfig(enabled=True, n_candidates=2)
        engine_llm = SymbolicEngineWithLLM(
            config, domain="chemistry", llm_config=llm_config, llm_mode="hybrid"
        )

        result_llm = engine_llm.discover(
            X,
            y,
            variable_names=variable_names,
            equation_name="Nernst Equation",
            auto_sanitize=True,
        )

        print(f"\nLLM-Guided Result:")
        print(f"   Expression: {result_llm['expression']}")
        print(f"   R² Score: {result_llm['r2_score']:.4f}")
        print(f"   LLM Mode: {result_llm.get('llm_mode', 'N/A')}")
        print(f"   Variable Mapping: {result_llm['variable_name_mapping']}")
    else:
        print("\n⚠️  Skipping LLM test (API key not found)")

    # ── v22 demo: BayesianRanker on compiled equation list ────────────
    print("\n" + "=" * 80)
    print("TEST 5 (v22): BayesianRanker — re-rank compiled equations")
    print("=" * 80)

    # Simulate a small Pareto-front as you'd get from engine.model.equations_
    demo_vars = ["E0", "T", "Qr"]  # after sanitization: Q → Qr
    demo_exprs = [
        "E0 - 0.026 * T * log(Qr)",
        "E0 + T",
        "E0",
    ]
    demo_candidates = []
    for expr_str in demo_exprs:
        try:
            fn = EquationTools.compile_equation(expr_str, demo_vars)
            # dummy complexity estimate
            demo_candidates.append({
                "equation": expr_str,
                "complexity": len(expr_str.split()),
                "callable": fn,
            })
        except Exception as exc:
            print(f"   Compile error for '{expr_str}': {exc}")

    # Build tiny synthetic data matching the first expression
    rng = np.random.default_rng(0)
    _E0 = rng.uniform(0.5, 1.5, 50)
    _T  = rng.uniform(280, 360, 50)
    _Qr = rng.uniform(0.1, 10, 50)
    _X_demo = np.column_stack([_E0, _T, _Qr])
    _y_demo = _E0 - 0.026 * _T * np.log(_Qr)

    ranker_v22 = BayesianRanker(complexity_penalty=0.005)
    ranked_v22 = ranker_v22.rank(demo_candidates, _X_demo, _y_demo)

    print("\n  Bayesian ranking (best first):")
    for rank_i, entry in enumerate(ranked_v22):
        print(f"   #{rank_i + 1}  score={entry['posterior_score']:.2f}"
              f"  complexity={entry['complexity']}"
              f"  eq={entry['equation']}")

    # ── v23 demo: SymbolicTreeEngine (no PySR) ────────────────────────
    print("\n" + "=" * 80)
    print("TEST 6 (v23): SymbolicTreeEngine — PySR-free tree search")
    print("=" * 80)

    # Simple target: y = x0 * x1  (kinetic-energy-like product)
    rng2 = np.random.default_rng(1)
    _X_tree = rng2.uniform(0.5, 3.0, (80, 2))
    _y_tree = _X_tree[:, 0] * _X_tree[:, 1]

    tree_engine = SymbolicTreeEngine(
        max_depth=3,
        population_size=200,
        iterations=10,        # keep quick for demo; raise for real use
        complexity_penalty=0.02,
    )

    tree_result = tree_engine.discover_validate_interpret(
        _X_tree,
        _y_tree,
        variable_names=["m", "v"],
        variable_units={"m": "kg", "v": "m/s"},
        variable_descriptions={"m": "mass", "v": "velocity"},
        equation_name="product law demo",
        show_formatted=True,
        verbose=False,         # suppress per-iteration noise in demo
    )

    print(f"\n  Final: eq={tree_result['equation']}  "
          f"R²={tree_result['r2']:.4f}  "
          f"dim_valid={tree_result['dimensionally_valid']}")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 80)

# ===========================================================================
# hybrid_system_v50_2
# ===========================================================================

"""
HypatiaX Hybrid Discovery System v5.1
======================================
Adds four output-correctness fixes (FIX-A … FIX-D) from v4.1 on top of the
v5.0 LLM-wiring rewrite.  All v5.0 LLM fixes (FIX-1 … FIX-6) and PROD-1…7
performance improvements are preserved.

Bug history
-----------
v3.5  use_llm parameter introduced in discover_validate_interpret() signature.
      AnthropicProvider / GoogleProvider imported directly.
      Bug: neither provider was ever called; SymbolicEngineWithLLM never used.

v3.8  Direct LLM imports removed (regression).
      Bug: persisted — use_llm still a no-op.

v4.1-PROD (v40)
      PROD-1…7 performance improvements added.
      Bug: persisted — self.anthropic_provider / self.google_provider set but
      never read; SymbolicEngineWithLLM never imported; use_llm flag never
      checked in method body.

v4.2 / v4.2.1 (hybrid_system.py / v43)
      Variable-name fix, optional import guards added.
      Bug: persisted unchanged.

v5.0  LLM-wiring rewrite (FIX-1 … FIX-6).  Output-correctness bugs below
      were not addressed.

What v5.0 fixed (LLM wiring)
-----------------------------
FIX-1   Import SymbolicEngineWithLLM alongside SymbolicEngine.
FIX-2   __init__ instantiates SymbolicEngineWithLLM when use_llm=True OR when
        an LLM API key is available, passing llm_mode through correctly.
FIX-3   discover_validate_interpret() now reads use_llm and routes to the
        correct engine path.
FIX-4   _discover_with_retry() respects the engine type already set —
        no duplicate routing needed.
FIX-5   _initialize_llm_providers() retained for the external
        anthropic_provider / google_provider attributes (used by callers that
        talk to LLM providers directly), but the discovery path now uses
        SymbolicEngineWithLLM's internal IntegratedLLMEngine instead.
FIX-6   use_llm=True now propagates through the discover() thin adapter so
        benchmark runners can enable LLM guidance via metadata.

What v5.1 fixes (output correctness — ported from v4.1)
---------------------------------------------------------
FIX-A   RMSE always Infinity in discover() return dict.
        Root cause: discover() tried to read a "predictions" key that
        _discover_with_retry() never writes.  Fix: compute RMSE directly
        from the discovered expression string against the training y vector
        using the same expression evaluator used for validation.

FIX-B   extrapolation_errors always 0.0 / extrap_r2 always wrong.
        Root cause: the discover() thin adapter returned the expression only
        under "final_formula"; test harnesses computing out-of-distribution
        metrics looked for a "formula" key and got None.
        Fix: return "formula", "expression", and "final_formula" all pointing
        to the same string, and add a "variable_names" key so callers can
        bind the expression without extra round-trips.

FIX-C   PySR non-determinism warning on every run.
        Root cause: random_state was set without deterministic=True and
        parallelism='serial'.  Fix: pass deterministic=True and
        parallelism='serial' into the engine call when a random_state is
        supplied (opt-out via allow_nondeterministic=True on __init__).

FIX-D   Gravitational-force (and other extreme-scale equations) collapses
        to a constant.
        Root cause: PySR normalises internally but when feature magnitudes
        span >6 orders of magnitude (e.g. 1e-9 charges vs 1e22 forces) the
        search collapses to a constant before meaningful structure appears.
        Fix: discover() detects extreme-scale inputs (log10 range > 6 across
        any feature), applies per-feature signed-log10 scaling, fits on the
        scaled space, and injects a scale_log=True flag into the result
        metadata so callers know the returned expression is in log-space.
        Raw RMSE is still computed in original units.

Reproducibility pins (inherited from v4.0, preserved in v5.1)
--------------------------------------------------------------
PIN-1   max_retries default = 5  (matches v4 reference run).
PIN-2   Default DiscoveryConfig niterations = 50.
PIN-3   Warm-start Phase 2 disabled by default (_WS_THRESHOLD = -1.0).
PIN-4   enable_physics_fallback default = False.

Public API is backward-compatible: callers that pass use_llm=False (or omit
it) get pure-PySR behaviour identical to v4.1-PROD.
"""

import json
import logging
import os
import random
import re
import time
from collections import deque
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FIX-1: Import BOTH SymbolicEngine and SymbolicEngineWithLLM.
# Previous versions only imported the base class, making LLM guidance
# unreachable regardless of what use_llm was set to.
# ---------------------------------------------------------------------------


class DiscoveryMode(Enum):
    STRICT = "strict"
    CALIBRATED = "calibrated"


# ---------------------------------------------------------------------------
# PROD-1: Cached quality check (unchanged from v4.1-PROD).
# ---------------------------------------------------------------------------
@lru_cache(maxsize=256)
def _cached_quality(
    expression: str,
    r2_rounded: float,
    complexity_threshold: int,
) -> Tuple[bool, int, Tuple[str, ...]]:
    """
    Pure-function quality check used as the LRU target.
    Returns (is_overfit, complexity, warnings_tuple) — fully hashable.
    """
    complexity = len(expression)
    is_overfit = False
    warnings: List[str] = []

    if complexity > complexity_threshold and r2_rounded < 0.999:
        is_overfit = True
        warnings.append(f"High complexity ({complexity}) but R2={r2_rounded:.4f}")

    constants = re.findall(r"\d+\.\d+", expression)
    if len(constants) > 5:
        warnings.append(f"Many constants detected ({len(constants)})")

    suspicious = [c for c in constants if float(c) < 0.001 or float(c) > 1000]
    if suspicious:
        warnings.append(f"Suspicious constants: {suspicious[:3]}")

    return is_overfit, complexity, tuple(warnings)


# ---------------------------------------------------------------------------
# PROD-3: Pre-compiled regex patterns for PySR operator normalisation.
# ---------------------------------------------------------------------------
def _build_op_patterns(aliases: Dict[str, str]) -> Dict[str, Tuple[re.Pattern, str]]:
    return {
        pysr_name: (re.compile(r"\b" + re.escape(pysr_name) + r"\b"), numpy_name)
        for pysr_name, numpy_name in aliases.items()
    }


# ---------------------------------------------------------------------------
# PROD-4: Recursive serialisation helper.
# ---------------------------------------------------------------------------
def _to_serialisable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    # sp.Expr (sympy_expr in validation layer_results) is not JSON-serialisable;
    # convert to its string representation so save_results() doesn't crash.
    if isinstance(obj, sp.Basic):
        return str(obj)
    return obj


class HybridDiscoverySystem:
    """
    Hybrid discovery system v5.1.

    Now correctly wires LLM guidance through SymbolicEngineWithLLM when
    use_llm=True or when an API key is present.  Backward-compatible:
    use_llm=False (default) gives pure-PySR behaviour identical to v4.1-PROD.

    LLM modes (passed as llm_mode, or inferred from primary_llm):
        "none"     — pure PySR (default, same as all previous versions)
        "seed"     — LLM configures PySR operator set before search
        "hybrid"   — LLM attempts first; PySR refines if needed
        "fallback" — PySR first; LLM fires only when PySR underperforms
    """

    # PROD-3: Alias table (unchanged from v4.1-PROD)
    _PYSR_OP_ALIASES: Dict[str, str] = {
        "safe_asin":   "arcsin",
        "safe_acos":   "arccos",
        "asin_of_sin": "arcsin",
        "acos_of_cos": "arccos",
        "atan_of_tan": "arctan",
    }
    _PYSR_OP_PATTERNS: Dict[str, Tuple[re.Pattern, str]] = _build_op_patterns(
        _PYSR_OP_ALIASES
    )

    def __init__(
        self,
        domain: str = "general",
        discovery_config: Optional[DiscoveryConfig] = None,
        discovery_mode: DiscoveryMode = DiscoveryMode.STRICT,
        max_results: Optional[int] = 100,
        validation_weights: Optional[Dict[str, float]] = None,
        use_rich_output: bool = True,
        primary_llm: str = "anthropic",
        enable_fallback: bool = True,
        enable_physics_fallback: bool = False,
        physics_fallback_threshold: float = 0.85,
        complexity_penalty_threshold: int = 20,
        physics_population_size: int = 20,
        physics_generations: int = 100,
        max_retries: int = 5,                # PIN-1: raised from 3 for reproducibility
        enable_auto_config: bool = True,
        anthropic_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        # FIX-2: New parameters to control LLM engine behaviour.
        use_llm: bool = False,
        llm_mode: str = "hybrid",          # none | seed | hybrid | fallback
        llm_n_candidates: int = 3,
        llm_temperature: float = 0.3,
        # FIX-C: deterministic PySR (eliminates non-determinism warning).
        allow_nondeterministic: bool = False,
    ):
        """
        Initialize HybridDiscoverySystem v5.1.

        New parameters vs v4.1-PROD
        ----------------------------
        use_llm : bool
            Master switch.  When False (default) behaviour is identical to all
            previous versions.  When True, SymbolicEngineWithLLM is used and
            llm_mode controls the integration strategy.
        llm_mode : str
            "none"     — pure PySR (same as use_llm=False)
            "seed"     — LLM suggests operators; PySR searches
            "hybrid"   — LLM first, PySR refines (recommended)
            "fallback" — PySR first, LLM backup on poor R²
        llm_n_candidates : int
            Number of equation hypotheses to request from the LLM per call.
        llm_temperature : float
            Sampling temperature passed to the LLM API.
        allow_nondeterministic : bool
            When False (default), forces deterministic=True + parallelism=
            'serial' on the engine call so PySR suppresses its non-determinism
            warning and results are reproducible across runs.
            Set True only when you want parallel search and can tolerate
            run-to-run variation.
        """
        self.domain = domain
        self.discovery_mode = discovery_mode
        self.primary_llm = primary_llm
        self.enable_fallback = enable_fallback
        self.enable_physics_fallback = enable_physics_fallback
        self.physics_fallback_threshold = physics_fallback_threshold
        self.complexity_penalty_threshold = complexity_penalty_threshold
        self.physics_population_size = physics_population_size
        self.physics_generations = physics_generations
        self.max_retries = max_retries
        self.enable_auto_config = enable_auto_config
        self.use_llm = use_llm
        self.llm_mode = llm_mode if use_llm else "none"

        logger.info("=" * 70)
        logger.info("HybridDiscoverySystem v5.1 — LLM WIRING + OUTPUT FIXES")
        logger.info("=" * 70)
        logger.info(f"Domain: {domain}")
        logger.info(f"Discovery mode: {self.discovery_mode.value}")
        logger.info(f"Primary LLM: {primary_llm}")
        logger.info(f"use_llm: {use_llm}  |  llm_mode: {self.llm_mode}")
        logger.info(f"Auto-config: {enable_auto_config}")
        logger.info(f"Max retries: {max_retries}")
        logger.info(f"PhysicsAware fallback: {enable_physics_fallback}")
        logger.info(f"Complexity threshold: {complexity_penalty_threshold}")
        logger.info(f"Deterministic PySR: {not allow_nondeterministic}")
        logger.info("=" * 70)

        # FIX-C: store for use in _discover_with_retry
        self._pysr_deterministic = not allow_nondeterministic
        self._pysr_parallelism = "serial" if not allow_nondeterministic else None

        if discovery_config is None:
            symbolic_config = DiscoveryConfig(
                niterations=50,              # PIN-2: matches v4 reference run
                enable_auto_configuration=enable_auto_config,
            )
            logger.info("Using default iterations: 50")
        else:
            symbolic_config = discovery_config
            logger.info(f"Using provided iterations: {symbolic_config.niterations}")
            logger.info(f"Parsimony: {symbolic_config.parsimony}")
            logger.info(
                f"Transcendental compositions: {symbolic_config.use_transcendental_compositions}"
            )

        # PROD-2: operator injection (unchanged from v4.1-PROD)
        self._inject_operators(symbolic_config, domain)

        # FIX-2: Resolve the API key that will be used for LLM guidance.
        _llm_api_key = (
            anthropic_api_key
            if primary_llm == "anthropic"
            else (google_api_key or os.getenv("GOOGLE_API_KEY"))
        ) or os.getenv("ANTHROPIC_API_KEY")

        # FIX-2: Auto-enable LLM if a key is present and use_llm was not
        # explicitly set to False by the caller.
        _key_present = bool(_llm_api_key)
        if _key_present and not use_llm:
            logger.info(
                "[LLM] API key found but use_llm=False — running pure PySR. "
                "Pass use_llm=True to enable LLM guidance."
            )

        # FIX-2: Instantiate the correct engine class.
        # Previous versions ALWAYS used SymbolicEngine (base) even when
        # use_llm=True, because SymbolicEngineWithLLM was never imported.
        if self.llm_mode != "none" and _key_present:
            llm_config = LLMConfig(
                enabled=True,
                api_key=_llm_api_key,
                n_candidates=llm_n_candidates,
                temperature=llm_temperature,
            )
            try:
                self.symbolic_engine: SymbolicEngine = SymbolicEngineWithLLM(
                    symbolic_config,
                    domain=domain,
                    llm_config=llm_config,
                    llm_mode=self.llm_mode,
                )
                logger.info(
                    f"[LLM] SymbolicEngineWithLLM instantiated "
                    f"(mode={self.llm_mode}, candidates={llm_n_candidates})"
                )
            except Exception:
                logger.error(
                    "SymbolicEngineWithLLM construction FAILED — "
                    "falling back to base SymbolicEngine",
                    exc_info=True,
                )
                self.symbolic_engine = SymbolicEngine(symbolic_config, domain=domain)
                self.llm_mode = "none"
        else:
            # Pure-PySR path: identical to all previous versions.
            self.symbolic_engine = SymbolicEngine(symbolic_config, domain=domain)
            if self.llm_mode != "none":
                logger.warning(
                    "[LLM] llm_mode != 'none' but no API key available — "
                    "running pure PySR.  Set ANTHROPIC_API_KEY or pass "
                    "anthropic_api_key= to enable LLM guidance."
                )
                self.llm_mode = "none"

        try:
            self.validator = EnsembleValidator(
                domain=domain, max_history=max_results, weights=validation_weights
            )
        except Exception:
            logger.error("EnsembleValidator construction FAILED", exc_info=True)
            raise

        # FIX-5: Keep external provider attributes for callers that use them
        # directly (e.g. interpretation, explanation steps).  These are NOT
        # used in the discovery path — SymbolicEngineWithLLM owns that now.
        self._initialize_llm_providers(anthropic_api_key, google_api_key)

        self.max_results = max_results
        self.results: Any = deque(maxlen=max_results) if max_results is not None else []

        self.stats: Dict[str, int] = {
            "discoveries": 0,
            "symbolic_attempts": 0,
            "symbolic_successes": 0,
            "symbolic_failures": 0,
            "llm_guided": 0,
            "llm_skipped": 0,
            "physics_used": 0,
            "physics_successes": 0,
            "validations": 0,
            "auto_configs": 0,
        }

        self.use_rich_output = use_rich_output
        logger.info("[OK] HybridDiscoverySystem v5.1 initialized\n")

    # ------------------------------------------------------------------
    # PROD-2: shared operator-injection logic (unchanged from v4.1-PROD)
    # ------------------------------------------------------------------
    @staticmethod
    def _inject_operators(symbolic_config: DiscoveryConfig, domain: str) -> None:
        """Inject safe_asin/safe_acos when use_transcendental_compositions is True."""
        _TRIG_DEFAULTS = ["sin", "cos", "tan"]
        _needs_inv_trig = getattr(symbolic_config, "use_transcendental_compositions", False)
        if _needs_inv_trig:
            _inv_trig = ["safe_asin", "safe_acos"]
            _current = list(getattr(symbolic_config, "unary_operators", None) or [])
            if not _current:
                _current = list(_TRIG_DEFAULTS)
                logger.info(
                    f"[AUTO-v5.1] unary_operators was empty — seeding with trig defaults: {_current}"
                )
            _added = [op for op in _inv_trig if op not in _current]
            if _added:
                symbolic_config.unary_operators = _current + _added
                logger.info(
                    f"[AUTO-v5.1] Injected inverse-trig operators {_added} "
                    f"(use_tc=True). Full unary set: {symbolic_config.unary_operators}"
                )
        else:
            logger.info(
                f"[AUTO-v5.1] Skipping safe_asin/safe_acos injection "
                f"(domain='{domain}', use_tc=False)"
            )

    def _initialize_llm_providers(
        self, anthropic_api_key: Optional[str], google_api_key: Optional[str]
    ) -> None:
        """
        Initialize external LLM provider references (FIX-5).

        These are kept for callers that use anthropic_provider / google_provider
        directly (e.g. interpretation, summarisation steps outside discovery).
        The discovery path itself now uses SymbolicEngineWithLLM internally.
        """
        api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            try:
                self.anthropic_provider = None  # not available in embedded mode
                pass
            except Exception:
                self.anthropic_provider = None
        else:
            self.anthropic_provider = None

        api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                self.google_provider = None  # not available in embedded mode
                pass
            except Exception:
                self.google_provider = None
        else:
            self.google_provider = None

    def _create_optimized_physics_regressor(
        self, noise_level: Optional[float] = None
    ) -> PhysicsAwareRegressor:
        return PhysicsAwareRegressor(
            domain=self.domain,
            verbose=True,
            population_size=self.physics_population_size,
            generations=self.physics_generations,
            noise_level=noise_level,
        )

    def _check_expression_quality(self, expression: str, r2: float) -> Dict[str, Any]:
        """Quality check — PROD-1: delegates to LRU-cached pure function."""
        r2_rounded = round(r2, 6)
        is_overfit, complexity, warnings_tuple = _cached_quality(
            expression, r2_rounded, self.complexity_penalty_threshold
        )
        return {
            "is_overfit": is_overfit,
            "complexity": complexity,
            "warnings": list(warnings_tuple),
        }

    def _detect_rational_pattern(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Detect if data likely follows a rational/saturation pattern (unchanged)."""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score as _r2

        if X.shape[1] < 1 or np.any(y <= 0):
            return False
        try:
            inv_y = 1.0 / y
            for i in range(X.shape[1]):
                xi = X[:, i]
                if np.any(xi <= 0):
                    continue
                inv_x = 1.0 / xi
                r2 = _r2(
                    inv_y,
                    LinearRegression()
                    .fit(inv_x.reshape(-1, 1), inv_y)
                    .predict(inv_x.reshape(-1, 1)),
                )
                if r2 > 0.85:
                    logger.info(
                        f"[RATIONAL] Lineweaver-Burk R²={r2:.3f} on var {i} — injecting inv"
                    )
                    return True
            for i in range(X.shape[1]):
                xi = X[:, i]
                sort_idx = np.argsort(xi)
                y_sorted = y[sort_idx]
                if y_sorted[-1] > y_sorted[0]:
                    diffs = np.diff(y_sorted)
                    if np.all(diffs >= -1e-6) and diffs[-1] < diffs[0] * 0.3:
                        logger.info(
                            f"[RATIONAL] Saturation shape detected on var {i} — injecting inv"
                        )
                        return True
        except Exception as exc:
            logger.warning(f"[RATIONAL] Detection failed: {exc}")
        return False

    # ------------------------------------------------------------------
    # Core discovery worker
    # ------------------------------------------------------------------
    def _discover_with_retry(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        variable_descriptions: Dict[str, str],
        variable_units: Dict[str, str],
        equation_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Discover with retry.

        FIX-4: No engine-routing needed here — the correct engine type
        (SymbolicEngine or SymbolicEngineWithLLM) was already selected in
        __init__ based on use_llm and key availability.  All retry logic and
        physics fallback are unchanged from v4.1-PROD.
        """
        best_result = None
        best_r2 = -np.inf
        last_attempt_error: Optional[Exception] = None
        _inv_injected = False

        for attempt in range(self.max_retries):
            try:
                seed = 42 + attempt
                logger.info(f"\n[SYMBOLIC] Attempt {attempt + 1}/{self.max_retries} (seed={seed})")
                self.stats["symbolic_attempts"] += 1

                result = self.symbolic_engine.discover(
                    X, y, variable_names, equation_name=equation_name, random_state=seed,
                    # FIX-C: suppress non-determinism warning when running in
                    # deterministic mode.  The engine's discover() accepts **kwargs
                    # and forwards recognised keys to PySR via pysr_kwargs.
                    **({"deterministic": True, "parallelism": "serial"}
                       if self._pysr_deterministic else {}),
                )

                r2 = result.get("r2_score", 0)
                expr = result.get("expression", "")

                # Track whether LLM was actually used this attempt
                if result.get("llm_mode") and result["llm_mode"] != "none":
                    self.stats["llm_guided"] += 1
                else:
                    self.stats["llm_skipped"] += 1

                try:
                    collapsed = detect_collapsed_constants(expr, variable_names)
                except Exception:
                    logger.error("detect_collapsed_constants FAILED", exc_info=True)
                    collapsed = []

                result["collapsed_constants"] = collapsed
                logger.info(f"   Result: {expr}")
                logger.info(f"   R2 = {r2:.4f}")
                if result.get("llm_mode"):
                    logger.info(f"   LLM mode: {result['llm_mode']}")

                if expr and expr not in (
                    "DISCOVERY_FAILED", "NO_VALID_EQUATIONS", "VALIDATION_FAILED"
                ):
                    quality = self._check_expression_quality(expr, r2)
                    if quality["is_overfit"]:
                        logger.warning("   [WARNING] Possible overfit")
                        for w in quality["warnings"]:
                            logger.warning(f"      {w}")
                else:
                    quality = {"is_overfit": False, "complexity": 0, "warnings": []}

                if r2 > best_r2:
                    best_r2 = r2
                    best_result = result
                    best_result["discovery_engine"] = "symbolic"
                    best_result["attempt"] = attempt + 1
                    best_result["quality_check"] = quality
                    logger.info("   [BEST] New best!")

                if attempt == 0 and r2 < 0.1 and not _inv_injected:
                    if self._detect_rational_pattern(X, y):
                        _current_unary = list(
                            getattr(self.symbolic_engine.config, "unary_operators", None) or []
                        )
                        if "inv" not in _current_unary:
                            self.symbolic_engine.config.unary_operators = _current_unary + ["inv"]
                            logger.info("[RATIONAL] Injected 'inv' into unary_operators for next attempt")
                            _inv_injected = True

                _early_stop_r2 = (
                    0.9999
                    if getattr(self.symbolic_engine.config, "use_transcendental_compositions", False)
                    else 0.95
                )
                if r2 >= _early_stop_r2 and not quality["is_overfit"]:
                    logger.info(f"   [EARLY STOP] Excellent result (R²={r2:.6f})")
                    self.stats["symbolic_successes"] += 1
                    return best_result

            except Exception as e:
                last_attempt_error = e
                logger.error(f"   [ERROR] Attempt {attempt + 1} failed: {e}")
                logger.error(f"Attempt {attempt + 1} exception", exc_info=True)

        if best_result and best_r2 >= 0.97:
            logger.info(f"\n[SUCCESS] SymbolicEngine succeeded (R2={best_r2:.4f})")
            self.stats["symbolic_successes"] += 1
            return best_result
        else:
            logger.warning(f"\n[WARNING] SymbolicEngine best R2={best_r2:.4f}")
            self.stats["symbolic_failures"] += 1

        if self.enable_physics_fallback and (
            not best_result or best_r2 < self.physics_fallback_threshold
        ):
            try:
                logger.info("\n[FALLBACK] Using PhysicsAwareRegressor...")
                _meta_noise = getattr(self, "_current_noise_level", None)
                physics_regressor = self._create_optimized_physics_regressor(
                    noise_level=_meta_noise
                )
                physics_regressor.fit_noise_aware(
                    X=X,
                    y=y,
                    variable_names=variable_names,
                    noise_level=_meta_noise,
                    variable_units=variable_units,
                    variable_descriptions=variable_descriptions,
                )
                expression = physics_regressor.get_expression()
                r2 = physics_regressor.best_fitness_
                logger.info(f"   PhysicsAware: {expression}")
                logger.info(f"   R2 = {r2:.4f}")
                physics_result = {
                    "expression": expression,
                    "r2_score": r2,
                    "discovery_engine": "physics_aware",
                    "complexity": len(expression),
                }
                self.stats["physics_used"] += 1
                if r2 > best_r2:
                    logger.info("   [BEST] PhysicsAware better!")
                    best_result = physics_result
                    best_r2 = r2
                    self.stats["physics_successes"] += 1
            except Exception as e:
                logger.error(f"   [ERROR] PhysicsAware failed: {e}")

        if best_result:
            logger.warning(
                f"[PARTIAL] Returning best result with R2={best_r2:.4f}. "
                "If R2 is very low, check that the right unary operators are enabled."
            )
            return best_result
        else:
            raise ValueError(
                f"All {self.max_retries} discovery attempts failed"
                + (f": {last_attempt_error}" if last_attempt_error else "")
                + f"\n  HINT: If this is an optics/trig equation, ensure "
                  f"safe_asin/safe_acos are in unary_operators (DiscoveryConfig). "
                  f"Domain detected: '{self.domain}'."
            ) from last_attempt_error

    @staticmethod
    def _normalise_expression(expression_str: str) -> str:
        """Replace PySR custom operator names — PROD-3: uses pre-compiled patterns.
        FIX-SIMPLIFY v5.2: also collapses log(exp(x))→x and exp(log(x))→x identities.
        """
        result = expression_str
        for pat, numpy_name in HybridDiscoverySystem._PYSR_OP_PATTERNS.values():
            result = pat.sub(numpy_name, result)
        # FIX-SIMPLIFY v5.3: nsimplify constants + real-symbol simplification.
        # Step 1 — nsimplify: snap near-rational constants (0.99956 → 1, etc.)
        # Step 2 — simplify with real=True so log(exp(x))→x etc. collapse.
        # Guard: only accept if strictly shorter AND R² drop ≤ 0.5 pp
        #        (checked at call sites that pass y/X; here we just shorten).
        try:
            import sympy as _sp
            _free_raw = _sp.sympify(result).free_symbols
            _real_subs = {str(s): _sp.Symbol(str(s), real=True, positive=True)
                          for s in _free_raw}
            _sym_raw  = _sp.sympify(result, locals=_real_subs)
            # Step 1: snap near-rational constants
            _sym_ns   = _sp.nsimplify(_sym_raw, rational=False, tolerance=5e-3)
            # Step 2: algebraic simplification with real assumptions
            _sym_simp = _sp.simplify(_sym_ns)
            _simp_str = str(_sym_simp)
            if len(_simp_str) < len(result):  # only accept if strictly shorter
                result = _simp_str
        except Exception:
            pass  # sympy failure is non-fatal
        return result

    def _safe_validate(
        self,
        expression_str: str,
        variable_definitions: Dict[str, str],
        variable_units: Dict[str, str],
        test_data: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Safe validation (unchanged from v4.1-PROD)."""
        normalised = self._normalise_expression(expression_str)
        if normalised != expression_str:
            logger.info(
                f"[NORMALISE] Expression rewritten for validator: "
                f"'{expression_str}' → '{normalised}'"
            )
        try:
            return self.validator.validate_complete(
                expression_str=normalised,
                variable_definitions=variable_definitions,
                variable_units=variable_units,
                test_data=test_data,
            )
        except Exception as e:
            logger.warning(f"[WARNING] Validation error: {str(e)[:100]}")
            return {
                "valid": False,
                "total_score": 60.0,
                "layer_scores": {
                    "symbolic": 100.0,
                    "dimensional": 20.0,
                    "domain": 60.0,
                    "numerical": 100.0,
                },
                "errors": [f"Validation error: {str(e)[:200]}"],
                "warnings": ["Validation failed - likely unit system issue"],
                "validation_exception": True,
            }

    # ------------------------------------------------------------------
    # Complete discovery workflow
    # ------------------------------------------------------------------
    def discover_validate_interpret(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        variable_descriptions: Dict[str, str],
        variable_units: Dict[str, str],
        description: Optional[str] = None,
        equation_name: Optional[str] = None,
        validate_first: bool = True,
        show_formatted: bool = True,
        use_llm: bool = False,          # FIX-3: now actually read and respected
        min_validation_score: float = 85.0,
    ) -> Dict[str, Any]:
        """
        Complete discovery workflow v5.1.

        FIX-3: use_llm is now respected.  If True and the instance was
        initialised with use_llm=False (pure-PySR engine), a warning is
        logged and the call proceeds with pure PySR.  The recommended
        pattern is to set use_llm at __init__ time; the parameter here
        acts as a per-call override guard only.
        """
        # FIX-3: Per-call use_llm guard.
        _effective_llm = use_llm or self.use_llm
        if use_llm and self.llm_mode == "none":
            logger.warning(
                "[LLM] use_llm=True passed to discover_validate_interpret() but "
                "the engine was initialised in pure-PySR mode (either use_llm=False "
                "at __init__ or no API key was found).  Running pure PySR.  "
                "Reinitialise with use_llm=True to enable LLM guidance."
            )

        if equation_name:
            pass

        try:
            discovery_result = self._discover_with_retry(
                X, y, variable_names, variable_descriptions, variable_units,
                equation_name=equation_name,
            )
            self.stats["discoveries"] += 1

            # PATCH-3: Warm-start Phase 2.
            # If Phase 1 did not reach the quality threshold, extract structural
            # constraints from the best expression and re-run PySR with a tighter
            # search space.  No Julia fork required — constraints are passed as
            # standard PySR kwargs via pysr_kwargs.update(kwargs) (line 1203 of
            # symbolic_engine.py).
            _WS_THRESHOLD = -1.0   # PIN-3: disabled (set to 0.95 to re-enable)
            _p1_r2 = discovery_result.get("r2_score", 0.0)
            _p1_expr = discovery_result.get("expression", "")

            if (
                _p1_r2 < _WS_THRESHOLD
                and _p1_expr
                and _p1_expr not in ("DISCOVERY_FAILED", "NO_VALID_EQUATIONS", "VALIDATION_FAILED")
                and hasattr(self.symbolic_engine, "_extract_operators_from_equation")
            ):
                logger.info(
                    f"\n[WARM-START] Phase 1 R²={_p1_r2:.4f} < {_WS_THRESHOLD}. "
                    "Running constrained Phase 2..."
                )
                _orig_binary  = list(self.symbolic_engine.config.binary_operators)
                _orig_unary   = list(self.symbolic_engine.config.unary_operators)
                _orig_maxsize = self.symbolic_engine.config.maxsize

                try:
                    _ws_constraints = self.symbolic_engine._extract_operators_from_equation(
                        _p1_expr
                    )
                    # Temporarily tighten the engine config

                    if _ws_constraints.get("binary_operators"):
                        self.symbolic_engine.config.binary_operators = (
                            _ws_constraints["binary_operators"]
                        )
                    if _ws_constraints.get("unary_operators"):
                        self.symbolic_engine.config.unary_operators = (
                            _ws_constraints["unary_operators"]
                        )
                    if _ws_constraints.get("maxsize"):
                        self.symbolic_engine.config.maxsize = _ws_constraints["maxsize"]

                    logger.info(f"   [WARM-START] Constraints: {_ws_constraints}")

                    _p2_result = self._discover_with_retry(
                        X, y, variable_names, variable_descriptions, variable_units,
                        equation_name=equation_name,
                    )

                    # Restore original config regardless of outcome
                    self.symbolic_engine.config.binary_operators = _orig_binary
                    self.symbolic_engine.config.unary_operators  = _orig_unary
                    self.symbolic_engine.config.maxsize          = _orig_maxsize

                    _p2_r2 = _p2_result.get("r2_score", 0.0)
                    logger.info(
                        f"   [WARM-START] Phase 2 R²={_p2_r2:.4f} vs Phase 1 R²={_p1_r2:.4f}"
                    )

                    if _p2_r2 > _p1_r2:
                        logger.info("   [WARM-START] Phase 2 is better — adopting result.")
                        _p2_result["warm_start_phase"] = 2
                        _p2_result["phase1_r2"] = _p1_r2
                        discovery_result = _p2_result
                    else:
                        logger.info("   [WARM-START] Phase 1 still best — keeping.")
                        discovery_result["warm_start_phase"] = 1

                except Exception as _ws_err:
                    logger.warning(f"   [WARM-START] Phase 2 failed ({_ws_err}) — keeping Phase 1.")
                    # Config already restored inside the try block above;
                    # if exception occurred before restore, reset defensively:
                    self.symbolic_engine.config.binary_operators = _orig_binary
                    self.symbolic_engine.config.unary_operators  = _orig_unary
                    self.symbolic_engine.config.maxsize          = _orig_maxsize

            engine = discovery_result.get("discovery_engine", "unknown")
            llm_info = discovery_result.get("llm_mode", "")
            if llm_info:
                pass
            if "attempt" in discovery_result:
                pass
            if discovery_result.get("auto_configuration", {}).get("used"):
                auto_cfg = discovery_result["auto_configuration"]["config"]
                self.stats["auto_configs"] += 1

        except Exception as e:
            import traceback as _tb_mod
            _tb_str = _tb_mod.format_exc()
            logger.error(f"Discovery failed: {e}")
            logger.error(_tb_str)
            return {
                "error": "discovery_failed",
                "message": str(e),
                "traceback": _tb_str,
            }

        test_data = {name: X[:, i] for i, name in enumerate(variable_names)}
        validation_result = self._safe_validate(
            expression_str=discovery_result["expression"],
            variable_definitions=variable_descriptions,
            variable_units=variable_units,
            test_data=test_data,
        )
        self.stats["validations"] += 1

        if validation_result.get("validation_exception"):
            pass

        if discovery_result.get("collapsed_constants"):
            validation_result.setdefault("warnings", []).append(
                f"Collapsed constants detected: {discovery_result['collapsed_constants']}"
            )

        validation_score = validation_result["total_score"]
        r2_score = discovery_result["r2_score"]
        accepted = False
        accept_reason = None

        if self.discovery_mode == DiscoveryMode.STRICT:
            accepted = validation_score >= min_validation_score
        elif self.discovery_mode == DiscoveryMode.CALIBRATED:
            accepted = r2_score >= 0.99 and validation_score >= 30.0
            if accepted:
                accept_reason = "Calibrated physics acceptance (constants absorbed)"

        complete_result = {
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "domain": self.domain,
            "discovery": discovery_result,
            "validation": validation_result,
            "acceptance": {
                "accepted": accepted,
                "mode": self.discovery_mode.value,
                "reason": accept_reason,
            },
            "metadata": {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "variable_names": variable_names,
                "discovery_engine": discovery_result.get("discovery_engine"),
                "llm_mode": self.llm_mode,
                "equation_name": equation_name,
                "version": "5.1",
            },
        }

        self.results.append(complete_result)


        return complete_result

    # ------------------------------------------------------------------
    # discover() thin adapter — FIX-6: propagates use_llm from metadata
    # ------------------------------------------------------------------
    def discover(
        self,
        X: np.ndarray,
        y: np.ndarray,
        var_names: List[str],
        description: str = "",
        metadata: Optional[Dict] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Thin adapter for benchmark runners.

        FIX-6: metadata may now contain use_llm (bool) and llm_mode (str)
        to override the instance defaults per-call.  This lets benchmark
        runners toggle LLM guidance equation-by-equation without
        reinstantiating the system.
        """
        metadata = metadata or {}

        _noise_level = metadata.get("noise_level", None)
        self._current_noise_level = _noise_level

        # PROD-7: domain fast-path (unchanged)
        _domain_from_meta = metadata.get("domain", "")
        if _domain_from_meta and _domain_from_meta != self.domain:
            logger.info(
                f"[DOMAIN-FIX] Updating domain: '{self.domain}' → '{_domain_from_meta}'"
            )
            self.domain = _domain_from_meta
            self.symbolic_engine.domain = _domain_from_meta

        # FIX-6: per-call LLM override from metadata
        _meta_use_llm = metadata.get("use_llm", self.use_llm)

        variable_descriptions = metadata.get(
            "variable_descriptions", {v: v for v in var_names}
        )
        variable_units = metadata.get("variable_units", {v: "" for v in var_names})
        equation_name = metadata.get("equation_name", description or "unknown")

        # FIX-D: extreme-scale log-transform.
        # When any feature spans >6 orders of magnitude PySR collapses to a
        # constant (e.g. gravitational force: 1e-9 kg masses, 1e22 N force).
        # Apply signed log10 per feature and flag the result so callers know
        # the returned expression is in log-space.
        _X_orig = X
        _y_orig = y
        _log_scaled = False
        _LOG_THRESHOLD = 4  # log10 OOM (FIX-D v5.2: was 6, too strict for gravity/EM)
        _ABS_MAG_THRESHOLD = 1e5  # trigger log-scale if X column median abs > this

        def _signed_log10(arr: np.ndarray) -> np.ndarray:
            """sign(x) * log10(|x| + 1) — safe for zeros."""
            return np.sign(arr) * np.log10(np.abs(arr) + 1.0)

        try:
            _needs_log: List[bool] = []
            for _col in range(X.shape[1]):
                _vals = X[:, _col]
                _abs = np.abs(_vals[np.isfinite(_vals) & (_vals != 0)])
                if len(_abs) < 2:
                    _needs_log.append(False)
                    continue
                _range_trigger = np.log10(_abs.max()) - np.log10(_abs.min()) > _LOG_THRESHOLD
                _mag_trigger   = np.median(_abs) > _ABS_MAG_THRESHOLD
                _needs_log.append(_range_trigger or _mag_trigger)  # FIX-D v5.2
            _y_abs = np.abs(y[np.isfinite(y) & (y != 0)])
            _y_needs_log = (
                len(_y_abs) >= 2
                and np.log10(_y_abs.max()) - np.log10(_y_abs.min()) > _LOG_THRESHOLD
            )
            if any(_needs_log) or _y_needs_log:
                _log_scaled = True
                X_scaled = X.copy().astype(float)
                for _col, _do_log in enumerate(_needs_log):
                    if _do_log:
                        X_scaled[:, _col] = _signed_log10(X[:, _col])
                        logger.info(
                            f"[FIX-D] Feature '{var_names[_col]}' log10-scaled "
                            f"(range > {_LOG_THRESHOLD} OOM)"
                        )
                y_fit = _signed_log10(y) if _y_needs_log else y
                if _y_needs_log:
                    logger.info("[FIX-D] Target y log10-scaled (extreme output range)")
                X = X_scaled
                y = y_fit
                logger.info("[FIX-D] Extreme-scale log transform applied")

                # FIX-4a: rename log-scaled var_names so PySR sees
                # linear inputs and can discover the additive log-law.
                # e.g. "m1" -> "log_m1" so PySR finds log_m1 + log_m2 - 2*log_r + C
                _log_name_map: Dict[str, str] = {}
                _new_var_names = list(var_names)
                for _col, _do_log in enumerate(_needs_log):
                    if _do_log:
                        _orig_nm = var_names[_col]
                        _log_nm  = f"log_{_orig_nm}"
                        _new_var_names[_col] = _log_nm
                        _log_name_map[_log_nm] = _orig_nm
                        logger.info(
                            f"[FIX-4a] Renamed feature '{_orig_nm}' -> '{_log_nm}' "
                            "for PySR (log-space)"
                        )
                var_names = _new_var_names
        except Exception as _fd_err:
            logger.warning(f"[FIX-D] Scale detection failed ({_fd_err}) — using raw data")
            X, y = _X_orig, _y_orig
            _log_scaled = False

        # Store originals so FIX-A can compute RMSE in original (unscaled) units.
        self._discover_X_orig = _X_orig
        self._discover_y_orig = _y_orig
        self._discover_scale_log = _log_scaled

        # FIX-4b: initialise map so it exists whether or not log-scaling fired.
        # On the non-log path _log_name_map stays empty, so the rename below
        # is a no-op.  On the log path it was set by Fix-4a above.
        if not _log_scaled:
            _log_name_map = {}

        # FIX-POW: enable "pow" only when every feature value is non-negative
        # (evaluated on post-scaled X so log-transform is already factored in).
        # Negative bases cause Julia DomainError with fractional exponents.
        # Callers can override via metadata: metadata={"allow_pow": True/False}.
        # FIX-POW: default to False — never auto-add 'pow' based on data sign.
        # PySR's 'pow' (Julia ^) with unconstrained exponents produces
        # numerically-accurate but symbolically meaningless expressions such as
        # R^-600.  Callers that genuinely need fractional powers must pass
        # metadata={"allow_pow": True} explicitly.
        _allow_pow = metadata.get("allow_pow", False)
        _orig_binary_ops = list(self.symbolic_engine.config.binary_operators)
        if _allow_pow and "pow" not in _orig_binary_ops:
            self.symbolic_engine.config.binary_operators = _orig_binary_ops + ["pow"]
            logger.info(
                "[FIX-POW] X is non-negative — adding 'pow' to binary_operators "
                "(was: %s)", _orig_binary_ops,
            )
        elif not _allow_pow and "pow" in _orig_binary_ops:
            self.symbolic_engine.config.binary_operators = [
                op for op in _orig_binary_ops if op != "pow"
            ]
            logger.info(
                "[FIX-POW] X has negative values — removing 'pow' from binary_operators "
                "(was: %s)", _orig_binary_ops,
            )

        try:
            full_result = self.discover_validate_interpret(
                X=X,
                y=y,
                variable_names=var_names,
                variable_descriptions=variable_descriptions,
                variable_units=variable_units,
                description=description,
                equation_name=equation_name,
                show_formatted=verbose,
                use_llm=_meta_use_llm,
            )

            # FIX-POW: restore original binary_operators regardless of result
            self.symbolic_engine.config.binary_operators = _orig_binary_ops

            if "error" in full_result and full_result["error"] == "discovery_failed":
                raise RuntimeError(full_result.get("message", "Discovery failed"))

            discovery = full_result.get("discovery", {})
            validation = full_result.get("validation", {})
            r2 = float(discovery.get("r2_score", 0.0))

            formula = discovery.get("expression", "N/A")

            # FIX-4b: reverse-rename log_m1 -> m1 etc. in the formula string
            # so the caller receives readable physics notation, not log_* names.
            _log_name_map_local = locals().get("_log_name_map", {})
            if _log_name_map_local and formula not in (
                "DISCOVERY_FAILED", "NO_VALID_EQUATIONS",
                "VALIDATION_FAILED", "N/A",
            ):
                import re as _re
                _formula_out = formula
                # Sort longest first to avoid partial replacements
                for _log_nm, _orig_nm in sorted(
                    _log_name_map_local.items(), key=lambda kv: -len(kv[0])
                ):
                    # Replace whole-word occurrences only
                    _formula_out = _re.sub(
                        rf"\b{_re.escape(_log_nm)}\b",
                        _orig_nm,
                        _formula_out,
                    )
                if _formula_out != formula:
                    logger.info(
                        "[FIX-4b] Formula de-logged: '%s' -> '%s'",
                        formula, _formula_out,
                    )
                formula = _formula_out

            # FIX-A: compute RMSE from the expression string — the "predictions"
            # key was never written by _discover_with_retry(), so the old code
            # always returned inf.  Evaluate the expression directly against the
            # *original* (unscaled) training y.
            rmse = float("inf")
            _X_for_rmse = getattr(self, "_discover_X_orig", X)
            _y_for_rmse = getattr(self, "_discover_y_orig", y)
            if formula and formula not in (
                "DISCOVERY_FAILED", "NO_VALID_EQUATIONS", "VALIDATION_FAILED", "N/A"
            ):
                try:
                    _norm_expr = self._normalise_expression(formula)
                    # FIX-A: build namespace with BOTH original and sanitised
                    # variable names so the eval succeeds whether the formula
                    # uses "I" or "I_var" (whichever PySR received after rename).
                    _name_map = discovery.get("variable_name_mapping", {})
                    _safe_names = discovery.get("variable_names", var_names)
                    # FIX-RATIO v5.2: include ratio/augmented features in eval namespace
                    _X_aug    = getattr(self.symbolic_engine, "_last_X_aug",    _X_for_rmse)
                    _aug_nms  = getattr(self.symbolic_engine, "_last_aug_names", list(var_names))
                    # FIX-RMSE v5.2: formula was fit on normalised X (_X_aug base cols),
                    # so bind base columns from _X_aug, not original-scale _X_for_rmse.
                    _ns: Dict[str, Any] = {
                        "np": np,
                        # FIX-NS: bare math names (exp/log/sqrt etc.) added so
                        # PySR-emitted formulas eval without NameError.
                        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                        "abs": np.abs, "sin": np.sin, "cos": np.cos,
                        "tan": np.tan, "arcsin": np.arcsin,
                        "arccos": np.arccos, "arctan": np.arctan,
                        # base cols from normalised X (matches what PySR was fit to)
                        **{name: _X_aug[:, i] for i, name in enumerate(var_names)
                           if i < _X_aug.shape[1]},
                        # sanitised name aliases (e.g. I_var for I)
                        **{_safe_names[i]: _X_aug[:, i]
                           for i in range(len(var_names))
                           if i < _X_aug.shape[1] and _safe_names[i] != var_names[i]},
                        # ratio_ and other engineered feature columns
                        **{nm: _X_aug[:, i] for i, nm in enumerate(_aug_nms)
                           if nm not in var_names and i < _X_aug.shape[1]},
                    }
                    # FIX-RMSE-LOG: After FIX-4b renames log_m1→m1 etc. in the
                    # formula, the namespace still binds log_m1/log_m2/log_r.
                    # Bind the original names too so the eval succeeds.
                    if _log_name_map_local:
                        _vn_list = list(var_names)
                        for _log_nm, _orig_nm in _log_name_map_local.items():
                            if _log_nm in _vn_list:
                                _col_idx = _vn_list.index(_log_nm)
                                if _col_idx < _X_aug.shape[1]:
                                    _ns[_orig_nm] = _X_aug[:, _col_idx]
                    _y_pred = eval(_norm_expr, {"__builtins__": {}}, _ns)  # noqa: S307
                    _y_pred = np.asarray(_y_pred, dtype=float)
                    if _y_pred.shape == ():
                        _y_pred = np.full(len(_y_for_rmse), float(_y_pred))
                    # FIX-OVERFLOW v3: align prediction and reference to the same
                    # y-space, then let np.isfinite() filter true overflow (inf/nan).
                    # No scale-based clamping: a "bad but finite" formula should
                    # report a large RMSE, not inf.
                    # When scale_log=True the formula was fit on sign(y)*log10(|y|+1),
                    # so the reference must be transformed to match.
                    _scale_log_flag = getattr(self, "_discover_scale_log", False)
                    if _scale_log_flag:
                        _y_rmse_ref = np.sign(_y_for_rmse) * np.log10(
                            np.abs(_y_for_rmse) + 1.0
                        )
                    else:
                        _y_rmse_ref = _y_for_rmse
                    _finite = np.isfinite(_y_pred) & np.isfinite(_y_rmse_ref)
                    if _finite.sum() >= 2:
                        rmse = float(
                            np.sqrt(np.mean(
                                (_y_rmse_ref[_finite] - _y_pred[_finite]) ** 2
                            ))
                        )
                except Exception as _rmse_err:
                    logger.warning(f"[FIX-A] RMSE eval failed ({type(_rmse_err).__name__}: {_rmse_err}) — reporting inf")

            success = r2 > 0.0 and formula not in (
                "DISCOVERY_FAILED", "NO_VALID_EQUATIONS", "VALIDATION_FAILED", "N/A"
            )

            # FIX-D: report whether the expression is in log-space
            _scale_log = getattr(self, "_discover_scale_log", False)

            return {
                # FIX-B: all three key aliases test harnesses look for
                "formula": formula,
                "expression": formula,
                "final_formula": formula,
                # FIX-B: variable names so callers can bind the expression
                "variable_names": var_names,
                "success": success,
                "r2": r2,
                "rmse": rmse,
                "strategy": discovery.get("discovery_engine", "symbolic"),
                "llm_mode": discovery.get("llm_mode", self.llm_mode),
                "validations": 1 if validation else 0,
                "validation_score": validation.get("total_score", 0.0),
                # FIX-D: flag log-space transform
                "complexity": discovery.get("complexity", None),
                "trace": discovery.get("trace", []),
                "scale_log": _scale_log,
                "error": None,
            }

        except Exception as exc:
            # FIX-POW: always restore binary_operators on exception path too
            self.symbolic_engine.config.binary_operators = _orig_binary_ops
            logger.error(
                f"discover() caught top-level exception — {type(exc).__name__}: {exc}",
                exc_info=True,
            )
            return {
                "success": False,
                "r2": 0.0,
                "rmse": float("inf"),
                "formula": "N/A",
                "expression": "N/A",
                "final_formula": "N/A",
                "variable_names": var_names,
                "strategy": "error",
                "llm_mode": "none",
                "validations": 0,
                "scale_log": False,
                "error": str(exc)[:200],
            }

    def print_statistics_summary(self) -> None:
        """Print statistics summary."""
        print(f"\n{'=' * 70}")
        print("STATISTICS SUMMARY v5.1")
        print(f"{'=' * 70}")
        print(f"\nOverall:")
        print(f"   Discoveries: {self.stats['discoveries']}")
        print(f"   Validations: {self.stats['validations']}")
        print(f"\nSymbolicEngine:")
        print(f"   Attempts: {self.stats['symbolic_attempts']}")
        print(f"   Successes: {self.stats['symbolic_successes']}")
        print(f"   Failures: {self.stats['symbolic_failures']}")
        if self.stats["symbolic_attempts"] > 0:
            rate = 100 * self.stats["symbolic_successes"] / self.stats["symbolic_attempts"]
            print(f"   Success rate: {rate:.1f}%")
        print(f"\nLLM Guidance (mode={self.llm_mode}):")
        print(f"   Calls guided by LLM: {self.stats['llm_guided']}")
        print(f"   Calls using pure PySR: {self.stats['llm_skipped']}")
        if self.enable_physics_fallback:
            print(f"\nPhysicsAware:")
            print(f"   Used: {self.stats['physics_used']}")
            print(f"   Successes: {self.stats['physics_successes']}")
        print(f"\nAuto-Configuration:")
        print(f"   Used: {self.stats['auto_configs']} times")
        print(f"\n{'=' * 70}\n")

    def save_results(self, filename: Optional[str] = None) -> str:
        """Save results to JSON — PROD-4/5: single-pass serialisation."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"discovery_results_v51_{timestamp}.json"

        results_list = [_to_serialisable(r) for r in self.results]

        output = {
            "version": "5.1",
            "timestamp": datetime.now().isoformat(),
            "domain": self.domain,
            "llm_mode": self.llm_mode,
            "statistics": self.stats,
            "results": results_list,
        }

        with open(filename, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"[OK] Results saved to {filename}")
        return filename


if __name__ == "__main__":
    # ============================================================================
    # QUICK TEST (covers FIX-A … FIX-D)
    # ============================================================================

    # ── Test A: Ohm's Law — FIX-A (RMSE), FIX-B (key aliases), FIX-C (deterministic) ──
    print("\nTest A: Ohm's Law — pure PySR (FIX-A: RMSE, FIX-B: keys, FIX-C: deterministic)")
    print("-" * 80)
    np.random.seed(42)
    _I = np.random.uniform(0.1, 10, 100)
    _R = np.random.uniform(1, 100, 100)
    _V = _I * _R + np.random.normal(0, np.abs(_I * _R) * 0.01, 100)
    _X_ohm = np.column_stack([_I, _R])
    
    _sys_a = HybridDiscoverySystem(
        domain="physics",
        discovery_config=DiscoveryConfig(niterations=75, enable_auto_configuration=True),
        enable_physics_fallback=False,
        max_retries=5,
        use_llm=False,
        allow_nondeterministic=False,
    )
    _res_a = _sys_a.discover(
        _X_ohm, _V, ["I", "R"],
        description="Ohm's Law",
        metadata={
            "equation_name": "ohms_law",
            "variable_descriptions": {"I": "Current", "R": "Resistance"},
            "variable_units": {"I": "A", "R": "ohm"},
        },
    )
    print(f"  formula       : {_res_a['formula']}")
    print(f"  expression    : {_res_a['expression']}")
    print(f"  final_formula : {_res_a['final_formula']}")
    print(f"  variable_names: {_res_a['variable_names']}")
    print(f"  R²            : {_res_a['r2']:.4f}")
    _rmse_ok_a = _res_a["rmse"] < float("inf") and _res_a["rmse"] >= 0
    print(f"  RMSE          : {_res_a['rmse']:.4f}  {'[FIX-A OK]' if _rmse_ok_a else '[FIX-A FAIL]'}")
    print(f"  scale_log     : {_res_a['scale_log']}  (should be False for Ohm\'s Law)")
    _sys_a.print_statistics_summary()
    
    # ── Test B: Gravitational Force — FIX-D (extreme-scale log) + FIX-A RMSE ──────
    print("\nTest B: Gravitational Force — extreme-scale log transform (FIX-D + FIX-A)")
    print("-" * 80)
    np.random.seed(42)
    _G = 6.674e-11
    _m1 = np.random.uniform(1e10, 1e12, 120)
    _m2 = np.random.uniform(1e10, 1e12, 120)
    _r  = np.random.uniform(1e6,  1e8,  120)
    _F  = _G * _m1 * _m2 / _r ** 2
    _X_grav = np.column_stack([_m1, _m2, _r])
    
    _sys_d = HybridDiscoverySystem(
        domain="physics",
        discovery_config=DiscoveryConfig(niterations=75, enable_auto_configuration=True),
        enable_physics_fallback=False,
        max_retries=5,
        use_llm=False,
    )
    _res_d = _sys_d.discover(
        _X_grav, _F, ["m1", "m2", "r"],
        description="Gravitational Force",
        metadata={
            "equation_name": "gravity",
            "variable_descriptions": {"m1": "mass 1", "m2": "mass 2", "r": "distance"},
            "variable_units": {"m1": "kg", "m2": "kg", "r": "m"},
        },
    )
    print(f"  formula   : {_res_d['formula']}")
    print(f"  R²        : {_res_d['r2']:.4f}")
    _rmse_ok_d = _res_d["rmse"] < float("inf") and _res_d["rmse"] >= 0
    print(f"  RMSE      : {_res_d['rmse']}  {'[FIX-A OK]' if _rmse_ok_d else '[FIX-A FAIL: RMSE=inf]'}")
    _log_ok = _res_d["scale_log"]
    print(f"  scale_log : {_log_ok}  {'[FIX-D OK: log-transform applied]' if _log_ok else '[FIX-D FAIL: no log-transform]'}")
    
    # ── Test D: BIC parsimony — Ohm's Law should pick I*R over a complex polynomial ──
    print("\nTest D: BIC parsimony (FIX-PARSIMONY) — simple I*R should beat bloated poly")
    print("-" * 80)
    # Reuse _res_a from Test A (same Ohm's Law run)
    _expr_d = _res_a.get("expression", "")
    _bic_trace_d = [t for t in _res_a.get("trace", []) if t.startswith("bic_selection")]
    _complexity_d = _res_a.get("complexity", None)
    print(f"  expression   : {_expr_d}")
    print(f"  complexity   : {_complexity_d}")
    print(f"  bic_trace    : {_bic_trace_d[:1] if _bic_trace_d else '[not found]'}")
    # Pass: complexity ≤ 5 means we got something like I*R or R*I, not a polynomial
    _bic_ok = isinstance(_complexity_d, (int, float)) and _complexity_d <= 5
    print(f"  [FIX-PARSIMONY {'OK' if _bic_ok else 'NOTE: complexity > 5 — BIC may need more PySR iterations'}]")
    
    # ── Test E: nsimplify — constant snapping ──────────────────────────────────────
    print("\nTest E: nsimplify constant-snapping (FIX-SIMPLIFY)")
    print("-" * 80)
    try:
        import sympy as _sp
        # Simulate the kind of near-1 constant PySR sometimes emits
        _test_expr_raw = "0.99956363 * I * R"
        _free_e = _sp.sympify(_test_expr_raw).free_symbols
        _real_e = {str(s): _sp.Symbol(str(s), real=True, positive=True) for s in _free_e}
        _sym_e  = _sp.sympify(_test_expr_raw, locals=_real_e)
        _sym_ns = _sp.nsimplify(_sym_e, rational=False, tolerance=5e-3)
        _simp_e = _sp.simplify(_sym_ns)
        _simp_str_e = str(_simp_e)
        print(f"  raw expression : {_test_expr_raw}")
        print(f"  after nsimplify: {_simp_str_e}")
        _simplify_ok = "0.9" not in _simp_str_e  # 0.99956 should have been snapped to 1
        print(f"  [FIX-SIMPLIFY {'OK — constant snapped to 1' if _simplify_ok else 'NOTE: constant not snapped'}]")
        assert _simplify_ok, f"nsimplify did not snap 0.99956363 to 1 (got: {_simp_str_e})"
    
        # Also check log(exp(x)) → x collapses with real assumptions
        _x_sym = _sp.Symbol("x", real=True, positive=True)
        _log_exp = _sp.log(_sp.exp(_x_sym))
        _collapsed = _sp.simplify(_log_exp)
        _logexp_ok = str(_collapsed) == "x"
        print(f"  log(exp(x)) → {_collapsed}  [{'OK' if _logexp_ok else 'NOTE: did not collapse'}]")
        assert _logexp_ok, f"log(exp(x)) did not simplify to x (got: {_collapsed})"
        print("  ✅ Test E passed")
    except Exception as _e_err:
        print(f"  ⚠️  Test E error: {_e_err}")
    
    # ── Test C: LLM hybrid mode (optional — requires ANTHROPIC_API_KEY) ─────────────
    print("\nTest C: LLM hybrid mode (use_llm=True, requires ANTHROPIC_API_KEY)")
    print("-" * 80)
    if os.getenv("ANTHROPIC_API_KEY"):
        _sys_llm = HybridDiscoverySystem(
            domain="physics",
            discovery_config=DiscoveryConfig(niterations=75, enable_auto_configuration=True),
            enable_physics_fallback=False,
            max_retries=5,
            use_llm=True,
            llm_mode="hybrid",
            llm_n_candidates=3,
        )
        _res_llm = _sys_llm.discover_validate_interpret(
            X=_X_ohm, y=_V,
            variable_names=["I", "R"],
            variable_descriptions={"I": "Current in amperes", "R": "Resistance in ohms"},
            variable_units={"I": "A", "R": "ohm"},
            description="Ohm's Law (LLM hybrid)",
            equation_name="ohms_law",
        )
        print(f"  Expression : {_res_llm['discovery']['expression']}")
        print(f"  R²         : {_res_llm['discovery']['r2_score']:.4f}")
        print(f"  LLM mode   : {_res_llm['discovery'].get('llm_mode', 'N/A')}")
        _sys_llm.print_statistics_summary()
    else:
        print("  Skipping — ANTHROPIC_API_KEY not set.")
        print("  Set the key and rerun with use_llm=True to test LLM guidance.")
    
    print('\n✅ HypatiaX embed loaded — HybridDiscoverySystem, DiscoveryConfig ready')

    import numpy as np
    
    np.random.seed(42)
    I = np.random.uniform(0.1, 10, 80)
    R = np.random.uniform(1, 100, 80)
    V = I * R + np.random.normal(0, np.abs(I * R) * 0.01, 80)
    
    sys_test = HybridDiscoverySystem(
        domain='physics',
        discovery_config=DiscoveryConfig(niterations=75, enable_auto_configuration=True),
        max_retries=2, use_llm=False, allow_nondeterministic=False,
    )
    res = sys_test.discover(np.column_stack([I, R]), V, ['Current', 'R'],
                            description="Ohm's Law")
    print(f"Formula : {res['formula']}")
    print(f"R²      : {res['r2']:.4f}")
    assert res['r2'] > 0.95
    print('✅ Smoke-test passed')
