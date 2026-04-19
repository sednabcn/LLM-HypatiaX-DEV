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
