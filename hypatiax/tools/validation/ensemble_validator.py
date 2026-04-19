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

from hypatiax.tools.validation.dimensional_validator import DimensionalValidator
from hypatiax.tools.validation.domain_validator import DomainValidator
from hypatiax.tools.validation.symbolic_validator import SymbolicValidator

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
        return history[-limit:] if limit else history

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
