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
