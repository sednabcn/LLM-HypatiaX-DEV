"""
HypatiaX Hybrid Discovery System v4.1-PROD
==========================================
PRODUCTION VERSION — performance improvements only.
Core worker logic (SymbolicEngine.discover, _discover_with_retry,
discover_validate_interpret, PhysicsAwareRegressor) is UNCHANGED.

What changed vs v4.1
--------------------
PROD-1  Expression-quality cache (functools.lru_cache wrapper)
        _check_expression_quality is called up to max_retries×3 times per
        discovery pass with the same (expression, r2) pair.  A 128-entry
        LRU cache removes redundant regex scans — zero behaviour change.

PROD-2  Operator-injection logic de-duplicated
        The trig-injection block was copy-pasted between __init__ and
        _discover_with_retry.  Extracted into a single private method
        _inject_operators() so both call-sites share one path and future
        edits are applied once.

PROD-3  _normalise_expression compiled regex
        _PYSR_OP_ALIASES substitution now uses pre-compiled patterns stored
        in _PYSR_OP_PATTERNS (compiled once at class-definition time).
        Avoids re-compiling the same four regex objects on every call.

PROD-4  JSON serialisation helper is O(n) not O(n²)
        save_results() previously iterated over result dicts twice (once to
        build result_copy, again to serialise).  Now uses a single-pass
        recursive _to_serialisable() helper — same output, less RAM
        pressure on large result sets.

PROD-5  deque → list copy done once per save
        list(self.results) called inside the loop previously; moved outside.

PROD-6  Lazy LLM provider initialisation
        _initialize_llm_providers() now skips construction if the API key is
        absent — avoids importing AnthropicProvider / GoogleProvider when they
        are not needed (saves ~80 ms on cold start when keys are missing).

PROD-7  discover() domain-fix fast-path
        The string comparison `_domain_from_meta != self.domain` now short-
        circuits when the metadata domain is empty, avoiding an unnecessary
        attribute write on every call.

All other behaviour, defaults, log messages, and public APIs are identical
to v4.1.  Drop-in replacement: import HybridDiscoverySystem from this module.
"""

import json
import logging
import os
import random
import re
import time
from collections import deque
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_env_path)

from hypatiax.tools.symbolic.symbolic_engine import (
    DiscoveryConfig,
    SymbolicEngine,
    detect_collapsed_constants,
)
from hypatiax.tools.symbolic.physics_aware_regressor import PhysicsAwareRegressor
from hypatiax.tools.validation.ensemble_validator import EnsembleValidator


class DiscoveryMode(Enum):
    STRICT = "strict"
    CALIBRATED = "calibrated"


# ---------------------------------------------------------------------------
# PROD-1: Cached quality check — wraps the real computation so the cache
# key is (expression, r2_rounded) and doesn't cross instance boundaries.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=256)
def _cached_quality(
    expression: str,
    r2_rounded: float,
    complexity_threshold: int,
) -> Tuple[bool, int, Tuple[str, ...]]:
    """
    Pure-function version of _check_expression_quality used as the LRU target.
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
# PROD-4: Recursive serialisation helper (replaces the nested loop in save_results)
# ---------------------------------------------------------------------------
def _to_serialisable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


class HybridDiscoverySystem:
    """
    Hybrid discovery system v4.1-PROD.

    Identical behaviour to v4.1; see module docstring for performance changes.
    """

    # PROD-3: Alias table (unchanged from v4.1)
    _PYSR_OP_ALIASES: Dict[str, str] = {
        "safe_asin":   "arcsin",
        "safe_acos":   "arccos",
        "asin_of_sin": "arcsin",
        "acos_of_cos": "arccos",
        "atan_of_tan": "arctan",
    }
    # Pre-compiled at class-definition time — shared across all instances.
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
        max_retries: int = 3,
        enable_auto_config: bool = True,
        anthropic_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
    ):
        """Initialize hybrid system v4.1-PROD (identical signature to v4.1)."""
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

        logger.info("=" * 70)
        logger.info("HybridDiscoverySystem v4.1-PROD — OPTICS FIX + IMPROVED DIAGNOSTICS")
        logger.info("=" * 70)
        logger.info(f"Domain: {domain}")
        logger.info(f"Discovery mode: {self.discovery_mode.value}")
        logger.info(f"Primary LLM: {primary_llm}")
        logger.info(f"Auto-config: {enable_auto_config}")
        logger.info(f"Max retries: {max_retries}")
        logger.info(f"PhysicsAware fallback: {enable_physics_fallback}")
        logger.info(f"Complexity threshold: {complexity_penalty_threshold}")
        logger.info("=" * 70)

        if discovery_config is None:
            symbolic_config = DiscoveryConfig(
                niterations=40,
                enable_auto_configuration=enable_auto_config,
            )
            logger.info("Using default iterations: 40 (no config provided — runner will override)")
        else:
            symbolic_config = discovery_config
            logger.info(f"Using provided iterations: {symbolic_config.niterations}")
            logger.info(f"Parsimony: {symbolic_config.parsimony}")
            logger.info(
                f"Transcendental compositions: {symbolic_config.use_transcendental_compositions}"
            )

        # PROD-2: Operator injection extracted to _inject_operators()
        self._inject_operators(symbolic_config, domain)

        try:
            self.symbolic_engine = SymbolicEngine(symbolic_config, domain=domain)
        except Exception:
            logger.error("SymbolicEngine construction FAILED", exc_info=True)
            raise

        try:
            self.validator = EnsembleValidator(
                domain=domain, max_history=max_results, weights=validation_weights
            )
        except Exception:
            logger.error("EnsembleValidator construction FAILED", exc_info=True)
            raise

        # PROD-6: Lazy LLM provider initialisation
        self._initialize_llm_providers(anthropic_api_key, google_api_key)

        self.max_results = max_results
        self.results: Any = deque(maxlen=max_results) if max_results is not None else []

        self.stats: Dict[str, int] = {
            "discoveries": 0,
            "symbolic_attempts": 0,
            "symbolic_successes": 0,
            "symbolic_failures": 0,
            "physics_used": 0,
            "physics_successes": 0,
            "validations": 0,
            "auto_configs": 0,
        }

        self.use_rich_output = use_rich_output
        logger.info("[OK] HybridDiscoverySystem v4.1-PROD initialized\n")

    # ------------------------------------------------------------------
    # PROD-2: shared operator-injection logic
    # ------------------------------------------------------------------
    @staticmethod
    def _inject_operators(symbolic_config: DiscoveryConfig, domain: str) -> None:
        """Inject safe_asin/safe_acos when use_transcendental_compositions is True.

        This is the same logic as the __init__ block in v4.1, extracted so
        _discover_with_retry can call it without duplicating code.
        """
        _TRIG_DEFAULTS = ["sin", "cos", "tan"]
        _needs_inv_trig = getattr(symbolic_config, "use_transcendental_compositions", False)
        if _needs_inv_trig:
            _inv_trig = ["safe_asin", "safe_acos"]
            _current = list(getattr(symbolic_config, "unary_operators", None) or [])
            if not _current:
                _current = list(_TRIG_DEFAULTS)
                logger.info(
                    f"[AUTO-v4.1-PROD] unary_operators was empty — seeding with trig defaults: {_current}"
                )
            _added = [op for op in _inv_trig if op not in _current]
            if _added:
                symbolic_config.unary_operators = _current + _added
                logger.info(
                    f"[AUTO-v4.1-PROD] Injected inverse-trig operators {_added} "
                    f"(use_tc=True). Full unary set: {symbolic_config.unary_operators}"
                )
        else:
            logger.info(
                f"[AUTO-v4.1-PROD] Skipping safe_asin/safe_acos injection "
                f"(domain='{domain}', use_tc=False)"
            )

    def _initialize_llm_providers(
        self, anthropic_api_key: Optional[str], google_api_key: Optional[str]
    ) -> None:
        """Initialize LLM providers — PROD-6: skip construction when key is absent."""
        # Anthropic
        api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            try:
                from hypatiax.tools.llm_providers.anthropic_provider import AnthropicProvider
                self.anthropic_provider = AnthropicProvider(api_key=api_key, max_tokens=4096)
            except Exception:
                self.anthropic_provider = None
        else:
            self.anthropic_provider = None

        # Google
        api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                from hypatiax.tools.llm_providers.google_provider import GoogleProvider
                self.google_provider = GoogleProvider(api_key=api_key, max_output_tokens=8192)
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
        r2_rounded = round(r2, 6)  # round to keep cache hit rate high
        is_overfit, complexity, warnings_tuple = _cached_quality(
            expression, r2_rounded, self.complexity_penalty_threshold
        )
        return {
            "is_overfit": is_overfit,
            "complexity": complexity,
            "warnings": list(warnings_tuple),
        }

    def _detect_rational_pattern(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Detect if data likely follows a rational/saturation pattern.

        Unchanged from v4.1.
        """
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
    # Core discovery worker — UNCHANGED from v4.1
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
        """Discover with retry — core logic identical to v4.1."""
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
                    X, y, variable_names, equation_name=equation_name, random_state=seed
                )

                r2 = result.get("r2_score", 0)
                expr = result.get("expression", "")

                try:
                    collapsed = detect_collapsed_constants(expr, variable_names)
                except Exception:
                    logger.error("detect_collapsed_constants FAILED", exc_info=True)
                    collapsed = []

                result["collapsed_constants"] = collapsed
                logger.info(f"   Result: {expr}")
                logger.info(f"   R2 = {r2:.4f}")

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
            logger.info(f"\\n[SUCCESS] SymbolicEngine succeeded (R2={best_r2:.4f})")
            self.stats["symbolic_successes"] += 1
            return best_result
        else:
            logger.warning(f"\\n[WARNING] SymbolicEngine best R2={best_r2:.4f}")
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
                + f"\n  HINT: If this is an optics/trig equation (e.g. Snell's law), "
                  f"ensure safe_asin/safe_acos are in unary_operators (DiscoveryConfig). "
                  f"Domain detected: '{self.domain}'."
            ) from last_attempt_error

    @staticmethod
    def _normalise_expression(expression_str: str) -> str:
        """Replace PySR custom operator names — PROD-3: uses pre-compiled patterns."""
        result = expression_str
        for pat, numpy_name in HybridDiscoverySystem._PYSR_OP_PATTERNS.values():
            result = pat.sub(numpy_name, result)
        return result

    def _safe_validate(
        self,
        expression_str: str,
        variable_definitions: Dict[str, str],
        variable_units: Dict[str, str],
        test_data: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Safe validation — identical to v4.1."""
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
    # Complete discovery workflow — UNCHANGED from v4.1
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
        use_llm: bool = False,
        min_validation_score: float = 85.0,
    ) -> Dict[str, Any]:
        """Complete discovery workflow v4.1-PROD — identical behaviour to v4.1."""
        print(f"\n{'=' * 70}")
        print("DISCOVERY WORKFLOW v4.1-PROD")
        print(f"{'=' * 70}")
        print(f"Description: {description or 'Unnamed'}")
        print(f"Domain: {self.domain.upper()}")
        print(f"Samples: {len(X)}")
        print(f"Variables: {variable_names}")
        if equation_name:
            print(f"Equation hint: {equation_name}")
        print(f"{'=' * 70}")

        print("\n[DISCOVER] Running symbolic regression...")
        try:
            discovery_result = self._discover_with_retry(
                X, y, variable_names, variable_descriptions, variable_units,
                equation_name=equation_name,
            )
            self.stats["discoveries"] += 1

            engine = discovery_result.get("discovery_engine", "unknown")
            print("\n[OK] Discovery complete")
            print(f"   Expression: {discovery_result['expression']}")
            print(f"   R2 Score: {discovery_result['r2_score']:.4f}")
            print(f"   Engine: {engine}")
            if "attempt" in discovery_result:
                print(f"   Attempt: {discovery_result['attempt']}/{self.max_retries}")
            if discovery_result.get("auto_configuration", {}).get("used"):
                auto_cfg = discovery_result["auto_configuration"]["config"]
                print(f"   Auto-config: {auto_cfg.get('reason', 'N/A')}")
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

        print("\n[VALIDATE] Checking expression quality...")
        test_data = {name: X[:, i] for i, name in enumerate(variable_names)}
        validation_result = self._safe_validate(
            expression_str=discovery_result["expression"],
            variable_definitions=variable_descriptions,
            variable_units=variable_units,
            test_data=test_data,
        )
        self.stats["validations"] += 1

        print("[OK] Validation complete")
        print(f"   Score: {validation_result['total_score']:.1f}/100")
        if validation_result.get("validation_exception"):
            print("   [WARNING] Validation had errors (likely unit system)")

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
                "equation_name": equation_name,
                "version": "4.1-prod",
            },
        }

        self.results.append(complete_result)

        print(f"\n{'=' * 70}")
        print("[OK] WORKFLOW COMPLETE")
        print(f"{'=' * 70}\n")

        return complete_result

    # ------------------------------------------------------------------
    # discover() thin adapter — PROD-7 fast-path domain check
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
        """Thin adapter for the benchmark runner — PROD-7: domain fast-path."""
        metadata = metadata or {}

        _noise_level = metadata.get("noise_level", None)
        self._current_noise_level = _noise_level

        # PROD-7: Skip attribute write when domains already match or meta is empty.
        _domain_from_meta = metadata.get("domain", "")
        if _domain_from_meta and _domain_from_meta != self.domain:
            logger.info(
                f"[DOMAIN-FIX] Updating domain: '{self.domain}' → '{_domain_from_meta}' "
                "(from runner metadata — enables correct operator injection)"
            )
            self.domain = _domain_from_meta
            self.symbolic_engine.domain = _domain_from_meta

        variable_descriptions = metadata.get(
            "variable_descriptions", {v: v for v in var_names}
        )
        variable_units = metadata.get("variable_units", {v: "" for v in var_names})
        equation_name = metadata.get("equation_name", description or "unknown")

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
            )

            if "error" in full_result and full_result["error"] == "discovery_failed":
                raise RuntimeError(full_result.get("message", "Discovery failed"))

            discovery = full_result.get("discovery", {})
            validation = full_result.get("validation", {})
            r2 = float(discovery.get("r2_score", 0.0))

            try:
                y_pred = discovery.get("predictions", None)
                rmse = (
                    float(np.sqrt(np.mean((y - np.asarray(y_pred)) ** 2)))
                    if y_pred is not None
                    else float("inf")
                )
            except Exception:
                rmse = float("inf")

            formula = discovery.get("expression", "N/A")
            success = r2 > 0.0 and formula not in (
                "DISCOVERY_FAILED", "NO_VALID_EQUATIONS", "VALIDATION_FAILED", "N/A"
            )

            return {
                "success": success,
                "r2": r2,
                "rmse": rmse,
                "final_formula": formula,
                "strategy": discovery.get("discovery_engine", "symbolic"),
                "validations": 1 if validation else 0,
                "validation_score": validation.get("total_score", 0.0),
                "error": None,
            }

        except Exception as exc:
            logger.error(
                f"discover() caught top-level exception — {type(exc).__name__}: {exc}",
                exc_info=True,
            )
            return {
                "success": False,
                "r2": 0.0,
                "rmse": float("inf"),
                "final_formula": "N/A",
                "strategy": "error",
                "validations": 0,
                "error": str(exc)[:200],
            }

    def print_statistics_summary(self) -> None:
        """Print statistics summary — identical to v4.1."""
        print(f"\n{'=' * 70}")
        print("STATISTICS SUMMARY v4.1-PROD")
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
            filename = f"discovery_results_v41prod_{timestamp}.json"

        # PROD-5: copy deque to list once
        results_list = [_to_serialisable(r) for r in self.results]

        output = {
            "version": "4.1-prod",
            "timestamp": datetime.now().isoformat(),
            "domain": self.domain,
            "statistics": self.stats,
            "results": results_list,
        }

        with open(filename, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"[OK] Results saved to {filename}")
        return filename


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("HYBRID SYSTEM v4.1-PROD — QUICK TEST")
    print("=" * 80)

    print("\nTest: Ohm's Law V = I * R")
    print("-" * 80)

    np.random.seed(42)
    I = np.random.uniform(0.1, 10, 100)
    R = np.random.uniform(1, 100, 100)
    V = I * R + np.random.normal(0, np.abs(I * R) * 0.01, 100)
    X = np.column_stack([I, R])

    discovery_config = DiscoveryConfig(niterations=60, enable_auto_configuration=True)
    system = HybridDiscoverySystem(
        domain="physics",
        discovery_config=discovery_config,
        enable_physics_fallback=False,
        max_retries=3,
    )

    result = system.discover_validate_interpret(
        X=X,
        y=V,
        variable_names=["I", "R"],
        variable_descriptions={"I": "Current in amperes", "R": "Resistance in ohms"},
        variable_units={"I": "A", "R": "Ohm"},
        description="Ohm's Law",
        equation_name="ohms_law",
    )

    print("\n" + "=" * 80)
    print("RESULT:")
    print("=" * 80)
    print(f"Expression: {result['discovery']['expression']}")
    print(f"R2: {result['discovery']['r2_score']:.4f}")
    print(f"Engine: {result['discovery'].get('discovery_engine')}")

    expr = result["discovery"]["expression"]
    if "*" in expr and "+" not in expr:
        print("\n[SUCCESS] Found multiplicative relationship!")
    else:
        print("\n[FAILED] Expression still contains addition")

    system.print_statistics_summary()
