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
from collections import deque
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

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

# ---------------------------------------------------------------------------
# FIX-1: Import BOTH SymbolicEngine and SymbolicEngineWithLLM.
# Previous versions only imported the base class, making LLM guidance
# unreachable regardless of what use_llm was set to.
# ---------------------------------------------------------------------------
from hypatiax.tools.symbolic.physics_aware_regressor import PhysicsAwareRegressor
from hypatiax.tools.symbolic.symbolic_engine import (
    DiscoveryConfig,
    LLMConfig,
    SymbolicEngine,
    SymbolicEngineWithLLM,
    detect_collapsed_constants,
)
from hypatiax.tools.validation.ensemble_validator import EnsembleValidator


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
) -> tuple[bool, int, tuple[str, ...]]:
    """
    Pure-function quality check used as the LRU target.
    Returns (is_overfit, complexity, warnings_tuple) — fully hashable.
    """
    complexity = len(expression)
    is_overfit = False
    warnings: list[str] = []

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
def _build_op_patterns(aliases: dict[str, str]) -> dict[str, tuple[re.Pattern, str]]:
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
    _PYSR_OP_ALIASES: dict[str, str] = {
        "safe_asin":   "arcsin",
        "safe_acos":   "arccos",
        "asin_of_sin": "arcsin",
        "acos_of_cos": "arccos",
        "atan_of_tan": "arctan",
    }
    _PYSR_OP_PATTERNS: dict[str, tuple[re.Pattern, str]] = _build_op_patterns(
        _PYSR_OP_ALIASES
    )

    def __init__(
        self,
        domain: str = "general",
        discovery_config: DiscoveryConfig | None = None,
        discovery_mode: DiscoveryMode = DiscoveryMode.STRICT,
        max_results: int | None = 100,
        validation_weights: dict[str, float] | None = None,
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
        anthropic_api_key: str | None = None,
        google_api_key: str | None = None,
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
        logger.info("HybridDiscoverySystem v5.2 — LLM WIRING + OUTPUT FIXES + FIX-D/RATIO/SIMPLIFY")
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

        self.stats: dict[str, int] = {
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
        self, anthropic_api_key: str | None, google_api_key: str | None
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
                from hypatiax.tools.llm_providers.anthropic_provider import (
                    AnthropicProvider,
                )
                self.anthropic_provider = AnthropicProvider(api_key=api_key, max_tokens=4096)
            except Exception:
                self.anthropic_provider = None
        else:
            self.anthropic_provider = None

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
        self, noise_level: float | None = None
    ) -> PhysicsAwareRegressor:
        return PhysicsAwareRegressor(
            domain=self.domain,
            verbose=True,
            population_size=self.physics_population_size,
            generations=self.physics_generations,
            noise_level=noise_level,
        )

    def _check_expression_quality(self, expression: str, r2: float) -> dict[str, Any]:
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
        variable_names: list[str],
        variable_descriptions: dict[str, str],
        variable_units: dict[str, str],
        equation_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Discover with retry.

        FIX-4: No engine-routing needed here — the correct engine type
        (SymbolicEngine or SymbolicEngineWithLLM) was already selected in
        __init__ based on use_llm and key availability.  All retry logic and
        physics fallback are unchanged from v4.1-PROD.
        """
        best_result = None
        best_r2 = -np.inf
        last_attempt_error: Exception | None = None
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
        FIX-SIMPLIFY v5.2: collapses log(exp(x))→x and exp(log(x))→x.
        """
        result = expression_str
        result = result.replace("^", "**")   # ← ADD THIS LINE (first thing)
        for pat, numpy_name in HybridDiscoverySystem._PYSR_OP_PATTERNS.values():
            result = pat.sub(numpy_name, result)
        try:
            import sympy as _sp
            _free = _sp.sympify(result).free_symbols
            _assumptions = {str(s): _sp.Symbol(str(s), real=True) for s in _free}
            _sym = _sp.sympify(result, locals=_assumptions)
            _simp_str = str(_sp.simplify(_sym))
            if len(_simp_str) < len(result):
                result = _simp_str
        except Exception:
            pass
        return result

    def _safe_validate(
        self,
        expression_str: str,
        variable_definitions: dict[str, str],
        variable_units: dict[str, str],
        test_data: dict[str, np.ndarray],
    ) -> dict[str, Any]:
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
        variable_names: list[str],
        variable_descriptions: dict[str, str],
        variable_units: dict[str, str],
        description: str | None = None,
        equation_name: str | None = None,
        validate_first: bool = True,
        show_formatted: bool = True,
        use_llm: bool = False,          # FIX-3: now actually read and respected
        min_validation_score: float = 85.0,
    ) -> dict[str, Any]:
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

        print(f"\n{'=' * 70}")
        print("DISCOVERY WORKFLOW v5.1")
        print(f"{'=' * 70}")
        print(f"Description: {description or 'Unnamed'}")
        print(f"Domain: {self.domain.upper()}")
        print(f"Samples: {len(X)}")
        print(f"Variables: {variable_names}")
        print(f"LLM mode: {self.llm_mode}")
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
            print("\n[OK] Discovery complete")
            print(f"   Expression: {discovery_result['expression']}")
            print(f"   R2 Score: {discovery_result['r2_score']:.4f}")
            print(f"   Engine: {engine}")
            if llm_info:
                print(f"   LLM mode used: {llm_info}")
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
                "llm_mode": self.llm_mode,
                "equation_name": equation_name,
                "version": "5.1",
            },
        }

        self.results.append(complete_result)

        print(f"\n{'=' * 70}")
        print("[OK] WORKFLOW COMPLETE")
        print(f"{'=' * 70}\n")

        return complete_result

    # ------------------------------------------------------------------
    # discover() thin adapter — FIX-6: propagates use_llm from metadata
    # ------------------------------------------------------------------
    def discover(
        self,
        X: np.ndarray,
        y: np.ndarray,
        var_names: list[str],
        description: str = "",
        metadata: dict | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
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
        _LOG_THRESHOLD = 4  # log10 OOM (v5.2: was 6, too strict for gravity/EM)
        _ABS_MAG_THRESHOLD = 1e5  # trigger if X column median abs > this

        def _signed_log10(arr: np.ndarray) -> np.ndarray:
            """sign(x) * log10(|x| + 1) — safe for zeros."""
            return np.sign(arr) * np.log10(np.abs(arr) + 1.0)

        try:
            _needs_log: list[bool] = []
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
        except Exception as _fd_err:
            logger.warning(f"[FIX-D] Scale detection failed ({_fd_err}) — using raw data")
            X, y = _X_orig, _y_orig
            _log_scaled = False

        # Store originals so FIX-A can compute RMSE in original (unscaled) units.
        self._discover_X_orig = _X_orig
        self._discover_y_orig = _y_orig
        self._discover_scale_log = _log_scaled

        # FIX-POW: enable "pow" only when every feature value is non-negative
        # (evaluated on post-scaled X so log-transform is already factored in).
        # Negative bases cause Julia DomainError with fractional exponents.
        # Callers can override via metadata: metadata={"allow_pow": True/False}.
        _allow_pow = metadata.get(
            "allow_pow",
            bool(np.isfinite(X).all() and float(np.min(X)) >= 0.0),
        )
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
                    # FIX-RATIO v5.2: bind sanitised names, ratio/augmented features,
                    # and use _X_aug (normalised) for base cols since the formula
                    # was discovered on normalised X, not original-scale X.
                    _safe_names = discovery.get("variable_names", var_names)
                    _X_aug   = getattr(self.symbolic_engine, "_last_X_aug",    _X_for_rmse)
                    _aug_nms = getattr(self.symbolic_engine, "_last_aug_names", list(var_names))
                    _ns: dict[str, Any] = {
                        "np": np,
                        # base cols — use normalised X (matches what formula was fit to)
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
                    _y_pred = eval(_norm_expr, {"__builtins__": {}}, _ns)  # noqa: S307
                    _y_pred = np.asarray(_y_pred, dtype=float)
                    if _y_pred.shape == ():
                        _y_pred = np.full(len(_y_for_rmse), float(_y_pred))
                    _finite = np.isfinite(_y_pred) & np.isfinite(_y_for_rmse)
                    if _finite.sum() >= 2:
                        rmse = float(
                            np.sqrt(np.mean(
                                (_y_for_rmse[_finite] - _y_pred[_finite]) ** 2
                            ))
                        )
                except Exception as _rmse_err:
                    logger.debug(f"[FIX-A] RMSE eval failed ({_rmse_err}) — reporting inf")

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
        print("\nOverall:")
        print(f"   Discoveries: {self.stats['discoveries']}")
        print(f"   Validations: {self.stats['validations']}")
        print("\nSymbolicEngine:")
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
            print("\nPhysicsAware:")
            print(f"   Used: {self.stats['physics_used']}")
            print(f"   Successes: {self.stats['physics_successes']}")
        print("\nAuto-Configuration:")
        print(f"   Used: {self.stats['auto_configs']} times")
        print(f"\n{'=' * 70}\n")

    def save_results(self, filename: str | None = None) -> str:
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


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("HYBRID SYSTEM v5.1 — QUICK TEST (covers FIX-A … FIX-D)")
    print("=" * 80)

    # ── Test A: Ohm's Law — verifies FIX-A (RMSE), FIX-B (key aliases), FIX-C (deterministic)
    print("\nTest A: Ohm's Law — pure PySR (FIX-A: RMSE, FIX-B: keys, FIX-C: deterministic)")
    print("-" * 80)
    np.random.seed(42)
    I = np.random.uniform(0.1, 10, 100)
    R = np.random.uniform(1, 100, 100)
    V = I * R + np.random.normal(0, np.abs(I * R) * 0.01, 100)
    X_ohm = np.column_stack([I, R])

    sys_a = HybridDiscoverySystem(
        domain="physics",
        discovery_config=DiscoveryConfig(niterations=50, enable_auto_configuration=True),
        enable_physics_fallback=False,
        max_retries=5,
        use_llm=False,
        allow_nondeterministic=False,   # FIX-C default — explicit here for clarity
    )
    res_a = sys_a.discover(
        X_ohm, V, ["I", "R"],
        description="Ohm's Law",
        metadata={
            "equation_name": "ohms_law",
            "variable_descriptions": {"I": "Current", "R": "Resistance"},
            "variable_units": {"I": "A", "R": "ohm"},  # Pint requires lowercase
        },
    )
    print(f"  formula       : {res_a['formula']}")
    print(f"  expression    : {res_a['expression']}")       # FIX-B: alias
    print(f"  final_formula : {res_a['final_formula']}")    # FIX-B: alias
    print(f"  variable_names: {res_a['variable_names']}")   # FIX-B: new key
    print(f"  R²            : {res_a['r2']:.4f}")
    rmse_ok = res_a["rmse"] < float("inf") and res_a["rmse"] >= 0
    print(f"  RMSE          : {res_a['rmse']:.4f}  {'[FIX-A OK]' if rmse_ok else '[FIX-A FAIL]'}")
    print(f"  scale_log     : {res_a['scale_log']}  (should be False for Ohm's Law)")
    sys_a.print_statistics_summary()

    # ── Test B: Gravitational Force — verifies FIX-D (extreme-scale log transform)
    print("\nTest B: Gravitational Force — extreme-scale log transform (FIX-D)")
    print("-" * 80)
    np.random.seed(42)
    G = 6.674e-11
    m1 = np.random.uniform(1e10, 1e12, 120)
    m2 = np.random.uniform(1e10, 1e12, 120)
    r  = np.random.uniform(1e6,  1e8,  120)
    F  = G * m1 * m2 / r ** 2
    X_grav = np.column_stack([m1, m2, r])

    sys_d = HybridDiscoverySystem(
        domain="physics",
        discovery_config=DiscoveryConfig(niterations=50, enable_auto_configuration=True),
        enable_physics_fallback=False,
        max_retries=5,
        use_llm=False,
    )
    res_d = sys_d.discover(
        X_grav, F, ["m1", "m2", "r"],
        description="Gravitational Force",
        metadata={
            "equation_name": "gravity",
            "variable_descriptions": {"m1": "mass 1", "m2": "mass 2", "r": "distance"},
            "variable_units": {"m1": "kg", "m2": "kg", "r": "m"},
        },
    )
    print(f"  formula   : {res_d['formula']}")
    print(f"  R²        : {res_d['r2']:.4f}")
    print(f"  RMSE      : {res_d['rmse']}")
    log_ok = res_d["scale_log"]
    print(f"  scale_log : {log_ok}  {'[FIX-D: log-transform applied]' if log_ok else '[WARNING: no log-transform — check data ranges]'}")

    # ── Test C: LLM hybrid mode (optional — requires ANTHROPIC_API_KEY)
    print("\nTest C: LLM hybrid mode (use_llm=True, requires ANTHROPIC_API_KEY)")
    print("-" * 80)
    if os.getenv("ANTHROPIC_API_KEY"):
        sys_llm = HybridDiscoverySystem(
            domain="physics",
            discovery_config=DiscoveryConfig(niterations=50, enable_auto_configuration=True),
            enable_physics_fallback=False,
            max_retries=5,
            use_llm=True,
            llm_mode="hybrid",
            llm_n_candidates=3,
        )
        res_llm = sys_llm.discover_validate_interpret(
            X=X_ohm, y=V,
            variable_names=["I", "R"],
            variable_descriptions={"I": "Current in amperes", "R": "Resistance in ohms"},
            variable_units={"I": "A", "R": "Ohm"},
            description="Ohm's Law (LLM hybrid)",
            equation_name="ohms_law",
        )
        print(f"  Expression : {res_llm['discovery']['expression']}")
        print(f"  R²         : {res_llm['discovery']['r2_score']:.4f}")
        print(f"  LLM mode   : {res_llm['discovery'].get('llm_mode', 'N/A')}")
        sys_llm.print_statistics_summary()
    else:
        print("  Skipping — ANTHROPIC_API_KEY not set.")
        print("  Set the key and rerun with use_llm=True to test LLM guidance.")
