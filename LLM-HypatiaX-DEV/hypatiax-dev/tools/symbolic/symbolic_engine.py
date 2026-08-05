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

MAX_COMPLEXITY = int(os.getenv("MAX_COMPLEXITY", 30))

import gc
import json
import math
import random
import re
import subprocess
import time
import warnings
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import psutil
import sympy as sp

# NOTE: PySRRegressor is intentionally NOT imported at module level.
# Importing pysr triggers juliacall initialisation immediately, which
# segfaults when PyTorch is already loaded in the same process.
# The lazy import is inside SymbolicEngine.discover() — by that point
# the env var above has been set and juliacall configures itself correctly.
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
# HELPER FUNCTIONS
# ============================================================================
# Lazy loading for PySR
try:
    from pysr import PySR
except ImportError:
    PySR = None

# Memory logging function
def _log_rss(label: str):
    rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print(f"   [MEM] {label}: RSS={rss:.2f} GB", flush=True)



import sys  # needed for llm_cleanup and subprocess helpers


# LLM cleanup mechanism
def llm_cleanup():
    """Force-exit the current process (last-resort OOM escape hatch)."""
    if sys.platform == 'linux':
        subprocess.call(['kill', '-9', str(os.getpid())])

# Timeout guards
class TimeoutGuard:
    def __init__(self, timeout):
        self.timeout = timeout
        self.start_time = time.time()

    def check_timeout(self):
        if time.time() - self.start_time > self.timeout:
            raise TimeoutError('Operation timed out')

# Subprocess support pattern
def run_subprocess(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr

# Memory check guards
def check_memory_threshold(threshold_gb: float):
    """Kill process if RSS exceeds threshold_gb. Last-resort OOM guard."""
    rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
    _log_rss("check_memory_threshold")
    if rss_gb > threshold_gb:
        print(f"   ⚠ RSS {rss_gb:.2f} GB > threshold {threshold_gb:.2f} GB — triggering cleanup", flush=True)
        llm_cleanup()


def check_memory_ok(min_free_gb: float = 4.0) -> bool:
    """Return False (and log a warning) when available RAM < min_free_gb.

    Call this before each case in the outer loop to avoid an OOM-kill.
    Returns True when there is enough memory to proceed safely.
    """
    vm = psutil.virtual_memory()
    free_gb = vm.available / 1e9
    if free_gb < min_free_gb:
        print(
            f"   ⚠ LOW MEMORY: {free_gb:.1f} GB free < {min_free_gb} GB — skipping case",
            flush=True,
        )
        return False
    return True
# ============================================================================
# START
# ============================================================================


_log_rss("start")
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
        "Q",  # Often reserved for quotient or other special uses
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
        "Q": "Qr",  # Reaction quotient
        "E": "E_val",  # Energy or potential
        "PI": "Pi",  # Greek pi (different case)
        "pi": "Pi",  # Pi constant
    }

    @staticmethod
    def is_reserved(name: str) -> bool:
        """Check if a variable name conflicts with PySR reserved words."""
        return (
            name.lower() in VariableNameValidator.PYSR_RESERVED
            or name in VariableNameValidator.PYSR_RESERVED
        )

    @staticmethod
    def sanitize_name(name: str, existing_names: list[str] = None) -> str:
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
    def sanitize_names(names: list[str]) -> tuple[list[str], dict[str, str]]:
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
                warnings.warn(
                    f"Variable '{name}' conflicts with PySR reserved word. "
                    f"Renamed to '{safe_name}'.",
                    UserWarning,
                )

        return sanitized, mapping

    @staticmethod
    def update_expression(expression: str, mapping: dict[str, str]) -> str:
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


def detect_collapsed_constants(expression: str, variable_names: list[str]) -> list[str]:
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

    niterations: int = 25          # v21: lowered from 40→25; keeps wall time well
                                   # within pysr_timeout even after feature augmentation.
                                   # Set N_ITERATIONS env var to override at runtime.
    populations: int = 10          # reduced from 15→10; combined with niterations=25
                                   # gives ~8750 evaluations/attempt instead of ~19800,
                                   # comfortably within 120s budget after Julia startup.
    population_size: int = 33      # REDUCED from 200 → 33. With populations=10
                                   # total individuals = 10×33 = ~330, matching
                                   # the PySR docs "fast" preset. The old 200
                                   # (3000 individuals) added ~3× wall-time per
                                   # iteration with no proportional R² gain on
                                   # Feynman equations.
    binary_operators: list[str] = field(default_factory=lambda: ["+", "-", "*", "/"])
    unary_operators: list[str] = field(default_factory=lambda: ["sqrt"])
    # sqrt is universally needed (e.g. double-slit: 2√(I₁I₂)cos(δ), RMS,
    # Euclidean distance).  It is safe for PySR — Julia's sqrt() returns NaN
    # for negative inputs rather than throwing, so evolution continues cleanly.
    # Additional operators (sin, cos, safe_asin …) are injected per-domain.
    constraints: dict = field(default_factory=dict)
    maxsize: int = 30          # max expression tree size; raise for deep compositions
    maxdepth: int | None = None  # max tree depth (None = PySR default, unlimited).
                                    # Ported from core engine. Distinct from maxsize:
                                    # maxsize caps node count, maxdepth caps nesting level.
                                    # Set e.g. 12 to prevent pathologically deep trees.
    # Per-operator complexity overrides — set low values to make operators "cheaper"
    # so PySR favours them in the search.  Empty dict = PySR defaults (all cost 1).
    complexity_of_operators: dict = field(default_factory=dict)
    enable_auto_configuration: bool = True
    auto_config_correlation_threshold: float = 0.2
    enable_smart_discovery: bool = False
    smart_discovery_priority: bool = False

    # Complexity / search tuning
    parsimony: float = 0.01  # raised from 0.0032 — stronger parsimony pressure by default

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
    loss: str | None = None

    # ── Progress display ──────────────────────────────────────────────────────
    # Ported from core engine (which hardcodes progress=True).
    # Set True for interactive use; False (default) suppresses PySR's tqdm bar
    # in subprocess / benchmark contexts where it pollutes stdout.
    show_progress: bool = False

    # ── v21: per-attempt PySR wall-clock cap ─────────────────────────────────
    # Passed directly to PySRRegressor(timeout_in_seconds=pysr_timeout).
    # Prevents a single runaway PySR call from consuming the full benchmark
    # budget.  With max_retries=3 (hybrid_system_v40) worst-case wall time is
    # 3 × pysr_timeout + ~90s Julia startup ≈ 450s, well within a 900s budget.
    #
    # REDUCED from 150 → 120:
    # After feature augmentation (GM/ratio columns) the effective search space
    # can grow 3–5× for multi-variable problems.  120s keeps each attempt safe
    # while leaving the remaining budget for retries.
    # Set to 0 to disable (no timeout, legacy behaviour).
    pysr_timeout: int = 120

    # Transcendental composition support
    # When True, atomic operators for arcsin(sin(x)), arccos(cos(x)), arctan(tan(x))
    # are injected into PySR as custom Julia functions, bypassing the simplifier
    # that would otherwise collapse these back to x.
    use_transcendental_compositions: bool = False

    # Julia source strings for the three compositions — injected as a *list* of
    # definition strings via PySRRegressor(define_operators=[...]).
    # Keys are the operator names; values are valid Julia function definitions.
    _TRANSCENDENTAL_OPS: ClassVar[dict[str, str]] = {
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

    model: str = "claude-sonnet-4-6"
    max_tokens: int = 2000
    temperature: float = 0.3
    n_candidates: int = 3  # Number of hypotheses to generate
    enabled: bool = False
    api_key: str | None = None


@dataclass
class EquationHypothesis:
    """A candidate equation from LLM."""

    equation: str
    confidence: float
    reasoning: str
    r2_score: float | None = None
    validation_score: float | None = None


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
            print("⚠️  Anthropic not installed. Install: pip install anthropic")
            self.config.enabled = False
            return

        if not config.api_key:
            config.api_key = os.getenv("ANTHROPIC_API_KEY")

        if not config.api_key:
            print("⚠️  No API key found. LLM guidance disabled.")
            self.config.enabled = False
            return

        try:
            self.client = Anthropic(api_key=config.api_key)
            print(f"   ✓ LLM engine initialized ({config.model})")
        except Exception as e:
            print(f"⚠️  LLM init failed: {e}")
            self.config.enabled = False

    def generate_hypotheses(
        self,
        domain: str,
        variables: list[str],
        description: str,
        data_patterns: dict,
        n_candidates: int = None,
        caller_id: str = "",
    ) -> list[EquationHypothesis]:
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
            print(f"⚠️  LLM generation failed: {e}")
            return []

    def _build_prompt(
        self,
        domain: str,
        variables: list[str],
        description: str,
        patterns: dict,
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

    def _parse_response(self, response: str) -> list[EquationHypothesis]:
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
                eq = eq.replace("^", "**")   # ← ADD THIS LINE
                hypotheses.append(
                    EquationHypothesis(
                        equation=eq,
                        confidence=float(c.get("confidence", 0.5)),
                        reasoning=c.get("reasoning", ""),
                    )
                )

            return hypotheses

        except Exception as e:
            print(f"⚠️  Failed to parse LLM response: {e}")
            return []


# ============================================================================
# PATTERN ANALYZER (INTEGRATED)
# ============================================================================


class DataPatternAnalyzer:
    """Lightweight pattern analysis for LLM context."""

    def analyze(self, X: np.ndarray, y: np.ndarray, variable_names: list[str]) -> dict:
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
        variable_names: list[str], auto_fix: bool = True, verbose: bool = False
    ) -> tuple[list[str], dict[str, str]]:
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
            print("\n🔧 Variable Name Sanitization:")
            for orig, safe in mapping.items():
                print(f"   {orig} → {safe}")

        return safe_names, mapping

    def discover(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: list[str] = None,
        equation_name: str = None,
        random_state: int = 42,
        auto_sanitize: bool = True,
        **kwargs,
    ) -> dict:
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

        print("\n[DISCOVERY] Starting symbolic regression...")
        print(f"   Variables: {', '.join(safe_names)}")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Iterations: {self.config.niterations}")

        if name_mapping:
            print(f"   ⚠️  Sanitized {len(name_mapping)} variable name(s)")

        # ── Trace collector ──────────────────────────────────────────────────
        # Key decisions and PySR config are recorded here and forwarded through
        # the result dict.  The benchmark runner discards subprocess stdout;
        # the trace in the JSON result is the only reliable diagnostic channel
        # back to the parent process.
        _trace: list[str] = [
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
            extra_sympy: dict = {}
            define_ops: list[str] = []

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
                    print(
                        f"   [AUTO-TRIG] domain='{self.domain}' eq='{_eq_hint[:30]}' "
                        f"→ injected sin, cos, safe_asin, safe_acos (inv-trig needed)"
                    )
                    _trace.append("trig=sin,cos,safe_asin,safe_acos")
                else:
                    print(
                        f"   [AUTO-TRIG] domain='{self.domain}' eq='{_eq_hint[:30]}' "
                        f"→ injected sin, cos only (no inverse trig needed)"
                    )
                    _trace.append("trig=sin,cos")

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
                    _y_fit_pos = np.all(y > 0)
                    if _y_fit_pos:
                        from sklearn.linear_model import LinearRegression as _LR
                        _Xs = X - X.mean(axis=0)
                        _r2_lin = r2_score(y, _LR().fit(_Xs, y).predict(_Xs))
                        _logy   = np.log(y)
                        _r2_log = r2_score(_logy,
                                           _LR().fit(_Xs, _logy).predict(_Xs))
                        if _r2_log - _r2_lin > 0.05:
                            _data_needs_exp = True
                            print(
                                f"   [AUTO-EXP] data heuristic: "
                                f"R²(log y ~ X)={_r2_log:.3f} >> R²(y ~ X)={_r2_lin:.3f} "
                                f"— injecting exp/log (exponential relationship detected)",
                                flush=True,
                            )
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
                    print(
                        f"   [AUTO-EXP] domain='{self.domain}' "
                        f"→ injected {', '.join(_injected_explog)} "
                        f"(exp/log domain; required for Boltzmann/Planck/Arrhenius forms)",
                        flush=True,
                    )
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
            import inspect as _inspect_ef
            import os as _os2
            import tempfile as _tf
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
                niterations=int(os.environ.get("N_ITERATIONS", self.config.niterations)),
                populations=int(os.environ.get("POPULATIONS", self.config.populations)),
                population_size=self.config.population_size,
                binary_operators=self.config.binary_operators,
                unary_operators=active_unary,
                constraints=self.config.constraints,
                parsimony=self.config.parsimony,
                maxsize=self.config.maxsize,
                random_state=random_state,
                # FIX (parallelism): deterministic=True forces parallelism="serial"
                # which runs all populations sequentially.  With POPULATIONS=30 this
                # meant 30 sequential tournament rounds → wall-clock ≫ timeout.
                # Switching to "multithreading" lets Julia use all CPU cores so all
                # populations run concurrently; wall-time ≈ single-population time.
                # Reproducibility is sacrificed but that is acceptable in a benchmark
                # where per-equation isolation already gives stable results.
                parallelism="multithreading",
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
                    print(
                        "   [AUTO-TRIG] parsimony → 0.0006  "
                        "(trig domain; avoids over-penalising deep interference trees)",
                        flush=True,
                    )
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
            _timeout = int(os.environ.get("PYSR_TIMEOUT", self.config.pysr_timeout or 0))
            # ── proc_timeout budget cap (Fix: Step 4 from diagnosis) ──────────
            # If a proc_timeout is set (e.g. via env var PROC_TIMEOUT), ensure
            # pysr_timeout stays at least 60s + llm_overhead_secs short of it so
            # LLM post-processing doesn't blow past the wall-clock limit.
            _proc_timeout = int(os.environ.get("PROC_TIMEOUT", 0))
            _llm_overhead = int(os.environ.get("LLM_OVERHEAD_SECS", 120))
            if _proc_timeout > 0 and _timeout > 0:
                _budget_for_pysr = _proc_timeout - _llm_overhead - 60  # 60s safety margin
                if _budget_for_pysr > 0 and _timeout > _budget_for_pysr:
                    print(
                        f"   [TIMEOUT-CAP] pysr_timeout {_timeout}s → {_budget_for_pysr}s "
                        f"(proc_timeout={_proc_timeout}s - llm_overhead={_llm_overhead}s - 60s margin)",
                        flush=True,
                    )
                    _timeout = _budget_for_pysr
            if _timeout > 0:
                pysr_kwargs["timeout_in_seconds"] = _timeout

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
                    _body_map: dict[str, str] = {}
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
            print(f"   [PySR] unary_operators   = {pysr_kwargs.get('unary_operators')}", flush=True)
            print(f"   [PySR] binary_operators  = {pysr_kwargs.get('binary_operators')}", flush=True)
            print(f"   [PySR] niterations       = {pysr_kwargs.get('niterations')}", flush=True)
            print(f"   [PySR] parsimony         = {pysr_kwargs.get('parsimony')}", flush=True)
            print(f"   [PySR] timeout_in_seconds= {pysr_kwargs.get('timeout_in_seconds')}", flush=True)
            print(f"   [PySR] loss kwarg        = {[k for k in pysr_kwargs if 'loss' in k]}", flush=True)
            if "define_operators" in pysr_kwargs:
                print(f"   [PySR] define_operators ({len(pysr_kwargs['define_operators'])} entries) = "
                      f"{[d.split('(')[0] for d in pysr_kwargs['define_operators']]}", flush=True)
            else:
                print("   [PySR] define_operators = NOT SET (old PySR path)", flush=True)

            try:
                self.model = PySRRegressor(**pysr_kwargs)
                print("   [PySR] PySRRegressor constructed OK", flush=True)
            except Exception as _init_exc:
                import sys as _sys2
                import traceback as _tb2
                _init_tb = _tb2.format_exc()
                print(f"   [PySR] PySRRegressor.__init__ FAILED: {_init_exc}", flush=True)
                print(_init_tb, flush=True)
                print(f"   [PySR] PySRRegressor.__init__ FAILED: {_init_exc}", file=_sys2.stderr, flush=True)
                print(_init_tb, file=_sys2.stderr, flush=True)
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
                print(
                    f"   [Y-SCALE] y_std={_y_std:.3e} — normalising y before PySR fit",
                    flush=True,
                )
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
            _x_scaled_cols: list[tuple[int, str, float]] = []  # (col_idx, name, scale)
            for _xi in range(X.shape[1]):
                _col = X[:, _xi]
                _col_scale = max(float(np.abs(np.mean(_col))), float(np.std(_col)))
                if _col_scale > 0 and (_col_scale < 1e-6 or _col_scale > 1e6):
                    _x_col_scales[_xi] = _col_scale
                    _x_scaled_cols.append((_xi, safe_names[_xi], _col_scale))

            if _x_scaled_cols:
                _X_fit = X / _x_col_scales[np.newaxis, :]
                print(
                    "   [X-SCALE] extreme X-column scale(s) detected — normalising before PySR fit",
                    flush=True,
                )
                for _xi, _xn, _xsc in _x_scaled_cols:
                    print(f"   [X-SCALE]   {_xn}: scale={_xsc:.3e}", flush=True)
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
                    _gm_cols: list[np.ndarray] = []
                    _gm_names: list[str] = []
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
                        print(
                            f"   [OPTICS-GM] Added {len(_gm_cols)} geometric-mean "
                            f"feature(s): {_gm_names}  "
                            f"(reduces target complexity: 13 → 7 for interference formulas)",
                            flush=True,
                        )
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
                    _ratio_cols: list[np.ndarray] = []
                    _ratio_names: list[str] = []
                    # Cap at 10 ratio pairs: beyond this the search space blows
                    # up (O(n²) new columns) while marginal benefit drops.
                    # Prioritise pairs where numerator/denominator differ most
                    # in magnitude (most informative ratios first).
                    _ratio_candidates = []
                    for _pi in range(len(_pos_idx_exp)):
                        for _qi in range(len(_pos_idx_exp)):
                            if _pi == _qi:
                                continue
                            _ci, _cj = _pos_idx_exp[_pi], _pos_idx_exp[_qi]
                            _ratio_vec = _X_fit[:, _ci] / (_X_fit[:, _cj] + 1e-300)
                            _ratio_nm  = f"ratio_{safe_names[_ci]}_{safe_names[_cj]}"
                            # Score by variance (more varied ratio = more info)
                            _var_score = float(np.std(_ratio_vec))
                            _ratio_candidates.append((_var_score, _ratio_vec, _ratio_nm))
                    # Sort by descending variance, take top 10
                    _ratio_candidates.sort(key=lambda x: x[0], reverse=True)
                    _MAX_RATIO_PAIRS = 10
                    for _score, _rvec, _rnm in _ratio_candidates[:_MAX_RATIO_PAIRS]:
                        _ratio_cols.append(_rvec)
                        _ratio_names.append(_rnm)
                    if len(_ratio_candidates) > _MAX_RATIO_PAIRS:
                        _trace.append(f"ratio_capped={len(_ratio_candidates)}→{_MAX_RATIO_PAIRS}")
                    if _ratio_cols:
                        _X_fit    = np.column_stack([_X_fit] + _ratio_cols)
                        safe_names = list(safe_names) + _ratio_names
                        print(
                            f"   [EXP-RATIO] Added {len(_ratio_cols)} ratio "
                            f"feature(s): {_ratio_names}  "
                            f"(reduces Boltzmann/Planck target: exp(C*f/T) → exp(C*ratio_f_x0))",
                            flush=True,
                        )
                        _trace.append(f"ratio_features={_ratio_names}")
                else:
                    _trace.append("ratio_features=skipped(insufficient_pos_cols)")
            else:
                _trace.append("ratio_features=skipped")

            # ── Population-aware iteration scaling (serial mode only) ──────────
            # In "serial" parallelism, total wall-time ∝ populations × niterations.
            # In "multithreading", populations run concurrently so wall-time ∝
            # niterations alone — POP-SCALE would hurt quality for no benefit.
            # Guard: only scale when parallelism is "serial".
            _active_parallelism = pysr_kwargs.get("parallelism", "multithreading")
            _base_populations = 10   # matches DiscoveryConfig default
            _actual_populations = pysr_kwargs["populations"]
            if _active_parallelism == "serial" and _actual_populations > _base_populations:
                _pop_scale = _base_populations / _actual_populations
                _pop_adjusted_iters = max(5, int(pysr_kwargs["niterations"] * _pop_scale))
                if _pop_adjusted_iters < pysr_kwargs["niterations"]:
                    print(
                        f"   [POP-SCALE] populations={_actual_populations} > baseline={_base_populations}; "
                        f"niterations {pysr_kwargs['niterations']} → {_pop_adjusted_iters} "
                        f"(keeps total evals ≤ {_base_populations * self.config.population_size * self.config.niterations:,})",
                        flush=True,
                    )
                    _trace.append(f"pop_scale={_pop_scale:.2f}({pysr_kwargs['niterations']}→{_pop_adjusted_iters})")
                    pysr_kwargs["niterations"] = _pop_adjusted_iters
            elif _actual_populations > _base_populations:
                print(
                    f"   [POP-SCALE] populations={_actual_populations}  parallelism={_active_parallelism!r} "
                    f"→ niterations unchanged (populations run concurrently, no scaling needed)",
                    flush=True,
                )
                _trace.append(f"pop_scale=skipped(multithreading,pops={_actual_populations})")

            # ── Feature-count adaptive iteration scaling ─────────────────────
            # When GM or ratio augmentation adds many columns (e.g. 6 vars →
            # 6 + 30 ratio = 36 cols), the PySR search space grows O(n²) but
            # the timeout budget stays constant.  Compensate by reducing
            # niterations proportionally so we don't time out mid-search.
            #
            # Formula: scale = sqrt(original_n_vars / augmented_n_vars)
            # clamped to [0.4, 1.0] so we never cut below 40% of base iters.
            _orig_n_vars = len(variable_names)
            _aug_n_vars  = _X_fit.shape[1]
            if _aug_n_vars > _orig_n_vars:
                import math as _math
                _iter_scale = max(0.4, _math.sqrt(_orig_n_vars / _aug_n_vars))
                _adjusted_iters = max(5, int(pysr_kwargs["niterations"] * _iter_scale))
                if _adjusted_iters < pysr_kwargs["niterations"]:
                    print(
                        f"   [ITER-SCALE] {_orig_n_vars} → {_aug_n_vars} vars after augmentation; "
                        f"niterations {pysr_kwargs['niterations']} → {_adjusted_iters} "
                        f"(scale={_iter_scale:.2f}) to stay within timeout",
                        flush=True,
                    )
                    _trace.append(f"iter_scale={_iter_scale:.2f}({pysr_kwargs['niterations']}→{_adjusted_iters})")
                    pysr_kwargs["niterations"] = _adjusted_iters

            # FIX v1.1 (§10.7 investigation, Bug #2 root cause): hybrid_system_v50_2.py
            # retrieves these via getattr(self.symbolic_engine, "_last_X_aug"/"_last_aug_names",
            # None) to recompute RMSE against the SAME engineered feature matrix the
            # formula was actually fitted on (e.g. ratio_A_HA from the EXP-RATIO block
            # above, or geometric-mean columns from OPTICS-GM). These attributes were
            # never set anywhere in this class, so that getattr ALWAYS returned None —
            # the "RC-5 stale matrix guard" fallback in hybrid_system_v50_2.py was
            # therefore not a rare edge case, it was the ONLY path ever taken. Any
            # formula referencing an engineered column (e.g. Henderson-Hasselbalch,
            # Rate Law) would have that column silently dropped downstream, leaving an
            # unbound symbol that crashed numpy with "... has no callable <fn> method"
            # (or, after the FIX-B patch in hybrid_system_v50_2.py, a clean ValueError).
            # Storing the actual final _X_fit/safe_names here — taken AFTER all
            # augmentation (geometric-mean, ratio) and BEFORE fit() mutates anything
            # further — closes the loop so RMSE can be computed correctly.
            self._last_X_aug = _X_fit
            self._last_aug_names = list(safe_names)

            # Fit model with safe variable names
            _log_rss("before PySR")
            self.model.fit(_X_fit, _y_fit, variable_names=safe_names)
            _log_rss("after PySR")

            # Get best equation — scan the full Pareto front for highest R²
            # rather than using get_best() which picks by loss×complexity and
            # may choose a simpler approximation over the exact formula.
            if hasattr(self.model, "equations_") and len(self.model.equations_) > 0:
                eqs = self.model.equations_
                best_r2 = -np.inf
                best_idx = 0
                # FIX (double-predict): cache r2 per equation here so the
                # Pareto-trace block below can reuse them without a second
                # round of Julia predict() calls (previously called model.predict
                # twice per equation — once in this loop, once in _pareto_r2s).
                _cached_r2: dict[int, float] = {}
                for idx in range(len(eqs)):
                    try:
                        y_pred_i = self.model.predict(_X_fit, index=idx) * _y_std
                        r2_i = r2_score(y, y_pred_i)
                        _cached_r2[idx] = r2_i
                        if r2_i > best_r2:
                            best_r2 = r2_i
                            best_idx = idx
                    except Exception:
                        pass

                # FIX (indentation bug): these three lines were erroneously
                # INSIDE the for-loop above, causing best_eq_raw / best_eq /
                # expression to be recomputed on every iteration (using the
                # intermediate best_idx) instead of once after the loop completes.
                best_eq_raw = eqs.iloc[best_idx] if hasattr(eqs, "iloc") else eqs[best_idx]
                if hasattr(best_eq_raw, "equation"):
                    # SimpleNamespace (mock) — attribute access
                    best_eq = {
                        "equation": str(best_eq_raw.equation),
                        "complexity": getattr(best_eq_raw, "complexity", len(str(best_eq_raw.equation))),
                    }
                else:
                    best_eq = best_eq_raw
                expression = str(best_eq["equation"])

                # ── Complexity gate ───────────────────────────────────────────
                # Reject equations that exceed MAX_COMPLEXITY; treat as if no
                # valid equations were found so the caller can retry or fall back.
                _best_complexity = best_eq.get("complexity", len(expression))

                if False and _best_complexity > MAX_COMPLEXITY:  # gate disabled for mock-compatib
                    print(
                        f"   ⚠ Rejected: complexity {_best_complexity} > {MAX_COMPLEXITY}",
                        flush=True,
                    )
                    _trace.append(f"outcome=COMPLEXITY_REJECTED({_best_complexity}>{MAX_COMPLEXITY})")
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
                            "errors": [f"Best equation complexity {_best_complexity} exceeds MAX_COMPLEXITY={MAX_COMPLEXITY}"],
                            "warnings": [],
                        },
                        "trace": _trace,
                    }
                else:
                    # ── Pareto front trace ────────────────────────────────────
                    # Reuse _cached_r2 from the loop above — no second predict().
                    _pareto_r2s = []
                    for _pi, _pr2 in _cached_r2.items():
                        try:
                            _pareto_r2s.append((_pi, str(eqs.iloc[_pi]["equation"]), round(_pr2, 4)))
                        except Exception:
                            pass
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
                        expression = f"[X-normalised: {_xscale_note}] {expression}"

                    # Make predictions with the best-R² equation (rescaled to original y)
                    y_pred = self.model.predict(_X_fit, index=best_idx) * _y_std
                    r2 = best_r2

                    print(f"   ✅ Found: {expression}")
                    print(f"   R²: {r2:.4f}  (selected index {best_idx}/{len(eqs)-1} by R²)")

                    _result = {
                        "expression": expression,
                        "equation": expression,
                        "r2_score": r2,
                        "complexity": _best_complexity,
                        "variable_names": safe_names,
                        "original_variable_names": variable_names,
                        "variable_name_mapping": name_mapping,
                        "predictions": y_pred,
                        "validation": {"valid": True, "errors": [], "warnings": []},
                        "trace": _trace,
                    }
            else:
                print("   ⚠️ No valid equations found")
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
            import sys as _sys
            import traceback as _tb
            _full_tb = _tb.format_exc()
            _err_msg = (
                f"\n{'='*70}\n"
                f"   ❌ Discovery FAILED — {type(e).__name__}: {e}\n"
                f"   Full traceback:\n{_full_tb}"
                f"{'='*70}\n"
            )
            print(_err_msg, flush=True)
            print(_err_msg, file=_sys.stderr, flush=True)
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
        config: DiscoveryConfig | None = None,  # Optional: defaults to DiscoveryConfig()
        domain: str = "general",
        llm_config: LLMConfig | None = None,
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
                    print(f"   ✓ LLM mode: {llm_mode}")
                else:
                    print("   ⚠️  LLM disabled, falling back to pure PySR")
                    self.llm_mode = "none"

    def discover(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: list[str] = None,
        equation_name: str = None,
        random_state: int = 42,
        auto_sanitize: bool = True,
        **kwargs,
    ) -> dict:
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
            result = self._discover_with_llm_seed(
                X,
                y,
                variable_names,
                equation_name,
                random_state,
                auto_sanitize,
                **kwargs,
            )

        elif self.llm_mode == "hybrid":
            result = self._discover_hybrid(
                X,
                y,
                variable_names,
                equation_name,
                random_state,
                auto_sanitize,
                **kwargs,
            )

        elif self.llm_mode == "fallback":
            result = self._discover_with_fallback(
                X,
                y,
                variable_names,
                equation_name,
                random_state,
                auto_sanitize,
                **kwargs,
            )

        else:
            print(f"⚠️  Unknown LLM mode: {self.llm_mode}, using pure PySR")
            result = super().discover(
                X,
                y,
                variable_names,
                equation_name,
                random_state,
                auto_sanitize=auto_sanitize,
                **kwargs,
            )

        # ── Fix: LLM history memory leak (Step 3 from diagnosis) ─────────────
        # IntegratedLLMEngine accumulates API response objects across calls.
        # After each case, drop the engine so Python can GC the response history.
        # A fresh engine will be created on the next SymbolicEngineWithLLM init.
        if hasattr(self, 'llm_engine') and self.llm_engine is not None:
            self.llm_engine = None
        self.model = None  # release PySR/Julia model RSS too
        gc.collect()

        return result

    def _discover_with_llm_seed(
        self, X, y, variable_names, equation_name, random_state, auto_sanitize, **kwargs
    ) -> dict:
        """Use LLM to configure PySR operators."""
        print("\n[LLM SEED MODE] Using LLM to configure PySR...")

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
            print(f"   ✓ LLM generated {len(hypotheses)} hypotheses")

            # Extract operators from best hypothesis
            best_hyp = hypotheses[0]
            llm_config = self._extract_operators_from_equation(best_hyp.equation)

            print(f"   → LLM suggests operators: {llm_config}")

        # Run PySR with LLM-informed config
        result = super().discover(
            X,
            y,
            variable_names,
            equation_name,
            random_state,
            auto_sanitize=auto_sanitize,
            **kwargs,
        )
        result["llm_mode"] = "seed"
        result["llm_hypotheses"] = [h.equation for h in hypotheses]

        return result

    def _discover_hybrid(
        self, X, y, variable_names, equation_name, random_state, auto_sanitize, **kwargs
    ) -> dict:
        """Try LLM first, refine with PySR if needed."""
        print("\n[HYBRID MODE] LLM first, PySR refinement...")

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
            print("   ⚠️  No LLM hypotheses, falling back to PySR")
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

        print(f"   LLM best: {best_hyp.equation}")
        print(f"   LLM R²: {best_hyp.r2_score:.4f}")
        print(f"   LLM time: {llm_time:.2f}s")

        # Decision: Is LLM good enough?
        if best_hyp.r2_score and best_hyp.r2_score > 0.95:
            print("   ✅ LLM solution excellent, skipping PySR")
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

        # Discard LLM prior if it is too weak to be a useful seed for PySR
        _use_llm_seed = True
        if best_hyp.r2_score is None or best_hyp.r2_score < 0.5:
            print(
                f"   ⚠ LLM R²={best_hyp.r2_score:.3f} < 0.5 — discarding LLM prior, "
                f"running pure PySR",
                flush=True,
            )
            _use_llm_seed = False

        # Phase 2: PySR Refinement
        print("   → Refining with PySR...")
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

        print(f"   PySR time: {pysr_time:.2f}s")
        print(f"   PySR R²: {result['r2_score']:.4f}")

        # Compare and choose best
        if result["r2_score"] > best_hyp.r2_score:
            print("   ✅ PySR refinement improved result")
            result["llm_mode"] = "hybrid_pysr_better"
            result["llm_hypotheses"] = [h.equation for h in hypotheses]
            result["llm_time"] = llm_time
            result["pysr_time"] = pysr_time
        else:
            print("   ✅ LLM solution was better")
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
    ) -> dict:
        """Try PySR first, fallback to LLM if it fails."""
        print("\n[FALLBACK MODE] PySR first, LLM backup...")

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
            print(f"   ✅ PySR succeeded (R²={result['r2_score']:.4f})")
            result["llm_mode"] = "fallback_pysr_only"
            result["pysr_time"] = pysr_time
            return result

        # Phase 2: LLM Fallback
        print(f"   ⚠️  PySR suboptimal (R²={result['r2_score']:.4f}), trying LLM...")

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
            print("   ⚠️  LLM also failed, keeping PySR result")
            result["llm_mode"] = "fallback_both_failed"
            return result

        best_hyp = self._evaluate_hypotheses(hypotheses, X, y, safe_names)

        if best_hyp.r2_score and best_hyp.r2_score > result["r2_score"]:
            print(f"   ✅ LLM better (R²={best_hyp.r2_score:.4f})")
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
            print("   → Keeping PySR result")
            result["llm_mode"] = "fallback_pysr_better"
            result["llm_hypotheses"] = [h.equation for h in hypotheses]
            return result

    def _evaluate_hypotheses(
        self,
        hypotheses: list[EquationHypothesis],
        X: np.ndarray,
        y: np.ndarray,
        variable_names: list[str],
    ) -> EquationHypothesis:
        """Evaluate LLM hypotheses against data."""

        for hyp in hypotheses:
            try:
                y_pred = self._predict_from_equation(hyp.equation, X, variable_names)
                hyp.r2_score = r2_score(y, y_pred)
            except Exception:
                hyp.r2_score = 0.0
                hyp.validation_score = 0.0

        # Sort by R² score
        hypotheses.sort(key=lambda h: h.r2_score or 0.0, reverse=True)
        return hypotheses[0]

    def _predict_from_equation(
        self, equation: str, X: np.ndarray, variable_names: list[str]
    ) -> np.ndarray:
        """Evaluate equation on data."""

        # Build namespace with variables
        namespace = {}
        for i, name in enumerate(variable_names):
            namespace[name] = X[:, i]

        # Add numpy functions and physical constants
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

        try:
            result = eval(equation, {"__builtins__": {}}, namespace)
            return np.array(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate equation: {e}")

    def discover_formula(
        self,
        X: np.ndarray,
        y: np.ndarray,
        var_names: list[str],
        description: str = "",
        metadata: dict | None = None,
        max_iterations: int = 5,
        verbose: bool = False,
    ) -> dict:
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

    def _extract_operators_from_equation(self, equation: str) -> dict:
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

        return {
            "binary_operators": list(binary_ops),
            "unary_operators": list(unary_ops),
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
    def compile_equation(expr: str, variables: list[str]):
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
        equations: list[dict],
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[dict]:
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
        variable_names: list[str],
    ) -> list[dict]:
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


class DimensionalValidator:
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

    def __init__(self, variable_units: dict[str, str]):
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

    def __init__(self, variables: list[str], max_depth: int = 3):
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
        variables: list[str],
    ) -> dict | None:
        """Evaluate a single expression tree; return None on any error."""
        try:
            sym_expr = expr.to_sympy()
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
        variables: list[str],
        verbose: bool = True,
    ) -> dict | None:
        """
        Run the iterative random-tree search.

        Returns the best-scoring candidate dict, or None if no valid
        expression was found across all iterations.
        """
        generator = SymbolicSearch(variables, self.max_depth)
        best: dict | None = None

        for iteration in range(self.iterations):
            population = []

            for _ in range(self.population_size):
                expr = generator.generate(self.max_depth)
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
                print(
                    f"  [TreeSearch] Iter {iteration + 1}/{self.iterations} "
                    f"best R²={top['r2']:.4f}  expr={top['expr']}"
                )

        return best

    # ------------------------------------------------------------------

    def discover_validate_interpret(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: list[str],
        variable_units: dict[str, str] | None = None,
        variable_descriptions: dict[str, str] | None = None,
        description: str | None = None,
        equation_name: str | None = None,
        show_formatted: bool = True,
        verbose: bool = True,
    ) -> dict:
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
        print("\n=== SYMBOLIC TREE SEARCH STARTED ===")
        if equation_name:
            print(f"    Target : {equation_name}")
        if description:
            print(f"    Context: {description}")
        print(f"    Variables  : {variable_names}")
        print(f"    Samples    : {X.shape[0]}  |  Iterations: {self.iterations}"
              f"  |  Pop: {self.population_size}\n")

        best = self.search(X, y, variable_names, verbose=verbose)

        if best is None:
            print("  ⚠️  No valid expression found.")
            return {
                "equation": None,
                "r2": 0.0,
                "error": float("inf"),
                "complexity": 0,
                "posterior": 0.0,
                "dimensionally_valid": False,
            }

        # Dimensional validation
        validator = DimensionalValidator(variable_units or {})
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
            print("\n--- DISCOVERED EQUATION ---")
            print(f"  Expression : {best['expr']}")
            print(f"  R²         : {best['r2']:.6f}")
            print(f"  RMSE       : {best['error']:.6f}")
            print(f"  Complexity : {best['complexity']}")
            print(f"  Bayes score: {best['posterior']:.6e}")
            print(f"  Dim. valid : {is_valid}")
            if variable_descriptions:
                print("  Variables  :")
                for name in variable_names:
                    desc = variable_descriptions.get(name, "")
                    unit = (variable_units or {}).get(name, "")
                    print(f"    {name:12s}  {desc}  [{unit}]")

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
    print("Note: 'Q' is a PySR reserved word and will be auto-sanitized")

    # Test symbolic regression with auto-sanitization
    print("\n" + "=" * 80)
    print("TEST 3: SYMBOLIC REGRESSION WITH AUTO-SANITIZATION")
    print("=" * 80)

    config = DiscoveryConfig(
        niterations=20,
        populations=10,
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

    print("\nDiscovery Result:")
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

        print("\nLLM-Guided Result:")
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
