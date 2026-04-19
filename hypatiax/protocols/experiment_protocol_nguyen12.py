"""
experiment_protocol_nguyen12.py
================================

Nguyen-12 Symbolic Regression Benchmark Protocol for HypatiaX
--------------------------------------------------------------

This module provides the complete Nguyen-12 benchmark protocol,
structured identically to ``experiment_protocol_benchmark.py`` so that
both files can be dropped in as interchangeable runners.

The Nguyen-12 suite consists of 12 equations spanning:
  * Univariate polynomials   (N1–N4)
  * Univariate transcendental (N5–N8)
  * Bivariate expressions    (N9–N12)

Experiment 3 results (exp3_nguyen12_hybrid50v.json):
  * HypatiaX strict recovery : 11/12 (91.7%)
  * PySR-only strict recovery : 10/12 (83.3%)
  * MW H > P  : U=51.0, p=0.893 (n.s.)
  * MW P > NN : U=113.0, p=0.0097 (significant)

Usage
-----
    # Quick start — load all 12 equations
    python experiment_protocol_nguyen12.py

    # Run HypatiaX vs PySR-only on all equations
    python experiment_protocol_nguyen12.py --run

    # Single equation
    python experiment_protocol_nguyen12.py --equation N5

    # Extrapolation equations only
    python experiment_protocol_nguyen12.py --extrap-only

    # Print protocol documentation
    python experiment_protocol_nguyen12.py --describe

Author  : HypatiaX Team
Version : 1.0
Date    : 2026-04-06

Source data
-----------
Results reported against exp3_nguyen12_hybrid50v.json
(1000 iter / 300s / 30 pops / seed 42 / LLM hybrid mode, 8 candidates).

Cite as : Bonet Chaple, R.P. (2026). HypatiaX: A Hybrid Framework for
          Analytical Expression Discovery. JMLR (under review).
"""

from __future__ import annotations

import warnings
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Module-level reproducibility
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Optional heavy imports — gracefully degrade for documentation-only use
# ---------------------------------------------------------------------------
try:
    import pysr  # pip install pysr
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    warnings.warn(
        "pysr not installed. Cannot run experiments.\n"
        "Install with: pip install pysr",
        stacklevel=2,
    )


# ============================================================================
# EQUATION DEFINITIONS
# ============================================================================

@dataclass
class NguyenEquation:
    """
    One equation from the Nguyen-12 symbolic regression benchmark.

    Fields
    ------
    nguyen_id        : Canonical ID, e.g. ``"N1"``.
    name             : Human-readable name, e.g. ``"Nguyen-1"``.
    description      : Plain-English description of the relationship.
    ground_truth     : Analytical formula string (Python syntax).
    formula_hint     : High-level structural hint passed to LLM.
    variable_names   : List of variable names, e.g. ``["x"]`` or ``["x", "y"]``.
    variable_ranges  : Dict mapping variable name → (lo, hi) training range.
    extrap_ranges    : Dict mapping variable name → (lo, hi) extrapolation range.
    n_variables      : Number of input variables (1 or 2).
    difficulty       : ``"easy"`` / ``"medium"`` / ``"hard"``.
    category         : ``"polynomial"`` / ``"transcendental"`` / ``"bivariate"``.
    extrapolation_test : Whether this equation is flagged for extrap evaluation.
    positive_domain  : If True, all variables and the target must be positive
                       (special handling for N11 ``x^y``).
    fn               : Callable implementing the ground-truth function.
    """
    nguyen_id: str
    name: str
    description: str
    ground_truth: str
    formula_hint: str
    variable_names: List[str]
    variable_ranges: Dict[str, Tuple[float, float]]
    extrap_ranges: Dict[str, Tuple[float, float]]
    n_variables: int
    difficulty: str
    category: str
    extrapolation_test: bool
    positive_domain: bool
    fn: Optional[Callable] = field(default=None, repr=False)

    def generate(
        self,
        num_samples: int = 200,
        noise_level: float = 0.0,
        seed: int = 42,
        split: float = 0.2,
    ) -> Tuple[str, np.ndarray, np.ndarray, List[str], Dict]:
        """
        Generate ``(description, X_train, y_train, variable_names, metadata)``
        tuple compatible with the HypatiaX runner interface.

        Parameters
        ----------
        num_samples : Total data points before train/test split.
        noise_level : Gaussian noise as fraction of y std (0 = noiseless).
        seed        : Random seed (each equation gets a deterministic offset).
        split       : Fraction held out as test set (not returned here).

        Returns
        -------
        Tuple of (description, X, y, variable_names, metadata) where
        X is the training portion and metadata carries all relevant info.
        """
        if self.fn is None:
            raise ValueError(
                f"No callable fn defined for {self.nguyen_id}. "
                "Reconstruct with fn= argument."
            )

        rng = np.random.RandomState(seed)

        # Generate training data
        var_arrays_tr = [
            rng.uniform(lo, hi, num_samples)
            for lo, hi in [self.variable_ranges[v] for v in self.variable_names]
        ]
        X_tr = (
            np.column_stack(var_arrays_tr)
            if len(var_arrays_tr) > 1
            else var_arrays_tr[0].reshape(-1, 1)
        )
        y_tr = self.fn(*var_arrays_tr)

        # Filter non-finite values
        valid = np.isfinite(y_tr)
        X_tr, y_tr = X_tr[valid], y_tr[valid]

        if noise_level > 0:
            noise_std = noise_level * np.std(y_tr)
            y_tr = y_tr + rng.normal(0, noise_std, len(y_tr))

        # Train / test split
        n_train = int(len(X_tr) * (1 - split))
        X_train = X_tr[:n_train]
        y_train = y_tr[:n_train]

        # Generate extrapolation data (separate from training)
        if self.extrapolation_test:
            var_arrays_ext = [
                rng.uniform(lo, hi, 100)
                for lo, hi in [self.extrap_ranges[v] for v in self.variable_names]
            ]
            X_ext = (
                np.column_stack(var_arrays_ext)
                if len(var_arrays_ext) > 1
                else var_arrays_ext[0].reshape(-1, 1)
            )
            try:
                y_ext = self.fn(*var_arrays_ext)
                valid_ext = np.isfinite(y_ext)
                X_ext = X_ext[valid_ext]
                y_ext = y_ext[valid_ext]
            except Exception:
                X_ext = y_ext = None
        else:
            X_ext = y_ext = None

        metadata = {
            "equation_name": self.nguyen_id,
            "nguyen_id": self.nguyen_id,
            "name": self.name,
            "description": self.description,
            "ground_truth": self.ground_truth,
            "formula_hint": self.formula_hint,
            "category": self.category,
            "difficulty": self.difficulty,
            "n_variables": self.n_variables,
            "variable_ranges": self.variable_ranges,
            "extrap_ranges": self.extrap_ranges,
            "extrapolation_test": self.extrapolation_test,
            "positive_domain": self.positive_domain,
            "benchmark": "nguyen12",
            "source": "Uy et al. (2011)",
            # Extrapolation arrays attached for runners that evaluate them
            "X_extrap": X_ext,
            "y_extrap": y_ext,
        }

        return (self.description, X_train, y_train, self.variable_names, metadata)


# ---------------------------------------------------------------------------
# Helper: build the 12 equations
# ---------------------------------------------------------------------------

def _build_nguyen_equations() -> List[NguyenEquation]:
    eqs: List[NguyenEquation] = []

    # ── UNIVARIATE POLYNOMIALS ─────────────────────────────────────────────

    eqs.append(NguyenEquation(
        nguyen_id="N1",
        name="Nguyen-1",
        description="Cubic polynomial",
        ground_truth="x**3 + x**2 + x",
        formula_hint="polynomial degree 3",
        variable_names=["x"],
        variable_ranges={"x": (-1.0, 1.0)},
        extrap_ranges={"x": (1.0, 3.0)},
        n_variables=1,
        difficulty="easy",
        category="polynomial",
        extrapolation_test=True,
        positive_domain=False,
        fn=lambda x: x**3 + x**2 + x,
    ))

    eqs.append(NguyenEquation(
        nguyen_id="N2",
        name="Nguyen-2",
        description="Quartic polynomial",
        ground_truth="x**4 + x**3 + x**2 + x",
        formula_hint="polynomial degree 4",
        variable_names=["x"],
        variable_ranges={"x": (-1.0, 1.0)},
        extrap_ranges={"x": (1.0, 3.0)},
        n_variables=1,
        difficulty="easy",
        category="polynomial",
        extrapolation_test=True,
        positive_domain=False,
        fn=lambda x: x**4 + x**3 + x**2 + x,
    ))

    eqs.append(NguyenEquation(
        nguyen_id="N3",
        name="Nguyen-3",
        description="Quintic polynomial",
        ground_truth="x**5 + x**4 + x**3 + x**2 + x",
        formula_hint="polynomial degree 5",
        variable_names=["x"],
        variable_ranges={"x": (-1.0, 1.0)},
        extrap_ranges={"x": (1.0, 3.0)},
        n_variables=1,
        difficulty="medium",
        category="polynomial",
        extrapolation_test=True,
        positive_domain=False,
        fn=lambda x: x**5 + x**4 + x**3 + x**2 + x,
    ))

    eqs.append(NguyenEquation(
        nguyen_id="N4",
        name="Nguyen-4",
        description="Sextic polynomial (hardest polynomial — Taylor trap)",
        ground_truth="x**6 + x**5 + x**4 + x**3 + x**2 + x",
        formula_hint="polynomial degree 6",
        variable_names=["x"],
        variable_ranges={"x": (-1.0, 1.0)},
        extrap_ranges={"x": (1.0, 3.0)},
        n_variables=1,
        difficulty="hard",
        category="polynomial",
        extrapolation_test=True,
        positive_domain=False,
        fn=lambda x: x**6 + x**5 + x**4 + x**3 + x**2 + x,
    ))

    # ── UNIVARIATE TRANSCENDENTAL ─────────────────────────────────────────

    eqs.append(NguyenEquation(
        nguyen_id="N5",
        name="Nguyen-5",
        description="Product of sine and cosine with unit offset",
        ground_truth="sin(x**2) * cos(x) - 1",
        formula_hint="trig product with offset",
        variable_names=["x"],
        variable_ranges={"x": (-1.0, 1.0)},
        extrap_ranges={"x": (1.0, 3.0)},
        n_variables=1,
        difficulty="medium",
        category="transcendental",
        extrapolation_test=True,
        positive_domain=False,
        fn=lambda x: np.sin(x**2) * np.cos(x) - 1,
    ))

    eqs.append(NguyenEquation(
        nguyen_id="N6",
        name="Nguyen-6",
        description="Sum of two sine functions",
        ground_truth="sin(x) + sin(x + x**2)",
        formula_hint="sum of sines",
        variable_names=["x"],
        variable_ranges={"x": (-1.0, 1.0)},
        extrap_ranges={"x": (1.0, 3.0)},
        n_variables=1,
        difficulty="medium",
        category="transcendental",
        extrapolation_test=True,
        positive_domain=False,
        fn=lambda x: np.sin(x) + np.sin(x + x**2),
    ))

    eqs.append(NguyenEquation(
        nguyen_id="N7",
        name="Nguyen-7",
        description="Sum of two logarithms",
        ground_truth="log(x + 1) + log(x**2 + 1)",
        formula_hint="sum of logs",
        variable_names=["x"],
        variable_ranges={"x": (0.0, 2.0)},
        extrap_ranges={"x": (2.0, 5.0)},
        n_variables=1,
        difficulty="hard",
        category="transcendental",
        extrapolation_test=True,
        positive_domain=True,
        fn=lambda x: np.log(x + 1) + np.log(x**2 + 1),
    ))

    eqs.append(NguyenEquation(
        nguyen_id="N8",
        name="Nguyen-8",
        description="Square root (easiest transcendental — typically LLM-only)",
        ground_truth="sqrt(x)",
        formula_hint="square root",
        variable_names=["x"],
        variable_ranges={"x": (0.0, 4.0)},
        extrap_ranges={"x": (4.0, 10.0)},
        n_variables=1,
        difficulty="easy",
        category="transcendental",
        extrapolation_test=True,
        positive_domain=True,
        fn=lambda x: np.sqrt(x),
    ))

    # ── BIVARIATE ─────────────────────────────────────────────────────────

    eqs.append(NguyenEquation(
        nguyen_id="N9",
        name="Nguyen-9",
        description="Sum of univariate sines in two variables",
        ground_truth="sin(x) + sin(y**2)",
        formula_hint="bivariate sines",
        variable_names=["x", "y"],
        variable_ranges={"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
        extrap_ranges={"x": (1.0, 3.0), "y": (1.0, 3.0)},
        n_variables=2,
        difficulty="medium",
        category="bivariate",
        extrapolation_test=True,
        positive_domain=False,
        fn=lambda x, y: np.sin(x) + np.sin(y**2),
    ))

    eqs.append(NguyenEquation(
        nguyen_id="N10",
        name="Nguyen-10",
        description="Product of sine and cosine in two variables",
        ground_truth="2 * sin(x) * cos(y)",
        formula_hint="product sin/cos",
        variable_names=["x", "y"],
        variable_ranges={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        extrap_ranges={"x": (1.0, 3.0), "y": (1.0, 3.0)},
        n_variables=2,
        difficulty="medium",
        category="bivariate",
        extrapolation_test=True,
        positive_domain=False,
        fn=lambda x, y: 2 * np.sin(x) * np.cos(y),
    ))

    eqs.append(NguyenEquation(
        nguyen_id="N11",
        name="Nguyen-11",
        description="Power law x^y (positive domain only)",
        ground_truth="x**y",
        formula_hint="power law",
        variable_names=["x", "y"],
        # Positive domain: x ∈ (0.1, 1.0), y ∈ (0.5, 3.0) avoids
        # complex-valued outputs and keeps the target well-conditioned.
        variable_ranges={"x": (0.1, 1.0), "y": (0.5, 3.0)},
        extrap_ranges={"x": (1.0, 3.0), "y": (3.0, 6.0)},
        n_variables=2,
        difficulty="medium",
        category="bivariate",
        extrapolation_test=True,
        positive_domain=True,
        fn=lambda x, y: np.power(np.abs(x) + 1e-8, y),
    ))

    eqs.append(NguyenEquation(
        nguyen_id="N12",
        name="Nguyen-12",
        description="Bivariate polynomial with mixed signs",
        ground_truth="x**4 - x**3 + y**2 / 2 - y",
        formula_hint="bivariate polynomial",
        variable_names=["x", "y"],
        variable_ranges={"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
        extrap_ranges={"x": (1.0, 3.0), "y": (1.0, 3.0)},
        n_variables=2,
        difficulty="hard",
        category="bivariate",
        extrapolation_test=True,
        positive_domain=False,
        fn=lambda x, y: x**4 - x**3 + y**2 / 2 - y,
    ))

    assert len(eqs) == 12, f"Expected 12 Nguyen equations, got {len(eqs)}"
    return eqs


NGUYEN_EQUATIONS: List[NguyenEquation] = _build_nguyen_equations()
NGUYEN_BY_ID: Dict[str, NguyenEquation] = {eq.nguyen_id: eq for eq in NGUYEN_EQUATIONS}


# ============================================================================
# PROTOCOL CLASS
# ============================================================================

class NguYenProtocol:
    """
    Protocol manager for the Nguyen-12 symbolic regression benchmark.

    Provides the same interface as ``BenchmarkProtocol`` in
    ``experiment_protocol_benchmark.py``:

        protocol.get_all_domains()
        protocol.load_test_data(domain, ...)

    Domains correspond to the three Nguyen categories:
        ``"nguyen_polynomial"``      → N1–N4
        ``"nguyen_transcendental"``  → N5–N8
        ``"nguyen_bivariate"``       → N9–N12

    Quick start
    -----------
    >>> proto = NguYenProtocol()
    >>> for domain in proto.get_all_domains():
    ...     for desc, X, y, var_names, meta in proto.load_test_data(domain):
    ...         print(meta["equation_name"], X.shape)
    """

    # Recovery thresholds matching experiment_protocol_benchmark.py constants
    RECOVERY_THRESHOLD_STRICT: float = 0.9999  # Used in Exp 3
    RECOVERY_THRESHOLD_RELAXED: float = 0.99
    RECOVERY_THRESHOLD_PRACTICAL: float = 0.995
    NOISE_LEVEL_DEFAULT: float = 0.01           # Exp 3 uses 1% noise
    NOISE_LEVEL_NOISELESS: float = 0.0

    # Published results from exp3_nguyen12_hybrid50v.json (seed 42)
    EXP3_RESULTS: Dict[str, Dict] = {
        "N1":  {"H_train": 0.9999256, "H_extrap": 0.9998852, "P_train": 0.9999104, "P_extrap": 0.9999994, "NN_extrap": -0.7837},
        "N2":  {"H_train": 0.9999249, "H_extrap": 0.9999858, "P_train": 0.9999112, "P_extrap": 0.9999999, "NN_extrap": -0.9019},
        "N3":  {"H_train": 0.9999138, "H_extrap": 0.9976055, "P_train": 0.9998861, "P_extrap": -426.225, "NN_extrap": -0.9126},
        "N4":  {"H_train": 0.9999146, "H_extrap": -27439417.4, "P_train": 0.9999002, "P_extrap": -47010.6, "NN_extrap": -0.8284},
        "N5":  {"H_train": 0.9999133, "H_extrap": 0.9999810, "P_train": 0.9999102, "P_extrap": 0.9999999, "NN_extrap": -5.5863},
        "N6":  {"H_train": 0.9999224, "H_extrap": 0.9999891, "P_train": 0.9999094, "P_extrap": 0.9999997, "NN_extrap": -12.654},
        "N7":  {"H_train": 0.9999138, "H_extrap": 0.7315580, "P_train": 0.9999103, "P_extrap": 0.9762010, "NN_extrap": 0.8564},
        "N8":  {"H_train": 0.9999094, "H_extrap": 1.0,       "P_train": 0.9999094, "P_extrap": 1.0,       "NN_extrap": 0.9541},
        "N9":  {"H_train": 0.9999263, "H_extrap": 0.9999999, "P_train": 0.9999176, "P_extrap": 1.0,       "NN_extrap": -6.7078},
        "N10": {"H_train": 0.9999271, "H_extrap": 0.9996501, "P_train": 0.9999160, "P_extrap": 0.9999999, "NN_extrap": -2.3794},
        "N11": {"H_train": 0.9999272, "H_extrap": 0.9999105, "P_train": 0.9999149, "P_extrap": 1.0,       "NN_extrap": -0.4226},
        "N12": {"H_train": 0.9994111, "H_extrap": -1.0537,   "P_train": 0.9986729, "P_extrap": -1.0556,   "NN_extrap": -1.1980},
    }

    _CATEGORY_DOMAIN_MAP: Dict[str, str] = {
        "polynomial":    "nguyen_polynomial",
        "transcendental": "nguyen_transcendental",
        "bivariate":     "nguyen_bivariate",
    }
    _DOMAIN_CATEGORY_MAP: Dict[str, str] = {v: k for k, v in _CATEGORY_DOMAIN_MAP.items()}

    def __init__(
        self,
        num_samples: int = 200,
        noise_level: float = NOISE_LEVEL_DEFAULT,
        seed: int = 42,
        category: Optional[str] = None,
        extrap_only: bool = False,
        noiseless: bool = False,
    ) -> None:
        """
        Initialise the Nguyen-12 protocol.

        Parameters
        ----------
        num_samples : Default data points per equation.
        noise_level : Gaussian noise as fraction of y std.
        seed        : Base random seed.
        category    : Restrict to ``"polynomial"`` / ``"transcendental"``
                      / ``"bivariate"``; ``None`` = all.
        extrap_only : Only load equations flagged for extrapolation testing.
        noiseless   : Shorthand for ``noise_level=0.0``.
        """
        self.default_num_samples = num_samples
        self.noise_level = 0.0 if noiseless else noise_level
        self.noiseless = noiseless or (noise_level == 0.0)
        self.seed = seed
        self.category = category
        self.extrap_only = extrap_only

        self._equations: List[NguyenEquation] = self._filter(
            category=category, extrap_only=extrap_only
        )

    # ── Filtering ─────────────────────────────────────────────────────────

    @staticmethod
    def _filter(
        category: Optional[str] = None,
        extrap_only: bool = False,
    ) -> List[NguyenEquation]:
        eqs = NGUYEN_EQUATIONS
        if category is not None:
            eqs = [e for e in eqs if e.category == category]
        if extrap_only:
            eqs = [e for e in eqs if e.extrapolation_test]
        return eqs

    # ── Runner interface (mirrors BenchmarkProtocol) ──────────────────────

    def get_all_domains(self) -> List[str]:
        """
        Return domain keys exposed by this protocol.

        Returns ``["nguyen_bivariate", "nguyen_polynomial", "nguyen_transcendental"]``
        (or a subset if filtered at construction time).
        """
        cats = sorted({eq.category for eq in self._equations})
        return [self._CATEGORY_DOMAIN_MAP[c] for c in cats]

    def load_test_data(
        self,
        domain: str,
        num_samples: Optional[int] = None,
        noise_level: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[Tuple[str, np.ndarray, np.ndarray, List[str], Dict]]:
        """
        Load test cases for *domain*.

        Mirrors ``BenchmarkProtocol.load_test_data()`` exactly:

            for desc, X, y, var_names, meta in protocol.load_test_data(domain):
                ...

        Parameters
        ----------
        domain      : One of the strings from ``get_all_domains()``.
        num_samples : Data points per equation (default: ``self.default_num_samples``).
        noise_level : Gaussian noise fraction (default: ``self.noise_level``).
        seed        : Base random seed (default: ``self.seed``).

        Returns
        -------
        List of ``(description, X, y, variable_names, metadata)`` tuples.
        ``metadata`` always contains:
            ``equation_name``, ``nguyen_id``, ``category``, ``difficulty``,
            ``benchmark``, ``ground_truth``, ``extrapolation_test``,
            ``X_extrap``, ``y_extrap``.

        Raises
        ------
        KeyError
            If *domain* is not one of the strings from ``get_all_domains()``.
        """
        available = self.get_all_domains()
        if domain not in available:
            raise KeyError(
                f"Unknown domain '{domain}'.\n"
                f"Available: {', '.join(available)}"
            )

        n     = num_samples if num_samples is not None else self.default_num_samples
        noise = noise_level if noise_level is not None else self.noise_level
        s     = seed if seed is not None else self.seed

        cat   = self._DOMAIN_CATEGORY_MAP[domain]
        items = [eq for eq in self._equations if eq.category == cat]

        results = []
        for offset, eq in enumerate(items):
            results.append(eq.generate(n, noise, seed=s + offset * 7))
        return results

    # ── Individual equation access ────────────────────────────────────────

    @staticmethod
    def get_equation(nguyen_id: str) -> NguyenEquation:
        """Return a single equation by ID, e.g. ``"N5"``."""
        if nguyen_id not in NGUYEN_BY_ID:
            raise KeyError(
                f"Unknown Nguyen ID '{nguyen_id}'. "
                f"Available: {sorted(NGUYEN_BY_ID)}"
            )
        return NGUYEN_BY_ID[nguyen_id]

    @staticmethod
    def load_single(
        nguyen_id: str,
        num_samples: int = 200,
        noise_level: float = 0.0,
        seed: int = 42,
    ) -> Tuple[str, np.ndarray, np.ndarray, List[str], Dict]:
        """Load data for a single equation by ID."""
        eq = NguYenProtocol.get_equation(nguyen_id)
        return eq.generate(num_samples, noise_level, seed)

    @staticmethod
    def load_all(
        num_samples: int = 200,
        noise_level: float = 0.0,
        seed: int = 42,
    ) -> List[Tuple[str, np.ndarray, np.ndarray, List[str], Dict]]:
        """Load data for all 12 Nguyen equations."""
        return [
            eq.generate(num_samples, noise_level, seed=seed + i * 7)
            for i, eq in enumerate(NGUYEN_EQUATIONS)
        ]

    # ── Reporting ─────────────────────────────────────────────────────────

    @staticmethod
    def generate_experiment_report(
        results: List[Dict],
        threshold: float = RECOVERY_THRESHOLD_STRICT,
    ) -> Dict:
        """
        Generate benchmark comparison report.

        Compatible with ``BenchmarkProtocol.generate_experiment_report()``
        so existing infrastructure works unchanged.

        Parameters
        ----------
        results   : List of result dicts from an experiment runner.
        threshold : R² threshold for counting a recovery as successful.

        Returns
        -------
        Dict with ``overall``, ``by_category``, ``by_system`` keys.
        """
        successful = [
            r for r in results
            if r.get("evaluation", {}).get("r2", -np.inf) >= threshold
        ]
        r2_all = [
            r["evaluation"]["r2"]
            for r in results
            if "r2" in r.get("evaluation", {})
        ]

        by_category: Dict[str, Dict] = {}
        by_system: Dict[str, Dict] = {}

        for r in results:
            meta     = r.get("metadata", {})
            category = meta.get("category", "unknown")
            system   = r.get("system", "hypatiax")
            r2       = r.get("evaluation", {}).get("r2")
            ok       = r2 is not None and r2 >= threshold

            for bucket, key in [(by_category, category), (by_system, system)]:
                if key not in bucket:
                    bucket[key] = {"total": 0, "recovered": 0, "r2_scores": []}
                bucket[key]["total"] += 1
                if ok:
                    bucket[key]["recovered"] += 1
                if r2 is not None:
                    bucket[key]["r2_scores"].append(r2)

        def _summarise(bucket: Dict) -> Dict:
            out = {}
            for key, val in bucket.items():
                scores = val["r2_scores"]
                total  = val["total"]
                rec    = val["recovered"]
                out[key] = {
                    "total":         total,
                    "recovered":     rec,
                    "recovery_rate": rec / total if total else 0.0,
                    "mean_r2":   float(np.mean(scores)) if scores else None,
                    "median_r2": float(np.median(scores)) if scores else None,
                    "std_r2":    float(np.std(scores)) if scores else None,
                }
            return out

        return {
            "overall": {
                "total":         len(results),
                "recovered":     len(successful),
                "recovery_rate": len(successful) / len(results) if results else 0.0,
                "mean_r2":       float(np.mean(r2_all)) if r2_all else None,
                "median_r2":     float(np.median(r2_all)) if r2_all else None,
                "r2_threshold":  threshold,
                "benchmark":     "nguyen12",
            },
            "by_category": _summarise(by_category),
            "by_system":   _summarise(by_system),
        }

    # ── Documentation ─────────────────────────────────────────────────────

    @staticmethod
    def describe() -> None:
        """Print full protocol documentation to stdout."""
        print()
        print("=" * 80)
        print("  NGUYEN-12 BENCHMARK PROTOCOL")
        print("  HypatiaX Experiment 3 — Source: exp3_nguyen12_hybrid50v.json")
        print("=" * 80)

        by_cat: Dict[str, List[NguyenEquation]] = {}
        for eq in NGUYEN_EQUATIONS:
            by_cat.setdefault(eq.category, []).append(eq)

        for cat, eqs in [
            ("polynomial",    by_cat.get("polynomial", [])),
            ("transcendental",by_cat.get("transcendental", [])),
            ("bivariate",     by_cat.get("bivariate", [])),
        ]:
            print(f"\n  {cat.upper()} ({len(eqs)} equations):")
            print("  " + "-" * 60)
            for eq in eqs:
                flag = "⭐" if eq.extrapolation_test else "  "
                pos  = "[+domain]" if eq.positive_domain else ""
                print(
                    f"    {flag} {eq.nguyen_id:<5} {eq.name:<14} "
                    f"{'[' + eq.difficulty + ']':<10} "
                    f"{eq.ground_truth:<30} {pos}"
                )

        print(f"\n  Total equations       : {len(NGUYEN_EQUATIONS)}")
        print(f"  Extrapolation-flagged : {sum(1 for e in NGUYEN_EQUATIONS if e.extrapolation_test)}")
        print(f"  Positive domain       : {sum(1 for e in NGUYEN_EQUATIONS if e.positive_domain)}")
        print()
        print("  PUBLISHED RESULTS (exp3_nguyen12_hybrid50v.json, seed 42):")
        print("  " + "-" * 60)
        print(f"  {'ID':<5} {'H_train':>9} {'H_extrap':>12} {'P_train':>9} {'P_extrap':>12} {'NN_extrap':>12}")
        for nid, row in NguYenProtocol.EXP3_RESULTS.items():
            print(
                f"  {nid:<5} {row['H_train']:>9.7f} {row['H_extrap']:>12.4g} "
                f"{row['P_train']:>9.7f} {row['P_extrap']:>12.4g} {row['NN_extrap']:>12.4g}"
            )
        print()
        print("  H strict recovery  : 11/12 (91.7%)")
        print("  P strict recovery  : 10/12 (83.3%)")
        print("  MW H > P  : U=51.0, p=0.893 (n.s.)")
        print("  MW P > NN : U=113.0, p=0.0097 (sig.)")
        print()


# ============================================================================
# MAIN — CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HypatiaX Nguyen-12 Benchmark Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print full protocol documentation and exit",
    )
    parser.add_argument(
        "--equation",
        metavar="ID",
        default=None,
        help="Load a single equation by ID (e.g. N5)",
    )
    parser.add_argument(
        "--category",
        choices=["polynomial", "transcendental", "bivariate"],
        default=None,
        help="Restrict to one category",
    )
    parser.add_argument(
        "--extrap-only",
        action="store_true",
        help="Only load extrapolation-flagged equations",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of data points per equation (default: 200)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=NguYenProtocol.NOISE_LEVEL_DEFAULT,
        help=f"Noise level as fraction of y std (default: {NguYenProtocol.NOISE_LEVEL_DEFAULT})",
    )
    parser.add_argument(
        "--noiseless",
        action="store_true",
        help="Run with zero noise (noise_level=0.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    args = parser.parse_args()

    # ── --describe ────────────────────────────────────────────────────────
    if args.describe:
        NguYenProtocol.describe()
        raise SystemExit(0)

    # ── --equation ────────────────────────────────────────────────────────
    if args.equation:
        eq_id = args.equation.upper()
        desc, X, y, var_names, meta = NguYenProtocol.load_single(
            eq_id,
            num_samples=args.num_samples,
            noise_level=0.0 if args.noiseless else args.noise,
            seed=args.seed,
        )
        print(f"\n  {eq_id}: {meta['ground_truth']}")
        print(f"  Variables   : {var_names}")
        print(f"  X shape     : {X.shape}")
        print(f"  y range     : [{y.min():.4f}, {y.max():.4f}]")
        print(f"  Difficulty  : {meta['difficulty']}")
        if meta.get("X_extrap") is not None:
            print(f"  X_extrap    : {meta['X_extrap'].shape}")
        raise SystemExit(0)

    # ── Standard run ─────────────────────────────────────────────────────
    protocol = NguYenProtocol(
        num_samples=args.num_samples,
        noise_level=0.0 if args.noiseless else args.noise,
        seed=args.seed,
        category=args.category,
        extrap_only=args.extrap_only,
        noiseless=args.noiseless,
    )

    if args.noiseless or args.noise == 0.0:
        print("\n" + "=" * 70)
        print("  ⚠️  NOISELESS MODE — noise_level=0.0")
        print("  Use R²≥0.9999 threshold for literature comparison")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print(f"  NOISY MODE — noise_level={protocol.noise_level}")
        print("=" * 70)

    print()
    print("=" * 80)
    print("  HypatiaX NGUYEN-12 BENCHMARK PROTOCOL")
    print("=" * 80)

    domains = protocol.get_all_domains()
    print(f"\n  {len(domains)} domain(s) via get_all_domains():\n")

    total_cases = 0
    for domain in domains:
        cases = protocol.load_test_data(domain)
        total_cases += len(cases)
        print(f"  [{domain}]  ({len(cases)} equations)")
        for desc, X, y, var_names, meta in cases:
            nid        = meta["equation_name"]
            ref        = NguYenProtocol.EXP3_RESULTS.get(nid, {})
            h_train    = ref.get("H_train", float("nan"))
            p_train    = ref.get("P_train", float("nan"))
            extrap_sym = "⭐" if meta.get("extrapolation_test") else "  "
            pos_sym    = "[+]" if meta.get("positive_domain") else "   "
            print(
                f"    {extrap_sym} {nid:<5} {meta['ground_truth']:<32} "
                f"{'['+meta['difficulty']+']':<10} {pos_sym}  "
                f"H_train={h_train:.7f}  P_train={p_train:.7f}"
            )

    print(f"\n  Total equations : {total_cases}")
    print(f"  ⭐ = extrapolation test,  [+] = positive domain only")
    print()
    print("  Published recovery (strict R²≥0.9999, hybrid50v run):")
    print("    HypatiaX  : 11/12  (91.7%)")
    print("    PySR-only : 10/12  (83.3%)")
    print("    MW H>P    : U=51.0, p=0.893  (n.s.)")
    print("    MW P>NN   : U=113.0, p=0.0097  (sig.)")
    print()
    print(f"  Full docs : python {__file__} --describe")
    print()
