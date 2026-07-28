"""
hypatiax/core/metrics.py
========================
Regression and symbolic-regression evaluation metrics for HypatiaX.

Paper: "HypatiaX: A Hybrid Symbolic-Neural Framework for
        Extrapolation-Reliable Analytical Discovery"  (JMLR v3.0, Apr 2026)

Public API
----------
compute_r2(y_true, y_pred)                  → float
compute_mse(y_true, y_pred)                 → float
compute_rmse(y_true, y_pred)                → float
compute_mae(y_true, y_pred)                 → float
compute_max_error(y_true, y_pred)           → float
compute_near_perfect_rate(scores, thr)      → float   # §10.2: R²>0.99 rate
compute_catastrophic_rate(scores, thr)      → float   # §10.2: R²<-1 rate
compute_speedup(t_baseline, t_system)       → float
evaluate_all(y_true, y_pred)               → dict

All functions accept array-like inputs (list, np.ndarray, pd.Series).
NumPy is the only hard dependency; scikit-learn is used when available
for cross-validation of R² but is never required at import time.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Union

import numpy as np

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ArrayLike = Union[Sequence[float], np.ndarray]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_array(x: ArrayLike, name: str = "array") -> np.ndarray:
    """Convert array-like to a 1-D float64 numpy array."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D after ravelling, got shape {arr.shape}")
    return arr


def _validate_pair(y_true: ArrayLike, y_pred: ArrayLike):
    """Return validated (y_true, y_pred) arrays of equal length."""
    yt = _to_array(y_true, "y_true")
    yp = _to_array(y_pred, "y_pred")
    if len(yt) != len(yp):
        raise ValueError(
            f"y_true and y_pred must have the same length "
            f"({len(yt)} vs {len(yp)})"
        )
    if len(yt) == 0:
        raise ValueError("Arrays must not be empty")
    return yt, yp


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Coefficient of determination (R²).

    R² = 1 - SS_res / SS_tot

    Returns
    -------
    float
        R² ∈ (-∞, 1].  Returns 0.0 when SS_tot == 0 (constant target),
        matching sklearn's behaviour.

    Examples
    --------
    >>> compute_r2([1, 2, 3], [1, 2, 3])
    1.0
    >>> round(compute_r2([1, 2, 3], [1, 2, 4]), 4)
    0.5
    """
    yt, yp = _validate_pair(y_true, y_pred)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot == 0.0:
        return 0.0 if ss_res != 0.0 else 1.0
    return float(1.0 - ss_res / ss_tot)


def compute_mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Mean Squared Error.

    MSE = (1/n) Σ (y_true - y_pred)²

    Examples
    --------
    >>> compute_mse([0, 1], [1, 1])
    0.5
    """
    yt, yp = _validate_pair(y_true, y_pred)
    return float(np.mean((yt - yp) ** 2))


def compute_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Root Mean Squared Error.

    RMSE = sqrt(MSE)

    Examples
    --------
    >>> round(compute_rmse([0, 1], [1, 1]), 6)
    0.707107
    """
    return math.sqrt(compute_mse(y_true, y_pred))


def compute_mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Mean Absolute Error.

    MAE = (1/n) Σ |y_true - y_pred|

    Examples
    --------
    >>> compute_mae([0, 1, 2], [0, 0, 2])
    0.3333333333333333
    """
    yt, yp = _validate_pair(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def compute_max_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Maximum absolute error (L∞ norm of residuals).

    Examples
    --------
    >>> compute_max_error([0, 1, 2], [0, 0, 2])
    1.0
    """
    yt, yp = _validate_pair(y_true, y_pred)
    return float(np.max(np.abs(yt - yp)))


# ---------------------------------------------------------------------------
# HypatiaX paper-specific aggregate metrics (§10.2, §10.4)
# ---------------------------------------------------------------------------

def compute_near_perfect_rate(
    scores: ArrayLike,
    threshold: float = 0.99,
) -> float:
    """
    Fraction of tasks whose R² exceeds *threshold*.

    Paper target (§10.2): 89.2 % of DeFi 74-task benchmark at R²>0.99.

    Parameters
    ----------
    scores : array-like of float
        Per-task R² scores.
    threshold : float
        Default 0.99.

    Returns
    -------
    float
        Value in [0, 1].

    Examples
    --------
    >>> compute_near_perfect_rate([0.995, 0.98, 0.999], threshold=0.99)
    0.6666666666666666
    """
    arr = _to_array(scores, "scores")
    if len(arr) == 0:
        return float("nan")
    return float(np.mean(arr > threshold))


def compute_catastrophic_rate(
    scores: ArrayLike,
    threshold: float = -1.0,
) -> float:
    """
    Fraction of tasks whose R² is below *threshold* (catastrophic failures).

    Paper target (§10.2): 0 catastrophic failures (R²<-1).

    Parameters
    ----------
    scores : array-like of float
        Per-task R² scores.
    threshold : float
        Default -1.0  (negative infinity of meaningful fit).

    Returns
    -------
    float
        Value in [0, 1].

    Examples
    --------
    >>> compute_catastrophic_rate([0.9, -2.0, 0.5])
    0.3333333333333333
    """
    arr = _to_array(scores, "scores")
    if len(arr) == 0:
        return float("nan")
    return float(np.mean(arr < threshold))


def compute_speedup(t_baseline: float, t_system: float) -> float:
    """
    Speedup ratio: baseline wall-time / system wall-time.

    Paper target (§10.4): 1.73× speedup for LLM-routed cases.

    Parameters
    ----------
    t_baseline : float
        Wall-clock time of the baseline method (seconds).
    t_system : float
        Wall-clock time of HypatiaX (seconds).

    Returns
    -------
    float
        Speedup ≥ 0.  Returns inf when t_system == 0.

    Examples
    --------
    >>> round(compute_speedup(1.73, 1.0), 2)
    1.73
    """
    if t_system < 0 or t_baseline < 0:
        raise ValueError("Times must be non-negative")
    if t_system == 0.0:
        return float("inf")
    return float(t_baseline / t_system)


# ---------------------------------------------------------------------------
# Convenience bundle
# ---------------------------------------------------------------------------

def evaluate_all(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    near_perfect_threshold: float = 0.99,
    catastrophic_threshold: float = -1.0,
) -> dict:
    """
    Compute all scalar metrics for a single prediction array and return
    them as a plain dict.

    Returned keys
    -------------
    r2, mse, rmse, mae, max_error,
    near_perfect  (bool — R² > near_perfect_threshold),
    catastrophic  (bool — R² < catastrophic_threshold)

    Examples
    --------
    >>> m = evaluate_all([1, 2, 3], [1, 2, 3])
    >>> m['r2']
    1.0
    >>> m['catastrophic']
    False
    """
    r2 = compute_r2(y_true, y_pred)
    return {
        "r2":           r2,
        "mse":          compute_mse(y_true, y_pred),
        "rmse":         compute_rmse(y_true, y_pred),
        "mae":          compute_mae(y_true, y_pred),
        "max_error":    compute_max_error(y_true, y_pred),
        "near_perfect": bool(r2 > near_perfect_threshold),
        "catastrophic": bool(r2 < catastrophic_threshold),
    }


# ---------------------------------------------------------------------------
# Backwards-compat shims
# ---------------------------------------------------------------------------
# statistical_analysis.py originally did:
#   from hypatiax.core import metrics
#   metrics.compute_r2(...)
# All names are already at module level so that usage works unchanged.

r2_score  = compute_r2    # sklearn-style alias
mean_squared_error   = compute_mse
root_mean_squared_error = compute_rmse
mean_absolute_error  = compute_mae


# ---------------------------------------------------------------------------
# Self-test (python -m hypatiax.core.metrics)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import doctest
    results = doctest.testmod(verbose=False)
    if results.failed:
        raise SystemExit(f"{results.failed} doctest(s) failed")

    # Spot-check against sklearn when available
    try:
        from sklearn.metrics import r2_score as _skl_r2
        _yt = np.random.default_rng(42).normal(size=200)
        _yp = _yt + np.random.default_rng(0).normal(scale=0.1, size=200)
        _ours  = compute_r2(_yt, _yp)
        _theirs = float(_skl_r2(_yt, _yp))
        assert abs(_ours - _theirs) < 1e-10, f"R² mismatch: {_ours} vs {_theirs}"
        print(f"✓ R² matches sklearn ({_ours:.6f})")
    except ImportError:
        print("sklearn not installed — skipping cross-check")

    print("✓ All metrics self-tests passed")
