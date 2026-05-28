"""
extrap_r2_far.py
================
Extracted from run_comparative_suite_benchmark_v2.py

Contains the two self-contained pieces that produce extrap_r2_far:

  1. build_extrap_split()   — splits (X, y) into (X_train, y_train, X_far, y_far)
                               given train_frac and multiplier.  Called once per
                               test-case in the --extrap pre-processing loop.

  2. compute_extrap_r2_far() — given the per-method MethodResult dict and the
                               held-out (X_far, y_far), re-evaluates each method's
                               formula string on the far region and returns
                               extrap_r2_far / extrap_rmse_far dicts.
                               Called inside run_test() after all methods have run.

The helper _runner_eval_formula() that evaluates a formula string against X is
also included verbatim, as compute_extrap_r2_far() depends on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 0.  EXTRAPOLATION REGIME — five-tier quality ladder
#     Matches extrapolation_test_protocol.py and the comments in run_test()
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExtrapolationRegime:
    """A single tier in the extrapolation-quality ladder."""
    name:      str    # e.g. "EXCELLENT"
    threshold: float  # upper bound of this tier (exclusive), % units
    label:     str    # human-readable label for display


#: Ordered from best to worst.  Each tier covers error_pct < threshold.
#: The last entry (CATASTROPHIC) has threshold=inf so it catches everything ≥ 500 %.
REGIMES: List[ExtrapolationRegime] = [
    ExtrapolationRegime("EXCELLENT",    50.0,       "< 50 %  EXCELLENT"),
    ExtrapolationRegime("GOOD",        100.0,       "< 100 % GOOD"),
    ExtrapolationRegime("MODERATE",    200.0,       "< 200 % MODERATE"),
    ExtrapolationRegime("POOR",        500.0,       "< 500 % POOR"),
    ExtrapolationRegime("CATASTROPHIC", float("inf"), "≥ 500 % CATASTROPHIC"),
]


def calculate_extrapolation_error(
    rmse_far: float,
    rmse_train: float,
) -> Optional[float]:
    """
    Compute extrapolation error percentage:  (RMSE_far / RMSE_train) × 100.

    Returns None when either argument is non-finite or rmse_train ≤ 0.
    A value of 100 % means the far-region RMSE exactly equals the training RMSE
    (GOOD tier boundary).
    """
    if (
        rmse_train is None
        or rmse_far is None
        or not math.isfinite(rmse_train)
        or not math.isfinite(rmse_far)
        or rmse_train <= 0.0
    ):
        return None
    return float(rmse_far / rmse_train * 100.0)


def extrapolation_error_status(error_pct: Optional[float]) -> str:
    """
    Map an extrapolation error percentage to its five-tier status label.

    Returns the regime *name* string (e.g. "EXCELLENT") so callers can
    do equality comparisons.  Returns "UNKNOWN" when error_pct is None or
    non-finite.
    """
    if error_pct is None or not math.isfinite(error_pct):
        return "UNKNOWN"
    for regime in REGIMES:
        if error_pct < regime.threshold:
            return regime.name
    return "CATASTROPHIC"  # shouldn't be reached; last tier has threshold=inf


# ──────────────────────────────────────────────────────────────────────────────
# 1.  BUILD EXTRAP SPLIT
#     Source: main() pre-processing loop, lines ~3997-4071
# ──────────────────────────────────────────────────────────────────────────────

def build_extrap_split(
    X: np.ndarray,
    y: np.ndarray,
    description: str = "",
    train_frac: float = 0.8,
    multiplier: float = 2.0,
) -> Tuple[
    np.ndarray,   # X_train
    np.ndarray,   # y_train
    np.ndarray,   # X_far  (may be empty)
    np.ndarray,   # y_far  (may be empty)
    Dict,         # extrap metadata dict (stored in record)
]:
    """
    Re-partition (X, y) for OOD extrapolation evaluation.

    Training covers the first `train_frac` of the sample range (sorted by
    X[:,0]).  The far region is the remainder, clipped at
    x_train_max + multiplier * train_range so the evaluation regime matches
    repro.yaml benchmarks.feynman.extrap_mult=2.0.

    Returns (X_train, y_train, X_far, y_far, metadata).
    X_far / y_far may have length 0 when no samples fall within the multiplier
    boundary — callers must guard with `len(X_far) > 1` before using them.
    """
    # Sort by first variable to get a contiguous training region.
    order    = np.argsort(X[:, 0])
    Xs       = X[order]
    ys       = y[order]
    n        = len(Xs)
    split    = max(1, int(n * train_frac))
    X_train  = Xs[:split]
    y_train  = ys[:split]

    # Multiplier-bounded far region.
    # far_ceiling = x_train_max + multiplier * train_range
    x_train_min  = float(Xs[0, 0])
    x_train_max  = float(Xs[split - 1, 0])
    train_range  = max(x_train_max - x_train_min, 1e-300)
    far_ceiling  = x_train_max + multiplier * train_range

    far_all      = Xs[split:]
    far_y_all    = ys[split:]
    far_mask     = far_all[:, 0] <= far_ceiling
    X_far        = far_all[far_mask]
    y_far        = far_y_all[far_mask]
    n_clipped    = int((~far_mask).sum())

    if n_clipped > 0:
        print(
            f"   ℹ️  '{description[:45]}': clipped {n_clipped} far sample(s) "
            f"beyond {multiplier}× boundary (x>{far_ceiling:.3g})"
        )
    if len(X_far) == 0:
        print(
            f"  ⚠️  '{description[:45]}': no far samples within "
            f"multiplier={multiplier}× — extrap_r2_far will be null"
        )

    metadata = {
        "extrap":            True,
        "extrap_train_frac": train_frac,
        "extrap_multiplier": multiplier,
        "extrap_n_train":    split,
        "extrap_n_test":     len(X_far),
        "extrap_x_train_max": x_train_max,
        "extrap_far_ceiling": far_ceiling,
    }

    return X_train, y_train, X_far, y_far, metadata


# ──────────────────────────────────────────────────────────────────────────────
# 2.  _RUNNER_EVAL_FORMULA  (dependency of compute_extrap_r2_far)
#     Source: BaseMethod._runner_eval_formula(), lines ~776-887
# ──────────────────────────────────────────────────────────────────────────────

def _runner_eval_formula(
    python_code: str,
    X: np.ndarray,
    var_names: List[str],
) -> Optional[np.ndarray]:
    """
    Evaluate *python_code* as a numpy expression that maps X columns to a 1-D
    prediction array.

    Supports three surface forms:
      • Bare expression:   ``(x0 * 50.0) / (x0 + 10.0)``
      • Assignment form:   ``y = (x0 * 50.0) / (x0 + 10.0)``
      • Def form:          ``def formula(x0, k): return x0 * k / (x0 + k)``

    Returns a numpy array of length len(X), or None on any failure.
    """
    try:
        import scipy.special as _spsp
    except ImportError:
        _spsp = None

    safe_globals: Dict[str, Any] = {
        "__builtins__": {},
        "np": np,
        "numpy": np,
        "math": math,
        "pi": np.pi,
        "e":  np.e,
        "inf": np.inf,
        "nan": np.nan,
        # common numpy ufuncs as bare names
        "exp":    lambda x: np.exp(np.clip(x, -500.0, 500.0)),
        "log":    np.log,
        "log10":  np.log10,
        "log2":   np.log2,
        "sqrt":   np.sqrt,
        "sin":    np.sin,
        "cos":    np.cos,
        "tan":    np.tan,
        "arcsin": lambda x: np.arcsin(np.clip(x, -1.0, 1.0)),
        "arccos": lambda x: np.arccos(np.clip(x, -1.0, 1.0)),
        "arctan": np.arctan,
        "arctan2": np.arctan2,
        "abs":    np.abs,
        "fabs":   np.abs,
        "floor":  np.floor,
        "ceil":   np.ceil,
        "sign":   np.sign,
        "power":  np.power,
        "tanh":   np.tanh,
        "sinh":   np.sinh,
        "cosh":   np.cosh,
        "erf":    (np.vectorize(math.erf) if _spsp is None else _spsp.erf),
        "erfc":   (np.vectorize(math.erfc) if _spsp is None else _spsp.erfc),
    }
    if _spsp is not None:
        safe_globals["scipy"]   = type("m", (), {"special": _spsp})()
        safe_globals["special"] = _spsp

    # Inject each variable as the corresponding X column.
    local_ns: Dict[str, Any] = {}
    for i, vn in enumerate(var_names):
        local_ns[vn] = X[:, i] if X.ndim == 2 else X

    code   = python_code.strip()
    y_pred = None

    # Strategy 1: bare expression
    try:
        y_pred = eval(code, safe_globals, local_ns)  # noqa: S307
    except (SyntaxError, Exception):
        pass

    # Strategy 2: assignment form — execute and grab last assigned var
    if y_pred is None:
        try:
            exec_ns = {**safe_globals, **local_ns}  # noqa: S102
            exec(code, exec_ns)                      # noqa: S102
            for candidate in ("y", "result", "output", "pred", "f"):
                if candidate in exec_ns and isinstance(
                    exec_ns[candidate], (np.ndarray, float, int)
                ):
                    y_pred = exec_ns[candidate]
                    break
        except Exception:
            pass

    # Strategy 3: def form — find the first callable and call it
    if y_pred is None and "def " in code:
        try:
            exec_ns: Dict[str, Any] = dict(safe_globals)
            exec(code, exec_ns)  # noqa: S102
            fn = next(
                (v for k, v in exec_ns.items() if callable(v) and k != "__builtins__"),
                None,
            )
            if fn is not None:
                fn_args = [local_ns[vn] for vn in var_names]
                y_pred  = fn(*fn_args)
        except Exception:
            pass

    if y_pred is None:
        return None

    arr = np.asarray(y_pred, dtype=float).flatten()
    if len(arr) != len(X):
        return None
    return arr


# ──────────────────────────────────────────────────────────────────────────────
# 3.  COMPUTE extrap_r2_far
#     Source: run_test() inner block, lines ~2868-2981
# ──────────────────────────────────────────────────────────────────────────────

def _far_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """
    R² on the far region with the same sign-flip correction used throughout
    the benchmark (BaseMethod._safe_r2).

    Returns None when the prediction is degenerate (non-finite, near-constant,
    or fewer than 2 samples).
    Returns float("-inf") when R² < -100 (pathological divergence).
    """
    y_pred = np.asarray(y_pred, dtype=float)
    if np.any(~np.isfinite(y_pred)) or np.any(np.abs(y_pred) > 1e100):
        return None
    if np.std(y_pred) < 1e-30:
        return None
    if len(y_true) < 2:
        return None

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    _scale = float(np.max(np.abs(y_true)) ** 2) * len(y_true)
    _tol   = 1e-10 * _scale if _scale > 0 else 1e-30

    if ss_tot < _tol:
        return 1.0 if ss_res < _tol else float("-inf")

    r2 = float(1 - ss_res / ss_tot)

    # Sign-flip correction: if negating the prediction improves R², use that.
    if r2 < 0:
        ss_flip  = np.sum((y_true - (-y_pred)) ** 2)
        r2_flip  = float(1 - ss_flip / ss_tot)
        if r2_flip > r2:
            r2 = r2_flip

    return r2 if r2 >= -100 else float("-inf")


def _far_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if not np.all(np.isfinite(y_pred)):
        return None
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_extrap_r2_far(
    results: Dict[str, Any],                                # {method_name: MethodResult}
    X_far: Optional[np.ndarray],
    y_far: Optional[np.ndarray],
    var_names: List[str],
    y_train: Optional[np.ndarray] = None,                   # training targets for RMSE_train
    y_pred_train: Optional[Dict[str, np.ndarray]] = None,   # {method: train predictions}
    verbose: bool = True,
) -> Tuple[
    Dict[str, Optional[float]],   # extrap_r2_far
    Dict[str, Optional[float]],   # extrap_rmse_far
    Dict[str, Optional[float]],   # extrap_error_pct  (RMSE_far / RMSE_train × 100)
]:
    """
    Re-evaluate each method's formula on the held-out far region and return
    ``(extrap_r2_far, extrap_rmse_far, extrap_error_pct)`` dicts keyed by method name.

    Called *after* all methods have run (results already populated), inside
    run_test(), before the record is assembled.

    Parameters
    ----------
    results       : ``{name: MethodResult}`` — each entry must expose ``.success``
                    and ``.formula``.
    X_far         : held-out far-region features, shape ``(n_far, n_vars)``.
    y_far         : held-out far-region targets, shape ``(n_far,)``.
    var_names     : variable names matching X columns (e.g. ``["x0", "x1"]``).
    y_train       : training targets, used together with ``y_pred_train`` to
                    compute RMSE_train per method.
    y_pred_train  : ``{method_name: np.ndarray}`` of per-method training
                    predictions pre-evaluated by the caller (only symbolic
                    methods included).  When provided, RMSE_train is derived
                    from these; otherwise ``extrap_error_pct`` is null for all
                    methods.
    verbose       : print per-method far-region R² / error-% summary table.

    Returns
    -------
    extrap_r2_far    : ``{method_name: float | None}``
    extrap_rmse_far  : ``{method_name: float | None}``
    extrap_error_pct : ``{method_name: float | None}``  — (RMSE_far/RMSE_train)×100

    Notes
    -----
    Methods that return NN architecture tags (``"ImprovedNN(…)"``,
    ``"[NN fallback"``, ``"N/A"``, ``""``) are skipped — they have no evaluable
    expression so all three values are null.  Only SymbolicEngineWithLLM and
    HybridDiscoverySystem v50_2 produce formula strings and yield non-null values.
    """
    extrap_r2_far:    Dict[str, Optional[float]] = {}
    extrap_rmse_far:  Dict[str, Optional[float]] = {}
    extrap_error_pct: Dict[str, Optional[float]] = {}

    # Guard: skip entirely when not in extrap mode or far region is degenerate.
    _do_extrap = (
        X_far is not None
        and y_far is not None
        and len(X_far) > 1
        and len(y_far) > 1
    )
    if not _do_extrap:
        return extrap_r2_far, extrap_rmse_far, extrap_error_pct

    for mname, res in results.items():
        # Failed methods → null across the board.
        if not res.success:
            extrap_r2_far[mname]    = None
            extrap_rmse_far[mname]  = None
            extrap_error_pct[mname] = None
            continue

        formula = (res.formula or "").strip()

        # NN / black-box methods return architecture tags — not evaluable.
        is_nn_tag = (
            formula.startswith("ImprovedNN(")
            or formula.startswith("[NN fallback")
            or formula in ("N/A", "")
        )
        if is_nn_tag:
            extrap_r2_far[mname]    = None
            extrap_rmse_far[mname]  = None
            extrap_error_pct[mname] = None
            continue

        # Symbolic methods: re-evaluate on the far region.
        try:
            y_far_pred = _runner_eval_formula(formula, X_far, var_names)
            if y_far_pred is None or len(y_far_pred) != len(y_far):
                extrap_r2_far[mname]    = None
                extrap_rmse_far[mname]  = None
                extrap_error_pct[mname] = None
            else:
                r2f      = _far_r2(y_far, y_far_pred)
                rmse_far = _far_rmse(y_far, y_far_pred) if r2f is not None else None
                extrap_r2_far[mname]   = r2f
                extrap_rmse_far[mname] = rmse_far

                # error_pct = (RMSE_far / RMSE_train) × 100
                error_pct: Optional[float] = None
                if (
                    rmse_far is not None
                    and y_pred_train is not None
                    and y_train is not None
                    and mname in y_pred_train
                ):
                    rmse_train = _far_rmse(y_train, y_pred_train[mname])
                    error_pct  = calculate_extrapolation_error(rmse_far, rmse_train)
                extrap_error_pct[mname] = error_pct

        except Exception:
            extrap_r2_far[mname]    = None
            extrap_rmse_far[mname]  = None
            extrap_error_pct[mname] = None

    if verbose and any(v is not None for v in extrap_r2_far.values()):
        print(
            f"\n  📐 Extrapolation R² on far region (n={len(X_far)} held-out samples):",
            flush=True,
        )
        for mn, r2f in extrap_r2_far.items():
            r2s  = f"{r2f:.4f}" if (r2f is not None and math.isfinite(r2f)) else "null"
            pctr = extrap_error_pct.get(mn)
            pcts = f"{pctr:.1f}%" if (pctr is not None and math.isfinite(pctr)) else "null"
            tier = extrapolation_error_status(pctr)
            print(
                f"     {mn:<42} extrap_r2_far={r2s}  error_pct={pcts}  [{tier}]",
                flush=True,
            )

    return extrap_r2_far, extrap_rmse_far, extrap_error_pct


# ──────────────────────────────────────────────────────────────────────────────
# Usage example (matches how it is wired in the benchmark)
# ──────────────────────────────────────────────────────────────────────────────
#
#   # --- pre-processing loop (before running methods) ---
#   X_train, y_train, X_far, y_far, extrap_meta = build_extrap_split(
#       X, y,
#       description = description,
#       train_frac  = args.extrap_train_frac,   # default 0.8
#       multiplier  = args.extrap_multiplier,    # default 2.0
#   )
#   metadata = {**metadata, **extrap_meta}
#
#   # --- after all methods have run on (X_train, y_train) ---
#   extrap_r2_far, extrap_rmse_far, extrap_error_pct = compute_extrap_r2_far(
#       results      = results,           # {name: MethodResult}
#       X_far        = X_far,
#       y_far        = y_far,
#       var_names    = var_names,
#       y_train      = y_train,           # NEW: needed for error_%
#       y_pred_train = y_pred_train_dict, # NEW: {method: train preds}
#       verbose      = True,
#   )
#
#   # --- assemble record ---
#   record["extrap_r2_far"]      = extrap_r2_far
#   record["extrap_rmse_far"]    = extrap_rmse_far
#   record["extrap_error_pct"]   = extrap_error_pct
#   record["extrap_train_frac"]  = metadata.get("extrap_train_frac")
#   record["extrap_multiplier"]  = metadata.get("extrap_multiplier")
#   record["extrap_n_train"]     = metadata.get("extrap_n_train")
#   record["extrap_n_test"]      = metadata.get("extrap_n_test")
#   record["extrap_x_train_max"] = metadata.get("extrap_x_train_max")
#   record["extrap_far_ceiling"] = metadata.get("extrap_far_ceiling")
