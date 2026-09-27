#!/usr/bin/env python3
"""
hypatiax_defi_benchmark_pca.py
================================
HypatiaX DeFi Benchmark — PCA-directed 40/60 split variant (FIX-C3)

This is a direct derivative of hypatiax_defi_benchmark_v4.py.
All logic (74 cases, LLM/NN/Hybrid methods, report) is identical to v4.0,
including the validation-selected hybrid (SECTION 1B) and the Fix-12
truncated-formula guard.

The single intentional change vs v4 is the data-split protocol applied to
every test case in run_benchmark():

  OLD (v4):  build_extrap_split() / _aggressive_split() — percentile cut
             on a single configured feature axis
  NEW (pca): pca_directed_split() — sort by PC1, take lowest 40% as train

This mirrors the FIX-C3 split used in the Feynman benchmark
(run_comparative_suite_benchmark_pca.py §10.7) so that the DeFi
extrapolation results (§6.4) are produced under the same protocol
and are directly comparable.

pca_directed_split is imported from hypatiax.tools.utils.pca_split_utils
(see SECTION 6 below) — it is NOT inlined in this file. It is the same
reference implementation verified by Gate A of ci_runner_disclosure.yml.

v4 sync (2026)
──────────────
Ported from hypatiax_defi_benchmark_v4.py, in addition to Fix 14
(ground-truth-leak removal, already present in this file since v3.2):

  SECTION 1B  Validation-selected hybrid. Replaces the old cascading
              routing logic (transcendental-token / extrapolation-probe
              overrides) with an internal-validation contest between
              {llm, nn, residual_nn, blend} candidates, selected using
              training data only (never the final test set). Hybrid
              `decision` values are now "v4_llm" / "v4_nn" /
              "v4_residual_nn" / "v4_blend" / "v4_nn_fallback".
  Fix 12      _is_truncated_formula() guard on the pure_llm arm — a
              formula that ends mid-line with no valid `return` cannot
              have executed correctly, and any R² recorded against it
              is invalid.
  Mean-by-seed  _generate_pooled_seed_report() (ported from v4) is now
              called automatically at the end of a multi-seed sweep,
              writing hypatiax_defi_benchmark_pca_pooled_seed_report.json.

This file additionally keeps two audit fields v4 dropped: `llm_model`
and `model_used` on the hybrid result, so a run mixing model versions
stays auditable (see SECTION 5B). These are purely additive and do not
change any R² or decision value relative to v4.

Fix 16  Ranked validation fallback after full-fit candidate failure
Fix 15  Hybrid fallback discarded a trustworthy LLM prediction (ported
          from v4.0). When the selected candidate (residual_nn / blend)
          produced NaN/Inf on the extrapolative test range, the recovery
          path always refit a fresh bare NN ("nn_fallback"), discarding
          llm_test even when it was already computed and trustworthy.
          Fallback now prefers the trustworthy LLM prediction
          ("llm_fallback") and only refits a bare NN when no trustworthy
          LLM prediction is available. `model_used` audit trail updated
          to map "llm_fallback" to the LLM model name.

Fix 18  Extrapolation guard was defeated by blend(alpha=0):
          - Old: _select_v4_candidate() excluded bare-NN keys ("nn:*") when the
            test domain was extrapolative, but the blend grid contained
            alpha=0.0, and blend(alpha=0) is the bare NN. Blend's validation
            R2 is >= the NN's by construction and wins ties, so whenever a
            trusted (train_r2 > 0.5) but imperfect formula existed and an NN
            scored well on the in-domain validation split, the selector
            returned "blend, alpha=0.00" -- a bare NN that had passed the guard
            as "LLM-anchored", reported as model_used="blend(...,alpha=0.00)"
            with extrapolation_unmitigated=False.
          - New: when extrapolative, the blend grid is restricted to
            alpha >= _V4_EXTRAP_MIN_BLEND_ALPHA (0.5). Non-extrapolative
            selection is unchanged.
          - Ported in lockstep with hypatiax_defi_benchmark_v4.py (the selector
            code is identical between the two variants).

Fix 19  NaN-safe log/sqrt/exp wrappers were dead code for np.-prefixed calls:
          - Old: _EXEC_GLOBALS gave LLM-generated formulas both a raw numpy
            module under "np"/"numpy" AND separate NaN-safe bare-name wrappers
            ("log", "sqrt", "exp", ...). Since the formula-generation prompt
            tells the model to "Use numpy (imported as np)", generated code
            calling np.log(...)/np.sqrt(...) resolved via attribute lookup
            straight to the real, unguarded numpy function, bypassing the
            wrapper of the same name in the same globals dict.
          - New: "np"/"numpy" are now a thin proxy (_SafeNumpyNamespace) that
            overrides log/log2/log10/sqrt/exp with the same NaN-safe
            implementations as the bare-name wrappers and forwards every other
            attribute unchanged. np.log(x) and log(x) now behave identically.
          - Ported in lockstep with hypatiax_defi_benchmark_v4.py.

Fix 20  "np.inf" false-positive in _formula_has_pathological_behavior:
          - Old: "np.inf" was blacklisted as an "unambiguous" degenerate
            pattern, but it's also the correct way to express a genuine
            mathematical singularity guarded by np.where (e.g. a breakeven
            fee rate that's infinite at zero volume share). A formula fitting
            training data at r2≈1.0 was rejected on this text match alone.
          - New: "np.inf" removed from the blacklist; _execute_formula's
            nan_frac check (based on actual output values, not source text)
            already provides equivalent protection against genuinely
            unguarded inf-blowups. "1/0" and "**1000" unchanged.
          - Ported in lockstep with hypatiax_defi_benchmark_v4.py.

Output files:
  hypatiax_defi_benchmark_pca_checkpoint.json
  hypatiax_defi_benchmark_pca_results.json
  hypatiax_defi_benchmark_pca_pooled_seed_report.json  (multi-seed sweeps only)
  split_protocol_disclosure.json  (written on completion)

Author : HypatiaX Team
Version: 2.0 — PCA 40/60 split variant of v4.0 (validation-selected hybrid)
Date   : 2026
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import json
import math as _math
import os
import re
import sys
from pathlib import Path

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from scipy import stats
from sklearn.preprocessing import StandardScaler

# ── project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))

# ── env / API key ─────────────────────────────────────────────────────────────
for _ep in [
    _ROOT / "hypatiax" / ".env",
    _ROOT / ".env",
    Path.cwd() / "hypatiax" / ".env",
    Path.cwd() / ".env",
]:
    if _ep.exists():
        load_dotenv(dotenv_path=_ep)
        print(f"✅ Loaded .env from: {_ep}")
        break

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42

import random
import time

# ── CASE RANGE INJECTION (auto-generated by add_case_range_benchmark.py) ──
def _apply_case_range(seq):
    """Return the slice of *seq* selected by CASE_RANGE_START/CASE_RANGE_END.

    Uses 1-based inclusive indexing to match CI --case-range N-M syntax.
    Returns seq unchanged when neither variable is set (local runs).
    """
    import os
    try:
        n = len(seq)
        start = max(0, int(os.getenv("CASE_RANGE_START", "1")) - 1)
        end   = min(n, int(os.getenv("CASE_RANGE_END",   str(n))))
        return seq[start:end]
    except Exception:
        return seq
# ────────────────────────────────────────────────────────────────────────────

# ── TASK_IDS / SEED injection (auto-generated by ci_experiment.yml) ─────────
def _apply_task_ids_defi(test_cases):
    """Filter DeFi test_cases against TASK_IDS env-var set by the CI worker shard.

    Two-stage matching (name → domain):

    Stage 1 — exact name match (fine-grained, case-level filtering).
        Used when TASK_IDS contains full case names, e.g.:
            TASK_IDS="Black-Scholes Call Price Correlated Portfolio VaR"
        Returns only the matching cases.

    Stage 2 — domain key match (coarse, domain-level filtering).
        Used by exp1 / suppA where TASK_IDS contains the 10 domain keys, e.g.:
            TASK_IDS="amm risk_var liquidity"
        Stage 1 produces 0 hits (no case name equals a domain key), so Stage 2
        filters tc['domain'] to the shard-assigned subset instead.
        After this fix, v3c.py's own domain filter is coherent with the
        protocol-level SHARD_IDS filter in
        experiment_protocol_defi._apply_shard_ids().  Both layers now
        independently select the correct domain subset.

    Silent fallback — no warning.
        For exp1b, TASK_IDS contains synthetic checkpoint-tracking IDs such as
        "portfolio_seed42".  These intentionally match nothing by name or domain;
        the correct case filter for exp1b is DEFI_TASK_FILTER="Portfolio" read
        in run_benchmark().  Emitting a RuntimeWarning here would be actively
        misleading, so the fallback is silent.

    Falls back to the full list when TASK_IDS is unset (local / Colab runs).
    """
    import os
    raw = os.environ.get("TASK_IDS", "").replace(",", " ").split()
    if not raw:
        return test_cases   # unset → local / Colab run, no filtering

    allowed = set(raw)

    # Stage 1: exact name match — fine-grained, case-level
    by_name = [tc for tc in test_cases if tc.get("name") in allowed]
    if by_name:
        return by_name

    # Stage 2: domain key match — coarse, domain-level (exp1 / suppA)
    by_domain = [tc for tc in test_cases if tc.get("domain") in allowed]
    if by_domain:
        return by_domain

    # Nothing matched — TASK_IDS are synthetic checkpoint IDs (exp1b pattern)
    # or genuinely stale.  Silent fallback: actual case filtering is handled
    # by DEFI_TASK_FILTER in run_benchmark().
    return test_cases

def _resolve_seed():
    """Return seed: PYSR_SEED → EXPERIMENT_SEED → NN_SEED → module SEED (42)."""
    import os
    for var in ("PYSR_SEED", "EXPERIMENT_SEED", "NN_SEED"):
        v = os.environ.get(var, "").strip()
        if v:
            try:
                return int(v)
            except ValueError:
                pass
    return None  # sentinel → caller keeps its own default
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

# ── protocol import ───────────────────────────────────────────────────────────
try:
    from hypatiax.protocols.experiment_protocol_defi import DeFiExperimentProtocol
    print("✅ Loaded experiment_protocol_defi.py")
except ImportError as _e:
    print(f"❌ Cannot import DeFiExperimentProtocol: {_e}")
    sys.exit(1)

# ── output paths ──────────────────────────────────────────────────────────────
# Respect OUT_BASE env var when set by CI (ci_experiment_simplify.yml).
# If OUT_BASE is set, write into OUT_BASE/RESULT_SUBDIR (canonical CI path).
# RESULT_SUBDIR defaults to the noiseless subdir matching the plan job metadata.
# When running locally (no OUT_BASE), behaviour is unchanged.
# --output-dir CLI flag overrides all of the above when provided.
_OUT_BASE      = os.environ.get("OUT_BASE", "").strip()
_RESULT_SUBDIR = os.environ.get("RESULT_SUBDIR", "comparison_results/noise-noiseless/noiseless").strip()
if _OUT_BASE:
    RESULTS_DIR = Path(_OUT_BASE) / _RESULT_SUBDIR
else:
    RESULTS_DIR = Path("hypatiax/data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = RESULTS_DIR / "hypatiax_defi_benchmark_pca_checkpoint.json"
FINAL_OUTPUT    = RESULTS_DIR / "hypatiax_defi_benchmark_pca_results.json"


def _configure_output_dir(output_dir: str | None) -> None:
    """Override the module-level output paths when --output-dir is supplied.

    This must be called before any function that reads CHECKPOINT_FILE or
    FINAL_OUTPUT (i.e. before run_benchmark / report_only).
    """
    if output_dir is None:
        return
    global RESULTS_DIR, CHECKPOINT_FILE, FINAL_OUTPUT
    RESULTS_DIR     = Path(output_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE = RESULTS_DIR / "hypatiax_defi_benchmark_pca_checkpoint.json"
    FINAL_OUTPUT    = RESULTS_DIR / "hypatiax_defi_benchmark_pca_results.json"
    print(f"📁 Output dir overridden via --output-dir: {RESULTS_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Neural network (self-contained, no external NN import needed)
# ─────────────────────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    """Small MLP: LayerNorm + SiLU, no Dropout (too few training samples)."""
    def __init__(self, in_dim: int, hidden: list[int] = None):
        super().__init__()
        hidden = hidden or [128, 64, 32]
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Model-identity constants ─────────────────────────────────────────────────
# Single source of truth for which model produced which result. Every result
# record below tags itself with one of these strings (or a composite of them
# for ensemble/fallback paths) so that runs mixing models are auditable after
# the fact, rather than silently assuming a single model throughout.
_HYBRID_LLM_MODEL_NAME = "claude-sonnet-4-6"          # model used inside _generate_llm_formula
_NN_MODEL_NAME          = "mlp_128_64_32"             # fixed MLP architecture used by _train_and_eval_nn

_NN_SEED = 2024   # fixed seed → deterministic NN scores across resume sessions


_NN_MAX_TIME_S = 120  # Wall-clock cap per NN training run (Issue 2 fix).
                      # Prevents runaway convergence failures like Portfolio Sharpe
                      # Ratio (29,088 s in the reported run, 96.7% of total NN time).
                      # 120 s is generous for a 200-sample MLP; consistent with the
                      # PySR timeout_in_seconds used in Experiments 1–3.


def _compute_augment_plan(X_train: np.ndarray) -> dict:
    """
    FIX (feature-count mismatch): decide which augmented columns to add
    from the TRAINING split ONLY, and return a fixed plan. The old
    _augment_features() evaluated np.all(xi > 0) / np.all(xi >= 0)
    independently on whatever array it was given, so train and test —
    which can have systematically different value ranges under the
    PCA-directed 40/60 split — could qualify a different number of
    columns for log/sqrt augmentation. That produced train/test feature
    matrices of different width, which StandardScaler then rejected
    ("X has 9 features, but StandardScaler is expecting 7 features").
    Deciding the plan from X_train alone and applying it identically to
    every other split guarantees a fixed, split-independent column count.
    """
    plan = {"log_cols": [], "sqrt_cols": [], "ratio": X_train.shape[1] == 2}
    for i in range(X_train.shape[1]):
        xi = X_train[:, i]
        if np.all(xi > 0):
            plan["log_cols"].append(i)
        if np.all(xi >= 0):
            plan["sqrt_cols"].append(i)
    return plan


def _apply_augment_plan(X: np.ndarray, plan: dict) -> np.ndarray:
    """
    Apply a previously computed augmentation plan (see _compute_augment_plan)
    to X — train or test — so every split produces the same number of
    columns regardless of that split's own values. Values are clipped
    before log/sqrt: a split (typically test, under PCA split) may contain
    values outside the range that qualified the column on train (e.g. a
    column that was all-positive on train but dips <= 0 on test); clipping
    keeps the column finite instead of reintroducing NaN and silently
    failing training/evaluation downstream.
    """
    cols = [X]
    eps = 1e-8
    for i in plan["log_cols"]:
        cols.append(np.log(np.clip(X[:, i], eps, None)).reshape(-1, 1))
    for i in plan["sqrt_cols"]:
        cols.append(np.sqrt(np.clip(X[:, i], 0.0, None) + eps).reshape(-1, 1))
    if plan["ratio"]:
        cols.append((X[:, 0] / (X[:, 1] + eps)).reshape(-1, 1))
        cols.append((X[:, 1] / (X[:, 0] + eps)).reshape(-1, 1))
    return np.hstack(cols)


def _augment_features(X: np.ndarray) -> np.ndarray:
    """
    DEPRECATED — kept only for any external caller that still imports this
    name directly on a single array. Internal training/eval now uses
    _compute_augment_plan(X_train) + _apply_augment_plan(X, plan) so train
    and test always get the same column layout. Calling this function
    directly on train and test separately reintroduces the original bug —
    do not use it that way.
    """
    return _apply_augment_plan(X, _compute_augment_plan(X))


def _train_and_eval_nn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    epochs: int = 300,
    hidden: list[int] = None,
    seed: int = _NN_SEED,
    max_time_s: float = _NN_MAX_TIME_S,
    augment: bool = True,
) -> dict:
    """
    Train a small MLP on (X_train, y_train) and evaluate on both splits.
    Returns dict: train_r2, test_r2, success, y_pred_train, y_pred_test.

    max_time_s: hard wall-clock limit.  Training stops early if exceeded.
    The early-stop flag is returned as 'timed_out' so callers can log it.
    augment: if True (default), adds log/sqrt/ratio features before training
             (v3c2-fix4: physics-informed feature augmentation).
    """
    torch.manual_seed(seed); np.random.seed(seed)
    hidden = hidden or [128, 64, 32]

    # FIX (crash-to-success:False): the entire body is now wrapped in
    # try/except so a training blow-up (non-finite values, a singular
    # matrix, a bad augment plan, etc.) degrades to a success:False result
    # instead of crashing the whole benchmark run — matching the Pure LLM
    # baseline's error-handling contract. The returned dict always has the
    # same keys (train_r2, test_r2, y_pred_train, y_pred_test) so existing
    # unconditional callers (nn_m["test_r2"], etc.) keep working unchanged;
    # only "success" and the new "error"/"error_type" keys signal failure.
    try:
        # v3c2-fix4: augment features BEFORE scaling
        # FIX (feature-count mismatch): plan is derived from X_train only and
        # applied identically to X_test, so the two splits always produce the
        # same number of columns (see _compute_augment_plan/_apply_augment_plan).
        if augment:
            _plan   = _compute_augment_plan(X_train)
            X_train = _apply_augment_plan(X_train, _plan)
            X_test  = _apply_augment_plan(X_test, _plan)

        if not (np.all(np.isfinite(X_train)) and np.all(np.isfinite(y_train))
                and np.all(np.isfinite(X_test)) and np.all(np.isfinite(y_test))):
            raise ValueError("Non-finite values in inputs before NN training")

        sx, sy = StandardScaler(), StandardScaler()
        Xtr = sx.fit_transform(X_train)
        ytr = sy.fit_transform(y_train.reshape(-1, 1)).flatten()

        model = _MLP(X_train.shape[1], hidden)
        opt   = torch.optim.Adam(model.parameters(), lr=0.001)
        crit  = nn.MSELoss()
        Xt = torch.FloatTensor(Xtr)
        yt = torch.FloatTensor(ytr).reshape(-1, 1)

        timed_out  = False
        _wall_start = time.time()
        model.train()
        for epoch in range(epochs):
            opt.zero_grad(); loss = crit(model(Xt), yt); loss.backward(); opt.step()
            # Check wall-clock every 25 epochs to avoid per-epoch overhead
            if epoch % 25 == 0 and (time.time() - _wall_start) >= max_time_s:
                timed_out = True
                print(f"    ⏱  NN wall-clock limit ({max_time_s}s) reached at epoch {epoch} — stopping early")
                break

        def _decode(raw): return sy.inverse_transform(raw.reshape(-1, 1)).flatten()
        def _r2(yt_, yp_):
            # FIX (magnitude-scaled threshold): mirrors the Pure LLM
            # baseline's scale = max(|y_true|)**2 * len(y) fix so tiny-y
            # equations (Photon ~1e-19, Zeeman ~1e-23) aren't misclassified
            # as constant targets by a flat 1e-10 cutoff.
            ss_r = np.sum((yt_ - yp_) ** 2); ss_t = np.sum((yt_ - yt_.mean()) ** 2)
            _tol = 1e-10 * (np.max(np.abs(yt_)) ** 2) * len(yt_)
            return float(1 - ss_r / ss_t) if ss_t > _tol else 0.0

        model.eval()
        with torch.no_grad():
            yp_tr = _decode(model(Xt).numpy().flatten())
            Xte   = torch.FloatTensor(sx.transform(X_test))  # X_test already augmented above
            yp_te = _decode(model(Xte).numpy().flatten())

        if not (np.all(np.isfinite(yp_tr)) and np.all(np.isfinite(yp_te))):
            raise ValueError("Non-finite predictions after inverse-transform")

        return {
            "train_r2":    _r2(y_train, yp_tr),
            "test_r2":     _r2(y_test,  yp_te),
            "success":     True,
            "timed_out":   timed_out,
            "y_pred_train": yp_tr,
            "y_pred_test":  yp_te,
        }
    except Exception as e:
        print(f"    ⚠  NN training/eval failed: {type(e).__name__}: {e}")
        return {
            "train_r2":    0.0,
            "test_r2":     0.0,
            "success":     False,
            "timed_out":   False,
            "error":       str(e),
            "error_type":  type(e).__name__,
            "y_pred_train": np.zeros_like(np.asarray(y_train, dtype=float)),
            "y_pred_test":  np.zeros_like(np.asarray(y_test, dtype=float)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Fix 5: UNIFIED formula evaluator (single code path)
# ─────────────────────────────────────────────────────────────────────────────

def compute_moneyness(S, K, mode="log"):
    """
    Correct moneyness definition.
    mode="log"   → preferred (Black-Scholes consistent, log-normal assumption)
    mode="ratio" → fallback (simple S/K, breaks symmetry)
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    if mode == "log":
        return np.log(np.where(K > 0, S / K, np.nan))
    elif mode == "ratio":
        return S / K
    else:
        raise ValueError(f"Unknown moneyness mode: {mode!r}")


# FIX 19 (ported from hypatiax_defi_benchmark_v4.py; see that module's
# docstring for the full note): NaN-safe implementations shared between the
# bare-name wrappers ("log", "sqrt", "exp", ...) below AND the "np"/"numpy"
# proxy, so that np.log(x) and log(x) are guaranteed to behave identically no
# matter which spelling LLM-generated code happens to use. The
# formula-generation prompt tells the model to use "np.", so a wrapper that
# only intercepts the bare name is a guard the generated code almost never
# actually passes through -- np.log(x) resolved via attribute lookup straight
# to the real, unguarded numpy.log, raising
# "RuntimeWarning: invalid value encountered in log" and producing -inf/nan
# at exactly the domain boundaries (Black-Scholes d1/d2, leverage ratios,
# moneyness, ...) the wrapper exists to guard.
_SAFE_MATH_OVERRIDES = {
    "exp":   lambda x: np.exp(np.clip(x, -500.0, 500.0)),
    "log":   lambda x: np.log(np.where(np.asarray(x) > 0, x, np.nan)),
    "log2":  lambda x: np.log2(np.where(np.asarray(x) > 0, x, np.nan)),
    "log10": lambda x: np.log10(np.where(np.asarray(x) > 0, x, np.nan)),
    "sqrt":  lambda x: np.sqrt(np.where(np.asarray(x) >= 0, x, np.nan)),
}


class _SafeNumpyNamespace:
    """FIX 19: proxy for the numpy module exposed to LLM-generated formula
    code as "np"/"numpy". Overrides log/log2/log10/sqrt/exp with the
    NaN-safe versions in _SAFE_MATH_OVERRIDES; every other attribute is
    forwarded unchanged to the real numpy module."""

    def __init__(self, real_module, overrides):
        object.__setattr__(self, "_real", real_module)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name):
        override = self._overrides.get(name)
        return override if override is not None else getattr(self._real, name)

    def __repr__(self):
        return f"_SafeNumpyNamespace({self._real!r})"


_SAFE_NP = _SafeNumpyNamespace(np, _SAFE_MATH_OVERRIDES)

_EXEC_GLOBALS = {
    "np": _SAFE_NP, "numpy": _SAFE_NP, "math": _math,
    "pi": np.pi, "e": np.e,
    "sin":     np.sin,   "cos":    np.cos,
    "tan":     np.tan,   "arcsin": lambda x: np.arcsin(np.clip(x, -1, 1)),
    "arccos":  lambda x: np.arccos(np.clip(x, -1, 1)),
    "arctan":  np.arctan, "arctan2": np.arctan2,
    "abs":     np.abs,   "sign":   np.sign,  "tanh":  np.tanh,
    "sinh":    np.sinh,  "cosh":   np.cosh,
    "minimum": np.minimum, "maximum": np.maximum, "clip": np.clip,
    # Issue 5 fix: NaN-safe normal CDF / PDF wrappers.
    # LLM-generated Black-Scholes / Greeks code uses these names.
    # The clipping ensures out-of-range inputs don't propagate NaN silently.
    "norm":    stats.norm,
    "norm_cdf": lambda x: stats.norm.cdf(np.clip(np.asarray(x, dtype=float), -37.0, 37.0)),
    "norm_pdf": lambda x: stats.norm.pdf(np.clip(np.asarray(x, dtype=float), -37.0, 37.0)),
    # FIX 6: correct moneyness (log-normal / Black-Scholes consistent)
    "moneyness": compute_moneyness,
    **_SAFE_MATH_OVERRIDES,  # bare "log"/"sqrt"/"exp"/... names, same impls as the np proxy
}


def _execute_formula(llm_code: str, X: np.ndarray,
                     constants: dict = None) -> np.ndarray | None:
    """
    Fix 5 — unified formula execution.
    ALL LLM formula calls go through this function.
    Returns np.ndarray of predictions, or None on any failure.
    constants: dict of scalar values to inject (e.g. {"K": 100.0}) so
               formulas that reference protocol constants don't NameError.
    """
    exec_globals = _EXEC_GLOBALS.copy()
    if constants:
        exec_globals.update(constants)
    local = {}
    try:
        exec(llm_code, exec_globals, local)
    except Exception:
        return None

    func = next((v for v in local.values() if callable(v)), None)
    if func is None:
        return None

    try:
        if X.ndim == 1 or X.shape[1] == 1:
            preds = func(X[:, 0] if X.ndim > 1 else X)
        else:
            args = [X[:, i] for i in range(X.shape[1])]
            try:
                preds = func(*args)
            except Exception:
                preds = np.array([func(*X[i]) for i in range(len(X))])
        preds = np.asarray(preds, dtype=float).flatten()
        # v3c2-fix2: allow up to 10% NaN/inf (e.g. log(0) at domain boundaries)
        nan_frac = np.mean(np.isnan(preds) | np.isinf(preds))
        if nan_frac > 0.1:
            return None
        # Replace remaining boundary NaN/inf with median for partial evaluation
        if nan_frac > 0:
            med = float(np.nanmedian(preds))
            preds = np.where(np.isfinite(preds), preds, med)
        # FIX 6: reject degenerate constant functions → useless predictions
        # Applied AFTER NaN replacement so we don't reject partially-valid formulas.
        if np.std(preds) < 1e-12:
            return None
        return preds
    except Exception:
        return None


# FIX 12 (ported from run_comparative_suite_benchmark_v2.py via v4.0): guard
# against truncated PureLLM formulas — code that ends mid-line with no valid
# `return <something>` cannot have executed correctly, and any R² recorded
# against it is invalid (root cause of the 100% recovery artefact in the
# March 2026 run, where 11/30 truncated formulas all scored R² ≈ 0.9976
# because the harness fell back to a cached value instead of failing).
#
# IMPORTANT: only `def`-form code requires a `return` statement. Bare
# expression / assignment-form code (no `def` at all) is a normal, complete
# surface form for PureLLMBaseline's output and must not be flagged as
# truncated just because it has no `return` line.
def _is_truncated_formula(code: str) -> bool:
    """Return True if code is incomplete / cannot have been executed."""
    if not code or not code.strip():
        return True
    if "def " not in code:
        return False          # bare-expression / assignment form — no return required
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("return"):
            rest = stripped[len("return"):].strip()
            if rest:          # return <something> — complete
                return False
    return True               # def-form with no valid return found


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    FIX 9: Robust metric suite beyond R² alone.
    Returns r2, mae, rmse, mape.  All values are floats.
    """
    err  = y_true - y_pred
    ss_r = np.sum(err ** 2)
    ss_t = np.sum((y_true - y_true.mean()) ** 2)
    r2   = float(1 - ss_r / ss_t) if ss_t > 1e-10 else 0.0
    return {
        "r2":   r2,
        "mae":  float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err / (np.abs(y_true) + 1e-8)))),
    }


def _eval_formula_r2(
    llm_code: str, X: np.ndarray, y_true: np.ndarray,
    constants: dict = None,
) -> tuple[float, bool]:
    """
    Fix 5 — evaluate LLM formula and return (r2, success).
    Returns (nan, False) on any failure.
    constants: forwarded to _execute_formula for NameError-free evaluation.
    """
    preds = _execute_formula(llm_code, X, constants=constants)
    if preds is None:
        return float("nan"), False
    # FIX 9: use _compute_metrics for consistency; return r2 + success
    metrics = _compute_metrics(y_true, preds)
    return metrics["r2"], True


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Routing helpers (Fixes 1, 2, 4)
# ─────────────────────────────────────────────────────────────────────────────

_TRANSCENDENTAL_TOKENS = [
    "math.exp", "np.exp", "exp(",
    "math.log", "np.log", "log(",
    "math.sqrt", "np.sqrt", "sqrt(",
    "norm.cdf", "norm.pdf", "scipy.stats",
    "np.maximum", "np.minimum", "max(", "min(",
    "math.sin", "np.sin", "math.cos", "np.cos",
    "**0.",
]


def _formula_has_transcendental(code: str) -> bool:
    """Fix 2 — True if code contains ops that are structurally bad for NN extrapolation."""
    low = code.lower()
    return any(t.lower() in low for t in _TRANSCENDENTAL_TOKENS)


def _sort_by_principal_direction(X: np.ndarray) -> np.ndarray:
    """
    FIX 8: Sort samples along the first principal component rather than
    a single feature axis.  This gives a correct extrapolation split for
    multivariate inputs where the primary direction of variation is not
    aligned with any single feature.
    """
    Xc = X - X.mean(axis=0)
    _, _, vh = np.linalg.svd(Xc, full_matrices=False)
    principal = vh[0]
    scores = Xc @ principal
    return np.argsort(scores)


def _extrapolation_probe(X_train: np.ndarray, y_train: np.ndarray,
                          probe_frac: float = 0.15) -> float:
    """
    Fix 1 — measure how much an MLP degrades at the edge of training data.
    Returns degradation = (in-distribution R²) - (probe R²).  Large → route to LLM.
    """
    n      = len(X_train)
    split  = int(n * (1 - probe_frac))
    # FIX 8: use principal direction for ordering instead of X[:, 0]
    order  = _sort_by_principal_direction(X_train)
    Xs, ys = X_train[order], y_train[order]
    Xf, yf = Xs[:split], ys[:split]
    Xp, yp = Xs[split:], ys[split:]

    if len(Xp) < 5 or len(Xf) < 10:
        return 0.0

    try:
        probe_result = _train_and_eval_nn(
            Xf, yf, Xp, yp, epochs=200, hidden=[64, 32], seed=_NN_SEED
        )
        return max(0.0, probe_result["train_r2"] - probe_result["test_r2"])
    except Exception:
        return 0.0


def _distance_llm_weight(X_test: np.ndarray, X_train: np.ndarray,
                          base_weight: float = 0.3) -> float:
    """
    Fix 4 — LLM blend weight: increases as test points move outside training range.
    Returns scalar in [base_weight, 1.0].
    """
    lo, hi  = X_train.min(axis=0), X_train.max(axis=0)
    outside = np.mean(np.any((X_test < lo) | (X_test > hi), axis=1))
    return float(base_weight + (1.0 - base_weight) * outside)


def _ensemble_llm_nn(
    llm_pred_test: np.ndarray,
    nn_pred_test: np.ndarray,
    llm_pred_train: np.ndarray,
    nn_pred_train: np.ndarray,
    y_train: np.ndarray,
    llm_r2: float = None,
    nn_r2: float = None,
) -> np.ndarray:
    """
    Uncertainty-weighted ensemble of LLM and NN predictions.
    FIX 7: uncertainty is estimated from TRAINING residuals only — no test leakage.
    """
    eps = 1e-8
    # FIX 7: use train residuals, not test residuals, to estimate uncertainty
    llm_unc = float(np.std(llm_pred_train - y_train))
    nn_unc  = float(np.std(nn_pred_train  - y_train))
    w_llm   = (1.0 / (llm_unc + eps)) * max(float(llm_r2 or 0.0), 0.0)
    w_nn    = (1.0 / (nn_unc  + eps)) * max(float(nn_r2  or 0.0), 0.0)
    if w_llm + w_nn <= 0:
        w_llm = w_nn = 0.5
    total = w_llm + w_nn
    return (w_llm / total) * llm_pred_test + (w_nn / total) * nn_pred_test


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LLM formula generation (built-in, no external baseline import)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_llm_formula(
    description: str, domain: str, var_names: list[str], metadata: dict
) -> dict:
    """
    Call the Anthropic API to generate a Python formula function.
    Returns dict with keys: python_code, formula, success, error.
    """
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except Exception as e:
        return {"python_code": None, "formula": None, "success": False,
                "error": f"API client error: {e}", "model": _HYBRID_LLM_MODEL_NAME}

    var_list = ", ".join(var_names)
    constants = metadata.get("constants", {})
    constants_block = ""
    if constants:
        constants_block = "\nKnown constants (use these exact values in your formula):\n"
        for k, v in constants.items():
            constants_block += f"  {k} = {v}\n"

    prompt   = f"""You are an expert in DeFi (decentralised finance) mathematics.

Task: Derive the mathematical formula for the following quantity.

Description : {description}
Domain      : {domain}
Variables   : {var_list}{constants_block}
Return ONLY a Python function called `formula` that accepts the variables as
positional numpy-array arguments (in the order listed) and returns a numpy array.
Use numpy (imported as np) for any mathematical operations.
Do not include imports, explanations, or markdown — just the function definition.

Example output:
def formula({var_list}):
    return <expression>
"""
    try:
        resp = client.messages.create(
            model=_HYBRID_LLM_MODEL_NAME,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        code = resp.content[0].text.strip()
        # Strip markdown fences if present
        if "```" in code:
            lines = [ln for ln in code.splitlines()
                     if not ln.strip().startswith("```")]
            code  = "\n".join(lines).strip()
        return {"python_code": code, "formula": code,
                "success": "def formula" in code, "error": None,
                "model": _HYBRID_LLM_MODEL_NAME}
    except Exception as e:
        return {"python_code": None, "formula": None,
                "success": False, "error": str(e), "model": _HYBRID_LLM_MODEL_NAME}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Hybrid routing (Fix 5b: trustworthiness guard)
# ─────────────────────────────────────────────────────────────────────────────

def _formula_has_pathological_behavior(code: str) -> bool:
    """
    FIX 10 (v3c2-fix2): Detect formulas with patterns known to cause degenerate output.
    Returns True if any bad pattern is found → formula should not be trusted.

    BUGFIX: bare "nan" substring was matching identifiers like "nominal", "channel",
    any variable name containing "nan" — causing valid LLM formulas to be wrongly
    rejected → train_r2 = NaN.  Fixed with whole-word matching via regex.
    "1/0", "**1000" are unambiguous — substring match is fine.

    FIX 20 (ported from hypatiax_defi_benchmark_v4.py; see that module for the
    full note): "np.inf" removed from the blacklist -- it's also the correct,
    standard way to represent a genuine mathematical singularity when guarded
    by np.where (e.g. a breakeven-fee-rate formula that's infinite at zero
    volume share), and _execute_formula's downstream nan_frac check already
    catches truly unguarded inf-blowups based on actual output values, making
    the static text match both unnecessary and a source of false positives.
    """
    # Whole-word match for bare "nan" — avoids false positives in identifiers
    if re.search(r'\bnan\b', code):
        return True
    # These patterns are unambiguous — substring match is fine
    unambiguous = ["1/0", "**1000"]
    low = code.lower()
    return any(p.lower() in low for p in unambiguous)


def _hybrid_predict_and_eval(
    description: str, domain: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    var_names: list[str], metadata: dict,
    seed: int = _NN_SEED,
) -> dict:
    """
    NOTE: dead code — run_benchmark's per-case loop calls
    _v4_hybrid_predict_and_eval() below, not this function. This function
    still makes its own independent _generate_llm_formula() call rather than
    reusing pure_llm's already-scored formula (FIX 17, applied only to
    _v4_hybrid_predict_and_eval since that's the one actually invoked);
    apply the same fix here first if this function is ever wired back in.

    Full hybrid pipeline for one test case.
    Returns dict: train_r2, test_r2, decision, success.

    Routing priority (in order):
      Fix 2  → transcendental token detection → LLM  (if formula trustworthy)
      Fix 1  → extrapolation probe degradation → LLM  (if formula trustworthy)
      Fix 3  → LLM-feature augmentation for NN path   (if formula trustworthy)
      Fix 4  → distance-gated blend for ensemble path
      Fix 5  → unified evaluator for ALL LLM formula calls
      Fix 5b → routing guard: only override if LLM train R² > 0
    """

    # Step 1: Generate LLM formula
    llm_result = _generate_llm_formula(description, domain, var_names, metadata)
    llm_code   = llm_result.get("python_code") or ""
    has_formula = bool(llm_code and "def formula" in llm_code)
    # Model that generated llm_code — recorded regardless of whether the LLM
    # path is ultimately used, so the audit trail shows what was *tried*.
    llm_model_name = llm_result.get("model", _HYBRID_LLM_MODEL_NAME)

    # Extract constants from metadata — injected into exec globals so formulas
    # that reference protocol constants (e.g. K=100) don't NameError at eval time.
    constants = metadata.get("constants") or {}

    # Step 2: Evaluate LLM on training data (Fix 5 — unified evaluator)
    llm_train_r2, llm_train_ok = (
        _eval_formula_r2(llm_code, X_train, y_train, constants=constants)
        if has_formula else (float("nan"), False)
    )

    # Fix 5b + FIX 10: formula is trustworthy only if it fits training data well
    # AND does not exhibit any known pathological patterns.
    # FIX 10: raised threshold from > 0.0 to > 0.5 for stronger trust gate.
    llm_trustworthy = (
        has_formula
        and llm_train_ok
        and llm_train_r2 > 0.5
        and not _formula_has_pathological_behavior(llm_code)
    )

    # Step 3: Initial routing decision based on training LLM R²
    # Issue 4 fix: if LLM achieved near-perfect training fit, prefer it
    # immediately — do NOT allow the cascade below to accidentally flip to NN.
    # Previous code only checked train_r2 > 0.95 which was correct in principle,
    # but the routing-override block (Step 4) could later change "llm" back to
    # "ensemble" or "nn" even when train_r2 == 1.0.  We lock the decision here
    # so the Step 4 overrides only apply when the initial decision is NOT "llm".
    if llm_train_ok and llm_train_r2 >= 0.95:
        decision = "llm"          # locked — Step 4 overrides will be skipped
    elif llm_train_ok and llm_train_r2 > 0.50:
        decision = "ensemble"
    else:
        decision = "nn"

    # Step 4: Routing overrides (only when formula is trustworthy — Fix 5b)
    # IMPORTANT: do NOT override a locked "llm" decision (Issue 4 fix).
    if llm_trustworthy and decision in ("nn", "ensemble"):
        # Fix 2: transcendental tokens → LLM wins
        if _formula_has_transcendental(llm_code):
            decision = "llm"

    if llm_trustworthy and decision in ("nn", "ensemble"):
        # Fix 1: NN edge-degradation probe → LLM wins
        try:
            degradation = _extrapolation_probe(X_train, y_train)
            if degradation >= 0.15:
                decision = "llm"
        except Exception:
            pass

    # Step 5: Evaluate on TEST set using Fix 5 unified evaluator
    nn_rerun_time_s = 0.0   # Issue 3: track NN cost paid by hybrid on fallback

    if decision == "llm":
        test_r2, ok = _eval_formula_r2(llm_code, X_test, y_test, constants=constants)
        if not ok or np.isnan(test_r2):
            _t_nn0 = time.time()
            nn_m    = _train_and_eval_nn(X_train, y_train, X_test, y_test, seed=seed)
            nn_rerun_time_s = time.time() - _t_nn0
            test_r2 = nn_m["test_r2"]
            decision = "nn_fallback"

    elif decision == "ensemble" and has_formula:
        # Fix 5: LLM test predictions via unified evaluator
        llm_test_preds = _execute_formula(llm_code, X_test, constants=constants)
        _t_nn0         = time.time()
        nn_m           = _train_and_eval_nn(X_train, y_train, X_test, y_test, seed=seed)
        nn_rerun_time_s = time.time() - _t_nn0

        if llm_test_preds is not None:
            # FIX 7: also get LLM train predictions for uncertainty estimation
            llm_train_preds = _execute_formula(llm_code, X_train, constants=constants)
            if llm_train_preds is None:
                test_r2  = nn_m["test_r2"]
                decision = "nn_fallback"
            else:
                llm_w         = _distance_llm_weight(X_test, X_train, base_weight=0.3)
                nn_w          = 1.0 - llm_w
                ensemble_pred = _ensemble_llm_nn(
                    llm_pred_test=llm_test_preds,
                    nn_pred_test=nn_m["y_pred_test"],
                    llm_pred_train=llm_train_preds,
                    nn_pred_train=nn_m["y_pred_train"],
                    y_train=y_train,
                    llm_r2=llm_train_r2 * llm_w,
                    nn_r2=nn_m["train_r2"] * nn_w,
                )
                m = _compute_metrics(y_test, ensemble_pred)
                test_r2 = m["r2"]
        else:
            test_r2  = nn_m["test_r2"]
            decision = "nn_fallback"

    else:
        # decision == "nn"
        # Fix 3: if formula trustworthy, augment X with LLM predictions
        nn_used_llm_features = False  # tracks whether X was augmented with LLM output
        if llm_trustworthy:
            try:
                llm_tr_preds = _execute_formula(llm_code, X_train, constants=constants)
                llm_te_preds = _execute_formula(llm_code, X_test,  constants=constants)
                if (llm_tr_preds is not None and llm_te_preds is not None):
                    X_tr_aug = np.column_stack([X_train, llm_tr_preds])
                    X_te_aug = np.column_stack([X_test,  llm_te_preds])
                    _t_nn0 = time.time()
                    nn_m     = _train_and_eval_nn(X_tr_aug, y_train, X_te_aug, y_test, seed=seed)
                    nn_rerun_time_s = time.time() - _t_nn0
                    nn_used_llm_features = True
                else:
                    _t_nn0 = time.time()
                    nn_m     = _train_and_eval_nn(X_train, y_train, X_test, y_test, seed=seed)
                    nn_rerun_time_s = time.time() - _t_nn0
            except Exception:
                _t_nn0 = time.time()
                nn_m         = _train_and_eval_nn(X_train, y_train, X_test, y_test, seed=seed)
                nn_rerun_time_s = time.time() - _t_nn0
        else:
            _t_nn0 = time.time()
            nn_m             = _train_and_eval_nn(X_train, y_train, X_test, y_test, seed=seed)
            nn_rerun_time_s = time.time() - _t_nn0
        test_r2 = nn_m["test_r2"]

    # ── Model-identity audit trail ──────────────────────────────────────────
    # `decision` reflects which path actually produced test_r2 (it may have
    # been overwritten to "nn_fallback" above). Map it to the concrete model
    # name(s) responsible, rather than leaving that implicit in `decision`.
    if decision == "llm":
        model_used = llm_model_name
    elif decision == "ensemble":
        model_used = f"ensemble({llm_model_name}+{_NN_MODEL_NAME})"
    elif decision == "nn_fallback":
        model_used = _NN_MODEL_NAME
    else:  # decision == "nn"
        model_used = (
            f"{_NN_MODEL_NAME}(llm_augmented_features={llm_model_name})"
            if nn_used_llm_features else _NN_MODEL_NAME
        )

    return {
        "train_r2":       float(llm_train_r2) if llm_train_ok else float("nan"),
        "test_r2":        float(test_r2),
        "decision":       decision,
        "llm_code":       llm_code if has_formula else None,
        "llm_train_r2":   float(llm_train_r2) if llm_train_ok else float("nan"),
        "llm_model":      llm_model_name,     # model that generated llm_code, whether or not it was ultimately used
        "model_used":     model_used,         # model(s) that actually produced test_r2 — the auditable ground truth
        "nn_rerun_time_s": round(nn_rerun_time_s, 3),  # Issue 3: NN cost paid by hybrid
        # NOTE: this is a placeholder, not the operative success field — the
        # caller (see FIX 13 in run_benchmark's per-case loop) recomputes
        # success as a real fit-quality gate from train_r2/test_r2 above and
        # discards this value entirely. Do not read hy_m["success"] directly.
        "success":        True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5B — v4.0 validation-selected hybrid (supersedes the cascading
# routing logic in _hybrid_predict_and_eval above, which is kept only as
# reference/history and is no longer called from run_benchmark()).
# ─────────────────────────────────────────────────────────────────────────────

_V4_VAL_FRAC = 0.25
_V4_MIN_VAL = 10
_V4_ARCHITECTURES = ([64, 32], [128, 64, 32])
_V4_MAX_TIME_S = 8
_V4_BLEND_GRID = np.linspace(0.0, 1.0, 21)
# FIX 18: when the test domain is extrapolative, a blend must keep at least this
# much weight on the LLM formula. blend(alpha=0) IS the bare NN, so leaving 0.0
# in the grid let a "blend" win on internal-validation R2 alone and silently
# defeat the extrapolation guard in _select_v4_candidate (see FIX 18 note there).
_V4_EXTRAP_MIN_BLEND_ALPHA = 0.5


def _split_internal_validation(X_train: np.ndarray, y_train: np.ndarray,
                               config: dict, val_frac: float = _V4_VAL_FRAC):
    """Create an internal validation split using TRAINING DATA ONLY.

    The validation region is the high end of the training region along the
    same configured extrapolation variable. This is deliberately harder than
    a random holdout and is never allowed to inspect the final (PCA-split)
    test set.
    """
    n = len(X_train)
    if n < 2 * _V4_MIN_VAL:
        return X_train, y_train, None, None
    var_idx = int(config.get("split_var_idx", 0))
    vals = (X_train[:, var_idx]
            if (X_train.ndim >= 2 and X_train.shape[1] > var_idx)
            else X_train.flatten())
    order = np.argsort(vals)
    n_val = max(_V4_MIN_VAL, int(round(n * val_frac)))
    n_val = min(n_val, n - _V4_MIN_VAL)
    inner = order[:-n_val]
    val = order[-n_val:]
    return X_train[inner], y_train[inner], X_train[val], y_train[val]


def _fit_nn_predict(X_fit: np.ndarray, y_fit: np.ndarray, X_eval: np.ndarray,
                    hidden: list[int], seed: int, epochs: int = 300,
                    max_time_s: float = _V4_MAX_TIME_S) -> tuple[np.ndarray, bool]:
    """Fit an MLP and return predictions on X_eval, using fit data only."""
    torch.manual_seed(seed); np.random.seed(seed)
    plan = _compute_augment_plan(X_fit)
    Xf = _apply_augment_plan(X_fit, plan)
    Xe = _apply_augment_plan(X_eval, plan)
    sx, sy = StandardScaler(), StandardScaler()
    Xfs = sx.fit_transform(Xf)
    yfs = sy.fit_transform(y_fit.reshape(-1, 1)).flatten()
    model = _MLP(Xf.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.MSELoss()
    Xt = torch.FloatTensor(Xfs); yt = torch.FloatTensor(yfs).reshape(-1, 1)
    start = time.time(); timed_out = False
    model.train()
    for epoch in range(epochs):
        opt.zero_grad(); loss = crit(model(Xt), yt); loss.backward(); opt.step()
        if epoch % 25 == 0 and time.time() - start >= max_time_s:
            timed_out = True
            break
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(sx.transform(Xe))).numpy().flatten()
    pred = sy.inverse_transform(pred.reshape(-1, 1)).flatten()
    return pred, timed_out


def _fit_linear_fallback_predict(X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray) -> np.ndarray:
    """Deterministic ordinary-least-squares affine fallback (Fix 21, ported
    from hypatiax_defi_benchmark_v4.py).

    Used only for the untrusted-LLM + extrapolative branch of
    `_select_v4_candidate` (see Fix 21 below): the case where no
    LLM-anchored candidate exists at all, so the extrapolation guard
    previously had nothing to prefer a bare NN over and fell back to the
    simplest available NN architecture chosen purely by internal-
    validation R^2. In-domain validation R^2 says nothing about behaviour
    once the benchmark's actual test points fall outside the training
    range (`_v4_extrapolates`), and this was observed to produce
    catastrophic NN blow-ups (e.g. trusted=False -> selected=nn: mean
    test R^2 -1.961, min -11.810 across a 20-case slice of the DeFi
    benchmark).

    A plain least-squares affine fit has no hyperparameters and no random
    seed to overfit the internal validation split with, and its
    extrapolation error grows at worst linearly in the inputs rather than
    the unbounded blow-ups an MLP can produce out-of-range. It will not
    capture genuine nonlinear structure -- it is a deliberately
    conservative fallback, not a replacement for a trustworthy LLM
    formula or an in-domain NN fit.
    """
    Xtr = np.asarray(X_train, dtype=float)
    Xte = np.asarray(X_test, dtype=float)
    ytr = np.asarray(y_train, dtype=float)
    if Xtr.ndim == 1:
        Xtr = Xtr.reshape(-1, 1)
    if Xte.ndim == 1:
        Xte = Xte.reshape(-1, 1)
    Xtr1 = np.column_stack([np.ones(len(Xtr)), Xtr])
    Xte1 = np.column_stack([np.ones(len(Xte)), Xte])
    coef, *_ = np.linalg.lstsq(Xtr1, ytr, rcond=None)
    return Xte1 @ coef


def _ridge_affine_fit(Xtr1: np.ndarray, ytr: np.ndarray, ridge_lambda: float) -> np.ndarray:
    """Regularized affine fit. Ridge (not raw lstsq) so an ill-conditioned or
    narrow-range feature matrix can't produce an arbitrarily large slope --
    that unconstrained slope is exactly what caused _fit_linear_fallback_predict
    to blow up to R^2 in the tens-of-thousands-negative range on some DeFi
    ratio/product formulas (see _fit_linear_fallback_predict_v2 docstring)."""
    n_features = Xtr1.shape[1]
    reg = ridge_lambda * np.eye(n_features)
    reg[0, 0] = 0.0  # never penalize the intercept
    XtX = Xtr1.T @ Xtr1
    Xty = Xtr1.T @ ytr
    return np.linalg.solve(XtX + reg, Xty)


def _clamp_to_training_range(pred: np.ndarray, y_train: np.ndarray, clamp_factor: float) -> np.ndarray:
    """Bound predictions to a multiple of the observed training-label range.
    This is the actual safety guarantee Fix 21 claimed but did not enforce:
    a fallback model can still be structurally wrong (e.g. affine fit to a
    ratio/product formula) and extrapolate its own bad slope to +-90000
    while being "affine" the whole way there. Clamping caps the damage any
    single candidate can do, regardless of why its extrapolation went bad."""
    lo, hi = float(np.min(y_train)), float(np.max(y_train))
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0)
    return np.clip(pred, lo - clamp_factor * span, hi + clamp_factor * span)


def _fit_linear_fallback_predict_v2(X_train: np.ndarray, y_train: np.ndarray,
                                    X_test: np.ndarray, ridge_lambda: float = 1e-2,
                                    clamp_factor: float = 3.0) -> np.ndarray:
    """Robust successor to _fit_linear_fallback_predict (Fix 22).

    _fit_linear_fallback_predict (Fix 21) is a plain, unregularized OLS
    affine fit with no output bound. Empirically re-run against the real
    74-case DeFi catalogue, it is *not* the bounded, safe fallback its
    docstring claims: on ratio/product-structured formulas (AMM spot price,
    leverage, Sharpe ratio, LTV, IL breakeven -- exactly the formula shapes
    common in this benchmark, and exactly the kind of formula a live LLM
    run showed the pure-LLM baseline itself sometimes fails to produce code
    for) it extrapolates to test R^2 in the hundreds or tens-of-thousands
    negative -- far worse than the bare-NN blowups (worst observed: -11.81)
    that Fix 21 was written to replace.

    Fix 22 keeps the deterministic, seed-free affine-fit idea (still
    preferred over an NN for this branch: bounded structural form, no
    hyperparameter search) but adds two changes:
      1. Ridge regularization instead of raw np.linalg.lstsq, so a narrow-
         range or collinear X_train can't produce an arbitrarily large slope.
      2. Hard output clamping to a bounded multiple of the observed
         training-label range, so even a structurally-wrong fit (the model
         is affine, the ground truth is a ratio) cannot blow up unboundedly.
    Optionally also fits in log-space when X_train and y_train are strictly
    positive (common for ratio/product DeFi formulas) and keeps whichever
    of {linear, log-linear} has the better internal train-refit residual,
    since a log-linear model structurally matches multiplicative formulas
    far better than an affine one.
    """
    Xtr = np.asarray(X_train, dtype=float)
    Xte = np.asarray(X_test, dtype=float)
    ytr = np.asarray(y_train, dtype=float)
    if Xtr.ndim == 1:
        Xtr = Xtr.reshape(-1, 1)
    if Xte.ndim == 1:
        Xte = Xte.reshape(-1, 1)

    def _affine_predict(Xtr_, ytr_, Xte_):
        Xtr1 = np.column_stack([np.ones(len(Xtr_)), Xtr_])
        Xte1 = np.column_stack([np.ones(len(Xte_)), Xte_])
        coef = _ridge_affine_fit(Xtr1, ytr_, ridge_lambda)
        return Xte1 @ coef, Xtr1 @ coef

    pred_lin_te, pred_lin_tr = _affine_predict(Xtr, ytr, Xte)
    lin_train_resid = float(np.mean((pred_lin_tr - ytr) ** 2))
    best_pred = pred_lin_te
    best_resid = lin_train_resid

    can_log = np.all(Xtr > 0) and np.all(Xte > 0) and np.all(ytr > 0)
    if can_log:
        log_pred_te_log, log_pred_tr_log = _affine_predict(np.log(Xtr), np.log(ytr), np.log(Xte))
        pred_log_tr = np.exp(log_pred_tr_log)
        log_train_resid = float(np.mean((pred_log_tr - ytr) ** 2))
        if np.all(np.isfinite(log_pred_te_log)) and log_train_resid < best_resid:
            best_pred = np.exp(log_pred_te_log)
            best_resid = log_train_resid

    if not np.all(np.isfinite(best_pred)):
        best_pred = np.nan_to_num(best_pred, nan=float(np.median(ytr)),
                                  posinf=float(np.max(ytr)), neginf=float(np.min(ytr)))
    return _clamp_to_training_range(best_pred, ytr, clamp_factor)


def _fit_linear_fallback_predict_local(X_train: np.ndarray, y_train: np.ndarray,
                                       X_test: np.ndarray, k: int = 8,
                                       buffer_mult: float = 6.0) -> np.ndarray:
    """Fix 23: local-neighborhood variant of _fit_linear_fallback_predict_v2.

    _fit_linear_fallback_predict_v2 clamps to the *global* y_train range,
    which is leakage-safe but too loose for cases where the extrapolative
    test split lands in a narrow slice near the edge of a much wider
    training range: R^2 divides by the tiny test-domain variance, so even
    an ordinary-sized absolute error explodes (observed: R^2 as low as
    -7284 on "Position margin ratio" even though _v2's raw predictions were
    not themselves unbounded).

    This variant clamps each test point's prediction to a bound derived
    from only its k nearest training neighbors (by standardized feature
    distance) rather than the full training range -- tighter where the
    global range is misleadingly wide, still leakage-safe (only ever reads
    y_train). Empirically this is a genuine precision/recall trade rather
    than a strict improvement: it controls the worst tail but can also
    clip otherwise-correct extrapolation, which is exactly why it is
    offered as a second internal-validation-ranked candidate in
    _select_v4_candidate rather than a blanket replacement for the global
    version -- the internal validation split decides, per case, which one
    actually behaves better on that case's held-out edge region.
    """
    base_pred = _fit_linear_fallback_predict_v2(X_train, y_train, X_test)
    Xtr = np.asarray(X_train, dtype=float)
    Xte = np.asarray(X_test, dtype=float)
    ytr = np.asarray(y_train, dtype=float)
    if Xtr.ndim == 1:
        Xtr = Xtr.reshape(-1, 1)
    if Xte.ndim == 1:
        Xte = Xte.reshape(-1, 1)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Xtr_s = (Xtr - mu) / sd
    Xte_s = (Xte - mu) / sd
    k_eff = min(k, len(Xtr))
    out = base_pred.copy()
    for i in range(len(Xte_s)):
        dist = np.sum((Xtr_s - Xte_s[i]) ** 2, axis=1)
        nn_idx = np.argsort(dist)[:k_eff]
        local_y = ytr[nn_idx]
        lo, hi = float(local_y.min()), float(local_y.max())
        span = hi - lo if hi > lo else max(abs(hi), 1.0)
        out[i] = float(np.clip(base_pred[i], lo - buffer_mult * span, hi + buffer_mult * span))
    return out


def _fit_candidate_full(candidate: str, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, llm_train: np.ndarray | None,
                        llm_test: np.ndarray | None, hidden: list[int],
                        blend_alpha: float, seed: int) -> dict:
    """Fit the selected v4 candidate on ALL available training data."""
    if candidate == "llm":
        return {"y_pred_test": llm_test, "train_pred": llm_train}
    if candidate == "linear_fallback":
        # Fix 22: robust conservative fallback for untrusted-LLM +
        # extrapolative cases -- see _fit_linear_fallback_predict_v2
        # docstring. Supersedes the unregularized, unclamped Fix 21 version,
        # which was empirically shown to blow up to R^2 in the hundreds/
        # thousands-negative on ratio-structured DeFi formulas.
        pred_te = _fit_linear_fallback_predict_v2(X_train, y_train, X_test)
        pred_tr = _fit_linear_fallback_predict_v2(X_train, y_train, X_train)
        return {"y_pred_test": pred_te, "train_pred": pred_tr, "timed_out": False}
    if candidate == "linear_fallback_local":
        # Fix 23: local-neighborhood clamp variant -- see
        # _fit_linear_fallback_predict_local docstring. Only ever reached
        # via _select_v4_candidate's internal-validation ranking against
        # "linear_fallback", never chosen unconditionally.
        pred_te = _fit_linear_fallback_predict_local(X_train, y_train, X_test)
        pred_tr = _fit_linear_fallback_predict_local(X_train, y_train, X_train)
        return {"y_pred_test": pred_te, "train_pred": pred_tr, "timed_out": False}
    if candidate == "nn":
        pred_te, timed = _fit_nn_predict(X_train, y_train, X_test, hidden, seed, max_time_s=_NN_MAX_TIME_S)
        pred_tr, _ = _fit_nn_predict(X_train, y_train, X_train, hidden, seed, max_time_s=_NN_MAX_TIME_S)
        return {"y_pred_test": pred_te, "train_pred": pred_tr, "timed_out": timed}
    if candidate in ("residual", "residual_nn"):
        if llm_train is None or llm_test is None:
            return None
        residual = y_train - llm_train
        pred_res_te, timed = _fit_nn_predict(X_train, residual, X_test, hidden, seed, max_time_s=_NN_MAX_TIME_S)
        pred_res_tr, _ = _fit_nn_predict(X_train, residual, X_train, hidden, seed, max_time_s=_NN_MAX_TIME_S)
        return {"y_pred_test": llm_test + pred_res_te, "train_pred": llm_train + pred_res_tr, "timed_out": timed}
    if candidate == "blend":
        if llm_test is None:
            return None
        pred_te, timed = _fit_nn_predict(X_train, y_train, X_test, hidden, seed, max_time_s=_NN_MAX_TIME_S)
        pred_tr, _ = _fit_nn_predict(X_train, y_train, X_train, hidden, seed, max_time_s=_NN_MAX_TIME_S)
        return {"y_pred_test": blend_alpha * llm_test + (1.0 - blend_alpha) * pred_te,
                "train_pred": blend_alpha * llm_train + (1.0 - blend_alpha) * pred_tr,
                "timed_out": timed}
    return None


def _v4_extrapolates(X_train: np.ndarray, X_test: np.ndarray) -> bool:
    """True if any feature in X_test falls outside the observed X_train range.

    This only inspects feature values (never y_test), so it introduces no
    label leakage into the selection step. It exists because a bare NN's
    internal-validation R² is measured on a split drawn from the training
    domain -- it says nothing about NN behavior once the benchmark's actual
    test points fall outside that domain, which is exactly where NN
    extrapolation failures (train R² ~0.9999, test R² deeply negative) occur.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    test_min = X_test.min(axis=0)
    test_max = X_test.max(axis=0)
    return bool(np.any(test_min < train_min) or np.any(test_max > train_max))


def _rank_fallback_candidates(Xi: np.ndarray, yi: np.ndarray,
                              Xv: np.ndarray, yv: np.ndarray) -> dict:
    """Fix 23: rank the deterministic fallback candidates on the same
    edge-holdout internal validation split used for llm/residual/blend/nn,
    rather than hardcoding one fallback for every case.

    _split_internal_validation draws Xv/yv from the *high end* of the
    training region along the configured extrapolation variable -- a
    genuine (if milder) held-out extrapolation probe, not a random in-
    domain split -- so ranking fallback candidates on it is safe in the
    same way ranking nn/residual/blend on it already was; it is not the
    in-domain-only signal that made ranking a bare NN unsafe in Fix 21.

    Returns {candidate_name: r2} for every fallback candidate that could
    be fit without raising.
    """
    out = {}
    try:
        pred_global = _fit_linear_fallback_predict_v2(Xi, yi, Xv)
        out["linear_fallback"] = float(_compute_metrics(yv, pred_global)["r2"])
    except Exception:
        pass
    try:
        pred_local = _fit_linear_fallback_predict_local(Xi, yi, Xv)
        out["linear_fallback_local"] = float(_compute_metrics(yv, pred_local)["r2"])
    except Exception:
        pass
    return out


def _select_v4_candidate(X_train: np.ndarray, y_train: np.ndarray, llm_code: str,
                         constants: dict, config: dict, seed: int,
                         extrapolative: bool = False) -> dict:
    """Select LLM/NN/residual-NN/blend strictly from an internal validation split.

    No final-test values are read here. If an LLM formula is unavailable or
    fails on validation, only candidates that can be evaluated remain
    eligible.

    `extrapolative` flags that the benchmark's test features fall outside the
    training domain (see `_v4_extrapolates`). When True, a bare NN candidate
    is prevented from winning solely because its internal-validation R² looks
    good -- llm/residual/blend candidates are preferred instead, as long as
    at least one of them is available. NN remains selectable when it's the
    only candidate at all (e.g. no usable LLM formula), and residual/blend
    candidates -- which stay anchored to the LLM formula -- are untouched by
    this guard, EXCEPT that (FIX 18) an extrapolative blend may not drop below
    `_V4_EXTRAP_MIN_BLEND_ALPHA` weight on the formula: blend(alpha=0) is
    the bare NN, so it must not be allowed to pass the guard as "anchored".
    """
    Xi, yi, Xv, yv = _split_internal_validation(X_train, y_train, config)
    if Xv is None:
        if llm_code:
            selected = "llm"
        elif extrapolative:
            # Fix 21: too few points for an internal validation split, no
            # trustworthy LLM formula, and the test domain extrapolates --
            # do not default to "nn" here either.
            selected = "linear_fallback"
        else:
            selected = "nn"
        return {"selected": selected, "hidden": _V4_ARCHITECTURES[1],
                "blend_alpha": 1.0, "validation_r2": {}, "validation_n": 0,
                "extrapolation_unmitigated": False}

    llm_i = _execute_formula(llm_code, Xi, constants=constants) if llm_code else None
    llm_v = _execute_formula(llm_code, Xv, constants=constants) if llm_code else None
    llm_ok = llm_i is not None and llm_v is not None

    candidates = {}
    if llm_ok:
        candidates["llm"] = {"r2": _compute_metrics(yv, llm_v)["r2"], "hidden": None, "alpha": 1.0}

    # FIX 18: blend(alpha=0) is exactly the bare NN, and the alpha grid always
    # contains 0.0, so blend's validation R2 is >= the NN's by construction and
    # wins ties (priority blend < nn). Without a floor, the extrapolation guard
    # below -- which only excludes keys starting with "nn:" -- was a no-op
    # whenever a trusted formula existed: an NN that looked good on the
    # in-domain validation split came back as "blend, alpha=0.00", labelled
    # extrapolation_unmitigated=False. Under extrapolation, keep the formula
    # in charge of the blend.
    alpha_grid = _V4_BLEND_GRID
    if extrapolative:
        alpha_grid = _V4_BLEND_GRID[_V4_BLEND_GRID >= _V4_EXTRAP_MIN_BLEND_ALPHA - 1e-9]

    for hidden in _V4_ARCHITECTURES:
        try:
            nn_v, _ = _fit_nn_predict(Xi, yi, Xv, hidden, seed)
            candidates[f"nn:{hidden}"] = {"r2": _compute_metrics(yv, nn_v)["r2"], "hidden": hidden, "alpha": 0.0}
            if llm_ok:
                residual = yi - llm_i
                res_v, _ = _fit_nn_predict(Xi, residual, Xv, hidden, seed)
                residual_pred = llm_v + res_v
                candidates[f"residual:{hidden}"] = {"r2": _compute_metrics(yv, residual_pred)["r2"], "hidden": hidden, "alpha": 1.0}
                best_alpha, best_r2 = float(alpha_grid[0]), -np.inf
                for a in alpha_grid:
                    bp = a * llm_v + (1.0 - a) * nn_v
                    r2 = _compute_metrics(yv, bp)["r2"]
                    if r2 > best_r2:
                        best_r2, best_alpha = r2, float(a)
                candidates[f"blend:{hidden}"] = {"r2": best_r2, "hidden": hidden, "alpha": best_alpha}
        except Exception:
            continue

    if not candidates:
        if extrapolative:
            # Fix 21/23: nothing fit at all (every candidate raised) and the
            # case extrapolates -- still must not default to "nn". Rank the
            # fallback candidates on the edge-holdout validation split
            # rather than hardcoding one (see _rank_fallback_candidates).
            fb_scores = _rank_fallback_candidates(Xi, yi, Xv, yv)
            if fb_scores:
                fb_winner = max(fb_scores, key=lambda k: (fb_scores[k], k == "linear_fallback"))
            else:
                fb_winner = "linear_fallback"
                fb_scores = {"linear_fallback": float("nan")}
            return {"selected": fb_winner, "hidden": None, "blend_alpha": 1.0,
                    "validation_r2": fb_scores,
                    "validation_n": len(yv), "extrapolation_unmitigated": False}
        return {"selected": "nn", "hidden": _V4_ARCHITECTURES[1], "blend_alpha": 0.0,
                "validation_r2": {}, "validation_n": len(yv),
                "extrapolation_unmitigated": False}

    # Deterministic tie-break: prefer the simpler candidate when validation R² is tied.
    priority = {"llm": 0, "residual": 1, "blend": 2, "linear_fallback": 3,
               "linear_fallback_local": 4, "nn": 5}
    pool = candidates
    extrapolation_unmitigated = False
    if extrapolative:
        # Don't let a bare NN win purely on internal-validation R² when the
        # benchmark test domain lies outside training range -- prefer any
        # LLM-anchored candidate if one exists.
        anchored = {k: v for k, v in candidates.items() if not k.startswith("nn:")}
        if anchored:
            pool = anchored
        else:
            # Fix 21/23: extrapolative test domain but no LLM-anchored
            # candidate exists at all (LLM formula unavailable/untrustworthy)
            # -- there is nothing for the guard above to prefer NN over.
            # Fix 21 originally fell back to a single hardcoded deterministic
            # linear fit, on the reasoning that in-domain validation R² --
            # the exact signal that misled the old bare-NN branch -- can't be
            # trusted to rank fallback candidates either. That reasoning
            # doesn't actually apply here: _split_internal_validation draws
            # Xv/yv from the *high end* of the training region along the
            # extrapolation variable, a genuine (if milder) held-out
            # extrapolation probe -- the same split already used to rank
            # nn/residual/blend above. Fix 23 uses that same split to rank
            # between the fallback candidates themselves (global-clamp vs.
            # local-neighborhood-clamp -- see _rank_fallback_candidates),
            # since neither is uniformly safer: empirically, global-clamp
            # ridge/log fitting kills genuine unbounded blow-ups but can
            # still fail badly when the test split is a narrow slice near
            # the edge of a much wider training range, and local-clamp fixes
            # that but can clip otherwise-correct extrapolation elsewhere.
            # A bare NN is still never in this pool. Because a genuine
            # mitigation is applied either way, extrapolation_unmitigated
            # stays False; the actual choice is auditable via `selected`
            # and `validation_r2` in the returned dict.
            fb_scores = _rank_fallback_candidates(Xi, yi, Xv, yv)
            if not fb_scores:
                fb_scores = {"linear_fallback": float("nan")}
            for name, r2v in fb_scores.items():
                candidates[name] = {"r2": r2v, "hidden": None, "alpha": 1.0}
            pool = {name: candidates[name] for name in fb_scores}
    winner_key = max(pool, key=lambda k: (pool[k]["r2"], -priority.get(k.split(":")[0], 9)))
    w = candidates[winner_key]
    prefix = winner_key.split(":")[0]
    # Keep the externally reported name consistent with the candidate
    # implementation: validation keys use "residual:<hidden>", while the
    # fitted candidate/audit trail use "residual_nn".
    selected_name = "residual_nn" if prefix == "residual" else prefix
    return {
        "selected": selected_name,
        "hidden": w.get("hidden") or _V4_ARCHITECTURES[1],
        "blend_alpha": float(w.get("alpha", 1.0)),
        "validation_r2": {k: float(v["r2"]) for k, v in candidates.items()},
        "validation_candidates": {
            k: {
                "candidate": ("residual_nn" if k.startswith("residual:") else k.split(":")[0]),
                "hidden": v.get("hidden"),
                "alpha": float(v.get("alpha", 1.0)),
                "r2": float(v["r2"]),
            }
            for k, v in candidates.items()
        },
        "validation_n": len(yv),
        "extrapolation_unmitigated": extrapolation_unmitigated,
    }


def _v4_hybrid_predict_and_eval(description: str, domain: str,
                                X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray,
                                var_names: list[str], metadata: dict,
                                config: dict, seed: int = _NN_SEED,
                                pure_llm_code: str | None = None,
                                pure_llm_model: str | None = None) -> dict:
    """Leakage-safe v4 hybrid: LLM formula + validation-selected correction model.

    Same selection logic as hypatiax_defi_benchmark_v4.py's
    _v4_hybrid_predict_and_eval(); this copy additionally reports
    `llm_model` / `model_used` (pca.py's pre-existing audit-trail fields,
    not present in v4) so a run mixing model versions stays auditable.

    FIX 17 (2026-09-18, ported from hypatiax_defi_benchmark_v4.py): when the
    caller supplies `pure_llm_code`, it's the exact LLM-generated formula the
    pure_llm arm already produced and scored via
    PureLLMBaseline.generate_formula()/test_formula_accuracy() — reusing it
    here means this arm's llm_trustworthy gate and llm_train_r2 are computed
    on the SAME formula pure_llm.test_r2 reports, instead of a second,
    independently-prompted API call made by _generate_llm_formula below,
    which was never guaranteed to agree with pure_llm's own call and could
    silently diverge (llm_train_r2 as low as -6.9 million on a case where
    pure_llm's formula scored test_r2 = 1.0), routing hybrid to a weaker NN
    fallback it didn't need. `pure_llm_code=None` (the default) preserves the
    old standalone-call behavior; `pure_llm_code=""` (pure_llm ran but had no
    usable/non-truncated formula) is treated as "no formula" rather than
    triggering a fresh, potentially-divergent call here. `pure_llm_model`
    carries pure_llm's reported model name through to this arm's `llm_model`
    / `model_used` audit fields so they describe the formula actually used
    rather than defaulting to _HYBRID_LLM_MODEL_NAME.
    """
    if pure_llm_code is None:
        llm_result = _generate_llm_formula(description, domain, var_names, metadata)
        llm_code = llm_result.get("python_code") or ""
        llm_model_name = llm_result.get("model", _HYBRID_LLM_MODEL_NAME)
    else:
        llm_code = pure_llm_code
        llm_model_name = pure_llm_model or _HYBRID_LLM_MODEL_NAME
    constants = metadata.get("constants") or {}
    llm_train = _execute_formula(llm_code, X_train, constants=constants) if llm_code else None
    llm_test = _execute_formula(llm_code, X_test, constants=constants) if llm_code else None
    llm_train_r2 = (_compute_metrics(y_train, llm_train)["r2"] if llm_train is not None else float("nan"))
    # Fix 21: hardened trust gate. Previously only checked llm_train_r2 > 0.5
    # and the pathological-pattern check; a formula could still pass with a
    # non-finite prediction on X_test (train R^2 computed on llm_train alone
    # can be a healthy number even when llm_test contains NaN/Inf for
    # out-of-range test points, e.g. a pole in the formula). This is a
    # strictly stronger gate than before -- it can only turn a previously
    # "trustworthy" formula untrustworthy when its predictions are actually
    # not usable, never the reverse.
    trustworthy = bool(llm_train is not None and llm_test is not None
                       and np.all(np.isfinite(llm_train)) and np.all(np.isfinite(llm_test))
                       and llm_train_r2 > 0.5
                       and not _formula_has_pathological_behavior(llm_code))
    if not trustworthy:
        llm_train = llm_test = None

    selection = _select_v4_candidate(X_train, y_train, llm_code if trustworthy else "",
                                     constants, config, seed,
                                     extrapolative=_v4_extrapolates(X_train, X_test))
    selected = selection["selected"]
    hidden = selection["hidden"]
    alpha = selection["blend_alpha"]
    extrapolation_unmitigated = bool(selection.get("extrapolation_unmitigated", False))

    if selected == "llm":
        pred_test = llm_test
        pred_train = llm_train
        timed_out = False
    else:
        fitted = _fit_candidate_full(selected, X_train, y_train, X_test,
                                     llm_train, llm_test, hidden, alpha, seed)
        # Guard against both hard failures (_fit_candidate_full returns None,
        # e.g. residual/blend requested without a usable LLM formula) and
        # soft failures (a live fit that silently produced NaN/Inf, e.g. an
        # MLP numerically blowing up on an out-of-range extrapolative test
        # point). Checking only `is None` would miss the second case: the
        # candidate would keep decision="v4_residual_nn"/"v4_blend" and
        # extrapolation_unmitigated=False even though nothing usable was
        # actually produced.
        _pred = fitted.get("y_pred_test") if fitted is not None else None
        _pred_bad = _pred is None or not np.all(np.isfinite(_pred))
        if _pred_bad:
            # Fix 15/16: if the validation-selected candidate fails at full-fit
            # time, walk the remaining candidates in descending validation R²
            # order. This stays leakage-safe because ranking comes exclusively
            # from the internal validation split; y_test is never consulted.
            # The trustworthy LLM is therefore a normal ranked fallback rather
            # than a special-case shortcut, and bare NN is only used if it was
            # actually the next viable validation-ranked candidate (or the
            # final defensive fallback when no ranked candidate can be fit).
            ranked = sorted(selection.get("validation_candidates", {}).items(),
                            key=lambda kv: kv[1].get("r2", selection["validation_r2"].get(kv[0], -np.inf)),
                            reverse=True)
            attempted = {selected}
            fallback_fit = None
            fallback_name = None
            fallback_hidden = hidden
            fallback_alpha = alpha

            for key, spec in ranked:
                candidate = spec["candidate"]
                if candidate in attempted:
                    continue
                attempted.add(candidate)
                if candidate == "llm":
                    if trustworthy and llm_test is not None:
                        fallback_fit = {"y_pred_test": llm_test, "train_pred": llm_train, "timed_out": False}
                    else:
                        continue
                else:
                    candidate_hidden = spec.get("hidden") or _V4_ARCHITECTURES[1]
                    candidate_alpha = float(spec.get("alpha", 1.0))
                    fallback_fit = _fit_candidate_full(candidate, X_train, y_train, X_test,
                                                       llm_train, llm_test, candidate_hidden,
                                                       candidate_alpha, seed)
                fp = fallback_fit.get("y_pred_test") if fallback_fit is not None else None
                if fp is not None and np.all(np.isfinite(fp)):
                    fallback_name = "llm_fallback" if candidate == "llm" else candidate
                    fallback_hidden = spec.get("hidden") or _V4_ARCHITECTURES[1]
                    fallback_alpha = float(spec.get("alpha", 1.0))
                    break
                fallback_fit = None

            if fallback_fit is not None and fallback_name is not None:
                pred_test = fallback_fit["y_pred_test"]
                pred_train = fallback_fit.get("train_pred")
                timed_out = bool(fallback_fit.get("timed_out", False))
                selected = fallback_name
                hidden = fallback_hidden
                alpha = fallback_alpha
            else:
                # Last resort only: there was no usable validation-ranked
                # candidate left. Preserve the historical defensive NN path.
                fitted = _fit_candidate_full("nn", X_train, y_train, X_test,
                                             None, None, _V4_ARCHITECTURES[1], 0.0, seed)
                selected = "nn_fallback"
                pred_test = fitted["y_pred_test"]
                pred_train = fitted.get("train_pred")
                timed_out = bool(fitted.get("timed_out", False))
            # The originally selected candidate failed, so the extrapolation
            # protection attached to that selection no longer describes the
            # model actually used.
            extrapolation_unmitigated = True
        else:
            pred_test = fitted["y_pred_test"]
            pred_train = fitted.get("train_pred")
            timed_out = bool(fitted.get("timed_out", False))

    test_r2 = _compute_metrics(y_test, pred_test)["r2"] if pred_test is not None else float("nan")
    train_r2 = _compute_metrics(y_train, pred_train)["r2"] if pred_train is not None else llm_train_r2

    # ── Model-identity audit trail (kept from pca.py's pre-v4 hybrid) ──────
    if selected in ("llm", "llm_fallback"):
        model_used = llm_model_name
    elif selected == "residual_nn":
        model_used = f"residual({llm_model_name}+{_NN_MODEL_NAME})"
    elif selected == "blend":
        model_used = f"blend({llm_model_name}+{_NN_MODEL_NAME},alpha={alpha:.2f})"
    elif selected in ("linear_fallback", "linear_fallback_local"):
        # Fix 21/22/23: deterministic extrapolation-safe fallback -- not an
        # LLM formula and not the NN model, so neither existing label fits.
        model_used = selected
    else:  # "nn" or "nn_fallback"
        model_used = _NN_MODEL_NAME

    return {
        "train_r2": float(train_r2), "test_r2": float(test_r2),
        "decision": f"v4_{selected}", "llm_train_r2": float(llm_train_r2),
        "llm_trustworthy": trustworthy, "selected_candidate": selected,
        "selected_hidden": hidden, "blend_alpha": alpha,
        "validation_r2": selection["validation_r2"],
        "validation_n": selection["validation_n"], "timed_out": timed_out,
        "llm_code": llm_code if llm_code else None,
        "llm_model": llm_model_name,   # model that generated llm_code
        "model_used": model_used,      # model(s) that actually produced test_r2
        "extrapolation_unmitigated": extrapolation_unmitigated,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Data splitting
# ─────────────────────────────────────────────────────────────────────────────

# FIX-C3: PCA-directed 40/60 split — imported from the local hypatiax
# package (NOT inlined). Identical to the reference implementation
# verified by Gate A of ci_runner_disclosure.yml. Called directly at the
# case loop (see pca_directed_split() call below); replaces v3c.py's
# _aggressive_split() for every case.
from hypatiax.tools.utils.pca_split_utils import pca_directed_split

# NOTE: the old v3c.py-style _aggressive_split() wrapper was removed here.
# It was dead code (never called) whose return-tuple order
# (X_train, X_test, y_train, y_test) did not match v3c.py's convention
# (X_train, y_train, X_test, y_test), and would have silently swapped
# y_train/X_test if anyone had wired it in. The call site below unpacks
# pca_directed_split() directly and correctly.


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Test catalogue (74 cases, 0 intractable)
# ─────────────────────────────────────────────────────────────────────────────

def _get_test_cases() -> list[dict]:
    """
    Return the 74-case catalogue (all tractable).
    """
    return [
        # ── EASY (24) ──────────────────────────────────────────────────────
        {"name": "Value at Risk at 95%",             "domain": "risk_var",    "difficulty": "easy",   "formula_type": "linear",          "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Value at Risk at 99%",             "domain": "risk_var",    "difficulty": "easy",   "formula_type": "linear",          "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Partial liquidation amount",         "domain": "liquidation", "difficulty": "easy",   "formula_type": "linear",          "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Constant product formula",       "domain": "amm",         "difficulty": "easy",   "formula_type": "rational_simple", "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Simple Staking APY",               "domain": "staking",     "difficulty": "easy",   "formula_type": "linear",          "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Loan-to-Value",                    "domain": "lending",     "difficulty": "easy",   "formula_type": "rational_simple", "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Spot price from AMM",              "domain": "amm",         "difficulty": "easy",   "formula_type": "rational_simple", "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "LP share percentage",              "domain": "amm",         "difficulty": "easy",   "formula_type": "rational_simple", "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Long position unrealized PnL",     "domain": "trading",     "difficulty": "easy",   "formula_type": "linear",          "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Short position unrealized PnL",    "domain": "trading",     "difficulty": "easy",   "formula_type": "linear",          "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Funding rate cost",                "domain": "trading",     "difficulty": "easy",   "formula_type": "linear",          "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Validator commission adjusted",    "domain": "staking",     "difficulty": "easy",   "formula_type": "linear",          "num_samples": 200, "config": {"split_var_idx": 1, "split_type": "high"}},
        {"name": "Slashing penalty",                 "domain": "staking",     "difficulty": "easy",   "formula_type": "linear",          "num_samples": 200, "config": {"split_var_idx": 1, "split_type": "high"}},
        {"name": "Protocol reserve accumulation",    "domain": "lending",     "difficulty": "easy",   "formula_type": "linear",          "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Leveraged position notional",      "domain": "trading",     "difficulty": "easy",   "formula_type": "linear",          "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Cross-margin available balance",   "domain": "liquidation", "difficulty": "easy",   "formula_type": "linear",          "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Realized PnL for long",            "domain": "liquidation", "difficulty": "easy",   "formula_type": "linear",          "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "LP fee earnings",                  "domain": "liquidity",   "difficulty": "easy",   "formula_type": "rational_simple", "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Multi-day Value at Risk",          "domain": "risk_var",    "difficulty": "easy",   "formula_type": "algebraic",       "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "ES scaling for multi-day",         "domain": "expected_shortfall", "difficulty": "easy", "formula_type": "algebraic",  "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Annualised Portfolio tracking error", "domain": "risk_var", "difficulty": "easy",   "formula_type": "algebraic",       "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Incremental VaR",                  "domain": "risk_var",    "difficulty": "easy",   "formula_type": "linear",          "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        # Fix 0: Reserve Ratio uses independent log-uniform sampling (handled in protocol)
        {"name": "Reserve ratio",                    "domain": "amm",         "difficulty": "easy",   "formula_type": "rational_simple", "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Impermanent loss breakeven fee rate", "domain": "liquidity",  "difficulty": "easy",   "formula_type": "rational_simple", "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},

        # ── MEDIUM (28) ────────────────────────────────────────────────────
        {"name": "Liquidation Price Long",           "domain": "trading",     "difficulty": "medium", "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 1, "split_type": "high"}},
        {"name": "Liquidation Price Short",          "domain": "trading",     "difficulty": "medium", "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 1, "split_type": "high"}},
        {"name": "Constant Product Price Impact",    "domain": "amm",         "difficulty": "medium", "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Effective Leverage",               "domain": "trading",     "difficulty": "medium", "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Borrowing Interest",               "domain": "lending",     "difficulty": "medium", "formula_type": "exponential",     "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Compounding Staking Returns",      "domain": "staking",     "difficulty": "medium", "formula_type": "exponential",     "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Portfolio Sharpe Ratio",           "domain": "risk",        "difficulty": "medium", "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 1, "split_type": "high"}},
        {"name": "APY calculation",                  "domain": "liquidity",   "difficulty": "medium", "formula_type": "exponential",     "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Capital efficiency",               "domain": "liquidity",   "difficulty": "medium", "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "AMM output amount",                "domain": "amm",         "difficulty": "medium", "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Price slippage percentage",        "domain": "amm",         "difficulty": "medium", "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Information ratio",                "domain": "risk_var",    "difficulty": "medium", "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Utilization rate of DeFi",         "domain": "lending",     "difficulty": "medium", "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Borrow APY from utilization",      "domain": "lending",     "difficulty": "medium", "formula_type": "linear",          "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Supply APY from borrow",           "domain": "lending",     "difficulty": "medium", "formula_type": "linear",          "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Health factor",                    "domain": "lending",     "difficulty": "medium", "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Impermanent loss percentage",      "domain": "amm",         "difficulty": "medium", "formula_type": "algebraic_with_sqrt", "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Options Delta",                    "domain": "derivatives", "difficulty": "medium", "formula_type": "norm_cdf",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Collateral ratio",                 "domain": "lending",     "difficulty": "medium", "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Funding rate cost (extended)",             "domain": "trading",     "difficulty": "medium", "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Staking reward for fixed lock-up",              "domain": "staking",     "difficulty": "medium", "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Expected Shortfall at 95%",            "domain": "expected_shortfall", "difficulty": "medium", "formula_type": "linear",   "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Expected Shortfall at 99%",            "domain": "expected_shortfall", "difficulty": "medium", "formula_type": "linear",   "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Optimal LP Position (Kelly)",                   "domain": "liquidity",   "difficulty": "medium", "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Concentrated liquidity position width",                  "domain": "liquidity",   "difficulty": "medium", "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Portfolio VaR for two correlated",      "domain": "risk_var",    "difficulty": "medium", "formula_type": "algebraic",       "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Position margin ratio",               "domain": "liquidation", "difficulty": "medium", "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Concentrated liquidity position width (v2)", "domain": "liquidity", "difficulty": "medium", "formula_type": "algebraic_with_sqrt", "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Spot price from AMM reserve",         "domain": "amm",         "difficulty": "medium", "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},

        # ── HARD (21, of which 6 intractable) ─────────────────────────────
        {"name": "Black-Scholes Call Price",         "domain": "derivatives", "difficulty": "hard",   "formula_type": "norm_cdf",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Black-Scholes Put Price",          "domain": "derivatives", "difficulty": "hard",   "formula_type": "norm_cdf",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Component ES",                     "domain": "expected_shortfall", "difficulty": "hard", "formula_type": "quadratic_form", "num_samples": 200, "config": {"split_var_idx": 1, "split_type": "high"}},
        {"name": "Gamma of option",                  "domain": "derivatives", "difficulty": "hard",   "formula_type": "norm_pdf",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Vega of option",                   "domain": "derivatives", "difficulty": "hard",   "formula_type": "norm_pdf",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Multi-Collateral LTV",             "domain": "lending",     "difficulty": "hard",   "formula_type": "weighted_aggregate", "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        # Fix 5 target case
        {"name": "Correlated Portfolio VaR",         "domain": "risk",        "difficulty": "hard",   "formula_type": "quadratic_form",  "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        # Fix 5 target case
        {"name": "Impermanent loss in constant product", "domain": "amm",     "difficulty": "hard",   "formula_type": "algebraic_with_sqrt", "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Constant product formula (multivariate)",         "domain": "amm",         "difficulty": "hard",   "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Convexity Adjustment",             "domain": "amm",         "difficulty": "hard",   "formula_type": "algebraic",       "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Liquidation price for leveraged long",  "domain": "liquidation", "difficulty": "hard", "formula_type": "rational",    "num_samples": 200, "config": {"split_var_idx": 1, "split_type": "high"}},
        {"name": "Liquidation price for leveraged short", "domain": "liquidation", "difficulty": "hard", "formula_type": "rational",    "num_samples": 200, "config": {"split_var_idx": 1, "split_type": "high"}},
        {"name": "Maximum safe leverage",            "domain": "liquidation", "difficulty": "hard",   "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 1, "split_type": "high"}},
        {"name": "Required collateral",              "domain": "liquidation", "difficulty": "hard",   "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 1, "split_type": "high"}},
        {"name": "Forward price for derivative",     "domain": "derivatives", "difficulty": "hard",   "formula_type": "exponential",     "num_samples": 200, "config": {"split_var_idx": 1, "split_type": "high"}},
        {"name": "Portfolio Expected Shortfall for correlated", "domain": "expected_shortfall", "difficulty": "hard", "formula_type": "quadratic_form", "num_samples": 200, "config": {"split_var_idx": 2, "split_type": "high"}},
        {"name": "Call option intrinsic",            "domain": "derivatives", "difficulty": "hard",   "formula_type": "piecewise_linear","num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Put-call parity",                  "domain": "derivatives", "difficulty": "hard",   "formula_type": "exponential",     "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Simple options moneyness",         "domain": "derivatives", "difficulty": "hard",   "formula_type": "rational",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Uniswap V3 virtual",               "domain": "amm",         "difficulty": "hard",   "formula_type": "algebraic_with_sqrt", "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
        {"name": "Theta of option",                  "domain": "derivatives", "difficulty": "hard",   "formula_type": "norm_pdf",        "num_samples": 200, "config": {"split_var_idx": 0, "split_type": "high"}},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_checkpoint() -> tuple[list, int]:
    if not CHECKPOINT_FILE.exists():
        return [], 0
    try:
        data = json.loads(CHECKPOINT_FILE.read_text())
        if not isinstance(data, list):
            return [], 0
        # Deduplicate — keep last occurrence of each (case name, seed) pair.
        # seed-loop fix: without the seed in the key, a multi-seed sweep would
        # collapse to one record per case (last seed wins) on resume.
        seen = {}
        for item in data:
            key = (item.get("equation_id", id(item)), item.get("seed"))
            seen[key] = item
        data = list(seen.values())
        return data, len(data)
    except Exception:
        return [], 0


def _save_checkpoint(results: list):
    def _default(obj):
        if isinstance(obj, (np.integer,)):         return int(obj)
        if isinstance(obj, (np.floating, float)):
            if np.isnan(obj) or np.isinf(obj):    return None
            return float(obj)
        if isinstance(obj, np.bool_):              return bool(obj)
        if isinstance(obj, np.ndarray):            return obj.tolist()
        raise TypeError(f"Not serialisable: {type(obj)}")
    CHECKPOINT_FILE.write_text(json.dumps(results, indent=2, default=_default))


def _save_final(results: list):
    def _default(obj):
        if isinstance(obj, (np.integer,)):         return int(obj)
        if isinstance(obj, (np.floating, float)):
            if np.isnan(obj) or np.isinf(obj):    return None
            return float(obj)
        if isinstance(obj, np.bool_):              return bool(obj)
        if isinstance(obj, np.ndarray):            return obj.tolist()
        raise TypeError(f"Not serialisable: {type(obj)}")
    FINAL_OUTPUT.write_text(json.dumps(results, indent=2, default=_default))
    print(f"\n✅ Final results saved → {FINAL_OUTPUT}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Statistical report (honest fixed-denominator)
# ─────────────────────────────────────────────────────────────────────────────

_INTRACTABLE_NAMES = {
    tc["name"] for tc in _get_test_cases() if tc.get("extrapolation_intractable")
}
_STANDARD_TOTAL = len(_get_test_cases()) - len(_INTRACTABLE_NAMES)  # = 66 + adjustments


def _generate_report(results: list):
    CLIP_LO    = -10.0
    standard   = [r for r in results if r["equation_id"] not in _INTRACTABLE_NAMES]
    intractable = [r for r in results if r["equation_id"] in _INTRACTABLE_NAMES]

    def _r2s(rlist, method):
        out = []
        for r in rlist:
            v = r["results"].get(method, {}).get("test_r2")
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                out.append(float(v))
        return out

    def _stats(scores, denom):
        arr     = np.array(scores, dtype=float)
        clipped = np.clip(arr, CLIP_LO, 1.0)
        return {
            "n":             len(arr),
            "median":        float(np.median(clipped)) if len(clipped) else float("nan"),
            "mean_clip":     float(np.mean(clipped))   if len(clipped) else float("nan"),
            "pct_09":        100 * int(np.sum(arr > 0.9))  / denom,
            "pct_099":       100 * int(np.sum(arr > 0.99)) / denom,
            "catastrophic":  int(np.sum(arr < CLIP_LO)),
        }

    # Fixed denominator = number of STANDARD cases run so far
    denom = max(len(standard), 1)

    print("\n" + "=" * 80)
    print("STATISTICAL REPORT — HypatiaX DeFi Benchmark v4.0 (PCA split)")
    print("=" * 80)
    print(f"Total cases run : {len(results)}")
    print(f"  Standard      : {len(standard)}  (used in aggregate, denominator = {denom})")
    print(f"  Intractable   : {len(intractable)}  (reported separately, excluded from aggregate)")
    print()
    print("NOTE: All percentages use FIXED denominator (standard cases only).")
    print("      NaN / failure counts as 0 toward the denominator. This makes")
    print("      LLM, NN, and Hybrid rates directly comparable.\n")

    for method in ["pure_llm", "neural_network", "hybrid"]:
        s = _stats(_r2s(standard, method), denom)
        label = {"pure_llm": "Pure LLM      ", "neural_network": "Neural Network",
                 "hybrid":   "Hybrid        "}[method]
        print(f"  {label}: median={s['median']:.4f}, "
              f"mean(clip-{abs(CLIP_LO):.0f})={s['mean_clip']:.4f}, "
              f">0.9: {s['pct_09']:.1f}%, "
              f">0.99: {s['pct_099']:.1f}%  "
              f"(catastrophic R²<{CLIP_LO}: {s['catastrophic']})")

    if intractable:
        print(f"\n── Intractable cases ({len(intractable)}) ─────────────────────────────────────")
        for r in intractable:
            hy = r["results"].get("hybrid", {}).get("test_r2")
            # BUGFIX: this read r["test_case"], a key that has never existed on
            # these records (only "equation_id" does) — dead code that would
            # have raised KeyError the first time this codebase actually had
            # an intractable case in `results`. Currently unreachable in
            # practice since _get_test_cases() ships 0 intractable cases, but
            # fixed here while touching this section rather than left latent.
            print(f"  {r['equation_id'][:55]:<55}  hybrid test R² = "
                  f"{'nan' if hy is None or (isinstance(hy, float) and np.isnan(hy)) else f'{hy:.4f}'}")

    # By-difficulty breakdown
    print("\n── By difficulty ─────────────────────────────────────────────────────────")
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in standard if r.get("difficulty") == diff]
        if not subset:
            continue
        for method in ["pure_llm", "hybrid"]:
            s = _stats(_r2s(subset, method), max(len(subset), 1))
            print(f"  {diff:6s} | {method:14s}: median={s['median']:.4f}, "
                  f">0.99: {s['pct_099']:.1f}%  (n={len(subset)})")

    # ── Timing and speedup ─────────────────────────────────────────────────
    print("\n── Timing summary (per-case mean) ────────────────────────────────────────")

    # Identify timed-out NN cases (Issue 2)
    timed_out_cases = [
        r["equation_id"] for r in results
        if r["results"].get("neural_network", {}).get("timed_out", False)
    ]
    if timed_out_cases:
        print(f"  ⚠️  NN wall-clock limit hit ({_NN_MAX_TIME_S}s) in "
              f"{len(timed_out_cases)} case(s): {', '.join(timed_out_cases)}")

    for method in ["pure_llm", "neural_network", "hybrid"]:
        times = [
            r["results"].get(method, {}).get("time_s", 0.0)
            for r in results
            if r["results"].get(method, {}).get("time_s") is not None
        ]
        times = [t for t in times if t and t > 0]
        if times:
            label = {"pure_llm": "Pure LLM      ", "neural_network": "Neural Network",
                     "hybrid":   "Hybrid        "}[method]
            print(f"  {label}: mean={np.mean(times):.1f}s, "
                  f"median={np.median(times):.1f}s  (n={len(times)})")

    nn_times  = [r["results"].get("neural_network", {}).get("time_s", 0.0) or 0.0 for r in results]
    hyb_times = [r["results"].get("hybrid",         {}).get("time_s", 0.0) or 0.0 for r in results]
    nn_times  = [t for t in nn_times  if t > 0]
    hyb_times = [t for t in hyb_times if t > 0]
    if nn_times and hyb_times and len(nn_times) == len(hyb_times):
        speedup_mean   = np.mean(nn_times) / np.mean(hyb_times)
        speedup_median = np.median(nn_times) / np.median(hyb_times)
        verdict = (f"Hybrid {speedup_mean:.2f}× faster than NN (mean)"
                   if speedup_mean > 1 else f"Hybrid {1/speedup_mean:.2f}× slower than NN (mean)")
        print(f"  Hybrid vs NN speedup: {verdict}")
        print(f"  Hybrid vs NN speedup (median): {speedup_median:.2f}×")
        print("  (Paper claims 73% reduction = 3.7×)")
        reduction_mean   = (1 - np.mean(hyb_times) / np.mean(nn_times)) * 100
        reduction_median = (1 - np.median(hyb_times) / np.median(nn_times)) * 100
        print(f"  This run: mean-based {reduction_mean:.1f}% reduction, "
              f"median-based {reduction_median:.1f}% reduction")

        # Issue 3: also report hybrid speedup excluding NN-fallback cases
        # (these are the only fair comparison — pure LLM call vs NN).
        # v4 decision values are "v4_llm" / "v4_nn" / "v4_residual_nn" /
        # "v4_blend" / "v4_nn_fallback" — filter on the whole "v4_nn"/
        # "v4_nn_fallback" strings rather than the old bare "nn"/"nn_fallback".
        llm_only_nn   = [r["results"].get("neural_network", {}).get("time_s", 0.0) or 0.0
                          for r in results
                          if r["results"].get("hybrid", {}).get("decision", "") not in
                          ("v4_nn", "v4_nn_fallback") and r["results"].get("neural_network")]
        llm_only_hyb  = [r["results"].get("hybrid", {}).get("time_s", 0.0) or 0.0
                          for r in results
                          if r["results"].get("hybrid", {}).get("decision", "") not in
                          ("v4_nn", "v4_nn_fallback") and r["results"].get("hybrid")]
        if llm_only_nn and llm_only_hyb and len(llm_only_nn) == len(llm_only_hyb):
            sp_llm = np.mean(llm_only_nn) / np.mean(llm_only_hyb)
            print(f"  Hybrid vs NN speedup (LLM-routed cases only, n={len(llm_only_nn)}): "
                  f"{sp_llm:.2f}× — this is the cleanest comparison for the paper")

    print("\n" + "=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — Main runner
# ─────────────────────────────────────────────────────────────────────────────

# Cases where Fix 5 is known to have been broken — use these for quick verification
_FIX5_VERIFY_CASES = {
    "Correlated Portfolio VaR",
    "Impermanent loss in constant product",
    "Capital efficiency",
    "Concentrated liquidity position width (v2)",
}


def run_benchmark(resume: bool = False, verify_fix5: bool = False,
                  verbose: bool = False, cases: list = None,
                  seeds: list = None):
    protocol   = DeFiExperimentProtocol()
    test_cases = _get_test_cases()
    total      = len(test_cases)

    # ── Env-driven overrides (set by run_all.py via experiment_protocol_defi_v3.py) ──
    # DEFI_TASK_FILTER: run only cases whose name contains this substring
    _task_filter = os.environ.get("DEFI_TASK_FILTER")
    if _task_filter and not cases:
        cases = [_task_filter]
        print(f"\n  DEFI_TASK_FILTER={_task_filter!r}: filtering to matching cases")
    # DEFI_SEEDS: comma-separated seed list for multi-seed sweep
    _seeds_env = os.environ.get("DEFI_SEEDS")
    if _seeds_env and seeds is None:
        seeds = [int(s) for s in _seeds_env.split(",")]
        print(f"\n  DEFI_SEEDS={_seeds_env!r}: will run seeds {seeds}")
    # TASK_IDS: exact-name shard filter set by CI worker (space/comma-separated)
    test_cases = _apply_task_ids_defi(test_cases)
    # SEED: env-driven override (PYSR_SEED → EXPERIMENT_SEED → NN_SEED → 42)
    _env_seed = _resolve_seed()
    if _env_seed is not None and seeds is None:
        seeds = [_env_seed]
        print(f"\n  SEED override from env: seeds={seeds}")

    if verify_fix5:
        test_cases = [tc for tc in test_cases if tc["name"] in _FIX5_VERIFY_CASES]
        total      = len(test_cases)
        print(f"\n🔍 Fix-5 verification mode: running {total} target cases only")

    # ONE_EQUATION smoke-test: run only the first case
    # Triggered by run_all_checkpoint.py --one-equation (sets ONE_EQUATION=1 in env).
    if os.environ.get("ONE_EQUATION") == "1" and not verify_fix5 and not cases:
        test_cases = test_cases[:1]
        total = 1
        print(f"\n🔥 Smoke-test mode (ONE_EQUATION=1): running 1 of {len(_get_test_cases())} cases only")

    # --cases filter: keep only cases whose name contains any of the given substrings
    if cases:
        filters = [c.lower() for c in cases]
        test_cases = [
            tc for tc in test_cases
            if any(f in tc["name"].lower() for f in filters)
        ]
        total = len(test_cases)
        if total == 0:
            print(f"❌ No test cases matched filters: {cases}")
            print("Available cases:")
            for tc in _get_test_cases():
                print(f"  {tc['name']}")
            return []
        print(f"\n🔍 Case filter active: running {total} case(s) — "
              f"{[tc['name'] for tc in test_cases]}")

    global CHECKPOINT_FILE, FINAL_OUTPUT

    # ── Multi-seed sweep support (mirrors hypatiax_defi_benchmark_v3c.py) ──
    # `seeds` (from DEFI_SEEDS env, a bare SEED override, or an explicit
    # seeds= kwarg) drives a real sweep: we loop over every seed in
    # `seeds`, reseeding all RNGs and writing a DISTINCT, seed-tagged
    # checkpoint/output file per seed whenever a sweep was explicitly
    # requested — not just when more than one seed is present, since a
    # single-seed CI shard run is still logically part of a sweep and must
    # get a file that survives alongside every other shard's (see
    # FIX-SINGLE-SEED-SHARD in v3c.py). Single-seed / no-seed runs keep the
    # plain, unsuffixed filenames.
    _base_results_dir = RESULTS_DIR
    _orig_checkpoint, _orig_final = CHECKPOINT_FILE, FINAL_OUTPUT
    seed_list  = seeds if seeds else [None]
    multi_seed = len(seed_list) > 1 or bool(_seeds_env)
    all_seed_results = []

    for _seed_idx, _seed in enumerate(seed_list, 1):
        if _seed is not None:
            random.seed(_seed); np.random.seed(_seed); torch.manual_seed(_seed)
            if multi_seed:
                print(f"\n🌱 Seed sweep {_seed_idx}/{len(seed_list)}: seed={_seed}")
            else:
                print(f"\n🌱 Seed = {_seed}")
            _nn_seed = _seed
        else:
            _nn_seed = _NN_SEED

        if multi_seed:
            CHECKPOINT_FILE = _base_results_dir / f"hypatiax_defi_benchmark_pca_checkpoint_seed{_seed}.json"
            FINAL_OUTPUT    = _base_results_dir / f"hypatiax_defi_benchmark_pca_results_seed{_seed}.json"
        else:
            CHECKPOINT_FILE, FINAL_OUTPUT = _orig_checkpoint, _orig_final

        # NSHARDS=1 FIX: on a fresh (non-resume) run, remove any stale checkpoint
        # and final-output JSON left over from a prior run.  Without this, a second
        # run in the same workspace smuggles the prior run's cases through
        # _load_checkpoint, producing duplicate records and a bloated output file
        # that ci_analysis misreads as multiple seeds.
        if not resume:
            if CHECKPOINT_FILE.exists():
                CHECKPOINT_FILE.unlink()
                print(f"  [fresh run] Removed stale checkpoint: {CHECKPOINT_FILE}")
            if FINAL_OUTPUT.exists():
                FINAL_OUTPUT.unlink()
                print(f"  [fresh run] Removed stale output: {FINAL_OUTPUT}")

        existing, n_done = _load_checkpoint() if resume else ([], 0)
        all_results       = list(existing)

        print("=" * 80)
        print("HypatiaX DeFi Extrapolation Benchmark v4.0 (PCA split)")
        print("=" * 80)
        print(f"Cases: {total} | Resuming from: {n_done + 1}" if resume else
              f"Cases: {total} | Fresh run")
        print(f"Checkpoint: {CHECKPOINT_FILE}")
        print(f"Output    : {FINAL_OUTPUT}")
        print("=" * 80)

        for i, tc in enumerate(test_cases, 1):
            # Skip already-done cases when resuming (separate file per seed,
            # so no need to also match on seed here).
            if resume and any(r.get("equation_id") == tc["name"] for r in all_results):
                print(f"[{i:02d}/{total}] ⏭  {tc['name']} — already done")
                continue

            is_intractable = tc.get("extrapolation_intractable", False)
            print(f"\n[{i:02d}/{total}] {tc['name']}  "
                  f"({tc['difficulty'].upper()}"
                  f"{' — INTRACTABLE' if is_intractable else ''})"
                  f"{f' seed={_seed}' if _seed is not None else ''}")

            try:
                # Load protocol data
                protocol_cases = protocol.load_test_data(
                    tc["domain"], num_samples=tc["num_samples"]
                )
                match = next(
                    ((d, X, y, v, m) for d, X, y, v, m in protocol_cases
                     if tc["name"].lower() in d.lower()),
                    None,
                )
                if not match:
                    print(f"  ⚠️  No protocol match for '{tc['name']}' — skipping")
                    continue

                desc, X_full, y_full, var_names, metadata = match
                metadata.update({
                    "extrapolation_test": True,
                    "difficulty":         tc["difficulty"],
                    "formula_type":       tc["formula_type"],
                })
                tc.setdefault("description", desc)

                # FIX-C3: PCA-directed 40/60 split replaces _aggressive_split.
                # Matches the Feynman benchmark protocol (run_comparative_suite_benchmark_pca.py).
                # FIX-C3b (audit finding): random_state was hardcoded to the literal
                # 42 here, so every seed in a multi-seed sweep (exp1b_pca /
                # DEFI_SEEDS=42,99,123,777,2024) trained/tested on the IDENTICAL
                # data partition — only NN init and LLM sampling varied. The sweep
                # therefore never measured split-sensitivity, only two much
                # narrower sources of stochasticity. Now uses the loop's actual
                # _seed (falling back to 42 to preserve single-run behavior when
                # no seed is supplied), so each swept seed gets its own split.
                _split_seed = _seed if _seed is not None else 42
                X_tr, X_te, y_tr, y_te = pca_directed_split(
                    X_full, y_full, test_size=0.6, random_state=_split_seed
                )
                print(f"  Split (PCA 40/60, random_state={_split_seed}) → train={len(X_tr)}, test={len(X_te)}")

                case_results = {}

                # FIX 17 (ported from hypatiax_defi_benchmark_v4.py): the exact
                # code string and model pure_llm scored, threaded into the hybrid
                # arm below so both arms are gated/scored on one shared LLM call
                # rather than two independent, differently-prompted ones. Code
                # stays "" (never None) and model stays None on any pure_llm
                # failure path so the hybrid arm treats "pure_llm had no usable
                # formula" the same way pure_llm did, instead of silently
                # reverting to its own standalone LLM call.
                _pure_llm_code_for_hybrid = ""
                _pure_llm_model_for_hybrid = None

                # ── Pure LLM ────────────────────────────────────────────────────
                try:
                    _t0_llm = time.time()
                    from hypatiax.core.base_pure_llm.baseline_pure_llm_defi_discovery import (
                        PureLLMBaseline,
                    )
                    import sys
                    import hypatiax
                    import hypatiax.core.base_pure_llm as _m

                    print(
                        f"MODULE_RESOLUTION: "
                        f"hypatiax={hypatiax.__file__} "
                        f"base_pure_llm_defi_discovery={_m.baseline_pure_llm_defi_discovery.__file__} "
                        f"sys_path0={sys.path[0]}",
                        flush=True,
                    )
                    # FIX-ITEM1-MODEL-MISMATCH: pin explicitly to the same
                    # constant the hybrid arm's inline LLM call uses, rather
                    # than relying on PureLLMBaseline's own default, which
                    # drifted out of sync with Fix 13/14 and caused every
                    # standalone pure_llm call to fail near-instantly on an
                    # invalid model string (see consolidation report §4 item 1).
                    llm_base  = PureLLMBaseline(model=_HYBRID_LLM_MODEL_NAME)
                    llm_res   = llm_base.generate_formula(desc, tc["domain"],
                                                          var_names, metadata)

                    # DEBUG-ITEM1-CACHE-HYPOTHESIS: see matching comment in
                    # hypatiax_defi_benchmark_v4.py. This script's non-PCA
                    # sibling shows exactly one real-network-latency pure_llm
                    # call per shard (the first case, ~0.6-0.8s) followed by
                    # ~73 near-zero-time (0.001s) failures too fast to be real
                    # API round trips -- consistent with an external patch
                    # (PureLLMBaseline's unused self._cache, "added by
                    # apply_patches") memoizing on a key that doesn't vary
                    # per-equation. This matters even more here: if seed 42's
                    # reported partial pure_llm success is a cache hit from
                    # this shard's own first case rather than an independently
                    # -derived answer for each equation, the same cached
                    # formula could be getting silently stamped onto every
                    # case in this shard.
                    import hashlib as _hashlib
                    _raw = llm_res.get("raw_response", "") or ""
                    print(
                        f"DEBUG pure_llm[{tc['name']}]: "
                        f"cache_id={id(getattr(llm_base, '_cache', None))} "
                        f"cache_keys={list(getattr(llm_base, '_cache', {}).keys())} "
                        f"python_code_present={'python_code' in llm_res} "
                        f"raw_response_hash={_hashlib.sha256(_raw.encode()).hexdigest()[:12]} "
                        f"raw_response_snippet={_raw[:80]!r}"
                    )

                    # FIX 12 (ported from v4.0): reject truncated formulas before
                    # scoring — see _is_truncated_formula() above for why.
                    _llm_code = llm_res.get("python_code", "") or llm_res.get("formula_code", "") or ""
                    if _is_truncated_formula(_llm_code):
                        case_results["pure_llm"] = {
                            "train_r2": float("nan"), "test_r2": float("nan"),
                            "executed": False, "success": False,
                            "time_s": round(time.time() - _t0_llm, 3),
                            "error": "truncated_formula: no valid return statement",
                            "model": llm_res.get("model"),
                            # DEBUG-ITEM1-CACHE-HYPOTHESIS: real exception text
                            # from generate_formula()'s except clause, previously
                            # discarded in favor of the generic message above —
                            # persisted here so it survives into the JSON we
                            # already collect, no separate logs needed.
                            "llm_internal_error": llm_res.get("error"),
                            "debug_cache_keys": list(getattr(llm_base, "_cache", {}).keys()),
                        }
                    else:
                        # FIX 17: this is the same code string (and model) pure_llm
                        # is about to score below — share it with the hybrid arm's
                        # call further down instead of letting hybrid re-derive its
                        # own formula from a second, independent LLM call.
                        _pure_llm_code_for_hybrid = _llm_code
                        _pure_llm_model_for_hybrid = llm_res.get("model")
                        llm_tr_m  = llm_base.test_formula_accuracy(llm_res, X_tr, y_tr,
                                                                   var_names, verbose=False)
                        llm_te_m  = llm_base.test_formula_accuracy(llm_res, X_te, y_te,
                                                                   var_names, verbose=False)
                        case_results["pure_llm"] = {
                            "train_r2": float(llm_tr_m["r2"]) if llm_tr_m.get("success") else float("nan"),
                            "test_r2":  float(llm_te_m["r2"]) if llm_te_m.get("success") else float("nan"),
                            "executed": llm_te_m.get("success", False),
                            # FIX 11: PureLLMBaseline.test_formula_accuracy()'s own "success"
                            # only means the generated code executed without raising — it does
                            # NOT gate on fit quality (observed: 11/74 exp1_pca cases report
                            # success=True with test_r2 as low as -126,483). Recompute success
                            # here as a fit-quality gate, reusing the >0.5 "trustworthy"
                            # threshold already established for the hybrid arm's LLM trust
                            # gate (FIX 10, above) so both arms share one pass definition.
                            "success": bool(
                                llm_te_m.get("success", False)
                                and not _math.isnan(llm_te_m.get("r2", float("nan")))
                                and llm_te_m["r2"] > 0.5
                            ),
                            "time_s":   round(time.time() - _t0_llm, 3),
                            # Model tracking: PureLLMBaseline.generate_formula() reports
                            # which model it used (self.model) at every return path —
                            # surface it here so a run mixing model versions is auditable.
                            "model":    llm_res.get("model"),
                        }
                except Exception as e:
                    # llm_res may not exist if generate_formula() itself raised —
                    # fall back to None rather than assuming a model was used.
                    _attempted_model = llm_res.get("model") if "llm_res" in dir() else None
                    case_results["pure_llm"] = {
                        "train_r2": float("nan"), "test_r2": float("nan"),
                        "executed": False, "success": False, "time_s": 0.0, "error": str(e),
                        "model": _attempted_model,
                    }

                # ── Neural Network ───────────────────────────────────────────────
                try:
                    _t0_nn = time.time()
                    nn_m = _train_and_eval_nn(X_tr, y_tr, X_te, y_te, seed=_nn_seed)
                    # FIX (audit item 1): the NN arm previously hardcoded
                    # "success": True regardless of what _train_and_eval_nn
                    # actually reported, so an internally-caught training
                    # failure (non-finite values, singular matrices, bad
                    # augmentation -> success=False, r2=0.0) still showed up
                    # as a "successful" run in the output. Gate on the NN's
                    # own success flag *and* the same test_r2 > 0.5
                    # fit-quality threshold used by the pure_llm and hybrid
                    # arms, so all three arms share one pass definition.
                    _nn_success = bool(
                        nn_m.get("success", True)
                        and not _math.isnan(nn_m.get("test_r2", float("nan")))
                        and nn_m["test_r2"] > 0.5
                    )
                    case_results["neural_network"] = {
                        "train_r2":    nn_m["train_r2"],
                        "test_r2":     nn_m["test_r2"],
                        "success":     _nn_success,
                        "timed_out":   nn_m.get("timed_out", False),
                        "time_s":      round(time.time() - _t0_nn, 3),
                        "y_pred_train": nn_m["y_pred_train"].tolist(),
                        "y_pred_test":  nn_m["y_pred_test"].tolist(),
                        "model":       _NN_MODEL_NAME,
                    }
                except Exception as e:
                    case_results["neural_network"] = {
                        "train_r2": float("nan"), "test_r2": float("nan"),
                        "success": False, "time_s": 0.0, "error": str(e),
                        "model": _NN_MODEL_NAME,
                    }

                # ── Hybrid (v4.0 validation-selected, all Fixes applied) ──────────
                try:
                    _t0_hyb = time.time()
                    hy_m = _v4_hybrid_predict_and_eval(
                        desc, tc["domain"], X_tr, y_tr, X_te, y_te, var_names, metadata,
                        config=tc["config"], seed=_nn_seed,
                        # FIX 17: reuse pure_llm's already-scored formula (and its
                        # model name, for the llm_model/model_used audit fields)
                        # instead of having the hybrid arm generate its own.
                        pure_llm_code=_pure_llm_code_for_hybrid,
                        pure_llm_model=_pure_llm_model_for_hybrid,
                    )
                    hyb_time = round(time.time() - _t0_hyb, 3)

                    _train_r2 = hy_m["train_r2"]
                    _test_r2  = hy_m["test_r2"]
                    _nan = lambda v: v is None or (isinstance(v, float) and _math.isnan(v))

                    # FIX 13 (Issue 9 — decision-attribution / masked-failure fix):
                    # the previous "fixed" success computation only excluded NaN
                    # test_r2 values, so a catastrophic-but-numeric test_r2 (e.g.
                    # an LLM formula that executes cleanly but extrapolates to a
                    # value like -126,483 — see FIX 11's comment above) still
                    # reported success=True. Recompute success as a real
                    # fit-quality gate, reusing the same >0.5 threshold as
                    # pure_llm's FIX 11, so both arms share one pass definition.
                    _hybrid_success = bool(
                        not _nan(_train_r2)
                        and not _nan(_test_r2)
                        and _test_r2 > 0.5
                    )

                    case_results["hybrid"] = {
                        "train_r2":          _train_r2,
                        "test_r2":           _test_r2,
                        "decision":          hy_m["decision"],       # "v4_llm" / "v4_nn" / "v4_residual_nn" / "v4_blend" / "v4_nn_fallback"
                        "success":           _hybrid_success,        # FIX 13 — fit-quality gate
                        "time_s":            round(hyb_time, 3),
                        "llm_trustworthy":   hy_m.get("llm_trustworthy"),
                        "selected_candidate": hy_m.get("selected_candidate"),
                        # Parity with v4.py: _v4_hybrid_predict_and_eval() already
                        # returns llm_train_r2 but it was dropped here, so the PCA
                        # output JSON couldn't show how well the (reused) pure-LLM
                        # formula fit the training split.
                        "llm_train_r2":      hy_m.get("llm_train_r2"),
                        "validation_r2":     hy_m.get("validation_r2"),
                        "validation_n":      hy_m.get("validation_n"),
                        "timed_out":         hy_m.get("timed_out", False),
                        # Model tracking: llm_model is what generated the candidate
                        # formula; model_used is what actually produced test_r2
                        # (they can differ — e.g. decision="v4_nn_fallback" still
                        # tried an LLM formula first). Both are needed to audit
                        # whether/how models were mixed within this one case.
                        "llm_model":         hy_m.get("llm_model"),
                        "model_used":        hy_m.get("model_used"),
                        # True when the test domain was extrapolative and no
                        # LLM-anchored candidate was available at all (or the
                        # selected candidate failed to fit and was force-
                        # replaced by bare NN) -- distinguishes "genuinely
                        # nothing better was available" from an ordinary
                        # confident NN win.
                        "extrapolation_unmitigated": hy_m.get("extrapolation_unmitigated", False),
                    }
                except Exception as e:
                    case_results["hybrid"] = {
                        "train_r2": float("nan"), "test_r2": float("nan"),
                        "success": False, "time_s": 0.0, "error": str(e),
                        "llm_trustworthy": None, "selected_candidate": None,
                        "llm_train_r2": None,
                        "validation_r2": None, "validation_n": None,
                        "timed_out": None, "llm_model": None, "model_used": None,
                        "extrapolation_unmitigated": None,
                    }

                # ── Augment with extrapolation gap and stability score ────────────
                for method, res in case_results.items():
                    tr_ = res.get("train_r2", float("nan"))
                    te_ = res.get("test_r2",  float("nan"))
                    res["extrapolation_gap"] = (
                        float(tr_ - te_) if not (np.isnan(tr_) or np.isnan(te_)) else float("nan")
                    )
                    res["stability_score"] = (
                        float(te_ / tr_)
                        if (not np.isnan(tr_) and not np.isnan(te_) and abs(tr_) > 1e-6)
                        else float("nan")
                    )

                # ── Print per-case summary ────────────────────────────────────────
                def _fmt(v):
                    return "   nan" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:6.4f}"

                for method, res in case_results.items():
                    dec = f" [{res.get('decision', '')}]" if method == "hybrid" else ""
                    print(f"  {method:15s}: train={_fmt(res.get('train_r2'))}, "
                          f"test={_fmt(res.get('test_r2'))}{dec}")

                record = {
                    "equation_id":            tc["name"],
                    "seed":                 _seed,
                    "difficulty":           tc["difficulty"],
                    "formula_type":         tc["formula_type"],
                    "extrapolation_intractable": is_intractable,
                    "results":              case_results,
                }
                all_results.append(record)
                _save_checkpoint(all_results)
                print(f"  💾 Checkpoint saved ({len(all_results)}/{total})")

            except Exception as outer_e:
                print(f"  ❌ Outer error: {outer_e}")
                continue

        # Final report + save (per seed — each seed gets its own report/output)
        _generate_report(all_results)
        _save_final(all_results)

        # Remove checkpoint on clean completion (not on verify-fix5 partial run)
        if not verify_fix5 and CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            print("🗑️  Checkpoint removed (run complete)")

        all_seed_results.append(all_results)

    # FIX-C3: write split_protocol_disclosure.json so Gate B of
    # ci_runner_disclosure.yml can verify protocol parity with the
    # Feynman benchmark (run_comparative_suite_benchmark_pca.py). One
    # disclosure file for the whole sweep — the split protocol doesn't
    # vary by seed.
    import datetime as _dt
    _disclosure = {
        "split_protocol":   "pca_40_60",
        "script":           Path(__file__).name,
        "test_size":        0.6,
        "train_size":       0.4,
        "random_split_used": False,       # FIX Bug 1: was missing — Gate B key-presence check failed
        "split_function":   "pca_directed_split",
        "split_level":      "outer_loop",
        "force_fresh":      True,
        "seeds":            seed_list,
        "description":      "PC1-directed sort; train on lowest 40%, test on highest 60%",
        "timestamp_utc":    _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    _disc_path = _base_results_dir / "split_protocol_disclosure.json"
    _disc_path.write_text(json.dumps(_disclosure, indent=2))
    print(f"📋 split_protocol_disclosure.json written → {_disc_path}")

    if len(all_seed_results) > 1:
        pooled = _generate_pooled_seed_report(all_seed_results)
        pooled_path = _base_results_dir / "hypatiax_defi_benchmark_pca_pooled_seed_report.json"
        pooled_path.write_text(json.dumps(pooled, indent=2))
        print(f"\n📊 Pooled seed report saved → {pooled_path}")

    return all_seed_results[0] if len(all_seed_results) == 1 else all_seed_results


def _generate_pooled_seed_report(seed_results: list[list]) -> dict:
    """Compute mean test R² by seed, per method (ported from v4.0).

    Each seed is first reduced to a per-case mean, then seed-level means are
    summarized. This avoids giving seeds with duplicate case rows extra
    weight. Both raw and clip-10 means are reported so the old 0.8427-style
    ambiguity is impossible to miss.
    """
    methods = ["pure_llm", "neural_network", "hybrid"]
    out = {"n_seeds": len(seed_results), "methods": {}}
    for method in methods:
        seed_means_raw, seed_means_clip = [], []
        case_values = {}
        for results in seed_results:
            vals = []
            for r in results:
                v = r.get("results", {}).get(method, {}).get("test_r2")
                if v is not None and np.isfinite(v):
                    vals.append(float(v)); case_values.setdefault(r.get("equation_id"), []).append(float(v))
            if vals:
                seed_means_raw.append(float(np.mean(vals)))
                seed_means_clip.append(float(np.mean(np.clip(vals, -10.0, 1.0))))
        out["methods"][method] = {
            "seed_mean_raw": seed_means_raw,
            "mean_of_seed_means_raw": float(np.mean(seed_means_raw)) if seed_means_raw else float("nan"),
            "sd_of_seed_means_raw": float(np.std(seed_means_raw, ddof=1)) if len(seed_means_raw) > 1 else 0.0,
            "mean_of_seed_means_clip10": float(np.mean(seed_means_clip)) if seed_means_clip else float("nan"),
            "n_seed_means": len(seed_means_raw),
        }
    print("\n" + "=" * 80)
    print("POOLED SEED REPORT — HypatiaX DeFi Benchmark (PCA 40/60 split)")
    print("=" * 80)
    print("Metric: test R². Seeds are summarized separately, then averaged.")
    print("Raw mean is un-clipped; clip-10 is shown only for historical comparability.")
    for method, m in out["methods"].items():
        print(f"  {method:15s}: raw mean={m['mean_of_seed_means_raw']:.4f} ± {m['sd_of_seed_means_raw']:.4f} | "
              f"mean(clip-10)={m['mean_of_seed_means_clip10']:.4f} | n={m['n_seed_means']}")
    print("=" * 80)
    return out


def report_only():
    if not FINAL_OUTPUT.exists():
        # Try checkpoint
        src = CHECKPOINT_FILE if CHECKPOINT_FILE.exists() else None
        if src is None:
            print(f"❌ No results file found at {FINAL_OUTPUT}")
            return
    else:
        src = FINAL_OUTPUT
    results = json.loads(src.read_text())
    _generate_report(results)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HypatiaX DeFi Extrapolation Benchmark v4.0 (PCA split)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python hypatiax_defi_benchmark_v3.py                              # full 74-case run
  python hypatiax_defi_benchmark_v3.py --resume                     # continue from checkpoint
  python hypatiax_defi_benchmark_v3.py --verify-fix5               # run only the 4 Fix-5 target cases
  python hypatiax_defi_benchmark_v3.py --report-only               # print report from saved results
  python hypatiax_defi_benchmark_v3.py --cases moneyness           # run only moneyness case(s)
  python hypatiax_defi_benchmark_v3.py --cases moneyness delta     # run multiple named cases
  python hypatiax_defi_benchmark_v3.py --output-dir /tmp/out       # write results to a custom directory
  python hypatiax_defi_benchmark_v3.py --output-dir ./results --resume  # resume from a custom directory
        """,
    )
    parser.add_argument("--resume",      action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--force-fresh", action="store_true", dest="force_fresh",
                        help=(
                            "Delete any existing checkpoint and results files before "
                            "running, guaranteeing fresh results regardless of how "
                            "the script is invoked. Overrides --resume."
                        ))
    parser.add_argument("--verify-fix5", action="store_true",
                        help="Run only the 4 cases targeted by Fix 5 to verify the fix")
    parser.add_argument("--report-only", action="store_true",
                        help="Print statistical report from saved results without running")
    parser.add_argument("--verbose",     action="store_true",
                        help="Extra per-case output")
    parser.add_argument("--cases",       nargs="+", metavar="SUBSTRING",
                        help=(
                            "Run only cases whose name contains any of the given "
                            "substrings (case-insensitive). "
                            "E.g. --cases moneyness   or   --cases 'black-scholes' delta"
                        ))
    parser.add_argument("--output-dir",  metavar="DIR", default=None,
                        help=(
                            "Directory for checkpoint and results JSON files. "
                            "Overrides OUT_BASE env var and the default "
                            "'hypatiax/data/results' path. Created if it does not exist. "
                            "E.g. --output-dir /tmp/benchmark_out"
                        ))
    parser.add_argument("--seeds",       nargs="+", type=int, metavar="SEED", default=None,
                        help=(
                            "One or more seeds to sweep (each case is run once per "
                            "seed, with results tagged by seed). Equivalent to setting "
                            "the DEFI_SEEDS env var. E.g. --seeds 42 99 123 777 2024"
                        ))

    args = parser.parse_args()

    # Apply --output-dir before any I/O touches CHECKPOINT_FILE / FINAL_OUTPUT.
    _configure_output_dir(args.output_dir)

    if args.report_only:
        report_only()
    else:
        # --force-fresh overrides --resume: purge stale files before calling run_benchmark
        if getattr(args, "force_fresh", False):
            import glob as _glob
            for _stale in [CHECKPOINT_FILE, FINAL_OUTPUT]:
                if _stale.exists():
                    _stale.unlink()
                    print(f"  [--force-fresh] Removed stale file: {_stale}")
            # Also purge any shard/seed-tagged variants in the same directory
            for _pat in ["hypatiax_defi_benchmark_pca_checkpoint*.json",
                         "hypatiax_defi_benchmark_pca_results*.json"]:
                for _f in CHECKPOINT_FILE.parent.glob(_pat):
                    _f.unlink()
                    print(f"  [--force-fresh] Removed stale file: {_f}")
            args.resume = False
            print("  [--force-fresh] Checkpoint cleared — running fresh.")
        run_benchmark(
            resume=args.resume,
            verify_fix5=args.verify_fix5,
            verbose=args.verbose,
            cases=args.cases,
            seeds=args.seeds,
        )
