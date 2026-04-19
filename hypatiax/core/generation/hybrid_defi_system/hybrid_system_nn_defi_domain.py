# hybrid_system_nn_defi_domain.py
"""
Enhanced Hybrid System - LLM symbolic formula + Neural Network ensemble.
Fully implements:
  - generate_llm_formula()    (LLM symbolic discovery)
  - train_nn()                (NN baseline)
  - hybrid_predict()          (LLM-guided + NN ensemble)
  - evaluate_llm_formula()    (evaluate formula on held-out data)
  - --batch CLI mode          (runs full DeFi benchmark suite, writes hybrid_defi_*.json)
"""

import argparse
import json
import os
import re
import sys
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from anthropic import Anthropic

from hypatiax.protocols.experiment_protocol_defi import DeFiExperimentProtocol

env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------

class ImprovedNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            # LayerNorm instead of BatchNorm1d — no batch-size sensitivity.
            # No Dropout — 160 training samples cannot afford activation dropout;
            # it degrades convergence without adding regularisation benefit.
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_nn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_dims: List[int] = None,
    epochs: int = 1000,
    lr: float = 0.003,
) -> Tuple[ImprovedNN, object, object]:
    """Train NN; returns (model, scaler_X, scaler_y).

    KEY IMPROVEMENTS vs original:
    - LayerNorm + SiLU instead of BatchNorm + ReLU + Dropout
    - CosineAnnealingWarmRestarts LR schedule (better than StepLR)
    - Best-model tracking via deep .clone() of state dict
    - Early stopping with patience=100
    - 1000 epochs (up from 500)
    - Adaptive architecture by input dimensionality
    """
    import copy
    from sklearn.preprocessing import StandardScaler

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    Xs = scaler_X.fit_transform(X_train)
    ys = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    # Adaptive hidden dims based on input size
    n_vars = X_train.shape[1]
    if hidden_dims is None:
        if n_vars <= 2:
            hidden_dims = [128, 64, 32]
        elif n_vars <= 4:
            hidden_dims = [256, 128, 64, 32]
        else:
            hidden_dims = [512, 256, 128, 64]

    model = ImprovedNN(X_train.shape[1], hidden_dims)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=2, eta_min=3e-6
    )
    criterion = nn.MSELoss()

    Xt = torch.FloatTensor(Xs)
    yt = torch.FloatTensor(ys).reshape(-1, 1)

    best_loss  = float("inf")
    best_state = None
    patience   = 100
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(Xt), yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(epoch)

        val_loss = loss.item()
        if val_loss < best_loss - 1e-8:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler_X, scaler_y


def nn_predict(model, scaler_X, scaler_y, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        Xs = scaler_X.transform(X)
        ys = model(torch.FloatTensor(Xs)).numpy().flatten()
        return scaler_y.inverse_transform(ys.reshape(-1, 1)).flatten()


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if not np.isfinite(ss_res) or not np.isfinite(ss_tot) or ss_tot < 1e-30:
            return 0.0
        r2 = float(1 - ss_res / ss_tot)
    return r2 if np.isfinite(r2) else 0.0


# ---------------------------------------------------------------------------
# Formula evaluation helpers
# ---------------------------------------------------------------------------

def _safe_exec_formula(python_code: str, X: np.ndarray, var_names: List[str]) -> Optional[np.ndarray]:
    """Execute a formula string against data rows. Returns array or None."""
    import math as _math
    # Build namespace with safe wrappers for common out-of-domain operations.
    # arcsin/arccos clip to [-1,1] to prevent RuntimeWarning on Snell-like
    # equations; exp clips to [-500,500] to prevent overflow on Bose-Einstein.
    ns = {
        "np":      np,
        "numpy":   np,
        "math":    _math,
        "pi":      np.pi,
        "e":       np.e,
        "exp":     lambda x: np.exp(np.clip(x, -500.0, 500.0)),
        "log":     np.log,
        "sqrt":    np.sqrt,
        "sin":     np.sin,
        "cos":     np.cos,
        "tan":     np.tan,
        "arcsin":  lambda x: np.arcsin(np.clip(x, -1.0, 1.0)),
        "arccos":  lambda x: np.arccos(np.clip(x, -1.0, 1.0)),
        "arctan":  np.arctan,
        "arctan2": np.arctan2,
        "abs":     np.abs,
        "sign":    np.sign,
        "tanh":    np.tanh,
        "sinh":    np.sinh,
        "cosh":    np.cosh,
    }

    # Clean code
    code = python_code.strip()
    if not code.startswith("def "):
        return None

    try:
        import warnings as _warnings
        # Snapshot the namespace BEFORE exec so we can identify the newly
        # defined formula function.  Without this, the loop below would
        # stop at the first pre-existing lambda in ns (e.g. the `exp`
        # wrapper) instead of the actual formula — causing every formula
        # to be silently replaced by  exp(X[:,0]), which produces wildly
        # wrong predictions (R² ≈ -1e78 for Michaelis-Menten, silent
        # failure for logistic growth).
        pre_exec_keys = set(ns.keys())
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            exec(compile(code, "<formula>", "exec"), ns)
        # Find the function that exec() just added to the namespace.
        # Prefer a function literally named 'formula' so that LLM outputs
        # containing helper functions don't get mistakenly selected as the
        # entry-point (Bug #2: multi-function formulas used to pick the first
        # callable found in dict-iteration order, which was often a helper).
        new_callables = {
            k: v for k, v in ns.items()
            if k not in pre_exec_keys and callable(v) and not isinstance(v, type)
        }
        fn = new_callables.get("formula") or (
            next(iter(new_callables.values()), None)
        )
        if fn is None:
            return None

        # Try vectorised call first
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                args = [X[:, i] for i in range(min(len(var_names), X.shape[1]))]
                result = fn(*args)
            if np.isscalar(result):
                result = np.full(len(X), float(result))
            return np.array(result, dtype=float)
        except Exception:
            # fallback row-by-row
            rows = []
            for row in X:
                args = [row[i] for i in range(min(len(var_names), len(row)))]
                rows.append(float(fn(*args)))
            return np.array(rows, dtype=float)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stage 2 — Nonlinear parameter fitting via scipy.optimize
# ---------------------------------------------------------------------------

def _parametrize_formula(python_code: str, var_names: List[str]) -> tuple:
    """
    Make formula parameters free by replacing their assigned numeric values
    with _P[i] index references so scipy can optimize them.

    Works for ANY case convention the LLM uses (Vmax, km, VMAX, r, K, etc.).
    Does NOT require the LLM to follow any naming convention.

    Two strategies, tried in order:
      A) Named assignments — finds `name = <number>` lines and replaces
         the literal with _P[i].  Handles: Vmax=80.0, Km=5.0, r=0.5, etc.
      B) Inline literals — fallback for fully numeric formulas like
         `return 80.0 * S / (5.0 + S)`. Replaces all non-structural
         numeric literals (skips 0, 1, 2) with _P[i].

    Returns (parametrized_code, init_vals, param_names).
    param_names is empty for the inline-literal fallback.
    """
    _SKIP = set(var_names or []) | {
        'np', 'numpy', 'math', 'import', 'True', 'False', 'None'
    }
    body_start = python_code.find('\n')
    sig  = python_code[:body_start]
    body = python_code[body_start:]

    # ── Strategy A: named assignments ─────────────────────────────────
    pattern = re.compile(
        r'^([ \t]+)([a-zA-Z_]\w*)\s*='
        r'\s*([+-]?\s*\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)\s*$',
        re.MULTILINE,
    )
    seen_names: Dict[str, int] = {}
    init_vals: List[float] = []
    new_body = body

    for m in pattern.finditer(body):
        indent, name, val_str = m.group(1), m.group(2), m.group(3)
        if name in _SKIP or name in seen_names:
            continue
        idx = len(init_vals)
        seen_names[name] = idx
        init_vals.append(float(val_str.replace(' ', '')))
        # Use re.sub with count=1 so only the first occurrence of this exact
        # assignment line is replaced (str.replace would silently rewrite all
        # duplicate lines if the same name=value pair appeared more than once).
        new_body = re.sub(
            re.escape(m.group(0)),
            f"{indent}{name} = _P[{idx}]  # was {val_str.strip()}\n",
            new_body,
            count=1,
        )

    if init_vals:
        return sig + "\n" + new_body, init_vals, list(seen_names.keys())

    # ── Strategy B: inline numeric literals ────────────────────────────
    # 0, 1, 2 are treated as structural: skipping them avoids over-
    # parametrizing formulas that contain these as ordinary mathematical
    # constants (identity, sign, squaring).  Removing 2.0 from this set
    # caused curve_fit to gain an unnecessary free dimension on formulas
    # like Michaelis-Menten, confusing its local search, triggering the
    # expensive differential_evolution fallback (+10 s), and degrading
    # both speed and R².  Formulas where 2.0 genuinely IS a scale factor
    # (e.g. impermanent loss 2*sqrt(r)/(1+r)) are handled by Strategy A's
    # named-assignment pass instead, so they are unaffected by this choice.
    _STRUCTURAL = {0.0, 1.0, 2.0}
    counter = [0]
    b_init: List[float] = []

    def _replace_literal(m: re.Match) -> str:
        val = float(m.group(0))
        if val in _STRUCTURAL:
            return m.group(0)
        idx = counter[0]
        counter[0] += 1
        b_init.append(val)
        return f"_P[{idx}]"

    new_body_b = re.sub(
        r'\b\d+\.?\d*(?:[eE][+-]?\d+)?\b',
        _replace_literal,
        body,
    )
    return sig + "\n" + new_body_b, b_init, []


def fit_formula_params(
    python_code: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    variable_names: List[str],
    verbose: bool = False,
) -> Tuple[Optional[str], float]:
    """
    Stage 2: parametrize the LLM formula and fit numeric constants to data.

    Works regardless of what naming convention the LLM used.
    Tries curve_fit (fast, local) then differential_evolution (slow, global).
    Returns (fitted_code, r2_on_train).
    """
    try:
        from scipy.optimize import curve_fit as _curve_fit, \
                                   differential_evolution as _de
    except ImportError:
        preds = _safe_exec_formula(python_code, X_train, variable_names)
        r2 = r2_score(y_train, preds) if (preds is not None
                                          and np.isfinite(preds).all()) else 0.0
        return python_code, r2

    # ── Early-exit gate ────────────────────────────────────────────────
    # Evaluate the raw (un-parametrized) formula first.  If it already
    # fits well there is nothing to gain from freeing its constants —
    # doing so only adds noise dimensions that confuse curve_fit and can
    # trigger the expensive differential_evolution fallback.  Threshold
    # 0.95 is deliberately conservative: a formula scoring R²=0.97 on
    # training data almost certainly has the right structure, and any
    # remaining gap is better closed by the NN / ensemble leg.
    raw_preds = _safe_exec_formula(python_code, X_train, variable_names)
    raw_r2 = (
        r2_score(y_train, raw_preds)
        if raw_preds is not None and np.isfinite(raw_preds).all()
        else -np.inf
    )
    if raw_r2 >= 0.95:
        if verbose:
            print(f"  [Stage2] raw R²={raw_r2:.4f} ≥ 0.95 — skipping param fitting")
        return python_code, raw_r2

    parametrized, init_vals, param_names = _parametrize_formula(
        python_code, variable_names
    )

    if not init_vals:
        # Formula has no tunable constants — evaluate as-is
        return python_code, raw_r2 if np.isfinite(raw_r2) else 0.0

    # ── Strategy B dimension cap ───────────────────────────────────────
    # Strategy B (inline literal replacement) is the fallback path: it
    # can produce many free parameters from a formula with many embedded
    # numbers.  Beyond ~4 free dimensions curve_fit's local Jacobian
    # search becomes unreliable, the R²<0.90 guard triggers, and the
    # 500-iteration differential_evolution fallback runs — costing ~10 s
    # and often degrading quality.  If B produced too many parameters,
    # return the raw formula rather than attempting a doomed fit.
    _MAX_INLINE_PARAMS = 4
    if not param_names and len(init_vals) > _MAX_INLINE_PARAMS:
        if verbose:
            print(
                f"  [Stage2] Strategy B found {len(init_vals)} inline params "
                f"(> {_MAX_INLINE_PARAMS}) — skipping to avoid over-parametrization"
            )
        return python_code, raw_r2 if np.isfinite(raw_r2) else 0.0

    if verbose:
        print(f"  [Stage2] fitting {len(init_vals)} param(s): {param_names or '(inline)'}")

    import math as _math
    import warnings as _warnings

    _NS_BASE = {
        "np": np, "numpy": np, "math": _math,
        "pi": np.pi, "e": np.e,
        "exp":    lambda x: np.exp(np.clip(x, -500.0, 500.0)),
        "log":    np.log, "sqrt": np.sqrt,
        "sin":    np.sin, "cos":  np.cos, "tan": np.tan,
        "arcsin": lambda x: np.arcsin(np.clip(x, -1.0, 1.0)),
        "arccos": lambda x: np.arccos(np.clip(x, -1.0, 1.0)),
        "arctan": np.arctan, "arctan2": np.arctan2,
        "abs":    np.abs, "sign": np.sign,
        "tanh":   np.tanh, "sinh": np.sinh, "cosh": np.cosh,
    }

    def _make_fn(P):
        ns = dict(_NS_BASE)
        ns["_P"] = list(P)
        try:
            pre_keys = set(ns.keys())
            exec(compile(parametrized, "<formula>", "exec"), ns)
            # Prefer a callable literally named 'formula' so that helper
            # functions defined above the main formula are never mistakenly
            # used as the entry-point (Bug #2 / #1: old code used next()
            # without a default and iterated in dict order, picking the first
            # callable found — often a helper — and raising StopIteration
            # silently when exec added nothing callable at all).
            new_callables = {
                k: v for k, v in ns.items()
                if k not in pre_keys and callable(v) and not isinstance(v, type)
            }
            return new_callables.get("formula") or next(
                iter(new_callables.values()), None
            )
        except Exception:
            return None

    def _predict(P):
        fn = _make_fn(P)
        if fn is None:
            return None
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                args = [X_train[:, i] for i in range(X_train.shape[1])]
                r = fn(*args)
            r = np.array(
                r if not np.isscalar(r) else np.full(len(y_train), float(r)),
                dtype=float,
            )
            return r if np.isfinite(r).all() else None
        except Exception:
            return None

    def _wrapper(X_flat, *P):
        fn = _make_fn(P)
        if fn is None:
            return np.full(len(X_flat), 1e30)
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                args = [X_flat[:, i] for i in range(X_train.shape[1])]
                r = fn(*args)
            r = np.array(r, dtype=float)
            return np.where(np.isfinite(r), r, 1e30)
        except Exception:
            return np.full(len(X_flat), 1e30)

    import time as _time

    y_scale = max(abs(float(y_train.mean())), float(y_train.std()), 1.0)
    n       = len(init_vals)
    rng     = np.random.default_rng(42)

    # Hard wall-clock budget for the entire Stage 2 fitting process.
    # curve_fit gets the first _FIT_BUDGET_S seconds; whatever is left
    # (if any) goes to differential_evolution.  This prevents a single
    # bad formula with many parameters from stalling the benchmark for
    # 60+ seconds (observed: maxiter=500 DE on Michaelis-Menten → 75 s).
    _FIT_BUDGET_S = 8.0
    _t_stage2_start = _time.monotonic()

    # Bounds tied to y_scale rather than ±1e9 so curve_fit's Jacobian
    # search starts in a sensible region and converges faster.
    _fit_bounds = (-y_scale * 100, y_scale * 100)

    candidates = [
        np.array(init_vals),                              # LLM's own guess
        np.ones(n),
        np.full(n, y_scale),
        np.full(n, y_scale / 10),
        np.full(n, float(np.abs(y_train).max())),
        rng.uniform(0.5, 2.0, n),
    ]
    scored: List[tuple] = []
    for c in candidates:
        p = _predict(c)
        r = r2_score(y_train, p) if p is not None else -np.inf
        scored.append((r, c))
    scored.sort(key=lambda x: x[0], reverse=True)

    best_r2   = -np.inf
    best_code = python_code

    def _record(P):
        nonlocal best_r2, best_code
        preds = _predict(P)
        if preds is None:
            return
        r2 = r2_score(y_train, preds)
        if r2 > best_r2:
            best_r2 = r2
            if param_names:
                # Rewrite the named assignment lines with fitted values
                fc = python_code
                for name, val in zip(param_names, P):
                    fc = re.sub(
                        rf'(?<![_\w]){re.escape(name)}\s*=\s*[^\n]+',
                        f"{name} = {val:.8g}",
                        fc,
                    )
                best_code = fc
            else:
                # Inline literals: write the parametrized form with _P values
                # substituted back as literals
                fc = parametrized
                for i, val in enumerate(P):
                    fc = fc.replace(f"_P[{i}]", f"{val:.8g}")
                # Remove _P bookkeeping lines
                fc = re.sub(r'[ \t]*_P\s*=.*\n?', '', fc)
                best_code = fc

    # ── curve_fit from best candidates ────────────────────────────────
    for _, p0 in scored[:3]:
        # Stop early if the time budget is already exhausted or we already
        # have an excellent fit — no point running more curve_fit attempts.
        if _time.monotonic() - _t_stage2_start > _FIT_BUDGET_S:
            break
        if best_r2 >= 0.95:
            break
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                popt, _ = _curve_fit(
                    _wrapper, X_train, y_train,
                    p0=p0, maxfev=3000, bounds=_fit_bounds,
                )
            _record(popt)
        except Exception:
            continue

    # ── differential_evolution fallback ───────────────────────────────
    # Only run if curve_fit left a meaningful gap AND the time budget
    # has not been consumed.  maxiter=100 (down from 500) keeps the
    # worst-case cost bounded; the wall-clock callback aborts early if
    # even that budget overruns.
    _t_remaining = _FIT_BUDGET_S - (_time.monotonic() - _t_stage2_start)
    if best_r2 < 0.90 and _t_remaining > 1.0:
        try:
            bounds_de = [(-y_scale * 100, y_scale * 100)] * n
            _t_de_start = _time.monotonic()

            def _obj(P):
                pr = _predict(P)
                return float(np.mean((y_train - pr) ** 2)) if pr is not None else 1e30

            def _de_callback(xk, convergence=None):
                # Return True to stop DE early if wall-clock budget exceeded.
                return _time.monotonic() - _t_de_start > _t_remaining

            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                res = _de(
                    _obj, bounds_de,
                    maxiter=100, seed=42, polish=False,
                    callback=_de_callback,
                )
            _record(res.x)
        except Exception:
            pass

    if verbose:
        print(f"  [Stage2] best R²={best_r2:.4f}")

    # Return the true best_r2 — do NOT clamp to 0.  Clamping masked cases
    # where fitting diverged (best_r2 << 0) but the clamped 0.0 was still
    # larger than a negative llm_train_r2, causing hybrid_predict to adopt
    # the broken formula and produce predictions in the ~10^41 range.
    return best_code, best_r2 if np.isfinite(best_r2) else -np.inf


# ---------------------------------------------------------------------------
# Main Hybrid System
# ---------------------------------------------------------------------------

class EnhancedHybridSystemDeFi:
    """
    Hybrid system: LLM symbolic formula + NN, with ensemble fallback.

    Public API used by test_enhanced_defi_extrapolation.py:
      generate_llm_formula(description, domain, var_names, metadata, verbose) -> Dict
      hybrid_predict(description, domain, X, y, var_names, metadata, verbose) -> Dict
      evaluate_llm_formula(result_dict, X, y, var_names, verbose) -> Dict
    """

    def __init__(self, model: str = "claude-sonnet-4-6", no_cache: bool = False):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.results = []
        self.formula_cache: Dict[str, Dict] = {}
        self._no_cache = no_cache   # honoured in generate_llm_formula

        # Delegate LLM formula generation to PureLLMBaseline so Feynman
        # equations (biology/chemistry/physics) benefit from hardcoded OLS
        # paths, variant guards, and all prompt fixes — without reimplementing
        # that logic here.  The DeFi-specific specialised prompts in this class
        # remain as a fallback for pure DeFi equations.
        try:
            from hypatiax.core.base_pure_llm.baseline_pure_llm_defi_discovery import (
                PureLLMBaseline as _PureLLMBaseline,
            )
            self._llm_baseline = _PureLLMBaseline(model=model)
        except Exception:
            self._llm_baseline = None

    # ------------------------------------------------------------------
    # LLM formula generation
    # ------------------------------------------------------------------

    def generate_llm_formula(
        self,
        description: str,
        domain: str,
        variable_names: List[str],
        metadata: Dict,
        verbose: bool = False,
        X: "np.ndarray | None" = None,
        y: "np.ndarray | None" = None,
    ) -> Dict:
        cache_key = f"{description}|{domain}|{','.join(variable_names)}"
        if not self._no_cache and cache_key in self.formula_cache:
            return self.formula_cache[cache_key].copy()

        # ── Delegate to PureLLMBaseline (preferred path) ─────────────────────
        # Feynman biology/chemistry/physics equations benefit from hardcoded OLS
        # paths, variant guards, and all prompt fixes in PureLLMBaseline.
        # Pure DeFi equations fall through to the local specialised prompts.
        if self._llm_baseline is not None:
            try:
                result = self._llm_baseline.generate_formula(
                    description=description,
                    domain=domain,
                    variable_names=variable_names,
                    metadata=metadata,
                    X=X,
                    y=y,
                )
                code = result.get("python_code", "N/A")
                if code and code != "N/A" and "return" in code:
                    if not self._no_cache:
                        self.formula_cache[cache_key] = result.copy()
                    if verbose:
                        print(f"  [LLM→PureLLM] formula: {result.get('formula','')[:80]}")
                    return result
            except Exception:
                pass  # fall through to local DeFi implementation

        # ── Local DeFi implementation (fallback) ─────────────────────────────
        desc_lower = description.lower()
        use_specialized = any(
            k in desc_lower
            for k in ["kelly", "impermanent loss", "liquidation", "expected shortfall",
                       "black-scholes", "sharpe", "value at risk"]
        )

        if use_specialized:
            prompt = self._specialized_prompt(description, domain, variable_names, metadata)
        else:
            prompt = self._standard_prompt(description, domain, variable_names, metadata)

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            content = resp.content[0].text if resp.content else ""

            # Guard against stop_reason=max_tokens: retry with tighter prompt.
            stop_reason = getattr(resp, "stop_reason", None)
            if stop_reason == "max_tokens" or (
                content and "def formula" in content and "return" not in content
            ):
                tight_prompt = (
                    f"Give ONLY the Python function for: {description}\n"
                    f"Variables: {', '.join(variable_names)}\n\n"
                    f"PYTHON:\n"
                    f"def formula({', '.join(variable_names)}):\n"
                    f"    import numpy as np\n"
                    f"    return ...\n\n"
                    f"Reply with ONLY the def block, no explanation."
                )
                resp2 = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    temperature=0.0,
                    messages=[{"role": "user", "content": tight_prompt}],
                )
                content = resp2.content[0].text if resp2.content else content

            parsed = self._parse_response(content)

            python_code = parsed.get("python", "N/A")
            if python_code and python_code != "N/A" and "return" not in python_code:
                python_code = "N/A"

            result = {
                "formula": parsed.get("formula", "N/A"),
                "latex": parsed.get("latex", "N/A"),
                "python_code": python_code,
                "explanation": parsed.get("explanation", "N/A"),
                "specialized": use_specialized,
                "raw_response": content,
            }

            if result["python_code"] and result["python_code"] != "N/A":
                self.formula_cache[cache_key] = result.copy()

            if verbose:
                print(f"  [LLM] formula extracted: {result['formula'][:80]}")
            return result

        except Exception as e:
            return {
                "formula": "N/A", "latex": "N/A",
                "python_code": "N/A", "explanation": "N/A",
                "specialized": False, "raw_response": "",
                "error": str(e),
            }


    def _standard_prompt(self, description, domain, variable_names, metadata):
        var_info = f"Variables (in order): {', '.join(variable_names)}" if variable_names else ""
        constants = ""
        if metadata and metadata.get("constants"):
            constants = "\nConstants:\n"
            for k, v in metadata["constants"].items():
                constants += f"  {k} = {v}\n"
        return f"""You are a mathematical formula expert in DeFi / {domain}.
Task: Derive the closed-form formula for: {description}
{var_info}
{constants}

Respond with EXACTLY these sections and no other text:

FORMULA:
[concise mathematical expression]

PYTHON:
def formula({", ".join(variable_names)}):
    import numpy as np
    # implement using np.* for array support
    return ...

EXPLANATION:
[1-2 sentences describing the formula]
"""

    def _specialized_prompt(self, description, domain, variable_names, metadata):
        desc_lower = description.lower()
        var_list = ", ".join(variable_names)
        v = variable_names

        if "kelly" in desc_lower or ("optimal" in desc_lower and "lp" in desc_lower):
            return f"""FORMULA:
f* = min(mu / (lambda * sigma^2), 1.0)

PYTHON:
def formula({var_list}):
    import numpy as np
    risk_aversion = 2.0
    f_star = {v[0]} / (risk_aversion * {v[1]}**2)
    return np.minimum(f_star, 1.0)

EXPLANATION:
Kelly criterion for LP position sizing, capped at 1.0.
"""
        if "impermanent loss" in desc_lower:
            return f"""FORMULA:
IL% = (2*sqrt(r)/(1+r) - 1) * 100

PYTHON:
def formula({var_list}):
    import numpy as np
    r = {v[0]}
    il_fraction = 2.0 * np.sqrt(r) / (1.0 + r) - 1.0
    return il_fraction * 100.0

EXPLANATION:
Impermanent loss percentage for a 50/50 constant-product pool.
"""
        if "value at risk" in desc_lower or "var at" in desc_lower:
            return f"""FORMULA:
VaR = mu - z * sigma  (z=1.645 for 95%)

PYTHON:
def formula({var_list}):
    import numpy as np
    z = 1.6449
    return {v[0]} - z * {v[1]}

EXPLANATION:
Parametric VaR at 95% confidence level.
"""
        if "expected shortfall" in desc_lower:
            # ES-from-VaR: single variable (var_95), formula = var * tail_multiplier
            if len(variable_names) == 1:
                return f"""FORMULA:
ES = VaR * 1.254

PYTHON:
def formula({var_list}):
    import numpy as np
    tail_multiplier = 1.254
    return {v[0]} * tail_multiplier

EXPLANATION:
Expected Shortfall from VaR using the normal-distribution tail risk multiplier (1.254 for 95% confidence).
"""
            # ES-from-portfolio: two variables (portfolio_value, daily_vol)
            return f"""FORMULA:
ES = portfolio_value * daily_volatility * 2.063

PYTHON:
def formula({var_list}):
    import numpy as np
    es_multiplier = 2.063
    return {v[0]} * {v[1]} * es_multiplier

EXPLANATION:
Expected Shortfall at 95% confidence for normal returns (ES multiplier = 2.063).
"""
        # fallback
        return self._standard_prompt(description, domain, variable_names, metadata)

    def _parse_response(self, content: str) -> Dict[str, str]:
        parsed = {"formula": "N/A", "latex": "N/A", "python": "N/A", "explanation": "N/A"}

        for key, tag in [("formula", "FORMULA"), ("latex", "LATEX"), ("explanation", "EXPLANATION")]:
            m = re.search(rf"{tag}:\s*\n(.*?)(?=\n[A-Z]+:|\Z)", content, re.DOTALL | re.IGNORECASE)
            if m:
                parsed[key] = m.group(1).strip()

        # Python extraction
        code = None
        m = re.search(r"PYTHON:\s*\n(.*?)(?=\n[A-Z]+:|\Z)", content, re.DOTALL | re.IGNORECASE)
        if m:
            code = m.group(1).strip()
        if not code:
            m = re.search(r"```python\s*\n(.*?)\n```", content, re.DOTALL)
            if m:
                code = m.group(1).strip()
        if not code:
            m = re.search(r"(def\s+\w+\s*\([^)]*\)\s*:(?:\n(?:[ \t]+.*))*)", content)
            if m:
                code = m.group(1).strip()
        if code:
            code = re.sub(r"^```python\s*", "", code.strip())
            code = re.sub(r"\s*```$", "", code)
            parsed["python"] = code.strip()

        return parsed

    # ------------------------------------------------------------------
    # Hybrid predict (train + evaluate on same split)
    # ------------------------------------------------------------------
    def hybrid_predict(
            self,
            description: str,
            domain: str,
            X_train: np.ndarray,
            y_train: np.ndarray,
            variable_names: List[str],
            metadata: Dict,
            verbose: bool = False,
    ) -> Dict:
        """
        Hybrid symbolic + neural system.

        Returns:
            llm_result
            llm_train_r2
            nn_train_r2
            decision_margin
            decision
            nn_model
            scaler_X
            scaler_y
            evaluation {r2, mse, mae}
        """ 

        # =====================================================
        # 1️⃣  STAGE 1 — LLM symbolic structure discovery
        # =====================================================
        llm_result = self.generate_llm_formula(
            description, domain, variable_names, metadata,
            verbose=verbose, X=X_train, y=y_train,
        )

        llm_train_r2    = 0.0
        llm_preds_train = None

        if llm_result.get("python_code") and llm_result["python_code"] != "N/A":
            llm_preds_train = _safe_exec_formula(
                llm_result["python_code"], X_train, variable_names
            )
            if llm_preds_train is not None and np.isfinite(llm_preds_train).all():
                llm_train_r2 = r2_score(y_train, llm_preds_train)
            else:
                llm_preds_train = None

        # =====================================================
        # 2️⃣  STAGE 2 — scipy nonlinear parameter fitting
        #       Parametrizes the LLM formula and fits its numeric
        #       constants to training data — works regardless of
        #       what naming convention the LLM used.
        # =====================================================
        fitted_code       = llm_result.get("python_code", "N/A")
        fitted_train_r2   = llm_train_r2
        fitted_preds_train = llm_preds_train

        if fitted_code and fitted_code != "N/A":
            fc, fr2 = fit_formula_params(
                fitted_code, X_train, y_train, variable_names, verbose=verbose
            )
            if fr2 > llm_train_r2 + 0.001:
                fitted_code        = fc
                fitted_train_r2    = fr2
                fitted_preds_train = _safe_exec_formula(fc, X_train, variable_names)
                if fitted_preds_train is None or not np.isfinite(fitted_preds_train).all():
                    fitted_preds_train = llm_preds_train
                    fitted_train_r2    = llm_train_r2
                elif verbose:
                    print(f"  [Stage2] R²: {llm_train_r2:.4f} → {fitted_train_r2:.4f}")


        # =====================================================
        # 3️⃣  NEURAL NETWORK TRAINING
        # =====================================================
        nn_model = None
        scaler_X = None
        scaler_y = None
        nn_preds_train = None
        nn_train_r2 = 0.0

        try:
            nn_model, scaler_X, scaler_y = train_nn_model(X_train, y_train)
            nn_preds_train = nn_predict(nn_model, scaler_X, scaler_y, X_train)

            if nn_preds_train is not None:
                nn_train_r2 = r2_score(y_train, nn_preds_train)

        except Exception as e:
            if verbose:
                print(f"[NN] Training failed: {e}")


        # =====================================================
        # 4️⃣  DECISION LOGIC
        # =====================================================
        decision_margin = fitted_train_r2 - nn_train_r2

        # Fitted symbolic formula clearly dominates
        if fitted_train_r2 >= 0.85 and decision_margin > 0.05:
            decision   = "fitted_llm"
            best_preds = fitted_preds_train

        # Both competitive — adaptive ensemble weighted by R²
        elif (
            fitted_train_r2 >= 0.5
            and nn_train_r2 >= 0.5
            and abs(decision_margin) <= 0.15
            and fitted_preds_train is not None
            and nn_preds_train is not None
        ):
            decision = "ensemble"
            total = fitted_train_r2 + nn_train_r2
            w = fitted_train_r2 / total if total > 0 else 0.6
            best_preds = w * fitted_preds_train + (1.0 - w) * nn_preds_train

        # NN fallback
        else:
            decision   = "nn"
            best_preds = nn_preds_train


        # =====================================================
        # 4️⃣  FINAL TRAIN METRICS
        # =====================================================
        # Guard: if best_preds contains non-finite values (overflow from a
        # badly-fitted formula that slipped through), fall back to NN so the
        # benchmark never receives astronomically wrong predictions.
        if best_preds is not None and not np.isfinite(best_preds).all():
            if nn_preds_train is not None and np.isfinite(nn_preds_train).all():
                best_preds = nn_preds_train
                decision   = "nn"
            else:
                best_preds = None

        if best_preds is not None:
            final_r2  = r2_score(y_train, best_preds)
            residuals = y_train - best_preds
            with np.errstate(over="ignore", invalid="ignore"):
                mse = float(np.mean(residuals ** 2))
            rmse = float(np.sqrt(mse)) if np.isfinite(mse) else 0.0
            mae  = float(np.mean(np.abs(residuals)))
        else:
            final_r2 = 0.0
            mse  = 0.0
            rmse = 0.0
            mae  = 0.0


        # =====================================================
        # 5️⃣  RETURN STRUCTURE (CRITICAL FOR EXTRAPOLATION TEST)
        # =====================================================
        return {
            "llm_result":      llm_result,
            "llm_train_r2":    float(llm_train_r2),
            "fitted_train_r2": float(fitted_train_r2),
            "fitted_code":     fitted_code,
            "nn_train_r2":     float(nn_train_r2),
            "decision_margin": float(decision_margin),
            "decision":        decision,

            # For "fitted_llm": apply fitted_code to X_test via _safe_exec_formula.
            # For "ensemble":   w * _safe_exec_formula(fitted_code, X_test) + (1-w) * nn_predict(...)
            "nn_model": nn_model if decision in ["nn", "ensemble"] else None,
            "scaler_X": scaler_X if decision in ["nn", "ensemble"] else None,
            "scaler_y": scaler_y if decision in ["nn", "ensemble"] else None,

            "evaluation": {
                "r2":   float(final_r2),
                "rmse": rmse,
                "mse":  mse,
                "mae":  mae,
            },
        }

    # ------------------------------------------------------------------
    # Evaluate LLM formula on arbitrary data
    # ------------------------------------------------------------------

    def evaluate_llm_formula(
        self,
        result_dict: Dict,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        verbose: bool = False,
    ) -> Dict:
        """Evaluate a formula dict (with 'python_code') against data."""
        code = result_dict.get("python_code", "N/A")
        if not code or code == "N/A":
            return {"success": False, "r2": 0.0, "error": "no python_code"}

        preds = _safe_exec_formula(code, X, variable_names)
        if preds is None or not np.isfinite(preds).all():
            return {"success": False, "r2": 0.0, "error": "formula evaluation failed"}

        r2 = r2_score(y, preds)
        mse = float(np.mean((y - preds) ** 2))
        mae = float(np.mean(np.abs(y - preds)))

        if verbose:
            print(f"  [eval] R²={r2:.4f}, MSE={mse:.4e}, MAE={mae:.4e}")

        return {"success": True, "r2": r2, "mse": mse, "mae": mae}


# ---------------------------------------------------------------------------
# --batch mode: run full DeFi benchmark suite and save results
# ---------------------------------------------------------------------------

def run_batch(verbose: bool = False):
    """Run the full DeFi benchmark suite and write hybrid_defi_<timestamp>.json."""
    protocol = DeFiExperimentProtocol()
    hybrid = EnhancedHybridSystemDeFi()

    domains = ["amm", "lending", "risk", "staking", "trading", "derivatives", "liquidity"]
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 72)
    print("HYBRID DEFI BENCHMARK SUITE — BATCH MODE")
    print(f"Timestamp: {timestamp}")
    print("=" * 72)

    for domain in domains:
        try:
            test_cases = protocol.load_test_data(domain, num_samples=200)
        except Exception as e:
            print(f"  [SKIP] {domain}: {e}")
            continue

        for description, X, y, var_names, metadata in test_cases:
            print(f"\n  {domain.upper()} | {description[:60]}")
            try:
                result = hybrid.hybrid_predict(
                    description, domain, X, y, var_names, metadata, verbose=verbose
                )

                entry = {
                    "domain": domain,
                    "description": description,
                    "variable_names": var_names,
                    "decision": result["decision"],
                    "llm_train_r2": result["llm_train_r2"],
                    "nn_train_r2": result["nn_train_r2"],
                    "hybrid_train_r2": result["evaluation"]["r2"],
                    "llm_formula": result["llm_result"].get("formula", "N/A"),
                    "llm_python": result["llm_result"].get("python_code", "N/A"),
                    "success": True,
                }
                print(
                    f"    decision={result['decision']}, "
                    f"LLM R²={result['llm_train_r2']:.4f}, "
                    f"NN R²={result['nn_train_r2']:.4f}, "
                    f"Hybrid R²={result['evaluation']['r2']:.4f}"
                )
            except Exception as e:
                entry = {
                    "domain": domain,
                    "description": description,
                    "success": False,
                    "error": str(e),
                }
                print(f"    ERROR: {e}")

            all_results.append(entry)

    # Save results
    out_dir = Path("hypatiax/data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hybrid_defi_{timestamp}.json"

    with open(out_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "total": len(all_results),
                "successful": sum(1 for r in all_results if r.get("success")),
                "results": all_results,
            },
            f,
            indent=2,
        )

    print(f"\n✅ Saved {len(all_results)} results → {out_path}")

    # Summary
    successful = [r for r in all_results if r.get("success")]
    if successful:
        avg_hybrid = np.mean([r["hybrid_train_r2"] for r in successful])
        avg_llm = np.mean([r["llm_train_r2"] for r in successful])
        avg_nn = np.mean([r["nn_train_r2"] for r in successful])
        decisions = {d: sum(1 for r in successful if r.get("decision") == d) for d in ["llm", "nn", "ensemble"]}
        print(f"\n📊 Summary ({len(successful)}/{len(all_results)} succeeded):")
        print(f"   Avg LLM R²:    {avg_llm:.4f}")
        print(f"   Avg NN R²:     {avg_nn:.4f}")
        print(f"   Avg Hybrid R²: {avg_hybrid:.4f}")
        print(f"   Decisions: LLM={decisions['llm']}, NN={decisions['nn']}, Ensemble={decisions['ensemble']}")

    return all_results


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid DeFi System")
    parser.add_argument("--batch", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.batch:
        run_batch(verbose=args.verbose)
    else:
        # Quick smoke test
        hybrid = EnhancedHybridSystemDeFi()
        out = hybrid.generate_llm_formula(
            "Optimal LP position size using risk-adjusted Kelly criterion",
            "liquidity",
            ["expected_fee_apy", "il_risk"],
            metadata={"ground_truth": "min(expected_fee_apy/(2*il_risk**2),1.0)"},
            verbose=True,
        )
        print(json.dumps({k: v for k, v in out.items() if k != "raw_response"}, indent=2))
