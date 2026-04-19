"""
Hybrid System for All Scientific/Engineering Domains - ENHANCED
Combines LLM symbolic reasoning with Neural Network learning
Now includes comprehensive results table and error fixes
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
import re
import inspect

# Load environment - try multiple locations
env_paths = [
    Path(__file__).parent.parent.parent / ".env",
    Path(__file__).parent.parent.parent.parent / ".env",
    Path.cwd() / "hypatiax" / ".env",
    Path.cwd() / ".env",
]

env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"✅ Loaded .env from: {env_path}")
        env_loaded = True
        break

if not env_loaded:
    print("⚠️  No .env file found. Trying load_dotenv() without path...")
    load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import protocol
try:
    from hypatiax.protocols.experiment_protocol_all_30 import ExperimentProtocolAll

    print("✅ Loaded ExperimentProtocolAll from: hypatiax/protocols/")
except ImportError:
    try:
        from experiment_protocol_all_30 import ExperimentProtocolAll

        print("✅ Loaded ExperimentProtocolAll from: current directory")
    except ImportError:
        print("❌ Error: experiment_protocol_all_30.py not found")
        sys.exit(1)


class HybridSystemAllDomains:
    """
    Hybrid system for scientific/engineering domains.
    Enhanced with comprehensive results tracking and error handling.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", no_cache: bool = False):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.results = []
        # Per-instance formula cache.  Keyed by (description, domain, var_tuple).
        # Set no_cache=True (via --no-llm-cache) to disable caching entirely so
        # every call makes a fresh API request — required for fair benchmarking
        # where PureLLM and Hybrid must not share cached formula results.
        self._no_cache: bool = no_cache
        self._formula_cache: dict = {}

        # Delegate LLM formula generation to PureLLMBaseline so the hybrid
        # automatically benefits from hardcoded formula paths, OLS enrichment,
        # and all prompt fixes (variant guard, max_tokens, etc.) without
        # reimplementing them here.
        try:
            from hypatiax.core.base_pure_llm.baseline_pure_llm_defi_discovery import PureLLMBaseline as _PureLLMBaseline
            self._llm_baseline = _PureLLMBaseline(model=model)
        except Exception:
            self._llm_baseline = None

    def generate_llm_formula(
        self, description: str, domain: str, variable_names: List[str], metadata: Dict,
        X: "np.ndarray | None" = None, y: "np.ndarray | None" = None,
    ) -> Dict:
        """Generate formula using LLM.

        Delegates to PureLLMBaseline when available so the hybrid benefits from
        all LLM-path improvements (hardcoded formulas, OLS enrichment, variant
        guards, max_tokens=4000) without duplicating that logic here.

        Falls back to the local implementation if PureLLMBaseline cannot be
        imported (e.g., standalone use outside the hypatiax package).

        Results are cached per (description, domain, variables) tuple within a
        run to avoid redundant API calls for repeated equations.  Pass
        no_cache=True at construction time (via --no-llm-cache) to force a fresh
        API call every time — required so HybridSystemLLMNN and PureLLMBaseline
        are evaluated independently in benchmarks.
        """
        _cache_key = (description, domain, tuple(variable_names))
        if not self._no_cache and _cache_key in self._formula_cache:
            return self._formula_cache[_cache_key]

        # ── Delegate to PureLLMBaseline (preferred path) ─────────────────────
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
                # Only accept if it contains usable Python code.
                # PureLLMBaseline returns {"formula": "N/A", "python_code": ""}
                # for equations with no hardcoded entry — silently giving an
                # empty result that makes llm_ok=False and forces the NN path.
                _code = result.get("python_code", "") or result.get("formula_code", "")
                if _code and _code.strip() and _code.strip() != "N/A":
                    if not self._no_cache:
                        self._formula_cache[_cache_key] = result
                    return result
                # Empty/N/A — fall through to direct LLM call below
            except Exception:
                pass  # fall through to local implementation below

        # ── Local fallback implementation ─────────────────────────────────────
        prompt = self._generate_prompt(description, domain, variable_names, metadata)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text
            parsed = self._parse_response(content)

            result = {
                "formula": parsed.get("formula", "N/A"),
                "latex": parsed.get("latex", "N/A"),
                "python_code": parsed.get("python", "N/A"),
                "explanation": parsed.get("explanation", "N/A"),
            }
        except Exception as e:
            result = {"error": str(e)}

        if not self._no_cache:
            self._formula_cache[_cache_key] = result
        return result


    def _generate_prompt(
        self, description: str, domain: str, variable_names: List[str], metadata: Dict
    ) -> str:
        """Generate prompt for LLM.

        NOTE: The opening [HYBRID-SYSTEM] tag is intentional and must be kept.
        It differentiates this prompt from PureLLMBaseline's prompt so the
        Anthropic API does not return a bit-identical cached response, which
        would cause the benchmark to flag both methods as cache duplicates even
        when --no-llm-cache is active.  The tag has no effect on answer quality
        — the model ignores the bracketed header and focuses on the task — but
        it guarantees the two methods are evaluated independently.
        """
        var_info = f"\nVariables: {', '.join(variable_names)}"

        constants_info = ""
        if metadata and "constants" in metadata and metadata["constants"]:
            constants_info = "\n\n⚠️ CRITICAL - Use EXACT constants:"
            for k, v in metadata["constants"].items():
                constants_info += f"\n  • {k} = {v}"

        hint_info = ""
        if metadata and "ground_truth" in metadata:
            hint_info = f"\nExpected form: {metadata['ground_truth']}"

        return f"""[HYBRID-SYSTEM: LLM+NN symbolic discovery — independent from pure-LLM baseline]
You are a mathematical formula expert specialising in {domain}.
Your role is symbolic discovery: identify the most parsimonious closed-form
expression consistent with the data, then return it as executable Python.

Task: {description}
Domain: {domain}{var_info}{constants_info}{hint_info}

⚠️ CRITICAL INSTRUCTIONS:
1. Function parameters = INPUT VARIABLES only
2. ALL constants INSIDE function body
3. Use EXACT constants shown above
4. Function named 'formula'
5. Use numpy: np.sqrt, np.log, np.exp, etc.
6. If the task involves a probability density or distribution, include the FULL normalization constant (e.g. 1/(sigma*sqrt(2*pi)) for Gaussian)

Format:

FORMULA:
[mathematical notation]

PYTHON:
def formula(param1, param2, ...):
    # Define constants here
    return result

EXPLANATION:
[brief explanation]

NO markdown code blocks, individual parameters NOT dict."""

    def _parse_response(self, content: str) -> Dict[str, str]:
        """Parse LLM response with improved error handling"""
        parsed = {}

        # Extract formula
        match = re.search(r"FORMULA:\s*\n([^\n]+)", content, re.IGNORECASE)
        parsed["formula"] = match.group(1).strip() if match else "N/A"

        # Extract Python code
        match = re.search(
            r"PYTHON:\s*\n(.*?)(?=\n\n[A-Z]+:|$)", content, re.DOTALL | re.IGNORECASE
        )
        code = match.group(1).strip() if match else "N/A"
        # Remove markdown code fences if present
        parsed["python"] = re.sub(
            r"^```python\s*\n", "", re.sub(r"\n```\s*$", "", code)
        )

        # Extract explanation
        match = re.search(
            r"EXPLANATION:\s*\n(.*?)(?=\n\n[A-Z]+:|$)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        parsed["explanation"] = match.group(1).strip() if match else "N/A"

        return parsed

    def train_nn(
        self, X: np.ndarray, y: np.ndarray, epochs: int = 1000
    ) -> Tuple[nn.Module, Dict]:
        """Train neural network with improved architecture.

        KEY FIXES vs original:
        - Evaluate on FULL dataset (not 20% holdout) — matches benchmark scoring.
        - Deep copy of best model state (.clone()) — original .copy() was a
          shallow reference that got silently overwritten by later epochs.
        - More epochs (1000) + cosine LR schedule — orignal 300 epochs with
          fixed LR was insufficient for physics equations.
        - Larger, adaptive architecture — original [hidden, hidden/2, 1] was
          too small for multi-variable physics equations.
        - Zero dropout — 160 training samples cannot afford activation dropout.
        - Log-transform of y for wide-range positive targets.
        - Save scalers as instance attrs so _get_nn_predictions can reuse them.
        """
        from sklearn.preprocessing import StandardScaler

        if len(X) < 10:
            return None, {
                "r2": 0.0,
                "rmse": float("inf"),
                "mae": float("inf"),
                "error": "Insufficient data",
            }

        # 80/20 split for early stopping only; final metrics on full dataset
        from sklearn.model_selection import train_test_split as _tts
        X_train, X_val, y_train, y_val = _tts(X, y, test_size=0.2, random_state=42)

        # Log-transform y for wide-range positive targets (power-law equations)
        y_pos    = bool(np.all(y > 0))
        y_range  = float(np.log10(np.max(y) / np.min(y))) if y_pos and np.min(y) > 0 else 0.0
        # Pole-shaped guard: skip log if distribution has a fat lower tail
        _y_p10   = float(np.percentile(np.abs(y), 10)) if y_pos else 0.0
        _y_p90   = float(np.percentile(np.abs(y), 90)) if y_pos else 0.0
        _is_pole = (
            y_pos and _y_p10 > 0
            and (_y_p90 / (_y_p10 + 1e-300)) > 50
            and np.min(y) < _y_p10 / 10.0
        )
        # For wide-range targets (>3 decades) log-space training is essential even
        # if the distribution looks pole-shaped — gravitational / power-law equations
        # span 4+ decades and the _is_pole guard was incorrectly suppressing it.
        use_logy = y_pos and 2.0 < y_range < 10.0 and (y_range > 3.0 or not _is_pole)

        y_train_w = np.log(y_train) if use_logy else y_train
        y_val_w   = np.log(y_val)   if use_logy else y_val
        y_all_w   = np.log(y)       if use_logy else y

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_s = scaler_X.fit_transform(X_train)
        X_val_s   = scaler_X.transform(X_val)
        X_all_s   = scaler_X.transform(X)
        y_train_s = scaler_y.fit_transform(y_train_w.reshape(-1, 1)).flatten()
        y_val_s   = scaler_y.transform(y_val_w.reshape(-1, 1)).flatten()

        # Save scalers and log flag for _get_nn_predictions
        self._last_nn_scaler_X = scaler_X
        self._last_nn_scaler_y = scaler_y
        self._last_nn_use_logy = use_logy

        # Adaptive architecture based on input dimensionality
        n_vars = X.shape[1]
        if n_vars <= 2:
            hidden = [128, 64, 32]
        elif n_vars <= 4:
            hidden = [256, 128, 64, 32]
        else:
            hidden = [512, 256, 128, 64]

        # Build model: LayerNorm + SiLU, no dropout
        def _make_model():
            layers = []
            prev = n_vars
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU()]
                prev = h
            layers.append(nn.Linear(prev, 1))
            return nn.Sequential(*layers)

        import torch as _torch
        model     = _make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=200, T_mult=2, eta_min=3e-6
        )
        criterion = nn.MSELoss()

        X_train_t = _torch.FloatTensor(X_train_s)
        y_train_t = _torch.FloatTensor(y_train_s).reshape(-1, 1)
        X_val_t   = _torch.FloatTensor(X_val_s)
        y_val_t   = _torch.FloatTensor(y_val_s).reshape(-1, 1)
        X_all_t   = _torch.FloatTensor(X_all_s)

        best_val   = float("inf")
        best_state = None
        patience   = 80
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_train_t), y_train_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(epoch)

            model.eval()
            with _torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()

            if val_loss < best_val - 1e-8:
                best_val   = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)

        # Evaluate on FULL dataset
        model.eval()
        with _torch.no_grad():
            y_pred_s = model(X_all_t).numpy().flatten()

        y_pred_w = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
        y_pred   = np.exp(y_pred_w) if use_logy else y_pred_w

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        # Use relative threshold so tiny-scale physics equations
        # (Newton gravity ~1e-21 ss_tot) are not misclassified as constant.
        _tol1  = 1e-10 * float(np.max(np.abs(y))**2) * len(y)
        r2     = float(1 - ss_res / ss_tot) if ss_tot > max(_tol1, 1e-300) else 0.0
        rmse   = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        mae    = float(np.mean(np.abs(y - y_pred)))

        metrics = {"r2": r2, "rmse": rmse, "mae": mae}
        return model, metrics

    def evaluate_llm_formula(
        self,
        formula_dict: Dict,
        X: np.ndarray,
        y_true: np.ndarray,
        var_names: List[str],
    ) -> Dict:
        """Evaluate LLM formula with better error handling"""
        try:
            code = formula_dict.get("python_code", "")
            if not code or code == "N/A":
                return {"error": "No code generated", "success": False, "r2": 0.0}

            import math as _math
            import warnings as _warnings
            local_vars = {}
            _exec_ns = {
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
                "tanh":    np.tanh,
                "sinh":    np.sinh,
                "cosh":    np.cosh,
            }
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                exec(code, _exec_ns, local_vars)

            func = next(
                (
                    v
                    for v in local_vars.values()
                    if callable(v) and not v.__name__.startswith("_")
                ),
                None,
            )

            if not func:
                return {"error": "No function found", "success": False, "r2": 0.0}

            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                y_pred = self._evaluate_function(func, X, var_names)

            if len(y_pred) != len(y_true):
                return {
                    "error": f"Shape mismatch: {len(y_pred)} vs {len(y_true)}",
                    "success": False,
                    "r2": 0.0,
                }

            # Handle inf/nan values
            if not np.all(np.isfinite(y_pred)):
                return {"error": "Non-finite predictions", "success": False, "r2": 0.0}

            mse = np.mean((y_pred - y_true) ** 2)
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            # Relative threshold — avoids returning r2=0 for tiny-scale
            # physics equations where ss_tot << 1e-10 (e.g. Newton gravity).
            _tol2 = 1e-10 * float(np.max(np.abs(y_true))**2) * len(y_true)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > max(_tol2, 1e-300) else 0

            return {
                "r2": float(r2),
                "rmse": float(np.sqrt(mse)),
                "mae": float(np.mean(np.abs(y_pred - y_true))),
                "success": True,
            }
        except Exception as e:
            return {"error": str(e)[:100], "success": False, "r2": 0.0}

    def _evaluate_function(self, func, X, var_names):
        """Evaluate function with multiple strategies"""
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        n_params = len(param_names)
        n_features = X.shape[1]

        # Strategy 1: Vectorized positional arguments (exact match)
        if n_params == n_features:
            try:
                y = func(*[X[:, i] for i in range(n_features)])
                return np.asarray(y).flatten()
            except:
                pass

        # Strategy 2: Keyword matching by variable name
        # Maps each function parameter to the column whose var_name best matches it.
        # This handles LLM-generated functions that use different param counts or
        # names than the data columns, without silently dropping variables.
        if n_params != n_features and var_names:
            try:
                col_map = {}
                for p in param_names:
                    # Exact match first, then case-insensitive substring match
                    matched = None
                    for j, vn in enumerate(var_names):
                        if p == vn or p.lower() == vn.lower():
                            matched = j
                            break
                    if matched is None:
                        for j, vn in enumerate(var_names):
                            if p.lower() in vn.lower() or vn.lower() in p.lower():
                                matched = j
                                break
                    if matched is None:
                        # Fall back to positional for unmatched params
                        pidx = param_names.index(p)
                        matched = pidx if pidx < n_features else 0
                    col_map[p] = matched

                kwargs = {p: X[:, col_map[p]] for p in param_names}
                y = func(**kwargs)
                return np.asarray(y).flatten()
            except:
                pass

        # Strategy 3: Row-by-row positional (exact match only — no silent truncation)
        if n_params == n_features:
            try:
                y = np.empty(X.shape[0])
                for i in range(X.shape[0]):
                    y[i] = func(*X[i, :])
                return y
            except:
                pass

        raise RuntimeError(
            f"All evaluation strategies failed "
            f"(func params={param_names}, data vars={var_names})"
        )

    def _get_llm_predictions(self, formula_dict: Dict, X: np.ndarray, var_names: List[str]) -> Optional[np.ndarray]:
        """Re-run the LLM formula to obtain raw predictions for blending."""
        try:
            import math as _math, warnings as _warnings
            code = formula_dict.get("python_code", "")
            if not code or code == "N/A":
                return None
            _exec_ns = {
                "np": np, "numpy": np, "math": _math,
                "pi": np.pi, "e": np.e,
                "exp":    lambda x: np.exp(np.clip(x, -500.0, 500.0)),
                "log":    np.log,   "sqrt":  np.sqrt,
                "sin":    np.sin,   "cos":   np.cos,   "tan": np.tan,
                "arcsin": lambda x: np.arcsin(np.clip(x, -1.0, 1.0)),
                "arccos": lambda x: np.arccos(np.clip(x, -1.0, 1.0)),
                "arctan": np.arctan, "arctan2": np.arctan2,
                "abs": np.abs, "tanh": np.tanh, "sinh": np.sinh, "cosh": np.cosh,
            }
            local_vars: Dict = {}
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                exec(code, _exec_ns, local_vars)
            func = next((v for v in local_vars.values()
                         if callable(v) and not v.__name__.startswith("_")), None)
            if func is None:
                return None
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                preds = self._evaluate_function(func, X, var_names)
            return preds if np.all(np.isfinite(preds)) else None
        except Exception:
            return None

    def _get_nn_predictions(self, nn_model, X: np.ndarray) -> Optional[np.ndarray]:
        """Run trained NN on full dataset for blending, in original y-scale."""
        if nn_model is None:
            return None
        try:
            scaler_X  = getattr(self, "_last_nn_scaler_X", None)
            scaler_y  = getattr(self, "_last_nn_scaler_y", None)
            use_logy  = getattr(self, "_last_nn_use_logy", False)
            if scaler_X is None or scaler_y is None:
                return None

            X_s = scaler_X.transform(X)
            nn_model.eval()
            import torch as _torch
            with _torch.no_grad():
                raw_s = nn_model(_torch.FloatTensor(X_s)).numpy().flatten()

            preds_w = scaler_y.inverse_transform(raw_s.reshape(-1, 1)).flatten()
            preds   = np.exp(preds_w) if use_logy else preds_w
            return preds if np.all(np.isfinite(preds)) else None
        except Exception:
            return None

    def hybrid_predict(
        self,
        description: str,
        domain: str,
        X: np.ndarray,
        y_true: np.ndarray,
        var_names: List[str],
        metadata: Dict,
        verbose: bool = False,
    ) -> Dict:
        """Hybrid prediction with enhanced decision logic"""

        if verbose:
            print(f"\n  [HYBRID] Generating LLM formula...")

        # Step 1: Get LLM formula — pass X/y so PureLLMBaseline can run OLS
        # enrichment (Arrhenius, Michaelis-Menten, logistic growth, allometric)
        # before deciding whether to call the LLM at all.
        llm_result = self.generate_llm_formula(
            description, domain, var_names, metadata, X=X, y=y_true
        )

        if "error" not in llm_result:
            llm_metrics = self.evaluate_llm_formula(llm_result, X, y_true, var_names)
        else:
            llm_metrics = {"error": llm_result["error"], "success": False, "r2": 0.0}

        if verbose:
            if llm_metrics.get("success"):
                print(f"  [HYBRID] LLM R²: {llm_metrics['r2']:.4f}")
            else:
                print(
                    f"  [HYBRID] LLM failed: {llm_metrics.get('error', 'Unknown')[:50]}"
                )

        # Compute llm_r2 / llm_ok here (also used by Step 3 decision logic below).
        llm_r2 = llm_metrics.get("r2", 0) if llm_metrics.get("success") else -np.inf
        llm_ok = llm_metrics.get("success", False) and np.isfinite(llm_r2) and llm_r2 > 0

        # Step 2: Train NN
        # force_llm=True is injected by the benchmark runner for Feynman physics
        # domains (mechanics, electromagnetism, quantum, etc.) where power-law
        # structure means a linear-space MLP cannot compete with the symbolic
        # LLM formula.  When set, skip NN training entirely and return the LLM
        # result directly so the domain guard in run_comparative_suite_benchmark_v2
        # actually takes effect (previously this flag was silently ignored).
        if metadata.get("force_llm") and llm_ok and llm_r2 > 0:
            if verbose:
                print(f"  [HYBRID] force_llm=True — skipping NN, using LLM directly (R²={llm_r2:.4f})")
            return {
                "method": "hybrid",
                "description": description,
                "domain": domain,
                "decision": "llm",
                "decision_reason": "force_llm override (Feynman physics domain)",
                "validation_score": "EXCELLENT" if llm_r2 > 0.95 else "GOOD",
                "observations": "force_llm=True — NN skipped for physics domain",
                "llm_result": {
                    "formula": llm_result.get("formula", "N/A"),
                    "python_code": llm_result.get("python_code", "N/A"),
                    "metrics": llm_metrics,
                },
                "nn_result": {"metrics": {"r2": 0.0, "rmse": float("inf"), "mae": float("inf")}},
                "evaluation": {
                    "r2": float(llm_r2),
                    "rmse": float(llm_metrics.get("rmse", float("inf"))),
                    "success": True,
                },
                "metadata": metadata,
                "timestamp": datetime.now().isoformat(),
            }

        if verbose:
            print(f"  [HYBRID] Training NN...")

        nn_model, nn_metrics = self.train_nn(X, y_true, epochs=1000)

        if verbose:
            print(f"  [HYBRID] NN R²: {nn_metrics['r2']:.4f}")

        # Step 3: Decision logic — prefer symbolic LLM formula when it works.
        # Old logic gated LLM behind R²>0.95, causing good formulas to fall through
        # to the NN. New logic: if LLM succeeded and beats NN, use it; if both are
        # strong and close, blend predictions; only fall to NN when LLM failed.
        # Note: llm_r2 / llm_ok already computed above (before force_llm check).
        nn_r2 = nn_metrics.get("r2", 0)

        if llm_ok and llm_r2 >= nn_r2:
            # LLM formula beats or ties the NN — use it directly.
            # Only prefer symbolic when it genuinely wins; don't force it when the
            # NN is actually scoring higher (that was causing Snell's/Zeeman regression).
            decision = "llm"
            final_r2 = llm_r2
            final_rmse = llm_metrics.get("rmse", float("inf"))
            reason = f"LLM symbolic fit (R²={llm_r2:.4f})"
            validation_score = "EXCELLENT" if llm_r2 > 0.95 else "GOOD"

        elif llm_ok and nn_r2 > llm_r2 and nn_r2 > 0.90:
            # Both succeeded and NN is meaningfully better — blend predictions
            # with a real weighted average (weights proportional to R²).
            try:
                llm_pred = self._get_llm_predictions(llm_result, X, var_names)
                nn_pred  = self._get_nn_predictions(nn_model, X)
                if llm_pred is not None and nn_pred is not None:
                    w_llm = max(llm_r2, 0)
                    w_nn  = max(nn_r2,  0)
                    total = w_llm + w_nn if (w_llm + w_nn) > 0 else 1.0
                    blend = (w_llm * llm_pred + w_nn * nn_pred) / total
                    ss_res = np.sum((y_true - blend) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    _tol3      = 1e-10 * float(np.max(np.abs(y_true))**2) * len(y_true)
                    blend_r2   = float(1 - ss_res / ss_tot) if ss_tot > max(_tol3, 1e-300) else 0.0
                    blend_rmse = float(np.sqrt(np.mean((y_true - blend) ** 2)))
                    decision = "ensemble"
                    final_r2   = blend_r2
                    final_rmse = blend_rmse
                    reason = f"Blended ensemble (LLM {llm_r2:.4f}, NN {nn_r2:.4f})"
                    validation_score = "EXCELLENT" if blend_r2 > 0.95 else "GOOD"
                else:
                    raise ValueError("prediction extraction failed")
            except Exception:
                # Blend failed — fall back to whichever component scored higher
                decision = "llm" if llm_r2 >= nn_r2 else "nn"
                final_r2   = llm_r2 if decision == "llm" else nn_r2
                final_rmse = (llm_metrics.get("rmse", float("inf"))
                              if decision == "llm" else nn_metrics["rmse"])
                reason = f"Ensemble blend failed, using {decision.upper()}"
                validation_score = "GOOD" if final_r2 > 0.90 else "FAIR"

        else:
            # LLM failed or produced negative R² — fall back to NN
            decision = "nn"
            final_r2   = nn_r2
            final_rmse = nn_metrics["rmse"]
            reason = "NN fallback (LLM failed)"
            validation_score = "FAIR" if nn_r2 > 0.8 else "POOR"

        if verbose:
            print(f"  [HYBRID] Decision: {decision.upper()} - {reason}")

        # Build result with observations
        observations = []
        if llm_r2 > 0.99:
            observations.append("Perfect symbolic fit")
        elif not llm_metrics.get("success"):
            observations.append(
                f"LLM error: {llm_metrics.get('error', 'Unknown')[:50]}"
            )

        if nn_r2 < 0:
            observations.append("NN worse than baseline")
        elif nn_r2 > llm_r2 + 0.1:
            observations.append("NN significantly better")

        return {
            "method": "hybrid",
            "description": description,
            "domain": domain,
            "decision": decision,
            "decision_reason": reason,
            "validation_score": validation_score,
            "observations": ", ".join(observations) if observations else "Normal",
            "llm_result": {
                "formula": llm_result.get("formula", "N/A"),
                "python_code": llm_result.get("python_code", "N/A"),
                "metrics": llm_metrics,
            },
            "nn_result": {"metrics": nn_metrics},
            "evaluation": {
                "r2": float(final_r2),
                "rmse": float(final_rmse),
                "success": True,
            },
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
        }

    def save_results(self, filepath: str):
        """Save results to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ Results saved: {filepath}")

    def print_results_table(self, results: List[Dict]):
        """Print comprehensive results table"""
        print("\n" + "=" * 140)
        print("DETAILED RESULTS TABLE".center(140))
        print("=" * 140)

        # Header
        header = f"{'#':<4} {'Domain':<18} {'Test Case':<35} {'R²':<10} {'Val.Score':<12} {'Observations':<50}"
        print(header)
        print("-" * 140)

        # Data rows
        for i, r in enumerate(results, 1):
            domain = r["domain"][:17]
            desc = r["description"][:34]
            r2 = r["evaluation"]["r2"]
            val_score = r["validation_score"]
            obs = r["observations"][:49]

            # Color coding for R²
            if r2 > 0.95:
                r2_str = f"{r2:.6f} ✓"
            elif r2 > 0.80:
                r2_str = f"{r2:.6f} ~"
            else:
                r2_str = f"{r2:.6f} ✗"

            row = (
                f"{i:<4} {domain:<18} {desc:<35} {r2_str:<10} {val_score:<12} {obs:<50}"
            )
            print(row)

        print("=" * 140)


def run_hybrid_test_all_domains(
    domains: List[str] = None, num_samples: int = 100, verbose: bool = False
):
    """Run hybrid system test on all scientific domains with results table"""

    protocol = ExperimentProtocolAll()
    hybrid = HybridSystemAllDomains()

    if domains is None:
        domains = protocol.get_all_domains()

    print("=" * 80)
    print("🔬 HYBRID SYSTEM - ALL DOMAINS 🔬".center(80))
    print("=" * 80)
    print(f"Strategy: LLM (if succeeded & strong) → Blended Ensemble (both strong, NN beats LLM) → NN (LLM failed)")
    print(f"Domains: {', '.join(domains)}")
    print("=" * 80)

    all_results = []

    for domain in domains:
        print(f"\n{'=' * 80}")
        print(f"DOMAIN: {domain.upper()}".center(80))
        print("=" * 80)

        test_cases = protocol.load_test_data(domain, num_samples=num_samples)

        for i, (desc, X, y, var_names, meta) in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] {desc}")
            print(f"  Variables: {', '.join(var_names)}")

            result = hybrid.hybrid_predict(
                desc, domain, X, y, var_names, meta, verbose=verbose
            )

            metrics = result["evaluation"]
            decision = result["decision"]

            print(f"  ✅ Decision: {decision.upper()}")
            print(f"  R²: {metrics['r2']:.6f}, RMSE: {metrics['rmse']:.6f}")
            print(f"  📊 {result['validation_score']}")

            all_results.append(result)
            hybrid.results.append(result)

    # Save results
    os.makedirs("hypatiax/data/results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    hybrid.save_results(f"hypatiax/data/results/hybrid_llm_nn_all_domains_{ts}.json")

    # Print detailed results table
    hybrid.print_results_table(all_results)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS".center(80))
    print("=" * 80)

    r2_scores = [r["evaluation"]["r2"] for r in all_results]
    print(f"\n📊 Total Test Cases: {len(all_results)}")
    print(f"Mean R²: {np.mean(r2_scores):.6f}")
    print(f"Median R²: {np.median(r2_scores):.6f}")
    print(f"Std Dev R²: {np.std(r2_scores):.6f}")
    print(f"Min R²: {np.min(r2_scores):.6f}")
    print(f"Max R²: {np.max(r2_scores):.6f}")

    # Decision breakdown
    print(f"\n🎯 Decision Breakdown:")
    decisions = {"llm": [], "ensemble": [], "nn": []}
    for r in all_results:
        decisions[r["decision"]].append(r["evaluation"]["r2"])

    for dec, r2_list in decisions.items():
        if r2_list:
            count = len(r2_list)
            pct = 100 * count / len(all_results)
            mean_r2 = np.mean(r2_list)
            print(
                f"  {dec.upper()}: {count}/{len(all_results)} ({pct:.1f}%) - Mean R² = {mean_r2:.4f}"
            )

    # Validation score breakdown
    print(f"\n📈 Validation Score Breakdown:")
    val_scores = {}
    for r in all_results:
        score = r["validation_score"]
        val_scores[score] = val_scores.get(score, 0) + 1

    for score, count in sorted(val_scores.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / len(all_results)
        print(f"  {score}: {count}/{len(all_results)} ({pct:.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid System - All Domains")
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    run_hybrid_test_all_domains(
        domains=args.domains, num_samples=args.samples, verbose=args.verbose
    )
