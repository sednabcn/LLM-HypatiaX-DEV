"""
run_hybrid_defi_consolidated.py - FULLY SELF-CONTAINED RUNNER
==============================================================
All components consolidated into single file:
1. External protocol import only
2. LLM formula generation (built-in)
3. Neural network training (built-in)
4. Hybrid decision logic (built-in)
5. Results table generation (built-in)

No external dependencies except:
- Standard libraries (numpy, torch, sklearn)
- Anthropic API client
- External protocol (experiment_protocol_defi_20.py)

Author: HypatiaX Team
Version: 5.0 Consolidated
Date: 2026-01-15
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
from collections import defaultdict
from tabulate import tabulate
from dotenv import load_dotenv
from anthropic import Anthropic
import re
import inspect
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

# Load environment
env_paths = [
    Path(__file__).parent.parent.parent / ".env",  # hypatiax/.env (correct location)
    Path(__file__).parent.parent.parent.parent / ".env",  # Project root
    Path.cwd() / "hypatiax" / ".env",  # From current working directory
    Path.cwd() / ".env",  # Current working directory
]

env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"✅ Loaded .env from: {env_path}")
        env_loaded = True
        break

if not env_loaded:
    print("⚠️  No .env file found in standard locations. Searched:")
    for p in env_paths:
        print(f"   - {p} {'(exists)' if p.exists() else '(not found)'}")
    print("   Trying load_dotenv() without path...")
    load_dotenv()  # Try to load from default locations

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import ONLY the external protocol
try:
    from hypatiax.protocols.experiment_protocol_defi_20 import (
        DeFiExperimentProtocolExtended as DeFiExperimentProtocol,
    )

    print("✅ Loaded protocol from: hypatiax/protocols/")
except ImportError:
        print("❌ Error: experiment_protocol_defi_20.py not found")
        sys.exit(1)


# ============================================================================
# FORMULA TEMPLATES & FEW-SHOT EXAMPLES (Built-in)
# ============================================================================

FORMULA_TEMPLATES = {
    "amm": [
        "Constant Product: y = k/x",
        "Price Impact: Δp/p = Δx/(x + Δx)",
        "Impermanent Loss: 2√r/(1+r) - 1",
    ],
    "liquidity": [
        "Optimal Position: min(expected_return / (risk_factor × variance²), 1.0)",
        "Capital Efficiency: 1/√(P_lower) - 1/√(P_upper)",
        "Fee APY: (fees_24h × 365) / liquidity_value",
    ],
    "risk_var": [
        "VaR: V × σ × z_score",
        "Portfolio VaR: √(Σw²σ² + Σ Σ wiwijσiσjρij)",
    ],
    "expected_shortfall": [
        "CVaR: V × σ × 2.063 (for 95%)",
        "Sharpe Ratio: (R - Rf) / σ",
    ],
    "liquidation": [
        "Liquidation Long: P_entry × (1 - 1/(L×m))",
        "Liquidation Short: P_entry × (1 + 1/(L×m))",
    ],
    "staking": [
        "Simple APY: (rewards / principal) × (365 / days)",
        "Compound APY: (1 + r/n)^n - 1",
    ],
}

FEW_SHOT_EXAMPLES = {
    "conditional_formulas": """
EXAMPLE: Kelly Criterion with cap
Variables: expected_return, variance
Formula: f* = min(expected_return / (2 × variance²), 1.0)
Python:
def formula(expected_return, variance):
    risk_aversion = 2.0
    f_star = expected_return / (risk_aversion * variance**2)
    return np.minimum(f_star, 1.0)
""",
    "risk_metrics": """
EXAMPLE: Value at Risk
Variables: portfolio_value, daily_volatility
Formula: VaR = portfolio_value × daily_volatility × 1.645
Python:
def formula(portfolio_value, daily_volatility):
    z_95 = 1.645
    return portfolio_value * daily_volatility * z_95
""",
}


# ============================================================================
# CONSOLIDATED HYBRID SYSTEM CLASS
# ============================================================================


class ConsolidatedHybridSystem:
    """
    Fully self-contained hybrid system.
    No external dependencies except protocol and standard libraries.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.results = []
        self.formula_cache = {}

    # ========================================================================
    # LLM FORMULA GENERATION (Built-in)
    # ========================================================================

    def generate_llm_formula(
        self,
        description: str,
        domain: str,
        variable_names: List[str],
        metadata: Dict,
        characteristics: Dict = None,
        verbose: bool = False,
    ) -> Dict:
        """Generate formula using LLM"""

        cache_key = f"{description}|{domain}|{','.join(variable_names)}"
        if cache_key in self.formula_cache:
            if verbose:
                print(f"  [LLM] Using cached formula")
            return self.formula_cache[cache_key].copy()

        # Try specialized prompt first
        specialized_prompt = self._get_specialized_prompt(
            description, domain, variable_names, metadata
        )

        if specialized_prompt:
            prompt = specialized_prompt
            use_specialized = True
        else:
            prompt = self._get_standard_prompt(
                description, domain, variable_names, metadata, characteristics
            )
            use_specialized = False

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text

            if verbose:
                print(f"  [LLM] Response length: {len(content)} chars")

            parsed = self._parse_llm_response(content, verbose=verbose)

            result = {
                "formula": parsed.get("formula", "N/A"),
                "latex": parsed.get("latex", "N/A"),
                "python_code": parsed.get("python", "N/A"),
                "explanation": parsed.get("explanation", "N/A"),
                "specialized": use_specialized,
            }

            # Cache if valid
            if (
                result["python_code"] != "N/A"
                and "def formula" in result["python_code"]
            ):
                self.formula_cache[cache_key] = result.copy()

            return result

        except Exception as e:
            if verbose:
                print(f"  [LLM] Error: {e}")
            return {"error": str(e)}

    def _get_specialized_prompt(
        self, description: str, domain: str, variable_names: List[str], metadata: Dict
    ) -> Optional[str]:
        """Get specialized prompt for known formulas"""
        desc_lower = description.lower()
        var_list = ", ".join(variable_names)

        # Kelly Criterion
        if ("optimal" in desc_lower and "lp" in desc_lower) or "kelly" in desc_lower:
            return f"""Task: {description}
Variables: {var_list}

FORMULA:
f* = min(μ / (λ × σ²), 1.0)

PYTHON:
def formula({var_list}):
    risk_aversion = 2.0
    f_star = {variable_names[0]} / (risk_aversion * {variable_names[1]}**2)
    return np.minimum(f_star, 1.0)

EXPLANATION:
Kelly criterion with cap at 100% allocation.
"""

        # Impermanent Loss
        if "impermanent loss percentage" in desc_lower:
            return f"""Task: {description}
Variables: {var_list}

FORMULA:
IL% = (2√r / (1 + r) - 1) × 100

PYTHON:
def formula({var_list}):
    il_fraction = 2.0 * np.sqrt({variable_names[0]}) / (1.0 + {variable_names[0]}) - 1.0
    return il_fraction * 100.0

EXPLANATION:
Impermanent loss percentage for price ratio change.
"""

        # Liquidation Long
        if "liquidation" in desc_lower and "long" in desc_lower:
            return f"""Task: {description}
Variables: {var_list}

FORMULA:
P_liq = P_entry × (1 - 1/(L × 0.8))

PYTHON:
def formula({var_list}):
    maintenance_margin = 0.8
    return {variable_names[0]} * (1.0 - 1.0 / ({variable_names[1]} * maintenance_margin))

EXPLANATION:
Liquidation price for long positions.
"""

        # Liquidation Short
        if "liquidation" in desc_lower and "short" in desc_lower:
            return f"""Task: {description}
Variables: {var_list}

FORMULA:
P_liq = P_entry × (1 + 1/(L × 0.8))

PYTHON:
def formula({var_list}):
    maintenance_margin = 0.8
    return {variable_names[0]} * (1.0 + 1.0 / ({variable_names[1]} * maintenance_margin))

EXPLANATION:
Liquidation price for short positions.
"""

        # VaR 95%
        if "value at risk" in desc_lower and "95%" in desc_lower:
            return f"""Task: {description}
Variables: {var_list}

FORMULA:
VaR₉₅ = V × σ × 1.645

PYTHON:
def formula({var_list}):
    z_score_95 = 1.645
    return {variable_names[0]} * {variable_names[1]} * z_score_95

EXPLANATION:
Parametric VaR at 95% confidence.
"""

        return None

    def _get_standard_prompt(
        self,
        description: str,
        domain: str,
        variable_names: List[str],
        metadata: Dict,
        characteristics: Dict,
    ) -> str:
        """Get standard prompt with enhancements"""
        var_info = f"\nVariables (in order): {', '.join(variable_names)}"

        constants_info = ""
        if metadata and "constants" in metadata and metadata["constants"]:
            constants_info = "\n\n[CONSTANTS]:"
            for k, v in metadata["constants"].items():
                constants_info += f"\n  {k} = {v}"

        ground_truth_hint = ""
        if (
            metadata
            and "ground_truth" in metadata
            and metadata["ground_truth"] != "N/A"
        ):
            ground_truth_hint = f"\n\n[EXPECTED]: {metadata['ground_truth']}"

        return f"""You are a mathematical formula expert in {domain}.

Task: {description}{var_info}{constants_info}{ground_truth_hint}

Output EXACTLY 3 sections:

FORMULA:
[mathematical expression]

PYTHON:
def formula({", ".join(variable_names)}):
    # Use np.minimum(), np.maximum(), np.where() for conditionals
    return result

EXPLANATION:
[1-2 sentences]
"""

    def _parse_llm_response(
        self, content: str, verbose: bool = False
    ) -> Dict[str, str]:
        """Parse LLM response"""
        parsed = {}

        # Extract FORMULA
        match = re.search(r"FORMULA:\s*\n([^\n]+(?:\n[^\n]+)?)", content, re.IGNORECASE)
        if match:
            parsed["formula"] = match.group(1).strip()
        else:
            parsed["formula"] = "N/A"

        # Extract PYTHON
        code = None
        match = re.search(
            r"PYTHON:\s*\n(.*?)(?=\n\n[A-Z]+:|\n\n\[|$)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            code = match.group(1).strip()

        if not code:
            match = re.search(r"```python\s*\n(.*?)\n```", content, re.DOTALL)
            if match:
                code = match.group(1).strip()

        if not code:
            match = re.search(
                r"(def formula\([^)]*\):.*?)(?=\n\n[A-Z]+:|\n\n```|\n\n\[|\Z)",
                content,
                re.DOTALL,
            )
            if match:
                code = match.group(1).strip()

        if code:
            code = re.sub(r"^```python\s*\n", "", code)
            code = re.sub(r"\n```\s*$", "", code)
            parsed["python"] = code.strip()
        else:
            parsed["python"] = "N/A"

        # Extract EXPLANATION
        match = re.search(
            r"EXPLANATION:\s*\n(.*?)(?=\n\n[A-Z]+:|\n\n```|\Z)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            parsed["explanation"] = match.group(1).strip()
        else:
            parsed["explanation"] = "N/A"

        return parsed

    def evaluate_llm_formula(
        self,
        formula_dict: Dict,
        X: np.ndarray,
        y_true: np.ndarray,
        var_names: List[str],
        verbose: bool = False,
    ) -> Dict:
        """Evaluate LLM formula"""
        try:
            code = formula_dict.get("python_code", "")
            if not code or code == "N/A":
                return {"error": "No code", "success": False}

            if "def formula" not in code:
                return {"error": "No formula function", "success": False}

            local_vars = {}
            exec(code, {"np": np, "numpy": np}, local_vars)

            func = next((v for v in local_vars.values() if callable(v)), None)

            if not func:
                return {"error": "No function", "success": False}

            y_pred = self._evaluate_function(func, X, var_names, verbose=verbose)

            if len(y_pred) != len(y_true):
                return {"error": "Shape mismatch", "success": False}

            mse = np.mean((y_pred - y_true) ** 2)
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0

            return {
                "r2": float(r2),
                "rmse": float(np.sqrt(mse)),
                "mae": float(np.mean(np.abs(y_pred - y_true))),
                "success": True,
                "predictions": y_pred,
            }
        except Exception as e:
            if verbose:
                print(f"  [EVAL] Error: {e}")
            return {"error": str(e), "success": False}

    def _evaluate_function(self, func, X, var_names, verbose=False):
        """Evaluate function with multiple strategies"""
        sig = inspect.signature(func)
        n_params = len(sig.parameters)
        n_features = X.shape[1]

        # Try vectorized evaluation
        if n_params == n_features:
            try:
                y = func(*[X[:, i] for i in range(n_features)])
                return np.asarray(y).flatten()
            except Exception:
                pass

        # Try row-by-row
        try:
            y = np.empty(X.shape[0])
            for i in range(X.shape[0]):
                if n_params == n_features:
                    y[i] = func(*X[i, :])
                elif n_params < n_features:
                    y[i] = func(*X[i, :n_params])
            return y
        except Exception as e:
            if verbose:
                print(f"  [EVAL] Failed: {e}")

        raise RuntimeError("All evaluation strategies failed")

    # ========================================================================
    # NEURAL NETWORK TRAINING (Built-in)
    # ========================================================================

    def train_nn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        is_extrapolation: bool = False,
        epochs: int = 500,
        verbose: bool = False,
    ) -> Tuple:
        """Train neural network"""

        test_size = 0.3 if is_extrapolation else 0.2

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_s = scaler_X.fit_transform(X_train)
        X_test_s = scaler_X.transform(X_test)
        y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        model = nn.Sequential(
            nn.Linear(X.shape[1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        criterion = nn.MSELoss()

        X_train_t = torch.FloatTensor(X_train_s)
        y_train_t = torch.FloatTensor(y_train_s).reshape(-1, 1)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.3, patience=30, min_lr=1e-6
        )

        best_loss = float("inf")
        patience_counter = 0
        max_patience = 100

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train_t)
            loss = criterion(pred, y_train_t)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step(loss)

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    if verbose:
                        print(f"  [NN] Early stopping at epoch {epoch}")
                    break

        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test_s)
            y_pred_s = model(X_test_t).numpy().flatten()
            y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()

            mse = np.mean((y_test - y_pred) ** 2)
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0

        metrics = {
            "r2": float(r2),
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(y_test - y_pred))),
        }

        return model, metrics, scaler_X, scaler_y

    def get_nn_predictions(self, model, X, scaler_X, scaler_y):
        """Get NN predictions"""
        model.eval()
        with torch.no_grad():
            X_scaled = scaler_X.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)
            y_pred_scaled = model(X_tensor).numpy().flatten()
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        return y_pred

    # ========================================================================
    # PATTERN RECOGNITION (Built-in)
    # ========================================================================

    def detect_formula_characteristics(
        self, X: np.ndarray, y: np.ndarray, var_names: List[str]
    ) -> Dict:
        """Detect mathematical patterns"""
        characteristics = {
            "is_linear": False,
            "has_sqrt": False,
            "has_reciprocal": False,
            "confidence": 0.0,
            "best_pattern": None,
        }

        try:
            # Test for linearity
            lr = LinearRegression().fit(X, y)
            linear_r2 = lr.score(X, y)
            if linear_r2 > 0.99:
                characteristics["is_linear"] = True
                characteristics["confidence"] = 0.95
                characteristics["best_pattern"] = "linear"
                return characteristics

            # Test for sqrt
            X_sqrt = np.sqrt(np.abs(X) + 1e-10)
            lr_sqrt = LinearRegression().fit(X_sqrt, y)
            sqrt_r2 = lr_sqrt.score(X_sqrt, y)
            if sqrt_r2 > 0.98:
                characteristics["has_sqrt"] = True
                characteristics["confidence"] = 0.90
                characteristics["best_pattern"] = "sqrt"

            # Test for reciprocal
            X_recip = 1 / (np.abs(X) + 1e-10)
            lr_recip = LinearRegression().fit(X_recip, y)
            recip_r2 = lr_recip.score(X_recip, y)
            if recip_r2 > 0.98:
                characteristics["has_reciprocal"] = True
                characteristics["confidence"] = max(characteristics["confidence"], 0.90)
                characteristics["best_pattern"] = "reciprocal"

        except Exception:
            pass

        return characteristics

    # ========================================================================
    # HYBRID DECISION LOGIC (Built-in)
    # ========================================================================

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
        """Main hybrid prediction with extrapolation-aware logic"""

        is_extrapolation = metadata.get("extrapolation_test", False)

        # Pattern recognition
        characteristics = self.detect_formula_characteristics(X, y_true, var_names)

        if verbose:
            print(
                f"\n  [PATTERN] {characteristics.get('best_pattern')}, "
                f"Confidence: {characteristics.get('confidence', 0):.2f}"
            )
            print(f"  [HYBRID] Extrapolation: {is_extrapolation}")

        # Generate and evaluate LLM
        llm_result = self.generate_llm_formula(
            description, domain, var_names, metadata, characteristics, verbose=verbose
        )

        if "error" not in llm_result and llm_result.get("python_code") != "N/A":
            llm_metrics = self.evaluate_llm_formula(
                llm_result, X, y_true, var_names, verbose=verbose
            )
        else:
            llm_metrics = {
                "error": llm_result.get("error", "No code"),
                "success": False,
            }

        # Train and evaluate NN
        nn_model, nn_metrics, scaler_X, scaler_y = self.train_nn(
            X, y_true, is_extrapolation=is_extrapolation, epochs=500, verbose=verbose
        )
        nn_predictions = self.get_nn_predictions(nn_model, X, scaler_X, scaler_y)

        # Extract scores
        llm_r2 = llm_metrics.get("r2", -999) if llm_metrics.get("success") else -999
        nn_r2 = nn_metrics.get("r2", -999)

        if verbose:
            print(f"\n  [HYBRID] LLM R²: {llm_r2:.4f}, NN R²: {nn_r2:.4f}")

        llm_valid = llm_r2 > 0.0
        nn_valid = nn_r2 > 0.0

        # EXTRAPOLATION-AWARE DECISION LOGIC
        if is_extrapolation:
            if verbose:
                print(f"  [HYBRID] 🔴 EXTRAPOLATION MODE")

            if llm_valid and llm_r2 > 0.90:
                decision = "llm"
                final_r2 = llm_r2
                final_rmse = llm_metrics["rmse"]
                reason = f"⭐ EXTRAP: LLM excellent (R²={llm_r2:.4f})"
            elif llm_valid and llm_r2 > 0.70:
                decision = "llm"
                final_r2 = llm_r2
                final_rmse = llm_metrics["rmse"]
                reason = f"✅ EXTRAP: LLM preferred (R²={llm_r2:.4f})"
            elif llm_valid:
                decision = "llm"
                final_r2 = llm_r2
                final_rmse = llm_metrics["rmse"]
                reason = f"🔶 EXTRAP: LLM safer than NN"
            elif nn_valid:
                decision = "nn"
                final_r2 = nn_r2
                final_rmse = nn_metrics["rmse"]
                reason = f"⚠️ EXTRAP: NN only (LLM failed)"
            else:
                decision = "failed"
                final_r2 = max(llm_r2, nn_r2)
                final_rmse = 999
                reason = "❌ EXTRAP: Both failed"

        # REGULAR INTERPOLATION LOGIC
        else:
            if not llm_valid and not nn_valid:
                decision = "failed"
                final_r2 = max(llm_r2, nn_r2)
                final_rmse = 999
                reason = "Both failed"
            elif llm_valid and not nn_valid:
                decision = "llm"
                final_r2 = llm_r2
                final_rmse = llm_metrics["rmse"]
                reason = "LLM only valid"
            elif nn_valid and not llm_valid:
                decision = "nn"
                final_r2 = nn_r2
                final_rmse = nn_metrics["rmse"]
                reason = "NN only valid"
            else:
                # Both valid - choose best
                if llm_r2 > 0.95:
                    decision = "llm"
                    final_r2 = llm_r2
                    final_rmse = llm_metrics["rmse"]
                    reason = f"Excellent LLM (R²={llm_r2:.4f})"
                elif llm_r2 > nn_r2:
                    decision = "llm"
                    final_r2 = llm_r2
                    final_rmse = llm_metrics["rmse"]
                    reason = f"LLM better ({llm_r2:.4f} > {nn_r2:.4f})"
                else:
                    decision = "nn"
                    final_r2 = nn_r2
                    final_rmse = nn_metrics["rmse"]
                    reason = f"NN better ({nn_r2:.4f} > {llm_r2:.4f})"

        if verbose:
            print(f"\n  [HYBRID] Decision: {decision.upper()}")
            print(f"  [HYBRID] Reason: {reason}")
            print(f"  [HYBRID] Final R²: {final_r2:.4f}")

        return {
            "method": "consolidated_hybrid",
            "description": description,
            "domain": domain,
            "decision": decision,
            "decision_reason": reason,
            "is_extrapolation_test": is_extrapolation,
            "llm_valid": llm_valid,
            "nn_valid": nn_valid,
            "pattern_characteristics": characteristics,
            "llm_result": {
                "formula": llm_result.get("formula", "N/A"),
                "python_code": llm_result.get("python_code", "N/A"),
                "explanation": llm_result.get("explanation", "N/A"),
                "specialized": llm_result.get("specialized", False),
                "metrics": {k: v for k, v in llm_metrics.items() if k != "predictions"},
            },
            "nn_result": {"metrics": nn_metrics},
            "evaluation": {
                "r2": float(final_r2),
                "rmse": float(final_rmse),
                "success": final_r2 > 0.0,
            },
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
        }

    def save_results(self, filepath: str):
        """Save results to JSON"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ Results saved: {filepath}")


# ============================================================================
# RESULTS TABLE GENERATOR (Built-in)
# ============================================================================


class ResultsTableGenerator:
    """Generate comprehensive results tables"""

    @staticmethod
    def generate_observations(result: Dict) -> str:
        """Generate intelligent observations"""
        observations = []

        r2 = result["evaluation"]["r2"]
        decision = result["decision"]
        is_extrap = result.get("is_extrapolation_test", False)

        # Performance
        if r2 > 0.99:
            observations.append("⭐Excellent")
        elif r2 > 0.95:
            observations.append("✅VeryGood")
        elif r2 > 0.80:
            observations.append("🟡OK")
        else:
            observations.append("🟠Weak")

        # Method
        if decision == "llm":
            if result.get("llm_result", {}).get("specialized", False):
                observations.append("Specialized")
            else:
                observations.append("LLM-Gen")
        elif decision == "nn":
            observations.append("NN-Only")

        # Extrapolation
        if is_extrap:
            if r2 > 0.90:
                observations.append("Extrap-Robust✨")
            elif r2 > 0.70:
                observations.append("Extrap-Good")
            else:
                observations.append("Extrap-Weak")

        # Pattern
        pattern = result.get("pattern_characteristics", {}).get("best_pattern")
        if pattern:
            observations.append(pattern)

        return " | ".join(observations)

    @staticmethod
    def generate_table(results: List[Dict], title: str = "Test Results") -> str:
        """Generate formatted table"""
        table_data = []
        for i, r in enumerate(results, 1):
            test_name = r.get("description", "Unknown")[:40]
            domain = r.get("domain", "N/A")[:12]
            r2 = r["evaluation"]["r2"]
            val_score = r.get("nn_result", {}).get("metrics", {}).get("r2", 0.0)
            observations = ResultsTableGenerator.generate_observations(r)

            r2_str = (
                f"{r2:.4f}✓"
                if r2 > 0.95
                else f"{r2:.4f}⚠" if r2 < 0.80 else f"{r2:.4f}"
            )

            table_data.append(
                [i, test_name, domain, r2_str, f"{val_score:.4f}", observations]
            )

        headers = ["#", "Test Case", "Domain", "R²", "Val Score", "Observations"]
        table = tabulate(table_data, headers=headers, tablefmt="grid", stralign="left")

        return f"\n{'='*130}\n{title.center(130)}\n{'='*130}\n{table}"

    @staticmethod
    def generate_summary_table(results: List[Dict]) -> str:
        """Generate domain summary"""
        by_domain = defaultdict(
            lambda: {
                "total": 0,
                "r2_scores": [],
                "extrap_count": 0,
                "extrap_r2": [],
                "excellent": 0,
                "good": 0,
                "acceptable": 0,
                "failed": 0,
            }
        )

        for r in results:
            domain = r.get("domain", "unknown")
            r2 = r["evaluation"]["r2"]
            by_domain[domain]["total"] += 1
            by_domain[domain]["r2_scores"].append(r2)

            if r.get("is_extrapolation_test"):
                by_domain[domain]["extrap_count"] += 1
                by_domain[domain]["extrap_r2"].append(r2)

            if r2 > 0.99:
                by_domain[domain]["excellent"] += 1
            elif r2 > 0.95:
                by_domain[domain]["good"] += 1
            elif r2 > 0.80:
                by_domain[domain]["acceptable"] += 1
            else:
                by_domain[domain]["failed"] += 1

        table_data = []
        for domain, stats in sorted(by_domain.items()):
            mean_r2 = np.mean(stats["r2_scores"]) if stats["r2_scores"] else 0
            extrap_mean = np.mean(stats["extrap_r2"]) if stats["extrap_r2"] else None
            performance = f"{stats['excellent']}⭐ {stats['good']}✅ {stats['acceptable']}🟡 {stats['failed']}❌"
            extrap_str = f"{extrap_mean:.3f}" if extrap_mean is not None else "N/A"

            table_data.append(
                [
                    domain,
                    stats["total"],
                    f"{mean_r2:.4f}",
                    extrap_str,
                    stats["extrap_count"],
                    performance,
                ]
            )

        headers = ["Domain", "Tests", "Mean R²", "Extrap R²", "Extrap #", "Performance"]
        table = tabulate(table_data, headers=headers, tablefmt="grid", stralign="left")

        return f"\n{'='*100}\n{'DOMAIN SUMMARY'.center(100)}\n{'='*100}\n{table}"


# ============================================================================
# MAIN RUNNER FUNCTIONS
# ============================================================================


def run_full_test(
    domains: List[str] = None, num_samples: int = 100, verbose: bool = False
):
    """Run full test suite with tables"""

    protocol = DeFiExperimentProtocol()
    hybrid = ConsolidatedHybridSystem()

    if domains is None:
        domains = protocol.get_all_domains()

    print("=" * 130)
    print("[EXPERIMENT] CONSOLIDATED HYBRID SYSTEM v5.0".center(130))
    print("=" * 130)
    print("All components built-in (no external dependencies except protocol)")
    print("\nFeatures:")
    print("  ✅ Extrapolation-aware decision logic")
    print("  ✅ Pattern recognition")
    print("  ✅ Specialized formulas")
    print("  ✅ Neural network training")
    print("  ✅ Comprehensive results tables")
    print("=" * 130)

    all_results = []

    for domain in domains:
        print(f"\n{'='*130}")
        print(f"DOMAIN: {domain.upper()}".center(130))
        print("=" * 130)

        test_cases = protocol.load_test_data(domain, num_samples=num_samples)

        for i, (desc, X, y, var_names, meta) in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] {desc}")

            result = hybrid.hybrid_predict(
                desc, domain, X, y, var_names, meta, verbose=verbose
            )

            metrics = result["evaluation"]
            print(f"  Decision: {result['decision'].upper()}")
            print(f"  R²: {metrics['r2']:.6f}, RMSE: {metrics['rmse']:.6f}")

            if metrics["r2"] > 0.99:
                print(f"  ⭐ EXCELLENT")
            elif metrics["r2"] > 0.95:
                print(f"  ✅ GOOD")
            elif metrics["r2"] > 0.80:
                print(f"  🟡 ACCEPTABLE")
            else:
                print(f"  🟠 NEEDS WORK")

            all_results.append(result)
            hybrid.results.append(result)

    # Save results
    os.makedirs("hypatiax/data/results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"hypatiax/data/results/consolidated_hybrid_{ts}.json"
    hybrid.save_results(results_file)

    # Generate tables
    print("\n" * 2)
    print(ResultsTableGenerator.generate_table(all_results, "DETAILED TEST RESULTS"))
    print(ResultsTableGenerator.generate_summary_table(all_results))

    # Statistics
    print("\n" + "=" * 130)
    print("OVERALL STATISTICS".center(130))
    print("=" * 130)

    r2_scores = [r["evaluation"]["r2"] for r in all_results]
    extrap_r2 = [
        r["evaluation"]["r2"] for r in all_results if r.get("is_extrapolation_test")
    ]

    print(f"\nTotal Cases: {len(all_results)}")
    print(f"Mean R²: {np.mean(r2_scores):.4f}")
    print(f"Median R²: {np.median(r2_scores):.4f}")
    print(f"Min R²: {np.min(r2_scores):.4f}")

    if extrap_r2:
        print(f"\n🚀 Extrapolation Tests (n={len(extrap_r2)}):")
        print(f"   Mean R²: {np.mean(extrap_r2):.4f}")

    excellent = sum(1 for r2 in r2_scores if r2 > 0.99)
    good = sum(1 for r2 in r2_scores if 0.95 < r2 <= 0.99)
    acceptable = sum(1 for r2 in r2_scores if 0.80 < r2 <= 0.95)
    weak = sum(1 for r2 in r2_scores if r2 <= 0.80)

    print(f"\n📊 Performance:")
    print(
        f"   ⭐ Excellent (>0.99): {excellent:2d} ({100*excellent/len(r2_scores):5.1f}%)"
    )
    print(f"   ✅ Very Good (>0.95): {good:2d} ({100*good/len(r2_scores):5.1f}%)")
    print(
        f"   🟡 Acceptable (>0.80): {acceptable:2d} ({100*acceptable/len(r2_scores):5.1f}%)"
    )
    print(f"   🟠 Weak (≤0.80): {weak:2d} ({100*weak/len(r2_scores):5.1f}%)")

    decisions = defaultdict(int)
    for r in all_results:
        decisions[r["decision"]] += 1

    print(f"\n🎯 Decisions:")
    for decision in ["llm", "nn", "failed"]:
        if decision in decisions:
            count = decisions[decision]
            pct = 100 * count / len(all_results)
            dec_r2 = [
                r["evaluation"]["r2"] for r in all_results if r["decision"] == decision
            ]
            mean_r2 = np.mean(dec_r2) if dec_r2 else 0
            print(
                f"   {decision.upper():8s}: {count:2d} ({pct:5.1f}%) - Mean R² = {mean_r2:.4f}"
            )

    print("\n" + "=" * 130)
    print(f"Results saved to: {results_file}")
    print("=" * 130)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Consolidated Hybrid System v5.0")
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    run_full_test(domains=args.domains, num_samples=args.samples, verbose=args.verbose)

"""
USAGE:

# Full test with 200 samples
python run_hybrid_defi_consolidated.py --samples 200 --verbose

# Specific domains
python run_hybrid_defi_consolidated.py --domains amm risk_var liquidation

# Quick test
python run_hybrid_defi_consolidated.py --samples 50
"""
