"""
Comparative Test Suite: When LLM-Guided PySR Outperforms All Other Methods
===========================================================================

Tests 5 scenarios where Hybrid LLM+PySR beats:
1. PySR + validation (pure symbolic regression)
2. Baseline pure LLM (no symbolic search)
3. Neural Network (pure ML)
4. LLM + NN ensemble (hybrid ML)

Tested on:
- All scientific domains (physics, chemistry, biology, etc.)
- DeFi domain (specialized financial formulas)
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

# PySR imports
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    print("⚠️  PySR not available - install with: pip install pysr")

# Load environment - try multiple locations
env_paths = [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env",
    Path(__file__).parent.parent.parent / ".env",
    Path(__file__).parent.parent.parent.parent / ".env",
    Path.cwd() / ".env",
    Path.cwd() / "hypatiax" / ".env",
    Path(__file__).parent.parent.parent / "hypatiax" / ".env",  # Added this path
]

env_loaded = False
api_key_found = False

for env_path in env_paths:
    if env_path.exists():
        print(f"📍 Found .env at: {env_path}")
        load_dotenv(dotenv_path=env_path, override=True)
        
        # Check if API key is now available
        if os.getenv("ANTHROPIC_API_KEY"):
            print(f"✅ Loaded .env from: {env_path}")
            print(f"✅ API key found: {os.getenv('ANTHROPIC_API_KEY')[:20]}...")
            env_loaded = True
            api_key_found = True
            break
        else:
            print(f"⚠️  .env file found but no ANTHROPIC_API_KEY in it")

if not env_loaded:
    print("⚠️  No .env file with API key found in standard locations. Trying environment variables...")
    load_dotenv(override=True)
    if os.getenv("ANTHROPIC_API_KEY"):
        print("✅ API key found in environment variables")
        api_key_found = True
    else:
        print("❌ No API key found. Checked locations:")
        for p in env_paths:
            exists = "✓" if p.exists() else "✗"
            print(f"   {exists} {p}")
        print("\n💡 Tip: Make sure your .env file contains:")
        print("   ANTHROPIC_API_KEY=sk-ant-...")


class ComparativeTestSuite:
    """
    Test suite comparing 5 different approaches:
    1. Hybrid LLM + PySR (our approach)
    2. PySR + validation only
    3. Pure LLM baseline
    4. Neural Network
    5. LLM + NN ensemble
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            # Try one more time with explicit path
            hypatiax_env = Path.cwd() / "hypatiax" / ".env"
            if hypatiax_env.exists():
                print(f"\n🔄 Attempting to reload from: {hypatiax_env}")
                load_dotenv(dotenv_path=hypatiax_env, override=True)
                api_key = os.getenv("ANTHROPIC_API_KEY")
                
                if api_key:
                    print(f"✅ Successfully loaded API key on retry")
                else:
                    # Check what's actually in the file
                    print(f"\n🔍 Checking contents of {hypatiax_env}:")
                    with open(hypatiax_env, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.strip() and not line.strip().startswith('#'):
                                key_name = line.split('=')[0].strip()
                                print(f"   Found key: {key_name}")
            
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not set. Please ensure it's in your .env file.\n"
                    f"Checked locations:\n" + 
                    "\n".join([f"  - {p}" for p in env_paths]) +
                    f"\n\n💡 Your .env is at: {Path.cwd() / 'hypatiax' / '.env'}\n" +
                    "   Make sure it contains: ANTHROPIC_API_KEY=sk-ant-..."
                )
        
        print(f"✅ Initializing ComparativeTestSuite with API key: {api_key[:20]}...")
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.results = []

    # ========================================================================
    # METHOD 1: HYBRID LLM + PySR (Our Approach)
    # ========================================================================
    
    def method_llm_guided_pysr(
        self, description: str, X: np.ndarray, y: np.ndarray, 
        var_names: List[str], metadata: Dict, verbose: bool = False
    ) -> Dict:
        """
        Hybrid approach: LLM guides PySR search space
        
        Advantages:
        - LLM provides domain knowledge (operators, functional forms)
        - PySR refines to exact symbolic formula
        - Validates against physical constraints
        - Best of both worlds: interpretability + accuracy
        """
        if not PYSR_AVAILABLE:
            return {"error": "PySR not available", "r2": 0.0}
        
        if verbose:
            print("  [LLM+PySR] Getting LLM guidance...")
        
        # Step 1: Get LLM guidance on expected operators and form
        guidance = self._get_llm_guidance(description, var_names, metadata)
        
        # Step 2: Configure PySR with LLM-suggested operators
        operators = guidance.get("operators", ["add", "sub", "mul", "div"])
        unary_operators = guidance.get("unary_operators", ["exp", "log", "sqrt"])
        
        # Add pow if not in operators (needed for power laws)
        if "pow" not in operators and any(hint in str(metadata).lower() for hint in ["power", "square", "^"]):
            operators.append("pow")
        
        if verbose:
            print(f"  [LLM+PySR] Operators: {operators}")
            print(f"  [LLM+PySR] Unary: {unary_operators}")
        
        # Step 3: Run PySR with guided search - IMPROVED SETTINGS
        model = PySRRegressor(
            niterations=100,  # Increased from 40
            binary_operators=operators,
            unary_operators=unary_operators,
            populations=15,  # Increased from 8
            population_size=50,  # Increased from 33
            maxsize=20,  # Increased from 15
            complexity_of_operators={op: 1 for op in operators},
            timeout_in_seconds=180,  # Increased from 60
            parsimony=0.001,  # Reduced from 0.01 for less penalty
            random_state=42,
            constraints={
                "pow": (-1, 1),  # Base can be complex, power must be simple
            } if "pow" in operators else {},
            nested_constraints={
                "exp": {"exp": 0, "log": 0},  # Don't nest exp/log
                "log": {"exp": 0, "log": 0},
            },
            warm_start=False,
            verbosity=0,
        )
        
        try:
            model.fit(X, y, variable_names=var_names)
            
            # Get best equation
            y_pred = model.predict(X)
            
            # Check for invalid predictions
            if not np.all(np.isfinite(y_pred)):
                return {
                    "method": "llm_guided_pysr", 
                    "error": "Non-finite predictions", 
                    "r2": 0.0, 
                    "success": False
                }
            
            r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
            rmse = np.sqrt(np.mean((y - y_pred)**2))
            
            # Step 4: Validate with LLM
            formula_str = str(model.get_best())
            validation = self._validate_with_llm(formula_str, description, metadata)
            
            return {
                "method": "llm_guided_pysr",
                "formula": formula_str,
                "r2": float(r2),
                "rmse": float(rmse),
                "validation": validation,
                "success": True
            }
        except Exception as e:
            if verbose:
                print(f"  [LLM+PySR] Error: {str(e)[:100]}")
            return {"method": "llm_guided_pysr", "error": str(e)[:100], "r2": 0.0, "success": False}

    # ========================================================================
    # METHOD 2: PySR + Validation Only
    # ========================================================================
    
    def method_pysr_validation(
        self, description: str, X: np.ndarray, y: np.ndarray,
        var_names: List[str], metadata: Dict, verbose: bool = False
    ) -> Dict:
        """
        Pure PySR with standard operators + post-validation
        
        Disadvantages vs LLM+PySR:
        - No domain knowledge guidance
        - May explore irrelevant operators
        - Slower convergence
        - May miss domain-specific functional forms
        """
        if not PYSR_AVAILABLE:
            return {"error": "PySR not available", "r2": 0.0}
        
        if verbose:
            print("  [PySR] Running standard PySR...")
        
        # Standard operators (no LLM guidance) - IMPROVED SETTINGS
        model = PySRRegressor(
            niterations=100,  # Increased from 40
            binary_operators=["add", "sub", "mul", "div", "pow"],
            unary_operators=["exp", "log", "sqrt", "square"],
            populations=15,  # Increased from 8
            population_size=50,  # Increased from 33
            maxsize=20,  # Increased from 15
            timeout_in_seconds=180,  # Increased from 60
            parsimony=0.001,  # Reduced from 0.01
            random_state=42,
            constraints={"pow": (-1, 1)},
            nested_constraints={
                "exp": {"exp": 0, "log": 0},
                "log": {"exp": 0, "log": 0},
            },
            verbosity=0,
        )
        
        try:
            model.fit(X, y, variable_names=var_names)
            y_pred = model.predict(X)
            
            # Check for invalid predictions
            if not np.all(np.isfinite(y_pred)):
                return {
                    "method": "pysr_validation", 
                    "error": "Non-finite predictions", 
                    "r2": 0.0, 
                    "success": False
                }
            
            r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
            rmse = np.sqrt(np.mean((y - y_pred)**2))
            
            return {
                "method": "pysr_validation",
                "formula": str(model.get_best()),
                "r2": float(r2),
                "rmse": float(rmse),
                "success": True
            }
        except Exception as e:
            if verbose:
                print(f"  [PySR] Error: {str(e)[:100]}")
            return {"method": "pysr_validation", "error": str(e)[:100], "r2": 0.0, "success": False}

    # ========================================================================
    # METHOD 3: Pure LLM Baseline
    # ========================================================================
    
    def method_pure_llm(
        self, description: str, X: np.ndarray, y: np.ndarray,
        var_names: List[str], metadata: Dict, verbose: bool = False
    ) -> Dict:
        """
        Pure LLM symbolic generation (no refinement)
        
        Disadvantages vs LLM+PySR:
        - May have small numerical errors
        - Cannot refine to optimal form
        - Limited to knowledge cutoff
        - No iterative improvement
        """
        if verbose:
            print("  [LLM] Generating formula...")
        
        prompt = self._generate_llm_prompt(description, var_names, metadata)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            parsed = self._parse_llm_response(content)
            
            # Evaluate
            code = parsed.get("python_code", "")
            if not code:
                return {"method": "pure_llm", "error": "No code", "r2": 0.0, "success": False}
            
            local_vars = {}
            exec(code, {"np": np}, local_vars)
            func = next((v for v in local_vars.values() if callable(v)), None)
            
            if not func:
                return {"method": "pure_llm", "error": "No function", "r2": 0.0, "success": False}
            
            y_pred = self._evaluate_function(func, X, var_names)
            r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
            rmse = np.sqrt(np.mean((y - y_pred)**2))
            
            return {
                "method": "pure_llm",
                "formula": parsed.get("formula", "N/A"),
                "r2": float(r2),
                "rmse": float(rmse),
                "success": True
            }
        except Exception as e:
            return {"method": "pure_llm", "error": str(e)[:100], "r2": 0.0, "success": False}

    # ========================================================================
    # METHOD 4: Neural Network
    # ========================================================================
    
    def method_neural_network(
        self, description: str, X: np.ndarray, y: np.ndarray,
        var_names: List[str], metadata: Dict, verbose: bool = False
    ) -> Dict:
        """
        Pure neural network approach
        
        Disadvantages vs LLM+PySR:
        - Black box (no interpretability)
        - Cannot extract symbolic formula
        - Requires more data
        - Poor extrapolation
        """
        if verbose:
            print("  [NN] Training neural network...")
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_s = scaler_X.fit_transform(X_train)
        X_test_s = scaler_X.transform(X_test)
        y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        X_train_t = torch.FloatTensor(X_train_s)
        y_train_t = torch.FloatTensor(y_train_s).reshape(-1, 1)
        
        for epoch in range(300):
            optimizer.zero_grad()
            pred = model(X_train_t)
            loss = criterion(pred, y_train_t)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test_s)
            y_pred_s = model(X_test_t).numpy().flatten()
            y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
            
            r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
            rmse = np.sqrt(np.mean((y_test - y_pred)**2))
        
        return {
            "method": "neural_network",
            "formula": "Black box NN",
            "r2": float(r2),
            "rmse": float(rmse),
            "success": True
        }

    # ========================================================================
    # METHOD 5: LLM + NN Ensemble
    # ========================================================================
    
    def method_llm_nn_ensemble(
        self, description: str, X: np.ndarray, y: np.ndarray,
        var_names: List[str], metadata: Dict, verbose: bool = False
    ) -> Dict:
        """
        LLM + NN ensemble (weighted average)
        
        Disadvantages vs LLM+PySR:
        - Still has black box component
        - No pure symbolic formula
        - More complex deployment
        - Harder to interpret
        """
        if verbose:
            print("  [LLM+NN] Running ensemble...")
        
        llm_result = self.method_pure_llm(description, X, y, var_names, metadata, verbose=False)
        nn_result = self.method_neural_network(description, X, y, var_names, metadata, verbose=False)
        
        # Weight by R² scores
        llm_r2 = llm_result.get("r2", 0)
        nn_r2 = nn_result.get("r2", 0)
        
        if llm_r2 + nn_r2 > 0:
            ensemble_r2 = max(llm_r2, nn_r2)  # Best performer
            ensemble_rmse = min(llm_result.get("rmse", 1e10), nn_result.get("rmse", 1e10))
        else:
            ensemble_r2 = 0
            ensemble_rmse = float('inf')
        
        return {
            "method": "llm_nn_ensemble",
            "formula": f"Ensemble (LLM: {llm_result.get('formula', 'N/A')[:30]}...)",
            "r2": float(ensemble_r2),
            "rmse": float(ensemble_rmse),
            "llm_r2": float(llm_r2),
            "nn_r2": float(nn_r2),
            "success": True
        }

    # ========================================================================
    # Helper Functions
    # ========================================================================
    
    def _get_llm_guidance(self, description: str, var_names: List[str], metadata: Dict) -> Dict:
        """Get LLM guidance on operators and functional form"""
        prompt = f"""You are a mathematical expert. Analyze this formula task:

Task: {description}
Variables: {', '.join(var_names)}
Domain: {metadata.get('domain', 'unknown')}

Suggest:
1. Binary operators needed (add, sub, mul, div, pow)
2. Unary operators needed (exp, log, sqrt, sin, cos, abs)
3. Expected functional form hints

Format:
BINARY_OPERATORS: [list]
UNARY_OPERATORS: [list]
FORM_HINT: [description]
"""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
            
            # Parse operators
            binary = re.search(r"BINARY_OPERATORS:\s*\[(.*?)\]", content)
            unary = re.search(r"UNARY_OPERATORS:\s*\[(.*?)\]", content)
            
            binary_ops = [op.strip().strip("'\"") for op in binary.group(1).split(",")] if binary else ["add", "sub", "mul", "div"]
            unary_ops = [op.strip().strip("'\"") for op in unary.group(1).split(",")] if unary else ["exp", "log"]
            
            return {
                "operators": binary_ops,
                "unary_operators": unary_ops,
                "form_hint": content
            }
        except:
            return {
                "operators": ["add", "sub", "mul", "div"],
                "unary_operators": ["exp", "log"]
            }
    
    def _validate_with_llm(self, formula: str, description: str, metadata: Dict) -> str:
        """Validate formula makes physical/mathematical sense"""
        prompt = f"""Validate this discovered formula:

Task: {description}
Formula: {formula}

Does this make physical/mathematical sense? (YES/NO + brief reason)
"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text[:100]
        except:
            return "Validation unavailable"
    
    def _generate_llm_prompt(self, description: str, var_names: List[str], metadata: Dict) -> str:
        """Generate prompt for pure LLM"""
        return f"""Generate a mathematical formula for:

Task: {description}
Variables: {', '.join(var_names)}
Domain: {metadata.get('domain', 'unknown')}

Format:
FORMULA: [mathematical notation]
PYTHON:
def formula({', '.join(var_names)}):
    return result
"""
    
    def _parse_llm_response(self, content: str) -> Dict:
        """Parse LLM response"""
        formula_match = re.search(r"FORMULA:\s*([^\n]+)", content, re.IGNORECASE)
        python_match = re.search(r"PYTHON:\s*\n(.*?)(?=\n\n|\Z)", content, re.DOTALL | re.IGNORECASE)
        
        return {
            "formula": formula_match.group(1).strip() if formula_match else "N/A",
            "python_code": python_match.group(1).strip() if python_match else "N/A"
        }
    
    def _evaluate_function(self, func, X, var_names):
        """Evaluate function"""
        sig = inspect.signature(func)
        n_params = len(sig.parameters)
        
        try:
            y = func(*[X[:, i] for i in range(n_params)])
            return np.asarray(y).flatten()
        except:
            y = np.empty(X.shape[0])
            for i in range(X.shape[0]):
                y[i] = func(*X[i, :n_params])
            return y

    # ========================================================================
    # Test Runner
    # ========================================================================
    
    def run_comparative_test(
        self, description: str, X: np.ndarray, y: np.ndarray,
        var_names: List[str], metadata: Dict, domain: str, verbose: bool = True
    ) -> Dict:
        """Run all 5 methods and compare"""
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Test: {description}")
            print(f"Domain: {domain}")
            print(f"{'='*80}")
        
        results = {}
        
        # Method 1: LLM + PySR
        if verbose:
            print("\n[1/5] Running LLM-Guided PySR...")
        results["llm_pysr"] = self.method_llm_guided_pysr(
            description, X, y, var_names, metadata, verbose
        )
        
        # Method 2: PySR + Validation
        if verbose:
            print("\n[2/5] Running PySR + Validation...")
        results["pysr_validation"] = self.method_pysr_validation(
            description, X, y, var_names, metadata, verbose
        )
        
        # Method 3: Pure LLM
        if verbose:
            print("\n[3/5] Running Pure LLM...")
        results["pure_llm"] = self.method_pure_llm(
            description, X, y, var_names, metadata, verbose
        )
        
        # Method 4: Neural Network
        if verbose:
            print("\n[4/5] Running Neural Network...")
        results["neural_network"] = self.method_neural_network(
            description, X, y, var_names, metadata, verbose
        )
        
        # Method 5: LLM + NN Ensemble
        if verbose:
            print("\n[5/5] Running LLM + NN Ensemble...")
        results["llm_nn_ensemble"] = self.method_llm_nn_ensemble(
            description, X, y, var_names, metadata, verbose
        )
        
        # Compare
        comparison = self._compare_results(results)
        
        if verbose:
            self._print_comparison_table(results, comparison)
        
        return {
            "description": description,
            "domain": domain,
            "results": results,
            "comparison": comparison,
            "winner": comparison["winner"],
            "timestamp": datetime.now().isoformat()
        }
    
    def _compare_results(self, results: Dict) -> Dict:
        """Compare all methods"""
        r2_scores = {name: res.get("r2", 0) for name, res in results.items()}
        winner = max(r2_scores, key=r2_scores.get)
        
        # Check if LLM+PySR wins
        llm_pysr_r2 = r2_scores.get("llm_pysr", 0)
        advantages = []
        
        for method, r2 in r2_scores.items():
            if method != "llm_pysr" and llm_pysr_r2 > r2:
                diff = llm_pysr_r2 - r2
                advantages.append(f"{method}: +{diff:.4f}")
        
        return {
            "winner": winner,
            "scores": r2_scores,
            "llm_pysr_advantages": advantages,
            "llm_pysr_wins": winner == "llm_pysr"
        }
    
    def _print_comparison_table(self, results: Dict, comparison: Dict):
        """Print comparison table"""
        print("\n" + "="*80)
        print("COMPARISON RESULTS".center(80))
        print("="*80)
        
        print(f"\n{'Method':<25} {'R²':<12} {'RMSE':<12} {'Status':<20}")
        print("-"*80)
        
        for method, res in results.items():
            r2 = res.get("r2", 0)
            rmse = res.get("rmse", float('inf'))
            status = "✓ WINNER" if comparison["winner"] == method else ""
            
            print(f"{method:<25} {r2:<12.6f} {rmse:<12.6f} {status:<20}")
        
        print("="*80)
        
        if comparison["llm_pysr_wins"]:
            print("\n🎯 LLM+PySR WINS!")
            print(f"Advantages over other methods:")
            for adv in comparison["llm_pysr_advantages"]:
                print(f"  • {adv}")
        else:
            print(f"\n⚠️  Winner: {comparison['winner'].upper()}")


# ============================================================================
# 5 STRATEGIC TEST SCENARIOS
# ============================================================================

class FiveStrategicTests:
    """
    5 scenarios where LLM+PySR should outperform all other methods
    Uses proper test data from experiment protocol
    """
    
    @staticmethod
    def get_all_tests_from_protocol(
        benchmark: str = "feynman",
        num_samples: int = 200,
        series: str = None,
    ):
        """Get tests from experiment_protocol_benchmark.BenchmarkProtocol.

        Args:
            benchmark   : "feynman" | "srbench" | "both"  (default: "feynman")
            num_samples : Data points per equation         (default: 200)
            series      : Feynman series filter: "I" | "II" | "III" |
                          "crossover" | None (all)
        """
        # Import the new benchmark protocol.
        # Supports both running from the project root and from the benchmarks/
        # sub-directory so the import path works in both contexts.
        try:
            from hypatiax.protocols.experiment_protocol_benchmark import BenchmarkProtocol
        except ImportError:
            from experiment_protocol_benchmark import BenchmarkProtocol

        protocol = BenchmarkProtocol(
            benchmark=benchmark,
            num_samples=num_samples,
            seed=42,
            feynman_series=series,
        )
        all_tests = []

        for domain in protocol.get_all_domains():
            test_cases = protocol.load_test_data(domain, num_samples=num_samples)
            for desc, X, y, var_names, meta in test_cases:
                # `meta["domain"]` is the physics sub-domain (e.g. "mechanics");
                # `domain` from the loop is the protocol key ("feynman_mechanics").
                # Use the bare sub-domain for display / filtering consistency.
                bare_domain = meta.get("domain", domain.replace("feynman_", ""))
                formula_type = meta.get("formula_type", "physics")
                all_tests.append({
                    "name": meta["equation_name"],
                    "domain": bare_domain,
                    "description": desc,
                    "ground_truth": meta.get("ground_truth", "unknown"),
                    "X": X,
                    "y": y,
                    "var_names": var_names,
                    "metadata": meta,
                    "why_wins": f"LLM understands {formula_type}, PySR refines",
                })

        return all_tests
        """
        TEST 1: Complex Symbolic Relationships (All Domains)
        
        Why LLM+PySR wins:
        - LLM provides domain knowledge (e.g., knows physics uses exp(-E/kT))
        - PySR refines to exact coefficients
        - Pure LLM may have numerical errors
        - NN cannot capture symbolic form
        - PySR alone may not find domain-specific operators
        
        Example: Arrhenius equation k = A*exp(-Ea/RT)
        """
        return {
            "name": "Complex Symbolic (All Domains)",
            "domain": "chemistry",
            "description": "Arrhenius Equation: k = A*exp(-Ea/(R*T))",
            "ground_truth": "A * exp(-Ea / (R * T))",
            "generator": lambda n: {
                "A": np.random.uniform(1e10, 1e12, n),
                "Ea": np.random.uniform(50000, 100000, n),
                "R": np.full(n, 8.314),
                "T": np.random.uniform(300, 500, n)
            },
            "formula": lambda d: d["A"] * np.exp(-d["Ea"] / (d["R"] * d["T"])),
            "var_names": ["A", "Ea", "R", "T"],
            "why_wins": "LLM knows exp(-E/kT) pattern, PySR refines coefficients"
        }
    
    @staticmethod
    def test_2_multi_term_interactions_defi():
        """
        TEST 2: Multi-term Interactions (DeFi Domain)
        
        Why LLM+PySR wins:
        - DeFi formulas often have multiple interacting terms
        - LLM understands financial relationships
        - PySR discovers exact weights
        - Pure approaches miss term interactions
        
        Example: Impermanent Loss = 2*sqrt(price_ratio) / (1 + price_ratio) - 1
        """
        return {
            "name": "Multi-term Interactions (DeFi)",
            "domain": "defi",
            "description": "Impermanent Loss Formula",
            "ground_truth": "2 * sqrt(p1/p0) / (1 + p1/p0) - 1",
            "generator": lambda n: {
                "p0": np.random.uniform(100, 1000, n),
                "p1": np.random.uniform(100, 1000, n)
            },
            "formula": lambda d: 2 * np.sqrt(d["p1"]/d["p0"]) / (1 + d["p1"]/d["p0"]) - 1,
            "var_names": ["p0", "p1"],
            "why_wins": "LLM understands DeFi math, PySR finds exact form"
        }
    
    @staticmethod
    def test_3_nonlinear_scaling_all_domains():
        """
        TEST 3: Nonlinear Scaling Laws (All Domains)
        
        Why LLM+PySR wins:
        - Scaling laws common in biology/physics
        - LLM knows to try power laws
        - PySR discovers exact exponents
        - NN overfits, pure LLM may guess wrong exponent
        
        Example: Metabolic rate ∝ Mass^(3/4)
        """
        return {
            "name": "Nonlinear Scaling (All Domains)",
            "domain": "biology",
            "description": "Allometric Scaling: Y = a*M^b",
            "ground_truth": "a * M^b",
            "generator": lambda n: {
                "a": np.full(n, 3.5),
                "M": np.random.uniform(1, 100, n),
                "b": np.full(n, 0.75)
            },
            "formula": lambda d: d["a"] * d["M"] ** d["b"],
            "var_names": ["a", "M", "b"],
            "why_wins": "LLM suggests power law, PySR finds exact exponent"
        }
    
    @staticmethod
    def test_4_ratio_based_defi():
        """
        TEST 4: Ratio-based Formulas (DeFi Domain)
        
        Why LLM+PySR wins:
        - DeFi heavy on ratios and fractions
        - LLM understands financial ratios
        - PySR optimizes structure
        - NN struggles with division
        
        Example: Utilization = Borrowed / (Borrowed + Available)
        """
        return {
            "name": "Ratio-based (DeFi)",
            "domain": "defi",
            "description": "Utilization Rate Formula",
            "ground_truth": "borrowed / (borrowed + available)",
            "generator": lambda n: {
                "borrowed": np.random.uniform(1000, 10000, n),
                "available": np.random.uniform(1000, 10000, n)
            },
            "formula": lambda d: d["borrowed"] / (d["borrowed"] + d["available"]),
            "var_names": ["borrowed", "available"],
            "why_wins": "LLM knows ratio structure, PySR optimizes"
        }
    
    @staticmethod
    def test_5_composite_functions_all_domains():
        """
        TEST 5: Composite Functions (All Domains)
        
        Why LLM+PySR wins:
        - Functions composed of multiple operations
        - LLM provides functional decomposition
        - PySR searches structured space
        - Other methods struggle with composition
        
        Example: Nernst equation E = E0 - (RT/nF)*ln(Q)
        """
        return {
            "name": "Composite Functions (All Domains)",
            "domain": "chemistry",
            "description": "Nernst Equation: E = E0 - (RT/nF)*ln(Q)",
            "ground_truth": "E0 - (R*T/(n*F)) * log(Q)",
            "generator": lambda n: {
                "E0": np.random.uniform(0.5, 1.5, n),
                "R": np.full(n, 8.314),
                "T": np.random.uniform(273, 373, n),
                "n": np.random.randint(1, 3, n).astype(float),
                "F": np.full(n, 96485),
                "Q": np.random.uniform(0.1, 10, n)
            },
            "formula": lambda d: d["E0"] - (d["R"]*d["T"]/(d["n"]*d["F"])) * np.log(d["Q"]),
            "var_names": ["E0", "R", "T", "n", "F", "Q"],
            "why_wins": "LLM provides structure, PySR refines components"
        }


# ============================================================================
# Main Test Runner
# ============================================================================

def run_five_strategic_tests(
    num_samples: int = 200,
    verbose: bool = True,
    domain_filter: str = 'all',
    benchmark: str = 'feynman',
    series: str = None,
):
    """
    Run strategic tests using experiment_protocol_benchmark.BenchmarkProtocol.

    Args:
        num_samples   : Number of data points per equation (default: 200)
        verbose       : Print detailed output
        domain_filter : Filter tests by domain
            - 'all'        : Run all tests
            - 'all_domains': Scientific domains only (chemistry, biology, physics, …)
            - 'defi'       : DeFi domain only
            - <string>     : Any specific bare domain name
        benchmark     : "feynman" | "srbench" | "both"  (default: "feynman")
        series        : Feynman series filter: "I" | "II" | "III" |
                        "crossover" | None (all)
    """

    suite = ComparativeTestSuite()
    tests_manager = FiveStrategicTests()

    # Get all tests from the benchmark protocol
    all_test_configs = tests_manager.get_all_tests_from_protocol(
        benchmark=benchmark,
        num_samples=num_samples,
        series=series,
    )
    
    # Filter by domain
    if domain_filter == 'all_domains':
        # Scientific domains only
        test_configs = [t for t in all_test_configs if t['domain'] in ['chemistry', 'biology', 'physics']]
    elif domain_filter == 'defi':
        # DeFi domain only
        test_configs = [t for t in all_test_configs if t['domain'] == 'defi']
    elif domain_filter == 'all':
        test_configs = all_test_configs
    else:
        # Specific domain
        test_configs = [t for t in all_test_configs if t['domain'] == domain_filter]
    
    print("\n" + "="*80)
    if domain_filter != 'all':
        print(f"STRATEGIC TESTS: LLM+PySR vs ALL ({domain_filter.upper()})".center(80))
    else:
        print("STRATEGIC TESTS: LLM+PySR vs ALL METHODS".center(80))
    print("="*80)
    print(f"Running {len(test_configs)} test cases from experiment protocol")
    print("="*80)
    
    all_results = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_configs)}: {config['name']}".center(80))
        print(f"Domain: {config['domain']} | Formula: {config['ground_truth']}")
        print("="*80)
        
        # Use data from protocol
        X = config['X']
        y = config['y']
        var_names = config['var_names']
        metadata = config['metadata']
        
        # Run test
        result = suite.run_comparative_test(
            config["description"],
            X, y,
            var_names,
            metadata,
            config["domain"],
            verbose=verbose
        )
        
        all_results.append(result)
    
    # Final Summary
    print("\n" + "="*80)
    print(f"FINAL SUMMARY: LLM+PySR Performance ({domain_filter.upper()})".center(80))
    print("="*80)
    
    wins = sum(1 for r in all_results if r["comparison"]["llm_pysr_wins"])
    total = len(test_configs)
    
    print(f"\n🎯 LLM+PySR Won: {wins}/{total} tests ({100*wins/total:.0f}%)")
    print("\nDetailed Breakdown:")
    print("-"*80)
    print(f"{'Test':<40} {'Winner':<20} {'LLM+PySR R²':<15}")
    print("-"*80)
    
    for i, result in enumerate(all_results, 1):
        test_name = test_configs[i-1]["name"]
        winner = result["winner"]
        llm_pysr_r2 = result["results"]["llm_pysr"].get("r2", 0)
        
        status = "✓" if winner == "llm_pysr" else "✗"
        print(f"{status} {test_name:<38} {winner:<20} {llm_pysr_r2:.6f}")
    
    print("="*80)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/strategic_tests_{domain_filter}_{ts}.json"
    with open(filename, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {filename}")
    
    return all_results


def run_all_domains_tests(num_samples: int = 200, verbose: bool = True):
    """Run tests for scientific domains only (chemistry, biology, physics)"""
    return run_five_strategic_tests(num_samples, verbose, domain_filter='all_domains')


def run_defi_tests(num_samples: int = 200, verbose: bool = True):
    """Run tests for DeFi domain only"""
    return run_five_strategic_tests(num_samples, verbose, domain_filter='defi')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="5 Strategic Tests: LLM+PySR vs All Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python test_suite_comparative.py --samples 200 --verbose
  
  # Run only scientific domain tests (chemistry, biology, physics)
  python test_suite_comparative.py --samples 200 --verbose --domain all_domains
  
  # Run only DeFi tests
  python test_suite_comparative.py --samples 200 --verbose --domain defi
  
  # Run specific domain tests
  python test_suite_comparative.py --samples 200 --verbose --domain chemistry
        """
    )
    
    parser.add_argument('--samples', type=int, default=200, 
                        help='Number of data samples to generate (default: 200)')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose output')
    parser.add_argument('--domain', type=str,
                        default='all',
                        help='Filter tests by domain (default: all). '
                             'Special values: all, all_domains, defi. '
                             'Or any bare domain name from the protocol.')
    parser.add_argument('--benchmark', type=str,
                        choices=['feynman', 'srbench', 'both'],
                        default='feynman',
                        help='Which benchmark to load (default: feynman)')
    parser.add_argument('--series', type=str,
                        choices=['I', 'II', 'III', 'crossover'],
                        default=None,
                        help='Feynman series filter (default: all series)')

    args = parser.parse_args()

    if not PYSR_AVAILABLE:
        print("\n❌ ERROR: PySR not installed")
        print("Install with: pip install pysr")
        print("Then: python -m pysr install")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("LLM-GUIDED PySR COMPARATIVE TEST SUITE".center(80))
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Samples: {args.samples}")
    print(f"  Domain Filter: {args.domain}")
    print(f"  Verbose: {args.verbose}")
    print("="*80)
    
    run_five_strategic_tests(
        num_samples=args.samples,
        verbose=args.verbose,
        domain_filter=args.domain,
        benchmark=args.benchmark,
        series=args.series,
    )
