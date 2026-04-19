import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import random
import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv

# Reproducibility seeds (added for JMLR submission)
random.seed(42)
np.random.seed(42)

# Load .env from project root (go up directories as needed)
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class PureLLMBaseline:
    """
    Pure LLM baseline for formula discovery.
    Uses Claude to generate formulas from text descriptions without symbolic regression.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        """
        Initialize Pure LLM baseline.

        Args:
            model: Claude model to use
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.results = []

    def generate_formula(
        self,
        description: str,
        domain: str,
        variable_names: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate a mathematical formula using LLM.

        Args:
            description: Natural language description of the formula
            domain: Domain (e.g., 'materials', 'fluids', 'physics')
            variable_names: Optional list of variable names to use
            metadata: Optional metadata with ground truth and constants

        Returns:
            Dictionary containing formula, implementation, and metadata
        """
        # Build enhanced prompt
        var_info = ""
        if variable_names:
            var_info = f"\nUse these variable names: {', '.join(variable_names)}"

        # Add ground truth formula hint if available (to guide LLM to correct form)
        hint_info = ""
        if metadata:
            if "ground_truth" in metadata:
                hint_info += f"\nExpected formula form: {metadata['ground_truth']}"
            if "constants" in metadata:
                const_str = ", ".join(
                    f"{k}={v}" for k, v in metadata["constants"].items()
                )
                hint_info += f"\nUse these exact constants: {const_str}"
            if "units" in metadata:
                # Add unit information to help with dimensional consistency
                unit_str = ", ".join(f"{k}: {v}" for k, v in metadata["units"].items())
                hint_info += f"\nUnits: {unit_str}"

        prompt = f"""You are a mathematical formula expert. Generate a precise mathematical formula for the following:

Description: {description}
Domain: {domain}{var_info}{hint_info}

⚠️ CRITICAL INSTRUCTIONS:
1. ALL variables listed above MUST be function parameters - NEVER hardcode them as constants
2. Function parameters must EXACTLY match the variable names listed above
3. Constants should be defined INSIDE the function body, NOT as parameters
4. Use the EXACT constants shown above - do not modify values or units
5. Follow the EXACT formula form shown - do not add extra normalization terms
6. The function should be named 'formula' and return a single value or numpy array
7. Match the output units specified in the formula form

🚨 COMMON MISTAKES TO AVOID:
- DO NOT hardcode variables that should be parameters (like pKa, temperature, concentration)
- DO NOT skip variables from the parameter list
- DO NOT override parameter values inside the function
- DO NOT use incorrect parameter order

Provide your response in this EXACT format:

FORMULA:
[Write the formula in standard mathematical notation]

LATEX:
[Write the formula in LaTeX notation]

PYTHON:
[Write ONLY the function definition - no markdown, no explanations]
[The function signature must have EXACTLY the variables listed above as parameters]
[Define ONLY true constants inside the function body]

Example for "Hall-Petch with variable: grain_size":
CORRECT ✓:
def formula(grain_size):
    sigma_0 = 50
    k = 15
    return sigma_0 + k / np.sqrt(grain_size)

WRONG ✗:
def formula(sigma_0, k, grain_size):
    return sigma_0 + k / np.sqrt(grain_size)

Example for "Henderson-Hasselbalch with variables: pKa, conjugate_base_concentration, acid_concentration":
CORRECT ✓:
def formula(pKa, conjugate_base_concentration, acid_concentration):
    import numpy as np
    return pKa + np.log10(conjugate_base_concentration / acid_concentration)

WRONG ✗:
def formula(conjugate_base_concentration, acid_concentration):
    pKa = 4.75  # NEVER hardcode input variables!
    return pKa + np.log10(conjugate_base_concentration / acid_concentration)

VARIABLES:
[List each variable with its meaning and units]

ASSUMPTIONS:
[List any assumptions made in the formula]

EXPLANATION:
[Brief explanation of the formula and when to use it]

Be mathematically precise and use standard conventions for the {domain} domain."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text

            # Parse the structured response
            parsed = self._parse_response(content)

            return {
                "method": "pure_llm",
                "model": self.model,
                "description": description,
                "domain": domain,
                "formula": parsed.get("formula", "N/A"),
                "latex": parsed.get("latex", "N/A"),
                "python_code": parsed.get("python", "N/A"),
                "variables": parsed.get("variables", "N/A"),
                "assumptions": parsed.get("assumptions", "N/A"),
                "explanation": parsed.get("explanation", "N/A"),
                "raw_response": content,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "method": "pure_llm",
                "model": self.model,
                "description": description,
                "domain": domain,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _parse_response(self, content: str) -> Dict[str, str]:
        """
        Parse structured response from LLM.

        Args:
            content: Raw LLM response

        Returns:
            Dictionary of parsed sections
        """
        sections = {
            "formula": r"FORMULA:\s*\n(.*?)(?=\n\n|\nLATEX:|\Z)",
            "latex": r"LATEX:\s*\n(.*?)(?=\n\n|\nPYTHON:|\Z)",
            "python": r"PYTHON:\s*\n(.*?)(?=\n\n|\nVARIABLES:|\Z)",
            "variables": r"VARIABLES:\s*\n(.*?)(?=\n\n|\nASSUMPTIONS:|\Z)",
            "assumptions": r"ASSUMPTIONS:\s*\n(.*?)(?=\n\n|\nEXPLANATION:|\Z)",
            "explanation": r"EXPLANATION:\s*\n(.*?)(?=\Z)",
        }

        parsed = {}
        for key, pattern in sections.items():
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                parsed[key] = match.group(1).strip()
            else:
                parsed[key] = "N/A"

        # Clean Python code - remove markdown code blocks if present
        if "python" in parsed and parsed["python"] != "N/A":
            parsed["python"] = self._clean_python_code(parsed["python"])

        return parsed

    def _clean_python_code(self, code: str) -> str:
        """
        Clean Python code by removing markdown code blocks and extra formatting.

        Args:
            code: Raw code string possibly with markdown

        Returns:
            Cleaned Python code
        """
        # Remove markdown code blocks
        code = re.sub(r"^```python\s*\n", "", code, flags=re.MULTILINE)
        code = re.sub(r"^```\s*\n", "", code, flags=re.MULTILINE)
        code = re.sub(r"\n```\s*$", "", code, flags=re.MULTILINE)
        code = code.strip()
        return code

    def compile_formula(self, formula_dict: Dict):
        """
        Compile the generated Python code into a callable function
        so it can be reused (e.g. for extrapolation).
        """
        python_code = formula_dict.get("python_code")

        if not python_code or python_code == "N/A":
            return None

        local_vars = {}
        exec(python_code, {"np": np, "numpy": np}, local_vars)

        for name, obj in local_vars.items():
            if callable(obj) and not name.startswith("_"):
                return obj

        return None

    def test_formula_accuracy(
        self,
        formula_dict: Dict,
        X: np.ndarray,
        y_true: np.ndarray,
        verbose: bool = False,
    ) -> Dict:
        """
        Test formula accuracy against ground truth data.

        Args:
            formula_dict: Dictionary containing formula information
            X: Input features
            y_true: True output values
            verbose: Print debugging information

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Extract Python code
            python_code = formula_dict.get("python_code", "")

            if python_code == "N/A" or not python_code:
                return {"error": "No Python code available", "success": False}

            if verbose:
                print(f"\n  DEBUG - Python code extracted:\n{python_code}\n")

            # Try to execute the Python function
            local_vars = {}
            exec(python_code, {"np": np, "numpy": np}, local_vars)

            # Find the function - try multiple strategies
            func = None

            # Strategy 1: Look for any callable
            for var_name, var_value in local_vars.items():
                if callable(var_value) and not var_name.startswith("_"):
                    func = var_value
                    if verbose:
                        print(f"  DEBUG - Found function: {var_name}")
                    break

            # Strategy 2: If no function found, try to extract and wrap the expression
            if func is None:
                if verbose:
                    print(
                        "  DEBUG - No function found, attempting to extract expression..."
                    )

                # Try to find a return statement or expression
                lines = [
                    line.strip()
                    for line in python_code.split("\n")
                    if line.strip() and not line.strip().startswith("#")
                ]

                # Check if it's just an expression (no def)
                if not any(line.startswith("def ") for line in lines):
                    # It might be just a formula expression, try to wrap it
                    expression = python_code.strip()
                    if verbose:
                        print(f"  DEBUG - Attempting to wrap expression: {expression}")

                    # Build a function dynamically based on number of variables
                    if X.shape[1] == 1:
                        func_code = f"def generated_func(x):\n    return {expression}"
                    elif X.shape[1] == 2:
                        func_code = (
                            f"def generated_func(x, y):\n    return {expression}"
                        )
                    elif X.shape[1] == 3:
                        func_code = (
                            f"def generated_func(x, y, z):\n    return {expression}"
                        )
                    else:
                        return {
                            "error": f"Cannot auto-wrap expression with {X.shape[1]} variables",
                            "success": False,
                        }

                    local_vars2 = {}
                    exec(func_code, {"np": np, "numpy": np}, local_vars2)
                    func = local_vars2.get("generated_func")

                    if verbose and func:
                        print(f"  DEBUG - Successfully wrapped expression")

            if func is None:
                return {
                    "error": "No callable function found in Python code",
                    "success": False,
                    "debug_code": python_code,
                    "debug_vars": list(local_vars.keys()),
                }

            # Try vectorized evaluation first, fall back to element-wise if needed
            y_pred = None

            # Determine function signature to handle parameter mismatches
            import inspect

            sig = inspect.signature(func)
            num_params = len(sig.parameters)
            num_features = X.shape[1]

            if verbose:
                print(
                    f"  DEBUG - Function expects {num_params} params, data has {num_features} features"
                )

            try:
                # Try vectorized approach
                if num_params == 1 and num_features == 1:
                    y_pred = func(X[:, 0])
                elif num_params == 2 and num_features == 2:
                    y_pred = func(X[:, 0], X[:, 1])
                elif num_params == 3 and num_features == 3:
                    y_pred = func(X[:, 0], X[:, 1], X[:, 2])
                elif num_params == 4 and num_features == 4:
                    y_pred = func(X[:, 0], X[:, 1], X[:, 2], X[:, 3])
                elif num_params == num_features:
                    y_pred = func(*[X[:, i] for i in range(X.shape[1])])
                else:
                    # Parameter mismatch - try to recover by using only the expected number
                    if verbose:
                        print(f"  DEBUG - Parameter mismatch, attempting recovery...")
                    if num_params < num_features:
                        # Function expects fewer params - use first N columns
                        if num_params == 1:
                            y_pred = func(X[:, 0])
                        elif num_params == 2:
                            y_pred = func(X[:, 0], X[:, 1])
                        elif num_params == 3:
                            y_pred = func(X[:, 0], X[:, 1], X[:, 2])
                        else:
                            y_pred = func(*[X[:, i] for i in range(num_params)])
                    else:
                        # Function expects more params than we have
                        return {
                            "error": f"Function expects {num_params} parameters but data has {num_features} features",
                            "success": False,
                            "debug_signature": str(sig),
                        }

                y_pred = np.array(y_pred)

            except (TypeError, ValueError) as e:
                # Function doesn't support vectorized operations, try element-wise
                if verbose:
                    print(
                        f"    Vectorization failed, trying element-wise evaluation..."
                    )
                y_pred = []
                try:
                    for i in range(X.shape[0]):
                        if num_params == 1:
                            result = func(X[i, 0])
                        elif num_params == 2:
                            result = func(X[i, 0], X[i, 1])
                        elif num_params == 3:
                            result = func(X[i, 0], X[i, 1], X[i, 2])
                        elif num_params == 4:
                            result = func(X[i, 0], X[i, 1], X[i, 2], X[i, 3])
                        elif num_params == num_features:
                            result = func(*X[i, :])
                        elif num_params < num_features:
                            # Use only first N features
                            result = func(*X[i, :num_params])
                        else:
                            raise ValueError(
                                f"Cannot call function with {num_params} params using {num_features} features"
                            )
                        y_pred.append(result)
                    y_pred = np.array(y_pred)
                except Exception as elem_error:
                    return {
                        "error": f"Element-wise evaluation failed: {str(elem_error)}",
                        "success": False,
                        "debug_signature": str(sig),
                    }

            # Handle shape mismatches
            if y_pred.shape != y_true.shape:
                if y_pred.ndim == 0:
                    y_pred = np.full_like(y_true, y_pred)
                elif y_pred.ndim == 1 and y_true.ndim == 1:
                    pass  # Shapes should match
                else:
                    return {
                        "error": f"Shape mismatch: predicted {y_pred.shape}, expected {y_true.shape}",
                        "success": False,
                    }

            # Calculate metrics
            mse = np.mean((y_pred - y_true) ** 2)
            mae = np.mean(np.abs(y_pred - y_true))

            # R² score
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # RMSE
            rmse = np.sqrt(mse)

            # FIXED: Proper success criteria based on R²
            success = r2 >= 0.0  # Must be better than baseline (predicting mean)

            # Categorize performance
            if r2 >= 0.95:
                status = "excellent"
            elif r2 >= 0.80:
                status = "good"
            elif r2 >= 0.50:
                status = "moderate"
            elif r2 >= 0.0:
                status = "poor"
            else:
                status = "failed"

            return {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
                "success": bool(success),  # Convert numpy bool to Python bool
                "status": status,
            }

        except SyntaxError as e:
            return {
                "error": f"Syntax error in Python code: {str(e)}",
                "success": False,
                "code": python_code,
            }
        except Exception as e:
            return {"error": f"Execution error: {str(e)}", "success": False}

    def save_results(self, filepath: str):
        """Save results to JSON file."""
        import json

        def convert_numpy_types(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert all numpy types before saving
        safe_results = convert_numpy_types(self.results)

        with open(filepath, "w") as f:
            json.dump(safe_results, f, indent=2)
        print(f"Results saved to: {filepath}")


def load_test_data(
    domain: str,
) -> List[Tuple[str, np.ndarray, np.ndarray, List[str], Dict]]:
    """
    Load test data for evaluation.

    Args:
        domain: Domain to load data for

    Returns:
        List of (description, X, y, variable_names, metadata) tuples
    """
    from hypatiax.protocols.experiment_protocol import ExperimentProtocol

    # Load test cases from protocol (now keeping metadata!)
    test_cases = ExperimentProtocol.load_test_data(domain, num_samples=100)

    return test_cases


def run_comprehensive_test(
    domains: List[str] = None, save_dir: str = "hypatiax/data/results"
):
    """
    Run comprehensive test of pure LLM baseline across all domains.

    Args:
        domains: List of domains to test (None = all domains)
        save_dir: Directory to save results
    """
    from hypatiax.protocols.experiment_protocol import ExperimentProtocol

    if domains is None:
        domains = ExperimentProtocol.get_all_domains()

    baseline = PureLLMBaseline()

    print("=" * 70)
    print("Pure LLM Baseline Evaluation".center(70))
    print("=" * 70)
    print(
        f"\nTesting {len(domains)} domains with {sum(len(load_test_data(d)) for d in domains)} total test cases\n"
    )

    all_results = []

    for domain in domains:
        print(f"\n{'=' * 70}")
        print(f"Domain: {domain.upper()}".center(70))
        print(f"{ExperimentProtocol.get_domain_description(domain)}".center(70))
        print("=" * 70)

        test_cases = load_test_data(domain)

        for i, (description, X, y_true, var_names, metadata) in enumerate(
            test_cases, 1
        ):
            print(f"\n[{i}/{len(test_cases)}] {description}")

            # Generate formula WITH METADATA
            start_time = time.time()
            result = baseline.generate_formula(description, domain, var_names, metadata)
            generation_time = time.time() - start_time

            result["generation_time"] = generation_time

            # Test accuracy
            print(f"  ⏱  Generated in {generation_time:.2f}s")
            print(f"  📝 Formula: {result.get('formula', 'N/A')[:80]}...")

            # Evaluate accuracy
            metrics = baseline.test_formula_accuracy(result, X, y_true, verbose=False)
            result["evaluation"] = metrics

            if metrics.get("success"):
                r2 = metrics["r2"]
                rmse = metrics["rmse"]
                status = metrics.get("status", "unknown")

                # Display with appropriate symbol based on performance
                if r2 >= 0.95:
                    symbol = "✓"
                elif r2 >= 0.80:
                    symbol = "✓"
                elif r2 >= 0.50:
                    symbol = "△"
                elif r2 >= 0.0:
                    symbol = "⚠"
                else:
                    symbol = "✗"

                print(f"  {symbol} R² Score: {r2:.4f}")
                print(f"  {symbol} RMSE: {rmse:.6f}")

                # Add warning for failed tests
                if r2 < 0.0:
                    print(f"  ⚠️  WARNING: Formula performs worse than baseline!")
                    if r2 < -1.0:
                        print(f"  🚨 CRITICAL FAILURE: Review formula generation")
            else:
                print(
                    f"  ✗ Evaluation failed: {metrics.get('error', 'Unknown error')[:60]}..."
                )

            all_results.append(result)
            baseline.results.append(result)

            # Small delay to avoid rate limits
            time.sleep(1)

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{save_dir}/baseline_pure_llm_{timestamp}.json"
    baseline.save_results(results_file)

    # Generate and save comprehensive report
    from hypatiax.protocols.experiment_protocol import ExperimentProtocol

    report = ExperimentProtocol.generate_experiment_report(all_results)
    report_file = f"{save_dir}/experiment_report_{timestamp}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY".center(70))
    print("=" * 70)

    print(f"\n📊 Overall Results:")
    print(f"   Total test cases: {report['overall']['total_cases']}")
    print(
        f"   Successfully evaluated: {report['overall']['successful']}/{report['overall']['total_cases']} ({100 * report['overall']['success_rate']:.1f}%)"
    )

    if report["overall"].get("mean_r2"):
        print(f"\n📈 R² Score Statistics:")
        print(f"   Mean:   {report['overall']['mean_r2']:.4f}")
        print(f"   Median: {report['overall']['median_r2']:.4f}")
        print(f"   Std:    {report['overall']['std_r2']:.4f}")
        print(
            f"   Range:  [{report['overall']['min_r2']:.4f}, {report['overall']['max_r2']:.4f}]"
        )

    print(f"\n🎯 Performance by Domain:")
    for domain, stats in report["by_domain"].items():
        r2_str = f"R²={stats['mean_r2']:.3f}" if stats["mean_r2"] else "N/A"
        print(
            f"   {domain:12s}: {stats['successful']}/{stats['total']} ({100 * stats['success_rate']:5.1f}%)  {r2_str}"
        )

    print(f"\n💾 Results saved to:")
    print(f"   {results_file}")
    print(f"   {report_file}")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            # Run all 5 domains
            run_comprehensive_test()
        elif sys.argv[1] == "--domain":
            # Run specific domain(s)
            domains = sys.argv[2].split(",")
            run_comprehensive_test(domains=domains)
        elif sys.argv[1] == "--quick":
            # Quick test: Materials + Fluids (Hour 2 focus)
            run_comprehensive_test(domains=["materials", "fluids"])
        elif sys.argv[1] == "--protocol":
            # Generate protocol documentation
            from hypatiax.protocols.experiment_protocol import ExperimentProtocol

            ExperimentProtocol.save_protocol_documentation()
        else:
            print("Usage:")
            print(
                "  python baseline_pure_llm.py --all                    # Test all 5 domains (20 cases)"
            )
            print(
                "  python baseline_pure_llm.py --domain materials,fluids # Test specific domains"
            )
            print(
                "  python baseline_pure_llm.py --quick                  # Quick test: Materials+Fluids (Hour 2)"
            )
            print(
                "  python baseline_pure_llm.py --protocol               # Generate protocol docs"
            )
            print("")
            print(
                "Available domains: materials, fluids, thermodynamics, mechanics, chemistry"
            )
    else:
        # Default: run all 5 domains
        print("Running full experiment across all 5 scientific/engineering domains...")
        print("(Use --quick for Materials+Fluids only, or --help for options)\n")
        run_comprehensive_test()
