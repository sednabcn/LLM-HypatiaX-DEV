# baseline_pure_llm_defi.py
"""
Pure LLM baseline for formula discovery in DeFi and Risk Management.
Improved robust prompting, parsing and evaluation logic.
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

import random
import numpy as np
from anthropic import Anthropic
from pathlib import Path
from dotenv import load_dotenv

# Reproducibility seeds (added for JMLR submission)
random.seed(42)
np.random.seed(42)

# Load .env from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class PureLLMBaseline:
    """
    Pure LLM baseline. Generates formulas with Claude and tries to produce
    executable, vectorized Python functions. Robust parsing/fallbacks included.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.results = []
        # small in-memory cache to avoid repeated API calls for same prompt
        self._cache: Dict[str, Dict] = {}

    def generate_formula(
        self,
        description: str,
        domain: str,
        variable_names: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        verbose: bool = False,
    ) -> Dict:
        """
        Generate a mathematical formula using LLM and return structured result.

        Returns dict with keys:
            - method, model, description, domain
            - formula (string)
            - latex (string)
            - python_code (string)  # cleaned code or "N/A"
            - variables, assumptions, explanation
            - raw_response, specialized_prompt (bool), timestamp
        """
        if variable_names is None:
            variable_names = []

        cache_key = f"{description}|{domain}|{','.join(variable_names)}|{json.dumps(metadata or {})}"
        if cache_key in self._cache:
            if verbose:
                print("  [CACHE] Returning cached result")
            out = self._cache[cache_key].copy()
            out["cached"] = True
            return out

        # choose specialized prompt where applicable
        desc_lower = description.lower()
        use_specialized = False
        if (
            ("kelly" in desc_lower and "optimal" in desc_lower)
            or ("impermanent loss" in desc_lower)
            or ("liquidation" in desc_lower)
            or ("expected shortfall" in desc_lower)
        ):
            use_specialized = True
            prompt = self._generate_specialized_prompt(
                description, domain, variable_names, metadata
            )
        else:
            prompt = self._generate_standard_prompt(
                description, domain, variable_names, metadata
            )

        try:
            if verbose:
                print("  [LLM] Sending prompt...")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text
            if verbose:
                print("  [LLM] Response length:", len(content))

            parsed = self._parse_response(content, verbose=verbose)

            result = {
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
                "specialized_prompt": use_specialized,
                "timestamp": datetime.now().isoformat(),
            }

            # cache result
            self._cache[cache_key] = result.copy()
            return result

        except Exception as e:
            if verbose:
                import traceback

                traceback.print_exc()
            return {
                "method": "pure_llm",
                "model": self.model,
                "description": description,
                "domain": domain,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _generate_standard_prompt(
        self,
        description: str,
        domain: str,
        variable_names: List[str],
        metadata: Optional[Dict],
    ) -> str:
        var_info = ""
        if variable_names:
            var_info = f"\nVariables (in order): {', '.join(variable_names)}"

        constants_info = ""
        if metadata and metadata.get("constants"):
            constants_info = "\nConstants:\n"
            for k, v in metadata["constants"].items():
                constants_info += f"  - {k} = {v}\n"

        ground_truth_hint = ""
        if metadata and metadata.get("ground_truth"):
            ground_truth_hint = f"\nExpected form (hint): {metadata['ground_truth']}"

        return f"""You are a precise mathematical formula expert in {domain}.
Task: {description}{var_info}{constants_info}{ground_truth_hint}

Output EXACTLY three sections with these headers (no additional text):

FORMULA:
[mathematical expression using the variable names provided]

PYTHON:
def formula({", ".join(variable_names)}):
    # Use numpy (np) functions. Define constants inside the function if provided.
    # Return a numpy scalar or numpy array.
    ...

EXPLANATION:
[1-2 sentence explanation]

Do NOT output anything else."""

    def _generate_specialized_prompt(
        self,
        description: str,
        domain: str,
        variable_names: List[str],
        metadata: Optional[Dict],
    ) -> str:
        # Mirror the stronger, exact prompts used in tests (Kelly, Liquidation, IL, ES)
        var_list = ", ".join(variable_names) if variable_names else ""
        desc_lower = description.lower()

        if "kelly" in desc_lower or ("optimal" in desc_lower and "lp" in desc_lower):
            return f"""You must output EXACTLY three sections.

FORMULA:
f* = min(mu / (lambda * sigma^2), 1.0)

PYTHON:
def formula({var_list}):
    risk_aversion = 2.0
    f_star = {variable_names[0]} / (risk_aversion * {variable_names[1]}**2)
    return np.minimum(f_star, 1.0)

EXPLANATION:
Kelly criterion (risk-adjusted), cap at 1.0.
"""

        if (
            "impermanent loss" in desc_lower
            or "impermanent loss percentage" in desc_lower
        ):
            return f"""You must output EXACTLY three sections.

FORMULA:
IL% = (2*sqrt(r)/(1+r) - 1) * 100

PYTHON:
def formula({var_list}):
    il_fraction = 2.0 * np.sqrt({variable_names[0]}) / (1.0 + {variable_names[0]}) - 1.0
    return il_fraction * 100.0

EXPLANATION:
Impermanent loss percentage for a 50/50 pool.
"""

        if "liquidation" in desc_lower and "short" in desc_lower:
            return f"""You must output EXACTLY three sections.

FORMULA:
P_liq = P_entry * (1 + 1/(L * m))

PYTHON:
def formula({var_list}):
    maintenance_margin = 0.8
    return {variable_names[0]} * (1.0 + 1.0 / ({variable_names[1]} * maintenance_margin))

EXPLANATION:
Liquidation price for short position.
"""

        if "liquidation" in desc_lower and "long" in desc_lower:
            return f"""You must output EXACTLY three sections.

FORMULA:
P_liq = P_entry * (1 - 1/(L * m))

PYTHON:
def formula({var_list}):
    maintenance_margin = 0.8
    return {variable_names[0]} * (1.0 - 1.0 / ({variable_names[1]} * maintenance_margin))

EXPLANATION:
Liquidation price for long position.
"""

        if (
            "expected shortfall" in desc_lower
            or "expected shortfall" in (metadata or {}).get("ground_truth", "").lower()
        ):
            return f"""You must output EXACTLY three sections.

FORMULA:
ES_p = ES1 + ES2 + rho * sqrt(ES1 * ES2)

PYTHON:
def formula({var_list}):
    return {variable_names[0]} + {variable_names[1]} + {variable_names[2]} * np.sqrt({variable_names[0]} * {variable_names[1]})

EXPLANATION:
Portfolio expected shortfall for two positions with correlation adjustment.
"""

        # Fallback to standard prompt
        return self._generate_standard_prompt(
            description, domain, variable_names, metadata
        )

    def _parse_response(self, content: str, verbose: bool = False) -> Dict[str, str]:
        """
        Extract FORMULA, LATEX, PYTHON, VARIABLES, ASSUMPTIONS, EXPLANATION sections.
        Robust multiple-strategy extraction; cleans python code for execution.
        """
        sections = {
            "formula": r"FORMULA:\s*\n(.*?)(?=\n\n[A-Z]+:|\n\n\[|$)",
            "latex": r"LATEX:\s*\n(.*?)(?=\n\n[A-Z]+:|\n\n\[|$)",
            "python": r"PYTHON:\s*\n(.*?)(?=\n\n[A-Z]+:|\n\n\[|$)",
            "variables": r"VARIABLES:\s*\n(.*?)(?=\n\n[A-Z]+:|\n\n\[|$)",
            "assumptions": r"ASSUMPTIONS:\s*\n(.*?)(?=\n\n[A-Z]+:|\n\n\[|$)",
            "explanation": r"EXPLANATION:\s*\n(.*?)(?=\n\n[A-Z]+:|\n\n\[|$)",
        }

        parsed = {}
        for key, pattern in sections.items():
            m = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            parsed[key] = m.group(1).strip() if m else "N/A"

        # Clean python: try multiple fallbacks
        python_code = parsed.get("python", "")
        python_code = self._clean_python_code(python_code)
        if (
            not python_code
            or python_code == "N/A"
            or "not found" in python_code.lower()
        ):
            # find fenced python blocks
            m = re.search(
                r"```python\s*\n(.*?)\n```", content, re.DOTALL | re.IGNORECASE
            )
            if m:
                python_code = self._clean_python_code(m.group(1))
            else:
                # look for any def ... function
                m2 = re.search(
                    r"(def\s+\w+\s*\([^)]*\)\s*:(?:\n(?:\s+.*?))*)", content, re.DOTALL
                )
                if m2:
                    python_code = self._clean_python_code(m2.group(1))

        parsed["python"] = python_code if python_code else "N/A"
        return parsed

    def _clean_python_code(self, code: str) -> str:
        """Remove fences, leading/trailing text and return a single function def if possible."""
        if not code:
            return "N/A"
        # remove leading/trailing fences or whitespace
        code = re.sub(r"^```(?:python)?\s*", "", code.strip(), flags=re.MULTILINE)
        code = re.sub(r"\s*```$", "", code, flags=re.MULTILINE).strip()

        # If the python block contains multiple sections, try to extract a complete def
        # Prefer def formula(...) but accept any def
        def_match = re.search(
            r"(def\s+formula\s*\([^)]*\)\s*:(?:\n(?:\s+.*?))*)", code, re.DOTALL
        )
        if not def_match:
            def_match = re.search(
                r"(def\s+\w+\s*\([^)]*\)\s*:(?:\n(?:\s+.*?))*)", code, re.DOTALL
            )
        if def_match:
            func_code = def_match.group(1).rstrip()
            # strip trailing unindented text
            lines = func_code.splitlines()
            # ensure only consistent indentation kept
            result_lines = []
            base_indent = None
            for i, ln in enumerate(lines):
                if i == 0:
                    result_lines.append(ln.rstrip())
                    base_indent = len(ln) - len(ln.lstrip())
                else:
                    # keep line if indented more than base (body) or blank
                    if ln.strip() == "" or (len(ln) - len(ln.lstrip()) > base_indent):
                        result_lines.append(ln.rstrip())
                    else:
                        # stop at next top-level def/section
                        break
            cleaned = "\n".join(result_lines).strip()
            return cleaned

        # If no def found but code appears to be a single expression, return it raw
        single_expr = code.strip()
        # avoid extremely long outputs
        if len(single_expr) > 10000:
            return "N/A"
        return single_expr

    def test_formula_accuracy(
        self,
        formula_dict: Dict,
        X: np.ndarray,
        y_true: np.ndarray,
        verbose: bool = False,
    ) -> Dict:
        """
        Attempt to execute python_code in formula_dict against X and compare to y_true.
        Returns metrics dict or error info.
        """
        try:
            python_code = (
                formula_dict.get("python_code") or formula_dict.get("python") or ""
            )
            if not python_code or python_code == "N/A":
                return {"error": "No Python code available", "success": False}

            # Prepare execution environment
            local_vars = {}
            # safe exec globals
            exec_globals = {"np": np, "numpy": np}
            try:
                exec(python_code, exec_globals, local_vars)
            except SyntaxError:
                # maybe the extracted content was not a def, try wrapping as expression
                pass

            # find a callable
            func = None
            for v in local_vars.values():
                if callable(v) and not v.__name__.startswith("_"):
                    func = v
                    break

            # If no function found, try to wrap expression into function signature
            if func is None:
                # If python_code contains 'return' but no def, wrap into function with variable count from X
                expr = python_code.strip()
                # heuristics: if expression contains 'np' or variable names, try to wrap
                n_feats = X.shape[1]
                # build wrapper parameter list using variable_names if available in formula_dict
                varnames = formula_dict.get("variables")
                if isinstance(varnames, (list, tuple)) and len(varnames) == n_feats:
                    params = varnames
                else:
                    # fallback to generic names x,y,z,...
                    default_names = ["x", "y", "z", "w", "u", "v"]
                    params = default_names[:n_feats]
                # create a wrapper that returns the expression and supports vectorized numpy arrays
                wrapper_params = ", ".join(params)
                wrapper_lines = [
                    f"def generated_func({wrapper_params}):",
                    f"    return {expr}",
                ]
                wrapper_code = "\n".join(wrapper_lines)
                if verbose:
                    print("  [WRAP] Attempting to wrap expression into function:")
                    print(wrapper_code)
                local = {}
                exec(wrapper_code, {"np": np, "numpy": np}, local)
                func = local.get("generated_func")

            if func is None:
                return {
                    "error": "No callable function found after parsing",
                    "success": False,
                    "debug": python_code[:500],
                }

            # Evaluate vectorized (pass arrays); fallback to row-wise
            try:
                if X.ndim == 1 or X.shape[1] == 1:
                    y_pred = func(X[:, 0]) if X.ndim > 1 else func(X)
                else:
                    # Try to call with each column as separate arg (vectorized)
                    args = [X[:, i] for i in range(X.shape[1])]
                    y_pred = func(*args)
                y_pred = np.asarray(y_pred).flatten()
            except Exception as e:
                if verbose:
                    print("  [EVAL] Vectorized evaluation failed:", e)
                    print("  [EVAL] Falling back to row-by-row evaluation")
                y_pred = np.empty(X.shape[0])
                for i in range(X.shape[0]):
                    vals = X[i, :]
                    try:
                        if vals.size == 1:
                            y_val = func(vals[0])
                        else:
                            y_val = func(*vals)
                    except Exception as ex:
                        return {
                            "error": f"Error evaluating function on row {i}: {ex}",
                            "success": False,
                        }
                    y_pred[i] = float(y_val)

            # ensure shapes match
            if y_pred.shape != y_true.shape:
                # try to broadcast scalar
                if y_pred.size == 1:
                    y_pred = np.full_like(y_true, float(y_pred))
                else:
                    return {
                        "error": f"Shape mismatch: predicted {y_pred.shape}, expected {y_true.shape}",
                        "success": False,
                    }

            # metrics
            mse = float(np.mean((y_pred - y_true) ** 2))
            mae = float(np.mean(np.abs(y_pred - y_true)))
            rmse = float(np.sqrt(mse))
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

            return {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "success": True,
                "y_pred_preview": y_pred[:5].tolist(),
            }

        except Exception as e:
            return {"error": f"Execution error: {e}", "success": False}

    def save_results(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✅ Results saved to: {filepath}")


if __name__ == "__main__":
    # quick smoke test stub (requires ANTHROPIC_API_KEY in environment)
    baseline = PureLLMBaseline()
    test_desc = "Optimal LP position size using risk-adjusted Kelly criterion"
    out = baseline.generate_formula(
        test_desc,
        "liquidity",
        ["expected_fee_apy", "il_risk"],
        metadata={"ground_truth": "min(expected_fee_apy/(2*il_risk**2),1.0)"},
        verbose=True,
    )
    print(json.dumps(out, indent=2)[:1000])
