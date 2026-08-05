"""
hypatia.py — LLM warm-start prior for HypatiaX / exp3_nguyen12_hypatiax.py
===========================================================================
Provides ``get_llm_prior(eq, X_train, y_train, ...)`` which queries Claude
via the Anthropic API and returns a ranked list of candidate symbolic
expressions in PySR-compatible Python syntax.

The module is intentionally self-contained: no HypatiaX internal imports are
required so that exp3 can drop this file next to the script and import it
directly.

Usage
-----
    from hypatia import get_llm_prior

    exprs = get_llm_prior(eq_dict, X_train, y_train)
    # exprs -> ["x**3 + x**2 + x", "x**3 + x**2", ...]

API key resolution order
------------------------
1. ``api_key`` argument to ``get_llm_prior``
2. ``ANTHROPIC_API_KEY`` environment variable
3. ``.env`` file in the working directory (python-dotenv, optional)

Dependencies
------------
    pip install anthropic numpy scikit-learn
    pip install python-dotenv   # optional – for .env support
"""

from __future__ import annotations

import json
import os
import re
import time
import warnings
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Optional dotenv support
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed; rely on env var or explicit api_key arg

# ---------------------------------------------------------------------------
# Anthropic client (lazy init so the module imports even without the package)
# ---------------------------------------------------------------------------
_anthropic_module = None

def _get_anthropic():
    global _anthropic_module
    if _anthropic_module is None:
        try:
            import anthropic as _a
            _anthropic_module = _a
        except ImportError:
            raise ImportError(
                "anthropic package required for LLM warm-start.\n"
                "  pip install anthropic"
            )
    return _anthropic_module


# ===========================================================================
# Public API
# ===========================================================================

def get_llm_prior(
    eq: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    api_key: str | None = None,
    n_candidates: int = 5,
    temperature: float = 0.25,
    model: str = "claude-sonnet-5",
    max_tokens: int = 1024,
    timeout: float = 60.0,
    verbose: bool = True,
) -> list[str]:
    """Return a list of candidate expressions for ``eq`` ordered by LLM confidence.

    Parameters
    ----------
    eq:
        One entry from the ``NGUYEN`` list in exp3.  Must have keys:
        ``id``, ``vars``, ``formula_hint``, and optionally ``formula``.
    X_train:
        Training features, shape ``(n, n_vars)``.
    y_train:
        Training targets, shape ``(n,)``.
    api_key:
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    n_candidates:
        How many candidate expressions to request from the LLM.
    temperature:
        Sampling temperature (lower = more deterministic).
    model:
        Anthropic model string.
    max_tokens:
        Max completion tokens.
    timeout:
        Seconds to wait for the API call before raising.
    verbose:
        Print progress to stdout.

    Returns
    -------
    List[str]
        Python expressions using the variable names in ``eq["vars"]``,
        compatible with PySR's ``populations_init`` / expression seeding.
        Empty list on any error (caller falls back to cold PySR).
    """
    resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not resolved_key:
        warnings.warn(
            "ANTHROPIC_API_KEY not set — LLM warm-start disabled for "
            f"{eq['id']}. Set the variable or pass api_key=.",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    # ------------------------------------------------------------------
    # 1. Data-pattern summary (injected into prompt as context)
    # ------------------------------------------------------------------
    patterns = _summarise_patterns(X_train, y_train, eq["vars"])

    # ------------------------------------------------------------------
    # 2. Build prompt
    # ------------------------------------------------------------------
    prompt = _build_prompt(eq, patterns, n_candidates)

    # ------------------------------------------------------------------
    # 3. Call API
    # ------------------------------------------------------------------
    if verbose:
        print(f"  [LLM] querying {model} for {eq['id']} ({n_candidates} candidates)...")

    t0 = time.time()
    try:
        anthropic = _get_anthropic()
        client = anthropic.Anthropic(api_key=resolved_key)
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
        )
        raw = message.content[0].text
    except Exception as exc:
        warnings.warn(
            f"  [LLM] API call failed for {eq['id']}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # 4. Parse & validate expressions
    # ------------------------------------------------------------------
    candidates = _parse_response(raw, eq["vars"])

    if verbose:
        print(f"  [LLM] {len(candidates)} valid expression(s) in {elapsed:.1f}s")
        for rank, expr in enumerate(candidates, 1):
            print(f"    {rank}. {expr}")

    return candidates


# ===========================================================================
# Internal helpers
# ===========================================================================

def _summarise_patterns(
    X: np.ndarray,
    y: np.ndarray,
    var_names: list[str],
) -> dict[str, Any]:
    """Compute lightweight statistics that fit in a short prompt section."""
    summary: dict[str, Any] = {}

    # Per-variable correlations with target
    corrs = {}
    for i, vn in enumerate(var_names):
        xi = X[:, i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = float(np.corrcoef(xi, y)[0, 1])
        corrs[vn] = round(c, 3) if np.isfinite(c) else 0.0
    summary["linear_correlations"] = corrs

    # y range and scale
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    summary["y_range"] = [round(y_min, 4), round(y_max, 4)]
    summary["y_scale"] = (
        "log-scale (always positive)" if y_min > 0 and y_max / max(y_min, 1e-9) > 100
        else "mixed/linear"
    )

    # Linearity test (R² of OLS)
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        lr = LinearRegression().fit(X, y)
        r2 = r2_score(y, lr.predict(X))
        summary["linear_r2"] = round(float(r2), 4)
        # r2 is numpy.float64; comparison yields numpy.bool_ which is NOT
        # JSON-serialisable — must cast to Python bool explicitly.
        summary["is_linear"] = bool(r2 > 0.97)
    except Exception:
        summary["linear_r2"] = None
        summary["is_linear"] = False

    # Periodicity hint via FFT entropy
    try:
        from scipy.fft import fft
        fft_mag = np.abs(fft(y - np.mean(y)))[1: len(y) // 2]
        if fft_mag.sum() > 0:
            p = fft_mag / fft_mag.sum()
            entropy = -np.sum(p * np.log(p + 1e-15))
            summary["fft_entropy"] = round(float(entropy), 3)
            # entropy is numpy.float64 (result of np.sum); same numpy.bool_
            # JSON-serialisation trap as is_linear above.
            summary["periodic_hint"] = bool(entropy < 3.0)  # low entropy → peaked spectrum
        else:
            summary["periodic_hint"] = False
    except Exception:
        summary["periodic_hint"] = False

    return summary


def _build_prompt(
    eq: dict[str, Any],
    patterns: dict[str, Any],
    n_candidates: int,
) -> str:
    var_names = eq["vars"]
    var_list = ", ".join(var_names)
    hint = eq.get("formula_hint", "unknown structure")
    eq_id = eq["id"]

    patterns_json = json.dumps(patterns, indent=2)

    return f"""You are an expert symbolic regression assistant embedded in the HypatiaX system.

TASK
----
Generate {n_candidates} candidate symbolic expressions for the Nguyen benchmark equation
{eq_id}. The expressions will seed PySR's evolutionary search (warm-start), so they
must be in Python syntax compatible with numpy operations.

EQUATION CONTEXT
----------------
- Equation ID : {eq_id}
- Variables   : {var_list}
- Formula hint: {hint}

DATA PATTERNS (measured on training split)
------------------------------------------
{patterns_json}

OUTPUT FORMAT
-------------
Return ONLY a JSON array of objects, ranked from highest to lowest confidence.
Each object must have:
  "expr"       : the expression as a valid Python string
  "confidence" : float in [0, 1]
  "note"       : one short phrase explaining the form

Rules for "expr":
- Use ONLY these variable names: {var_list}
- Use Python operators: + - * / ** (NOT ^)
- Allowed functions (prefix with "np."): np.sin, np.cos, np.log, np.exp, np.sqrt, np.abs
- DO NOT include "y =" or any assignment; bare expression only
- DO NOT use any variable not in the list above
- Keep expressions concise (complexity ≤ 20 nodes)

Example output for a hypothetical problem with variables [x]:
[
  {{"expr": "x**3 + x**2 + x", "confidence": 0.90, "note": "degree-3 polynomial"}},
  {{"expr": "x**3 + x**2",     "confidence": 0.70, "note": "degree-3 without linear"}},
  {{"expr": "np.sin(x) + x",   "confidence": 0.20, "note": "trigonometric alternative"}}
]

JSON array (no preamble, no markdown fences):"""


def _parse_response(raw: str, var_names: list[str]) -> list[str]:
    """Extract and validate expressions from the LLM response.

    Returns a list of syntactically valid, variable-safe expression strings.
    Malformed or unsafe entries are silently dropped.
    """
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    # Locate the JSON array
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        return []

    try:
        candidates = json.loads(text[start:end])
    except json.JSONDecodeError:
        return []

    if not isinstance(candidates, list):
        return []

    # Sort by confidence descending
    candidates.sort(key=lambda c: float(c.get("confidence", 0)), reverse=True)

    valid_exprs: list[str] = []
    allowed_names = set(var_names) | {
        "np", "sin", "cos", "log", "exp", "sqrt", "abs",
        "True", "False", "None",
    }

    for item in candidates:
        if not isinstance(item, dict):
            continue
        expr = item.get("expr", "").strip()
        if not expr:
            continue

        # Safety: reject expressions referencing undeclared names
        try:
            import ast
            tree = ast.parse(expr, mode="eval")
            names_used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            # Allow numpy attribute access (np.sin etc.) — handled at eval time
            unexpected = names_used - allowed_names
            if unexpected:
                continue  # references an undefined symbol
        except SyntaxError:
            continue

        # Smoke-test: evaluate with dummy arrays to catch runtime errors
        try:
            dummy = {vn: np.ones(5) for vn in var_names}
            dummy["np"] = np
            eval(compile(ast.parse(expr, mode="eval"), "<expr>", "eval"), dummy)
        except Exception:
            continue

        valid_exprs.append(expr)

    return valid_exprs

if __name__ == "__main__":
    pass  # TODO: add entry point
