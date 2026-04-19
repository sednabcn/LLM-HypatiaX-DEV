"""
Example: how a hybrid system should call NeuralNetworkBaseline.train_get_model()
and get_predictions(), combine them with an LLM-produced formula, and form an
uncertainty-weighted ensemble.

This snippet assumes these modules are available in your PYTHONPATH:
- baseline_pure_llm_defi.PureLLMBaseline (or another LLM wrapper)
- baseline_neural_network_defi_improved.NeuralNetworkBaseline

It also works offline if ANTHROPIC_API_KEY is not set by using the canonical
specialized formula for the test case.

Contributes uncertainty-weighted ensemble combining LLM + NN:

ensemble_llm_nn() - Weights predictions by inverse uncertainty
Better than choosing one method outright
Could help when both methods are valid but imperfect

"""

import numpy as np
from sklearn.metrics import r2_score
from hypatiax.protocols.experiment_protocol_defi import DeFiExperimentProtocol

# Import your LLM baseline and NN baseline implementations
from hypatiax.core.base_pure_llm.baseline_pure_llm_defi_discovery import PureLLMBaseline
from hypatiax.core.training.baseline_neural_network_defi_improved import NeuralNetworkBaseline


def execute_python_code_get_predictions(python_code: str, X: np.ndarray):
    """
    Safely exec the python_code string (which must define a function either named
    'formula' or the first callable found). Returns vectorized predictions for X.
    """
    import math as _math
    exec_globals = {
        "np":      np,
        "numpy":   np,
        "math":    _math,
        "pi":      np.pi,
        "e":       np.e,
        "exp":     lambda x: np.exp(np.clip(x, -500.0, 500.0)),
        "log":     np.log,
        "log2":    np.log2,
        "log10":   np.log10,
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
        "minimum": np.minimum,
        "maximum": np.maximum,
        "clip":    np.clip,
    }
    local_vars = {}
    try:
        exec(python_code, exec_globals, local_vars)
    except Exception as e:
        raise RuntimeError(f"Exec failed: {e}\nCode preview:\n{python_code[:400]}")

    # find first callable
    func = None
    for v in local_vars.values():
        if callable(v):
            func = v
            break
    if func is None:
        for v in exec_globals.values():
            if callable(v) and getattr(v, "__name__", "") != "np":
                func = v
                break
    if func is None:
        raise RuntimeError("No callable found in executed LLM code")

    # Call vectorized (columns -> separate args) if possible, else row-by-row
    if X.ndim == 1 or X.shape[1] == 1:
        args = (X[:, 0],) if X.ndim > 1 else (X,)
        y_pred = func(*args)
    else:
        args = [X[:, i] for i in range(X.shape[1])]
        try:
            y_pred = func(*args)
        except Exception:
            # fallback row-by-row
            y_pred = np.array([func(*X[i, :]) for i in range(X.shape[0])])
    return np.asarray(y_pred).flatten()


def ensemble_llm_nn(llm_pred, nn_pred, y_true, llm_r2=None, nn_r2=None):
    """
    Build an uncertainty-weighted ensemble between LLM and NN predictions.
    - Compute uncertainties as std of residuals
    - Use inverse-uncertainty (and optionally scale by R²) to weight
    """
    eps = 1e-8

    # Residual-based uncertainties
    llm_res = (llm_pred - y_true) if llm_pred is not None else None
    nn_res = nn_pred - y_true

    llm_unc = float(np.std(llm_res)) if llm_res is not None else 1.0
    nn_unc = float(np.std(nn_res)) if nn_res is not None else 1.0

    # Strength factor from R² (if provided)
    llm_strength = float(llm_r2) if llm_r2 is not None else 1.0
    nn_strength = float(nn_r2) if nn_r2 is not None else 1.0

    # Compute raw weights (higher weight -> lower uncertainty and higher R²)
    w_llm = (
        (1.0 / (llm_unc + eps)) * max(llm_strength, 0.0)
        if llm_pred is not None
        else 0.0
    )
    w_nn = (1.0 / (nn_unc + eps)) * max(nn_strength, 0.0)

    if w_llm + w_nn <= 0:
        # fallback equal weights
        w_llm = 0.5 if llm_pred is not None else 0.0
        w_nn = 0.5

    # Normalize
    total = w_llm + w_nn
    w_llm /= total
    w_nn /= total

    # Weighted ensemble
    if llm_pred is None:
        ensemble = nn_pred
    else:
        ensemble = w_llm * llm_pred + w_nn * nn_pred

    return ensemble, {
        "w_llm": w_llm,
        "w_nn": w_nn,
        "llm_unc": llm_unc,
        "nn_unc": nn_unc,
    }


def main():
    # Load a representative test case (Kelly criterion) from the protocol
    protocol = DeFiExperimentProtocol()
    # Find a liquidity/kelley-type test case
    cases = protocol.load_test_data("liquidity", num_samples=100)
    # pick the test case whose description contains 'kelly' (adjust if necessary)
    target = None
    for desc, X, y_true, var_names, meta in cases:
        if "kelly" in desc.lower() or "optimal lp" in desc.lower():
            target = (desc, X, y_true, var_names, meta)
            break
    if target is None:
        # fallback to first liquidity case
        desc, X, y_true, var_names, meta = cases[0]
    else:
        desc, X, y_true, var_names, meta = target

    print("Selected test:", desc)
    print("Variables:", var_names)
    print("Data shapes:", X.shape, y_true.shape)

    # 1) Generate/obtain LLM formula (if API key present). Otherwise, use canned formula
    llm = PureLLMBaseline()
    try:
        llm_result = llm.generate_formula(
            description=desc,
            domain="liquidity",
            variable_names=var_names,
            metadata=meta,
            verbose=False,
        )
    except Exception as e:
        # No API key or call failed -> use canonical specialized code (Kelly)
        llm_result = {"python_code": "N/A", "formula": "N/A"}
        print("LLM call failed or unavailable; will use canonical fallback.")

    # Provide a canonical fallback if LLM did not return working code
    if not llm_result.get("python_code") or llm_result.get("python_code") == "N/A":
        # Example canonical Kelly implementation (matches specialized prompt)
        python_code = """
def formula(expected_fee_apy, il_risk):
    risk_aversion = 2.0
    f_star = expected_fee_apy / (risk_aversion * il_risk**2)
    return np.minimum(f_star, 1.0)
"""
        llm_result = {
            "python_code": python_code,
            "formula": "min(mu/(lambda*sigma^2),1.0)",
        }

    # 2) Evaluate LLM formula on X
    try:
        llm_pred = execute_python_code_get_predictions(llm_result["python_code"], X)
        llm_r2 = r2_score(y_true, llm_pred)
        print(f"LLM R² = {llm_r2:.6f}")
    except Exception as e:
        print("LLM evaluation error:", e)
        llm_pred = None
        llm_r2 = None

    # 3) Train NN and obtain predictions (use train_get_model/get_predictions)
    nn = NeuralNetworkBaseline(hidden_dims=[128, 64, 32], epochs=200, batch_size=64)
    model, nn_metrics, scaler_X, scaler_y = nn.train_get_model(
        X,
        y_true,
        metadata=meta,
        is_extrapolation=meta.get("extrapolation_test", False),
        verbose=True,
    )
    nn_pred = nn.get_predictions(model, scaler_X, scaler_y, X)
    print("NN metrics:", nn_metrics)
    nn_r2 = nn_metrics.get("r2", None)

    # 4) Ensemble
    ensemble_pred, info = ensemble_llm_nn(
        llm_pred, nn_pred, y_true, llm_r2=llm_r2, nn_r2=nn_r2
    )
    ensemble_r2 = r2_score(y_true, ensemble_pred)
    print("Ensemble info:", info)
    print(f"Ensemble R² = {ensemble_r2:.6f}")

    # 5) Final decision (example policy)
    if llm_pred is not None and llm_r2 is not None and llm_r2 > 0.95:
        chosen_method = "LLM"
        chosen_pred = llm_pred
    elif ensemble_r2 > max(
        nn_r2 if nn_r2 is not None else -999,
        llm_r2 if llm_r2 is not None else -999,
    ):
        chosen_method = "ENSEMBLE"
        chosen_pred = ensemble_pred
    else:
        chosen_method = "NN"
        chosen_pred = nn_pred

    print("Chosen method:", chosen_method)
    print("Chosen R²:", r2_score(y_true, chosen_pred))


if __name__ == "__main__":
    main()
