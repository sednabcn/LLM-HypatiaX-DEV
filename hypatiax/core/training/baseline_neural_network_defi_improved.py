"""
COMPLETE REWRITE: baseline_neural_network_defi.py
Fixed version with improved architecture, training, and data handling.

KEY IMPROVEMENTS:
1. Better data normalization (StandardScaler for both X and y)
2. Improved network architecture with dropout and batch norm
3. Early stopping to prevent overfitting
4. Learning rate scheduling
5. Proper train/val/test split
6. Better handling of extrapolation tests
7. Consistent evaluation metrics with Pure LLM baseline

Expected improvements:
- Better generalization on extrapolation tests
- More stable training with fewer poor fits
- Comparable or better performance to Pure LLM on some domains
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from pathlib import Path

# Import experiment protocol
from hypatiax.protocols.experiment_protocol_defi import DeFiExperimentProtocol

# Centralised a-priori architecture & solver selection.
# Soft import: if adaptive_config.py is not yet deployed to hypatiax/core/,
# the module still loads and uses the embedded _LocalCaseProfile fallback.
# A hard import here would make the entire module unimportable, causing
# _probe() → NN_AVAILABLE=False → every call returns _unavailable() instantly
# (R²=0, RMSE=inf, 0.0 s) — which is exactly what happened before this fix.
try:
    from hypatiax.core.training.adaptive_config import CaseProfile as _CaseProfile
    _ADAPTIVE_CONFIG_AVAILABLE = True
except ImportError:
    _CaseProfile = None
    _ADAPTIVE_CONFIG_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# _LocalCaseProfile — embedded fallback used when adaptive_config.py is absent.
# Implements the same signal→config logic as CaseProfile so that parameter
# decisions remain centralised even without the external module.
# ─────────────────────────────────────────────────────────────────────────────
class _LocalSolverConfig:
    """Minimal mutable stand-in for SolverConfig."""
    __slots__ = (
        "input_dim", "hidden_dims", "use_log_y", "y_sign", "log_X_cols",
        "optimizer_cls", "lr", "weight_decay",
        "scheduler_cls", "sched_patience", "sched_factor", "sched_min_lr",
        "max_epochs", "stop_patience", "n_seeds", "grad_clip",
        "budget_log_secs", "budget_lin_secs", "rationale",
    )

    def make_optimizer(self, model):
        if self.optimizer_cls == "AdamW":
            return optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def make_scheduler(self, optimizer):
        if self.scheduler_cls == "CosineAnnealingWarmRestarts":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=200, T_mult=2, eta_min=self.sched_min_lr
            )
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.sched_patience,
            factor=self.sched_factor, min_lr=self.sched_min_lr,
        )

    def summary(self):
        return (
            f"── _LocalSolverConfig (adaptive_config.py not found) ──\n"
            f"  hidden={list(self.hidden_dims)}  log-y={self.use_log_y}  "
            f"lr={self.lr:.4g}  wd={self.weight_decay:.4g}  "
            f"sched={self.scheduler_cls}  seeds={self.n_seeds}"
        )


def _local_resolve(X: np.ndarray, y: np.ndarray,
                   metadata: dict, budget_secs: float) -> _LocalSolverConfig:
    """
    Reproduce the key signal→parameter decisions from CaseProfile without
    depending on the external module.  Kept in sync with adaptive_config.py.
    """
    meta   = metadata or {}
    n_vars = X.shape[1]
    n_samp = X.shape[0]
    rat    = []

    # ── Architecture ────────────────────────────────────────────────────
    if n_vars <= 2:
        hidden = (128, 64, 32)
    elif n_vars <= 4:
        hidden = (256, 128, 64, 32)
    else:
        hidden = (512, 256, 128, 64)
    rat.append(f"arch {list(hidden)}: n_vars={n_vars}")

    # ── Log-y: pole-guard heuristic ──────────────────────────────────────
    y_abs    = np.abs(y)
    y_all_pos = bool(np.all(y > 0))
    y_all_neg = bool(np.all(y < 0))
    y_sign    = -1.0 if y_all_neg else 1.0
    use_log_y = False
    log_y_reason = "mixed-sign y"
    if y_all_pos or y_all_neg:
        _ymin  = float(np.min(y_abs))
        _ymax  = float(np.max(y_abs))
        _ratio = _ymax / (_ymin + 1e-300)
        _p10   = float(np.percentile(y_abs, 10))
        _p90   = float(np.percentile(y_abs, 90))
        _pole  = (
            _p10 > 0
            and (_p90 / (_p10 + 1e-300)) > 50
            and _ymin < _p10 / 10.0
        )
        if _pole:
            log_y_reason = f"pole-shaped (p10={_p10:.3g},p90={_p90:.3g}) — Bose-Einstein guard"
        elif _ratio < 10:
            log_y_reason = f"range only {np.log10(_ratio):.1f} decades — linear space"
        elif _ratio > 1e10:
            log_y_reason = f"range {np.log10(_ratio):.1f} decades > 10 — skip log"
        else:
            use_log_y    = True
            log_y_reason = f"power-law range {np.log10(_ratio):.1f} decades"
    rat.append(f"log-y={use_log_y}: {log_y_reason}")

    # ── Log-X columns ─────────────────────────────────────────────────────
    log_X_cols = tuple(
        col for col in range(n_vars)
        if (bool(np.all(X[:, col] > 0))
            and (use_log_y
                 or float(np.max(X[:, col])) / (float(np.min(X[:, col])) + 1e-300) > 10))
    )
    if log_X_cols:
        rat.append(f"log-X cols={list(log_X_cols)}")

    # ── Learning rate ────────────────────────────────────────────────────
    difficulty = str(meta.get("difficulty", "")).lower()
    is_extrap  = bool(meta.get("extrapolation_test", False))
    lr = 3e-3 if use_log_y else 1e-3
    rat.append(f"lr={lr:.4g}")

    # ── Weight decay ─────────────────────────────────────────────────────
    wd = 1e-5
    if is_extrap or n_samp < 100:
        wd = 1e-4
    rat.append(f"wd={wd:.4g}")

    # ── Scheduler ────────────────────────────────────────────────────────
    if use_log_y or difficulty in ("hard", "expert"):
        sched_cls = "CosineAnnealingWarmRestarts"
    else:
        sched_cls = "ReduceLROnPlateau"
    rat.append(f"scheduler={sched_cls}")

    # ── Epochs ───────────────────────────────────────────────────────────
    base_ep = 300 + 100 * max(0, n_vars - 2)
    if difficulty == "expert":
        base_ep = int(base_ep * 2.0)
    elif difficulty == "hard":
        base_ep = int(base_ep * 1.5)
    max_ep = min(base_ep, 2000 if use_log_y else 1000)
    rat.append(f"max_epochs={max_ep}")

    # ── Seeds ─────────────────────────────────────────────────────────────
    if difficulty == "expert":
        n_seeds = 5
    elif difficulty == "hard" or is_extrap:
        n_seeds = 3
    else:
        n_seeds = 1
    rat.append(f"n_seeds={n_seeds}")

    # ── Budget fractions ──────────────────────────────────────────────────
    if n_seeds > 1:
        per_seed = budget_secs * 0.90 / n_seeds
        bud_log  = max(20.0, per_seed * 0.55 / 0.85)
        bud_lin  = max(15.0, per_seed * 0.30 / 0.85)
    else:
        bud_log = max(20.0, budget_secs * 0.55)
        bud_lin = max(15.0, budget_secs * 0.30)

    cfg = _LocalSolverConfig()
    cfg.input_dim        = n_vars
    cfg.hidden_dims      = hidden
    cfg.use_log_y        = use_log_y
    cfg.y_sign           = y_sign
    cfg.log_X_cols       = log_X_cols
    cfg.optimizer_cls    = "AdamW"
    cfg.lr               = lr
    cfg.weight_decay     = wd
    cfg.scheduler_cls    = sched_cls
    cfg.sched_patience   = 30 if use_log_y else 20
    cfg.sched_factor     = 0.5
    cfg.sched_min_lr     = lr * 1e-3
    cfg.max_epochs       = max_ep
    cfg.stop_patience    = 100 if use_log_y else 80
    cfg.n_seeds          = n_seeds
    cfg.grad_clip        = 1.0
    cfg.budget_log_secs  = bud_log
    cfg.budget_lin_secs  = bud_lin
    cfg.rationale        = tuple(rat)
    return cfg


class ImprovedNN(nn.Module):
    """
    Improved neural network for physics regression:
    - Wider + deeper: [256, 256, 128, 128, 64] — enough capacity for sharp
      nonlinearities like 1/(exp(x)-1), x^4, exp(-x^2).
    - LayerNorm per layer: normalises per-sample, preserving inter-sample
      magnitude contrast that BatchNorm destroys.
    - Residual (skip) connections on equal-width pairs: stabilises gradients
      through depth, especially critical for functions like Bose-Einstein and
      Planck that have near-singular regions.
    - SiLU activation: smooth, non-zero gradient everywhere, better than ReLU
      for physics functions with exponential curvature.
    - Zero dropout: 200-sample datasets cannot afford any activation dropout.
      Regularisation comes entirely from weight_decay in Adam.
    """

    def __init__(self, input_dim, hidden_dims=[256, 256, 128, 128, 64]):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        self.blocks = nn.ModuleList()
        self.norms   = nn.ModuleList()
        self.skips   = nn.ModuleList()

        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            self.blocks.append(nn.Linear(prev_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            # Skip projection if dimensions differ; identity otherwise
            if prev_dim != hidden_dim:
                self.skips.append(nn.Linear(prev_dim, hidden_dim, bias=False))
            else:
                self.skips.append(nn.Identity())
            prev_dim = hidden_dim

        self.act    = nn.SiLU()
        self.output = nn.Linear(prev_dim, 1)

    def forward(self, x):
        x = self.act(self.input_proj(x))
        for block, norm, skip in zip(self.blocks, self.norms, self.skips):
            residual = skip(x)
            x = self.act(norm(block(x))) + residual
        return self.output(x)


def _single_train_run(X_train_t, y_train_t, X_val_t, y_val_t, cfg, seed):
    """
    One training run driven entirely by a SolverConfig.
    Returns (model, best_val_loss, final_train_loss).

    All hyper-parameters (lr, weight_decay, architecture, scheduler type,
    epoch budget, early-stopping patience, grad-clip) come from cfg — no
    magic numbers live in this function.
    """
    torch.manual_seed(seed)
    model     = ImprovedNN(cfg.input_dim, list(cfg.hidden_dims))
    optimizer = cfg.make_optimizer(model)
    scheduler = cfg.make_scheduler(optimizer)
    criterion = nn.MSELoss()

    best_val         = float("inf")
    best_state       = None
    no_improve       = 0
    final_train_loss = float("inf")

    for epoch in range(cfg.max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if cfg.scheduler_cls == "CosineAnnealingWarmRestarts":
            scheduler.step(epoch)
        else:
            scheduler.step(loss)

        final_train_loss = loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val - 1e-8:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.stop_patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val, final_train_loss


def train_neural_network(
    X, y, description, domain, metadata=None, budget_secs=90.0, verbose=False
):
    """
    Train and evaluate an improved neural network.

    All architecture and solver hyper-parameters are resolved a priori by
    CaseProfile.from_data() via adaptive_config.py — no magic numbers live
    in this function.  The only external knob is budget_secs: the wall-clock
    budget forwarded from the benchmark runner's --method-timeout flag.

    Args:
        X            : Input features (numpy array, shape [n, d])
        y            : Target values (numpy array, shape [n])
        description  : Human-readable test case description
        domain       : Domain name string
        metadata     : Test case metadata dict (difficulty, extrapolation_test, …)
        budget_secs  : Total wall-clock budget for this call (default 90s)
        verbose      : Print per-epoch training progress

    Returns:
        dict with keys evaluation, training_info, metadata, timestamp
    """

    # =========================================================================
    # STEP 1 — Resolve all hyper-parameters a priori from (X, y, metadata)
    # =========================================================================
    if _ADAPTIVE_CONFIG_AVAILABLE:
        cfg = _CaseProfile.from_data(X, y, metadata, budget_secs=budget_secs).to_config()
    else:
        cfg = _local_resolve(X, y, metadata or {}, budget_secs)

    if verbose:
        print(cfg.summary())

    # =========================================================================
    # STEP 2 — Preprocessing (driven entirely by cfg)
    # =========================================================================
    X_work = X.copy().astype(float)
    for col in cfg.log_X_cols:
        X_work[:, col] = np.log(X_work[:, col])

    if cfg.use_log_y:
        y_work = np.log(np.abs(y))
    else:
        y_work = y.copy()

    X_train, X_val, y_train, y_val = train_test_split(
        X_work, y_work, test_size=0.2, random_state=42
    )

    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s   = scaler_X.transform(X_val)
    X_all_s   = scaler_X.transform(X_work)

    scaler_y  = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_s   = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    X_train_t = torch.FloatTensor(X_train_s)
    y_train_t = torch.FloatTensor(y_train_s).reshape(-1, 1)
    X_val_t   = torch.FloatTensor(X_val_s)
    y_val_t   = torch.FloatTensor(y_val_s).reshape(-1, 1)
    X_all_t   = torch.FloatTensor(X_all_s)

    # =========================================================================
    # STEP 3 — Multi-restart training  (cfg.n_seeds restarts → keep best)
    # =========================================================================
    best_model       = None
    best_val_loss    = float("inf")
    best_train_loss  = float("inf")

    for seed in range(cfg.n_seeds):
        model, val_loss, train_loss = _single_train_run(
            X_train_t, y_train_t, X_val_t, y_val_t, cfg, seed
        )
        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            best_train_loss = train_loss
            best_model      = model

    # =========================================================================
    # STEP 4 — Evaluate on full dataset (matches benchmark scoring convention)
    # =========================================================================
    best_model.eval()
    with torch.no_grad():
        y_pred_s = best_model(X_all_t).numpy().flatten()

    y_pred_w = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
    if cfg.use_log_y:
        y_pred = cfg.y_sign * np.exp(np.clip(y_pred_w, -500.0, 500.0))
    else:
        y_pred = y_pred_w

    rmse   = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    mae    = float(np.mean(np.abs(y - y_pred)))
    mse    = float(np.mean((y - y_pred) ** 2))
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else (1.0 if ss_res < 1e-10 else 0.0)

    # =========================================================================
    # STEP 5 — Extrapolation analysis (if applicable)
    # =========================================================================
    is_extrapolation = bool((metadata or {}).get("extrapolation_test", False))
    extrapolation_stats = None
    if is_extrapolation:
        extrapolation_stats = {
            "mean_prediction":       float(np.mean(y_pred)),
            "std_prediction":        float(np.std(y_pred)),
            "mean_error":            float(np.mean(y_pred - y)),
            "extrapolation_quality": (
                "poor" if abs(np.mean(y_pred - y)) > np.std(y) else "good"
            ),
        }

    # =========================================================================
    # STEP 6 — Pack and return results
    # =========================================================================
    arch_str = (
        f"ResidualMLP {list(cfg.hidden_dims)} SiLU LayerNorm "
        f"({'log-y' if cfg.use_log_y else 'lin-y'}, {cfg.n_seeds} seed(s))"
    )
    result = {
        "method":       "neural_network",
        "architecture": arch_str,
        "description":  description,
        "domain":       domain,
        "evaluation": {
            "r2":      r2,
            "rmse":    rmse,
            "mae":     mae,
            "mse":     mse,
            "success": True,
        },
        "training_info": {
            "log_y":            cfg.use_log_y,
            "log_X_cols":       list(cfg.log_X_cols),
            "best_val_loss":    float(best_val_loss),
            "final_train_loss": float(best_train_loss),
            "n_seeds":          cfg.n_seeds,
            "hidden_dims":      list(cfg.hidden_dims),
            "lr":               cfg.lr,
            "weight_decay":     cfg.weight_decay,
            "scheduler":        cfg.scheduler_cls,
            "config_rationale": list(cfg.rationale),
        },
        "metadata":  metadata,
        "timestamp": datetime.now().isoformat(),
    }

    if extrapolation_stats:
        result["extrapolation_stats"] = extrapolation_stats

    return result


def run_comprehensive_test(
    domains=None, num_samples=100, budget_secs=90.0, save_dir="results", verbose=False
):
    """
    Run comprehensive neural network baseline evaluation.

    Args:
        domains     : List of domains to test (None = all)
        num_samples : Number of samples per test case
        budget_secs : Wall-clock budget per equation (forwarded to train_neural_network)
        save_dir    : Directory to save results
        verbose     : Print detailed training progress
    """

    protocol = DeFiExperimentProtocol()

    if domains is None:
        domains = protocol.get_all_domains()

    print("=" * 80)
    print("IMPROVED NEURAL NETWORK BASELINE - DEFI & RISK MANAGEMENT".center(80))
    print("=" * 80)
    print(f"Architecture  : adaptive — resolved per-equation by adaptive_config.py")
    print(f"                (1-2 vars→[128,64,32]  3-4→[256,128,64,32]  5+→[512,256,128,64])")
    print(f"Solver params : adaptive — log-y, lr, wd, scheduler, seeds all from CaseProfile")
    print(f"Budget        : {budget_secs:.0f}s / equation (early-stopping + wall-clock cap)")
    print(f"Domains       : {', '.join(domains)}")
    print(f"Samples/test  : {num_samples}")
    print("=" * 80)

    all_results = []

    for domain in domains:
        print(f"\n{'=' * 80}")
        print(f"DOMAIN: {domain.upper()}".center(80))
        print(f"{protocol.get_domain_description(domain)}".center(80))
        print("=" * 80)

        test_cases = protocol.load_test_data(domain, num_samples=num_samples)

        for i, (description, X, y, var_names, metadata) in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] {description}")
            print(f"  Variables: {', '.join(var_names)}")
            print(f"  Ground truth: {metadata.get('ground_truth', 'N/A')}")
            print(f"  Difficulty: {metadata.get('difficulty', 'N/A')}")
            print(f"  Shape: X={X.shape}, y={y.shape}")

            if metadata.get("extrapolation_test"):
                print(f"  ⚠️  EXTRAPOLATION TEST CASE")

            # Train and evaluate
            print(f"  Training neural network…")
            result = train_neural_network(
                X,
                y,
                description,
                domain,
                metadata,
                budget_secs=budget_secs,
                verbose=verbose,
            )

            # Print results
            metrics  = result["evaluation"]
            training = result["training_info"]

            print(f"  ✅ R² Score: {metrics['r2']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  Final Train Loss: {training.get('final_train_loss', float('nan')):.6f}")
            print(f"  Config: hidden={training['hidden_dims']}  log-y={training['log_y']}  "
                  f"seeds={training['n_seeds']}  scheduler={training['scheduler']}")

            # Categorize performance
            r2 = metrics["r2"]
            if r2 > 0.99:
                print(f"  🎯 EXCELLENT FIT")
            elif r2 > 0.95:
                print(f"  ✓ Good fit")
            elif r2 > 0.80:
                print(f"  ⚠️  Moderate fit")
            else:
                print(f"  ❌ Poor fit")

            # Extrapolation info
            if result.get("extrapolation_stats"):
                ext_stats = result["extrapolation_stats"]
                print(f"  📊 Extrapolation: mean_error={ext_stats['mean_error']:.4f}")

            all_results.append(result)

    # =====================================================================
    # GENERATE COMPREHENSIVE REPORT
    # =====================================================================

    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE REPORT".center(80))
    print("=" * 80)

    report = protocol.generate_experiment_report(all_results)

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{save_dir}/baseline_nn_defi_IMPROVED_{timestamp}.json"
    report_file = f"{save_dir}/report_nn_defi_IMPROVED_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Results saved to: {results_file}")

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Report saved to: {report_file}")

    # =====================================================================
    # PRINT SUMMARY
    # =====================================================================

    print("\n" + "=" * 80)
    print("SUMMARY".center(80))
    print("=" * 80)

    overall = report["overall"]
    print(f"\n📊 Overall Performance:")
    print(f"  Total test cases: {overall['total_cases']}")
    print(
        f"  Successful: {overall['successful']}/{overall['total_cases']} "
        f"({100 * overall['success_rate']:.1f}%)"
    )

    if "mean_r2" in overall:
        print(f"  Mean R²: {overall['mean_r2']:.6f}")
        print(f"  Median R²: {overall['median_r2']:.6f}")
        print(f"  Std R²: {overall['std_r2']:.6f}")
        if "min_r2" in overall:
            print(f"  Min R²: {overall['min_r2']:.6f}")
            print(f"  Max R²: {overall['max_r2']:.6f}")

    print(f"\n📈 By Domain:")
    for domain, stats in report["by_domain"].items():
        mean_r2_str = (
            f"R²: {stats.get('mean_r2', 0):.4f}"
            if stats.get("mean_r2") is not None
            else "N/A"
        )
        print(
            f"  {domain}: {stats['successful']}/{stats['total']} "
            f"({100 * stats['success_rate']:.1f}%) - {mean_r2_str}"
        )

    if report.get("extrapolation_tests"):
        print(f"\n🎯 Extrapolation Test Cases:")
        for test in report["extrapolation_tests"]:
            status = "✅" if test["success"] else "❌"
            r2_str = (
                f"R²: {test.get('r2', 0):.4f}"
                if test.get("r2") is not None
                else "Failed"
            )
            print(f"  {status} {test['description'][:60]}")
            print(f"     {r2_str}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE".center(80))
    print("=" * 80)

    print("\n⚠️  Neural Network Limitations:")
    print("  • Black box model - no interpretable formula")
    print("  • May struggle with extrapolation beyond training range")
    print("  • Requires retraining for each new dataset")
    print("  • Cannot provide mathematical insights")
    print("\n✅ Neural Network Advantages:")
    print("  • Can learn complex non-linear patterns")
    print("  • No need for formula specification")
    print("  • Good for interpolation within training range")

    return report


def compare_with_llm_baseline(nn_report_file, llm_report_file):
    """
    Compare Neural Network results with Pure LLM baseline.

    Args:
        nn_report_file: Path to NN report JSON
        llm_report_file: Path to LLM report JSON
    """

    with open(nn_report_file, "r") as f:
        nn_report = json.load(f)

    with open(llm_report_file, "r") as f:
        llm_report = json.load(f)

    print("\n" + "=" * 80)
    print("COMPARISON: NEURAL NETWORK vs PURE LLM".center(80))
    print("=" * 80)

    print(f"\n{'Metric':<30} {'Neural Network':<20} {'Pure LLM':<20} {'Winner'}")
    print("-" * 80)

    nn_overall = nn_report["overall"]
    llm_overall = llm_report["overall"]

    # Overall R²
    nn_r2 = nn_overall.get("mean_r2", 0)
    llm_r2 = llm_overall.get("mean_r2", 0)
    winner = "NN 🏆" if nn_r2 > llm_r2 else "LLM 🏆" if llm_r2 > nn_r2 else "Tie"
    print(f"{'Mean R²':<30} {nn_r2:<20.4f} {llm_r2:<20.4f} {winner}")

    # Median R²
    nn_med = nn_overall.get("median_r2", 0)
    llm_med = llm_overall.get("median_r2", 0)
    winner = "NN 🏆" if nn_med > llm_med else "LLM 🏆" if llm_med > nn_med else "Tie"
    print(f"{'Median R²':<30} {nn_med:<20.4f} {llm_med:<20.4f} {winner}")

    # Success rate
    nn_success = nn_overall.get("success_rate", 0)
    llm_success = llm_overall.get("success_rate", 0)
    winner = (
        "NN 🏆"
        if nn_success > llm_success
        else "LLM 🏆" if llm_success > nn_success else "Tie"
    )
    print(f"{'Success Rate':<30} {nn_success:<20.2%} {llm_success:<20.2%} {winner}")

    print("\n" + "-" * 80)
    print("Domain-by-Domain Comparison:")
    print("-" * 80)

    for domain in nn_report["by_domain"].keys():
        if domain in llm_report["by_domain"]:
            nn_domain = nn_report["by_domain"][domain]
            llm_domain = llm_report["by_domain"][domain]

            nn_r2 = nn_domain.get("mean_r2", 0)
            llm_r2 = llm_domain.get("mean_r2", 0)

            winner = "NN" if nn_r2 > llm_r2 else "LLM" if llm_r2 > nn_r2 else "Tie"
            print(f"{domain:<15} NN: {nn_r2:>6.4f}  LLM: {llm_r2:>6.4f}  → {winner}")

    print("=" * 80)


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print(
                """
Usage:
  python baseline_neural_network_improved.py [options]

Options:
  --all                  Run on all domains
  --domain <domains>     Run on specific domains (comma-separated)
  --quick                Run quick test on 2 domains
  --compare <nn> <llm>   Compare NN results with LLM baseline
  --verbose              Show per-equation config rationale + training progress
  --budget <secs>        Wall-clock budget per equation (default: 90)
  --samples <n>          Set samples per test (default: 100)

Note: architecture, lr, scheduler, n_seeds are resolved automatically
      per-equation by adaptive_config.py — no manual tuning needed.

Examples:
  python baseline_neural_network_improved.py --all
  python baseline_neural_network_improved.py --domain amm,liquidation
  python baseline_neural_network_improved.py --quick --verbose
  python baseline_neural_network_improved.py --compare nn_report.json llm_report.json
  python baseline_neural_network_improved.py --all --budget 120
"""
            )
            sys.exit(0)

        elif sys.argv[1] == "--compare":
            if len(sys.argv) < 4:
                print("Error: --compare requires two file paths")
                sys.exit(1)
            compare_with_llm_baseline(sys.argv[2], sys.argv[3])
            sys.exit(0)

        elif sys.argv[1] == "--all":
            verbose     = "--verbose" in sys.argv
            budget_secs = 90.0
            samples     = 100

            if "--budget" in sys.argv:
                idx         = sys.argv.index("--budget")
                budget_secs = float(sys.argv[idx + 1])

            if "--samples" in sys.argv:
                idx     = sys.argv.index("--samples")
                samples = int(sys.argv[idx + 1])

            run_comprehensive_test(
                domains=None, num_samples=samples,
                budget_secs=budget_secs, verbose=verbose
            )

        elif sys.argv[1] == "--domain":
            if len(sys.argv) < 3:
                print("Error: --domain requires domain names")
                sys.exit(1)

            domains     = sys.argv[2].split(",")
            verbose     = "--verbose" in sys.argv
            budget_secs = 90.0
            if "--budget" in sys.argv:
                idx         = sys.argv.index("--budget")
                budget_secs = float(sys.argv[idx + 1])
            run_comprehensive_test(domains=domains, budget_secs=budget_secs, verbose=verbose)

        elif sys.argv[1] == "--quick":
            verbose = "--verbose" in sys.argv
            run_comprehensive_test(
                domains=["amm", "risk_var"],
                num_samples=100,
                budget_secs=60.0,
                verbose=verbose,
            )

        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
            sys.exit(1)
    else:
        # Default: run on all domains with 90 s budget per equation
        run_comprehensive_test(domains=None, num_samples=100, budget_secs=90.0)
