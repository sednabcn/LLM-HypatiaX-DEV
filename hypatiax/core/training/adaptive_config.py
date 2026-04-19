"""
adaptive_config.py
==================
A priori architecture & solver parameter selection for HypatiaX.

PHILOSOPHY
----------
Every hyper-parameter decision is made ONCE, BEFORE training, from
observable signals derived purely from (X, y, metadata).  No grid search,
no runtime branching, no duplicated magic numbers scattered across files.

The module exposes a single entry point:

    cfg = CaseProfile.from_data(X, y, metadata, budget_secs).to_config()

`cfg` is a frozen dataclass that every training loop can read uniformly:

    model     = ImprovedNN(cfg.input_dim, cfg.hidden_dims)
    optimizer = cfg.make_optimizer(model)
    scheduler = cfg.make_scheduler(optimizer)
    ...

SIGNAL → PARAMETER MAP  (all reasoning is documented inline)
─────────────────────────────────────────────────────────────
Signal                           → Parameter(s)
─────────────────────────────────────────────────────────────
n_vars                           → hidden_dims, max_epochs
n_samples                        → weight_decay, dropout_p
y_dynamic_range (decades)        → use_log_y, lr
y_pole_shaped                    → use_log_y=False (override), lr↓
X_cols_wide                      → log_X_cols
budget_secs                      → max_epochs (hard cap), n_seeds
metadata["difficulty"]           → n_seeds, max_epochs multiplier
metadata["extrapolation_test"]   → weight_decay↑ (stronger regularisation)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Optional torch — only needed at config *use* time, not at import time.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# CONSTANTS  —  single source of truth for every magic number
# ============================================================================

# Architecture width tiers keyed by n_vars
_HIDDEN_BY_NVARS = {
    1: [128, 64, 32],
    2: [128, 64, 32],
    3: [256, 128, 64, 32],
    4: [256, 128, 64, 32],
    5: [512, 256, 128, 64],
}
_HIDDEN_DEFAULT   = [512, 256, 128, 64]   # n_vars >= 6

# Epoch budget
_EPOCHS_BASE      = 300
_EPOCHS_PER_EXTRA_VAR = 100              # +100 epochs for each var beyond 2
_EPOCHS_MAX_LOG   = 2000                 # hard cap for log-space training
_EPOCHS_MAX_LIN   = 1000                 # hard cap for linear training

# Log-space y thresholds
_LOG_Y_MIN_DECADES = 2.0    # skip log-y if range < 2 decades (almost linear)
_LOG_Y_MAX_DECADES = 10.0   # skip log-y if range > 10 decades (divergent/pole)

# Pole detection thresholds (Bose-Einstein / Fermi-Dirac guard)
_POLE_P10_P90_RATIO = 50.0  # bulk distribution spans > 50x
_POLE_MIN_P10_RATIO = 10.0  # minimum is more than 10x below the 10th percentile

# Log-X threshold
_LOG_X_MIN_RATIO    = 10.0  # log-transform a column if max/min > 10x

# Learning rates
_LR_DEFAULT         = 1e-3
_LR_LOG_SPACE       = 3e-3   # faster convergence in log-space (smoother loss landscape)
_LR_POLE            = 5e-4   # slower near divergent functions

# Weight decay
_WD_DEFAULT         = 1e-5
_WD_EXTRAP          = 1e-4   # extra regularisation for extrapolation tests
_WD_SMALL_N         = 1e-4   # stronger when n_samples < 100

# Scheduler patience
_SCHED_PATIENCE_DEFAULT = 20
_SCHED_PATIENCE_LOG     = 30

# Early-stopping patience (in epochs)
_STOP_PATIENCE_DEFAULT  = 80
_STOP_PATIENCE_LOG      = 100

# Seed counts
_SEEDS_DEFAULT      = 1
_SEEDS_HARD         = 3      # "hard" equations get more seeds
_SEEDS_EXPERT       = 5      # "expert" equations

# Budget fractions (of total wall-clock budget_secs)
_BUDGET_FRAC_LOG    = 0.55
_BUDGET_FRAC_LIN    = 0.30
_BUDGET_FRAC_SEED   = 0.90   # multi-seed: divide this fraction equally


# ============================================================================
# SIGNAL EXTRACTION  —  all derived from (X, y) with no training needed
# ============================================================================

@dataclass
class CaseSignals:
    """
    All observable signals computed from (X, y, metadata) before any training.
    These are the *inputs* to every parameter decision.
    """

    # ── Problem geometry ──────────────────────────────────────────────────
    n_vars:        int        # number of input features
    n_samples:     int        # number of data points

    # ── Output distribution ───────────────────────────────────────────────
    y_all_pos:     bool       # all y > 0
    y_all_neg:     bool       # all y < 0
    y_sign:        float      # +1.0 or -1.0
    y_min_abs:     float      # min |y|
    y_max_abs:     float      # max |y|
    y_ratio:       float      # max|y| / min|y|  (dynamic range)
    y_decades:     float      # log10(y_ratio)
    y_p10:         float      # 10th-percentile of |y|
    y_p90:         float      # 90th-percentile of |y|
    y_pole_shaped: bool       # distribution consistent with a divergent pole

    # ── Input distribution ────────────────────────────────────────────────
    log_X_cols:    List[int]  # columns to log-transform a priori

    # ── Metadata ──────────────────────────────────────────────────────────
    difficulty:          str    # "easy" | "medium" | "hard" | "expert" | ""
    is_extrapolation:    bool
    domain:              str

    # ── Derived decisions ─────────────────────────────────────────────────
    use_log_y:     bool       # transform y → log(y) before training
    log_y_reason:  str        # human-readable explanation for log_y decision

    @classmethod
    def from_data(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> "CaseSignals":
        """Compute all signals from raw data.  Zero side-effects."""
        meta   = metadata or {}
        n_samp, n_vars = X.shape

        # ── y distribution ──────────────────────────────────────────────
        y_abs    = np.abs(y)
        y_all_pos = bool(np.all(y > 0))
        y_all_neg = bool(np.all(y < 0))
        y_sign    = 1.0 if not y_all_neg else -1.0
        y_min_abs = float(np.min(y_abs))
        y_max_abs = float(np.max(y_abs))
        y_ratio   = y_max_abs / (y_min_abs + 1e-300)
        y_decades = float(np.log10(y_ratio)) if y_ratio > 1 else 0.0
        y_p10     = float(np.percentile(y_abs, 10))
        y_p90     = float(np.percentile(y_abs, 90))

        # Pole-shape: bulk spans >50x AND minimum is far below the 10th percentile.
        # Fires for Bose-Einstein / Fermi-Dirac, NOT for power-laws like Coulomb.
        y_pole_shaped = (
            y_p10 > 0
            and (y_p90 / (y_p10 + 1e-300)) > _POLE_P10_P90_RATIO
            and y_min_abs < y_p10 / _POLE_MIN_P10_RATIO
        )

        # ── Log-y decision ───────────────────────────────────────────────
        # Rule 1: only consider when y is strictly one-signed.
        # Rule 2: skip if range < 2 decades (almost linear — log adds no value).
        # Rule 3: skip if range > 10 decades (divergent, log poorly represents).
        # Rule 4: skip if pole-shaped (Bose-Einstein guard).
        if not (y_all_pos or y_all_neg):
            use_log_y  = False
            log_y_reason = "mixed-sign y — log undefined"
        elif y_decades < _LOG_Y_MIN_DECADES:
            use_log_y  = False
            log_y_reason = f"y range only {y_decades:.1f} decades < {_LOG_Y_MIN_DECADES} — linear space"
        elif y_decades > _LOG_Y_MAX_DECADES:
            use_log_y  = False
            log_y_reason = f"y range {y_decades:.1f} decades > {_LOG_Y_MAX_DECADES} — divergent / skip log"
        elif y_pole_shaped:
            use_log_y  = False
            log_y_reason = (
                f"pole-shaped y (p10={y_p10:.3g}, p90={y_p90:.3g}, min={y_min_abs:.3g})"
                " — Bose-Einstein/Fermi-Dirac guard"
            )
        else:
            use_log_y  = True
            log_y_reason = f"power-law range {y_decades:.1f} decades — log-space training"

        # ── Log-X columns ────────────────────────────────────────────────
        # When log_y is active: transform ALL strictly-positive columns
        # (in y=∏xᵢ^aᵢ, every log(xᵢ) contributes regardless of its own range).
        # When log_y is inactive: only transform wide-range (>10x) positive columns.
        log_X_cols = []
        for col in range(n_vars):
            col_data = X[:, col]
            col_pos  = bool(np.all(col_data > 0))
            col_wide = float(np.max(col_data)) / (float(np.min(col_data)) + 1e-300) > _LOG_X_MIN_RATIO
            if col_pos and (use_log_y or col_wide):
                log_X_cols.append(col)

        return cls(
            n_vars=n_vars,
            n_samples=n_samp,
            y_all_pos=y_all_pos,
            y_all_neg=y_all_neg,
            y_sign=y_sign,
            y_min_abs=y_min_abs,
            y_max_abs=y_max_abs,
            y_ratio=y_ratio,
            y_decades=y_decades,
            y_p10=y_p10,
            y_p90=y_p90,
            y_pole_shaped=y_pole_shaped,
            log_X_cols=log_X_cols,
            difficulty=str(meta.get("difficulty", "")).lower(),
            is_extrapolation=bool(meta.get("extrapolation_test", False)),
            domain=str(meta.get("domain", "")),
            use_log_y=use_log_y,
            log_y_reason=log_y_reason,
        )


# ============================================================================
# SOLVER CONFIG  —  the resolved parameter set, fully determined a priori
# ============================================================================

@dataclass(frozen=True)
class SolverConfig:
    """
    Immutable resolved parameters for one training run.
    Produced by CaseProfile.to_config().
    """

    # ── Architecture ──────────────────────────────────────────────────────
    input_dim:     int
    hidden_dims:   Tuple[int, ...]

    # ── Preprocessing ─────────────────────────────────────────────────────
    use_log_y:     bool
    y_sign:        float
    log_X_cols:    Tuple[int, ...]

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer_cls: str         # "AdamW" | "Adam"
    lr:            float
    weight_decay:  float

    # ── Scheduler ─────────────────────────────────────────────────────────
    scheduler_cls: str         # "ReduceLROnPlateau" | "CosineAnnealingWarmRestarts"
    sched_patience: int        # for ReduceLROnPlateau
    sched_factor:  float       # for ReduceLROnPlateau
    sched_min_lr:  float

    # ── Training loop ─────────────────────────────────────────────────────
    max_epochs:    int
    stop_patience: int         # early-stopping patience
    n_seeds:       int         # number of independent restarts
    grad_clip:     float       # gradient-clipping max-norm (0 = disabled)

    # ── Budget ────────────────────────────────────────────────────────────
    budget_log_secs:   float   # wall-clock for log-space training phase
    budget_lin_secs:   float   # wall-clock for linear fallback phase

    # ── Human-readable rationale ──────────────────────────────────────────
    rationale: Tuple[str, ...]  # ordered list of decisions and their reasons

    # ------------------------------------------------------------------
    # Factory helpers — create PyTorch objects from the frozen config.
    # These are the ONLY place that imports torch, keeping the module
    # usable in environments without PyTorch (e.g. analysis scripts).
    # ------------------------------------------------------------------

    def make_optimizer(self, model) -> "torch.optim.Optimizer":
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch not available")
        if self.optimizer_cls == "AdamW":
            return optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def make_scheduler(self, optimizer) -> "torch.optim.lr_scheduler._LRScheduler":
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch not available")
        if self.scheduler_cls == "CosineAnnealingWarmRestarts":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=200, T_mult=2, eta_min=self.sched_min_lr
            )
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.sched_patience,
            factor=self.sched_factor,
            min_lr=self.sched_min_lr,
        )

    def summary(self) -> str:
        lines = [
            "── SolverConfig ──────────────────────────────────────────────",
            f"  Architecture : input({self.input_dim}) → {list(self.hidden_dims)} → 1",
            f"  Log-y        : {self.use_log_y}",
            f"  Log-X cols   : {list(self.log_X_cols)}",
            f"  Optimizer    : {self.optimizer_cls}(lr={self.lr}, wd={self.weight_decay})",
            f"  Scheduler    : {self.scheduler_cls}(patience={self.sched_patience})",
            f"  Training     : max_epochs={self.max_epochs}, patience={self.stop_patience}, seeds={self.n_seeds}",
            f"  Budget       : log={self.budget_log_secs:.0f}s  lin={self.budget_lin_secs:.0f}s",
            "  Rationale:",
        ]
        for r in self.rationale:
            lines.append(f"    • {r}")
        lines.append("─" * 62)
        return "\n".join(lines)


# ============================================================================
# CASE PROFILE  —  maps signals to config (the core of this module)
# ============================================================================

@dataclass
class CaseProfile:
    """
    Holds pre-computed signals and resolves them into a SolverConfig.

    Usage::

        sig = CaseSignals.from_data(X, y, metadata)
        cfg = CaseProfile(sig, budget_secs=90).to_config()
        # --- then in training ---
        model     = ImprovedNN(cfg.input_dim, cfg.hidden_dims)
        optimizer = cfg.make_optimizer(model)
        scheduler = cfg.make_scheduler(optimizer)
    """

    signals:     CaseSignals
    budget_secs: float = 90.0

    def to_config(self) -> SolverConfig:
        """
        Translate signals into a fully-resolved SolverConfig.
        Every decision is documented with its reason.
        """
        s   = self.signals
        rat = []   # rationale accumulator

        # ── 1. ARCHITECTURE ──────────────────────────────────────────────
        hidden_dims = _HIDDEN_BY_NVARS.get(s.n_vars, _HIDDEN_DEFAULT)
        rat.append(
            f"Architecture {hidden_dims}: n_vars={s.n_vars} "
            f"(1-2→[128,64,32], 3-4→[256,128,64,32], 5+→[512,256,128,64])"
        )

        # ── 2. LOG TRANSFORMS ────────────────────────────────────────────
        rat.append(f"Log-y={s.use_log_y}: {s.log_y_reason}")
        if s.log_X_cols:
            rat.append(
                f"Log-X cols={s.log_X_cols}: "
                f"{'all positive cols (log_y active)' if s.use_log_y else 'wide-range positive cols'}"
            )

        # ── 3. LEARNING RATE ─────────────────────────────────────────────
        if s.y_pole_shaped:
            lr = _LR_POLE
            rat.append(f"LR={lr}: pole-shaped y — slower initial rate")
        elif s.use_log_y:
            lr = _LR_LOG_SPACE
            rat.append(f"LR={lr}: log-space training — smoother loss, higher LR safe")
        else:
            lr = _LR_DEFAULT
            rat.append(f"LR={lr}: default linear-space rate")

        # ── 4. WEIGHT DECAY ──────────────────────────────────────────────
        wd = _WD_DEFAULT
        if s.is_extrapolation:
            wd = max(wd, _WD_EXTRAP)
            rat.append(f"weight_decay={wd}: extrapolation test — stronger L2 regularisation")
        if s.n_samples < 100:
            wd = max(wd, _WD_SMALL_N)
            rat.append(f"weight_decay={wd}: n_samples={s.n_samples} < 100 — stronger L2")
        if wd == _WD_DEFAULT:
            rat.append(f"weight_decay={wd}: default")

        # ── 5. OPTIMIZER ─────────────────────────────────────────────────
        # AdamW for all cases — decoupled weight decay is strictly better.
        optimizer_cls = "AdamW"
        rat.append("Optimizer: AdamW (decoupled weight decay, always preferred over Adam)")

        # ── 6. SCHEDULER ─────────────────────────────────────────────────
        # CosineAnnealingWarmRestarts for log-space or hard/expert tasks:
        # allows periodic LR spikes to escape plateaus on sharp functions.
        # ReduceLROnPlateau for everything else (simpler, stable).
        if s.use_log_y or s.difficulty in ("hard", "expert"):
            scheduler_cls   = "CosineAnnealingWarmRestarts"
            sched_patience  = _SCHED_PATIENCE_LOG
            rat.append(
                f"Scheduler: CosineAnnealingWarmRestarts — "
                f"{'log-space' if s.use_log_y else s.difficulty} needs LR restarts"
            )
        else:
            scheduler_cls   = "ReduceLROnPlateau"
            sched_patience  = _SCHED_PATIENCE_DEFAULT
            rat.append(f"Scheduler: ReduceLROnPlateau(patience={sched_patience}) — default stable schedule")

        sched_factor  = 0.5
        sched_min_lr  = lr * 1e-3

        # ── 7. EPOCHS ────────────────────────────────────────────────────
        base_epochs = _EPOCHS_BASE + _EPOCHS_PER_EXTRA_VAR * max(0, s.n_vars - 2)
        # Difficulty multiplier
        if s.difficulty == "expert":
            base_epochs = int(base_epochs * 2.0)
            rat.append(f"Epochs multiplier ×2.0: difficulty=expert")
        elif s.difficulty == "hard":
            base_epochs = int(base_epochs * 1.5)
            rat.append(f"Epochs multiplier ×1.5: difficulty=hard")

        # Cap to the appropriate hard ceiling
        max_epochs = min(base_epochs, _EPOCHS_MAX_LOG if s.use_log_y else _EPOCHS_MAX_LIN)
        rat.append(f"max_epochs={max_epochs}: base={base_epochs}, n_vars={s.n_vars}")

        # ── 8. EARLY-STOPPING PATIENCE ───────────────────────────────────
        stop_patience = _STOP_PATIENCE_LOG if s.use_log_y else _STOP_PATIENCE_DEFAULT
        rat.append(
            f"stop_patience={stop_patience}: "
            f"{'log-space needs more epochs to plateau' if s.use_log_y else 'default'}"
        )

        # ── 9. SEEDS ─────────────────────────────────────────────────────
        if s.difficulty == "expert":
            n_seeds = _SEEDS_EXPERT
        elif s.difficulty == "hard" or s.is_extrapolation:
            n_seeds = _SEEDS_HARD
        else:
            n_seeds = _SEEDS_DEFAULT
        rat.append(
            f"n_seeds={n_seeds}: "
            f"difficulty={s.difficulty or 'default'}, extrapolation={s.is_extrapolation}"
        )

        # ── 10. GRADIENT CLIPPING ────────────────────────────────────────
        grad_clip = 1.0   # always clip — prevents runaway gradients on log-space functions
        rat.append(f"grad_clip={grad_clip}: always active (especially important in log-space)")

        # ── 11. WALL-CLOCK BUDGET ────────────────────────────────────────
        # If multi-seed: divide _BUDGET_FRAC_SEED of the total budget equally.
        if n_seeds > 1:
            per_seed_budget  = (self.budget_secs * _BUDGET_FRAC_SEED) / n_seeds
            budget_log_secs  = max(20, per_seed_budget * _BUDGET_FRAC_LOG / (_BUDGET_FRAC_LOG + _BUDGET_FRAC_LIN))
            budget_lin_secs  = max(15, per_seed_budget * _BUDGET_FRAC_LIN / (_BUDGET_FRAC_LOG + _BUDGET_FRAC_LIN))
            rat.append(
                f"Budget per seed: log={budget_log_secs:.0f}s, lin={budget_lin_secs:.0f}s "
                f"({n_seeds} seeds × {per_seed_budget:.0f}s)"
            )
        else:
            budget_log_secs = max(20, self.budget_secs * _BUDGET_FRAC_LOG)
            budget_lin_secs = max(15, self.budget_secs * _BUDGET_FRAC_LIN)
            rat.append(
                f"Budget: log={budget_log_secs:.0f}s, lin={budget_lin_secs:.0f}s "
                f"(total={self.budget_secs:.0f}s)"
            )

        return SolverConfig(
            input_dim      = s.n_vars,
            hidden_dims    = tuple(hidden_dims),
            use_log_y      = s.use_log_y,
            y_sign         = s.y_sign,
            log_X_cols     = tuple(s.log_X_cols),
            optimizer_cls  = optimizer_cls,
            lr             = lr,
            weight_decay   = wd,
            scheduler_cls  = scheduler_cls,
            sched_patience = sched_patience,
            sched_factor   = sched_factor,
            sched_min_lr   = sched_min_lr,
            max_epochs     = max_epochs,
            stop_patience  = stop_patience,
            n_seeds        = n_seeds,
            grad_clip      = grad_clip,
            budget_log_secs = budget_log_secs,
            budget_lin_secs = budget_lin_secs,
            rationale      = tuple(rat),
        )

    @classmethod
    def from_data(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        metadata: Optional[dict] = None,
        budget_secs: float = 90.0,
    ) -> "CaseProfile":
        """Convenience constructor — computes signals then wraps in a CaseProfile."""
        sig = CaseSignals.from_data(X, y, metadata)
        return cls(signals=sig, budget_secs=budget_secs)


# ============================================================================
# DROP-IN TRAINING LOOP  —  reference implementation using SolverConfig
# ============================================================================

def train_with_config(
    X: np.ndarray,
    y: np.ndarray,
    cfg: SolverConfig,
    ImprovedNN,                     # your nn.Module class
    verbose: bool = False,
) -> Tuple[object, float, float]:
    """
    Reference training loop that consumes a SolverConfig uniformly.

    Returns:
        (best_model, best_r2, final_train_loss)

    This function intentionally has NO parameter decisions of its own — every
    value comes from cfg.  Adding a new hyper-parameter means adding it to
    CaseProfile.to_config() and reading it here.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to train")
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    best_model, best_r2, best_train_loss = None, float("-inf"), float("inf")

    for seed in range(cfg.n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ── Preprocessing ────────────────────────────────────────────────
        X_work = X.copy().astype(float)
        for col in cfg.log_X_cols:
            X_work[:, col] = np.log(X_work[:, col])

        if cfg.use_log_y:
            y_work = np.log(np.abs(y))
        else:
            y_work = y.copy()

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_work, y_work, test_size=0.2, random_state=42 + seed
        )

        sx = StandardScaler().fit(X_tr)
        sy = StandardScaler().fit(y_tr.reshape(-1, 1))
        X_tr_s  = sx.transform(X_tr)
        X_val_s = sx.transform(X_val)
        y_tr_s  = sy.transform(y_tr.reshape(-1, 1)).flatten()
        y_val_s = sy.transform(y_val.reshape(-1, 1)).flatten()

        X_all_s = sx.transform(X_work)

        X_t = torch.FloatTensor(X_tr_s)
        y_t = torch.FloatTensor(y_tr_s).reshape(-1, 1)
        X_vt = torch.FloatTensor(X_val_s)
        y_vt = torch.FloatTensor(y_val_s).reshape(-1, 1)

        # ── Model + Solver (all from cfg) ─────────────────────────────────
        model     = ImprovedNN(cfg.input_dim, list(cfg.hidden_dims))
        optimizer = cfg.make_optimizer(model)
        scheduler = cfg.make_scheduler(optimizer)
        criterion = nn.MSELoss()

        best_val   = float("inf")
        best_state = None
        no_improve = 0
        final_train_loss = float("inf")
        _deadline  = time.time() + cfg.budget_log_secs

        for epoch in range(cfg.max_epochs):
            if time.time() > _deadline:
                if verbose:
                    print(f"   [seed={seed}] budget reached at epoch {epoch}")
                break

            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_t), y_t)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            # Scheduler step — ReduceLROnPlateau needs the loss value
            if cfg.scheduler_cls == "ReduceLROnPlateau":
                scheduler.step(loss)
            else:
                scheduler.step(epoch)

            final_train_loss = loss.item()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_vt), y_vt).item()
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

        # ── Evaluate on full dataset ──────────────────────────────────────
        model.eval()
        with torch.no_grad():
            y_pred_s = model(torch.FloatTensor(X_all_s)).numpy().flatten()
        y_pred_w = sy.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
        if cfg.use_log_y:
            y_pred = cfg.y_sign * np.exp(np.clip(y_pred_w, -500, 500))
        else:
            y_pred = y_pred_w

        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else (1.0 if ss_res < 1e-10 else 0.0)

        if verbose:
            print(f"   [seed={seed}] R²={r2:.4f}  train_loss={final_train_loss:.6f}")

        if r2 > best_r2:
            best_r2          = r2
            best_train_loss  = final_train_loss
            best_model       = model

    return best_model, best_r2, best_train_loss


# ============================================================================
# CONVENIENCE: one-liner for the full pipeline
# ============================================================================

def resolve(
    X: np.ndarray,
    y: np.ndarray,
    metadata: Optional[dict] = None,
    budget_secs: float = 90.0,
    verbose: bool = False,
) -> SolverConfig:
    """
    One-liner to go from raw data → SolverConfig.

        cfg = adaptive_config.resolve(X, y, metadata, budget_secs=90)

    Optionally prints the full rationale when verbose=True.
    """
    cfg = CaseProfile.from_data(X, y, metadata, budget_secs).to_config()
    if verbose:
        print(cfg.summary())
    return cfg


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  adaptive_config  —  self-test".center(70))
    print("=" * 70)

    rng = np.random.default_rng(0)

    test_cases = [
        {
            "name": "Power-law (Coulomb-like, 2 vars, 200 samples)",
            "X": np.column_stack([rng.uniform(1, 100, 200), rng.uniform(1, 100, 200)]),
            "y": lambda X: 8.99e9 * X[:, 0] * X[:, 1],
            "meta": {"difficulty": "medium"},
        },
        {
            "name": "Bose-Einstein (pole-shaped, 2 vars, 200 samples)",
            "X": np.column_stack([rng.uniform(0.1, 3.0, 200), rng.uniform(0.1, 1.0, 200)]),
            "y": lambda X: 1.0 / (np.exp(X[:, 0] / X[:, 1]) - 1 + 1e-6),
            "meta": {"difficulty": "hard"},
        },
        {
            "name": "Extrapolation test (5 vars, hard, 150 samples)",
            "X": rng.uniform(0.1, 10, (150, 5)),
            "y": lambda X: X[:, 0] * X[:, 1] ** 2 / (X[:, 2] * X[:, 3] + X[:, 4]),
            "meta": {"difficulty": "hard", "extrapolation_test": True},
        },
        {
            "name": "Small dataset (1 var, easy, 50 samples)",
            "X": rng.uniform(1, 100, (50, 1)),
            "y": lambda X: 9.81 * X[:, 0],
            "meta": {"difficulty": "easy"},
        },
    ]

    for tc in test_cases:
        X   = tc["X"]
        y   = tc["y"](X)
        cfg = resolve(X, y, tc["meta"], budget_secs=90, verbose=False)
        print(f"\n▶ {tc['name']}")
        print(cfg.summary())
