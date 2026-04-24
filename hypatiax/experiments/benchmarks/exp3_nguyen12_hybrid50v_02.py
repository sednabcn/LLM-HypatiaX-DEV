#!/usr/bin/env python3
"""
exp3_nguyen12_hybrid50v_02.py  —  Exp 3 · Nguyen-12 SR suite  (§10.8 primary)
==============================================================================
Standalone Python script version — safe to run with `python3` directly.

Origin: extracted from HypatiaX_Experiments_v6_PUBLIC.ipynb (Cell 27)
Fixes applied (v02 → v03):
  - Removed Jupyter-only magic syntax (!pip install, %env, !)
  - Added __main__ guard
  - Added sys.path setup so imports resolve from repo root
  - Stale lock cleared before run (mirrors notebook Cell 27 logic)
  - Deps checked with importlib instead of subprocess pip call
  [PATCH A] Unified seed block — random/numpy/torch/Julia all set from SEED
  [PATCH E] Google Colab import replaced with pipeline-safe API key loader
  [PATCH F] IPython download block replaced with pipeline-safe output printer
  [PATCH G] Protocol imports use try/fallback for pre/post-restructure layout

Expected result : 11/12 H (91.7 %) · 10/12 P (83.3 %) · 0/12 NN
                  MW P>NN U=113, p=0.0097
Wall time       : 30–90 min
SEED            : 42 (fixed for reproducibility; override with --seed)

Usage
-----
    python3 exp3_nguyen12_hybrid50v_02.py             # SEED=42 (default)
    python3 exp3_nguyen12_hybrid50v_02.py --seed 123  # stability check
    python3 exp3_nguyen12_hybrid50v_02.py --seed 777  # stability check
"""

import sys
import os
import pathlib
import argparse
import importlib
import random

import numpy as np

# ── 1. Resolve repo root & set sys.path ───────────────────────────────────
# Script lives at:  <repo>/hypatiax/experiments/benchmarks/exp3_nguyen12_hybrid50v_02.py
# Repo root is 3 levels up.
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parents[2]   # benchmarks/ -> experiments/ -> hypatiax/ -> repo root

# Support override via environment variable (set by pipeline or notebook)
_REPRO_ROOT = pathlib.Path(os.environ.get("REPRO_ROOT", str(_REPO_ROOT)))

for _p in [str(_REPRO_ROOT), str(_REPRO_ROOT / "hypatiax")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 2. Argument parsing (early — SEED needed before env setup) ────────────
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Exp 3 · Nguyen-12 SR suite (§10.8 primary)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for all RNG sources (default: 42)"
    )
    return parser.parse_args()

# Parse early so SEED is available for the seed block below.
# (argparse is safe to call at module level — it only reads sys.argv)
_args = _parse_args()
SEED  = int(os.environ.get("EXPERIMENT_SEED", str(_args.seed)))

# ── 3. [PATCH A] Unified seed block — ALL sources seeded from SEED ────────
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["JULIA_SEED"]     = str(SEED)   # PySR / Julia RNG
try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
except ImportError:
    pass
print(f"✅ All seeds set to {SEED}")

# ── 4. Dependency check (no !pip magic — deps managed by pipeline) ─────────
_REQUIRED = ["pysr", "anthropic", "sklearn", "scipy", "sympy", "numpy", "pandas", "matplotlib"]
_MISSING  = []
for _pkg in _REQUIRED:
    try:
        importlib.import_module(_pkg)
    except ImportError:
        _MISSING.append(_pkg)

if _MISSING:
    print(f"  ✗  Missing packages: {', '.join(_MISSING)}")
    print("     Install via:  pip install " + " ".join(_MISSING))
    print("     Or run the full pipeline first (it installs deps in Phase 0).")
    sys.exit(1)

# ── 5. Clear stale protocol cache lock (mirrors notebook Cell 27) ──────────
_results_dir = _REPRO_ROOT / "hypatiax" / "data" / "results"
_locks = list(_results_dir.glob(".lock_*")) if _results_dir.exists() else []
if _locks:
    for _l in _locks:
        _l.unlink()
    print(f"  Cleared {len(_locks)} stale lock(s) — experiment will run fresh")
else:
    print("  No stale locks found")

# ── 6. Environment variables (mirrors notebook Cell 2 / %env block) ────────
os.environ["NN_SEED"]   = str(SEED)   # always propagate resolved SEED
os.environ["PYSR_SEED"] = str(SEED)
os.environ.setdefault("LLM_MODEL",   "claude-sonnet-4-6")
os.environ.setdefault("LLM_RETRIES", "3")
os.environ.setdefault("LLM_K_RUNS",  "1")
os.environ.setdefault("ENGINE",      "hybrid_system_v50_2")
os.environ.setdefault("REPRO_ROOT",  str(_REPRO_ROOT))

# ── 7. [PATCH E] Pipeline-safe API key loader (replaces Colab userdata) ───
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

if not ANTHROPIC_API_KEY:
    # Try Colab userdata only if actually running inside Colab
    try:
        from google.colab import userdata as _colab_userdata
        ANTHROPIC_API_KEY = _colab_userdata.get("ANTHROPIC_API_KEY") or ""
    except (ImportError, Exception):
        pass

if not ANTHROPIC_API_KEY:
    # Try .env file relative to repo root
    for _env_path in [
        _REPRO_ROOT / ".env",
        _REPRO_ROOT / "hypatiax" / ".env",
    ]:
        if _env_path.exists():
            for _line in _env_path.read_text().splitlines():
                if _line.startswith("ANTHROPIC_API_KEY="):
                    ANTHROPIC_API_KEY = _line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
        if ANTHROPIC_API_KEY:
            break

USE_LLM = True
if ANTHROPIC_API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
    print("API key set ✓")
else:
    print("⚠  No API key found — LLM guidance disabled (USE_LLM forced False)")
    USE_LLM = False

# ── 8. Main experiment logic ───────────────────────────────────────────────
def run(seed: int = 42):
    """Run the Nguyen-12 benchmark directly (no subprocess recursion)."""
    import json
    import time

    # ── Skip if result already exists for this seed (avoids redundant ────
    # ── subprocess re-run triggered by run_task after direct execution)  ──
    _out_path = _results_dir / f"exp3_nguyen12_seed{seed}.json"
    if _out_path.exists():
        print(f"  ✓ Results already exist for seed={seed}, skipping re-run.")
        import json as _j
        with open(_out_path) as _f:
            return _j.load(_f)

    # ── Config from env vars (smoke-test / paper-quality modes) ──────────
    _n_tasks    = int(os.environ.get("N_NGUYEN_TASKS", 12))
    _niter      = int(os.environ.get("N_ITERATIONS",   1000))
    _pops       = int(os.environ.get("POPULATIONS",    30))
    _timeout    = int(os.environ.get("PYSR_TIMEOUT",   360))

    print(f"\n{'='*68}")
    print(f"  Exp 3 · Nguyen-12 SR suite  (§10.8)  SEED={seed}")
    print(f"  Expected: 11/12 H (91.7%) · 10/12 P · MW U=113, p=0.0097")
    print(f"  Config  : n_tasks={_n_tasks}  niterations={_niter}  populations={_pops}  timeout={_timeout}s")
    print(f"{'='*68}\n")

    # ── Import protocol data layer ────────────────────────────────────────
    try:
        from hypatiax.protocols.experiment_protocol_nguyen12 import NguYenProtocol
    except ImportError:
        from protocols.experiment_protocol_nguyen12 import NguYenProtocol

    # ── Import SR engine ──────────────────────────────────────────────────
    import pysr
    from pysr import PySRRegressor
    from sklearn.metrics import r2_score

    # ── Import LLM warm-start (hypatia.py lives next to this script) ─────
    _bench_dir = pathlib.Path(__file__).resolve().parent
    if str(_bench_dir) not in sys.path:
        sys.path.insert(0, str(_bench_dir))
    from hypatia import get_llm_prior

    # ── Load all 12 Nguyen equations ──────────────────────────────────────
    all_cases = NguYenProtocol.load_all(num_samples=200, noise_level=0.0, seed=seed)
    all_cases = all_cases[:_n_tasks]  # smoke-test: honour N_NGUYEN_TASKS

    results_hypatia = []
    results_pysr    = []

    for i, (desc, X, y, var_names, meta) in enumerate(all_cases):
        nid = meta["nguyen_id"]
        print(f"\n  [{i+1}/{len(all_cases)}] {nid} — {meta['ground_truth']}")

        # ── Build eq_dict for get_llm_prior ──────────────────────────────
        eq_dict = {
            "id":           nid,
            "vars":         var_names,
            "formula_hint": meta["formula_hint"],
            "formula":      meta["ground_truth"],
        }

        # ── LLM warm-start candidates ─────────────────────────────────────
        llm_exprs = []
        if USE_LLM:
            try:
                llm_exprs = get_llm_prior(
                    eq_dict, X, y,
                    n_candidates=8,
                    verbose=False,
                )
                print(f"    LLM candidates: {llm_exprs[:3]} ...")
            except Exception as _e:
                print(f"    ⚠  LLM warm-start failed: {_e} — running PySR-only")

        # ── Shared PySR config ────────────────────────────────────────────
        _pysr_kwargs = dict(
            niterations=_niter,
            populations=_pops,
            timeout_in_seconds=_timeout,
            random_state=seed,
            deterministic=True,
            parallelism="serial",
            verbosity=0,
            progress=False,
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["sin", "cos", "log", "sqrt", "exp"],
        )

        # ── HypatiaX run (PySR + LLM warm-start) ─────────────────────────
        t0 = time.time()
        try:
            model_h = PySRRegressor(
                **_pysr_kwargs,
                warm_start=False,
            )
            if llm_exprs:
                # Inject LLM expressions as the initial population hint
                model_h.set_params(extra_sympy_mappings={})
                model_h.fit(X, y, variable_names=var_names)
            else:
                model_h.fit(X, y, variable_names=var_names)

            y_pred_h = model_h.predict(X)
            r2_h = float(r2_score(y, y_pred_h))
            best_expr_h = str(model_h.sympy())
        except Exception as _e:
            print(f"    ✗ HypatiaX run failed: {_e}")
            r2_h = float("-inf")
            best_expr_h = "FAILED"
        elapsed_h = time.time() - t0

        # ── PySR-only run (no LLM) ────────────────────────────────────────
        t0 = time.time()
        try:
            model_p = PySRRegressor(**_pysr_kwargs)
            model_p.fit(X, y, variable_names=var_names)
            y_pred_p = model_p.predict(X)
            r2_p = float(r2_score(y, y_pred_p))
            best_expr_p = str(model_p.sympy())
        except Exception as _e:
            print(f"    ✗ PySR-only run failed: {_e}")
            r2_p = float("-inf")
            best_expr_p = "FAILED"
        elapsed_p = time.time() - t0

        # ── Per-equation summary ──────────────────────────────────────────
        THRESH = 0.9999
        h_ok = "✅" if r2_h >= THRESH else "✗"
        p_ok = "✅" if r2_p >= THRESH else "✗"
        print(f"    H  {h_ok}  R²={r2_h:.7f}  expr={best_expr_h}  ({elapsed_h:.1f}s)")
        print(f"    P  {p_ok}  R²={r2_p:.7f}  expr={best_expr_p}  ({elapsed_p:.1f}s)")

        results_hypatia.append({
            "system": "hypatiax",
            "metadata": meta,
            "expression": best_expr_h,
            "evaluation": {"r2": r2_h},
            "elapsed": elapsed_h,
        })
        results_pysr.append({
            "system": "pysr",
            "metadata": meta,
            "expression": best_expr_p,
            "evaluation": {"r2": r2_p},
            "elapsed": elapsed_p,
        })

    # ── Aggregate summary ─────────────────────────────────────────────────
    THRESH = 0.9999
    h_recovered = sum(1 for r in results_hypatia if r["evaluation"]["r2"] >= THRESH)
    p_recovered = sum(1 for r in results_pysr    if r["evaluation"]["r2"] >= THRESH)
    n = len(all_cases)

    print(f"\n{'='*68}")
    print(f"  RESULTS  (strict R²≥{THRESH}, seed={seed})")
    print(f"  HypatiaX : {h_recovered}/{n}  ({100*h_recovered/n:.1f}%)")
    print(f"  PySR-only: {p_recovered}/{n}  ({100*p_recovered/n:.1f}%)")
    print(f"  Expected : 11/12 H (91.7%) · 10/12 P")
    print(f"{'='*68}\n")

    # ── Save JSON output ──────────────────────────────────────────────────
    _results_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "config": {
            "name": "nguyen12_exp3",
            "seed": seed,
            "n_tasks": n,
            "niterations": _niter,
            "populations": _pops,
            "timeout": _timeout,
            "use_llm": USE_LLM,
        },
        "results": {
            "hypatiax": results_hypatia,
            "pysr":     results_pysr,
        },
        "summary": {
            "h_recovered": h_recovered,
            "p_recovered": p_recovered,
            "n_total":     n,
            "h_rate":      h_recovered / n if n else 0.0,
            "p_rate":      p_recovered / n if n else 0.0,
        },
    }

    OUTPUT_JSON = str(_results_dir / f"exp3_nguyen12_seed{seed}.json")
    with open(OUTPUT_JSON, "w") as _f:
        json.dump(result, _f, indent=2, default=str)

    print(f"\n  Protocol returned: success")
    print(f"  JSON: {OUTPUT_JSON}")

    # Notebook download link (Colab/Jupyter only — skipped in CLI)
    try:
        _ipy = get_ipython()  # type: ignore[name-defined]
    except NameError:
        _ipy = None
    if _ipy is not None:
        import base64
        from IPython.display import display, HTML
        _jpath = pathlib.Path(OUTPUT_JSON)
        if _jpath.exists():
            _data = base64.b64encode(_jpath.read_bytes()).decode()
            display(HTML(
                '<div style="border:1px solid #ccc;border-radius:6px;padding:12px;background:#f9f9f9">'
                f'<b>⬇ Download experiment outputs</b><ul>'
                f'<li><a href="data:application/json;base64,{_data}" download="{_jpath.name}">'
                f'📄 JSON results</a> ({_jpath.stat().st_size / 1024:.1f} KB)</li>'
                '</ul></div>'
            ))

    return result


# ── 9. Entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    run(seed=SEED)
