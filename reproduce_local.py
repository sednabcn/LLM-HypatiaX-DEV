#!/usr/bin/env python3
"""
reproduce_local.py — HypatiaX · Local Reproduction Script for Reviewers
=========================================================================
Paper: "HypatiaX: A Hybrid Symbolic-Neural Framework for
        Extrapolation-Reliable Analytical Discovery"  (JMLR v3.0, Apr 2026)

This script reproduces all paper results on a local machine.
It is self-contained: it patches requirements.txt and loads the API key
from a .env file automatically — no manual setup beyond step 1 below.

Quick start (3 steps)
---------------------
  1. Create a .env file in this directory:
         echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env

  2. Install dependencies (first run only):
         pip install -r requirements.txt

  3. Run:
         python3 reproduce_local.py               # recommended (skips slow steps, ~2–4 h)
         python3 reproduce_local.py --full        # all experiments including slow (~15–25 h)
         python3 reproduce_local.py --only exp3   # single step

Usage
-----
    python3 reproduce_local.py                  # fast mode: skip Feynman, noise, instability
    python3 reproduce_local.py --full           # full paper-quality run (all experiments)
    python3 reproduce_local.py --only exp3      # run one step by id
    python3 reproduce_local.py --continue-on-fail  # log failures but keep going

Step IDs (use with --only):
    Setup   : deps  patches-gen  patches-apply  validate  check-hypatiax-protocols
    Phase 1 : exp1  exp1b  exp2  exp3  exp3b
    Phase 2 : suppB  suppA  instability  extrap
    Phase 3 : provenance  verify  hashlock
    Phase 4 : figures  tables

Wall-time estimates (--skip-slow / default)
-------------------------------------------
    deps + setup     :  ~5 min
    exp1  (DeFi 74)  :  ~2–4 h
    exp1b (seed sw.) :  ~30–60 min
    exp3  (Nguyen)   :  ~30–90 min
    exp3b (seed 123) :  ~30–90 min
    suppA (routing)  :  ~30–60 min
    extrap           :  ~20–40 min
    figures + tables :  ~10 min
    ─────────────────────────────
    Total (fast)     :  ~4–8 h
    Total (--full)   :  ~15–25 h

Exit codes
----------
    0  All steps passed
    1  One or more steps failed
    2  Fatal configuration error
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Locate repo root ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
os.chdir(REPO_ROOT)

# ── 1. Load ANTHROPIC_API_KEY from .env (if not already in environment) ───────
#    Priority: existing env → .env file in repo root
def _load_api_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return   # already set — nothing to do
    dotenv = REPO_ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY=") and not line.startswith("#"):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    os.environ["ANTHROPIC_API_KEY"] = key
                    return
    # Also try hypatiax/config_secrets.py (Kaggle / Colab paths)
    try:
        import hypatiax.config_secrets  # noqa: F401  side-effect: sets ANTHROPIC_API_KEY
    except Exception:
        pass

_load_api_key()

# ── 2. Patch requirements.txt — remove/fix lines that break local installs ────
#    • defi-risk    : private SSH-only repo, unavailable outside the author's env
#    • optimum-onnx : ==0.0.3 conflicts with transformers==5.0.0; upgrade to 0.1.0
def _patch_requirements() -> None:
    req = REPO_ROOT / "requirements.txt"
    if not req.exists():
        return
    original = req.read_text()
    patched  = original

    # Remove private SSH dep
    lines = patched.splitlines(keepends=True)
    kept  = [l for l in lines if "defi-risk" not in l]
    if len(kept) < len(lines):
        patched = "".join(kept)
        print(f"  ✂  Removed defi-risk (private SSH dep) from requirements.txt")

    # Upgrade conflicting optimum-onnx pin
    if "optimum-onnx==0.0.3" in patched:
        patched = patched.replace("optimum-onnx==0.0.3", "optimum-onnx==0.1.0")
        print(f"  ✂  Upgraded optimum-onnx 0.0.3 → 0.1.0 (compatible with transformers==5.0.0)")

    if patched != original:
        req.write_text(patched)

_patch_requirements()


# ── Step registry ─────────────────────────────────────────────────────────────
@dataclass
class Step:
    id: str
    label: str
    cmd: list[str]
    phase: str
    slow: bool = False                   # skipped unless --full
    env_extra: dict = field(default_factory=dict)
    expected: str = ""                   # expected result shown in log


STEPS: list[Step] = [
    # ── Phase 0: Setup ──────────────────────────────────────────────────────
    Step("deps",          "Install dependencies",
         ["pip", "install", "-q", "-r", "requirements.txt"],
         phase="0 · Setup"),

    Step("patches-gen",   "Generate patches",
         ["python3", "scripts/patches/generate_patches.py"],
         phase="0 · Setup"),

    Step("patches-apply", "Apply patches (FIX-C1…FIX-5b)",
         ["python3", "scripts/patches/apply_patches.py"],
         phase="0 · Setup"),

    Step("validate",      "Validate patched source",
         ["python3", "scripts/patches/validate_code.py"],
         phase="0 · Setup"),

    Step("check-hypatiax-protocols",
         "Verify hypatiax/protocols/ input-data modules",
         ["python3", "scripts/patches/check_hypatiax_protocols.py"],
         phase="0 · Setup",
         expected="All 9 hypatiax/protocols/ input-data modules present"),

    # ── Phase 1: Core experiments ────────────────────────────────────────────
    Step("exp1",
         "Exp 1 · DeFi 74-task benchmark v3.0 (§10.2–10.4, §10.6)",
         ["python3", "protocols/experiment_protocol_ablation_exp1.py"],
         phase="1 · Core experiments",
         expected="89.2% R²>0.99 · 0 catastrophic · 1.73× speedup"),

    Step("exp1b",
         "Exp 1b · Portfolio Variance seed sweep (§10.5)",
         ["python3", "protocols/experiment_protocol_defi_v3.py",
          "--task", "portfolio_variance",
          "--seeds", "42", "99", "123", "777", "2024"],
         phase="1 · Core experiments",
         expected="P(H>P) ≈ 0.76"),

    Step("exp2",
         "Exp 2 · Feynman 30-equation extrapolation (§10.7)",
         ["python3", "protocols/experiment_protocol_feynman_exp2.py"],
         phase="1 · Core experiments",
         slow=True,
         expected="9/30 (30%)  [wall time 4–8 h]"),

    Step("exp3",
         "Exp 3 · Nguyen-12 SEED=42 (§10.8 primary)",
         ["python3", "protocols/experiment_protocol_nguyen12_exp3.py"],
         phase="1 · Core experiments",
         expected="11/12 H (91.7%) · 10/12 P · MW U=113, p=0.0097"),

    Step("exp3b",
         "Exp 3b · Nguyen-12 SEED=123 (§10.8 stability check)",
         ["python3", "protocols/experiment_protocol_nguyen12_exp3.py",
          "--seed", "123"],
         phase="1 · Core experiments",
         expected="consistent with SEED=42"),

    # ── Phase 2: Supplementary benchmarks ───────────────────────────────────
    Step("suppB",
         "Supp B · Noise & sample-complexity sweep",
         ["python3", "protocols/experiment_protocol_noise_sweep.py"],
         phase="2 · Supplementary benchmarks",
         slow=True,
         expected="EHD 100% at all σ · plateau ≈ N=500"),

    Step("suppA",
         "Supp A · Hybrid routing improvements (Fix 1–5b)",
         ["python3", "protocols/experiment_protocol_hybrid_routing.py"],
         phase="2 · Supplementary benchmarks",
         expected="+6pp Fix1, +5pp Fix2, +1pp Fix3"),

    Step("instability",
         "§10.9 · Stability under stochastic inference (K=30)",
         ["python3", "protocols/experiment_protocol_instability_rf02_04.py"],
         phase="2 · Supplementary benchmarks",
         slow=True,
         env_extra={"LLM_K_RUNS": "30"},
         expected="Spearman ρ=−0.70, p<0.001 · 70 tasks · C-Collapse anomaly (RF-06)"),

    Step("extrap",
         "§10.8 · Extrapolation comparative (near/med/far OOD)",
         ["python3", "protocols/experiment_protocol_extrapolation_comparative.py"],
         phase="2 · Supplementary benchmarks"),

    # ── Phase 3: Audit & verification ───────────────────────────────────────
    Step("provenance",
         "§11 · Provenance audit",
         ["python3", "protocols/experiment_protocol_provenance_audit.py"],
         phase="3 · Audit & verification"),

    Step("verify",
         "Verify results against paper targets",
         ["python3", "scripts/patches/verify_results.py", "--report", "--json"],
         phase="3 · Audit & verification"),

    Step("hashlock",
         "Hash lock check",
         ["python3", "reproducibility/hash_lock.py", "--check"],
         phase="3 · Audit & verification"),

    # ── Phase 4: Outputs ─────────────────────────────────────────────────────
    Step("figures",
         "Generate all figures",
         ["python3", "figures/generate_figures.py",
          "--outdir", "hypatiax/data/results/figures"],
         phase="4 · Outputs"),

    Step("tables",
         "Generate all tables",
         ["python3", "scripts/patches/generate_tables.py",
          "--outdir", "hypatiax/data/results/tables"],
         phase="4 · Outputs"),

    # ── Phase 4-B: Paper audit notebooks ─────────────────────────────────────
    # Copies all three .tex files into notebooks/ so NB-01..05 can scan them.
    # Searches paper/ subdir first, then repo root (where they currently live).
    # Files:
    #   • main paper    : jmlr-hypatiax*.tex  or  jmlr_paper*.tex
    #   • Supp A        : supp_routing_improvements.tex
    #   • Supp B        : supp_benchmark_report.tex
    Step("audit-setup",
         "Paper audit · Copy main paper + supplements into notebooks/",
         ["python3", "-c",
          "import shutil, pathlib; "
          "nb = pathlib.Path('notebooks'); nb.mkdir(exist_ok=True); "
          "search_dirs = [pathlib.Path('paper'), pathlib.Path('.')]; "
          "copied = []; "
          # ── main paper ──
          "main = next((f for d in search_dirs "
          "             for pat in ('jmlr-hypatiax*.tex','jmlr_paper*.tex') "
          "             for f in d.glob(pat) if f.is_file()), None); "
          "(shutil.copy(main, nb / main.name), copied.append(main.name)) "
          "  if main else print('WARNING: main paper .tex not found'); "
          # ── supplements ──
          "[shutil.copy(src, nb / name) or copied.append(name) "
          " for name in ('supp_routing_improvements.tex','supp_benchmark_report.tex') "
          " for src in [next((d/name for d in search_dirs if (d/name).is_file()), None)] "
          " if src]; "
          "print(f'audit-setup: copied {len(copied)} file(s) to notebooks/: {copied}')"],
         phase="4-B · Paper audit"),

    Step("audit-NB-01",
         "Paper audit · NB-01 Citation & Bibliography",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-01_Citation_Bibliography_Audit.ipynb"],
         phase="4-B · Paper audit"),

    Step("audit-NB-02",
         "Paper audit · NB-02 Cross-Reference & Label",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-02_CrossReference_Label_Audit.ipynb"],
         phase="4-B · Paper audit"),

    Step("audit-NB-03",
         "Paper audit · NB-03 Section Structure & Numbering",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-03_Section_Structure_Numbering.ipynb"],
         phase="4-B · Paper audit"),

    Step("audit-NB-04",
         "Paper audit · NB-04 Numerical Consistency",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-04_Numerical_Consistency_Checker.ipynb"],
         phase="4-B · Paper audit"),

    Step("audit-NB-05",
         "Paper audit · NB-05 Figure & Image Dependencies",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-05_Figure_Image_Dependency_Checker.ipynb"],
         phase="4-B · Paper audit"),
]


# ── Result tracking ───────────────────────────────────────────────────────────
@dataclass
class Result:
    id: str
    label: str
    status: str           # "pass" | "fail" | "skip"
    elapsed: float = 0.0
    log_path: Optional[Path] = None
    returncode: int = 0


# ── Runner ────────────────────────────────────────────────────────────────────
def run_step(step: Step, log_dir: Path, env: dict) -> Result:
    log_path = log_dir / f"{step.id}.log"
    merged_env = {**env, **step.env_extra}

    print(f"\n┌─── [{step.id}] {step.label}")
    print(f"│    {time.strftime('%H:%M:%S')}")
    if step.expected:
        print(f"│    Expected : {step.expected}")
    if step.env_extra:
        for k, v in step.env_extra.items():
            print(f"│    env+     : {k}={v}")
    print(f"│    cmd      : {' '.join(step.cmd)}")

    t0 = time.time()
    try:
        with open(log_path, "w") as log_fh:
            proc = subprocess.run(
                step.cmd,
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log_fh.write(proc.stdout)
            # Stream last 20 lines to console
            for line in proc.stdout.splitlines()[-20:]:
                print(f"│  {line}")

        elapsed = time.time() - t0
        ok = proc.returncode == 0
        sym = "✓" if ok else "✗"
        tail = f"  — see logs/{step.id}.log" if not ok else ""
        print(f"└─── {sym} {'done' if ok else 'FAILED'}  ({elapsed:.0f}s){tail}")
        return Result(step.id, step.label,
                      "pass" if ok else "fail",
                      elapsed, log_path, proc.returncode)

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"└─── ✗ ERROR: {exc}")
        return Result(step.id, step.label, "fail", elapsed, log_path)


def banner(msg: str) -> None:
    print("\n" + "═" * 68)
    print(f"  {msg}")
    print("═" * 68)


# ── Summary ───────────────────────────────────────────────────────────────────
def _print_summary(results: list[Result], elapsed: float) -> None:
    passed  = [r for r in results if r.status == "pass"]
    failed  = [r for r in results if r.status == "fail"]
    skipped = [r for r in results if r.status == "skip"]

    hh, rem = divmod(int(elapsed), 3600)
    mm, ss  = divmod(rem, 60)

    banner("Reproduction summary")
    sym = {"pass": "✓", "fail": "✗", "skip": "─"}
    for r in results:
        t = f"  {r.elapsed:6.0f}s" if r.status != "skip" else "        "
        print(f"  {sym[r.status]} [{r.id:28s}] {r.label[:48]:48s}{t}")

    print()
    print(f"  ✓ passed : {len(passed)}")
    print(f"  ✗ failed : {len(failed)}")
    print(f"  ─ skipped: {len(skipped)}")
    print(f"  Wall time: {hh:02d}:{mm:02d}:{ss:02d}")
    print()

    results_dir = REPO_ROOT / "hypatiax" / "data" / "results"
    if results_dir.exists():
        data  = list(results_dir.rglob("*.json")) + list(results_dir.rglob("*.csv"))
        figs  = list(results_dir.rglob("*.pdf"))
        tbls  = list(results_dir.rglob("*.tex"))
        print(f"  Results → {results_dir}")
        print(f"    Data files (JSON+CSV) : {len(data)}")
        print(f"    Figures (PDF)         : {len(figs)}")
        print(f"    Tables  (TeX)         : {len(tbls)}")
        print()

    if failed:
        print("  Failed steps:")
        for r in failed:
            print(f"    [{r.id}] → {r.log_path}")
        print()
        print("  Re-run a single step:")
        print("    python3 reproduce_local.py --only <id>")
    else:
        print("  ✓ All steps passed. Results ready for verification.")
        print(f"  Results    : hypatiax/data/results/")
        print(f"  Figures    : hypatiax/data/results/figures/")
        print(f"  Tables     : hypatiax/data/results/tables/")
        print(f"  Logs       : logs/")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="HypatiaX · Local reproduction script for reviewers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--full", action="store_true",
                        help="Run ALL experiments including slow steps "
                             "(Feynman exp2, noise sweep suppB, instability). "
                             "Default: slow steps are skipped.")
    parser.add_argument("--only", metavar="ID",
                        help="Run only this step id (e.g. exp3)")
    parser.add_argument("--continue-on-fail", action="store_true",
                        help="Log failures but continue remaining steps")
    parser.add_argument("--seed", type=int, default=42, metavar="N",
                        help="Override random seed for all steps (default: 42)")
    args = parser.parse_args()

    skip_slow = not args.full   # slow steps are skipped by default

    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    # ── Header ────────────────────────────────────────────────────────────────
    banner("HypatiaX · Local Reproduction Script  (JMLR Apr 2026)")
    print(f"  Repo      : {REPO_ROOT}")
    print(f"  Python    : {sys.version.split()[0]}")
    print(f"  Date      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode      : {'FULL (all experiments)' if args.full else 'FAST (slow steps skipped)'}")
    print(f"  Seed      : {args.seed}")

    # ── API key check ─────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("\n  ERROR: ANTHROPIC_API_KEY is not set.")
        print("  Create a .env file in the repo root:")
        print("    echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env")
        print("  Or export it directly:")
        print("    export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(2)
    print(f"  API key   : set ({len(api_key)} chars) ✓")

    # ── Protocol module check ─────────────────────────────────────────────────
    hypatiax_proto = REPO_ROOT / "hypatiax" / "protocols"
    required = [
        "experiment_protocol_defi.py",
        "experiment_protocol_defi_20.py",
        "experiment_protocol_nguyen12.py",
        "experiment_protocol_all_18_a.py",
        "experiment_protocol_all_20.py",
        "experiment_protocol_all_30.py",
        "experiment_protocol_benchmark.py",
        "experiment_protocol_benchmark_v2.py",
        "experiment_protocol_comparative.py",
    ]
    missing = [f for f in required if not (hypatiax_proto / f).exists()]
    if missing:
        print(f"\n  ERROR: {len(missing)} input-data module(s) missing from hypatiax/protocols/:")
        for f in missing:
            print(f"    ✗  {f}")
        sys.exit(2)
    print(f"  Protocols : all {len(required)} hypatiax/protocols/ modules ✓")

    # ── Build environment ─────────────────────────────────────────────────────
    seed_str = str(args.seed)
    env = {**os.environ}
    env["NN_SEED"]               = seed_str
    env["PYSR_SEED"]             = seed_str
    env["PYTHONHASHSEED"]        = seed_str
    env.setdefault("LLM_MODEL",             "claude-sonnet-4-20250514")
    env.setdefault("LLM_RETRIES",           "3")
    env.setdefault("LLM_K_RUNS",            "1")   # overridden to 30 for instability
    env.setdefault("N_TASKS_DEFI",          "74")
    env.setdefault("N_TASKS_INSTABILITY",   "70")  # FIX-T1: must be 70
    env.setdefault("PCA_TRAIN_FRAC",        "0.40")
    env.setdefault("NN_TIME_LIMIT",         "120")
    env.setdefault("ENGINE_NAME",           "hybrid_system_v50_2")  # FIX-C2
    env.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    env["PYTHONPATH"]  = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["RESULTS_DIR"] = str(REPO_ROOT / "hypatiax" / "data" / "results")
    env["REPRO_ROOT"]  = str(REPO_ROOT)

    print(f"  NN_SEED={seed_str}  PYSR_SEED={seed_str}  PYTHONHASHSEED={seed_str}")
    print(f"  LLM_MODEL={env['LLM_MODEL']}")
    print(f"  ENGINE={env['ENGINE_NAME']}  N_TASKS_INSTABILITY={env['N_TASKS_INSTABILITY']}")

    # ── Validate --only ───────────────────────────────────────────────────────
    step_ids = [s.id for s in STEPS]
    if args.only and args.only not in step_ids:
        print(f"\n  ERROR: unknown step id '{args.only}'.")
        print(f"  Valid ids: {', '.join(step_ids)}")
        sys.exit(2)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    results: list[Result] = []
    current_phase = ""
    t_total = time.time()

    for step in STEPS:
        if step.phase != current_phase:
            banner(f"Phase {step.phase}")
            current_phase = step.phase

        if args.only and step.id != args.only:
            results.append(Result(step.id, step.label, "skip"))
            print(f"  ── skip [{step.id}]  (--only {args.only})")
            continue

        if skip_slow and step.slow:
            results.append(Result(step.id, step.label, "skip"))
            print(f"  ── skip [{step.id}]  (slow step — use --full to run)")
            continue

        result = run_step(step, log_dir, env)
        results.append(result)

        if result.status == "fail" and not args.continue_on_fail:
            print(f"\n  Pipeline aborted at [{step.id}].")
            print(f"  Re-run this step alone : python3 reproduce_local.py --only {step.id}")
            print(f"  Or keep going          : python3 reproduce_local.py --continue-on-fail")
            _print_summary(results, time.time() - t_total)
            sys.exit(1)

    _print_summary(results, time.time() - t_total)
    sys.exit(1 if any(r.status == "fail" for r in results) else 0)


if __name__ == "__main__":
    main()
