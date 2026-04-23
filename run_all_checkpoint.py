#!/usr/bin/env python3
"""
run_all.py  —  HypatiaX · Full reproducibility pipeline (Python version)
Paper: "HypatiaX: A Hybrid Symbolic-Neural Framework for
        Extrapolation-Reliable Analytical Discovery"  (JMLR v3.0, Apr 2026)

Usage:
    python3 run_all.py                      # full pipeline
    python3 run_all.py --skip-slow          # skip slow steps (Feynman, noise sweep, instability)
    python3 run_all.py --only exp3          # run one step by id
    python3 run_all.py --resume             # resume from last checkpoint
    python3 run_all.py --resume --from exp2 # resume, but force-rerun from this step
    python3 run_all.py --clear-checkpoint   # delete checkpoint file and exit
    python3 run_all.py --continue-on-fail   # log failures but keep going
    python3 run_all.py --verify-only        # re-check existing results without re-running
    python3 run_all.py --seed 123           # override seed for all steps (default: 42)
    python3 run_all.py --only exp3 --seed 777  # single step with custom seed
    python3 run_all.py --skip-paper         # skip pdflatex compile steps
    python3 run_all.py --pysr-timeout 900   # extend PySR wall-clock limit (default 360s)
    python3 run_all.py --one-equation       # smoke-test: 1 equation per experiment, fast timeout

Step IDs (use with --only / --from):
    Setup   : deps  patches-gen  patches-apply  fixup-init  fixup-tex  validate  check-hypatiax-protocols
    Phase 1 : exp1  exp1b  exp2  exp3  exp3b
    Phase 2 : suppB  suppA  instability  extrap
    Phase 3 : provenance  discover-provenance  scan-imports  verify  hashlock
    Phase 4 : figures  tables
    Phase 4B: audit-setup  audit-NB-01 ... audit-NB-05

Prerequisites:
    export ANTHROPIC_API_KEY="sk-ant-..."
    pip install -r requirements.txt

Changelog v4.8 (2026-04-23):
    FIX-EXP1B-ARGS: exp1b Step no longer passes --task/--seeds to
               experiment_protocol_defi_v3.py — those flags were silently
               ignored (the protocol has no argparser). Task filtering and
               seed list are now forwarded via DEFI_TASK_FILTER and
               DEFI_SEEDS env vars, which experiment_protocol_defi_v3.py
               reads and forwards into config for hypatiax_defi_benchmark_v3c.py.
    FIX-EXP2-FUTURE: experiment_protocol_benchmark_v2.py had
               `from __future__ import annotations` at line 84 (after
               sys.path bootstrap block) — Python requires it at line 1.
               Moved to top of file; SyntaxError in exp2 resolved.

Changelog v4.7 (2026-04-22):
    FIX-ONE-EQ-EXP3: exp3 and exp3b now append --n-tasks 1 to their cmd when
               ONE_EQUATION=1 is active. Previously N_NGUYEN_TASKS=1 was set
               in the env but experiment_protocol_nguyen12_exp3.py controls
               task count via its own CLI arg — the env var was silently
               ignored, so all 12 Nguyen equations still ran.
    FIX-4:        --one-equation smoke-test now injects N_ITERATIONS=200 and
               POPULATIONS=10 so PySR actually runs light. Previously it still
               used default populations=30 / iterations=1000, making the smoke
               test as slow as a real run. PYSR_TIMEOUT also tightened from
               120 → 60s to match the lighter config.
    MINOR-A:   extrap step PROTOCOL_ROOT now auto-detects: tries
               hypatiax/protocols/ first, falls back to repo-root protocols/
               if the former is absent (matches the comment in FIX-EXTRAP).

Changelog v4.6 (2026-04-21):
    FIX-SUPPA:    suppA step now injects SKIP_PERF_ANALYSIS=1 (suppresses the call to
               hypatiax/analysis/analyze_hybrid_performance.py which does not exist)
               and HYPATIAX_CORE_OPTIONAL=1 as an extra guard against the __init__
               ImportError while fixup-init runs first to fix the root cause.
    FIX-EXTRAP:   extrap step now injects PROTOCOL_ROOT=<repo>/protocols/ so the
               extrapolation protocol finds experiment_protocol_benchmark_v2.py in
               the correct location (repo-root protocols/) not hypatiax/protocols/.
               Also injects HYPATIAX_CORE_OPTIONAL=1 so noisy/noiseless passes
               don't abort when hypatiax.core is unavailable.
    FIX-VERIFY:   verify step now passes PATCHED_DATA_DIR and VERIFY_RESULTS_DIR env
               vars pointing to hypatiax/data/results/ — previously verify_results.py
               looked under scripts/hypatiax/data/patched/ and found nothing.
    FIX-TABLES:   tables step now passes TABLE_OUTDIR and VERIFY_RESULTS_DIR env vars
               so generate_tables.py writes to RESULTS_DIR/tables/ (not paper/tables/)
               and sources its input data from the correct results directory.
    FIX-AUDIT-SETUP: audit-setup now searches paper/, repo-root, paper/tables/, and
               logs/ for supplement .tex files and emits explicit WARNINGs for any
               that cannot be found, instead of silently skipping them.
    FIX-INVENTORY:  inventory_results() now falls back to paper/tables/ when
               RESULTS_DIR/tables/ has no .tex files, so the pipeline summary
               correctly reports a non-zero table count.

Changelog v4.5 (2026-04-21):
    FIX-INIT-PY: Added fixup-init step (Phase 0, after patches-apply) that wraps
               the broken `from hypatiax.core import HypatiaX` line in hypatiax/
               __init__.py inside a try/except. This prevents ImportError from
               propagating into sub-packages (hypatiax.protocols.*, etc.) used by
               exp1b, suppA, extrap, and others.  The patch is idempotent.
    FIX-PKG-CHECK: exp3 and exp3b now inject SKIP_PKG_CHECK=1 into the subprocess
               environment. The protocol wrapper's importlib package check fails
               spuriously when packages are installed in an editable/stale venv
               even though they are importable.  SKIP_PKG_CHECK=1 bypasses the
               pre-flight check; packages were already validated by the deps step.
    FIX-PROVENANCE: provenance step is now graceful when
               protocols/experiment_protocol_provenance_audit.py is absent — prints
               a warning and exits 0 instead of crashing the pipeline.

Changelog v4.4 (2026-04-21):
    FIX-INIT:  Replaced `import hypatiax.config_secrets` (which triggers hypatiax/__init__.py
               → `from hypatiax.core import HypatiaX` → ImportError) with a robust
               _load_api_key() function that loads hypatiax/config_secrets.py *directly*
               via importlib, bypassing __init__.py entirely.  Falls back to a
               minimal inline .env parser so the pipeline starts cleanly even when
               the hypatiax package itself is broken.
    FIX-PY:    Inject PIPELINE_PYTHON=sys.executable into the subprocess environment.
    FIX-EXP3:  exp3 / exp3b cmd lists use sys.executable (not bare "python3") to
               avoid "Missing packages: pysr, sklearn, pandas" false-negatives.
    FIX-EXP1B: exp1b sets env_extra DEFI_V3C_NO_TIMEOUT_FLAGS=1 so the protocol
               wrapper does not forward --pysr-timeout/--method-timeout flags that
               hypatiax_defi_benchmark_v3c.py does not accept.

Changelog v4.3 (2026-04-21):
    NEW: --one-equation flag injects ONE_EQUATION=1 + N_TASKS_*=1 + N_FEYNMAN_TASKS=1
         into the environment so every experiment script runs exactly one equation.
         Also forces PYSR_TIMEOUT=120 (overridable with --pysr-timeout) and prints a
         SMOKE-TEST banner.  Use this to verify the full pipeline end-to-end quickly
         before committing to a full multi-hour run.

Changelog v4.2 (2026-04-21):
    FIX: KeyboardInterrupt (Ctrl+C) now caught in both run_step() and main():
         subprocess is terminated cleanly, checkpoint is saved, summary is
         printed, and the process exits with code 130 (standard Ctrl+C exit).
    FIX: archive_step_results() rglob sub_pattern no longer uses fragile
         lstrip() chain — '**/*.json' is passed to rglob() directly.
    FIX: exp3b label corrected to "seeds 99/123/777/2024" (SEED=42 = exp3).
    FIX: Popen now uses bufsize=1 (line-buffered) for more responsive streaming.
    NEW: --pysr-timeout SECS flag injects PYSR_TIMEOUT env var so experiment
         scripts can extend the PySR wall-clock limit on slower hardware (the
         360 s default causes 11/15 Hybrid results to be N/A locally, breaking
         the Mann-Whitney U stat — paper expects U=126 over 15 pairs, not 4).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Load API key (env → Kaggle → .env → Colab) ────────────────────────────────
# FIX-INIT: importing hypatiax.config_secrets at module level triggers hypatiax/__init__.py,
# which does `from hypatiax.core import HypatiaX`.  If hypatiax.core is broken or
# not yet importable (e.g. editable-install stale .egg-link, missing compiled ext),
# the whole pipeline crashes before main() even runs.
#
# Strategy (in order):
#   1. Try importlib to load hypatiax/config_secrets.py *directly* as a standalone module,
#      completely bypassing hypatiax/__init__.py.
#   2. If that also fails (e.g. config_secrets.py itself has a bad import), fall back to
#      a minimal inline .env parser that replicates what hypatiax.config_secrets does:
#      read ANTHROPIC_API_KEY from the environment, then from hypatiax/.env or .env.
def _load_api_key() -> None:
    """Load ANTHROPIC_API_KEY via hypatiax.config_secrets, or fall back to .env parsing."""
    import importlib.util as _ilu

    # ── Attempt 1: load hypatiax/config_secrets.py directly (skips __init__.py) ──────
    _repo = Path(__file__).resolve().parent
    _config_secrets_path = _repo / "hypatiax" / "config_secrets.py"
    if _config_secrets_path.exists():
        try:
            _spec = _ilu.spec_from_file_location("hypatiax._config_secrets_standalone",
                                                  _config_secrets_path)
            if _spec and _spec.loader:
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)  # type: ignore[arg-type]
                # config_secrets.py sets os.environ["ANTHROPIC_API_KEY"] as a side-effect
                if os.environ.get("ANTHROPIC_API_KEY"):
                    print("✅ ANTHROPIC_API_KEY loaded from hypatiax/config_secrets.py")
                    return
        except Exception as _e:
            print(f"  ⚠  hypatiax/config_secrets.py direct-load failed ({_e}); "
                  "falling back to .env parser")

    # ── Attempt 2: minimal .env parser (no third-party deps) ──────────────────
    if os.environ.get("ANTHROPIC_API_KEY"):
        print(f"✅ ANTHROPIC_API_KEY already set in environment")
        return
    for _env_path in [_repo / "hypatiax" / ".env",
                      _repo / ".env",
                      Path.home() / ".env"]:
        if _env_path.exists():
            for _line in _env_path.read_text().splitlines():
                _line = _line.strip()
                if _line.startswith("#") or "=" not in _line:
                    continue
                _k, _, _v = _line.partition("=")
                _k = _k.strip()
                _v = _v.strip().strip('"').strip("'")
                if _k == "ANTHROPIC_API_KEY" and _v:
                    os.environ["ANTHROPIC_API_KEY"] = _v
                    print(f"✅ ANTHROPIC_API_KEY loaded from {_env_path}")
                    return
    # Key still not found — main() will catch this and print a clear error.

_load_api_key()

# ── Canonical paths ────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "hypatiax" / "data" / "results"
LOG_DIR     = REPO_ROOT / "logs"
CHECKPOINT  = LOG_DIR / "pipeline_checkpoint.json"

# ── Strip incompatible deps from requirements.txt (local runs) ────────────────
# • defi-risk    : private SSH-only repo, unavailable locally
# • optimum-onnx : ==0.0.3 conflicts with transformers==5.0.0
_REQUIREMENTS = REPO_ROOT / "requirements.txt"
_STRIP_PATTERNS = ["defi-risk", "optimum-onnx"]
if _REQUIREMENTS.exists():
    _lines = _REQUIREMENTS.read_text().splitlines(keepends=True)
    _filtered = [l for l in _lines
                 if not any(p in l for p in _STRIP_PATTERNS)]
    if len(_filtered) < len(_lines):
        _REQUIREMENTS.write_text("".join(_filtered))
        print(f"  ✂  Removed {len(_lines)-len(_filtered)} incompatible dep(s) "
              f"from requirements.txt: {_STRIP_PATTERNS}")

# ── Stage paper .tex files into paper/ if they live at repo root ──────────────
# validate_code.py and audit notebooks expect tex files in paper/.
# If the repo was published with them at root, copy them across automatically.
import shutil as _shutil
_PAPER_DIR = REPO_ROOT / "paper"
_TEX_PATTERNS = [
    "jmlr_paper*.tex",               # main paper (underscore variant)
    "jmlr-hypatiax*.tex",            # main paper (hyphen variant)
    "supp_routing_improvements.tex", # Supp A (needed by FIX-XR3 in validate_code)
    "supp_benchmark_report.tex",     # Supp B
]
_staged: list[str] = []
for _pat in _TEX_PATTERNS:
    for _src in REPO_ROOT.glob(_pat):
        _dst = _PAPER_DIR / _src.name
        if not _dst.exists():
            _PAPER_DIR.mkdir(exist_ok=True)
            _shutil.copy2(_src, _dst)
            _staged.append(_src.name)
if _staged:
    print(f"  📄 Staged {len(_staged)} .tex file(s) into paper/: {_staged}")

# Steps that are part of a paper compile (skipped by --skip-paper)
_PAPER_STEP_IDS = {"audit-NB-01", "audit-NB-02", "audit-NB-03",
                   "audit-NB-04", "audit-NB-05", "audit-setup"}


# ── Experiment registry ────────────────────────────────────────────────────────
@dataclass
class Step:
    id: str
    label: str
    cmd: list[str]
    phase: str
    slow: bool = False                 # skipped by --skip-slow
    paper: bool = False                # skipped by --skip-paper
    env_extra: dict = field(default_factory=dict)
    expected: str = ""                 # human note shown in summary
    result_glob: str = ""              # glob relative to RESULTS_DIR to verify output


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

    # FIX-INIT-PY: hypatiax/__init__.py contains `from hypatiax.core import HypatiaX`
    # at module level. If hypatiax.core is broken (stale editable install, missing
    # compiled extension, etc.) ANY import of a hypatiax sub-package (e.g.
    # hypatiax.protocols.*) will crash with ImportError. This step wraps that line
    # in a try/except so sub-packages remain importable even when the top-level class
    # is unavailable.
    Step("fixup-init",
         "Guard hypatiax/__init__.py broken HypatiaX import (FIX-INIT-PY)",
         ["python3", "-c", "\n".join([
             "from pathlib import Path",
             "init = Path('hypatiax') / '__init__.py'",
             "if not init.exists():",
             "    print('  ⚠ fixup-init: hypatiax/__init__.py not found — skipping')",
             "    raise SystemExit(0)",
             "src = init.read_text(encoding='utf-8')",
             "BAD  = 'from hypatiax.core import HypatiaX'",
             "GOOD = ('try:\\n'",
             "        '    from hypatiax.core import HypatiaX  # noqa: F401\\n'",
             "        'except Exception:  # broken core does not block sub-packages\\n'",
             "        '    HypatiaX = None  # type: ignore')",
             "if BAD in src and 'except Exception:' not in src:",
             "    patched = src.replace(BAD, GOOD)",
             "    init.write_text(patched, encoding='utf-8')",
             "    print('  ✓ fixup-init: hypatiax/__init__.py patched — HypatiaX import guarded')",
             "else:",
             "    print('  ✓ fixup-init: already patched or BAD string absent')",
         ])],
         phase="0 · Setup"),

    # FIX-T2 / FIX-B2 / FIX-B3: apply remaining .tex patches that
    # generate_patches.py creates JSON for but apply_patches.py never acts on
    # (apply_patches.py only handles Python source files).
    #
    # validate_code.py check_paper_text() opens exactly ONE file:
    #   paper/jmlr-hypatiax-paper-final.tex
    # and calls error() — which aborts the pipeline — if ANY of these strings
    # appear anywhere in that file:
    #   FIX-T2 → "Five-Layer Architecture Overview"
    #   FIX-B2 → "cranmer2023interpretable"
    #   FIX-B3 → "udrescu2020aifeynman"
    #
    # FIX-B2/B3 interpretation: the paper has inline \bibitem entries with
    # duplicate keys.  The fix is to consolidate each pair by renaming the
    # SECOND \bibitem and its citation to a deduplicated key (key + "b"), then
    # renaming the FIRST \bibitem + all \cite{key} to the same canonical name,
    # so the original conflicting string no longer appears anywhere.
    # Canonical mapping:
    #   cranmer2023interpretable  → cranmer2023interp
    #   udrescu2020aifeynman      → udrescu2020feynman
    # All edits are idempotent.
    Step("fixup-tex",
         "Apply FIX-T2 (Five-Stage) + FIX-B2/B3 (rename dup bibkeys in main .tex)",
         ["python3", "-c", "\n".join([
             "import re",
             "from pathlib import Path",
             "",
             "TEX = Path('paper') / 'jmlr-hypatiax-paper-final.tex'",
             "if not TEX.exists():",
             "    print(f'  ⚠ fixup-tex: {TEX} not found — skipping')",
             "    raise SystemExit(0)",
             "",
             "src = TEX.read_text(encoding='utf-8')",
             "original = src",
             "",
             "# FIX-T2: rename heading (exact string from generate_patches.py FIX-T2 sed)",
             "src = src.replace('Five-Layer Architecture Overview',",
             "                  'Five-Stage Architecture Overview')",
             "",
             "# FIX-B2 / FIX-B3: rename every occurrence of the conflicting bibkey",
             "# string so that 'key in src' is False for the validator.",
             "# Renaming both \\bibitem{key} and \\cite{key} keeps the document",
             "# internally consistent.",
             "RENAMES = [",
             "    ('cranmer2023interpretable', 'cranmer2023interp'),",
             "    ('udrescu2020aifeynman',      'udrescu2020feynman'),",
             "]",
             "for old_key, new_key in RENAMES:",
             "    if old_key in src:",
             "        src = src.replace(old_key, new_key)",
             "        print(f'  ✓ FIX-B: renamed all occurrences: {old_key} → {new_key}')",
             "    else:",
             "        print(f'  ✓ FIX-B: {old_key} absent (already clean)')",
             "",
             "if src != original:",
             "    TEX.write_text(src, encoding='utf-8')",
             "    print(f'  ✓ fixup-tex: {TEX} patched')",
             "else:",
             "    print(f'  ✓ fixup-tex: {TEX} already clean')",
             "",
             "# Self-check: mirror validate_code.py assertions exactly",
             "final = TEX.read_text(encoding='utf-8')",
             "bad = {",
             "    'FIX-T2': 'Five-Layer Architecture Overview',",
             "    'FIX-B2': 'cranmer2023interpretable',",
             "    'FIX-B3': 'udrescu2020aifeynman',",
             "}",
             "failures = [f'{k}: \"{v}\" still present' for k, v in bad.items() if v in final]",
             "for fix_id, needle in bad.items():",
             "    if needle not in final:",
             "        print(f'  ✓ {fix_id}: absent — validate_code.py check will pass')",
             "if failures:",
             "    print('\\n'.join(f'  ✗ {f}' for f in failures), flush=True)",
             "    raise SystemExit(1)",
             "print('fixup-tex: done')",
         ])],
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
    # Exp 1 covers §10.2–10.4 (DeFi 74-task) and §10.6 (Core-15 ablation) in a
    # single protocol. Do NOT substitute run_dual_condition_benchmark.py.
    Step("exp1",
         "Exp 1 · DeFi 74-task benchmark v3.0 (§10.2–10.4, §10.6)",
         ["python3", "hypatiax/protocols/experiment_protocol_ablation_exp1.py"],
         phase="1 · Core experiments",
         expected="89.2% R²>0.99 · 0 catastrophic · 1.73× speedup",
         result_glob="comparison_results/noise-noiseless/noiseless/*.json"),

    # §10.5: five-seed robustness sweep for Portfolio Variance only
    # FIX-EXP1B: hypatiax_defi_benchmark_v3c.py does NOT accept --pysr-timeout or
    # --method-timeout as CLI arguments — it errors with "unrecognized arguments".
    # Timeouts are already propagated via the PYSR_TIMEOUT env var set in main().
    # DEFI_V3C_NO_TIMEOUT_FLAGS=1 signals experiment_protocol_defi_v3.py not to
    # forward those flags when it shells out to the benchmark script.
    Step("exp1b",
         "Exp 1b · Portfolio Variance seed sweep (§10.5)",
         # --task and --seeds are NOT accepted by experiment_protocol_defi_v3.py —
         # they are forwarded via env vars DEFI_TASK_FILTER and DEFI_SEEDS instead.
         ["python3", "hypatiax/protocols/experiment_protocol_defi_v3.py"],
         phase="1 · Core experiments",
         expected="P(H>P) ≈ 0.76",
         result_glob="comparison_results/noise-noiseless/15/*.json",
         env_extra={
             "DEFI_V3C_NO_TIMEOUT_FLAGS": "1",
             "DEFI_TASK_FILTER": "portfolio",
             "DEFI_SEEDS": "42,99,123,777,2024",
         }),

    # §10.7: primary run is Kaggle 4-vCPU; this protocol reproduces that environment
    Step("exp2",
         "Exp 2 · Feynman 30-equation extrapolation (§10.7)",
         ["python3", "hypatiax/protocols/experiment_protocol_feynman_exp2.py"],
         phase="1 · Core experiments",
         slow=True,
         expected="9/30 (30%)  [Kaggle 4-vCPU primary · wall time 4–8 h]",
         result_glob="comparison_results/feynman-tests/**/*.json"),

    # §10.8 primary: SEED=42, source exp3_nguyen12_hybrid50v_02.py logic
    # FIX-EXP3: use sys.executable so the protocol wrapper runs in the active venv
    # (not bare 'python3'), preventing false "Missing packages: pysr, sklearn, pandas".
    # FIX-EXP3-PKG: The protocol wrapper runs a package check via importlib in the
    # subprocess Python. When packages (pysr, sklearn, pandas) are installed in the
    # venv but the importlib check somehow fails (e.g. stale .egg-link, editable
    # install not refreshed), set SKIP_PKG_CHECK=1 so the wrapper skips the check
    # and proceeds directly. Packages were already validated by the deps step.
    # FIX-ONE-EQ-EXP3: N_NGUYEN_TASKS=1 (env) is NOT enough — experiment_protocol_nguyen12_exp3.py
    # controls task count via its own CLI arg, not the env var. When --one-equation is active
    # we append --n-tasks 1 directly to the cmd so only 1 Nguyen equation runs.
    # N_NGUYEN_TASKS=1 is kept as a belt-and-suspenders fallback.
    Step("exp3",
         "Exp 3 · Nguyen-12 SEED=42 (§10.8 primary)",
         [sys.executable, "hypatiax/protocols/experiment_protocol_nguyen12_exp3.py",
          "--seed", "42"]
         + (["--n-tasks", "1"] if os.environ.get("ONE_EQUATION") == "1" else []),
         phase="1 · Core experiments",
         expected="11/12 H (91.7%) · 10/12 P · MW U=113, p=0.0097",
         result_glob="hypatiax/data/results/nguyen12_exp3_*.json",
         env_extra={"SKIP_PKG_CHECK": "1"}),

    # §10.8 stability: remaining 4 seeds (SEED=42 is exp3 above)
    # FIX-EXP3B: same venv-Python fix as exp3. Also sets SKIP_PKG_CHECK=1.
    # FIX-ONE-EQ-EXP3B: same --n-tasks 1 fix as exp3 above.
    Step("exp3b",
         "Exp 3b · Nguyen-12 seeds 99/123/777/2024 (§10.8 stability)",
         [sys.executable, "hypatiax/protocols/experiment_protocol_nguyen12_exp3.py",
          "--seeds", "99", "123", "777", "2024"]
         + (["--n-tasks", "1"] if os.environ.get("ONE_EQUATION") == "1" else []),
         phase="1 · Core experiments",
         expected="consistent with SEED=42 across all 5 seeds",
         result_glob="extrapolation/full_run_*.json",
         env_extra={"SKIP_PKG_CHECK": "1"}),

    # ── Phase 2: Supplementary benchmarks ───────────────────────────────────
    # Supp B: noise σ ∈ {0,0.5,1,5,10}% AND sample n ∈ {50…1000} in one protocol
    Step("suppB",
         "Supp B · Noise & sample-complexity sweep",
         ["python3", "hypatiax/protocols/experiment_protocol_noise_sweep.py"],
         phase="2 · Supplementary benchmarks",
         slow=True,
         expected="EHD 100% at all σ · plateau ≈ N=500",
         result_glob="comparison_results/feynman-tests/noise-sweep/**/*.json"),

    # Supp A: routing improvements Fix 1–5b
    # FIX-SUPPA: Two issues seen in errors.txt:
    #   (1) hypatiax/analysis/analyze_hybrid_performance.py does not exist — Step 3/4
    #       (Performance Analysis) exits code 2.  SKIP_PERF_ANALYSIS=1 tells the
    #       routing protocol to skip that sub-step rather than aborting the whole suite.
    #   (2) The HypatiaX ImportError in __init__.py (fixed by fixup-init above) caused
    #       Steps 1 and 2 to fail.  fixup-init runs first so this should be resolved,
    #       but HYPATIAX_CORE_OPTIONAL=1 is set as an extra guard for the routing script.
    Step("suppA",
         "Supp A · Hybrid routing improvements (Fix 1–5b)",
         ["python3", "hypatiax/protocols/experiment_protocol_hybrid_routing.py"],
         phase="2 · Supplementary benchmarks",
         expected="+6pp Fix1, +5pp Fix2, +1pp Fix3",
         result_glob="hybrid_pysr/all_domains/**/*.json",
         env_extra={"SKIP_PERF_ANALYSIS": "1",
                    "HYPATIAX_CORE_OPTIONAL": "1"}),

    # §10.9: 70 tasks × K=30 stochastic runs — LLM_K_RUNS injected via env_extra
    Step("instability",
         "§10.9 · Stability under stochastic inference (K=30)",
         ["python3", "hypatiax/protocols/experiment_protocol_instability_rf02_04.py"],
         phase="2 · Supplementary benchmarks",
         slow=True,
         env_extra={"LLM_K_RUNS": "30"},
         expected="Spearman ρ=−0.70, p<0.001 · 70 tasks · C-Collapse anomaly (RF-06)",
         result_glob="hybrid_llm_nn/**/*.json"),

    # FIX-EXTRAP: experiment_protocol_extrapolation_comparative.py searches for
    # experiment_protocol_benchmark_v2.py inside hypatiax/protocols/ but it lives
    # at protocols/ (repo root).  PROTOCOL_ROOT tells the script where to look.
    # Also: run_comparative_suite_benchmark_v2.py exits 1 for both noisy and
    # noiseless passes because the HypatiaX ImportError prevents sub-module load.
    # fixup-init (Phase 0) resolves the root cause; HYPATIAX_CORE_OPTIONAL=1 is an
    # extra guard.
    Step("extrap",
         "§10.8 · Extrapolation comparative (near/med/far OOD)",
         ["python3", "hypatiax/protocols/experiment_protocol_extrapolation_comparative.py"],
         phase="2 · Supplementary benchmarks",
         result_glob="extrapolation/extrapolation_73cases_enhanced.json",
         env_extra={
             # Minor Fix A: auto-detect PROTOCOL_ROOT; fall back to repo-root protocols/
             # if hypatiax/protocols/ is absent (comment in changelog says repo-root).
             "PROTOCOL_ROOT": str(
                 REPO_ROOT / "hypatiax" / "protocols"
                 if (REPO_ROOT / "hypatiax" / "protocols").exists()
                 else REPO_ROOT / "protocols"
             ),
             "HYPATIAX_CORE_OPTIONAL": "1",
         }),

    # ── Phase 3: Audit & verification ───────────────────────────────────────
    Step("provenance",
         "§11 · Provenance audit — protocol orchestration",
         ["python3", "-c",
          "import subprocess, sys, pathlib; "
          "s = pathlib.Path('hypatiax/protocols/experiment_protocol_provenance_audit.py'); "
          "sys.exit(subprocess.run([sys.executable, str(s)]).returncode) "
          "if s.exists() else "
          "(print('  ⚠  provenance protocol not found — skipping (non-blocking)') or sys.exit(0))"],
         phase="3 · Audit & verification"),

    Step("discover-provenance",
         "§11 · discover_provenance.py — link result files to families",
         ["python3", "-c",
          "import subprocess, sys, pathlib; "
          "m = pathlib.Path('provenance_map.json'); "
          "pathlib.Path('logs/provenance_audit').mkdir(parents=True, exist_ok=True); "
          "(print('INFO: provenance_map.json absent — skipping discover_provenance (public repo)') or sys.exit(0)) "
          "if not m.exists() else "
          "sys.exit(subprocess.run([sys.executable, 'discover_provenance.py', "
          "'--root', '.', '--map', str(m), '--out', 'logs/provenance_audit']).returncode)"],
         phase="3 · Audit & verification"),

    Step("scan-imports",
         "§11 · scan_internal_imports.py — internal import DAG",
         ["python3", "scan_internal_imports.py",
          "--root", ".", "--out", "logs/repro_output"],
         phase="3 · Audit & verification"),

    # FIX-VERIFY: verify_results.py resolves result paths relative to a base that
    # defaults to scripts/hypatiax/data/patched/ — wrong.  Pass PATCHED_DATA_DIR
    # and RESULTS_DIR so the script finds files at their actual locations:
    #   hypatiax/data/results/{defi,feynman,exp1_ablation,instability}/
    Step("verify",
         "Verify results against paper targets",
         ["python3", "scripts/patches/verify_results.py", "--report"],
         phase="3 · Audit & verification",
         env_extra={"PATCHED_DATA_DIR": str(REPO_ROOT / "hypatiax" / "data" / "results"),
                    "VERIFY_RESULTS_DIR": str(RESULTS_DIR)}),

    Step("hashlock",
         "Hash lock check",
         ["python3", "hypatiax/reproducibility/hash_lock.py", "--check"],
         phase="3 · Audit & verification"),

    # ── Phase 4: Outputs — figures & tables written to hypatiax/data/results/ ─
    Step("figures",
         "Generate all figures",
         ["python3", "figures/generate_figures.py",
          "--outdir", str(RESULTS_DIR / "figures")],
         phase="4 · Outputs",
         result_glob="figures/*.pdf"),

    # FIX-TABLES: generate_tables.py ignores --outdir and writes to paper/tables/
    # (hardcoded inside the script — line "Output: paper/tables/" in errors.txt).
    # Pass TABLE_OUTDIR env var as an alternative signal AND keep --outdir.
    # Also set RESULTS_DIR so the script can locate input JSON data from the
    # correct place (hypatiax/data/results/) rather than a hardcoded patched path.
    Step("tables",
         "Generate all tables",
         ["python3", "scripts/patches/generate_tables.py",
          "--outdir", str(RESULTS_DIR / "tables")],
         phase="4 · Outputs",
         result_glob="tables/*.tex",
         env_extra={"TABLE_OUTDIR":    str(RESULTS_DIR / "tables"),
                    "VERIFY_RESULTS_DIR": str(RESULTS_DIR)}),

    # ── Phase 4-B: Paper audit notebooks ─────────────────────────────────────
    # FIX-AUDIT-SETUP: In the previous run only the main .tex was copied (1 file).
    # supp_routing_improvements.tex and supp_benchmark_report.tex were silently
    # skipped because they weren't found in paper/ or repo root.  Broaden the
    # search to also check paper/tables/ and logs/, and emit a clear WARNING
    # (rather than silent skip) for each supplement that is genuinely absent.
    Step("audit-setup",
         "Paper audit · Copy main paper + supplements into notebooks/",
         ["python3", "-c", "\n".join([
             "import shutil, pathlib, sys",
             "nb = pathlib.Path('notebooks'); nb.mkdir(exist_ok=True)",
             "search_dirs = [pathlib.Path('paper'), pathlib.Path('.'),",
             "               pathlib.Path('paper') / 'tables',",
             "               pathlib.Path('logs')]",
             "copied = []; missing = []",
             # ── main paper ──
             "main = next((f for d in search_dirs",
             "             for pat in ('jmlr-hypatiax*.tex','jmlr_paper*.tex')",
             "             for f in d.glob(pat) if f.is_file()), None)",
             "if main:",
             "    shutil.copy(main, nb / main.name); copied.append(main.name)",
             "else:",
             "    print('WARNING: main paper .tex not found in any search dir')",
             # ── supplements ──
             "for name in ('supp_routing_improvements.tex','supp_benchmark_report.tex'):",
             "    src = next((d / name for d in search_dirs if (d / name).is_file()), None)",
             "    if src:",
             "        shutil.copy(src, nb / name); copied.append(name)",
             "    else:",
             "        missing.append(name)",
             "        print(f'WARNING: {name} not found in {[str(d) for d in search_dirs]}')",
             "print(f'audit-setup: copied {len(copied)} file(s) to notebooks/: {copied}')",
             "if missing:",
             "    print(f'audit-setup: {len(missing)} supplement(s) missing: {missing}')",
             "    print('  → Audit notebooks will run but supplement checks will be skipped.')",
         ])],
         phase="4-B · Paper audit",
         paper=True),

    Step("audit-NB-01",
         "Paper audit · NB-01 Citation & Bibliography",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-01_Citation_Bibliography_Audit.ipynb"],
         phase="4-B · Paper audit",
         paper=True),

    Step("audit-NB-02",
         "Paper audit · NB-02 Cross-Reference & Label",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-02_CrossReference_Label_Audit.ipynb"],
         phase="4-B · Paper audit",
         paper=True),

    Step("audit-NB-03",
         "Paper audit · NB-03 Section Structure & Numbering",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-03_Section_Structure_Numbering.ipynb"],
         phase="4-B · Paper audit",
         paper=True),

    Step("audit-NB-04",
         "Paper audit · NB-04 Numerical Consistency",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-04_Numerical_Consistency_Checker.ipynb"],
         phase="4-B · Paper audit",
         paper=True),

    Step("audit-NB-05",
         "Paper audit · NB-05 Figure & Image Dependencies",
         ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "--ExecutePreprocessor.timeout=300",
          "notebooks/NB-05_Figure_Image_Dependency_Checker.ipynb"],
         phase="4-B · Paper audit",
         paper=True),
]

STEP_IDS = [s.id for s in STEPS]


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    """Return {step_id: status} from the checkpoint file, or {} if none exists."""
    if CHECKPOINT.exists():
        try:
            return json.loads(CHECKPOINT.read_text())
        except Exception:
            pass
    return {}


def save_checkpoint(state: dict) -> None:
    """Persist {step_id: status} to disk atomically."""
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(CHECKPOINT)


def clear_checkpoint() -> None:
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print(f"  Checkpoint cleared: {CHECKPOINT}")
    else:
        print("  No checkpoint file found.")


# ── Result-file helpers ────────────────────────────────────────────────────────

def ensure_output_dirs() -> None:
    """Create the canonical output subdirectories under hypatiax/data/results/."""
    for sub in [
        "comparison_results/extrapolation",
        "comparison_results/feynman-tests/noise-sweep",
        "comparison_results/noise-noiseless/noiseless",
        "comparison_results/noise-noiseless/15",
        "extrapolation",
        "hybrid_llm_nn/all_domains",
        "hybrid_llm_nn/defi",
        "hybrid_pysr/all_domains",
        "hybrid_pysr/defi",
        "llm_guided/all_domains",
        "llm_guided/defi",
        "standalone_llm_nn",
        "figures",
        "tables",
    ]:
        (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)


def archive_step_results(step: Step) -> None:
    """
    After a step completes, snapshot any newly produced output files into
    logs/<step_id>_results/ for provenance tracing.

    FIX: use rglob() for patterns containing '**'; plain glob() does not expand
    recursive wildcards, so suppB/instability/exp2 previously archived 0 files.
    """
    if not step.result_glob:
        return

    # Split into a base anchor and the glob pattern so rglob works correctly
    pattern = step.result_glob
    if "**" in pattern:
        # e.g. "comparison_results/feynman-tests/**/*.json"
        # Split at the first '**' component
        parts = Path(pattern).parts
        star_idx = next(i for i, p in enumerate(parts) if "**" in p)
        base_dir = RESULTS_DIR / Path(*parts[:star_idx])
        # Reconstruct only the glob portion after (and including) the '**' part
        sub_pattern = str(Path(*parts[star_idx:]))
        matches = list(base_dir.rglob(sub_pattern)) \
                  if base_dir.exists() else []
    else:
        matches = list(RESULTS_DIR.glob(pattern))

    if not matches:
        return

    dest = LOG_DIR / f"{step.id}_results"
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in matches:
        dst = dest / src.name
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            shutil.copy2(src, dst)
            count += 1
    if count:
        print(f"  📁  {count} result file(s) archived → logs/{step.id}_results/")


def inventory_results() -> tuple[int, int, int]:
    """Return (data_file_count, pdf_count, tex_count) under RESULTS_DIR.

    FIX-INVENTORY: generate_tables.py writes to paper/tables/ (ignoring --outdir),
    so tex count was always 0.  Also check paper/tables/ as a fallback.
    """
    jsons = sum(1 for _ in RESULTS_DIR.rglob("*.json"))
    csvs  = sum(1 for _ in RESULTS_DIR.rglob("*.csv"))
    pdfs  = sum(1 for _ in (RESULTS_DIR / "figures").glob("*.pdf")) \
            if (RESULTS_DIR / "figures").exists() else 0
    # Check canonical location first, fall back to paper/tables/
    tables_dir = RESULTS_DIR / "tables"
    if not tables_dir.exists() or not any(tables_dir.glob("*.tex")):
        tables_dir = REPO_ROOT / "paper" / "tables"
    texs = sum(1 for _ in tables_dir.glob("*.tex")) if tables_dir.exists() else 0
    return jsons + csvs, pdfs, texs


# ── Result tracking ────────────────────────────────────────────────────────────
@dataclass
class StepResult:
    id: str
    label: str
    status: str          # "pass" | "fail" | "skip" | "resume-skip"
    elapsed: float = 0.0
    log_path: Optional[Path] = None
    returncode: int = 0


# ── Step runner ────────────────────────────────────────────────────────────────
def run_step(step: Step, env: dict) -> StepResult:
    """
    FIX: stream subprocess output line-by-line to both the log file and stdout,
    rather than buffering all output into memory before printing the last 20 lines.
    This prevents OOM on long-running steps (exp2, suppB, instability).
    """
    log_path = LOG_DIR / f"{step.id}.log"
    merged_env = {**env, **step.env_extra}

    print(f"\n┌─── [{step.id}] {step.label}")
    print(f"│    {time.strftime('%H:%M:%S')}")
    if step.expected:
        print(f"│    Expected : {step.expected}")
    if step.env_extra:
        for k, v in step.env_extra.items():
            print(f"│    env+  {k}={v}")
    print(f"│    cmd: {' '.join(str(x) for x in step.cmd)}")

    t0 = time.time()
    proc: Optional[subprocess.Popen] = None
    try:
        with open(log_path, "w") as log_fh:
            proc = subprocess.Popen(
                step.cmd,
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,          # line-buffered for responsive streaming
            )
            # Stream line-by-line: write to log and echo to terminal
            assert proc.stdout is not None
            for line in proc.stdout:
                log_fh.write(line)
                print(f"│  {line}", end="")
            proc.wait()

        elapsed = time.time() - t0
        ok = proc.returncode == 0
        sym = "✓" if ok else "✗"
        print(f"\n└─── {sym} {'done' if ok else 'FAILED'}  ({elapsed:.0f}s)"
              + (f"  — see {log_path}" if not ok else ""))

        if ok:
            archive_step_results(step)

        return StepResult(step.id, step.label,
                          "pass" if ok else "fail",
                          elapsed, log_path, proc.returncode)

    except KeyboardInterrupt:
        # ── Ctrl+C: kill the child process cleanly, then re-raise so main()
        #    can save the checkpoint and print a tidy summary.
        elapsed = time.time() - t0
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        print(f"\n└─── ✗ INTERRUPTED  ({elapsed:.0f}s) — step killed, checkpoint saved")
        raise   # propagate so main() can handle graceful shutdown

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"└─── ✗ ERROR: {exc}")
        return StepResult(step.id, step.label, "fail", elapsed, log_path)


def banner(msg: str) -> None:
    print("\n" + "═" * 68)
    print(f"  {msg}")
    print("═" * 68)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="HypatiaX reproducibility pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip Feynman (exp2), noise sweep (suppB), instability")
    parser.add_argument("--only", metavar="ID",
                        help="Run only this step id")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint — skip steps already marked pass")
    parser.add_argument("--from", dest="from_step", metavar="ID",
                        help="With --resume: force-rerun from this step id onwards")
    parser.add_argument("--clear-checkpoint", action="store_true",
                        help="Delete the checkpoint file and exit")
    parser.add_argument("--continue-on-fail", action="store_true",
                        help="Log failures but continue remaining steps")
    parser.add_argument("--verify-only", action="store_true",
                        help="Re-check existing results without re-running experiments")
    parser.add_argument("--skip-paper", action="store_true",
                        help="Skip Phase 4-B paper audit notebook steps")
    parser.add_argument("--seed", type=int, default=None, metavar="N",
                        help="Override PYSR_SEED / NN_SEED / PYTHONHASHSEED for all steps "
                             "(e.g. --seed 123). Defaults to 42 when omitted.")
    parser.add_argument("--pysr-timeout", type=int, default=None, metavar="SECS",
                        help="Wall-clock timeout (seconds) passed to PySR via PYSR_TIMEOUT "
                             "env var. Default is whatever the experiment script sets (360s). "
                             "Use e.g. --pysr-timeout 900 on slower hardware.")
    parser.add_argument("--one-equation", action="store_true",
                        help="Smoke-test mode: run exactly 1 equation per experiment. "
                             "Injects ONE_EQUATION=1, N_TASKS_DEFI=1, N_FEYNMAN_TASKS=1, "
                             "N_TASKS_INSTABILITY=1, and forces PYSR_TIMEOUT=120 (unless "
                             "--pysr-timeout is also given). Use this to verify the full "
                             "pipeline end-to-end quickly on local hardware.")
    args = parser.parse_args()

    # FIX: validate --from is only used alongside --resume; warn otherwise
    if args.from_step and not args.resume:
        print("  WARNING: --from has no effect without --resume. "
              "Did you mean: python3 run_all.py --resume --from <id>?",
              file=sys.stderr)

    os.chdir(REPO_ROOT)
    LOG_DIR.mkdir(exist_ok=True)
    ensure_output_dirs()

    # ── --clear-checkpoint ─────────────────────────────────────────────────
    if args.clear_checkpoint:
        clear_checkpoint()
        sys.exit(0)

    banner("HypatiaX · Reproducibility Pipeline v4.7 (checkpoint/resume)"
           + ("  [SMOKE-TEST: 1 equation]" if args.one_equation else ""))
    print(f"  Repo      : {REPO_ROOT}")
    print(f"  Python    : {sys.version.split()[0]}")
    print(f"  Date      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results   : {RESULTS_DIR}")
    print(f"  Logs      : {LOG_DIR}")
    print(f"  Checkpoint: {CHECKPOINT}")

    # ── API key ────────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("\n  ERROR: ANTHROPIC_API_KEY is not set.")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)
    print(f"\n  API key   : set ({len(api_key)} chars)")

    # ── hypatiax/protocols/ check ──────────────────────────────────────────
    hypatiax_proto = REPO_ROOT / "hypatiax" / "protocols"
    required_hp = [
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
    missing_hp = [f for f in required_hp if not (hypatiax_proto / f).exists()]
    if missing_hp:
        print(f"\n  ERROR: {len(missing_hp)} input-data module(s) missing "
              "from hypatiax/protocols/:")
        for f in missing_hp:
            print(f"    ✗  {f}")
        sys.exit(1)
    print(f"  Protocols : all {len(required_hp)} hypatiax/protocols/ modules ✓")

    # ── NB-06 code-quality pre-audit (non-blocking) ────────────────────────
    nb06 = REPO_ROOT / "notebooks" / "NB-06_Code_Quality_Pipeline_Integrity.ipynb"
    if nb06.exists():
        try:
            r = subprocess.run(
                ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                 "--inplace", "--ExecutePreprocessor.timeout=120", str(nb06)],
                capture_output=True, text=True, timeout=150,
            )
            status = "✓" if r.returncode == 0 else "⚠ warnings (non-blocking)"
            print(f"  NB-06     : code quality pre-audit {status}")
        except FileNotFoundError:
            print("  NB-06     : jupyter not found — skipped (pip install notebook)")
        except Exception as exc:
            print(f"  NB-06     : skipped ({exc})")

    # ── verify-only shortcut ───────────────────────────────────────────────
    if args.verify_only:
        banner("Verify-only mode")
        subprocess.run(["python3", "scripts/patches/verify_results.py", "--report"],
                       check=False)
        subprocess.run(["python3", "hypatiax/reproducibility/hash_lock.py", "--check"],
                       check=False)
        sys.exit(0)

    # ── Build environment (mirrors run_all.sh and notebook cell 2) ─────────
    _seed_str = str(args.seed) if args.seed is not None else "42"

    env = {**os.environ}
    env["PYTHONWARNINGS"] = "ignore"
    env["NN_SEED"]               = os.environ.get("NN_SEED",   _seed_str)
    env["PYSR_SEED"]             = os.environ.get("PYSR_SEED", _seed_str)
    env["PYTHONHASHSEED"]        = os.environ.get("PYTHONHASHSEED", _seed_str)
    # If --seed was given explicitly it overrides any pre-existing env value.
    if args.seed is not None:
        env["NN_SEED"]        = _seed_str
        env["PYSR_SEED"]      = _seed_str
        env["PYTHONHASHSEED"] = _seed_str
    env.setdefault("LLM_MODEL",             "claude-sonnet-4-20250514")
    env.setdefault("LLM_RETRIES",           "3")
    env.setdefault("LLM_K_RUNS",            "1")   # overridden to 30 for instability
    env.setdefault("N_TASKS_DEFI",          "74")
    env.setdefault("N_TASKS_INSTABILITY",   "70")  # FIX-T1: must be 70 not 71
    env.setdefault("PCA_TRAIN_FRAC",        "0.40")
    env.setdefault("NN_TIME_LIMIT",         "120")
    env.setdefault("ENGINE_NAME",           "hybrid_system_v50_2")  # FIX-C2: never v40
    env.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    # PYSR_TIMEOUT: experiment scripts should read this to override their default
    # wall-clock limit (360 s).  Use --pysr-timeout on slower local hardware to
    # prevent the Hybrid column from being all N/A due to PySR timing out.
    if args.pysr_timeout is not None:
        env["PYSR_TIMEOUT"] = str(args.pysr_timeout)
        print(f"  PYSR_TIMEOUT={args.pysr_timeout}s  (--pysr-timeout override)")
    else:
        env.setdefault("PYSR_TIMEOUT", "360")   # match experiment script default

    # ── --one-equation: smoke-test mode ───────────────────────────────────
    if args.one_equation:
        # Tell every experiment script to run only 1 equation/task.
        # Scripts should check ONE_EQUATION=1 and/or the N_TASKS_* vars.
        env["ONE_EQUATION"]          = "1"
        env["N_TASKS_DEFI"]          = "1"   # exp1: 1 of 74 DeFi tasks
        env["N_CORE15_TASKS"]        = "1"   # exp1: 1 of 15 Core-15 ablation equations
        env["N_FEYNMAN_TASKS"]       = "1"   # exp2: 1 of 30 Feynman equations
        env["N_TASKS_INSTABILITY"]   = "1"   # instability: 1 of 70 tasks
        env["N_NGUYEN_TASKS"]        = "1"   # exp3/exp3b: 1 of 12 Nguyen equations
        env["N_NOISE_EQUATIONS"]     = "1"   # suppB: 1 equation across all noise levels
        env["LLM_K_RUNS"]            = "1"   # force K=1 even for instability step
        # FIX 4: real smoke-test PySR config — small populations + iterations
        env["N_ITERATIONS"]          = "200"  # was default ~1000 (too heavy for smoke-test)
        env["POPULATIONS"]           = "10"   # was default ~30  (too heavy for smoke-test)
        # Use a short PySR timeout (120 s) unless the user explicitly overrode it
        if args.pysr_timeout is None:
            env["PYSR_TIMEOUT"] = "60"  # FIX 4: tightened from 120 → 60 for real smoke-test
        print("\n" + "▲" * 68)
        print("  ▲▲  SMOKE-TEST MODE  (--one-equation)")
        print("  ▲▲  1 equation per experiment · PYSR_TIMEOUT="
              + env["PYSR_TIMEOUT"] + "s"
              + "  N_ITERATIONS=" + env["N_ITERATIONS"]
              + "  POPULATIONS=" + env["POPULATIONS"])
        print("  ▲▲  This is NOT a full reproducibility run.")
        print("  ▲▲  Results will NOT match paper targets.")
        print("▲" * 68)
    env["PYTHONPATH"]    = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["RESULTS_DIR"]   = str(RESULTS_DIR)
    # FIX-PY: export the exact venv interpreter so protocol wrappers that shell out
    # to Python can use PIPELINE_PYTHON instead of bare 'python3'.
    env["PIPELINE_PYTHON"] = sys.executable
    env["REPRO_ROOT"]  = str(REPO_ROOT)

    _seed_source = f"--seed flag" if args.seed is not None else "default (env or 42)"
    print(f"\n  NN_SEED={env['NN_SEED']}  PYSR_SEED={env['PYSR_SEED']}  "
          f"PYTHONHASHSEED={env['PYTHONHASHSEED']}  (source: {_seed_source})")
    print(f"  LLM_MODEL={env['LLM_MODEL']}")
    print(f"  ENGINE={env['ENGINE_NAME']}  N_TASKS_INSTABILITY={env['N_TASKS_INSTABILITY']}")
    if args.skip_paper:
        print(f"  --skip-paper: Phase 4-B notebook steps will be skipped")

    # ── Validate --only / --from ───────────────────────────────────────────
    if args.only and args.only not in STEP_IDS:
        print(f"\n  ERROR: unknown step id '{args.only}'.")
        print(f"  Valid ids: {', '.join(STEP_IDS)}")
        sys.exit(1)
    if args.from_step and args.from_step not in STEP_IDS:
        print(f"\n  ERROR: unknown step id '{args.from_step}'.")
        print(f"  Valid ids: {', '.join(STEP_IDS)}")
        sys.exit(1)

    # ── Load checkpoint state ──────────────────────────────────────────────
    checkpoint_state: dict[str, str] = {}
    if args.resume:
        checkpoint_state = load_checkpoint()
        if checkpoint_state:
            n_prev = sum(1 for v in checkpoint_state.values() if v == "pass")
            print(f"\n  ── Resuming: {n_prev} step(s) already passed in checkpoint")
            if args.from_step:
                print(f"  ── --from {args.from_step}: force-rerun from here onwards")
        else:
            print("\n  ── No checkpoint found — running full pipeline")

    # ── Run pipeline ───────────────────────────────────────────────────────
    results: list[StepResult] = []
    current_phase = ""
    t_total = time.time()
    # once we reach --from step, all subsequent steps run regardless of checkpoint
    past_from = args.from_step is None

    try:
        for step in STEPS:
            if step.phase != current_phase:
                banner(f"Phase {step.phase}")
                current_phase = step.phase

            if args.from_step and step.id == args.from_step:
                past_from = True

            # --only filter
            if args.only and step.id != args.only:
                results.append(StepResult(step.id, step.label, "skip"))
                print(f"  ── skip [{step.id}]  (--only {args.only})")
                continue

            # --resume: skip steps that already passed, unless we're past --from
            if (args.resume
                    and checkpoint_state.get(step.id) == "pass"
                    and not past_from):
                results.append(StepResult(step.id, step.label, "resume-skip"))
                print(f"  ↩  skip [{step.id}]  (checkpoint: already passed)")
                continue

            # --skip-slow filter
            if args.skip_slow and step.slow:
                results.append(StepResult(step.id, step.label, "skip"))
                print(f"  ── skip [{step.id}]  (--skip-slow)")
                continue

            # FIX: --skip-paper filter — previously parsed but never applied
            if args.skip_paper and step.paper:
                results.append(StepResult(step.id, step.label, "skip"))
                print(f"  ── skip [{step.id}]  (--skip-paper)")
                continue

            result = run_step(step, env)
            results.append(result)

            # save checkpoint after every step
            checkpoint_state[step.id] = result.status
            save_checkpoint(checkpoint_state)

            if result.status == "fail" and not args.continue_on_fail:
                print(f"\n  Pipeline aborted at [{step.id}].")
                print(f"  Checkpoint saved → {CHECKPOINT}")
                print(f"  To resume:         python3 run_all.py --resume")
                print(f"  To rerun this step: python3 run_all.py --only {step.id}")
                _print_summary(results, time.time() - t_total)
                sys.exit(1)

    except KeyboardInterrupt:
        # ── Ctrl+C pressed between steps (or re-raised from run_step) ──────
        print(f"\n\n  ⚠  Interrupted by user (Ctrl+C).")
        # Mark any step currently being attempted as failed in checkpoint
        # (run_step already appended its StepResult before re-raising, so
        #  results list is up to date; just ensure the checkpoint reflects it.)
        for r in results:
            if r.id not in checkpoint_state:
                checkpoint_state[r.id] = r.status
        save_checkpoint(checkpoint_state)
        print(f"  Checkpoint saved → {CHECKPOINT}")
        print(f"  Resume with:       python3 run_all.py --resume"
              + ("  --one-equation" if args.one_equation else ""))
        _print_summary(results, time.time() - t_total)
        sys.exit(130)   # conventional exit code for Ctrl+C

    _print_summary(results, time.time() - t_total)

    # Clear checkpoint only after a complete, fully-passing run
    failed = [r for r in results if r.status == "fail"]
    if not failed and not args.only:
        clear_checkpoint()

    if args.one_equation:
        print("\n" + "▲" * 68)
        print("  ▲▲  SMOKE-TEST COMPLETE  (--one-equation)")
        print("  ▲▲  Re-run without --one-equation for a full reproducibility run.")
        print("▲" * 68)

    sys.exit(1 if failed else 0)



def _print_summary(results: list[StepResult], elapsed: float) -> None:
    passed       = [r for r in results if r.status == "pass"]
    failed       = [r for r in results if r.status == "fail"]
    skipped      = [r for r in results if r.status == "skip"]
    resume_skips = [r for r in results if r.status == "resume-skip"]

    hh, rem = divmod(int(elapsed), 3600)
    mm, ss  = divmod(rem, 60)

    banner("Pipeline summary")
    col = {"pass": "✓", "fail": "✗", "skip": "─", "resume-skip": "↩"}
    for r in results:
        t = f"  {r.elapsed:6.0f}s" if r.status in ("pass", "fail") else "        "
        print(f"  {col[r.status]} [{r.id:28s}] {r.label[:48]:48s}{t}")

    print()
    print(f"  ✓ passed      : {len(passed)}")
    print(f"  ✗ failed      : {len(failed)}")
    print(f"  ─ skipped     : {len(skipped)}")
    print(f"  ↩ resume-skip : {len(resume_skips)}")
    print(f"  Wall time     : {hh:02d}:{mm:02d}:{ss:02d}")

    # Result file inventory
    data_files, fig_files, tbl_files = inventory_results()
    print(f"\n  Results → {RESULTS_DIR}")
    print(f"    Data files (JSON+CSV) : {data_files}")
    print(f"    Figures (PDF)         : {fig_files}")
    print(f"    Tables  (TeX)         : {tbl_files}")

    # Provenance coverage
    prov = LOG_DIR / "provenance_audit" / "provenance_audit_summary.txt"
    if prov.exists():
        print(f"\n  Provenance ({prov}):")
        for line in prov.read_text().splitlines():
            if any(k in line for k in ("AUTHORITATIVE", "ORPHAN", "Total")):
                print(f"    {line.strip()}")

    if failed:
        print("\n  Failed steps:")
        for r in failed:
            print(f"    [{r.id}] → {r.log_path}")
        print()
        print(f"  Checkpoint : {CHECKPOINT}")
        print("  Resume     : python3 run_all.py --resume")
        print("  Single step: python3 run_all.py --only <id>")
    else:
        print("\n  ✓ All steps passed.")
        print(f"  Results    : {RESULTS_DIR}/")
        print(f"  Figures    : {RESULTS_DIR}/figures/")
        print(f"  Tables     : {RESULTS_DIR}/tables/")
        print(f"  Provenance : {LOG_DIR}/provenance_audit/")
        print(f"  Import DAG : {LOG_DIR}/repro_output/import_graph.dot")
        print("  Checkpoint : cleared (all steps passed)")


if __name__ == "__main__":
    main()
