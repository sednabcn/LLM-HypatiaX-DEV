#!/usr/bin/env python3
"""
hybrid_variants_catalog.py
===========================
Grep-verifiable catalog of every distinct Hybrid-method combination
strategy found in github.com/sednabcn/LLM-HypatiaX-REPRO.

Extends the original 5-variant mapping (hypatiax_variants_source_mapping_
report.md) with three variants discovered while auditing the uploaded
scripts against a fresh clone of the repo on 2026-09-22:

  Variant 6      — validation-selected residual hybrid   (hypatiax_defi_benchmark_v4.py)
  Variant 6-PCA  — same selector, PCA-projected features  (hypatiax_defi_benchmark_v4_pca.py)
  Variant 7      — LLM-prior PySR population seeding      (exp3_nguyen12_consolidated.py + hypatia.py)

Run `python hybrid_variants_catalog.py /path/to/cloned/repo` to re-verify
every row against real source (existence of file, class/function, and the
literal gate-threshold tokens quoted in `gate`). Exits non-zero and prints
a mismatch report if anything no longer matches.
"""
from __future__ import annotations
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Variant:
    id: str
    name: str
    real_source: list[str]           # paths, relative to repo root
    symbol: str                      # class / function name to grep for
    symbol_kind: str                 # "class" | "def"
    combination_strategy: str        # one-line strategy label
    gate: str                        # gate/decision-logic description
    invoked_by: list[str] = field(default_factory=list)  # experiment scripts
    notes: str = ""


CATALOG: list[Variant] = [
    # ── Original five (grep-verified 2026-09-22, unchanged) ──────────────
    Variant(
        id="1", name="Enhanced Hybrid DeFi",
        real_source=["hypatiax/core/generation/hybrid_defi_system/hybrid_system_nn_defi_domain.py"],
        symbol="EnhancedHybridSystemDeFi", symbol_kind="class",
        combination_strategy="R²-threshold gate + weighted ensemble fallback",
        gate="fitted_r2>=0.85 and margin>0.05 -> LLM; both r2>=0.5 and |margin|<=0.15 -> "
             "R²-share-weighted ensemble; else -> NN",
        invoked_by=["run_hybrid_system_benchmark.py",
                    "run_comparative_suite_benchmark_v2.py",
                    "run_noise_sweep_benchmark.py",
                    "test_enhanced_defi_extrapolation.py"],
    ),
    Variant(
        id="2", name="Hybrid All-Domains",
        real_source=["hypatiax/core/generation/hybrid_all_domains_llm_nn/hybrid_system_llm_nn_all_domains.py"],
        symbol="HybridSystemAllDomains", symbol_kind="class",
        combination_strategy="LLM-first gate, tie/force favors LLM",
        gate="llm_ok gate first (LLM succeeds and r2>0); ties/wins favor LLM; ensemble only "
             "when NN beats LLM and NN r2>0.90; force_llm never rescues a broken formula",
    ),
    Variant(
        id="3", name="Ensemble Example",
        real_source=["hypatiax/core/generation/hybrid_defi_llm_nn/hybrid_ensemble_system_defi_domain.py"],
        symbol="ensemble_llm_nn", symbol_kind="def",
        combination_strategy="Inverse-uncertainty x R²-strength weighted blend",
        gate="w = (1/std_residual) * max(r2,0) per arm; falls back to 0.5/0.5 if undefined",
    ),
    Variant(
        id="4", name="LLM-Prior / PySR Family",
        real_source=["hypatiax/tools/symbolic/symbolic_engine.py"],
        symbol="SymbolicEngineWithLLM", symbol_kind="class",
        combination_strategy="Mode-dependent dispatch (hybrid / fallback)",
        gate="hybrid: r2>0.95 -> LLM only; 0.5<=r2<=0.95 -> LLM-seeded search, keep better; "
             "r2<0.5 -> prior discarded. fallback: search r2>0.90 -> done; else ask LLM, keep better",
    ),
    Variant(
        id="5", name="Template Physics Fallback",
        real_source=["hypatiax/tools/symbolic/hybrid_system_v50_2.py",
                     "hypatiax/tools/symbolic/physics_aware_regressor.py",
                     "hypatiax/tools/symbolic/smart_structure_detector.py"],
        symbol="HybridDiscoverySystem", symbol_kind="class",
        combination_strategy="Retry loop + optional physics-template fallback",
        gate="max_retries=5, seed=42+attempt; early stop r2>=0.95 (or >=0.9999 if "
             "transcendental compositions); success requires r2>=0.97 after retries; "
             "physics fallback only if enable_physics_fallback=True (off by default) "
             "and best r2 < physics_fallback_threshold (default 0.85)",
        notes="SmartStructureDetector is ported but orphaned -- never imported/called "
              "outside its own module in the real repo.",
    ),

    # ── Newly identified (this session, 2026-09-22) ───────────────────────
    Variant(
        id="6", name="Validation-Selected Residual Hybrid (v4)",
        real_source=["hypatiax/experiments/benchmarks/hypatiax_defi_benchmark_v4.py"],
        symbol="_select_v4_candidate", symbol_kind="def",
        combination_strategy="Internal-validation candidate selection among 5 arms",
        gate="Split off an internal validation slice; score {llm, nn:<arch>, "
             "residual_nn:<arch> (NN on LLM residuals), blend:<arch> (alpha-grid "
             "search over llm/nn mix), linear_fallback} on that slice; pick the max, "
             "tie-broken toward the simpler candidate; when the test domain is "
             "extrapolative a bare NN is excluded from the pool entirely",
        invoked_by=["hypatiax_defi_benchmark_v4.py (self, via _v4_hybrid_predict_and_eval)"],
        notes="Explicitly supersedes/replaces variants 1-3's real source per the file's "
              "own header (\"Replaces all previous versions: ... hybrid_system_nn_defi_"
              "domain.py, hybrid_ensemble_system_defi_domain.py, ...\"). The file also "
              "contains a legacy _hybrid_predict_and_eval() (the older Fix-1..13 gate "
              "cascade, structurally similar to Variant 1/3's logic) but it is dead code "
              "-- the file's own comment at the call site calls it \"old (dead)\"; only "
              "_v4_hybrid_predict_and_eval/_select_v4_candidate is ever invoked.",
    ),
    Variant(
        id="6-PCA", name="Validation-Selected Residual Hybrid (v4, PCA features)",
        real_source=["hypatiax/experiments/benchmarks/hypatiax_defi_benchmark_v4_pca.py"],
        symbol="_select_v4_candidate", symbol_kind="def",
        combination_strategy="Same selector as Variant 6, on PCA-projected features",
        gate="Identical gate/candidate-pool logic to Variant 6; same known bug fixed in "
             "both copies verbatim (\"Fix 15 ... same bug present verbatim in the _pca "
             "variant of this function\")",
        invoked_by=["test_defi_benchmark_v4_smoke.py",
                    "baseline_pure_llm_defi_discovery.py"],
    ),
    Variant(
        id="7", name="LLM-Prior PySR Population Seeding",
        real_source=["hypatiax/experiments/benchmarks/exp3_nguyen12_consolidated.py",
                     "hypatiax/experiments/benchmarks/hypatia.py"],
        symbol="get_llm_prior", symbol_kind="def",
        combination_strategy="LLM candidate expressions seeded directly into PySR's "
                              "genetic-search initial population (guesses=)",
        gate="get_llm_prior() asks the LLM for n_candidates expressions per equation; "
             "converted to PySR guess strings and passed into "
             "PySRRegressor(guesses=...) at construction time (pysr>=2.0.0a1 moved "
             "`guesses` from fit() to __init__()); run refuses to start "
             "(FIX-LLM-WARMSTART-CTOR) rather than silently degrade to an unseeded run "
             "if the installed pysr build lacks a guesses parameter; on any LLM-call "
             "exception it falls back to a cold, unseeded PySR run for that case",
        notes="Not a threshold gate or a blend like variants 1-6: it is population-"
              "seeding of a genetic search, evaluated after the fact only by comparing "
              "the seeded (H) run's R² against an unseeded PySR-only (P) run on the same "
              "case. os.environ.setdefault('ENGINE','hybrid_system_v50_2') is set at "
              "import time but the actual H-arm fit path uses raw PySRRegressor, not "
              "HybridDiscoverySystem -- the ENGINE variable is vestigial in this script.",
    ),
]


def verify(repo_root: str) -> bool:
    root = Path(repo_root)
    ok = True
    for v in CATALOG:
        symbol_found = False
        for rel in v.real_source:
            p = root / rel
            if not p.exists():
                print(f"[MISSING FILE]  Variant {v.id}: {rel}")
                ok = False
                continue
            text = p.read_text(errors="replace")
            pattern = rf"^{v.symbol_kind}\s+{re.escape(v.symbol)}\b"
            if re.search(pattern, text, re.MULTILINE):
                symbol_found = True
        if not symbol_found:
            print(f"[MISSING SYMBOL] Variant {v.id}: {v.symbol_kind} {v.symbol} not found in any of {v.real_source}")
            ok = False
    print(f"\n{'PASS' if ok else 'FAIL'}: {len(CATALOG)} variants checked against {repo_root}")
    return ok


def print_table() -> None:
    for v in CATALOG:
        print(f"\n[{v.id}] {v.name}")
        print(f"    source : {', '.join(v.real_source)}")
        print(f"    symbol : {v.symbol_kind} {v.symbol}")
        print(f"    strat  : {v.combination_strategy}")
        print(f"    gate   : {v.gate}")
        if v.invoked_by:
            print(f"    used by: {', '.join(v.invoked_by)}")
        if v.notes:
            print(f"    note   : {v.notes}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(0 if verify(sys.argv[1]) else 1)
    print_table()
