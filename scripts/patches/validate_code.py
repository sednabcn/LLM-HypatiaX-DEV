#!/usr/bin/env python3
"""
validate_code.py — Pre-run code quality and correctness checks

Checks:
  1. No stale hybrid_system_v50_2 imports
  2. No hardcoded API keys
  3. Duplicate DeFi case names (FIX-C1)
  4. Critical file existence
  5. Python imports resolve (hybrid_system_v50_2)

Exit 0 = all good, Exit 1 = blocking issues found.
"""
import re
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent

ERRORS   = []
WARNINGS = []

def error(msg):   ERRORS.append(msg);   print(f"  ❌ {msg}")
def warning(msg): WARNINGS.append(msg); print(f"  ⚠  {msg}")
def ok(msg):      print(f"  ✅ {msg}")

# ── Check 1: No stale v50_2 imports ─────────────────────────────────────────────
def check_v50_2_imports():
    print("\n[1] Checking for stale hybrid_system_v50_2 imports...")
    stale = []
    for f in (ROOT / "hypatiax").rglob("*.py"):
        if "hybrid_system_v50_2" in f.name:
            continue   # skip the file itself
        src = f.read_text(errors="replace")
        if re.search(r"hybrid_system_v50_2[^_]", src):
            stale.append(str(f.relative_to(ROOT)))
    if stale:
        error(f"Stale v50_2 imports in: {', '.join(stale)}")
        print("     Fix: sed -i 's/hybrid_system_v50_2[^_]/hybrid_system_v50_2/g' <file>")
    else:
        ok("No stale v50_2 imports")

# ── Check 2: No exposed API keys ──────────────────────────────────────────────
def check_api_keys():
    print("\n[2] Scanning for exposed API keys...")
    for f in ROOT.rglob("*.py"):
        src = f.read_text(errors="replace")
        if re.search(r"sk-ant-api[0-9a-zA-Z\-]+", src):
            error(f"Exposed Anthropic API key in: {f.relative_to(ROOT)}")
    for f in ROOT.rglob("*.ipynb"):
        src = f.read_text(errors="replace")
        if re.search(r"sk-ant-api[0-9a-zA-Z\-]+", src):
            error(f"Exposed Anthropic API key in notebook: {f.relative_to(ROOT)}")
    if not any("API key" in e for e in ERRORS):
        ok("No exposed API keys")

# ── Check 3: Duplicate DeFi case names ────────────────────────────────────────
def check_defi_duplicates():
    print("\n[3] Checking DeFi case name uniqueness (FIX-C1)...")
    bench = next(ROOT.rglob("hypatiax_defi_benchmark_v3c.py"), None)
    if not bench:
        warning("hypatiax_defi_benchmark_v3c.py not found — skipping")
        return
    src   = bench.read_text(errors="replace")
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', src)
    dupes = {n: c for n, c in Counter(names).items() if c > 1}
    if dupes:
        for name, count in dupes.items():
            error(f"Duplicate DeFi case name '{name}' appears {count}×  → apply FIX-C1")
    else:
        ok(f"All {len(names)} DeFi case names are unique")

# ── Check 4: Critical files exist ─────────────────────────────────────────────
def check_critical_files():
    print("\n[4] Checking critical file existence...")
    required = [
        "hypatiax/tools/symbolic/hybrid_system_v50_2.py",
        "hypatiax/experiments/benchmarks/hypatiax_defi_benchmark_v3c.py",
        "hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_v2.py",
        "hypatiax/experiments/benchmarks/run_dual_sweep_benchmarks.py",
    ]
    for rel in required:
        p = ROOT / rel
        if p.exists():
            ok(rel)
        else:
            error(f"Missing: {rel}")

# ── Check 5: v50_2 import resolves ────────────────────────────────────────────
def check_v50_import():
    print("\n[5] Checking hybrid_system_v50_2 importability...")
    v50 = ROOT / "hypatiax" / "tools" / "symbolic" / "hybrid_system_v50_2.py"
    if not v50.exists():
        error("hybrid_system_v50_2.py not found")
        return
    # Light syntax check
    import ast
    try:
        ast.parse(v50.read_text())
        ok("hybrid_system_v50_2.py parses OK")
    except SyntaxError as e:
        error(f"Syntax error in hybrid_system_v50_2.py: {e}")

# ── Check 6: Supp A section reference ─────────────────────────────────────────
def check_supp_a():
    print("\n[6] Checking Supp A section reference (FIX-XR3)...")
    supp = ROOT / "paper" / "supp_routing_improvements.tex"
    if not supp.exists():
        warning("supp_routing_improvements.tex not found — skipping")
        return
    src = supp.read_text(errors="replace")
    if "Section 7.3 (Component 3)" in src:
        error("Supp A still says 'Section 7.3' — should be '7.4' (FIX-XR3)")
    else:
        ok("Supp A section reference OK")

# ── Check 7: Paper text fixes ──────────────────────────────────────────────────
def check_paper_text():
    print("\n[7] Checking paper text fixes...")
    tex = ROOT / "paper" / "jmlr-hypatiax-paper-final.tex"
    if not tex.exists():
        warning("Main paper .tex not found — skipping")
        return
    src = tex.read_text(errors="replace")

    if "across all 71 cases" in src:
        error("FIX-T1 not applied: '71 cases' should be '70 tasks' in §10.9")
    else:
        ok("FIX-T1: '70 tasks' wording OK")

    if "Five-Layer Architecture Overview" in src:
        warning("FIX-T2 not applied: §8.3 still says 'Five-Layer' — should be 'Five-Stage'")
    else:
        ok("FIX-T2: 'Five-Stage' terminology OK")

    if "cranmer2023interpretable" in src:
        error("FIX-B2 not applied: duplicate bibkey cranmer2023interpretable still in paper")
    else:
        ok("FIX-B2: duplicate bibkey removed")

    if "udrescu2020aifeynman" in src:
        error("FIX-B3 not applied: duplicate bibkey udrescu2020aifeynman still in paper")
    else:
        ok("FIX-B3: duplicate bibkey removed")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("═" * 55)
    print("  Code Validator — HypatiaX JMLR")
    print("═" * 55)

    check_v50_2_imports()
    check_api_keys()
    check_defi_duplicates()
    check_critical_files()
    check_v50_import()
    check_supp_a()
    check_paper_text()

    print(f"\n{'═'*55}")
    print(f"  Errors: {len(ERRORS)}   Warnings: {len(WARNINGS)}")
    print(f"{'═'*55}")

    if ERRORS:
        print("\n❌ Blocking errors found — fix before running benchmarks")
        sys.exit(1)
    elif WARNINGS:
        print("\n⚠  Warnings present — review before submission")
        sys.exit(0)
    else:
        print("\n✅ All checks passed — safe to run")
        sys.exit(0)

if __name__ == "__main__":
    main()
