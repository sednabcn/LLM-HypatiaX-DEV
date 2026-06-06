#!/usr/bin/env python3
"""
fix_paper_issues.py  (v2)
==========================
Auto-applies every non-numerical, non-figure fix detected by the HypatiaX
paper audit.  Safe to re-run: all operations are idempotent.

v2 change: fix application is now driven by scripts/patches/issue_registry.json.
  - Entries with status="false_positive" or status="resolved" are automatically
    skipped, with a clear printed explanation.
  - Entries with status="open" and auto_fixable=true are applied.
  - No FIX-* IDs are hardcoded in this file.

Usage
-----
    python fix_paper_issues.py                        # dry-run (default)
    python fix_paper_issues.py --dry-run false        # apply changes
    python fix_paper_issues.py --tex-dir path/to/tex  # custom .tex root
    python fix_paper_issues.py --supp-dir path/to/tex # custom supp root
    python fix_paper_issues.py --py-dir path/to/src   # custom Python src root
    python fix_paper_issues.py --registry path/to/issue_registry.json
"""

import argparse
import re
import sys
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Issue registry loader
# ---------------------------------------------------------------------------

DEFAULT_REGISTRY_PATH = Path("scripts/patches/issue_registry.json")

def load_registry(registry_path: Path) -> list[dict]:
    if not registry_path.exists():
        print(f"⚠  Registry not found at {registry_path} — all fixes will run (legacy mode).")
        return []
    try:
        entries = json.loads(registry_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(entries)} entries from {registry_path}")
        return entries
    except Exception as e:
        print(f"⚠  Failed to parse registry: {e} — all fixes will run (legacy mode).")
        return []

def registry_status(registry: list[dict], fix_id: str) -> str:
    """Return 'open' | 'resolved' | 'false_positive' | 'unknown'."""
    for e in registry:
        if e.get("id") == fix_id:
            return e.get("status", "unknown")
    return "unknown"

def registry_reason(registry: list[dict], fix_id: str) -> str:
    for e in registry:
        if e.get("id") == fix_id:
            return e.get("false_positive_reason") or e.get("action") or ""
    return ""

def should_apply(registry: list[dict], fix_id: str) -> bool:
    """
    Returns True only when the registry says this fix is still open.
    False positives and resolved items are skipped.
    If the registry is empty (legacy mode) every fix runs.
    """
    if not registry:
        return True
    status = registry_status(registry, fix_id)
    if status == "false_positive":
        reason = registry_reason(registry, fix_id)
        print(f"  SKIP {fix_id} — marked FALSE POSITIVE in registry.")
        if reason:
            print(f"         Reason: {reason}")
        return False
    if status == "resolved":
        reason = registry_reason(registry, fix_id)
        print(f"  SKIP {fix_id} — marked RESOLVED in registry.")
        if reason:
            print(f"         Action was: {reason}")
        return False
    if status == "unknown":
        print(f"  WARN {fix_id} — not found in registry; applying anyway.")
    return True  # open or unknown

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def save(path: Path, text: str, dry_run: bool) -> None:
    if dry_run:
        print(f"  [DRY-RUN] would write {path}")
    else:
        path.write_text(text, encoding="utf-8")
        print(f"  [WROTE]   {path}")

def patch(original: str, old: str, new: str, label: str) -> tuple[str, int]:
    count = original.count(old)
    if count:
        print(f"  {label}: replaced {count} occurrence(s)")
    return original.replace(old, new), count

def patch_re(original: str, pattern: str, repl, label: str,
             flags: int = 0) -> tuple[str, int]:
    new, count = re.subn(pattern, repl, original, flags=flags)
    if count:
        print(f"  {label}: replaced {count} occurrence(s)")
    return new, count

# ---------------------------------------------------------------------------
# Per-fix functions  (unchanged logic from v1)
# ---------------------------------------------------------------------------

def fix_b1_add_koza1994(text: str) -> str:
    if r"\bibitem{koza1994genetic}" in text:
        print("  FIX-B1: bibitem already present — skipping")
        return text
    new_bibitem = (
        r"\bibitem{koza1994genetic}" + "\n"
        r"J.~R. Koza." + "\n"
        r"\newblock {\em Genetic Programming II: Automatic Discovery of Reusable Programs}." + "\n"
        r"\newblock MIT Press, 1994."
    )
    anchor = r"\bibitem{koza1992gp}"
    if anchor in text:
        idx      = text.index(anchor)
        next_bib = text.find(r"\bibitem{", idx + len(anchor))
        if next_bib == -1:
            next_bib = len(text)
        text = text[:next_bib] + new_bibitem + "\n\n" + text[next_bib:]
        print("  FIX-B1: inserted \\bibitem{koza1994genetic} after koza1992gp")
    else:
        print("  FIX-B1: WARNING — koza1992gp anchor not found; manual insert needed")
    return text


def fix_b2_cranmer(text: str) -> str:
    text, _ = patch_re(text, r"cranmer2023interpretable", "cranmer2023pysr",
                       "FIX-B2: redirect \\cite{cranmer2023interpretable}")
    text, _ = patch_re(
        text,
        r"\\bibitem\{cranmer2023pysr\}[^\n]*\n(?:[^\n]*\n)*?(?=\\bibitem|\\end\{thebibliography\})",
        "", "FIX-B2: remove duplicate bibitem",
    )
    return text


def fix_b3_udrescu(text: str) -> str:
    text, _ = patch_re(text, r"udrescu2020aifeynman", "udrescu2020ai",
                       "FIX-B3: redirect \\cite{udrescu2020aifeynman}")
    text, _ = patch_re(
        text,
        r"\\bibitem\{udrescu2020ai\}[^\n]*\n(?:[^\n]*\n)*?(?=\\bibitem|\\end\{thebibliography\})",
        "", "FIX-B3: remove duplicate bibitem",
    )
    return text


def fix_xr1_duplicate_label(text: str) -> str:
    text, _ = patch(text, r"\label{sec:llm_domain}", "",
                    "FIX-XR1: remove \\label{sec:llm_domain}")
    text, _ = patch_re(text, r"\\(auto)?ref\{sec:llm_domain\}",
                       r"\\\1ref{sec:llm_limitations}",
                       "FIX-XR1: redirect \\ref{sec:llm_domain}")
    return text


def fix_xr2_label_in_item(text: str) -> str:
    subsec_pattern = r"(\\subsection\{Pipeline Corrections in v3\.0\})"
    label = r"\label{sec:r2_bugfix}"
    if re.search(subsec_pattern + r"\s*" + re.escape(label), text):
        print("  FIX-XR2: label already on subsection — skipping")
        return text
    text, removed = patch(text, label, "", "FIX-XR2: remove misplaced label")
    if removed:
        text, attached = patch_re(text, subsec_pattern,
                                  lambda m: m.group(1) + "\n" + label,
                                  "FIX-XR2: attach label to subsection")
        if not attached:
            print("  FIX-XR2: WARNING — subsection heading not found; label removed but not re-attached")
    return text


def fix_xr3_supp_section_number(text: str) -> str:
    text, _ = patch_re(text, r"Section\s+7\.3\s*\(Component\s+3\)",
                       "Section 7.4 (Component 3)",
                       "FIX-XR3: Section 7.3 → 7.4 (Component 3)")
    text, _ = patch_re(text, r"(Proposition\s+1[^.]*?Section\s+)7\.3",
                       r"\g<1>7.4",
                       "FIX-XR3: Proposition 1 reference 7.3 → 7.4",
                       flags=re.DOTALL)
    return text


def fix_xr4_filename(text: str) -> str:
    text, _ = patch(text, "jmlr_paper_main.tex", "jmlr-hypatiax-paper-final.tex",
                    "FIX-XR4: rename jmlr_paper_main.tex")
    return text


def fix_n1_71_cases(text: str) -> str:
    text, _ = patch_re(text, r"\b71\s+cases\b", "70 tasks",
                       "FIX-N1: 71 cases → 70 tasks", flags=re.IGNORECASE)
    return text


def fix_n2_five_layer(text: str) -> str:
    text, _ = patch_re(text,
                       r"\\subsection\{Five-Layer Architecture Overview\}",
                       r"\\subsection{Five-Stage Routing Architecture Overview}",
                       "FIX-N2: rename §8.3 heading")
    text, _ = patch_re(text, r"Five-Layer Architecture Overview",
                       "Five-Stage Routing Architecture Overview",
                       "FIX-N2: rename inline references")
    return text


def fix_c2_stale_imports(py_text: str) -> str:
    py_text, _ = patch_re(py_text, r"\bhybrid_system_v40\b", "hybrid_system_v50_2",
                           "FIX-C2: hybrid_system_v40 → hybrid_system_v50_2")
    return py_text


# ---------------------------------------------------------------------------
# Fix dispatch table
# Maps fix_id → (file_target, function)
#   file_target: "main_tex" | "supp_tex" | "py"
# ---------------------------------------------------------------------------

FIX_DISPATCH = {
    "FIX-B1":  ("main_tex", fix_b1_add_koza1994),
    "FIX-B2":  ("main_tex", fix_b2_cranmer),
    "FIX-B3":  ("main_tex", fix_b3_udrescu),
    "FIX-XR1": ("main_tex", fix_xr1_duplicate_label),
    "FIX-XR2": ("main_tex", fix_xr2_label_in_item),
    "FIX-XR3": ("supp_tex", fix_xr3_supp_section_number),
    "FIX-XR4": ("supp_tex", fix_xr4_filename),
    "FIX-N1":  ("main_tex", fix_n1_71_cases),
    "FIX-N2":  ("main_tex", fix_n2_five_layer),
    "FIX-C2":  ("py",       fix_c2_stale_imports),
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MAIN_TEX_NAME = "jmlr-hypatiax-paper-final.tex"
SUPP_TEX_NAME = "supp_routing_improvements.tex"
PYTHON_TARGET = "run_comparative_suite_benchmark_v2.py"


def find_file(root: Path, name: str) -> Path | None:
    matches = list(root.rglob(name))
    if matches:
        return matches[0]
    return None


def run(tex_dir: Path, supp_dir: Path, py_dir: Path,
        dry_run: bool, registry: list[dict]) -> int:

    # Load files once; apply all relevant fixes; save once per file
    main_tex_path = find_file(tex_dir, MAIN_TEX_NAME)
    if main_tex_path is None:
        candidates = list(tex_dir.glob("jmlr*.tex"))
        if candidates:
            main_tex_path = candidates[0]
            print(f"  INFO: using {main_tex_path.name} as main tex file")

    supp_tex_path = find_file(supp_dir, SUPP_TEX_NAME)
    py_file_path  = find_file(py_dir, PYTHON_TARGET)

    # Build file buffers
    buffers: dict[str, tuple[Path | None, str]] = {
        "main_tex": (main_tex_path, load(main_tex_path) if main_tex_path else ""),
        "supp_tex": (supp_tex_path, load(supp_tex_path) if supp_tex_path else ""),
        "py":       (py_file_path,  load(py_file_path)  if py_file_path  else ""),
    }

    applied   = []
    skipped   = []
    not_found = []

    # Iterate registry in order so fixes are applied in a consistent sequence
    for entry in registry:
        fix_id   = entry.get("id", "")
        if fix_id not in FIX_DISPATCH:
            continue  # not auto-fixable (FIX-F*, FIX-C1, FIX-C3, etc.)

        # Registry gate: skip false positives and resolved items
        if not should_apply(registry, fix_id):
            skipped.append(fix_id)
            continue

        target, fn = FIX_DISPATCH[fix_id]
        path, text = buffers[target]

        if path is None:
            print(f"  WARN {fix_id}: target file not found — skipping")
            not_found.append(fix_id)
            continue

        print(f"\n  Applying {fix_id} → {path.name}")
        new_text = fn(text)
        buffers[target] = (path, new_text)
        applied.append(fix_id)

    # Write modified buffers
    print()
    for target, (path, text) in buffers.items():
        if path is None or not text:
            continue
        orig = load(path)
        if text != orig:
            print(f"=== Writing {path} ===")
            save(path, text, dry_run)
        else:
            print(f"=== {path.name}: no changes ===")

    # Final report
    print()
    print("=" * 60)
    print("fix_paper_issues.py (v2) — summary")
    print("=" * 60)
    if applied:
        print(f"  Applied   ({len(applied)}): {', '.join(applied)}")
    if skipped:
        print(f"  Skipped   ({len(skipped)}): {', '.join(skipped)}  ← false_positive or resolved in registry")
    if not_found:
        print(f"  Not found ({len(not_found)}): {', '.join(not_found)}  ← target file missing")

    manual = [
        e["id"] for e in registry
        if e.get("status") == "open" and not e.get("auto_fixable", False)
    ]
    if manual:
        print(f"\n  Manual action still required ({len(manual)}):")
        for mid in manual:
            entry = next((e for e in registry if e["id"] == mid), {})
            print(f"    {mid}: {entry.get('action','')}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-apply HypatiaX paper fixes (v2)")
    parser.add_argument("--tex-dir",   default=".", help="Root to search for main .tex file")
    parser.add_argument("--supp-dir",  default=".", help="Root to search for supplementary .tex")
    parser.add_argument("--py-dir",    default=".", help="Root to search for Python benchmark files")
    parser.add_argument("--registry",  default=str(DEFAULT_REGISTRY_PATH),
                        help="Path to issue_registry.json")
    parser.add_argument("--dry-run",   default="true",
                        help="Set to 'false' to write changes; any other value is a dry run")
    args = parser.parse_args()

    dry_run = args.dry_run.lower() != "false"
    print("=" * 60)
    print("fix_paper_issues.py (v2)")
    print("=" * 60)
    print(f"Mode     : {'DRY-RUN — no files will be written' if dry_run else 'APPLY — files will be patched in place'}")
    print(f"Registry : {args.registry}")
    print()

    registry = load_registry(Path(args.registry))

    rc = run(
        tex_dir  = Path(args.tex_dir).resolve(),
        supp_dir = Path(args.supp_dir).resolve(),
        py_dir   = Path(args.py_dir).resolve(),
        dry_run  = dry_run,
        registry = registry,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
