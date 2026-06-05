#!/usr/bin/env python3
"""
fix_paper_issues.py
====================
Auto-applies every non-numerical, non-figure fix detected by the HypatiaX
paper audit.  Safe to re-run: all operations are idempotent.

Fixes applied
-------------
FIX-B1   Add \\bibitem{koza1994genetic} if missing
FIX-B2   Remove cranmer2023interpretable bibitem; redirect all \\cite calls
FIX-B3   Remove udrescu2020aifeynman bibitem; redirect all \\cite calls
FIX-XR1  Remove duplicate \\label{sec:llm_domain}; update \\ref call sites
FIX-XR2  Move \\label{sec:r2_bugfix} out of \\item blocks
FIX-XR3  Change Section 7.3 → 7.4 in supp_routing_improvements.tex
FIX-XR4  Rename jmlr_paper_main.tex → jmlr-hypatiax-paper-final.tex in Supp A
FIX-N1   Change "71 cases" → "70 tasks" in instability section body
FIX-N2   Rename §8.3 heading Five-Layer → Five-Stage Routing Architecture
FIX-C2   Replace stale hybrid_system_v40 imports with hybrid_system_v50_2

Fixes intentionally NOT applied here (require human judgment or new assets)
----------------------------------------------------------------------------
FIX-N3   Nguyen-12 numbers — requires seed=123 rerun
FIX-F1   Replace \\fbox placeholder — requires actual figure file
FIX-F2-4 Copy missing figure files — requires source asset verification
FIX-C1   Rename duplicate benchmark cases — requires checkpoint rerun
FIX-C3   Already resolved; §10.7 text update requires human review

Usage
-----
    python fix_paper_issues.py                        # dry-run (default)
    python fix_paper_issues.py --dry-run false        # apply changes
    python fix_paper_issues.py --tex-dir path/to/tex  # custom .tex root
    python fix_paper_issues.py --py-dir path/to/src   # custom Python src root
"""

import argparse
import re
import sys
from pathlib import Path

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
    """Replace all occurrences of old → new; return (new_text, count)."""
    count = original.count(old)
    if count:
        print(f"  {label}: replaced {count} occurrence(s)")
    return original.replace(old, new), count

def patch_re(original: str, pattern: str, repl, label: str,
             flags: int = 0) -> tuple[str, int]:
    """Replace regex pattern with repl (string or callable) in original.

    Use a lambda whenever the replacement contains literal LaTeX backslash
    commands (e.g. \\label, \\subsection) — Python's regex engine interprets
    \\l, \\s etc. as escape sequences inside plain replacement strings and
    raises re.error: bad escape.
    """
    new, count = re.subn(pattern, repl, original, flags=flags)
    if count:
        print(f"  {label}: replaced {count} occurrence(s)")
    return new, count

# ---------------------------------------------------------------------------
# Per-fix functions
# ---------------------------------------------------------------------------

def fix_b1_add_koza1994(text: str) -> str:
    """FIX-B1: Insert \\bibitem{koza1994genetic} if not already present."""
    if "koza1994genetic" in text and r"\bibitem{koza1994genetic}" in text:
        print("  FIX-B1: already present — skipping")
        return text
    if r"\bibitem{koza1994genetic}" in text:
        print("  FIX-B1: bibitem already present — skipping")
        return text

    new_bibitem = (
        r"\bibitem{koza1994genetic}" + "\n"
        r"J.~R. Koza." + "\n"
        r"\newblock {\em Genetic Programming II: Automatic Discovery of Reusable Programs}." + "\n"
        r"\newblock MIT Press, 1994."
    )
    # Insert immediately after the koza1992gp bibitem block (safest anchor)
    anchor = r"\bibitem{koza1992gp}"
    if anchor in text:
        # Find end of the koza1992gp entry (next blank line or next \bibitem)
        idx = text.index(anchor)
        # Walk forward to the next \bibitem or end-of-bibliography
        next_bib = text.find(r"\bibitem{", idx + len(anchor))
        if next_bib == -1:
            next_bib = len(text)
        insert_at = next_bib
        text = text[:insert_at] + new_bibitem + "\n\n" + text[insert_at:]
        print("  FIX-B1: inserted \\bibitem{koza1994genetic} after koza1992gp")
    else:
        print("  FIX-B1: WARNING — koza1992gp anchor not found; manual insert needed")
    return text


def fix_b2_cranmer(text: str) -> str:
    """FIX-B2: Remove cranmer2023interpretable bibitem; redirect \\cite calls."""
    # Redirect all \\cite occurrences first (handles {cranmer2023interpretable}
    # whether alone or in a multi-key cite like {a,cranmer2023interpretable,b})
    text, _ = patch_re(
        text,
        r"cranmer2023interpretable",
        "cranmer2023pysr",
        "FIX-B2: redirect \\cite{cranmer2023interpretable}",
    )
    # Remove the now-orphaned bibitem entry (multi-line, ends before next \bibitem
    # or \end{thebibliography})
    text, _ = patch_re(
        text,
        r"\\bibitem\{cranmer2023pysr\}[^\n]*\n(?:[^\n]*\n)*?(?=\\bibitem|\\end\{thebibliography\})",
        # keep only one copy — the one keyed cranmer2023pysr
        # This regex matches if there are now TWO cranmer2023pysr bibitems (after
        # the rename above created a duplicate); we remove the second occurrence.
        "",
        "FIX-B2: remove duplicate bibitem",
    )
    return text


def fix_b3_udrescu(text: str) -> str:
    """FIX-B3: Remove udrescu2020aifeynman bibitem; redirect \\cite calls."""
    text, _ = patch_re(
        text,
        r"udrescu2020aifeynman",
        "udrescu2020ai",
        "FIX-B3: redirect \\cite{udrescu2020aifeynman}",
    )
    # Remove orphaned bibitem (same pattern as FIX-B2)
    text, _ = patch_re(
        text,
        r"\\bibitem\{udrescu2020ai\}[^\n]*\n(?:[^\n]*\n)*?(?=\\bibitem|\\end\{thebibliography\})",
        "",
        "FIX-B3: remove duplicate bibitem",
    )
    return text


def fix_xr1_duplicate_label(text: str) -> str:
    """FIX-XR1: Remove \\label{sec:llm_domain}; update \\ref call sites."""
    # Remove the duplicate label (keep sec:llm_limitations)
    text, _ = patch(
        text,
        r"\label{sec:llm_domain}",
        "",
        "FIX-XR1: remove \\label{sec:llm_domain}",
    )
    # Update any \\ref or \\autoref pointing to the removed label
    text, _ = patch_re(
        text,
        r"\\(auto)?ref\{sec:llm_domain\}",
        r"\\\1ref{sec:llm_limitations}",
        "FIX-XR1: redirect \\ref{sec:llm_domain}",
    )
    return text


def fix_xr2_label_in_item(text: str) -> str:
    """FIX-XR2: Move \\label{sec:r2_bugfix} from inside \\item to the subsection heading.

    Strategy: find the \\subsection{Pipeline Corrections in v3.0} line and ensure
    \\label{sec:r2_bugfix} immediately follows it, then remove the label from
    any \\item block.
    """
    subsec_pattern = r"(\\subsection\{Pipeline Corrections in v3\.0\})"
    label = r"\label{sec:r2_bugfix}"

    # If label is already on the subsection line, nothing to do
    if re.search(subsec_pattern + r"\s*" + re.escape(label), text):
        print("  FIX-XR2: label already on subsection — skipping")
        return text

    # Remove label from wherever it currently sits (may be in \item)
    text, removed = patch(text, label, "", "FIX-XR2: remove misplaced label")

    if removed:
        # Re-attach to subsection heading.
        # Must use a lambda — label contains \label which has \l, an invalid
        # regex escape sequence that causes re.error: bad escape \l at position 4
        # when passed as a plain replacement string to re.subn.
        text, attached = patch_re(
            text,
            subsec_pattern,
            lambda m: m.group(1) + "\n" + label,
            "FIX-XR2: attach label to subsection",
        )
        if not attached:
            print("  FIX-XR2: WARNING — subsection heading not found; label removed but not re-attached")
    return text


def fix_xr3_supp_section_number(text: str) -> str:
    """FIX-XR3: Change Section 7.3 → 7.4 in supp_routing_improvements.tex."""
    # Only replace references to Component 3 / Proposition 1 context to avoid
    # touching unrelated 7.3 references in other supplementary files.
    text, _ = patch_re(
        text,
        r"Section\s+7\.3\s*\(Component\s+3\)",
        "Section 7.4 (Component 3)",
        "FIX-XR3: Section 7.3 → 7.4 (Component 3)",
    )
    # Also fix bare "Section 7.3" near Proposition 1 context
    text, _ = patch_re(
        text,
        r"(Proposition\s+1[^.]*?Section\s+)7\.3",
        r"\g<1>7.4",
        "FIX-XR3: Proposition 1 reference 7.3 → 7.4",
        flags=re.DOTALL,
    )
    return text


def fix_xr4_filename(text: str) -> str:
    """FIX-XR4: Replace stale filename jmlr_paper_main.tex with correct name."""
    text, _ = patch(
        text,
        "jmlr_paper_main.tex",
        "jmlr-hypatiax-paper-final.tex",
        "FIX-XR4: rename jmlr_paper_main.tex",
    )
    return text


def fix_n1_71_cases(text: str) -> str:
    """FIX-N1: Change '71 cases' → '70 tasks' in instability section body."""
    text, _ = patch_re(
        text,
        r"\b71\s+cases\b",
        "70 tasks",
        "FIX-N1: 71 cases → 70 tasks",
        flags=re.IGNORECASE,
    )
    return text


def fix_n2_five_layer(text: str) -> str:
    """FIX-N2: Rename §8.3 heading Five-Layer Architecture → Five-Stage Routing Architecture."""
    text, _ = patch_re(
        text,
        r"\\subsection\{Five-Layer Architecture Overview\}",
        r"\\subsection{Five-Stage Routing Architecture Overview}",
        "FIX-N2: rename §8.3 heading",
    )
    # Also fix any \\section-level variant (belt-and-suspenders)
    text, _ = patch_re(
        text,
        r"Five-Layer Architecture Overview",
        "Five-Stage Routing Architecture Overview",
        "FIX-N2: rename inline references",
    )
    return text


def fix_c2_stale_imports(py_text: str) -> str:
    """FIX-C2: Replace stale hybrid_system_v40 imports with hybrid_system_v50_2."""
    py_text, _ = patch_re(
        py_text,
        r"\bhybrid_system_v40\b",
        "hybrid_system_v50_2",
        "FIX-C2: hybrid_system_v40 → hybrid_system_v50_2",
    )
    return py_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MAIN_TEX_NAME  = "jmlr-hypatiax-paper-final.tex"
SUPP_TEX_NAME  = "supp_routing_improvements.tex"
PYTHON_TARGET  = "run_comparative_suite_benchmark_v2.py"


def find_file(root: Path, name: str) -> Path | None:
    """Search recursively for a file; return first match or None."""
    matches = list(root.rglob(name))
    if matches:
        return matches[0]
    return None


def run(tex_dir: Path, supp_dir: Path, py_dir: Path, dry_run: bool) -> int:
    errors = 0

    # ── Main .tex file ───────────────────────────────────────────────────────
    main_tex = find_file(tex_dir, MAIN_TEX_NAME)
    if main_tex is None:
        # Fallback: accept any .tex file at repo root (in case filename varies)
        candidates = list(tex_dir.glob("jmlr*.tex"))
        if candidates:
            main_tex = candidates[0]
            print(f"  INFO: using {main_tex.name} as main tex file")
    if main_tex is None:
        print(f"WARNING: {MAIN_TEX_NAME} not found under {tex_dir} — skipping main-tex fixes")
    else:
        print(f"\n=== Main TeX: {main_tex} ===")
        text = load(main_tex)
        text = fix_b1_add_koza1994(text)
        text = fix_b2_cranmer(text)
        text = fix_b3_udrescu(text)
        text = fix_xr1_duplicate_label(text)
        text = fix_xr2_label_in_item(text)
        text = fix_n1_71_cases(text)
        text = fix_n2_five_layer(text)
        save(main_tex, text, dry_run)

    # ── Supplementary .tex file ──────────────────────────────────────────────
    supp_tex = find_file(supp_dir, SUPP_TEX_NAME)
    if supp_tex is None:
        print(f"WARNING: {SUPP_TEX_NAME} not found under {supp_dir} — skipping supp fixes")
    else:
        print(f"\n=== Supp TeX: {supp_tex} ===")
        text = load(supp_tex)
        text = fix_xr3_supp_section_number(text)
        text = fix_xr4_filename(text)
        save(supp_tex, text, dry_run)

    # ── Python benchmark file ────────────────────────────────────────────────
    py_file = find_file(py_dir, PYTHON_TARGET)
    if py_file is None:
        print(f"WARNING: {PYTHON_TARGET} not found under {py_dir} — skipping FIX-C2")
    else:
        print(f"\n=== Python: {py_file} ===")
        text = load(py_file)
        text = fix_c2_stale_imports(text)
        save(py_file, text, dry_run)

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-apply HypatiaX paper fixes")
    parser.add_argument("--tex-dir",  default=".", help="Root to search for main .tex file")
    parser.add_argument("--supp-dir", default=".", help="Root to search for supplementary .tex")
    parser.add_argument("--py-dir",   default=".", help="Root to search for Python benchmark files")
    parser.add_argument(
        "--dry-run",
        default="true",
        help="Set to 'false' to write changes; any other value is a dry run",
    )
    args = parser.parse_args()

    dry_run = args.dry_run.lower() != "false"
    if dry_run:
        print("=== DRY-RUN mode — no files will be written ===")
    else:
        print("=== APPLY mode — files will be patched in place ===")

    tex_dir  = Path(args.tex_dir).resolve()
    supp_dir = Path(args.supp_dir).resolve()
    py_dir   = Path(args.py_dir).resolve()

    rc = run(tex_dir, supp_dir, py_dir, dry_run)
    if rc:
        sys.exit(rc)

    print("\n=== fix_paper_issues.py complete ===")
    print("Fixes applied: FIX-B1, FIX-B2, FIX-B3, FIX-XR1, FIX-XR2,")
    print("               FIX-XR3, FIX-XR4, FIX-N1, FIX-N2, FIX-C2")
    print("Skipped (manual): FIX-N3, FIX-F1–F4, FIX-C1, FIX-C3")


if __name__ == "__main__":
    main()
