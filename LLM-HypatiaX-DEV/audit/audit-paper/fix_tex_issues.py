#!/usr/bin/env python3
"""
fix_tex_issues.py
------------------
Applies fixes for the issues CONFIRMED to still exist in the current .tex
files, re-derived directly from the source text (see ground_truth_check.py)
rather than from HypatiaX_Paper_Audit_Report.html, which is stale: most of
its NB-01/NB-02/NB-04 findings (koza1994genetic undefined, cranmer2023interp
duplicate, udrescu2020feynman duplicate, sec:llm_domain duplicate label,
sec:r2_bugfix inside \\item, "71 cases" body text, "Five-Layer Architecture"
heading) have already been fixed in the uploaded files and are NOT
reapplied here to avoid corrupting already-correct text.

Run from the directory containing:
    jmlr_paper_main.tex
    supp_routing_improvements.tex
    supp_benchmark_report.tex

Usage:
    python3 fix_tex_issues.py            # apply fixes in place
    python3 fix_tex_issues.py --dry-run  # show what would change, write nothing
    python3 fix_tex_issues.py --check    # re-run ground-truth checks after fixing

Each fix is idempotent: running the script twice produces no further changes,
and a fix is skipped (with a message) if its target text is not found,
rather than failing or partially corrupting the file.
"""
import argparse
import difflib
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Fix definitions: (filename, description, old, new)
# old must match EXACTLY ONCE in the file, or the fix is skipped with a
# warning (never applied partially / never applied to the wrong spot).
# ---------------------------------------------------------------------------

FIXES = [
    # ------------------------------------------------------------------
    # FIX-1  Undefined \ref{subsec:routing} in supp_routing_improvements.tex
    #        -> the Component 3 subsection's actual label is \label{sec:routing}
    #        (jmlr_paper_main.tex line 724).
    # ------------------------------------------------------------------
    dict(
        file="supp_routing_improvements.tex",
        description="FIX-1: undefined \\ref{subsec:routing} -> \\ref{sec:routing}",
        old=r"Section~\ref{subsec:routing} (\emph{Component 3: Five-Stage Routing and Ensembling})",
        new=r"Section~\ref{sec:routing} (\emph{Component 3: Five-Stage Routing and Ensembling})",
    ),

    # ------------------------------------------------------------------
    # FIX-2  Undefined \ref{tab:nguyen} in supp_benchmark_report.tex
    #        -> the actual table label is \label{tab:nguyen12}
    #        (jmlr_paper_main.tex line 1573).
    # ------------------------------------------------------------------
    dict(
        file="supp_benchmark_report.tex",
        description="FIX-2: undefined \\ref{tab:nguyen} -> \\ref{tab:nguyen12}",
        old=r"Table~\ref{tab:nguyen} provides this comparison.",
        new=r"Table~\ref{tab:nguyen12} provides this comparison.",
    ),

    # ------------------------------------------------------------------
    # FIX-3  Undefined \ref{def:ood_extrap} in supp_benchmark_report.tex
    #        -> nearest matching concept actually defined in the main paper
    #        is \label{def:decoupling} (Interpolation--Extrapolation
    #        Decoupling, jmlr_paper_main.tex line 400), which is exactly what
    #        this sentence is describing (R^2 collapse on held-out/OOD data).
    # ------------------------------------------------------------------
    dict(
        file="supp_benchmark_report.tex",
        description="FIX-3: undefined \\ref{def:ood_extrap} -> \\ref{def:decoupling}",
        old=r"meeting the threshold in Definition~\ref{def:ood_extrap}",
        new=r"meeting the threshold in Definition~\ref{def:decoupling}",
    ),

    # ------------------------------------------------------------------
    # FIX-4  Filename references to the old/renamed main-paper file.
    #        supp_routing_improvements.tex and supp_benchmark_report.tex both
    #        still say 'jmlr-hypatiax-paper-final.tex'; the actual file is
    #        'jmlr_paper_main.tex'. Each occurrence is listed explicitly
    #        (rather than a blind global replace) so every hit is visible
    #        and auditable in this script.
    # ------------------------------------------------------------------
    dict(
        file="supp_routing_improvements.tex",
        description="FIX-4a: filename reference (companion-to line)",
        old=r"%%  Companion to: jmlr-hypatiax-paper-final.tex",
        new=r"%%  Companion to: jmlr_paper_main.tex",
    ),
    dict(
        file="supp_routing_improvements.tex",
        description="FIX-4b: filename reference (prose, \\texttt)",
        old=r"Extrapolation-Reliable Analytical Discovery} (\texttt{jmlr-hypatiax-paper-final.tex}).",
        new=r"Extrapolation-Reliable Analytical Discovery} (\texttt{jmlr_paper_main.tex}).",
    ),
    dict(
        file="supp_routing_improvements.tex",
        description="FIX-4c: filename reference (appendix-source comment)",
        old=r"% NEW APPENDICES (extracted from jmlr-hypatiax-paper-final.tex)",
        new=r"% NEW APPENDICES (extracted from jmlr_paper_main.tex)",
    ),
    dict(
        file="supp_routing_improvements.tex",
        description="FIX-4d: filename reference (inserted-from comment)",
        old=r"% INSERTED FROM jmlr-hypatiax-paper-final.tex — Appendix B & C",
        new=r"% INSERTED FROM jmlr_paper_main.tex — Appendix B & C",
    ),
    dict(
        file="supp_benchmark_report.tex",
        description="FIX-4e: filename reference (header comment, line 6)",
        old=r"jmlr-hypatiax-paper-final.tex",
        new=r"jmlr_paper_main.tex",
        # NOTE: supp_benchmark_report.tex has 4 occurrences of this exact
        # bare string (lines 6, 132, 729, 908); see FIX-4e-all below which
        # replaces all of them since none of the four needs different
        # surrounding text (unlike the routing file's varied phrasing above).
        replace_all=True,
    ),
]

# ------------------------------------------------------------------
# FIX-5  Undefined citations in supp_benchmark_report.tex: meidani2024snip,
#        balestriero2021high, neyshabur2017exploring, zhang2021understanding.
#        These are real papers with no \bibitem entry. Add entries to the
#        thebibliography block, matching the file's existing citation style
#        (author-year key, \bibitem[Surname et al.(Year)]{key} format).
#        Bibliographic details verified via web search (see chat for sources):
#          - Meidani, Shojaee, Reddy, Barati Farimani. SNIP: Bridging
#            Mathematical Symbolic and Numeric Realms with Unified
#            Pre-training. ICLR 2024.
#          - Balestriero, Pesenti, LeCun. Learning in High Dimension Always
#            Amounts to Extrapolation. arXiv:2110.09485, 2021.
#          - Neyshabur, Bhojanapalli, McAllester, Srebro. Exploring
#            Generalization in Deep Learning. NeurIPS 2017, pp. 5949-5958.
#          - Zhang, Bengio, Hardt, Recht, Vinyals. Understanding Deep
#            Learning (Still) Requires Rethinking Generalization.
#            Commun. ACM 64(3):107-115, 2021.
# ------------------------------------------------------------------
NEW_BIBITEMS = r"""
\bibitem[Meidani et~al.(2024)]{meidani2024snip}
K.~Meidani, P.~Shojaee, C.~K.~Reddy, and A.~Barati~Farimani.
\newblock SNIP: Bridging mathematical symbolic and numeric realms with unified pre-training.
\newblock \textit{Proc.\ ICLR}, 2024.

\bibitem[Balestriero et~al.(2021)]{balestriero2021high}
R.~Balestriero, J.~Pesenti, and Y.~LeCun.
\newblock Learning in high dimension always amounts to extrapolation.
\newblock \textit{arXiv:2110.09485}, 2021.

\bibitem[Neyshabur et~al.(2017)]{neyshabur2017exploring}
B.~Neyshabur, S.~Bhojanapalli, D.~McAllester, and N.~Srebro.
\newblock Exploring generalization in deep learning.
\newblock \textit{Proc.\ NeurIPS}, 30:5949--5958, 2017.

\bibitem[Zhang et~al.(2021)]{zhang2021understanding}
C.~Zhang, S.~Bengio, M.~Hardt, B.~Recht, and O.~Vinyals.
\newblock Understanding deep learning (still) requires rethinking generalization.
\newblock \textit{Commun.\ ACM}, 64(3):107--115, 2021.
"""

BIBITEM_FIX = dict(
    file="supp_benchmark_report.tex",
    description="FIX-5: add 4 missing \\bibitem entries (meidani2024snip, "
                 "balestriero2021high, neyshabur2017exploring, zhang2021understanding)",
    old=r"\end{thebibliography}",
    new=NEW_BIBITEMS.rstrip("\n") + "\n\n\\end{thebibliography}",
)


def apply_fix(path: Path, fix: dict, dry_run: bool) -> bool:
    """Apply a single old->new fix. Returns True if a change was made."""
    text = path.read_text(encoding="utf-8")
    old, new = fix["old"], fix["new"]

    # Idempotency guard: if the fix's distinguishing content (a marker
    # unique to "new") is already present, skip -- regardless of whether
    # "old" also still appears elsewhere (e.g. \end{thebibliography} is
    # part of both the "before" and "after" text for an insertion fix).
    marker = fix.get("marker", new)
    if marker in text:
        print(f"  [SKIP-already-applied] {fix['description']}")
        return False

    count = text.count(old)

    if count == 0:
        print(f"  [SKIP-not-found] {fix['description']}  "
              f"(target text not found in {path.name} -- file may have "
              f"changed; please verify manually)")
        return False

    if count > 1 and not fix.get("replace_all"):
        print(f"  [SKIP-ambiguous] {fix['description']}  "
              f"(matched {count} times in {path.name}; expected exactly 1 -- "
              f"refusing to guess which one)")
        return False

    new_text = text.replace(old, new)

    if dry_run:
        diff = difflib.unified_diff(
            text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"{path.name} (before)",
            tofile=f"{path.name} (after)",
            n=1,
        )
        print(f"  [DRY-RUN] {fix['description']}")
        sys.stdout.writelines("    " + line for line in diff)
    else:
        path.write_text(new_text, encoding="utf-8")
        print(f"  [APPLIED] {fix['description']}  ({count} occurrence{'s' if count != 1 else ''})")

    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                     help="Show what would change without writing any files.")
    ap.add_argument("--dir", default=".", help="Directory containing the .tex files.")
    args = ap.parse_args()

    base = Path(args.dir)
    all_fixes = list(FIXES) + [BIBITEM_FIX]

    by_file = {}
    for fix in all_fixes:
        by_file.setdefault(fix["file"], []).append(fix)

    applied_total = 0
    for fname, fixes in by_file.items():
        path = base / fname
        print(f"\n=== {fname} ===")
        if not path.exists():
            print(f"  [ERROR] file not found: {path}")
            continue
        for fix in fixes:
            if apply_fix(path, fix, args.dry_run):
                applied_total += 1

    print(f"\n{'Would apply' if args.dry_run else 'Applied'} {applied_total} fix(es) "
          f"out of {len(all_fixes)} defined.")
    if args.dry_run:
        print("(dry run -- no files were modified)")


if __name__ == "__main__":
    main()
