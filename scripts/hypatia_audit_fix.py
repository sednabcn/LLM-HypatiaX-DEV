#!/usr/bin/env python3
"""
hypatia_audit_fix.py
====================
Detection + fix script for the 4 live open issues flagged by the
HypatiaX paper audit (run 27792594147).

Issues handled
--------------
FIX-B1  CRITICAL  koza1994genetic missing \bibitem          → VERIFY (was false-positive)
FIX-F-new HIGH    hypatiaX_algorithm1_routing_cascade_v2    → DETECT only (hand-crafted fig)
FIX-N1  MEDIUM    "71 cases" vs "70 tasks" in Spearman fn   → AUTO-FIX
FIX-N2  MEDIUM    "Layer~N" terminology vs "five-stage"      → AUTO-FIX (terminology align)

Usage
-----
    python hypatia_audit_fix.py [--fix] [--tex PATH] [--bib PATH] [--figures-dir PATH]

    Without --fix  → detect-only mode, exit 0 if all clear, 1 if issues found.
    With    --fix  → apply safe text fixes and write patched files next to originals
                     (suffix _fixed).  Missing figure is always manual.
"""

import argparse
import re
import sys
from pathlib import Path


# ── configuration ────────────────────────────────────────────────────────────

DEFAULTS = {
    "tex": "jmlr_paper_main.tex",
    "bib": "references.bib",
    "figures_dir": "figures",
}

# FIX-N1: exact patterns
N1_BAD_PATTERN   = re.compile(r"\b71 cases\b")
N1_GOOD_TEXT     = "70 tasks"

# FIX-N2: the subsection body uses Layer~N; abstract/intro/§7 use "five-stage routing"
# We flag lines that say "Layer~<digit>" inside the routing-architecture subsection.
# The subsection runs from \label{sec:validation_framework} through \subsection{System Variants}.
N2_LAYER_RE      = re.compile(r"Layer~\d")
N2_SECTION_START = "sec:validation_framework"
N2_SECTION_END_RE = re.compile(r"\\subsection\{")

# FIX-B1: the bibitem must exist
B1_CITE_KEY      = "koza1994genetic"
B1_BIBITEM_RE    = re.compile(r"\\bibitem\[.*?\]\{koza1994genetic\}")
B1_CITE_RE       = re.compile(r"\\cite[tp]?\{[^}]*koza1994genetic[^}]*\}")

# FIX-F-new: figure file name expected on disk
FIGURE_FILENAME  = "hypatiaX_algorithm1_routing_cascade_v2"
FIGURE_INCLUDE_RE = re.compile(r"\\includegraphics\[.*?\]\{hypatiaX_algorithm1_routing_cascade_v2\}")


# ── helpers ──────────────────────────────────────────────────────────────────

def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def write(path: Path, content: str):
    path.write_text(content, encoding="utf-8")

def fixed_path(path: Path) -> Path:
    return path.with_stem(path.stem + "_fixed")


# ── detectors ────────────────────────────────────────────────────────────────

def check_b1(tex_text: str, bib_text: str) -> dict:
    """
    FIX-B1: koza1994genetic cited but no \\bibitem.
    The audit flagged this as CRITICAL, but the bib file DOES contain the entry,
    and the main .tex also has \\bibitem[Koza(1994)]{koza1994genetic} at line ~2000.
    This check verifies both are present → should resolve as FALSE POSITIVE.
    """
    result = {"id": "FIX-B1", "severity": "CRITICAL"}

    cite_lines = [
        i + 1
        for i, line in enumerate(tex_text.splitlines())
        if B1_CITE_RE.search(line)
    ]
    bibitem_lines = [
        i + 1
        for i, line in enumerate(tex_text.splitlines())
        if B1_BIBITEM_RE.search(line)
    ]
    bib_entry_present = f"@book{{{B1_CITE_KEY}" in bib_text or f"@article{{{B1_CITE_KEY}" in bib_text

    result["cite_lines"]           = cite_lines
    result["bibitem_in_tex_lines"] = bibitem_lines
    result["bib_entry_present"]    = bib_entry_present
    result["is_false_positive"]    = bool(bibitem_lines) and bib_entry_present

    if result["is_false_positive"]:
        result["verdict"] = (
            "FALSE POSITIVE — \\bibitem{koza1994genetic} found in tex "
            f"(line {bibitem_lines[0]}) and @book entry confirmed in .bib. "
            "No action needed; registry sync required."
        )
    else:
        missing = []
        if not bibitem_lines:
            missing.append("\\bibitem missing from .tex")
        if not bib_entry_present:
            missing.append("@book entry missing from .bib")
        result["verdict"] = (
            f"REAL ISSUE — {'; '.join(missing)}. "
            "Action: add \\bibitem{koza1994genetic} block or redirect cites to koza1992gp."
        )

    return result


def check_f_new(tex_text: str, figures_dir: Path) -> dict:
    """
    FIX-F-new: hypatiaX_algorithm1_routing_cascade_v2 must exist as PDF/PNG.
    Checks for exact match first, then case-insensitive match (Linux FS is
    case-sensitive so a mismatch still breaks the build).
    """
    result = {"id": "FIX-F-new", "severity": "HIGH"}

    include_lines = [
        i + 1
        for i, line in enumerate(tex_text.splitlines())
        if FIGURE_INCLUDE_RE.search(line)
    ]

    extensions = [".pdf", ".png", ".jpg", ".jpeg"]

    # 1. Exact match (what LaTeX will find)
    exact_files = []
    for ext in extensions:
        candidate = figures_dir / f"{FIGURE_FILENAME}{ext}"
        if candidate.exists():
            exact_files.append(str(candidate))

    # 2. Case-insensitive scan of the figures directory
    case_mismatch_files = []
    if figures_dir.exists():
        target_lower = FIGURE_FILENAME.lower()
        for f in figures_dir.iterdir():
            stem_lower = f.stem.lower()
            if stem_lower == target_lower and f.suffix.lower() in extensions:
                if str(f) not in exact_files:
                    case_mismatch_files.append(f.name)

    result["include_lines"]       = include_lines
    result["exact_files"]         = exact_files
    result["case_mismatch_files"] = case_mismatch_files

    if exact_files:
        result["is_false_positive"] = True
        result["verdict"] = (
            f"FALSE POSITIVE — figure file found on disk (exact match): {exact_files}. "
            "No build breakage expected."
        )
    elif case_mismatch_files:
        result["is_false_positive"] = False
        result["subtype"] = "case_mismatch"
        result["verdict"] = (
            f"REAL ISSUE (case mismatch) — LaTeX includes '{FIGURE_FILENAME}' "
            f"but disk has: {case_mismatch_files}. "
            "On Linux/macOS this WILL break the build. "
            f"Action: rename the file to '{FIGURE_FILENAME}.pdf' (or .png)."
        )
    else:
        result["is_false_positive"] = False
        result["subtype"] = "missing"
        result["verdict"] = (
            f"REAL ISSUE — figure file not found in {figures_dir}/ (tried {extensions}, "
            "case-insensitive scan also found nothing). "
            "Action: place the hand-crafted PDF/PNG there manually. "
            "This WILL break the LaTeX build."
        )

    return result


def check_n1(tex_text: str) -> dict:
    """
    FIX-N1: 'XX cases' in a footnote/body line where the correct value is '70 tasks'.
    The audit says line 1637; we search the whole file for safety.
    """
    result = {"id": "FIX-N1", "severity": "MEDIUM"}

    bad_lines = [
        (i + 1, line.strip())
        for i, line in enumerate(tex_text.splitlines())
        if N1_BAD_PATTERN.search(line)
    ]

    result["bad_lines"] = bad_lines
    result["is_false_positive"] = len(bad_lines) == 0

    if bad_lines:
        result["verdict"] = (
            f"REAL ISSUE — '71 cases' found on {len(bad_lines)} line(s): "
            + ", ".join(str(ln) for ln, _ in bad_lines)
            + ". Action: replace with '70 tasks'."
        )
    else:
        result["verdict"] = "FALSE POSITIVE — '71 cases' not present; no fix needed."

    return result


def check_n2(tex_text: str) -> dict:
    """
    FIX-N2: Body of §8.3 (sec:validation_framework) uses 'Layer~N' while
    the rest of the paper consistently says 'five-stage routing'.
    The subsection title itself is already correct ('Five-Stage Routing Architecture Overview').
    The BODY uses Layer~1, Layer~3, Layer~4, Layer~5 — these are the flagged lines.
    """
    result = {"id": "FIX-N2", "severity": "MEDIUM"}

    lines = tex_text.splitlines()
    in_section = False
    layer_lines = []

    for i, line in enumerate(lines):
        if N2_SECTION_START in line:
            in_section = True
            continue
        if in_section:
            # Stop at the next \subsection (but not the one that opened us)
            if N2_SECTION_END_RE.search(line):
                break
            if N2_LAYER_RE.search(line):
                layer_lines.append((i + 1, line.strip()))

    result["layer_lines"]      = layer_lines
    result["is_false_positive"] = len(layer_lines) == 0

    if layer_lines:
        result["verdict"] = (
            f"REAL ISSUE — {len(layer_lines)} line(s) in §sec:validation_framework "
            "use 'Layer~N' terminology inconsistent with 'five-stage routing' used everywhere else. "
            "Action: rename Stage~1, Stage~2 etc. (or rephrase as prose)."
        )
    else:
        result["verdict"] = "FALSE POSITIVE — no 'Layer~N' found in the section; no fix needed."

    return result


# ── fixers ───────────────────────────────────────────────────────────────────

def fix_n1(tex_text: str) -> tuple[str, int]:
    """Replace '71 cases' → '70 tasks'."""
    new_text, count = N1_BAD_PATTERN.subn(N1_GOOD_TEXT, tex_text)
    return new_text, count


def fix_n2(tex_text: str) -> tuple[str, list]:
    """
    Within the sec:validation_framework subsection body, rename
    'Layer~N' → 'Stage~N' to align with the rest of the paper's
    'five-stage routing' language.  Stage numbering is preserved
    (Layer~1 → Stage~1, etc.).
    """
    lines = tex_text.splitlines()
    in_section = False
    changes = []

    for i, line in enumerate(lines):
        if N2_SECTION_START in line:
            in_section = True
            continue
        if in_section:
            if N2_SECTION_END_RE.search(line) and i > 0:
                in_section = False
            if N2_LAYER_RE.search(line):
                new_line = re.sub(r"Layer~(\d)", r"Stage~\1", line)
                if new_line != line:
                    changes.append((i + 1, line.strip(), new_line.strip()))
                    lines[i] = new_line

    return "\n".join(lines), changes


# ── reporting ─────────────────────────────────────────────────────────────────

def print_result(r: dict, verbose: bool = True):
    status = "✅ FALSE POSITIVE" if r["is_false_positive"] else "❌ REAL ISSUE"
    print(f"\n{'─'*60}")
    print(f"  {r['id']}  [{r['severity']}]  →  {status}")
    if verbose:
        print(f"  {r['verdict']}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HypatiaX audit: detect & fix open issues")
    parser.add_argument("--fix",          action="store_true", help="Apply auto-fixes")
    parser.add_argument("--tex",          default=DEFAULTS["tex"],         help="Main .tex file")
    parser.add_argument("--bib",          default=DEFAULTS["bib"],         help="Bibliography .bib file")
    parser.add_argument("--figures-dir",  default=DEFAULTS["figures_dir"], help="Figures directory")
    parser.add_argument("--quiet",        action="store_true", help="Suppress per-finding detail")
    args = parser.parse_args()

    tex_path     = Path(args.tex)
    bib_path     = Path(args.bib)
    figures_dir  = Path(args.figures_dir)
    verbose      = not args.quiet

    # ── load files ──
    if not tex_path.exists():
        print(f"ERROR: tex file not found: {tex_path}", file=sys.stderr)
        sys.exit(2)
    if not bib_path.exists():
        print(f"WARNING: bib file not found: {bib_path} — skipping FIX-B1 bib check",
              file=sys.stderr)
        bib_text = ""
    else:
        bib_text = read(bib_path)

    tex_text = read(tex_path)

    print("=" * 60)
    print("  HypatiaX Audit — Detection Run")
    print(f"  tex: {tex_path}   bib: {bib_path}")
    print("=" * 60)

    # ── run checks ──
    b1      = check_b1(tex_text, bib_text)
    f_new   = check_f_new(tex_text, figures_dir)
    n1      = check_n1(tex_text)
    n2      = check_n2(tex_text)

    results = [b1, f_new, n1, n2]
    for r in results:
        print_result(r, verbose)

    real_issues     = [r for r in results if not r["is_false_positive"]]
    false_positives = [r for r in results if r["is_false_positive"]]

    print(f"\n{'─'*60}")
    print(f"  Summary: {len(real_issues)} real issue(s), {len(false_positives)} false positive(s)")

    # ── apply fixes ──
    if args.fix:
        print(f"\n{'─'*60}")
        print("  Applying auto-fixes …")

        modified = False

        if n1 and not n1["is_false_positive"]:
            tex_text, count = fix_n1(tex_text)
            if count:
                print(f"  FIX-N1: replaced {count} occurrence(s) of '71 cases' → '70 tasks'")
                modified = True
            else:
                print("  FIX-N1: nothing to replace (already correct)")

        if n2 and not n2["is_false_positive"]:
            tex_text, changes = fix_n2(tex_text)
            if changes:
                print(f"  FIX-N2: updated {len(changes)} line(s) — Layer~N → Stage~N:")
                for ln, before, after in changes:
                    print(f"    line {ln}:")
                    print(f"      before: {before}")
                    print(f"      after:  {after}")
                modified = True
            else:
                print("  FIX-N2: nothing to replace (already correct)")

        if modified:
            out_path = fixed_path(tex_path)
            write(out_path, tex_text)
            print(f"\n  ✅ Fixed tex written to: {out_path}")
        else:
            print("  No text changes were needed.")

        # FIX-B1 — resolved as FP, no tex change needed
        if b1["is_false_positive"]:
            print(
                "\n  FIX-B1: confirmed false positive — \\bibitem present. "
                "Registry sync recommended but no file change required."
            )
        else:
            print(
                "\n  FIX-B1: real issue detected — manual fix required. "
                "Add \\bibitem{koza1994genetic} or redirect cite to koza1992gp."
            )

        # FIX-F-new — case mismatch can be auto-renamed; truly missing is always manual
        if f_new["is_false_positive"]:
            print(f"\n  FIX-F-new: figure already on disk (exact match) — no action needed.")
        elif f_new.get("subtype") == "case_mismatch":
            # rename the mismatched file to the exact name LaTeX expects
            mismatched = f_new["case_mismatch_files"]
            renamed = []
            for name in mismatched:
                old = figures_dir / name
                suffix = Path(name).suffix
                new = figures_dir / f"{FIGURE_FILENAME}{suffix}"
                try:
                    old.rename(new)
                    renamed.append(f"{old} → {new}")
                    modified = True
                except Exception as e:
                    renamed.append(f"FAILED to rename {old}: {e}")
            print(
                f"\n  FIX-F-new: ✅ case-mismatch auto-renamed:\n"
                + "\n".join(f"    {r}" for r in renamed)
            )
        else:
            print(
                f"\n  FIX-F-new: ⚠  Figure {FIGURE_FILENAME}.(pdf|png) is missing. "
                f"Place it in {figures_dir}/ manually (hand-crafted; no generator exists)."
            )

    # ── exit code ──
    sys.exit(1 if real_issues else 0)


if __name__ == "__main__":
    main()
