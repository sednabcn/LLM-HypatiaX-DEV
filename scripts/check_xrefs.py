#!/usr/bin/env python3
"""Static \\label / \\ref / \\cite cross-reference checker for the HypatiaX paper.

Usage:
    check_xrefs.py MAIN.tex SUPP_A.tex SUPP_B.tex REFERENCES.bib

This is a warn-only check (real LaTeX errors are caught by the compile step
itself) and is shared by `make validate` and the ci_create_pdf.yml validate
job, so the logic lives in exactly one place instead of being duplicated —
and re-implemented with fragile inline awk/sed — in the workflow YAML.

When run inside GitHub Actions ($GITHUB_ACTIONS=true):
  - undefined refs/cites are reported as ::warning:: annotations
  - undef_refs / undef_cites counts are appended to $GITHUB_OUTPUT
Always exits 0 — this check never fails the build on its own.
"""
import os
import re
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 5:
        print("usage: check_xrefs.py MAIN.tex SUPP_A.tex SUPP_B.tex REFERENCES.bib", file=sys.stderr)
        return 2

    main_tex, supp_a, supp_b, bib_file = sys.argv[1:5]
    tex_files = [main_tex, supp_a, supp_b]

    label_set = set()
    for tex in tex_files:
        p = Path(tex)
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="replace")
            label_set |= set(re.findall(r"\\label\{([^}]+)\}", text))

    main_path = Path(main_tex)
    main_src = main_path.read_text(encoding="utf-8", errors="replace") if main_path.exists() else ""

    refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", main_src))
    cites = {
        key.strip()
        for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", main_src)
        for key in group.split(",")
    }

    bib_path = Path(bib_file)
    bib_keys = set()
    if bib_path.exists():
        bib_text = bib_path.read_text(encoding="utf-8", errors="replace")
        bib_keys = set(re.findall(r"@\w{2,20}\{([^,\s]+)", bib_text))

    undef_refs = sorted(refs - label_set)
    undef_cites = sorted(cites - bib_keys)

    print(f"Labels found: {len(label_set)}")
    print(f"\\ref targets in main paper: {len(refs)}")
    print(f"\\cite keys in main paper: {len(cites)}")
    print(f".bib entries: {len(bib_keys)}")

    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"

    if undef_refs:
        for r in undef_refs:
            msg = f"Undefined \\ref target: {r}"
            print(f"::warning::{msg}" if in_actions else f"WARN     {msg}")
    else:
        print("OK       All \\ref targets have matching \\label definitions")

    if undef_cites:
        for c in undef_cites:
            msg = f"Citation key not in .bib: {c}"
            print(f"::warning::{msg}" if in_actions else f"WARN     {msg}")
    else:
        print(f"OK       All \\cite keys found in {bib_file}")

    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a", encoding="utf-8") as fh:
            fh.write(f"undef_refs={len(undef_refs)}\n")
            fh.write(f"undef_cites={len(undef_cites)}\n")

    return 0  # warn-only; never fails the build


if __name__ == "__main__":
    sys.exit(main())
