#!/usr/bin/env python3
"""
Let gen_randomsplit() in scripts/generate_tables.py fall back to the exp2 per-domain runs
(comparison_results/feynman-tests/exp2/protocol_core_noiseless_*.json) when no
*random*80_20* / protocol_core_random_* files exist.

Dry run by default. --apply writes the change and keeps generate_tables.py.bak.
Each anchor must occur exactly once, otherwise nothing is changed.

  python patch_randomsplit_fallback.py scripts/generate_tables.py
  python patch_randomsplit_fallback.py scripts/generate_tables.py --apply
"""
import shutil
import sys

SIG_OLD = "def _load_split_equations(glob_pat: str, r2_field: str | None = None) -> tuple[list[tuple], Path | None]:"
SIG_NEW = ("def _load_split_equations(glob_pat: str, r2_field: str | None = None,\n"
           "                          subdirs: tuple | None = None) -> tuple[list[tuple], Path | None]:")

LOOP_OLD = ('        for subdir in ("comparison_results/feynman-tests/exp2_multi", '
            '"comparison_results/feynman-tests/exp2_pca_4060", "exp2_multi", ""):')
LOOP_NEW = ('        for subdir in (subdirs or ("comparison_results/feynman-tests/exp2_multi", '
            '"comparison_results/feynman-tests/exp2_pca_4060", "exp2_multi", "")):')

SKIP_OLD = '    if not rows:\n        skip_table("randomsplit.tex",'
SKIP_NEW = ('    if not rows:\n'
            '        # fallback: exp2 random-80/20 per-domain runs (one file per domain, same record shape).\n'
            '        # The result differs from the July 22/23 runs the paper cites if any method is non-deterministic.\n'
            '        rows, src = _load_split_equations("protocol_core_noiseless_2*.json",\n'
            '                                          subdirs=("comparison_results/feynman-tests/exp2",))\n'
            '    if not rows:\n        skip_table("randomsplit.tex",')


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path, apply = sys.argv[1], "--apply" in sys.argv
    text = open(path, encoding="utf-8").read()
    for label, old in (("signature", SIG_OLD), ("subdir loop", LOOP_OLD), ("skip call", SKIP_OLD)):
        n = text.count(old)
        if n != 1:
            sys.exit(f"STOP: anchor '{label}' found {n} times (expected 1). The file differs from what I was shown; "
                     "no changes made.")
    new = text.replace(SIG_OLD, SIG_NEW).replace(LOOP_OLD, LOOP_NEW).replace(SKIP_OLD, SKIP_NEW)
    print("3 anchors found, patch is applicable.")
    if not apply:
        print("dry run: re-run with --apply to write it.")
        return
    shutil.copy2(path, path + ".bak")
    open(path, "w", encoding="utf-8").write(new)
    print(f"patched {path} (backup {path}.bak)")
    print("next: regenerate tables, then run the checker as the paper_check CI job does, and read the randomsplit row.")
    print("      MATCH -> the Aug runs reproduce the paper table. DIFF/PARTIAL -> the paper's numbers came from the")
    print("      July runs that are not in the repo; note that DIFF fails the CI job unless you record an exception.")


if __name__ == "__main__":
    main()
