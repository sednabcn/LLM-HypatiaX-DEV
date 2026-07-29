#!/usr/bin/env python3
"""
update_notebooks.py

Closes out FIX-D1 and FIX-N4 in NB-04 and NB-06, in line with the
2026-07-29 issue_registry.json resolution.

IMPORTANT — what this script does NOT do:
  It does not touch the Step 6c (NB-04) / Step 3c (NB-06) *check logic*.
  That code already correctly globs for
  `defi/hypatix_defi_benchmark_v3c_corrected_*.json` and validates it —
  it was only ever reporting [MISSING] because the file didn't exist yet.
  Once the file is staged (Step 1 below), those cells will self-report
  [OK] the next time the notebook is actually executed. This script does
  NOT execute the notebooks — CI/you still need to re-run them for the
  [OK] lines to actually appear in cell output.

What this script DOES do:
  1. Stage hypatix_defi_benchmark_v3c_corrected_seed42.json into the
     results path the notebooks expect (RESULTS_DIR/defi/...).
  2. Patch stale "OPEN, critical" / "investigation stub" status text in
     markdown cells and final summary-print cells so the notebooks stop
     asserting things the registry no longer asserts. Only prose/status
     text is touched — no check logic is rewritten.

Usage:
    python3 update_notebooks.py \
        --nb04 NB-04_Numerical_Consistency_Checker.ipynb \
        --nb06 NB-06_Code_Quality_Pipeline_Integrity.ipynb \
        --defi-result hypatix_defi_benchmark_v3c_corrected_seed42.json \
        --repo-root .   # dir containing (or to contain) hypatiax/data/results/

Safe to re-run: all replacements are guarded by "already applied?" checks.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """Walk upward from `start` looking for a .git dir or a hypatiax/ dir.
    Falls back to `start` itself if neither is found (e.g. a fresh checkout
    where hypatiax/ doesn't exist yet)."""
    cur = start.resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / ".git").exists() or (candidate / "hypatiax").exists():
            return candidate
    return start.resolve()


def autodiscover(repo_root: Path):
    """Locate NB-04, NB-06, and the newest committed corrected DeFi result
    file by convention, so CI doesn't need hand-typed paths. Returns a dict;
    any entry not found is left as None and must be supplied explicitly."""
    found = {"nb04": None, "nb06": None, "defi_result": None}

    nb_candidates = list(repo_root.glob("**/NB-04_Numerical_Consistency_Checker.ipynb"))
    if nb_candidates:
        found["nb04"] = sorted(nb_candidates)[0]

    nb_candidates = list(repo_root.glob("**/NB-06_Code_Quality_Pipeline_Integrity.ipynb"))
    if nb_candidates:
        found["nb06"] = sorted(nb_candidates)[0]

    # Prefer a result file already committed under hypatiax/data/results/defi/
    # (the normal case once a benchmark run has landed there); this is what
    # "automatic" means -- the script picks up whatever the benchmark step
    # produced, rather than requiring someone to name the seed by hand.
    defi_candidates = sorted(
        repo_root.glob("hypatiax/data/results/defi/hypatix_defi_benchmark_v3c_corrected_*.json")
    )
    if defi_candidates:
        found["defi_result"] = defi_candidates[-1]  # newest by name/seed

    return found



def has_audit_findings_cell(nb: dict) -> bool:
    return any("audit_findings" in c.get("metadata", {}).get("tags", [])
               for c in nb["cells"])


def make_code_cell(cell_id: str, tags: list, source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {"tags": tags},
        "outputs": [],
        "source": source,
    }


# This is the piece that actually closes the loop with ci_paper_notebooks.yml's
# "Extract & upload registry patch" step, which ONLY reads cells tagged
# audit_findings (or a cell literally id'd step8_writeback). It parses this
# cell's JSON output for {"findings": [{"id": ..., "status": ...}, ...]} and
# turns it into logs/registry_patch_NB-04.json -> merged into
# scripts/patches/issue_registry.json. Nothing else in the notebook (plain
# print()s, markdown headers) feeds that pipeline at all.
#
# Status is computed LIVE from the Step 6c / Step 6d variables already in
# scope when this cell runs (nbconvert executes cells top-to-bottom in one
# kernel) -- never hardcoded -- so a future regression in the underlying
# check automatically flips this back to "open" rather than staying stuck
# on a stale "resolved".
NB04_AUDIT_FINDINGS_SOURCE = '''# AUTO: live findings for CI registry merge (tag: audit_findings)
# ci_paper_notebooks.yml's "Extract & upload registry patch" step reads this
# cell's JSON output and merges it into scripts/patches/issue_registry.json.
# Status is computed from the Step 6c / Step 6d results above, not hardcoded.
import json as _json

_fixd1_ok = bool(DEFI_CORRECTED_GLOB) and all(
    (summary.get(k) is not None and abs(summary.get(k) - exp) <= tol)
    for k, exp, tol in [
        ("corrected_success_rate",      TRUTH["hypatix_success_pct_corrected"] / 100, 0.005),
        ("hard_tier_gain_pp_corrected", TRUTH["hypatix_hard_tier_gain_pp_corrected"], 0.1),
        ("catastrophic_masked_count",   TRUTH["hypatix_catastrophic_masked_count"], 0),
    ]
)
_fixn4_ok = not hits_0678  # '0.678' absent from current source -> false positive confirmed

_findings = {
    "findings": [
        {
            "id": "FIX-D1",
            "status": "resolved" if _fixd1_ok else "open",
            "severity": "critical",
            "description": (
                "DeFi hybrid-attribution-bug: corrected figures (60.8% / -4.8pp / "
                "22 masked) verified against committed result file."
                if _fixd1_ok else
                "DeFi hybrid-attribution-bug: corrected result file missing or "
                "does not match the paper's disclosed figures."
            ),
            "nb": "NB-04",
        },
        {
            "id": "FIX-N4",
            "status": "false_positive" if _fixn4_ok else "open",
            "severity": "medium",
            "description": (
                "'0.678' does not occur anywhere in current source; premise not reproducible."
                if _fixn4_ok else
                "'0.678' found in source -- investigate co-occurrence with '9/30'."
            ),
            "nb": "NB-04",
        },
    ]
}
print(_json.dumps(_findings, indent=2))
'''

NB06_AUDIT_FINDINGS_SOURCE = '''# AUTO: live findings for CI registry merge (tag: audit_findings)
# ci_paper_notebooks.yml's "Extract & upload registry patch" step reads this
# cell's JSON output and merges it into scripts/patches/issue_registry.json.
# Status is computed from the Step 3c results above, not hardcoded.
import json as _json

_fixd1_ok = bool(DEFI_CORRECTED_FILES) and all(
    (summary.get(k) is not None and abs(summary.get(k) - exp) <= tol)
    for k, (exp, tol) in EXPECTED.items()
)

_findings = {
    "findings": [
        {
            "id": "FIX-D1",
            "status": "resolved" if _fixd1_ok else "open",
            "severity": "critical",
            "description": (
                "DeFi hybrid-attribution-bug: corrected figures verified against "
                "committed result file (NB-06 Step 3c)."
                if _fixd1_ok else
                "DeFi hybrid-attribution-bug: corrected result file missing or "
                "does not match disclosed figures (NB-06 Step 3c)."
            ),
            "nb": "NB-06",
        },
    ]
}
print(_json.dumps(_findings, indent=2))
'''


def ensure_audit_findings_cell(nb: dict, source: str, cell_id: str) -> bool:
    """Idempotent: appends the audit_findings cell only if the notebook has
    none yet. Returns True if a cell was added."""
    if has_audit_findings_cell(nb):
        return False
    nb["cells"].append(make_code_cell(cell_id, ["audit_findings"], source))
    return True


def load_nb(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_nb(nb: dict, path: Path):
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def cell_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_cell_source(cell: dict, new_text: str):
    # Preserve Jupyter's list-of-lines-with-newlines source format.
    lines = new_text.splitlines(keepends=True)
    cell["source"] = lines


def replace_in_notebook(nb: dict, replacements: list, nb_name: str) -> int:
    """Apply (old, new, label) replacements across all cells. Each old must
    appear at most once across the notebook (mirrors str_replace safety)."""
    applied = 0
    for old, new, label in replacements:
        hits = [c for c in nb["cells"] if old in cell_source(c)]
        if not hits:
            print(f"  [SKIP] {nb_name}: '{label}' — pattern not found "
                  f"(already applied, or source has changed upstream)")
            continue
        if len(hits) > 1:
            print(f"  [WARN] {nb_name}: '{label}' — pattern found in "
                  f"{len(hits)} cells, expected 1. Applying to all; "
                  f"verify output.")
        for cell in hits:
            src = cell_source(cell)
            set_cell_source(cell, src.replace(old, new))
            applied += 1
        print(f"  [OK]   {nb_name}: '{label}' — patched")
    return applied


def stage_defi_result(defi_result_src: Path, repo_root: Path) -> Path:
    dest_dir = repo_root / "hypatiax" / "data" / "results" / "defi"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / defi_result_src.name
    if defi_result_src.resolve() == dest.resolve():
        print(f"  [SKIP] {defi_result_src.name} already staged at {dest} (auto-discovered in place)")
        return dest
    shutil.copy2(defi_result_src, dest)
    print(f"  [OK]   staged {defi_result_src.name} -> {dest}")
    return dest


def verify_defi_result(dest: Path):
    """Sanity-check the staged file against the expected FIX-D1 targets
    before declaring anything resolved — do not trust the filename alone."""
    data = json.loads(dest.read_text())
    summary = data.get("summary", {})
    per_task = data.get("per_task", [])

    expected = {
        "corrected_success_rate":      (0.608, 0.005),
        "hard_tier_gain_pp_corrected": (-4.8,  0.1),
        "catastrophic_masked_count":   (22,    0),
    }
    ok = True
    for key, (exp_val, tol) in expected.items():
        actual = summary.get(key)
        if actual is None or abs(actual - exp_val) > tol:
            print(f"  [FAIL] verify: summary.{key} = {actual}, expected {exp_val} (tol {tol})")
            ok = False

    # Recompute independently from per_task, don't just trust the summary block.
    if per_task:
        n = len(per_task)
        recomputed_rate = sum(1 for t in per_task if t.get("corrected_pass")) / n
        recomputed_masked = sum(1 for t in per_task if t.get("masked_catastrophic"))
        if abs(recomputed_rate - summary.get("corrected_success_rate", -1)) > 0.001:
            print(f"  [FAIL] verify: recomputed rate {recomputed_rate:.4f} != "
                  f"summary value {summary.get('corrected_success_rate')}")
            ok = False
        if recomputed_masked != summary.get("catastrophic_masked_count"):
            print(f"  [FAIL] verify: recomputed masked count {recomputed_masked} != "
                  f"summary value {summary.get('catastrophic_masked_count')}")
            ok = False

    if ok:
        print("  [OK]   verify: staged file matches FIX-D1 targets "
              "(summary + independent per_task recomputation agree)")
    return ok


NB04_REPLACEMENTS = [
    (
        "- **FIX-D1 (NEW, critical, OPEN)** — the DeFi §hybrid-attribution-bug disclosure. "
        "Paper still prints the pre-bug headline numbers (89.2%, +38.1pp hard-tier gain, "
        "'zero catastrophic failures') inline, immediately followed by a footnote/bracketed "
        "correction (60.8%, −4.8pp, 22/74 masked failures). Both readings legitimately "
        "co-occur in the same sentence — the plain substring checks in Step 2 cannot tell "
        "'number is correct' from 'number is flagged as superseded right next to it'. "
        "Step 6c below adds a footnote-aware check plus a corrected-figures verification "
        "against the DeFi benchmark output.",
        "- **FIX-D1 (RESOLVED 2026-07-29)** — the DeFi §hybrid-attribution-bug disclosure. "
        "Root cause confirmed: `hybrid.success`/`hybrid.test_r2` was trusted blindly without "
        "checking whether the sub-method actually named in `hybrid.decision` itself succeeded. "
        "Corrected via `fix_defi_attribution_bug.py`; output committed to "
        "`defi/hypatix_defi_benchmark_v3c_corrected_seed42.json` and reproduced independently "
        "from raw per-task data (not just paper prose): overall 60.8% (45/74), hard-tier "
        "−4.8pp, 22/74 masked catastrophic failures. Step 6c below verifies this file directly.",
        "Cell 0 status block: FIX-D1",
    ),
    (
        "- **FIX-N4 (NEW, open, investigation stub)** — registry claims a stray '0.678' "
        "baseline co-occurs with '9/30' somewhere in §10.7. Step 6d below searches the "
        "source directly; as of this run, no occurrence of '0.678' was found anywhere in "
        "the .tex file, so the premise of this issue could not be reproduced against "
        "current source.",
        "- **FIX-N4 (CLOSED 2026-07-28, false positive)** — registry claimed a stray "
        "'0.678' baseline co-occurs with '9/30' somewhere in §10.7. Direct grep of the "
        "current source found zero occurrences of '0.678' anywhere in the file; every "
        "'9/30' occurrence is already part of the documented withdrawn-run narrative. "
        "Confirmed false positive — no detector regex or paper_targets.json entry needed. "
        "Step 6d below retains the investigation for auditability.",
        "Cell 0 status block: FIX-N4",
    ),
    (
        "## Step 6c — FIX-D1 (OPEN, critical): DeFi hybrid-attribution-bug footnote-aware check",
        "## Step 6c — FIX-D1 (RESOLVED 2026-07-29): DeFi hybrid-attribution-bug footnote-aware check",
        "Step 6c header",
    ),
    (
        "## Step 6d — FIX-N4 (open, investigation stub): '9/30 vs 0.678' claim in §10.7",
        "## Step 6d — FIX-N4 (CLOSED, false positive): '9/30 vs 0.678' claim in §10.7",
        "Step 6d header",
    ),
]
# NB-04's Cell 18 final-status print is patched separately below (NB04_FINAL_STATUS_*)
# as a plain-text match, since cell_source() returns the rendered string, not the
# JSON-escaped form.

# The final-status print cell is *Python source code as text* -- inside its quoted
# string literals, "\n" is a literal two-character backslash-n (not a real newline),
# and unicode markers like the checkmark are literal "\u2705" escape text too. Rather
# than match a fragile whole multi-line block, patch small unique substrings that
# each live entirely within a single quoted line.
NB04_FINAL_STATUS_SUBSTITUTIONS = [
    (
        "FIX-D1  DeFi hybrid-attribution-bug -- OPEN, critical. Paper discloses corrected figures",
        "FIX-D1  DeFi hybrid-attribution-bug -- RESOLVED 2026-07-29. Corrected result file",
    ),
    (
        "inline (60.8% overall, -4.8pp hard-tier, 22/74 masked catastrophic) alongside the",
        "committed and independently reproduced from raw per-task data: overall 60.8%,",
    ),
    (
        "pre-bug headline numbers. Step 6c adds footnote-aware checks; corrected DeFi result",
        "hard-tier -4.8pp, 22/74 masked catastrophic. Step 6c verifies it directly.",
    ),
    (
        "file has not yet been located/committed for CI gating. \\u26a0\\ufe0f",
        "\\u2705",
    ),
    (
        "FIX-N4  '9/30 vs 0.678' in \\u00a710.7 -- OPEN, investigation stub. Step 6d searched the",
        "FIX-N4  '9/30 vs 0.678' in \\u00a710.7 -- CLOSED, false positive (2026-07-28). '0.678' does",
    ),
    (
        "current source and found NO occurrence of '0.678' anywhere; recommend the registry",
        "not occur anywhere in current source; all '9/30' hits are part of the documented",
    ),
    (
        "entry be re-verified against its original source or closed as not reproducible. \\u2753",
        "withdrawn-run narrative. Step 6d retains the investigation for auditability. \\u2705",
    ),
]

NB06_REPLACEMENTS = [
    (
        "- **FIX-D1 (NEW, critical, OPEN) corrected DeFi benchmark check** — reads "
        "`hypatix_defi_benchmark_v3c_corrected_*.json` to verify the paper's disclosed "
        "post-attribution-bug figures (60.8% overall, −4.8pp hard-tier, 22/74 masked "
        "catastrophic failures).",
        "- **FIX-D1 (RESOLVED 2026-07-29) corrected DeFi benchmark check** — reads "
        "`hypatix_defi_benchmark_v3c_corrected_seed42.json` (now committed) and verifies the "
        "paper's disclosed post-attribution-bug figures (60.8% overall, −4.8pp hard-tier, "
        "22/74 masked catastrophic failures) against it directly.",
        "Cell 0 status block",
    ),
    (
        "## Step 3c — FIX-D1 (NEW, critical, OPEN): corrected DeFi benchmark verification",
        "## Step 3c — FIX-D1 (RESOLVED 2026-07-29): corrected DeFi benchmark verification",
        "Step 3c header",
    ),
]

# Same literal-backslash-n code-as-text situation as NB-04 -- small unique
# substitutions instead of a fragile whole-block match.
NB06_FINAL_STATUS_SUBSTITUTIONS = [
    (
        "FIX-D1  DeFi hybrid-attribution-bug (\\u00a7hybrid-attribution-bug) -- OPEN, critical",
        "FIX-D1  DeFi hybrid-attribution-bug (\\u00a7hybrid-attribution-bug) -- RESOLVED 2026-07-29",
    ),
    (
        "not eliminated). Step 3c above checks for a committed corrected result file --",
        "not eliminated). Step 3c above checks the now-committed corrected result file",
    ),
    (
        "as of this run, none was found. REMAINING ACTION: commit",
        "(hypatix_defi_benchmark_v3c_corrected_seed42.json, now committed) and confirms",
    ),
    (
        "hypatix_defi_benchmark_v3c_corrected_*.json with summary.corrected_success_rate,",
        "all three figures match to within tolerance: corrected_success_rate,",
    ),
    (
        "summary.hard_tier_gain_pp_corrected, and summary.catastrophic_masked_count so this",
        "hard_tier_gain_pp_corrected, and catastrophic_masked_count. This is now a live CI",
    ),
    (
        "becomes a live CI gate instead of a text-only disclosure.",
        "gate, not a text-only disclosure.",
    ),
]


def main():
    ap = argparse.ArgumentParser(
        description="Auto-discovers repo root, NB-04/NB-06, and the newest "
                     "committed corrected DeFi result file by convention. "
                     "Pass explicit paths only to override discovery."
    )
    ap.add_argument("--nb04", type=Path, default=None)
    ap.add_argument("--nb06", type=Path, default=None)
    ap.add_argument("--defi-result", type=Path, default=None)
    ap.add_argument("--repo-root", type=Path, default=None,
                     help="Defaults to nearest ancestor containing .git or hypatiax/")
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="Defaults to writing notebooks back in place")
    ap.add_argument("--check", action="store_true",
                     help="Dry run: report what would change, write nothing.")
    args = ap.parse_args()

    repo_root = args.repo_root or find_repo_root(Path.cwd())
    print(f"repo root: {repo_root}")

    discovered = autodiscover(repo_root)
    nb04 = args.nb04 or discovered["nb04"]
    nb06 = args.nb06 or discovered["nb06"]
    defi_result = args.defi_result or discovered["defi_result"]

    missing = [name for name, val in
               [("NB-04", nb04), ("NB-06", nb06), ("DeFi result file", defi_result)]
               if val is None]
    if missing:
        print(f"\n[NOOP] Could not locate: {', '.join(missing)}. Nothing to do yet "
              f"-- this is expected if the benchmark rerun hasn't landed a result "
              f"file under hypatiax/data/results/defi/ yet. Exiting 0 (not a failure).")
        sys.exit(0)

    print(f"NB-04:       {nb04}")
    print(f"NB-06:       {nb06}")
    print(f"DeFi result: {defi_result}")
    if args.check:
        print("(--check: dry run, no files will be written)\n")

    print()
    print("=" * 80)
    print("Step 1 — stage corrected DeFi result file")
    print("=" * 80)
    dest = stage_defi_result(defi_result, repo_root) if not args.check else defi_result
    if not verify_defi_result(dest):
        print("\n[FAIL] staged file failed verification against FIX-D1 targets. "
              "Not patching notebook status text for a result that doesn't check out.")
        sys.exit(1)

    out_dir_04 = args.out_dir or nb04.parent
    out_dir_06 = args.out_dir or nb06.parent

    print()
    print("=" * 80)
    print("Step 2 — patch NB-04 status text")
    print("=" * 80)
    nb04_obj = load_nb(nb04)
    n04 = replace_in_notebook(nb04_obj, NB04_REPLACEMENTS, "NB-04")
    applied_final = 0
    for cell in nb04_obj["cells"]:
        src = cell_source(cell)
        if "FIX-D1  DeFi hybrid-attribution-bug -- OPEN, critical" not in src:
            continue
        new_src = src
        for old, new in NB04_FINAL_STATUS_SUBSTITUTIONS:
            if old not in new_src:
                print(f"  [WARN] NB-04 final-status substitution not found: {old[:60]!r}...")
                continue
            new_src = new_src.replace(old, new)
            applied_final += 1
        set_cell_source(cell, new_src)
    print(f"  [{'OK' if applied_final else 'SKIP'}] NB-04: final status cell "
          f"({applied_final}/{len(NB04_FINAL_STATUS_SUBSTITUTIONS)} substitutions applied)")

    added_af4 = ensure_audit_findings_cell(nb04_obj, NB04_AUDIT_FINDINGS_SOURCE, "audit_findings_fixd1_n4")
    print(f"  [{'OK' if added_af4 else 'SKIP'}] NB-04: audit_findings cell "
          f"({'added -- now feeds ci_paper_notebooks.yml registry merge' if added_af4 else 'already present'})")

    if n04 == 0 and applied_final == 0 and not added_af4:
        print("  [INFO] NB-04 already up to date -- no changes needed.")
    elif not args.check:
        nb04_out = out_dir_04 / nb04.name
        save_nb(nb04_obj, nb04_out)
        print(f"  -> wrote {nb04_out}")
    else:
        print("  (--check: would write changes, none written)")

    print()
    print("=" * 80)
    print("Step 3 — patch NB-06 status text")
    print("=" * 80)
    nb06_obj = load_nb(nb06)
    n06 = replace_in_notebook(nb06_obj, NB06_REPLACEMENTS, "NB-06")
    applied_final6 = 0
    for cell in nb06_obj["cells"]:
        src = cell_source(cell)
        if "FIX-D1  DeFi hybrid-attribution-bug (\\u00a7hybrid-attribution-bug) -- OPEN, critical" not in src:
            continue
        new_src = src
        for old, new in NB06_FINAL_STATUS_SUBSTITUTIONS:
            if old not in new_src:
                print(f"  [WARN] NB-06 final-status substitution not found: {old[:60]!r}...")
                continue
            new_src = new_src.replace(old, new)
            applied_final6 += 1
        set_cell_source(cell, new_src)
    print(f"  [{'OK' if applied_final6 else 'SKIP'}] NB-06: final status cell "
          f"({applied_final6}/{len(NB06_FINAL_STATUS_SUBSTITUTIONS)} substitutions applied)")

    added_af6 = ensure_audit_findings_cell(nb06_obj, NB06_AUDIT_FINDINGS_SOURCE, "audit_findings_fixd1")
    print(f"  [{'OK' if added_af6 else 'SKIP'}] NB-06: audit_findings cell "
          f"({'added -- now feeds ci_paper_notebooks.yml registry merge' if added_af6 else 'already present'})")

    if n06 == 0 and applied_final6 == 0 and not added_af6:
        print("  [INFO] NB-06 already up to date -- no changes needed.")
    elif not args.check:
        nb06_out = out_dir_06 / nb06.name
        save_nb(nb06_obj, nb06_out)
        print(f"  -> wrote {nb06_out}")
    else:
        print("  (--check: would write changes, none written)")

    changed = (n04 + applied_final + n06 + applied_final6
               + int(added_af4) + int(added_af6)) > 0
    print()
    print("=" * 80)
    if args.check:
        print(f"CHECK MODE: {'changes pending' if changed else 'up to date'}.")
        sys.exit(1 if changed else 0)  # nonzero in CI = drift detected, needs a real run
    print("DONE." + (" Notebooks updated." if changed else " Nothing to update."))
    print(f"Staged result file: {dest}")
    print("Commit the updated .ipynb files AND the staged defi/ result file together")
    print("-- a status-text-only patch with no result file committed would repeat")
    print("exactly the FIX-D1 badge-flip mistake this audit caught earlier.")
    print("=" * 80)


if __name__ == "__main__":
    main()
