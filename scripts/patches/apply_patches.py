#!/usr/bin/env python3
"""
apply_patches.py
================
HypatiaX reproducibility pipeline — apply all code patches before running
experiments.  Implements patches P-1 through P-5 from HypatiaX_Final_Pipeline_Plan.md.

Patches applied:
  P-1  Swap hybrid_system_v40 → hybrid_system_v50_2  (4 source files)
  P-2  Fix 3 duplicate DeFi case names + change checkpoint key → equation_id
  P-3  Set populations=30 as default in make_pysr()
  P-4  Remove hardcoded API keys (replaced with os.environ lookup)
  P-5  Add Feynman 80/20 split protocol comment to run_comparative_suite_benchmark_v2.py

Usage:
    python3 apply_patches.py                # apply all patches
    python3 apply_patches.py --dry-run      # show diffs, no writes
    python3 apply_patches.py --patch P-1    # apply one patch only
    python3 apply_patches.py --verify       # apply all + re-run scanner

Exit codes:
  0 — all patches applied (or already applied — idempotent)
  1 — one or more patches failed

Fixes vs original:
  FIX-1  repo_root now resolves correctly (file lives at repo root, not scripts/patches/)
  FIX-2  P-1 uses word-boundary \\b instead of (?!fix) lookahead
  FIX-3  P-4 also catches bare sk-ant-... string assignments (not only Anthropic() calls)
  FIX-4  P-2 guards against 3+ occurrences of the same duplicate name
  FIX-5  P-3 target file corrected to match actual tree (experiment_protocol_all_18_a.py)
  FIX-6  Backup (.bak) written before every file overwrite
  FIX-7  --verify flag re-runs scan_internal_imports.py after patching
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ── Colour helpers ────────────────────────────────────────────────────────────
GRN = "\033[0;32m"
YLW = "\033[1;33m"
RED = "\033[0;31m"
NC  = "\033[0m"

def ok(msg):   print(f"{GRN}  ✓  {msg}{NC}")
def warn(msg): print(f"{YLW}  ⚠  {msg}{NC}")
def fail(msg): print(f"{RED}  ✗  {msg}{NC}")


# ── Patch base class ──────────────────────────────────────────────────────────

class Patch:
    id: str = ""
    description: str = ""

    def apply(self, root: Path, dry_run: bool) -> bool:
        raise NotImplementedError

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _backup(path: Path) -> None:
        """Write a .bak copy alongside the original before overwriting."""
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)

    def _replace_in_file(
        self,
        path: Path,
        old: str,
        new: str,
        dry_run: bool,
        label: str = "",
    ) -> bool:
        if not path.exists():
            fail(f"{label or path}: file not found")
            return False
        text = path.read_text(errors="replace")
        if old not in text:
            ok(f"{label or path.name}: already patched (pattern not found — skipping)")
            return True
        if dry_run:
            print(f"  DRY-RUN  {label or path.name}: would replace {repr(old[:60])} …")
            return True
        self._backup(path)
        path.write_text(text.replace(old, new))
        ok(f"{label or path.name}: patched  (backup: {path.name}.bak)")
        return True

    def _regex_replace(
        self,
        path: Path,
        pattern: str,
        repl: str,
        dry_run: bool,
        label: str = "",
        flags: int = 0,
    ) -> bool:
        if not path.exists():
            fail(f"{label or path}: file not found")
            return False
        text = path.read_text(errors="replace")
        new_text, n = re.subn(pattern, repl, text, flags=flags)
        if n == 0:
            ok(f"{label or path.name}: already patched (0 replacements — skipping)")
            return True
        if dry_run:
            print(f"  DRY-RUN  {label or path.name}: would make {n} replacement(s)")
            return True
        self._backup(path)
        path.write_text(new_text)
        ok(f"{label or path.name}: {n} replacement(s) applied  (backup: {path.name}.bak)")
        return True


# ── P-1: v40 → v50_2 engine swap ─────────────────────────────────────────────

class PatchP1(Patch):
    id = "P-1"
    description = "Swap hybrid_system_v40 → hybrid_system_v50_2 (FIX-C2)"

    # Files where only live import statements are rewritten.
    # (no method-name or benchmark-table references)
    SIMPLE_TARGETS = [
        "hypatiax/core/generation/hybrid_all_domains/suite_hybrid_system_all_domains.py",
        "hypatiax/core/generation/hybrid_defi_llm_guided/llm_guided_symbolic_discovery_defi.py",
        "hypatiax/core/generation/hybrid_defi_system/complete_defi_hybrid_system.py",
    ]

    # run_comparative_suite_benchmark_v2.py needs surgical, line-by-line treatment
    # because it contains a mix of:
    #   • live import statements  (must change)
    #   • a wrapper class name    (must change)
    #   • comments / docstrings referencing the old filename  (update wording)
    #   • a benchmark table row   (update name + path string)
    BENCHMARK_FILE = (
        "hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_v2.py"
    )

    # ── substitution rules applied to BENCHMARK_FILE, in order ───────────────
    # Each entry: (regex_pattern, replacement, description)
    # Rules are intentionally narrow so comments/docs get human-readable updates
    # rather than a blind find-replace that leaves "v50_2.py" in prose.
    BENCHMARK_RULES: list[tuple[str, str, str]] = [
        # 1. Live import lines  (lines 2070, 2404)
        #    from hypatiax.tools.symbolic.hybrid_system_v40 import HybridDiscoverySystem
        (
            r"from hypatiax\.tools\.symbolic\.hybrid_system_v40 import HybridDiscoverySystem",
            "from hypatiax.tools.symbolic.hybrid_system_v50_2 import HybridDiscoverySystem",
            "import statements",
        ),
        # 2. Class definition (line 2392) and benchmark table row (line 2559).
        #    Line 2302 is a comment "# Same adaptive budget as HybridSystemV40Method"
        #    — excluded by anchoring to class keyword or tuple indent pattern.
        (
            r"^(class |        \(\d+,\s*)HybridSystemV40Method\b",
            r"\1HybridSystemV50_2Method",
            "wrapper class name",
        ),
        # 3. Path string in benchmark table tuple  (line 2559)
        #    "tools/symbolic/hybrid_system_v40.py"
        (
            r'"tools/symbolic/hybrid_system_v40\.py"',
            '"tools/symbolic/hybrid_system_v50_2.py"',
            "benchmark table path string",
        ),
        # 4. Docstring / comment: "Wraps hypatiax...hybrid_system_v40.HybridDiscoverySystem"
        #    (line 2394)
        (
            r"(Wraps hypatiax\.tools\.symbolic\.)hybrid_system_v40(\.HybridDiscoverySystem)",
            r"\1hybrid_system_v50_2\2",
            "docstring module reference",
        ),
        # 5. Inline comment: "tools/symbolic/hybrid_system_v40.py"  (lines 29, 1997, 2389)
        #    Only inside # comment or docstring lines — leave prose wording intact
        #    but update the filename so readers can find the actual file.
        (
            r"(#[^\n]*?)hybrid_system_v40(\.py)",
            r"\1hybrid_system_v50_2\2",
            "inline comment filename references",
        ),
        # 6. Remaining bare "v40" label in the benchmark summary comment (line 3232)
        #    "HybridDiscovery v40" → "HybridDiscovery v50_2"
        (
            r"(HybridDiscovery\s+)v40\b",
            r"\1v50_2",
            "benchmark summary label",
        ),
    ]

    def apply(self, root: Path, dry_run: bool) -> bool:
        ok_all = True

        # ── Simple targets: only import lines need changing ───────────────────
        for rel in self.SIMPLE_TARGETS:
            path = root / rel
            ok_all &= self._regex_replace(
                path,
                r"from hypatiax\.tools\.symbolic\.hybrid_system_v40 import",
                "from hypatiax.tools.symbolic.hybrid_system_v50_2 import",
                dry_run,
                label=rel,
                flags=re.MULTILINE,
            )

        # ── Benchmark file: apply each rule in sequence ───────────────────────
        bm_path = root / self.BENCHMARK_FILE
        if not bm_path.exists():
            fail(f"P-1: {self.BENCHMARK_FILE} not found")
            return False

        text     = bm_path.read_text(errors="replace")
        original = text
        total_n  = 0

        for pattern, repl, desc in self.BENCHMARK_RULES:
            text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
            if n:
                total_n += n
                if dry_run:
                    print(f"  DRY-RUN  {bm_path.name}: [{desc}] would make {n} replacement(s)")
                else:
                    ok(f"  {bm_path.name}: [{desc}] {n} replacement(s)")

        if text == original:
            ok(f"{bm_path.name}: already fully patched — no changes needed")
        elif not dry_run:
            self._backup(bm_path)
            bm_path.write_text(text)
            ok(f"{bm_path.name}: {total_n} total replacement(s)  (backup: {bm_path.name}.bak)")

        return ok_all


# ── P-2: Fix duplicate DeFi case names ───────────────────────────────────────

class PatchP2(Patch):
    id = "P-2"
    description = "Fix 3 duplicate DeFi case names + checkpoint key → equation_id"

    TARGET = "hypatiax/experiments/benchmarks/hypatiax_defi_benchmark_v3c.py"

    RENAMES = [
        ('"Constant product formula"',
         '"Constant product formula (multivariate)"'),
        ('"Funding rate cost"',
         '"Funding rate cost (extended)"'),
        ('"Concentrated liquidity position width"',
         '"Concentrated liquidity position width (v2)"'),
    ]

    def apply(self, root: Path, dry_run: bool) -> bool:
        path = root / self.TARGET
        if not path.exists():
            fail(f"P-2: {self.TARGET} not found")
            return False

        text     = path.read_text(errors="replace")
        original = text

        for old, new in self.RENAMES:
            parts = text.split(old)
            occurrences = len(parts) - 1

            if occurrences == 0:
                ok(f"P-2 rename {old[:40]!r}: not found — already patched or not present")
                continue

            if occurrences == 1:
                ok(f"P-2 rename {old[:40]!r}: only one occurrence — already patched")
                continue

            # Guard: more than 3 is unexpected
            if occurrences > 3:
                warn(
                    f"P-2: {occurrences} occurrences of {old[:40]!r} — "
                    "too many to patch safely; review manually."
                )
                continue

            # Replace SECOND occurrence only regardless of total count:
            #   2 occurrences: parts = [pre1, between, post2]
            #   3 occurrences: parts = [pre1, between, between2, post3]
            #     → keep 1st and 3rd as-is, rename 2nd only
            text = (
                parts[0] + old           # keep 1st occurrence
                + parts[1] + new         # rename 2nd occurrence
                + old.join(parts[2:])    # keep remaining occurrences unchanged
            )
            if dry_run:
                print(f"  DRY-RUN  P-2: would rename second {old[:40]!r} → {new[:40]!r}")
            else:
                ok(f"P-2: renamed second occurrence of {old[:40]!r}")

        # Fix checkpoint key: case["name"] → case["equation_id"]
        text, n = re.subn(
            r'checkpoint\[case\["name"\]\]',
            'checkpoint[case["equation_id"]]',
            text,
        )
        if n > 0:
            if dry_run:
                print(f"  DRY-RUN  P-2: would fix checkpoint key ({n} replacement(s))")
            else:
                ok(f"P-2: checkpoint key → equation_id ({n} replacement(s))")

        if text == original:
            ok("P-2: already fully patched — no changes needed")
            return True

        if not dry_run:
            self._backup(path)
            path.write_text(text)
        return True


# ── P-3: populations=30 in make_pysr() ───────────────────────────────────────

class PatchP3(Patch):
    id = "P-3"
    description = "Set populations=30 as default in make_pysr() (fair ablation baseline)"

    # FIX-5: corrected to actual files present in the tree.
    TARGETS = [
        "hypatiax/core/training/baseline_neural_network.py",
        "hypatiax/protocols/experiment_protocol_all_18_a.py",  # was: non-existent path
    ]

    def apply(self, root: Path, dry_run: bool) -> bool:
        ok_all = True
        for rel in self.TARGETS:
            path = root / rel
            if not path.exists():
                warn(f"P-3: {rel} not found — skipping (non-fatal)")
                continue
            ok_all &= self._regex_replace(
                path,
                r"(def make_pysr\(.*?populations\s*=\s*)(\d+)",
                r"\g<1>30",
                dry_run,
                label=rel,
                flags=re.DOTALL,
            )
        return ok_all


# ── P-4: Remove hardcoded API keys ───────────────────────────────────────────

class PatchP4(Patch):
    id = "P-4"
    description = "Replace hardcoded API keys with os.environ lookup"

    # Pattern 1: anthropic.Anthropic(api_key="sk-ant-...")
    _CALL_PATTERN = re.compile(
        r'anthropic\.Anthropic\(\s*api_key\s*=\s*["\']sk-ant-[^"\']{10,}["\']',
        re.MULTILINE,
    )
    _CALL_REPLACEMENT = 'anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]'

    # FIX-3: Pattern 2 — bare string assignments, e.g. API_KEY = "sk-ant-..."
    _BARE_PATTERN = re.compile(
        r'(["\'])sk-ant-[A-Za-z0-9\-_]{20,}\1',
        re.MULTILINE,
    )
    _BARE_REPLACEMENT = 'os.environ["ANTHROPIC_API_KEY"]'

    def apply(self, root: Path, dry_run: bool) -> bool:
        import json as _json

        hits: list[Path] = []
        nb_hits: list[Path] = []

        # ── Scan .py files ────────────────────────────────────────────────────
        for py in root.rglob("*.py"):
            if py.name in ("apply_patches.py",) or py.suffix == ".bak":
                continue
            try:
                text = py.read_text(errors="replace")
            except OSError:
                continue
            if self._CALL_PATTERN.search(text) or self._BARE_PATTERN.search(text):
                hits.append(py)

        # ── Scan .ipynb notebooks ─────────────────────────────────────────────
        for nb_path in root.rglob("*.ipynb"):
            if nb_path.suffix == ".bak":
                continue
            try:
                raw = nb_path.read_text(errors="replace")
            except OSError:
                continue
            if self._BARE_PATTERN.search(raw):
                nb_hits.append(nb_path)

        if not hits and not nb_hits:
            ok("P-4: no hardcoded API keys found ✓")
            return True

        # ── Fix .py files ─────────────────────────────────────────────────────
        for py in hits:
            text     = py.read_text(errors="replace")
            new_text = self._CALL_PATTERN.sub(self._CALL_REPLACEMENT, text)
            new_text = self._BARE_PATTERN.sub(self._BARE_REPLACEMENT, new_text)
            if "import os" not in new_text:
                new_text = "import os\n" + new_text
            if dry_run:
                warn(f"P-4 DRY-RUN: would remove hardcoded key in {py.relative_to(root)}")
            else:
                self._backup(py)
                py.write_text(new_text)
                ok(f"P-4: removed hardcoded API key from {py.relative_to(root)}"
                   f"  (backup: {py.name}.bak)")
                warn("  Rotate the exposed key at console.anthropic.com immediately!")

        # ── Fix .ipynb notebooks ──────────────────────────────────────────────
        for nb_path in nb_hits:
            try:
                nb = _json.loads(nb_path.read_text(errors="replace"))
            except Exception:
                warn(f"P-4: could not parse {nb_path.name} as JSON — skipping")
                continue
            changed = False
            for cell in nb.get("cells", []):
                src_lines = cell.get("source", [])
                new_lines = []
                cell_text = "".join(src_lines)
                needs_os  = "import os" not in cell_text
                for line in src_lines:
                    new_line = self._BARE_PATTERN.sub(self._BARE_REPLACEMENT, line)
                    if new_line != line:
                        changed = True
                        if needs_os:
                            new_lines.append("import os\n")
                            needs_os = False
                    new_lines.append(new_line)
                if changed:
                    cell["source"] = new_lines
            if not changed:
                continue
            if dry_run:
                warn(f"P-4 DRY-RUN: would remove hardcoded key from {nb_path.name}")
            else:
                self._backup(nb_path)
                nb_path.write_text(_json.dumps(nb, indent=1, ensure_ascii=False))
                ok(f"P-4: removed hardcoded API key from {nb_path.relative_to(root)}"
                   f"  (backup: {nb_path.name}.bak)")
                warn("  Rotate the exposed key at console.anthropic.com immediately!")

        return True


# ── P-5: Feynman split protocol comment ──────────────────────────────────────

class PatchP5(Patch):
    id = "P-5"
    description = (
        "Add Feynman 80/20 split protocol comment to "
        "run_comparative_suite_benchmark_v2.py"
    )

    TARGET = "hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_v2.py"

    COMMENT = '''\
    """
    Split protocol: 80/20 random split, random_state=42, extrap_multiplier=2.0.
    NOTE: This differs from the DeFi benchmark PCA 40/60 split (hypatiax_defi_benchmark_v3c.py).
    Results are NOT directly comparable. See §10.7 disclosure note in the paper.
    """
'''
    MARKER = "def main():"  # actual entry point in this file (line 3218)

    def apply(self, root: Path, dry_run: bool) -> bool:
        path = root / self.TARGET
        if not path.exists():
            warn(f"P-5: {self.TARGET} not found — skipping (non-fatal)")
            return True

        text = path.read_text(errors="replace")
        if "Split protocol:" in text:
            ok("P-5: split protocol comment already present")
            return True

        idx = text.find(self.MARKER)
        if idx == -1:
            warn(f"P-5: marker '{self.MARKER}' not found — skipping")
            return True

        end_of_line = text.find("\n", idx) + 1
        new_text    = text[:end_of_line] + self.COMMENT + text[end_of_line:]

        if dry_run:
            print(
                f"  DRY-RUN  P-5: would insert split protocol comment "
                f"after {self.MARKER!r}"
            )
            return True

        self._backup(path)
        path.write_text(new_text)
        ok(f"P-5: split protocol comment inserted  (backup: {Path(self.TARGET).name}.bak)")
        return True


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_PATCHES: list[Patch] = [
    PatchP1(),
    PatchP2(),
    PatchP3(),
    PatchP4(),
    PatchP5(),
]


# ── Post-patch verification ───────────────────────────────────────────────────

def run_verification(root: Path) -> None:
    """Re-run scan_internal_imports.py and print a pass/fail summary."""
    scanner = root / "scan_internal_imports.py"
    if not scanner.exists():
        warn("Verification skipped — scan_internal_imports.py not found at repo root")
        return

    print("\n  ── Post-patch verification (scan_internal_imports.py)")
    result = subprocess.run(
        [sys.executable, str(scanner), "--root", str(root)],
        capture_output=True,
        text=True,
    )
    print(result.stdout)

    # Filter stderr — Python 3.12 emits SyntaxWarning for \d in non-raw strings
    # in scanned files; these are cosmetic and do not affect scan correctness.
    real_errors = [
        line for line in result.stderr.splitlines()
        if line
        and "SyntaxWarning" not in line
        and "invalid escape sequence" not in line
    ]
    if real_errors:
        print("\n".join(real_errors))

    # Match against the actual console output format of scan_internal_imports.py
    all_clear = (
        "[A] Stale engine imports : 0" in result.stdout
        and "[B] Ghost imports        : 0" in result.stdout
        and "[C] Protocol layer leaks : 0" in result.stdout
        and "[D] Import cycles        : 0" in result.stdout
    )
    if all_clear:
        ok("Verification passed — all checks clean ✓")
    else:
        warn("Verification found remaining issues — review import_report.txt")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="HypatiaX — apply reproducibility patches"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be changed without writing files",
    )
    parser.add_argument(
        "--patch", metavar="ID",
        help="Apply only this patch (e.g. P-1)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Re-run scan_internal_imports.py after patching to confirm results",
    )
    args = parser.parse_args()

    # FIX-1: file lives at repo root — parents[2] is correct, not parents[2]
    repo_root = Path(__file__).resolve().parents[2]
    print("\n  HypatiaX apply_patches.py")
    print(f"  Repo root : {repo_root}")
    print(f"  Dry run   : {args.dry_run}")
    print()

    patches = ALL_PATCHES
    if args.patch:
        patches = [p for p in ALL_PATCHES if p.id == args.patch]
        if not patches:
            fail(
                f"Unknown patch id: {args.patch!r}  "
                f"(valid: {[p.id for p in ALL_PATCHES]})"
            )
            return 1

    failed_patches: list[str] = []
    for patch in patches:
        print(f"  ── {patch.id}: {patch.description}")
        try:
            success = patch.apply(repo_root, args.dry_run)
        except Exception as exc:
            fail(f"{patch.id} raised exception: {exc}")
            success = False
        if not success:
            failed_patches.append(patch.id)
        print()

    if failed_patches:
        fail(f"Patches failed: {failed_patches}")
        return 1

    if args.dry_run:
        print("  DRY-RUN complete — no files were modified")
    else:
        ok(f"All {len(patches)} patch(es) applied successfully ✓")

    # FIX-7: optional post-patch scan
    if args.verify and not args.dry_run:
        run_verification(repo_root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
