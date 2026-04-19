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
    python3 scripts/patches/apply_patches.py          # apply all patches
    python3 scripts/patches/apply_patches.py --dry-run # show diffs, no writes
    python3 scripts/patches/apply_patches.py --patch P-1  # apply one patch only

Exit codes:
  0 — all patches applied (or already applied — idempotent)
  1 — one or more patches failed
"""

import argparse
import re
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
        """Return True on success."""
        raise NotImplementedError

    def _replace_in_file(self, path: Path, old: str, new: str,
                         dry_run: bool, label: str = "") -> bool:
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
        path.write_text(text.replace(old, new))
        ok(f"{label or path.name}: patched")
        return True

    def _regex_replace(self, path: Path, pattern: str, repl: str,
                       dry_run: bool, label: str = "", flags=0) -> bool:
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
        path.write_text(new_text)
        ok(f"{label or path.name}: {n} replacement(s) applied")
        return True


# ── P-1: v40 → v50_2 engine swap ─────────────────────────────────────────────

class PatchP1(Patch):
    id = "P-1"
    description = "Swap hybrid_system_v40 → hybrid_system_v50_2 (FIX-C2)"

    # Files that contain v40 imports per Pipeline Plan §PART 1 / P-1
    TARGETS = [
        "hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_v2.py",
        "hypatiax/core/generation/hybrid_all_domains/suite_hybrid_system_all_domains.py",
        "hypatiax/core/generation/hybrid_defi_llm_guided/llm_guided_symbolic_discovery_defi.py",
        "hypatiax/core/generation/hybrid_defi_system/complete_defi_hybrid_system.py",
    ]

    def apply(self, root: Path, dry_run: bool) -> bool:
        ok_all = True
        for rel in self.TARGETS:
            path = root / rel
            # Replace all occurrences of v40 (but NOT v40fix — that's a separate file)
            ok_all &= self._regex_replace(
                path,
                r"hybrid_system_v40(?!fix)",
                "hybrid_system_v50_2",
                dry_run,
                label=rel,
                flags=re.MULTILINE,
            )
        return ok_all


# ── P-2: Fix duplicate DeFi case names ───────────────────────────────────────

class PatchP2(Patch):
    id = "P-2"
    description = "Fix 3 duplicate DeFi case names + checkpoint key → equation_id"

    TARGET = "hypatiax/experiments/benchmarks/hypatiax_defi_benchmark_v3c.py"

    RENAMES = [
        # (old_name_in_hard_tier, new_name)
        ('"Constant product formula"',           '"Constant product formula (multivariate)"'),
        ('"Funding rate cost"',                  '"Funding rate cost (extended)"'),
        ('"Concentrated liquidity position width"',
         '"Concentrated liquidity position width (v2)"'),
    ]

    def apply(self, root: Path, dry_run: bool) -> bool:
        path = root / self.TARGET
        if not path.exists():
            fail(f"P-2: {self.TARGET} not found")
            return False

        text = path.read_text(errors="replace")
        original = text

        # Apply renames — only rename the SECOND occurrence of each duplicate
        for old, new in self.RENAMES:
            parts = text.split(old)
            if len(parts) < 3:
                ok(f"P-2 rename {old[:40]}: already patched or no duplicate")
                continue
            # Replace only the last occurrence (2nd duplicate)
            text = old.join(parts[:-1]) + new + parts[-1]
            if not dry_run:
                ok(f"P-2: renamed second occurrence of {old[:40]!r}")
            else:
                print(f"  DRY-RUN  P-2: would rename second {old[:40]!r} → {new[:40]!r}")

        # Fix checkpoint key: case["name"] → case["equation_id"]
        text, n = re.subn(
            r'checkpoint\[case\["name"\]\]',
            'checkpoint[case["equation_id"]]',
            text,
        )
        if n > 0:
            if not dry_run:
                ok(f"P-2: checkpoint key → equation_id ({n} replacement(s))")
            else:
                print(f"  DRY-RUN  P-2: would fix checkpoint key ({n} replacement(s))")

        if text == original:
            ok("P-2: already fully patched — no changes needed")
            return True

        if not dry_run:
            path.write_text(text)
        return True


# ── P-3: populations=30 in make_pysr() ───────────────────────────────────────

class PatchP3(Patch):
    id = "P-3"
    description = "Set populations=30 as default in make_pysr() (fair ablation baseline)"

    TARGETS = [
        "hypatiax/core/training/baseline_neural_network.py",
        "protocols/experiment_protocol_ablation_exp1.py",
    ]

    def apply(self, root: Path, dry_run: bool) -> bool:
        ok_all = True
        for rel in self.TARGETS:
            path = root / rel
            if not path.exists():
                warn(f"P-3: {rel} not found — skipping")
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

    # Pattern: anthropic.Anthropic(api_key="sk-ant-...")
    API_KEY_PATTERN = re.compile(
        r'anthropic\.Anthropic\(\s*api_key\s*=\s*["\']sk-ant-[^"\']{10,}["\']',
        re.MULTILINE,
    )
    SAFE_REPLACEMENT = 'anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]'

    def apply(self, root: Path, dry_run: bool) -> bool:
        # Scan all .py files in repo for hardcoded keys
        hits = []
        for py in root.rglob("*.py"):
            try:
                text = py.read_text(errors="replace")
            except Exception:
                continue
            if self.API_KEY_PATTERN.search(text):
                hits.append(py)

        if not hits:
            ok("P-4: no hardcoded API keys found ✓")
            return True

        ok_all = True
        for py in hits:
            text = py.read_text(errors="replace")
            new_text = self.API_KEY_PATTERN.sub(self.SAFE_REPLACEMENT, text)
            # Ensure os is imported
            if "import os" not in new_text:
                new_text = "import os\n" + new_text
            if dry_run:
                warn(f"P-4 DRY-RUN: would remove hardcoded key in {py.relative_to(root)}")
            else:
                py.write_text(new_text)
                ok(f"P-4: removed hardcoded API key from {py.relative_to(root)}")
                warn("  ⚠  Rotate the exposed key at console.anthropic.com immediately!")

        return ok_all


# ── P-5: Feynman split protocol comment ──────────────────────────────────────

class PatchP5(Patch):
    id = "P-5"
    description = "Add Feynman 80/20 split protocol comment to run_comparative_suite_benchmark_v2.py"

    TARGET = "hypatiax/experiments/benchmarks/run_comparative_suite_benchmark_v2.py"

    COMMENT = '''\
    """
    Split protocol: 80/20 random split, random_state=42, extrap_multiplier=2.0.
    NOTE: This differs from the DeFi benchmark PCA 40/60 split (hypatiax_defi_benchmark_v3c.py).
    Results are NOT directly comparable. See §10.7 disclosure note in the paper.
    """
'''
    # Insert after `def run_experiment(` line
    MARKER = "def run_experiment("

    def apply(self, root: Path, dry_run: bool) -> bool:
        path = root / self.TARGET
        if not path.exists():
            warn(f"P-5: {self.TARGET} not found — skipping")
            return True  # non-fatal

        text = path.read_text(errors="replace")
        if "Split protocol:" in text:
            ok("P-5: split protocol comment already present")
            return True

        idx = text.find(self.MARKER)
        if idx == -1:
            warn(f"P-5: marker '{self.MARKER}' not found — skipping")
            return True

        # Find end of the def signature line
        end_of_line = text.find("\n", idx) + 1
        new_text = text[:end_of_line] + self.COMMENT + text[end_of_line:]

        if dry_run:
            print(f"  DRY-RUN  P-5: would insert split protocol comment after {self.MARKER!r}")
            return True

        path.write_text(new_text)
        ok(f"P-5: split protocol comment inserted into {self.TARGET}")
        return True


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_PATCHES: list[Patch] = [
    PatchP1(),
    PatchP2(),
    PatchP3(),
    PatchP4(),
    PatchP5(),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="HypatiaX — apply reproducibility patches")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be changed without writing files")
    parser.add_argument("--patch", metavar="ID",
                        help="Apply only this patch (e.g. P-1)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]  # scripts/patches/ → repo root
    print(f"\n  HypatiaX apply_patches.py")
    print(f"  Repo root : {repo_root}")
    print(f"  Dry run   : {args.dry_run}")
    print()

    patches = ALL_PATCHES
    if args.patch:
        patches = [p for p in ALL_PATCHES if p.id == args.patch]
        if not patches:
            fail(f"Unknown patch id: {args.patch!r}  (valid: {[p.id for p in ALL_PATCHES]})")
            return 1

    failed_patches = []
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
        print(f"  DRY-RUN complete — no files were modified")
    else:
        ok(f"All {len(patches)} patch(es) applied successfully ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
