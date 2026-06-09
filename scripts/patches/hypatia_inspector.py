#!/usr/bin/env python3
"""
hypatia_inspector.py  — HypatiaX Repo Inspector & Auto-Fixer
=============================================================
Walks the entire HypatiaX repository tree, detects every known issue
class from the audit report, then auto-applies all safe fixes — driven
entirely by scripts/patches/issue_registry.json.

  • No FIX-* IDs are hardcoded. The registry is the single source of truth.
  • Every fix function is idempotent (safe to re-run).
  • Backs up each file before modifying it (.bak suffix).
  • --apply mode writes real changes; default is a read-only dry-run.
  • --update-registry patches the registry JSON after successful fixes.

Detectors (one per audit notebook)
───────────────────────────────────
  NB-01  bibliography   missing \\bibitem, duplicate alias keys
  NB-02  cross_ref      duplicate/misplaced \\label, stale §-numbers,
                        stale filenames in supplementary
  NB-03  structure      \\section / \\subsection without \\label,
                        unused equation labels
  NB-04  numerical      count mismatches, heading terminology drift
  NB-05  figures        \\fbox placeholders, missing figure files
  NB-06  code_quality   duplicate benchmark case names, stale imports

Auto-fixable (status=open, auto_fixable=true in registry)
──────────────────────────────────────────────────────────
  FIX-B1   Insert missing \\bibitem{koza1994genetic}
  FIX-B2   Redirect \\cite{cranmer2023interpretable} → cranmer2023pysr
  FIX-B3   Redirect \\cite{udrescu2020aifeynman}     → udrescu2020ai
  FIX-XR1  Remove \\label{sec:llm_domain}; fix \\ref call sites
  FIX-XR2  Move \\label{sec:r2_bugfix} out of \\item block
  FIX-XR3  Section 7.3 → 7.4 in supp_routing_improvements.tex
  FIX-XR4  jmlr_paper_main.tex → jmlr-hypatiax-paper-final.tex in Supp A
  FIX-N1   "71 cases" → "70 tasks"
  FIX-N2   "Five-Layer Architecture Overview" → "Five-Stage Routing…"
  FIX-C2   hybrid_system_v40 → hybrid_system_v50_2
  FIX-F2   Copy fig18_r2_heatmap_improved.pdf  (if source found)
  FIX-F3   Copy fig09_r2_heatmap_regimes.pdf   (if source found)
  FIX-F4   Copy fig1_seed_sweep.pdf             (if source found)

Manual-only (detected & reported, never touched by this script)
───────────────────────────────────────────────────────────────
  FIX-F1   Replace \\fbox placeholder with real figure
  FIX-F5   Verify algorithm file exists and compiles
  FIX-C1   Rename duplicate benchmark cases + rerun checkpoint
  FIX-S1   Add \\label to all 12 \\section commands
  FIX-S2   Add \\label to all ~55 \\subsection commands
  FIX-EQ   Add \\eqref refs or remove 8 unused equation labels

Usage
─────
  # Detect everything, apply nothing (safe read-only):
  python hypatia_inspector.py --repo /path/to/repo

  # Apply all auto-fixable issues:
  python hypatia_inspector.py --repo /path/to/repo --apply

  # Custom roots + write findings JSON + update registry after fixes:
  python hypatia_inspector.py --repo . --apply \\
      --tex-dir src/latex --supp-dir supplementary --py-dir src \\
      --registry scripts/patches/issue_registry.json \\
      --json-out findings.json --update-registry

  # Only show critical/high issues:
  python hypatia_inspector.py --repo . --severity critical,high

Exit codes
──────────
  0  all open issues resolved (or zero open issues in dry-run)
  1  open issues remain / unrecoverable error
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

# ── ANSI colour helpers ───────────────────────────────────────────────────────
import os
_TTY = sys.stdout.isatty() or bool(os.environ.get("FORCE_COLOR"))
def _c(code: str, t: str) -> str: return f"\033[{code}m{t}\033[0m" if _TTY else t
RED    = lambda t: _c("31;1", t)
ORANGE = lambda t: _c("33;1", t)
YELLOW = lambda t: _c("33",   t)
GREEN  = lambda t: _c("32;1", t)
BLUE   = lambda t: _c("34",   t)
BOLD   = lambda t: _c("1",    t)
DIM    = lambda t: _c("2",    t)
CYAN   = lambda t: _c("36",   t)

SEV_COLOR = {"critical": RED, "high": ORANGE, "medium": YELLOW, "low": BLUE}

# ── Registry helpers ──────────────────────────────────────────────────────────

DEFAULT_REGISTRY = Path("scripts/patches/issue_registry.json")

def load_registry(path: Path) -> list[dict]:
    if not path.exists():
        print(YELLOW(f"⚠  Registry not found at {path} — discovery-only mode."))
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
        print(DIM(f"   Loaded {len(entries)} registry entries from {path}"))
        return entries
    except Exception as exc:
        print(YELLOW(f"⚠  Registry parse error: {exc} — discovery-only mode."))
        return []

def _reg(registry: list[dict], fix_id: str) -> dict | None:
    return next((e for e in registry if e.get("id") == fix_id), None)

def reg_status(registry, fix_id):   e = _reg(registry, fix_id); return e.get("status","unknown") if e else "unknown"
def reg_reason(registry, fix_id):   e = _reg(registry, fix_id); return (e or {}).get("false_positive_reason") or (e or {}).get("action","")
def is_auto_fixable(registry, fix_id):
    e = _reg(registry, fix_id)
    return bool(e and e.get("auto_fixable") and e.get("status") == "open")

def should_skip(registry, fix_id) -> tuple[bool, str]:
    e = _reg(registry, fix_id)
    if not e: return False, ""
    st = e.get("status","open")
    if st == "false_positive": return True, f"false positive — {e.get('false_positive_reason','')}"
    if st == "resolved":       return True, f"already resolved — {e.get('action','')}"
    return False, ""

def save_registry(path: Path, registry: list[dict]) -> None:
    path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

# ── File I/O helpers ──────────────────────────────────────────────────────────

def tex_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.tex") if ".bak" not in p.suffixes)

def py_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if ".bak" not in p.suffixes)

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def write(p: Path, content: str, dry_run: bool) -> None:
    if dry_run:
        print(DIM(f"              [dry] would write {p.name}"))
    else:
        bak = p.with_suffix(p.suffix + ".bak")
        shutil.copy2(p, bak)
        p.write_text(content, encoding="utf-8")

def find_file(root: Path, name: str) -> Path | None:
    hits = [p for p in root.rglob(name) if ".bak" not in p.suffixes]
    return hits[0] if hits else None

# ── Finding dataclass ─────────────────────────────────────────────────────────

class Finding:
    __slots__ = ("fix_id","severity","summary","detail","files","auto_fixable","status","skip_reason")
    def __init__(self, fix_id, severity, summary, detail="", files=None, auto_fixable=False):
        self.fix_id       = fix_id
        self.severity     = severity
        self.summary      = summary
        self.detail       = detail
        self.files        = files or []
        self.auto_fixable = auto_fixable
        self.status       = "detected"
        self.skip_reason  = ""

# ─────────────────────────────────────────────────────────────────────────────
# DETECTORS
# ─────────────────────────────────────────────────────────────────────────────

class Inspector:
    def __init__(self, repo: Path, tex_root: Path, supp_root: Path, py_root: Path):
        self.repo      = repo
        self.tex_root  = tex_root
        self.supp_root = supp_root
        self.py_root   = py_root
        self.findings: list[Finding] = []

    def _add(self, f: Finding): self.findings.append(f)

    # ── NB-01: Bibliography ───────────────────────────────────────────────────

    def detect_bibliography(self):
        txf = tex_files(self.tex_root)
        if not txf: return

        bibitems: set[str] = set()
        citations: dict[str, list[str]] = {}

        for p in txf:
            src = read(p)
            for m in re.finditer(r"\\bibitem(?:\[.*?\])?\{([^}]+)\}", src):
                bibitems.add(m.group(1))
            for m in re.finditer(r"\\cite(?:p|t|alt|author|year)?\{([^}]+)\}", src):
                for key in (k.strip() for k in m.group(1).split(",")):
                    citations.setdefault(key, []).append(str(p))

        # FIX-B1: cited key with no bibitem
        for key, files in citations.items():
            if key not in bibitems:
                self._add(Finding(
                    "FIX-B1", "critical",
                    f"\\cite{{{key}}} has no \\bibitem — will produce [?] in compiled PDF",
                    f"Cited in: {', '.join(set(files))}",
                    list(set(files)),
                    auto_fixable=(key == "koza1994genetic"),
                ))

        # FIX-B2 / FIX-B3: duplicate alias bibitems
        ALIASES = [
            ("cranmer2023interpretable", "cranmer2023pysr", "FIX-B2"),
            ("udrescu2020aifeynman",     "udrescu2020ai",   "FIX-B3"),
        ]
        for alias, canon, fix_id in ALIASES:
            if alias in bibitems and canon in bibitems:
                cite_files = list({str(p) for p in txf
                                   if alias in read(p)})
                self._add(Finding(
                    fix_id, "high",
                    f"Duplicate bibitem alias: \\bibitem{{{alias}}} and \\bibitem{{{canon}}} are the same paper",
                    f"Remove \\bibitem{{{alias}}}; redirect all \\cite{{{alias}}} → {canon}.",
                    cite_files, auto_fixable=True,
                ))

    # ── NB-02: Cross-references ───────────────────────────────────────────────

    def detect_crossrefs(self):
        txf    = tex_files(self.tex_root)
        labels: dict[str, list[tuple[str,int]]] = {}
        refs:   set[str] = set()

        for p in txf:
            for i, line in enumerate(read(p).splitlines(), 1):
                for m in re.finditer(r"\\label\{([^}]+)\}", line):
                    labels.setdefault(m.group(1), []).append((str(p), i))
                for m in re.finditer(r"\\(?:auto)?ref\{([^}]+)\}", line):
                    refs.add(m.group(1))

        # FIX-XR1: duplicate labels on same section
        if "sec:llm_domain" in labels:
            self._add(Finding(
                "FIX-XR1", "medium",
                "\\label{sec:llm_domain} duplicates sec:llm_limitations on Section 3",
                "Remove \\label{sec:llm_domain}; redirect all \\ref{sec:llm_domain} → sec:llm_limitations.",
                [loc[0] for loc in labels["sec:llm_domain"]], auto_fixable=True,
            ))

        # FIX-XR2: label inside \item block
        for p in txf:
            lines = read(p).splitlines()
            in_item = False
            for i, line in enumerate(lines, 1):
                if re.search(r"\\item\b", line):       in_item = True
                if re.search(r"\\(?:sub)*section\b", line): in_item = False
                if in_item and "\\label{sec:r2_bugfix}" in line:
                    self._add(Finding(
                        "FIX-XR2", "medium",
                        f"\\label{{sec:r2_bugfix}} is inside \\item block at {p.name}:{i}",
                        "Move label to \\subsection{Pipeline Corrections in v3.0} heading.",
                        [str(p)], auto_fixable=True,
                    ))
                    break

        # FIX-XR3 / FIX-XR4: stale content in supplementary
        supp_txf = tex_files(self.supp_root) if self.supp_root.exists() else []
        for p in supp_txf:
            src = read(p)
            if "supp_routing_improvements" in p.name:
                if re.search(r"Section\s+7\.3|7\.3\s*\(Component", src):
                    self._add(Finding("FIX-XR3", "medium",
                        f"Stale 'Section 7.3' in {p.name} — should be §7.4",
                        "Change 7.3 → 7.4 in supp_routing_improvements.tex.",
                        [str(p)], auto_fixable=True))
            if "jmlr_paper_main.tex" in src:
                self._add(Finding("FIX-XR4", "medium",
                    f"Stale filename 'jmlr_paper_main.tex' in {p.name}",
                    "Replace with 'jmlr-hypatiax-paper-final.tex' throughout Supp A.",
                    [str(p)], auto_fixable=True))

        # Informational: undefined \ref targets
        undef = sorted(r for r in refs if r not in labels)
        if undef:
            self._add(Finding("FIX-XR0", "low",
                f"{len(undef)} undefined \\ref target(s)",
                "Keys: " + ", ".join(undef[:20]) + (" …" if len(undef)>20 else ""),
                auto_fixable=False))

    # ── NB-03: Structure ──────────────────────────────────────────────────────

    def detect_structure(self):
        txf = tex_files(self.tex_root)
        all_refs: set[str] = set()
        for p in txf:
            for m in re.finditer(r"\\(?:eq)?ref\{([^}]+)\}", read(p)):
                all_refs.add(m.group(1))

        miss_sec, miss_sub, unused_eq = [], [], []
        for p in txf:
            lines = read(p).splitlines()
            for i, line in enumerate(lines, 1):
                if re.match(r"\s*\\section\{", line):
                    ctx = "".join(lines[i-1:i+1])
                    if "\\label{" not in ctx:
                        miss_sec.append((str(p), i, line.strip()))
                if re.match(r"\s*\\subsection\{", line):
                    ctx = "".join(lines[i-1:i+1])
                    if "\\label{" not in ctx:
                        miss_sub.append((str(p), i, line.strip()))
                for m in re.finditer(r"\\label\{(eq:[^}]+)\}", line):
                    if m.group(1) not in all_refs:
                        unused_eq.append((str(p), i, m.group(1)))

        if miss_sec:
            self._add(Finding("FIX-S1","low",
                f"{len(miss_sec)} top-level \\section(s) have no \\label",
                "\n  ".join(f"{f}:{ln}  {t[:70]}" for f,ln,t in miss_sec[:10]),
                list({f for f,_,_ in miss_sec}), auto_fixable=False))
        if miss_sub:
            self._add(Finding("FIX-S2","low",
                f"{len(miss_sub)} \\subsection(s) have no \\label",
                "\n  ".join(f"{f}:{ln}  {t[:70]}" for f,ln,t in miss_sub[:10]),
                list({f for f,_,_ in miss_sub}), auto_fixable=False))
        if unused_eq:
            self._add(Finding("FIX-EQ","low",
                f"{len(unused_eq)} equation label(s) defined but never referenced",
                "Labels: " + ", ".join(lbl for _,_,lbl in unused_eq),
                auto_fixable=False))

    # ── NB-04: Numerical consistency ──────────────────────────────────────────

    def detect_numerical(self):
        for p in tex_files(self.tex_root):
            for i, line in enumerate(read(p).splitlines(), 1):
                if re.search(r"\b71\s+cases\b", line, re.IGNORECASE):
                    self._add(Finding("FIX-N1","medium",
                        f"'71 cases' at {p.name}:{i} — should be '70 tasks'",
                        "Change '71 cases' → '70 tasks' (instability section body, line ~1637).",
                        [str(p)], auto_fixable=True))
                if re.search(r"Five-Layer Architecture Overview", line, re.IGNORECASE):
                    self._add(Finding("FIX-N2","medium",
                        f"Heading 'Five-Layer Architecture Overview' at {p.name}:{i}",
                        "Abstract/intro/§7 all use 'five-stage routing'. "
                        "Rename to 'Five-Stage Routing Architecture Overview'.",
                        [str(p)], auto_fixable=True))

    # ── NB-05: Figures ────────────────────────────────────────────────────────

    def detect_figures(self):
        txf = tex_files(self.tex_root)
        figs_dir = self.repo / "figures"

        # FIX-F1: \fbox placeholder
        for p in txf:
            src = read(p)
            if re.search(r"\\fbox\{", src) and re.search(
                    r"hypatiaX_three_systems|architecture", src, re.IGNORECASE):
                self._add(Finding("FIX-F1","high",
                    f"\\fbox{{}} placeholder found in {p.name} — final architecture figure missing",
                    "Replace with final PDF/PNG from Figures/architecture_figures/.",
                    [str(p)], auto_fixable=False))

        # FIX-F2/F3/F4: missing figure files
        FIGS = {
            "FIX-F2": ("fig18_r2_heatmap_improved.pdf", "Figures/figures-cosmetic-last/"),
            "FIX-F3": ("fig09_r2_heatmap_regimes.pdf",  "Figures/figures-cosmetic-last/"),
            "FIX-F4": ("fig1_seed_sweep.pdf",            "Figures/figures-portfolio-variance/"),
        }
        for fix_id, (fname, src_subdir) in FIGS.items():
            target = figs_dir / fname
            if not target.exists():
                src_path = self.repo / src_subdir / fname
                self._add(Finding(fix_id,"medium",
                    f"Missing figure: figures/{fname}",
                    f"Source {'✓ found' if src_path.exists() else '✗ not found'}: "
                    f"{src_subdir}{fname}",
                    [str(src_path)] if src_path.exists() else [],
                    # auto-fixable only when source exists
                    auto_fixable=src_path.exists()))

        # FIX-F5: algorithm file
        if not list(self.repo.rglob("hypatiaX_algorithm1_routing_cascade_v2*")):
            self._add(Finding("FIX-F5","medium",
                "hypatiaX_algorithm1_routing_cascade_v2 not found in repo tree",
                "Verify the algorithm file exists and compiles correctly.",
                auto_fixable=False))

    # ── NB-06: Code quality ───────────────────────────────────────────────────

    def detect_code_quality(self):
        pyf = py_files(self.py_root)

        # FIX-C1: duplicate case names in defi benchmark
        for p in [f for f in pyf if "defi_benchmark" in f.name]:
            names = re.findall(r'"([^"]{10,})"', read(p))
            names += re.findall(r"'([^']{10,})'", read(p))
            seen: dict[str,int] = {}
            dups = []
            for n in names:
                seen[n] = seen.get(n,0) + 1
                if seen[n] == 2: dups.append(n)
            if dups:
                self._add(Finding("FIX-C1","medium",
                    f"{len(dups)} duplicate case name(s) in {p.name}",
                    "Duplicates: " + "; ".join(f'"{d}"' for d in dups),
                    [str(p)], auto_fixable=False))

        # FIX-C2: stale v40 imports
        OLD, NEW = "hybrid_system_v40", "hybrid_system_v50_2"
        for p in pyf:
            if OLD in read(p):
                hits = [i for i,ln in enumerate(read(p).splitlines(),1) if OLD in ln]
                self._add(Finding("FIX-C2","medium",
                    f"Stale import '{OLD}' in {p.name} at lines {hits}",
                    f"Replace with '{NEW}' throughout.",
                    [str(p)], auto_fixable=True))

    # ── Run all ───────────────────────────────────────────────────────────────

    def run_all(self) -> list[Finding]:
        print(BOLD("\n── Detection pass ──────────────────────────────────────────"))
        for label, fn in [
            ("NB-01  Bibliography   ", self.detect_bibliography),
            ("NB-02  Cross-refs     ", self.detect_crossrefs),
            ("NB-03  Structure      ", self.detect_structure),
            ("NB-04  Numerical      ", self.detect_numerical),
            ("NB-05  Figures        ", self.detect_figures),
            ("NB-06  Code quality   ", self.detect_code_quality),
        ]:
            before = len(self.findings)
            fn()
            n = len(self.findings) - before
            print(f"  {label} {CYAN(str(n))} finding(s)")
        return self.findings


# ─────────────────────────────────────────────────────────────────────────────
# FIXERS  — one method per FIX-* id, all idempotent
# ─────────────────────────────────────────────────────────────────────────────

class Fixer:
    def __init__(self, repo: Path, tex_root: Path, supp_root: Path, py_root: Path,
                 dry_run: bool = True):
        self.repo      = repo
        self.tex_root  = tex_root
        self.supp_root = supp_root
        self.py_root   = py_root
        self.dry_run   = dry_run
        self.applied:  list[str] = []
        self.skipped:  list[str] = []

    # ── Low-level patch helpers ───────────────────────────────────────────────

    def _patch_str(self, p: Path, old: str, new: str, label: str) -> int:
        src = read(p)
        n   = src.count(old)
        if n == 0: return 0
        print(DIM(f"              {label}: {n} replacement(s) in {p.name}"))
        write(p, src.replace(old, new), self.dry_run)
        return n

    def _patch_re(self, p: Path, pattern: str, repl, label: str, flags: int = 0) -> int:
        src       = read(p)
        new, n    = re.subn(pattern, repl, src, flags=flags)
        if n == 0: return 0
        print(DIM(f"              {label}: {n} replacement(s) in {p.name}"))
        write(p, new, self.dry_run)
        return n

    def _tex_files(self):   return tex_files(self.tex_root)
    def _supp_files(self):  return tex_files(self.supp_root) if self.supp_root.exists() else []
    def _py_files(self):    return py_files(self.py_root)

    # ── Fix implementations ───────────────────────────────────────────────────

    def apply_FIX_B1(self):
        """Insert \\bibitem{koza1994genetic} after koza1992gp in any tex file."""
        NEW_BIB = (
            r"\bibitem{koza1994genetic}" + "\n"
            r"J.~R. Koza." + "\n"
            r"\newblock {\em Genetic Programming II: Automatic Discovery of Reusable Programs}." + "\n"
            r"\newblock MIT Press, 1994."
        )
        for p in self._tex_files():
            src = read(p)
            if r"\bibitem{koza1994genetic}" in src:
                print(DIM(f"              FIX-B1: \\bibitem{{koza1994genetic}} already present in {p.name}"))
                self.applied.append("FIX-B1 (already present)")
                return
            if r"\bibitem{koza1992gp}" in src:
                idx      = src.index(r"\bibitem{koza1992gp}")
                next_bib = src.find(r"\bibitem{", idx + 20)
                ins      = next_bib if next_bib != -1 else len(src)
                new_src  = src[:ins] + NEW_BIB + "\n\n" + src[ins:]
                print(DIM(f"              FIX-B1: inserted \\bibitem{{koza1994genetic}} in {p.name}"))
                write(p, new_src, self.dry_run)
                self.applied.append(f"FIX-B1 — inserted bibitem in {p.name}")
                return
        self.skipped.append("FIX-B1: koza1992gp anchor not found — manual insert required")

    def apply_FIX_B2(self):
        for p in self._tex_files():
            # Step 1: redirect any surviving alias cite keys
            self._patch_re(p, r"cranmer2023interpretable", "cranmer2023pysr",
                           "FIX-B2 redirect cite")
            # Step 2: remove alias bibitem block if it still exists
            self._patch_re(
                p,
                r"\\bibitem(?:\[.*?\])?\{cranmer2023interpretable\}[^\n]*\n(?:[^\n]*\n)*?(?=\\bibitem|\\end\{thebibliography\}|\Z)",
                "", "FIX-B2 remove alias bibitem")
            # Step 3: remove exact duplicate bibitem{cranmer2023pysr} blocks
            #         (keeps the first occurrence, removes subsequent identical blocks)
            src = read(p)
            pattern = re.compile(
                r"(\\bibitem(?:\[[^\]]*\])?\{cranmer2023pysr\}[^\n]*\n(?:[^\n]*\n)*?)"
                r"(?=\\bibitem|\\end\{thebibliography\}|\Z)",
                re.DOTALL)
            blocks = pattern.findall(src)
            if len(blocks) > 1:
                # Rebuild bib section keeping only the first match
                new_src = src
                for dup in blocks[1:]:
                    new_src = new_src.replace(dup, "", 1)
                if new_src != src:
                    print(DIM(f"              FIX-B2: removed {len(blocks)-1} duplicate bibitem(s) in {p.name}"))
                    write(p, new_src, self.dry_run)
        self.applied.append("FIX-B2")

    def apply_FIX_B3(self):
        for p in self._tex_files():
            # Step 1: redirect any surviving alias cite keys
            self._patch_re(p, r"udrescu2020aifeynman", "udrescu2020ai",
                           "FIX-B3 redirect cite")
            # Step 2: remove alias bibitem block if it still exists
            self._patch_re(
                p,
                r"\\bibitem(?:\[.*?\])?\{udrescu2020aifeynman\}[^\n]*\n(?:[^\n]*\n)*?(?=\\bibitem|\\end\{thebibliography\}|\Z)",
                "", "FIX-B3 remove alias bibitem")
            # Step 3: remove exact duplicate bibitem{udrescu2020ai} blocks
            src = read(p)
            pattern = re.compile(
                r"(\\bibitem(?:\[[^\]]*\])?\{udrescu2020ai\}[^\n]*\n(?:[^\n]*\n)*?)"
                r"(?=\\bibitem|\\end\{thebibliography\}|\Z)",
                re.DOTALL)
            blocks = pattern.findall(src)
            if len(blocks) > 1:
                new_src = src
                for dup in blocks[1:]:
                    new_src = new_src.replace(dup, "", 1)
                if new_src != src:
                    print(DIM(f"              FIX-B3: removed {len(blocks)-1} duplicate bibitem(s) in {p.name}"))
                    write(p, new_src, self.dry_run)
        self.applied.append("FIX-B3")

    def apply_FIX_XR1(self):
        for p in self._tex_files():
            self._patch_str(p, r"\label{sec:llm_domain}", "",
                            "FIX-XR1 remove duplicate label")
            self._patch_re(p, r"\\(auto)?ref\{sec:llm_domain\}",
                           r"\\\1ref{sec:llm_limitations}",
                           "FIX-XR1 redirect \\ref")
        self.applied.append("FIX-XR1")

    def apply_FIX_XR2(self):
        TARGET_LABEL  = r"\label{sec:r2_bugfix}"
        TARGET_SUBSEC = "Pipeline Corrections in v3.0"
        for p in self._tex_files():
            src = read(p)
            if TARGET_LABEL not in src: continue
            already = re.search(
                re.escape(r"\subsection{" + TARGET_SUBSEC + "}") + r"\s*" + re.escape(TARGET_LABEL),
                src)
            if already:
                print(DIM(f"              FIX-XR2: label already on subsection in {p.name}"))
                self.applied.append("FIX-XR2 (already correct)")
                return
            cleaned  = src.replace(TARGET_LABEL, "")
            escaped  = TARGET_LABEL.replace("\\","\\\\")
            new_src  = re.sub(
                r"(\\subsection\{" + re.escape(TARGET_SUBSEC) + r"\})",
                r"\1\n" + escaped,
                cleaned,
            )
            if new_src != src:
                print(DIM(f"              FIX-XR2: moved label to subsection in {p.name}"))
                write(p, new_src, self.dry_run)
                self.applied.append(f"FIX-XR2 — {p.name}")
                return
        self.skipped.append("FIX-XR2: subsection heading not found")

    def apply_FIX_XR3(self):
        for p in self._supp_files():
            if "supp_routing_improvements" not in p.name: continue
            n  = self._patch_re(p, r"Section\s+7\.3(\s*\(Component\s+3\))",
                                r"Section 7.4\1", "FIX-XR3 7.3→7.4 (Component 3)")
            n += self._patch_re(p, r"(Proposition\s+1[^.]*?Section\s+)7\.3",
                                r"\g<1>7.4", "FIX-XR3 Prop 1 ref", flags=re.DOTALL)
            # Catch any remaining bare "Section~7.3" or "Section 7.3" patterns
            n += self._patch_re(p, r"(Section[~\s]+)7\.3\b",
                                r"\g<1>7.4", "FIX-XR3 bare section ref")
            if n: self.applied.append(f"FIX-XR3 — {p.name}")

    def apply_FIX_XR4(self):
        # Fix stale filename in ALL supplementary files, not just routing
        for p in self._supp_files():
            n = self._patch_str(p, "jmlr_paper_main.tex", "jmlr-hypatiax-paper-final.tex",
                                "FIX-XR4 filename")
            n += self._patch_re(p, r"\\texttt\{jmlr_paper_main\}",
                                r"\\texttt{jmlr-hypatiax-paper-final}", "FIX-XR4 texttt")
            if n: self.applied.append(f"FIX-XR4 — {p.name}")

    # ── Section / subsection label fixes ──────────────────────────────────────

    # Expected top-level section labels (title fragment → label key)
    _SECTION_LABELS: dict[str, str] = {
        "Introduction":                      "sec:intro",
        "Related Work":                      "sec:related",
        "Empirical Evidence":                "sec:llm_limitations",
        "Theoretical Framework":             "sec:theory",
        "Problem Formulation":               "sec:problem",
        "Benchmark Design":                  "sec:benchmark",
        "Methodology":                       "sec:method",
        "HypatiaX Architecture":             "sec:architecture",
        "Experimental Setup":                "sec:setup",
        "Results":                           "sec:results",
        "Discussion":                        "sec:discussion",
        "Conclusion":                        "sec:conclusion",
        "Reproducibility":                   "sec:reproducibility",
        "Full Benchmark Case":               "app:cases",
        "Benchmark Version History":         "app:versions",
        "Corrected Runtime Claim":           "app:timing",
    }

    # Subsection labels (title fragment → label key)
    _SUBSEC_LABELS: dict[str, str] = {
        "Contributions":                     "subsec:contributions",
        "Symbolic Regression":               "subsec:sr",
        "LLMs for Mathematical":             "subsec:llm_math",
        "Equation Learners":                 "subsec:eql",
        "Hybrid Symbolic":                   "subsec:hybrid_related",
        "Extrapolation in Neural":           "subsec:nn_extrap",
        "Experimental Design":               "subsec:exp_design",
        "Baseline Results":                  "subsec:baseline",
        "Failure Mode Taxonomy":             "subsec:failure_modes",
        "Case Studies":                      "subsec:case_studies",
        "Success Pattern":                   "subsec:success_patterns",
        "Extrapolation Testing":             "subsec:extrap_testing",
        "Formal Definitions":               "subsec:formal_defs",
        "Main Theoretical Result":           "subsec:main_result",
        "Implications for Discovery":        "subsec:implications",
        "Functional Form Recovery":          "subsec:functional_form",
        "Overview and Scope":               "subsec:overview",
        "Difficulty Classification":         "subsec:difficulty_class",
        "Data Generation":                   "subsec:data_gen",
        "Extrapolation Protocol":            "sec:split",
        "Architecture Overview":             "subsec:arch_overview",
        "Component 1":                       "sec:llm_gen",
        "Component 2":                       "sec:nn",
        "Component 3":                       "sec:routing",
        "Unified Formula Executor":          "sec:executor",
        "Design Principles":                 "subsec:design_principles",
        "Assumptions":                       "sec:assumptions",
        "Five-Stage Routing Architecture":   "sec:validation_framework",
        "System Variants":                   "subsec:variants",
        "Hybrid DeFi Variant":               "sec:hybrid_defi_spec",
        "Symbolic Discovery Core":           "subsec:symbolic_core",
        "Multi-Layer Validation":            "sec:validation_detail",
        "Benchmark Summary":                 "subsec:bench_summary",
        "Methods Compared":                  "subsec:methods",
        "Evaluation Metrics":               "subsec:metrics",
        "Runtime Measurement":               "sec:timing_setup",
        "Implementation Details":            "subsec:impl",
        "Five-System Comparative":           "sec:five_systems",
        "Overall Extrapolation":             "subsec:overall",
        "Performance by Difficulty":         "sec:difficulty",
        "Runtime Analysis":                  "sec:timing",
        "Portfolio Variance":                "sec:portfolio_seed_sweep",
        "Ablation":                          "sec:ablation_core15",
        "Feynman Extrapolation":             "sec:feynman30",
        "Nguyen-12":                         "sec:nguyen12",
        "Stability Under Stochastic":        "sec:instability_results",
        "Three Core Findings":               "subsec:three_findings",
        "Arrhenius and Scale":               "sec:arrhenius_failure",
        "Stability--Accuracy":               "subsec:stability_accuracy",
        "Why Transcendental":                "subsec:transcendental",
        "OOD Metric":                        "sec:ood_proxy",
        "Design Principles for Analytical":  "subsec:design_principles_disc",
        "Comparison with Related":           "subsec:comparison",
        "Ethical Considerations":            "subsec:ethics",
        "Broader Applicability":             "subsec:broader",
        "Limitations and Future":            "sec:limitations",
    }

    def _add_labels_to_headings(self, p: "Path", cmd: str,
                                label_map: "dict[str, str]") -> int:
        """Insert \\label{key} after any \\cmd{Title} that is missing its label.
        Returns the number of labels inserted."""
        src   = read(p)
        lines = src.splitlines()
        out   = []
        added = 0
        i = 0
        while i < len(lines):
            ln = lines[i]
            m  = re.match(
                r'^(\s*\\' + cmd + r'\*?\s*(?:\[[^\]]*\])?\s*\{)(.+?)(\})\s*$',
                ln)
            if m:
                title     = m.group(2)
                label_key = next(
                    (lbl for frag, lbl in label_map.items()
                     if frag.lower() in title.lower()),
                    None)
                out.append(ln)
                if label_key:
                    # Check next 3 lines for existing label
                    lookahead = "\n".join(lines[i+1:i+4])
                    if f"\\label{{{label_key}}}" not in lookahead:
                        out.append(f"\\label{{{label_key}}}")
                        added += 1
                        print(DIM(f"              FIX-S: +\\label{{{label_key}}} after '{title[:50]}'"))
            else:
                out.append(ln)
            i += 1
        if added:
            new_src = "\n".join(out)
            # Deduplicate: if a label now appears twice, remove the second
            for lbl in label_map.values():
                tag   = f"\\label{{{lbl}}}"
                parts = new_src.split(tag)
                if len(parts) > 2:          # more than one occurrence
                    new_src = parts[0] + tag + "".join(parts[1:]).replace(tag, "", len(parts)-2)
            write(p, new_src, self.dry_run)
        return added

    def apply_FIX_S1(self):
        """Add \\label to every top-level \\section missing one."""
        total = 0
        for p in self._tex_files():
            total += self._add_labels_to_headings(p, "section", self._SECTION_LABELS)
        if total:
            self.applied.append(f"FIX-S1 — {total} section label(s) added")
        else:
            self.applied.append("FIX-S1 (all section labels already present)")

    def apply_FIX_S2(self):
        """Add \\label to every \\subsection missing one."""
        total = 0
        for p in self._tex_files():
            total += self._add_labels_to_headings(p, "subsection", self._SUBSEC_LABELS)
        if total:
            self.applied.append(f"FIX-S2 — {total} subsection label(s) added")
        else:
            self.applied.append("FIX-S2 (all subsection labels already present)")

    def apply_FIX_N1(self):
        for p in self._tex_files():
            n = self._patch_re(p, r"\b71\s+cases\b", "70 tasks",
                               "FIX-N1", flags=re.IGNORECASE)
            if n: self.applied.append(f"FIX-N1 — {p.name}")

    def apply_FIX_N2(self):
        for p in self._tex_files():
            n  = self._patch_re(p, r"\\subsection\{Five-Layer Architecture Overview\}",
                                r"\\subsection{Five-Stage Routing Architecture Overview}",
                                "FIX-N2 subsection heading")
            n += self._patch_re(p, r"Five-Layer Architecture Overview",
                                "Five-Stage Routing Architecture Overview",
                                "FIX-N2 inline refs", flags=re.IGNORECASE)
            if n: self.applied.append(f"FIX-N2 — {p.name}")

    def apply_FIX_C2(self):
        for p in self._py_files():
            n = self._patch_re(p, r"\bhybrid_system_v40\b", "hybrid_system_v50_2",
                               "FIX-C2 v40→v50_2")
            if n: self.applied.append(f"FIX-C2 — {p.name}")

    def apply_FIX_F2(self, fname="fig18_r2_heatmap_improved.pdf",
                     src_sub="Figures/figures-cosmetic-last/"):
        self._copy_figure(fname, src_sub, "FIX-F2")

    def apply_FIX_F3(self, fname="fig09_r2_heatmap_regimes.pdf",
                     src_sub="Figures/figures-cosmetic-last/"):
        self._copy_figure(fname, src_sub, "FIX-F3")

    def apply_FIX_F4(self, fname="fig1_seed_sweep.pdf",
                     src_sub="Figures/figures-portfolio-variance/"):
        self._copy_figure(fname, src_sub, "FIX-F4")

    def _copy_figure(self, fname: str, src_sub: str, fix_id: str):
        src  = self.repo / src_sub / fname
        dest = self.repo / "figures" / fname
        if dest.exists():
            print(DIM(f"              {fix_id}: {fname} already in figures/"))
            self.applied.append(f"{fix_id} (already present)")
            return
        if not src.exists():
            self.skipped.append(f"{fix_id}: source {src} not found — copy manually")
            return
        if not self.dry_run:
            (self.repo / "figures").mkdir(exist_ok=True)
            shutil.copy2(src, dest)
            print(DIM(f"              {fix_id}: copied {fname} → figures/"))
        else:
            print(DIM(f"              [dry] {fix_id}: would copy {src} → figures/{fname}"))
        self.applied.append(f"{fix_id} — {fname}")

    # ── Dispatch ──────────────────────────────────────────────────────────────

    _DISPATCH: dict[str, str] = {
        "FIX-B1":  "apply_FIX_B1",  "FIX-B2":  "apply_FIX_B2",  "FIX-B3":  "apply_FIX_B3",
        "FIX-XR1": "apply_FIX_XR1", "FIX-XR2": "apply_FIX_XR2", "FIX-XR3": "apply_FIX_XR3",
        "FIX-XR4": "apply_FIX_XR4", "FIX-N1":  "apply_FIX_N1",  "FIX-N2":  "apply_FIX_N2",
        "FIX-C2":  "apply_FIX_C2",  "FIX-F2":  "apply_FIX_F2",  "FIX-F3":  "apply_FIX_F3",
        "FIX-F4":  "apply_FIX_F4",
    }

    def apply(self, fix_id: str):
        m = self._DISPATCH.get(fix_id)
        if m:
            getattr(self, m)()
        else:
            self.skipped.append(f"{fix_id}: no applier — manual action required")


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
STATUS_ICON = {"detected":"⚠️ ", "fixed":"✅", "manual":"🛠️ ", "skipped":"⏭️ "}

def print_report(findings: list[Finding], registry: list[dict],
                 fixer: Fixer, sev_filter: set[str] | None) -> int:

    findings.sort(key=lambda f: (SEV_ORDER.get(f.severity,9), f.fix_id))
    seen_skip: set[str] = set()   # deduplicate skip lines

    print(BOLD("\n── Inspection Report ───────────────────────────────────────"))
    print(f"  Date: {date.today()}    Total findings: {len(findings)}\n")

    remaining = 0

    for f in findings:
        if sev_filter and f.severity not in sev_filter:
            continue
        icon    = STATUS_ICON.get(f.status, "❓")
        sev_fn  = SEV_COLOR.get(f.severity, lambda t: t)
        sev_tag = sev_fn(f.severity.upper().ljust(8))
        id_tag  = BOLD(f.fix_id.ljust(10))
        print(f"  {icon}  {id_tag}  [{sev_tag}]  {f.summary}")

        if f.status == "skipped" and f.fix_id not in seen_skip:
            print(DIM(f"              ↳ SKIP: {f.skip_reason[:100]}"))
            seen_skip.add(f.fix_id)
        elif f.status == "manual":
            first_line = (f.detail or "").splitlines()[0] if f.detail else ""
            print(DIM(f"              ↳ Manual: {first_line[:100]}"))
        elif f.detail and f.status == "detected":
            for line in f.detail.splitlines()[:2]:
                print(DIM(f"              {line}"))

        if f.files and f.status not in ("skipped",):
            for fp in f.files[:2]:
                print(DIM(f"              📄 {fp}"))

        if f.status not in ("skipped",) and not (f.auto_fixable and f.status == "fixed"):
            if f.status in ("detected", "manual"):
                remaining += 1
        print()

    # Applied & skipped lines from fixer
    if fixer.applied:
        print(BOLD("── Applied fixes ───────────────────────────────────────────"))
        for line in fixer.applied:
            icon = "🔍" if line.startswith("[dry]") or not fixer.dry_run is False else "✅"
            icon = "✅" if not fixer.dry_run else "🔍"
            print(f"  {icon}  {line}")
        print()

    if fixer.skipped:
        print(BOLD("── Could not auto-fix (manual required) ────────────────────"))
        for line in fixer.skipped:
            print(f"  🛠️   {line}")
        print()

    # Summary bar
    by_status: dict[str,int] = {}
    for f in findings:
        if sev_filter and f.severity not in sev_filter: continue
        by_status[f.status] = by_status.get(f.status,0) + 1

    print(BOLD("── Summary ─────────────────────────────────────────────────"))
    print(f"  Findings detected:      {len(findings)}")
    print(f"  🟡 Skipped (FP/resolved): {by_status.get('skipped',0)}")
    print(f"  ✅ Auto-fixed:           {len(fixer.applied)}")
    print(f"  🛠️  Manual required:      {by_status.get('manual',0)}")
    print(f"  ⚠️  Still open:           {remaining}")
    print()
    return remaining


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY UPDATER
# ─────────────────────────────────────────────────────────────────────────────

def update_registry(registry: list[dict], fixed_ids: list[str], reg_path: Path,
                    dry_run: bool) -> None:
    """Mark successfully applied fixes as 'resolved' in the registry JSON."""
    changed = False
    today   = date.today().isoformat()
    for e in registry:
        if e.get("id") in fixed_ids and e.get("status") == "open":
            e["status"]  = "resolved"
            e["updated"] = today
            changed      = True
    if not changed:
        return
    if dry_run:
        print(DIM(f"   [dry] would mark {fixed_ids} as resolved in {reg_path}"))
    else:
        save_registry(reg_path, registry)
        print(GREEN(f"   Registry updated — {len(fixed_ids)} fix(es) marked resolved → {reg_path}"))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HypatiaX repo inspector + auto-fixer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--repo",             default=".",       help="Repository root (default: cwd)")
    p.add_argument("--tex-dir",          default="",        help="TeX source root (default: repo root)")
    p.add_argument("--supp-dir",         default="supplementary", help="Supplementary TeX root")
    p.add_argument("--py-dir",           default="",        help="Python source root (default: repo root)")
    p.add_argument("--registry",         default="",        help="Path to issue_registry.json")
    p.add_argument("--apply",            action="store_true", help="Apply auto-fixable fixes (default: dry-run)")
    p.add_argument("--update-registry",  action="store_true", help="Mark applied fixes as resolved in registry")
    p.add_argument("--severity",         default="",        help="Filter: critical,high,medium,low")
    p.add_argument("--json-out",         default="",        help="Write findings JSON to this path")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    repo      = Path(args.repo).resolve()
    tex_root  = (repo / args.tex_dir).resolve() if args.tex_dir  else repo
    supp_root = (repo / args.supp_dir).resolve()
    py_root   = (repo / args.py_dir).resolve()  if args.py_dir   else repo
    reg_path  = Path(args.registry).resolve()   if args.registry else repo / DEFAULT_REGISTRY

    # Sanity-check: the main paper tex must exist at repo root.  If it doesn't,
    # the caller is likely running from the wrong directory (e.g. notebooks/
    # instead of the repo root), which causes FileNotFoundError deep inside
    # notebook cells that open the file by bare name.  Fail fast with a clear
    # message rather than a cryptic traceback.
    PAPER_TEX = repo / "jmlr-hypatiax-paper-final.tex"
    if not PAPER_TEX.exists():
        print(f"\n[ERROR] jmlr-hypatiax-paper-final.tex not found under --repo {repo}")
        print(f"        Expected: {PAPER_TEX}")
        print(f"        Run hypatia_inspector.py from the repository root, or pass")
        print(f"        --repo /path/to/repo pointing at the directory that contains")
        print(f"        jmlr-hypatiax-paper-final.tex.")
        sys.exit(1)

    sev_filter = {s.strip().lower() for s in args.severity.split(",")} if args.severity else None

    print(BOLD(f"\n{'='*62}"))
    print(BOLD(f"  HypatiaX Inspector  {'(DRY RUN)' if not args.apply else '(APPLY MODE)'}"))
    print(BOLD(f"{'='*62}"))
    print(f"  Repo:      {repo}")
    print(f"  TeX root:  {tex_root}")
    print(f"  Supp root: {supp_root}")
    print(f"  Py root:   {py_root}")
    print(f"  Registry:  {reg_path}")

    registry  = load_registry(reg_path)
    inspector = Inspector(repo, tex_root, supp_root, py_root)
    findings  = inspector.run_all()

    fixer      = Fixer(repo, tex_root, supp_root, py_root, dry_run=not args.apply)
    applied_ids: set[str] = set()

    print(BOLD("\n── Fix pass ────────────────────────────────────────────────"))
    for f in findings:
        skip, reason = should_skip(registry, f.fix_id)
        if skip:
            f.status = "skipped"; f.skip_reason = reason
            if f.fix_id not in applied_ids:
                print(DIM(f"  ⏭️  {f.fix_id} — {reason[:90]}"))
                applied_ids.add(f.fix_id)
            continue

        if f.auto_fixable and is_auto_fixable(registry, f.fix_id):
            if f.fix_id not in applied_ids:
                tag = "" if args.apply else "[dry] "
                print(f"  🔧  {tag}{f.fix_id} — applying …")
                fixer.apply(f.fix_id)
                applied_ids.add(f.fix_id)
            f.status = "fixed" if args.apply else "detected"
        else:
            if f.fix_id not in applied_ids:
                print(DIM(f"  🛠️  {f.fix_id} — manual action required"))
                applied_ids.add(f.fix_id)
            f.status = "manual"

    remaining = print_report(findings, registry, fixer, sev_filter)

    # Optionally write findings JSON
    if args.json_out:
        out = [{"fix_id": f.fix_id, "severity": f.severity, "status": f.status,
                "summary": f.summary, "auto_fixable": f.auto_fixable,
                "files": f.files, "skip_reason": f.skip_reason}
               for f in findings]
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(GREEN(f"  JSON findings → {args.json_out}"))

    # Optionally update registry
    if args.update_registry and fixer.applied:
        cleanly_fixed = [a.split(" — ")[0].split(" (")[0] for a in fixer.applied
                         if not a.startswith("[dry]")]
        update_registry(registry, cleanly_fixed, reg_path, dry_run=not args.apply)

    if remaining == 0:
        print(GREEN("  ✅  All open issues resolved. Ready for CI gate.\n"))
        return 0
    else:
        hint = "Re-run with --apply to fix auto-fixable issues." if not args.apply else \
               "Fix the remaining manual items listed above."
        print(YELLOW(f"  ⚠   {remaining} open issue(s) remain. {hint}\n"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
