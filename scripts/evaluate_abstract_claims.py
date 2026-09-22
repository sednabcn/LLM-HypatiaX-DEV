#!/usr/bin/env python3
"""
evaluate_abstract_claims.py

Repository-wide audit of abstract/paper numerical claims for LLM-HypatiaX.

What it does
------------
1. Clones or updates the public reproducibility repository.
2. Scans source/result/report files recursively.
3. Searches for evidence for C1-C7 and NV1-NV6.
4. Never treats a missing source as a verified claim.
5. Writes:
     abstract_claim_audit.csv
     abstract_claim_audit.md
     abstract_claim_audit.json

Usage
-----
python3 evaluate_abstract_claims.py

Optional:
python3 evaluate_abstract_claims.py --repo /path/to/LLM-HypatiaX-REPRO
python3 evaluate_abstract_claims.py --repo-url https://github.com/sednabcn/LLM-HypatiaX-REPRO.git
python3 evaluate_abstract_claims.py --refresh
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any

DEFAULT_URL = "https://github.com/sednabcn/LLM-HypatiaX-REPRO.git"
DEFAULT_DIR = Path("LLM-HypatiaX-REPRO")
OUT_CSV = Path("abstract_claim_audit.csv")
OUT_MD = Path("abstract_claim_audit.md")
OUT_JSON = Path("abstract_claim_audit.json")

# These are the claims currently under audit.  The expected values are
# intentionally kept here so the script can distinguish PAPER CLAIM from
# REPOSITORY EVIDENCE.  Change only when the manuscript/abstract changes.
CLAIMS = [
    {
        "id": "C1",
        "claim": "Neural MLP median R²",
        "expected": "-0.47 (abstract) / -0.4675 (main table)",
        "queries": ["-0.47", "-0.4675", "Neural MLP", "median R2", "median R²"],
    },
    {
        "id": "C2",
        "claim": "Pure LLM near-perfect rate",
        "expected": "62.2%",
        "queries": ["62.2%", "62.2", "near-perfect", "pure LLM"],
    },
    {
        "id": "C3",
        "claim": "HypatiaX median R²",
        "expected": "1.00",
        "queries": ["median R2", "median R²", "1.00", "HypatiaX"],
    },
    {
        "id": "C4",
        "claim": "HypatiaX near-perfect/success rate",
        "expected": "paper contains 59/74, 52/74 and 44/74=59.5%; earlier rebuilt table gives 48/73=65.75%",
        "queries": ["59/74", "52/74", "44/74", "59.5%", "48/73", "65.75%", "success rate"],
    },
    {
        "id": "C5",
        "claim": "Abstract delta consistency",
        "expected": "62.2% + 28.3pp and 5.4% + 85.1pp -> 90.5%",
        "queries": ["28.3pp", "85.1pp", "90.5%", "90.5", "62.2"],
    },
    {
        "id": "C6",
        "claim": "Hybrid versus standalone baselines on extrapolation R²",
        "expected": "Hybrid beats LLM/NN standalone; PySR-only must be checked separately",
        "queries": ["PySR-only", "PySR", "LLM standalone", "NN standalone", "extrapolation", "R²"],
    },
    {
        "id": "C7",
        "claim": "DeFi task count",
        "expected": "74 tasks in paper versus source file extrapolation_73cases_enhanced.json",
        "queries": ["74", "73", "extrapolation_73cases_enhanced", "DeFi tasks"],
    },
    {
        "id": "NV1",
        "claim": "Arrhenius per-equation result",
        "expected": "Paper/abstract value to be verified against repository evidence",
        "queries": ["Arrhenius", "arrhenius", "activation energy", "exp(-"],
    },
    {
        "id": "NV2",
        "claim": "Portfolio Variance per-equation result",
        "expected": "Paper/abstract value to be verified against repository evidence",
        "queries": ["Portfolio Variance", "portfolio_variance", "portfolio variance", "covariance"],
    },
    {
        "id": "NV3",
        "claim": "Michaelis-Menten per-equation result",
        "expected": "Paper/abstract value to be verified against repository evidence",
        "queries": ["Michaelis-Menten", "Michaelis Menten", "Michaelis", "Menten"],
    },
    {
        "id": "NV4",
        "claim": "Nguyen-12 result",
        "expected": "Paper/abstract value to be verified against repository evidence",
        "queries": ["Nguyen-12", "Nguyen12", "nguyen12", "Nguyen"],
    },
    {
        "id": "NV5",
        "claim": "Timing / speedup result",
        "expected": "Paper/abstract timing claim to be verified against repository evidence",
        "queries": ["1.73x", "1.73×", "1.73", "speedup", "timing", "runtime"],
    },
    {
        "id": "NV6",
        "claim": "Feynman-30 result",
        "expected": "Paper/abstract value to be verified against repository evidence",
        "queries": ["Feynman-30", "Feynman 30", "feynman30", "Feynman"],
    },
]

TEXT_EXTS = {
    ".json", ".jsonl", ".csv", ".tsv", ".txt", ".md", ".tex", ".py", ".yaml",
    ".yml", ".toml", ".sh", ".ipynb", ".rst", ".log"
}

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", ".tox", ".idea", ".vscode"
}

NUM_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?")


def run(cmd: list[str], cwd: Path | None = None) -> str:
    p = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")
    return p.stdout


def ensure_repo(repo: Path, url: str, refresh: bool) -> Path:
    if (repo / ".git").exists():
        if refresh:
            run(["git", "fetch", "--all", "--tags", "--prune"], cwd=repo)
            branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo).strip()
            if branch and branch != "HEAD":
                run(["git", "pull", "--ff-only", "origin", branch], cwd=repo)
        return repo

    repo.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", url, str(repo)])
    return repo


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iter_files(repo: Path):
    for p in repo.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in TEXT_EXTS:
            yield p


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def context_snippet(text: str, start: int, end: int, radius: int = 180) -> str:
    a = max(0, start - radius)
    b = min(len(text), end + radius)
    return re.sub(r"\s+", " ", text[a:b]).strip()


def search_evidence(repo: Path, queries: list[str], limit: int = 12):
    hits = []
    qlower = [q.lower() for q in queries]
    for path in iter_files(repo):
        text = read_text(path)
        if not text:
            continue
        low = text.lower()
        matched = [q for q, ql in zip(queries, qlower) if ql in low]
        if not matched:
            continue

        # Capture one useful snippet around the first matched query.
        q = matched[0]
        pos = low.find(q.lower())
        snippet = context_snippet(text, pos, pos + len(q))
        rel = str(path.relative_to(repo))
        hits.append({
            "file": rel,
            "sha256": sha256_file(path),
            "matched": matched,
            "snippet": snippet,
        })

        if len(hits) >= limit:
            break
    return hits


def load_json_files(repo: Path):
    for p in iter_files(repo):
        if p.suffix.lower() != ".json":
            continue
        try:
            yield p, json.loads(read_text(p))
        except Exception:
            continue


def flatten_numbers(obj: Any, prefix: str = ""):
    """Yield (path, value) for numeric leaves."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten_numbers(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten_numbers(v, f"{prefix}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, float(obj)


def find_numeric_evidence(repo: Path, terms: list[str], limit: int = 20):
    out = []
    for p, obj in load_json_files(repo):
        raw = read_text(p)
        low = raw.lower()
        if not any(t.lower() in low for t in terms):
            continue
        nums = list(flatten_numbers(obj))
        rel = str(p.relative_to(repo))
        for path, value in nums[:2000]:
            if math.isfinite(value):
                out.append({
                    "file": rel,
                    "json_path": path,
                    "value": value,
                })
                if len(out) >= limit:
                    return out
    return out


def git_revision(repo: Path) -> str:
    try:
        return run(["git", "rev-parse", "HEAD"], cwd=repo).strip()
    except Exception:
        return "UNKNOWN"


def classify(claim_id: str, hits: list[dict], numeric_hits: list[dict]) -> str:
    if not hits and not numeric_hits:
        return "NOT FOUND"

    # C1-C7 are deliberately conservative. Presence of evidence is not
    # automatically a MATCH because matching the manuscript requires the
    # extracted value and denominator to agree.
    if claim_id in {"C1", "C2", "C3", "C4", "C5", "C6", "C7"}:
        return "EVIDENCE FOUND — VALUE REQUIRES COMPARISON"

    return "EVIDENCE FOUND — VALUE REQUIRES COMPARISON"


def build_report(repo: Path):
    rows = []

    for c in CLAIMS:
        hits = search_evidence(repo, c["queries"])
        numeric_hits = find_numeric_evidence(repo, c["queries"])

        status = classify(c["id"], hits, numeric_hits)

        rows.append({
            "id": c["id"],
            "claim": c["claim"],
            "paper_expected": c["expected"],
            "status": status,
            "evidence_files": "; ".join(x["file"] for x in hits[:8]),
            "evidence_matches": "; ".join(
                f'{x["file"]}: {",".join(x["matched"])}' for x in hits[:8]
            ),
            "numeric_json_evidence": "; ".join(
                f'{x["file"]}:{x["json_path"]}={x["value"]}'
                for x in numeric_hits[:12]
            ),
            "snippets": " || ".join(x["snippet"] for x in hits[:5]),
        })

    return rows


def write_outputs(rows, repo: Path):
    fields = [
        "id", "claim", "paper_expected", "status",
        "evidence_files", "evidence_matches",
        "numeric_json_evidence", "snippets"
    ]

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    revision = git_revision(repo)

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump({
            "repository": DEFAULT_URL,
            "local_repo": str(repo.resolve()),
            "git_revision": revision,
            "claims": rows,
        }, f, indent=2, ensure_ascii=False)

    lines = [
        "# HypatiaX Abstract / Paper Claim Audit",
        "",
        f"- Repository: `{DEFAULT_URL}`",
        f"- Revision: `{revision}`",
        f"- Local checkout: `{repo.resolve()}`",
        "",
        "> **Audit rule:** source presence is not treated as numerical verification. "
        "A claim is `MATCH` only after its value, denominator, population, method and "
        "evaluation condition agree with the manuscript claim.",
        "",
        "## Results",
        "",
        "| ID | Claim | Paper value/description | Status | Evidence |",
        "|---|---|---|---|---|",
    ]

    for r in rows:
        ev = r["evidence_files"] or "—"
        lines.append(
            f'| {r["id"]} | {r["claim"]} | {r["paper_expected"]} | '
            f'**{r["status"]}** | `{ev}` |'
        )

    lines += [
        "",
        "## Evidence details",
        "",
    ]

    for r in rows:
        lines += [
            f'### {r["id"]} — {r["claim"]}',
            "",
            f'**Expected:** {r["paper_expected"]}',
            "",
            f'**Status:** {r["status"]}',
            "",
            "**Repository evidence:**",
            "",
            r["snippets"] or "No text evidence found.",
            "",
            "**Numeric JSON evidence:**",
            "",
            r["numeric_json_evidence"] or "No numeric JSON evidence found.",
            "",
        ]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--repo-url", default=DEFAULT_URL)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    try:
        repo = ensure_repo(args.repo, args.repo_url, args.refresh)
    except Exception as e:
        print(f"ERROR: cannot prepare repository: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"Repository: {repo.resolve()}")
    print(f"Revision:   {git_revision(repo)}")
    print("Scanning repository evidence...")

    rows = build_report(repo)
    write_outputs(rows, repo)

    print()
    for r in rows:
        print(f'{r["id"]:4} {r["status"]:<42} {r["claim"]}')

    print()
    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_MD}")
    print(f"Wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
