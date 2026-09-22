#!/usr/bin/env python3
"""
rebuild_abstract_from_results.py

Build a NEW abstract from repository-derived results.

IMPORTANT
---------
This script deliberately does NOT copy numerical claims from an existing
abstract. It treats the repository result files as the source of truth.

It:
  1. clones/updates LLM-HypatiaX-REPRO;
  2. discovers result JSON/CSV/JSONL files;
  3. identifies candidate records for the major experiments;
  4. calculates transparent statistics where the record schema permits;
  5. writes a machine-readable evidence ledger;
  6. writes a numerical-results draft;
  7. writes an abstract containing ONLY values that passed explicit checks.

Because repository schemas can evolve, the script never invents a number.
If a metric cannot be derived unambiguously it is marked CANNOT_DERIVE and
is omitted from the generated abstract.

Usage
-----
python3 rebuild_abstract_from_results.py
python3 rebuild_abstract_from_results.py --repo ~/Downloads/GITHUB/LLM-HypatiaX-REPRO --refresh
python3 rebuild_abstract_from_results.py --repo /path/to/repo --out-dir abstract_rebuild

Outputs
-------
abstract_rebuild/
  result_inventory.csv
  result_evidence.json
  abstract_numbers.md
  generated_abstract.md
  audit_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

DEFAULT_URL = "https://github.com/sednabcn/LLM-HypatiaX-REPRO.git"
DEFAULT_REPO = Path("LLM-HypatiaX-REPRO")

TEXT_EXTENSIONS = {
    ".json", ".jsonl", ".csv", ".tsv", ".txt", ".md", ".tex", ".log",
    ".yaml", ".yml"
}

SKIP_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", "node_modules"
}

# Names/terms used only for discovery. They are NOT paper claims.
DOMAIN_TERMS = {
    "defi": ["defi", "liquidity", "var", "value at risk", "liquidation"],
    "nguyen12": ["nguyen12", "nguyen-12", "nguyen"],
    "feynman30": ["feynman", "feynman30", "feynman-30"],
    "arrhenius": ["arrhenius"],
    "portfolio_variance": ["portfolio_variance", "portfolio variance"],
    "michaelis_menten": ["michaelis", "michaelis-menten", "michaelis menten"],
    "timing": ["timing", "runtime", "speedup", "seconds"],
}

METHOD_TERMS = {
    "hypatiax": ["hypatiax", "hybrid"],
    "llm": ["pure_llm", "pure llm", "llm standalone", "llm"],
    "nn": ["neural", "nn", "mlp", "neural network"],
    "pysr": ["pysr"],
}


def run(cmd: list[str], cwd: Path | None = None) -> str:
    p = subprocess.run(
        cmd, cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if p.returncode:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}"
        )
    return p.stdout


def prepare_repo(repo: Path, url: str, refresh: bool) -> Path:
    if (repo / ".git").exists():
        if refresh:
            run(["git", "fetch", "--all", "--prune"], cwd=repo)
            branch = run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo
            ).strip()
            if branch and branch != "HEAD":
                run(["git", "pull", "--ff-only", "origin", branch], cwd=repo)
        return repo

    repo.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", url, str(repo)])
    return repo


def git_revision(repo: Path) -> str:
    try:
        return run(["git", "rev-parse", "HEAD"], cwd=repo).strip()
    except Exception:
        return "UNKNOWN"


def iter_files(repo: Path) -> Iterable[Path]:
    for p in repo.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_PARTS for part in p.parts):
            continue
        if p.suffix.lower() in TEXT_EXTENSIONS:
            yield p


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(read_text(path))
    except Exception:
        return None


def json_records(obj: Any) -> list[Any]:
    """
    Turn common JSON shapes into a list of candidate records.
    No assumptions about field names are made here.
    """
    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in (
            "records", "results", "data", "cases", "tasks", "runs",
            "experiments", "results_data", "benchmark_results"
        ):
            value = obj.get(key)
            if isinstance(value, list):
                return value

        # A dict whose values are records.
        vals = list(obj.values())
        if vals and all(isinstance(v, dict) for v in vals):
            return vals

        return [obj]

    return []


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def flatten_record(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            nk = normalize_key(str(k))
            path = f"{prefix}.{nk}" if prefix else nk
            if isinstance(v, (dict, list)):
                out.update(flatten_record(v, path))
            else:
                out[path] = v

    elif isinstance(obj, list):
        # Do not flatten arbitrary arrays into thousands of fields.
        for i, v in enumerate(obj[:100]):
            path = f"{prefix}[{i}]"
            if isinstance(v, (dict, list)):
                out.update(flatten_record(v, path))
            else:
                out[path] = v

    return out


def numeric(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        if s.endswith("%"):
            try:
                return float(s[:-1]) / 100.0
            except ValueError:
                return None
        try:
            x = float(s)
            return x if math.isfinite(x) else None
        except ValueError:
            return None
    return None


def find_values(record: dict[str, Any], patterns: list[str]) -> list[tuple[str, float]]:
    hits = []
    pats = [normalize_key(x) for x in patterns]

    for key, value in record.items():
        nk = normalize_key(key)
        if any(p in nk for p in pats):
            x = numeric(value)
            if x is not None:
                hits.append((key, x))

    return hits


def discover_json_records(repo: Path):
    """
    Return every parseable JSON record with provenance.

    The script keeps provenance at file + record index level so that every
    generated number can be traced back to raw repository data.
    """
    for path in repo.rglob("*.json"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue

        obj = load_json(path)
        if obj is None:
            continue

        records = json_records(obj)
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                continue

            yield {
                "file": str(path.relative_to(repo)),
                "record_index": i,
                "record": record,
                "flat": flatten_record(record),
            }


def classify_domain(file_name: str, flat: dict[str, Any]) -> set[str]:
    hay = (file_name + " " + " ".join(
        f"{k} {v}" for k, v in flat.items()
    )).lower()

    result = set()
    for domain, terms in DOMAIN_TERMS.items():
        if any(term.lower() in hay for term in terms):
            result.add(domain)
    return result


def classify_method(flat: dict[str, Any]) -> set[str]:
    hay = " ".join(f"{k} {v}" for k, v in flat.items()).lower()
    result = set()

    for method, terms in METHOD_TERMS.items():
        if any(term.lower() in hay for term in terms):
            result.add(method)

    return result


def collect_candidates(repo: Path):
    candidates = []

    for item in discover_json_records(repo):
        domains = classify_domain(item["file"], item["flat"])
        methods = classify_method(item["flat"])

        if domains or methods:
            candidates.append({
                "file": item["file"],
                "record_index": item["record_index"],
                "domains": sorted(domains),
                "methods": sorted(methods),
                "flat": item["flat"],
            })

    return candidates


def candidate_rows(candidates):
    rows = []

    for c in candidates:
        row = {
            "file": c["file"],
            "record_index": c["record_index"],
            "domains": ",".join(c["domains"]),
            "methods": ",".join(c["methods"]),
        }

        # Preserve the most relevant numeric fields for inspection.
        numeric_fields = []
        for k, v in c["flat"].items():
            x = numeric(v)
            if x is not None:
                if any(token in k for token in (
                    "r2", "r_squared", "score", "success", "runtime",
                    "time", "pass", "n_", "count", "seed"
                )):
                    numeric_fields.append(f"{k}={x}")

        row["numeric_fields"] = "; ".join(numeric_fields[:80])
        rows.append(row)

    return rows


def find_metric_values(candidates, domain: str, method: str | None,
                       patterns: list[str]):
    values = []

    for c in candidates:
        if domain not in c["domains"]:
            continue
        if method and method not in c["methods"]:
            continue

        hits = find_values(c["flat"], patterns)
        for key, value in hits:
            values.append({
                "file": c["file"],
                "record_index": c["record_index"],
                "key": key,
                "value": value,
            })

    return values


def unique_numeric(values: list[tuple[str, float]]) -> list[float]:
    return sorted(set(round(float(v[1]), 15) for v in values))


def median_result(values: list[tuple[str, float]]) -> dict:
    xs = unique_numeric(values)

    if not xs:
        return {
            "status": "CANNOT_DERIVE",
            "reason": "No unambiguous numeric observations found."
        }

    return {
        "status": "DERIVED",
        "n": len(xs),
        "median": median(xs),
        "min": min(xs),
        "max": max(xs),
    }


def rate_result(values: list[tuple[str, float]], threshold: float) -> dict:
    xs = unique_numeric(values)

    if not xs:
        return {
            "status": "CANNOT_DERIVE",
            "reason": "No numeric observations found."
        }

    passed = sum(x > threshold for x in xs)
    return {
        "status": "DERIVED",
        "n": len(xs),
        "passed": passed,
        "threshold": threshold,
        "rate": passed / len(xs),
    }


def build_number_ledger(repo: Path, candidates):
    """
    This ledger is intentionally explicit. A metric is only placed into the
    generated abstract when its provenance and population are sufficiently
    clear.

    We do NOT use numbers from an old abstract here.
    """
    ledger = []

    # Generic experiment metrics. The exact key matching is intentionally
    # broad because result schemas may contain r2/test_r2/extrap_r2 variants.
    specs = [
        ("NN_MEDIAN_R2", "all", "nn",
         ["test_r2", "r2_test", "test_r_squared", "r_squared_test"]),
        ("LLM_MEDIAN_R2", "all", "llm",
         ["test_r2", "r2_test", "test_r_squared", "r_squared_test"]),
        ("HYPATIAX_MEDIAN_R2", "all", "hypatiax",
         ["test_r2", "r2_test", "test_r_squared", "r_squared_test"]),
        ("PYSR_MEDIAN_R2", "all", "pysr",
         ["test_r2", "r2_test", "test_r_squared", "r_squared_test"]),
    ]

    for name, domain, method, patterns in specs:
        vals = []
        for c in candidates:
            if method not in c["methods"]:
                continue
            vals.extend(find_values(c["flat"], patterns))

        result = median_result(vals)
        result.update({
            "id": name,
            "method": method,
            "metric": "median test R2",
            "evidence": vals[:100],
        })
        ledger.append(result)

    # DeFi task population is derived from actual records, not the manuscript.
    defi_records = [
        c for c in candidates
        if "defi" in c["domains"]
    ]

    ledger.append({
        "id": "DEFI_RECORD_COUNT",
        "status": "DERIVED" if defi_records else "CANNOT_DERIVE",
        "n": len(defi_records),
        "metric": "number of discovered DeFi JSON records",
        "evidence": [
            {
                "file": c["file"],
                "record_index": c["record_index"]
            }
            for c in defi_records[:1000]
        ],
    })

    # Domain-specific existence/provenance checks.
    for domain in (
        "arrhenius",
        "portfolio_variance",
        "michaelis_menten",
        "nguyen12",
        "feynman30",
    ):
        records = [c for c in candidates if domain in c["domains"]]
        ledger.append({
            "id": domain.upper(),
            "status": "EVIDENCE_FOUND" if records else "CANNOT_DERIVE",
            "record_count": len(records),
            "metric": "domain result provenance",
            "evidence": [
                {
                    "file": c["file"],
                    "record_index": c["record_index"]
                }
                for c in records[:100]
            ],
        })

    # Timing: only derive if a clearly labelled runtime field exists.
    timing_values = []
    for c in candidates:
        if "timing" not in c["domains"]:
            continue
        timing_values.extend(find_values(
            c["flat"],
            ["runtime_seconds", "runtime_s", "elapsed_seconds",
             "elapsed_time", "runtime", "seconds"]
        ))

    ledger.append({
        "id": "TIMING_MEDIAN",
        "status": "DERIVED" if timing_values else "CANNOT_DERIVE",
        "metric": "runtime",
        "result": median_result(timing_values),
        "evidence": timing_values[:100],
    })

    return ledger


def safe_number(x: float, digits: int = 4) -> str:
    if abs(x) >= 100:
        return f"{x:.1f}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def write_outputs(repo: Path, out_dir: Path, candidates, ledger):
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory = candidate_rows(candidates)
    with (out_dir / "result_inventory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = ["file", "record_index", "domains", "methods", "numeric_fields"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(inventory)

    with (out_dir / "result_evidence.json").open("w", encoding="utf-8") as f:
        json.dump({
            "repository": str(repo.resolve()),
            "revision": git_revision(repo),
            "ledger": ledger,
        }, f, indent=2, ensure_ascii=False)

    # Human-readable numerical ledger.
    lines = [
        "# Abstract numerical ledger",
        "",
        f"- Repository: `{repo.resolve()}`",
        f"- Git revision: `{git_revision(repo)}`",
        "",
        "The values below come from repository result records. "
        "No old abstract number is used as input.",
        "",
        "| ID | Status | Result |",
        "|---|---|---|",
    ]

    for item in ledger:
        if item["id"] in {
            "NN_MEDIAN_R2", "LLM_MEDIAN_R2",
            "HYPATIAX_MEDIAN_R2", "PYSR_MEDIAN_R2"
        }:
            result = item
            if result["status"] == "DERIVED":
                value = (
                    f'median={safe_number(result["median"])}; '
                    f'n={result["n"]}'
                )
            else:
                value = result.get("reason", "")
        elif item["id"] == "DEFI_RECORD_COUNT":
            value = f'n={item.get("n", 0)}'
        elif item["id"] == "TIMING_MEDIAN":
            r = item.get("result", {})
            value = (
                safe_number(r["median"])
                if r.get("status") == "DERIVED"
                else r.get("reason", "")
            )
        else:
            value = (
                f'records={item.get("record_count", 0)}'
                if "record_count" in item
                else ""
            )

        lines.append(f'| {item["id"]} | {item["status"]} | {value} |')

    (out_dir / "abstract_numbers.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    # Generate an abstract only from derived numbers.
    nn = next((x for x in ledger if x["id"] == "NN_MEDIAN_R2"), None)
    llm = next((x for x in ledger if x["id"] == "LLM_MEDIAN_R2"), None)
    hyp = next((x for x in ledger if x["id"] == "HYPATIAX_MEDIAN_R2"), None)
    psr = next((x for x in ledger if x["id"] == "PYSR_MEDIAN_R2"), None)
    defi = next((x for x in ledger if x["id"] == "DEFI_RECORD_COUNT"), None)

    paragraphs = []

    # Method sentence contains no fabricated quantitative claim.
    paragraphs.append(
        "We present HypatiaX, a hybrid LLM–neural symbolic-regression "
        "architecture for analytical expression discovery from multimodal "
        "data. The reproducibility pipeline evaluates the system against "
        "standalone language-model, neural-network, and symbolic-regression "
        "baselines across analytical, scientific, and financial tasks."
    )

    quantitative = []

    if hyp and hyp.get("status") == "DERIVED":
        quantitative.append(
            f"Across the repository records used for this analysis, "
            f"HypatiaX has a median test R² of {safe_number(hyp['median'])} "
            f"(n={hyp['n']})."
        )

    if nn and nn.get("status") == "DERIVED":
        quantitative.append(
            f"The corresponding neural-network records have a median test "
            f"R² of {safe_number(nn['median'])} (n={nn['n']})."
        )

    if llm and llm.get("status") == "DERIVED":
        quantitative.append(
            f"The LLM records have a median test R² of "
            f"{safe_number(llm['median'])} (n={llm['n']})."
        )

    if psr and psr.get("status") == "DERIVED":
        quantitative.append(
            f"PySR records have a median test R² of "
            f"{safe_number(psr['median'])} (n={psr['n']})."
        )

    if defi and defi.get("status") == "DERIVED":
        quantitative.append(
            f"The current repository contains {defi['n']} discovered DeFi "
            f"result records; this count is reported directly from the "
            f"repository rather than inherited from the manuscript."
        )

    if quantitative:
        paragraphs.append(" ".join(quantitative))

    # Do NOT force unsupported domain-specific claims into the abstract.
    domain_names = {
        "ARRHENIUS": "Arrhenius",
        "PORTFOLIO_VARIANCE": "Portfolio Variance",
        "MICHAELIS_MENTEN": "Michaelis–Menten",
        "NGUYEN12": "Nguyen-12",
        "FEYNMAN30": "Feynman-30",
    }

    unsupported = []
    for item in ledger:
        if item["id"] in domain_names and item["status"] != "DERIVED":
            unsupported.append(domain_names[item["id"]])

    if unsupported:
        paragraphs.append(
            "Domain-specific numerical claims for "
            + ", ".join(unsupported)
            + " are not included here because the current result records do "
              "not provide an unambiguous derived statistic for them."
        )

    paragraphs.append(
        "All quantitative statements in this abstract are generated from "
        "the repository evidence ledger at the audited Git revision; "
        "unsupported numerical claims are omitted rather than substituted "
        "with values from an earlier manuscript version."
    )

    abstract = "# Generated Abstract\n\n" + "\n\n".join(paragraphs) + "\n"

    (out_dir / "generated_abstract.md").write_text(
        abstract, encoding="utf-8"
    )

    # Audit summary.
    derived = sum(
        1 for x in ledger
        if x.get("status") in {"DERIVED", "EVIDENCE_FOUND"}
    )
    unresolved = len(ledger) - derived

    summary = f"""# Abstract rebuild audit

Repository
: `{repo.resolve()}`

Git revision
: `{git_revision(repo)}`

Ledger entries
: {len(ledger)}

Derived/evidence entries
: {derived}

Unresolved entries
: {unresolved}

## Rule

The generated abstract is built from repository-derived evidence only.
Existing abstract numbers are not used as numerical inputs.

An unresolved metric is omitted rather than guessed.

## Important

The discovery stage identifies result records and computes statistics only
when their fields can be interpreted unambiguously. Domain-specific claims
such as Arrhenius, Portfolio Variance, Michaelis-Menten, Nguyen-12, timing,
and Feynman-30 should be promoted into the abstract only after their exact
paper metric and population are mapped to the corresponding repository
records.
"""

    (out_dir / "audit_summary.md").write_text(
        summary, encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--repo-url", default=DEFAULT_URL)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("abstract_rebuild"))
    args = parser.parse_args()

    try:
        repo = prepare_repo(args.repo, args.repo_url, args.refresh)
    except Exception as exc:
        print(f"ERROR: repository preparation failed:\n{exc}", file=sys.stderr)
        sys.exit(2)

    print(f"Repository: {repo.resolve()}")
    print(f"Revision:   {git_revision(repo)}")
    print("Discovering result records...")

    candidates = collect_candidates(repo)
    print(f"Candidate result records: {len(candidates)}")

    ledger = build_number_ledger(repo, candidates)
    write_outputs(repo, args.out_dir, candidates, ledger)

    print()
    print(f"Wrote {args.out_dir / 'result_inventory.csv'}")
    print(f"Wrote {args.out_dir / 'result_evidence.json'}")
    print(f"Wrote {args.out_dir / 'abstract_numbers.md'}")
    print(f"Wrote {args.out_dir / 'generated_abstract.md'}")
    print(f"Wrote {args.out_dir / 'audit_summary.md'}")
    print()
    print("IMPORTANT: review abstract_numbers.md before using the generated abstract.")


if __name__ == "__main__":
    main()
