#!/usr/bin/env python3
"""find_sources.py -- for every UNVERIFIED paper table, find the source file on THIS machine and say
whether CI can see it (tracked by git, not ignored, in a folder generate_tables.py reads).

Run from the repo root:   python3 find_sources.py            (searches INSIDE the repo only, every folder incl. data_old)
Paste the output back; nothing is modified.
"""
import json, os, subprocess, sys
from pathlib import Path

REPO = Path.cwd().resolve()
try:
    import yaml
    RESULTS = (REPO / yaml.safe_load(open("config/experiments.yml"))["results_root"]).resolve()
except Exception:
    RESULTS = REPO / "hypatiax/data/results"
SKIP = {".git", "node_modules", "site-packages", "__pycache__", ".cache", "Library", ".venv", "venv", "miniconda3", "anaconda3", ".local"}

def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True)

tracked = {(REPO / p).resolve() for p in git("ls-files").stdout.splitlines()}

def status(p: Path):
    p = p.resolve()
    inside = REPO in p.parents
    if not inside:
        return "OUTSIDE repo"
    if p in tracked:
        return "tracked"
    ign = git("check-ignore", "-q", str(p)).returncode == 0
    return "IGNORED by .gitignore" if ign else "untracked (git add it)"

def rel(p):  # path relative to results root if inside, else as is
    try: return str(p.resolve().relative_to(RESULTS))
    except Exception: return str(p)

def load(p):
    try: return json.loads(p.read_text())
    except Exception: return None

# (table, glob test, dirs the generator reads under RESULTS, required shape check)
def shape_portfolio(d): return isinstance(d, dict) and isinstance(d.get("seeds", d.get("results")), list) and len(d.get("seeds", d.get("results"))) >= 5
def shape_scal(d): return isinstance(d, dict) and bool(d.get("dataset_sizes")) and "per_size" in d
def shape_dict(d): return isinstance(d, dict)
def shape_list_dict(d): return isinstance(d, (list, dict))

TARGETS = [
  ("portfolio_seed_sweep", lambda n: n.startswith("portfolio_variance") and n.endswith(".json"), ["", ], shape_portfolio, "dict with >=5 'seeds'/'results'"),
  ("timing_breakdown",     lambda n: "timing_breakdown" in n and n.endswith(".json"), ["", "routing"], shape_dict, "dict of numbers"),
  ("scalability",          lambda n: n.startswith("scalability_") and n.endswith(".json"), ["", "routing"], shape_scal, "keys dataset_sizes + per_size"),
  ("validation_stats",     lambda n: "validation_log" in n and n.endswith(".json"), ["", "validation"], shape_dict, "dict; generator glob is 'validation_log*.json' (must START with validation_log)"),
  ("fix5_cases",           lambda n: n.startswith("fix5_cases") and n.endswith(".json"), ["", "routing", "fixes"], shape_list_dict, "list of cases or dict['cases']"),
  ("randomsplit",          lambda n: n.endswith(".json") and (("random" in n and "80_20" in n) or n.startswith("protocol_core_random_")), ["comparison_results/feynman-tests/exp2_multi", "comparison_results/feynman-tests/exp2_pca_4060", "exp2_multi", ""], lambda d: isinstance(d, dict) and isinstance(d.get("tests"), list), "dict with 'tests' list"),
]
CONFIG = [("formulas_detailed", "ground_truth_formulas.json"), ("hyperparameters_complete", "hyperparameters_complete.json"), ("jmlr", "jmlr_readiness.json")]

roots = [REPO] + [Path(a) for a in sys.argv[1:]]          # DEV repo only, unless you pass extra folders
seen, files = set(), []
for r in roots:
    for dp, dn, fn in os.walk(r):
        dn[:] = [d for d in dn if d not in SKIP]
        for f in fn:
            if f.endswith((".json", ".yml")):
                p = Path(dp) / f
                if p.resolve() not in seen:
                    seen.add(p.resolve()); files.append(p)

print(f"results root used by the generator: {RESULTS}\nscanned {len(files)} json/yml files\n")

def name_ok(table, n):
    if table == "validation_stats": return n.startswith("validation_log")
    return True

def report(table, hits, expect_dirs, note, cfg=False):
    print(f"=== {table}   (generator reads: {', '.join(d or '<results root>' for d in expect_dirs)})")
    if not hits:
        print("    NOT FOUND on this machine\n"); return
    for p, extra in hits[:40]:
        base = REPO if cfg else RESULTS
        ok_dir = any(p.resolve().parent == (base / d).resolve() for d in expect_dirs)
        nm = "" if name_ok(table, p.name) else "; FILE NAME NOT MATCHED by generator glob (rename to validation_log*.json)"
        print(f"    {p.relative_to(REPO) if REPO in p.resolve().parents else p}\n      {status(p)}; {'in a folder the generator reads' if ok_dir else 'NOT in a folder the generator reads'}; {extra}{nm}")
    print()

for table, match, dirs, shape, note in TARGETS:
    hits = []
    for p in files:
        if match(p.name):
            d = load(p)
            hits.append((p, ("shape OK" if shape(d) else f"shape WRONG, need {note}") if d is not None else "unreadable json"))
    report(table, hits, dirs, note)

for table, name in CONFIG:
    hits = [(p, "present") for p in files if p.name == name]
    report(table, hits, ["config"], "", cfg=True)

# DeFi seed files (timing_full / timing_llm_routed_full): non-PCA must hold exactly 74 records
hits = []
for p in files:
    if p.name.startswith("hypatiax_defi_benchmark") and "seed" in p.name and p.suffix == ".json":
        d = load(p); n = len(d) if isinstance(d, list) else "?"
        pca = "_pca_" in p.name
        ok = (n == 74) or pca
        hits.append((p, f"{n} records{' (PCA variant)' if pca else ''}{'' if ok else '  <-- generator IGNORES (needs 74)'}"))
report("timing_full / timing_llm_routed_full", hits,
       ["", "defi", "comparison_results/noise-noiseless/noiseless/defi", "comparison_results/noise-noiseless/noiseless/defi/multi-seeds"], "")

# noiseless / noisy shards: which carry the fields hardcoded/arch/time_noise/winrate need
print("=== protocol_core_noiseless_* / protocol_core_noisy_*  (hardcoded, arch, time_noise, winrate)")
rows = []
for p in files:
    if p.name.startswith(("protocol_core_noiseless_", "protocol_core_noisy_")) and "checkpoint" not in p.name:
        d = load(p); tests = d.get("tests", []) if isinstance(d, dict) else []
        hc = dec = 0
        for t in tests:
            res = t.get("results", {}) or {}
            pl = res.get("pure_llm") or res.get("PureLLM")
            hc += bool(isinstance(pl, dict) and pl.get("is_hardcoded"))
            dec += any(isinstance(r, dict) and (r.get("decision") or r.get("strategy")) for r in res.values())
        rows.append((p, len(tests), hc, dec))
for p, n, hc, dec in sorted(rows, key=lambda r: r[0].name)[:40]:
    print(f"    {p.name}: {n} tests, is_hardcoded=True in {hc}, decision/strategy in {dec}  [{status(p)}]")
print(f"    ({len(rows)} shards found; need hc>0 for tab:hardcoded, dec>0 for tab:arch)")
