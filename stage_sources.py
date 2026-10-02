#!/usr/bin/env python3
"""stage_sources.py -- put source files where generate_tables.py can read them (dry-run unless --apply).

Run from the repo root.  Subcommands:

  defi      [--src DIR] [--apply]   five+ DeFi seed files with 74 records -> tab:timing_full, tab:timing_llm_routed_full
  validation --src FILE [--apply]   a validation log -> tab:validation_stats (renamed to validation_log_*.json)
  portfolio --src FILE [--apply]    portfolio sweep -> tab:portfolio_seed_sweep (shape is checked first)
  noise     [--src FILE] [--apply]  find noise_sweep_*.json files and show how many sigma levels each holds (time_noise, winrate, ...)
  shards    [--root DIR ...]        (scans the repo only unless --root is given) which noiseless shards carry is_hardcoded / decision fields (tab:hardcoded, tab:arch)
  inspect   FILE                    print the structure of a JSON file

Scans (noise, shards) look INSIDE the repo only. --src is the only way to read from outside, and every copy lands inside the repo.
Nothing is committed.  After --apply it prints the `git add -f` lines (-f because results folders are often ignored).
"""
import argparse, json, os, re, shutil, subprocess, sys
from pathlib import Path

REPO = Path.cwd().resolve()
try:
    import yaml
    RESULTS = (REPO / yaml.safe_load(open("config/experiments.yml"))["results_root"]).resolve()
except Exception:
    RESULTS = REPO / "hypatiax/data/results"
SKIP = {".git", "node_modules", "site-packages", "__pycache__", ".cache", "venv", ".venv", "miniconda3", "anaconda3", ".local", "pytest-of-" }
DEFI_DEST = RESULTS / "comparison_results/noise-noiseless/noiseless/defi/multi-seeds"
STAGED = []

def walk(root):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP and not d.startswith("pytest-of-")]
        for f in fn:
            yield Path(dp) / f

def load(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None

def stage(src: Path, dst: Path, apply: bool):
    note = ""
    if dst.exists():
        note = "  (REPLACES existing file)" if dst.read_bytes() != src.read_bytes() else "  (identical, already there)"
    print(f"  {src}\n    -> {dst.relative_to(REPO) if REPO in dst.parents else dst}{note}")
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst); STAGED.append(dst)

def finish(apply):
    if not apply:
        print("\nDRY RUN: nothing copied. Re-run with --apply to copy.")
    elif STAGED:
        print("\nCopied. Now:")
        for p in STAGED: print(f"  git add -f {p.relative_to(REPO)}")

# ---------------------------------------------------------------- defi
def cmd_defi(a):
    cands = {}
    roots = [Path(a.src)] if a.src else [REPO]
    for r in roots:
        for p in walk(r):
            if re.fullmatch(r"hypatiax_defi_benchmark_v4_results_seed\d+\.json", p.name):
                d = load(p)
                ok = isinstance(d, list) and len(d) == a.n and isinstance(d[0], dict) and "results" in d[0]
                cands.setdefault(p.parent, []).append((p, ok, len(d) if isinstance(d, list) else "?"))
    good = {d: [x for x in v if x[1]] for d, v in cands.items()}
    good = {d: v for d, v in good.items() if len({re.search(r"seed(\d+)", x[0].name).group(1) for x in v}) >= 2}
    if not good:
        print(f"No folder with >=2 v4 seed files of exactly {a.n} records that have a 'results' key.")
        for d, v in cands.items(): print("  ", d, [(x[0].name, x[2]) for x in v])
        return finish(a.apply)
    if len(good) > 1 and not a.src:
        print("Several folders qualify; pick one with --src DIR:")
        for d, v in good.items(): print(f"  {d}  seeds={sorted(re.search(r'seed(\d+)', x[0].name).group(1) for x in v)}")
        return
    d, v = next(iter(good.items()))
    print(f"Source folder: {d}")
    for p, _, _ in sorted(v, key=lambda x: x[0].name): stage(p, DEFI_DEST / p.name, a.apply)
    print("\nThe generator keeps one file per seed number, skips any non-PCA file whose record count != "
          f"{a.n}, and searches this folder, so 5-record copies elsewhere cannot shadow these.")
    finish(a.apply)

# ---------------------------------------------------------------- validation
def cmd_validation(a):
    src = Path(a.src); d = load(src)
    if not isinstance(d, dict): sys.exit("not a JSON object")
    print("top-level keys:", list(d)[:15])
    name = src.name if src.name.startswith("validation_log") else "validation_log_" + src.stem + ".json"
    stage(src, RESULTS / "validation" / name, a.apply); finish(a.apply)

# ---------------------------------------------------------------- portfolio
NEED = ("seed", "pysr_far_r2|p_far_r2", "hypatiax_far_r2|h_far_r2")
def cmd_portfolio(a):
    src = Path(a.src); d = load(src)
    seeds = d.get("seeds", d.get("results")) if isinstance(d, dict) else None
    if not isinstance(seeds, list) or len(seeds) < 5:
        print("The generator needs a dict whose 'seeds' (or 'results') is a list of >= 5 entries.")
        print("This file's structure:"); describe(d); return
    missing = [k for k in NEED if not any(any(x in s for x in k.split("|")) for s in seeds if isinstance(s, dict))]
    print(f"{len(seeds)} seed entries; fields missing from every entry: {missing or 'none'}")
    if missing: describe(seeds[0]); return
    stage(src, RESULTS / "portfolio_variance_seed_sweep.json", a.apply); finish(a.apply)

# ---------------------------------------------------------------- inspect
def describe(o, depth=0, maxd=3, key="<root>"):
    pad = "  " * depth
    if isinstance(o, dict):
        print(f"{pad}{key}: dict[{len(o)}]  keys={list(o)[:12]}")
        if depth < maxd:
            for k in list(o)[:6]: describe(o[k], depth + 1, maxd, k)
    elif isinstance(o, list):
        print(f"{pad}{key}: list[{len(o)}]")
        if o and depth < maxd: describe(o[0], depth + 1, maxd, "[0]")
    else:
        print(f"{pad}{key}: {type(o).__name__} = {str(o)[:50]}")
def cmd_inspect(a): describe(load(a.file))


# ---------------------------------------------------------------- noise
NOISE_DEST = RESULTS / "comparison_results/feynman-tests/noise-sweep"
def cmd_noise(a):
    if a.src:
        d = load(a.src)
        lv = d.get("noise_levels") if isinstance(d, dict) else None
        if not lv or "per_noise" not in d: sys.exit("not a noise_sweep file (needs 'noise_levels' and 'per_noise')")
        print(f"{len(lv)} sigma levels: {sorted(lv)}")
        stage(Path(a.src), NOISE_DEST / Path(a.src).name, a.apply); return finish(a.apply)
    rows = []
    for p in walk(REPO):
        if p.name.startswith("noise_sweep_") and p.suffix == ".json" and "checkpoint" not in p.name:
            d = load(p)
            if isinstance(d, dict) and "noise_levels" in d:
                rows.append((len(d["noise_levels"]), p, sorted(d["noise_levels"])))
    print("The generator reads ONE noise_sweep_*.json (the newest by name) and never merges files.")
    print("Look for a file that holds all the sigma levels the paper prints:\n")
    for n, p, lv in sorted(rows, key=lambda r: -r[0])[:15]:
        print(f"  {n} levels {lv}\n    {p}")
    if not rows: print("  none found")

# ---------------------------------------------------------------- shards
def cmd_shards(a):
    roots = [Path(r) for r in a.root] or [REPO]
    seen, hits, total = set(), [], 0
    for r in roots:
        for p in walk(r):
            if p.name.startswith(("protocol_core_noiseless_", "protocol_core_noisy_")) and p.suffix == ".json" and "checkpoint" not in p.name:
                try: key = (p.name, p.stat().st_size)
                except OSError: continue
                if key in seen: continue
                seen.add(key); total += 1
                d = load(p); tests = d.get("tests", []) if isinstance(d, dict) else []
                hc = dec = 0
                for t in tests:
                    res = t.get("results", {}) or {}
                    pl = res.get("pure_llm") or res.get("PureLLM")
                    hc += bool(isinstance(pl, dict) and pl.get("is_hardcoded"))
                    dec += any(isinstance(x, dict) and (x.get("decision") or x.get("strategy")) for x in res.values())
                if hc or dec: hits.append((p, len(tests), hc, dec))
    print(f"{total} distinct shards (by name+size); {len(hits)} carry is_hardcoded or decision/strategy\n")
    for p, n, hc, dec in sorted(hits, key=lambda h: -(h[2] + h[3]))[:25]:
        tag = "  [pytest fixture?]" if n <= 2 else ""
        print(f"  {p}\n     {n} tests, is_hardcoded={hc}, decision/strategy={dec}{tag}")

def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("defi"); s.add_argument("--src"); s.add_argument("--n", type=int, default=74); s.add_argument("--apply", action="store_true"); s.set_defaults(f=cmd_defi)
    s = sp.add_parser("validation"); s.add_argument("--src", required=True); s.add_argument("--apply", action="store_true"); s.set_defaults(f=cmd_validation)
    s = sp.add_parser("portfolio"); s.add_argument("--src", required=True); s.add_argument("--apply", action="store_true"); s.set_defaults(f=cmd_portfolio)
    s = sp.add_parser("noise"); s.add_argument("--src"); s.add_argument("--apply", action="store_true"); s.set_defaults(f=cmd_noise)
    s = sp.add_parser("shards"); s.add_argument("--root", action="append", default=[]); s.set_defaults(f=cmd_shards)
    s = sp.add_parser("inspect"); s.add_argument("file"); s.set_defaults(f=cmd_inspect)
    a = ap.parse_args(); a.f(a)

if __name__ == "__main__":
    main()
