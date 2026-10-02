#!/usr/bin/env python3
"""Run from the repo root:  python inspect_noise.py . > inspect_noise_out.txt
Prints a short summary of the noise-sweep result files so tab:arch / tab:noise_sensitivity can be checked.
Read-only; changes nothing."""
import hashlib, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
files = [p for p in repo.rglob("*.json")
         if re.search(r"noise(?!less)", str(p), re.I) and not re.search(r"sample", str(p), re.I)]
print(f"== {len(files)} noise json files")
bydir = defaultdict(list)
for p in files:
    bydir[str(p.parent)].append(p.name)
for d, names in sorted(bydir.items()):
    print(f"  {d}: {len(names)} files, e.g. {sorted(names)[:3]}")

md = defaultdict(list)
for p in files:
    md[hashlib.md5(p.read_bytes()).hexdigest()].append(str(p))
dups = {k: v for k, v in md.items() if len(v) > 1}
print(f"== identical-content duplicates: {len(dups)} groups")
for v in list(dups.values())[:5]:
    print("   ", v)

def walk(o, path=()):
    if isinstance(o, dict):
        yield path, o
        for k, v in o.items():
            yield from walk(v, path + (str(k),))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, path + (str(i),))

def short(d, n=14):
    out = {}
    for k, v in list(d.items())[:n]:
        out[k] = v if isinstance(v, (int, float, bool, type(None))) else (v[:60] if isinstance(v, str) else type(v).__name__)
    return out

key_re = re.compile(r"^(decision|route|routing|routing_decision|path|primary_path|selected_method|strategy|chosen)$", re.I)
shown = {"M3": 0, "M4": 0}
noise_vals, succ_keys, route_vals = Counter(), Counter(), {"M3": Counter(), "M4": Counter()}
for p in files:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    for path, d in walk(data):
        hay = " ".join(path + tuple(str(v) for v in d.values() if isinstance(v, str))).replace("_", "").lower()
        which = "M3" if ("enhancedhybridsystemdefi" in hay or re.search(r"\bm3\b", hay)) else \
                "M4" if ("hybridsystemllmnn" in hay or re.search(r"\bm4\b", hay)) else None
        for k, v in d.items():
            if re.search(r"noise", str(k), re.I) and isinstance(v, (int, float)) and not isinstance(v, bool):
                noise_vals[round(float(v), 4)] += 1
            if re.fullmatch(r"success|passed|ok|recovered|exact|correct\w*", str(k), re.I):
                succ_keys[str(k)] += 1
        if which and any(key_re.match(str(k)) and isinstance(v, str) for k, v in d.items()):
            for k, v in d.items():
                if key_re.match(str(k)) and isinstance(v, str):
                    route_vals[which][v] += 1
            if shown[which] < 1:
                shown[which] += 1
                print(f"\n== ONE RAW {which} RECORD  (file {p}, path {'/'.join(path)})")
                print(json.dumps(short(d), indent=1, default=str))
print("\n== routing values  M3:", dict(route_vals["M3"]), " M4:", dict(route_vals["M4"]))
print("== noise values seen anywhere (value: count of dicts):", dict(sorted(noise_vals.items())))
print("== success-like keys seen:", dict(succ_keys))
