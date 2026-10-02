import json, sys
from collections import defaultdict
from pathlib import Path
base = Path(sys.argv[1]) / "comparison_results"
seen = defaultdict(list)
for p in sorted(base.rglob("protocol_core_noiseless_*.json")):
    if any(s in p.name for s in ("checkpoint", "_sig", "MISSING")):
        continue
    try:
        tests = json.loads(p.read_text()).get("tests", [])
    except Exception:
        continue
    for t in tests:
        k = (t.get("domain"), t.get("description") or t.get("name"))
        seen[k].append((p.relative_to(base), t))
n = 0
for k, v in seen.items():
    if len(v) > 1 and len({json.dumps(t, sort_keys=True) for _, t in v}) > 1:
        n += 1
        print(k)
        for p, _ in v:
            print("   ", p)
print(f"\n{n} record(s) differ between files; {sum(len(v) > 1 for v in seen.values())} appear in more than one file")
