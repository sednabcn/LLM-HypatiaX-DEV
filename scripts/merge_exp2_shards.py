import json, glob, os, sys, collections
d = "hypatiax/data/results/comparison_results/feynman-tests/exp2"
files = sorted(f for f in glob.glob(f"{d}/protocol_core_noiseless_2*.json") if "pca" not in f and "merged" not in f)
ok = lambda v: isinstance(v,(int,float)) and v==v
latest, hist, top = {}, collections.defaultdict(list), {}
for f in files:
    data = json.load(open(f)); top = {k:v for k,v in data.items() if k!="tests"}
    for rec in data.get("tests", []):
        k = (rec.get("domain","?"), rec.get("description") or rec.get("name") or "?")
        latest[k] = rec; hist[k].append((os.path.basename(f)[-20:-5], rec.get("results",{})))
print(len(files),"shards |",len(latest),"distinct tests |",sum(len(v)>1 for v in hist.values()),"in >1 shard")
mk = sys.argv[1] if len(sys.argv)>1 else None
allkeys = sorted({m for r in latest.values() for m in (r.get("results") or {})})
print("method keys:", allkeys)
if not mk:
    mk = next((m for m in allkeys if "pure" in m.lower() and "llm" in m.lower()), None)
print("Pure LLM key:", mk)
g = lambda res: (res.get(mk) or {}).get("r2", (res.get(mk) or {}).get("test_r2"))
vals = {k: g(r.get("results",{})) for k,r in latest.items()}
for t in (0.99, 0.999999):
    print(f"passes r2>={t}: {sum(ok(v) and v>=t for v in vals.values())}/{len(vals)}")
for k,v in vals.items():
    if not (ok(v) and v>=0.999999):
        print("FAIL", k, v, "| history:", [(s,g(r)) for s,r in hist[k]])
top["tests"] = list(latest.values())
out = f"{d}/protocol_core_noiseless_99999999_999999_merged.json"
json.dump(top, open(out,"w"), indent=1); print("wrote", out)
