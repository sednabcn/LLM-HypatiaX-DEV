"""Shared loader for protocol_core_*.json shards (real schema, from regenerate_tables.py):
   file["protocol"]["noise_level"], file["protocol"]["threshold"],
   file["tests"][i]["description"], file["tests"][i]["results"][method] -> {r2, metadata{decision,...}}"""
import glob, json, os

M3_PREFIX, M4_PREFIX = "EnhancedHybridSystemDeFi", "HybridSystemLLMNN"

def which(method):
    if method.startswith(M3_PREFIX): return "M3"
    if method.startswith(M4_PREFIX): return "M4"
    return None

def load(d, pattern="protocol_core_*.json"):
    rows, seen = [], set()
    files = sorted(glob.glob(os.path.join(d, pattern)))
    if not files: raise SystemExit(f"no files match {d}/{pattern}")
    for f in files:
        j = json.load(open(f)); p = j["protocol"]
        for t in j["tests"]:
            for method, rec in t.get("results", {}).items():
                key = (p["noise_level"], t["description"], method)
                if key in seen: raise SystemExit(f"duplicate record {key} in {f}")
                seen.add(key)
                r2 = rec.get("r2")
                md = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
                rows.append(dict(noise=p["noise_level"], thr=p["threshold"], method=method,
                                 which=which(method), test=t["description"], file=os.path.basename(f),
                                 r2=r2 if isinstance(r2, (int, float)) else float("-inf"),
                                 decision=md.get("decision"), nn_applied=md.get("nn_applied")))
    return rows

def inspect(d, pattern="protocol_core_*.json"):
    f = sorted(glob.glob(os.path.join(d, pattern)))[0]; j = json.load(open(f))
    print("file:", f); print("protocol:", j["protocol"]); print("tests:", len(j["tests"]))
    t = j["tests"][0]; print("test keys:", list(t)); print("methods:", list(t["results"]))
    m = next(iter(t["results"].values())); print("record keys:", list(m))
