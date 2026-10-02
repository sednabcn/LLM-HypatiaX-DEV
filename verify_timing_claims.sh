#!/bin/bash
# Run from the DEV root. Checks claims removed from the paper because they were not verified here.
python3 - <<'PY'
import glob, json, re
b="hypatiax/data/results/comparison_results/noise-noiseless/noiseless/defi/"
fs=[b+"hypatiax_defi_benchmark_v3_results_seed42.json"]+sorted(glob.glob(b+"multi-seeds/hypatiax_defi_benchmark_v3_results_seed*.json"))
for f in fs:
    r=json.load(open(f)); s=re.search(r"seed(\d+)",f).group(1)
    to={m:sum(bool(x["results"][m].get("timed_out")) for x in r) for m in ("neural_network",)}
    seeds={x["seed"] for x in r}
    llm=[x["results"]["pure_llm"]["time_s"] for x in r]
    print("seed",s,"n",len(r),"seed fields",seeds,"timed_out",to,"pure_llm range %.1f-%.1f"%(min(llm),max(llm)))
PY
