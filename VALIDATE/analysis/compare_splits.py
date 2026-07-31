# compare_splits.py
import json
r_v3c = json.loads(open("results_v3c.json").read())
r_pca = json.loads(open("results_pca.json").read())

def get(r, key="Portfolio Sharpe Ratio"):
    return next(x for x in r if x["equation_id"] == key)

# if y_test (ground truth) values are logged per-case, compare them directly —
# same order + same values = same partition was used
c1 = get(r_v3c)["results"]
c2 = get(r_pca)["results"]

# check whatever ground-truth/test-target field each JSON actually carries
print(json.dumps({k: type(v).__name__ for k, v in c1.get("neural_network", {}).items()}, indent=2))
