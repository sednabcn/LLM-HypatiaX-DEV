# quick_split_diff.py
import json
r_v3c = json.loads(open("results_v3c.json").read())
r_pca = json.loads(open("results_pca.json").read())

def get(r, key="Portfolio Sharpe Ratio"):
    return next(x for x in r if x["equation_id"] == key)

c1 = get(r_v3c)["results"]["neural_network"]["y_pred_test"]
c2 = get(r_pca)["results"]["neural_network"]["y_pred_test"]

print("lengths:", len(c1), len(c2))
print("identical:", c1 == c2)
print("max abs diff (if same length):",
      max(abs(a-b) for a,b in zip(c1,c2)) if len(c1)==len(c2) else "N/A - different test sets")
