# compare_splits_v2.py
import json
r_v3c = json.loads(open("results_v3c.json").read())
r_pca = json.loads(open("results_pca.json").read())

def get(r, key="Portfolio Sharpe Ratio"):
    return next(x for x in r if x["equation_id"] == key)

c1 = get(r_v3c)["results"]["neural_network"]
c2 = get(r_pca)["results"]["neural_network"]

print("train_r2:", c1["train_r2"], "vs", c2["train_r2"])
print("test_r2: ", c1["test_r2"],  "vs", c2["test_r2"])
print("time_s:   ", c1["time_s"],  "vs", c2["time_s"])

t1, t2 = c1["y_pred_train"], c2["y_pred_train"]
print("train lengths:", len(t1), len(t2))
if len(t1) == len(t2):
    diffs = [abs(a-b) for a,b in zip(t1,t2)]
    print("train max diff:", max(diffs))
