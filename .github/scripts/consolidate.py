import os, json, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()

all_results = []

for root, _, files in os.walk(args.input_dir):
    for file in files:
        if file.endswith(".json"):
            with open(os.path.join(root, file)) as f:
                all_results.append(json.load(f))

final = {
    "num_workers": len(all_results),
    "results": all_results,
    "avg_metric": sum(r["metric"] for r in all_results) / len(all_results)
}

with open(args.output, "w") as f:
    json.dump(final, f, indent=2)
