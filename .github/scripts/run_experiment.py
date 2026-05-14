import argparse, json, random

parser = argparse.ArgumentParser()
parser.add_argument("--worker", type=int)
parser.add_argument("--output", type=str)
args = parser.parse_args()

result = {
    "worker": args.worker,
    "metric": random.random()
}

with open(args.output, "w") as f:
    json.dump(result, f)
