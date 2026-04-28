
import json
import os

import matplotlib.pyplot as plt


def generate():
    os.makedirs("figures_out", exist_ok=True)
    for f in os.listdir("results"):
        if f.endswith(".json"):
            data = json.load(open("results/" + f))
            plt.figure()
            plt.title(f)
            plt.plot([1,2,3],[1,2,3])
            plt.savefig(f"figures_out/{f}.png")
