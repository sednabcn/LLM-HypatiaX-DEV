import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

d = json.load(open("july2026_results_summary.json"))
sweep = d["noise_sweep"]
noise_keys = sorted(sweep.keys(), key=float)
noise_pct = [float(k) * 100 for k in noise_keys]

methods = ["PureLLM", "HLLMNN", "SymLLM", "EHSDeFi", "HDSv50_2", "ImpNN"]
colors = {
    "PureLLM": "#7f7f7f",
    "HLLMNN": "#1b6f3a",
    "SymLLM": "#2b6ca3",
    "EHSDeFi": "#c96a1f",
    "HDSv50_2": "#8e44ad",
    "ImpNN": "#c0392b",
}
markers = {
    "PureLLM": "o", "HLLMNN": "s", "SymLLM": "^",
    "EHSDeFi": "D", "HDSv50_2": "v", "ImpNN": "x",
}
labels = {
    "PureLLM": "PureLLM (M1)", "HLLMNN": "HLLMNN (M4)", "SymLLM": "SymLLM (M5)",
    "EHSDeFi": "EHSDeFi (M3)", "HDSv50_2": "HDSv50_2 (M6)", "ImpNN": "ImpNN (M2)",
}

fig, ax = plt.subplots(figsize=(6.3, 4.0))
for m in methods:
    ys = [sweep[k][m]["recovery_rate"] for k in noise_keys]
    ax.plot(noise_pct, ys, marker=markers[m], color=colors[m], linewidth=1.6,
             markersize=6, label=labels[m])

ax.set_xlabel(r"Noise level $\sigma$ (% of std($y$))", fontsize=11)
ax.set_ylabel(r"Recovery rate at $R^2 \geq 0.999999$ (%)", fontsize=11)
ax.set_ylim(-5, 105)
ax.set_xlim(-0.05, 1.05)
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9, ncol=2)
ax.set_title(r"Recovery rate vs. noise level (July 2026 rerun, $n{=}200$, 30 equations)",
             fontsize=10.5)
fig.tight_layout()
fig.savefig("recovery_vs_noise.pdf")
print("saved recovery_vs_noise.pdf")
