
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_CSV = "hypatiax/data/figures/instability_extrapolation.csv"
OUTPUT_FIG = "fig_instability_vs_extrapolation.png"

sns.set(style="whitegrid", context="talk")


def classify_regime(row):
    ii = row["II"]
    mean = row["mean"]

    if ii == 0 and mean > 0.99:
        return "A-Symbolic"
    elif ii < 0.05:
        return "B-Biased"
    elif ii < 0.10:
        return "C-Marginal"
    else:
        return "C-Collapse"


def main():
    df = pd.read_csv(INPUT_CSV)

    # EXPECT: you add this column externally or compute it
    if "extrapolation_r2" not in df.columns:
        raise ValueError("CSV must contain 'extrapolation_r2' column")

    df["regime"] = df.apply(classify_regime, axis=1)

    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        data=df,
        x="II",
        y="extrapolation_r2",
        hue="regime",
        style="regime",
        s=100
    )

    plt.xlabel("Instability Index (II)")
    plt.ylabel("Extrapolation $R^2$")
    plt.title("Instability vs Extrapolation Performance")

    # Decision boundary lines (important!)
    plt.axvline(0.05, linestyle="--", alpha=0.5)
    plt.axvline(0.10, linestyle="--", alpha=0.5)

    plt.axhline(0.9, linestyle="--", alpha=0.5)

    plt.legend(title="Regime")

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=300)
    plt.close()

    print(f"✅ Saved: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
