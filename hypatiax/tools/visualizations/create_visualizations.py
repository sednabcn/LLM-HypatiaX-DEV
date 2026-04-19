#!/usr/bin/env python3
"""
Visualization Only Script - Reads Merged Data
==============================================
Creates publication-quality plots from all_systems_merged.json

Usage:
    python create_visualizations.py

Requires:
    - all_systems_merged.json (created by unified_analysis_script.py)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication quality
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "sans-serif"
sns.set_style("whitegrid")


def load_merged_data():
    """Load the merged JSON file."""
    merged_file = Path.cwd() / "all_systems_merged.json"

    if not merged_file.exists():
        print(f"❌ File not found: {merged_file}")
        print("\nRun unified_analysis_script.py first to create the merged data!")
        return None

    with open(merged_file, "r") as f:
        data = json.load(f)

    print(f"✅ Loaded: {merged_file.name}")
    print(f"   Total tests: {data.get('total_tests', 'unknown')}")
    return data


def extract_plot_data(data):
    """Extract data for plotting."""
    method_mapping = {
        "Hybrid System v40": "Hybrid_v40",
        "Pure LLM (Enhanced)": "Pure_LLM",
        "Pure LLM (Basic)": "Pure_LLM_Basic",
        "Neural Network": "Neural_Network",
        "LLM+NN Ensemble (Smart)": "Ensemble_Smart",
    }

    systems = {
        "Hybrid_v40": [],
        "Pure_LLM": [],
        "Neural_Network": [],
        "Ensemble_Smart": [],
    }

    for test in data.get("tests", []):
        for method_orig, system_key in method_mapping.items():
            if system_key not in systems:
                continue

            if method_orig in test.get("results", {}):
                result = test["results"][method_orig]

                if "extrapolation_errors" in result:
                    errors = result["extrapolation_errors"]
                    medium_error = errors.get("medium", np.nan)

                    if np.isfinite(medium_error) and 0 < medium_error < 1e6:
                        systems[system_key].append(medium_error)

    # Filter to only systems with data
    systems = {k: v for k, v in systems.items() if len(v) > 0}

    print(f"\nExtracted data:")
    for name, values in systems.items():
        print(
            f"  • {name:20s}: {len(values):2d} points (median: {np.median(values):.1f}%)"
        )

    return systems


def create_boxplot(systems, output_dir):
    """Create simple boxplot comparison."""
    print("\n📊 Creating boxplot...")

    # Prepare data
    system_info = [
        ("Hybrid_v40", "Hybrid\nSystem v40", "darkblue"),
        ("Pure_LLM", "Pure\nLLM", "green"),
        ("Ensemble_Smart", "Ensemble\nSmart", "purple"),
        ("Neural_Network", "Neural\nNetwork", "red"),
    ]

    data_to_plot = []
    labels = []
    colors = []

    for key, label, color in system_info:
        if key in systems and len(systems[key]) > 0:
            data_to_plot.append(systems[key])
            labels.append(label)
            colors.append(color)

    if len(data_to_plot) == 0:
        print("❌ No data to plot!")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = list(range(1, len(data_to_plot) + 1))

    # Box plots
    bp = ax.boxplot(
        data_to_plot,
        positions=positions,
        patch_artist=True,
        showfliers=True,
        widths=0.6,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    # Color boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Labels and styling
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Extrapolation Error (%)\nMedium Regime (2×)", fontsize=12)
    ax.set_title(
        "Extrapolation Performance Comparison", fontsize=14, fontweight="bold", pad=15
    )
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y", which="both")

    # Add statistics labels
    for pos, errors in zip(positions, data_to_plot):
        median = np.median(errors)
        q1 = np.percentile(errors, 25)
        q3 = np.percentile(errors, 75)

        # Median value above box
        ax.text(
            pos,
            median * 1.15,
            f"{median:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

        # N count below
        ax.text(
            pos,
            ax.get_ylim()[0] * 1.5,
            f"n={len(errors)}",
            ha="center",
            va="top",
            fontsize=8,
            style="italic",
        )

    # Reference lines
    ax.axhline(
        y=10,
        color="green",
        linestyle="--",
        alpha=0.4,
        linewidth=1,
        label="10% (excellent)",
    )
    ax.axhline(
        y=100,
        color="orange",
        linestyle="--",
        alpha=0.4,
        linewidth=1,
        label="100% (2× error)",
    )

    ax.legend(loc="upper right", fontsize=9)

    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.15)

    # Save
    output_dir.mkdir(exist_ok=True)
    pdf_file = output_dir / "figure_boxplot_comparison.pdf"
    png_file = output_dir / "figure_boxplot_comparison.png"

    plt.savefig(pdf_file, bbox_inches="tight")
    plt.savefig(png_file, bbox_inches="tight", dpi=200)

    print(f"✅ Saved: {pdf_file}")
    print(f"✅ Saved: {png_file}")

    plt.close()


def create_barplot(systems, output_dir):
    """Create bar plot with error bars."""
    print("\n📊 Creating barplot...")

    system_info = [
        ("Hybrid_v40", "Hybrid v40", "darkblue"),
        ("Pure_LLM", "Pure LLM", "green"),
        ("Ensemble_Smart", "Ensemble Smart", "purple"),
        ("Neural_Network", "Neural Net", "red"),
    ]

    names = []
    medians = []
    q1s = []
    q3s = []
    colors_list = []

    for key, label, color in system_info:
        if key in systems and len(systems[key]) > 0:
            data = systems[key]
            names.append(label)
            medians.append(np.median(data))
            q1s.append(np.percentile(data, 25))
            q3s.append(np.percentile(data, 75))
            colors_list.append(color)

    if len(names) == 0:
        print("❌ No data to plot!")
        return

    # Calculate error bars (from median)
    lower_err = [m - q for m, q in zip(medians, q1s)]
    upper_err = [q - m for m, q in zip(q3s, medians)]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(names))

    bars = ax.bar(
        x_pos,
        medians,
        yerr=[lower_err, upper_err],
        color=colors_list,
        alpha=0.7,
        capsize=5,
        error_kw={"linewidth": 2},
    )

    # Labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Median Extrapolation Error (%)\nMedium Regime (2×)", fontsize=12)
    ax.set_title(
        "Median Performance with Interquartile Range",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # Value labels on bars
    for i, (bar, med) in enumerate(zip(bars, medians)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height * 1.15,
            f"{med:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.12)

    # Save
    pdf_file = output_dir / "figure_barplot_comparison.pdf"
    png_file = output_dir / "figure_barplot_comparison.png"

    plt.savefig(pdf_file, bbox_inches="tight")
    plt.savefig(png_file, bbox_inches="tight", dpi=200)

    print(f"✅ Saved: {pdf_file}")
    print(f"✅ Saved: {png_file}")

    plt.close()


def create_scatter_plot(data, output_dir):
    """Create R² vs Extrapolation scatter plot."""
    print("\n📊 Creating scatter plot...")

    method_mapping = {
        "Hybrid System v40": ("Hybrid v40", "darkblue", "o"),
        "Pure LLM (Enhanced)": ("Pure LLM", "green", "s"),
        "Neural Network": ("Neural Net", "red", "^"),
        "LLM+NN Ensemble (Smart)": ("Ensemble", "purple", "d"),
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for method_orig, (label, color, marker) in method_mapping.items():
        r2_values = []
        extrap_values = []

        for test in data.get("tests", []):
            if method_orig in test.get("results", {}):
                result = test["results"][method_orig]

                r2 = result.get("r2", np.nan)
                if "extrapolation_errors" in result:
                    extrap = result["extrapolation_errors"].get("medium", np.nan)

                    if (
                        np.isfinite(r2)
                        and np.isfinite(extrap)
                        and -1 < r2 < 1.1
                        and extrap < 1e6
                    ):
                        r2_values.append(r2)
                        extrap_values.append(extrap)

        if len(r2_values) > 0:
            ax.scatter(
                r2_values,
                extrap_values,
                c=color,
                marker=marker,
                s=100,
                alpha=0.7,
                label=f"{label} (n={len(r2_values)})",
                edgecolors="black",
            )

    ax.set_xlabel("R² Score (Interpolation)", fontsize=12)
    ax.set_ylabel("Extrapolation Error (%)\nMedium Regime (2×)", fontsize=12)
    ax.set_title(
        "Interpolation vs Extrapolation Performance",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.12)

    # Save
    pdf_file = output_dir / "figure_scatter_r2_vs_extrap.pdf"
    png_file = output_dir / "figure_scatter_r2_vs_extrap.png"

    plt.savefig(pdf_file, bbox_inches="tight")
    plt.savefig(png_file, bbox_inches="tight", dpi=200)

    print(f"✅ Saved: {pdf_file}")
    print(f"✅ Saved: {png_file}")

    plt.close()


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("VISUALIZATION SCRIPT - CREATING PUBLICATION FIGURES")
    print("=" * 80)

    # Load data
    data = load_merged_data()
    if data is None:
        return

    # Extract plot data
    systems = extract_plot_data(data)

    if len(systems) == 0:
        print("\n❌ No valid data found for plotting!")
        return

    # Create output directory
    output_dir = Path.cwd() / "figures"
    output_dir.mkdir(exist_ok=True)

    # Generate all plots
    try:
        create_boxplot(systems, output_dir)
    except Exception as e:
        print(f"❌ Boxplot failed: {e}")

    try:
        create_barplot(systems, output_dir)
    except Exception as e:
        print(f"❌ Barplot failed: {e}")

    try:
        create_scatter_plot(data, output_dir)
    except Exception as e:
        print(f"❌ Scatter plot failed: {e}")

    print("\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 All figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
