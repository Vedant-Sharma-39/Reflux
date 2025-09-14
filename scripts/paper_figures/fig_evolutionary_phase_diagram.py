# FILE: scripts/paper_figures/fig_evolutionary_phase_diagram.py
#
# Generates an evolutionary phase diagram for phenotypic switching strategies.
# This script analyzes how the optimal switching rate (k_opt) adapts to
# environments with varying mean duration (mean_patch_width) and randomness
# (Fano factor), analogous to Fig. 3 in Skanata & Kussell, PRL (2016).

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# --- Publication Settings ---
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def cm_to_inch(cm):
    return cm / 2.54


# --- End Publication Settings ---


def get_project_root():
    """Dynamically finds the project root directory and adds it to the path."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


PROJECT_ROOT = get_project_root()
from src.config import EXPERIMENTS, PARAM_GRID
from src.io.data_loader import load_aggregated_data


def main():
    """
    Main function to load data, perform analysis, and generate the phase diagram.
    """
    campaign_id = EXPERIMENTS["evolutionary_phase_diagram"]["campaign_id"]
    print(f"Generating Evolutionary Phase Diagram from campaign: {campaign_id}")

    df = load_aggregated_data(campaign_id, PROJECT_ROOT)
    if df.empty:
        print(
            f"Error: Data for campaign '{campaign_id}' is empty or not found.",
            file=sys.stderr,
        )
        sys.exit(1)

    figure_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    # --- CHANGE: Output filenames ---
    output_path_pdf = os.path.join(figure_dir, "fig_evolutionary_phase_diagram.pdf")
    output_path_eps = os.path.join(figure_dir, "fig_evolutionary_phase_diagram.eps")

    df_filtered = df[df["termination_reason"] == "converged"].copy()

    env_params = df_filtered["env_definition"].str.extract(
        r"gamma_mean_(\d+\.?\d*)_fano_(\d+\.?\d*)"
    )
    env_params.columns = ["mean_patch_width", "fano_factor"]

    df_processed = pd.concat([df_filtered, env_params], axis=1)

    df_processed.dropna(subset=["mean_patch_width", "fano_factor"], inplace=True)
    df_processed["mean_patch_width"] = df_processed["mean_patch_width"].astype(float)
    df_processed["fano_factor"] = df_processed["fano_factor"].astype(float)

    if df_processed.empty:
        print(
            "Error: No valid data found after filtering for converged, Gamma-distributed environments.",
            file=sys.stderr,
        )
        sys.exit(1)

    optimal_indices = df_processed.groupby(["mean_patch_width", "fano_factor"])[
        "avg_front_speed"
    ].idxmax()
    df_optimal = df_processed.loc[optimal_indices].copy()

    df_optimal["log10_k_opt"] = np.log10(df_optimal["k_total"])

    try:
        pivot_k_opt = df_optimal.pivot_table(
            index="mean_patch_width", columns="fano_factor", values="log10_k_opt"
        ).sort_index(ascending=False)
    except Exception as e:
        print(f"Error creating pivot table: {e}", file=sys.stderr)
        print(
            "Check if you have sufficient variation in environmental parameters in your data.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- CHANGE: Plotting setup for publication ---
    sns.set_theme(style="ticks", context="paper")
    fig, ax = plt.subplots(figsize=(cm_to_inch(11.4), cm_to_inch(10)))

    cbar_kws = {"label": r"Optimal Switching Rate, $\log_{10}(k_{opt})$"}
    sns.heatmap(
        pivot_k_opt,
        ax=ax,
        cmap="viridis",
        linewidths=0.5,
        annot=True,
        fmt=".1f",
        # --- CHANGE: Font sizes ---
        annot_kws={"size": 6, "color": "white", "weight": "bold"},
        cbar_kws=cbar_kws,
    )
    ax.figure.axes[-1].yaxis.label.set_size(8)

    k_scan = PARAM_GRID["k_total_final_log"]
    slow_threshold = np.log10(min(k_scan)) + 0.5
    fast_threshold = np.log10(max(k_scan)) - 0.5

    X, Y = np.meshgrid(pivot_k_opt.columns, pivot_k_opt.index)

    contours = ax.contour(
        X,
        Y,
        pivot_k_opt,
        levels=[slow_threshold, fast_threshold],
        colors="white",
        linewidths=2.5,
        linestyles="--",
    )
    # --- CHANGE: Font sizes ---
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f")

    ax.text(
        3,
        200,
        "Slow Switching\n(Constitutive-like)",
        color="white",
        ha="center",
        va="center",
        fontsize=8,
        weight="bold",
    )
    ax.text(
        40,
        200,
        "Adaptive\nSwitching",
        color="white",
        ha="center",
        va="center",
        fontsize=8,
        weight="bold",
    )
    ax.text(
        40,
        50,
        "Fast Switching\n(Memoryless-like)",
        color="white",
        ha="center",
        va="center",
        fontsize=8,
        weight="bold",
    )

    # --- CHANGE: Font sizes ---
    ax.set_title(
        "Evolutionary Phase Diagram of Switching Strategies", fontsize=12, pad=15
    )
    ax.set_xlabel("Environmental Randomness (Fano Factor)", fontsize=8)
    ax.set_ylabel("Mean Environmental Duration (Patch Width)", fontsize=8)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)

    # --- CHANGE: Save to PDF and EPS ---
    plt.savefig(output_path_pdf, bbox_inches="tight")
    plt.savefig(output_path_eps, bbox_inches="tight")
    print(
        f"\nEvolutionary Phase Diagram saved successfully to: {output_path_pdf} and {output_path_eps}"
    )


if __name__ == "__main__":
    main()
