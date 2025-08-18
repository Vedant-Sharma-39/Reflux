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


def get_project_root():
    """Dynamically finds the project root directory and adds it to the path."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


# --- Main Setup ---
PROJECT_ROOT = get_project_root()
from src.config import EXPERIMENTS, PARAM_GRID
from src.io.data_loader import load_aggregated_data


def main():
    """
    Main function to load data, perform analysis, and generate the phase diagram.
    """
    # 1. --- Configuration and Data Loading ---
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
    output_path = os.path.join(figure_dir, "fig_evolutionary_phase_diagram.png")

    # 2. --- Data Pre-processing and Feature Engineering ---
    #
    # CRITICAL STEP: Filter for successful, converged runs only.
    # This removes simulations that timed out ('max_cycles_reached') and did not
    # produce a stable, meaningful fitness measurement.
    df_filtered = df[df["termination_reason"] == "converged"].copy()

    # Extract environmental parameters from the 'env_definition' string.
    # This regex captures the mean and fano factor from names like 'gamma_mean_60_fano_10'.
    env_params = df_filtered["env_definition"].str.extract(
        r"gamma_mean_(\d+\.?\d*)_fano_(\d+\.?\d*)"
    )
    env_params.columns = ["mean_patch_width", "fano_factor"]

    df_processed = pd.concat([df_filtered, env_params], axis=1)

    # Drop rows that don't match the gamma distribution format (e.g., periodic controls like 'symmetric_refuge_60w')
    # and convert the extracted parameters to numeric types for analysis.
    df_processed.dropna(subset=["mean_patch_width", "fano_factor"], inplace=True)
    df_processed["mean_patch_width"] = df_processed["mean_patch_width"].astype(float)
    df_processed["fano_factor"] = df_processed["fano_factor"].astype(float)

    if df_processed.empty:
        print(
            "Error: No valid data found after filtering for converged, Gamma-distributed environments.",
            file=sys.stderr,
        )
        print(
            "This could mean your simulations need to run longer (increase 'max_cycles')."
        )
        sys.exit(1)

    # 3. --- Core Analysis: Find the Optimal Strategy for Each Environment ---
    # For each unique environment (mean_patch_width, fano_factor), find the
    # row (and thus the k_total) that corresponds to the maximum average fitness (avg_front_speed).
    optimal_indices = df_processed.groupby(["mean_patch_width", "fano_factor"])[
        "avg_front_speed"
    ].idxmax()
    df_optimal = df_processed.loc[optimal_indices].copy()

    # Use log10 of the optimal k for better visualization, as k spans orders of magnitude.
    df_optimal["log10_k_opt"] = np.log10(df_optimal["k_total"])

    # 4. --- Prepare Data for Plotting ---
    # Create a pivot table to structure the data into a 2D grid for the heatmap.
    try:
        pivot_k_opt = df_optimal.pivot_table(
            index="mean_patch_width", columns="fano_factor", values="log10_k_opt"
        ).sort_index(
            ascending=False
        )  # y-axis should have large values at the bottom
    except Exception as e:
        print(f"Error creating pivot table: {e}", file=sys.stderr)
        print(
            "Check if you have sufficient variation in environmental parameters in your data.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 5. --- Plotting the Phase Diagram ---
    sns.set_theme(style="ticks", context="talk")
    fig, ax = plt.subplots(figsize=(14, 11))

    # Use a heatmap to represent the optimal switching rate in the phase space.
    sns.heatmap(
        pivot_k_opt,
        ax=ax,
        cmap="viridis",
        linewidths=0.5,
        annot=True,  # Annotate each cell with its value
        fmt=".1f",  # Format annotation to one decimal place
        annot_kws={"size": 12, "color": "white", "weight": "bold"},
        cbar_kws={"label": r"Optimal Switching Rate, $\log_{10}(k_{opt})$"},
    )

    # --- Add Contour Lines to Delineate Phases ---
    # Define thresholds based on the range of k values scanned in the simulation.
    k_scan = PARAM_GRID["k_total_final_log"]
    # Define phase boundaries slightly inside the min/max k values.
    slow_threshold = np.log10(min(k_scan)) + 0.5
    fast_threshold = np.log10(max(k_scan)) - 0.5

    # Create grid coordinates for the contour plot.
    X, Y = np.meshgrid(pivot_k_opt.columns, pivot_k_opt.index)

    # Draw contour lines.
    contours = ax.contour(
        X,
        Y,
        pivot_k_opt,
        levels=[slow_threshold, fast_threshold],
        colors="white",
        linewidths=3.5,
        linestyles="--",
    )
    ax.clabel(contours, inline=True, fontsize=14, fmt="%.1f")

    # --- Add Text Annotations for the Phases ---
    # These coordinates may need slight tuning depending on your exact data range.
    ax.text(
        3,
        200,
        "Slow Switching\n(Constitutive-like)",
        color="white",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
    )
    ax.text(
        40,
        200,
        "Adaptive\nSwitching",
        color="white",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
    )
    ax.text(
        40,
        50,
        "Fast Switching\n(Memoryless-like)",
        color="white",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
    )

    # --- Final Touches ---
    ax.set_title(
        "Evolutionary Phase Diagram of Switching Strategies", fontsize=24, pad=20
    )
    ax.set_xlabel("Environmental Randomness (Fano Factor)", fontsize=18)
    ax.set_ylabel("Mean Environmental Duration (Patch Width)", fontsize=18)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nEvolutionary Phase Diagram saved successfully to: {output_path}")


if __name__ == "__main__":
    main()
