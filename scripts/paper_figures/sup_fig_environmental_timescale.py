# FILE: scripts/paper_figures/sup_fig_fitness_vs_timescale_by_selection.py
#
# Generates a comprehensive 1x3 Supplementary Figure. Each panel shows the
# fitness landscape (fitness vs. k_total) for a different selection strength (b_m),
# with individual curves representing different environmental timescales (patch width W).
# This provides a clear, direct view of how the optimal strategy shifts.

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


PROJECT_ROOT = get_project_root()
# Imports are safe after setting the path
from src.config import EXPERIMENTS
from src.io.data_loader import load_aggregated_data


def find_nearest(array, value):
    """Finds the nearest value in a sorted array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def main():
    # 1. --- Configuration and Data Loading ---
    campaign_id = EXPERIMENTS["bet_hedging_final"]["campaign_id"]
    print(f"Generating 3-panel fitness landscape figure from campaign: {campaign_id}")

    df_full = load_aggregated_data(campaign_id, PROJECT_ROOT)
    if df_full.empty:
        sys.exit("Error: Data for campaign is missing or empty.")

    # 2. --- Data Filtering & Processing ---
    # Select three representative selection strengths
    b_m_targets = [0.9, 0.5, 0.2]
    b_m_vals = [find_nearest(df_full["b_m"].unique(), val) for val in b_m_targets]

    # Filter for converged, unbiased switching runs
    phi_val = find_nearest(df_full["phi"].unique(), 0.0)
    df_filtered = df_full[
        (df_full["termination_reason"] == "converged")
        & np.isclose(df_full["phi"], phi_val)
    ].copy()

    # Extract patch width (W)
    df_filtered["patch_width"] = (
        df_filtered["env_definition"].str.extract(r"(\d+)w").astype(float)
    )
    df_plot_base = df_filtered[
        df_filtered["patch_width"].isin([30.0, 60.0, 120.0])
    ].copy()

    if df_plot_base.empty:
        sys.exit("Error: No data available after initial filtering.")

    # 3. --- Visualization Setup ---
    sns.set_theme(style="ticks", context="talk")
    # Create a 1x3 layout with a shared Y-axis for direct comparison
    fig, axes = plt.subplots(
        1, 3, figsize=(24, 8), sharey=True, constrained_layout=True
    )
    fig.suptitle(
        "Fitness Landscapes Across Selection Strengths and Environmental Timescales",
        fontsize=28,
        y=1.05,
    )

    palette = sns.color_palette("rocket_r", n_colors=3)
    panel_titles = [
        "(A) Weak Selection",
        "(B) Medium Selection",
        "(C) Strong Selection",
    ]

    # 4. --- Loop Through Panels and Plot Data ---
    for i, bm_val in enumerate(b_m_vals):
        ax = axes[i]
        df_panel = df_plot_base[np.isclose(df_plot_base["b_m"], bm_val)]

        if df_panel.empty:
            ax.text(
                0.5,
                0.5,
                f"No data for b_m={bm_val}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        sns.lineplot(
            data=df_panel,
            x="k_total",
            y="avg_front_speed",
            hue="patch_width",
            palette=palette,
            lw=4,
            marker="o",
            ms=10,
            ax=ax,
        )

        ax.set_xscale("log")
        ax.set_title(f"{panel_titles[i]} ($b_m = {bm_val:.2f}$)", fontsize=20, pad=15)
        ax.set_xlabel("Switching Rate, $k_{total}$", fontsize=16)
        ax.grid(True, which="both", ls=":", alpha=0.7)

        # Handle legends: show on the first plot, hide on others
        if i == 0:
            ax.legend(title="Patch Width, $W$", fontsize=14)
            ax.set_ylabel("Absolute Fitness (Front Speed)", fontsize=16)
        else:
            ax.get_legend().remove()
            ax.set_ylabel("")  # Remove redundant y-axis label

    # --- Save the Figure ---
    figure_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    output_path = os.path.join(
        figure_dir, "sup_fig_fitness_vs_timescale_by_selection.png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved successfully to: {output_path}")


if __name__ == "__main__":
    main()
