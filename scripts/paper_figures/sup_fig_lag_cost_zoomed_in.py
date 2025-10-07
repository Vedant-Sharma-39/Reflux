# FILE: scripts/paper_figures/sup_fig_lag_cost_zoomed_in.py
#
# Generates a diagnostic 1x3 panel figure to clearly visualize the fitness
# cost of lag duration (τ) at different fixed switching rates (k).
# Each panel's Y-axis is independently scaled to zoom in on the trend.

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Project Setup ---

def get_project_root() -> Path:
    """Dynamically finds the project root directory and adds it to the path."""
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root

PROJECT_ROOT = get_project_root()
from src.io.data_loader import load_aggregated_data
from src.config import EXPERIMENTS

def find_nearest(array: np.ndarray, value: float) -> float:
    """Finds the nearest value in a sorted array."""
    idx = np.abs(np.asarray(array) - value).argmin()
    return array[idx]

# --- Main Analysis and Plotting ---

def main():
    # 1. --- Configuration and Data Loading ---
    campaign_id = EXPERIMENTS["lag_vs_selection_definitive"]["campaign_id"]
    print(f"Loading data from definitive campaign: '{campaign_id}'")
    df_raw = load_aggregated_data(campaign_id, PROJECT_ROOT)

    if df_raw.empty:
        sys.exit(f"Error: Data for campaign '{campaign_id}' is missing.")

    figure_dir = PROJECT_ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    output_path = figure_dir / "sup_fig_lag_cost_zoomed_in.pdf"

    # 2. --- Data Filtering and Preparation ---
    df_converged = df_raw[df_raw["termination_reason"] == "converged"].copy()
    df_analysis_base = df_converged[np.isclose(df_converged["phi"], 0.0)].copy()

    env_for_plots = "symmetric_refuge_60w"
    b_m_all = np.sort(df_analysis_base["b_m"].unique())
    b_m_for_panels = find_nearest(b_m_all, 0.50)

    df_data_base = df_analysis_base[
        (df_analysis_base["env_definition"] == env_for_plots) &
        np.isclose(df_analysis_base["b_m"], b_m_for_panels)
    ].copy()

    k_all = np.sort(df_data_base["k_total"].unique())
    k_vals_to_plot = [
        find_nearest(k_all, 0.1),
        find_nearest(k_all, 1.0),
        find_nearest(k_all, 10.0),
    ]

    # 3. --- Plotting ---
    print("Generating zoomed-in 1x3 panel figure...")
    sns.set_theme(context="paper", style="ticks")
    
    # Create a 1x3 layout. sharex is useful, but critically, sharey=False
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"The Fitness Cost of Lag Duration (τ) is Revealed by Fixing Switching Rate (k)\n($b_m$={b_m_for_panels:.2f}, Environment: {env_for_plots})",
        fontsize=14
    )

    panel_colors = sns.color_palette("viridis", n_colors=len(k_vals_to_plot))

    for i, (k_val, ax) in enumerate(zip(k_vals_to_plot, axes)):
        df_panel = df_data_base[np.isclose(df_data_base["k_total"], k_val)]
        
        if df_panel.empty:
            ax.text(0.5, 0.5, f"No data for k={k_val}", ha='center', va='center')
            continue

        sns.lineplot(
            data=df_panel,
            x="switching_lag_duration",
            y="avg_front_speed",
            marker='o',
            ms=5,
            lw=2,
            color=panel_colors[i],
            ci='sd', # Show standard deviation as confidence interval
            ax=ax
        )

        ax.set_title(f"Slow Switching ($k = {k_val:.1f}$)" if i == 0 else
                     f"Medium Switching ($k = {k_val:.1f}$)" if i == 1 else
                     f"Fast Switching ($k = {k_val:.1f}$)")
        
        ax.set_xscale('log')
        ax.grid(True, which='both', linestyle=':')
        
        # Central x-axis label
        if i == 1:
            ax.set_xlabel("Switching Lag Duration (τ)")
        else:
            ax.set_xlabel("")

        # Leftmost y-axis label
        if i == 0:
            ax.set_ylabel("Mean Fitness (Front Speed)")
        else:
            ax.set_ylabel("")

    # 4. --- Save the Final Figure ---
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Diagnostic figure showing the true cost of lag saved to: {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()