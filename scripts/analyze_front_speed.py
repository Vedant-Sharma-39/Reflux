# FILE: scripts/analyze_front_speed.py
#
# [REWRITTEN FOR STATE-SPACE ANALYSIS]
# This script generates a detailed visualization of the front speed as a function
# of the system's internal state variables: the average mutant fraction (rho_M)
# and the selection coefficient (s).
#
# It produces a multi-panel plot, with one panel for each switching bias (phi),
# allowing for a clear analysis of how the Speed = f(rho_M, s) relationship
# changes with the underlying switching strategy.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, argparse

# --- Robust Path Setup & Data Aggregation ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(project_root, "src"))
    from config import EXPERIMENTS
    from data_utils import aggregate_data_cached
except (NameError, ImportError) as e:
    sys.exit(f"FATAL: Could not import configuration or helpers. Error: {e}")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {"font.size": 14, "axes.labelsize": 16, "axes.titlesize": 18, "legend.fontsize": 12}
)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze front speed as a function of internal state (rho_M, s)."
    )
    parser.add_argument(
        "experiment_name", default="exp1_front_speed_deleterious_scan", nargs="?"
    )
    args = parser.parse_args()

    config = EXPERIMENTS[args.experiment_name]
    CAMPAIGN_ID = config["CAMPAIGN_ID"]
    ANALYSIS_DIR = os.path.join(project_root, "data", CAMPAIGN_ID, "analysis")
    FIGS_DIR = os.path.join(
        project_root, "figures", CAMPAIGN_ID, "speed_vs_state_analysis"
    )
    os.makedirs(FIGS_DIR, exist_ok=True)

    # --- Load and Process Data ---
    df_raw = aggregate_data_cached(CAMPAIGN_ID, project_root)
    if df_raw is None or df_raw.empty:
        sys.exit("FATAL: No data found.")

    df_raw["s"] = df_raw["b_m"] - 1.0

    group_keys = ["s", "phi", "k_total"]
    df_avg = (
        df_raw.groupby(group_keys)
        .agg(
            avg_front_speed=("avg_front_speed", "mean"), avg_rho_M=("avg_rho_M", "mean")
        )
        .reset_index()
        .dropna()
    )

    print(f"--- Starting State-Space Analysis for Campaign: {CAMPAIGN_ID} ---")
    print(f"Figures will be saved to: {FIGS_DIR}")

    # ==========================================================================
    # FIGURE: Speed vs. Composition, colored by Selection, faceted by Bias
    # ==========================================================================

    # --- [THE FIX] ---
    # Use the simpler and more direct g.map() method.

    g = sns.FacetGrid(
        df_avg, col="phi", col_wrap=3, height=7, aspect=1.1, sharex=True, sharey=True
    )

    # .map() will pass the columns 'avg_rho_M', 'avg_front_speed', and 's'
    # as positional arguments to plt.scatter.
    # We use a reversed colormap for better intuition (yellow = neutral).
    g.map(
        plt.scatter,
        "avg_rho_M",
        "avg_front_speed",
        "s",
        cmap="viridis_r",
        s=40,
        alpha=0.8,
        edgecolor="none",
    )

    # --- Finalize the Plot ---
    g.set_axis_labels("Average Mutant Fraction ($\\rho_M$)", "Average Front Speed")
    g.set_titles("Switching Bias ($\\phi$) = {col_name}")

    # Add a single, shared colorbar for the entire figure
    fig = g.fig
    fig.subplots_adjust(right=0.88)
    s_vals = df_avg["s"].unique()
    norm = plt.Normalize(vmin=s_vals.min(), vmax=s_vals.max())
    sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Selection Coefficient (s)", size=14)

    fig.suptitle(
        "Front Speed as a Function of Internal State (Composition & Selection)",
        fontsize=22,
        y=1.02,
    )

    plot_filename = "Fig_Speed_vs_Composition_and_Selection.png"
    plt.savefig(os.path.join(FIGS_DIR, plot_filename), dpi=300)
    plt.close(fig)

    print(
        f"\nState-space analysis plot saved to: {os.path.join(FIGS_DIR, plot_filename)}"
    )


if __name__ == "__main__":
    main()
