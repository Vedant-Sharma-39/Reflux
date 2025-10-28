# FILE: scripts/paper_figures/fig_transient_lag_analysis.py
#
# Generates the definitive 1x2 panel figure for journal submission.
# This version focuses on the fitness landscape (Panel A) and the resulting
# fitness cost at fixed switching rates (Panel B).

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


# --- Helper Functions ---

def find_nearest(array: np.ndarray, value: float) -> float:
    """Finds the nearest value in a sorted array."""
    idx = np.abs(np.asarray(array) - value).argmin()
    return array[idx]

# This helper function is no longer needed for the new Panel B
# def find_robust_optimum(group: pd.DataFrame) -> pd.Series: ...


# --- Main Analysis and Plotting ---

def main():
    # 1. --- Configuration and Data Loading ---
    campaign_id = EXPERIMENTS["lag_vs_environment_scan"]["campaign_id"]
    print(f"Loading data from definitive campaign: '{campaign_id}'")
    df_raw = load_aggregated_data(campaign_id, PROJECT_ROOT)

    if df_raw.empty:
        sys.exit(f"Error: Data for campaign '{campaign_id}' is missing.")

    figure_dir = PROJECT_ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    output_path = figure_dir / "fig_switching_lag.pdf"

    # 2. --- Data Filtering and Preparation ---
    df_converged = df_raw[df_raw["termination_reason"] == "converged"].copy()
    df_analysis_base = df_converged[np.isclose(df_converged["phi"], 0.0)].copy()

    env_for_plots = "symmetric_refuge_60w"
    b_m_all = np.sort(df_analysis_base["b_m"].unique())

    # --- Data for Panel A ---
    # Panel A shows the landscape for a single b_m value
    b_m_for_panel_A = find_nearest(b_m_all, 0.50)
    df_panel_A_data = df_analysis_base[
        (df_analysis_base["env_definition"] == env_for_plots) &
        np.isclose(df_analysis_base["b_m"], b_m_for_panel_A)
    ].copy()
    
    # --- Data for Panel B ---
    # Panel B shows max fitness vs. lag for *different* b_m values.
    # We must start from the base data *before* filtering by b_m.
    df_panel_B_base = df_analysis_base[
        (df_analysis_base["env_definition"] == env_for_plots)
    ].copy()

    # Select the b_m values shown in the target legend
    b_m_vals_for_lines = [find_nearest(b_m_all, v) for v in [0.20, 0.50, 0.70, 0.90, 1.00]]
    
    # Filter for only these b_m values
    df_panel_B_filtered = df_panel_B_base[
        df_panel_B_base["b_m"].isin(b_m_vals_for_lines)
    ].copy()

    # For each (b_m, switching_lag_duration) pair, find the MAX fitness
    # (this implicitly finds the optimal k_total for that point)
    df_panel_B_data = df_panel_B_filtered.groupby(
        ["b_m", "switching_lag_duration"]
    )["avg_front_speed"].max().reset_index()


    # 3. --- Plotting ---
    print("Generating refined 1x2 panel figure for publication...")
    sns.set_theme(
        context="paper",
        style="ticks",
        rc={
            "font.size": 8, "axes.titlesize": 11, "axes.labelsize": 9,
            "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
            "legend.title_fontsize": 7, "axes.edgecolor": "black",
            "grid.linestyle": "--", "grid.alpha": 0.6,
        },
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8))

    # --- Panel A: Fitness Landscape (Unchanged logic) ---
    axA = axes[0]
    if not df_panel_A_data.empty:
        pivot_data = df_panel_A_data.pivot_table(
            index="switching_lag_duration", columns="k_total", values="avg_front_speed"
        ).sort_index(ascending=False)

        sns.heatmap(
            pivot_data, ax=axA, cmap="RdBu",
            cbar_kws={'label': 'Mean Population Fitness'},
            # Set vmin/vmax to match target image if needed, e.g., vmin=0.0, vmax=1.5
            linewidths=0.0 # Target image doesn't seem to have white grid lines
        )

        x_vals = pivot_data.columns
        x_ticks_to_show = np.array([1e-2, 1e-1, 1e0, 1e1, 1e2])
        x_tick_indices = x_vals.get_indexer(x_ticks_to_show, method='nearest')
        axA.set_xticks(x_tick_indices + 0.5)
        axA.set_xticklabels([f"$10^{{{int(np.log10(v))}}}$" for v in x_ticks_to_show], rotation=0, ha='center')

        y_vals = pivot_data.index
        y_ticks_to_show = np.array([1e2, 1e1, 1e0, 1e-1, 1e-2])
        y_tick_indices = y_vals.get_indexer(y_ticks_to_show, method='nearest')
        axA.set_yticks(y_tick_indices + 0.5)
        axA.set_yticklabels([f"$10^{{{int(np.log10(v))}}}$" for v in y_ticks_to_show], rotation=0)

        axA.set_title(f"(A) Fitness Landscape ($b_m$={b_m_for_panel_A:.2f})")
        axA.set_xlabel("Switching Rate ($k_{\mathrm{total}}$)")
        axA.set_ylabel("Switching Lag Duration ($\\tau$)")
    else:
        axA.text(0.5, 0.5, "No data for Panel A", ha='center', va='center')

    # --- Panel B: Cost of Lag vs. b_m (REVISED LOGIC) ---
    axB = axes[1]
    # Use a palette that matches the target (dark-purple -> yellow), like 'magma'
    b_m_palette = sns.color_palette("magma", n_colors=len(b_m_vals_for_lines))
    
    sns.lineplot(
        data=df_panel_B_data,
        x="switching_lag_duration",
        y="avg_front_speed",
        hue="b_m",  # Hue is now b_m
        hue_order=b_m_vals_for_lines, # Ensures legend order
        palette=b_m_palette,          # Use new palette
        marker='o',
        ms=4,
        lw=1.5,
        ax=axB,
    )
    axB.set(xscale='log', yscale='linear')
    # Update title to match target
    axB.set_title("(B) Lag Imposes a Hard Fitness Cost")
    axB.set_xlabel("Switching Lag Duration ($\\tau$)")
    # Update y-axis label to match target
    axB.set_ylabel("Mean Population Fitness")
    
    # Update legend to match target
    handles, labels = axB.get_legend_handles_labels()
    axB.legend(
        handles=handles,
        labels=[f"{float(l):.2f}" for l in labels], # Format labels as 0.20, 0.50 etc.
        title="Fitness Cost ($b_m$)", 
        loc='lower left', 
        frameon=False,
    )
    axB.grid(True, which='major', axis='y') # Target plot only has y-grid
    sns.despine(ax=axB) # Remove top and right spines to match target
    sns.despine(ax=axA) # Do the same for Panel A

    fig.subplots_adjust(wspace=0.35) # Adjust spacing

    # 4. --- Save the Final Figure ---
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Final polished figure saved to: {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()