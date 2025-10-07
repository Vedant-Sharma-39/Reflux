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
    campaign_id = EXPERIMENTS["lag_vs_selection_definitive"]["campaign_id"]
    print(f"Loading data from definitive campaign: '{campaign_id}'")
    df_raw = load_aggregated_data(campaign_id, PROJECT_ROOT)

    if df_raw.empty:
        sys.exit(f"Error: Data for campaign '{campaign_id}' is missing.")

    figure_dir = PROJECT_ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    output_path = figure_dir / "fig_transient_lag_analysis_final_2panel.pdf"

    # 2. --- Data Filtering and Preparation ---
    df_converged = df_raw[df_raw["termination_reason"] == "converged"].copy()
    df_analysis_base = df_converged[np.isclose(df_converged["phi"], 0.0)].copy()

    env_for_plots = "symmetric_refuge_60w"
    b_m_all = np.sort(df_analysis_base["b_m"].unique())
    b_m_for_panels = find_nearest(b_m_all, 0.50)

    # Data for both panels will be based on this b_m value
    df_base_for_panels = df_analysis_base[
        (df_analysis_base["env_definition"] == env_for_plots) &
        np.isclose(df_analysis_base["b_m"], b_m_for_panels)
    ].copy()
    
    # --- MODIFICATION: Prepare data specifically for the new Panel B ---
    # Select three representative, fixed switching rates to plot as lines
    k_all = np.sort(df_base_for_panels["k_total"].unique())
    k_vals_for_lines = [
        find_nearest(k_all, 0.1),
        find_nearest(k_all, 1.0),
        find_nearest(k_all, 10.0),
    ]
    # Filter the main data to get only the rows corresponding to these k values
    df_panel_b_data = df_base_for_panels[
        df_base_for_panels["k_total"].isin(k_vals_for_lines)
    ].copy()


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

    # --- Panel A: Fitness Landscape (Unchanged) ---
    axA = axes[0]
    if not df_base_for_panels.empty:
        pivot_data = df_base_for_panels.pivot_table(
            index="switching_lag_duration", columns="k_total", values="avg_front_speed"
        ).sort_index(ascending=False)

        sns.heatmap(
            pivot_data, ax=axA, cmap="RdBu_r",
            cbar_kws={'label': 'Mean Population Fitness'},
            linewidths=0.5, linecolor='white'
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

        axA.set_title(f"(A) Fitness Landscape ($b_m$={b_m_for_panels:.2f})")
        axA.set_xlabel("Switching Rate ($k_{\mathrm{total}}$)")
        axA.set_ylabel("Switching Lag Duration ($\\tau$)")
    else:
        axA.text(0.5, 0.5, "No data for Panel A", ha='center', va='center')

    # --- MODIFICATION: Panel B now shows fitness vs lag for fixed k rates ---
    axB = axes[1]
    k_palette = sns.color_palette("viridis", n_colors=len(k_vals_for_lines))
    
    sns.lineplot(
        data=df_panel_b_data,
        x="switching_lag_duration",
        y="avg_front_speed",
        hue="k_total",
        hue_order=k_vals_for_lines, # Ensures legend order matches palette
        palette=k_palette,
        marker='o',
        ms=4,
        lw=1.5,
        ax=axB,
    )
    axB.set(xscale='log', yscale='linear')
    axB.set_title("(B) Cost of Lag at Fixed Switching Rates")
    axB.set_xlabel("Switching Lag Duration ($\\tau$)")
    axB.set_ylabel("Mean Fitness")
    axB.legend(
        title="Switching Rate ($k$)", loc='lower left', frameon=False,
    )
    axB.grid(True, which='major')

    fig.subplots_adjust(wspace=0.3)

    # 4. --- Save the Final Figure ---
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Final polished figure saved to: {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()