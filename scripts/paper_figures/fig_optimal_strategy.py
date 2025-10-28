# FILE: scripts/paper_figures/plot_homogeneous_fitness.py
#
# Generates the final, definitive two-panel synthesis figure (Figure 6).
# Panel A shows the absolute fitness landscape in the fluctuating (w=60) environment,
# establishing the concept of an optimal switching rate.
# Panel B directly compares the *absolute* fitness cost of selection in the stable
# vs. the fluctuating environment for key strategic regimes.

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib
import warnings

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
from src.config import EXPERIMENTS
from src.io.data_loader import load_aggregated_data


def find_nearest(array, value):
    """Finds the nearest value in a sorted array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def process_filtered_data(df, env_name=None):
    """
    Helper function to filter for successful runs, add 's' column,
    and optionally filter by environment name.
    """
    valid_reasons = ["boundary_hit", "converged"]
    df_successful = df[df["termination_reason"].isin(valid_reasons)].copy()
    if df_successful.empty:
        return pd.DataFrame()
    
    df_filtered = df_successful[np.isclose(df_successful["phi"], 0.0)].copy()
    if df_filtered.empty:
        return pd.DataFrame()
        
    if env_name:
         df_filtered = df_filtered[df_filtered["env_definition"] == env_name].copy()

    if df_filtered.empty:
        return pd.DataFrame()

    df_filtered["s"] = df_filtered["b_m"] - 1.0
    return df_filtered


def main():
    # 1. --- Load Data ---
    bh_campaign_id = EXPERIMENTS["bet_hedging_final"]["campaign_id"]
    homo_campaign_id = EXPERIMENTS["homogeneous_fitness_cost"]["campaign_id"]
    
    print("Loading bet-hedging (fluctuating) data...")
    df_bh = load_aggregated_data(bh_campaign_id, PROJECT_ROOT)
    print("Loading homogeneous (stable) data...")
    df_homo = load_aggregated_data(homo_campaign_id, PROJECT_ROOT)
    
    if df_bh.empty or df_homo.empty:
        sys.exit("Error: Data for one or both campaigns is missing.")
        
    df_bh.columns = df_bh.columns.str.strip()
    df_homo.columns = df_homo.columns.str.strip()
    
    figure_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    output_path_pdf = os.path.join(
        figure_dir, "fig_optimal_strategy.pdf"
    )
    output_path_png = os.path.join(
        figure_dir, "fig_optimal_strategy.png"
    )

    # 2. --- Prepare Data for Panel A ---
    # Panel A shows absolute fitness in the fluctuating env for one s-value
    df_bh_abs = process_filtered_data(df_bh, "symmetric_refuge_60w")
    s_all_bh = np.sort(df_bh_abs["b_m"].unique())
    s_val_bh = find_nearest(s_all_bh, 0.5) # This is b_m, so s = -0.5
    df_panel_A = df_bh_abs[np.isclose(df_bh_abs["b_m"], s_val_bh)]

    # 3. --- DYNAMICALLY LINK PANELS ---
    # Find the optimal k_total from Panel A's data
    if df_panel_A.empty:
        sys.exit("Error: No data for Panel A, cannot determine optimal k.")
        
    # CORRECTED LOGIC: Find the k with the highest *mean* fitness,
    # which is what the line plot is actually showing.
    df_panel_A_mean = df_panel_A.groupby("k_total")["avg_front_speed"].mean().reset_index()
    optimal_k_index = df_panel_A_mean['avg_front_speed'].idxmax()
    optimal_k_value = df_panel_A_mean.loc[optimal_k_index, 'k_total']
    print(f"Dynamically identified optimal k_total (from mean fitness): {optimal_k_value}")

    # 4. --- Prepare Data for Panel B ---
    # Panel B compares absolute fitness in stable vs. fluctuating envs
    df_norm_homo = process_filtered_data(df_homo)
    df_norm_homo["Environment"] = "Stable"
    
    df_norm_bh = process_filtered_data(df_bh, "symmetric_refuge_60w")
    df_norm_bh["Environment"] = "Fluctuating"
    
    df_combined = pd.concat([df_norm_homo, df_norm_bh])

    # Define the k-regimes based on Panel A's *dynamic* optimum
    k_all = np.sort(df_combined["k_total"].unique())
    
    # --- CHANGE: Set "Slow" k to be further from optimal for better separation ---
    k_slow = find_nearest(k_all, 0.02) # Changed from 0.05
    k_optimal = optimal_k_value # <-- This is the dynamic link
    k_fast = find_nearest(k_all, 1.0)
    
    k_regime_values = [k_slow, k_optimal, k_fast]
    k_regime_names = ["Slow", "Optimal", "Fast"]
    k_regime_map = dict(zip(k_regime_values, k_regime_names))
    
    print(f"Regimes set: Slow (k={k_slow}), Optimal (k={k_optimal}), Fast (k={k_fast})")
    
    df_plot_B = df_combined[df_combined["k_total"].isin(k_regime_values)].copy()
    
    # Suppress Pandas warning for the next operation
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
        df_plot_B["Regime"] = df_plot_B["k_total"].map(k_regime_map)
    
    # Create the combined legend key
    df_plot_B["LegendKey"] = df_plot_B["Regime"] + " - " + df_plot_B["Environment"]


    # 5. --- Plotting ---
    sns.set_theme(style="ticks", context="paper")
    
    # Define shared colors and font properties
    regime_colors = {
        "Slow": "#4B0082", # Purple
        "Optimal": "#008080", # Teal
        "Fast": "#FFA500" # Orange
    }
    title_font = {'fontsize': 12, 'fontweight': 'bold'}
    label_font = {'fontsize': 10}
    tick_font_size = 9

    fig, axes = plt.subplots(
        1, 2, figsize=(cm_to_inch(17.8), cm_to_inch(8.5)), constrained_layout=True
    )

    # --- Panel A: Optimal Strategy in Fluctuating Environment ---
    axA = axes[0]
    sns.lineplot(
        data=df_panel_A,
        x="k_total",
        y="avg_front_speed",
        ax=axA,
        color="black",
        linewidth=2,
        marker="o",             # Changed from "."
        markersize=5,           # Adjusted size
        mfc="white",            # Added marker face color
        mec="black",            # Added marker edge color
        mew=0.5,                # Added marker edge width
        errorbar="sd"           # Add confidence interval
    )

    # Add vertical regime lines
    for (regime_name, k_val) in zip(k_regime_names, k_regime_values):
        axA.axvline(
            x=k_val, color=regime_colors[regime_name], linestyle="--", lw=2, alpha=0.9
        )

    axA.set_title(f"(A) Optimal Strategy in Fluctuating Environment", **title_font)
    axA.set_xlabel("Switching Rate, $k_{\mathrm{total}}$", **label_font)
    axA.set_ylabel("Absolute Fitness ($v_f$)", **label_font) # Match target
    axA.tick_params(axis="both", labelsize=tick_font_size)
    axA.grid(True, which="both", linestyle="--", alpha=0.6) # Lightened grid
    axA.set_xscale("log")
    axA.set_ylim(bottom=0.95) # Give space below 1.0


    # --- Panel B: Cost of Strategy in Different Environments ---
    axB = axes[1]

    # Define the exact mapping for the legend
    hue_order = [
        "Slow - Stable", "Slow - Fluctuating",
        "Optimal - Stable", "Optimal - Fluctuating",
        "Fast - Stable", "Fast - Fluctuating"
    ]
    
    # Map colors and styles to the combined key
    palette_map = {
        "Slow - Stable": regime_colors["Slow"], "Slow - Fluctuating": regime_colors["Slow"],
        "Optimal - Stable": regime_colors["Optimal"], "Optimal - Fluctuating": regime_colors["Optimal"],
        "Fast - Stable": regime_colors["Fast"], "Fast - Fluctuating": regime_colors["Fast"]
    }
    
    style_map = {
        "Slow - Stable": (1,0), "Slow - Fluctuating": (5, 3), # Solid, Dashed
        "Optimal - Stable": (1,0), "Optimal - Fluctuating": (5, 3),
        "Fast - Stable": (1,0), "Fast - Fluctuating": (5, 3)
    }

    # Ensure the "Regime" column exists before plotting
    if "Regime" not in df_plot_B.columns:
        sys.exit("Error: 'Regime' column was not created. Check k_regime_map.")
        
    # Filter out any potential NaN regimes if mapping failed
    df_plot_B_final = df_plot_B.dropna(subset=['Regime'])
    if df_plot_B_final.empty and not df_plot_B.empty:
         sys.exit(f"Error: Regime mapping failed. k_values in data: {df_plot_B['k_total'].unique()}")

    sns.lineplot(
        data=df_plot_B_final, # Use final filtered data
        x="s",
        y="avg_front_speed", # Plot absolute fitness
        ax=axB,
        hue="LegendKey",
        hue_order=hue_order,
        style="LegendKey",
        style_order=hue_order,
        palette=palette_map,
        dashes=style_map,
        linewidth=2.5,
        errorbar="sd",
        markers=False, # No markers on target
        legend="full"
    )

    axB.set_title("(B) Cost of Strategy in Different Environments", **title_font)
    axB.set_xlabel("Selection Strength, $s = b_m - 1$", **label_font)
    axB.set_ylabel("Absolute Fitness ($v_f$)", **label_font) # Match target
    
    # Add axis limits for better scaling
    axB.set_ylim(1.15, 1.7)
    axB.set_xlim(-0.95, 0.05) # Add padding
    
    axB.tick_params(axis="both", labelsize=tick_font_size)
    axB.grid(True, which="both", linestyle="--", alpha=0.6) # Lightened grid
    
    # Customize legend
    legB = axB.legend(fontsize=8, title="Legend", frameon=True)
    plt.setp(legB.get_title(), fontsize=9, fontweight='bold')

    # 6. --- Save Figure ---
    plt.savefig(output_path_pdf, bbox_inches="tight")
    plt.savefig(output_path_png, bbox_inches="tight", dpi=300)
    print(f"Figure saved to: {output_path_pdf} and {output_path_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()

