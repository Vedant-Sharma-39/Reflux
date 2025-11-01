# FILE: scripts/paper_figures/fig_optimal_strategy.py
#
# Generates the final, definitive two-panel synthesis figure (Figure 1).
# Fully publication-ready for high-impact journals.
# - Two-panel (1x2) layout
# - Legends placed in optimal empty spaces (top-right Panel A, bottom-right Panel B)
# - Markers only on 'Fluctuating' lines in Panel B
# - Solid lines in Panel B made visually distinct (slightly thicker + alpha)
# - LaTeX math, vector output, high readability

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
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 12

def cm_to_inch(cm):
    return cm / 2.54


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

    output_path_pdf = os.path.join(figure_dir, "fig_optimal_strategy.pdf")
    output_path_png = os.path.join(figure_dir, "fig_optimal_strategy.png")

    # 2. --- Prepare Data for Panel A ---
    df_bh_abs = process_filtered_data(df_bh, "symmetric_refuge_60w")
    s_all_bh = np.sort(df_bh_abs["b_m"].unique())
    s_val_bh = find_nearest(s_all_bh, 0.5)  # b_m = 0.5 → s = -0.5
    df_panel_A = df_bh_abs[np.isclose(df_bh_abs["b_m"], s_val_bh)]

    if df_panel_A.empty:
        sys.exit("Error: No data for Panel A, cannot determine optimal k.")
        
    df_panel_A_mean = df_panel_A.groupby("k_total")["avg_front_speed"].mean().reset_index()
    optimal_k_index = df_panel_A_mean['avg_front_speed'].idxmax()
    optimal_k_value = df_panel_A_mean.loc[optimal_k_index, 'k_total']
    print(f"Dynamically identified optimal k_total (from mean fitness): {optimal_k_value}")

    # 3. --- Prepare Data for Panel B ---
    df_norm_homo = process_filtered_data(df_homo)
    df_norm_homo["Environment"] = "Stable"
    
    df_norm_bh = process_filtered_data(df_bh, "symmetric_refuge_60w")
    df_norm_bh["Environment"] = "Fluctuating"
    
    df_combined = pd.concat([df_norm_homo, df_norm_bh])

    k_all = np.sort(df_combined["k_total"].unique())
    k_slow = find_nearest(k_all, 0.02) 
    k_optimal = optimal_k_value 
    k_fast = find_nearest(k_all, 1.0)
    
    k_regime_values = [k_slow, k_optimal, k_fast]
    k_regime_names = ["Slow", "Optimal", "Fast"]
    k_regime_map = dict(zip(k_regime_values, k_regime_names))
    
    print(f"Regimes set: Slow (k={k_slow:.5f}), Optimal (k={k_optimal:.5f}), Fast (k={k_fast:.5f})")
    
    df_plot_B = df_combined[df_combined["k_total"].isin(k_regime_values)].copy()
    
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
        df_plot_B["Regime"] = df_plot_B["k_total"].map(k_regime_map)

    # 4. --- Plotting Setup ---
    sns.set_theme(style="ticks", context="paper")

    regime_colors = {
        "Slow": "#4B0082",   # Indigo
        "Optimal": "#008080", # Teal
        "Fast": "#FFA500"    # Orange
    }

    title_font = {'fontsize': 14, 'fontweight': 'bold'}
    label_font = {'fontsize': 13}
    tick_font_size = 12
    legend_font_size = 11

    fig, axes = plt.subplots(
        1, 2, 
        figsize=(cm_to_inch(17.8), cm_to_inch(8.5)),
        constrained_layout=True
    )

    # --- Panel A: Optimal Strategy ---
    axA = axes[0]
    sns.lineplot(
        data=df_panel_A,
        x="k_total",
        y="avg_front_speed",
        ax=axA,
        color="black",
        linewidth=2.5,
        marker="o",
        markersize=6,
        mfc="white",
        mec="black",
        mew=0.7,
        errorbar=("ci", 95),
        legend=False
    )

    for regime_name, k_val in zip(k_regime_names, k_regime_values):
        axA.axvline(x=k_val, color=regime_colors[regime_name], linestyle="--", lw=2.5, alpha=0.9)
        try:
            y_val = df_panel_A_mean[np.isclose(df_panel_A_mean["k_total"], k_val)]["avg_front_speed"].values[0]
            axA.plot(k_val, y_val, 'o', color=regime_colors[regime_name],
                     markersize=9, mec='black', mew=1.2)
        except IndexError:
            print(f"Warning: Could not find y-value for k={k_val}")

    axA.set_title('(A) Optimal Strategy', **title_font)
    axA.set_xlabel(r"Switching Rate, $k_{\mathrm{total}}$", **label_font)
    axA.set_ylabel(r"Absolute Fitness (Front Speed, $v_f$)", **label_font)
    axA.tick_params(axis="both", labelsize=tick_font_size)
    axA.grid(True, which="both", linestyle=":", alpha=0.4)
    axA.set_xscale("log")
    axA.set_ylim(bottom=0.95)

    # --- Panel A Legend: Top Right ---
    legend_elements_A = [
        Line2D([0], [0], color=regime_colors['Slow'], lw=2.5, ls='--', label='Slow'),
        Line2D([0], [0], color=regime_colors['Optimal'], lw=2.5, ls='--', label='Optimal'),
        Line2D([0], [0], color=regime_colors['Fast'], lw=2.5, ls='--', label='Fast')
    ]
    axA.legend(handles=legend_elements_A, title="Regime", loc='upper right',
               frameon=False, fontsize=legend_font_size, title_fontsize=legend_font_size)

    # --- Panel B: Cost of Strategy ---
    axB = axes[1]
    df_plot_B_final = df_plot_B.dropna(subset=['Regime'])
    if df_plot_B_final.empty:
        sys.exit("Error: No data for Panel B after regime mapping.")

    # Plot each regime and environment explicitly to avoid Seaborn style/marker bugs
    for regime in k_regime_names:
        color = regime_colors[regime]
        # Stable: solid line, no markers, slightly thicker + semi-transparent for contrast
        df_s = df_plot_B_final[
            (df_plot_B_final["Regime"] == regime) & 
            (df_plot_B_final["Environment"] == "Stable")
        ]
        if not df_s.empty:
            sns.lineplot(
                data=df_s, x="s", y="avg_front_speed",
                ax=axB, color=color, linewidth=3.0,  # Slightly thicker
                errorbar=("ci", 95), linestyle='-', marker=None, legend=False,
                alpha=0.8  # Semi-transparent to distinguish from Fluctuating
            )
        # Fluctuating: dashed line, with markers
        df_f = df_plot_B_final[
            (df_plot_B_final["Regime"] == regime) & 
            (df_plot_B_final["Environment"] == "Fluctuating")
        ]
        if not df_f.empty:
            sns.lineplot(
                data=df_f, x="s", y="avg_front_speed",
                ax=axB, color=color, linewidth=2.5,
                errorbar=("ci", 95), linestyle='--', marker='o',
                markersize=7, markeredgecolor='black', markeredgewidth=0.7, legend=False
            )

    axB.set_title('(B) Cost of Strategy', **title_font)
    axB.set_xlabel(r"Selection Strength, $s = b_m - 1$", **label_font)
    axB.set_ylabel(r"Absolute Fitness (Front Speed, $v_f$)", **label_font)
    axB.set_ylim(1.15, 1.7)
    axB.set_xlim(-0.95, 0.05)
    axB.tick_params(axis="both", labelsize=tick_font_size)
    axB.grid(True, which="both", linestyle=":", alpha=0.4)

    # --- Panel B Legend: Bottom Right ---
    legend_elements_B = [
        Line2D([0], [0], color='gray', lw=2.5, ls='-', label='Stable'),
        Line2D([0], [0], color='gray', lw=2.5, ls='--', marker='o',
               mec='black', mew=0.7, mfc='gray', markersize=7, label='Fluctuating')
    ]
    axB.legend(handles=legend_elements_B, title="Environment", loc='lower right',
               frameon=False, fontsize=legend_font_size, title_fontsize=legend_font_size)

    # 5. --- Save Figure ---
    plt.savefig(output_path_pdf, bbox_inches="tight", dpi=600)
    plt.savefig(output_path_png, bbox_inches="tight", dpi=600)
    print(f"Figure saved successfully to:\n  {output_path_pdf}\n  {output_path_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()