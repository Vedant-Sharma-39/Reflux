# FILE: scripts/paper_figures/fig_transient_lag_analysis.py
#
# Generates the definitive 1x2 panel figure for journal submission.
#
# v3: Swapped Panel A and Panel B as requested.
#     Updated to full publication-ready quality with LaTeX fonts,
#     larger text, and improved spacing.

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# --- Publication Settings ---
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 12 # Increased base font size
# --- End Publication Settings ---


# --- Project Setup ---

def cm_to_inch(cm):
    """Converts centimeters to inches for figure sizing."""
    return cm / 2.54

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
    
    # --- Updated Output Paths ---
    output_path_pdf = figure_dir / "fig_switching_lag.pdf"
    output_path_png = figure_dir / "fig_switching_lag.png"

    # 2. --- Data Filtering and Preparation ---
    # Using the convergence filter from your original script
    df_converged = df_raw[df_raw["var_front_speed"] <= 0.001].copy()
    df_analysis_base = df_converged[np.isclose(df_converged["phi"], 0.0)].copy()

    env_for_plots = "symmetric_refuge_60w"
    b_m_all = np.sort(df_analysis_base["b_m"].unique())

    # --- Data for Heatmap (Original Panel A) ---
    b_m_for_heatmap = find_nearest(b_m_all, 0.50)
    df_heatmap_data = df_analysis_base[
        (df_analysis_base["env_definition"] == env_for_plots) &
        np.isclose(df_analysis_base["b_m"], b_m_for_heatmap)
    ].copy()
    
    # --- Data for Line Plot (Original Panel B) ---
    df_lineplot_base = df_analysis_base[
        (df_analysis_base["env_definition"] == env_for_plots)
    ].copy()

    b_m_vals_for_lines = [find_nearest(b_m_all, v) for v in [0.20, 0.50, 0.70, 0.90, 1.00]]
    
    df_lineplot_filtered = df_lineplot_base[
        df_lineplot_base["b_m"].isin(b_m_vals_for_lines)
    ].copy()

    df_lineplot_data = df_lineplot_filtered.groupby(
        ["b_m", "switching_lag_duration"]
    )["avg_front_speed"].max().reset_index()


    # 3. --- Plotting ---
    print("Generating refined 1x2 panel figure for publication (Panels A/B swapped)...")
    
    # --- Standardized Font Definitions ---
    title_font = {'fontsize': 14, 'fontweight': 'bold'}
    label_font = {'fontsize': 13}
    tick_font_size = 12
    legend_font_size = 11

    # Set theme *before* creating figure
    sns.set_theme(context="paper", style="ticks")

    # --- Increased figure size and enabled constrained_layout ---
    fig, axes = plt.subplots(
        1, 2, 
        figsize=(cm_to_inch(17.8), cm_to_inch(8.5)), # Standard 1x2 size
        constrained_layout=True # Fixes cramping
    )

    # --- Panel A (NOW THE LINE PLOT) ---
    axA = axes[0]
    b_m_palette = sns.color_palette("magma", n_colors=len(b_m_vals_for_lines))
    
    sns.lineplot(
        data=df_lineplot_data,
        x="switching_lag_duration",
        y="avg_front_speed",
        hue="b_m",
        hue_order=b_m_vals_for_lines,
        palette=b_m_palette,
        marker='o',
        ms=6, 
        lw=2.0, 
        ax=axA,
    )
    axA.set(xscale='log', yscale='linear')

    # --- Use standard fonts and LaTeX labels ---
    axA.set_title(r"(A) Lag Imposes a Hard Fitness Cost", **title_font)
    axA.set_xlabel(r"Switching Lag Duration ($\tau$)", **label_font)
    axA.set_ylabel(r"Mean Population Fitness ($v_f$)", **label_font)
    axA.tick_params(axis='both', which='major', labelsize=tick_font_size)
    
    handles, labels = axA.get_legend_handles_labels()
    axA.legend(
        handles=handles,
        labels=[f"{float(l):.2f}" for l in labels],
        title=r"Fitness Cost ($b_m$)", # LaTeX title
        loc='lower left', 
        frameon=False,
        fontsize=legend_font_size, # Use standard font size
        title_fontsize=legend_font_size # Use standard font size
    )
    axA.grid(True, which='major', axis='y', linestyle=':', alpha=0.6) # Lighter grid
    sns.despine(ax=axA)
    
    
    # --- Panel B (NOW THE HEATMAP) ---
    axB = axes[1]
    if not df_heatmap_data.empty:
        pivot_data = df_heatmap_data.pivot_table(
            index="switching_lag_duration", columns="k_total", values="avg_front_speed"
        ).sort_index(ascending=False)

        # --- Use LaTeX for colorbar label ---
        cbar_kws = {'label': r'Mean Population Fitness ($v_f$)'}

        sns.heatmap(
            pivot_data, ax=axB, cmap="RdBu",
            cbar_kws=cbar_kws,
            linewidths=0.0
        )
        
        # --- Use LaTeX for tick labels ---
        x_vals = pivot_data.columns
        x_ticks_to_show = np.array([1e-2, 1e-1, 1e0, 1e1, 1e2])
        x_tick_indices = x_vals.get_indexer(x_ticks_to_show, method='nearest')
        axB.set_xticks(x_tick_indices + 0.5)
        axB.set_xticklabels([fr"$10^{{{int(np.log10(v))}}}$" for v in x_ticks_to_show], rotation=0, ha='center')

        y_vals = pivot_data.index
        y_ticks_to_show = np.array([1e2, 1e1, 1e0, 1e-1, 1e-2])
        y_tick_indices = y_vals.get_indexer(y_ticks_to_show, method='nearest')
        axB.set_yticks(y_tick_indices + 0.5)
        axB.set_yticklabels([fr"$10^{{{int(np.log10(v))}}}$" for v in y_ticks_to_show], rotation=0)

        # --- Use standard fonts and LaTeX labels ---
        axB.set_title(fr"(B) Fitness Landscape ($b_m={b_m_for_heatmap:.2f}$)", **title_font)
        axB.set_xlabel(r"Switching Rate ($k_{\mathrm{total}}$)", **label_font)
        axB.set_ylabel(r"Switching Lag Duration ($\tau$)", **label_font)
        
        # --- Increase tick label font size ---
        axB.tick_params(axis='both', which='major', labelsize=tick_font_size)
        axB.figure.axes[-1].yaxis.label.set_size(label_font['fontsize'])
        axB.figure.axes[-1].tick_params(labelsize=tick_font_size)
        sns.despine(ax=axB)

    else:
        axB.text(0.5, 0.5, "No data for Panel B", ha='center', va='center')


    # 4. --- Save the Final Figure ---
    plt.savefig(output_path_pdf, dpi=600, bbox_inches="tight")
    plt.savefig(output_path_png, dpi=600, bbox_inches="tight")
    print(f"\n✅ Final polished figure (A/B swapped) saved to:\n  {output_path_pdf}\n  {output_path_png}")
    plt.close(fig)

if __name__ == "__main__":
    main()