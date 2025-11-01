# FILE: scripts/paper_figures/fig_reversibility.py
#
# Generates the definitive Figure 3 from the "bet_hedging_final" experiment.
#
# v8 (CORRECTED LAYOUT):
# 1. Uses GridSpec with improved height ratios and spacing.
# 2. Removes redundant legend from Panel D to avoid duplication.
# 3. Ensures shared legend is clean, horizontal, centered, and well-spaced.
# 4. Fixes axis label alignment and prevents overlap.
# 5. Highlights phi = -1.0, 0.0, 0.75 with thicker, opaque lines.
# 6. Uses marker size 5 (ms=5) as requested.

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import warnings
from matplotlib.lines import Line2D

# --- Publication Settings ---
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 12
# --- End Publication Settings ---


def cm_to_inch(cm):
    return cm / 2.54


def get_project_root():
    """Dynamically finds the project root directory."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def find_nearest(array, value):
    """Finds the nearest value in a sorted array."""
    array = np.asarray(array)
    if array.size == 0:
        raise ValueError(
            f"Cannot find nearest value in an empty array. Check data filters."
        )
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def plot_strategy_panel(ax, df_bm_slice, title, label_font, tick_font_size):
    """
    Helper function to plot a single fitness-vs-k panel.
    Plots highlighted lines thicker, matching the target image.
    """
    
    phis = sorted(df_bm_slice["phi"].unique())
    palette = sns.color_palette("coolwarm_r", n_colors=len(phis))
    phi_to_color = dict(zip(phis, palette))
    
    # --- Highlight -1.0, 0.0, and 0.75 (as in target) ---
    highlight_phis = [find_nearest(phis, p) for p in [-1.0, 0.0, 0.75]]

    # Plot all lines in a loop to control style
    for phi_val in phis:
        is_highlight = np.isclose(phi_val, highlight_phis).any()
        df_phi = df_bm_slice[np.isclose(df_bm_slice["phi"], phi_val)]
        
        sns.lineplot(
            data=df_phi,
            x="k_total",
            y="avg_front_speed",
            ax=ax,
            color=phi_to_color[phi_val],
            label=f"{phi_val:.2f}",
            lw=3.0 if is_highlight else 1.5,
            alpha=1.0 if is_highlight else 0.7,
            zorder=5 if is_highlight else 2,
            marker="o",
            ms=5,
            mfc="white",
            mec="black",
            mew=0.7,
            errorbar=("ci", 95),
            legend="full"
        )
    
    ax.set_xscale("log")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(r"Switching Rate, $k_{\mathrm{total}}$", **label_font)
    ax.set_ylabel(r"Absolute Fitness ($v_f$)", **label_font)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.tick_params(axis="both", which="major", labelsize=tick_font_size)
    
    # Remove individual legends
    if ax.get_legend() is not None:
        ax.get_legend().remove()


def main():
    project_root = get_project_root()
    from src.config import EXPERIMENTS, PARAM_GRID

    try:
        campaign_id = EXPERIMENTS["bet_hedging_final"]["campaign_id"]
    except KeyError as e:
        print(
            f"Error: Required experiment key 'bet_hedging_final' not found in src/config.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    summary_path = os.path.join(
        project_root,
        "data",
        campaign_id,
        "analysis",
        f"{campaign_id}_summary_aggregated.csv",
    )
    if not os.path.exists(summary_path):
        print(
            f"Error: Data file not found for campaign '{campaign_id}'. Run 'make consolidate CAMPAIGN={campaign_id}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading data from: {summary_path}")
    df = pd.read_csv(summary_path, low_memory=False)

    # --- Data Processing ---
    df = df[~np.isclose(df["phi"], 1.0)].copy()
    df_filtered = df[df["termination_reason"] == "converged"].copy()

    env_name = PARAM_GRID["env_definitions"]["symmetric_refuge_60w"]["name"]
    df_plot_data = df_filtered[
        (df_filtered["env_definition"] == env_name) & (df_filtered["k_total"] > 0)
    ].copy()

    if df_plot_data.empty:
        print(
            f"\nERROR: No data found for environment '{env_name}' in campaign '{campaign_id}' after filtering.",
            file=sys.stderr,
        )
        sys.exit(1)

    figure_dir = os.path.join(project_root, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    
    output_path_pdf = os.path.join(figure_dir, "fig_reversibility.pdf")
    output_path_png = os.path.join(figure_dir, "fig_reversibility.png")
    print(f"\nGenerating definitive Figure 3 from campaign: {campaign_id}")

    # --- Plotting Setup ---
    sns.set_theme(style="ticks", context="paper")
    
    title_font = {'fontsize': 14, 'fontweight': 'bold'}
    label_font = {'fontsize': 13}
    tick_font_size = 12
    legend_font_size = 11

    # --- Use GridSpec with improved proportions ---
    fig = plt.figure(figsize=(cm_to_inch(17.8), cm_to_inch(19.0)))
    gs = fig.add_gridspec(3, 2, height_ratios=[10, 10, 3.5])  # Increased legend row height
    
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    ax_legend = fig.add_subplot(gs[2, :])


    bm_all = np.sort(df_plot_data["b_m"].unique())
    bm_targets = [0.9, 0.5, 0.2]
    bm_vals = [find_nearest(bm_all, val) for val in bm_targets]

    panel_map = {
        bm_vals[0]: (axA, r"(A) Weak Disadvantage ($b_m = " + f"{bm_vals[0]:.2f}$" + r")"),
        bm_vals[1]: (axB, r"(B) Medium Disadvantage ($b_m = " + f"{bm_vals[1]:.2f}$" + r")"),
        bm_vals[2]: (axC, r"(C) Strong Disadvantage ($b_m = " + f"{bm_vals[2]:.2f}$" + r")"),
    }
    
    handles, labels = [], []
    for bm_val, (ax, title) in panel_map.items():
        df_panel = df_plot_data[np.isclose(df_plot_data["b_m"], bm_val)]
        if not df_panel.empty:
            plot_strategy_panel(ax, df_panel, title, label_font, tick_font_size)
            if not handles:
                handles_all, labels_all = ax.get_legend_handles_labels()
                sorted_legend = sorted(zip(labels_all, handles_all), key=lambda x: float(x[0]))
                labels = [l for l, h in sorted_legend]
                handles = [h for l, h in sorted_legend]

    # --- Panel D: Optimal Strategy Performance (NO LOCAL LEGEND) ---
    df_periodic_filtered = df_filtered[df_filtered["env_definition"] == env_name]
    phi_irr_val = find_nearest(df_periodic_filtered["phi"].unique(), -1.0)
    df_baseline_runs = df_periodic_filtered[
        np.isclose(df_periodic_filtered["phi"], phi_irr_val)
    ].copy()
    df_baseline_stats = (
        df_baseline_runs.groupby("b_m")["avg_front_speed"]
        .agg(["mean", "std"])
        .reset_index()
    )

    df_rev_stats = (
        df_plot_data.groupby(["b_m", "phi", "k_total"])["avg_front_speed"]
        .agg(["mean", "std"])
        .reset_index()
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pareto_idx = df_rev_stats.groupby("b_m")["mean"].idxmax()
    df_pareto_stats = df_rev_stats.loc[pareto_idx]

    # Baseline (irreversible)
    axD.errorbar(
        x=df_baseline_stats["b_m"],
        y=df_baseline_stats["mean"],
        yerr=df_baseline_stats["std"],
        label=r"Irreversible ($\phi=-1.0$)",
        color="crimson",
        marker="s",
        lw=2.5,
        ls="--",
        capsize=5,
        ms=7,
        mfc='crimson',
        mec='black',
        mew=0.7
    )
    
    # Optimal reversible
    axD.errorbar(
        x=df_pareto_stats["b_m"],
        y=df_pareto_stats["mean"],
        yerr=df_pareto_stats["std"],
        label=r"Optimal Reversible",
        color="royalblue",
        marker="o",
        lw=3.0,
        ls='-',
        ms=8,
        mfc='white',
        mec='royalblue',
        mew=1.0,
        capsize=5,
    )
    
    # Reversibility advantage (fill)
    baseline_mean_aligned = df_baseline_stats.set_index("b_m").reindex(df_pareto_stats["b_m"])["mean"]
    axD.fill_between(
        df_pareto_stats["b_m"],
        df_pareto_stats["mean"],
        baseline_mean_aligned,
        color="gold",
        alpha=0.3,
        label=r"Reversibility Advantage",
    )

    axD.set_title("(D) Optimal Strategy Performance", **title_font)
    axD.set_xlabel(r"Mutant Fitness in Hostile Patch, $b_m$", **label_font)
    axD.set_ylabel(r"Absolute Fitness ($v_f$)", **label_font)
    axD.grid(True, which="both", ls=":", alpha=0.4)
    axD.tick_params(axis="both", which="major", labelsize=tick_font_size)

    # IMPORTANT: Do NOT add a legend to Panel D — it's handled globally below

    # --- Shared horizontal legend at bottom ---
    ax_legend.axis('off')
    ax_legend.legend(
        handles, 
        labels, 
        title=r"Switching Bias, $\phi$",
        loc='center',
        frameon=False, 
        fontsize=legend_font_size,
        title_fontsize=legend_font_size,
        ncol=4,
        columnspacing=1.2,
        handletextpad=0.4
    )
    
    sns.despine(fig)
    
    # --- Final layout adjustments ---
    fig.subplots_adjust(
        left=0.10,
        right=0.95,
        top=0.90,   # Reduced from 0.92 to avoid title crowding
        bottom=0.16, # Increased from 0.15 for legend
        hspace=0.7, # Increased vertical spacing
        wspace=0.25
    )

    plt.savefig(output_path_pdf, bbox_inches="tight", dpi=600)
    plt.savefig(output_path_png, bbox_inches="tight", dpi=600)
    print(f"\nDefinitive Figure 3 saved to:\n  {output_path_pdf}\n  {output_path_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()