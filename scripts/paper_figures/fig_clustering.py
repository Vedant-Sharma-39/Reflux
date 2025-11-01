"""
Main Figure: Absolute vs. Relative Benefit of Clustering for Disadvantaged Mutants

This script analyzes the 'deleterious_invasion_dynamics' experiment to create a
comprehensive two-panel figure.

v2: Updated to full publication-ready quality with LaTeX fonts,
    larger text, and improved spacing.
"""

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
matplotlib.rcParams["font.size"] = 12 # Base font size
# --- End Publication Settings ---


def cm_to_inch(cm):
    return cm / 2.54


# --- Helper Functions ---


def get_project_root() -> Path:
    """Dynamically finds the project root directory."""
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = get_project_root()
from src.config import EXPERIMENTS
from src.io.data_loader import load_aggregated_data


def main():
    campaign_id = EXPERIMENTS["deleterious_invasion_dynamics"]["campaign_id"]
    print(f"Generating figure from campaign: {campaign_id}")
    df_full = load_aggregated_data(campaign_id, PROJECT_ROOT)

    if df_full.empty:
        sys.exit(f"Error: Data for campaign '{campaign_id}' is empty.")

    figure_dir = PROJECT_ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    
    # --- Updated Output Paths ---
    output_path_pdf = figure_dir / "fig_clustering.pdf"
    output_path_png = figure_dir / "fig_clustering.png"

    # --- Data Processing ---
    unique_sizes = sorted(df_full["initial_mutant_patch_size"].unique())
    initial_size_val = unique_sizes[len(unique_sizes) // 2]
    df = df_full[df_full["initial_mutant_patch_size"] == initial_size_val].copy()

    if df.empty:
        sys.exit(f"Error: No data found for initial_size={initial_size_val}.")

    df_extinctions = df[df["outcome"] == "extinction"].copy()
    if df_extinctions.empty:
        sys.exit("Warning: No extinction data found; cannot generate plot.")

    # Calculate mean cluster size
    fragments_df = (
        df.groupby(["correlation_length", "b_m"])
        .agg(mean_fragments=("num_fragments", "mean"))
        .reset_index()
    )
    fragments_df["mean_cluster_size"] = (
        initial_size_val / fragments_df["mean_fragments"]
    )

    # Calculate mean invasion depth
    depth_df = (
        df_extinctions.groupby(["correlation_length", "b_m"])
        .agg(mean_q_max=("q_at_outcome", "mean"))
        .reset_index()
    )

    # Combine metrics into a single analysis DataFrame
    analysis_df = pd.merge(depth_df, fragments_df, on=["correlation_length", "b_m"])

    # Calculate relative invasion depth against the most fragmented state
    baseline_depths = analysis_df.loc[
        analysis_df.groupby("b_m")["mean_cluster_size"].idxmin()
    ]
    baseline_depths = baseline_depths.rename(columns={"mean_q_max": "baseline_q_max"})
    analysis_df = pd.merge(
        analysis_df, baseline_depths[["b_m", "baseline_q_max"]], on="b_m"
    )
    analysis_df["relative_invasion_depth"] = (
        analysis_df["mean_q_max"] / analysis_df["baseline_q_max"]
    )

    # --- Plotting Setup ---
    
    # --- Standardized Font Definitions ---
    title_font = {'fontsize': 14, 'fontweight': 'bold'}
    label_font = {'fontsize': 13}
    tick_font_size = 12
    legend_font_size = 11

    sns.set_theme(context="paper", style="ticks")
    
    fig, axes = plt.subplots(
        1, 2, 
        figsize=(cm_to_inch(17.8), cm_to_inch(8.5)), # Standard 1x2 size
        constrained_layout=True # Fixes cramping
    )
    ax1, ax2 = axes

    palette = sns.color_palette("magma", n_colors=len(analysis_df["b_m"].unique()))

    # --- Panel A: Absolute Invasion Depth ---
    sns.lineplot(
        data=analysis_df,
        x="mean_cluster_size",
        y="mean_q_max",
        hue="b_m",
        palette=palette,
        marker="o",
        lw=2.5,
        ms=6,
        mfc='white', # White marker fill
        mec='black', # Black marker edge
        mew=0.7,   # Marker edge width
        ax=ax1,
        legend=False,
    )
    # --- Use LaTeX and Standard Fonts ---
    ax1.set_title(r"(A) Absolute Invasion Depth", **title_font)
    ax1.set_ylabel(r"Mean Peak Mutant Fraction ($\langle q_{\max} \rangle$)", **label_font)

    # --- Panel B: Relative Benefit of Clustering ---
    sns.lineplot(
        data=analysis_df,
        x="mean_cluster_size",
        y="relative_invasion_depth",
        hue="b_m",
        palette=palette,
        marker="o",
        lw=2.5,
        ms=6,
        mfc='white',
        mec='black',
        mew=0.7,
        ax=ax2,
        legend=True,
    )
    # --- Use LaTeX and Standard Fonts ---
    ax2.set_title(r"(B) Relative Benefit of Clustering", **title_font)
    ax2.set_ylabel(r"Relative Invasion Depth", **label_font)
    ax2.axhline(1.0, color="grey", linestyle="--", lw=1.5)

    # --- Shared Axis Properties and Legend ---
    for ax in axes:
        ax.set_xlabel(r"Mean Initial Cluster Size", **label_font)
        ax.set_xscale("log")
        ax.grid(True, which="both", ls=":", alpha=0.4) # Lighter grid
        ax.tick_params(axis="both", which='major', labelsize=tick_font_size)

    # --- Improve and place the legend ---
    leg = ax2.get_legend()
    leg.set_title(r"Selection Strength ($b_m$)")
    leg.set_loc("upper left")
    leg.set_frame_on(False) # Cleaner look
    plt.setp(leg.get_title(), fontsize=legend_font_size)
    plt.setp(leg.get_texts(), fontsize=legend_font_size)

    sns.despine(fig) # Remove top/right spines

    # --- Save Final Figure ---
    plt.savefig(output_path_pdf, bbox_inches="tight", dpi=600)
    plt.savefig(output_path_png, bbox_inches="tight", dpi=600)
    print(f"\nFigure saved to:\n  {output_path_pdf}\n  {output_path_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()