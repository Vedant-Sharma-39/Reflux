"""
Main Figure: Absolute vs. Relative Benefit of Clustering for Disadvantaged Mutants

This script analyzes the 'deleterious_invasion_dynamics' experiment to create a
comprehensive two-panel figure.

- Panel A shows the absolute invasion depth, demonstrating that weaker negative
  selection allows mutants to persist longer.
- Panel B shows the relative invasion depth (normalized by the most fragmented
  state), revealing the key insight: the survival boost from clustering is most
  pronounced for the most disadvantaged mutants.
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


def cm_to_inch(cm):
    return cm / 2.54


# --- Helper Functions ---


def get_project_root() -> Path:
    """Dynamically finds the project root directory."""
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    output_path_pdf = figure_dir / "fig5_fragmentation_benefit.pdf"
    output_path_eps = figure_dir / "fig5_fragmentation_benefit.eps"

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
    sns.set_theme(
        context="paper",
        style="ticks",
        rc={
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 9,
            "axes.edgecolor": "black",
            "grid.linestyle": ":",
        },
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(cm_to_inch(17.8), cm_to_inch(8)), constrained_layout=True
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
        lw=2,
        ms=5,
        ax=ax1,
        legend=False,
    )
    ax1.set_title("(A) Absolute Invasion Depth")
    ax1.set_ylabel("Mean Peak Mutant Fraction ($\langle q_{max} \\rangle$)")

    # --- Panel B: Relative Benefit of Clustering ---
    sns.lineplot(
        data=analysis_df,
        x="mean_cluster_size",
        y="relative_invasion_depth",
        hue="b_m",
        palette=palette,
        marker="o",
        lw=2,
        ms=6,
        ax=ax2,
        legend=True,
    )
    ax2.set_title("(B) Relative Benefit of Clustering")
    ax2.set_ylabel("Relative Invasion Depth")
    ax2.axhline(1.0, color="grey", linestyle="--", lw=1.5)

    # --- Shared Axis Properties and Legend ---
    for ax in axes:
        ax.set_xlabel("Mean Initial Cluster Size")
        ax.set_xscale("log")
        ax.grid(True, which="major", ls=":", axis="x")

    # Improve and place the legend inside Panel B
    leg = ax2.get_legend()
    leg.set_title("Selection Strength ($b_m$)")
    leg.set_loc("upper left")

    # --- Save Final Figure ---
    plt.savefig(output_path_pdf, bbox_inches="tight")
    plt.savefig(output_path_eps, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path_pdf} and {output_path_eps}")
    plt.close(fig)


if __name__ == "__main__":
    main()
