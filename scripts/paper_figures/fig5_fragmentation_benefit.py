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

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_project_root():
    """Dynamically finds the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS
from src.io.data_loader import load_aggregated_data


def main():
    # --- 1. Data Loading ---
    campaign_id = EXPERIMENTS["deleterious_invasion_dynamics"]["campaign_id"]
    print(f"Generating figure from campaign: {campaign_id}")
    df_full = load_aggregated_data(campaign_id, PROJECT_ROOT)

    if df_full.empty:
        print(f"Error: Data for campaign '{campaign_id}' is empty.", file=sys.stderr)
        sys.exit(1)

    figure_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    output_path = os.path.join(figure_dir, "fig5_fragmentation_benefit.png")

    # --- 2. Data Filtering & Processing ---
    unique_sizes = sorted(df_full["initial_mutant_patch_size"].unique())
    initial_size_val = unique_sizes[len(unique_sizes) // 2]

    df = df_full[df_full["initial_mutant_patch_size"] == initial_size_val].copy()

    if df.empty:
        print(
            f"Error: No data found for initial_size={initial_size_val}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- 3. Analysis ---
    df_extinctions = df[df["outcome"] == "extinction"].copy()
    if df_extinctions.empty:
        print(
            "Warning: No extinction data found; cannot generate plot.", file=sys.stderr
        )
        return

    fragments_df = (
        df.groupby(["correlation_length", "b_m"])
        .agg(mean_fragments=("num_fragments", "mean"))
        .reset_index()
    )
    fragments_df["mean_cluster_size"] = (
        initial_size_val / fragments_df["mean_fragments"]
    )

    depth_df = (
        df_extinctions.groupby(["correlation_length", "b_m"])
        .agg(mean_q_max=("q_at_outcome", "mean"))
        .reset_index()
    )

    analysis_df = pd.merge(
        depth_df, fragments_df, on=["correlation_length", "b_m"], how="left"
    )

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

    # --- 4. Plotting: The Final 1x2 Figure ---
    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    ax1, ax2 = axes

    fig.suptitle(
        f"Clustering Boosts Survival of Disadvantaged Mutants (Initial # = {int(initial_size_val)})",
        fontsize=24,
        y=1.02,
    )

    palette = sns.color_palette("viridis", n_colors=len(analysis_df["b_m"].unique()))

    # Panel A: Absolute Invasion Depth
    sns.lineplot(
        data=analysis_df,
        x="mean_cluster_size",
        y="mean_q_max",
        hue="b_m",
        palette=palette,
        marker="o",
        lw=3,
        ms=9,
        ax=ax1,
        legend=False,
    )
    ax1.set_title("(A) Absolute Invasion Depth", fontsize=18)
    ax1.set_ylabel("Mean Max q Reached (before extinction)", fontsize=16)

    # Panel B: Relative Invasion Depth
    sns.lineplot(
        data=analysis_df,
        x="mean_cluster_size",
        y="relative_invasion_depth",
        hue="b_m",
        palette=palette,
        marker="o",
        lw=3.5,
        ms=10,
        ax=ax2,
    )
    ax2.set_title("(B) Relative Benefit of Clustering", fontsize=18)
    ax2.set_ylabel(
        "Relative Invasion Depth\n(Normalized to Most Fragmented State)", fontsize=16
    )
    ax2.axhline(1.0, color="grey", linestyle="--", lw=2)

    # --- Final Touches for both axes ---
    for ax in axes:
        ax.set_xlabel("Mean Initial Cluster Size", fontsize=16)
        ax.set_xscale("log")
        ax.grid(True, which="both", ls=":", axis="x")

    # Create a single, shared legend below the plots
    handles, labels = ax2.get_legend_handles_labels()
    ax2.get_legend().remove()  # Remove the default legend from the plot
    fig.legend(
        handles,
        labels,
        title="Selection, $b_m$",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(labels),
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
