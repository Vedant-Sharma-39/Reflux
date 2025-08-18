"""
Supplementary Figure: The Impact of Initial Spatial Distribution on Fixation

This script analyzes the 'initial_distribution_selection' experiment. It generates
a figure to illustrate how the initial spatial clustering (fragmentation) of a
fixed number of mutants affects their probability of fixation.

The key visualization connects three quantities:
1.  The Tunable Knob: The correlation length of the generative random field.
2.  The Physical Readout: The resulting number of initial mutant fragments.
3.  The Evolutionary Outcome: The probability of mutant fixation.

The analysis is performed for both disadvantaged (b_m < 1) and advantaged (b_m > 1)
mutants to show how selection interacts with the initial spatial structure.
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

# Imports from your project structure
from src.config import EXPERIMENTS
from src.io.data_loader import load_aggregated_data


def main():
    # --- 1. Data Loading ---
    try:
        campaign_id = EXPERIMENTS["initial_distribution_selection"]["campaign_id"]
    except KeyError:
        print(
            "Error: 'initial_distribution_selection' not found in config.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Generating Initial Distribution figure from campaign: {campaign_id}")
    df = load_aggregated_data(campaign_id, PROJECT_ROOT)

    if df.empty:
        print(
            f"Error: Data for campaign '{campaign_id}' is empty or not found.",
            file=sys.stderr,
        )
        print("Please run 'make consolidate' for this campaign first.")
        sys.exit(1)

    figure_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    output_path = os.path.join(figure_dir, "sup_fig_initial_distribution.png")

    # --- 2. Data Processing ---
    # Convert the 'outcome' column to a binary 'is_fixation' column for aggregation.
    df["is_fixation"] = (df["outcome"] == "fixation").astype(int)

    # Group by the experimental parameters and calculate key statistics.
    # We need the mean fixation probability, its standard error, and the mean number of fragments.
    prob_df = (
        df.groupby(["correlation_length", "b_m"])
        .agg(
            fixation_prob=("is_fixation", "mean"),
            # Standard error of a proportion = sqrt(p*(1-p)/n)
            # A robust way is SEM = std/sqrt(n)
            fixation_prob_err=(
                "is_fixation",
                lambda x: np.std(x, ddof=1) / np.sqrt(len(x)),
            ),
            mean_fragments=("num_fragments", "mean"),
        )
        .reset_index()
    )

    # --- 3. Plotting ---
    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(
        1, 2, figsize=(18, 7), constrained_layout=True, sharey=True
    )
    fig.suptitle(
        "Fixation Probability Increases with Initial Mutant Clustering",
        fontsize=24,
        y=1.07,
    )

    b_m_vals = sorted(prob_df["b_m"].unique())
    if len(b_m_vals) < 2:
        print(
            f"Warning: Expected at least 2 b_m values, but found {len(b_m_vals)}. Plot may be incomplete."
        )
        # Adjust layout if only one panel is needed
        if len(b_m_vals) == 1:
            axes = [axes[0]]
            fig.set_size_inches(9, 7)

    titles = [
        f"(A) Disadvantaged Mutants ($b_m = {b_m_vals[0]}$)",
        f"(B) Advantaged Mutants ($b_m = {b_m_vals[1]}$)",
    ]

    for i, (ax, b_m_val) in enumerate(zip(axes, b_m_vals)):
        df_panel = prob_df[np.isclose(prob_df["b_m"], b_m_val)].sort_values(
            "correlation_length"
        )

        # --- Primary Axis: Fixation Probability ---
        color_prob = "royalblue"
        ax.errorbar(
            df_panel["correlation_length"],
            df_panel["fixation_prob"],
            yerr=df_panel["fixation_prob_err"],
            fmt="-o",
            color=color_prob,
            label="Fixation Probability",
            capsize=5,
            lw=3,
            ms=8,
            zorder=10,
        )
        ax.set_xlabel("GRF Correlation Length (Clustering)", fontsize=16)
        if i == 0:
            ax.set_ylabel("Fixation Probability", color=color_prob, fontsize=16)
        ax.tick_params(axis="y", labelcolor=color_prob)
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, which="both", ls=":", axis="x")

        # --- Secondary Axis: Number of Fragments ---
        ax2 = ax.twinx()
        color_frag = "crimson"
        ax2.plot(
            df_panel["correlation_length"],
            df_panel["mean_fragments"],
            "s--",
            color=color_frag,
            label="Mean Initial Fragments",
            lw=2.5,
            ms=7,
        )
        if i == len(axes) - 1:
            ax2.set_ylabel("Mean Initial Fragments", color=color_frag, fontsize=16)
        ax2.tick_params(axis="y", labelcolor=color_frag)
        # Ensure y-axis for fragments is also log scale to show the relationship clearly
        ax2.set_yscale("log")

        ax.set_title(titles[i], fontsize=18)

    # --- Create a single, clean legend below the figure ---
    lines, labels = axes[0].get_legend_handles_labels()
    lines2, labels2 = axes[0].twinx().get_legend_handles_labels()
    fig.legend(
        lines + lines2,
        labels + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=2,
        fontsize=14,
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")


if __name__ == "__main__":
    main()
