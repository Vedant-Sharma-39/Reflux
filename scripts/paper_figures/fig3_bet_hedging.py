# FILE: scripts/paper_figures/fig3_bet_hedging.py

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 3: Spatial Bet-Hedging."
    )
    parser.add_argument(
        "campaign_id", help="Campaign ID for the spatial bet-hedging scan."
    )
    args = parser.parse_args()

    project_root = get_project_root()
    summary_path = os.path.join(
        project_root,
        "data",
        args.campaign_id,
        "analysis",
        f"{args.campaign_id}_summary_aggregated.csv",
    )

    if not os.path.exists(summary_path):
        sys.exit(f"Error: Summary file not found: {summary_path}")

    df = pd.read_csv(summary_path)
    print(f"Loaded {len(df)} simulation results.")

    # Process Data
    df["s"] = df["b_m"] - 1.0

    # Create Plot
    sns.set_theme(style="ticks", context="talk")

    g = sns.relplot(
        data=df,
        x="k_total",
        y="avg_front_speed",
        hue="phi",
        col="s",
        row="patch_width",
        kind="line",
        marker="o",
        height=4,
        aspect=1.2,
        palette="coolwarm",
        facet_kws={"margin_titles": True, "sharey": False},
        legend="full",
    )

    g.set_xlabels(r"Switching Rate, $k_{total}$")
    g.set_ylabels("Mean Front Speed, $v$")
    g.set(xscale="log")
    g.fig.suptitle(
        "Figure 3: Front Speed in Patchy Environments (Spatial Bet-Hedging)",
        y=1.03,
        fontsize=24,
    )
    g.legend.set_title(r"Bias, $\phi$")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

    # Save Figure
    output_dir = os.path.join(project_root, "data", args.campaign_id, "analysis")
    output_path = os.path.join(output_dir, "figure3_bet_hedging.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure 3 saved to {output_path}")


if __name__ == "__main__":
    main()
