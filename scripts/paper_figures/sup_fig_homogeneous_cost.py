# FILE: scripts/paper_figures/sup_fig_homogeneous_cost.py

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Figure: Fitness Cost in Homogeneous Environments."
    )
    parser.add_argument("campaign_id")
    args = parser.parse_args()
    project_root = get_project_root()
    summary_path = os.path.join(
        project_root,
        "data",
        args.campaign_id,
        "analysis",
        f"{args.campaign_id}_summary_aggregated.csv",
    )
    output_path = os.path.join(
        project_root,
        "data",
        args.campaign_id,
        "analysis",
        "sup_fig_homogeneous_cost.png",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading data from: {os.path.basename(summary_path)}")
    try:
        df = pd.read_csv(summary_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame()

    if df.empty:
        print(
            f"Warning: No data for campaign '{args.campaign_id}'. Cannot generate figure."
        )
        sys.exit(0)

    df["s"] = df["b_m"] - 1.0

    # Find the non-switching (k=0) speed for each 's' to use as a baseline
    df_k0 = df[df["k_total"] == 0].set_index("s")["avg_front_speed"].rename("v_k0")

    # Join this baseline speed back to the main dataframe
    df_plot = df.join(df_k0, on="s")

    # Calculate the relative speed
    df_plot["relative_speed"] = df_plot["avg_front_speed"] / df_plot["v_k0"]

    sns.set_theme(style="whitegrid", context="talk")
    g = sns.relplot(
        data=df_plot,
        x="k_total",
        y="relative_speed",
        hue="phi",
        col="s",
        kind="line",
        palette="coolwarm_r",
        marker="o",
        height=5,
        aspect=1.1,
        facet_kws={"margin_titles": True},
    )
    g.set_xlabels(r"Switching Rate, $k_{total}$")
    g.set_ylabels("Relative Front Speed, $v / v_{k=0}$")
    g.set(xscale="log")
    g.fig.suptitle("Fitness Cost of Switching in a Homogeneous Environment", y=1.03)
    g.legend.set_title(r"Bias, $\phi$")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.02, 1))

    # Add a horizontal line at y=1 for reference
    for ax in g.axes.flatten():
        ax.axhline(1.0, ls=":", color="black", zorder=0)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nHomogeneous Cost Figure saved to {output_path}")


if __name__ == "__main__":
    main()
