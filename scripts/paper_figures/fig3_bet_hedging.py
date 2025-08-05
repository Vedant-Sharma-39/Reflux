import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots for Figure 3: Spatial Bet-Hedging."
    )
    parser.add_argument(
        "campaign_id", help="Campaign ID for the spatial bet-hedging scan."
    )
    args = parser.parse_args()

    # --- Load Data ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    summary_path = os.path.join(
        project_root,
        "data",
        args.campaign_id,
        "analysis",
        f"{args.campaign_id}_summary_aggregated.csv",
    )
    if not os.path.exists(summary_path):
        print(f"Error: Summary file not found: {summary_path}")
        return

    df = pd.read_csv(summary_path)
    print(f"Loaded {len(df)} simulation results.")

    # --- Process Data ---
    # Convert patch sequence from string back to a meaningful value if needed
    # For now, we assume 'patch_width' is a direct column. If not, this is where you'd parse it.
    df["s"] = df["b_m"] - 1.0

    # --- Create Plot ---
    sns.set_theme(style="ticks", context="talk")

    # Use seaborn's relplot to create a faceted grid of line plots
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
        facet_kws={"margin_titles": True},
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

    # --- Save Figure ---
    output_dir = os.path.join(project_root, "data", args.campaign_id, "analysis")
    output_path = os.path.join(output_dir, "figure3_bet_hedging.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure 3 saved to {output_path}")


if __name__ == "__main__":
    main()
