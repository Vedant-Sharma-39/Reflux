# FILE: scripts/paper_figures/fig3_bet_hedging.py

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Determine project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'src'))

from io.data_loader import load_aggregated_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 3: Spatial Bet-Hedging."
    )
    parser.add_argument(
        "campaign_id",
        default="bet_hedging",
        nargs="?",
        help="Campaign ID for the bet-hedging experiment (default: bet_hedging)",
    )
    args = parser.parse_args()

    # Load the aggregated data using the centralized loader
    df = load_aggregated_data(args.campaign_id, project_root)
    if df is None or df.empty:
        sys.exit(f"Could not load data for campaign '{args.campaign_id}'. Aborting.")

    print(f"Loaded {len(df)} simulation results.")
    df["s"] = df["b_m"] - 1.0

    # Find optimal k_total for each curve to mark it on the plot
    optima = df.loc[df.groupby(["s", "patch_width", "phi"])["avg_front_speed"].idxmax()]

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
        palette="coolwarm_r",
        facet_kws={"margin_titles": True, "sharey": False},
        legend="full",
    )

    # Add theoretical reference lines, format titles, and plot optima
    for (row_val, col_val), ax in g.axes_dict.items():
        patch_width = row_val
        s_val = col_val
        ax.set_title(f"patch_width = {patch_width}, s = {s_val:.2f}", fontsize=16)

        # Filter optima for the current panel
        panel_optima = optima[(optima['patch_width'] == patch_width) & (optima['s'] == s_val)]

        # Plot the optimal points on this specific axis
        sns.scatterplot(
            ax=ax,
            data=panel_optima,
            x='k_total',
            y='avg_front_speed',
            marker='*',
            s=500,
            hue='phi',
            palette='coolwarm_r',
            edgecolor='black',
            zorder=10,
            legend=False,
        )
        
        # Add vertical line for theoretical expectation k ~ v/L
        panel_data = df[(df['patch_width'] == patch_width) & (df['s'] == s_val)]
        if not panel_data.empty:
            v_proxy = panel_data['avg_front_speed'].max()
            k_theory = v_proxy / patch_width
            ax.axvline(k_theory, ls='--', color='gray', zorder=1, lw=2)

    g.set_xlabels(r"Switching Rate, $k_{total}$")
    g.set_ylabels("Mean Front Speed, $v$")
    g.set(xscale="log")
    g.fig.suptitle(
        "Figure 3: Optimal Switching in Patchy Environments (Spatial Bet-Hedging)",
        y=1.03,
        fontsize=24,
    )
    g.legend.set_title(r"Bias, $\phi$")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

    output_dir = os.path.join(project_root, "data", args.campaign_id, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "figure3_bet_hedging.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure 3 saved to {output_path}")


if __name__ == "__main__":
    main()
