# FILE: scripts/paper_figures/fig3_bet_hedging.py (Final Polished Version)

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
        description="Generate Figure 3: Spatial Bet-Hedging."
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
    output_dir = os.path.join(project_root, "data", args.campaign_id, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "figure3_bet_hedging.png")

    print(
        f"Loading data from: {os.path.basename(summary_path)}... (This may take a moment)"
    )
    try:
        df = pd.read_csv(summary_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame()

    if df.empty:
        print(
            f"Warning: No data found for campaign '{args.campaign_id}'. Cannot generate Figure 3."
        )
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Figure 3: No Data Available", ha="center", va="center")
        plt.savefig(output_path, dpi=300)
        sys.exit(0)

    print(f"Loaded {len(df)} simulation results.")
    df["s"] = df["b_m"] - 1.0

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
        markersize=8,
        markeredgecolor="white",
        height=4,
        aspect=1.2,
        palette="coolwarm_r",
        facet_kws={"margin_titles": True, "sharey": False},
        legend="full",
    )

    # --- Plot optima stars and theoretical lines on each facet ---
    palette = sns.color_palette("coolwarm_r", n_colors=df["phi"].nunique())
    phi_to_color = dict(zip(sorted(df["phi"].unique()), palette))

    for (row_val, col_val), ax in g.axes_dict.items():
        patch_width = row_val
        s_val = col_val

        panel_data = df[(df["patch_width"] == patch_width) & np.isclose(df["s"], s_val)]
        if panel_data.empty:
            continue

        # Add theoretical reference line: k_opt ~ v_max / L
        v_proxy = panel_data["avg_front_speed"].max()
        k_theory = v_proxy / patch_width
        ax.axvline(k_theory, ls="--", color="gray", zorder=0, lw=2)

        # Find max speed for each phi in this panel
        optima_idx = panel_data.groupby(["phi"])["avg_front_speed"].idxmax()
        panel_optima = panel_data.loc[optima_idx].sort_values(
            "avg_front_speed", ascending=False
        )

        # Plot stars stacked on the y-axis at the lowest k_total
        x_coord_star = panel_data["k_total"].min()
        for _, row in panel_optima.iterrows():
            ax.scatter(
                x=x_coord_star,
                y=row["avg_front_speed"],
                marker="*",
                s=500,
                facecolor=phi_to_color[row["phi"]],
                edgecolor="black",
                zorder=10,
                linewidth=1.5,
            )

    g.set_xlabels(r"Switching Rate, $k_{total}$")
    g.set_ylabels("Mean Front Speed, $v$")
    g.set(xscale="log")
    g.set_titles(
        row_template="patch_width = {row_name}", col_template="s = {col_name:.2f}"
    )
    g.fig.suptitle(
        "Figure 3: Optimal Switching in Patchy Environments (Spatial Bet-Hedging)",
        y=1.03,
        fontsize=24,
    )
    g.legend.set_title(r"Bias, $\phi$")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.02, 1))

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure 3 saved to {output_path}")


if __name__ == "__main__":
    main()
