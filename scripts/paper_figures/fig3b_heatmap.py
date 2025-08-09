# FILE: scripts/paper_figures/fig3b_heatmap.py

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
        description="Generate Figure 3b: Heatmap of Optimal Front Speed."
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
    output_path = os.path.join(output_dir, "figure3b_optimal_speed_heatmap.png")

    print(
        f"Loading data from: {os.path.basename(summary_path)}... (This may take a moment)"
    )
    try:
        df = pd.read_csv(summary_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame()

    if df.empty:
        print(
            f"Warning: No data found for campaign '{args.campaign_id}'. Cannot generate heatmap."
        )
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Figure 3b: No Data Available", ha="center", va="center")
        plt.savefig(output_path, dpi=300)
        sys.exit(0)

    print(f"Loaded {len(df)} simulation results.")
    df["s"] = df["b_m"] - 1.0
    df["log10_k_total"] = np.log10(df["k_total"])

    # For each (s, k, patch_width), find the row with the max speed (collapsing the phi dimension)
    opt_idx = df.groupby(["s", "k_total", "patch_width"])["avg_front_speed"].idxmax()
    df_opt = df.loc[opt_idx]

    sns.set_theme(style="white", context="talk")
    g = sns.FacetGrid(
        df_opt,
        col="patch_width",
        height=6,
        aspect=1.1,
        col_wrap=3,
        sharex=True,
        sharey=True,
    )

    # Map the heatmap and then overlay the annotations
    def plot_heatmap(data, color, **kwargs):
        pivot_speed = data.pivot_table(
            index="s", columns="log10_k_total", values="avg_front_speed"
        )
        pivot_phi = data.pivot_table(index="s", columns="log10_k_total", values="phi")

        ax = plt.gca()
        sns.heatmap(
            pivot_speed,
            ax=ax,
            cmap="viridis",
            cbar_kws={"label": "Max Front Speed, v_max"},
        )
        sns.heatmap(
            pivot_phi,
            ax=ax,
            cmap="coolwarm_r",
            annot=True,
            fmt=".1f",
            cbar=False,
            annot_kws={"size": 10, "weight": "bold", "color": "white"},
        )
        ax.invert_yaxis()

    g.map_dataframe(plot_heatmap)
    g.set_axis_labels(r"log$_{10}(k_{total})$", "Selection, $s$")
    g.set_titles("patch_width = {col_name}")
    g.fig.suptitle(
        "Optimal Front Speed and Corresponding Bias ($\\phi$)", fontsize=24, y=1.03
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure 3b heatmap saved to {output_path}")


if __name__ == "__main__":
    main()
