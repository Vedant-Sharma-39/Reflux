# FILE: scripts/paper_figures/fig3b_selection_comparison.py

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
    parser = argparse.ArgumentParser(description="Generate Figure 3b: Front Speed vs. Selection for fixed k_total.")
    parser.add_argument("campaign_id")
    args = parser.parse_args()
    project_root = get_project_root()

    summary_path = os.path.join(project_root, "data", args.campaign_id, "analysis", f"{args.campaign_id}_summary_aggregated.csv")
    output_path = os.path.join(project_root, "data", args.campaign_id, "analysis", "figure3b_selection_comparison.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Loading data from: {os.path.basename(summary_path)}... (This may take a moment)")
    try:
        df = pd.read_csv(summary_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame()

    if df.empty:
        print(f"Warning: No data found for campaign '{args.campaign_id}'. Cannot generate Figure 3b."); sys.exit(0)

    print(f"Loaded {len(df)} simulation results.")
    df["s"] = df["b_m"] - 1.0

    # --- Select a fixed, intermediate switching rate for the plot ---
    k_total_available = sorted(df['k_total'].unique())
    # Choose a value around the middle of the log scale, e.g., ~0.1
    k_total_slice = min(k_total_available, key=lambda x: abs(x - 0.1))
    
    df_plot = df[np.isclose(df['k_total'], k_total_slice)].copy()

    sns.set_theme(style="whitegrid", context="talk")
    g = sns.relplot(
        data=df_plot,
        x="s",
        y="avg_front_speed",
        hue="phi",
        row="patch_width",
        kind="line",
        marker="o",
        height=4,
        aspect=1.5,
        palette="coolwarm_r",
        facet_kws={"margin_titles": True},
        legend="full"
    )

    g.set_xlabels("Selection, $s$")
    g.set_ylabels("Mean Front Speed, $v$")
    g.set_titles("patch_width = {row_name}")
    g.fig.suptitle(f"Front Speed vs. Selection at Fixed Switching Rate (k_total ≈ {k_total_slice:.2f})", y=1.03)
    g.legend.set_title(r"Bias, $\phi$")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure 3b saved to {output_path}")


if __name__ == "__main__":
    main()