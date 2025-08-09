# FILE: scripts/paper_figures/fig3_bet_hedging.py (Excludes s=0)

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
    parser = argparse.ArgumentParser(description="Generate Figure 3: Optimal Switching in Patchy Environments.")
    parser.add_argument("campaign_id")
    args = parser.parse_args()
    project_root = get_project_root()

    summary_path = os.path.join(project_root, "data", args.campaign_id, "analysis", f"{args.campaign_id}_summary_aggregated.csv")
    output_path = os.path.join(project_root, "data", args.campaign_id, "analysis", "figure3_bet_hedging_speed.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Loading data from: {os.path.basename(summary_path)}... (This may take a moment)")
    try:
        df = pd.read_csv(summary_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame()

    if df.empty:
        print(f"Warning: No data found for campaign '{args.campaign_id}'. Cannot generate Figure 3."); sys.exit(0)

    print(f"Loaded {len(df)} simulation results.")
    df["s"] = df["b_m"] - 1.0

    # --- KEY CHANGE: Filter out the s=0.0 (neutral) case ---
    df_plot = df[~np.isclose(df['s'], 0.0)].copy()
    # --- END CHANGE ---

    df_switching = df_plot[df_plot['k_total'] > 0]
    df_control = df_plot[df_plot['k_total'] == 0]

    sns.set_theme(style="ticks", context="talk")
    g = sns.relplot(
        data=df_switching,
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

    for (row_val, col_val), ax in g.axes_dict.items():
        patch_width = row_val
        s_val = col_val
        control_data = df_control[(df_control['patch_width'] == patch_width) & np.isclose(df_control['s'], s_val)]
        if not control_data.empty:
            control_speed = control_data['avg_front_speed'].mean()
            ax.axhline(control_speed, ls=':', color='gray', zorder=0, lw=2, label=r'$k_{total}=0$')

    g.set_xlabels(r"Switching Rate, $k_{total}$")
    g.set_ylabels("Mean Front Speed, $v$")
    g.set(xscale="log")
    g.set_titles(row_template="patch_width = {row_name}", col_template="s = {col_name:.2f}")
    g.fig.suptitle("Figure 3: Optimal Switching in Patchy Environments", y=1.03, fontsize=24)
    g.legend.set_title(r"Bias, $\phi$")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.02, 1))

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure 3 saved to {output_path}")

if __name__ == "__main__":
    main()