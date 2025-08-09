# FILE: scripts/paper_figures/fig3_bet_hedging.py (Corrected Argument Parsing)

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
    # --- THE CRITICAL FIX ---
    # The script now correctly accepts TWO arguments, as passed by manage.py
    parser = argparse.ArgumentParser(description="Generate Figure 3: Relative Fitness of Spatial Bet-Hedging.")
    parser.add_argument("main_campaign_id", help="Campaign ID for the main bet-hedging scan.")
    parser.add_argument("control_campaign_id", help="Campaign ID for the non-switching control runs.")
    args = parser.parse_args()
    # --- END FIX ---
    
    project_root = get_project_root()

    path_main = os.path.join(project_root, "data", args.main_campaign_id, "analysis", f"{args.main_campaign_id}_summary_aggregated.csv")
    path_control = os.path.join(project_root, "data", args.control_campaign_id, "analysis", f"{args.control_campaign_id}_summary_aggregated.csv")
    output_path = os.path.join(project_root, "data", args.main_campaign_id, "analysis", "figure3_bet_hedging_relative_fitness.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Loading main data from: {os.path.basename(path_main)}")
    try: df_main = pd.read_csv(path_main)
    except (FileNotFoundError, pd.errors.EmptyDataError): df_main = pd.DataFrame()
    
    print(f"Loading control data from: {os.path.basename(path_control)}")
    try: df_control = pd.read_csv(path_control)
    except (FileNotFoundError, pd.errors.EmptyDataError): df_control = pd.DataFrame()

    if df_main.empty or df_control.empty:
        print(f"Warning: Missing data. Main campaign has {len(df_main)} rows, control has {len(df_control)} rows.")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Figure 3: Missing Data", ha='center', va='center')
        plt.savefig(output_path, dpi=300); sys.exit(0)

    # --- Calculate v_max_pure from Control Data ---
    df_control['s'] = df_control['b_m'] - 1.0
    # The 'width' value is the patch size for mutants. 0 means all WT.
    v_pure_wt = df_control[df_control['initial_mutant_patch_size'] == 0].set_index(['s', 'patch_width'])['avg_front_speed']
    v_pure_m = df_control[df_control['initial_mutant_patch_size'] == df_control['width']].set_index(['s', 'patch_width'])['avg_front_speed']
    v_max_pure = pd.concat([v_pure_wt, v_pure_m], axis=1).max(axis=1).rename('v_max_pure')

    # --- Calculate Relative Fitness in Main Data ---
    df_main['s'] = df_main['b_m'] - 1.0
    df_plot = df_main.join(v_max_pure, on=['s', 'patch_width'])
    df_plot['relative_fitness'] = df_plot['avg_front_speed'] / df_plot['v_max_pure']
    df_plot.dropna(subset=['relative_fitness'], inplace=True) # Drop rows where controls were missing

    sns.set_theme(style="ticks", context="talk")
    g = sns.relplot(
        data=df_plot,
        x="k_total",
        y="relative_fitness",
        hue="phi",
        col="s",
        row="patch_width",
        kind="line",
        marker="o",
        height=4, aspect=1.2,
        palette="coolwarm_r",
        facet_kws={"margin_titles": True, "sharey": True},
        legend="full",
    )

    for (row_val, col_val), ax in g.axes_dict.items():
        ax.axhline(1.0, ls=':', color='black', zorder=0, lw=2)
        panel_data = df_plot[(df_plot['patch_width'] == row_val) & np.isclose(df_plot['s'], col_val)]
        if panel_data.empty: continue
        optimum = panel_data.loc[panel_data['relative_fitness'].idxmax()]
        ax.scatter(optimum['k_total'], optimum['relative_fitness'], marker='*', s=500,
                   facecolor='gold', edgecolor='black', zorder=10)

    g.set_xlabels(r"Switching Rate, $k_{total}$")
    g.set_ylabels(r"Relative Fitness Gain, $v / v_{\max, pure}$")
    g.set(xscale="log")
    g.set_titles(row_template="patch width = {row_name}", col_template="s = {col_name:.2f}")
    g.fig.suptitle("Benefit of Bet-Hedging vs. Best Pure Strategy", y=1.03, fontsize=24)
    g.legend.set_title(r"Bias, $\phi$")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.02, 1))

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure 3 saved to {output_path}")

if __name__ == "__main__":
    main()