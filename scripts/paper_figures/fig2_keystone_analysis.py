# FILE: scripts/paper_figures/fig2_keystone_analysis.py

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
    parser = argparse.ArgumentParser(description="Generate Keystone Figure 2: Linking Phase Transition to Bet-Hedging.")
    parser.add_argument("phase_diagram_campaign")
    parser.add_argument("bet_hedging_campaign")
    parser.add_argument("control_campaign")
    args = parser.parse_args()
    project_root = get_project_root()

    # --- Load All Required Datasets ---
    path_pd = os.path.join(project_root, "data", args.phase_diagram_campaign, "analysis", f"{args.phase_diagram_campaign}_summary_aggregated.csv")
    path_bh = os.path.join(project_root, "data", args.bet_hedging_campaign, "analysis", f"{args.bet_hedging_campaign}_summary_aggregated.csv")
    path_ctrl = os.path.join(project_root, "data", args.control_campaign, "analysis", f"{args.control_campaign}_summary_aggregated.csv")
    output_path = os.path.join(project_root, "data", args.bet_hedging_campaign, "analysis", "figure2_keystone_plot.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        df_pd = pd.read_csv(path_pd)
        df_bh = pd.read_csv(path_bh)
        df_ctrl = pd.read_csv(path_ctrl)
    except FileNotFoundError as e:
        print(f"Error: Missing a required data file. {e}", file=sys.stderr); sys.exit(1)

    # --- Plotting Setup ---
    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("The Biological Advantage of Bet-Hedging Emerges from the Crossover Regime", fontsize=24, y=1.03)
    palette = sns.color_palette("coolwarm_r", n_colors=3)

    # --- Panel (a): Phase Boundary Character ---
    axA = axes[0]
    df_pd['s'] = df_pd['b_m'] - 1.0
    k_slice = df_pd[np.isclose(df_pd['log10_k_total'], -1.0)]['k_total'].iloc[0]
    df_pd_slice = df_pd[df_pd['k_total'] == k_slice]
    phi_to_plot = [-1.0, -0.5, 0.0]
    
    for i, phi in enumerate(phi_to_plot):
        data = df_pd_slice[np.isclose(df_pd_slice['phi'], phi)].sort_values('s')
        sns.lineplot(ax=axA, data=data, x='s', y='avg_rho_M', label=rf"$\phi = {phi:.2f}$", color=palette[i], lw=3.5, marker='o', markersize=8)
        
    axA.set_title("(a) Phase Boundary Sharpness vs. Reversibility", fontsize=18)
    axA.set_xlabel("Selection, $s$")
    axA.set_ylabel(r"Steady-State Mutant Density, $\langle\rho_M\rangle$")
    axA.legend(title=r"Switching Bias")
    axA.grid(True, ls=':')

    # --- Panel (b): Bet-Hedging Advantage ---
    axB = axes[1]
    df_ctrl['s'] = df_ctrl['b_m'] - 1.0
    v_pure_wt = df_ctrl[df_ctrl['initial_mutant_patch_size'] == 0].set_index(['s', 'patch_width'])['avg_front_speed']
    v_pure_m = df_ctrl[df_ctrl['initial_mutant_patch_size'] == df_ctrl['width']].set_index(['s', 'patch_width'])['avg_front_speed']
    v_max_pure = pd.concat([v_pure_wt, v_pure_m], axis=1).max(axis=1).rename('v_max_pure')

    df_bh['s'] = df_bh['b_m'] - 1.0
    df_bh_plot = df_bh.join(v_max_pure, on=['s', 'patch_width'])
    df_bh_plot['relative_fitness'] = df_bh_plot['avg_front_speed'] / df_bh_plot['v_max_pure']
    df_bh_plot.dropna(subset=['relative_fitness'], inplace=True)

    s_slice = -0.25
    patch_slice = 60
    df_bh_slice = df_bh_plot[np.isclose(df_bh_plot['s'], s_slice) & (df_bh_plot['patch_width'] == patch_slice)]

    for i, phi in enumerate(phi_to_plot):
        data = df_bh_slice[np.isclose(df_bh_slice['phi'], phi)]
        sns.lineplot(ax=axB, data=data, x='k_total', y='relative_fitness', label=rf"$\phi = {phi:.2f}$", color=palette[i], lw=3.5, marker='o', markersize=8)

    axB.axhline(1.0, ls=':', color='black', zorder=0, lw=2, label="Best Pure Strategy")
    axB.set_title(f"(b) Bet-Hedging Advantage (s={s_slice}, L={patch_slice})", fontsize=18)
    axB.set_xlabel(r"Switching Rate, $k_{total}$")
    axB.set_ylabel(r"Relative Fitness Gain, $v / v_{\max, pure}$")
    axB.set_xscale('log')
    axB.legend(title=r"Switching Bias")
    axB.grid(True, which="both", ls=':')

    sns.despine(fig)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    print(f"\nKeystone Figure 2 saved to {output_path}")

if __name__ == "__main__":
    main()