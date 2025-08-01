# FILE: scripts/verify_optimal_state_analysis.py
#
# A script to verify the "characterize optimal state" analysis using existing
# data from the 'exp1_front_speed_deleterious_scan' campaign.
#
# It filters the data for a fixed selection strength (s ~ -0.4) and generates
# plots to visualize the landscape of front speed and composition as a function
# of the control parameters phi and k_total. This serves as a preview and

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, argparse

# --- Robust Path Setup & Data Aggregation ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(project_root, "src"))
    from config import EXPERIMENTS
    from data_utils import aggregate_data_cached
except (NameError, ImportError) as e:
    sys.exit(f"FATAL: Could not import configuration or helpers. Error: {e}")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18})

def main():
    parser = argparse.ArgumentParser(description="Verify optimal state analysis on existing data.")
    parser.add_argument("experiment_name", default="exp1_front_speed_deleterious_scan", nargs="?")
    args = parser.parse_args()

    # --- Load and Process Data ---
    config = EXPERIMENTS[args.experiment_name]
    CAMPAIGN_ID = config["CAMPAIGN_ID"]
    ANALYSIS_DIR = os.path.join(project_root, "data", CAMPAIGN_ID, "analysis")
    FIGS_DIR = os.path.join(project_root, "figures", CAMPAIGN_ID, "verification_analysis_s_neg0.4")
    os.makedirs(FIGS_DIR, exist_ok=True)

    df_raw = aggregate_data_cached(CAMPAIGN_ID, project_root)
    if df_raw is None or df_raw.empty: sys.exit("FATAL: No data found.")
    
    df_raw["s"] = df_raw["b_m"] - 1.0
    
    # Average over replicates
    df_avg = df_raw.groupby(["s", "phi", "k_total"]).agg(
        avg_front_speed=("avg_front_speed", "mean"),
        avg_rho_M=("avg_rho_M", "mean")
    ).reset_index().dropna()
    
    # --- Select the data slice for s ≈ -0.4 ---
    target_s = -0.4
    actual_s = df_avg['s'].unique()[np.argmin(np.abs(df_avg['s'].unique() - target_s))]
    df_plot = df_avg[np.isclose(df_avg['s'], actual_s)].copy()
    
    print(f"--- Verifying Analysis for s = {actual_s:.3f} using data from {CAMPAIGN_ID} ---")
    print(f"Figures will be saved to: {FIGS_DIR}")
    if df_plot.empty:
        sys.exit(f"FATAL: No data found for s close to {target_s}.")

    # ==========================================================================
    # VERIFICATION PLOTS
    # ==========================================================================
    
    # --- FIGURE 1: Heatmaps of observables vs. (phi, k_total) ---
    print("Generating heatmaps of observables...")
    observables = ["avg_front_speed", "avg_rho_M"]
    fig, axes = plt.subplots(1, 2, figsize=(22, 8), constrained_layout=True)
    fig.suptitle(f"Observable Landscape at s ≈ {actual_s:.3f}", fontsize=24)
    
    for ax, obs in zip(axes, observables):
        try:
            pivot = df_plot.pivot_table(index='phi', columns='k_total', values=obs)
            sns.heatmap(pivot, ax=ax, cmap='plasma', cbar_kws={'label': obs.replace("_", " ").title()})
            ax.set_title(f"{obs.replace('_', ' ').title()} vs. ($\\phi, k_{{total}}$)")
        except Exception as e:
            ax.text(0.5, 0.5, f"Could not generate heatmap.\nError: {e}", ha='center', va='center')

    plt.savefig(os.path.join(FIGS_DIR, "Fig1_Verification_Heatmaps.png"), dpi=300)
    plt.close(fig)

    # --- FIGURE 2: Line plots of observables vs. phi at different k_total ---
    print("Generating line plots of observables vs. phi...")
    k_unique = df_plot['k_total'].unique()
    k_slices = np.quantile(k_unique, [0.1, 0.5, 0.9]) if len(k_unique) > 3 else k_unique
    k_to_plot = [k_unique[np.argmin(np.abs(k_unique - k_q))] for k_q in k_slices]
    df_sliced = df_plot[df_plot['k_total'].isin(k_to_plot)]

    fig, axes = plt.subplots(1, 2, figsize=(22, 8), sharex=True, constrained_layout=True)
    fig.suptitle(f"Observable Profiles vs. $\\phi$ at s ≈ {actual_s:.3f}", fontsize=24)
    
    for i, obs in enumerate(observables):
        sns.lineplot(data=df_sliced, x='phi', y=obs, hue='k_total', palette='coolwarm',
                     marker='o', ax=axes[i], legend='full')
        axes[i].set_title(obs.replace("_", " ").title())
        axes[i].grid(True, ls='--')

    plt.savefig(os.path.join(FIGS_DIR, "Fig2_Verification_vs_Phi.png"), dpi=300)
    plt.close(fig)

    print(f"\nVerification analysis complete. Check the generated plots in {FIGS_DIR}.")

if __name__ == "__main__":
    main()