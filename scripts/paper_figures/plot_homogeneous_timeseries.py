# FILE: scripts/paper_figures/plot_relaxation_theory_comparison.py
#
# Compares the simulated relaxation dynamics with the solution of the
# mean-field ordinary differential equation (ODE).

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import gzip
from scipy.integrate import odeint

def get_project_root():
    """Dynamically finds the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# --- Add project root to path ---
PROJECT_ROOT = get_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import EXPERIMENTS
# We reuse the timeseries loading function from the previous script
from scripts.paper_figures.plot_homogeneous_timeseries import load_timeseries_for_params

def relaxation_ode(f, t, s_eff, k_total, phi):
    """
    Defines the ODE for the mean-field model of mutant fraction dynamics.
    df/dt = s_eff*f*(1-f) - k_wt_m*f + k_m_wt*(1-f)
    """
    k_wt_m = (k_total / 2.0) * (1.0 - phi)
    k_m_wt = (k_total / 2.0) * (1.0 + phi)
    
    selection_term = s_eff * f * (1.0 - f)
    switching_term = k_m_wt * (1.0 - f) - k_wt_m * f
    
    return selection_term + switching_term

def main():
    # --- Data Loading ---
    ts_campaign_id = EXPERIMENTS["homogeneous_timeseries"]["campaign_id"]
    pd_campaign_id = EXPERIMENTS["phase_diagram"]["campaign_id"]

    ts_summary_path = os.path.join(PROJECT_ROOT, "data", ts_campaign_id, "analysis", f"{ts_campaign_id}_summary_aggregated.csv")
    s_eff_summary_path = os.path.join(PROJECT_ROOT, "data", pd_campaign_id, "analysis", "s_eff_summary.csv")
    figure_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    print("Loading timeseries summary data...")
    df_summary = pd.read_csv(ts_summary_path)
    print("Loading s_eff summary data...")
    df_s_eff = pd.read_csv(s_eff_summary_path)

    # --- Setup Plotting ---
    b_m_vals = sorted(df_summary["b_m"].unique())
    phi_vals = sorted(df_summary["phi"].unique())
    k_total_vals = sorted(df_summary["k_total"].unique())
    initial_sizes = sorted(df_summary["initial_mutant_patch_size"].unique())
    width = df_summary["width"].iloc[0]
    palette = sns.color_palette("coolwarm", n_colors=len(initial_sizes))

    # --- Create one figure per phi value ---
    for phi in phi_vals:
        print(f"\nGenerating theory comparison plot for phi = {phi:.2f}...")
        fig, axes = plt.subplots(
            len(b_m_vals), len(k_total_vals), 
            figsize=(8 * len(k_total_vals), 6 * len(b_m_vals)),
            constrained_layout=True, sharex=True, sharey=True
        )
        fig.suptitle(f"Relaxation Dynamics vs. Mean-Field Theory (φ = {phi:.2f})", fontsize=28, y=1.04)

        for i, b_m in enumerate(b_m_vals):
            for j, k_total in enumerate(k_total_vals):
                ax = axes[i, j]
                ax.set_title(f"b$_m$={b_m:.2f}, k$_t$={k_total:.2g}", fontsize=16)

                # Find the corresponding s_eff for this (b_m, phi)
                s_eff_row = df_s_eff[(np.isclose(df_s_eff['b_m'], b_m)) & (np.isclose(df_s_eff['phi'], phi))]
                s_eff = s_eff_row['s_eff'].iloc[0] if not s_eff_row.empty else 0.0

                for k, size in enumerate(initial_sizes):
                    # --- Plot Simulation Data (Mean Trace) ---
                    query = {"b_m": b_m, "phi": phi, "k_total": k_total, "initial_mutant_patch_size": size}
                    timeseries_list = load_timeseries_for_params(df_summary, ts_campaign_id, PROJECT_ROOT, query)
                    if not timeseries_list: continue
                    
                    all_times = np.concatenate([df_ts["time"].values for df_ts in timeseries_list])
                    common_time = np.linspace(0, np.percentile(all_times, 95), 200)
                    interpolated_rhos = [np.interp(common_time, df_ts["time"], df_ts["mutant_fraction"]) for df_ts in timeseries_list]
                    mean_rho = np.mean(interpolated_rhos, axis=0)
                    
                    ax.plot(common_time, mean_rho, color=palette[k], lw=3, label=f"Sim (Initial: {size/width:.0%})")
                    
                    # --- Solve and Plot Theoretical ODE ---
                    f0 = size / width # Initial condition
                    ode_solution = odeint(relaxation_ode, f0, common_time, args=(s_eff, k_total, phi))
                    ax.plot(common_time, ode_solution, color='black', linestyle='--', lw=2.5, label=f"Theory (s_eff={s_eff:.2f})")

                # De-duplicate legend
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), fontsize=10)

                ax.grid(True, linestyle=":")
                ax.set_ylim(-0.05, 1.05)
                if i == len(b_m_vals) - 1: ax.set_xlabel("Time", fontsize=14)
                if j == 0: ax.set_ylabel(r"Mutant Fraction, $\langle\rho_M\rangle$", fontsize=14)

        output_filename = os.path.join(figure_dir, f"relaxation_theory_comparison_phi_{phi:.2f}.png")
        plt.savefig(output_filename, dpi=150, bbox_inches="tight")
        print(f"  -> Saved figure to {output_filename}")
        plt.close(fig)

if __name__ == "__main__":
    main()