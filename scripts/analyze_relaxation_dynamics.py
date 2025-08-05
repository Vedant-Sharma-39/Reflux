# FILE: scripts/analyze_relaxation_dynamics.py
#
# Analyzes the output of the 'exp_relaxation_dynamics' campaign.
# It fits an exponential model to the time series of the wild-type
# fraction to extract the relaxation time (tau) and visualizes how this
# timescale depends on s and k_total.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, argparse
from scipy.optimize import curve_fit
from tqdm import tqdm
import ast

# --- Robust Path Setup & Data Aggregation ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))

from config import EXPERIMENTS
from data_utils import aggregate_data_cached


# --- Exponential Model for Fitting ---
def relaxation_model(t, tau, rho_wt_final):
    """Models the rise of WT fraction from 0: rho_WT(t) = rho_WT_final * (1 - exp(-t/tau))."""
    return rho_wt_final * (1 - np.exp(-t / tau))


def main():
    parser = argparse.ArgumentParser(description="Analyze relaxation timescales.")
    parser.add_argument("experiment_name", default="exp_relaxation_dynamics", nargs="?")
    args = parser.parse_args()

    CAMPAIGN_ID = EXPERIMENTS[args.experiment_name]["CAMPAIGN_ID"]
    FIGS_DIR = os.path.join(project_root, "figures", CAMPAIGN_ID)
    os.makedirs(FIGS_DIR, exist_ok=True)

    df_raw = aggregate_data_cached(CAMPAIGN_ID, project_root)
    if df_raw is None or df_raw.empty:
        sys.exit("FATAL: No data found.")
    df_raw["s"] = df_raw["b_m"] - 1.0

    print("Fitting relaxation time for each simulation run...")
    relaxation_times = []

    for _, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
        ts_data = row.get("timeseries")
        if isinstance(ts_data, str):
            try:
                ts_data = ast.literal_eval(ts_data)
            except:
                continue
        if not isinstance(ts_data, list) or len(ts_data) < 10:
            continue

        df_ts = pd.DataFrame(ts_data)
        if df_ts.empty:
            continue

        # We fit the rise of the WT fraction from an initial state of rho_M=1 (rho_WT=0)
        t = df_ts["time"].values
        rho_wt = 1.0 - df_ts["mutant_fraction"].values

        try:
            # Guess: tau=100, rho_wt_final=final value
            p0 = [100.0, rho_wt[-1]]
            popt, _ = curve_fit(
                relaxation_model, t, rho_wt, p0=p0, bounds=([1e-3, 0], [np.inf, 1.0])
            )
            tau = popt[0]
            relaxation_times.append(
                {
                    "s": row["s"],
                    "phi": row["phi"],
                    "k_total": row["k_total"],
                    "tau": tau,
                }
            )
        except RuntimeError:
            continue

    df_tau = pd.DataFrame(relaxation_times)
    df_tau_avg = (
        df_tau.groupby(["s", "phi", "k_total"])
        .agg(mean_tau=("tau", "mean"))
        .reset_index()
    )

    print("Generating heatmaps of relaxation time...")
    for phi in sorted(df_tau_avg["phi"].unique()):
        df_slice = df_tau_avg[df_tau_avg["phi"] == phi]

        fig, ax = plt.subplots(figsize=(14, 10))
        try:
            pivot = df_slice.pivot_table(
                index="s", columns="k_total", values="mean_tau"
            ).sort_index(ascending=False)
            sns.heatmap(
                pivot,
                ax=ax,
                cmap="magma",
                norm=plt.matplotlib.colors.LogNorm(),
                cbar_kws={"label": "Mean Relaxation Time ($\\tau$)"},
            )
            ax.set_title(f"Relaxation Timescale Landscape for $\\phi = {phi:.2f}$")

            plot_filename = f"Fig_Relaxation_Time_phi_{phi:.2f}.png".replace(".", "p")
            plt.savefig(
                os.path.join(FIGS_DIR, plot_filename), dpi=300, bbox_inches="tight"
            )
            plt.close(fig)
        except Exception as e:
            print(f"Could not generate heatmap for phi={phi}. Error: {e}")
            plt.close(fig)

    print(f"\nAnalysis complete. Figures saved to: {FIGS_DIR}")


if __name__ == "__main__":
    main()
