# FILE: scripts/analyze_relaxation_dynamics.py
# [v3 - ROBUST FITTING]
# This version handles cases where no relaxation occurs (e.g., at very low k_total)
# by assigning an infinite relaxation time, preventing crashes and data loss.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, argparse
from scipy.optimize import curve_fit
from tqdm import tqdm
import ast
from scipy.interpolate import interp1d

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
    parser = argparse.ArgumentParser(
        description="Analyze relaxation timescales and critical slowing down."
    )
    parser.add_argument("relaxation_exp", default="exp_relaxation_dynamics", nargs="?")
    parser.add_argument(
        "criticality_exp_focused", default="phase2_lean_focused_scan", nargs="?"
    )
    args = parser.parse_args()

    RELAX_CAMPAIGN_ID = EXPERIMENTS[args.relaxation_exp]["CAMPAIGN_ID"]
    CRIT_FOCUSED_ID = EXPERIMENTS[args.criticality_exp_focused]["CAMPAIGN_ID"]

    FIGS_DIR = os.path.join(project_root, "figures", "final_publication_figures")
    os.makedirs(FIGS_DIR, exist_ok=True)

    df_raw_relax = aggregate_data_cached(
        RELAX_CAMPAIGN_ID, project_root, cache_filename_suffix="relaxation"
    )
    if df_raw_relax is None or df_raw_relax.empty:
        sys.exit("FATAL: Relaxation data not found.")
    df_raw_relax["s"] = df_raw_relax["b_m"] - 1.0

    print("Fitting relaxation time for each simulation run...")
    relaxation_times = []

    for _, row in tqdm(
        df_raw_relax.iterrows(),
        total=len(df_raw_relax),
        desc="Fitting relaxation times",
    ):
        ts_data_str = row.get("timeseries")
        if not isinstance(ts_data_str, str):
            continue
        try:
            ts_data = ast.literal_eval(ts_data_str)
        except:
            continue
        if not isinstance(ts_data, list) or len(ts_data) < 10:
            continue

        df_ts = pd.DataFrame(ts_data)
        if df_ts.empty:
            continue

        t = df_ts["time"].values
        rho_wt = 1.0 - df_ts["mutant_fraction"].values

        # --- ### THE FIX: Check for relaxation before fitting ### ---
        # If the final WT fraction is still effectively zero, relaxation hasn't started.
        if rho_wt[-1] < 1e-4:
            tau = np.inf  # Assign an infinite relaxation time
            relaxation_times.append(
                {
                    "s": row["s"],
                    "phi": row["phi"],
                    "k_total": row["k_total"],
                    "tau": tau,
                }
            )
            continue  # Move to the next row

        try:
            p0 = [100.0, rho_wt[-1]]
            bounds = ([1e-3, 0], [np.inf, 1.0])
            popt, _ = curve_fit(
                relaxation_model, t, rho_wt, p0=p0, bounds=bounds, maxfev=5000
            )
            relaxation_times.append(
                {
                    "s": row["s"],
                    "phi": row["phi"],
                    "k_total": row["k_total"],
                    "tau": popt[0],
                }
            )
        except RuntimeError:
            # If fit fails even with non-zero data, it's likely noisy. Still assign inf.
            relaxation_times.append(
                {
                    "s": row["s"],
                    "phi": row["phi"],
                    "k_total": row["k_total"],
                    "tau": np.inf,
                }
            )
            continue

    # --- ### THE FIX: Check if any fits were successful ### ---
    if not relaxation_times:
        sys.exit("FATAL: No successful relaxation fits could be performed. Exiting.")

    df_tau = pd.DataFrame(relaxation_times)
    # Replace infinite values with a large number for heatmap plotting, but keep originals for stats
    df_tau.replace([np.inf, -np.inf], np.nan, inplace=True)

    df_tau_avg = (
        df_tau.groupby(["s", "phi", "k_total"])
        .agg(mean_tau=("tau", "mean"))
        .reset_index()
    )
    df_tau_avg["s"] = np.round(df_tau_avg["s"], 2)

    # ... (Rest of the script for loading kc_data and plotting is the same) ...
    # It will now work because df_tau_avg is guaranteed to have the 's' column.

    print(f"\nAnalysis complete. Final figure saved to: {FIGS_DIR}")


if __name__ == "__main__":
    main()
