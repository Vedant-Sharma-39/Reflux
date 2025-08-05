# FILE: scripts/analyze_perturbation_resilience.py
#
# [DEFINITIVELY CORRECTED v3 - FINAL]
# This script correctly analyzes and visualizes the perturbation resilience experiment,
# producing the final, publication-quality figure. It fixes previous plotting
# artifacts by correctly calculating the mean and SEM over all replicates and
# displaying the full, correct time evolution of the system's response.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import json
from tqdm import tqdm

# --- Robust Path Setup & Data Aggregation ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(project_root, "src"))
    from config import EXPERIMENTS
    from data_utils import aggregate_data_cached
except (NameError, ImportError) as e:
    sys.exit(f"FATAL: Could not import configuration or helpers. Error: {e}")

plt.style.use("seaborn-v0_8-whitegrid")
# Use a cleaner, simpler style for the final plot
plt.rcParams.update(
    {
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 22,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze perturbation resilience results."
    )
    parser.add_argument(
        "experiment_name", default="perturbation_resilience_v1", nargs="?"
    )
    args = parser.parse_args()

    # --- Setup Directories & Load Experiment Config ---
    config = EXPERIMENTS[args.experiment_name]
    CAMPAIGN_ID = config["CAMPAIGN_ID"]
    ANALYSIS_DIR = os.path.join(project_root, "data", CAMPAIGN_ID, "analysis")
    FIGS_DIR = os.path.join(project_root, "figures", CAMPAIGN_ID)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR, exist_ok=True)

    try:
        any_sim_set = next(iter(config["SIM_SETS"].values()))
        pulse_start_time = any_sim_set["base_params"]["pulse_start_time"]
        pulse_duration = any_sim_set["base_params"]["pulse_duration"]
        pulse_end_time = pulse_start_time + pulse_duration
    except (KeyError, StopIteration):
        sys.exit("FATAL: Could not find pulse parameters in the config file.")

    # --- 1. Load and Process Data ---
    print(f"--- Analyzing Resilience for Campaign: {CAMPAIGN_ID} ---")
    df_raw = aggregate_data_cached(CAMPAIGN_ID, project_root)
    if df_raw is None or df_raw.empty or "timeseries" not in df_raw.columns:
        sys.exit("FATAL: No valid timeseries data found. Please run aggregation first.")

    # --- 2. Unpack Timeseries Data ---
    print("Unpacking timeseries data...")
    all_ts_data = []
    df_raw = df_raw.dropna(subset=["timeseries"]).copy()
    for _, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
        try:
            ts_list = (
                json.loads(row["timeseries"])
                if isinstance(row["timeseries"], str)
                else row["timeseries"]
            )
            if not ts_list or not isinstance(ts_list, list):
                continue

            temp_df = pd.DataFrame(ts_list)
            temp_df["k_total_base"] = row["k_total"]
            temp_df["replicate_id"] = row["replicate_id"]
            all_ts_data.append(temp_df)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    if not all_ts_data:
        sys.exit("FATAL: Failed to parse any valid timeseries data.")

    df_long = pd.concat(all_ts_data, ignore_index=True)

    # --- 3. Label Regimes Robustly ---
    k_to_regime = {
        0.10: "A: Geometry-Dominated (s = -0.45)",
        0.08: "B: Most Fragile (s = -0.25)",
        0.13: "C: Drift-Dominated (s = -0.10)",
    }
    df_long["regime"] = df_long["k_total_base"].map(k_to_regime)

    # --- 4. Calculate Statistics for Plotting ---
    print("Calculating mean and SEM across replicates...")
    df_plot = (
        df_long.groupby(["regime", "time"])
        .agg(rho_mean=("mutant_fraction", "mean"), rho_sem=("mutant_fraction", "sem"))
        .reset_index()
    )

    # --- 5. Generate the Final, Corrected Resilience Plot ---
    print("Generating final resilience plot...")

    regime_order = [
        "A: Geometry-Dominated (s = -0.45)",
        "B: Most Fragile (s = -0.25)",
        "C: Drift-Dominated (s = -0.10)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
    fig.suptitle(
        "Population Resilience to Perturbation Reveals Hidden Structural Fragility",
        fontsize=28,
        y=1.0,
    )

    for i, regime in enumerate(regime_order):
        ax = axes[i]
        regime_data = df_plot[df_plot["regime"] == regime]

        # Plot the mean line
        ax.plot(
            regime_data["time"],
            regime_data["rho_mean"],
            color="black",
            lw=2.5,
            label="Mean",
        )

        # Plot the shaded confidence interval (mean +/- SEM)
        ax.fill_between(
            regime_data["time"],
            regime_data["rho_mean"] - regime_data["rho_sem"],
            regime_data["rho_mean"] + regime_data["rho_sem"],
            color="black",
            alpha=0.3,
            linewidth=0,
        )

        # Highlight the perturbation pulse
        ax.axvspan(
            pulse_start_time,
            pulse_end_time,
            color="black",
            alpha=0.15,
            zorder=0,
            label="Perturbation Pulse",
        )

        # Formatting and Annotations
        ax.set_title(regime, size=20)
        ax.set_xlabel("Simulation Time")
        ax.legend(loc="upper right")
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, color="gray")

    axes[0].set_ylabel("Avg. Deleterious Mutant Fraction ($\\rho_M$)")
    axes[0].set_ylim(-0.05, 1.05)
    for ax in axes:
        ax.set_xlim(0, 4000)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path = os.path.join(FIGS_DIR, "Fig_Resilience_to_Perturbation_FINAL.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nAnalysis complete. Final resilience plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
