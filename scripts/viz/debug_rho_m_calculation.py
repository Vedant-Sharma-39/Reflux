# FILE: scripts/viz/analyze_steady_state_fluctuations.py
# A diagnostic script to rigorously check if the system reaches a stable
# steady state by analyzing the time series of the mutant fraction (rho_M).
# [FIXED] Corrected Matplotlib mathtext syntax in plot labels.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from tqdm import tqdm

# --- Robust Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "src"))

from linear_model import GillespieSimulation
from metrics import MetricsManager, FrontDynamicsTracker

# ==============================================================================
# 1. DEFINE PARAMETERS FOR THE ANALYSIS
# ==============================================================================
# Choose an interesting parameter set to test (e.g., one near a transition)
PARAMS_TO_TEST = {
    "width": 128,
    "length": 50000,  # Long enough to not hit the end
    "b_m": 0.8,
    "k_total": 0.5,
    "phi": -0.5,
}

# Long run to ensure we see the steady state
SIM_CONFIG = {
    "total_run_time": 4000.0,
    "num_replicates": 5,
    "log_interval": 10.0,  # Log data every 10 time units
}

# Analysis parameters
STEADY_STATE_START_TIME = 2000.0  # Time after which we consider it "steady state"
ROLLING_WINDOW_SIZE = 50  # Number of data points for rolling stats

# Visualization settings
FIGURES_DIR = os.path.join(project_root, "figures", "steady_state_diagnostics")
os.makedirs(FIGURES_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")


# ==============================================================================
# 2. MAIN EXECUTION FUNCTION
# ==============================================================================
def main():
    print(f"--- Analyzing Steady State for Parameters ---")
    print(PARAMS_TO_TEST)

    all_replicates_data = []

    # --- Run Simulations ---
    for rep in range(SIM_CONFIG["num_replicates"]):
        print(f"  Running replicate {rep + 1}/{SIM_CONFIG['num_replicates']}...")

        sim = GillespieSimulation(**PARAMS_TO_TEST)
        manager = MetricsManager()
        manager.register_simulation(sim)
        tracker = FrontDynamicsTracker(sim, log_interval=SIM_CONFIG["log_interval"])
        manager.add_tracker(tracker)

        # Use tqdm for a progress bar based on simulation time
        pbar = tqdm(total=SIM_CONFIG["total_run_time"], unit=" sim time")
        last_time = 0
        while sim.time < SIM_CONFIG["total_run_time"]:
            did_step, _ = sim.step()
            if not did_step:
                print("  Warning: Simulation stalled.")
                break
            manager.after_step()
            pbar.update(sim.time - last_time)
            last_time = sim.time
        pbar.close()

        rep_df = tracker.get_dataframe()
        rep_df["replicate"] = rep
        all_replicates_data.append(rep_df)

    if not all_replicates_data:
        print("No data was generated. Exiting.")
        return

    full_df = pd.concat(all_replicates_data, ignore_index=True)

    # --- Generate Diagnostic Plots ---

    # Plot 1: Full Time Series for all replicates
    print("\nGenerating Plot 1: Full Time Series...")
    plt.figure(figsize=(16, 8))
    sns.lineplot(
        data=full_df,
        x="time",
        y="mutant_fraction",
        hue="replicate",
        palette="viridis",
        legend=None,
    )
    plt.axvline(
        STEADY_STATE_START_TIME,
        color="r",
        linestyle="--",
        lw=3,
        label=f"Steady State Window Start ({STEADY_STATE_START_TIME})",
    )
    plt.title("Time Evolution of Mutant Fraction ($\\rho_M$) for All Replicates")
    plt.xlabel("Simulation Time")
    plt.ylabel("Mutant Fraction ($\\rho_M$)")  # [FIXED]
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(
        os.path.join(FIGURES_DIR, "diag_1_full_timeseries.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # Get data from the steady state window
    steady_state_df = full_df[full_df["time"] >= STEADY_STATE_START_TIME].copy()

    # Plot 2: Rolling Mean and Standard Deviation for a single replicate
    print("Generating Plot 2: Rolling Statistics...")
    rep0_df = steady_state_df[steady_state_df["replicate"] == 0].copy()
    if not rep0_df.empty:
        rep0_df["rho_rolling_mean"] = (
            rep0_df["mutant_fraction"].rolling(window=ROLLING_WINDOW_SIZE).mean()
        )
        rep0_df["rho_rolling_std"] = (
            rep0_df["mutant_fraction"].rolling(window=ROLLING_WINDOW_SIZE).std()
        )

        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.set_xlabel("Simulation Time")
        ax1.set_ylabel("Mutant Fraction ($\\rho_M$)", color="tab:blue")  # [FIXED]
        ax1.plot(
            rep0_df["time"],
            rep0_df["mutant_fraction"],
            color="gray",
            alpha=0.5,
            label="Raw $\\rho_M$",
        )
        ax1.plot(
            rep0_df["time"],
            rep0_df["rho_rolling_mean"],
            color="tab:blue",
            lw=3,
            label=f"Rolling Mean (w={ROLLING_WINDOW_SIZE})",
        )
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Rolling Std. Dev.", color="tab:red")
        ax2.plot(
            rep0_df["time"],
            rep0_df["rho_rolling_std"],
            color="tab:red",
            lw=3,
            label=f"Rolling Std. Dev. (w={ROLLING_WINDOW_SIZE})",
        )
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.legend(loc="upper right")

        plt.title("Rolling Statistics of $\\rho_M$ in Steady State (Replicate 0)")
        plt.savefig(
            os.path.join(FIGURES_DIR, "diag_2_rolling_stats.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    # Plot 3: Distribution of rho_M values in the steady state
    print("Generating Plot 3: Steady State Distribution...")
    plt.figure(figsize=(10, 7))
    sns.histplot(data=steady_state_df, x="mutant_fraction", kde=True, bins=30)

    mean_rho = steady_state_df["mutant_fraction"].mean()
    std_rho = steady_state_df["mutant_fraction"].std()

    plt.axvline(
        mean_rho, color="r", linestyle="--", lw=3, label=f"Mean = {mean_rho:.3f}"
    )
    plt.title(
        f"Distribution of $\\rho_M$ in Steady State (t > {STEADY_STATE_START_TIME})"
    )
    plt.xlabel("Mutant Fraction ($\\rho_M$)")  # [FIXED]
    plt.legend()

    # Add text box for statistics
    stats_text = (
        f"Mean = {mean_rho:.4f}\nStd Dev = {std_rho:.4f}\nVariance = {std_rho**2:.4f}"
    )
    plt.text(
        0.05,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.savefig(
        os.path.join(FIGURES_DIR, "diag_3_steadystate_distribution.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print("\nSteady state diagnostics complete.")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
