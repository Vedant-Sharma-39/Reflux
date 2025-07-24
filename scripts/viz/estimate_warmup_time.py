# FILE: scripts/viz/analyze_sampling_precision.py
# [v3] The definitive diagnostic script to rigorously determine all
#      key sampling parameters for a simulation campaign.

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
WORST_CASE_PARAMS = {
    "width": 128,
    "length": 50000,
    "b_m": 0.5,
    "k_total": 0.01,
    "phi": 0.0,
}
SIM_CONFIG = {"total_run_time": 10000.0, "log_interval": 10.0}
TARGET_SEM_FRAC = 0.05  # Target a standard error that is 5% of the mean value
ROLLING_WINDOW_SIZE = 100
FIGURES_DIR = os.path.join(project_root, "figures", "sampling_precision_diagnostics")
os.makedirs(FIGURES_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")


# ==============================================================================
# 2. ANALYSIS HELPER FUNCTION
# ==============================================================================
def calculate_acf(series):
    n = len(series)
    data = np.asarray(series) - np.mean(series)
    acf = np.correlate(data, data, "full")[-n:]
    acf /= np.arange(n, 0, -1) * np.var(series)
    return acf


# ==============================================================================
# 3. MAIN EXECUTION FUNCTION
# ==============================================================================
def main():
    print(f"--- Running Sampling Precision Analysis ---")
    print(f"Worst-Case Parameters: {WORST_CASE_PARAMS}")

    # --- Run Simulation ---
    sim = GillespieSimulation(**WORST_CASE_PARAMS)
    manager = MetricsManager()
    manager.register_simulation(sim)
    tracker = FrontDynamicsTracker(sim, log_interval=SIM_CONFIG["log_interval"])
    manager.add_tracker(tracker)
    pbar = tqdm(total=SIM_CONFIG["total_run_time"], unit=" sim time")
    last_time = 0
    while sim.time < SIM_CONFIG["total_run_time"]:
        did_step, _ = sim.step()
        if not did_step:
            break
        manager.after_step()
        pbar.update(sim.time - last_time)
        last_time = sim.time
    pbar.close()

    df = tracker.get_dataframe()
    if df.empty:
        print("No data generated. Exiting.")
        return

    # --- Create the multi-panel figure ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20), constrained_layout=True)
    fig.suptitle("Definitive Sampling Parameter Analysis", fontsize=28)

    # --- PANEL A: Warmup Time Determination ---
    print("\nStep 1: Analyzing warmup time...")
    df["rho_rolling_mean"] = (
        df["mutant_fraction"].rolling(window=ROLLING_WINDOW_SIZE).mean()
    )
    ax1.plot(
        df["time"],
        df["mutant_fraction"],
        color="gray",
        alpha=0.3,
        label="Raw $\\rho_M(t)$",
    )
    ax1.plot(
        df["time"],
        df["rho_rolling_mean"],
        color="dodgerblue",
        lw=3,
        label=f"Rolling Mean",
    )
    WARMUP_TIME = 500.0
    ax1.axvline(
        WARMUP_TIME,
        color="red",
        linestyle="--",
        lw=3,
        label=f"Chosen Warmup Time = {WARMUP_TIME}",
    )
    ax1.set_title("Panel A: Visual Determination of Warmup Time")
    ax1.set_xlabel("Simulation Time")
    ax1.set_ylabel("Mutant Fraction ($\\rho_M$)")
    ax1.legend(fontsize=14)
    ax1.set_xlim(0, WARMUP_TIME * 4)
    ax1.set_ylim(0, df["mutant_fraction"].max() * 1.2)
    print(f" -> Visual analysis suggests warmup is complete by t={WARMUP_TIME}.")

    # --- PANEL B: Autocorrelation Analysis ---
    print("\nStep 2: Analyzing autocorrelation...")
    steady_state_series = df[df["time"] >= WARMUP_TIME]["mutant_fraction"].values
    acf = calculate_acf(steady_state_series)
    time_lags = np.arange(len(acf)) * SIM_CONFIG["log_interval"]
    try:
        tau_index = np.where(acf < 1 / np.e)[0][0]
        tau = time_lags[tau_index]
    except IndexError:
        tau = time_lags[-1]
    ax2.plot(time_lags, acf, "o-", label="Autocorrelation Function (ACF)")
    ax2.axhline(1 / np.e, color="red", linestyle="--", label="1/e threshold")
    ax2.axvline(tau, color="green", linestyle="--", label=f"$\\tau \\approx$ {tau:.1f}")
    ax2.set_title("Panel B: Autocorrelation of Steady-State Fluctuations")
    ax2.set_xlabel("Time Lag ($\\Delta t$)")
    ax2.set_ylabel("Autocorrelation")
    ax2.set_xlim(0, tau * 5 if tau > 0 else 100)
    ax2.legend()
    print(f" -> Autocorrelation time (tau) is approx. {tau:.2f} time units.")

    # --- PANEL C: Precision vs. Sampling Time ---
    print("\nStep 3: Analyzing precision vs. sampling time...")
    mean_rho = steady_state_series.mean()
    var_rho = steady_state_series.var(ddof=1)
    sigma_rho = np.sqrt(var_rho)
    target_sem = mean_rho * TARGET_SEM_FRAC

    sampling_durations = np.linspace(
        tau * 10, len(steady_state_series) * SIM_CONFIG["log_interval"], 100
    )
    n_eff_values = sampling_durations / tau
    sem_values = sigma_rho / np.sqrt(n_eff_values)

    ax3.plot(sampling_durations, sem_values, lw=3, label="Predicted SEM")
    ax3.axhline(
        target_sem,
        color="red",
        linestyle="--",
        label=f"Target SEM ({TARGET_SEM_FRAC*100:.0f}% of mean)",
    )

    # Find the sampling time that meets the target
    try:
        required_time_index = np.where(sem_values < target_sem)[0][0]
        required_sampling_time = sampling_durations[required_time_index]
        ax3.axvline(
            required_sampling_time,
            color="green",
            linestyle="--",
            label=f"Time to reach target $\\approx$ {required_sampling_time:.0f}",
        )
    except IndexError:
        required_sampling_time = -1
        print("Warning: Target SEM was not reached in this simulation duration.")

    ax3.set_title("Panel C: Predicted Precision vs. Sampling Duration")
    ax3.set_xlabel("Total Sampling Time (after warmup)")
    ax3.set_ylabel("Standard Error of the Mean (SEM)")
    ax3.legend()
    ax3.set_yscale("log")
    ax3.set_xscale("log")

    # --- Final Printout and Recommendation ---
    final_warmup_time = np.ceil(WARMUP_TIME / 100) * 100
    if required_sampling_time > 0:
        # Add a more modest 1.5x safety factor now that it's visualized
        final_sampling_time = np.ceil(required_sampling_time * 1.5 / 1000) * 1000
    else:  # Fallback if target not reached
        final_sampling_time = 8000.0

    num_samples = 200
    sample_interval = np.ceil(final_sampling_time / num_samples)

    print("\n--- FINAL RECOMMENDATION ---")
    print(
        f"Desired SEM: < {target_sem:.4f} (based on measured $\\sigma={sigma_rho:.3f}$)"
    )
    print("-" * 30)
    print(f"Recommended Warmup Time:   {final_warmup_time:.1f}")
    print(f"Recommended Sampling Time: {final_sampling_time:.1f}")
    print(
        f" -> For ~{num_samples} samples, set sample_interval = {sample_interval:.1f}"
    )
    print("-" * 30)
    print("Update the `base_params` in your config.py with these values:")
    print(f'  "warmup_time": {final_warmup_time},')
    print(f'  "num_samples": {num_samples},')
    print(f'  "sample_interval": {sample_interval},')

    plt.savefig(os.path.join(FIGURES_DIR, "diag_full_precision_analysis.png"), dpi=200)
    plt.close()
    print(
        f"\nDefinitive analysis plot saved to: {FIGURES_DIR}/diag_full_precision_analysis.png"
    )


if __name__ == "__main__":
    main()
