# FILE: scripts/viz/debug_correlation_fitting.py
#
# A dedicated script to interactively debug the fitting of the correlation
# function, g(r). This allows for visual inspection of the fit for specific
# parameter sets and development of a more robust fitting strategy.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import argparse
from scipy.optimize import curve_fit
from scipy.stats import linregress
import collections

# --- Robust Path and Config Import ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "src"))
try:
    from config import EXPERIMENTS
except ImportError:
    print("FATAL: Could not import EXPERIMENTS from src/config.py.")
    sys.exit(1)

# --- Constants for Fitting ---
XI_UPPER_BOUND = 512
GINF_ESTIMATION_RANGE = (40, 60)
MAX_R_FOR_PLOTTING = 128


# --- Fitting Models ---
def physical_decay_model(r, C, xi, g_inf):
    """Exponential decay model for g(r)."""
    return C * np.exp(-r / xi) + g_inf


# ==============================================================================
# The NEW, Robust Fitting Strategy
# ==============================================================================
def robust_fit_g_r(r_data, g_data):
    """
    A more robust fitting function for g(r) that uses pre-analysis to handle
    different physical regimes.

    Returns:
        tuple: (C, xi, g_inf, status_string)
    """
    if len(r_data) < 5:
        return 0, 1.0, np.mean(g_data), "no_fit_short_data"

    # --- Pre-analysis Step 1: Check for a decay signal ---
    # Look for a negative slope in the initial part of the data
    initial_decay_mask = (r_data > 0) & (r_data <= 10)
    has_decay_signal = False
    if np.sum(initial_decay_mask) > 3:
        # Use linregress on log-y vs x for exponential decay
        slope_initial, _, r_val, _, _ = linregress(
            r_data[initial_decay_mask],
            np.log(np.maximum(1e-9, g_data[initial_decay_mask])),
        )
        # A significant negative slope indicates decay
        if slope_initial < -0.01 and r_val**2 > 0.1:
            has_decay_signal = True

    # --- Conditional Logic Based on Pre-analysis ---

    # Case 1: No significant decay signal detected
    if not has_decay_signal:
        mean_g = np.mean(g_data)
        # Subcase 1a: Highly correlated (flat g(r) near 1)
        if mean_g > 0.5:
            return 0.0, XI_UPPER_BOUND, 1.0, "manual_correlated"
        # Subcase 1b: Uncorrelated (noisy g(r) near 0)
        else:
            return 0.0, 1.0, 0.0, "manual_uncorrelated"

    # Case 2: Decay signal detected, proceed with curve_fit
    try:
        # Estimate g_inf from the tail
        tail_mask = (r_data >= GINF_ESTIMATION_RANGE[0]) & (
            r_data <= GINF_ESTIMATION_RANGE[1]
        )
        g_inf_guess = np.mean(g_data[tail_mask]) if np.any(tail_mask) else g_data[-1]

        # Estimate C from the initial point
        g_initial = g_data[r_data > 0][0]
        C_guess = max(0, g_initial - g_inf_guess)

        p0 = [C_guess, 10.0, g_inf_guess]
        bounds = ([0, 1.0, 0], [1.0, XI_UPPER_BOUND, 1.0])

        fit_mask = r_data <= MAX_R_FOR_PLOTTING
        popt, _ = curve_fit(
            physical_decay_model,
            r_data[fit_mask],
            g_data[fit_mask],
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )
        return popt[0], popt[1], popt[2], "fit_success"
    except (RuntimeError, ValueError) as e:
        # Fallback if curve_fit fails despite signal
        return 0.0, 1.0, np.mean(g_data), f"fit_failed_fallback ({e})"


def main():
    parser = argparse.ArgumentParser(
        description="Debug g(r) fitting for a single parameter set."
    )
    parser.add_argument(
        "experiment_name", choices=EXPERIMENTS.keys(), help="Experiment to analyze."
    )
    parser.add_argument("--bm", type=float, required=True, help="b_m value.")
    parser.add_argument("--phi", type=float, required=True, help="phi value.")
    parser.add_argument("--k", type=float, required=True, help="k_total value.")
    args = parser.parse_args()

    # --- 1. Load and Find Data ---
    CAMPAIGN_ID = EXPERIMENTS[args.experiment_name]["CAMPAIGN_ID"]
    data_file = os.path.join(project_root, "data", f"{CAMPAIGN_ID}_aggregated.csv")

    if not os.path.exists(data_file):
        print(f"FATAL: Aggregated data file not found at {data_file}")
        print("Please run the aggregation part of the main analysis script first.")
        sys.exit(1)

    print("Loading and processing data...")
    df_raw = pd.read_csv(data_file)

    # Calculate average g(r) on the fly for the selected parameters
    mask = (
        (np.isclose(df_raw["b_m"], args.bm))
        & (np.isclose(df_raw["phi"], args.phi))
        & (np.isclose(df_raw["k_total"], args.k))
    )
    df_subset = df_raw[mask].dropna(subset=["g_r"]).copy()

    if df_subset.empty:
        print("FATAL: No data found for the specified parameters.")
        sys.exit(1)

    all_g_r_points = collections.defaultdict(list)
    df_subset["g_r_parsed"] = df_subset["g_r"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    for _, row in df_subset.iterrows():
        for r, g_val in row["g_r_parsed"]:
            all_g_r_points[r].append(g_val)

    avg_g_r = sorted([(r, np.mean(v)) for r, v in all_g_r_points.items()])
    r_data, g_data = np.array(avg_g_r).T

    # --- 2. Perform the Robust Fit ---
    C_fit, xi_fit, g_inf_fit, status = robust_fit_g_r(r_data, g_data)

    # --- 3. Generate the Debug Plot ---
    fig, ax = plt.subplots(figsize=(14, 9))

    # Plot raw data
    ax.plot(
        r_data, g_data, "o", color="steelblue", label="Averaged g(r) Data", alpha=0.8
    )

    # Plot the robust fit
    r_smooth = np.logspace(np.log10(r_data.min()), np.log10(r_data.max()), 200)
    fit_params = (C_fit, xi_fit, g_inf_fit)
    ax.plot(
        r_smooth,
        physical_decay_model(r_smooth, *fit_params),
        "--",
        color="crimson",
        linewidth=2.5,
        label="Robust Fit",
    )

    # --- 4. Annotate the Plot ---
    title = (
        f"g(r) Fit Debug for $b_m={args.bm}, \\phi={args.phi}, k_{{total}}={args.k}$"
    )
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Geodesic Distance (r)", fontsize=16)
    ax.set_ylabel("Correlation Function g(r)", fontsize=16)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":")

    # Build text box for annotations
    text_str = (
        f"--- Robust Fit Results ---\n"
        f"Status: {status}\n"
        f"$\\xi = {xi_fit:.3f}$\n"
        f"$C = {C_fit:.3f}$\n"
        f"$g_\\infty = {g_inf_fit:.3f}$"
    )

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.7)
    ax.text(
        0.05,
        0.05,
        text_str,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="bottom",
        bbox=props,
    )

    ax.legend(fontsize=14)
    plt.tight_layout()

    # Save the figure
    figures_dir = os.path.join(project_root, "figures", "correlation_fit_debug")
    os.makedirs(figures_dir, exist_ok=True)
    filename = os.path.join(
        figures_dir, f"debug_fit_bm{args.bm}_phi{args.phi}_k{args.k}.png"
    )
    plt.savefig(filename, dpi=150)
    print(f"\nDebug plot saved to: {filename}")

    plt.show()


if __name__ == "__main__":
    # Example usage from the command line:
    # python scripts/viz/debug_correlation_fitting.py criticality_mapping_v1 --bm 0.8 --phi 0.0 --k 0.01
    # python scripts/viz/debug_correlation_fitting.py criticality_mapping_v1 --bm 0.8 --phi 0.0 --k 1.0
    # python scripts/viz/debug_correlation_fitting.py criticality_mapping_v1 --bm 0.8 --phi 0.0 --k 100.0
    main()
