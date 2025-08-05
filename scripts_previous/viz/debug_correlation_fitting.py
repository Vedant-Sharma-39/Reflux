# FILE: scripts/viz/debug_correlation_fitting.py
#
# [DEFINITIVE PRODUCTION VERSION v8 - FINAL w/ Classification]
# This version implements a more robust "classify-then-fit" logic.
# It first checks if significant decay is present. If not, it classifies the
# state as non-decaying and assigns physically meaningful parameters (xi -> inf).
# If decay is present, it proceeds with the fit. It also samples a random
# subset of parameter combinations for efficient analysis.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import argparse
import collections
from scipy.optimize import curve_fit
from tqdm import tqdm

# --- Robust Path and Config Import ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "src"))
try:
    from config import EXPERIMENTS
except ImportError:
    print("FATAL: Could not import EXPERIMENTS from src/config.py.")
    sys.exit(1)

# --- Definitive Fitting Constants ---
MAX_R_FOR_VIS = 60
DECAY_FIT_RANGE = (1, 5)
PLATEAU_RANGE = (20, 40)
XI_LARGE_THRESHOLD = 1e6  # A large, but finite, number for robust comparison
DECAY_THRESHOLD_RATIO = (
    1.1  # g_initial must be at least 10% > g_plateau to be considered "decaying"
)
NUM_RANDOM_SAMPLES = 5  # Number of (b_m, phi) pairs to sample for the showcase


# --- Fitting Models & Helpers ---
def physical_decay_model(r, C, xi, g_inf):
    # Use np.divide to handle xi=inf gracefully (exp(0)=1)
    return C * np.exp(np.divide(-r, xi, out=np.zeros_like(r), where=xi != 0)) + g_inf


def simple_decay_model(r, C, xi):
    return C * np.exp(-r / xi)


# ==============================================================================
# The DEFINITIVE Strategy: Classify, then Fit or Assign
# ==============================================================================
def fit_and_interpret_g_r(r_data, g_data):
    """
    Analyzes the correlation function g(r). First classifies if the data shows
    significant decay. If it does, it fits an exponential model. If not, it
    assigns parameters corresponding to a non-decaying state.

    Returns:
        tuple: (parameters, status_string) where parameters is a tuple (C, xi, g_inf).
    """
    if len(r_data) < 5:
        return (0, 1.0, 0.0), "no_fit_short_data"

    vis_mask = (r_data > 0) & (r_data <= MAX_R_FOR_VIS)
    if not np.any(vis_mask):
        return (0, 1.0, 0.0), "no_data_in_range"
    r_vis, g_vis = r_data[vis_mask], g_data[vis_mask]

    try:
        # Step 1: Pre-classification - check for significant decay
        decay_mask = (r_vis >= DECAY_FIT_RANGE[0]) & (r_vis <= DECAY_FIT_RANGE[1])
        plateau_mask = (r_vis >= PLATEAU_RANGE[0]) & (r_vis <= PLATEAU_RANGE[1])

        # Ensure there's data in both regions to make a comparison
        if np.any(decay_mask) and np.any(plateau_mask):
            g_initial_avg = np.mean(g_vis[decay_mask])
            g_plateau_avg = np.mean(g_vis[plateau_mask])

            # If initial value is not significantly higher than plateau, it's non-decaying
            if g_initial_avg / g_plateau_avg < DECAY_THRESHOLD_RATIO:
                g_inf_est = np.mean(g_vis)  # Use all visible data for a stable estimate
                return (0.0, np.inf, g_inf_est), "classified_non_decaying"

        # Step 2: If we are here, it means data appears to be decaying. Proceed with fit.
        g_inf_est = np.mean(g_vis[plateau_mask]) if np.any(plateau_mask) else 0.0
        if np.sum(decay_mask) < 2:
            return (0.0, np.inf, g_inf_est), "no_initial_data_for_fit"

        r_decay = r_vis[decay_mask]
        g_prime_decay = g_vis[decay_mask] - g_inf_est

        # Fit only positive part of decay
        if g_prime_decay[0] < 0:
            g_inf_est = np.mean(g_vis)
            return (0.0, np.inf, g_inf_est), "no_positive_decay_to_fit"

        bounds = ([0, 1e-9], [1.1, XI_LARGE_THRESHOLD])  # C, xi
        popt, _ = curve_fit(
            simple_decay_model,
            r_decay,
            g_prime_decay,
            p0=[max(0, g_prime_decay[0]), 5.0],
            bounds=bounds,
        )
        C_fit, xi_fit = popt

        # Step 3: Interpret the fit result as a final check
        if xi_fit >= XI_LARGE_THRESHOLD * 0.99:
            return (0.0, np.inf, g_inf_est), "fit_indicates_no_decay"

        return (C_fit, xi_fit, g_inf_est), "fit_success"

    except Exception:
        # A general fallback if any part of the process fails
        g_inf_fallback = np.mean(g_data) if len(g_data) > 0 else 0.0
        return (0, np.inf, g_inf_fallback), "fit_failed_exception"


def main():
    parser = argparse.ArgumentParser(
        description="Generate a gallery of g(r) fit plots for an experiment."
    )
    parser.add_argument(
        "experiment_name", choices=EXPERIMENTS.keys(), help="Experiment to analyze."
    )
    args = parser.parse_args()

    # --- 1. Load and Pre-process Data ---
    CAMPAIGN_ID = EXPERIMENTS[args.experiment_name]["CAMPAIGN_ID"]
    data_file = os.path.join(project_root, "data", f"{CAMPAIGN_ID}_aggregated.csv")
    if not os.path.exists(data_file):
        sys.exit(f"FATAL: Aggregated data file not found: {data_file}")

    print("Loading and pre-processing data...")
    df_raw = pd.read_csv(data_file)
    df_raw_g = df_raw.dropna(subset=["g_r"]).copy()
    df_raw_g["g_r_parsed"] = df_raw_g["g_r"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    all_g_r_points = collections.defaultdict(lambda: collections.defaultdict(list))
    for _, row in df_raw_g.iterrows():
        params = (row["b_m"], row["phi"], row["k_total"])
        for r, g_val in row["g_r_parsed"]:
            all_g_r_points[params][r].append(g_val)
    avg_g_r_data = [
        {"b_m": p[0], "phi": p[1], "k_total": p[2], "avg_g_r": sorted(d.items())}
        for p, r_data in all_g_r_points.items()
        for d in [{r: np.mean(v) for r, v in r_data.items()}]
    ]
    df_avg = pd.DataFrame(avg_g_r_data)

    # --- 2. Get unique parameter pairs and randomly sample them ---
    base_figures_dir = os.path.join(project_root, "figures", "correlation_fit_gallery")
    os.makedirs(base_figures_dir, exist_ok=True)

    unique_pairs = (
        df_avg[["b_m", "phi"]].drop_duplicates().sort_values(by=["phi", "b_m"])
    )

    if len(unique_pairs) > NUM_RANDOM_SAMPLES:
        print(
            f"Found {len(unique_pairs)} unique (b_m, phi) pairs. Randomly sampling {NUM_RANDOM_SAMPLES}."
        )
        sampled_pairs = unique_pairs.sample(n=NUM_RANDOM_SAMPLES, random_state=42)
    else:
        print(f"Found {len(unique_pairs)} unique (b_m, phi) pairs. Using all of them.")
        sampled_pairs = unique_pairs

    k_low_focus = np.logspace(-2, -0.7, 4)
    k_high_focus = np.logspace(0.5, 2, 6)
    k_showcase = np.concatenate([k_low_focus, k_high_focus])

    print(f"\nGenerating focused showcase for {len(sampled_pairs)} (b_m, phi) pairs...")

    for _, pair_row in tqdm(
        sampled_pairs.iterrows(),
        total=len(sampled_pairs),
        desc="Processing (b_m, phi) pairs",
    ):
        bm_val, phi_val = pair_row["b_m"], pair_row["phi"]

        figures_dir = os.path.join(
            base_figures_dir, f"phi_{phi_val:.2f}_bm_{bm_val:.2f}"
        )
        os.makedirs(figures_dir, exist_ok=True)

        df_slice = df_avg[
            (np.isclose(df_avg["b_m"], bm_val)) & (np.isclose(df_avg["phi"], phi_val))
        ]
        if df_slice.empty:
            continue

        for k_target in k_showcase:
            row = df_slice.iloc[(df_slice["k_total"] - k_target).abs().argmin()]
            k_actual = row["k_total"]
            if not row["avg_g_r"]:
                continue
            r_data, g_data = np.array(row["avg_g_r"]).T

            params, status = fit_and_interpret_g_r(r_data, g_data)

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(
                r_data,
                g_data,
                "o",
                color="steelblue",
                label="Averaged g(r) Data",
                alpha=0.8,
            )

            r_max_plot = min(r_data.max(), MAX_R_FOR_VIS)
            r_smooth = np.logspace(
                np.log10(max(1e-9, r_data[r_data > 0].min())), np.log10(r_max_plot), 200
            )

            ax.plot(
                r_smooth,
                physical_decay_model(r_smooth, *params),
                "--",
                color="crimson",
                linewidth=2.5,
                label="Exponential Fit",
            )

            # Format xi for display, handling infinity
            if np.isinf(params[1]):
                xi_str = "\$\\infty\$"
            else:
                xi_str = f"{params[1]:.3f}"
            text_str = f"Fit Status: {status}\n$C = {params[0]:.3f}$\n$\\xi = {xi_str}$\n$g_\\infty = {params[2]:.3f}$"

            ax.axvspan(
                0,
                MAX_R_FOR_VIS,
                color="gray",
                alpha=0.1,
                zorder=-1,
                label=f"Data Range for Vis (r < {MAX_R_FOR_VIS})",
            )
            ax.axvspan(
                DECAY_FIT_RANGE[0],
                DECAY_FIT_RANGE[1],
                color="red",
                alpha=0.15,
                zorder=-1,
                label=f"Decay Fit Range",
            )
            ax.axvspan(
                PLATEAU_RANGE[0],
                PLATEAU_RANGE[1],
                color="blue",
                alpha=0.15,
                zorder=-1,
                label=f"g_inf Est. Range",
            )

            title = f"Fit Showcase: $b_m={bm_val:.2f}, \\phi={phi_val:.2f}, k_{{total}} \\approx {k_actual:.2f}$"
            ax.set_title(title, fontsize=18)
            ax.set_xlabel("Geodesic Distance (r)")
            ax.set_ylabel("Correlation Function g(r)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1e-6, 2.0)
            ax.grid(True, which="both", linestyle=":")

            props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            ax.text(
                0.05,
                0.3,
                text_str,
                transform=ax.transAxes,
                fontsize=12,
                bbox=props,
                verticalalignment="top",
            )
            ax.legend(fontsize=12, loc="lower left")
            plt.tight_layout()

            filename = os.path.join(figures_dir, f"showcase_k{k_actual:.3f}.png")
            plt.savefig(filename, dpi=150)
            plt.close(fig)

    print("\nDefinitive showcase generation complete.")


if __name__ == "__main__":
    main()
