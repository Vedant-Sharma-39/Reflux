# FILE: scripts/analyze_diffusion.py
# Aggregates and analyzes the results from the diffusion/roughness experiment.
# It calculates the growth exponent (beta) and roughness exponent (alpha).

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import ast

# --- Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))

# IMPORTANT: Create a config_diffusion.py or link to your main config
# For now, we hardcode the campaign ID for simplicity.
CAMPAIGN_ID = "diffusion_v2_refined_neutral"
THEORY_BETA = 1 / 3
THEORY_ALPHA = 1 / 2


plt.style.use("seaborn-v0_8-whitegrid")

def read_json_file(filepath):
    """Worker to read a single JSON file and parse the trajectory."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        # Trajectory is stored as a string, so we need to evaluate it
        if "roughness_trajectory" in data and isinstance(
            data["roughness_trajectory"], str
        ):
            data["roughness_trajectory"] = ast.literal_eval(
                data["roughness_trajectory"]
            )
        return data
    except Exception:
        return None


def main():
    print(f"--- Analyzing Diffusion Results for Campaign: {CAMPAIGN_ID} ---")
    results_dir = os.path.join(project_root, "data", CAMPAIGN_ID, "results")
    output_dir = os.path.join(project_root, "data", CAMPAIGN_ID, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load and Aggregate Data ---
    filepaths = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".json")
    ]
    if not filepaths:
        print(f"Error: No result files found in {results_dir}. Exiting.")
        return

    with Pool(processes=max(1, cpu_count() - 2)) as pool:
        all_results = list(
            tqdm(pool.imap_unordered(read_json_file, filepaths), total=len(filepaths))
        )

    valid_results = [r for r in all_results if r and r.get("roughness_trajectory")]
    if not valid_results:
        print("No valid trajectories found in result files. Exiting.")
        return

    # --- 2. Unroll trajectories into a long-form DataFrame ---
    long_form_data = []
    for res in valid_results:
        width = res.get("width")
        for q, w_sq in res["roughness_trajectory"]:
            long_form_data.append({"L": width, "q": q, "W_sq": w_sq})

    df = pd.DataFrame(long_form_data)
    print(f"Created long-form DataFrame with {len(df)} data points.")

    # --- 3. Bin and Average Data ---
    # To average W_sq across replicates, we must bin the q values.
    max_q = df["q"].max()
    bins = np.logspace(0, np.log10(max_q + 1), 100)  # Use log-spaced bins
    df["q_bin"] = pd.cut(df["q"], bins)

    # Group by system width (L) and q-bin, then average
    avg_df = (
        df.groupby(["L", "q_bin"], observed=False)
        .agg(
            q_mean=("q", "mean"),
            W_sq_mean=("W_sq", "mean"),
            W_sq_sem=("W_sq", lambda x: x.std(ddof=1) / np.sqrt(x.count())),
        )
        .dropna()
        .reset_index()
    )

    # --- 4. Plot 1: Growth Phase (W^2 vs q) ---
    print("Generating Growth Phase plot (W^2 vs q)...")
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, avg_df["L"].nunique()))

    for i, (l_val, group) in enumerate(avg_df.groupby("L")):
        ax.plot(
            group["q_mean"],
            group["W_sq_mean"],
            "o",
            markersize=4,
            color=colors[i],
            alpha=0.6,
            label=f"L = {l_val}",
        )
        
        ax.set_xlabel("Mean Front Position (q)", fontsize=16)
    ax.set_ylabel("Squared Interface Width <W²>", fontsize=16)
    ax.set_title("Interface Roughening - Growth Phase", fontsize=20, pad=15)
    ax.legend(fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    # Set explicit limits to focus the view
    ax.set_xlim(1, 4000)
    ax.set_ylim(0.5, 500)

    # Fit to the largest system size for the cleanest growth regime
    fit_data = avg_df[avg_df["L"] == avg_df["L"].max()]
    # Select a range for fitting that avoids initial transients and saturation
    fit_range = fit_data[(fit_data["q_mean"] > 10) & (fit_data["q_mean"] < 100)]

    if len(fit_range) > 2:
        log_q = np.log10(fit_range["q_mean"])
        log_w = np.log10(fit_range["W_sq_mean"])
        slope, intercept, _, _, _ = linregress(log_q, log_w)
        beta_measured = slope / 2.0

        # Plot the fit
        q_fit_plot = np.logspace(np.log10(5), np.log10(200), 50)
        w_fit_plot = (10**intercept) * (q_fit_plot**slope)
        ax.plot(
            q_fit_plot,
            w_fit_plot,
            "r--",
            linewidth=2,
            label=f"Fit (slope={slope:.2f}, β={beta_measured:.2f})",
        )

    # Plot theoretical reference line
    q_theory = np.array([10, 100])
    w_theory = 0.5 * q_theory ** (2 * THEORY_BETA)  # Offset for clarity
    ax.plot(
        q_theory,
        w_theory,
        "k:",
        linewidth=2.5,
        label=f"Theory (slope={2*THEORY_BETA:.2f})",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean Front Position (q)")
    ax.set_ylabel("Squared Interface Width <W²>")
    ax.set_title("Interface Roughening - Growth Phase")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(output_dir, "diffusion_growth_phase.png"), dpi=200)
    plt.close()

    # --- 5. Plot 2: Saturation Phase (W_sat^2 vs L) ---
    print("Generating Saturation Phase plot (W_sat^2 vs L)...")
    saturation_data = []
    for l_val, group in avg_df.groupby("L"):
        # Estimate saturation by averaging the last few points
        last_q = group["q_mean"].max()
        sat_points = group[group["q_mean"] > 0.8 * last_q]
        if not sat_points.empty:
            w_sat_sq = sat_points["W_sq_mean"].mean()
            saturation_data.append({"L": l_val, "W_sat_sq": w_sat_sq})

    sat_df = pd.DataFrame(saturation_data)

    if len(sat_df) > 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(
            sat_df["L"],
            sat_df["W_sat_sq"],
            "o",
            color="navy",
            markersize=8,
            label="Measured Saturation Roughness",
        )

        log_L = np.log10(sat_df["L"])
        log_w_sat = np.log10(sat_df["W_sat_sq"])
        slope, intercept, _, _, _ = linregress(log_L, log_w_sat)
        alpha_measured = slope / 2.0

        L_fit_plot = np.logspace(
            np.log10(sat_df["L"].min()), np.log10(sat_df["L"].max()), 50
        )
        w_fit_plot = (10**intercept) * (L_fit_plot**slope)
        ax.plot(
            L_fit_plot,
            w_fit_plot,
            "r--",
            linewidth=2,
            label=f"Fit (slope={slope:.2f}, α={alpha_measured:.2f})",
        )

        # Plot theoretical reference line
        L_theory = np.array([sat_df["L"].min(), sat_df["L"].max()])
        w_theory = 0.1 * L_theory ** (2 * THEORY_ALPHA)  # Offset for clarity
        ax.plot(
            L_theory,
            w_theory,
            "k:",
            linewidth=2.5,
            label=f"Theory (slope={2*THEORY_ALPHA:.2f})",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("System Width (L)")
        ax.set_ylabel("Saturated Squared Width <W²_sat>")
        ax.set_title("Interface Roughening - Saturation Phase")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.savefig(os.path.join(output_dir, "diffusion_saturation_phase.png"), dpi=200)
        plt.close()

    print("\nDiffusion analysis complete.")


if __name__ == "__main__":
    main()
