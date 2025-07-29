# FILE: scripts/viz/debug_front_morphology.py
#
# A clean, single-core diagnostic tool to visually inspect front roughness
# scaling data for a SINGLE specified selection value.

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm
import ast
import json
import argparse

# --- Configuration and Setup ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from config import EXPERIMENTS
except (NameError, ImportError) as e:
    sys.exit(f"FATAL: Could not import configuration. Error: {e}")

THEORY_BETA, THEORY_ALPHA = 1 / 3, 1 / 2
plt.style.use("seaborn-v0_8-whitegrid")


def read_json_worker(filepath):
    """Hardened worker for reading a single JSON file."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
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
    parser = argparse.ArgumentParser(
        description="Debug front roughness plots for a specific selection value."
    )
    parser.add_argument(
        "experiment_name", default="front_morphology_vs_selection", nargs="?"
    )
    parser.add_argument(
        "--bm", type=float, required=True, help="The b_m value to isolate and plot."
    )
    args = parser.parse_args()

    config = EXPERIMENTS[args.experiment_name]
    CAMPAIGN_ID = config["CAMPAIGN_ID"]
    s_val = args.bm - 1.0

    print(f"--- Generating Debug Morphology Plots for s = {s_val:.3f} ---")

    # Step 1: Load data for the specific b_m value
    results_dir = os.path.join(project_root, "data", CAMPAIGN_ID, "results")
    if not os.path.isdir(results_dir):
        sys.exit(f"FATAL: Results directory not found at {results_dir}")

    # Find files efficiently by checking the filename for the b_m value
    filepaths = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f"bm{args.bm:.3f}" in f and f.endswith(".json")
    ]
    if not filepaths:
        sys.exit(f"FATAL: No result files found for b_m = {args.bm}.")

    # Use a simple map since this is a single-core diagnostic
    all_results = list(
        map(read_json_worker, tqdm(filepaths, desc=f"Loading JSONs for b_m={args.bm}"))
    )
    df_s = pd.DataFrame([r for r in all_results if r and r.get("roughness_trajectory")])
    if df_s.empty:
        sys.exit("No valid roughness trajectories found for this b_m value.")

    # Step 2: Process data and prepare for plotting
    long_form_data = []
    for _, row in df_s.iterrows():
        for q, w_sq in row.get("roughness_trajectory", []):
            long_form_data.append({"L": row["width"], "q": q, "W_sq": w_sq})

    df_agg = pd.DataFrame(long_form_data)
    max_q = df_agg["q"].max()
    if max_q <= 0:
        sys.exit("No valid time evolution found in data.")
    bins = np.logspace(0, np.log10(max_q), 100)
    df_agg["q_bin"] = pd.cut(df_agg["q"], bins)
    avg_df = (
        df_agg.groupby(["L", "q_bin"], observed=True)
        .agg(q_mean=("q", "mean"), W_sq_mean=("W_sq", "mean"))
        .dropna()
        .reset_index()
    )

    # Step 3: Create the two-panel debug figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle(
        f"Front Morphology Debug for s = {s_val:.3f} (b_m = {args.bm})", fontsize=20
    )

    # Panel 1: Growth Phase (Beta)
    colors = plt.cm.viridis(np.linspace(0, 1, avg_df["L"].nunique()))
    for i, (l_val, group) in enumerate(avg_df.groupby("L")):
        ax1.plot(
            group["q_mean"],
            group["W_sq_mean"],
            "o",
            ms=5,
            color=colors[i],
            alpha=0.7,
            label=f"L = {l_val}",
        )

    fit_data = avg_df[avg_df["L"] == avg_df["L"].max()]
    fit_range = fit_data[
        (fit_data["q_mean"] > 10)
        & (fit_data["q_mean"] < 100)
        & (fit_data["W_sq_mean"] > 0)
    ]
    if len(fit_range) > 3:
        slope, intercept, _, _, _ = linregress(
            np.log10(fit_range["q_mean"]), np.log10(fit_range["W_sq_mean"])
        )
        beta = slope / 2.0
        q_fit = np.logspace(1, 2, 50)
        w_fit = (10**intercept) * (q_fit**slope)
        ax1.plot(q_fit, w_fit, "r--", lw=2.5, label=f"Fit (β={beta:.2f})")
    ax1.plot(
        np.array([10, 100]),
        0.5 * np.array([10, 100]) ** (2 * THEORY_BETA),
        "k:",
        lw=2.5,
        label=f"Theory (β={THEORY_BETA:.2f})",
    )
    ax1.set(
        xscale="log",
        yscale="log",
        xlabel="Mean Front Position (q)",
        ylabel="Squared Interface Width <W²>",
        title="Growth Phase",
    )
    ax1.legend()

    # Panel 2: Saturation Phase (Alpha)
    saturation_data = []
    for l_val, group in avg_df.groupby("L"):
        sat_points = group[group["q_mean"] > 0.8 * group["q_mean"].max()]
        if not sat_points.empty:
            saturation_data.append(
                {"L": l_val, "W_sat_sq": sat_points["W_sq_mean"].mean()}
            )
    sat_df = pd.DataFrame(saturation_data)
    if len(sat_df) > 2:
        sat_df = sat_df[sat_df["W_sat_sq"] > 0]
        ax2.plot(
            sat_df["L"],
            sat_df["W_sat_sq"],
            "o",
            color="navy",
            ms=10,
            label="Measured Saturation",
        )
        slope, intercept, _, _, _ = linregress(
            np.log10(sat_df["L"]), np.log10(sat_df["W_sat_sq"])
        )
        alpha = slope / 2.0
        L_fit = np.logspace(
            np.log10(sat_df["L"].min()), np.log10(sat_df["L"].max()), 50
        )
        w_fit = (10**intercept) * (L_fit**slope)
        ax2.plot(L_fit, w_fit, "r--", lw=2.5, label=f"Fit (α={alpha:.2f})")
        ax2.plot(
            L_fit,
            0.1 * L_fit ** (2 * THEORY_ALPHA),
            "k:",
            lw=2.5,
            label=f"Theory (α={THEORY_ALPHA:.2f})",
        )
    ax2.set(
        xscale="log",
        yscale="log",
        xlabel="System Width (L)",
        ylabel="Saturated Squared Width <W²_sat>",
        title="Saturation Phase",
    )
    ax2.legend()

    for ax in (ax1, ax2):
        ax.grid(True, which="both", ls="--")

    # Step 4: Save the figure
    FIGS_DIR = os.path.join(project_root, "figures", CAMPAIGN_ID, "morphology_debug")
    os.makedirs(FIGS_DIR, exist_ok=True)
    plot_path = os.path.join(FIGS_DIR, f"debug_morphology_s_{s_val:.4f}.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nDebug plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
