#!/usr/bin/env python3
# FILE: scripts/paper_figures/calculate_and_compare_radial_drift_final.py
#
# Compares the drift parameter (m_perp) estimated from radial growth experiments
# against the value calibrated from linear boundary experiments.
#
# This script implements two distinct methods for calculating drift from radial data:
# 1. Log-Fit Method: Fits a line to the mean sector angle <φ> vs. ln(radius).
#    The drift is then m_perp = slope / 2.
# 2. Derivative-Subtraction Method: Numerically estimates the derivative d<W>/dr
#    and subtracts the geometric inflation term <W>/r. The remaining "selection
#    term" should be constant and equal to 2 * m_perp.
#
# Key features:
# - Analysis is filtered to the highest initial width (W_0) condition to ensure
#   a clean and consistent comparison.
# - A radial fit window (RADIAL_FIT_RADIUS_MIN) is used to exclude initial
#   transient dynamics, leading to a more accurate drift measurement.
# - Generates diagnostic plots for each fit and a final summary plot comparing
#   the two radial methods against the linear-calibrated values.

import sys
import json
import gzip
import os
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from functools import partial
from scipy.stats import linregress

# ---------------------- Configuration ----------------------
RADIAL_CAMPAIGN_ID = "aif_online_scan_v1"
LINEAR_CAMPAIGN_ID = "boundary_experiment_v1"

# Binning and Fitting Parameters
BIN_SIZE_RADIAL = 15.0
RADIAL_FIT_RADIUS_MIN = 500.0  # Increased to exclude initial non-linear transients
RADIAL_FIT_RADIUS_MAX = 1000.0
MIN_POINTS_FOR_FIT = 5
MIN_RUNS_PER_BIN = 5
MUTANT_TYPES = [2, 3]  # Resistant, Compensated

# ---------------------- Project Paths ----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Input paths
LINEAR_PARAMS_CSV = PROJECT_ROOT / "data" / LINEAR_CAMPAIGN_ID / "analysis" / "boundary_diffusion_params_linear_final.csv"
RADIAL_DATA_DIR = PROJECT_ROOT / "data" / RADIAL_CAMPAIGN_ID
RADIAL_ANALYSIS_DIR = RADIAL_DATA_DIR / "analysis"
RADIAL_TRAJ_DIR = RADIAL_DATA_DIR / "trajectories"

# Output paths
FIG_DIR = PROJECT_ROOT / "figures" / "drift_comparison_linear_vs_radial"
FIG_DIR.mkdir(parents=True, exist_ok=True)
# Output CSV will be dynamically named based on initial width

# ---------------------- Helper Functions ----------------------
def _safe_r2(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    """Calculates R² for a simple linear fit y ≈ intercept + slope * x, handling edge cases."""
    if x.size < 2:
        return np.nan
    yhat = intercept + slope * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return np.nan if ss_tot <= 1e-15 else 1.0 - ss_res / ss_tot

# ---------------------- Data Loading & Per-Run Processing ----------------------
def load_linear_params() -> pd.DataFrame:
    """Loads the pre-calculated m_perp from the linear experiments."""
    if not LINEAR_PARAMS_CSV.exists():
        sys.exit(f"[ERROR] Linear params file not found: {LINEAR_PARAMS_CSV}")
    df = pd.read_csv(LINEAR_PARAMS_CSV).rename(columns={"b_m": "b_res", "m_perp": "m_perp_linear"})
    return df[["b_res", "m_perp_linear"]].dropna()

def load_radial_param_map() -> pd.DataFrame:
    """Loads the summary metadata for the radial campaign."""
    csv_path = RADIAL_ANALYSIS_DIR / f"{RADIAL_CAMPAIGN_ID}_summary_aggregated.csv"
    if not csv_path.exists():
        sys.exit(f"[ERROR] Radial summary file not found: {csv_path}")
    cols = ["task_id", "b_res", "sector_width_initial"]
    df = pd.read_csv(csv_path, usecols=cols)
    df = df.rename(columns={"task_id": "run_id", "sector_width_initial": "initial_width"})
    return df.drop_duplicates("run_id")

def process_single_radial_run(task: dict, bin_size: float) -> Optional[Dict]:
    """
    Loads one radial run, finds the main mutant lineage, bins its trajectory,
    and returns the first observation per bin.
    """
    traj_path = Path(task["traj_path"])
    if not traj_path.exists():
        return None
    with gzip.open(traj_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    rows = data if isinstance(data, list) else data.get("sector_trajectory", [])
    if not rows:
        return None
    df = pd.DataFrame(rows)

    mutant_mask = df["type"].isin(MUTANT_TYPES)
    if not mutant_mask.any():
        return None

    # Identify the dominant mutant lineage by finding the root_sid with the
    # largest cumulative width at the first radius where mutants appear.
    rmin = df.loc[mutant_mask, "radius"].min()
    sl = df[(np.abs(df["radius"] - rmin) < 1e-9) & mutant_mask]
    if sl.empty:
        return None
    root_sid = sl.groupby("root_sid")["width_cells"].sum().idxmax()

    lineage_df = df[(df["root_sid"] == root_sid) & mutant_mask].copy()
    if lineage_df.empty:
        return None

    # Bin by radius and keep the *first* observation per bin
    lineage_df["bin_idx"] = np.floor(lineage_df["radius"] / bin_size).astype(np.int64)
    lineage_df.sort_values("radius", inplace=True)
    df_first = lineage_df.drop_duplicates(subset=["bin_idx"], keep="first")
    if df_first.empty:
        return None

    return {
        "b_res": task["b_res"],
        "radius_bins": (df_first["bin_idx"].to_numpy() * bin_size).astype(float),
        "width_bins": df_first["width_cells"].to_numpy(float),
    }

# ---------------------- Plotting Functions ----------------------
def plot_radial_log_fit(df_agg: pd.DataFrame, fit_params: Dict, b_res: float, initial_width: int):
    """Generates and saves a diagnostic plot for the log-fit method."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        df_agg["log_radius"], df_agg["mean_angle"], yerr=df_agg["sem_angle"],
        fmt="o", ms=6, lw=1.5, capsize=4, color="teal", ecolor="teal", alpha=0.8,
        label="Mean Angle ± SEM"
    )
    r_min, r_max = RADIAL_FIT_RADIUS_MIN, RADIAL_FIT_RADIUS_MAX
    fit_line_x = np.linspace(np.log(r_min), np.log(r_max), 100)
    fit_line_y = fit_params["intercept"] + fit_params["slope"] * fit_line_x
    ax.plot(fit_line_x, fit_line_y, "r--", lw=2.5, label=f"Linear Fit (Slope = {fit_params['slope']:.4g})")
    ax.axvspan(np.log(r_min), np.log(r_max), color="gray", alpha=0.15, label="Fit Window")

    m_perp_radial = fit_params['slope'] / 2.0
    ax.text(
        0.04, 0.96, f"$m_\\perp = {m_perp_radial:.4f}$\n$R^2 = {fit_params['r2']:.3f}$",
        transform=ax.transAxes, va="top", ha="left", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
    )
    ax.set(
        title=f"Radial Drift (Log-Fit Method, $b_{{res}}$={b_res:.4f}, $W_0$={initial_width})",
        xlabel="ln(Radius)", ylabel="Mean Sector Angle <φ>"
    )
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"radial_drift_log_fit_b{b_res:.4f}_w{initial_width}.png", dpi=150)
    plt.close(fig)

def plot_radial_derivative_method(df_fit: pd.DataFrame, m_perp_est: float, b_res: float, initial_width: int):
    """Generates and saves a diagnostic plot for the derivative-subtraction method."""
    fig, ax = plt.subplots(figsize=(12, 7))
    selection_term = df_fit['dW_dr_numeric'] - df_fit['inflation_term']

    ax.plot(df_fit['radius_bin'], df_fit['dW_dr_numeric'], 'o-', color='darkorange', label='$d\\langle W \\rangle/dr$ (Total Width Change)')
    ax.plot(df_fit['radius_bin'], df_fit['inflation_term'], 's--', color='dodgerblue', label='$\\langle W \\rangle/r$ (Geometric Inflation)')
    ax.plot(df_fit['radius_bin'], selection_term, '^-', color='green', label='$d\\langle W \\rangle/dr - \\langle W \\rangle/r$ (Selection Term)')
    ax.axhline(2 * m_perp_est, color='red', linestyle=':', lw=2.5, label=f'Mean Selection Term ($2 \\times m_\\perp = {2*m_perp_est:.4f}$)')
    ax.axvspan(RADIAL_FIT_RADIUS_MIN, RADIAL_FIT_RADIUS_MAX, color="gray", alpha=0.12, label="Fit Window")
    ax.set(
        title=f"Radial Drift (Derivative Method, $b_{{res}}$={b_res:.4f}, $W_0$={initial_width})",
        xlabel="Radius", ylabel="Rate of Change"
    )
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"radial_drift_derivative_fit_b{b_res:.4f}_w{initial_width}.png", dpi=150)
    plt.close(fig)

def plot_comparison(df_merged: pd.DataFrame, initial_width: int):
    """Generates and saves the final comparison plot across all methods."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df_merged["b_res"], df_merged["m_perp_linear"], 'o-', ms=8, lw=2.5, color="crimson", label="Linear ($d\\langle W \\rangle / dr$)")
    ax.plot(df_merged["b_res"], df_merged["m_perp_radial_log"], 's--', ms=8, lw=2, color="navy", markerfacecolor="skyblue", label="Radial (ln(r) Fit)")
    ax.plot(df_merged["b_res"], df_merged["m_perp_radial_deriv"], '^:', ms=9, lw=2, color="darkgreen", markerfacecolor="lightgreen", label="Radial (Derivative Subtraction)")
    ax.set(
        title=f"Drift Comparison: Linear vs. Radial ($W_0$={initial_width} cells)",
        xlabel="Selection Parameter ($b_{res}$)",
        ylabel="Drift Parameter ($m_\\perp$)"
    )
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(frameon=False, fontsize=12)
    ax.axhline(0, color='k', linestyle=':', linewidth=1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"drift_comparison_all_methods_w{initial_width}.png", dpi=200)
    plt.close(fig)

# ---------------------- Main Execution ----------------------
def main():
    print("--- Step 1: Loading pre-calculated linear drift parameters ---")
    df_linear = load_linear_params()

    print("\n--- Step 2: Loading radial data and filtering for highest initial width ---")
    df_map_radial = load_radial_param_map().dropna(subset=['initial_width'])
    if df_map_radial.empty:
        sys.exit("No radial runs with initial width information found.")

    max_initial_width = int(df_map_radial['initial_width'].max())
    print(f"Found highest initial width: {max_initial_width} cells. Using this subset for analysis.")

    df_map_filtered = df_map_radial[df_map_radial['initial_width'] == max_initial_width]
    tasks = [dict(row, traj_path=str(RADIAL_TRAJ_DIR / f"traj_{row['run_id']}.json.gz")) for _, row in df_map_filtered.iterrows()]
    if not tasks:
        sys.exit(f"No tasks found for initial width {max_initial_width}.")

    num_workers = max(1, (os.cpu_count() or 2) - 2)
    worker = partial(process_single_radial_run, bin_size=BIN_SIZE_RADIAL)
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(worker, tasks), total=len(tasks), desc="Processing filtered radial runs"))

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        sys.exit("No valid radial trajectories found in the filtered set.")

    # Expand results into a long-form DataFrame
    all_points = []
    for res in valid_results:
        df = pd.DataFrame({"radius_bin": res["radius_bins"], "width_bin": res["width_bins"]})
        df["b_res"] = res["b_res"]
        all_points.append(df)
    df_all_radial = pd.concat(all_points, ignore_index=True)

    print(f"\n--- Step 3: Calculating drift from radial data (W0 = {max_initial_width}) ---")
    radial_fits = []
    for b_res, group in tqdm(df_all_radial.groupby("b_res"), desc="Fitting radial drift by b_res"):
        # Aggregate per-bin across runs
        agg = group.groupby("radius_bin")["width_bin"].agg(
            n_runs="count",
            mean_width="mean",
            sem_width="sem",
        ).reset_index()
        agg = agg[agg["n_runs"] >= MIN_RUNS_PER_BIN].copy()
        agg = agg[agg['radius_bin'] > 0].copy()
        if agg.empty:
            continue

        # Define the fitting window
        fit_mask = (agg["radius_bin"] >= RADIAL_FIT_RADIUS_MIN) & (agg["radius_bin"] <= RADIAL_FIT_RADIUS_MAX)
        m_perp_radial_log, m_perp_radial_deriv = np.nan, np.nan

        # --- Method 1: Log-Fit on Mean Angle ---
        agg["mean_angle"] = agg["mean_width"] / agg["radius_bin"]
        agg["sem_angle"] = agg["sem_width"] / agg["radius_bin"]
        agg["log_radius"] = np.log(agg["radius_bin"])
        df_fit_log = agg.loc[fit_mask].dropna()
        if len(df_fit_log) >= MIN_POINTS_FOR_FIT:
            fit_result = linregress(df_fit_log["log_radius"], df_fit_log["mean_angle"])
            m_perp_radial_log = fit_result.slope / 2.0
            r2 = _safe_r2(df_fit_log["log_radius"], df_fit_log["mean_angle"], fit_result.slope, fit_result.intercept)
            fit_params = {"slope": fit_result.slope, "intercept": fit_result.intercept, "r2": r2}
            plot_radial_log_fit(agg, fit_params, b_res, max_initial_width)

        # --- Method 2: Derivative-Subtraction ---
        df_fit_deriv = agg.copy() # Use full range for gradient, then select window
        if len(df_fit_deriv) >= 3: # Need at least 3 points for a stable gradient
            df_fit_deriv['dW_dr_numeric'] = np.gradient(df_fit_deriv['mean_width'], df_fit_deriv['radius_bin'])
            df_fit_deriv['inflation_term'] = df_fit_deriv['mean_width'] / df_fit_deriv['radius_bin']
            
            df_fit_window = df_fit_deriv.loc[fit_mask].dropna()
            if len(df_fit_window) >= MIN_POINTS_FOR_FIT:
                selection_term = df_fit_window['dW_dr_numeric'] - df_fit_window['inflation_term']
                m_perp_radial_deriv = np.mean(selection_term) / 2.0
                plot_radial_derivative_method(df_fit_window, m_perp_radial_deriv, b_res, max_initial_width)

        radial_fits.append({
            "b_res": b_res,
            "m_perp_radial_log": m_perp_radial_log,
            "m_perp_radial_deriv": m_perp_radial_deriv,
        })

    if not radial_fits:
        sys.exit("Could not compute any radial fits.")
    df_radial = pd.DataFrame(radial_fits)

    print("\n--- Step 4: Comparing all three methods ---")
    df_merged = pd.merge(df_linear, df_radial, on="b_res", how="inner")
    if df_merged.empty:
        sys.exit("No common b_res values found between linear and radial experiments. Cannot compare.")

    OUT_CSV = FIG_DIR / f"drift_comparison_summary_w{max_initial_width}.csv"
    df_merged.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"✅ Saved comparison data to: {OUT_CSV}")

    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(f"\n--- Final Drift Comparison (Initial Width = {max_initial_width}) ---")
        print(df_merged)

    plot_comparison(df_merged, max_initial_width)
    print(f"✅ Generated final comparison plot for W0 = {max_initial_width}.")

if __name__ == "__main__":
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    main()