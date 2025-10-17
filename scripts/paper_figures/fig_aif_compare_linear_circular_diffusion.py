#!/usr/bin/env python3
# FILE: scripts/paper_figures/compare_linear_vs_circular_diffusion.py
#
# THIS IS THE FINAL ROBUST VERSION: Fixes the KeyError by using consistent
# column names ('replicate', 'sector_width_initial') throughout the script.

import sys, json, gzip, os
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from scipy.stats import linregress
import argparse

# --- Project Setup & Constants ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from src.config import EXPERIMENTS

# Analysis Parameters
LINEAR_EXP_NAME = "small_boundary_exp"
CIRCULAR_EXP_ID = "aif_online_scan_v1"
MIN_TRAJ_FOR_FIT, FIT_START_PERCENT, FIT_END_PERCENT = 10, 0.30, 0.90
COMMON_POINTS, MIN_R_SQUARED, NUM_SAMPLE_TRAJ = 75, 0.90, 5
RESISTANT, COMPENSATED = 2, 3
MUTANTS = (RESISTANT, COMPENSATED)

# =============================================================================
# --- DATA LOADING & PROCESSING ---
# =============================================================================

def load_trajectory_data(campaign_id: str, task_id: str) -> Optional[List]:
    # This function is correct.
    base_dir = PROJECT_ROOT / "data" / campaign_id / "trajectories"
    possible_names = [f"traj_{task_id}.json.gz", f"traj_boundary_{task_id}.json.gz", f"traj_sector_{task_id}.json.gz"]
    file_path = next((base_dir / name for name in possible_names if (base_dir / name).exists()), None)
    if not file_path: return None
    try:
        with gzip.open(file_path, "rt") as f: data = json.load(f)
        if isinstance(data, dict):
            return next((data[key] for key in ["trajectory", "sector_trajectory"] if key in data), None)
        return data
    except Exception: return None

def analyze_linear_campaign_for_Dx(campaign_id: str) -> Dict[float, float]:
    # This function is correct.
    print(f"--- Analyzing Linear Campaign '{campaign_id}' to get ground truth Dx ---")
    summary_path = PROJECT_ROOT / "data" / campaign_id / "analysis" / f"{campaign_id}_summary_aggregated.csv"
    if not os.path.exists(summary_path): sys.exit(f"ERROR: Summary file not found for linear campaign: {summary_path}")
    df_summary = pd.read_csv(summary_path)
    dx_map = {}
    for b_res, group in tqdm(df_summary.groupby("b_m"), desc="Calculating Dx from linear data"):
        trajectories = [load_trajectory_data(campaign_id, str(tid)) for tid in group["task_id"]]
        trajectories = [np.array(t) for t in trajectories if t and len(t) > 10]
        if len(trajectories) < MIN_TRAJ_FOR_FIT: continue
        min_max_time = min(t[-1, 0] for t in trajectories if len(t) > 0)
        fit_start, fit_end = min_max_time * FIT_START_PERCENT, min_max_time * FIT_END_PERCENT
        if fit_end <= fit_start: continue
        common_time = np.linspace(fit_start, fit_end, COMMON_POINTS)
        resampled = [np.interp(common_time, t[:, 0], t[:, 1]) for t in trajectories]
        var_width = np.var(resampled, axis=0, ddof=1)
        fit = linregress(common_time, var_width)
        if fit.rvalue**2 > MIN_R_SQUARED: dx_map[b_res] = fit.slope / 4.0
    print("Found Dx values from linear experiment:", dx_map)
    return dx_map

def process_circular_task(task_info: dict) -> Optional[pd.DataFrame]:
    """Worker to process one circular trajectory file with robust sector ID."""
    
    rows = load_trajectory_data(task_info['campaign_id'], task_info['task_id'])
    if not rows: return None
    df_run = pd.DataFrame(rows)
    if df_run.empty or 'radius' not in df_run.columns or 'root_sid' not in df_run.columns: return None
    longest_lived_sid = df_run.loc[df_run['radius'].idxmax()]['root_sid']
    mask = (np.isin(df_run["type"], MUTANTS)) & (df_run["root_sid"] == longest_lived_sid)
    if not mask.any(): return None
    traj_df = df_run.loc[mask, ["radius", "width_cells"]].rename(columns={"width_cells": "width"})
    traj_df = traj_df.sort_values("radius").reset_index(drop=True)
    # --- FIX: Use the original column names from the summary file ---
    traj_df['b_res'] = task_info['b_res']
    traj_df['sector_width_initial'] = task_info['sector_width_initial']
    traj_df['replicate'] = task_info['replicate']
    return traj_df

def sample_trajectories(n_samples, r_initial, w_initial, r_final, growth_slope, Dx, dr=0.1):
    # This function is correct
    if r_final <= r_initial or Dx is None or np.isnan(Dx): return np.array([]), np.array([])
    radii = np.arange(r_initial, r_final, dr)
    widths = np.full((n_samples, len(radii)), w_initial, dtype=float)
    noise_std_dev = np.sqrt(4 * Dx * dr)
    for i in range(1, len(radii)):
        widths[:, i] = np.maximum(0, widths[:, i-1] + (growth_slope * dr) + np.random.normal(0, noise_std_dev, n_samples))
    return radii, widths

# =============================================================================
# --- MAIN WORKFLOW & PLOTTING ---
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare linear and circular diffusion models.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--linear-campaign", default="boundary_analysis", help="Name of the linear experiment in config.py")
    parser.add_argument("--circular-campaign", default="aif_online_scan_v1", help="ID of the circular experiment campaign")
    args = parser.parse_args()

    try:
        linear_campaign_id = EXPERIMENTS[args.linear_campaign]["campaign_id"]
    except KeyError:
        sys.exit(f"Error: Linear experiment '{args.linear_campaign}' not found in src/config.py.")

    Dx_from_linear_map = analyze_linear_campaign_for_Dx(linear_campaign_id)

    print(f"\n--- Loading Circular Campaign '{args.circular_campaign}' Data ---")
    summary_path = PROJECT_ROOT / "data" / args.circular_campaign / "analysis" / f"{args.circular_campaign}_summary_aggregated.csv"
    if not os.path.exists(summary_path):
        sys.exit(f"ERROR: Summary file not found for circular campaign: {summary_path}")
    
    # Load the original column names
    df_params = pd.read_csv(summary_path, usecols=["task_id", "b_res", "sector_width_initial", "replicate"])
    tasks = df_params.to_dict('records')
    for task in tasks: task['campaign_id'] = args.circular_campaign

    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap_unordered(process_circular_task, tasks), total=len(tasks), desc="Loading circular data"))
    
    df_all_traj = pd.concat([df for df in results if df is not None and not df.empty], ignore_index=True)
    if df_all_traj.empty:
        sys.exit("CRITICAL ERROR: Failed to load any valid circular trajectories.")

    print("\n--- Analyzing Conditions and Generating Comparison Plots ---")
    analysis_summary = []
    fig_dir = PROJECT_ROOT / "figures" / "diagnostics" / "linear_vs_circular_comparison"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- FIX: Use the correct column names for grouping ---
    for condition, group in tqdm(df_all_traj.groupby(['b_res', 'sector_width_initial']), desc="Analyzing & Plotting"):
        b_res, w0 = condition
        Dx_linear = Dx_from_linear_map.get(b_res)
        
        # --- FIX: Use the correct column name to group replicates ---
        trajectories = [g[['radius', 'width']].to_numpy() for _, g in group.groupby('replicate')]
        if len(trajectories) < MIN_TRAJ_FOR_FIT: continue
        
        # The rest of the analysis logic is correct and does not need to be changed
        r_initial = group['radius'].min()
        max_radii = [t[-1, 0] for t in trajectories if len(t)>0]
        if not max_radii: continue
        r_final = np.percentile(max_radii, 10)
        
        fit_start_r, fit_end_r = r_initial + (r_final - r_initial) * FIT_START_PERCENT, r_initial + (r_final - r_initial) * FIT_END_PERCENT
        if fit_end_r <= fit_start_r: continue

        common_r = np.linspace(fit_start_r, fit_end_r, COMMON_POINTS)
        resampled_w = [np.interp(common_r, t[:, 0], t[:, 1]) for t in trajectories]
        mean_w, fit_growth = np.mean(resampled_w, axis=0), linregress(common_r, mean_w)
        growth_slope = fit_growth.slope if fit_growth.rvalue**2 > MIN_R_SQUARED else 0.0

        common_inv_r = np.linspace(1/fit_end_r, 1/fit_start_r, COMMON_POINTS)
        resampled_a = []
        for t in trajectories:
            valid = t[:, 0] > 1e-6
            if np.sum(valid) < 2: continue
            resampled_a.append(np.interp(common_inv_r, 1/t[valid, 0][::-1], (t[valid, 1]/t[valid, 0])[::-1]))
        if len(resampled_a) < MIN_TRAJ_FOR_FIT: continue
        
        var_a, fit_angular_diff = np.var(resampled_a, axis=0, ddof=1), linregress(common_inv_r, var_a)
        Dx_circular = fit_angular_diff.slope / 4.0 if fit_angular_diff.rvalue**2 > MIN_R_SQUARED else np.nan
        
        analysis_summary.append({"b_res": b_res, "initial_width": w0, "Dx_from_linear": Dx_linear, "Dx_from_circular": Dx_circular})

        # Plotting remains the same...
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(f"Theory vs. Measurement for b_res={b_res:.4f}, w0={w0}", fontsize=18)
        
        r_sim_l, w_sim_linear = sample_trajectories(NUM_SAMPLE_TRAJ, r_initial, w0, r_final, growth_slope, Dx_linear)
        r_sim_c, w_sim_circular = sample_trajectories(NUM_SAMPLE_TRAJ, r_initial, w0, r_final, growth_slope, Dx_circular)

        axes[0, 0].set_title(f"A: Prediction from Linear (Dx={Dx_linear:.3f})")
        if r_sim_l.size > 0:
             for i in range(NUM_SAMPLE_TRAJ): axes[0, 0].plot(r_sim_l, w_sim_linear[i, :], c='red', lw=1.5, alpha=0.6)
        for t in trajectories: axes[0, 0].plot(t[:, 0], t[:, 1], c='gray', alpha=0.2, lw=0.8)

        axes[0, 1].set_title(f"B: Self-Consistent Fit (Dx={Dx_circular:.3f})")
        if r_sim_c.size > 0:
            for i in range(NUM_SAMPLE_TRAJ): axes[0, 1].plot(r_sim_c, w_sim_circular[i, :], c='blue', lw=1.5, alpha=0.6)
        for t in trajectories: axes[0, 1].plot(t[:, 0], t[:, 1], c='gray', alpha=0.2, lw=0.8)
        
        axes[1, 0].set_title("C: Direct Measurement of Dx from Var(Φ) vs 1/r")
        axes[1, 0].plot(common_inv_r, var_a, 'o'); axes[1, 0].plot(common_inv_r, fit_angular_diff.intercept + fit_angular_diff.slope * common_inv_r, 'r-', label=f"Fit (Slope={fit_angular_diff.slope:.2f}, $R^2$={fit_angular_diff.rvalue**2:.2f})\nImplied Dx = {Dx_circular:.3f}")
        
        axes[1, 1].set_title("D: Measurement of Growth Rate from <W> vs r")
        axes[1, 1].plot(common_r, mean_w, 'o'); axes[1, 1].plot(common_r, fit_growth.intercept + fit_growth.slope * common_r, 'r-', label=f"Fit (Slope={growth_slope:.4f}, $R^2$={fit_growth.rvalue**2:.2f})")
        
        for ax in axes.flat: ax.grid(True, linestyle=":")
        for ax in [axes[0,0], axes[0,1]]: ax.set_xlabel("Radius (r)"); ax.set_ylabel("Arc Width (W)"); ax.set_xlim(left=0, right=r_final*1.05); ax.set_ylim(bottom=0)
        axes[1,0].set_xlabel("Inverse Radius (1/r)"); axes[1,0].set_ylabel("Var(Angle) Var(Φ)"); axes[1,0].legend()
        axes[1,1].set_xlabel("Radius (r)"); axes[1,1].set_ylabel("Mean Arc Width <W>"); axes[1,1].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(fig_dir / f"diag_compare_b{b_res:.4f}_w{w0}.png")
        plt.close(fig)
        
    if analysis_summary:
        df_final = pd.DataFrame(analysis_summary)
        print("\n--- FINAL COMPARISON OF DIFFUSION CONSTANTS ---")
        print(df_final.to_string(index=False))

if __name__ == "__main__":
    main()