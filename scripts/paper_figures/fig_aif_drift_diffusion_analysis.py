#!/usr/bin/env python3
# FILE: scripts/paper_figures/fig_aif_circular_diffusion_analysis.py
#
# A script to test the Hallatschek & Nelson (2010) model for boundary
# wandering in circular expanding populations using AIF simulation data.
#
# For each condition, it produces a 4-panel diagnostic plot showing:
#   A) Sample raw trajectories of arc width vs. radius.
#   B) The linear fit for arc width growth rate (<W> vs. r).
#   C) The key theoretical test: the linear fit for Var(Φ) vs. 1/r.
#   D) A sanity check plot of angle <Φ> vs. radius.

import sys, json, gzip, os
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from scipy.stats import linregress

# --- Project Setup & Constants ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from src.config import EXPERIMENTS

RESISTANT, COMPENSATED = 2, 3
MUTANTS = (RESISTANT, COMPENSATED)
MIN_TRAJECTORIES_FOR_FIT = 10
FIT_START_RADIUS_PERCENT = 0.30  # Start fit later to avoid initial transients
FIT_END_RADIUS_PERCENT = 0.90
COMMON_RADIUS_POINTS = 75
MIN_R_SQUARED = 0.90
NUM_SAMPLE_TRAJECTORIES = 5 # Number of raw trajectories to show in Panel A

# =============================================================================
# --- DATA LOADING (Re-used and adapted) ---
# =============================================================================

def infer_root_sid_numpy(types: np.ndarray, radii: np.ndarray, widths: np.ndarray, roots: np.ndarray) -> Optional[int]:
    mask_mut = np.isin(types, MUTANTS)
    if not mask_mut.any(): return None
    rmin = radii[mask_mut].min()
    sl = mask_mut & (np.abs(radii - rmin) < 1e-9)
    if not sl.any(): return None
    agg: Dict[int, float] = {}
    for rt, w in zip(roots[sl], widths[sl]):
        agg[rt] = agg.get(rt, 0.0) + w
    return max(agg, key=agg.get) if agg else None

def process_single_task(task_info: dict) -> Optional[pd.DataFrame]:
    """Worker function to load the full, unbinned trajectory for one replicate."""
    traj_path = Path(task_info["traj_path"])
    if not traj_path.exists(): return None
    try:
        with gzip.open(traj_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else data.get("sector_trajectory", [])
        if not rows: return None
        
        df_run = pd.DataFrame(rows)
        types = df_run["type"].to_numpy(dtype=np.int32)
        radii = df_run["radius"].to_numpy(dtype=float)
        widths = df_run["width_cells"].to_numpy(dtype=float)
        roots = df_run["root_sid"].to_numpy(dtype=np.int32)

        root_to_track = infer_root_sid_numpy(types, radii, widths, roots)
        if root_to_track is None: return None

        mask = (np.isin(types, MUTANTS)) & (roots == root_to_track)
        if not mask.any(): return None
        
        traj_df = pd.DataFrame({"radius": radii[mask], "width": widths[mask]}).sort_values("radius").reset_index(drop=True)
        traj_df['b_res'] = task_info['b_res']
        traj_df['initial_width'] = task_info['sector_width_initial']
        traj_df['replicate_id'] = task_info['replicate']
        return traj_df
    except Exception:
        return None

# =============================================================================
# --- NEW CORE ANALYSIS & PLOTTING LOGIC ---
# =============================================================================

def analyze_and_plot_condition(
    condition_name: tuple, group_df: pd.DataFrame, fig: plt.Figure
) -> Optional[Dict]:
    """
    Analyzes all trajectories for a single condition and generates a 4-panel
    diagnostic plot within the provided figure.
    """
    b_res, w0 = condition_name
    
    # Create a 2x2 grid for this condition's plots
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    all_trajectories = [g[['radius', 'width']].to_numpy() for _, g in group_df.groupby('replicate_id')]

    if len(all_trajectories) < MIN_TRAJECTORIES_FOR_FIT:
        axA.text(0.5, 0.5, "Not enough data for fits", ha='center', va='center')
        return None

    # --- Data Preparation for Analysis ---
    max_radii = np.array([t[-1, 0] for t in all_trajectories if len(t) > 0])
    min_max_radius = np.percentile(max_radii, 10) # Use a robust minimum of the max radii

    fit_start_radius = group_df['radius'].min() * (1 + FIT_START_RADIUS_PERCENT)
    fit_end_radius = min_max_radius * FIT_END_RADIUS_PERCENT
    if fit_end_radius <= fit_start_radius: return None

    # Grid for linear analysis (vs. r)
    common_radius = np.linspace(fit_start_radius, fit_end_radius, num=COMMON_RADIUS_POINTS)
    
    # Grid for angular analysis (vs. 1/r) - MUST BE ASCENDING for interp
    common_inv_radius = np.linspace(1/fit_end_radius, 1/fit_start_radius, num=COMMON_RADIUS_POINTS)

    # Resample widths and angles
    resampled_widths = []
    resampled_angles = []
    for t in all_trajectories:
        # Interpolate width vs. radius
        resampled_w = np.interp(common_radius, t[:, 0], t[:, 1])
        resampled_widths.append(resampled_w)
        
        # Interpolate angle vs. inverse radius
        # Note: np.interp needs ascending x-values, so we flip both arrays
        inv_r = 1 / t[:, 0]
        angle = t[:, 1] / t[:, 0]
        resampled_a = np.interp(common_inv_radius, inv_r[::-1], angle[::-1])
        resampled_angles.append(resampled_a)

    # --- Panel A: Sample Trajectories ---
    axA.set_title("A: Sample Raw Trajectories")
    for i, t in enumerate(all_trajectories):
        if i < NUM_SAMPLE_TRAJECTORIES:
            axA.plot(t[:, 0], t[:, 1], lw=1, alpha=0.7, label=f"Rep. {i}" if i==0 else None)
    # Overlay the mean
    axA.plot(common_radius, np.mean(resampled_widths, axis=0), color='k', lw=2, ls='--', label='Mean')
    axA.set_xlabel("Radius (r)"); axA.set_ylabel("Arc Width (W)"); axA.legend(fontsize=8)

    # --- Panel B: Linear Drift Analysis (<W> vs r) ---
    axB.set_title("B: Arc Width Growth Rate (Linear Model)")
    mean_width = np.mean(resampled_widths, axis=0)
    fit_growth = linregress(common_radius, mean_width)
    axB.plot(common_radius, mean_width, 'o', ms=4)
    axB.plot(common_radius, fit_growth.intercept + fit_growth.slope * common_radius, 'r-',
             label=f"Slope={fit_growth.slope:.4f}\n$R^2$={fit_growth.rvalue**2:.2f}")
    axB.set_xlabel("Radius (r)"); axB.set_ylabel("Mean Arc Width <W(r)>"); axB.legend(fontsize=8)
    
    # --- Panel C: Angular Diffusion Analysis (Var(Φ) vs 1/r) ---
    axC.set_title("C: Angular Diffusion (Circular Model)")
    var_angle = np.var(resampled_angles, axis=0, ddof=1)
    fit_angular_diff = linregress(common_inv_radius, var_angle)
    axC.plot(common_inv_radius, var_angle, 'o', ms=4)
    axC.plot(common_inv_radius, fit_angular_diff.intercept + fit_angular_diff.slope * common_inv_radius, 'r-',
             label=f"Slope={fit_angular_diff.slope:.2f}\n$R^2$={fit_angular_diff.rvalue**2:.2f}")
    axC.set_xlabel("Inverse Radius (1/r)"); axC.set_ylabel("Var(Angle) Var(Φ)"); axC.legend(fontsize=8)

    # --- Panel D: Angle vs. Radius (Sanity Check) ---
    axD.set_title("D: Mean Angle vs. Radius")
    axD.plot(common_radius, np.mean(resampled_widths, axis=0) / common_radius, 'o-', ms=4)
    axD.set_xlabel("Radius (r)"); axD.set_ylabel("Mean Angle <Φ(r)>")
    
    # Calculate final results
    growth_rate_slope = fit_growth.slope if fit_growth.rvalue**2 > MIN_R_SQUARED else np.nan
    # From paper, slope of Var(Φ) vs (1/r) is -4*Dx. We use absolute radius, so slope is positive.
    Dx_from_angular = fit_angular_diff.slope / 4.0 if fit_angular_diff.rvalue**2 > MIN_R_SQUARED else np.nan
    
    return {
        "b_res": b_res, "initial_width": w0,
        "growth_rate_slope": growth_rate_slope,
        "Dx_from_angular_fit": Dx_from_angular,
    }

# =============================================================================
# --- MAIN WORKFLOW ---
# =============================================================================

def main():
    campaign_id = "aif_online_scan_v1"
    print(f"--- Running Circular Drift-Diffusion Analysis for: {campaign_id} ---")

    data_dir = PROJECT_ROOT / "data" / campaign_id
    analysis_dir = data_dir / "analysis"
    traj_dir = data_dir / "trajectories"
    fig_dir = PROJECT_ROOT / "figures" / "diagnostics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    summary_csv = analysis_dir / f"{campaign_id}_summary_aggregated.csv"
    df_params = pd.read_csv(summary_csv, low_memory=False, usecols=["task_id", "b_res", "sector_width_initial", "replicate"])
    tasks = [row.to_dict() for _, row in df_params.iterrows()]
    for task in tasks: task["traj_path"] = str(traj_dir / f"traj_{task['task_id']}.json.gz")

    num_workers = max(1, os.cpu_count() - 2)
    print(f"\nLoading {len(tasks)} trajectories using {num_workers} workers...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_task, tasks), total=len(tasks)))
    
    all_trajectories_list = [df for df in results if df is not None and not df.empty]
    if not all_trajectories_list: sys.exit("No valid trajectories were loaded.")
    df_all_traj = pd.concat(all_trajectories_list, ignore_index=True)

    print("\nAnalyzing conditions and generating diagnostic plots...")
    grouped = list(df_all_traj.groupby(['b_res', 'initial_width']))
    
    analysis_results = []
    for i, (condition, group) in enumerate(tqdm(grouped, desc="Analyzing & Plotting Conditions")):
        # Create a new figure for each condition
        fig = plt.figure(figsize=(12, 10))
        b_res, w0 = condition
        fig.suptitle(f"Drift & Diffusion Analysis for b_res={b_res:.4f}, initial_width={w0}", fontsize=16, y=0.98)
        
        result = analyze_and_plot_condition(condition, group, fig)
        if result:
            analysis_results.append(result)
        
        output_path = fig_dir / f"diag_circular_dd_b{b_res:.4f}_w{w0}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    if analysis_results:
        df_results = pd.DataFrame(analysis_results)
        print("\n--- Analysis Results Summary ---")
        print(df_results.to_string(index=False))

    print(f"\n✅ Analysis complete. Diagnostic plots saved in: {fig_dir}")

if __name__ == "__main__":
    main()