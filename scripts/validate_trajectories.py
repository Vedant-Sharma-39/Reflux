#!/usr/bin/env python3
# FILE: scripts/validate_trajectories_final_optimized.py
#
# v8 adds a new diagnostic plot for a log(Φ) vs log(r) power-law fit
# to test an alternative physical model.

import sys
import json
import gzip
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # For custom legends
from tqdm import tqdm

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN_ID = "aif_online_scan_v1"
SUMMARY_CSV_PATH = PROJECT_ROOT / "data" / CAMPAIGN_ID / "analysis" / f"{CAMPAIGN_ID}_summary_aggregated.csv"
MUTANT_TYPES = [2, 3]

# =============================================================================
# --- USER: SET ANALYSIS PARAMETERS ---
# =============================================================================
# NOTE: Set the B_RES value to match the data you are analyzing (e.g., 0.95 or 0.97)
B_RES_TO_ANALYZE = 0.97
INITIAL_WIDTH_TO_ANALYZE = 256
# Overall minimum radius for any calculation (e.g., for displacement)
MIN_RADIUS_FOR_FIT = 200.0 
SMOOTHING_WINDOW = 20
DISPLACEMENT_WINDOW = 15
NUM_BINS_FIT = 25
# =============================================================================

# --- Helper Functions ---
def process_and_smooth_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    df['angle'] = df['width_cells'] / df['radius']
    extinction_idx = df.index[df['angle'] <= 1e-6].tolist()
    if extinction_idx: df = df.iloc[:extinction_idx[0]]
    if len(df) < SMOOTHING_WINDOW + DISPLACEMENT_WINDOW: return pd.DataFrame()
    
    df['smooth_angle'] = df['angle'].rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean()
    df['inv_radius'] = 1 / df['radius']
    
    df['d_angle'] = df['smooth_angle'].diff(DISPLACEMENT_WINDOW)
    df['d_inv_radius'] = np.abs(df['inv_radius'].diff(DISPLACEMENT_WINDOW))
    df['radius_for_filter'] = df['radius']
    return df[['radius_for_filter', 'd_angle', 'd_inv_radius']].dropna()

def bin_data_for_variance(x_data: pd.Series, y_data: pd.Series, num_bins: int) -> pd.DataFrame:
    df = pd.DataFrame({'x': x_data, 'y': y_data})
    if df['x'].nunique() < 2: return pd.DataFrame()
    try:
        df['bin'] = pd.cut(df['x'], bins=num_bins)
    except ValueError:
        return pd.DataFrame()

    df['y_sq'] = df['y']**2
    binned_stats = df.groupby('bin', observed=False).agg(
        mean_y=('y', 'mean'),
        mean_y_sq=('y_sq', 'mean'),
        count=('y', 'count')
    ).reset_index()
    
    binned_stats = binned_stats.dropna()
    binned_stats['variance'] = binned_stats['mean_y_sq'] - binned_stats['mean_y']**2
    binned_stats['bin_center'] = binned_stats['bin'].apply(lambda b: b.mid).astype(float)
    
    return binned_stats[['bin_center', 'variance', 'count']]

def calculate_mean_trajectory(list_of_dfs, radius_bins):
    all_data = pd.concat(list_of_dfs, ignore_index=True)
    all_data['bin'] = pd.cut(all_data['radius'], bins=radius_bins)
    stats = all_data.groupby('bin', observed=False)['angle'].agg(['mean', 'std', 'sem']).dropna().reset_index()
    stats['radius'] = stats['bin'].apply(lambda b: b.mid).astype(float)
    return stats

def weighted_linear_regression(x, y, weights):
    # This version is for a y=mx model (no intercept)
    x, y, weights = map(np.asarray, (x, y, weights))
    valid = np.isfinite(weights) & (weights > 0) & np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2: return 0, 0
    x, y, weights = x[valid], y[valid], weights[valid]
    
    w_x2_sum = np.sum(weights * x**2)
    w_xy_sum = np.sum(weights * x * y)
    slope = w_xy_sum / w_x2_sum if w_x2_sum != 0 else 0
    y_mean = np.average(y, weights=weights)
    ss_total = np.sum(weights * (y - y_mean)**2)
    ss_residual = np.sum(weights * (y - slope * x)**2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    return slope, r_squared

def find_optimal_asymptotic_radius(mean_traj_df, prefix):
    print("\n--- Finding optimal asymptotic radius for drift fit ---")
    test_radii = np.arange(500, 2000, 50)
    results = []

    for r_min in test_radii:
        subset_df = mean_traj_df[mean_traj_df['radius'] >= r_min].copy()
        if len(subset_df) < 5: continue
        valid_mask = np.isfinite(subset_df['sem']) & (subset_df['sem'] > 0)
        if valid_mask.sum() < 5: continue
        weights = 1 / subset_df.loc[valid_mask, 'sem']**2
        x_data = subset_df.loc[valid_mask, 'inv_radius']
        y_data = subset_df.loc[valid_mask, 'mean']
        slope, intercept = np.polyfit(x_data, y_data, 1, w=weights)
        y_pred = slope * x_data + intercept
        r_sq = 1 - (np.sum(weights * (y_data - y_pred)**2) / np.sum(weights * (y_data - np.average(y_data, weights=weights))**2))
        results.append({'r_min': r_min, 'r_sq': r_sq, 'slope': slope})
    
    if not results:
        print("Warning: Could not find a stable region. Defaulting to 1000.")
        return 1000

    results_df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df['r_min'], results_df['r_sq'], 'o-', label='R² of fit')
    ax.set_title(f'Goodness-of-Fit (R²) vs. Minimum Fit Radius for {prefix}', fontsize=16)
    ax.set_xlabel('Minimum Radius for Drift Fit (r_min)', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.grid(True)
    ax.axhline(0.95, color='r', linestyle='--', label='R² = 0.95 Threshold')
    ax.legend()
    plot_path = PROJECT_ROOT / f"r_sq_optimization_{prefix}.png"
    fig.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved R-squared optimization plot to: {plot_path}")

    high_r_sq_df = results_df[results_df['r_sq'] > 0.95]
    optimal_radius = high_r_sq_df['r_min'].iloc[0] if not high_r_sq_df.empty else 1000
    print(f"Optimal asymptotic radius found: {optimal_radius}")
    return optimal_radius

def generate_trajectory(r_start, r_end, dr, initial_angle, D_x, drift_rate):
    radii = np.arange(r_start, r_end, dr)
    angles = np.zeros_like(radii, dtype=float); angles[0] = initial_angle
    for i in range(1, len(radii)):
        d_inv_r = abs(1/radii[i] - 1/radii[i-1])
        drift_d_angle = drift_rate * d_inv_r
        variance = 4 * D_x * d_inv_r
        diffusion_d_angle = np.random.normal(scale=np.sqrt(max(0, variance)))
        angles[i] = max(0, angles[i-1] + drift_d_angle + diffusion_d_angle)
    return radii, angles

def create_diagnostic_plots(mean_traj, drift_fit_params, diffusion_bins, diffusion_fit_params, prefix, full_mean_traj, min_radius_for_drift_fit):
    slope, intercept, drift_r_sq = drift_fit_params
    diffusion_constant, diffusion_r_sq = diffusion_fit_params

    # --- Drift Fit Plot (<Φ> vs 1/r) ---
    fig_drift, ax_drift = plt.subplots(figsize=(12, 8))
    ax_drift.errorbar(full_mean_traj['inv_radius'], full_mean_traj['mean'], yerr=full_mean_traj['sem'], fmt='o',
                      label='Binned Mean Angle (All Data)', capsize=3, color='lightblue', ecolor='lightgray', zorder=1)
    ax_drift.errorbar(mean_traj['inv_radius'], mean_traj['mean'], yerr=mean_traj['sem'], fmt='o',
                      label=f'Data for Fit (r > {min_radius_for_drift_fit})', capsize=5, color='darkblue', ecolor='lightblue', zorder=2)
    fit_x = np.linspace(min(mean_traj['inv_radius']), max(mean_traj['inv_radius']), 100)
    fit_y = slope * fit_x + intercept
    ax_drift.plot(fit_x, fit_y, 'r-', lw=2,
                  label=f'Weighted Fit (R²={drift_r_sq:.3f})\nSlope = {slope:.4f}', zorder=3)
    ax_drift.set_title(f'Diagnostic: Asymptotic Drift Calculation (<Φ> vs 1/r) for {prefix}', fontsize=16)
    ax_drift.set_xlabel('Inverse Radius (1/r)', fontsize=12)
    ax_drift.set_ylabel('Mean Sector Angle <Φ>', fontsize=12)
    ax_drift.legend(); ax_drift.grid(True)
    drift_plot_path = PROJECT_ROOT / f"diagnostic_drift_fit_vs_inv_r_{prefix}.png"
    fig_drift.savefig(drift_plot_path, dpi=120, bbox_inches='tight'); plt.close(fig_drift)
    print(f"Saved drift diagnostic plot to: {drift_plot_path}")

    # --- Diffusion Fit Plot ---
    fig_diff, ax_diff = plt.subplots(figsize=(10, 7))
    ax_diff.plot(diffusion_bins['bin_center'], diffusion_bins['variance'], 'o', color='darkgreen', label='Binned Variance of ΔΦ')
    fit_x_diff = np.array([0, diffusion_bins['bin_center'].max()])
    fit_y_diff = diffusion_constant * fit_x_diff
    ax_diff.plot(fit_x_diff, fit_y_diff, 'r-', lw=2, label=f'Weighted Fit (R²={diffusion_r_sq:.3f})\nSlope (Dx) = {diffusion_constant:.4f}')
    ax_diff.set_title(f'Diagnostic: Diffusion Calculation for {prefix}', fontsize=16)
    ax_diff.set_xlabel('4 * |Δ(1/r)|', fontsize=12)
    ax_diff.set_ylabel('Var(ΔΦ)', fontsize=12)
    ax_diff.legend(); ax_diff.grid(True)
    diffusion_plot_path = PROJECT_ROOT / f"diagnostic_diffusion_fit_{prefix}.png"
    fig_diff.savefig(diffusion_plot_path, dpi=120, bbox_inches='tight'); plt.close(fig_diff)
    print(f"Saved diffusion diagnostic plot to: {diffusion_plot_path}")

    # --- NEW Power-Law Fit Plot (log-log scale) ---
    fig_power, ax_power = plt.subplots(figsize=(10, 7))
    log_data = full_mean_traj[full_mean_traj['mean'] > 0].copy()
    log_data['log_r'] = np.log10(log_data['radius'])
    log_data['log_phi'] = np.log10(log_data['mean'])
    # Propagate error: d(log10(y)) = dy / (y * ln(10))
    log_data['log_phi_err'] = log_data['sem'] / (log_data['mean'] * np.log(10))
    
    ax_power.errorbar(log_data['log_r'], log_data['log_phi'], yerr=log_data['log_phi_err'], fmt='o',
                      label='All Binned Data (log-log)', color='purple', ecolor='plum', capsize=3)
    
    asymptotic_log_data = log_data[log_data['radius'] >= min_radius_for_drift_fit]
    if len(asymptotic_log_data) >= 2:
        slope_alpha, intercept_log = np.polyfit(asymptotic_log_data['log_r'], asymptotic_log_data['log_phi'], 1)
        fit_x_log = np.linspace(min(asymptotic_log_data['log_r']), max(asymptotic_log_data['log_r']), 100)
        fit_y_log = slope_alpha * fit_x_log + intercept_log
        ax_power.plot(fit_x_log, fit_y_log, 'g-', lw=2,
                      label=f'Power-Law Fit on Asymptotic Data\nExponent α = {slope_alpha:.3f}')
        print(f"Power-Law Fit (log Φ vs log r): Exponent α = {slope_alpha:.4f}")

    ax_power.set_title(f'Diagnostic: Power-Law Relation Check for {prefix}', fontsize=16)
    ax_power.set_xlabel('log10(Radius)', fontsize=12)
    ax_power.set_ylabel('log10(Mean Sector Angle)', fontsize=12)
    ax_power.legend(); ax_power.grid(True)
    power_plot_path = PROJECT_ROOT / f"diagnostic_power_law_fit_{prefix}.png"
    fig_power.savefig(power_plot_path, dpi=120, bbox_inches='tight'); plt.close(fig_power)
    print(f"Saved power-law diagnostic plot to: {power_plot_path}")

def main():
    prefix = f"b{B_RES_TO_ANALYZE}_w{INITIAL_WIDTH_TO_ANALYZE}"
    
    # --- Data Loading ---
    param_map = pd.read_csv(SUMMARY_CSV_PATH, usecols=["task_id", "b_res", "sector_width_initial"])
    condition_map = param_map[(param_map['b_res'] == B_RES_TO_ANALYZE) & (param_map['sector_width_initial'] == INITIAL_WIDTH_TO_ANALYZE)]
    if condition_map.empty: sys.exit(f"No runs found for b_res={B_RES_TO_ANALYZE} and w0={INITIAL_WIDTH_TO_ANALYZE}")
    traj_dir = PROJECT_ROOT / "data" / CAMPAIGN_ID / "trajectories"
    all_displacements, real_trajectories_for_mean = [], []
    for run_id in tqdm(condition_map['task_id'], desc="Processing trajectories"):
        file_path = traj_dir / f"traj_{run_id}.json.gz"
        if not file_path.exists(): continue
        with gzip.open(file_path, "rt") as f: data = json.load(f)
        rows = data if isinstance(data, list) else data.get("sector_trajectory", [])
        if not rows: continue
        df = pd.DataFrame(rows).sort_values(by='radius').reset_index(drop=True)
        df_mutants = df[df['type'].isin(MUTANT_TYPES)]; 
        if df_mutants.empty: continue
        min_radius = df_mutants['radius'].min()
        founding_root_id = df_mutants[df_mutants['radius'] == min_radius]['root_sid'].iloc[0]
        lineage_df = df_mutants[df_mutants['root_sid'] == founding_root_id].copy()
        displacements = process_and_smooth_trajectory(lineage_df)
        if not displacements.empty: all_displacements.append(displacements)
        lineage_df['angle'] = lineage_df['width_cells'] / lineage_df['radius']
        real_trajectories_for_mean.append(lineage_df[['radius', 'angle']])
    if not all_displacements or not real_trajectories_for_mean: sys.exit("Not enough valid data.")
    
    # --- PART 1.5: OBJECTIVELY FIND THE BEST FIT REGION ---
    radius_bins_mean = np.arange(MIN_RADIUS_FOR_FIT, 3001, 50)
    mean_traj_full = calculate_mean_trajectory(real_trajectories_for_mean, radius_bins_mean)
    mean_traj_full['inv_radius'] = 1 / mean_traj_full['radius']
    MIN_RADIUS_FOR_DRIFT_FIT = find_optimal_asymptotic_radius(mean_traj_full, prefix)
    
    print(f"\n--- Definitive Analysis v8 (with Power-Law Diagnostic) ---")
    print(f"Using objectively determined asymptotic radius: r > {MIN_RADIUS_FOR_DRIFT_FIT}")

    # --- PART 2: Parameter Calculation ---
    mean_traj_asymptotic = mean_traj_full[mean_traj_full['radius'] >= MIN_RADIUS_FOR_DRIFT_FIT].copy()
    valid_drift_mask = np.isfinite(mean_traj_asymptotic['sem']) & (mean_traj_asymptotic['sem'] > 0)
    drift_weights = 1 / mean_traj_asymptotic.loc[valid_drift_mask, 'sem']**2
    x_drift = mean_traj_asymptotic.loc[valid_drift_mask, 'inv_radius']
    y_drift = mean_traj_asymptotic.loc[valid_drift_mask, 'mean']
    slope, intercept = np.polyfit(x_drift, y_drift, 1, w=drift_weights)
    drift_parameter = -slope 
    y_pred = slope * x_drift + intercept
    drift_r_sq = 1 - (np.sum(drift_weights * (y_drift - y_pred)**2) / np.sum(drift_weights * (y_drift - np.average(y_drift, weights=drift_weights))**2))

    df_disp_all = pd.concat(all_displacements, ignore_index=True)
    df_disp = df_disp_all[df_disp_all['radius_for_filter'] >= MIN_RADIUS_FOR_FIT].copy()
    diffusion_bins = bin_data_for_variance(4 * df_disp['d_inv_radius'], df_disp['d_angle'], NUM_BINS_FIT)
    diffusion_weights = diffusion_bins['count']
    diffusion_constant, diffusion_r_sq = weighted_linear_regression(diffusion_bins['bin_center'], diffusion_bins['variance'], diffusion_weights)
    
    print("\n--- Best-Fit Parameters (Asymptotic Model) ---")
    print(f"Drift Fit (<Φ> vs 1/r, on r > {MIN_RADIUS_FOR_DRIFT_FIT}): Slope (m)={slope:.4f}, Intercept={intercept:.4f}, R²={drift_r_sq:.3f}")
    print(f"==> Inferred Asymptotic Drift Parameter for Sim: {drift_parameter:.4f}")
    print(f"Diffusion Fit (<Var(ΔΦ)> vs 4|Δ(1/r)|): Slope={diffusion_constant:.4f}, R²={diffusion_r_sq:.3f}")
    print(f"==> Inferred Diffusion Constant (D_x): {diffusion_constant:.4f}")

    create_diagnostic_plots(
        mean_traj_asymptotic[valid_drift_mask],
        (slope, intercept, drift_r_sq),
        diffusion_bins,
        (diffusion_constant, diffusion_r_sq),
        prefix,
        mean_traj_full,
        MIN_RADIUS_FOR_DRIFT_FIT
    )

    # --- PART 3: VISUALIZATION ---
    print("\nGenerating final validation plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 14, 'figure.figsize': (14, 8)})
    fig_val, ax_val = plt.subplots()
    simulated_trajectories = []
    
    initial_angles_for_sim = []
    for traj in real_trajectories_for_mean:
        asymptotic_part = traj[traj['radius'] >= MIN_RADIUS_FOR_DRIFT_FIT]
        if not asymptotic_part.empty:
            initial_angles_for_sim.append(asymptotic_part['angle'].iloc[0])
    if not initial_angles_for_sim:
        sys.exit(f"Error: No real trajectory data found beyond r={MIN_RADIUS_FOR_DRIFT_FIT} to initialize simulation.")
    avg_initial_angle_asymptotic = np.mean(initial_angles_for_sim)
    
    print(f"Starting simulation at r={MIN_RADIUS_FOR_DRIFT_FIT} with average angle={avg_initial_angle_asymptotic:.4f}")

    for _ in tqdm(range(len(real_trajectories_for_mean)), desc="Generating simulated trajectories"):
        radii, angles = generate_trajectory(
            r_start=MIN_RADIUS_FOR_DRIFT_FIT, r_end=3000.0, dr=10.0, 
            initial_angle=avg_initial_angle_asymptotic, 
            D_x=diffusion_constant, drift_rate=drift_parameter
        )
        simulated_trajectories.append(pd.DataFrame({'radius': radii, 'angle': angles}))

    radius_bins_plot = np.arange(0, 3001, 50)
    real_stats = calculate_mean_trajectory(real_trajectories_for_mean, radius_bins_plot)
    sim_stats = calculate_mean_trajectory(simulated_trajectories, radius_bins_plot)
    
    legend_elements = [Line2D([0], [0], color='royalblue', alpha=0.2, lw=4, label='Real Data (±1σ)'), Line2D([0], [0], color='crimson', alpha=0.3, lw=4, label='Simulated Data (±1σ)'), Line2D([0], [0], color='blue', lw=3, label='Mean (Real)'), Line2D([0], [0], color='darkred', lw=3, ls='--', label='Mean (Simulated)')]
    ax_val.fill_between(real_stats['radius'], real_stats['mean'] - real_stats['std'], real_stats['mean'] + real_stats['std'], color='royalblue', alpha=0.2)
    ax_val.fill_between(sim_stats['radius'], sim_stats['mean'] - sim_stats['std'], sim_stats['mean'] + sim_stats['std'], color='crimson', alpha=0.3)
    ax_val.plot(real_stats['radius'], real_stats['mean'], color='blue', lw=3)
    ax_val.plot(sim_stats['radius'], sim_stats['mean'], color='darkred', lw=3, ls='--')
    ax_val.set_title(f"Definitive Validation for {prefix}", fontsize=20, pad=15)
    ax_val.set_xlabel("Colony Radius (cells)", fontsize=16)
    ax_val.set_ylabel("Sector Angle (radians)", fontsize=16)
    ax_val.legend(handles=legend_elements, fontsize=12, loc='best')
    ax_val.set_ylim(bottom=0); ax_val.set_xlim(left=0)
    validation_path = PROJECT_ROOT / f"validation_definitive_optimized_{prefix}.png"
    print(f"\nSaving final validation plot to: {validation_path}")
    fig_val.savefig(validation_path, dpi=150, bbox_inches='tight'); plt.close(fig_val)
    print("\n✅ Definitive analysis and validation complete.")

if __name__ == "__main__":
    main()