import os
import sys
import json
import gzip
from pathlib import Path
from typing import Optional, Dict, NamedTuple, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import linregress
from tqdm import tqdm
import multiprocessing
from functools import partial

# --- Constants ---
RADIAL_CAMPAIGN_ID = "aif_online_scan_v1"
TARGET_BM = 0.97
INFERENCE_W0 = 256
PREDICTION_W0 = 64
BIN_SIZE = 15.0
MUTANT_TYPES = [2, 3]

RADIAL_FIT_RADIUS_MIN = 500.0
RADIAL_FIT_RADIUS_MAX = 2000.0
MIN_POINTS_FOR_FIT = 4
SIM_PATHS = 2000
SIM_SEED = 42

# --- Path Setup ---
try:
    THIS_FILE = Path(__file__).resolve()
    PROJECT_ROOT = THIS_FILE.parents[2]
except NameError:
    PROJECT_ROOT = Path(".").resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / RADIAL_CAMPAIGN_ID
ANALYSIS_DIR = DATA_DIR / "analysis"
TRAJ_DIR = DATA_DIR / "trajectories"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILENAME = f"figure_thesis_inflation_selection_balance_v4_w{INFERENCE_W0}_w{PREDICTION_W0}" # Updated version in filename

# --- PUBLICATION-QUALITY COLOR PALETTE ---
COLOR_EMP_MEAN = '#0072B2'      # Blue (Lattice Sim Mean)
COLOR_EMP_PERC = '#aaddff'      # Lighter Blue (Lattice Sim Percentile)
COLOR_SIM_MEAN_COND = '#D55E00' # Orange/Vermillion (Cond. RW Model Mean)
COLOR_SIM_PERC = '#FFBB78'      # Lighter Orange (Cond. RW Model Percentile) - RESTORED
COLOR_SIM_MEAN_UNCOND = '#444444'# Dark Gray (Uncond. RW Model Mean)
COLOR_SURVIVAL = '#009E73'      # Green (Survival)

# --- Data Structures and Core Functions (Unchanged) ---

class ProcessedData(NamedTuple):
    radius_grid: np.ndarray
    traj_array: np.ndarray
    mean_surv: np.ndarray
    std_surv: np.ndarray

# (infer_root_sid_numpy, process_single_radial_run functions remain the same)
def infer_root_sid_numpy(types: np.ndarray, radii: np.ndarray, widths: np.ndarray, roots: np.ndarray) -> Optional[int]:
    mask_mut = np.isin(types, MUTANT_TYPES)
    if not mask_mut.any(): return None
    valid_radii = radii[mask_mut]
    if valid_radii.size == 0: return None
    rmin = valid_radii.min()
    sl = mask_mut & (np.abs(radii - rmin) < 1e-9)
    if not sl.any(): return None
    agg: Dict[int, float] = {}
    valid_roots = roots[sl]
    valid_widths = widths[sl]
    if valid_roots.size == 0: return None
    for rt, w in zip(valid_roots, valid_widths):
        rt_int = int(rt)
        agg[rt_int] = agg.get(rt_int, 0.0) + w
    return max(agg, key=agg.get) if agg else None

def process_single_radial_run(task: Dict, bin_size: float) -> Optional[Dict]:
    traj_path = Path(task["traj_path"])
    if not traj_path.exists(): return None
    try:
        with gzip.open(traj_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else data.get("sector_trajectory", [])
        if not rows: return None
        df = pd.DataFrame(rows)
        if not all(col in df.columns for col in ["type", "radius", "width_cells", "root_sid"]):
            return None
        df = df.astype({"type": np.int32, "radius": float, "width_cells": float, "root_sid": np.int32}, errors='ignore')
        types, r, w, roots = df["type"].to_numpy(np.int32), df["radius"].to_numpy(float), df["width_cells"].to_numpy(float), df["root_sid"].to_numpy(np.int32)
        if np.isnan(r).any() or np.isnan(w).any():
             valid_mask = ~np.isnan(r) & ~np.isnan(w)
             types, r, w, roots = types[valid_mask], r[valid_mask], w[valid_mask], roots[valid_mask]
             if r.size == 0: return None
        root = infer_root_sid_numpy(types, r, w, roots)
        if root is None: return None
        mask = (np.isin(types, MUTANT_TYPES)) & (roots == root)
        if not mask.any(): return None
        df_lineage = df[mask].copy()
        df_lineage["bin_idx"] = np.floor((df_lineage["radius"] + 1e-9) / bin_size).astype(np.int64)
        df_lineage.sort_values("radius", inplace=True)
        df_first = df_lineage.drop_duplicates(subset=["bin_idx"], keep="first")
        if df_first.empty: return None
        adv_bins_raw = (df_first["bin_idx"].to_numpy() * bin_size).astype(float)
        width_bins_raw = df_first["width_cells"].to_numpy(float)
        finite_mask = np.isfinite(adv_bins_raw) & np.isfinite(width_bins_raw)
        adv_bins = adv_bins_raw[finite_mask]
        width_bins = width_bins_raw[finite_mask]
        if adv_bins.size == 0: return None
        return {
            "run_id": task["run_id"],
            "adv_bins": adv_bins,
            "width_bins": width_bins
        }
    except Exception as e:
        # print(f"Error processing {traj_path}: {e}") # Debugging
        return None

# (load_and_process_data, simulate_radial_rw_conditioned, infer_parameters functions remain the same)
def load_and_process_data(bin_size: float, b_m_filter: float, initial_width_filter: int) -> Optional[ProcessedData]:
    summary_path = ANALYSIS_DIR / f"{RADIAL_CAMPAIGN_ID}_summary_aggregated.csv"
    if not summary_path.exists():
        print(f"Error: Radial summary file not found: {summary_path}")
        return None
    try:
        df_map = pd.read_csv(summary_path, usecols=["task_id", "b_res", "sector_width_initial"]).rename(columns={"task_id": "run_id", "sector_width_initial": "initial_width"})
        df_map.dropna(subset=['initial_width'], inplace=True)
        df_map['initial_width'] = df_map['initial_width'].astype(int)
        df_map_filtered = df_map[
            np.isclose(df_map['b_res'], b_m_filter, atol=0.01) &
            (df_map['initial_width'] == initial_width_filter)
        ]
        if df_map_filtered.empty:
            print(f"Warning: No runs found for b_m≈{b_m_filter} and W₀={initial_width_filter}.")
            return None
        tasks = [dict(row, traj_path=str(TRAJ_DIR / f"traj_{row['run_id']}.json.gz")) for _, row in df_map_filtered.iterrows()]
        n_cpu = max(1, (os.cpu_count() or 2) - 1)
        print(f"Loading W₀={initial_width_filter} using {n_cpu} processes...")
        with multiprocessing.Pool(processes=n_cpu) as pool:
            results = list(tqdm(pool.imap(partial(process_single_radial_run, bin_size=bin_size), tasks),
                                  total=len(tasks), desc=f"Loading W₀={initial_width_filter}", unit="file"))
        valid_results = [res for res in results if res and "adv_bins" in res and res["adv_bins"].size > 0]
        if not valid_results:
            print(f"Warning: No valid trajectories could be loaded for W₀={initial_width_filter}.")
            return None
        all_long_dfs = [pd.DataFrame({"run_id": res["run_id"], "radius_bin": res["adv_bins"], "width_bin": res["width_bins"]}) for res in valid_results]
        df_long = pd.concat(all_long_dfs, ignore_index=True)
        min_r = df_long['radius_bin'].min()
        max_r = df_long['radius_bin'].max()
        if pd.isna(min_r) or pd.isna(max_r):
             print(f"Warning: Could not determine valid radius range for W₀={initial_width_filter}.")
             return None
        radius_grid = np.arange(min_r, max_r + bin_size, bin_size)
        processed_trajectories = []
        run_ids_processed = set()
        for run_id, group in df_long.groupby('run_id'):
             if run_id in run_ids_processed: continue
             s = pd.Series(group['width_bin'].values, index=group['radius_bin'].values)
             s = s[~s.index.duplicated(keep='first')]
             s = s.reindex(radius_grid)
             last_valid_idx = s.last_valid_index()
             if pd.notna(last_valid_idx):
                 last_valid_loc = s.index.get_loc(last_valid_idx)
                 s.iloc[last_valid_loc:] = s.iloc[last_valid_loc:].fillna(0)
             s = s.fillna(0)
             processed_trajectories.append(s.to_numpy())
             run_ids_processed.add(run_id)
        if not processed_trajectories:
            print(f"Warning: No trajectories remained after processing for W₀={initial_width_filter}.")
            return None
        traj_array = np.array(processed_trajectories)
        stats_arr = np.where(traj_array > 1e-9, traj_array, np.nan)
        mean_surv = np.full(radius_grid.shape, np.nan)
        std_surv = np.full(radius_grid.shape, np.nan)
        valid_counts = np.sum(~np.isnan(stats_arr), axis=0)
        valid_indices = valid_counts > 0
        with np.errstate(invalid='ignore'):
             if np.any(valid_indices):
                 mean_surv[valid_indices] = np.nanmean(stats_arr[:, valid_indices], axis=0)
                 valid_std_indices = valid_counts > 1
                 if np.any(valid_std_indices):
                     std_surv[valid_std_indices] = np.nanstd(stats_arr[:, valid_std_indices], axis=0)
    except Exception as e:
         print(f"Unexpected error during data loading/processing for W₀={initial_width_filter}: {type(e).__name__} - {e}")
         import traceback
         traceback.print_exc()
         return None
    return ProcessedData(radius_grid, traj_array, mean_surv, std_surv)

def simulate_radial_rw_conditioned(n_paths, r_grid, w0, m_perp, D_X, seed):
    rng = np.random.default_rng(seed)
    r = np.asarray(r_grid, dtype=float)
    nT = r.size
    phi = np.zeros((n_paths, nT))
    alive = np.ones((n_paths, nT), dtype=bool)
    r0 = r[0]
    phi[:, 0] = w0 / r0 if r0 > 1e-9 else 0.0
    for t in range(1, nT):
        dr = r[t] - r[t-1]
        rt_prev = r[t-1]
        if rt_prev <= 1e-9: continue
        currently_alive = alive[:, t-1]
        if not np.any(currently_alive):
            alive[:, t:] = False
            break
        n_alive = np.sum(currently_alive)
        dW_noise = rng.normal(loc=0.0, scale=np.sqrt(max(0, dr)), size=n_alive)
        drift_term = (2.0 * m_perp / rt_prev) * dr
        diffusion_term = (np.sqrt(max(0, 4.0 * D_X)) / rt_prev) * dW_noise
        phi_prev_alive = phi[currently_alive, t-1]
        phi_new = phi_prev_alive + drift_term + diffusion_term
        survived_mask = phi_new > 1e-12
        phi[currently_alive, t] = np.where(survived_mask, phi_new, 0.0)
        temp_alive = np.zeros(n_paths, dtype=bool)
        temp_alive[currently_alive] = survived_mask
        alive[:, t] = temp_alive
    W_sim = phi * r[None, :]
    return np.where(alive, W_sim, np.nan)

def infer_parameters(empirical_data: ProcessedData) -> Tuple[float, float]:
    df_stats = pd.DataFrame({
        'radius_bin': empirical_data.radius_grid,
        'mean_width': empirical_data.mean_surv,
        'var_width': empirical_data.std_surv**2
    })
    df_stats = df_stats.dropna(subset=['mean_width', 'radius_bin', 'var_width'])
    df_stats = df_stats[df_stats['mean_width'] > 1e-9].copy()
    df_stats = df_stats[df_stats['radius_bin'] > 1e-9].copy()
    df_stats['angle'] = df_stats['mean_width'] / df_stats['radius_bin']
    df_stats['angle_var'] = df_stats['var_width'] / df_stats['radius_bin']**2
    fit_df = df_stats[
        (df_stats['radius_bin'] >= RADIAL_FIT_RADIUS_MIN) &
        (df_stats['radius_bin'] <= RADIAL_FIT_RADIUS_MAX)
    ].copy()
    if len(fit_df) < MIN_POINTS_FOR_FIT:
        print(f"Warning: Only {len(fit_df)} points available for parameter fitting (min {MIN_POINTS_FOR_FIT}). Returning NaN.")
        return np.nan, np.nan
    fit_df['log_r'] = np.log(fit_df['radius_bin'])
    valid_angle_fit = fit_df.dropna(subset=['log_r', 'angle'])
    if len(valid_angle_fit) < MIN_POINTS_FOR_FIT:
         print(f"Warning: Not enough valid points for m_perp fit ({len(valid_angle_fit)}). Returning NaN.")
         return np.nan, np.nan
    slope_m, intercept_m, r_value_m, p_value_m, std_err_m = linregress(valid_angle_fit['log_r'], valid_angle_fit['angle'])
    m_perp = slope_m / 2.0
    r0_fit = fit_df['radius_bin'].iloc[0]
    fit_df['inv_r_term'] = (1.0 / r0_fit) - (1.0 / fit_df['radius_bin'])
    valid_var_fit = fit_df.dropna(subset=['inv_r_term', 'angle_var'])
    if len(valid_var_fit) < MIN_POINTS_FOR_FIT:
         print(f"Warning: Not enough valid points for D_X fit ({len(valid_var_fit)}). Returning NaN.")
         return np.nan, np.nan
    slope_D, intercept_D, r_value_D, p_value_D, std_err_D = linregress(valid_var_fit['inv_r_term'], valid_var_fit['angle_var'])
    D_X = slope_D / 4.0
    if D_X < 0:
        print(f"Warning: Inferred D_X is negative ({D_X:.2f}). Capping at 0.01.")
        D_X = 0.01
    print(f"Inferred parameters: m_perp = {m_perp:.4f}, D_X = {D_X:.4f}")
    print(f"Fit stats for m_perp: R^2={r_value_m**2:.3f}, p={p_value_m:.3g}")
    print(f"Fit stats for D_X: R^2={r_value_D**2:.3f}, p={p_value_D:.3g}")
    return m_perp, D_X


# --- REVISED Plotting Function (v4: Single Param Box, RW band restored) ---

def plot_panel(ax: plt.Axes, empirical_data: ProcessedData, params: Dict[str, float], title: str,
               is_prediction_panel: bool, add_legend: bool, add_right_ylabel: bool, show_param_box: bool): # Added show_param_box flag
    """Plots lattice sim data, conditioned RW model, and unconditioned RW mean."""
    m_perp, D_X = params['m_perp'], params['D_X']

    # --- Find start point for simulation/unconditioned mean ---
    start_mask = empirical_data.radius_grid >= RADIAL_FIT_RADIUS_MIN
    first_valid_idx_overall = np.where(~np.isnan(empirical_data.mean_surv) & (empirical_data.mean_surv > 1e-9))[0]

    if len(first_valid_idx_overall) == 0:
        print(f"Warning: No valid lattice sim mean width data found for {title}. Plotting lattice data only.")
        plot_lattice_only = True
        r_start, w_start = np.nan, np.nan
        r_grid_sim = np.array([])
        mean_w_sim_cond = np.array([])
        mean_w_sim_uncond = np.array([])
        W_sim = np.array([[]])
        p10_sim, p90_sim = np.array([]), np.array([]) # Initialize sim percentiles
    else:
        plot_lattice_only = False
        default_start_idx = -1
        if np.any(start_mask):
            potential_start_indices = np.where(start_mask)[0]
            for idx in potential_start_indices:
                if idx < len(empirical_data.mean_surv) and ~np.isnan(empirical_data.mean_surv[idx]) and empirical_data.mean_surv[idx] > 1e-9:
                    default_start_idx = idx
                    break
        if default_start_idx == -1:
            start_idx = first_valid_idx_overall[0]
            print(f"Warning: No valid data >= {RADIAL_FIT_RADIUS_MIN} units. Using first valid data point r={empirical_data.radius_grid[start_idx]:.1f} units as simulation start for {title}.")
        else:
            start_idx = default_start_idx
        r_start, w_start = empirical_data.radius_grid[start_idx], empirical_data.mean_surv[start_idx]
        max_r_empirical = empirical_data.radius_grid[~np.isnan(empirical_data.radius_grid)].max()
        if np.isnan(max_r_empirical) or r_start >= max_r_empirical:
            print(f"Warning: Cannot define RW model grid for {title}. Skipping RW model plots.")
            plot_lattice_only = True
            r_grid_sim = np.array([])
            mean_w_sim_cond = np.array([])
            mean_w_sim_uncond = np.array([])
            W_sim = np.array([[]])
            p10_sim, p90_sim = np.array([]), np.array([])
        else:
            r_grid_sim = np.linspace(r_start, max_r_empirical, 400)
            # --- Run CONDITIONED RW model ---
            W_sim = simulate_radial_rw_conditioned(SIM_PATHS, r_grid_sim, w_start, m_perp, D_X, seed=SIM_SEED)
            with np.errstate(invalid="ignore"):
                mean_w_sim_cond = np.nanmean(W_sim, axis=0)
                # RESTORED: Calculate simulated percentiles
                p10_sim = np.nanpercentile(W_sim, 10, axis=0)
                p90_sim = np.nanpercentile(W_sim, 90, axis=0)
                p10_sim = np.nan_to_num(p10_sim, nan=0.0) # Fallback if <10% survive

            # --- Calculate UNCONDITIONED RW mean trajectory ---
            if r_start > 1e-9:
                ratio_r = r_grid_sim / r_start
                log_ratio_r = np.log(ratio_r)
                mean_w_sim_uncond = w_start * ratio_r + 2.0 * m_perp * r_grid_sim * log_ratio_r
            else:
                mean_w_sim_uncond = np.full_like(r_grid_sim, np.nan)

    # === Plotting ===

    # --- Left Y-Axis (Width) ---

    # Calculate Lattice Sim Percentiles
    with np.errstate(invalid="ignore"):
        p10_emp = np.nanpercentile(np.where(empirical_data.traj_array > 1e-9, empirical_data.traj_array, np.nan), 10, axis=0)
        p90_emp = np.nanpercentile(np.where(empirical_data.traj_array > 1e-9, empirical_data.traj_array, np.nan), 90, axis=0)
        p10_emp = np.nan_to_num(p10_emp, nan=0.0)

    # Plot Lattice Sim Percentile Band
    ax.fill_between(empirical_data.radius_grid, p10_emp, p90_emp,
                    color=COLOR_EMP_PERC, alpha=0.4,
                    label='10-90th pct. (Lattice Sim.)', zorder=2)

    # Plot Lattice Sim mean (blue)
    ax.plot(empirical_data.radius_grid, empirical_data.mean_surv,
            color=COLOR_EMP_MEAN, lw=2.5,
            label='Mean (Lattice Sim.)', zorder=5)

    if not plot_lattice_only:
        # RESTORED: Plot Simulated Percentile Band for prediction panel
        if is_prediction_panel:
            ax.fill_between(r_grid_sim, p10_sim, p90_sim,
                            color=COLOR_SIM_PERC, alpha=0.4,
                            label='10-90th pct. (Cond. RW Model)', zorder=1) # Restored label

        # Plot Conditioned RW Model mean (orange)
        sim_lw = 2.8 if is_prediction_panel else 2.5
        ax.plot(r_grid_sim, mean_w_sim_cond,
                color=COLOR_SIM_MEAN_COND, lw=sim_lw, ls='-',
                label='Mean (Cond. RW Model)', zorder=4)

        # Plot Unconditioned RW Model mean (dashed gray)
        if is_prediction_panel:
            ax.plot(r_grid_sim, mean_w_sim_uncond,
                    color=COLOR_SIM_MEAN_UNCOND, lw=2.0, ls='--',
                    label='Mean (Uncond. RW Model)', zorder=3)

    # --- Right Y-axis (Survival Fraction) ---
    ax2 = ax.twinx()

    # Plot LATTICE SIM survival fraction
    if empirical_data.traj_array.shape[0] > 0:
        valid_counts_emp = np.sum(~np.isnan(np.where(empirical_data.traj_array > 1e-9, empirical_data.traj_array, np.nan)), axis=0)
        first_data_idx_emp = np.where(valid_counts_emp > 0)[0]
        if first_data_idx_emp.size > 0:
            norm_factor_emp = valid_counts_emp[first_data_idx_emp[0]]
            survival_fraction_emp = valid_counts_emp / norm_factor_emp
            ax2.plot(empirical_data.radius_grid[first_data_idx_emp[0]:],
                     survival_fraction_emp[first_data_idx_emp[0]:],
                     color=COLOR_SURVIVAL, lw=2.0, ls=':',
                     label='Survival (Lattice Sim.)', zorder=3)

    if not plot_lattice_only:
        # Plot CONDITIONED RW MODEL survival fraction
        n_surv_sim = np.sum(~np.isnan(W_sim), axis=0)
        if n_surv_sim.size > 0 and n_surv_sim[0] > 0:
            survival_fraction_sim = n_surv_sim / n_surv_sim[0]
        else:
            survival_fraction_sim = np.zeros_like(n_surv_sim)
        ax2.plot(r_grid_sim, survival_fraction_sim,
                 color=COLOR_SURVIVAL, lw=2.0, ls='-.',
                 label='Survival (Cond. RW Model)', zorder=3) # Green

    # Configure Right Y-axis
    ax2.tick_params(axis='y', colors=COLOR_SURVIVAL, labelsize=11)
    ax2.set_ylim(-0.02, 1.05)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    if add_right_ylabel:
        ax2.set_ylabel("Survival Fraction ($S$)", fontsize=13, color=COLOR_SURVIVAL, rotation=-90, labelpad=20, fontweight='bold')
    else:
        ax2.set_yticks([])
        ax2.set_yticklabels([])

    # --- Axes and Labels ---
    ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
    ax.set_xlabel("Colony Radius ($r$, cell widths)", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5, color='gray')
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.tick_params(which='minor', length=2, color='gray')

    # --- Parameter Box (Conditional) ---
    # *** CHANGE: Only show if requested ***
    if show_param_box:
        param_box_pos = (0.97, 0.97)
        param_text = f"$m_\\perp = {m_perp:.3f}$\n$D_X = {D_X:.2f}$"
        ax.text(param_box_pos[0], param_box_pos[1], param_text,
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="darkgray", alpha=0.8),
                fontsize=10)

    # --- Set Axis Limits ---
    min_r_data = empirical_data.radius_grid[~np.isnan(empirical_data.radius_grid)].min()
    max_r_data = empirical_data.radius_grid[~np.isnan(empirical_data.radius_grid)].max()
    if pd.isna(min_r_data): min_r_data = 0
    if pd.isna(max_r_data): max_r_data = RADIAL_FIT_RADIUS_MAX
    ax.set_xlim(left=max(0, min_r_data - 0.05 * (max_r_data - min_r_data)),
                right=max_r_data + 0.05 * (max_r_data - min_r_data))
    if p90_emp.size > 0:
        max_y_emp = np.nanmax(p90_emp)
    else:
        max_y_emp = 0
    # RESTORED: Use p90_sim if available for max_y calculation
    if not plot_lattice_only and p90_sim.size > 0:
        max_y_sim = np.nanmax(p90_sim)
    elif not plot_lattice_only and mean_w_sim_cond.size > 0:
        max_y_sim = np.nanmax(mean_w_sim_cond) # Fallback if percentiles weren't calculated somehow
    else:
        max_y_sim = 0
    max_y = max(max_y_emp, max_y_sim, 1.0)
    ax.set_ylim(bottom=-0.05 * max_y, top=max_y * 1.1)

    # --- Legend (only if requested) ---
    if add_legend:
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        available_handles = {**dict(zip(labels1, handles1)), **dict(zip(labels2, handles2))}

        # RESTORED: Add back Cond RW Percentile to desired order
        desired_order = [
            'Mean (Lattice Sim.)',
            '10-90th pct. (Lattice Sim.)',
            'Mean (Cond. RW Model)',
            '10-90th pct. (Cond. RW Model)', # <-- RESTORED
            'Mean (Uncond. RW Model)',
            'Survival (Lattice Sim.)',
            'Survival (Cond. RW Model)'
        ]
        final_handles = []
        final_labels = []
        for label in desired_order:
            if label in available_handles:
                final_handles.append(available_handles[label])
                final_labels.append(label)
        ax.legend(handles=final_handles, labels=final_labels,
                  loc='upper left', fontsize=9.5, frameon=True,
                  facecolor='white', framealpha=0.85, ncol=1)


# --- Main Execution ---

def main():
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 13, "axes.titlesize": 15,
        "legend.fontsize": 9.5, "figure.titlesize": 17,
        "axes.labelweight": "bold", "axes.titleweight": "bold",
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white"
    })

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    fig.suptitle(f"Inflation-Selection Balance Stabilizes Mutant Lineages ($b_m \\approx {TARGET_BM}$)",
                 fontsize=17, fontweight='bold')

    # --- Panel A: Parameter Inference (from W0=256) ---
    print("Processing data for Panel A (Inference)...")
    data_A = load_and_process_data(BIN_SIZE, TARGET_BM, INFERENCE_W0)

    if data_A is None:
        print("Error: Failed to load data for Panel A. Exiting.")
        axA.set_title(f"(A) Parameter Inference ($W_0 = {INFERENCE_W0}$ cell widths)\n--- DATA FAILED TO LOAD ---", color='red', fontsize=12)
        axB.set_title(f"(B) Model Prediction ($W_0 = {PREDICTION_W0}$ cell widths)\n--- DEPENDENT ON PANEL A ---", color='red', fontsize=12)
        plt.tight_layout(rect=[0, 0.02, 1, 0.93])
        plt.savefig(FIG_DIR / f"ERROR_{OUTPUT_FILENAME}.png")
        return

    m_perp, D_X = infer_parameters(data_A)
    if pd.isna(m_perp) or pd.isna(D_X):
        print("Error: Could not infer parameters from W0=256 data. Exiting.")
        axA.set_title(f"(A) Parameter Inference ($W_0 = {INFERENCE_W0}$ cell widths)\n--- PARAMETER INFERENCE FAILED ---", color='red', fontsize=12)
        axB.set_title(f"(B) Model Prediction ($W_0 = {PREDICTION_W0}$ cell widths)\n--- DEPENDENT ON PANEL A ---", color='red', fontsize=12)
        plt.tight_layout(rect=[0, 0.02, 1, 0.93])
        plt.savefig(FIG_DIR / f"ERROR_{OUTPUT_FILENAME}.png")
        return

    params = {'m_perp': m_perp, 'D_X': D_X}

    # Plot Panel A (Lattice Sim + RW Model Fit)
    plot_panel(axA, data_A, params,
               title=f"(A) Parameter Inference ($W_0 = {INFERENCE_W0}$ cell widths)",
               is_prediction_panel=False, add_legend=False, add_right_ylabel=False,
               show_param_box=True) # *** CHANGE: Show box here ***
    axA.set_ylabel("Sector Width ($W$, cell widths)", fontsize=13, fontweight='bold')

    # --- Panel B: Model Prediction (for W0=64) ---
    print("\nProcessing data for Panel B (Prediction)...")
    data_B = load_and_process_data(BIN_SIZE, TARGET_BM, PREDICTION_W0)

    if data_B is None:
        print("Error: Failed to load data for Panel B. Skipping Panel B plot.")
        axB.set_title(f"(B) Model Prediction ($W_0 = {PREDICTION_W0}$ cell widths)\n--- DATA FAILED TO LOAD ---", color='red', fontsize=12)
    else:
        # Plot Panel B (Lattice Sim vs. RW Model Prediction)
        plot_panel(axB, data_B, params,
                   title=f"(B) Model Prediction ($W_0 = {PREDICTION_W0}$ cell widths)",
                   is_prediction_panel=True, add_legend=True, add_right_ylabel=True,
                   show_param_box=False) # *** CHANGE: Do NOT show box here ***
        axB.set_ylabel("Sector Width ($W$, cell widths)", fontsize=13, fontweight='bold')

    # --- Final Figure Adjustments ---
    sns.despine(fig=fig)
    fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.93], pad=1.5, h_pad=2.0)

    out_path_png = FIG_DIR / f"{OUTPUT_FILENAME}.png"
    out_path_pdf = FIG_DIR / f"{OUTPUT_FILENAME}.pdf"

    print(f"\nSaving PNG figure to: {out_path_png}")
    try:
        fig.savefig(out_path_png, dpi=300, bbox_inches='tight', facecolor='white')
        print("PNG figure saved successfully.")
    except Exception as e:
        print(f"Error saving PNG figure: {e}")

    print(f"Saving PDF figure to: {out_path_pdf}")
    try:
        fig.savefig(out_path_pdf, bbox_inches='tight', facecolor='white')
        print("PDF figure saved successfully.")
    except Exception as e:
        print(f"Error saving PDF figure: {e}")


if __name__ == "__main__":
    if not ANALYSIS_DIR.exists() or not TRAJ_DIR.exists():
         print(f"Error: Required data directories not found.")
         print(f"Looked for analysis summary in: {ANALYSIS_DIR}")
         print(f"Looked for trajectories in: {TRAJ_DIR}")
         print(f"Please check the PROJECT_ROOT path definition and data location.")
         ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    main()