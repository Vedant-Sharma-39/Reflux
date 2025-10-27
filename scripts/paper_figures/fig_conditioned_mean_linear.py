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
import warnings

# -----------------------------------------------------------------
# !!! USER: SET ALL PARAMETERS HERE !!!
#
# --- General Parameters ---
TARGET_BM = 0.97  # The b_m value to use for ALL panels
INFERENCE_W0 = 256 # Initial width for Panels A & C (Inference)
PREDICTION_W0 = 64  # Initial width for Panels B & D (Prediction)
SIM_PATHS = 2000
SIM_SEED = 42

# --- Linear Campaign Parameters ---
LINEAR_CAMPAIGN_ID = "boundary_experiment_v1"
LINEAR_ADV_COL = "advancement"
LINEAR_WIDTH_COL = "width"
LINEAR_BIN_SIZE = 20.0
LINEAR_FIT_MIN = 10.0
LINEAR_FIT_MAX_LIMIT = 1800.0
LINEAR_MIN_POINTS = 5
LINEAR_MIN_RUNS = 3

# --- Radial Campaign Parameters ---
RADIAL_CAMPAIGN_ID = "aif_online_scan_v1"
RADIAL_BIN_SIZE = 15.0
RADIAL_MUTANT_TYPES = [2, 3] # from your original script
RADIAL_FIT_MIN = 500.0
RADIAL_FIT_MAX = 2000.0
RADIAL_MIN_POINTS = 4
#
# -----------------------------------------------------------------

# --- Path Setup ---
try:
    THIS_FILE = Path(__file__).resolve()
    PROJECT_ROOT = THIS_FILE.parents[2]
except NameError:
    PROJECT_ROOT = Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Linear Paths
DATA_DIR_LIN = PROJECT_ROOT / "data" / LINEAR_CAMPAIGN_ID
ANALYSIS_DIR_LIN = DATA_DIR_LIN / "analysis"
TRAJ_DIR_LIN = DATA_DIR_LIN / "trajectories"
# Radial Paths
DATA_DIR_RAD = PROJECT_ROOT / "data" / RADIAL_CAMPAIGN_ID
ANALYSIS_DIR_RAD = DATA_DIR_RAD / "analysis"
TRAJ_DIR_RAD = DATA_DIR_RAD / "trajectories"
# Figure Path
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILENAME = f"figure_2_linear_radial_comparison"

# --- Color Palette ---
COLOR_EMP_MEAN = '#0072B2'      # Blue (Lattice Sim Mean)
COLOR_EMP_PERC = '#aaddff'      # Lighter Blue (Lattice Sim Percentile)
COLOR_SIM_MEAN_COND = '#D55E00' # Orange/Vermillion (Cond. RW Model Mean)
COLOR_SIM_PERC = '#FFBB78'      # Lighter Orange (Cond. RW Model Percentile)
COLOR_SIM_MEAN_UNCOND = '#444444'# Dark Gray (Uncond. RW Model Mean)
COLOR_SURVIVAL = '#009E73'      # Green (Survival)

# --- Data Structures ---
class ProcessedData(NamedTuple):
    x_grid: np.ndarray # Advancement or Radius
    traj_array: np.ndarray
    mean_surv: np.ndarray
    std_surv: np.ndarray

# --- Helper functions ---
def _safe_r2(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    if x.size < 2: return np.nan
    yhat = intercept + slope * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return np.nan if ss_tot <= 1e-15 else 1.0 - ss_res / ss_tot

def _np_wls_linear(x: np.ndarray, y: np.ndarray, w_sigma: np.ndarray) -> Tuple[float, float]:
    coeffs = np.polyfit(x, y, deg=1, w=w_sigma)
    return coeffs[0], coeffs[1]

# -----------------------------------------------------------------
# --- LINEAR DATA FUNCTIONS ---
# -----------------------------------------------------------------

def process_single_run_linear(task: dict, bin_size: float) -> Optional[dict]:
    traj_path = Path(task["traj_path"])
    if not traj_path.exists(): return None
    try:
        with gzip.open(traj_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        trajectory_list = data if isinstance(data, list) else data.get("trajectory")
        if not trajectory_list: return None
        df = pd.DataFrame(trajectory_list, columns=[LINEAR_ADV_COL, LINEAR_WIDTH_COL])
        if df.empty: return None
        df = df.dropna(subset=[LINEAR_ADV_COL, LINEAR_WIDTH_COL]).copy()
        if df.empty: return None
        df["bin_idx"] = np.floor(df[LINEAR_ADV_COL].to_numpy(float) / bin_size).astype(np.int64)
        df.sort_values([LINEAR_ADV_COL], kind="stable", inplace=True)
        df_first = df.drop_duplicates(subset=["bin_idx"], keep="first")
        if df_first.empty: return None
        return {
            "adv_bins": (df_first["bin_idx"].to_numpy() * bin_size).astype(float),
            "width_bins": df_first[LINEAR_WIDTH_COL].to_numpy(float)
        }
    except Exception:
        return None

def load_param_map_linear() -> pd.DataFrame:
    csv_path = ANALYSIS_DIR_LIN / f"{LINEAR_CAMPAIGN_ID}_summary_aggregated.csv"
    if not csv_path.exists():
        sys.exit(f"[ERROR] Linear Summary file not found: {csv_path}")
    df = pd.read_csv(csv_path, usecols=["task_id", "b_m", "initial_mutant_patch_size"])
    df = df.rename(columns={"task_id": "run_id"})
    return df.drop_duplicates("run_id")

def load_and_process_data_linear(bin_size: float, b_m_filter: float, initial_width_filter: int) -> Optional[ProcessedData]:
    df_map_all = load_param_map_linear()
    df_map_all.dropna(subset=['initial_mutant_patch_size'], inplace=True)
    df_map_all['initial_mutant_patch_size'] = df_map_all['initial_mutant_patch_size'].astype(int)
    df_map = df_map_all[
        np.isclose(df_map_all['b_m'], b_m_filter, atol=1e-5) &
        (df_map_all['initial_mutant_patch_size'] == initial_width_filter)
    ].copy()
    if df_map.empty:
        print(f"Warning: No LINEAR runs found for b_m≈{b_m_filter} AND W₀={initial_width_filter}.")
        return None
    tasks = [
        dict(row, traj_path=str(TRAJ_DIR_LIN / f"traj_{row['run_id']}.json.gz"))
        for _, row in df_map.iterrows()
    ]
    if not tasks: return None
    n_cpu = max(1, (os.cpu_count() or 2) - 1)
    print(f"Loading LINEAR b_m={b_m_filter}, W₀={initial_width_filter} using {n_cpu} processes...")
    with multiprocessing.Pool(processes=n_cpu) as pool:
        results = list(tqdm(pool.imap(partial(process_single_run_linear, bin_size=bin_size), tasks),
                              total=len(tasks), desc=f"Loading LINEAR W₀={initial_width_filter}", unit="file"))
    valid_results = [res for res in results if res and "adv_bins" in res and res["adv_bins"].size > 0]
    if not valid_results:
        print(f"Warning: No valid LINEAR trajectories could be loaded.")
        return None
    all_long_dfs = [pd.DataFrame({"run_id": i, "x_grid": res["adv_bins"], "width_bin": res["width_bins"]})
                    for i, res in enumerate(valid_results)]
    df_long = pd.concat(all_long_dfs, ignore_index=True)
    min_x = df_long['x_grid'].min()
    max_x = df_long['x_grid'].max()
    if pd.isna(min_x) or pd.isna(max_x): return None
    x_grid = np.arange(min_x, max_x + bin_size, bin_size)
    processed_trajectories = []
    for run_id, group in df_long.groupby('run_id'):
         s = pd.Series(group['width_bin'].values, index=group['x_grid'].values)
         s = s[~s.index.duplicated(keep='first')]
         s = s.reindex(x_grid)
         last_valid_idx = s.last_valid_index()
         if pd.notna(last_valid_idx):
             last_valid_loc = s.index.get_loc(last_valid_idx)
             s.iloc[last_valid_loc:] = s.iloc[last_valid_loc:].fillna(0)
         s = s.fillna(0)
         processed_trajectories.append(s.to_numpy())
    if not processed_trajectories: return None
    traj_array = np.array(processed_trajectories)
    stats_arr = np.where(traj_array > 1e-9, traj_array, np.nan)
    with np.errstate(invalid='ignore'):
        mean_surv = np.nanmean(stats_arr, axis=0)
        std_surv = np.nanstd(stats_arr, axis=0)
        mean_surv[np.sum(~np.isnan(stats_arr), axis=0) == 0] = np.nan
        std_surv[np.sum(~np.isnan(stats_arr), axis=0) < 2] = np.nan
    return ProcessedData(x_grid, traj_array, mean_surv, std_surv)

# -----------------------------------------------------------------
# --- RADIAL DATA FUNCTIONS ---
# -----------------------------------------------------------------

def infer_root_sid_numpy(types: np.ndarray, radii: np.ndarray, widths: np.ndarray, roots: np.ndarray) -> Optional[int]:
    mask_mut = np.isin(types, RADIAL_MUTANT_TYPES)
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

def process_single_run_radial(task: dict, bin_size: float) -> Optional[dict]:
    traj_path = Path(task["traj_path"])
    if not traj_path.exists(): return None
    try:
        with gzip.open(traj_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else data.get("sector_trajectory", [])
        if not rows: return None
        df = pd.DataFrame(rows)
        if not all(col in df.columns for col in ["type", "radius", "width_cells", "root_sid"]): return None
        df = df.astype({"type": np.int32, "radius": float, "width_cells": float, "root_sid": np.int32}, errors='ignore')
        types, r, w, roots = df["type"].to_numpy(np.int32), df["radius"].to_numpy(float), df["width_cells"].to_numpy(float), df["root_sid"].to_numpy(np.int32)
        if np.isnan(r).any() or np.isnan(w).any():
             valid_mask = ~np.isnan(r) & ~np.isnan(w)
             types, r, w, roots = types[valid_mask], r[valid_mask], w[valid_mask], roots[valid_mask]
             if r.size == 0: return None
        root = infer_root_sid_numpy(types, r, w, roots)
        if root is None: return None
        mask = (np.isin(types, RADIAL_MUTANT_TYPES)) & (roots == root)
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
        return {"adv_bins": adv_bins, "width_bins": width_bins}
    except Exception:
        return None

def load_param_map_radial() -> pd.DataFrame:
    csv_path = ANALYSIS_DIR_RAD / f"{RADIAL_CAMPAIGN_ID}_summary_aggregated.csv"
    if not csv_path.exists():
        sys.exit(f"[ERROR] Radial Summary file not found: {csv_path}")
    df = pd.read_csv(csv_path, usecols=["task_id", "b_res", "sector_width_initial"])
    df = df.rename(columns={"task_id": "run_id", "sector_width_initial": "initial_width", "b_res": "b_m"})
    return df.drop_duplicates("run_id")

def load_and_process_data_radial(bin_size: float, b_m_filter: float, initial_width_filter: int) -> Optional[ProcessedData]:
    df_map_all = load_param_map_radial()
    df_map_all.dropna(subset=['initial_width'], inplace=True)
    df_map_all['initial_width'] = df_map_all['initial_width'].astype(int)
    df_map = df_map_all[
        np.isclose(df_map_all['b_m'], b_m_filter, atol=0.01) &
        (df_map_all['initial_width'] == initial_width_filter)
    ].copy()
    if df_map.empty:
        print(f"Warning: No RADIAL runs found for b_m≈{b_m_filter} AND W₀={initial_width_filter}.")
        return None
    tasks = [
        dict(row, traj_path=str(TRAJ_DIR_RAD / f"traj_{row['run_id']}.json.gz"))
        for _, row in df_map.iterrows()
    ]
    if not tasks: return None
    n_cpu = max(1, (os.cpu_count() or 2) - 1)
    print(f"Loading RADIAL b_m={b_m_filter}, W₀={initial_width_filter} using {n_cpu} processes...")
    with multiprocessing.Pool(processes=n_cpu) as pool:
        results = list(tqdm(pool.imap(partial(process_single_run_radial, bin_size=bin_size), tasks),
                              total=len(tasks), desc=f"Loading RADIAL W₀={initial_width_filter}", unit="file"))
    valid_results = [res for res in results if res and "adv_bins" in res and res["adv_bins"].size > 0]
    if not valid_results:
        print(f"Warning: No valid RADIAL trajectories could be loaded.")
        return None
    all_long_dfs = [pd.DataFrame({"run_id": i, "x_grid": res["adv_bins"], "width_bin": res["width_bins"]})
                    for i, res in enumerate(valid_results)]
    df_long = pd.concat(all_long_dfs, ignore_index=True)
    min_x = df_long['x_grid'].min()
    max_x = df_long['x_grid'].max()
    if pd.isna(min_x) or pd.isna(max_x): return None
    x_grid = np.arange(min_x, max_x + bin_size, bin_size)
    processed_trajectories = []
    for run_id, group in df_long.groupby('run_id'):
         s = pd.Series(group['width_bin'].values, index=group['x_grid'].values)
         s = s[~s.index.duplicated(keep='first')]
         s = s.reindex(x_grid)
         last_valid_idx = s.last_valid_index()
         if pd.notna(last_valid_idx):
             last_valid_loc = s.index.get_loc(last_valid_idx)
             s.iloc[last_valid_loc:] = s.iloc[last_valid_loc:].fillna(0)
         s = s.fillna(0)
         processed_trajectories.append(s.to_numpy())
    if not processed_trajectories: return None
    traj_array = np.array(processed_trajectories)
    stats_arr = np.where(traj_array > 1e-9, traj_array, np.nan)
    with np.errstate(invalid='ignore'):
        mean_surv = np.nanmean(stats_arr, axis=0)
        std_surv = np.nanstd(stats_arr, axis=0)
        mean_surv[np.sum(~np.isnan(stats_arr), axis=0) == 0] = np.nan
        std_surv[np.sum(~np.isnan(stats_arr), axis=0) < 2] = np.nan
    return ProcessedData(x_grid, traj_array, mean_surv, std_surv)

# -----------------------------------------------------------------
# --- INFERENCE FUNCTIONS ---
# -----------------------------------------------------------------

def infer_parameters_linear(empirical_data: ProcessedData) -> Tuple[float, float, float]:
    adv_grid = empirical_data.x_grid
    traj_arr = empirical_data.traj_array
    stats_arr = np.where(traj_arr > 1e-9, traj_arr, np.nan)
    with np.errstate(invalid='ignore'):
        df_agg = pd.DataFrame({
            "adv_bin": adv_grid,
            "width_mean": empirical_data.mean_surv,
            "width_var": empirical_data.std_surv**2,
            "n_runs": np.sum(~np.isnan(stats_arr), axis=0),
            "ms_mean": np.nanmean(np.square(stats_arr), axis=0),
            "m4_mean": np.nanmean(np.power(stats_arr, 4), axis=0),
        })
    df_agg = df_agg[df_agg["n_runs"] >= LINEAR_MIN_RUNS].sort_values("adv_bin").dropna()
    if df_agg.empty: return np.nan, np.nan, np.nan
    df_agg["width_se"] = np.sqrt(np.maximum(df_agg["width_var"], 0.0) / df_agg["n_runs"])
    df_agg["var_x"] = np.maximum(df_agg["ms_mean"] - np.square(df_agg["width_mean"]), 0.0)
    ms_var = np.maximum(df_agg["m4_mean"] - np.square(df_agg["ms_mean"]), 0.0) / df_agg["n_runs"]
    var_mean_var = ms_var + 4.0 * np.square(df_agg["width_mean"]) * (np.maximum(df_agg["width_var"], 0.0) / df_agg["n_runs"])
    df_agg["var_se"] = np.sqrt(np.maximum(var_mean_var, 0.0))
    total_runs = empirical_data.traj_array.shape[0]
    survival_df = df_agg[df_agg["n_runs"] == total_runs]
    dynamic_adv_max = survival_df["adv_bin"].max() if not survival_df.empty else (LINEAR_FIT_MIN - 1.0)
    effective_adv_max = min(LINEAR_FIT_MAX_LIMIT, dynamic_adv_max)
    fit_mask = (df_agg["adv_bin"] >= LINEAR_FIT_MIN) & (df_agg["adv_bin"] <= effective_adv_max)
    df_fit = df_agg.loc[fit_mask].dropna()
    if len(df_fit) < LINEAR_MIN_POINTS: return np.nan, np.nan, effective_adv_max
    drift_lr = linregress(df_fit["adv_bin"].to_numpy(), df_fit["width_mean"].to_numpy())
    m_perp_est = drift_lr.slope / 2.0
    x, y, sigma = df_fit["adv_bin"].to_numpy(float), df_fit["var_x"].to_numpy(float), df_fit["var_se"].to_numpy(float)
    mask_ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 1e-12)
    x_w, y_w, sigma_w = x[mask_ok], y[mask_ok], sigma[mask_ok]
    if x_w.size < LINEAR_MIN_POINTS: D_X_est = np.nan
    else:
        slope_var, intercept_var = _np_wls_linear(x_w, y_w, 1.0 / sigma_w)
        D_X_est = slope_var / 4.0
    if D_X_est < 0: D_X_est = 0.01
    print(f"Inferred LINEAR params: m_perp = {m_perp_est:.4f}, D_X = {D_X_est:.4f}")
    return m_perp_est, D_X_est, effective_adv_max

def infer_parameters_radial(empirical_data: ProcessedData) -> Tuple[float, float, float]:
    df_stats = pd.DataFrame({
        'radius_bin': empirical_data.x_grid, 
        'mean_width': empirical_data.mean_surv, 
        'var_width': empirical_data.std_surv**2
    })
    df_stats = df_stats.dropna(subset=['mean_width', 'radius_bin', 'var_width'])
    df_stats = df_stats[df_stats['mean_width'] > 1e-9].copy()
    df_stats = df_stats[df_stats['radius_bin'] > 1e-9].copy()
    df_stats['angle'] = df_stats['mean_width'] / df_stats['radius_bin']
    df_stats['angle_var'] = df_stats['var_width'] / df_stats['radius_bin']**2
    fit_df = df_stats[
        (df_stats['radius_bin'] >= RADIAL_FIT_MIN) & 
        (df_stats['radius_bin'] <= RADIAL_FIT_MAX)
    ].copy()
    if len(fit_df) < RADIAL_MIN_POINTS: return np.nan, np.nan, np.nan
    fit_df['log_r'] = np.log(fit_df['radius_bin'])
    valid_angle_fit = fit_df.dropna(subset=['log_r', 'angle'])
    if len(valid_angle_fit) < RADIAL_MIN_POINTS: return np.nan, np.nan, np.nan
    slope_m, intercept_m, r_value_m, p_value_m, std_err_m = linregress(valid_angle_fit['log_r'], valid_angle_fit['angle'])
    m_perp = slope_m / 2.0
    r0_fit = fit_df['radius_bin'].iloc[0]
    fit_df['inv_r_term'] = (1.0 / r0_fit) - (1.0 / fit_df['radius_bin'])
    valid_var_fit = fit_df.dropna(subset=['inv_r_term', 'angle_var'])
    if len(valid_var_fit) < RADIAL_MIN_POINTS: return np.nan, np.nan, np.nan
    slope_D, intercept_D, r_value_D, p_value_D, std_err_D = linregress(valid_var_fit['inv_r_term'], valid_var_fit['angle_var'])
    D_X = slope_D / 4.0
    if D_X < 0: D_X = 0.01
    print(f"Inferred RADIAL params: m_perp = {m_perp:.4f}, D_X = {D_X:.4f}")
    return m_perp, D_X, (RADIAL_FIT_MIN, RADIAL_FIT_MAX)

# -----------------------------------------------------------------
# --- SIMULATION FUNCTIONS ---
# -----------------------------------------------------------------

def simulate_linear_rw_conditioned(n_paths, a_grid, w0, m_perp, D_X, seed):
    rng = np.random.default_rng(seed)
    a = np.asarray(a_grid, dtype=float)
    nT = a.size
    W = np.zeros((n_paths, nT))
    alive = np.ones((n_paths, nT), dtype=bool)
    W[:, 0] = w0 
    for t in range(1, nT):
        da = a[t] - a[t-1]
        if da <= 0: continue
        currently_alive = alive[:, t-1]
        if not np.any(currently_alive):
            alive[:, t:] = False
            break
        n_alive = np.sum(currently_alive)
        dW_noise = rng.normal(loc=0.0, scale=np.sqrt(max(0, da)), size=n_alive)
        drift_term = (2.0 * m_perp) * da
        diffusion_term = (np.sqrt(max(0, 4.0 * D_X))) * dW_noise
        W_prev_alive = W[currently_alive, t-1]
        W_new = W_prev_alive + drift_term + diffusion_term
        survived_mask = W_new > 1e-12 
        W[currently_alive, t] = np.where(survived_mask, W_new, 0.0)
        temp_alive = np.zeros(n_paths, dtype=bool)
        temp_alive[currently_alive] = survived_mask 
        alive[:, t] = temp_alive
    return np.where(alive, W, np.nan)

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

# -----------------------------------------------------------------
# --- PLOTTING FUNCTION (GENERALIZED) ---
# -----------------------------------------------------------------

# *** SYNTAX FIX: Moved arguments with default values to the end ***
def plot_panel(ax: plt.Axes, 
               empirical_data: ProcessedData, 
               params: Dict[str, float], 
               title: str,
               x_label: str,
               w_label: str,
               is_prediction_panel: bool, 
               add_legend: bool, 
               add_right_ylabel: bool, 
               show_param_box: bool,
               sim_function: callable,
               uncond_mean_function: callable,
               fit_window: Optional[Tuple[float, float]] = None):
    
    m_perp, D_X = params['m_perp'], params['D_X']
    x_grid = empirical_data.x_grid
    
    # Find start point for simulation
    fit_min = fit_window[0] if fit_window else x_grid[0]
    start_mask = x_grid >= fit_min
    first_valid_idx_overall = np.where(~np.isnan(empirical_data.mean_surv) & (empirical_data.mean_surv > 1e-9))[0]

    if len(first_valid_idx_overall) == 0:
        print(f"Warning: No valid lattice sim mean width data found for {title}. Plotting lattice data only.")
        plot_lattice_only = True
        x_grid_sim, mean_w_sim_cond, mean_w_sim_uncond = np.array([]), np.array([]), np.array([])
        W_sim, p10_sim, p90_sim = np.array([[]]), np.array([]), np.array([])
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
        else:
            start_idx = default_start_idx
        
        x_start, w_start = x_grid[start_idx], empirical_data.mean_surv[start_idx]
        max_x_empirical = x_grid[~np.isnan(x_grid)].max()
        
        if np.isnan(max_x_empirical) or x_start >= max_x_empirical:
            plot_lattice_only = True
            x_grid_sim, mean_w_sim_cond, mean_w_sim_uncond = np.array([]), np.array([]), np.array([])
            W_sim, p10_sim, p90_sim = np.array([[]]), np.array([]), np.array([])
        else:
            x_grid_sim = np.linspace(x_start, max_x_empirical, 400)
            W_sim = sim_function(SIM_PATHS, x_grid_sim, w_start, m_perp, D_X, seed=SIM_SEED)
            with np.errstate(invalid="ignore"):
                mean_w_sim_cond = np.nanmean(W_sim, axis=0)
                p10_sim = np.nanpercentile(W_sim, 10, axis=0)
                p90_sim = np.nanpercentile(W_sim, 90, axis=0)
                p10_sim = np.nan_to_num(p10_sim, nan=0.0)
            mean_w_sim_uncond = uncond_mean_function(x_grid_sim, x_start, w_start, m_perp)

    # === Plotting ===
    with np.errstate(invalid="ignore"):
        p10_emp = np.nanpercentile(np.where(empirical_data.traj_array > 1e-9, empirical_data.traj_array, np.nan), 10, axis=0)
        p90_emp = np.nanpercentile(np.where(empirical_data.traj_array > 1e-9, empirical_data.traj_array, np.nan), 90, axis=0)
        p10_emp = np.nan_to_num(p10_emp, nan=0.0)
    ax.fill_between(x_grid, p10_emp, p90_emp,
                    color=COLOR_EMP_PERC, alpha=0.4,
                    label='10-90th pct. (Lattice Sim.)', zorder=2)
    ax.plot(x_grid, empirical_data.mean_surv,
            color=COLOR_EMP_MEAN, lw=2.5,
            label='Mean (Lattice Sim.)', zorder=5)
    if not plot_lattice_only:
        if is_prediction_panel:
            ax.fill_between(x_grid_sim, p10_sim, p90_sim,
                            color=COLOR_SIM_PERC, alpha=0.4,
                            label='10-90th pct. (Cond. RW Model)', zorder=1)
        sim_lw = 2.8 if is_prediction_panel else 2.5
        ax.plot(x_grid_sim, mean_w_sim_cond,
                color=COLOR_SIM_MEAN_COND, lw=sim_lw, ls='-',
                label='Mean (Cond. RW Model)', zorder=4)
        if is_prediction_panel:
            ax.plot(x_grid_sim, mean_w_sim_uncond,
                    color=COLOR_SIM_MEAN_UNCOND, lw=2.0, ls='--',
                    label='Mean (Uncond. RW Model)', zorder=3)
    ax2 = ax.twinx()
    if empirical_data.traj_array.shape[0] > 0:
        valid_counts_emp = np.sum(~np.isnan(np.where(empirical_data.traj_array > 1e-9, empirical_data.traj_array, np.nan)), axis=0)
        first_data_idx_emp = np.where(valid_counts_emp > 0)[0]
        if first_data_idx_emp.size > 0:
            norm_factor_emp = valid_counts_emp[first_data_idx_emp[0]]
            survival_fraction_emp = valid_counts_emp / norm_factor_emp
            ax2.plot(x_grid[first_data_idx_emp[0]:],
                     survival_fraction_emp[first_data_idx_emp[0]:],
                     color=COLOR_SURVIVAL, lw=2.0, ls=':',
                     label='Survival (Lattice Sim.)', zorder=3)
    if not plot_lattice_only:
        n_surv_sim = np.sum(~np.isnan(W_sim), axis=0)
        if n_surv_sim.size > 0 and n_surv_sim[0] > 0:
            survival_fraction_sim = n_surv_sim / n_surv_sim[0]
        else:
            survival_fraction_sim = np.zeros_like(n_surv_sim)
        ax2.plot(x_grid_sim, survival_fraction_sim,
                 color=COLOR_SURVIVAL, lw=2.0, ls='-.',
                 label='Survival (Cond. RW Model)', zorder=3) 
    ax2.tick_params(axis='y', colors=COLOR_SURVIVAL, labelsize=11)
    ax2.set_ylim(-0.02, 1.05)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    if add_right_ylabel:
        ax2.set_ylabel("Survival Fraction ($S$)", fontsize=13, color=COLOR_SURVIVAL, rotation=-90, labelpad=20, fontweight='bold')
    else:
        ax2.set_yticks([])
        ax2.set_yticklabels([])
    ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
    ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
    ax.set_ylabel(w_label, fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5, color='gray')
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.tick_params(which='minor', length=2, color='gray')
    if fit_window is not None:
        ax.axvspan(fit_window[0], fit_window[1], 
                   color="gray", alpha=0.15, zorder=0, 
                   label="Fit window (100% survival)")
    if show_param_box:
        param_box_pos = (0.97, 0.97)
        param_text = f"$m_\\perp = {m_perp:.3f}$\n$D_X = {D_X:.2f}$"
        ax.text(param_box_pos[0], param_box_pos[1], param_text,
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="darkgray", alpha=0.8),
                fontsize=10)
    min_x_data = x_grid[~np.isnan(x_grid)].min()
    max_x_data = x_grid[~np.isnan(x_grid)].max()
    if pd.isna(min_x_data): min_x_data = 0
    if pd.isna(max_x_data): max_x_data = 1000
    ax.set_xlim(left=max(0, min_x_data - 0.05 * (max_x_data - min_x_data)),
                right=max_x_data + 0.05 * (max_x_data - min_x_data))
    if p90_emp.size > 0: max_y_emp = np.nanmax(p90_emp)
    else: max_y_emp = 0
    if not plot_lattice_only and p90_sim.size > 0: max_y_sim = np.nanmax(p90_sim)
    else: max_y_sim = 0
    max_y = max(max_y_emp, max_y_sim, 1.0)
    ax.set_ylim(bottom=-0.05 * max_y, top=max_y * 1.1)
    
    # --- Legend ---
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    available_handles = {**dict(zip(labels1, handles1)), **dict(zip(labels2, labels2))}
    desired_order = [
        'Mean (Lattice Sim.)', '10-90th pct. (Lattice Sim.)',
        'Mean (Cond. RW Model)', '10-90th pct. (Cond. RW Model)',
        'Mean (Uncond. RW Model)',
        'Survival (Lattice Sim.)', 'Survival (Cond. RW Model)',
        'Fit window (100% survival)'
    ]
    final_handles = [available_handles[lbl] for lbl in desired_order if lbl in available_handles]
    final_labels = [lbl for lbl in desired_order if lbl in available_handles]
    if add_legend:
        if 'Fit window (100% survival)' in final_labels:
            idx = final_labels.index('Fit window (100% survival)')
            final_handles.pop(idx)
            final_labels.pop(idx)
        ax.legend(handles=final_handles, labels=final_labels,
                  loc='upper left', fontsize=9.5, frameon=True,
                  facecolor='white', framealpha=0.85, ncol=1)
    elif fit_window is not None:
        if 'Fit window (100% survival)' in available_handles:
             ax.legend(handles=[available_handles['Fit window (100% survival)']], 
                       labels=['Fit window (100% survival)'],
                       loc='upper left', fontsize=9.5, frameon=True,
                       facecolor='white', framealpha=0.85)

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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharey=False)
    axA, axB = axes[0, 0], axes[0, 1]
    axC, axD = axes[1, 0], axes[1, 1]
    
    fig.suptitle(f"Conditioning on Survival Explains Apparent Stagnancy of Deleterious Patches ($b_m \\approx {TARGET_BM}$)",
                 fontsize=17, fontweight='bold')

    # ------------------
    # --- TOP ROW: LINEAR ---
    # ------------------
    print("--- Processing LINEAR (Top Row) ---")
    data_A = load_and_process_data_linear(LINEAR_BIN_SIZE, TARGET_BM, INFERENCE_W0)
    if data_A is None:
        axA.set_title(f"(A) Linear Inference ($W_0 = {INFERENCE_W0}$)\n--- DATA FAILED TO LOAD ---", color='red', fontsize=12)
        axB.set_title(f"(B) Linear Prediction ($W_0 = {PREDICTION_W0}$)\n--- DEPENDENT ON PANEL A ---", color='red', fontsize=12)
    else:
        m_perp_lin, D_X_lin, adv_fit_max_A = infer_parameters_linear(data_A)
        if pd.isna(m_perp_lin) or pd.isna(D_X_lin):
            axA.set_title(f"(A) Linear Inference ($W_0 = {INFERENCE_W0}$)\n--- INFERENCE FAILED ---", color='red', fontsize=12)
            axB.set_title(f"(B) Linear Prediction ($W_0 = {PREDICTION_W0}$)\n--- DEPENDENT ON PANEL A ---", color='red', fontsize=12)
        else:
            params_lin = {'m_perp': m_perp_lin, 'D_X': D_X_lin}
            plot_panel(axA, data_A, params_lin,
                       title=f"(A) Linear Inference ($W_0 = {INFERENCE_W0}$)",
                       x_label="Front Advancement ($a$, cell widths)",
                       w_label="Patch Width ($W$, cell widths)",
                       is_prediction_panel=False, add_legend=False, add_right_ylabel=False,
                       show_param_box=True, 
                       fit_window=(LINEAR_FIT_MIN, adv_fit_max_A),
                       sim_function=simulate_linear_rw_conditioned,
                       uncond_mean_function=lambda x, x0, w0, m: w0 + 2.0 * m * (x - x0))
            
            data_B = load_and_process_data_linear(LINEAR_BIN_SIZE, TARGET_BM, PREDICTION_W0)
            if data_B is None:
                axB.set_title(f"(B) Linear Prediction ($W_0 = {PREDICTION_W0}$)\n--- DATA FAILED TO LOAD ---", color='red', fontsize=12)
            else:
                plot_panel(axB, data_B, params_lin,
                           title=f"(B) Linear Prediction ($W_0 = {PREDICTION_W0}$)",
                           x_label="Front Advancement ($a$, cell widths)",
                           w_label="Patch Width ($W$, cell widths)",
                           is_prediction_panel=True, add_legend=True, add_right_ylabel=True,
                           show_param_box=False,
                           sim_function=simulate_linear_rw_conditioned,
                           uncond_mean_function=lambda x, x0, w0, m: w0 + 2.0 * m * (x - x0),
                           fit_window=None)

    # ------------------
    # --- BOTTOM ROW: RADIAL ---
    # ------------------
    print("\n--- Processing RADIAL (Bottom Row) ---")
    data_C = load_and_process_data_radial(RADIAL_BIN_SIZE, TARGET_BM, INFERENCE_W0)
    if data_C is None:
        axC.set_title(f"(C) Radial Inference ($W_0 = {INFERENCE_W0}$)\n--- DATA FAILED TO LOAD ---", color='red', fontsize=12)
        axD.set_title(f"(D) Radial Prediction ($W_0 = {PREDICTION_W0}$)\n--- DEPENDENT ON PANEL C ---", color='red', fontsize=12)
    else:
        m_perp_rad, D_X_rad, fit_window_C = infer_parameters_radial(data_C)
        if pd.isna(m_perp_rad) or pd.isna(D_X_rad):
            axC.set_title(f"(C) Radial Inference ($W_0 = {INFERENCE_W0}$)\n--- INFERENCE FAILED ---", color='red', fontsize=12)
            axD.set_title(f"(D) Radial Prediction ($W_0 = {PREDICTION_W0}$)\n--- DEPENDENT ON PANEL C ---", color='red', fontsize=12)
        else:
            params_rad = {'m_perp': m_perp_rad, 'D_X': D_X_rad}
            plot_panel(axC, data_C, params_rad,
                       title=f"(C) Radial Inference ($W_0 = {INFERENCE_W0}$)",
                       x_label="Colony Radius ($r$, cell widths)",
                       w_label="Sector Width ($W$, cell widths)",
                       is_prediction_panel=False, add_legend=False, add_right_ylabel=False,
                       show_param_box=True, 
                       fit_window=fit_window_C,
                       sim_function=simulate_radial_rw_conditioned,
                       uncond_mean_function=lambda r, r0, w0, m: (w0 * r / r0) + (2.0 * m * r * np.log(r / r0)))
            
            data_D = load_and_process_data_radial(RADIAL_BIN_SIZE, TARGET_BM, PREDICTION_W0)
            if data_D is None:
                axD.set_title(f"(D) Radial Prediction ($W_0 = {PREDICTION_W0}$)\n--- DATA FAILED TO LOAD ---", color='red', fontsize=12)
            else:
                plot_panel(axD, data_D, params_rad,
                           title=f"(D) Radial Prediction ($W_0 = {PREDICTION_W0}$)",
                           x_label="Colony Radius ($r$, cell widths)",
                           w_label="Sector Width ($W$, cell widths)",
                           is_prediction_panel=True, add_legend=True, add_right_ylabel=True,
                           show_param_box=False,
                           sim_function=simulate_radial_rw_conditioned,
                           uncond_mean_function=lambda r, r0, w0, m: (w0 * r / r0) + (2.0 * m * r * np.log(r / r0)),
                           fit_window=None)

    # --- Final Figure Adjustments ---
    sns.despine(fig=fig)
    fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.93], pad=1.5, h_pad=2.5, w_pad=2.5)
    out_path_png = FIG_DIR / f"{OUTPUT_FILENAME}.png"
    out_path_pdf = FIG_DIR / f"{OUTPUT_FILENAME}.pdf"
    print(f"\nSaving PNG figure to: {out_path_png}")
    fig.savefig(out_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saving PDF figure to: {out_path_pdf}")
    fig.savefig(out_path_pdf, bbox_inches='tight', facecolor='white')
    print("✅ Figures saved successfully.")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()