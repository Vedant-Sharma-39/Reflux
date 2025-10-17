#!/usr/bin/env python3
# FILE: scripts/paper_figures/fig_aif_conditioned_means.py
#
# A high-performance, parallelized script to analyze online trajectories.
# It uses multiprocessing to dramatically speed up the analysis of many files.

import sys, json, gzip, os
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import multiprocessing
from functools import partial

# --- Project Setup & Constants ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from src.config import EXPERIMENTS

RESISTANT, COMPENSATED = 2, 3
MUTANTS = (RESISTANT, COMPENSATED)
DEFAULT_BIN_SIZE = 10.0

# =============================================================================
# --- CORE ANALYSIS LOGIC (Unchanged) ---
# =============================================================================

def bin_series_first_np(radius: np.ndarray, width: np.ndarray, bin_size: float) -> Tuple[np.ndarray, np.ndarray]:
    if radius.size == 0: return np.array([]), np.array([])
    order = np.argsort(radius, kind="stable")
    r_sorted, w_sorted = radius[order], width[order]
    bin_idx = np.floor(r_sorted / bin_size).astype(np.int64)
    uniq_bins, first_pos = np.unique(bin_idx, return_index=True)
    return uniq_bins.astype(float) * bin_size, w_sorted[first_pos]

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

# =============================================================================
# --- PARALLEL WORKER FUNCTION ---
# =============================================================================

def process_single_task(task_info: dict, bin_size: float) -> Optional[pd.DataFrame]:
    """
    Worker function to process a single trajectory file.
    Loads one file, processes it, and returns a binned DataFrame with parameters.
    """
    traj_path = Path(task_info["traj_path"])
    if not traj_path.exists(): return None

    try:
        with gzip.open(traj_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else data.get("sector_trajectory", [])
        if not rows: return None
        
        df_run = pd.DataFrame(rows)
        
        # Process this single run to get one binned trajectory
        types = df_run["type"].to_numpy(dtype=np.int32)
        radii = df_run["radius"].to_numpy(dtype=float)
        widths = df_run["width_cells"].to_numpy(dtype=float)
        roots = df_run["root_sid"].to_numpy(dtype=np.int32)

        root_to_track = infer_root_sid_numpy(types, radii, widths, roots)
        if root_to_track is None: return None

        mask = (np.isin(types, MUTANTS)) & (roots == root_to_track)
        if not mask.any(): return None

        bins, vals = bin_series_first_np(radii[mask], widths[mask], bin_size)
        
        binned_df = pd.DataFrame({"radius_bin": bins, "width": vals})
        binned_df['b_res'] = task_info['b_res']
        binned_df['initial_width'] = task_info['sector_width_initial']
        
        return binned_df
    except Exception:
        # Silently fail for a single corrupted file
        return None

# =============================================================================
# --- DATA I/O & PLOTTING (Unchanged, but consolidated) ---
# =============================================================================

def load_parameter_map(analysis_dir: Path, campaign: str) -> pd.DataFrame:
    summary_csv = analysis_dir / f"{campaign}_summary_aggregated.csv"
    if not summary_csv.exists():
        sys.exit(f"[ERROR] Summary file not found: {summary_csv}\nRun: make consolidate CAMPAIGN={campaign}")
    param_cols = ["task_id", "b_res", "sector_width_initial", "replicate"]
    df = pd.read_csv(summary_csv, low_memory=False, usecols=lambda c: c in param_cols)
    return df.rename(columns={"task_id": "run_id"}).drop_duplicates("run_id")

def plot_panel(stats: pd.DataFrame, out_path: Path, b_res: float, w0: int, smooth_window: int = 1):
    if stats.empty: return
    x, n = stats["radius_bin"], stats["n"]
    mu, se, med, q25, q75 = stats["mean"], stats["sem"], stats["median"], stats["q25"], stats["q75"]
    frac_surv = n / n.max()

    if smooth_window > 1:
        mu, med, q25, q75 = [s.rolling(smooth_window, center=True, min_periods=1).mean() for s in [mu, med, q25, q75]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(x, mu - se, mu + se, color="royalblue", alpha=0.2, label="Mean ± SEM (cond.)")
    ax.plot(x, mu, color="royalblue", lw=2.5, label="Mean (cond.)")
    ax.fill_between(x, q25, q75, color="crimson", alpha=0.2, label="Median IQR (cond.)")
    ax.plot(x, med, color="crimson", lw=2.5, ls='--', label="Median (cond.)")

    ax.set_xlabel("Radius (cells)"); ax.set_ylabel("Width (cells)"); ax.set_ylim(bottom=0)
    ax.set_title(f"Conditioned Sector Width vs. Radius\n($b_{{res}}$={b_res:.4f}, Initial Width={w0})")
    
    ax2 = ax.twinx()
    ax2.plot(x, n, color="gray", lw=1.5, ls=':', alpha=0.8, label="N (survivors)")
    ax2.step(x, frac_surv, where="post", color="green", lw=2, ls='--', label="Survival Fraction")
    ax2.set_ylabel("Count / Survival Fraction"); ax2.set_ylim(bottom=0)

    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
    
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    print(f"  -> Saved plot: {out_path.name}")

# =============================================================================
# --- MAIN WORKFLOW ---
# =============================================================================

def main():
    campaign_id = "aif_online_scan_v1"
    print(f"--- Analyzing Conditioned Means for Campaign: {campaign_id} (Parallel) ---")

    data_dir = PROJECT_ROOT / "data" / campaign_id
    analysis_dir = data_dir / "analysis"
    traj_dir = data_dir / "trajectories"
    fig_dir = PROJECT_ROOT / "figures" / "aif_conditioned_means"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create a list of tasks for the workers
    df_params = load_parameter_map(analysis_dir, campaign_id)
    tasks = []
    for _, row in df_params.iterrows():
        task_info = row.to_dict()
        task_info["traj_path"] = str(traj_dir / f"traj_{row['run_id']}.json.gz")
        tasks.append(task_info)
    
    if not tasks: sys.exit("No tasks found to process.")

    # 2. Run processing in parallel
    num_workers = max(1, os.cpu_count() - 2)
    print(f"\nProcessing {len(tasks)} runs in parallel using {num_workers} workers...")
    
    # Use functools.partial to pre-fill the 'bin_size' argument for the worker
    worker_func = partial(process_single_task, bin_size=DEFAULT_BIN_SIZE)
    
    binned_series_list = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        pbar = tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc="Processing runs")
        for result_df in pbar:
            if result_df is not None and not result_df.empty:
                binned_series_list.append(result_df)
    
    if not binned_series_list: sys.exit("No valid lineages found after processing.")
    
    df_all_binned = pd.concat(binned_series_list)

    # 3. Group by condition and generate plots (this part is fast)
    print("\nAggregating results and generating plots...")
    for (b_res, w0), group_df in tqdm(df_all_binned.groupby(['b_res', 'initial_width']), desc="Plotting conditions"):
        
        stats_df = group_df.groupby('radius_bin')['width'].agg(
            n='size', mean='mean', sem='sem', median='median',
            q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
        ).reset_index()

        out_path = fig_dir / f"conditioned_mean_b{b_res:.4f}_w{w0}.png"
        plot_panel(stats_df, out_path, b_res, w0, smooth_window=3)

    print("\n✅ Parallel analysis complete.")

if __name__ == "__main__":
    main()