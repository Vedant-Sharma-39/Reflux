#!/usr/bin/env python3
# FILE: scripts/paper_figures/fig_aif_radial_theory_vs_data_final_angle_sim.py
#
# Compare real radial trajectories (width) to theory-driven simulations.
#
# --- FINAL WORKFLOW (v6.1 - NameError Bugfix) ---
# 1. Loads radially-calibrated drift (m_perp) and linearly-calibrated diffusion (D_X).
# 2. Loads all experimental RADIAL trajectory data.
# 3. Iterates through EVERY initial width (W0) and b_res condition.
# 4. For each condition, filters data to start the comparison at a stable radius.
# 5. Simulates radial growth using the single best-fit experimental parameter set.
# 6. Overlays the empirical data with the single theory simulation, using a final,
#    publication-quality plotting style.

import sys, json, gzip, os
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from functools import partial
import multiprocessing
from tqdm import tqdm

# ---------- Project Setup ----------
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import EXPERIMENTS
except (NameError, ImportError):
    PROJECT_ROOT = Path(".").resolve()

RADIAL_CAMPAIGN_ID = "aif_online_scan_v1"
LINEAR_PARAMS_CAMPAIGN_ID = "boundary_experiment_v1"

# Input Paths
RADIAL_DATA_DIR = PROJECT_ROOT / "data" / RADIAL_CAMPAIGN_ID
RADIAL_ANALYSIS_DIR = RADIAL_DATA_DIR / "analysis"
RADIAL_TRAJ_DIR = RADIAL_DATA_DIR / "trajectories"
LINEAR_PARAMS_CSV = PROJECT_ROOT / "data" / LINEAR_PARAMS_CAMPAIGN_ID / "analysis" / "boundary_diffusion_params_linear_final.csv"
RADIAL_PARAMS_DIR = PROJECT_ROOT / "figures" / "drift_comparison_linear_vs_radial"

# Output Path
FIG_DIR = PROJECT_ROOT / "figures" / "aif_radial_theory_vs_data_v6_radial_theory_only"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Model constants ----------
RESISTANT, COMPENSATED = 2, 3
MUTANTS = (RESISTANT, COMPENSATED)
BIN_SIZE = 10.0
N_SIMS = 4000
SEED = 12345
SIMULATION_START_RADIUS = 500.0

# ---------- Helpers & Data Loading ----------
def bin_series_first_np(radius: np.ndarray, width: np.ndarray, bin_size: float):
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

def process_single_run(task: dict, bin_size: float) -> Optional[pd.DataFrame]:
    traj_path = Path(task["traj_path"])
    if not traj_path.exists(): return None
    with gzip.open(traj_path, "rt", encoding="utf-8") as f: data = json.load(f)
    rows = data if isinstance(data, list) else data.get("sector_trajectory", [])
    if not rows: return None
    df = pd.DataFrame(rows)
    types, r, w, roots = df["type"].to_numpy(np.int32), df["radius"].to_numpy(float), df["width_cells"].to_numpy(float), df["root_sid"].to_numpy(np.int32)
    root = infer_root_sid_numpy(types, r, w, roots)
    if root is None: return None
    mask = (np.isin(types, MUTANTS)) & (roots == root)
    if not mask.any(): return None
    r_bins, w_bins = bin_series_first_np(r[mask], w[mask], bin_size)
    keep = r_bins > 0
    if not np.any(keep): return None
    out = pd.DataFrame({"radius_bin": r_bins[keep], "width": w_bins[keep]})
    out["b_res"] = task["b_res"]
    out["initial_width"] = int(task["initial_width"])
    return out

# --- BUGFIX IS HERE: Restored the missing function ---
def load_radial_param_map() -> pd.DataFrame:
    """Loads the summary metadata for the radial campaign to map runs to parameters."""
    csv_path = RADIAL_ANALYSIS_DIR / f"{RADIAL_CAMPAIGN_ID}_summary_aggregated.csv"
    if not csv_path.exists():
        sys.exit(f"[ERROR] Radial summary file not found: {csv_path}")
    cols = ["task_id", "b_res", "sector_width_initial"]
    df = pd.read_csv(csv_path, usecols=cols)
    df = df.rename(columns={"task_id": "run_id", "sector_width_initial": "initial_width"})
    return df.drop_duplicates("run_id")
# --- END BUGFIX ---

def load_all_parameters(initial_width_for_radial_calib: int) -> pd.DataFrame:
    if not LINEAR_PARAMS_CSV.exists():
        sys.exit(f"[ERROR] Linear calibrated parameter file not found: {LINEAR_PARAMS_CSV}")
    df_linear = pd.read_csv(LINEAR_PARAMS_CSV).rename(columns={"b_m": "b_res"})

    radial_params_csv = RADIAL_PARAMS_DIR / f"drift_comparison_summary_w{initial_width_for_radial_calib}.csv"
    if not radial_params_csv.exists():
        sys.exit(f"[ERROR] Radial drift parameter file not found for W0={initial_width_for_radial_calib}: {radial_params_csv}")
    df_radial = pd.read_csv(radial_params_csv)

    df_linear = df_linear[["b_res", "D_X"]]
    df_radial = df_radial[["b_res", "m_perp_radial_log"]]

    df_merged = pd.merge(df_linear, df_radial, on="b_res", how="inner")
    if df_merged.empty:
        sys.exit("[ERROR] No common b_res values found between linear and radial parameter files.")
    return df_merged

# ---------- Theory simulation ----------
def simulate_angles_radial_rigorous(m_perp: float, D_X: float, r_grid: np.ndarray, phi0: float, n_paths: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    r, nT = np.asarray(r_grid, float), r_grid.size
    phi = np.zeros((n_paths, nT), dtype=float)
    alive = np.zeros_like(phi, dtype=bool)
    phi[:, 0], alive[:, 0] = phi0, phi0 > 0
    for t in range(1, nT):
        dr, rt = r[t] - r[t - 1], r[t - 1]
        if rt <= 0: continue
        dW_noise = rng.normal(loc=0.0, scale=np.sqrt(dr), size=n_paths)
        a = alive[:, t - 1]
        if not np.any(a):
            phi[:, t:], alive[:, t:] = 0.0, False
            break
        drift = (2.0 * m_perp / rt) * dr
        diffusion = (np.sqrt(4.0 * D_X) / rt) * dW_noise[a]
        phi_new = phi[a, t - 1] + drift + diffusion
        survived = phi_new > 0.0
        out = np.zeros(a.sum(), dtype=float)
        out[survived] = phi_new[survived]
        phi[a, t], alive[a, t] = out, survived
    return phi, alive

# ---------- Plotting (FINAL) ----------
class HandlerColouredPatch(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        p = Rectangle(xy=(0, 0), width=width, height=height, facecolor=orig_handle.get_facecolor(), lw=0)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def compare_plot_final(r_grid: np.ndarray, stats_emp: pd.DataFrame, sim_data_radial: dict, out_path: Path, b_res: float, w0: int, start_radius: float):
    x, mu_emp, se_emp, n_emp = stats_emp["radius_bin"], stats_emp["mean"], stats_emp["sem"], stats_emp["n"]
    frac_emp = n_emp / n_emp.iloc[0]
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.fill_between(x, mu_emp - se_emp, mu_emp + se_emp, color='gray', alpha=0.4)
    ax.plot(x, mu_emp, color='black', lw=3)
    ax.fill_between(r_grid, sim_data_radial["mean"] - sim_data_radial["sem"], sim_data_radial["mean"] + sim_data_radial["sem"], color='crimson', alpha=0.15)
    ax.plot(r_grid, sim_data_radial["mean"], color='crimson', lw=2.5, ls=':')
    ax2 = ax.twinx()
    ax2.plot(x, frac_emp, color="dimgray", lw=2.5, ls="-")

    title = f"Radial Theory vs. Data (b_res={b_res:.4f}, $W_0$={w0})\nComparison starts at r={start_radius:.0f}"
    ax.set(xlabel="Radius, r [µm]", ylabel="Width, w [µm]", ylim=(0, None), title=title)
    ax.grid(True, linestyle=':', color='lightgray')
    ax2.set_ylabel("Survival Fraction (from r$_0$)", color="dimgray")
    ax2.tick_params(axis='y', labelcolor='dimgray')
    ax2.set(ylim=(-0.05, 1.05))

    legend_handles = [
        Patch(facecolor='gray', alpha=0.4),
        Line2D([0], [0], color='black', lw=3),
        Line2D([0], [0], color='crimson', lw=2.5, ls=':'),
        Line2D([0], [0], color='dimgray', lw=2.5, ls='-')
    ]
    legend_labels = [
        "Empirical Mean + SEM", "Empirical Mean",
        r"Theory (Radial $m_\perp$)",
        "Empirical Survival"
    ]
    ax.legend(legend_handles, legend_labels, loc='upper left', frameon=True, fontsize=12,
              handler_map={Patch: HandlerColouredPatch()})
    ax.text(-0.08, 1.08, 'c', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  -> Saved figure: {out_path.name}")

def _calculate_sim_stats(sim_phi: np.ndarray, sim_alive: np.ndarray, r_grid: np.ndarray) -> dict:
    sim_W = sim_phi * r_grid[None, :]
    n_sim = sim_alive.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n_sim > 0, (sim_W * sim_alive).sum(axis=0) / n_sim, np.nan)
        var = np.where(n_sim > 1, ((sim_alive * (sim_W - mean[None, :])**2).sum(axis=0) / (n_sim - 1)), np.nan)
        sem = np.where(n_sim > 0, np.sqrt(var / n_sim), np.nan)
    return {"mean": mean, "sem": sem}

# ---------- Main ----------
def main():
    print("--- Step 1: Loading all radial trajectory data ---")
    df_map_radial = load_radial_param_map()
    tasks = [dict(row, traj_path=str(RADIAL_TRAJ_DIR / f"traj_{row['run_id']}.json.gz")) for _, row in df_map_radial.iterrows()]
    if not tasks: sys.exit("No radial trajectories found.")

    worker = partial(process_single_run, bin_size=BIN_SIZE)
    num_workers = max(1, os.cpu_count() - 2 if os.cpu_count() else 1)
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = [res for res in tqdm(pool.imap(worker, tasks), total=len(tasks), desc="Loading radial data") if res is not None]
    if not results: sys.exit("No valid radial lineages found.")
    df_all_radial = pd.concat(results, ignore_index=True)

    max_w0_for_calib = int(df_all_radial['initial_width'].max())
    print(f"\n--- Step 2: Loading parameters (using radial calibration from W0 = {max_w0_for_calib}) ---")
    df_params = load_all_parameters(max_w0_for_calib)

    rng = np.random.default_rng(SEED)
    print(f"\n--- Step 3: Generating plots for all conditions (simulations starting at r={SIMULATION_START_RADIUS}) ---")
    grouped = df_all_radial.groupby(["b_res", "initial_width"])
    for (b_res, w0), g in tqdm(grouped, total=len(grouped), desc="Generating plots"):
        stats_full = g.groupby("radius_bin")["width"].agg(n="size", mean="mean", sem="sem").reset_index()
        stats = stats_full[stats_full["radius_bin"] >= SIMULATION_START_RADIUS].copy().reset_index(drop=True)
        if len(stats) < 3:
            continue

        r_grid = stats["radius_bin"].to_numpy()
        param_row = df_params.iloc[(df_params["b_res"] - b_res).abs().idxmin()]
        m_perp_rad, D_X = float(param_row["m_perp_radial_log"]), float(param_row["D_X"])
        r0, w0_sim_start = r_grid[0], stats['mean'].iloc[0]
        phi0 = (w0_sim_start / r0) if r0 > 0 else 0.0

        sim_phi_rad, alive_rad = simulate_angles_radial_rigorous(m_perp_rad, D_X, r_grid, phi0, N_SIMS, seed=rng.integers(1, 1e9))
        stats_sim_rad = _calculate_sim_stats(sim_phi_rad, alive_rad, r_grid)

        out_path = FIG_DIR / f"compare_final_b{b_res:.4f}_w{w0}.png"
        compare_plot_final(r_grid, stats, stats_sim_rad, out_path, b_res, w0, SIMULATION_START_RADIUS)

    print("\n✅ Completed: Final theory vs. data overlays generated.")

if __name__ == "__main__":
    main()