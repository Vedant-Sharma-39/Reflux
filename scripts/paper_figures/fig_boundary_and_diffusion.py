#!/usr/bin/env python3
# FILE: scripts/paper_figures/figure1_boundary_and_diffusion_linear_final.py
#
# Drift (m_perp) and Diffusion (D_X) from a *linear boundary* experiment.
#
# This version estimates D_X from the *variance slope*:
#     Var[X] = E[X^2] - (E[X])^2  =>  d Var[X] / d r = 4 D_X
# so D_X = (1/4) * slope(Var vs advancement).
#
# Key features
# - Robust binning: one (first) observation per bin per run, then aggregate across runs.
# - 100% survival fit window: we fit only up to the largest bin seen by *all* runs.
# - OLS for drift via E[X] vs advancement; WLS for diffusion via Var[X] vs advancement.
# - Error bars: ±SE for both E[X] and Var[X] (delta-method approximation for Var[X] SE).
# - Clean diagnostics with fit-window shading and R² annotations.

import sys, json, gzip, os, warnings
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from functools import partial
from scipy.stats import linregress

# ---------------------- Configuration ----------------------
CAMPAIGN_ID = "boundary_experiment_v1"
ADVANCEMENT_COL = "advancement"
WIDTH_COL = "width"

# Binning / fitting
BIN_SIZE = 20.0
ADVANCEMENT_FIT_MIN = 10.0
ADVANCEMENT_FIT_MAX_LIMIT = 1800.0
MIN_POINTS_FOR_FIT = 5
MIN_RUNS_PER_BIN = 3

# ---------------------- Project paths ----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.config import EXPERIMENTS  # noqa: F401

DATA_DIR = PROJECT_ROOT / "data" / CAMPAIGN_ID
ANALYSIS_DIR = DATA_DIR / "analysis"
TRAJ_DIR = DATA_DIR / "trajectories"
FIG_DIR = PROJECT_ROOT / "figures" / "figure1_boundary_and_diffusion_linear"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = ANALYSIS_DIR / "boundary_diffusion_params_linear_final.csv"

# ---------------------- Helpers ----------------------
def _safe_r2(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    """R² for a simple linear fit y ≈ intercept + slope * x."""
    if x.size < 2:
        return np.nan
    yhat = intercept + slope * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return np.nan if ss_tot <= 1e-15 else 1.0 - ss_res / ss_tot

def _np_wls_linear(x: np.ndarray, y: np.ndarray, w_sigma: np.ndarray) -> Tuple[float, float]:
    """
    Weighted least squares using numpy.polyfit (degree=1).
    polyfit expects weights ~ 1/sigma, not 1/variance.
    Returns (slope, intercept).
    """
    coeffs = np.polyfit(x, y, deg=1, w=w_sigma)
    return coeffs[0], coeffs[1]

# ---------------------- Per-run processing ----------------------
def process_single_run(task: dict, bin_size: float) -> Optional[dict]:
    traj_path = Path(task["traj_path"])
    if not traj_path.exists():
        return None

    with gzip.open(traj_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    trajectory_list = data if isinstance(data, list) else data.get("trajectory")
    if not trajectory_list:
        return None

    df = pd.DataFrame(trajectory_list, columns=[ADVANCEMENT_COL, WIDTH_COL])
    if df.empty:
        return None

    df = df.dropna(subset=[ADVANCEMENT_COL, WIDTH_COL]).copy()
    if df.empty:
        return None

    # Bin by advancement and keep the *first* observation per bin per run
    df["bin_idx"] = np.floor(df[ADVANCEMENT_COL].to_numpy(float) / bin_size).astype(np.int64)
    df.sort_values([ADVANCEMENT_COL], kind="stable", inplace=True)
    df_first = df.drop_duplicates(subset=["bin_idx"], keep="first")
    if df_first.empty:
        return None

    adv_bins = (df_first["bin_idx"].to_numpy() * bin_size).astype(float)
    width_bins = df_first[WIDTH_COL].to_numpy(float)

    return {"b_m": task["b_m"], "adv_bins": adv_bins, "width_bins": width_bins}

# ---------------------- I/O ----------------------
def load_param_map() -> pd.DataFrame:
    csv_path = ANALYSIS_DIR / f"{CAMPAIGN_ID}_summary_aggregated.csv"
    if not csv_path.exists():
        sys.exit(f"[ERROR] Summary file not found: {csv_path}")
    df = pd.read_csv(csv_path, usecols=["task_id", "b_m"])
    df = df.rename(columns={"task_id": "run_id"})
    return df.drop_duplicates("run_id")

# ---------------------- Plotting ----------------------
def plot_fits(
    df_all_points: pd.DataFrame,
    df_agg: pd.DataFrame,
    fit_drift: Dict[str, float],
    fit_var: Dict[str, float],
    m_perp: float,
    D_X: float,
    b_m: float,
    outdir: Path,
    adv_fit_min: float,
    adv_fit_max: float,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Linear Boundary: Drift & Diffusion  (b_m = {b_m:.4f})", fontsize=16)

    # ---- Panel 1: Drift from mean width (OLS) ----
    ax1.scatter(
        df_all_points["adv_bin"], df_all_points["width_bin"],
        s=6, alpha=0.08, color="gray", label="Per-run samples"
    )
    ax1.errorbar(
        df_agg["adv_bin"], df_agg["width_mean"],
        yerr=df_agg["width_se"],
        fmt="o", ms=5, lw=1, capsize=3,
        color="tab:blue", ecolor="tab:blue", alpha=0.95,
        label="Mean width ± SE"
    )

    adv_line = np.array([adv_fit_min, adv_fit_max], dtype=float)
    mean_fit_line = fit_drift["intercept"] + fit_drift["slope"] * adv_line
    ax1.plot(adv_line, mean_fit_line, "r--", lw=2.5,
             label=f"OLS fit: slope={fit_drift['slope']:.4g}  (m⊥={m_perp:.3g})")
    ax1.axvspan(adv_fit_min, adv_fit_max, color="gray", alpha=0.12, label="Fit window (100% survival)")
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.set(title="Drift from Mean Width", xlabel="Front Advancement", ylabel="Mean Patch Width")
    ax1.legend(frameon=False, loc="best")
    ax1.text(
        0.02, 0.98,
        f"R² = {fit_drift['r2']:.3f}",
        transform=ax1.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, lw=0.5)
    )

    # ---- Panel 2: Diffusion from variance (WLS) ----
    ax2.errorbar(
        df_agg["adv_bin"], df_agg["var_x"],
        yerr=df_agg["var_se"],
        fmt="o", ms=5, lw=1, capsize=3,
        color="tab:green", ecolor="tab:green", alpha=0.95,
        label="Var[X] ± SE"
    )

    var_fit_line = fit_var["intercept"] + fit_var["slope"] * adv_line
    ax2.plot(adv_line, var_fit_line, "r--", lw=2.5,
             label=f"WLS fit: slope={fit_var['slope']:.4g}  (D_X={D_X:.3g})")
    ax2.axvspan(adv_fit_min, adv_fit_max, color="gray", alpha=0.12)
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.set(title="Diffusion from Variance (Var[X])", xlabel="Front Advancement", ylabel="Var[X]")
    ax2.legend(frameon=False, loc="best")
    ax2.text(
        0.02, 0.98,
        f"Weighted R² ≈ {fit_var['r2']:.3f}",
        transform=ax2.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, lw=0.5)
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = outdir / f"linear_fits_final_bm_{b_m:.4f}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# ---------------------- Main ----------------------
def main():
    print(f"=== Estimating drift/diffusion for LINEAR campaign: {CAMPAIGN_ID} ===")
    df_map = load_param_map()
    tasks = [
        dict(row, traj_path=str(TRAJ_DIR / f"traj_{row['run_id']}.json.gz"))
        for _, row in df_map.iterrows()
    ]
    if not tasks:
        sys.exit("No tasks found.")

    num_workers = max(1, (os.cpu_count() or 2) - 2)
    worker = partial(process_single_run, bin_size=BIN_SIZE)
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(worker, tasks), total=len(tasks), desc="Processing runs"))
    results = [r for r in results if r is not None]
    if not results:
        sys.exit("No valid trajectories found.")

    rows_out = []
    df_results = pd.DataFrame(results)

    print("\nAggregating and fitting by b_m using 100% survival window + WLS on Var[X] ...")
    for b_m, g_res in tqdm(df_results.groupby("b_m"), desc="Fitting by b_m"):
        total_runs = len(g_res)

        # Expand to long format of per-run samples
        all_pts = []
        for adv_arr, w_arr in zip(g_res["adv_bins"], g_res["width_bins"]):
            if len(adv_arr) == 0:
                continue
            all_pts.append(pd.DataFrame({"adv_bin": adv_arr, "width_bin": w_arr}))
        if not all_pts:
            continue
        df_all_points = pd.concat(all_pts, ignore_index=True)

        # Aggregate per-bin across runs
        grp = df_all_points.groupby("adv_bin")
        df_agg = grp["width_bin"].agg(
            width_mean="mean",
            width_var=lambda x: np.var(x, ddof=1) if len(x) > 1 else np.nan,  # sample var across runs
            n_runs="count",
            ms_mean=lambda x: np.mean(np.square(x)),       # E[X^2]
            m4_mean=lambda x: np.mean(np.power(x, 4)),     # E[X^4]
        ).reset_index()

        # Filter to bins with enough runs
        df_agg = df_agg[df_agg["n_runs"] >= MIN_RUNS_PER_BIN].sort_values("adv_bin")
        if df_agg.empty:
            warnings.warn(f"No bins with >= {MIN_RUNS_PER_BIN} runs for b_m={b_m}.")
            continue

        # Standard errors:
        # SE(mean) = sqrt(Var / n)
        df_agg["width_se"] = np.sqrt(np.maximum(df_agg["width_var"], 0.0) / df_agg["n_runs"])

        # Var[X] using raw moments (guard negatives)
        df_agg["var_x"] = np.maximum(df_agg["ms_mean"] - np.square(df_agg["width_mean"]), 0.0)

        # Var( mean(X^2) ) = ( E[X^4] - (E[X^2])^2 ) / n   → SE(mean(X^2)) = sqrt( . )
        ms_var = np.maximum(df_agg["m4_mean"] - np.square(df_agg["ms_mean"]), 0.0) / df_agg["n_runs"]

        # Delta-method approximation for Var(Var[X]):
        # Var(Var) ≈ Var(E[X^2]) + 4 * (E[X])^2 * Var(E[X])   (ignoring covariance term)
        # where Var(E[X]) = Var(X) / n_runs
        var_mean_var = ms_var + 4.0 * np.square(df_agg["width_mean"]) * (np.maximum(df_agg["width_var"], 0.0) / df_agg["n_runs"])
        df_agg["var_se"] = np.sqrt(np.maximum(var_mean_var, 0.0))

        # 100% survival window: bins present in *all* runs
        survival_df = df_agg[df_agg["n_runs"] == total_runs]
        dynamic_adv_max = survival_df["adv_bin"].max() if not survival_df.empty else (ADVANCEMENT_FIT_MIN - 1.0)
        effective_adv_max = min(ADVANCEMENT_FIT_MAX_LIMIT, dynamic_adv_max)

        print(
            f"  -> b_m={b_m:.4f}: {total_runs} runs. "
            f"100% survival up to adv={dynamic_adv_max:.1f}. "
            f"Fit window: [{ADVANCEMENT_FIT_MIN}, {effective_adv_max:.1f}]"
        )

        # Select window
        fit_mask = (df_agg["adv_bin"] >= ADVANCEMENT_FIT_MIN) & (df_agg["adv_bin"] <= effective_adv_max)
        df_fit = df_agg.loc[
            fit_mask, ["adv_bin", "width_mean", "width_se", "var_x", "var_se"]
        ].dropna()
        if len(df_fit) < MIN_POINTS_FOR_FIT:
            print(f"  -> [WARN] Not enough points ({len(df_fit)}) in window to fit.")
            rows_out.append(dict(b_m=b_m, m_perp=np.nan, D_X=np.nan, n_runs=total_runs,
                                 adv_fit_max_used=effective_adv_max))
            continue

        # ---- Drift from mean (OLS) ----
        drift_lr = linregress(df_fit["adv_bin"].to_numpy(), df_fit["width_mean"].to_numpy())
        m_perp_est = drift_lr.slope / 2.0
        r2_drift = _safe_r2(
            df_fit["adv_bin"].to_numpy(),
            df_fit["width_mean"].to_numpy(),
            slope=drift_lr.slope,
            intercept=drift_lr.intercept,
        )

        # ---- Diffusion from variance (WLS) ----
        x = df_fit["adv_bin"].to_numpy(float)
        y = df_fit["var_x"].to_numpy(float)
        sigma = df_fit["var_se"].to_numpy(float)

        # Keep only finite sigma, positive-ish
        mask_ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 1e-12)
        x_w, y_w, sigma_w = x[mask_ok], y[mask_ok], sigma[mask_ok]
        if x_w.size < MIN_POINTS_FOR_FIT:
            print(f"  -> [WARN] Not enough finite-weight points for WLS.")
            rows_out.append(dict(b_m=b_m, m_perp=m_perp_est, D_X=np.nan, n_runs=total_runs,
                                 adv_fit_max_used=effective_adv_max))
            continue

        w_sigma = 1.0 / sigma_w  # np.polyfit expects weights ≈ 1/sigma
        slope_var, intercept_var = _np_wls_linear(x_w, y_w, w_sigma)

        # Weighted R² (~) using weights ~ 1/sigma^2
        w = 1.0 / (sigma_w ** 2)
        w /= w.sum()
        yhat = intercept_var + slope_var * x_w
        ss_res = np.sum((1.0 / (sigma_w ** 2)) * (y_w - yhat) ** 2)
        ybar = np.sum(w * y_w)
        ss_tot = np.sum((1.0 / (sigma_w ** 2)) * (y_w - ybar) ** 2)
        r2_var = np.nan if ss_tot <= 1e-15 else 1.0 - ss_res / ss_tot

        D_X_est = slope_var / 4.0

        # ---- Plots ----
        fit_drift = {"slope": drift_lr.slope, "intercept": drift_lr.intercept, "r2": r2_drift}
        fit_var = {"slope": slope_var, "intercept": intercept_var, "r2": r2_var}
        plot_fits(
            df_all_points=df_all_points,
            df_agg=df_agg,
            fit_drift=fit_drift,
            fit_var=fit_var,
            m_perp=m_perp_est,
            D_X=D_X_est,
            b_m=b_m,
            outdir=FIG_DIR,
            adv_fit_min=ADVANCEMENT_FIT_MIN,
            adv_fit_max=effective_adv_max,
        )

        rows_out.append(
            dict(
                b_m=b_m,
                m_perp=m_perp_est,
                D_X=D_X_est,
                n_runs=total_runs,
                adv_fit_max_used=effective_adv_max,
                drift_slope=drift_lr.slope,
                drift_intercept=drift_lr.intercept,
                drift_R2=r2_drift,
                var_slope=slope_var,
                var_intercept=intercept_var,
                var_R2=r2_var,
            )
        )

    df_out = pd.DataFrame(rows_out).sort_values("b_m").reset_index(drop=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Saved parameter table: {OUT_CSV}")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print("\n--- Final Parameters (Linear; 100% survival; Var-slope WLS) ---")
        print(df_out)

if __name__ == "__main__":
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    main()
