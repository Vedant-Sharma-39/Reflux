# FILE: scripts/paper_figures/figure1_boundary_dynamics_only.py
#
# A focused script to calculate and visualize the effective drift and diffusion
# of a mutant sector boundary.
#
# NEW: Includes an optional '--diagnostics' flag to generate a detailed
# plot showing the linear regression for each 's' value.

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import gzip
from scipy.stats import linregress
from tqdm import tqdm
import matplotlib

# --- Publication Settings ---
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

def cm_to_inch(cm):
    return cm / 2.54

# --- Analysis Constants ---
MIN_TRAJECTORIES_FOR_ANALYSIS = 10
FIT_START_TIME_PERCENT = 0.25
FIT_END_TIME_PERCENT = 0.80
COMMON_TIME_POINTS = 75
MIN_R_SQUARED_DRIFT = 0.95
MIN_R_SQUARED_DIFFUSION = 0.90
FIG_SIZE = (cm_to_inch(9), cm_to_inch(7.5))
DPI = 300

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS

def load_trajectory_file(
    project_root: str, campaign_id: str, task_id: str
) -> list | None:
    """
    Loads a single gzipped trajectory data file for a given task_id.
    The file is a JSON dictionary containing a "trajectory" key.
    """
    base_dir = os.path.join(project_root, "data", campaign_id, "trajectories")
    possible_names = [f"traj_boundary_{task_id}.json.gz", f"traj_{task_id}.json.gz"]
    file_path = None
    for name in possible_names:
        if os.path.exists(os.path.join(base_dir, name)):
            file_path = os.path.join(base_dir, name)
            break
    if not file_path:
        return None
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
            return data # Correctly extracts the list from the dict
    except (json.JSONDecodeError, gzip.BadGzipFile, KeyError):
        return None

def analyze_trajectories_for_s(
    group: pd.DataFrame, project_root: str, campaign_id: str,
    diag_axes: tuple | None = None
) -> dict | None:
    """
    Loads trajectories for 's', fits them to find v_drift and D_eff,
    and optionally generates a diagnostic plot on the provided axes.
    """
    s_val = group["s"].iloc[0]
    all_trajectories = []
    for _, row in group.iterrows():
        traj_data = load_trajectory_file(project_root, campaign_id, row["task_id"])
        if traj_data and len(traj_data) > 10:
            all_trajectories.append(np.array(traj_data))

    if len(all_trajectories) < MIN_TRAJECTORIES_FOR_ANALYSIS:
        return None

    min_max_time = min(t[-1, 0] for t in all_trajectories if len(t) > 0)
    fit_start_time = min_max_time * FIT_START_TIME_PERCENT
    fit_end_time = min_max_time * FIT_END_TIME_PERCENT
    if fit_end_time <= fit_start_time:
        return None

    common_time = np.linspace(fit_start_time, fit_end_time, num=COMMON_TIME_POINTS)
    resampled_widths = [np.interp(common_time, t_data[:, 0], t_data[:, 1]) for t_data in all_trajectories]
    mean_trajectory = np.mean(resampled_widths, axis=0)
    var_trajectory = np.var(resampled_widths, axis=0, ddof=1)

    # --- Fit for Drift and Diffusion ---
    fit_drift = linregress(common_time, mean_trajectory)
    fit_diff = linregress(common_time, var_trajectory)

    # --- Generate Diagnostic Plots (if requested) ---
    if diag_axes:
        ax_drift, ax_diff = diag_axes
        # Plot for Drift
        ax_drift.plot(common_time, mean_trajectory, 'o', ms=4, label="Mean Width Data")
        drift_line = fit_drift.intercept + fit_drift.slope * common_time
        ax_drift.plot(common_time, drift_line, 'r-', lw=2,
                      label=f"Fit (Slope={fit_drift.slope:.3f}, $R^2$={fit_drift.rvalue**2:.2f})")
        ax_drift.set_title(f"Mean Width <W(t)> (s={s_val:.3f})")
        ax_drift.set_ylabel("<W(t)>")
        ax_drift.legend(fontsize=8)
        # Plot for Diffusion
        ax_diff.plot(common_time, var_trajectory, 'o', ms=4, label="Var(Width) Data")
        diff_line = fit_diff.intercept + fit_diff.slope * common_time
        ax_diff.plot(common_time, diff_line, 'r-', lw=2,
                     label=f"Fit (Slope={fit_diff.slope:.3f}, $R^2$={fit_diff.rvalue**2:.2f})")
        ax_diff.set_title(f"Variance of Width (s={s_val:.3f})")
        ax_diff.set_ylabel("Var(W(t))")
        ax_diff.legend(fontsize=8)
        for ax in diag_axes:
            ax.set_xlabel("Time")
            ax.grid(True, linestyle=":")

    # --- Return Results if Fits are Good ---
    if fit_drift.rvalue**2 < MIN_R_SQUARED_DRIFT or fit_diff.rvalue**2 < MIN_R_SQUARED_DIFFUSION:
        return None

    v_drift = fit_drift.slope / 2.0
    d_eff = fit_diff.slope / 4.0

    return {"s": s_val, "v_drift": v_drift, "D_eff": d_eff}

def main():
    parser = argparse.ArgumentParser(description="Generate Figure: Effective Boundary Drift and Diffusion.")
    parser.add_argument("--calib-campaign", help="Override campaign ID for drift/diffusion analysis.")
    parser.add_argument("--diagnostics", action="store_true", help="Generate a diagnostic plot of all linear fits.")
    args = parser.parse_args()

    calib_experiment_name = "boundary_analysis"
    calib_campaign_id = args.calib_campaign or EXPERIMENTS.get(calib_experiment_name, {}).get("campaign_id")
    if not calib_campaign_id:
        sys.exit(f"Error: Campaign ID for '{calib_experiment_name}' not found.")

    print(f"Using boundary dynamics campaign: '{calib_campaign_id}'")
    df_calib_summary = pd.read_csv(os.path.join(PROJECT_ROOT, "data", calib_campaign_id, "analysis", f"{calib_campaign_id}_summary_aggregated.csv"))
    df_calib_summary["s"] = df_calib_summary["b_m"] - 1.0

    # --- Setup Diagnostic Figure (if requested) ---
    s_groups = list(df_calib_summary.groupby("s"))
    diag_fig, diag_axes = (None, None)
    if args.diagnostics:
        num_s = len(s_groups)
        diag_fig, diag_axes = plt.subplots(num_s, 2, figsize=(12, 5 * num_s), constrained_layout=True)
        if num_s == 1: diag_axes = np.array([diag_axes]) # Ensure it's always 2D
        diag_fig.suptitle(f"Diagnostic Fits for Campaign: {calib_campaign_id}", fontsize=16)

    # --- Main Analysis Loop ---
    analysis_results = []
    for i, (s_val, group) in enumerate(tqdm(s_groups, desc="Analyzing s for Drift/Diffusion")):
        diag_ax_pair = (diag_axes[i, 0], diag_axes[i, 1]) if args.diagnostics else None
        result = analyze_trajectories_for_s(group, PROJECT_ROOT, calib_campaign_id, diag_axes=diag_ax_pair)
        if result:
            analysis_results.append(result)

    # --- Save Diagnostic Figure ---
    if args.diagnostics:
        diag_dir = os.path.join(PROJECT_ROOT, "figures", "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        diag_path = os.path.join(diag_dir, f"{calib_campaign_id}_fits_diagnostics.png")
        diag_fig.savefig(diag_path, dpi=150, bbox_inches="tight")
        print(f"\n✅ Diagnostic plot saved to: {diag_path}")

    if not analysis_results:
        sys.exit("Analysis failed. No valid drift/diffusion values could be calculated.")

    # --- Generate and Save Final Scientific Figure ---
    drift_diff_df = pd.DataFrame(analysis_results).sort_values(by="s")
    
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax1 = plt.subplots(1, 1, figsize=FIG_SIZE, constrained_layout=True)
    fig.suptitle("Effective Boundary Motion vs. Selection", fontsize=12)

    sns.lineplot(data=drift_diff_df, x="s", y="D_eff", ax=ax1, marker="o", label=r"$D_{eff}$", color="crimson", zorder=2)
    ax1.set_ylabel(r"Effective Diffusion, $D_{eff}$", color="crimson", fontsize=8)
    ax1.tick_params(axis="y", labelcolor="crimson", labelsize=7)

    ax2 = ax1.twinx()
    sns.lineplot(data=drift_diff_df, x="s", y="v_drift", ax=ax2, marker="s", label=r"$v_{drift}$", color="navy", zorder=2)
    ax2.set_ylabel(r"Drift Velocity, $v_{drift}$", color="navy", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="navy", labelsize=7)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best", fontsize=7).set_title(None)
    ax1.set_xlabel("Selection Coefficient, $s = b_m - 1$", fontsize=8)
    ax1.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.7, zorder=1)
    
    figure_dir = os.path.join(PROJECT_ROOT, "figures")
    output_path_pdf = os.path.join(figure_dir, f"{calib_campaign_id}_boundary_dynamics.pdf")
    plt.savefig(output_path_pdf, bbox_inches="tight")
    print(f"✅ Final figure saved to {output_path_pdf}")
    plt.close('all') # Close all figures

if __name__ == "__main__":
    main()