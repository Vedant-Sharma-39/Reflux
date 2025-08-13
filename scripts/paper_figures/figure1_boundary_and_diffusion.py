# FILE: scripts/paper_figures/fig1_boundary_analysis.py
# This is the definitive, physically corrected, and user-friendly version.

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


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# --- Add project root to path to allow importing from src ---
PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS


def load_trajectory_file(project_root, campaign_id, task_id, prefix="traj"):
    """Loads a single gzipped trajectory data file for a given task_id."""
    file_path = os.path.join(
        project_root, "data", campaign_id, "trajectories", f"{prefix}_{task_id}.json.gz"
    )
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, gzip.BadGzipFile):
        return None


def analyze_trajectories_for_s(
    group: pd.DataFrame, project_root: str, campaign_id: str, debug_axes=None
):
    """
    Loads all trajectories for a given 's', calculates mean and variance over
    a stable time window, and fits them to find v_drift and D_eff for a single interface.
    """
    all_trajectories = []
    for _, row in group.iterrows():
        traj_data = load_trajectory_file(project_root, campaign_id, row["task_id"])
        if traj_data and len(traj_data) > 10:
            all_trajectories.append(np.array(traj_data))

    if len(all_trajectories) < 20:
        return None

    min_max_time = min(t[-1, 0] for t in all_trajectories)
    fit_start_time = min_max_time * 0.25
    fit_end_time = min_max_time * 0.80

    if fit_end_time <= fit_start_time:
        return None

    common_time = np.linspace(fit_start_time, fit_end_time, num=75)

    resampled_widths = [
        np.interp(common_time, t_data[:, 0], t_data[:, 1])
        for t_data in all_trajectories
    ]
    resampled_widths = np.array(resampled_widths)

    mean_trajectory = np.mean(resampled_widths, axis=0)
    var_trajectory = np.var(resampled_widths, axis=0, ddof=1)

    v_slope, v_int, v_r, _, _ = linregress(common_time, mean_trajectory)
    d_slope, d_int, d_r, _, _ = linregress(common_time, var_trajectory)

    s_val = group["s"].iloc[0]

    if debug_axes is not None:
        ax_drift, ax_diff = debug_axes
        ax_drift.plot(common_time, mean_trajectory, "o", label="Averaged Data")
        ax_drift.plot(
            common_time,
            v_int + v_slope * common_time,
            "r-",
            label=f"Fit (Slope={v_slope:.2f})",
        )
        ax_drift.set_title(f"s={s_val:.2f} (Drift of Width)")
        ax_drift.legend()
        ax_diff.plot(common_time, var_trajectory, "o", label="Averaged Data")
        ax_diff.plot(
            common_time,
            d_int + d_slope * common_time,
            "r-",
            label=f"Fit (Slope={d_slope:.1f})",
        )
        ax_diff.set_title(f"s={s_val:.2f} (Variance of Width)")
        ax_diff.legend()
        if np.isclose(s_val, 0.0):
            initial_width = group["initial_mutant_patch_size"].iloc[0]
            y_range = max(10, np.std(mean_trajectory) * 4)
            ax_drift.set_ylim(initial_width - y_range, initial_width + y_range)

    if v_r**2 < 0.95 or d_r**2 < 0.90:
        return None

    v_drift = v_slope / 2.0
    d_eff = d_slope / 4.0

    return {"s": s_val, "v_drift": v_drift, "D_eff": d_eff}


def analyze_roughness(df_summary, project_root, campaign_id):
    if df_summary.empty:
        return pd.DataFrame()
    df_summary["s"] = df_summary["b_m"] - 1.0
    saturation_results = []
    for params, group in tqdm(
        df_summary.groupby(["s", "width"]), desc="Calculating saturated roughness"
    ):
        s, width = params
        w2_sats = []
        for _, row in group.iterrows():
            roughness_data = load_trajectory_file(
                project_root, campaign_id, row["task_id"], "traj"
            )
            if not roughness_data:
                continue
            w2_vals = np.array([t[1] for t in roughness_data])
            if len(w2_vals) > 20:
                w2_sats.append(np.mean(w2_vals[-len(w2_vals) // 4 :]))
        if w2_sats:
            saturation_results.append(
                {"s": s, "L": width, "W2_sat_mean": np.mean(w2_sats)}
            )
    return pd.DataFrame(saturation_results)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 1: Boundary Dynamics & KPZ Scaling. "
        "Defaults to campaigns defined in src/config.py."
    )
    parser.add_argument(
        "--calib-campaign", help="Override campaign ID for drift/diffusion analysis."
    )
    parser.add_argument("--kpz-campaign", help="Override campaign ID for KPZ analysis.")
    parser.add_argument(
        "--debug", action="store_true", help="Generate a diagnostic plot of the fits."
    )
    args = parser.parse_args()

    calib_campaign_id = (
        args.calib_campaign or EXPERIMENTS["boundary_analysis"]["campaign_id"]
    )
    kpz_campaign_id = args.kpz_campaign or EXPERIMENTS["kpz_scaling"]["campaign_id"]

    print(f"Using calibration campaign: '{calib_campaign_id}'")
    print(f"Using KPZ scaling campaign: '{kpz_campaign_id}'")

    figure_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    # --- BUG FIX: Correctly load summary file from data directory ---
    path_calib = os.path.join(
        PROJECT_ROOT,
        "data",
        calib_campaign_id,
        "analysis",
        f"{calib_campaign_id}_summary_aggregated.csv",
    )
    path_kpz = os.path.join(
        PROJECT_ROOT,
        "data",
        kpz_campaign_id,
        "analysis",
        f"{kpz_campaign_id}_summary_aggregated.csv",
    )

    output_path = os.path.join(figure_dir, "figure1_boundary_analysis.png")
    debug_output_path = os.path.join(figure_dir, "figure1_boundary_analysis_DEBUG.png")

    try:
        df_calib_summary = pd.read_csv(path_calib)
        df_kpz_summary = pd.read_csv(path_kpz)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("Error: A summary file is empty or not found. Cannot generate Figure 1.")
        sys.exit(1)

    df_calib_summary["s"] = df_calib_summary["b_m"] - 1.0
    s_values_to_plot = sorted(df_calib_summary["s"].unique())

    debug_fig, debug_axes = (None, None)
    if args.debug:
        print("Debug mode enabled. Will generate diagnostic plot.")
        debug_fig, debug_axes = plt.subplots(
            len(s_values_to_plot),
            2,
            figsize=(14, 4 * len(s_values_to_plot)),
            constrained_layout=True,
        )
        debug_fig.suptitle("Debug: Fits on Averaged Trajectories", fontsize=20)

    print("Performing drift/diffusion analysis...")
    analysis_results = []
    for i, s_val in enumerate(tqdm(s_values_to_plot, desc="Analyzing s values")):
        group = df_calib_summary[np.isclose(df_calib_summary["s"], s_val)]
        debug_ax_pair = (
            debug_axes[i]
            if args.debug and len(s_values_to_plot) > 1
            else (debug_axes if args.debug else None)
        )
        result = analyze_trajectories_for_s(
            group, PROJECT_ROOT, calib_campaign_id, debug_axes=debug_ax_pair
        )
        if result:
            analysis_results.append(result)

    if args.debug:
        debug_fig.savefig(debug_output_path, dpi=150, bbox_inches="tight")
        print(f"Debug plot saved to: {debug_output_path}")
        plt.close(debug_fig)

    drift_diff_df = pd.DataFrame(analysis_results)
    roughness_df = analyze_roughness(df_kpz_summary, PROJECT_ROOT, kpz_campaign_id)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    fig.suptitle("Figure 1: Boundary Dynamics and Front Morphology", fontsize=20)
    axA, axB = axes[0], axes[1]

    if not drift_diff_df.empty:
        sns.lineplot(
            data=drift_diff_df,
            x="s",
            y="D_eff",
            ax=axA,
            marker="o",
            label=r"$D_{eff}$",
            color="crimson",
        )
        axA.set_ylabel(r"Effective Diffusion, $D_{eff}$", color="crimson")
        axA2 = axA.twinx()
        sns.lineplot(
            data=drift_diff_df,
            x="s",
            y="v_drift",
            ax=axA2,
            marker="s",
            label=r"$v_{drift}$",
            color="navy",
        )
        axA2.set_ylabel(r"Drift Velocity, $v_{drift}$", color="navy")
        axA.set_xlim(right=0.05)  # Extend x-axis slightly past zero
    else:
        axA.text(
            0.5,
            0.5,
            "No data for Panel A",
            ha="center",
            va="center",
            transform=axA.transAxes,
            color="gray",
        )

    axA.set_title("Effective Boundary Motion")
    axA.set_xlabel("Selection Coefficient, $s = b_m - 1$")
    fig.legend(loc="upper center", bbox_to_anchor=(0.25, 0.9), ncol=1)

    if not roughness_df.empty:
        neutral_roughness = roughness_df[np.isclose(roughness_df["s"], 0.0)]
        if not neutral_roughness.empty:
            sns.scatterplot(
                data=neutral_roughness,
                x="L",
                y="W2_sat_mean",
                ax=axB,
                s=100,
                label="Simulation Data",
                zorder=10,
            )
            log_L = np.log10(neutral_roughness["L"])
            log_W2 = np.log10(neutral_roughness["W2_sat_mean"])
            slope, intercept, _, _, _ = linregress(log_L, log_W2)
            L_fit = np.logspace(
                np.log10(neutral_roughness["L"].min()),
                np.log10(neutral_roughness["L"].max()),
                100,
            )
            W2_fit = 10 ** (intercept + slope * np.log10(L_fit))
            # --- MODIFIED: Include KPZ exponent in the label ---
            axB.plot(
                L_fit,
                W2_fit,
                "r--",
                label=f"Fit: $W^2 \\propto L^{{{slope:.2f}}}$ (KPZ exponent $\\approx 1.0$)",
            )
        else:
            axB.text(
                0.5,
                0.5,
                "No neutral (s=0) data for KPZ",
                ha="center",
                va="center",
                transform=axB.transAxes,
                color="gray",
            )
    else:
        axB.text(
            0.5,
            0.5,
            "No data for Panel B",
            ha="center",
            va="center",
            transform=axB.transAxes,
            color="gray",
        )

    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.legend()
    axB.set_title(r"Interface Roughness Scaling (KPZ)")
    axB.set_xlabel("System Width, $L$")
    axB.set_ylabel(r"Saturated Roughness, $\langle W^2_{sat} \rangle$")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure 1 saved to {output_path}")


if __name__ == "__main__":
    main()
