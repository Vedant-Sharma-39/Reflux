# FILE: scripts/paper_figures/fig1_boundary_analysis.py
# This is the definitive, physically corrected, and user-friendly version.
# UPDATED to work with the refactored experiment structure in src/config.py.

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


# --- End Publication Settings ---


# --- Constants for Configuration and Analysis ---
MIN_TRAJECTORIES_FOR_ANALYSIS = 20
FIT_START_TIME_PERCENT = 0.25
FIT_END_TIME_PERCENT = 0.80
COMMON_TIME_POINTS = 75
MIN_R_SQUARED_DRIFT = 0.95
MIN_R_SQUARED_DIFFUSION = 0.90
ROUGHNESS_SATURATION_WINDOW_PERCENT = 0.25

# --- Plotting Constants ---
# --- CHANGE: Figure size to 2-column width (17.8 cm) ---
FIG_SIZE = (cm_to_inch(17.8), cm_to_inch(7.5))
DEBUG_FIG_WIDTH = 14
DEBUG_AXES_HEIGHT_PER_S_VALUE = 4
DPI = 300


def get_project_root():
    """Returns the absolute path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import EXPERIMENTS
except ImportError:
    print("Error: Could not import EXPERIMENTS from src/config.py.")
    sys.exit(1)


def load_trajectory_file(
    project_root: str, campaign_id: str, task_id: str, prefix: str = "traj"
) -> list | None:
    """
    Loads a single gzipped trajectory data file for a given task_id.
    """
    file_path = os.path.join(
        project_root, "data", campaign_id, "trajectories", f"{prefix}_{task_id}.json.gz"
    )
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, gzip.BadGzipFile) as e:
        return None


def analyze_trajectories_for_s(
    group: pd.DataFrame, project_root: str, campaign_id: str, debug_axes=None
) -> dict | None:
    """
    Loads all trajectories for a given 's' value, calculates mean and variance over
    a stable time window, and fits them to find v_drift and D_eff for a single interface.
    """
    all_trajectories = []
    for _, row in group.iterrows():
        traj_data = load_trajectory_file(
            project_root, campaign_id, row["task_id"], prefix="traj_boundary"
        )
        if traj_data and len(traj_data) > 10:
            all_trajectories.append(np.array(traj_data))

    if len(all_trajectories) < MIN_TRAJECTORIES_FOR_ANALYSIS:
        return None

    min_max_time = min(t[-1, 0] for t in all_trajectories)
    fit_start_time = min_max_time * FIT_START_TIME_PERCENT
    fit_end_time = min_max_time * FIT_END_TIME_PERCENT

    if fit_end_time <= fit_start_time:
        return None

    common_time = np.linspace(fit_start_time, fit_end_time, num=COMMON_TIME_POINTS)
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
        ax_drift.plot(
            common_time, mean_trajectory, "o", label="Averaged Data", markersize=4
        )
        ax_drift.plot(
            common_time,
            v_int + v_slope * common_time,
            "r-",
            label=f"Fit (Slope={v_slope:.2f}, $R^2$={v_r**2:.2f})",
        )
        ax_drift.set_title(f"s={s_val:.2f} (Drift of Width)")
        ax_drift.set_xlabel("Time")
        ax_drift.set_ylabel("Mean Width")
        ax_drift.legend(fontsize=8)
        ax_drift.grid(True, linestyle=":", alpha=0.7)

        ax_diff.plot(
            common_time, var_trajectory, "o", label="Averaged Data", markersize=4
        )
        ax_diff.plot(
            common_time,
            d_int + d_slope * common_time,
            "r-",
            label=f"Fit (Slope={d_slope:.1f}, $R^2$={d_r**2:.2f})",
        )
        ax_diff.set_title(f"s={s_val:.2f} (Variance of Width)")
        ax_diff.set_xlabel("Time")
        ax_diff.set_ylabel("Variance of Width")
        ax_diff.legend(fontsize=8)
        ax_diff.grid(True, linestyle=":", alpha=0.7)

        if np.isclose(s_val, 0.0):
            initial_width = group["initial_mutant_patch_size"].iloc[0]
            y_range = max(10, np.std(mean_trajectory) * 4)
            ax_drift.set_ylim(initial_width - y_range, initial_width + y_range)

    if v_r**2 < MIN_R_SQUARED_DRIFT or d_r**2 < MIN_R_SQUARED_DIFFUSION:
        return None

    v_drift = v_slope / 2.0
    d_eff = d_slope / 4.0

    return {"s": s_val, "v_drift": v_drift, "D_eff": d_eff}


def analyze_roughness(
    df_summary: pd.DataFrame, project_root: str, campaign_id: str
) -> pd.DataFrame:
    """
    Analyzes the saturated roughness (W^2_sat) for different system widths (L).
    """
    if df_summary.empty:
        return pd.DataFrame()

    if "b_m" in df_summary.columns and "s" not in df_summary.columns:
        df_summary["s"] = df_summary["b_m"] - 1.0

    saturation_results = []
    for params, group in tqdm(
        df_summary.groupby(["s", "width"]), desc="Calculating saturated roughness"
    ):
        s, width = params
        w2_sats = []
        for _, row in group.iterrows():
            roughness_data = load_trajectory_file(
                project_root, campaign_id, row["task_id"], "traj_roughness"
            )
            if not roughness_data:
                continue

            w2_vals = np.array([t[1] for t in roughness_data])
            if len(w2_vals) > 20:
                num_points_for_saturation = max(
                    1,
                    len(w2_vals)
                    // (100 // int(ROUGHNESS_SATURATION_WINDOW_PERCENT * 100)),
                )
                w2_sats.append(np.mean(w2_vals[-num_points_for_saturation:]))

        if w2_sats:
            saturation_results.append(
                {"s": s, "L": width, "W2_sat_mean": np.mean(w2_sats)}
            )
    return pd.DataFrame(saturation_results)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 1: Boundary Dynamics & KPZ Scaling. "
    )
    parser.add_argument(
        "--calib-campaign", help="Override campaign ID for drift/diffusion analysis."
    )
    parser.add_argument("--kpz-campaign", help="Override campaign ID for KPZ analysis.")
    parser.add_argument(
        "--debug", action="store_true", help="Generate a diagnostic plot of the fits."
    )
    args = parser.parse_args()

    calib_experiment_name = "boundary_velocity_analysis"
    kpz_experiment_name = "boundary_roughness_scaling"

    calib_campaign_id = args.calib_campaign or EXPERIMENTS.get(
        calib_experiment_name, {}
    ).get("campaign_id")
    kpz_campaign_id = args.kpz_campaign or EXPERIMENTS.get(kpz_experiment_name, {}).get(
        "campaign_id"
    )

    if not calib_campaign_id or not kpz_campaign_id:
        print(
            "Error: Campaign ID not found. Check src/config.py or specify via command line."
        )
        sys.exit(1)

    print(f"Using calibration campaign: '{calib_campaign_id}'")
    print(f"Using KPZ scaling campaign: '{kpz_campaign_id}'")

    figure_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figure_dir, exist_ok=True)

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

    # --- CHANGE: Output filenames ---
    output_path_pdf = os.path.join(figure_dir, "figure1_boundary_analysis.pdf")
    output_path_eps = os.path.join(figure_dir, "figure1_boundary_analysis.eps")
    debug_output_path = os.path.join(figure_dir, "figure1_boundary_analysis_DEBUG.png")

    try:
        df_calib_summary = pd.read_csv(path_calib)
        df_kpz_summary = pd.read_csv(path_kpz)
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"Error loading data: {e}. Please ensure analysis pipelines have run.")
        sys.exit(1)

    if "b_m" in df_calib_summary.columns:
        df_calib_summary["s"] = df_calib_summary["b_m"] - 1.0
    else:
        sys.exit("Error: 'b_m' column not found in calibration summary.")

    if "b_m" in df_kpz_summary.columns:
        df_kpz_summary["s"] = df_kpz_summary["b_m"] - 1.0

    s_values_to_plot = sorted(df_calib_summary["s"].unique())
    debug_fig, debug_axes = (None, None)
    if args.debug:
        fig_height = DEBUG_AXES_HEIGHT_PER_S_VALUE * len(s_values_to_plot)
        debug_fig, debug_axes = plt.subplots(
            len(s_values_to_plot),
            2,
            figsize=(DEBUG_FIG_WIDTH, fig_height),
            constrained_layout=True,
        )
        debug_fig.suptitle("Debug: Fits on Averaged Trajectories", fontsize=20)
        if len(s_values_to_plot) == 1:
            debug_axes = debug_axes[np.newaxis, :]

    analysis_results = []
    for i, s_val in enumerate(
        tqdm(s_values_to_plot, desc="Analyzing s for Drift/Diffusion")
    ):
        group = df_calib_summary[np.isclose(df_calib_summary["s"], s_val)]
        current_debug_ax_pair = (
            (debug_axes[i, 0], debug_axes[i, 1]) if args.debug else None
        )
        result = analyze_trajectories_for_s(
            group, PROJECT_ROOT, calib_campaign_id, debug_axes=current_debug_ax_pair
        )
        if result:
            analysis_results.append(result)

    if args.debug:
        debug_fig.savefig(debug_output_path, dpi=DPI, bbox_inches="tight")
        plt.close(debug_fig)

    drift_diff_df = pd.DataFrame(analysis_results)
    roughness_df = analyze_roughness(df_kpz_summary, PROJECT_ROOT, kpz_campaign_id)

    # --- CHANGE: Plotting setup for publication ---
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, constrained_layout=True)
    fig.suptitle("Figure 1: Boundary Dynamics and Front Morphology", fontsize=12)
    axA, axB = axes[0], axes[1]

    if not drift_diff_df.empty:
        drift_diff_df = drift_diff_df.sort_values(by="s")
        sns.lineplot(
            data=drift_diff_df,
            x="s",
            y="D_eff",
            ax=axA,
            marker="o",
            label=r"$D_{eff}$",
            color="crimson",
            zorder=2,
        )
        axA.set_ylabel(r"Effective Diffusion, $D_{eff}$", color="crimson", fontsize=8)
        axA.tick_params(axis="y", labelcolor="crimson", labelsize=7)
        axA.grid(True, linestyle=":", alpha=0.7)

        axA2 = axA.twinx()
        sns.lineplot(
            data=drift_diff_df,
            x="s",
            y="v_drift",
            ax=axA2,
            marker="s",
            label=r"$v_{drift}$",
            color="navy",
            zorder=2,
        )
        axA2.set_ylabel(r"Drift Velocity, $v_{drift}$", color="navy", fontsize=8)
        axA2.tick_params(axis="y", labelcolor="navy", labelsize=7)

        lines_A, labels_A = axA.get_legend_handles_labels()
        lines_A2, labels_A2 = axA2.get_legend_handles_labels()
        axA.legend(lines_A + lines_A2, labels_A + labels_A2, loc="best", fontsize=7)
        axA.get_legend().set_title(None)

        axA.set_xlim(drift_diff_df["s"].min() - 0.01, drift_diff_df["s"].max() + 0.01)
        axA2.set_xlim(axA.get_xlim())
    else:
        axA.text(
            0.5,
            0.5,
            "No data for Panel A",
            ha="center",
            va="center",
            transform=axA.transAxes,
            color="gray",
            fontsize=10,
        )

    axA.set_title("(A) Effective Boundary Motion", fontsize=10)
    axA.set_xlabel("Selection Coefficient, $s = b_m - 1$", fontsize=8)
    axA.tick_params(axis="x", labelsize=7)
    axA.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.7, zorder=1)

    if not roughness_df.empty:
        neutral_roughness = roughness_df[np.isclose(roughness_df["s"], 0.0)]
        if not neutral_roughness.empty:
            sns.scatterplot(
                data=neutral_roughness,
                x="L",
                y="W2_sat_mean",
                ax=axB,
                s=50,
                label="Simulation Data",
                zorder=10,
                edgecolor="black",
                alpha=0.8,
            )

            log_L = np.log10(neutral_roughness["L"])
            log_W2 = np.log10(neutral_roughness["W2_sat_mean"])
            finite_mask = np.isfinite(log_L) & np.isfinite(log_W2)
            if finite_mask.sum() >= 2:
                slope, intercept, r_value, _, _ = linregress(
                    log_L[finite_mask], log_W2[finite_mask]
                )
                L_fit = np.logspace(
                    np.log10(neutral_roughness["L"].min()),
                    np.log10(neutral_roughness["L"].max()),
                    100,
                )
                W2_fit = 10 ** (intercept + slope * np.log10(L_fit))
                axB.plot(
                    L_fit,
                    W2_fit,
                    "r--",
                    label=f"Fit: $\\langle W^2_{{sat}} \\rangle \\propto L^{{{slope:.2f}}}$ ($R^2={r_value**2:.2f}$)",
                    linewidth=1.5,
                )
            else:
                axB.text(
                    0.5,
                    0.7,
                    "Not enough valid data for fit.",
                    ha="center",
                    va="center",
                    transform=axB.transAxes,
                    color="gray",
                    fontsize=8,
                )
        else:
            axB.text(
                0.5,
                0.5,
                "No neutral (s=0) data found.",
                ha="center",
                va="center",
                transform=axB.transAxes,
                color="gray",
                fontsize=10,
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
            fontsize=10,
        )

    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.legend(loc="upper left", fontsize=7)
    axB.set_title(r"(B) Interface Roughness Scaling (KPZ)", fontsize=10)
    axB.set_xlabel("System Width, $L$", fontsize=8)
    axB.set_ylabel(r"Saturated Roughness, $\langle W^2_{sat} \rangle$", fontsize=8)
    axB.tick_params(axis="both", which="major", labelsize=7)
    axB.grid(True, which="both", linestyle=":", alpha=0.7)

    # --- CHANGE: Save to PDF and EPS ---
    plt.savefig(output_path_pdf, bbox_inches="tight")
    plt.savefig(output_path_eps, bbox_inches="tight")
    print(f"\nFigure 1 saved to {output_path_pdf} and {output_path_eps}")
    plt.close(fig)


if __name__ == "__main__":
    main()
