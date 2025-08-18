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

# --- Constants for Configuration and Analysis ---
MIN_TRAJECTORIES_FOR_ANALYSIS = 20
FIT_START_TIME_PERCENT = 0.25
FIT_END_TIME_PERCENT = 0.80
COMMON_TIME_POINTS = 75
MIN_R_SQUARED_DRIFT = 0.95  # R^2 threshold for drift velocity fit
MIN_R_SQUARED_DIFFUSION = 0.90  # R^2 threshold for effective diffusion fit
ROUGHNESS_SATURATION_WINDOW_PERCENT = 0.25 # Percentage of end points to average for saturation

# --- Plotting Constants ---
FIG_SIZE = (18, 7)
DEBUG_FIG_WIDTH = 14
DEBUG_AXES_HEIGHT_PER_S_VALUE = 4
DPI = 300
FONT_SCALE = 1.5


def get_project_root():
    """Returns the absolute path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# --- Add project root to path to allow importing from src ---
PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import EXPERIMENTS after adding project root to path
try:
    from src.config import EXPERIMENTS
except ImportError:
    print("Error: Could not import EXPERIMENTS from src/config.py.")
    print("Please ensure src/config.py exists and is accessible.")
    sys.exit(1)


def load_trajectory_file(project_root: str, campaign_id: str, task_id: str, prefix: str = "traj") -> list | None:
    """
    Loads a single gzipped trajectory data file for a given task_id.
    Returns the loaded JSON data as a list of lists (trajectory points)
    or None if the file is not found or corrupted.
    """
    file_path = os.path.join(
        project_root, "data", campaign_id, "trajectories", f"{prefix}_{task_id}.json.gz"
    )
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, gzip.BadGzipFile) as e:
        # print(f"Warning: Could not load {file_path}. Error: {e}") # Uncomment for verbose debugging
        return None


def analyze_trajectories_for_s(
    group: pd.DataFrame, project_root: str, campaign_id: str, debug_axes=None
) -> dict | None:
    """
    Loads all trajectories for a given 's' value, calculates mean and variance over
    a stable time window, and fits them to find v_drift and D_eff for a single interface.

    Args:
        group (pd.DataFrame): A DataFrame group containing tasks for a specific 's' value.
        project_root (str): Path to the project root directory.
        campaign_id (str): The ID of the campaign to load trajectory data from.
        debug_axes (tuple, optional): A tuple of (drift_ax, diff_ax) for plotting debug
                                      information. Defaults to None.

    Returns:
        dict | None: A dictionary containing 's', 'v_drift', and 'D_eff' if successful,
                     otherwise None (e.g., not enough trajectories, poor fit).
    """
    all_trajectories = []
    for _, row in group.iterrows():
        # Use 'traj_boundary' for this analysis as it contains width data
        traj_data = load_trajectory_file(
            project_root, campaign_id, row["task_id"], prefix="traj_boundary"
        )
        # Ensure trajectory has at least 10 points to be considered valid
        if traj_data and len(traj_data) > 10:
            all_trajectories.append(np.array(traj_data))

    if len(all_trajectories) < MIN_TRAJECTORIES_FOR_ANALYSIS:
        # print(f"Warning: Not enough trajectories ({len(all_trajectories)}) for s={group['s'].iloc[0]}. Required: {MIN_TRAJECTORIES_FOR_ANALYSIS}")
        return None

    # Determine a common time window for stable measurements
    min_max_time = min(t[-1, 0] for t in all_trajectories)
    fit_start_time = min_max_time * FIT_START_TIME_PERCENT
    fit_end_time = min_max_time * FIT_END_TIME_PERCENT

    if fit_end_time <= fit_start_time:
        # print(f"Warning: Invalid fit time window for s={group['s'].iloc[0]}.")
        return None

    common_time = np.linspace(fit_start_time, fit_end_time, num=COMMON_TIME_POINTS)

    # Resample all trajectories to the common time points
    resampled_widths = [
        np.interp(common_time, t_data[:, 0], t_data[:, 1])
        for t_data in all_trajectories
    ]
    resampled_widths = np.array(resampled_widths)

    # Calculate mean and variance trajectories
    mean_trajectory = np.mean(resampled_widths, axis=0)
    var_trajectory = np.var(resampled_widths, axis=0, ddof=1)  # ddof=1 for sample variance

    # Perform linear regression
    v_slope, v_int, v_r, _, _ = linregress(common_time, mean_trajectory)
    d_slope, d_int, d_r, _, _ = linregress(common_time, var_trajectory)

    s_val = group["s"].iloc[0]

    # Debug plotting if axes are provided
    if debug_axes is not None:
        ax_drift, ax_diff = debug_axes
        ax_drift.plot(common_time, mean_trajectory, "o", label="Averaged Data", markersize=4)
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
        ax_drift.grid(True, linestyle=':', alpha=0.7)

        ax_diff.plot(common_time, var_trajectory, "o", label="Averaged Data", markersize=4)
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
        ax_diff.grid(True, linestyle=':', alpha=0.7)

        # For s=0.0, set an appropriate y-limit to show fluctuations around initial width
        if np.isclose(s_val, 0.0):
            # Use initial_mutant_patch_size from the first row of the group (should be constant for a given s)
            initial_width = group["initial_mutant_patch_size"].iloc[0]
            # Set a dynamic y-range based on the initial width and observed mean trajectory spread
            y_range = max(10, np.std(mean_trajectory) * 4) # Ensure a minimum range
            ax_drift.set_ylim(initial_width - y_range, initial_width + y_range)


    # Check R-squared thresholds for reliable fits
    if v_r**2 < MIN_R_SQUARED_DRIFT or d_r**2 < MIN_R_SQUARED_DIFFUSION:
        # print(f"Warning: Low R^2 for s={s_val}. v_r^2={v_r**2:.2f}, d_r^2={d_r**2:.2f}")
        return None

    # Calculate v_drift and D_eff from slopes (assuming 2 * v * t and 4 * D * t scaling)
    v_drift = v_slope / 2.0
    d_eff = d_slope / 4.0

    return {"s": s_val, "v_drift": v_drift, "D_eff": d_eff}


def analyze_roughness(df_summary: pd.DataFrame, project_root: str, campaign_id: str) -> pd.DataFrame:
    """
    Analyzes the saturated roughness (W^2_sat) for different system widths (L)
    based on 'traj_roughness' files.

    Args:
        df_summary (pd.DataFrame): Summary DataFrame for the KPZ campaign.
        project_root (str): Path to the project root directory.
        campaign_id (str): The ID of the KPZ campaign to load trajectory data from.

    Returns:
        pd.DataFrame: A DataFrame with 's', 'L', and 'W2_sat_mean' values.
    """
    if df_summary.empty:
        return pd.DataFrame()

    # Ensure 's' column exists based on 'b_m'
    if 'b_m' in df_summary.columns and 's' not in df_summary.columns:
        df_summary["s"] = df_summary["b_m"] - 1.0

    saturation_results = []
    # Group by 's' and 'width' (which corresponds to L)
    for params, group in tqdm(
        df_summary.groupby(["s", "width"]), desc="Calculating saturated roughness"
    ):
        s, width = params  # width here is the system size L
        w2_sats = []
        for _, row in group.iterrows():
            # Use 'traj_roughness' for this analysis
            roughness_data = load_trajectory_file(
                project_root, campaign_id, row["task_id"], "traj_roughness"
            )
            if not roughness_data:
                continue

            # Assuming roughness_data format is [[time, W^2], [time, W^2], ...]
            w2_vals = np.array([t[1] for t in roughness_data])
            
            # Check if there are enough data points to reliably calculate saturation
            if len(w2_vals) > 20: # Arbitrary but reasonable minimum for averaging
                # Average the last ROUGHNESS_SATURATION_WINDOW_PERCENT of W^2 values
                num_points_for_saturation = max(1, len(w2_vals) // (100 // int(ROUGHNESS_SATURATION_WINDOW_PERCENT * 100)))
                w2_sats.append(np.mean(w2_vals[-num_points_for_saturation:]))
        
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

    # --- Experiment names (as defined in src/config.py) ---
    calib_experiment_name = "boundary_velocity_analysis"
    kpz_experiment_name = "boundary_roughness_scaling"

    # Get campaign IDs, using args or defaults from EXPERIMENTS config
    calib_campaign_id = (
        args.calib_campaign or EXPERIMENTS.get(calib_experiment_name, {}).get("campaign_id")
    )
    kpz_campaign_id = (
        args.kpz_campaign or EXPERIMENTS.get(kpz_experiment_name, {}).get("campaign_id")
    )

    if not calib_campaign_id:
        print(f"Error: Campaign ID for '{calib_experiment_name}' not found. Please specify with --calib-campaign or check src/config.py.")
        sys.exit(1)
    if not kpz_campaign_id:
        print(f"Error: Campaign ID for '{kpz_experiment_name}' not found. Please specify with --kpz-campaign or check src/config.py.")
        sys.exit(1)

    print(f"Using calibration campaign: '{calib_campaign_id}' (from experiment '{calib_experiment_name}')")
    print(f"Using KPZ scaling campaign: '{kpz_campaign_id}' (from experiment '{kpz_experiment_name}')")

    figure_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    # Define paths to summary CSV files
    path_calib = os.path.join(
        PROJECT_ROOT, "data", calib_campaign_id, "analysis",
        f"{calib_campaign_id}_summary_aggregated.csv"
    )
    path_kpz = os.path.join(
        PROJECT_ROOT, "data", kpz_campaign_id, "analysis",
        f"{kpz_campaign_id}_summary_aggregated.csv"
    )

    output_path = os.path.join(figure_dir, "figure1_boundary_analysis.png")
    debug_output_path = os.path.join(figure_dir, "figure1_boundary_analysis_DEBUG.png")

    df_calib_summary = pd.DataFrame()
    df_kpz_summary = pd.DataFrame()

    try:
        df_calib_summary = pd.read_csv(path_calib)
        print(f"Loaded calibration summary from: {path_calib}")
    except FileNotFoundError:
        print(f"Error: Calibration summary file not found at {path_calib}.")
        print("Please ensure you have run the analysis pipeline for the calibration campaign.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: Calibration summary file at {path_calib} is empty.")
        sys.exit(1)

    try:
        df_kpz_summary = pd.read_csv(path_kpz)
        print(f"Loaded KPZ summary from: {path_kpz}")
    except FileNotFoundError:
        print(f"Error: KPZ summary file not found at {path_kpz}.")
        print("Please ensure you have run the analysis pipeline for the KPZ campaign.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: KPZ summary file at {path_kpz} is empty.")
        sys.exit(1)

    # Ensure 's' column is correctly calculated for both dataframes
    if 'b_m' in df_calib_summary.columns:
        df_calib_summary["s"] = df_calib_summary["b_m"] - 1.0
    else:
        print("Warning: 'b_m' column not found in calibration summary. Cannot calculate 's'.")
        sys.exit(1) # 's' is essential for this plot

    if 'b_m' in df_kpz_summary.columns:
        df_kpz_summary["s"] = df_kpz_summary["b_m"] - 1.0
    # No else needed here, analyze_roughness will handle if 's' isn't explicitly there but 'b_m' is

    s_values_to_plot = sorted(df_calib_summary["s"].unique())

    # --- Debug Plot Setup ---
    debug_fig, debug_axes = (None, None)
    if args.debug:
        print("Debug mode enabled. Generating diagnostic plot of fits...")
        # Adjust figure height based on number of s values
        fig_height = DEBUG_AXES_HEIGHT_PER_S_VALUE * len(s_values_to_plot)
        debug_fig, debug_axes = plt.subplots(
            len(s_values_to_plot), 2,
            figsize=(DEBUG_FIG_WIDTH, fig_height),
            constrained_layout=True
        )
        debug_fig.suptitle("Debug: Fits on Averaged Trajectories for Each 's'", fontsize=20)
        # Ensure debug_axes is always a 2D array, even for a single s value
        if len(s_values_to_plot) == 1:
            debug_axes = debug_axes[np.newaxis, :] # Make it (1, 2)

    print("Performing drift/diffusion analysis...")
    analysis_results = []
    # Use tqdm for progress bar
    for i, s_val in enumerate(tqdm(s_values_to_plot, desc="Analyzing s values for Drift/Diffusion")):
        group = df_calib_summary[np.isclose(df_calib_summary["s"], s_val)]
        
        current_debug_ax_pair = None
        if args.debug:
            current_debug_ax_pair = (debug_axes[i, 0], debug_axes[i, 1])
            
        result = analyze_trajectories_for_s(
            group, PROJECT_ROOT, calib_campaign_id, debug_axes=current_debug_ax_pair
        )
        if result:
            analysis_results.append(result)

    if args.debug:
        debug_fig.savefig(debug_output_path, dpi=DPI, bbox_inches="tight")
        print(f"Debug plot saved to: {debug_output_path}")
        plt.close(debug_fig) # Close the debug figure to free memory

    drift_diff_df = pd.DataFrame(analysis_results)
    print(f"Drift/Diffusion analysis complete. Found {len(drift_diff_df)} valid results.")

    print("Performing roughness analysis...")
    roughness_df = analyze_roughness(df_kpz_summary, PROJECT_ROOT, kpz_campaign_id)
    print(f"Roughness analysis complete. Found {len(roughness_df)} valid results.")


    # --- Main Figure Plotting ---
    sns.set_theme(style="whitegrid", context="paper", font_scale=FONT_SCALE)
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, constrained_layout=True)
    fig.suptitle("Figure 1: Boundary Dynamics and Front Morphology", fontsize=20)
    axA, axB = axes[0], axes[1]

    # Panel A: Drift Velocity and Effective Diffusion
    if not drift_diff_df.empty:
        # Sort by 's' for cleaner lines
        drift_diff_df = drift_diff_df.sort_values(by="s")
        
        sns.lineplot(
            data=drift_diff_df,
            x="s",
            y="D_eff",
            ax=axA,
            marker="o",
            label=r"$D_{eff}$",
            color="crimson",
            zorder=2
        )
        axA.set_ylabel(r"Effective Diffusion, $D_{eff}$", color="crimson")
        axA.tick_params(axis='y', labelcolor="crimson")
        axA.grid(True, linestyle=':', alpha=0.7)


        axA2 = axA.twinx() # Create a twin axis for v_drift
        sns.lineplot(
            data=drift_diff_df,
            x="s",
            y="v_drift",
            ax=axA2,
            marker="s",
            label=r"$v_{drift}$",
            color="navy",
            zorder=2
        )
        axA2.set_ylabel(r"Drift Velocity, $v_{drift}$", color="navy")
        axA2.tick_params(axis='y', labelcolor="navy")
        
        # Combine legends from both axes
        lines_A, labels_A = axA.get_legend_handles_labels()
        lines_A2, labels_A2 = axA2.get_legend_handles_labels()
        axA.legend(lines_A + lines_A2, labels_A + labels_A2, loc="upper center", bbox_to_anchor=(0.25, 1.05), ncol=1)
        axA.get_legend().set_title(None) # Remove legend title if seaborn adds one

        axA.set_xlim(drift_diff_df["s"].min() - 0.01, drift_diff_df["s"].max() + 0.01)
        # Ensure both axes share the same x-limits for consistent alignment
        axA2.set_xlim(axA.get_xlim())
    else:
        axA.text(
            0.5, 0.5, "No data for Panel A: Effective Boundary Motion",
            ha="center", va="center", transform=axA.transAxes, color="gray",
            fontsize=FONT_SCALE*1.2
        )

    axA.set_title("Effective Boundary Motion")
    axA.set_xlabel("Selection Coefficient, $s = b_m - 1$")
    axA.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1) # Add a line at s=0

    # Panel B: Interface Roughness Scaling (KPZ)
    if not roughness_df.empty:
        neutral_roughness = roughness_df[np.isclose(roughness_df["s"], 0.0)]
        
        if not neutral_roughness.empty:
            sns.scatterplot(
                data=neutral_roughness,
                x="L",
                y="W2_sat_mean",
                ax=axB,
                s=100, # Marker size
                label="Simulation Data",
                zorder=10, # Ensure points are on top
                edgecolor='black', # Add outline for visibility
                alpha=0.8
            )
            
            # Perform log-log linear regression for scaling exponent
            log_L = np.log10(neutral_roughness["L"])
            log_W2 = np.log10(neutral_roughness["W2_sat_mean"])
            
            # Filter out NaN/inf values which can occur if W2_sat_mean is zero or negative
            finite_mask = np.isfinite(log_L) & np.isfinite(log_W2)
            if finite_mask.sum() >= 2: # Need at least 2 points for regression
                slope, intercept, r_value, _, _ = linregress(log_L[finite_mask], log_W2[finite_mask])
                
                # Plot the fit line
                # Generate L values for plotting the fit line, covering the range of data
                L_fit = np.logspace(
                    np.log10(neutral_roughness["L"].min()),
                    np.log10(neutral_roughness["L"].max()),
                    100 # Number of points for the fit line
                )
                W2_fit = 10 ** (intercept + slope * np.log10(L_fit))
                axB.plot(
                    L_fit,
                    W2_fit,
                    "r--",
                    label=f"Fit: $\\langle W^2_{{sat}} \\rangle \\propto L^{{{slope:.2f}}}$ ($R^2={r_value**2:.2f}$)",
                    linewidth=2
                )
            else:
                axB.text(
                    0.5, 0.7, "Not enough valid data points for KPZ fit.",
                    ha="center", va="center", transform=axB.transAxes, color="gray",
                    fontsize=FONT_SCALE
                )
                print("Warning: Not enough valid data points for KPZ fit (Panel B).")
        else:
            axB.text(
                0.5, 0.5, "No neutral (s=0) data for KPZ scaling. Check 'width' values.",
                ha="center", va="center", transform=axB.transAxes, color="gray",
                fontsize=FONT_SCALE*1.2
            )
    else:
        axB.text(
            0.5, 0.5, "No data for Panel B: Interface Roughness Scaling",
            ha="center", va="center", transform=axB.transAxes, color="gray",
            fontsize=FONT_SCALE*1.2
        )

    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.legend(loc="upper left")
    axB.set_title(r"Interface Roughness Scaling (KPZ)")
    axB.set_xlabel("System Width, $L$")
    axB.set_ylabel(r"Saturated Roughness, $\langle W^2_{sat} \rangle$")
    axB.grid(True, which="both", linestyle=':', alpha=0.7)


    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"\nFigure 1 saved to {output_path}")
    plt.close(fig) # Close the main figure to free memory


if __name__ == "__main__":
    main()