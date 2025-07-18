# scripts/analyze_calibration_results.py
# [UPGRADED v3] Aggregates raw trajectory data and performs a robust analysis.
# - Caches aggregation to CSV, with INCREMENTAL updates for new files.
# - PARALLELIZES the analysis and fitting for each b_m value.

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import re
import argparse
import ast

# --- Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))
try:
    from config_calibration import CAMPAIGN_ID
except ImportError:
    print("Error: Could not import from src/config_calibration.py.")
    sys.exit(1)


# --- Multiprocessing Worker for JSON Reading ---
def read_json_file(filepath):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            data["source_file"] = os.path.basename(filepath)  # Add source for tracking
            return data
    except:
        return None


# --- [NEW] Multiprocessing Worker for Parallel Analysis ---
def analyze_bm_group(args_tuple):
    """
    Analyzes all data for a single b_m value. Designed to be run in parallel.
    Takes a tuple to be compatible with pool.imap.
    """
    b_m, group_df, output_dir = args_tuple
    try:
        num_initial_replicates = group_df["replicate_id"].nunique()
        if num_initial_replicates == 0:
            return None

        all_points_data = [
            {"replicate_id": row["replicate_id"], "q": q, "width": w}
            for _, row in group_df.iterrows()
            for q, w in row["trajectory"]
        ]
        if not all_points_data:
            return None

        binned_data = pd.DataFrame(all_points_data)
        max_q = binned_data["q"].max()
        bins = np.linspace(0, max_q, num=100)
        binned_data["q_bin"] = pd.cut(binned_data["q"], bins, right=False)

        avg_trajectory = binned_data.groupby("q_bin", observed=False)[
            ["q", "width"]
        ].mean()
        survival_counts = binned_data.groupby("q_bin", observed=False)[
            "replicate_id"
        ].nunique()
        avg_trajectory["survival_prob"] = survival_counts / num_initial_replicates
        avg_trajectory.dropna(inplace=True)

        survival_threshold = 0.90
        reliable_data = avg_trajectory[
            avg_trajectory["survival_prob"] >= survival_threshold
        ]

        if len(reliable_data) < 10:
            return None

        q_fit, w_fit = reliable_data["q"].values, reliable_data["width"].values
        slope, intercept, r_val, _, _ = linregress(q_fit, w_fit)

        v_drift = slope / 2.0

        # --- Create and save the plot inside the worker ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            avg_trajectory["q"],
            avg_trajectory["width"],
            ".",
            markersize=5,
            alpha=0.5,
            color="steelblue",
            label="Averaged Data (all survivors)",
        )
        ax.plot(
            q_fit,
            w_fit,
            ".",
            markersize=6,
            color="orangered",
            label=f"Data for Fit (> {survival_threshold*100:.0f}% survival)",
        )
        q_plot_extended = avg_trajectory["q"].values
        ax.plot(
            q_plot_extended,
            intercept + slope * q_plot_extended,
            "r-",
            linewidth=2,
            label=f"Fit (slope={slope:.3f})",
        )
        ax.set_title(f"Avg. Sector Shrinkage for $b_m = {b_m}$")
        ax.set_xlabel("Mean Front Position ($q$)")
        ax.set_ylabel("⟨Sector Width⟩")
        ax.legend()
        ax.grid(True, linestyle="--")
        plt.savefig(
            os.path.join(output_dir, f"robust_shrinkage_bm_{b_m:.3f}.png"), dpi=150
        )
        plt.close(fig)

        return {"b_m": b_m, "v_drift": v_drift}
    except Exception as e:
        print(f"Error processing b_m={b_m}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description=f"Analyze calibration results for campaign '{CAMPAIGN_ID}'."
    )
    parser.add_argument(
        "--force-reaggregate",
        action="store_true",
        help="Ignore cache and re-aggregate all raw JSON files.",
    )
    args = parser.parse_args()

    print(f"--- Analyzing Calibration Results for Campaign: {CAMPAIGN_ID} ---")
    results_dir = os.path.join(project_root, "data", CAMPAIGN_ID, "results")
    output_dir = os.path.join(project_root, "data", CAMPAIGN_ID, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    cached_df_path = os.path.join(output_dir, "aggregated_raw_data.csv")

    # --- [MODIFIED] Incremental Caching Logic ---
    all_json_files = {f for f in os.listdir(results_dir) if f.endswith(".json")}
    cached_df = None
    files_to_process = all_json_files

    if not args.force_reaggregate and os.path.exists(cached_df_path):
        print(f"\nLoading existing data from CSV cache: {cached_df_path}")
        cached_df = pd.read_csv(cached_df_path)
        cached_df["trajectory"] = cached_df["trajectory"].apply(ast.literal_eval)

        processed_files = set(cached_df["source_file"])
        files_to_process = all_json_files - processed_files
        print(f"Found {len(files_to_process)} new JSON files to process.")
    else:
        print("\nCache not found or --force-reaggregate used. Processing all files.")

    if files_to_process:
        print(
            f"Aggregating {len(files_to_process)} JSON files (this may take a while)..."
        )
        filepaths_to_process = [os.path.join(results_dir, f) for f in files_to_process]

        with Pool(processes=max(1, cpu_count() - 1)) as pool:
            new_results = list(
                tqdm(
                    pool.imap_unordered(read_json_file, filepaths_to_process),
                    total=len(filepaths_to_process),
                )
            )

        valid_new_results = []
        for r in new_results:
            if r is None or not r.get("trajectory"):
                continue
            if "replicate_id" not in r:
                match = re.search(r"_rep(\d+)", r.get("task_id", ""))
                if match:
                    r["replicate_id"] = int(match.group(1))
                else:
                    continue
            valid_new_results.append(r)

        if valid_new_results:
            new_df = pd.DataFrame(valid_new_results)
            # Combine new data with existing cached data
            results_df = (
                pd.concat([cached_df, new_df], ignore_index=True)
                if cached_df is not None
                else new_df
            )

            print(
                f"Saving updated cache with {len(results_df)} total records to: {cached_df_path}"
            )
            results_df.to_csv(cached_df_path, index=False)
        else:
            print("No valid new data found. Using existing cache.")
            results_df = cached_df
    else:
        print("No new files to process. Using existing cache.")
        results_df = cached_df

    if results_df is None or results_df.empty:
        print("No data available for analysis. Exiting.")
        return

    # --- [MODIFIED] Parallel Analysis Section ---
    print(
        f"\n--- Performing Parallel Analysis for {results_df['b_m'].nunique()} b_m values ---"
    )

    # Prepare arguments for each parallel task
    analysis_tasks = [
        (b_m, group, output_dir) for b_m, group in results_df.groupby("b_m")
    ]

    drift_velocity_results = []
    with Pool(processes=max(1, cpu_count() - 2)) as pool:
        # Use tqdm to show progress on the analysis step
        results_iterator = pool.imap_unordered(analyze_bm_group, analysis_tasks)
        drift_velocity_results = list(tqdm(results_iterator, total=len(analysis_tasks)))

    # Filter out any failed tasks
    valid_dv_results = [res for res in drift_velocity_results if res is not None]
    if not valid_dv_results:
        print("\nCould not calculate any drift velocities. No final plot generated.")
        return

    # --- Final Aggregation and Plotting (in main process) ---
    calib_df = pd.DataFrame(valid_dv_results)

    # 2. Create the new 's' column for the selection coefficient.
    calib_df["s"] = calib_df["b_m"] - 1.0

    # 3. Now, sort the DataFrame by the 's' column. No need for a complex key.
    calib_df = calib_df.sort_values(by="s", ignore_index=True)

    output_csv = os.path.join(output_dir, "calibration_curve_data.csv")
    calib_df.to_csv(output_csv, index=False)
    print(f"\nFinal calibration data saved to {output_csv}")

    # The plotting section also needs a small adjustment to use the new 's' column
    plt.figure(figsize=(8, 6))
    plt.plot(
        calib_df["s"],
        calib_df["v_drift"],
        "o-",
        color="navy",
        label="Measured $v_{drift}$",
    )
    plt.plot(
        calib_df["s"],
        calib_df["s"],
        "--",
        color="gray",
        alpha=0.8,
        label="Theory ($v_{drift} = s$)",
    )

    plt.title("Calibration Curve: Effective Drift vs. Selection")
    plt.xlabel("Selection Coefficient ($s = b_m - 1$)")
    plt.ylabel("Effective Drift Velocity ($v_{drift}$)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.legend()
    final_plot_path = os.path.join(output_dir, "FINAL_CALIBRATION_CURVE.png")
    plt.savefig(final_plot_path, dpi=200, bbox_inches="tight")
    print(f"Final calibration curve plot saved to {final_plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
