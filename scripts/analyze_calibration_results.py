# scripts/analyze_calibration_results.py
# [UPGRADED v5] Fixes BrokenPipeError by separating computation from I/O.
# - Interactive menu for selecting experiment if no command-line argument is given.
# - Plotting is REMOVED from the parallel worker to ensure stability.
# - All other features (caching, parallel analysis) are retained.

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
    from config import EXPERIMENTS
except ImportError:
    print("FATAL: Could not import EXPERIMENTS from src/config.py.")
    sys.exit(1)


# --- Multiprocessing Worker for JSON Reading ---
def read_json_file(filepath):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            data["source_file"] = os.path.basename(filepath)
            return data
    except Exception:
        return None


# --- [MODIFIED] Multiprocessing Worker for Parallel Analysis (NO PLOTTING) ---
def analyze_bm_group(args_tuple):
    """
    Analyzes all data for a single b_m value. Returns a dictionary of results
    and data needed for plotting, but DOES NOT create the plot itself.
    """
    b_m, group_df, _ = args_tuple  # output_dir is no longer needed here
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
        slope, intercept, _, _, _ = linregress(q_fit, w_fit)
        v_drift = slope / 2.0

        # Return all data needed for plotting later in the main process
        return {
            "b_m": b_m,
            "v_drift": v_drift,
            "plot_data": {
                "avg_trajectory": avg_trajectory.to_dict("list"),
                "q_fit": q_fit.tolist(),
                "w_fit": w_fit.tolist(),
                "slope": slope,
                "intercept": intercept,
                "survival_threshold": survival_threshold,
            },
        }
    except Exception as e:
        print(f"Error processing b_m={b_m}: {e}")
        return None


def main():
    # --- [MODIFIED] Argument Parsing with Interactive Fallback ---
    calibration_experiments = [
        k for k, v in EXPERIMENTS.items() if v.get("run_mode") == "calibration"
    ]
    if not calibration_experiments:
        print(
            "FATAL: No experiments with `run_mode: 'calibration'` found in src/config.py."
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Analyze calibration results for a specified campaign."
    )
    parser.add_argument(
        "experiment_name",
        nargs="?",  # Make the argument optional
        help="The name of the calibration experiment to analyze.",
        choices=calibration_experiments,
    )
    parser.add_argument(
        "--force-reaggregate",
        action="store_true",
        help="Ignore cache and re-aggregate all raw JSON files.",
    )
    args = parser.parse_args()

    experiment_name = args.experiment_name
    if not experiment_name:
        print("Please choose a calibration experiment to analyze:")
        for i, name in enumerate(calibration_experiments):
            print(f"  [{i+1}] {name}")
        try:
            choice = int(input("Enter number: ")) - 1
            if 0 <= choice < len(calibration_experiments):
                experiment_name = calibration_experiments[choice]
            else:
                print("Invalid choice.")
                sys.exit(1)
        except (ValueError, IndexError):
            print("Invalid input.")
            sys.exit(1)

    experiment_config = EXPERIMENTS[experiment_name]
    CAMPAIGN_ID = experiment_config["CAMPAIGN_ID"]

    print(f"\n--- Analyzing Calibration Results for Campaign: {CAMPAIGN_ID} ---")

    results_dir = os.path.join(project_root, "data", CAMPAIGN_ID, "results")
    output_dir = os.path.join(project_root, "data", CAMPAIGN_ID, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    cached_df_path = os.path.join(output_dir, "aggregated_raw_data.csv")

    # --- Incremental Caching Logic (Unchanged) ---
    all_json_files = {f for f in os.listdir(results_dir) if f.endswith(".json")}
    cached_df = None
    if not args.force_reaggregate and os.path.exists(cached_df_path):
        print(f"\nLoading existing data from CSV cache: {cached_df_path}")
        cached_df = pd.read_csv(cached_df_path)
        cached_df["trajectory"] = cached_df["trajectory"].apply(ast.literal_eval)
        files_to_process = all_json_files - set(cached_df.get("source_file", []))
        print(f"Found {len(files_to_process)} new JSON files to process.")
    else:
        print("\nCache not found or --force-reaggregate used. Processing all files.")
        files_to_process = all_json_files

    if files_to_process:
        # ... (Aggregation logic is unchanged and correct) ...
        print(f"Aggregating {len(files_to_process)} JSON files...")
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
            results_df = cached_df
    else:
        results_df = cached_df

    if results_df is None or results_df.empty:
        print("No data available for analysis. Exiting.")
        return

    # --- Parallel Analysis Section ---
    print(
        f"\n--- Performing Parallel Analysis for {results_df['b_m'].nunique()} b_m values ---"
    )
    analysis_tasks = [
        (b_m, group, output_dir) for b_m, group in results_df.groupby("b_m")
    ]

    with Pool(processes=max(1, cpu_count() - 2)) as pool:
        results_iterator = pool.imap_unordered(analyze_bm_group, analysis_tasks)
        analysis_results = list(tqdm(results_iterator, total=len(analysis_tasks)))

    valid_results = [res for res in analysis_results if res is not None]
    if not valid_results:
        print("\nCould not calculate any drift velocities.")
        return

    # --- [NEW] Serial Plotting of Individual Fits ---
    print("\n--- Generating individual fit plots ---")
    for res in tqdm(valid_results, desc="Plotting fits"):
        b_m = res["b_m"]
        p_data = res["plot_data"]
        avg_traj_df = pd.DataFrame(p_data["avg_trajectory"])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            avg_traj_df["q"],
            avg_traj_df["width"],
            ".",
            markersize=5,
            alpha=0.5,
            color="steelblue",
            label="Averaged Data",
        )
        ax.plot(
            p_data["q_fit"],
            p_data["w_fit"],
            ".",
            markersize=6,
            color="orangered",
            label=f"Data for Fit (> {p_data['survival_threshold']*100:.0f}% survival)",
        )
        q_plot_extended = np.array(avg_traj_df["q"])
        ax.plot(
            q_plot_extended,
            p_data["intercept"] + p_data["slope"] * q_plot_extended,
            "r-",
            linewidth=2,
            label=f"Fit (slope={p_data['slope']:.3f})",
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

    # --- Final Aggregation and Plotting (Unchanged) ---
    calib_df = pd.DataFrame(
        [{"b_m": r["b_m"], "v_drift": r["v_drift"]} for r in valid_results]
    )
    calib_df["s"] = calib_df["b_m"] - 1.0
    calib_df = calib_df.sort_values(by="s", ignore_index=True)

    output_csv = os.path.join(output_dir, "calibration_curve_data.csv")
    calib_df.to_csv(output_csv, index=False)
    print(f"\nFinal calibration data saved to {output_csv}")

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
1
