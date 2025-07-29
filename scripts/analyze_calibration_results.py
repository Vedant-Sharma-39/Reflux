# FILE: scripts/analyze_boundary_dynamics.py
#
# [DEFINITIVE VERSION w/ DEBUG PLOTS]
# This version adds a debug plotting stage to visually verify the survival-
# filtered fitting process for both drift and diffusion.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, json, argparse, ast
from tqdm import tqdm
from scipy.stats import linregress
from multiprocessing import Pool, cpu_count

# --- Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))
from config import EXPERIMENTS

SURVIVAL_THRESHOLD = 0.90  # Only use data where 90% of replicates are still alive


def read_json_worker(filepath):
    """Worker for parallel JSON reading."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            data["source_file"] = os.path.basename(filepath)
            return data
    except Exception:
        return None


def aggregate_data_cached(campaign_id, analysis_dir, force_reaggregate=False):
    """Standard parallel and cached data aggregation function."""
    results_dir = os.path.join(project_root, "data", campaign_id, "results")
    cached_csv_path = os.path.join(analysis_dir, f"{campaign_id}_aggregated.csv")

    all_json_files = (
        {f for f in os.listdir(results_dir) if f.endswith(".json")}
        if os.path.isdir(results_dir)
        else set()
    )
    cached_df = None
    if not force_reaggregate and os.path.exists(cached_csv_path):
        print(f"\nLoading existing data from CSV cache: {cached_csv_path}")
        cached_df = pd.read_csv(cached_csv_path, low_memory=False)
        files_to_process = all_json_files - set(cached_df.get("source_file", []))
        print(f"Found {len(files_to_process)} new JSON files to process.")
    else:
        print("\nCache not found or --force-reaggregate used. Processing all files.")
        files_to_process = all_json_files

    if files_to_process:
        filepaths = [os.path.join(results_dir, f) for f in files_to_process]
        with Pool(processes=max(1, cpu_count() - 1)) as pool:
            new_results = list(
                tqdm(
                    pool.imap_unordered(read_json_worker, filepaths),
                    total=len(filepaths),
                )
            )
        valid_new = [r for r in new_results if r is not None]
        if valid_new:
            new_df = pd.DataFrame(valid_new)
            results_df = (
                pd.concat([cached_df, new_df], ignore_index=True)
                if cached_df is not None
                else new_df
            )
            if "trajectory" in results_df.columns:
                results_df["trajectory"] = results_df["trajectory"].astype(str)
            results_df.to_csv(cached_csv_path, index=False)
        else:
            results_df = cached_df
    else:
        results_df = cached_df

    return results_df


def analyze_trajectories_for_s(s_val, group_df):
    """
    Analyzes trajectories and returns fit results AND the data needed to plot them.
    """
    all_points = []
    num_initial_replicates = group_df["replicate_id"].nunique()
    if num_initial_replicates == 0:
        return None

    for _, row in group_df.iterrows():
        traj = row.get("trajectory")
        if pd.isna(traj):
            continue
        if isinstance(traj, str):
            try:
                traj = ast.literal_eval(traj)
            except (ValueError, SyntaxError):
                continue
        if not isinstance(traj, list):
            continue
        replicate_id = row.get("replicate_id", -1)
        for q, w in traj:
            all_points.append({"replicate_id": replicate_id, "q": q, "width": w})

    if not all_points:
        return None

    df = pd.DataFrame(all_points)
    bins = np.linspace(df["q"].min(), df["q"].max(), 50)
    df["q_bin"] = pd.cut(df["q"], bins)

    binned_stats = (
        df.groupby("q_bin", observed=True)
        .agg(
            q_mean=("q", "mean"),
            width_mean=("width", "mean"),
            width_var=("width", "var"),
            survival_counts=("replicate_id", "nunique"),
        )
        .dropna()
    )

    binned_stats["survival_prob"] = (
        binned_stats["survival_counts"] / num_initial_replicates
    )
    reliable_data = binned_stats[binned_stats["survival_prob"] >= SURVIVAL_THRESHOLD]

    if len(reliable_data) < 5:
        return None

    slope_v, intercept_v, r_v, _, _ = linregress(
        reliable_data["q_mean"], reliable_data["width_mean"]
    )
    v_drift = slope_v / 2.0
    slope_d, intercept_d, r_d, _, _ = linregress(
        reliable_data["q_mean"], reliable_data["width_var"]
    )
    d_eff = slope_d / 2.0

    return {
        "s": s_val,
        "v_drift": v_drift,
        "d_eff": d_eff,
        "r_sq_v": r_v**2,
        "r_sq_d": r_d**2,
        "plot_data": {
            "binned_stats": binned_stats.to_dict("list"),
            "reliable_data": reliable_data.to_dict("list"),
            "slope_v": slope_v,
            "intercept_v": intercept_v,
            "slope_d": slope_d,
            "intercept_d": intercept_d,
            "survival_threshold": SURVIVAL_THRESHOLD,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze boundary dynamics with survival filtering and debug plots."
    )
    parser.add_argument(
        "experiment_name", default="calibration_boundary_dynamics_v1", nargs="?"
    )
    parser.add_argument(
        "--force-reaggregate",
        action="store_true",
        help="Force re-aggregation of raw JSON data.",
    )
    args = parser.parse_args()

    config = EXPERIMENTS[args.experiment_name]
    CAMPAIGN_ID = config["CAMPAIGN_ID"]
    ANALYSIS_DIR = os.path.join(project_root, "data", CAMPAIGN_ID, "analysis")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    FIGS_DIR = os.path.join(project_root, "figures", CAMPAIGN_ID)
    os.makedirs(FIGS_DIR, exist_ok=True)
    DEBUG_FIGS_DIR = os.path.join(FIGS_DIR, "debug_fits")
    os.makedirs(DEBUG_FIGS_DIR, exist_ok=True)

    df_raw = aggregate_data_cached(CAMPAIGN_ID, ANALYSIS_DIR, args.force_reaggregate)
    if df_raw is None or df_raw.empty:
        sys.exit("FATAL: No data found.")

    df_raw["s"] = df_raw["b_m"] - 1.0
    analysis_tasks = [(s_val, group) for s_val, group in df_raw.groupby("s")]

    with Pool(processes=max(1, cpu_count() - 2)) as pool:
        results = list(
            tqdm(
                pool.starmap(analyze_trajectories_for_s, analysis_tasks),
                total=len(analysis_tasks),
                desc="Analyzing Selection Strengths",
            )
        )

    valid_results = [res for res in results if res is not None]
    if not valid_results:
        sys.exit("Analysis complete, but no valid results were generated.")

    # --- [NEW] Generate Debug Plots ---
    print("\n--- Generating debug plots ---")
    for res in tqdm(valid_results, desc="Plotting fits"):
        s, p_data = res["s"], res["plot_data"]
        binned_df = pd.DataFrame(p_data["binned_stats"])
        reliable_df = pd.DataFrame(p_data["reliable_data"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
        fig.suptitle(f"Boundary Dynamics Fit for s = {s:.3f}", fontsize=16)

        # Panel A: Drift (Mean Width)
        ax1.plot(
            binned_df["q_mean"],
            binned_df["width_mean"],
            ".",
            ms=5,
            alpha=0.4,
            color="steelblue",
            label="All Binned Data",
        )
        ax1.plot(
            reliable_df["q_mean"],
            reliable_df["width_mean"],
            ".",
            ms=7,
            color="orangered",
            label=f"Fit Data (> {p_data['survival_threshold']*100:.0f}% survival)",
        )
        q_plot = np.array(binned_df["q_mean"])
        ax1.plot(
            q_plot,
            p_data["intercept_v"] + p_data["slope_v"] * q_plot,
            "r-",
            lw=2,
            label=f"Fit (v_drift={res['v_drift']:.3f})",
        )
        ax1.set(
            xlabel="Mean Front Position (q)",
            ylabel="⟨Sector Width⟩",
            title="Drift Velocity Fit",
        )
        ax1.legend()
        ax1.grid(True, ls="--")

        # Panel B: Diffusion (Width Variance)
        ax2.plot(
            binned_df["q_mean"],
            binned_df["width_var"],
            ".",
            ms=5,
            alpha=0.4,
            color="seagreen",
            label="All Binned Data",
        )
        ax2.plot(
            reliable_df["q_mean"],
            reliable_df["width_var"],
            ".",
            ms=7,
            color="purple",
            label=f"Fit Data (> {p_data['survival_threshold']*100:.0f}% survival)",
        )
        ax2.plot(
            q_plot,
            p_data["intercept_d"] + p_data["slope_d"] * q_plot,
            "m-",
            lw=2,
            label=f"Fit (D_eff={res['d_eff']:.3f})",
        )
        ax2.set(
            xlabel="Mean Front Position (q)",
            ylabel="Var(Sector Width)",
            title="Effective Diffusion Fit",
        )
        ax2.legend()
        ax2.grid(True, ls="--")

        plt.savefig(os.path.join(DEBUG_FIGS_DIR, f"debug_fit_s_{s:.4f}.png"), dpi=150)
        plt.close(fig)

    # --- Final Aggregation and Summary Plot ---
    df_final = pd.DataFrame(
        [
            {"s": r["s"], "v_drift": r["v_drift"], "d_eff": r["d_eff"]}
            for r in valid_results
        ]
    )
    output_path = os.path.join(ANALYSIS_DIR, "boundary_dynamics_summary.csv")
    df_final.to_csv(output_path, index=False)
    print(f"\nSaved boundary dynamics summary to: {output_path}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    fig.suptitle("Boundary Dynamics Calibration (Survival-Filtered)", fontsize=20)
    axes[0].plot(df_final["s"], df_final["v_drift"], "o-", c="navy", label="Measured")
    axes[0].plot(
        df_final["s"], df_final["s"], "--", c="gray", alpha=0.8, label="Theory (v=s)"
    )
    axes[0].set(
        xlabel="Selection (s)",
        ylabel="Effective Drift ($v_{drift}$)",
        title="A. Sector Drift Velocity",
    )
    axes[0].legend()
    axes[1].plot(df_final["s"], df_final["d_eff"], "o-", c="darkgreen")
    axes[1].set(
        xlabel="Selection (s)",
        ylabel="Effective Diffusion ($D_{eff}$)",
        title="B. Sector Boundary Diffusion",
    )
    axes[1].axhline(0, color="k", lw=0.5, ls="-")
    for ax in axes:
        ax.grid(True, ls="--")
        ax.axvline(0, color="k", lw=0.5, ls="-")
    plot_path = os.path.join(FIGS_DIR, "Fig_Boundary_Dynamics_Summary.png")
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved summary plot to: {plot_path}")
    print(f"Saved debug plots to: {DEBUG_FIGS_DIR}")


if __name__ == "__main__":
    main()
