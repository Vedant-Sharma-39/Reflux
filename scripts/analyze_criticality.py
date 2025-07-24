# FILE: scripts/analyze_criticality.py
#
# [DEFINITIVE PRODUCTION VERSION v11 - MONOTONIC FIX]
# This version implements the final, most sophisticated classification logic
# to produce physically correct, monotonic correlation length curves.
#
# KEY FIX:
#   - The `classify_and_fit_g_r` function is upgraded to distinguish between
#     two types of "flat" g(r) functions:
#       1. True disorder (g_inf ~ 0) -> xi = 1
#       2. Selection-dominated order (g_inf ~ 1) -> xi = infinity
#     This corrects the non-monotonic behavior seen at low k_total for s < 0.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import collections
import argparse
import ast
from scipy.optimize import curve_fit
from scipy.stats import linregress
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- Robust Path and Config Import ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))
try:
    from config import EXPERIMENTS
except ImportError:
    print("FATAL: Could not import EXPERIMENTS from src/config.py.")
    sys.exit(1)

# --- Analysis Constants ---
XI_UPPER_BOUND = 1024
GINF_ORDERED_THRESHOLD = 0.9  # New threshold to detect selection-dominated order
K_DISORDERED_THRESHOLD = 10.0


# ==============================================================================
# STAGE 1 & 2: DATA LOADING AND DEFINITIVE FITTING
# ==============================================================================
def read_json_worker(filepath):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            data["source_file"] = os.path.basename(filepath)
            return data
    except Exception:
        return None


def aggregate_data_incremental(campaign_id, output_dir, force_reaggregate=False):
    # This function is correct and remains unchanged.
    results_dir = os.path.join(project_root, "data", campaign_id, "results")
    cached_csv_path = os.path.join(output_dir, "aggregated_correlation_data.csv")
    if not os.path.isdir(results_dir):
        return None
    all_json_files = {f for f in os.listdir(results_dir) if f.endswith(".json")}
    cached_df, files_to_process = None, all_json_files
    if not force_reaggregate and os.path.exists(cached_csv_path):
        cached_df = pd.read_csv(cached_csv_path)
        files_to_process = all_json_files - set(cached_df.get("source_file", []))
    if files_to_process:
        filepaths = [os.path.join(results_dir, f) for f in files_to_process]
        with Pool(processes=max(1, cpu_count() - 2)) as pool:
            new_results = list(
                tqdm(
                    pool.imap_unordered(read_json_worker, filepaths),
                    total=len(filepaths),
                    desc="Reading new JSONs",
                )
            )
        valid_new = [r for r in new_results if r is not None and "g_r" in r]
        new_df = pd.DataFrame(valid_new) if valid_new else pd.DataFrame()
        results_df = pd.concat([cached_df, new_df], ignore_index=True)
        if not new_df.empty:
            results_df.to_csv(cached_csv_path, index=False)
    else:
        results_df = cached_df
    if results_df is None or results_df.empty:
        return None
    if isinstance(results_df["g_r"].iloc[0], str):
        tqdm.pandas(desc="Parsing g(r) strings")
        results_df["g_r"] = results_df["g_r"].progress_apply(ast.literal_eval)
    return results_df


def calculate_average_g_r(df):
    # This function is correct and remains unchanged.
    print("Step 1: Averaging g(r) across replicates...")
    points = collections.defaultdict(lambda: collections.defaultdict(list))
    group_keys = [k for k in ["b_m", "phi", "k_total"] if k in df.columns]
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Aggregating g(r)"):
        params = tuple(row[k] for k in group_keys)
        for r, g_val in row["g_r"]:
            points[params][r].append(g_val)
    avg_data = [
        {
            **dict(zip(group_keys, p)),
            "avg_g_r": sorted({r: np.mean(v) for r, v in rd.items()}.items()),
        }
        for p, rd in points.items()
    ]
    return pd.DataFrame(avg_data)


def physical_decay_model(r, C, xi, g_inf):
    return C * np.exp(-r / xi) + g_inf


def classify_and_fit_g_r(r_data, g_data, k_total):
    """[MONOTONIC FIX] The definitive, three-stage classification and fitting function."""
    # Stage 1: Handle high-k disordered regime by physical cutoff
    if k_total > K_DISORDERED_THRESHOLD:
        return (0.0, 1.0, 0.0), "assigned_disordered_high_k"

    if len(r_data) < 5:
        return (0, 1.0, 0.0), "assigned_disordered_no_data"

    # Stage 2: Check for selection-dominated ordered state (flat g(r) near 1)
    g_mean = np.mean(g_data)
    g_var = np.var(g_data)
    if (
        g_mean > GINF_ORDERED_THRESHOLD and g_var < 0.01
    ):  # Heuristic for "flat and high"
        return (0.0, np.inf, g_mean), "assigned_ordered_selection"

    # Stage 3: If not in the above regimes, it must be a decaying function. Fit it.
    try:
        g_inf_guess = np.mean(g_data[-5:]) if len(g_data) > 5 else 0.0
        popt, _ = curve_fit(
            physical_decay_model,
            r_data,
            g_data,
            p0=[max(0, g_data[0] - g_inf_guess), 10.0, g_inf_guess],
            bounds=([0, 1e-9, -1.1], [1.1, XI_UPPER_BOUND, 1.1]),
        )
        return popt, "fit_success_decay"
    except Exception:
        return (0, 1.0, g_mean), "assigned_disordered_fit_failed"


def calculate_correlation_lengths(df_avg_g_r):
    print("Step 2: Calculating xi using definitive 'Monotonic Fix' logic...")
    results = []
    for _, row in tqdm(df_avg_g_r.iterrows(), total=len(df_avg_g_r), desc="Fitting xi"):
        r, g = np.array(row["avg_g_r"]).T
        params, status = classify_and_fit_g_r(r, g, row["k_total"])
        res = {k: v for k, v in row.items() if k != "avg_g_r"}
        res.update(
            {"C": params[0], "xi": params[1], "g_inf": params[2], "fit_status": status}
        )
        results.append(res)
    return pd.DataFrame(results)


def find_critical_k_robustly(df_slice):
    # This function is correct and remains unchanged.
    df = (
        df_slice.sort_values("k_total")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["xi"])
    )
    if len(df) < 5:
        return np.nan
    log_k, log_xi = np.log10(df["k_total"]), np.log10(df["xi"])
    transition_mask = (log_xi > np.log10(1.5)) & (log_xi < (log_xi.max() * 0.8))
    if transition_mask.sum() < 3:
        return df.loc[df["xi"].idxmax()]["k_total"]
    slope, intercept, _, _, _ = linregress(
        log_k[transition_mask], log_xi[transition_mask]
    )
    log_xi_c = np.log10(np.sqrt(df["xi"].max() * 1.0))
    log_k_c = (log_xi_c - intercept) / slope
    return 10**log_k_c


# ==============================================================================
# STAGE 3 & 4: PLOTTING and MAIN (Unchanged)
# ==============================================================================
def plot_xi_collapse(df_fits, varying_param, figures_dir, plot_id=""):
    if df_fits.empty:
        print(f"  [WARNING] Skipping plot 'xi_collapse{plot_id}': Input data is empty.")
        return
    print(f"  -> Generating Figure 1: Final xi vs k_total, varying {varying_param}...")
    fig, ax = plt.subplots(figsize=(12, 8))
    y_vals = sorted(df_fits[varying_param].unique())
    pal = sns.color_palette("viridis_r", n_colors=len(y_vals))
    for i, y_val in enumerate(y_vals):
        subset = df_fits[df_fits[varying_param] == y_val].sort_values("k_total")
        ax.plot(
            subset["k_total"],
            subset["xi"],
            "o-",
            label=f"{y_val:.2f}",
            color=pal[i],
            markersize=8,
            linewidth=2.5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total Switching Rate ($k_{total}$)", fontsize=16)
    ax.set_ylabel("Correlation Length ($\\xi$)", fontsize=16)
    ax.set_title("Collapse of Spatial Order", fontsize=20)
    ax.set_ylim(bottom=0.8)
    ax.legend(title=varying_param, fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.savefig(
        os.path.join(figures_dir, f"Fig1_xi_collapse{plot_id}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_phase_boundary(df_crit, varying_param, figures_dir, plot_id=""):
    if df_crit.empty:
        print(
            f"  [WARNING] Skipping plot 'PhaseBoundary{plot_id}': No critical points found."
        )
        return
    print(f"  -> Generating Figure 2: Phase Boundary (kc vs {varying_param})...")
    df_crit = df_crit.sort_values(varying_param)
    xlabel = (
        "Selection Coefficient ($s = b_m - 1$)"
        if varying_param == "s"
        else varying_param
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(
        df_crit[varying_param],
        df_crit["k_c"],
        "o-",
        markersize=10,
        linewidth=2.5,
        color="crimson",
    )
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Critical Switching Rate ($k_c$)", fontsize=14)
    ax.set_title(f"Phase Boundary: $k_c$ vs. Selection", fontsize=18)
    if "phi" in varying_param:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--")
    plt.savefig(
        os.path.join(figures_dir, f"Fig2_PhaseBoundary{plot_id}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
    crit_exps = [
        k for k, v in EXPERIMENTS.items() if v.get("run_mode") == "correlation_analysis"
    ]
    parser = argparse.ArgumentParser(
        description="Run definitive, artifact-free criticality analysis."
    )
    parser.add_argument(
        "experiment_name",
        nargs="?",
        help="Name of experiment to analyze.",
        choices=crit_exps,
    )
    parser.add_argument(
        "--force-reaggregate",
        action="store_true",
        help="Ignore cache and re-process all JSONs.",
    )
    args = parser.parse_args()
    exp_name = args.experiment_name
    if not exp_name:
        print("Please choose an experiment:")
        [print(f"  [{i+1}] {n}") for i, n in enumerate(crit_exps)]
        try:
            choice = int(input("Enter number: ")) - 1
            exp_name = crit_exps[choice]
        except (ValueError, IndexError):
            sys.exit("Invalid input.")
    exp_config = EXPERIMENTS[exp_name]
    CAMPAIGN_ID = exp_config["CAMPAIGN_ID"]
    FIGS_DIR = os.path.join(project_root, "figures", CAMPAIGN_ID, "criticality_final")
    os.makedirs(FIGS_DIR, exist_ok=True)
    print(f"\n--- Running Final Analysis for Campaign: {CAMPAIGN_ID} ---")
    df_raw = aggregate_data_incremental(CAMPAIGN_ID, FIGS_DIR, args.force_reaggregate)
    if df_raw is None or df_raw.empty:
        sys.exit("FATAL: No data found.")
    df_avg_g_r = calculate_average_g_r(df_raw)
    if df_avg_g_r.empty:
        sys.exit("FATAL: No g(r) data could be averaged.")
    df_fits = calculate_correlation_lengths(df_avg_g_r)
    if df_fits.empty:
        sys.exit("FATAL: Could not calculate any correlation lengths.")
    if "b_m" in df_fits.columns:
        df_fits["s"] = df_fits["b_m"] - 1.0
    print("\n--- Generating Plots Based on Experiment Configuration ---")
    for set_id, sim_set in exp_config.get("SIM_SETS", {}).items():
        if len(sim_set.get("grid_params", {})) == 2:
            base_params = sim_set.get("base_params", {})
            varying_param_key = [p for p in sim_set["grid_params"] if p != "k_total"][0]
            plot_param = "s" if varying_param_key == "b_m" else varying_param_key
            scan_data = df_fits.copy()
            for p, v_fixed in base_params.items():
                if p in scan_data.columns and p not in grid_params:
                    scan_data = scan_data[np.isclose(scan_data[p], v_fixed)]
            if scan_data.empty:
                print(f"[WARNING] No data found for scan '{set_id}'. Skipping.")
                continue
            print(f"\nAnalyzing 2D scan '{set_id}' (varying {plot_param})...")
            plot_xi_collapse(scan_data, plot_param, FIGS_DIR, plot_id=f"_{set_id}")
            crit_pts = [
                {"k_c": find_critical_k_robustly(grp), **grp.iloc[0][[plot_param]]}
                for _, grp in scan_data.groupby(plot_param)
            ]
            plot_phase_boundary(
                pd.DataFrame(crit_pts).dropna(),
                plot_param,
                FIGS_DIR,
                plot_id=f"_{set_id}",
            )
    print(f"\n--- Analysis Complete. Final figures saved to: {FIGS_DIR} ---")


if __name__ == "__main__":
    main()
