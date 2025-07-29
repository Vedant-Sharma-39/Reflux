# FILE: scripts/analyze_irreversible_criticality.py
# [DEFINITIVE v3 - PEAK FITTING]
# This version implements the correct analysis for the irreversible system.
# It defines k_c as the peak of the interface density, found by fitting a
# parabola to the data points around the peak.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, json, argparse
from tqdm import tqdm
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count

# --- Robust Path Setup ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from config import EXPERIMENTS
except (NameError, ImportError) as e:
    print(f"FATAL: Could not import configuration. Error: {e}", file=sys.stderr)
    sys.exit(1)

plt.style.use("seaborn-v0_8-whitegrid")


def read_json_worker(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception:
        return None


def aggregate_data_cached(campaign_id, analysis_dir, force_reaggregate=False):
    results_dir = os.path.join(project_root, "data", campaign_id, "results")
    cached_csv_path = os.path.join(analysis_dir, f"{campaign_id}_aggregated.csv")
    if not force_reaggregate and os.path.exists(cached_csv_path):
        return pd.read_csv(cached_csv_path)
    filepaths = (
        [
            os.path.join(results_dir, f)
            for f in os.listdir(results_dir)
            if f.endswith(".json")
        ]
        if os.path.isdir(results_dir)
        else []
    )
    if not filepaths:
        return None
    with Pool(processes=max(1, cpu_count() - 2)) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(read_json_worker, filepaths),
                total=len(filepaths),
                desc="Reading JSONs",
            )
        )
    df = pd.DataFrame([r for r in results if r is not None])
    df.to_csv(cached_csv_path, index=False)
    return df


def quadratic_model(log_k, a, b, c):
    """A simple parabola for fitting peaks in log-space."""
    return a * log_k**2 + b * log_k + c


def find_kc_by_peak_fit(s_val, group_df):
    """Finds k_c by fitting a parabola to the peak of the interface density."""
    data = group_df.sort_values("k_total")
    k_values, density_values = (
        data["k_total"].values,
        data["avg_interface_density"].values,
    )
    if len(k_values) < 5:
        return None

    peak_idx = np.argmax(density_values)
    window_size = 5  # Use a wider window for more stable fits
    start_idx = max(0, peak_idx - window_size)
    end_idx = min(len(k_values), peak_idx + window_size + 1)

    k_fit, density_fit = k_values[start_idx:end_idx], density_values[start_idx:end_idx]
    if len(k_fit) < 3:
        return {"s": s_val, "k_c": k_values[peak_idx], "plot_data": None}

    log_k_fit = np.log(k_fit)
    fit_params = None
    try:
        # A parabola must curve downwards, so 'a' must be negative.
        popt, pcov = curve_fit(
            quadratic_model,
            log_k_fit,
            density_fit,
            bounds=([-np.inf, -np.inf, -np.inf], [0, np.inf, np.inf]),
        )
        if np.isinf(pcov).any():
            raise RuntimeError("Unreliable fit")

        log_kc = -popt[1] / (2 * popt[0])
        if min(log_k_fit) < log_kc < max(log_k_fit):
            k_c, fit_params = np.exp(log_kc), popt
        else:
            k_c = k_values[peak_idx]
    except (RuntimeError, ValueError):
        k_c = k_values[peak_idx]

    return {
        "s": s_val,
        "k_c": k_c,
        "plot_data": {
            "k_values": k_values.tolist(),
            "density_values": density_values.tolist(),
            "fit_params": fit_params.tolist() if fit_params is not None else None,
            "k_fit_window": k_fit.tolist(),
            "density_fit_window": density_fit.tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze irreversible criticality by peak fitting."
    )
    parser.add_argument(
        "experiment_name", default="criticality_irreversible_v3_focused", nargs="?"
    )
    args = parser.parse_args()

    config = EXPERIMENTS[args.experiment_name]
    CAMPAIGN_ID = config["CAMPAIGN_ID"]
    ANALYSIS_DIR = os.path.join(project_root, "data", CAMPAIGN_ID, "analysis")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    FIGS_DIR = os.path.join(project_root, "figures", CAMPAIGN_ID)
    os.makedirs(FIGS_DIR, exist_ok=True)
    DEBUG_FIGS_DIR = os.path.join(FIGS_DIR, "debug_peak_fits")
    os.makedirs(DEBUG_FIGS_DIR, exist_ok=True)

    df_raw = aggregate_data_cached(CAMPAIGN_ID, ANALYSIS_DIR)
    if df_raw is None or df_raw.empty:
        sys.exit("FATAL: No data found.")

    df_raw["s"] = df_raw["b_m"] - 1.0
    df_avg = (
        df_raw.groupby(["s", "k_total"])
        .agg(avg_interface_density=("avg_interface_density", "mean"))
        .reset_index()
    )

    analysis_tasks = [(s_val, grp) for s_val, grp in df_avg.groupby("s")]
    with Pool(processes=max(1, cpu_count() - 2)) as pool:
        results = list(
            tqdm(
                pool.starmap(find_kc_by_peak_fit, analysis_tasks),
                total=len(analysis_tasks),
                desc="Fitting peaks",
            )
        )

    valid_results = [res for res in results if res is not None]
    df_crit = (
        pd.DataFrame([{"s": r["s"], "k_c": r["k_c"]} for r in valid_results])
        .dropna()
        .sort_values("s")
    )
    df_crit.to_csv(
        os.path.join(ANALYSIS_DIR, "irreversible_criticality_summary.csv"), index=False
    )

    for res in tqdm(valid_results, desc="Plotting fits"):
        s, p_data, k_c = res["s"], res["plot_data"], res["k_c"]
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(
            p_data["k_values"], p_data["density_values"], "o", label="Simulated Data"
        )
        if p_data.get("fit_params") is not None:
            ax.plot(
                p_data["k_fit_window"],
                p_data["density_fit_window"],
                "o",
                c="orange",
                ms=10,
                label="Data for Fit",
            )
            k_smooth = np.logspace(
                np.log10(min(p_data["k_fit_window"])),
                np.log10(max(p_data["k_fit_window"])),
                100,
            )
            ax.plot(
                k_smooth,
                quadratic_model(np.log(k_smooth), *p_data["fit_params"]),
                "r-",
                label="Parabolic Fit",
            )
        ax.axvline(k_c, color="k", ls="--", label=f"$k_c \\approx {k_c:.3f}$")
        ax.set(
            xscale="log",
            xlabel="Total Switching Rate ($k_{total}$)",
            ylabel="Average Interface Density",
            title=f"Irreversible Transition Peak for s = {s:.3f}",
        )
        ax.legend()
        plt.savefig(os.path.join(DEBUG_FIGS_DIR, f"debug_fit_s_{s:.4f}.png"), dpi=150)
        plt.close(fig)

    plt.figure(figsize=(10, 7))
    plt.plot(df_crit["s"], df_crit["k_c"], "o-", c="blue", markersize=8)
    plt.yscale("log")
    plt.title("Phase Boundary for Irreversible Switching ($\\phi=-1$)")
    plt.xlabel("Selection (s)")
    plt.ylabel("Critical Switching Rate ($k_c$)")
    plt.savefig(os.path.join(FIGS_DIR, "Fig_Irreversible_Phase_Boundary.png"), dpi=300)

    print(
        f"\nAnalysis complete. Summary and plots saved in {ANALYSIS_DIR} and {FIGS_DIR}"
    )


if __name__ == "__main__":
    main()
