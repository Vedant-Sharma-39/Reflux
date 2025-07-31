# FILE: scripts/analyze_phase_boundaries.py
#
# A definitive analysis script for the 'phase1_find_kc_vs_s_coarse' campaign.
# It uses a multi-model "classify-then-fit" strategy to robustly calculate k_c
# for each (s, phi) slice in the data.
#
# The final output is a single, comparative plot showing the k_c vs. s phase
# boundary for each simulated value of phi. It also supports generating
# detailed debug plots for every data slice to validate the fitting process.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    sys.exit(f"FATAL: Could not import configuration. Error: {e}")

plt.style.use("seaborn-v0_8-whitegrid")


# --- Model & Helper Functions ---
def quadratic_model(log_k, a, b, c):
    return a * log_k**2 + b * log_k + c


def sigmoid_model(log_k, A, B, log_kc, n):
    return A / (1 + np.exp(-n * (log_k - log_kc))) + B


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


def find_kc_for_slice(params, group_df):
    """[PARALLEL WORKER] Classifies data shape and applies the correct model."""
    s_val, phi_val = params
    data = group_df.sort_values("k_total")
    k_vals, density_vals = data["k_total"].values, data["avg_interface_density"].values
    if len(k_vals) < 5:
        return None

    max_density, min_density = np.max(density_vals), np.min(density_vals)
    density_range = max_density - min_density
    peak_idx = np.argmax(density_vals)

    fit_type = "none"
    if density_range < 0.01 and max_density < 0.02:
        fit_type = "flat"
    elif (
        peak_idx > 1
        and peak_idx < len(density_vals) - 2
        and max_density > 1.5 * density_vals[-1]
    ):
        fit_type = "peak"
    else:
        fit_type = "sigmoid"

    k_c, fit_params = np.nan, None
    plot_data = {"k_values": k_vals.tolist(), "density_values": density_vals.tolist()}

    if fit_type == "peak":
        window = 5
        start, end = max(0, peak_idx - window), min(len(k_vals), peak_idx + window + 1)
        k_fit_win, d_fit_win = k_vals[start:end], density_vals[start:end]
        k_c, fit_params = k_vals[peak_idx], None
        if len(k_fit_win) >= 3:
            try:
                valid_mask = (k_fit_win > 0) & (d_fit_win > 0)
                log_k_fit, d_fit_final = (
                    np.log(k_fit_win[valid_mask]),
                    d_fit_win[valid_mask],
                )
                if len(log_k_fit) < 3:
                    raise ValueError()
                popt, _ = curve_fit(
                    quadratic_model,
                    log_k_fit,
                    d_fit_final,
                    bounds=([-np.inf, -np.inf, -np.inf], [0, np.inf, np.inf]),
                )
                log_kc = -popt[1] / (2 * popt[0])
                if min(log_k_fit) < log_kc < max(log_k_fit):
                    k_c, fit_params = np.exp(log_kc), popt
                plot_data.update(
                    {
                        "k_fit_window": k_fit_win.tolist(),
                        "density_fit_window": d_fit_win.tolist(),
                    }
                )
            except (RuntimeError, ValueError):
                pass
    elif fit_type == "sigmoid":
        try:
            log_k_vals = np.log(k_vals[k_vals > 0])
            density_fit_vals = density_vals[k_vals > 0]
            if len(log_k_vals) < 4:
                raise ValueError()
            p0 = [density_range, min_density, np.log(np.median(k_vals)), 1.0]
            popt, _ = curve_fit(
                sigmoid_model, log_k_vals, density_fit_vals, p0=p0, maxfev=10000
            )
            k_c, fit_params = np.exp(popt[2]), popt
        except (RuntimeError, ValueError):
            half_max = min_density + density_range / 2.0
            k_c = np.interp(half_max, density_vals, k_vals)
            fit_type = "sigmoid_interp"

    plot_data.update(
        {
            "fit_params": fit_params.tolist() if fit_params is not None else None,
            "fit_type": fit_type,
        }
    )
    return {"s": s_val, "phi": phi_val, "k_c": k_c, "plot_data": plot_data}


# FILE: scripts/analyze_phase_boundaries.py

# ... (keep all the other functions like main, find_kc_for_slice, etc. the same) ...


# --- [MODIFIED] Debug Plotting with the "Perfect" Semilog Scale ---
def generate_debug_fit_plot(result, debug_dir):
    s, phi, k_c, p_data = result["s"], result["phi"], result["k_c"], result["plot_data"]
    fit_type = p_data.get("fit_type", "none")

    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title(f"Transition for s = {s:.3f}, $\\phi$ = {phi:.2f}", fontsize=18)

    # --- Plot Data and Fits ---
    ax.plot(
        p_data["k_values"], p_data["density_values"], "o", ms=8, label="Simulated Data"
    )

    fit_label = None
    if fit_type == "peak" and p_data.get("fit_params"):
        # Plot data used for the fit with a different style
        ax.plot(
            p_data["k_fit_window"],
            p_data["density_fit_window"],
            "o",
            c="orange",
            ms=12,
            label="Data for Fit",
        )
        # Generate a smooth curve for the fit
        k_smooth = np.logspace(
            np.log10(min(p_data["k_fit_window"])),
            np.log10(max(p_data["k_fit_window"])),
            200,
        )
        ax.plot(
            k_smooth,
            quadratic_model(np.log(k_smooth), *p_data["fit_params"]),
            "r-",
            lw=3,
        )
        fit_label = "Parabolic Fit"
    elif fit_type == "sigmoid" and p_data.get("fit_params"):
        k_smooth = np.logspace(
            np.log10(min(p_data["k_values"]) * 0.5),
            np.log10(max(p_data["k_values"]) * 2),
            200,
        )
        ax.plot(
            k_smooth,
            hybrid_sigmoid_model(np.log(k_smooth), *p_data["fit_params"]),
            "r-",
            lw=3,
        )
        fit_label = "Sigmoid Fit"

    # --- Set the "Perfect" Scales and Labels ---
    ax.set_xscale("log")
    ax.set_yscale("linear")  # Use linear scale for the y-axis

    # Set axis limits to give some padding, ensuring 0 is visible
    min_y, max_y = np.min(p_data["density_values"]), np.max(p_data["density_values"])
    ax.set_ylim(min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y))

    ax.set_xlabel("Total Switching Rate ($k_{total}$)", fontsize=14)
    ax.set_ylabel("Average Interface Density", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)

    # --- Build Legend ---
    handles, labels = ax.get_legend_handles_labels()
    if fit_label:
        handles.append(plt.Line2D([0], [0], color="r", lw=3))
        labels.append(fit_label)
    if not np.isnan(k_c):
        handles.append(plt.Line2D([0], [0], color="k", lw=2.5, ls="--"))
        labels.append(f"$k_c \\approx {k_c:.3f}$")
        ax.axvline(k_c, color="k", ls="--", lw=2.5)

    ax.legend(handles, labels, fontsize=14)
    ax.grid(True, which="both", ls="--", color="#cccccc")

    # --- Save Figure ---
    s_str = f"s_{s:.3f}".replace(".", "p").replace("-", "neg")
    phi_str = f"phi_{phi:.2f}".replace(".", "p").replace("-", "neg")
    filename = f"debug_parametric_{fit_type}_{s_str}_{phi_str}.png"
    plt.savefig(os.path.join(debug_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Analyze coarse-grained scans to find phase boundaries."
    )
    parser.add_argument(
        "experiment_name", default="phase1_find_kc_vs_s_coarse", nargs="?"
    )
    parser.add_argument(
        "--generate-debug-plots",
        action="store_true",
        help="Generate detailed fit plots for each (s, phi) slice.",
    )
    args = parser.parse_args()

    config = EXPERIMENTS[args.experiment_name]
    CAMPAIGN_ID = config["CAMPAIGN_ID"]
    ANALYSIS_DIR = os.path.join(project_root, "data", CAMPAIGN_ID, "analysis")
    FIGS_DIR = os.path.join(project_root, "figures", CAMPAIGN_ID)
    DEBUG_DIR = os.path.join(FIGS_DIR, "debug_fits_classified")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR, exist_ok=True)
    if args.generate_debug_plots:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        print(f"Debug plots will be saved to: {DEBUG_DIR}")

    df_raw = aggregate_data_cached(CAMPAIGN_ID, ANALYSIS_DIR)
    if df_raw is None or df_raw.empty:
        sys.exit("FATAL: No data found.")

    df_raw["s"] = df_raw["b_m"] - 1.0
    df_avg = (
        df_raw.groupby(["s", "phi", "k_total"])
        .agg(avg_interface_density=("avg_interface_density", "mean"))
        .reset_index()
    )
    analysis_tasks = [(params, group) for params, group in df_avg.groupby(["s", "phi"])]

    print(f"\n--- Analyzing {len(analysis_tasks)} (s, phi) slices in parallel ---")
    with Pool(processes=max(1, cpu_count() - 2)) as pool:
        results = list(
            tqdm(
                pool.starmap(find_kc_for_slice, analysis_tasks),
                total=len(analysis_tasks),
                desc="Finding critical points",
            )
        )

    valid_results = [res for res in results if res is not None]
    if not valid_results:
        sys.exit("FATAL: No critical points could be determined.")

    if args.generate_debug_plots:
        print("\n--- Generating debug plots ---")
        for res in tqdm(valid_results, desc="Plotting fits"):
            generate_debug_fit_plot(res, DEBUG_DIR)

    df_crit = (
        pd.DataFrame(
            [{"s": r["s"], "phi": r["phi"], "k_c": r["k_c"]} for r in valid_results]
        )
        .dropna()
        .sort_values(["phi", "s"])
    )
    summary_path = os.path.join(ANALYSIS_DIR, "phase_boundaries_summary.csv")
    df_crit.to_csv(summary_path, index=False)
    print(f"\nSaved phase boundary data to: {summary_path}")

    # --- [NEW] Generate Final Comparative Plot ---
    fig, ax = plt.subplots(figsize=(12, 8))

    phi_slices = sorted(df_crit["phi"].unique())
    palette = sns.color_palette("viridis", n_colors=len(phi_slices))

    for i, phi_val in enumerate(phi_slices):
        phi_group = df_crit[df_crit["phi"] == phi_val].sort_values("s")
        if not phi_group.empty:
            ax.plot(
                phi_group["s"],
                phi_group["k_c"],
                "o-",
                color=palette[i],
                markersize=8,
                linewidth=2.5,
                label=f"$\\phi = {phi_val:.2f}$",
            )

    ax.set_yscale("log")
    ax.set_xlabel("Selection Coefficient ($s = b_m - 1$)", fontsize=14)
    ax.set_ylabel("Critical Switching Rate ($k_c$)", fontsize=14)
    ax.set_title(
        "Phase Boundaries: $k_c$ vs. Selection for Different Biases", fontsize=18
    )
    ax.legend(title="Switching Bias ($\\phi$)", fontsize=12)
    ax.grid(True, which="both", ls="--")

    plot_path = os.path.join(FIGS_DIR, "Fig_Phase_Boundaries_Comparative.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Final comparative plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
