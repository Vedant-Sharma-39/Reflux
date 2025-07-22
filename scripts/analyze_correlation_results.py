# FILE: scripts/analyze_correlation_results.py
#
# [DEFINITIVE FINAL v7] This script performs the final, publication-ready analysis.
# It uses a highly robust method to find xi and a physically-motivated,
# range-limited fit to accurately measure the critical exponent eta.
#
# [MODIFICATION] This version adds a debug plotting capability to visualize
# the calculation of the critical switching rate, k_c.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import collections
from scipy.optimize import curve_fit
from scipy.stats import linregress
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- Robust Path and Config Import ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))
from config import EXPERIMENTS

# --- Configuration ---
EXPERIMENT_NAME = "spatial_structure_v1"
CAMPAIGN_ID = EXPERIMENTS[EXPERIMENT_NAME]["CAMPAIGN_ID"]
FIGURES_DIR = os.path.join(project_root, "figures", CAMPAIGN_ID)
os.makedirs(FIGURES_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

# --- [NEW] Publication-quality plot style ---
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica",
            "Arial",
            "Liberation Sans",
            "DejaVu Sans",
            "sans-serif",
        ],
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 22,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 24,
    }
)

# --- Analysis Constants ---
SIMULATION_WIDTH = 256
MAX_R_FOR_PLOTTING = 128
XI_UPPER_BOUND = 2 * SIMULATION_WIDTH
GINF_ESTIMATION_RANGE = (40, 60)
# --- [FIX] Adjusting fitting range based on user feedback to capture the initial decay ---
MIN_R_FOR_ETA_FITTING = 2
MAX_R_FOR_ETA_FITTING = 10


# ==============================================================================
# 1. & 2. Data Loading and Processing Functions (Unchanged and Correct)
# ==============================================================================
# ... (These sections are identical to the previous definitive version) ...
def read_json_worker(filepath):  # (code is identical)
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def aggregate_raw_data(campaign_id, force_reaggregate=False):  # (code is identical)
    results_dir = os.path.join(project_root, "data", campaign_id, "results")
    output_csv = os.path.join(project_root, "data", f"{campaign_id}_aggregated.csv")
    if not os.path.isdir(results_dir):
        return None
    if os.path.exists(output_csv) and not force_reaggregate:
        return pd.read_csv(output_csv)
    file_paths = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".json")
    ]
    if not file_paths:
        return pd.DataFrame()
    with Pool(processes=max(1, cpu_count() - 2)) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(read_json_worker, file_paths),
                total=len(file_paths),
                desc="Reading JSONs",
            )
        )
    valid_results = [r for r in results if r is not None and "g_r" in r]
    if not valid_results:
        return pd.DataFrame()
    df = pd.DataFrame(valid_results)
    df.to_csv(output_csv, index=False)
    return df


def physical_decay_model(r, C, xi, g_inf):
    return C * np.exp(-r / xi) + g_inf


def calculate_average_g_r(df):  # (code is identical)
    print("Step 1: Averaging g(r)...")
    all_g_r_points = collections.defaultdict(lambda: collections.defaultdict(list))
    tqdm.pandas(desc="Parsing g(r)")
    df["g_r_parsed"] = df["g_r"].progress_apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Aggregating"):
        params = (row["b_m"], row["k_total"])
        for r, g_val in row["g_r_parsed"]:
            all_g_r_points[params][r].append(g_val)
    avg_g_r_data = [
        {"b_m": p[0], "k_total": p[1], "avg_g_r": sorted(d.items())}
        for p, r_data in all_g_r_points.items()
        for d in [{r: np.mean(v) for r, v in r_data.items()}]
    ]
    return pd.DataFrame(avg_g_r_data)


def calculate_correlation_lengths_final(df_avg_g_r):  # (code is identical)
    print("Step 2: Calculating xi...")
    results = []
    for _, row in tqdm(
        df_avg_g_r.iterrows(), total=len(df_avg_g_r), desc="Final robust fitting"
    ):
        r_data, g_data = np.array(row["avg_g_r"]).T
        if len(r_data) < 10:
            results.append(
                {
                    "b_m": row["b_m"],
                    "k_total": row["k_total"],
                    "C_fit": 0,
                    "xi": 1.0,
                    "g_inf_fit": 0,
                }
            )
            continue
        initial_decay_mask = (r_data > 0) & (r_data <= 10)
        has_decay_signal = False
        if np.sum(initial_decay_mask) > 3:
            slope_initial, _, r_val, _, _ = linregress(
                r_data[initial_decay_mask], g_data[initial_decay_mask]
            )
            if slope_initial < -1e-4 and r_val**2 > 0.05:
                has_decay_signal = True
        if not has_decay_signal:
            C_fit, xi_fit, g_inf_fit = 0, 1.0, np.mean(g_data)
        else:
            tail_mask = (r_data >= GINF_ESTIMATION_RANGE[0]) & (
                r_data <= GINF_ESTIMATION_RANGE[1]
            )
            g_inf_guess = (
                np.mean(g_data[tail_mask]) if np.any(tail_mask) else g_data[-1]
            )
            g_initial = g_data[r_data > 0][0]
            C_guess = max(0, g_initial - g_inf_guess)
            p0 = [C_guess, 10.0, g_inf_guess]
            bounds = ([0, 1.0, 0], [1.0, XI_UPPER_BOUND, 1.0])
            try:
                fit_mask = r_data <= MAX_R_FOR_PLOTTING
                popt, _ = curve_fit(
                    physical_decay_model,
                    r_data[fit_mask],
                    g_data[fit_mask],
                    p0=p0,
                    bounds=bounds,
                    maxfev=10000,
                )
                C_fit, xi_fit, g_inf_fit = popt
            except (RuntimeError, ValueError):
                C_fit, xi_fit, g_inf_fit = 0, 1.0, np.mean(g_data)
        results.append(
            {
                "b_m": row["b_m"],
                "k_total": row["k_total"],
                "C_fit": C_fit,
                "xi": xi_fit,
                "g_inf_fit": g_inf_fit,
            }
        )
    return pd.DataFrame(results).dropna()


# ==============================================================================
# 3. PLOTTING FUNCTIONS
# ==============================================================================
def find_closest_k(df, k_target):
    return df.iloc[(df["k_total"] - k_target).abs().argmin()]


# --- [MODIFIED] This function is now enhanced for debug plotting ---
def find_critical_k_robustly(df_fits_subset, debug_plot=False):
    """
    Calculates the critical switching rate k_c.
    If debug_plot is True, it also generates and saves a plot visualizing
    the entire calculation process.
    """
    b_m = df_fits_subset["b_m"].iloc[0]
    df_fits_subset = df_fits_subset.sort_values("k_total")

    # Define the k-range where the transition is expected to be linear in log-log space
    k_ranges = {0.5: (0.05, 0.5), 0.8: (0.1, 1.0), 0.95: (0.03, 0.5)}
    k_min, k_max = k_ranges.get(b_m, (0.1, 1.0))
    transition_df = df_fits_subset[
        (df_fits_subset["k_total"] >= k_min) & (df_fits_subset["k_total"] <= k_max)
    ]

    # --- Fallback for insufficient data in the transition region ---
    if len(transition_df) < 3:
        log_k = np.log10(df_fits_subset["k_total"])
        log_xi = np.log10(df_fits_subset["xi"])
        grad = np.gradient(log_xi, log_k)
        k_c_val = df_fits_subset.iloc[np.argmin(grad)]["k_total"]

        if debug_plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(
                df_fits_subset["k_total"],
                df_fits_subset["xi"],
                "o-",
                label=f"All data for $b_m={b_m}$",
            )
            ax.axvline(
                k_c_val,
                color="red",
                linestyle="--",
                label=f"Estimated $k_c \\approx {k_c_val:.3f}$",
            )
            ax.set_title(f"Debug: $k_c$ Calculation (Fallback) for $b_m = {b_m}$")
            ax.text(
                0.5,
                0.5,
                "Fallback method used:\nNot enough points in transition region.\n$k_c$ is point of minimum gradient.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", fc="white", ec="black"),
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Total Switching Rate ($k_{total}$)")
            ax.set_ylabel("Correlation Length ($\\xi$)")
            ax.legend()
            debug_fig_dir = os.path.join(FIGURES_DIR, "k_c_fit_debug")
            os.makedirs(debug_fig_dir, exist_ok=True)
            plt.savefig(
                os.path.join(debug_fig_dir, f"kc_fit_bm_{b_m}_fallback.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

        return k_c_val

    # --- Main linear fit logic ---
    x_fit = np.log10(transition_df["k_total"])
    y_fit = np.log10(transition_df["xi"])
    slope, intercept, r_val, _, _ = linregress(x_fit, y_fit)

    # Estimate xi_max from the "ordered" phase (k < k_min)
    ordered_phase_df = df_fits_subset[df_fits_subset["k_total"] < k_min]
    xi_max = ordered_phase_df["xi"].max() if not ordered_phase_df.empty else 100
    xi_min = 1.0  # By definition in the "disordered" phase

    # Critical correlation length is the geometric mean of the two phases
    xi_c = np.sqrt(xi_max * xi_min)
    log_xi_c = np.log10(xi_c)

    # Invert the linear equation to find the corresponding k_c
    log_k_c = (log_xi_c - intercept) / slope
    k_c_final = 10**log_k_c

    # --- Debug plotting logic ---
    if debug_plot:
        fig, ax = plt.subplots(figsize=(14, 9))

        # 1. Plot all data points
        ax.plot(
            df_fits_subset["k_total"],
            df_fits_subset["xi"],
            "o-",
            color="skyblue",
            markersize=8,
            label=f"All data for $b_m={b_m}$",
        )

        # 2. Highlight the points used for the linear fit
        ax.plot(
            transition_df["k_total"],
            transition_df["xi"],
            "s",
            color="navy",
            markersize=10,
            label="Data for linear fit",
        )

        # 3. Plot the extended linear fit line
        k_plot_range = np.logspace(
            np.log10(df_fits_subset["k_total"].min()),
            np.log10(df_fits_subset["k_total"].max()),
            100,
        )
        log_k_plot_range = np.log10(k_plot_range)
        log_xi_fit = slope * log_k_plot_range + intercept
        xi_fit = 10**log_xi_fit
        ax.plot(
            k_plot_range,
            xi_fit,
            "k--",
            linewidth=2,
            label=f"Linear Fit (log-log, $R^2={r_val**2:.3f}$)",
        )

        # 4. Draw horizontal lines for xi_max, xi_min, and xi_c
        ax.axhline(
            xi_max,
            color="green",
            linestyle=":",
            linewidth=2,
            label=f"$\\xi_{{max}} \\approx {xi_max:.2f}$",
        )
        ax.axhline(
            xi_min,
            color="orange",
            linestyle=":",
            linewidth=2,
            label=f"$\\xi_{{min}} = {xi_min:.2f}$",
        )
        ax.axhline(
            xi_c,
            color="purple",
            linestyle="--",
            linewidth=2.5,
            label=f"$\\xi_c = \\sqrt{{\\xi_{{max}}\\xi_{{min}}}} \\approx {xi_c:.2f}$",
        )

        # 5. Draw a vertical line showing the final calculated k_c
        ax.axvline(
            k_c_final,
            color="red",
            linestyle="--",
            linewidth=2.5,
            label=f"$k_c \\approx {k_c_final:.3f}$",
        )

        # Finalize plot aesthetics
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Debug: $k_c$ Calculation for $b_m = {b_m}$")
        ax.set_xlabel("Total Switching Rate ($k_{total}$)")
        ax.set_ylabel("Correlation Length ($\\xi$)")
        ax.legend(loc="best", fontsize=12)
        ax.grid(True, which="both", ls="--")
        ax.set_facecolor("#f5f5f5")

        # Save the figure to a dedicated debug subfolder
        debug_fig_dir = os.path.join(FIGURES_DIR, "k_c_fit_debug")
        os.makedirs(debug_fig_dir, exist_ok=True)
        plt.savefig(
            os.path.join(debug_fig_dir, f"kc_fit_bm_{b_m}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    return k_c_final


def plot_figure_2_collapse(df_fits):
    print("  -> Generating Figure 2: Collapse of Spatial Order...")
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.lineplot(
        data=df_fits,
        x="k_total",
        y="xi",
        hue="b_m",
        marker="o",
        palette="viridis",
        ax=ax,
        linewidth=3,
        markersize=10,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Collapse of Spatial Order")
    ax.set_xlabel("Total Switching Rate ($k_{total}$)")
    ax.set_ylabel("Correlation Length ($\\xi$)")
    ax.legend(title="$b_m$ (Mutant Fitness)")
    ax.grid(True, which="both", ls="--", linewidth=0.7)
    plt.savefig(
        os.path.join(FIGURES_DIR, "Fig2_CorrelationLength_Collapse.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_figure_3_phase_diagram(df_fits):
    print("  -> Generating Figure 3: Phase Diagram...")
    df_fits["fitness_cost"] = 1 - df_fits["b_m"]
    pivot_df = df_fits.pivot_table(index="k_total", columns="fitness_cost", values="xi")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        pivot_df,
        annot=False,
        cmap="magma",
        norm=plt.matplotlib.colors.LogNorm(),
        cbar_kws={"label": "Correlation Length ($\\xi$)"},
        ax=ax,
    )
    ax.invert_yaxis()
    ax.set_title("Phase Diagram of Spatial Organization")
    ax.set_xlabel("Fitness Cost ($1 - b_m$)")
    ax.set_ylabel("Total Switching Rate ($k_{total}$)")
    plt.savefig(
        os.path.join(FIGURES_DIR, "Fig3_PhaseDiagram_xi.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_figure_4_critical(df_avg_g_r, df_fits):
    """
    [FINAL VERIFIED VERSION] This version performs a direct, simple linear fit on the
    log-log data within the visually identified scaling regime (r=2 to r=10),
    as per the final analysis. This avoids the instabilities of previous methods.
    """
    print("  -> Generating Figure 4: Probing the Critical Point...")
    b_m_critical = 0.8
    xi_subset = df_fits[np.isclose(df_fits["b_m"], b_m_critical)].sort_values("k_total")
    if len(xi_subset) < 3:
        return

    k_c = find_critical_k_robustly(xi_subset)
    critical_k_in_data = find_closest_k(xi_subset, k_c)["k_total"]

    critical_data_row = df_avg_g_r[
        np.isclose(df_avg_g_r["b_m"], b_m_critical)
        & np.isclose(df_avg_g_r["k_total"], critical_k_in_data)
    ]
    if critical_data_row.empty:
        return

    r_crit, g_crit = np.array(critical_data_row["avg_g_r"].iloc[0]).T

    # --- FINAL, SIMPLIFIED FITTING LOGIC ---
    fit_mask = (r_crit >= MIN_R_FOR_ETA_FITTING) & (r_crit <= MAX_R_FOR_ETA_FITTING)
    if np.sum(fit_mask) < 4:
        print(
            f"  -> Warning: Not enough points in range [{MIN_R_FOR_ETA_FITTING}, {MAX_R_FOR_ETA_FITTING}] to fit. Skipping."
        )
        return

    log_r = np.log10(r_crit[fit_mask])
    log_g = np.log10(g_crit[fit_mask])
    slope, intercept, r_val, _, _ = linregress(log_r, log_g)
    eta = -slope

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(16, 10))
    plot_mask = r_crit <= MAX_R_FOR_PLOTTING
    ax.plot(
        r_crit[plot_mask],
        g_crit[plot_mask],
        "o",
        color="darkorange",
        markersize=12,
        label="Simulation Data at Criticality",
    )
    r_fit_line = np.logspace(
        np.log10(MIN_R_FOR_ETA_FITTING), np.log10(MAX_R_FOR_ETA_FITTING), 50
    )
    g_fit_line = (10**intercept) * (r_fit_line ** (-eta))
    ax.plot(
        r_fit_line,
        g_fit_line,
        "k--",
        linewidth=4,
        label=f"Power-Law Fit ($R^2={r_val**2:.3f}$)",
    )
    ax.axvspan(
        MIN_R_FOR_ETA_FITTING,
        MAX_R_FOR_ETA_FITTING,
        color="gray",
        alpha=0.2,
        zorder=-1,
        label="Scaling Regime for Fit",
    )
    ax.text(
        0.95,
        0.95,
        f"$\\eta \\approx {eta:.2f}$",
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.7),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(
        f"Critical Behavior at $k_c \\approx {k_c:.2f}$ for $b_m = {b_m_critical}$"
    )
    ax.set_xlabel("Geodesic Distance (r)")
    ax.set_ylabel("Correlation Function g(r)")
    ax.legend()
    plt.savefig(
        os.path.join(FIGURES_DIR, "Fig4_Critical_PowerLaw_Final.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ==============================================================================
# 3.5. [NEW] DEBUG PLOTTING FUNCTIONS
# ==============================================================================
def plot_all_k_c_fit_debugs(df_fits):
    """
    Generates a debug plot for each b_m value to visualize the k_c calculation.
    """
    print("\nStep 3.5: Generating debug plots for k_c calculation...")

    # This will create a sub-directory for the debug plots to keep things tidy
    debug_fig_dir = os.path.join(FIGURES_DIR, "k_c_fit_debug")
    os.makedirs(debug_fig_dir, exist_ok=True)
    print(f"  -> Debug plots will be saved to: {debug_fig_dir}")

    for b_m_val in sorted(df_fits["b_m"].unique()):
        subset = df_fits[df_fits["b_m"] == b_m_val].sort_values("k_total")
        if len(subset) < 5:  # Not enough data to create a meaningful plot
            continue
        print(f"  -> Plotting debug for b_m = {b_m_val}")
        # The modified function is now self-contained for plotting.
        # It handles saving the plot internally when the flag is True.
        find_critical_k_robustly(subset, debug_plot=True)


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
def main():
    print(f"--- Running Full, Definitive Analysis for Campaign: {CAMPAIGN_ID} ---")
    df_raw = aggregate_raw_data(CAMPAIGN_ID)
    if df_raw is None or df_raw.empty:
        print(f"\nFATAL: No data found. Exiting.")
        return

    df_avg_g_r = calculate_average_g_r(df_raw)
    df_fits = calculate_correlation_lengths_final(df_avg_g_r)

    print("\nStep 3: Generating final publication figures...")
    plot_figure_2_collapse(df_fits)
    plot_figure_3_phase_diagram(df_fits)
    plot_figure_4_critical(df_avg_g_r, df_fits)

    # --- [NEW] Call the debug plotting function ---
    plot_all_k_c_fit_debugs(df_fits)

    print(f"\n--- Analysis complete. Figures saved to: {FIGURES_DIR} ---")
    print(
        "\nSuggestion for next steps: With these robust results, you are ready to write the manuscript."
    )
    print(
        "If more precision is needed for the critical exponent, a new, highly focused simulation campaign"
    )
    k_c_bm_08_subset = df_fits[np.isclose(df_fits["b_m"], 0.8)]
    if not k_c_bm_08_subset.empty:
        k_c_val = find_critical_k_robustly(k_c_bm_08_subset)
        print(
            f"with many replicates around k_c={k_c_val:.4f} for b_m=0.8 could be run."
        )


if __name__ == "__main__":
    main()
