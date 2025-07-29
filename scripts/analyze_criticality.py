# FILE: scripts/analyze_criticality_hierarchical.py
#
# [DEFINITIVE ANALYSIS v32 - FINAL NARRATIVE PLOTS]
# This is the final version. It implements the refined figure plan, separating
# the global model validation into its own figure and creating a new "zoom-out" vs.
# "zoom-in" showcase for the hierarchical method.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, argparse, json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy.optimize import curve_fit
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- Robust Path and Config Import ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))
try:
    from config import EXPERIMENTS
except ImportError:
    sys.exit("FATAL: Could not import EXPERIMENTS from src/config.py.")


# ==============================================================================
# STAGE 0 & 1: DATA LOADING & HIERARCHICAL FITTING
# ==============================================================================
def read_json_worker(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception:
        return None


def aggregate_data(campaign_id, analysis_dir, force_reaggregate=False):
    # ... (unchanged)
    results_dir = os.path.join(project_root, "data", campaign_id, "results")
    cached_csv_path = os.path.join(analysis_dir, f"{campaign_id}_aggregated_raw.csv")
    if not os.path.isdir(results_dir):
        print(f"Warning: Results directory not found for {campaign_id}")
        return None
    if not force_reaggregate and os.path.exists(cached_csv_path):
        return pd.read_csv(cached_csv_path)
    filepaths = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".json")
    ]
    if not filepaths:
        return None
    with Pool(processes=max(1, cpu_count() - 2)) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(read_json_worker, filepaths),
                total=len(filepaths),
                desc=f"Aggregating {campaign_id}",
            )
        )
    df = pd.DataFrame([r for r in results if r is not None])
    if df.empty:
        return None
    df.to_csv(cached_csv_path, index=False)
    return df


def average_metrics(df_raw):
    # ... (unchanged)
    group_keys = [k for k in ["b_m", "k_total"] if k in df_raw.columns]
    agg_dict = {"avg_interface_density": ("avg_interface_density", "mean")}
    final_agg_dict = {k: v for k, v in agg_dict.items() if v[0] in df_raw.columns}
    if not final_agg_dict:
        return pd.DataFrame()
    return df_raw.groupby(group_keys).agg(**final_agg_dict).reset_index()


def sigmoid_model(log_k, A, B, log_kc, n):
    return A / (1 + np.exp(-n * (log_k - log_kc))) + B


def analytical_derivative_sigmoid(log_k, A, n, log_kc):
    return (A * n * np.exp(-n * (log_k - log_kc))) / (
        (1 + np.exp(-n * (log_k - log_kc))) ** 2
    )


def fit_global_parameters_poly(df_global_avg):
    print("\n--- Stage 1: Learning global plateau behavior ---")
    plateau_params = []
    for s_val, group in tqdm(df_global_avg.groupby("s"), desc="Global fits"):
        if len(group) < 10:
            continue
        log_k, density = (
            np.log(group["k_total"].values),
            group["avg_interface_density"].values,
        )
        p0 = [
            max(density) - min(density),
            min(density),
            np.log(group["k_total"].median()),
            1.0,
        ]
        try:
            popt, _ = curve_fit(sigmoid_model, log_k, density, p0=p0, maxfev=10000)
            # [THE UPGRADE] Store all fitted parameters for the global susceptibility plot
            plateau_params.append(
                {
                    "s": s_val,
                    "A": popt[0],
                    "B": popt[1],
                    "log_kc_global": popt[2],
                    "n_global": popt[3],
                }
            )
        except RuntimeError:
            pass
    df_plateaus = pd.DataFrame(plateau_params).sort_values("s")
    poly_A = np.poly1d(np.polyfit(df_plateaus["s"], df_plateaus["A"], 9))
    poly_B = np.poly1d(np.polyfit(df_plateaus["s"], df_plateaus["B"], 9))
    return poly_A, poly_B, df_plateaus


def fit_local_constrained(df_local_avg, model_A, model_B):
    # ... (unchanged)
    print("\n--- Stage 2: Performing constrained fit on high-resolution data ---")

    def constrained_sigmoid_model(log_k, log_kc, n, A_fixed, B_fixed):
        return A_fixed / (1 + np.exp(-n * (log_k - log_kc))) + B_fixed

    fit_results = []
    for s_val, group in tqdm(df_local_avg.groupby("s"), desc="Constrained fits"):
        if len(group) < 5:
            continue
        A_fixed, B_fixed = model_A(s_val), model_B(s_val)
        log_k, density = (
            np.log(group["k_total"].values),
            group["avg_interface_density"].values,
        )
        p0 = [np.log(group["k_total"].median()), 4.0]
        try:
            popt, _ = curve_fit(
                lambda lk, lkc, n: constrained_sigmoid_model(
                    lk, lkc, n, A_fixed, B_fixed
                ),
                log_k,
                density,
                p0=p0,
                maxfev=10000,
            )
            log_kc, n = popt
            k_smooth = np.logspace(
                np.log10(group["k_total"].min()), np.log10(group["k_total"].max()), 200
            )
            fit_results.append(
                {
                    "s": s_val,
                    "k_c_fit": np.exp(log_kc),
                    "k_smooth": k_smooth,
                    "density_smooth": constrained_sigmoid_model(
                        np.log(k_smooth), log_kc, n, A_fixed, B_fixed
                    ),
                    "susceptibility_smooth": analytical_derivative_sigmoid(
                        np.log(k_smooth), A_fixed, n, log_kc
                    ),
                    "raw_k": group["k_total"].values,
                    "raw_density": group["avg_interface_density"].values,
                }
            )
        except RuntimeError:
            pass
    return fit_results


# ==============================================================================
# STAGE 3: PUBLICATION PLOTTING SUITE
# ==============================================================================
def plot_annotated_phase_boundary(fit_results, figs_dir):
    # ... (unchanged)
    print("  -> Generating Figure 1: The Annotated Phase Boundary...")
    df_crit = pd.DataFrame(
        [{"s": res["s"], "k_c_fit": res["k_c_fit"]} for res in fit_results]
    ).sort_values("s")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(
        df_crit["s"],
        df_crit["k_c_fit"],
        "o-",
        markersize=8,
        color="crimson",
        linewidth=2.5,
        zorder=10,
    )
    min_kc_idx = df_crit["k_c_fit"].idxmin()
    s_min, kc_min = df_crit.loc[min_kc_idx, "s"], df_crit.loc[min_kc_idx, "k_c_fit"]
    ax.plot(
        s_min,
        kc_min,
        "*",
        color="gold",
        markersize=20,
        markeredgecolor="black",
        label="Most Fragile State",
        zorder=11,
    )
    ax.axvspan(
        -0.8, -0.22, color="royalblue", alpha=0.1, label="Geometry-Dominated Regime"
    )
    ax.axvspan(
        -0.15, 0.0, color="mediumseagreen", alpha=0.1, label="Drift-Dominated Regime"
    )
    ax.set(
        xlabel="Selection Coefficient ($s = b_m - 1$)",
        ylabel="Critical Switching Rate ($k_c$)",
        title="Non-Monotonic Phase Boundary Reveals Three Physical Regimes",
        yscale="log",
    )
    ax.legend(fontsize=12)
    ax.grid(True, which="both", ls="--")
    plt.savefig(
        os.path.join(figs_dir, "Fig1_Annotated_Phase_Boundary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_mechanisms_diagnostic(df_crit, df_deff, df_wsat, figs_dir):
    # ... (unchanged, now becomes Figure 2)
    print("  -> Generating Figure 2: The Underlying Physical Mechanisms...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
    fig.suptitle("Disentangling the Forces Stabilizing the Ordered State", fontsize=24)
    ax1 = axes[0]
    ax2 = ax1.twinx()
    ax1.plot(
        df_deff["s"],
        df_deff["d_eff"],
        "o-",
        c="darkgreen",
        markersize=8,
        label="$D_{eff}$ (Drift Strength)",
    )
    ax2.plot(
        df_crit["s"],
        df_crit["k_c_fit"],
        "o--",
        c="gray",
        markersize=6,
        alpha=0.7,
        label="$k_c$ (for reference)",
    )
    ax1.set(
        xlabel="Selection (s)",
        ylabel="Effective Diffusion ($D_{eff}$)",
        title="A. Drift-Dominated Regime",
    )
    ax2.set(ylabel="$k_c$ (log scale)", yscale="log")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1 = axes[1]
    ax2 = ax1.twinx()
    ax1.plot(
        df_wsat["s"],
        df_wsat["W_sat"],
        "o-",
        c="darkred",
        markersize=8,
        label="$W_{sat}$ (Front Roughness)",
    )
    ax2.plot(
        df_crit["s"],
        df_crit["k_c_fit"],
        "o--",
        c="gray",
        markersize=6,
        alpha=0.7,
        label="$k_c$ (for reference)",
    )
    ax1.set(
        xlabel="Selection (s)",
        ylabel="Saturated Roughness ($W_{sat}$)",
        title="B. Geometry-Dominated Regime",
    )
    ax2.set(ylabel="$k_c$ (log scale)", yscale="log")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    for ax in [axes[0], axes[1]]:
        ax.grid(True, ls="--", alpha=0.6)
        ax.axvline(0, color="black", lw=0.5)
    plt.savefig(
        os.path.join(figs_dir, "Fig2_Mechanisms_Diagnostic.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_hierarchical_method_summary(
    df_plateaus, model_A, model_B, df_global_avg, fit_results_local, figs_dir
):
    print("  -> Generating Figure 3: A Showcase of the Analytical Method...")
    fig, axes = plt.subplots(1, 3, figsize=(28, 8), constrained_layout=True)
    fig.suptitle(
        "The Hierarchical Fitting Method: A Robust Path to Precision", fontsize=24
    )
    cmap = cm.magma

    # --- Panel A: Learning the Global Model ---
    s_smooth = np.linspace(df_plateaus["s"].min(), df_plateaus["s"].max(), 200)
    axes[0].plot(
        df_plateaus["s"], df_plateaus["A"], "o", c="C0", label="Measured Amplitude"
    )
    axes[0].plot(s_smooth, model_A(s_smooth), "-", c="C0", label="Smooth Fit A(s)")
    ax2 = axes[0].twinx()
    ax2.plot(
        df_plateaus["s"],
        df_plateaus["B"],
        "s",
        c="C1",
        markerfacecolor="none",
        label="Measured Baseline",
    )
    ax2.plot(s_smooth, model_B(s_smooth), "--", c="C1", label="Smooth Fit B(s)")
    axes[0].set(
        xlabel="Selection (s)",
        ylabel="Amplitude A",
        title="A. Learning Global Plateau Models",
    )
    ax2.set_ylabel("Baseline B")
    axes[0].legend(loc="upper left")
    ax2.legend(loc="upper right")

    # --- Panel B: Global Susceptibility Landscape ("Zoomed Out") ---
    s_values_global = sorted(df_global_avg["s"].unique())
    norm_global = mcolors.Normalize(
        vmin=min(s_values_global), vmax=max(s_values_global)
    )
    for _, row in df_plateaus.iterrows():
        k_smooth_global = np.logspace(-2, 2, 400)
        susc = analytical_derivative_sigmoid(
            np.log(k_smooth_global), row["A"], row["n_global"], row["log_kc_global"]
        )
        axes[1].plot(k_smooth_global, susc, "-", color=cmap(norm_global(row["s"])))
    axes[1].set(
        xscale="log",
        xlabel="$k_{total}$",
        ylabel="Susceptibility (from Model)",
        title="B. Susceptibility Landscape (Global View)",
    )

    # --- Panel C: High-Resolution Susceptibility ("Zoomed In") ---
    s_values_local = [res["s"] for res in fit_results_local]
    norm_local = mcolors.Normalize(vmin=min(s_values_local), vmax=max(s_values_local))
    for res in fit_results_local:
        color = cmap(norm_local(res["s"]))
        axes[2].plot(res["k_smooth"], res["susceptibility_smooth"], "-", color=color)
        peak_idx = np.argmax(res["susceptibility_smooth"])
        axes[2].plot(
            res["k_smooth"][peak_idx],
            res["susceptibility_smooth"][peak_idx],
            "*",
            ms=10,
            color=color,
            markeredgecolor="black",
        )
    axes[2].set(
        xscale="log",
        xlabel="$k_{total}$",
        ylabel="Susceptibility (from Model)",
        title="C. Precision Measurement of $k_c$ (Local View)",
    )

    for ax in axes:
        ax.grid(True, which="both", ls="--")
    plt.savefig(
        os.path.join(figs_dir, "Fig3_Hierarchical_Method_Showcase.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    GLOBAL_EXP, LOCAL_EXP = "criticality_v2", "criticality_v3"
    BOUNDARY_EXP, MORPHOLOGY_EXP = (
        "calibration_boundary_dynamics_v1",
        "calibration_front_morphology_v1",
    )
    FIGS_DIR = os.path.join(project_root, "figures", "FINAL_PUBLICATION_SUITE")
    os.makedirs(FIGS_DIR, exist_ok=True)

    print("\n--- Loading all available datasets ---")
    local_analysis_dir = os.path.join(
        project_root, "data", EXPERIMENTS[LOCAL_EXP]["CAMPAIGN_ID"], "analysis"
    )
    os.makedirs(local_analysis_dir, exist_ok=True)
    df_global_raw = aggregate_data(
        EXPERIMENTS[GLOBAL_EXP]["CAMPAIGN_ID"], local_analysis_dir
    )
    df_local_raw = aggregate_data(
        EXPERIMENTS[LOCAL_EXP]["CAMPAIGN_ID"], local_analysis_dir
    )
    if df_global_raw is None or df_local_raw is None:
        sys.exit("FATAL: Core criticality data is missing.")

    df_deff, df_wsat = None, None
    try:
        boundary_analysis_dir = os.path.join(
            project_root, "data", EXPERIMENTS[BOUNDARY_EXP]["CAMPAIGN_ID"], "analysis"
        )
        df_deff = pd.read_csv(
            os.path.join(boundary_analysis_dir, "boundary_dynamics_summary.csv")
        )
    except (KeyError, FileNotFoundError):
        print("[WARNING] Boundary dynamics data not found.")
    try:
        morphology_analysis_dir = os.path.join(
            project_root, "data", EXPERIMENTS[MORPHOLOGY_EXP]["CAMPAIGN_ID"], "analysis"
        )
        df_wsat = pd.read_csv(
            os.path.join(morphology_analysis_dir, "front_morphology_summary.csv")
        )
    except (KeyError, FileNotFoundError):
        print("[WARNING] Front morphology data not found.")

    df_global_avg = average_metrics(df_global_raw)
    df_global_avg["s"] = df_global_avg["b_m"] - 1.0
    df_local_avg = average_metrics(df_local_raw)
    df_local_avg["s"] = df_local_avg["b_m"] - 1.0

    model_A, model_B, df_plateaus = fit_global_parameters_poly(df_global_avg)
    fit_results_local = fit_local_constrained(df_local_avg, model_A, model_B)
    if not fit_results_local:
        sys.exit("FATAL: Hierarchical analysis failed.")

    print("\n--- Generating Publication-Ready Plot Suite ---")
    plot_annotated_phase_boundary(fit_results_local, FIGS_DIR)
    plot_hierarchical_method_summary(
        df_plateaus, model_A, model_B, df_global_avg, fit_results_local, FIGS_DIR
    )
    if df_deff is not None and df_wsat is not None:
        df_crit = pd.DataFrame(
            [{"s": res["s"], "k_c_fit": res["k_c_fit"]} for res in fit_results_local]
        )
        plot_mechanisms_diagnostic(df_crit, df_deff, df_wsat, FIGS_DIR)
    else:
        print("[INFO] Skipping mechanisms plot due to missing diagnostic data.")

    print(f"\n--- Analysis Complete. Final figures saved to: {FIGS_DIR} ---")


if __name__ == "__main__":
    main()
