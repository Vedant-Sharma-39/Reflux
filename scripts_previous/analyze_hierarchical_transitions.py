# FILE: scripts/analyze_criticality.py
#
# [DEFINITIVE HIERARCHICAL ANALYSIS - GENERALIZED FOR MULTI-PHI & MULTI-METRIC]
# This script is the final, primary tool for analyzing the two-phase
# hierarchical simulation campaigns (e.g., phase1_... and phase2_...).
#
# It automatically detects all `phi` slices present in the data and performs
# the complete hierarchical analysis for each one independently. It runs this
# analysis for two key metrics:
#   1. avg_interface_density (spatial order)
#   2. avg_rho_M (population composition)
#
# For each `phi` slice, it generates a full suite of publication-ready figures,
# including the final comparative susceptibility plot that highlights the
# decoupling of the two critical points.

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
    group_keys = [k for k in ["b_m", "phi", "k_total"] if k in df_raw.columns]
    agg_dict = {
        "avg_interface_density": ("avg_interface_density", "mean"),
        "avg_rho_M": ("avg_rho_M", "mean"),
    }
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


def fit_global_sigmoid_models(df_global_slice, metric_name):
    """Generalized function to fit sigmoid models to any metric."""
    plateau_params = []
    for s_val, group in tqdm(
        df_global_slice.groupby("s"), desc=f"Global fits for {metric_name}", leave=False
    ):
        if len(group) < 10:
            continue
        log_k, metric = np.log(group["k_total"].values), group[metric_name].values
        try:
            popt, _ = curve_fit(
                sigmoid_model,
                log_k,
                metric,
                p0=[
                    max(metric) - min(metric),
                    min(metric),
                    np.log(group["k_total"].median()),
                    1.0,
                ],
                maxfev=10000,
            )
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
    if not plateau_params:
        return None, None, None
    df_plateaus = pd.DataFrame(plateau_params).sort_values("s")
    poly_deg = min(9, len(df_plateaus) - 2)
    if poly_deg < 1:
        return None, None, None
    poly_A = np.poly1d(np.polyfit(df_plateaus["s"], df_plateaus["A"], poly_deg))
    poly_B = np.poly1d(np.polyfit(df_plateaus["s"], df_plateaus["B"], poly_deg))
    return poly_A, poly_B, df_plateaus


def fit_local_constrained_general(df_local_slice, model_A, model_B, metric_name):
    """A generalized version of the constrained fitting function."""

    def constrained_sigmoid(log_k, log_kc, n, A, B):
        return A / (1 + np.exp(-n * (log_k - log_kc))) + B

    fit_results = []
    for s_val, group in tqdm(
        df_local_slice.groupby("s"),
        desc=f"Constrained fits for {metric_name}",
        leave=False,
    ):
        if len(group) < 5 or metric_name not in group.columns:
            continue
        A_fixed, B_fixed = model_A(s_val), model_B(s_val)
        log_k, metric_vals = np.log(group["k_total"].values), group[metric_name].values
        try:
            popt, _ = curve_fit(
                lambda lk, lkc, n: constrained_sigmoid(lk, lkc, n, A_fixed, B_fixed),
                log_k,
                metric_vals,
                p0=[np.log(group["k_total"].median()), 4.0],
                maxfev=10000,
            )
            log_kc, n = popt
            k_min, k_max = group["k_total"].min(), group["k_total"].max()
            k_smooth = np.logspace(np.log10(k_min * 0.5), np.log10(k_max * 2), 200)
            fit_results.append(
                {
                    "s": s_val,
                    "k_c_fit": np.exp(log_kc),
                    "k_smooth": k_smooth,
                    "susceptibility_smooth": analytical_derivative_sigmoid(
                        np.log(k_smooth), A_fixed, n, log_kc
                    ),
                }
            )
        except RuntimeError:
            pass
    return fit_results


# ==============================================================================
# STAGE 3: PUBLICATION PLOTTING SUITE
# ==============================================================================
def plot_model_validation(phi_val, df_plateaus, model_A, model_B, figs_dir):
    print(f"  -> Generating Figure A: Model Validation for phi={phi_val:.2f}...")
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(
        f"Learning Global Plateau Models for Interface Density ($\\phi = {phi_val:.2f}$)",
        fontsize=20,
    )
    s_smooth = np.linspace(df_plateaus["s"].min(), df_plateaus["s"].max(), 200)
    ax.plot(
        df_plateaus["s"],
        df_plateaus["A"],
        "o",
        c="C0",
        ms=8,
        label="Measured Amplitude",
    )
    ax.plot(
        s_smooth, model_A(s_smooth), "-", c="C0", lw=2.5, label="Smooth Poly Fit A(s)"
    )
    ax2 = ax.twinx()
    ax2.plot(
        df_plateaus["s"],
        df_plateaus["B"],
        "s",
        c="C1",
        ms=8,
        markerfacecolor="none",
        label="Measured Baseline",
    )
    ax2.plot(
        s_smooth, model_B(s_smooth), "--", c="C1", lw=2.5, label="Smooth Poly Fit B(s)"
    )
    ax.set(xlabel="Selection (s)", ylabel="Amplitude A (Interface Density)")
    ax2.set_ylabel("Baseline B (Interface Density)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, which="both", ls="--")
    filename = f"Fig_A_Model_Validation_phi_{phi_val:.2f}.png".replace(".", "p")
    plt.savefig(os.path.join(figs_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


def plot_global_fits_on_data(
    phi_val, df_global_slice, df_plateaus_density, df_plateaus_rho, figs_dir
):
    """Creates a diagnostic plot showing the raw global data overlaid with the fitted sigmoid curves."""
    print(
        f"  -> Generating Figure B: Global Data with Fitted Models for phi={phi_val:.2f}..."
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9), constrained_layout=True)
    fig.suptitle(
        f"Global System Behavior & Model Fits ($\\phi = {phi_val:.2f}$)", fontsize=24
    )
    cmap = cm.magma
    s_vals = sorted(df_global_slice["s"].unique())
    norm = mcolors.Normalize(vmin=min(s_vals), vmax=max(s_vals))

    if df_plateaus_density is not None:
        for _, row in df_plateaus_density.iterrows():
            s_val, color = row["s"], cmap(norm(row["s"]))
            data_slice = df_global_slice[df_global_slice["s"] == s_val]
            ax1.plot(
                data_slice["k_total"],
                data_slice["avg_interface_density"],
                "o",
                ms=5,
                color=color,
                alpha=0.6,
            )
            k_smooth = np.logspace(-2.5, 2.5, 400)
            fit_params = (row["A"], row["B"], row["log_kc_global"], row["n_global"])
            ax1.plot(
                k_smooth,
                sigmoid_model(np.log(k_smooth), *fit_params),
                "-",
                lw=2,
                color=color,
            )
    ax1.set(
        xscale="log",
        xlabel="$k_{total}$",
        ylabel="Avg. Interface Density",
        title="A. Spatial Order",
    )

    if df_plateaus_rho is not None:
        for _, row in df_plateaus_rho.iterrows():
            s_val, color = row["s"], cmap(norm(row["s"]))
            data_slice = df_global_slice[df_global_slice["s"] == s_val]
            ax2.plot(
                data_slice["k_total"],
                data_slice["avg_rho_M"],
                "o",
                ms=5,
                color=color,
                alpha=0.6,
            )
            k_smooth = np.logspace(-2.5, 2.5, 400)
            fit_params = (row["A"], row["B"], row["log_kc_global"], row["n_global"])
            ax2.plot(
                k_smooth,
                sigmoid_model(np.log(k_smooth), *fit_params),
                "-",
                lw=2,
                color=color,
            )
    ax2.set(
        xscale="log",
        xlabel="$k_{total}$",
        ylabel="Avg. Mutant Fraction $\\langle\\rho_M\\rangle$",
        title="B. Population Composition",
    )

    for ax in (ax1, ax2):
        ax.grid(True, which="both", ls="--")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], aspect=30, pad=0.02)
    cbar.set_label("Selection (s)")
    filename = f"Fig_B_Global_Data_and_Fits_phi_{phi_val:.2f}.png".replace(".", "p")
    plt.savefig(os.path.join(figs_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


def plot_susceptibility_analysis(
    phi_val,
    df_plateaus_density,
    df_plateaus_rho,
    fit_results_local_density,
    fit_results_local_rho,
    figs_dir,
):
    print(f"  -> Generating Figure C: Susceptibility Analysis for phi={phi_val:.2f}...")
    fig, axes = plt.subplots(1, 3, figsize=(32, 8), constrained_layout=True)
    fig.suptitle(
        f"Susceptibility Analysis & Precision Measurement of Critical Points ($\\phi = {phi_val:.2f}$)",
        fontsize=24,
    )
    cmap = cm.magma

    s_vals_local = [res["s"] for res in fit_results_local_density]
    if not s_vals_local:
        return
    norm = mcolors.Normalize(vmin=min(s_vals_local), vmax=max(s_vals_local))
    s_vals_global = sorted(df_plateaus_density["s"].unique())
    norm_global = mcolors.Normalize(vmin=min(s_vals_global), vmax=max(s_vals_global))

    for _, row in df_plateaus_density.iterrows():
        k_smooth = np.logspace(-2.5, 2.5, 400)
        susc = analytical_derivative_sigmoid(
            np.log(k_smooth), row["A"], row["n_global"], row["log_kc_global"]
        )
        axes[0].plot(k_smooth, susc, "-", color=cmap(norm_global(row["s"])))
    axes[0].set(
        xscale="log",
        xlabel="$k_{total}$",
        ylabel="Susceptibility (Order)",
        title="A. Susceptibility Landscape (Global)",
    )

    for res in fit_results_local_density:
        color = cmap(norm(res["s"]))
        axes[1].plot(res["k_smooth"], res["susceptibility_smooth"], "-", color=color)
        peak_idx = np.argmax(res["susceptibility_smooth"])
        axes[1].plot(
            res["k_smooth"][peak_idx],
            res["susceptibility_smooth"][peak_idx],
            "*",
            ms=14,
            color=color,
            markeredgecolor="black",
            label="Order $k_c$" if res["s"] == s_vals_local[0] else "",
        )
    if fit_results_local_rho:
        for res in fit_results_local_rho:
            color = cmap(norm(res["s"]))
            axes[1].plot(
                res["k_smooth"],
                res["susceptibility_smooth"],
                "--",
                color=color,
                alpha=0.7,
            )
            peak_idx = np.argmax(res["susceptibility_smooth"])
            axes[1].plot(
                res["k_smooth"][peak_idx],
                res["susceptibility_smooth"][peak_idx],
                "v",
                ms=10,
                color=color,
                markeredgecolor="white",
                label="Composition $k_c$" if res["s"] == s_vals_local[0] else "",
            )
    axes[1].set(
        xscale="log",
        xlabel="$k_{total}$",
        title="B. Precision Measurement of Critical Points (Local)",
    )
    axes[1].legend()

    if df_plateaus_rho is not None:
        for _, row in df_plateaus_rho.iterrows():
            k_smooth = np.logspace(-2.5, 2.5, 400)
            susc = analytical_derivative_sigmoid(
                np.log(k_smooth), row["A"], row["n_global"], row["log_kc_global"]
            )
            axes[2].plot(k_smooth, susc, "-", color=cmap(norm_global(row["s"])))
    axes[2].set(
        xscale="log",
        xlabel="$k_{total}$",
        ylabel="Susceptibility (Composition)",
        title="C. Compositional Change Rate",
    )

    for ax in axes:
        ax.grid(True, which="both", ls="--")
    filename = f"Fig_C_Susceptibility_Analysis_phi_{phi_val:.2f}.png".replace(".", "p")
    plt.savefig(os.path.join(figs_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


def plot_annotated_phase_boundary(phi_val, fit_results, figs_dir):
    print(
        f"  -> Generating Figure D: Annotated Phase Boundary for phi={phi_val:.2f}..."
    )
    df_crit = pd.DataFrame(fit_results).sort_values("s")
    if df_crit.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(
        df_crit["s"], df_crit["k_c_fit"], "o-", ms=8, color="crimson", lw=2.5, zorder=10
    )
    min_kc_idx = df_crit["k_c_fit"].idxmin()
    s_min, kc_min = df_crit.loc[min_kc_idx, "s"], df_crit.loc[min_kc_idx, "k_c_fit"]
    ax.plot(
        s_min,
        kc_min,
        "*",
        color="gold",
        ms=20,
        markeredgecolor="black",
        label="Most Fragile State",
        zorder=11,
    )
    ax.axvspan(-0.8, -0.22, color="royalblue", alpha=0.1, label="Geometry-Dominated")
    ax.axvspan(-0.15, 0.0, color="mediumseagreen", alpha=0.1, label="Drift-Dominated")
    ax.set(
        xlabel="Selection ($s = b_m - 1$)",
        ylabel="Critical Switching Rate ($k_c$)",
        title=f"Phase Boundary for $\\phi = {phi_val:.2f}$",
        yscale="log",
    )
    ax.legend(fontsize=12)
    ax.grid(True, which="both", ls="--")
    filename = f"Fig_D_Phase_Boundary_phi_{phi_val:.2f}.png".replace(".", "p")
    plt.savefig(os.path.join(figs_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run generalized hierarchical analysis for multiple phi-slices."
    )
    parser.add_argument("global_exp", help="Name of the global (coarse) experiment.")
    parser.add_argument("focused_exp", help="Name of the focused (fine) experiment.")
    args = parser.parse_args()

    global_config, focused_config = (
        EXPERIMENTS[args.global_exp],
        EXPERIMENTS[args.focused_exp],
    )
    FIGS_DIR = os.path.join(
        project_root,
        "figures",
        f"hierarchical_analysis_{focused_config['CAMPAIGN_ID']}",
    )
    ANALYSIS_DIR = os.path.join(
        project_root, "data", f"hierarchical_analysis_{focused_config['CAMPAIGN_ID']}"
    )
    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    df_global_raw = aggregate_data(global_config["CAMPAIGN_ID"], ANALYSIS_DIR)
    df_focused_raw = aggregate_data(focused_config["CAMPAIGN_ID"], ANALYSIS_DIR)
    if df_global_raw is None or df_focused_raw is None:
        sys.exit("FATAL: One or both required datasets are missing.")

    df_global_avg = average_metrics(df_global_raw)
    df_global_avg["s"] = df_global_avg["b_m"] - 1.0
    df_focused_avg = average_metrics(df_focused_raw)
    df_focused_avg["s"] = df_focused_avg["b_m"] - 1.0

    phi_slices = sorted(df_global_avg["phi"].unique())

    for phi in phi_slices:
        print(f"\n{'='*20} Analyzing slice: phi = {phi:.2f} {'='*20}")
        global_slice = df_global_avg[df_global_avg["phi"] == phi].copy()
        focused_slice = df_focused_avg[df_focused_avg["phi"] == phi].copy()

        if global_slice.empty or focused_slice.empty:
            continue

        print("\n--- Stage 1: Learning global models ---")
        model_A_density, model_B_density, df_plateaus_density = (
            fit_global_sigmoid_models(global_slice, "avg_interface_density")
        )
        model_A_rho, model_B_rho, df_plateaus_rho = fit_global_sigmoid_models(
            global_slice, "avg_rho_M"
        )

        if model_A_density is None:
            print("  -> Skipping: could not learn global models for interface density.")
            continue

        print("\n--- Stage 2: Performing constrained fit on high-resolution data ---")
        fit_results_local_density = fit_local_constrained_general(
            focused_slice, model_A_density, model_B_density, "avg_interface_density"
        )
        fit_results_local_rho = None
        if model_A_rho is not None:
            fit_results_local_rho = fit_local_constrained_general(
                focused_slice, model_A_rho, model_B_rho, "avg_rho_M"
            )

        if not fit_results_local_density:
            print(
                "  -> Skipping: could not perform high-resolution constrained fits for density."
            )
            continue

        print("\n--- Stage 3: Generating Publication-Ready Plot Suite ---")
        plot_model_validation(
            phi, df_plateaus_density, model_A_density, model_B_density, FIGS_DIR
        )
        plot_global_fits_on_data(
            phi, global_slice, df_plateaus_density, df_plateaus_rho, FIGS_DIR
        )
        plot_susceptibility_analysis(
            phi,
            df_plateaus_density,
            df_plateaus_rho,
            fit_results_local_density,
            fit_results_local_rho,
            FIGS_DIR,
        )
        plot_annotated_phase_boundary(phi, fit_results_local_density, FIGS_DIR)

    print(f"\n--- Analysis Complete. Final figures saved to: {FIGS_DIR} ---")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
