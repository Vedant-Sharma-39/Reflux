# FILE: scripts/analyze_front_morphology.py
#
# [REWRITTEN & CORRECTED FOR MULTIPROCESSING]
# This version fixes a common multiprocessing issue by having each worker
# process load its own required data instead of receiving large DataFrame
# chunks from the main process. This avoids serialization (pickling) errors
# and is more memory-efficient.
#
# It still calculates KPZ exponents in parallel and includes the optional
# diagnostic plotting feature.

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import ast
import argparse

# --- Configuration and Setup ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from config import EXPERIMENTS
except (NameError, ImportError) as e:
    sys.exit(f"FATAL: Could not import configuration. Error: {e}")

THEORY_BETA, THEORY_ALPHA = 1 / 3, 1 / 2
plt.style.use("seaborn-v0_8-whitegrid")


# ==============================================================================
# STAGE 1: PARALLEL ANALYSIS (CORRECTED WORKER)
# ==============================================================================
def read_json_worker(filepath: str):
    """Hardened worker function to read a single JSON file."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        if "roughness_trajectory" in data and isinstance(
            data["roughness_trajectory"], str
        ):
            data["roughness_trajectory"] = ast.literal_eval(
                data["roughness_trajectory"]
            )
        return data
    except Exception:
        return None


def analyze_s_value(s_val: float, campaign_id: str, proj_root: str):
    """
    [PARALLEL WORKER - CORRECTED]
    Analyzes a single 's' value. It loads its own data to avoid
    multiprocessing serialization issues.
    """
    try:
        b_m = s_val + 1.0
        results_dir = os.path.join(proj_root, "data", campaign_id, "results")

        # Worker finds its own files based on the s_val it's assigned
        b_m_str = f"bm{b_m:.3f}"
        filepaths = [
            os.path.join(results_dir, f)
            for f in os.listdir(results_dir)
            if b_m_str in f and f.endswith(".json")
        ]

        if not filepaths:
            return None

        # Load and process only the relevant data
        group_results = [read_json_worker(fp) for fp in filepaths]
        group_df = pd.DataFrame(
            [r for r in group_results if r and r.get("roughness_trajectory")]
        )

        if group_df.empty:
            return None

        # --- The rest of the analysis logic is the same as before ---
        long_form_data = []
        for _, row in group_df.iterrows():
            for q, w_sq in row.get("roughness_trajectory", []):
                long_form_data.append({"L": row["width"], "q": q, "W_sq": w_sq})
        if not long_form_data:
            return None

        df = pd.DataFrame(long_form_data)
        if df.empty or df["q"].max() <= 1:
            return None

        bins = np.logspace(0, np.log10(df["q"].max()), 100)
        df["q_bin"] = pd.cut(df["q"], bins)
        avg_df = (
            df.groupby(["L", "q_bin"], observed=True)
            .agg(q_mean=("q", "mean"), W_sq_mean=("W_sq", "mean"))
            .dropna()
            .reset_index()
        )

        # Beta Calculation
        beta_measured, beta_fit_params = np.nan, None
        beta_fit_df = avg_df[avg_df["L"] == avg_df["L"].max()]
        beta_fit_df = beta_fit_df[
            (beta_fit_df["q_mean"] > 10)
            & (beta_fit_df["q_mean"] < 100)
            & (beta_fit_df["W_sq_mean"] > 0)
        ]
        if len(beta_fit_df) > 3:
            slope, intercept, _, _, _ = linregress(
                np.log10(beta_fit_df["q_mean"]), np.log10(beta_fit_df["W_sq_mean"])
            )
            beta_measured = slope / 2.0
            beta_fit_params = {"slope": slope, "intercept": intercept}

        # Alpha Calculation
        alpha_measured, alpha_fit_params = np.nan, None
        saturation_data = []
        for l_val, group in avg_df.groupby("L"):
            sat_points = group[group["q_mean"] > 0.8 * group["q_mean"].max()]
            if not sat_points.empty:
                saturation_data.append(
                    {"L": l_val, "W_sat_sq": sat_points["W_sq_mean"].mean()}
                )
        sat_df = pd.DataFrame(saturation_data)
        sat_df = sat_df[sat_df["W_sat_sq"] > 0]
        if len(sat_df) > 2:
            slope, intercept, _, _, _ = linregress(
                np.log10(sat_df["L"]), np.log10(sat_df["W_sat_sq"])
            )
            alpha_measured = slope / 2.0
            alpha_fit_params = {"slope": slope, "intercept": intercept}

        return {
            "s": s_val,
            "alpha": alpha_measured,
            "beta": beta_measured,
            "plot_data": {
                "avg_df": avg_df.to_dict(),
                "sat_df": sat_df.to_dict(),
                "beta_fit_params": beta_fit_params,
                "alpha_fit_params": alpha_fit_params,
            },
        }
    except Exception as e:
        print(f"Warning: Failed to analyze s={s_val}. Error: {e}", file=sys.stderr)
        return None


# ==============================================================================
# STAGE 2: PLOTTING AND MAIN EXECUTION
# ==============================================================================
def generate_debug_plot(s_val: float, plot_data: dict, debug_dir: str):
    """Generates a detailed 2-panel diagnostic plot for a single 's' value."""
    # This function is correct but needs to reconstruct DataFrames from dicts
    avg_df = pd.DataFrame(plot_data["avg_df"])
    sat_df = pd.DataFrame(plot_data["sat_df"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9), constrained_layout=True)
    fig.suptitle(f"Front Morphology Fit Diagnosis for s = {s_val:.3f}", fontsize=20)

    # Panel 1: Beta (Growth) Fit
    colors = plt.cm.viridis(np.linspace(0, 1, avg_df["L"].nunique()))
    for i, (l_val, group) in enumerate(avg_df.groupby("L")):
        ax1.plot(
            group["q_mean"],
            group["W_sq_mean"],
            "o",
            ms=5,
            color=colors[i],
            alpha=0.7,
            label=f"L = {l_val}",
        )

    if plot_data["beta_fit_params"]:
        params = plot_data["beta_fit_params"]
        beta = params["slope"] / 2.0
        q_fit = np.logspace(1, 2.5, 50)
        w_fit = (10 ** params["intercept"]) * (q_fit ** params["slope"])
        ax1.plot(q_fit, w_fit, "r--", lw=2.5, label=f"Fit ($\\beta$={beta:.3f})")

    q_theory = np.array([10, 200])
    w_theory = 0.5 * q_theory ** (2 * THEORY_BETA)
    ax1.plot(
        q_theory, w_theory, "k:", lw=2.5, label=f"Theory ($\\beta$={THEORY_BETA:.2f})"
    )
    ax1.set(
        xscale="log",
        yscale="log",
        xlabel="Mean Front Position (q)",
        ylabel="Squared Interface Width $\\langle W^2 \\rangle$",
        title="A. Growth Phase Exponent ($\\beta$)",
    )
    ax1.legend()

    # Panel 2: Alpha (Saturation) Fit
    if not sat_df.empty:
        ax2.plot(
            sat_df["L"],
            sat_df["W_sat_sq"],
            "o",
            color="navy",
            ms=10,
            label="Measured $W^2_{sat}$",
        )
        if plot_data["alpha_fit_params"]:
            params = plot_data["alpha_fit_params"]
            alpha = params["slope"] / 2.0
            L_fit = np.array(sorted(sat_df["L"]))
            w_fit = (10 ** params["intercept"]) * (L_fit ** params["slope"])
            ax2.plot(L_fit, w_fit, "r--", lw=2.5, label=f"Fit ($\\alpha$={alpha:.3f})")
        L_theory = np.array([sat_df["L"].min(), sat_df["L"].max()])
        w_theory = 0.1 * L_theory ** (2 * THEORY_ALPHA)
        ax2.plot(
            L_theory,
            w_theory,
            "k:",
            lw=2.5,
            label=f"Theory ($\\alpha$={THEORY_ALPHA:.2f})",
        )
    ax2.set(
        xscale="log",
        yscale="log",
        xlabel="System Width (L)",
        ylabel="Saturated Squared Width $\\langle W^2_{sat} \\rangle$",
        title="B. Roughness Exponent ($\\alpha$)",
    )
    ax2.legend()

    for ax in (ax1, ax2):
        ax.grid(True, which="both", ls="--")
    plot_path = os.path.join(debug_dir, f"debug_morphology_fit_s_{s_val:.4f}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)


def plot_summary_results(df_final: pd.DataFrame, figs_dir: str):
    """Generates the final summary plot of exponents vs. selection."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig.suptitle("KPZ Scaling Exponents vs. Selection Coefficient", fontsize=20)
    axes[0].plot(
        df_final["s"],
        df_final["alpha"],
        "o-",
        color="crimson",
        label="Measured $\\alpha$",
    )
    axes[0].axhline(
        THEORY_ALPHA,
        ls="--",
        color="gray",
        label=f"KPZ Theory ($\\alpha$={THEORY_ALPHA:.2f})",
    )
    axes[0].set(
        xlabel="Selection (s)",
        ylabel="Roughness Exponent ($\\alpha$)",
        title="A. Roughness Exponent vs. Selection",
    )
    axes[1].plot(
        df_final["s"],
        df_final["beta"],
        "o-",
        color="darkblue",
        label="Measured $\\beta$",
    )
    axes[1].axhline(
        THEORY_BETA,
        ls="--",
        color="gray",
        label=f"KPZ Theory ($\\beta$={THEORY_BETA:.2f})",
    )
    axes[1].set(
        xlabel="Selection (s)",
        ylabel="Growth Exponent ($\\beta$)",
        title="B. Growth Exponent vs. Selection",
    )
    for ax in axes:
        ax.legend()
        ax.grid(True, which="both", ls="--")
        ax.set_ylim(0, 1.0)
        ax.axvline(0, color="k", lw=0.5, ls="-")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(figs_dir, "Fig_KPZ_Exponents_vs_Selection.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nFinal summary plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze front morphology and calculate KPZ exponents."
    )
    parser.add_argument(
        "experiment_name", default="front_morphology_vs_selection", nargs="?"
    )
    parser.add_argument(
        "--generate-debug-plots",
        action="store_true",
        help="Generate detailed diagnostic plots for each 's' value.",
    )
    args = parser.parse_args()

    config = EXPERIMENTS[args.experiment_name]
    CAMPAIGN_ID = config["CAMPAIGN_ID"]
    analysis_dir = os.path.join(project_root, "data", CAMPAIGN_ID, "analysis")
    figs_dir = os.path.join(project_root, "figures", CAMPAIGN_ID)
    debug_dir = os.path.join(figs_dir, "debug_morphology_fits")
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    if args.generate_debug_plots:
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug plots will be saved to: {debug_dir}")

    # --- Execute Analysis Pipeline (Corrected) ---
    # Get the list of s values directly from the config to create tasks.
    # This is more efficient than scanning all filenames.
    s_values_to_analyze = [bm - 1.0 for bm in config["PARAM_GRID"]["b_m_scan"]]
    analysis_tasks = [(s, CAMPAIGN_ID, project_root) for s in s_values_to_analyze]

    print("\n--- Starting parallel analysis of exponents ---")
    with Pool(processes=max(1, cpu_count() - 2)) as pool:
        results = list(
            tqdm(
                pool.starmap(analyze_s_value, analysis_tasks),
                total=len(analysis_tasks),
                desc="Calculating Exponents",
            )
        )

    valid_results = [res for res in results if res is not None]
    if not valid_results:
        sys.exit("Analysis complete, but no valid exponent data was generated.")

    if args.generate_debug_plots:
        print("\n--- Generating diagnostic plots ---")
        for res in tqdm(valid_results, desc="Plotting fits"):
            generate_debug_plot(res["s"], res["plot_data"], debug_dir)

    df_final = (
        pd.DataFrame(
            [
                {"s": r["s"], "alpha": r["alpha"], "beta": r["beta"]}
                for r in valid_results
            ]
        )
        .sort_values("s")
        .dropna()
    )
    summary_path = os.path.join(analysis_dir, "kpz_scaling_exponents_summary.csv")
    df_final.to_csv(summary_path, index=False)
    print(f"\nSaved summary of exponents to {summary_path}")

    plot_summary_results(df_final, figs_dir)


if __name__ == "__main__":
    main()
