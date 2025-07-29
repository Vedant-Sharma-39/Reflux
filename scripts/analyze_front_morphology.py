# FILE: scripts/analyze_front_morphology.py
#
# A clean, high-performance script to analyze the global front morphology as a
# function of selection 's'. It calculates the KPZ scaling exponents (alpha and beta)
# for each selection value in parallel.

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
# 1. DATA LOADING (PARALLEL)
# ==============================================================================
def read_json_worker(filepath):
    """Hardened worker for reading a single JSON file."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        # Safely parse trajectory string if it exists
        if "roughness_trajectory" in data and isinstance(
            data["roughness_trajectory"], str
        ):
            data["roughness_trajectory"] = ast.literal_eval(
                data["roughness_trajectory"]
            )
        return data
    except Exception as e:
        print(
            f"Warning: Could not process file {os.path.basename(filepath)}. Error: {e}",
            file=sys.stderr,
        )
        return None


def load_and_preprocess_data(campaign_id):
    """Loads all raw JSON data for a campaign in parallel."""
    results_dir = os.path.join(project_root, "data", campaign_id, "results")
    if not os.path.isdir(results_dir):
        sys.exit(f"Error: Results directory not found at {results_dir}.")

    filepaths = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".json")
    ]
    if not filepaths:
        sys.exit("Error: No result files found.")

    with Pool(processes=max(1, cpu_count() - 2)) as pool:
        all_results = list(
            tqdm(
                pool.imap_unordered(read_json_worker, filepaths),
                total=len(filepaths),
                desc="Loading JSONs",
            )
        )

    df_raw = pd.DataFrame(
        [r for r in all_results if r and r.get("roughness_trajectory")]
    ).dropna(subset=["b_m", "width"])
    if df_raw.empty:
        sys.exit("No valid roughness trajectories found in the data.")

    df_raw["s"] = df_raw["b_m"] - 1.0
    return df_raw


# ==============================================================================
# 2. ANALYSIS (PARALLEL)
# ==============================================================================
def analyze_s_value(s_val, group_df):
    """[PARALLEL WORKER] Calculates alpha and beta for a single s-value."""
    try:
        long_form_data = []
        for _, row in group_df.iterrows():
            for q, w_sq in row.get("roughness_trajectory", []):
                long_form_data.append({"L": row["width"], "q": q, "W_sq": w_sq})
        if not long_form_data:
            return None

        df = pd.DataFrame(long_form_data)
        max_q = df["q"].max()
        if max_q <= 1:
            return None

        bins = np.logspace(0, np.log10(max_q), 100)
        df["q_bin"] = pd.cut(df["q"], bins)
        avg_df = (
            df.groupby(["L", "q_bin"], observed=True)
            .agg(q_mean=("q", "mean"), W_sq_mean=("W_sq", "mean"))
            .dropna()
            .reset_index()
        )

        # --- Beta Calculation ---
        beta_measured = np.nan
        beta_fit_df = avg_df[avg_df["L"] == avg_df["L"].max()]
        beta_fit_df = beta_fit_df[
            (beta_fit_df["q_mean"] > 10)
            & (beta_fit_df["q_mean"] < 100)
            & (beta_fit_df["W_sq_mean"] > 0)
        ]
        if len(beta_fit_df) > 3:
            slope, _, _, _, _ = linregress(
                np.log10(beta_fit_df["q_mean"]), np.log10(beta_fit_df["W_sq_mean"])
            )
            beta_measured = slope / 2.0

        # --- Alpha Calculation ---
        alpha_measured = np.nan
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
            slope, _, _, _, _ = linregress(
                np.log10(sat_df["L"]), np.log10(sat_df["W_sat_sq"])
            )
            alpha_measured = slope / 2.0

        return {"s": s_val, "alpha": alpha_measured, "beta": beta_measured}
    except Exception as e:
        print(f"Warning: Failed to analyze s={s_val}. Error: {e}", file=sys.stderr)
        return None


# ==============================================================================
# 3. PLOTTING AND MAIN EXECUTION
# ==============================================================================
def plot_summary(df_final, output_dir):
    """Generates the final summary plot of exponents vs. selection."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig.suptitle("Scaling Exponents vs. Selection Coefficient", fontsize=20)

    axes[0].plot(
        df_final["s"], df_final["alpha"], "o-", color="crimson", label="Measured α"
    )
    axes[0].axhline(
        THEORY_ALPHA, ls="--", color="gray", label=f"KPZ Theory (α={THEORY_ALPHA:.2f})"
    )
    axes[0].set(
        xlabel="Selection (s)", ylabel="Roughness Exponent (α)", title="α vs. Selection"
    )
    axes[0].legend()

    axes[1].plot(
        df_final["s"], df_final["beta"], "o-", color="darkblue", label="Measured β"
    )
    axes[1].axhline(
        THEORY_BETA, ls="--", color="gray", label=f"KPZ Theory (β={THEORY_BETA:.2f})"
    )
    axes[1].set(
        xlabel="Selection (s)", ylabel="Growth Exponent (β)", title="β vs. Selection"
    )
    axes[1].legend()

    for ax in axes:
        ax.grid(True, which="both", ls="--")
        ax.set_ylim(0, 1.0)
        ax.axvline(0, color="k", lw=0.5, ls="-")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_dir, "Fig_Exponents_vs_Selection.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Final summary plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze front morphology vs. selection."
    )
    parser.add_argument(
        "experiment_name", default="front_morphology_vs_selection", nargs="?"
    )
    args = parser.parse_args()

    config = EXPERIMENTS[args.experiment_name]
    CAMPAIGN_ID = config["CAMPAIGN_ID"]
    output_dir = os.path.join(project_root, "data", CAMPAIGN_ID, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load data
    df_raw = load_and_preprocess_data(CAMPAIGN_ID)

    # Step 2: Analyze in parallel
    print("\n--- Starting parallel analysis of exponents ---")
    analysis_tasks = [(s, group) for s, group in df_raw.groupby("s")]
    with Pool(processes=max(1, cpu_count() - 2)) as pool:
        results = list(
            tqdm(
                pool.starmap(analyze_s_value, analysis_tasks),
                total=len(analysis_tasks),
                desc="Calculating Exponents",
            )
        )

    # Step 3: Aggregate, save, and plot
    df_final = pd.DataFrame([res for res in results if res]).sort_values("s").dropna()
    summary_path = os.path.join(output_dir, "scaling_exponents_vs_selection.csv")
    df_final.to_csv(summary_path, index=False)
    print(f"\nSaved summary of exponents to {summary_path}")

    plot_summary(df_final, output_dir)


if __name__ == "__main__":
    main()
