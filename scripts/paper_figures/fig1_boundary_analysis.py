# FILE: scripts/paper_figures/fig1_boundary_analysis.py (Final Corrected Version)

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import linregress
from tqdm import tqdm


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def process_trajectories(df, column_name):
    if column_name not in df.columns:
        print(
            f"Warning: Column '{column_name}' not found. Skipping processing.",
            file=sys.stderr,
        )
        return pd.Series([None] * len(df), index=df.index)
    trajectories = []
    for item in tqdm(df[column_name], desc=f"Parsing '{column_name}'"):
        if pd.isna(item):
            trajectories.append(None)
            continue
        try:
            trajectories.append(json.loads(item))
        except (json.JSONDecodeError, TypeError):
            trajectories.append(None)
    return trajectories


def calculate_drift_diffusion(df_calib):
    if df_calib.empty:
        print("Warning: Calibration data is empty. Skipping drift/diffusion analysis.")
        return pd.DataFrame()

    print("Analyzing sector drift and diffusion from 'calibration' data...")
    df_calib["s"] = df_calib["b_m"] - 1.0
    df_calib["trajectory_data"] = process_trajectories(df_calib, "trajectory")
    analysis_results = []
    for params, group in tqdm(
        df_calib.groupby(["s"]), desc="Calculating D_eff and v_drift"
    ):
        # --- THE CRITICAL FIX ---
        # Unpack the single-element tuple returned by groupby
        s = params[0]
        # --- END FIX ---
        for _, row in group.dropna(subset=["trajectory_data"]).iterrows():
            if not row["trajectory_data"]:
                continue
            q_vals = np.array([t[0] for t in row["trajectory_data"]])
            width_vals = np.array([t[1] for t in row["trajectory_data"]])
            if len(q_vals) > 5:
                v_slope, _, _, _, _ = linregress(q_vals, width_vals)
                d_slope, _, d_r, _, _ = linregress(q_vals, width_vals**2)
                if d_r**2 > 0.8:
                    analysis_results.append(
                        {"s": s, "v_drift": v_slope, "D_eff": d_slope}
                    )
    return pd.DataFrame(analysis_results)


def analyze_roughness(df_kpz):
    if df_kpz.empty:
        print("Warning: KPZ scaling data is empty. Skipping roughness analysis.")
        return pd.DataFrame()

    print("Analyzing interface roughness from 'diffusion' data...")
    df_kpz["s"] = df_kpz["b_m"] - 1.0
    df_kpz["roughness_data"] = process_trajectories(df_kpz, "roughness_sq_trajectory")
    saturation_results = []
    for params, group in tqdm(
        df_kpz.groupby(["s", "width"]), desc="Calculating saturated roughness"
    ):
        s, width = params
        w2_sats = []
        for _, row in group.dropna(subset=["roughness_data"]).iterrows():
            if not row["roughness_data"]:
                continue
            w2_vals = np.array([t[1] for t in row["roughness_data"]])
            if len(w2_vals) > 20:
                w2_sats.append(np.mean(w2_vals[-len(w2_vals) // 4 :]))
        if w2_sats:
            saturation_results.append(
                {"s": s, "L": width, "W2_sat_mean": np.mean(w2_sats)}
            )
    return pd.DataFrame(saturation_results)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 1: Boundary Dynamics & KPZ Scaling."
    )
    parser.add_argument("calib_campaign")
    parser.add_argument("kpz_campaign")
    args = parser.parse_args()
    project_root = get_project_root()
    path_calib = os.path.join(
        project_root,
        "data",
        args.calib_campaign,
        "analysis",
        f"{args.calib_campaign}_summary_aggregated.csv",
    )
    path_kpz = os.path.join(
        project_root,
        "data",
        args.kpz_campaign,
        "analysis",
        f"{args.kpz_campaign}_summary_aggregated.csv",
    )
    output_path = os.path.join(
        project_root,
        "data",
        args.calib_campaign,
        "analysis",
        "figure1_boundary_analysis.png",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        df_calib = pd.read_csv(path_calib)
        df_kpz = pd.read_csv(path_kpz)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_calib, df_kpz = pd.DataFrame(), pd.DataFrame()

    if df_calib.empty and df_kpz.empty:
        print(
            "Warning: Both calibration and KPZ data are empty. Cannot generate Figure 1."
        )
        fig, ax = plt.subplots()
        ax.text(
            0.5,
            0.5,
            "Figure 1: No Data Available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        plt.savefig(output_path, dpi=300)
        sys.exit(0)

    drift_diff_df = calculate_drift_diffusion(df_calib)
    roughness_df = analyze_roughness(df_kpz)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    fig.suptitle("Figure 1: Boundary Dynamics and Front Morphology", fontsize=20)
    axA, axB = axes[0], axes[1]

    if not drift_diff_df.empty:
        sns.lineplot(
            data=drift_diff_df,
            x="s",
            y="D_eff",
            ax=axA,
            marker="o",
            label=r"$D_{eff}$ (Diffusion)",
            color="crimson",
        )
        axA.set_ylabel(r"Effective Diffusion, $D_{eff}$", color="crimson")
        axA2 = axA.twinx()
        sns.lineplot(
            data=drift_diff_df,
            x="s",
            y="v_drift",
            ax=axA2,
            marker="s",
            label=r"$v_{drift}$ (Drift)",
            color="navy",
        )
        axA2.plot(axA2.get_xlim(), axA2.get_xlim(), "k--", alpha=0.6, label="v = s")
        axA2.set_ylabel(r"Drift Velocity, $v_{drift}$", color="navy")
    else:
        axA.text(
            0.5,
            0.5,
            "No data for Panel A",
            ha="center",
            va="center",
            transform=axA.transAxes,
            color="gray",
        )
    axA.set_title("Effective Boundary Motion")
    axA.set_xlabel("Selection Coefficient, $s = b_m - 1$")

    if not roughness_df.empty:
        sns.lineplot(
            data=roughness_df[np.isclose(roughness_df["s"], 0.0)],
            x="L",
            y="W2_sat_mean",
            ax=axB,
            marker="o",
            label=r"$s=0.0$",
        )
        sns.lineplot(
            data=roughness_df[roughness_df["s"] < 0.0],
            x="L",
            y="W2_sat_mean",
            ax=axB,
            marker="s",
            label=r"$s < 0$",
        )
        axB.set_xscale("log")
        axB.set_yscale("log")
        axB.legend()
    else:
        axB.text(
            0.5,
            0.5,
            "No data for Panel B",
            ha="center",
            va="center",
            transform=axB.transAxes,
            color="gray",
        )
    axB.set_title(r"Interface Roughness Scaling (KPZ)")
    axB.set_xlabel("System Width, $L$")
    axB.set_ylabel(r"Saturated Roughness, $\langle W^2_{sat} \rangle$")

    plt.savefig(output_path, dpi=300)
    print(f"\nFigure 1 saved to {output_path}")


if __name__ == "__main__":
    main()
