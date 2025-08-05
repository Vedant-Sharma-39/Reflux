import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import linregress


def process_trajectories(df, column_name):
    """Safely parse JSON string trajectories into lists of floats."""
    trajectories = []
    for item in df[column_name]:
        try:
            # The data is stored as a string representation of a list
            trajectories.append(json.loads(item))
        except (json.JSONDecodeError, TypeError):
            trajectories.append(None)
    return trajectories


def calculate_deff(df_calib):
    """Calculates effective diffusion coefficient from sector width trajectories."""
    print("Calculating D_eff from calibration data...")
    df_calib["s"] = df_calib["b_m"] - 1.0
    df_calib["trajectory_data"] = process_trajectories(df_calib, "trajectory")

    deff_results = []
    for _, row in df_calib.dropna(subset=["trajectory_data"]).iterrows():
        q_vals = np.array([t[0] for t in row["trajectory_data"]])
        width_vals = np.array([t[1] for t in row["trajectory_data"]])

        # We expect width^2 to be linear with q. Slope is proportional to D_eff.
        if len(q_vals) > 5:
            slope, intercept, r_value, _, _ = linregress(q_vals, width_vals**2)
            if r_value**2 > 0.85:  # Filter for good linear fits
                deff_results.append({"s": row["s"], "D_eff": slope})

    return pd.DataFrame(deff_results)


def analyze_roughness(df_diff):
    """Analyzes interface roughness saturation from diffusion data."""
    print("Analyzing interface roughness from diffusion data...")
    df_diff["s"] = df_diff["b_m"] - 1.0
    df_diff["roughness_data"] = process_trajectories(df_diff, "roughness_trajectory")

    saturation_results = []
    for _, row in df_diff.dropna(subset=["roughness_data"]).iterrows():
        q_vals = np.array([t[0] for t in row["roughness_data"]])
        w2_vals = np.array([t[1] for t in row["roughness_data"]])

        # Use the last 25% of the data to estimate saturation
        if len(w2_vals) > 20:
            w2_sat = np.mean(w2_vals[-len(w2_vals) // 4 :])
            saturation_results.append(
                {"s": row["s"], "width": row["width"], "W2_sat": w2_sat}
            )

    return pd.DataFrame(saturation_results)


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots for Figure 1: Boundary Dynamics."
    )
    parser.add_argument(
        "campaign_ids",
        nargs="+",
        help="Campaign IDs for calibration and diffusion runs.",
    )
    args = parser.parse_args()

    # --- Load Data ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    df_calib = pd.DataFrame()
    df_diff = pd.DataFrame()

    for campaign_id in args.campaign_ids:
        summary_path = os.path.join(
            project_root,
            "data",
            campaign_id,
            "analysis",
            f"{campaign_id}_summary_aggregated.csv",
        )
        if not os.path.exists(summary_path):
            print(f"Warning: Summary file not found for {campaign_id}. Skipping.")
            continue

        df_temp = pd.read_csv(summary_path)
        if df_temp["run_mode"].iloc[0] == "calibration":
            df_calib = pd.concat([df_calib, df_temp], ignore_index=True)
        elif df_temp["run_mode"].iloc[0] == "diffusion":
            df_diff = pd.concat([df_diff, df_temp], ignore_index=True)

    if df_calib.empty or df_diff.empty:
        print("Error: Must provide data for both calibration and diffusion campaigns.")
        return

    # --- Process Data ---
    deff_df = calculate_deff(df_calib)
    roughness_df = analyze_roughness(df_diff)

    # --- Create Plots ---
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    fig.suptitle("Figure 1: Boundary Dynamics and Front Morphology", fontsize=20)

    # Plot 1: D_eff vs s
    sns.lineplot(
        data=deff_df,
        x="s",
        y="D_eff",
        ax=axes[0],
        marker="o",
        errorbar="sd",
        err_style="bars",
    )
    axes[0].set_title(r"Effective Diffusion Coefficient of Boundary")
    axes[0].set_xlabel("Selection Coefficient, $s = b_m - 1$")
    axes[0].set_ylabel(r"Effective Diffusion, $D_{eff}$")
    axes[0].axhline(0, color="grey", linestyle="--", lw=1.5)
    axes[0].axvline(0, color="grey", linestyle="--", lw=1.5)

    # Plot 2: Saturated Roughness W^2 vs System Width L
    roughness_df["L"] = roughness_df["width"]
    sns.lineplot(
        data=roughness_df[roughness_df["s"] == 0.0],
        x="L",
        y="W2_sat",
        ax=axes[1],
        marker="o",
        label=r"$s=0.0$ (Neutral)",
    )
    sns.lineplot(
        data=roughness_df[roughness_df["s"] < -0.1],
        x="L",
        y="W2_sat",
        ax=axes[1],
        marker="s",
        label=r"$s < -0.1$ (Deleterious)",
    )
    axes[1].set_title(r"Interface Roughness Scaling (KPZ)")
    axes[1].set_xlabel("System Width, $L$")
    axes[1].set_ylabel(r"Saturated Roughness, $W^2_{sat}$")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].legend()

    # --- Save Figure ---
    output_dir = os.path.join(project_root, "data", args.campaign_ids[0], "analysis")
    output_path = os.path.join(output_dir, "figure1_boundary_analysis.png")
    plt.savefig(output_path, dpi=300)
    print(f"Figure 1 saved to {output_path}")


if __name__ == "__main__":
    main()
