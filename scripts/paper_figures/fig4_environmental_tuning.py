# FILE: scripts/paper_figures/fig4_environmental_tuning.py
# Generates Figure 4. This definitive version correctly infers the patch width
# by handling cases where the `env_definition` column is empty in the source data.

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress


def get_project_root():
    """Dynamically finds the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def find_nearest(array, value):
    """Finds the nearest value in a sorted array."""
    array = np.asarray(array)
    if array.size == 0:
        raise ValueError("Cannot find nearest value in an empty array.")
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def main():
    project_root = get_project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.config import EXPERIMENTS, PARAM_GRID

    main_campaign_id = EXPERIMENTS["bet_hedging_final"]["campaign_id"]
    controls_campaign_id = EXPERIMENTS["bet_hedging_controls"]["campaign_id"]
    campaign_ids = [main_campaign_id, controls_campaign_id]

    dfs = []
    for cid in campaign_ids:
        summary_path = os.path.join(
            project_root, "data", cid, "analysis", f"{cid}_summary_aggregated.csv"
        )
        if os.path.exists(summary_path):
            print(f"Loading data from: {summary_path}")
            dfs.append(pd.read_csv(summary_path))
        else:
            print(
                f"Warning: Could not find summary file at {summary_path}",
                file=sys.stderr,
            )

    if not dfs:
        print("Error: No data files found. Cannot generate plot.", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)

    figure_dir = os.path.join(project_root, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    output_path = os.path.join(figure_dir, "fig4_environmental_tuning.png")

    print(f"\nGenerating Figure 4 from campaigns: {campaign_ids}")

    df_filtered = df[df["termination_reason"] != "stalled_or_boundary_hit"].copy()

    bm_target = 0.5
    phi_target = 0.0
    bm_val = find_nearest(df_filtered["b_m"].unique(), bm_target)
    phi_val = find_nearest(df_filtered["phi"].unique(), phi_target)

    df_plot_data = df_filtered[
        (np.isclose(df_filtered["b_m"], bm_val))
        & (np.isclose(df_filtered["phi"], phi_val))
        & (df_filtered["k_total"] > 0)
    ].copy()

    # --- DEFINITIVE FIX: Engineer 'patch_width' using a robust two-step process ---
    # Step 1: Attempt to map from the `env_definition` column. This will work for the
    # `fig3_controls` data but will produce `NaN` for `fig3_bet_hedging_final` data
    # because that column is blank in its CSV.
    env_name_map = {
        PARAM_GRID["env_definitions"]["symmetric_strong_scan_bm_30w"]["name"]: 30,
        PARAM_GRID["env_definitions"]["symmetric_strong_scan_bm_60w"]["name"]: 60,
        PARAM_GRID["env_definitions"]["symmetric_strong_scan_bm_120w"]["name"]: 120,
    }
    df_plot_data["patch_width"] = df_plot_data["env_definition"].map(env_name_map)

    # Step 2: Fill in the missing values. We know any row from the main campaign
    # that is missing a patch_width must be the 60px environment.
    is_main_exp = df_plot_data["campaign_id"] == main_campaign_id
    df_plot_data.loc[is_main_exp, "patch_width"] = df_plot_data.loc[
        is_main_exp, "patch_width"
    ].fillna(60)

    # Now, drop any rows that are still missing a patch_width (none should remain)
    df_plot_data.dropna(subset=["patch_width"], inplace=True)
    df_plot_data["patch_width"] = df_plot_data["patch_width"].astype(int)

    patch_widths = np.sort(df_plot_data["patch_width"].unique())
    if len(patch_widths) == 0:
        print(
            "\nFATAL ERROR: No data found for the required patch widths after filtering.",
            file=sys.stderr,
        )
        print("This indicates a problem with the data loading or filtering logic.")
        sys.exit(1)

    # --- Plotting Setup ---
    sns.set_theme(style="ticks", context="talk")
    fig = plt.figure(figsize=(22, 7), constrained_layout=True)
    gs = fig.add_gridspec(1, 4)
    fig.suptitle(
        f"Figure 4: Optimal Strategy is Tuned to Environmental Fluctuation Timescale\n($b_m={bm_val:.2f}, \\phi={phi_val:.2f}$)",
        fontsize=28,
        y=1.08,
    )

    # --- Panel A: Fitness Landscapes ---
    axA = fig.add_subplot(gs[0, 0:2])
    palette = sns.color_palette("mako_r", n_colors=len(patch_widths))
    sns.lineplot(
        data=df_plot_data,
        x="k_total",
        y="avg_front_speed",
        hue="patch_width",
        palette=palette,
        marker="o",
        lw=3,
        ms=9,
        ax=axA,
    )
    axA.set_xscale("log")
    axA.set_title("(A) Fitness Landscapes for Different Frequencies", fontsize=20)
    axA.set_xlabel("Switching Rate, $k$", fontsize=16)
    axA.set_ylabel("Long-Term Fitness", fontsize=16)
    axA.grid(True, which="both", ls=":")
    axA.legend(title="Patch Width, w")

    # --- Analysis for Panels B & C ---
    df_mean = (
        df_plot_data.groupby(["patch_width", "k_total"])["avg_front_speed"]
        .mean()
        .reset_index()
    )
    opt_idx = df_mean.groupby("patch_width")["avg_front_speed"].idxmax()
    df_opt = df_mean.loc[opt_idx].copy()
    df_opt["env_freq"] = 1.0 / df_opt["patch_width"]

    # --- Panel B: Optimal Rate vs. Frequency ---
    axB = fig.add_subplot(gs[0, 2])
    sns.lineplot(
        data=df_opt,
        x="env_freq",
        y="k_total",
        marker="o",
        ms=12,
        lw=3,
        ax=axB,
        color="crimson",
    )
    if len(df_opt["env_freq"]) > 1 and len(df_opt["k_total"]) > 1:
        slope, intercept, r_value, _, _ = linregress(
            df_opt["env_freq"], df_opt["k_total"]
        )
        x_fit = np.linspace(df_opt["env_freq"].min(), df_opt["env_freq"].max(), 100)
        axB.plot(
            x_fit,
            slope * x_fit + intercept,
            "k--",
            label=f"Linear Fit ($R^2={r_value**2:.3f}$)",
        )
    axB.set_title("(B) Optimal Rate Tracks Frequency", fontsize=20)
    axB.set_xlabel("Environmental Frequency, 1/w", fontsize=16)
    axB.set_ylabel("Optimal Rate, $k_{opt}$", fontsize=16)
    axB.grid(True, ls=":")
    axB.legend()

    # --- Panel C: Data Collapse ---
    axC = fig.add_subplot(gs[0, 3])
    df_collapse = df_plot_data.copy()
    df_collapse["k_rescaled"] = df_collapse["k_total"] * df_collapse["patch_width"]
    df_collapse = pd.merge(
        df_collapse,
        df_opt[["patch_width", "avg_front_speed"]].rename(
            columns={"avg_front_speed": "f_max"}
        ),
        on="patch_width",
    )
    df_collapse["fitness_rescaled"] = (
        df_collapse["avg_front_speed"] / df_collapse["f_max"]
    )
    sns.lineplot(
        data=df_collapse,
        x="k_rescaled",
        y="fitness_rescaled",
        hue="patch_width",
        palette=palette,
        marker="o",
        lw=3,
        ms=9,
        ax=axC,
        legend=False,
    )
    axC.set_xscale("log")
    axC.set_title("(C) Universal Adaptive Landscape", fontsize=20)
    axC.set_xlabel("Rescaled Rate, $k \\cdot w$", fontsize=16)
    axC.set_ylabel("Rescaled Fitness, $F/F_{max}$", fontsize=16)
    axC.grid(True, which="both", ls=":")

    sns.despine(fig)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure 4 (Environmental Tuning) saved to: {output_path}")


if __name__ == "__main__":
    main()
