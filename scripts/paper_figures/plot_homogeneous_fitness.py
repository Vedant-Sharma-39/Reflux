# FILE: scripts/paper_figures/plot_final_two_panel_synthesis.py
#
# Generates the final, definitive two-panel synthesis figure.
# Panel A shows the absolute fitness landscape in the fluctuating (w=60) environment,
# establishing the concept of an optimal switching rate.
# Panel B directly compares the relative fitness cost of selection in the stable
# vs. the fluctuating environment for key strategic regimes.

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


def get_project_root():
    """Dynamically finds the project root directory and adds it to the path."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


PROJECT_ROOT = get_project_root()
from src.config import EXPERIMENTS
from src.io.data_loader import load_aggregated_data


def find_nearest(array, value):
    """Finds the nearest value in a sorted array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def process_data_for_relative_fitness(df, env_name_filter=None):
    """Helper function to calculate relative fitness, filtering for successful runs."""
    valid_reasons = ["boundary_hit", "converged"]
    df_successful = df[df["termination_reason"].isin(valid_reasons)].copy()
    if df_successful.empty:
        return pd.DataFrame()
    df_filtered = df_successful[np.isclose(df_successful["phi"], 0.0)].copy()
    if df_filtered.empty:
        return pd.DataFrame()
    df_filtered["s"] = df_filtered["b_m"] - 1.0
    df_baseline = (
        df_filtered[np.isclose(df_filtered["s"], 0.0)][["k_total", "avg_front_speed"]]
        .groupby("k_total")
        .mean()
        .reset_index()
    )
    df_baseline = df_baseline.rename(columns={"avg_front_speed": "F_baseline"})
    df_norm = pd.merge(df_filtered, df_baseline, on="k_total", how="left")
    df_norm.dropna(subset=["F_baseline"], inplace=True)
    df_norm["relative_fitness"] = df_norm["avg_front_speed"] / df_norm["F_baseline"]
    return df_norm


def main():
    # 1. --- Load Data ---
    bh_campaign_id = EXPERIMENTS["bet_hedging_final"]["campaign_id"]
    homo_campaign_id = EXPERIMENTS["homogeneous_fitness_cost"]["campaign_id"]
    df_bh = load_aggregated_data(bh_campaign_id, PROJECT_ROOT)
    df_homo = load_aggregated_data(homo_campaign_id, PROJECT_ROOT)
    if df_bh.empty or df_homo.empty:
        sys.exit("Error: Data for one or both campaigns is missing.")
    df_bh.columns = df_bh.columns.str.strip()
    df_homo.columns = df_homo.columns.str.strip()
    figure_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    output_path = os.path.join(
        figure_dir, "fig5_interplay_of_switching_and_selection.png"
    )

    # 2. --- Process Data ---
    # Data for Panel A
    df_bh_abs = df_bh[
        (df_bh["env_definition"] == "symmetric_refuge_60w")
        & np.isclose(df_bh["phi"], 0.0)
        & (df_bh["termination_reason"] == "converged")
    ].copy()

    # Data for Panel B
    df_norm_homo = process_data_for_relative_fitness(df_homo)
    df_norm_homo["Environment"] = "Stable"
    df_norm_bh = process_data_for_relative_fitness(
        df_bh, env_name_filter="symmetric_refuge_60w"
    )
    df_norm_bh["Environment"] = "Fluctuating"
    df_combined = pd.concat([df_norm_homo, df_norm_bh])

    # 3. --- Define Regimes and Visual Styles ---
    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(28, 10))
    fig.suptitle(
        "The Bet-Hedging Trade-Off: Benefit in Fluctuating vs. Cost in Stable Environments",
        fontsize=32,
        y=1.03,
    )

    k_all = np.sort(df_combined["k_total"].unique())
    k_regimes_vals = [
        find_nearest(k_all, 0.05),
        find_nearest(k_all, 0.3),
        find_nearest(k_all, 10.0),
    ]
    custom_palette = {"Slow": "#4B0082", "Optimal": "#008080", "Fast": "#FFA500"}

    # --- Panel A: The Benefit ---
    axA = axes[0]
    s_all_bh = np.sort(df_bh_abs["b_m"].unique())
    s_val_bh = find_nearest(s_all_bh, 0.5)
    df_bh_abs_s_slice = df_bh_abs[np.isclose(df_bh_abs["b_m"], s_val_bh)]
    sns.lineplot(
        data=df_bh_abs_s_slice,
        x="k_total",
        y="avg_front_speed",
        ax=axA,
        color="black",
        linewidth=4,
        marker="o",
        markersize=10,
    )

    for i, (regime_name, k_val) in enumerate(
        zip(["Slow", "Optimal", "Fast"], k_regimes_vals)
    ):
        axA.axvline(
            x=k_val, color=custom_palette[regime_name], linestyle="--", lw=3, alpha=0.8
        )

    axA.set_title(
        f"(A) Benefit: Optimal Strategy in a Fluctuating Environment (b$_m$={s_val_bh:.2f})",
        fontsize=22,
        pad=15,
    )
    axA.set(
        xscale="log",
        xlabel="Switching Rate, $k_{total}$",
        ylabel="Absolute Fitness (Front Speed)",
    )
    axA.grid(True, which="both", linestyle=":")

    # --- Panel B: The Cost ---
    axB = axes[1]
    df_plot = df_combined[df_combined["k_total"].isin(k_regimes_vals)]
    df_plot["Regime"] = df_plot["k_total"].map(
        {
            k_regimes_vals[0]: "Slow",
            k_regimes_vals[1]: "Optimal",
            k_regimes_vals[2]: "Fast",
        }
    )

    sns.lineplot(
        data=df_plot,
        x="s",
        y="relative_fitness",
        ax=axB,
        hue="Regime",
        hue_order=["Slow", "Optimal", "Fast"],
        style="Environment",
        style_order=["Stable", "Fluctuating"],
        palette=custom_palette,
        linewidth=4,
        ci=95,
    )

    axB.axhline(1.0, color="red", lw=2.5, linestyle=":", zorder=0)
    axB.set_title(
        "(B) Cost: Selection Penalty in Stable vs. Fluctuating Environments",
        fontsize=22,
        pad=15,
    )
    axB.set(
        xlabel="Selection Strength, $s = b_m - 1$",
        ylabel="Relative Fitness (of viable fronts)",
        ylim=(0.4, 1.1),
    )
    axB.grid(True, which="both", linestyle=":")
    axB.legend(fontsize=16, title="Legend")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
