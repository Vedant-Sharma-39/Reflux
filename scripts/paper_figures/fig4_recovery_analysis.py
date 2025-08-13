# FILE: scripts/paper_figures/fig4_environmental_tuning.py
# Generates Figure 4, demonstrating that the optimal switching strategy is
# tuned to the timescale of environmental fluctuations.

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
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def main():
    project_root = get_project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.config import EXPERIMENTS

    try:
        campaign_id = EXPERIMENTS["bet_hedging_final"]["campaign_id"]
    except KeyError:
        print(
            "Error: 'bet_hedging_final' experiment not found in src/config.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    summary_path = os.path.join(
        project_root,
        "data",
        campaign_id,
        "analysis",
        f"{campaign_id}_summary_aggregated.csv",
    )
    figure_dir = os.path.join(project_root, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    output_path = os.path.join(figure_dir, "fig4_environmental_tuning.png")

    print(f"Generating Figure 4 from campaign: {campaign_id}")
    df = pd.read_csv(summary_path)

    # --- Use the definitive two-step filtering protocol ---
    STALL_THRESHOLD = 0.20
    stall_counts = df[df["termination_reason"] == "stalled_or_boundary_hit"][
        "phi"
    ].value_counts()
    total_counts = df["phi"].value_counts()
    stall_rates = stall_counts.div(total_counts, fill_value=0)
    phi_to_exclude = stall_rates[stall_rates > STALL_THRESHOLD].index.tolist()
    if phi_to_exclude:
        df = df[~df["phi"].isin(phi_to_exclude)].copy()
    df_filtered = df[df["termination_reason"] != "stalled_or_boundary_hit"].copy()

    df_filtered["s"] = df_filtered["b_m"] - 1.0

    # --- Select data for this specific figure ---
    s_target = -0.5
    phi_target = 0.0

    s_all = np.sort(df_filtered["s"].unique())
    phi_all = np.sort(df_filtered["phi"].unique())

    s_val = find_nearest(s_all, s_target)
    phi_val = find_nearest(phi_all, phi_target)

    df_plot_data = df_filtered[
        (np.isclose(df_filtered["s"], s_val))
        & (np.isclose(df_filtered["phi"], phi_val))
        & (df_filtered["k_total"] > 0)
    ].copy()

    patch_widths = np.sort(df_plot_data["patch_width"].unique())

    # --- Plotting Setup ---
    sns.set_theme(style="ticks", context="talk")
    fig = plt.figure(figsize=(22, 7), constrained_layout=True)
    gs = fig.add_gridspec(1, 4)
    fig.suptitle(
        f"Optimal Switching Strategy is Tuned to the Environmental Timescale (s={s_val:.2f}, $\\phi={phi_val:.2f}$)",
        fontsize=28,
        y=1.06,
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
    df_opt = df_mean.loc[opt_idx]
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
    # Add a linear regression fit
    slope, intercept, r_value, _, _ = linregress(df_opt["env_freq"], df_opt["k_total"])
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
