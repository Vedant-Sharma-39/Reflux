# FILE: scripts/paper_figures/sup_fig_stationary_distributions.py
# Generates a figure showing the stationary distribution of the mutant fraction
# for different switching rate regimes, analogous to Hufton et al. Fig. 1.

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
        # The bet_hedging_final campaign is the best source for this,
        # as it provides many replicates for each parameter set.
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
    output_path = os.path.join(figure_dir, "sup_fig_stationary_distributions.png")

    print(f"Generating stationary distribution figure from campaign: {campaign_id}")
    df = pd.read_csv(summary_path)

    # --- Use the definitive two-step filtering protocol for consistency ---
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
        & (df_filtered["patch_width"] == 60)
        & (df_filtered["k_total"] > 0)
    ].copy()

    # --- Plotting Setup ---
    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    fig.suptitle(
        f"Switching Rate Controls the Shape of the Stationary Distribution (s={s_val:.2f}, $\\phi={phi_val:.2f}$)",
        fontsize=28,
        y=1.05,
    )

    k_vals = np.sort(df_plot_data["k_total"].unique())
    k_slow = k_vals[0]
    k_intermediate = k_vals[len(k_vals) // 2]
    k_fast = k_vals[-1]

    k_regimes = [k_slow, k_intermediate, k_fast]
    titles = ["(A) Slow Switching", "(B) Intermediate Switching", "(C) Fast Switching"]

    for i, k_val in enumerate(k_regimes):
        ax = axes[i]
        data_slice = df_plot_data[np.isclose(df_plot_data["k_total"], k_val)]

        sns.histplot(
            data=data_slice, x="avg_rho_M", kde=True, stat="density", bins=15, ax=ax
        )

        ax.set_title(f"{titles[i]}\n$k={k_val:.3f}$", fontsize=20)
        ax.set_xlabel(r"Final Mutant Fraction, $\langle\rho_M\rangle$", fontsize=16)
        ax.set_xlim(-0.05, 1.05)

    axes[0].set_ylabel("Probability Density", fontsize=16)

    sns.despine(fig)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSupplementary Figure (Stationary Distributions) saved to: {output_path}")


if __name__ == "__main__":
    main()
