# FILE: scripts/paper_figures/fig4_data_collapse.py
# Generates a focused, two-panel figure demonstrating the universal scaling collapse
# of mutant fraction curves for a single, representative selection strength.

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
        campaign_id = EXPERIMENTS["phase_diagram"]["campaign_id"]
    except KeyError:
        print(
            "Error: 'phase_diagram' experiment not found in src/config.py.",
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
    output_path = os.path.join(figure_dir, "fig4_data_collapse.png")

    print(f"Generating data collapse figure from campaign: {campaign_id}")
    try:
        df = pd.read_csv(summary_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(
            f"Error: Data not found or empty for campaign '{campaign_id}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    df["s"] = df["b_m"] - 1.0

    # --- The Core Scaling Calculation ---
    # ρ_theory is the expected mutant fraction for k >> 1, given by (1 - phi) / 2
    theoretical_max = (1 - df["phi"]) / 2.0
    df["rho_M_scaled"] = np.divide(
        df["avg_rho_M"],
        theoretical_max,
        out=np.full_like(df["avg_rho_M"], np.nan),
        where=theoretical_max != 0,
    )

    sns.set_theme(style="ticks", context="talk")
    # --- MODIFIED: A single row of two panels ---
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 7), sharex=True, sharey=False, constrained_layout=True
    )
    fig.suptitle(
        "Universal Scaling Behavior Governs the Mutant Invasion Transition",
        fontsize=26,
        y=1.04,
    )

    s_all = np.sort(df["s"].unique())
    # --- MODIFIED: Focus on the single, dramatic s = -0.80 case ---
    s_val = find_nearest(s_all, -0.8)

    df_s_slice = df[np.isclose(df["s"], s_val)]

    ax_raw, ax_scaled = axes[0], axes[1]

    # --- Panel A (Left): Raw Data ---
    sns.lineplot(
        data=df_s_slice,
        x="k_total",
        y="avg_rho_M",
        hue="phi",
        palette="viridis",
        marker="o",
        lw=3,
        ms=9,
        ax=ax_raw,
        legend=True,
    )
    ax_raw.set_xscale("log")
    ax_raw.set_ylabel(r"Final Mutant Fraction, $\langle\rho_M\rangle$", fontsize=18)
    ax_raw.set_title(f"(A) Raw Data (s = {s_val:.2f})", fontsize=20)
    ax_raw.set_xlabel("Switching Rate, $k$", fontsize=18)
    if ax_raw.get_legend() is not None:
        ax_raw.get_legend().set_title(r"Bias, $\phi$")

    # --- Panel B (Right): Scaled Data (The Collapse) ---
    sns.lineplot(
        data=df_s_slice,
        x="k_total",
        y="rho_M_scaled",
        hue="phi",
        palette="viridis",
        marker="o",
        lw=3,
        ms=9,
        ax=ax_scaled,
        legend=False,
    )
    ax_scaled.set_xscale("log")
    ax_scaled.set_ylabel(
        r"Scaled Fraction, $\langle\rho_M\rangle / \rho_{theory}$", fontsize=18
    )
    ax_scaled.set_title("(B) Universal Scaling Collapse", fontsize=20)
    ax_scaled.axhline(1.0, color="black", ls="--", lw=2.5, label=r"$\rho_{theory}$")
    ax_scaled.set_ylim(-0.05, 1.2)
    ax_scaled.set_xlabel("Switching Rate, $k$", fontsize=18)
    ax_scaled.legend()

    sns.despine(fig)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure 4 (Data Collapse) saved to: {output_path}")


if __name__ == "__main__":
    main()
