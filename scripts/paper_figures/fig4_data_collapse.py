# FILE: scripts/paper_figures/fig4_data_collapse.py (Final 1x2 Version)

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_project_root():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def main():
    project_root = get_project_root()
    from src.config import EXPERIMENTS

    campaign_id = EXPERIMENTS["phase_diagram"]["campaign_id"]
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

    df = pd.read_csv(summary_path)
    df["s"] = df["b_m"] - 1.0
    df = df[(df["phi"] < 0.99)].copy()

    # --- Robust Empirical Rho_max Calculation (for Panel B) ---
    top_k_values = sorted(df["k_total"].unique())[-4:]
    df_rho_max = (
        df[df["k_total"].isin(top_k_values)]
        .groupby(["s", "phi"])["avg_rho_M"]
        .median()
        .reset_index()
    )
    df_rho_max = df_rho_max.rename(columns={"avg_rho_M": "rho_max_empirical"})
    df = pd.merge(df, df_rho_max, on=["s", "phi"], how="left")
    df["rho_M_scaled"] = df["avg_rho_M"] / df["rho_max_empirical"]

    # --- Create the 1x2 Summary Figure ---
    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
    fig.suptitle(
        "Universal Scaling Collapse of Mutant Invasion Dynamics",
        fontsize=28,
        y=1.05,
    )

    # --- Panel A: Raw Data ---
    axA = axes[0]
    s_val_collapse = df["s"][np.abs(df["s"] - (-0.8)).idxmin()]
    df_s_slice = df[np.isclose(df["s"], s_val_collapse)]
    sns.lineplot(
        data=df_s_slice,
        x="k_total",
        y="avg_rho_M",
        hue="phi",
        palette="viridis",
        marker="o",
        lw=3,
        ms=9,
        ax=axA,
    )
    axA.set(
        xscale="log",
        xlabel="Switching Rate, $k$",
        ylabel=r"Final Mutant Fraction, $\langle\rho_M\rangle$",
    )
    axA.set_title(f"(A) Raw Data (s = {s_val_collapse:.2f})", fontsize=20)
    axA.legend(title="Bias, φ")

    # --- Panel B: Universal Scaling Collapse ---
    axB = axes[1]
    sns.lineplot(
        data=df_s_slice,
        x="k_total",
        y="rho_M_scaled",
        hue="phi",
        palette="viridis",
        marker="o",
        lw=3,
        ms=9,
        ax=axB,
        legend=False,
    )
    axB.axhline(1.0, color="black", ls="--", lw=2.5, label=r"$\rho_{max}$")
    axB.set(
        xscale="log",
        xlabel="Switching Rate, $k$",
        ylabel=r"Scaled Fraction, $\langle\rho_M\rangle / \rho_{max}$",
    )
    axB.set_title("(B) Universal Scaling Collapse", fontsize=20)
    axB.legend()

    # --- Final Touches ---
    for ax in axes.flat:
        ax.grid(True, linestyle=":")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure 4 (Data Collapse) saved to: {output_path}")


if __name__ == "__main__":
    main()
