# FILE: scripts/paper_figures/fig4_summary.py (Final Empirical Summary)

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


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
    output_path = os.path.join(figure_dir, "fig4_summary_of_findings.png")

    df = pd.read_csv(summary_path)
    df["s"] = df["b_m"] - 1.0
    df = df[(df["phi"] > -0.99) & (df["phi"] < 0.99)].copy()

    # --- Robust Empirical Rho_max Calculation ---
    top_k_values = sorted(df["k_total"].unique())[-4:]
    df_rho_max = (
        df[df["k_total"].isin(top_k_values)]
        .groupby(["s", "phi"])["avg_rho_M"]
        .median()
        .reset_index()
    )
    df_rho_max = df_rho_max.rename(columns={"avg_rho_M": "rho_max_empirical"})
    df_rho_max["F_empirical"] = (
        2 * df_rho_max["rho_max_empirical"] - 1 + df_rho_max["phi"]
    )
    df = pd.merge(df, df_rho_max, on=["s", "phi"], how="left")
    df["rho_M_scaled"] = df["avg_rho_M"] / df["rho_max_empirical"]

    # --- Create the 2x2 Summary Figure ---
    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(22, 20), constrained_layout=True)
    fig.suptitle(
        "Summary of Findings: The Empirical Force of Spatial Selection",
        fontsize=32,
        y=1.03,
    )

    # --- Panel A: Raw Data ---
    axA = axes[0, 0]
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
    axB = axes[0, 1]
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

    # --- Panel C: Empirical Force vs. Selection Strength ---
    axC = axes[1, 0]
    sns.lineplot(
        data=df_rho_max,
        x="s",
        y="F_empirical",
        hue="phi",
        palette="viridis",
        marker="o",
        ms=12,
        lw=4,
        ax=axC,
    )
    axC.axhline(0, color="black", linestyle="--", lw=2)
    axC.set_title("(C) Force vs. Selection Strength", fontsize=20)
    axC.set_xlabel("Selection Strength, s", fontsize=16)
    axC.set_ylabel(r"Empirical Force, $F_{emp}$", fontsize=16)
    axC.legend(title="Bias, φ")

    # --- Panel D: Empirical Force vs. Switching Bias ---
    axD = axes[1, 1]
    s_values_all = sorted(df_rho_max["s"].unique())
    s_palette = sns.color_palette("coolwarm_r", n_colors=len(s_values_all))
    sns.lineplot(
        data=df_rho_max,
        x="phi",
        y="F_empirical",
        hue="s",
        palette=s_palette,
        marker="o",
        ms=12,
        lw=4,
        ax=axD,
    )
    axD.axhline(0, color="black", linestyle="--", lw=2)
    axD.set_title("(D) Force vs. Switching Bias", fontsize=20)
    axD.set_xlabel("Switching Bias, φ", fontsize=16)
    axD.set_ylabel(r"Empirical Force, $F_{emp}$", fontsize=16)
    handles, labels = axD.get_legend_handles_labels()
    # Format labels to be cleaner
    formatted_labels = [f"s = {float(l):.2f}" for l in labels]
    axD.legend(handles[::-1], formatted_labels[::-1], title="Selection, s")

    # --- Final Touches ---
    for ax in axes.flat:
        ax.grid(True, linestyle=":")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFinal summary figure saved to: {output_path}")


if __name__ == "__main__":
    main()
