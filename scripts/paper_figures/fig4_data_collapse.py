# FILE: scripts/paper_figures/fig4_data_collapse.py (Final 1x2 Version)

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# --- Publication Settings ---
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def cm_to_inch(cm):
    return cm / 2.54


# --- End Publication Settings ---


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
    # --- CHANGE: Output filenames ---
    output_path_pdf = os.path.join(figure_dir, "fig4_data_collapse.pdf")
    output_path_eps = os.path.join(figure_dir, "fig4_data_collapse.eps")

    df = pd.read_csv(summary_path)
    df["s"] = df["b_m"] - 1.0
    df = df[(df["phi"] < 0.99)].copy()

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

    # --- CHANGE: Plotting setup for publication ---
    sns.set_theme(style="ticks", context="paper")
    fig, axes = plt.subplots(
        1, 2, figsize=(cm_to_inch(17.8), cm_to_inch(7)), constrained_layout=True
    )
    fig.suptitle(
        "Universal Scaling Collapse of Mutant Invasion Dynamics",
        fontsize=12,
        y=1.05,
    )

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
        lw=2,
        ms=5,
        ax=axA,
    )
    axA.set(
        xscale="log",
        xlabel="Switching Rate, $k$",
        ylabel=r"Final Mutant Fraction, $\langle\rho_M\rangle$",
    )
    # --- CHANGE: Font sizes ---
    axA.set_title(f"(A) Raw Data (s = {s_val_collapse:.2f})", fontsize=10)
    axA.xaxis.label.set_size(8)
    axA.yaxis.label.set_size(8)
    axA.tick_params(axis="both", labelsize=7)
    legA = axA.legend(title="Bias, φ")
    plt.setp(legA.get_title(), fontsize=8)
    plt.setp(legA.get_texts(), fontsize=7)

    axB = axes[1]
    sns.lineplot(
        data=df_s_slice,
        x="k_total",
        y="rho_M_scaled",
        hue="phi",
        palette="viridis",
        marker="o",
        lw=2,
        ms=5,
        ax=axB,
        legend=False,
    )
    axB.axhline(1.0, color="black", ls="--", lw=1.5, label=r"$\rho_{max}$")
    axB.set(
        xscale="log",
        xlabel="Switching Rate, $k$",
        ylabel=r"Scaled Fraction, $\langle\rho_M\rangle / \rho_{max}$",
    )
    # --- CHANGE: Font sizes ---
    axB.set_title("(B) Universal Scaling Collapse", fontsize=10)
    axB.xaxis.label.set_size(8)
    axB.yaxis.label.set_size(8)
    axB.tick_params(axis="both", labelsize=7)
    axB.legend(fontsize=7)

    for ax in axes.flat:
        ax.grid(True, linestyle=":")

    # --- CHANGE: Save to PDF and EPS ---
    plt.savefig(output_path_pdf, bbox_inches="tight")
    plt.savefig(output_path_eps, bbox_inches="tight")
    print(
        f"\nFigure 4 (Data Collapse) saved to: {output_path_pdf} and {output_path_eps}"
    )


if __name__ == "__main__":
    main()
