# FILE: scripts/paper_figures/fig2_phase_diagram.py (Standardized & Robust Version)
import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 2: Phase Diagram.")
    parser.add_argument("campaign_id")
    args = parser.parse_args()
    project_root = get_project_root()
    summary_path = os.path.join(
        project_root,
        "data",
        args.campaign_id,
        "analysis",
        f"{args.campaign_id}_summary_aggregated.csv",
    )

    try:
        df = pd.read_csv(summary_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame()

    if df.empty:
        print(
            f"Warning: No data found for campaign '{args.campaign_id}'. Cannot generate Figure 2."
        )
        fig, _ = plt.subplots()
        output_path = os.path.join(
            project_root,
            "data",
            args.campaign_id,
            "analysis",
            "figure2_phase_diagram.png",
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.text(0.5, 0.5, "Figure 2: No Data Available", ha="center", va="center")
        plt.savefig(output_path, dpi=300)
        sys.exit(0)

    print(f"Loaded {len(df)} simulation results.")
    df["s"] = df["b_m"] - 1.0
    df["log10_k_total"] = np.log10(df["k_total"])
    phi_slices = sorted(df["phi"].unique())
    plot_phis = (
        [phi_slices[0], phi_slices[len(phi_slices) // 2], phi_slices[-1]]
        if len(phi_slices) > 3
        else phi_slices
    )

    sns.set_theme(style="white", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(
        2,
        len(plot_phis),
        figsize=(7 * len(plot_phis), 10),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    fig.suptitle("Figure 2: Phase Diagram of Mutant Invasion", fontsize=24, y=1.03)

    for i, phi in enumerate(plot_phis):
        df_slice = df[np.isclose(df["phi"], phi)]
        rho_pivot = df_slice.pivot_table(
            index="s", columns="log10_k_total", values="avg_rho_M"
        )
        var_pivot = df_slice.pivot_table(
            index="s", columns="log10_k_total", values="var_rho_M"
        )

        ax_rho = axes[0, i] if len(plot_phis) > 1 else axes[0]
        sns.heatmap(
            rho_pivot,
            ax=ax_rho,
            cmap="viridis",
            vmin=0,
            vmax=1.0,
            cbar_kws={"label": r"Mean Mutant Density, $\langle\rho_M\rangle$"},
        )
        ax_rho.set_title(rf"$\phi = {phi:.2f}$")
        ax_rho.set_ylabel("Selection, $s$")
        ax_rho.invert_yaxis()

        ax_var = axes[1, i] if len(plot_phis) > 1 else axes[1]
        sns.heatmap(
            var_pivot,
            ax=ax_var,
            cmap="magma",
            cbar_kws={"label": r"Variance, Var($\rho_M$)"},
        )
        ax_var.set_ylabel("Selection, $s$")
        ax_var.set_xlabel(r"log$_{10}(k_{total})$")
        ax_var.invert_yaxis()

    output_dir = os.path.join(project_root, "data", args.campaign_id, "analysis")
    output_path = os.path.join(output_dir, "figure2_phase_diagram.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure 2 saved to {output_path}")


if __name__ == "__main__":
    main()
