# FILE: scripts/paper_figures/fig2_phase_diagram.py

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --- Setup Project Root Path ---
def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


project_root = get_project_root()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.io.data_loader import aggregate_data_cached


def plot_theoretical_boundary(ax, pivot_table):
    """Plots the mean-field prediction s_critical = k_total on the heatmap axes."""
    log_k_vals = pivot_table.columns.values
    s_vals = pivot_table.index.values
    theory_k = np.linspace(log_k_vals.min(), log_k_vals.max(), 100)
    theory_s = np.power(10, theory_k)
    ax.plot(theory_k, theory_s, "r--", lw=2.5, label=r"Theory: $s_{crit} = k_{total}$")
    ax.set_ylim(s_vals.min(), s_vals.max())
    ax.set_xlim(log_k_vals.min(), log_k_vals.max())


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 2: Phase Diagram."
    )
    parser.add_argument(
        "campaign_id",
        default="phase_diagram",
        nargs="?",
        help="Campaign ID for the phase diagram experiment (default: phase_diagram)",
    )
    args = parser.parse_args()

    df = load_aggregated_data(args.campaign_id, project_root)
    if df is None or df.empty:
        sys.exit(f"Could not load data for campaign '{args.campaign_id}'. Aborting.")
    if df.empty:
        sys.exit(f"Error: No data loaded for campaign '{args.campaign_id}'.")

    print(f"Loaded {len(df)} simulation results.")
    df["s"] = df["b_m"] - 1.0
    df = df[df["k_total"] > 0].copy()
    df["log10_k_total"] = np.log10(df["k_total"])

    phi_slices = sorted(df["phi"].unique())
    plot_phis = (
        phi_slices
        if len(phi_slices) <= 3
        else [phi_slices[0], phi_slices[len(phi_slices) // 2], phi_slices[-1]]
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
        plot_theoretical_boundary(ax_rho, rho_pivot)

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
        plot_theoretical_boundary(ax_var, var_pivot)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=1)

    output_dir = os.path.join(project_root, "data", args.campaign_id, "analysis")
    output_path = os.path.join(output_dir, "figure2_phase_diagram.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure 2 saved to {output_path}")


if __name__ == "__main__":
    main()
