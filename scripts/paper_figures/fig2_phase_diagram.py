import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots for Figure 2: Phase Diagram."
    )
    parser.add_argument(
        "campaign_id", help="Campaign ID for the hierarchical criticality scan."
    )
    args = parser.parse_args()

    # --- Load Data ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    summary_path = os.path.join(
        project_root,
        "data",
        args.campaign_id,
        "analysis",
        f"{args.campaign_id}_summary_aggregated.csv",
    )
    if not os.path.exists(summary_path):
        print(f"Error: Summary file not found: {summary_path}")
        return

    df = pd.read_csv(summary_path)
    print(f"Loaded {len(df)} simulation results.")

    # --- Process Data ---
    df["s"] = df["b_m"] - 1.0
    df["log10_k_total"] = np.log10(df["k_total"])

    # We will plot for a few representative phi values
    phi_slices = sorted(df["phi"].unique())
    plot_phis = [phi_slices[0], phi_slices[len(phi_slices) // 2], phi_slices[-1]]

    # --- Create Plots ---
    sns.set_theme(style="white", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(
        2,
        len(plot_phis),
        figsize=(8 * len(plot_phis), 12),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    fig.suptitle("Figure 2: Phase Diagram of Mutant Invasion", fontsize=24, y=1.03)

    for i, phi in enumerate(plot_phis):
        df_slice = df[np.isclose(df["phi"], phi)]

        # Pivot for heatmap
        rho_pivot = df_slice.pivot_table(
            index="s", columns="log10_k_total", values="avg_rho_M"
        )
        var_pivot = df_slice.pivot_table(
            index="s", columns="log10_k_total", values="var_rho_M"
        )

        # Find critical line (peak of variance)
        kc_line = var_pivot.idxmax(axis=1)

        # Plot 1: Mutant Density (rho_M)
        ax_rho = axes[0, i]
        sns.heatmap(
            rho_pivot,
            ax=ax_rho,
            cmap="viridis",
            cbar_kws={"label": r"Mean Mutant Density, $\langle\rho_M\rangle$"},
        )
        ax_rho.plot(
            kc_line,
            rho_pivot.index,
            color="red",
            lw=2.5,
            linestyle="--",
            label=r"$k_c(s)$ (Peak Variance)",
        )
        ax_rho.set_title(rf"$\phi = {phi:.2f}$")
        ax_rho.set_ylabel("Selection, $s$")

        # Plot 2: Variance of Mutant Density
        ax_var = axes[1, i]
        sns.heatmap(
            var_pivot,
            ax=ax_var,
            cmap="magma",
            cbar_kws={"label": r"Variance, Var($\rho_M$)"},
        )
        ax_var.set_title(rf"$\phi = {phi:.2f}$")
        ax_var.set_ylabel("Selection, $s$")
        ax_var.set_xlabel(r"log$_{10}(k_{total})$")

    # --- Save Figure ---
    output_dir = os.path.join(project_root, "data", args.campaign_id, "analysis")
    output_path = os.path.join(output_dir, "figure2_phase_diagram.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure 2 saved to {output_path}")


if __name__ == "__main__":
    main()
