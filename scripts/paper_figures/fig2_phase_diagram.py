# FILE: scripts/paper_figures/fig2_phase_diagram.py
# Generates the definitive 2x2 Figure 2 with polished labels, scaled colorbars,
# and using phi = 0.5 for the generalist bias case.

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


def calculate_susceptibility(sorted_pivot_table):
    """Calculates susceptibility by taking the gradient along the 's' axis."""
    sus_values = np.abs(np.gradient(sorted_pivot_table.values, axis=0))
    return pd.DataFrame(
        sus_values, index=sorted_pivot_table.index, columns=sorted_pivot_table.columns
    )


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
    output_path = os.path.join(figure_dir, "fig2_phase_diagram_overview.png")

    print(f"Generating definitive Figure 2 (2x2) from campaign: {campaign_id}")
    try:
        df = pd.read_csv(summary_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(
            f"Error: Data not found or empty for campaign '{campaign_id}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    df["s"] = df["b_m"] - 1.0

    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), constrained_layout=True)
    fig.suptitle(
        "Switching Bias is a Master Regulator of the Mutant Invasion Phase",
        fontsize=28,
        y=1.03,
    )

    phi_all = np.sort(df["phi"].unique())

    # --- MODIFIED: Use phi = 0.5 instead of 1.0 for the third heatmap ---
    heatmap_phis = [find_nearest(phi_all, p) for p in [-1.0, 0.0, 0.5]]
    heatmap_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    heatmap_titles = [
        "(A) Strong Specialist Bias ($\\phi = -1.00$)",
        "(B) Unbiased Switching ($\\phi = 0.00$)",
        "(C) Generalist Bias ($\\phi = 0.50$)",  # Updated title
    ]

    sample_pivot = None
    for ax, phi, title in zip(heatmap_axes, heatmap_phis, heatmap_titles):
        df_slice = df[np.isclose(df["phi"], phi)]
        rho_pivot = df_slice.pivot_table(
            index="s", columns="k_total", values="avg_rho_M"
        ).sort_index(ascending=False)
        if sample_pivot is None:
            sample_pivot = rho_pivot

        vmax_theoretical = (1 - phi) / 2.0 if phi >= -1 else 1.0  # Handle phi=-1 case
        vmax_plot = max(vmax_theoretical, 0.01)

        sns.heatmap(
            rho_pivot,
            ax=ax,
            cmap="viridis",
            vmin=0,
            vmax=vmax_plot,
            cbar=True,
            cbar_kws={"label": r"Final Mutant Fraction, $\langle\rho_M\rangle$"},
        )
        ax.set_title(title, fontsize=20)

    # --- Panel D: The Shifting Phase Boundary ---
    ax_d = axes[1, 1]
    line_phis = [find_nearest(phi_all, p) for p in [-1.0, 0.0, 0.5]]
    palette = sns.color_palette("viridis", n_colors=len(line_phis))

    for i, phi in enumerate(line_phis):
        df_slice = df[np.isclose(df["phi"], phi)]
        # Use a pivot on log10_k_total for better boundary detection on a log scale
        df_slice["log10_k_total"] = np.log10(df_slice["k_total"])
        rho_pivot_log = df_slice.pivot_table(
            index="s", columns="log10_k_total", values="avg_rho_M"
        ).sort_index(ascending=False)
        sus_pivot = calculate_susceptibility(rho_pivot_log)

        # Find the k_total that corresponds to the peak susceptibility for each s
        boundary_log_k = sus_pivot.idxmax(axis=1)
        boundary_k = 10**boundary_log_k

        ax_d.plot(
            boundary_k.values,
            boundary_k.index,
            marker="o",
            lw=3,
            ms=9,
            color=palette[i],
            label=f"$\\phi={phi:.2f}$",
        )

    ax_d.set_xscale("log")
    ax_d.set_title("(D) Phase Boundary Shifts with Bias", fontsize=20)
    ax_d.set_xlabel("Critical Switching Rate, $k_c$", fontsize=16)
    ax_d.set_ylabel("Selection Strength, $s$", fontsize=16)
    ax_d.legend(title="Switching Bias", fontsize=14)
    ax_d.grid(True, which="both", ls=":")

    # --- Final Polishing: Clean Labels ---
    if sample_pivot is not None:
        y_labels = [f"{s:.2f}" for s in sample_pivot.index]
        x_labels = [f"{k:.3g}".rstrip("0").rstrip(".") for k in sample_pivot.columns]

        for ax in heatmap_axes:
            ax.set_xlabel("Switching Rate, $k$", fontsize=16)
            ax.set_ylabel("Selection Strength, $s$", fontsize=16)

            # --- FIX IS HERE ---
            # Set the tick locations *before* setting the labels
            ax.set_yticks(np.arange(len(y_labels)) + 0.5)
            ax.set_yticklabels(y_labels, rotation=0)

            ax.set_xticks(np.arange(len(x_labels)) + 0.5)
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
            # --- END FIX ---

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nDefinitive Figure 2 (2x2 Overview with phi=0.5) saved to: {output_path}")


if __name__ == "__main__":
    main()
