# FILE: scripts/paper_figures/fig2_phase_diagram.py
# Generates the definitive 2x2 Figure 2 with polished labels, scaled colorbars,
# and using phi = 0.5 for the generalist bias case.

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
    # --- CHANGE: Output filenames updated to PDF and EPS ---
    output_path_pdf = os.path.join(figure_dir, "fig2_phase_diagram_overview.pdf")
    output_path_eps = os.path.join(figure_dir, "fig2_phase_diagram_overview.eps")

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

    # --- CHANGE: Use 'paper' context for smaller fonts ---
    sns.set_theme(style="ticks", context="paper")
    # --- CHANGE: Set figure size to 2-column width (17.8cm) ---
    fig, axes = plt.subplots(
        2, 2, figsize=(cm_to_inch(17.8), cm_to_inch(16)), constrained_layout=True
    )
    fig.suptitle(
        "Switching Bias is a Master Regulator of the Mutant Invasion Phase",
        # --- CHANGE: Font size within 6-12pt range ---
        fontsize=12,
        y=1.03,
    )

    phi_all = np.sort(df["phi"].unique())

    heatmap_phis = [find_nearest(phi_all, p) for p in [-1.0, 0.0, 0.5]]
    heatmap_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    heatmap_titles = [
        "(A) Strong Specialist Bias ($\\phi = -1.00$)",
        "(B) Unbiased Switching ($\\phi = 0.00$)",
        "(C) Generalist Bias ($\\phi = 0.50$)",
    ]

    sample_pivot = None
    for ax, phi, title in zip(heatmap_axes, heatmap_phis, heatmap_titles):
        df_slice = df[np.isclose(df["phi"], phi)]
        rho_pivot = df_slice.pivot_table(
            index="s", columns="k_total", values="avg_rho_M"
        ).sort_index(ascending=False)
        if sample_pivot is None:
            sample_pivot = rho_pivot

        vmax_theoretical = (1 - phi) / 2.0 if phi >= -1 else 1.0
        vmax_plot = max(vmax_theoretical, 0.01)

        cbar_label = r"Final Mutant Fraction, $\langle\rho_M\rangle$"
        cbar_kws = {"label": cbar_label}
        sns.heatmap(
            rho_pivot,
            ax=ax,
            cmap="YlGnBu",
            vmin=0,
            vmax=vmax_plot,
            cbar=True,
            cbar_kws=cbar_kws,
        )
        ax.set_title(title, fontsize=10)
        # --- CHANGE: Font size for cbar label ---
        ax.figure.axes[-1].yaxis.label.set_size(8)

    ax_d = axes[1, 1]
    line_phis = [find_nearest(phi_all, p) for p in [-1.0, 0.0, 0.5]]
    palette = sns.color_palette("viridis", n_colors=len(line_phis))

    for i, phi in enumerate(line_phis):
        df_slice = df[np.isclose(df["phi"], phi)]
        df_slice["log10_k_total"] = np.log10(df_slice["k_total"])
        rho_pivot_log = df_slice.pivot_table(
            index="s", columns="log10_k_total", values="avg_rho_M"
        ).sort_index(ascending=False)
        sus_pivot = calculate_susceptibility(rho_pivot_log)

        boundary_log_k = sus_pivot.idxmax(axis=1)
        boundary_k = 10**boundary_log_k

        ax_d.plot(
            boundary_k.values,
            boundary_k.index,
            marker="o",
            lw=2,
            ms=5,
            color=palette[i],
            label=f"$\\phi={phi:.2f}$",
        )

    ax_d.set_xscale("log")
    # --- CHANGE: Font sizes ---
    ax_d.set_title("(D) Phase Boundary Shifts with Bias", fontsize=10)
    ax_d.set_xlabel("Critical Switching Rate, $k_c$", fontsize=8)
    ax_d.set_ylabel("Selection Strength, $s$", fontsize=8)
    ax_d.legend(title="Switching Bias", fontsize=7, title_fontsize=8)
    ax_d.grid(True, which="both", ls=":")
    ax_d.tick_params(axis="both", which="major", labelsize=7)

    if sample_pivot is not None:
        y_labels = [f"{s:.2f}" for s in sample_pivot.index]
        x_labels_full = [
            f"{k:.3g}".rstrip("0").rstrip(".") for k in sample_pivot.columns
        ]

        # --- IMPROVEMENT: Create sparse x-tick labels to reduce clutter ---
        label_frequency = 4  # Show one label for every 4 ticks.
        num_labels = len(x_labels_full)
        x_labels_sparse = []
        for i, label in enumerate(x_labels_full):
            # Show label if it's the first, the last, or on the desired frequency.
            if (i % label_frequency == 0) or (i == num_labels - 1):
                x_labels_sparse.append(label)
            else:
                x_labels_sparse.append("")

        for ax in heatmap_axes:
            # --- CHANGE: Font sizes ---
            ax.set_xlabel("Switching Rate, $k$", fontsize=8)
            ax.set_ylabel("Selection Strength, $s$", fontsize=8)
            ax.set_yticks(np.arange(len(y_labels)) + 0.5)
            ax.set_yticklabels(y_labels, rotation=0, fontsize=7)
            ax.set_xticks(np.arange(len(x_labels_full)) + 0.5)
            # Use the improved, sparse labels
            ax.set_xticklabels(x_labels_sparse, rotation=45, ha="right", fontsize=7)

    # --- CHANGE: Save to PDF and EPS ---
    plt.savefig(output_path_pdf, bbox_inches="tight")
    plt.savefig(output_path_eps, bbox_inches="tight")
    print(f"\nDefinitive Figure 2 saved to: {output_path_pdf} and {output_path_eps}")


if __name__ == "__main__":
    main()
