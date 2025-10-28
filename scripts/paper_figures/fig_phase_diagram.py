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
from scipy.ndimage import gaussian_filter # <-- IMPORT FOR SMOOTHING

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
    """
    Calculates susceptibility by taking the gradient along the 'k' axis (axis=1),
    which is the horizontal axis of the heatmap.
    """
    # --- IMPROVED LOGIC: Calculate gradient along axis=1 (k_total) ---
    sus_values = np.abs(np.gradient(sorted_pivot_table.values, axis=1))
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
    output_path_pdf = os.path.join(figure_dir, "fig_phase_diagram_overview.pdf")
    output_path_eps = os.path.join(figure_dir, "fig_phase_diagram_overview.eps")

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

    sns.set_theme(style="ticks", context="paper")
    fig, axes = plt.subplots(
        2, 2, figsize=(cm_to_inch(17.8), cm_to_inch(16)), constrained_layout=True
    )
    
    # --- REMOVED: suptitle removed to match reference image ---
    # fig.suptitle(...)

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
        
        if rho_pivot.empty:
            ax.text(0.5, 0.5, f"No data for phi={phi}", ha='center', va='center')
            continue
            
        if sample_pivot is None:
            sample_pivot = rho_pivot

        vmax_theoretical = (1 - phi) / 2.0 if phi >= -1 else 1.0
        vmax_plot = max(vmax_theoretical, 0.01)

        cbar_label = r"Avg. Mutant Fraction ($\langle\rho_M\rangle$)"
        cbar_kws = {"label": cbar_label}
        
        # --- IMPROVED: Colormap changed to RdBu_r to match reference ---
        sns.heatmap(
            rho_pivot,
            ax=ax,
            cmap="RdBu_r", # <-- CHANGED
            vmin=0,
            vmax=vmax_plot,
            cbar=True,
            cbar_kws=cbar_kws,
        )
        ax.set_title(title, fontsize=10)
        ax.figure.axes[-1].yaxis.label.set_size(8) # cbar label

    ax_d = axes[1, 1]
    line_phis = [find_nearest(phi_all, p) for p in [-1.0, 0.0, 0.5]]
    
    # --- CHANGE: Palette changed to match reference (dark blue -> green) ---
    palette = sns.color_palette("viridis", n_colors=len(line_phis))

    for i, phi in enumerate(line_phis):
        # --- FIX: Add .copy() to prevent SettingWithCopyWarning ---
        df_slice = df[np.isclose(df["phi"], phi)].copy()
        
        # We need to pivot on log_k for evenly spaced gradient calculation
        df_slice["log10_k_total"] = np.log10(df_slice["k_total"])
        rho_pivot_log = df_slice.pivot_table(
            index="s", columns="log10_k_total", values="avg_rho_M"
        ).sort_index(ascending=False)
        
        if rho_pivot_log.empty:
            continue
            
        # --- NEW: Smooth the data before taking gradient ---
        # Apply a Gaussian filter. Sigma controls the amount of smoothing.
        smoothed_data = gaussian_filter(rho_pivot_log.values, sigma=1.0)
        smoothed_pivot = pd.DataFrame(
            smoothed_data, 
            index=rho_pivot_log.index, 
            columns=rho_pivot_log.columns
        )

        # --- Calculate susceptibility on SMOOTHED data ---
        sus_pivot = calculate_susceptibility(smoothed_pivot)

        # Find the log_k column that has the max gradient for each s row
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
    ax_d.set_title("(D) Phase Boundary Shifts with Bias", fontsize=10)
    ax_d.set_xlabel("Critical Switching Rate ($k_c$)", fontsize=8) # Match ref
    ax_d.set_ylabel("Selection Strength ($s$)", fontsize=8)
    ax_d.legend(title="Switching Bias ($\phi$)", fontsize=7, title_fontsize=8) # Match ref
    ax_d.grid(True, which="both", ls=":")
    ax_d.tick_params(axis="both", which="major", labelsize=7)

    # --- IMPROVED: Clean, log-spaced tick labels for heatmaps ---
    if sample_pivot is not None:
        # X-axis (k_total)
        x_vals = sample_pivot.columns
        x_ticks_to_show = np.array([1e-2, 1e-1, 1e0, 1e1, 1e2])
        x_tick_indices = x_vals.get_indexer(x_ticks_to_show, method='nearest')
        x_tick_labels = [f"$10^{{{int(np.log10(v))}}}$" for v in x_ticks_to_show]
        # Ensure we don't show ticks for data that doesn't exist
        valid_x_indices = [i for i in x_tick_indices if i != -1]
        valid_x_labels = [label for i, label in zip(x_tick_indices, x_tick_labels) if i != -1]
        
        # Y-axis (s)
        y_vals = sample_pivot.index
        y_ticks_to_show = np.array([0.0, -0.2, -0.4, -0.7, -0.9])
        y_tick_indices = y_vals.get_indexer(y_ticks_to_show, method='nearest')
        y_tick_labels = [f"{v:.2f}" for v in y_ticks_to_show]
        valid_y_indices = [i for i in y_tick_indices if i != -1]
        valid_y_labels = [label for i, label in zip(y_tick_indices, y_tick_labels) if i != -1]


        for ax in heatmap_axes:
            ax.set_xlabel("Switching Rate ($k_{\mathrm{total}}$)", fontsize=8) # Match ref
            ax.set_ylabel("Selection Strength ($s$)", fontsize=8)
            
            # --- FIX: Convert list to numpy array before adding float ---
            ax.set_yticks(np.array(valid_y_indices) + 0.5)
            ax.set_yticklabels(valid_y_labels, rotation=0, fontsize=7)
            
            # --- FIX: Convert list to numpy array before adding float ---
            ax.set_xticks(np.array(valid_x_indices) + 0.5)
            ax.set_xticklabels(valid_x_labels, rotation=0, ha="center", fontsize=7)

    plt.savefig(output_path_pdf, bbox_inches="tight")
    plt.savefig(output_path_eps, bbox_inches="tight")
    print(f"\nDefinitive Figure 2 saved to: {output_path_pdf} and {output_path_eps}")


if __name__ == "__main__":
    main()

