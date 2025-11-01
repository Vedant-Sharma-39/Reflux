# FILE: scripts/paper_figures/fig2_phase_diagram.py
#
# Generates the definitive 2x2 Figure 2.
# This version is updated to full publication-ready quality, matching the
# style of 'fig_optimal_strategy.py'.
#
# v2: Improves readability and reduces cramping:
# 1. Increased overall figure size.
# 2. Increased all font sizes (titles, labels, ticks, legend).
# 3. Adjusted `constrained_layout` padding for better spacing.
# 4. Refined heatmap tick labels for better visibility.
# 5. Slightly larger markers and lines in Panel (D).

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
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 16 # Base font size increased
# --- End Publication Settings ---


def cm_to_inch(cm):
    return cm / 2.54


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
    
    output_path_pdf = os.path.join(figure_dir, "fig_phase_diagram.pdf")
    output_path_png = os.path.join(figure_dir, "fig_phase_diagram.png")

    print(f"Generating definitive Figure 2 (2x2) from campaign: {campaign_id}")
    try:
        df = pd.read_csv(summary_path, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(
            f"Error: Data not found or empty for campaign '{campaign_id}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    df["s"] = df["b_m"] - 1.0

    # --- Scaled-up Font Definitions ---
    title_font = {'fontsize': 16, 'fontweight': 'bold'} # Increased
    label_font = {'fontsize': 14} # Increased
    tick_font_size = 12 # Base size now 14, so this is relatively smaller
    legend_font_size = 12 # Increased

    sns.set_theme(style="ticks", context="paper")
    
    # --- Increased Figure Size & Adjusted Padding ---
    fig, axes = plt.subplots(
        2, 2, figsize=(cm_to_inch(20), cm_to_inch(18)), # Increased dimensions
        constrained_layout=True
    )
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.05, hspace=0.1, wspace=0.1) # Fine-tuned padding


    phi_all = np.sort(df["phi"].unique())

    heatmap_phis = [find_nearest(phi_all, p) for p in [-1.0, 0.0, 0.5]]
    heatmap_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    heatmap_titles = [
        r"(A) Strong Specialist Bias ($\phi = -1.00$)",
        r"(B) Unbiased Switching ($\phi = 0.00$)",
        r"(C) Generalist Bias ($\phi = 0.50$)",
    ]

    sample_pivot = None
    for ax, phi, title in zip(heatmap_axes, heatmap_phis, heatmap_titles):
        df_slice = df[np.isclose(df["phi"], phi)]
        rho_pivot = df_slice.pivot_table(
            index="s", columns="k_total", values="avg_rho_M"
        ).sort_index(ascending=False)
        
        if rho_pivot.empty:
            ax.text(0.5, 0.5, f"No data for phi={phi}", ha='center', va='center', fontsize=label_font['fontsize'])
            continue
            
        if sample_pivot is None:
            sample_pivot = rho_pivot

        vmax_theoretical = (1 - phi) / 2.0 if phi >= -1 else 1.0
        vmax_plot = max(vmax_theoretical, 0.01)

        cbar_label = r"Avg. Mutant Fraction ($\langle\rho_M\rangle$)"
        cbar_kws = {"label": cbar_label}
        
        sns.heatmap(
            rho_pivot,
            ax=ax,
            cmap="RdBu_r",
            vmin=0,
            vmax=vmax_plot,
            cbar=True,
            cbar_kws=cbar_kws,
            linewidths=0.0, # Remove lines between cells for cleaner look
            linecolor='none',
        )
        ax.set_title(title, **title_font)
        ax.figure.axes[-1].yaxis.label.set_size(label_font['fontsize']) # Increased
        ax.figure.axes[-1].tick_params(labelsize=tick_font_size) # Increased


    ax_d = axes[1, 1]
    line_phis = [find_nearest(phi_all, p) for p in [-1.0, 0.0, 0.5]]
    palette = sns.color_palette("viridis", n_colors=len(line_phis))

    for i, phi in enumerate(line_phis):
        df_slice = df[np.isclose(df["phi"], phi)].copy()
        
        df_slice["log10_k_total"] = np.log10(df_slice["k_total"])
        rho_pivot_log = df_slice.pivot_table(
            index="s", columns="log10_k_total", values="avg_rho_M"
        ).sort_index(ascending=False)
        
        if rho_pivot_log.empty:
            continue
            
        smoothed_data = gaussian_filter(rho_pivot_log.values, sigma=1.5)
        smoothed_pivot = pd.DataFrame(
            smoothed_data, 
            index=rho_pivot_log.index, 
            columns=rho_pivot_log.columns
        )

        sus_pivot = calculate_susceptibility(smoothed_pivot)

        boundary_log_k_values = []
        log_k_axis = sus_pivot.columns.to_numpy(dtype=float)
        s_values = sus_pivot.index
        
        for s_index in s_values:
            sus_row = sus_pivot.loc[s_index].to_numpy(dtype=float)
            peak_idx = np.argmax(sus_row)
            
            if peak_idx == 0 or peak_idx == len(sus_row) - 1:
                boundary_log_k = log_k_axis[peak_idx]
            else:
                x_points = log_k_axis[peak_idx-1 : peak_idx+2]
                y_points = sus_row[peak_idx-1 : peak_idx+2]
                
                try:
                    p = np.polyfit(x_points, y_points, 2)
                    a, b, c = p
                    if a < 0:
                        sub_pixel_peak_log_k = -b / (2 * a)
                        if sub_pixel_peak_log_k > x_points[0] and sub_pixel_peak_log_k < x_points[2]:
                            boundary_log_k = sub_pixel_peak_log_k
                        else:
                            boundary_log_k = log_k_axis[peak_idx]
                    else:
                        boundary_log_k = log_k_axis[peak_idx]
                except np.linalg.LinAlgError:
                    boundary_log_k = log_k_axis[peak_idx]
                    
            boundary_log_k_values.append(boundary_log_k)
        
        boundary_k = 10**np.array(boundary_log_k_values)

        ax_d.plot(
            boundary_k,
            s_values,
            marker="o",
            lw=2.5, # Slightly thicker lines
            ms=6,  # Slightly larger markers
            mfc='white',
            mec=palette[i],
            mew=1.0,
            color=palette[i],
            label=fr"$\phi={phi:.2f}$",
        )

    ax_d.set_xscale("log")
    ax_d.set_title(r"(D) Phase Boundary Shifts with Bias", **title_font)
    ax_d.set_xlabel(r"Critical Switching Rate ($k_c$)", **label_font)
    ax_d.set_ylabel(r"Selection Strength ($s$)", **label_font)
    ax_d.legend(
        title=r"Switching Bias ($\phi$)", 
        fontsize=legend_font_size, 
        title_fontsize=legend_font_size,
        frameon=False
    )
    ax_d.grid(True, which="both", ls=":", alpha=0.4)
    ax_d.tick_params(axis="both", which="major", labelsize=tick_font_size)

    if sample_pivot is not None:
        x_vals = sample_pivot.columns
        x_ticks_to_show = np.array([1e-2, 1e-1, 1e0, 1e1, 1e2])
        x_tick_indices = x_vals.get_indexer(x_ticks_to_show, method='nearest')
        x_tick_labels = [fr"$10^{{{int(np.log10(v))}}}$" for v in x_ticks_to_show]
        
        valid_x_indices = [i for i in x_tick_indices if i != -1]
        valid_x_labels = [label for i, label in zip(x_tick_indices, x_tick_labels) if i != -1]
        
        y_vals = sample_pivot.index
        # --- Adjusted y-ticks for better visual distribution ---
        y_ticks_to_show = np.array([0.0, -0.2, -0.4, -0.6, -0.8, -0.9])
        y_tick_indices = y_vals.get_indexer(y_ticks_to_show, method='nearest')
        y_tick_labels = [f"{v:.2f}" for v in y_ticks_to_show]

        valid_y_indices = [i for i in y_tick_indices if i != -1]
        valid_y_labels = [label for i, label in zip(y_tick_indices, y_tick_labels) if i != -1]

        for ax in heatmap_axes:
            ax.set_xlabel(r"Switching Rate ($k_{\mathrm{total}}$)", **label_font)
            ax.set_ylabel(r"Selection Strength ($s$)", **label_font)
            
            # --- Centered ticks more accurately on cells ---
            ax.set_yticks(np.array(valid_y_indices) + 0.5)
            ax.set_yticklabels(valid_y_labels, rotation=0, fontsize=tick_font_size)
            
            ax.set_xticks(np.array(valid_x_indices) + 0.5)
            ax.set_xticklabels(valid_x_labels, rotation=0, ha="center", fontsize=tick_font_size)

    sns.despine(fig) # Remove top and right spines

    plt.savefig(output_path_pdf, bbox_inches="tight", dpi=600)
    plt.savefig(output_path_png, bbox_inches="tight", dpi=600)
    print(f"\nDefinitive Figure 2 saved to:\n  {output_path_pdf}\n  {output_path_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()