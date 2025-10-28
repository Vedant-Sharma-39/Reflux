# FILE: scripts/paper_figures/sup_fig_mean_field (Corrected with Averaging)
#
# Generates a multi-panel figure to demonstrate the robustness of the
# mean-field theory across different selection strengths.
#
# THIS VERSION CORRECTLY AVERAGES ACROSS REPLICATES and plots error bars
# to represent the standard deviation of the simulation outcomes.

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit


def get_project_root():
    """Dynamically finds the project root directory and adds it to the path."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def find_nearest(array, value):
    """Finds the nearest value in a sorted array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def calculate_theoretical_rho_m(k_total, s, phi):
    """
    Calculates the theoretical steady-state mutant fraction based on the
    mean-field quadratic model.
    """
    if np.isclose(s, 0):
        return np.full_like(k_total, (1 - phi) / 2.0, dtype=float)

    k_total = np.asarray(k_total, dtype=float)
    sqrt_term = s**2 + k_total**2 + 2 * s * k_total * phi
    sqrt_term[sqrt_term < 0] = 0

    term_sqrt = np.sqrt(sqrt_term)
    numerator = (s + k_total) - term_sqrt
    denominator = 2 * s

    return numerator / denominator


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
    output_path = os.path.join(figure_dir, "sup_fig_mean_field.png")

    try:
        df_raw = pd.read_csv(summary_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(
            f"Error: Data for campaign '{campaign_id}' not found. Please run and consolidate it first.",
            file=sys.stderr,
        )
        sys.exit(1)

    df_raw["s"] = df_raw["b_m"] - 1.0
    df_raw = df_raw[df_raw["phi"] < 0.99].copy()

    # --- THIS IS THE KEY CORRECTION: AVERAGE ACROSS REPLICATES ---
    df = (
        df_raw.groupby(["s", "phi", "k_total"])
        .agg(
            rho_mean=("avg_rho_M", "mean"),
            rho_std=("avg_rho_M", "std"),
        )
        .reset_index()
    )
    # --- END OF CORRECTION ---

    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(
        2, 2, figsize=(20, 18), constrained_layout=True, sharey=True
    )
    fig.suptitle(
        "Mean-Field Theory Accurately Describes Dynamics Across Selection Regimes",
        fontsize=28,
        y=1.03,
    )

    all_s_values = np.sort(df["s"].unique())
    s_targets = [-0.8, -0.4, -0.2, -0.1]
    s_vals_to_plot = [find_nearest(all_s_values, target) for target in s_targets]
    panel_titles = [
        "(A) Strong Selection",
        "(B) Moderate Selection",
        "(C) Weak Selection",
        "(D) Very Weak Selection",
    ]
    phis_to_plot = [-1.0, -0.5, 0.0, 0.5]
    palette = sns.color_palette("viridis_r", n_colors=len(phis_to_plot))

    for ax, s_val, title in zip(axes.flat, s_vals_to_plot, panel_titles):
        df_s_slice = df[np.isclose(df["s"], s_val)]
        if df_s_slice.empty:
            ax.text(
                0.5,
                0.5,
                f"No data for s ≈ {s_val:.2f}",
                ha="center",
                transform=ax.transAxes,
            )
            continue

        b_m_val = s_val + 1.0
        ax.set_title(f"{title} (b$_m$ = {b_m_val:.2f})", fontsize=20)

        k_theory = np.logspace(
            np.log10(df_s_slice["k_total"].min()),
            np.log10(df_s_slice["k_total"].max()),
            200,
        )

        for i, phi_val in enumerate(phis_to_plot):
            df_phi_slice = df_s_slice[np.isclose(df_s_slice["phi"], phi_val)]
            if df_phi_slice.empty:
                continue

            # --- CHANGE: Plot mean with error bars instead of individual points ---
            ax.errorbar(
                df_phi_slice["k_total"],
                df_phi_slice["rho_mean"],
                yerr=df_phi_slice["rho_std"],
                fmt="o",  # Plot as points
                ms=10,
                color=palette[i],
                capsize=4,  # Add caps to error bars
            )

            rho_asymptote = (1 - phi_val) / 2.0
            ax.axhline(
                rho_asymptote,
                color=palette[i],
                linestyle="--",
                lw=2.5,
                alpha=0.9,
                zorder=0,
            )

            fit_func = lambda k, s_eff: calculate_theoretical_rho_m(k, s_eff, phi_val)
            try:
                bounds = (0, 10)

                sigma = df_phi_slice["rho_std"].fillna(1e-3).replace(0, 1e-3)

                popt, _ = curve_fit(
                    fit_func,
                    df_phi_slice["k_total"],
                    df_phi_slice["rho_mean"],  # Fit to the mean
                    p0=[1],
                    bounds=bounds,
                )
                s_eff = popt[0]

                rho_fit = calculate_theoretical_rho_m(k_theory, s_eff, phi_val)
                ax.plot(
                    k_theory,
                    rho_fit,
                    "-",
                    color=palette[i],
                    lw=3.5,
                    alpha=0.8,
                    label=f"φ={phi_val:.2f} (s_eff={s_eff:.3f})",
                )
            except (RuntimeError, ValueError):
                print(
                    f"Warning: Could not find optimal fit for s={s_val:.2f}, phi={phi_val}"
                )
                ax.plot(
                    [], [], "-", color=palette[i], label=f"φ={phi_val:.2f} (Fit Failed)"
                )

        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle=":")
        ax.legend(fontsize=12, title="Fit Parameters")
        ax.set_ylim(bottom=-0.05, top=1.05)

    for ax in axes[-1, :]:
        ax.set_xlabel("Switching Rate, $k_{total}$", fontsize=16)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Final Mutant Fraction, $\langle\rho_M\rangle$", fontsize=16)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure comparing theory across 'b_m' values saved to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
