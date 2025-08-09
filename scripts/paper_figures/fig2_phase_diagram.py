# FILE: scripts/paper_figures/fig2_phase_diagram.py (Using Susceptibility with Safe LogNorm)

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import SymLogNorm # <-- Use the safe version

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def calculate_susceptibility(pivot_table):
    """Calculates the susceptibility by taking the numerical derivative along the 's' axis."""
    susceptibility_values = np.abs(np.gradient(pivot_table.values, axis=0))
    return pd.DataFrame(susceptibility_values, index=pivot_table.index, columns=pivot_table.columns)

def main():
    parser = argparse.ArgumentParser(description="Generate Figure 2: Phase Diagram of Mutant Invasion.")
    parser.add_argument("campaign_id")
    args = parser.parse_args()
    project_root = get_project_root()
    summary_path = os.path.join(project_root, "data", args.campaign_id, "analysis", f"{args.campaign_id}_summary_aggregated.csv")
    output_path = os.path.join(project_root, "data", args.campaign_id, "analysis", "figure2_phase_diagram_susceptibility.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading data from: {os.path.basename(summary_path)}")
    try:
        df = pd.read_csv(summary_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame()

    if df.empty:
        print(f"Warning: No data found for campaign '{args.campaign_id}'. Cannot generate Figure 2.")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Figure 2: No Data Available", ha='center', va='center')
        plt.savefig(output_path, dpi=300); sys.exit(0)
    
    print(f"Loaded {len(df)} simulation results.")

    # --- Data Processing ---
    df["s"] = df["b_m"] - 1.0
    df = df[df["k_total"] > 0].copy()
    df["log10_k_total"] = np.log10(df["k_total"])

    # Select the three most important phi values
    phi_slices_available = sorted(df["phi"].unique())
    ideal_phis = [-1.0, 0.0, 1.0]
    plot_phis = [min(phi_slices_available, key=lambda x: abs(x - ip)) for ip in ideal_phis]
    plot_phis = sorted(list(set(plot_phis)))

    # --- Plotting ---
    sns.set_theme(style="white", context="paper", font_scale=1.4)
    fig, axes = plt.subplots(
        2, len(plot_phis),
        figsize=(18, 10),
        constrained_layout=True,
        sharex=True, sharey=True
    )
    fig.suptitle("Figure 2: Phase Diagram of Mutant Invasion", fontsize=28, y=1.05)

    for i, phi in enumerate(plot_phis):
        df_slice = df[np.isclose(df["phi"], phi)]
        rho_pivot = df_slice.pivot_table(index="s", columns="log10_k_total", values="avg_rho_M")
        
        sus_pivot = calculate_susceptibility(rho_pivot)

        ax_rho = axes[0, i]
        ax_sus = axes[1, i]

        sns.heatmap(rho_pivot, ax=ax_rho, cmap="viridis", vmin=0, vmax=1.0, cbar_kws={"label": r"$\langle\rho_M\rangle$"})
        ax_rho.set_title(rf"$\phi = {phi:.2f}$", fontsize=18)
        
        # --- THE CRITICAL FIX ---
        # Use SymLogNorm for the susceptibility plot to handle zero values gracefully.
        min_pos_sus = sus_pivot[sus_pivot > 0].min().min() if not sus_pivot[sus_pivot > 0].isnull().all().all() else 1e-9
        sns.heatmap(
            sus_pivot, ax=ax_sus, cmap="inferno", 
            norm=SymLogNorm(linthresh=min_pos_sus, vmin=0, base=10), 
            cbar_kws={"label": r"Susceptibility, $\chi = |\partial \langle\rho_M\rangle / \partial s|$"}
        )
        # --- END FIX ---
        
        ax_sus.set_xlabel(r"$\log_{10}(k_{total})$")
        
        if i == 0:
            ax_rho.set_ylabel("Selection, $s$")
            ax_sus.set_ylabel("Selection, $s$")

        # Format axes for readability
        ax_rho.set_yticklabels([f'{y:.2f}' for y in rho_pivot.index.values])
        ax_sus.set_xticklabels([f'{x:.1f}' for x in sus_pivot.columns.values])

    for ax in axes.flatten():
        ax.invert_yaxis()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure 2 (using Susceptibility) saved to {output_path}")

if __name__ == "__main__":
    main()