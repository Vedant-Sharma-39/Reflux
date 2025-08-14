# FILE: scripts/paper_figures/sup_fig_homogeneous_cost.py
#
# Generates a supplementary figure to quantify the inherent fitness cost of
# phenotypic switching in a stable, homogeneous environment where the
# wild-type/generalist is always favored.

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_project_root():
    """Dynamically finds the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# --- Add project root to path to allow importing from src ---
PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS

# Assuming you have a data_loader utility as per your structure
from src.io.data_loader import load_aggregated_data


def find_nearest(array, value):
    """Finds the nearest value in a sorted array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def main():
    # --- 1. Data Loading ---
    campaign_id = EXPERIMENTS["homogeneous_fitness_cost"]["campaign_id"]
    print(f"Generating Homogeneous Fitness Cost figure from campaign: {campaign_id}")
    df = load_aggregated_data(campaign_id, PROJECT_ROOT)

    if df.empty:
        print(
            f"Error: Data for campaign '{campaign_id}' is empty or not found.",
            file=sys.stderr,
        )
        sys.exit(1)

    figure_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    output_path = os.path.join(figure_dir, "sup_fig_homogeneous_cost.png")

    # --- 2. Data Processing and Filtering ---
    # Use the definitive two-step filtering protocol
    STALL_THRESHOLD = 0.20
    stall_counts = df[df["termination_reason"] == "stalled_or_boundary_hit"][
        "phi"
    ].value_counts()
    total_counts = df["phi"].value_counts()
    stall_rates = stall_counts.div(total_counts, fill_value=0)
    phi_to_exclude = stall_rates[stall_rates > STALL_THRESHOLD].index.tolist()

    if phi_to_exclude:
        df = df[~df["phi"].isin(phi_to_exclude)].copy()

    df_filtered = df[df["termination_reason"] != "stalled_or_boundary_hit"].copy()
    df_filtered["s"] = df_filtered["b_m"] - 1.0

    # Select representative selection strengths for panels
    s_all = np.sort(df_filtered["s"].unique())
    s_targets = [-0.1, -0.8]
    s_vals = [find_nearest(s_all, s) for s in s_targets]

    # --- 3. Plotting ---
    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    fig.suptitle(
        "The Inherent Fitness Cost of Phenotypic Switching in a Stable Environment",
        fontsize=24,
        y=1.05,
    )

    # --- Panel A & B: Fitness vs. Switching Rate ---
    panel_map = {
        s_vals[0]: (axes[0], f"(A) Weak Selection (s = {s_vals[0]:.2f})"),
        s_vals[1]: (axes[1], f"(B) Strong Selection (s = {s_vals[1]:.2f})"),
    }

    for s_val, (ax, title) in panel_map.items():
        df_panel = df_filtered[np.isclose(df_filtered["s"], s_val)]

        # Plot a reference line for the non-switching wild-type (fitness should be ~1.0, b_wt)
        ax.axhline(1.0, color="gray", linestyle="--", label="Non-Switching WT Fitness")

        sns.lineplot(
            data=df_panel,
            x="k_total",
            y="avg_front_speed",
            hue="phi",
            palette="coolwarm_r",  # Red (specialist bias) is bad, blue (generalist bias) is better
            legend="full",
            marker="o",
            lw=2.5,
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_title(title, fontsize=18)
        ax.set_xlabel("Switching Rate, $k$", fontsize=16)
        ax.set_ylabel("Long-Term Fitness", fontsize=16)
        ax.grid(True, which="both", ls=":")
        if ax.get_legend() is not None:
            ax.get_legend().set_title(r"Bias, $\phi$")

    # --- Panel C: Quantifying the Fitness Cost ---
    axC = axes[2]
    df_cost = df_filtered[df_filtered["k_total"] > 0].copy()

    # Baseline is the fitness of the non-switching WT (k=0, phi=any value)
    # Theoretically, this fitness is 1.0. We use the simulated value for robustness.
    df_k0 = df_filtered[df_filtered["k_total"] == 0]
    if not df_k0.empty:
        baseline_fitness = df_k0["avg_front_speed"].mean()
    else:
        baseline_fitness = 1.0  # Theoretical value

    df_cost["fitness_cost"] = baseline_fitness - df_cost["avg_front_speed"]

    sns.lineplot(
        data=df_cost,
        x="s",
        y="fitness_cost",
        hue="k_total",
        palette="crest",
        legend="full",
        marker="o",
        lw=2.5,
        ax=axC,
    )

    axC.set_title("(C) Fitness Cost Increases with Selection Strength", fontsize=18)
    axC.set_xlabel("Selection Strength, $s$", fontsize=16)
    axC.set_ylabel(r"Fitness Cost ($\Delta F = F_{k=0} - F_k$)", fontsize=16)
    axC.grid(True, ls=":")
    axC.set_ylim(bottom=0)  # Cost cannot be negative
    if axC.get_legend() is not None:
        axC.get_legend().set_title(r"Switching Rate, $k$")

    sns.despine(fig)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nHomogeneous cost figure saved to: {output_path}")


if __name__ == "__main__":
    main()
