# FILE: scripts/paper_figures/sup_fig_fitness_cost.py
# Quantifies the fitness cost of switching in a stable, homogeneous environment.

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_project_root():
    """Dynamically finds the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    project_root = get_project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.config import EXPERIMENTS

    try:
        campaign_id = EXPERIMENTS["homogeneous_fitness_cost"]["campaign_id"]
    except KeyError:
        print(
            "Error: 'homogeneous_fitness_cost' experiment not found in config.",
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
    df = pd.read_csv(summary_path)

    # In a homogeneous WT-favored env, baseline fitness is 1.0.
    # Cost is the deviation from this maximum.
    df["fitness_cost"] = 1.0 - df["avg_front_speed"]

    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    fig.suptitle(
        "The Inherent Fitness Cost of Switching in a Stable Environment",
        fontsize=24,
        y=1.05,
    )

    # --- Panel A: Cost vs. Switching Rate ---
    axA = axes[0]
    df_panel_A = df[
        np.isclose(df["b_m"], 0.5)
    ]  # Use a representative selection strength
    sns.lineplot(
        data=df_panel_A,
        x="k_total",
        y="avg_front_speed",
        hue="phi",
        palette="coolwarm_r",
        marker="o",
        lw=2.5,
        ax=axA,
    )
    axA.set_xscale("log")
    axA.set_title("(A) Fitness Decreases with Switching Rate\n($b_m=0.5$)", fontsize=18)
    axA.set_xlabel("Switching Rate, $k_{total}$", fontsize=16)
    axA.set_ylabel("Long-Term Fitness (Front Speed)", fontsize=16)
    axA.axhline(1.0, color="k", ls="--", label="Optimal (No Switching)")
    axA.legend(title="Bias, $\\phi$")
    axA.grid(True, which="both", ls=":")

    # --- Panel B: Cost vs. Mutant Fitness ---
    axB = axes[1]
    df_panel_B = df[df["k_total"] > 0]
    sns.lineplot(
        data=df_panel_B,
        x="b_m",
        y="fitness_cost",
        hue="k_total",
        palette="viridis_r",
        marker="o",
        lw=2.5,
        ax=axB,
    )
    axB.set_title("(B) Cost Profile vs. Mutant Viability", fontsize=18)
    axB.set_xlabel("Mutant Fitness in Hostile Patch, $b_m$", fontsize=16)
    axB.set_ylabel("Fitness Cost (1.0 - Speed)", fontsize=16)
    axB.set_ylim(bottom=0)
    axB.legend(title="Rate, $k_{total}$")
    axB.grid(True, ls=":")

    output_path = os.path.join(project_root, "figures", "sup_fig_fitness_cost.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSupplementary Figure (Fitness Cost) saved to: {output_path}")


if __name__ == "__main__":
    main()
