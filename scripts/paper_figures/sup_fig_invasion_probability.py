# FILE: scripts/paper_figures/sup_fig_invasion_probability.py
# Analyzes the probability of a small mutant patch surviving and fixing.

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
        campaign_id = EXPERIMENTS["invasion_probability_analysis"]["campaign_id"]
    except KeyError:
        print(
            "Error: 'invasion_probability_analysis' experiment not found in config.",
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

    # Calculate fixation probability
    df["fixated"] = (df["outcome"] == "fixation").astype(int)
    prob_df = (
        df.groupby(["k_total", "phi", "env_definition", "b_m"])["fixated"]
        .mean()
        .reset_index()
        .rename(columns={"fixated": "fixation_prob"})
    )

    sns.set_theme(style="ticks", context="talk")
    g = sns.relplot(
        data=prob_df,
        x="k_total",
        y="fixation_prob",
        hue="phi",
        style="env_definition",
        col="b_m",
        kind="line",
        palette="coolwarm_r",
        marker="o",
        height=6,
        aspect=1.2,
        legend="full",
        lw=3,
    )

    g.set_axis_labels("Switching Rate, $k_{total}$", "Fixation Probability")
    g.set(xscale="log")
    g.fig.suptitle(
        "Invasion Probability from a Small Mutant Patch", fontsize=24, y=1.03
    )
    g.set_titles("Selection, $b_m$ = {col_name}")

    output_path = os.path.join(
        project_root, "figures", "sup_fig_invasion_probability.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSupplementary Figure (Invasion Probability) saved to: {output_path}")


if __name__ == "__main__":
    main()
