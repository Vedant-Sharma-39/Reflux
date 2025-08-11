# FILE: scripts/paper_figures/fig_optimal_strategy.py
# This script generates a publication-quality figure showing how the optimal
# bet-hedging strategy (k_opt, phi_opt) adapts to environmental pressures (s, patch_width).

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure: The Adaptive Landscape of Bet-Hedging Strategies."
    )
    parser.add_argument("campaign_id")
    args = parser.parse_args()
    project_root = get_project_root()
    summary_path = os.path.join(
        project_root,
        "data",
        args.campaign_id,
        "analysis",
        f"{args.campaign_id}_summary_aggregated.csv",
    )
    output_path = os.path.join(
        project_root,
        "data",
        args.campaign_id,
        "analysis",
        "fig_optimal_strategy_landscape.png",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading data from: {os.path.basename(summary_path)}")
    try:
        df = pd.read_csv(summary_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame()

    if df.empty:
        print(
            f"Error: No data for campaign '{args.campaign_id}'. Cannot generate figure.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded {len(df)} simulation results.")

    if "termination_reason" in df.columns:
        df = df[df["termination_reason"] == "converged"].copy()
        print(f"Filtered for converged runs, {len(df)} results remain.")
    if df.empty:
        print(
            "Error: No converged runs found in the dataset. Cannot generate plot.",
            file=sys.stderr,
        )
        sys.exit(1)

    df["s"] = df["b_m"] - 1.0
    df_switching = df[df["k_total"] > 0].copy()

    # --- CORE ANALYSIS: Find the optimal strategy for each environment ---
    # For each combination of (selection, patch_width), find the parameters
    # (k_total, phi) that resulted in the highest average front speed.
    opt_idx = df_switching.groupby(["s", "patch_width"])["avg_front_speed"].idxmax()
    df_opt = df_switching.loc[opt_idx]
    df_opt["log10_k_opt"] = np.log10(df_opt["k_total"])

    # --- PIVOT for HEATMAPS ---
    pivot_k = df_opt.pivot_table(
        index="s", columns="patch_width", values="log10_k_opt"
    ).sort_index(ascending=False)
    pivot_phi = df_opt.pivot_table(
        index="s", columns="patch_width", values="phi"
    ).sort_index(ascending=False)

    # --- PLOTTING ---
    sns.set_theme(style="white", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    fig.suptitle(
        "The Adaptive Landscape of Spatial Bet-Hedging Strategies", fontsize=24, y=1.05
    )

    # Panel A: Optimal Switching Timescale (k_opt)
    axA = axes[0]
    sns.heatmap(
        pivot_k,
        ax=axA,
        cmap="plasma",
        cbar_kws={"label": r"Optimal Switching Rate, $\log_{10}(k_{opt})$"},
    )
    axA.set_title("(a) The Optimal Switching Timescale", fontsize=18)
    axA.set_xlabel("Patch Width (Environmental Timescale)")
    axA.set_ylabel("Selection, $s$ (Cost of Mismatch)")

    # Panel B: Optimal Switching Bias (phi_opt)
    axB = axes[1]
    sns.heatmap(
        pivot_phi,
        ax=axB,
        cmap="coolwarm_r",
        vmin=-1,
        vmax=1,
        cbar_kws={"label": r"Optimal Bias, $\phi_{opt}$"},
    )
    axB.set_title("(b) The Optimal Switching Bias", fontsize=18)
    axB.set_xlabel("Patch Width (Environmental Timescale)")
    axB.set_ylabel("")  # Y-axis is shared, no need to repeat
    axB.set_yticklabels([])

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nNew Adaptive Landscape Figure saved to {output_path}")


if __name__ == "__main__":
    main()
