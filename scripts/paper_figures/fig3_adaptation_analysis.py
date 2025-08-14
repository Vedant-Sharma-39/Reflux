# FILE: scripts/paper_figures/fig3_adaptation_analysis.py
# Generates the definitive Figure 3. This version is updated to work with the
# migrated data and explicitly removes data points where phi=1.0 from the analysis.

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
    if array.size == 0:
        raise ValueError(
            f"Cannot find nearest value in an empty array. Check data filters."
        )
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def plot_strategy_panel(ax, df_bm_slice, title):
    """Helper function to plot a single fitness-vs-k panel for a fixed b_m."""
    sns.lineplot(
        data=df_bm_slice,
        x="k_total",
        y="avg_front_speed",
        hue="phi",
        palette="coolwarm_r",
        legend="full",
        marker="o",
        lw=3,
        ms=8,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Switching Rate, $k$", fontsize=16)
    ax.set_ylabel("Long-Term Fitness", fontsize=16)
    ax.grid(True, which="both", ls=":")
    if ax.get_legend() is not None:
        ax.get_legend().set_title(r"Bias, $\phi$")


def main():
    project_root = get_project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.config import EXPERIMENTS

    try:
        main_campaign_id = EXPERIMENTS["bet_hedging_final"]["campaign_id"]
        controls_campaign_id = EXPERIMENTS["bet_hedging_controls"]["campaign_id"]
    except KeyError as e:
        print(f"Error: Required key {e} not found in src/config.py.", file=sys.stderr)
        sys.exit(1)

    # --- Load and Combine Data ---
    dfs = []
    for campaign_id in [main_campaign_id, controls_campaign_id]:
        summary_path = os.path.join(
            project_root,
            "data",
            campaign_id,
            "analysis",
            f"{campaign_id}_summary_aggregated.csv",
        )
        if os.path.exists(summary_path):
            print(f"Loading data from: {summary_path}")
            dfs.append(pd.read_csv(summary_path))
        else:
            print(
                f"Warning: Could not find summary file at {summary_path}",
                file=sys.stderr,
            )

    if not dfs:
        print("Error: No data files found. Cannot generate plot.", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)

    # --- THE FIX IS HERE: EXCLUDE PHI = 1.0 ---
    print(f"\nTotal rows loaded: {len(df)}")
    df = df[~np.isclose(df["phi"], 1.0)].copy()
    print(f"Rows after removing phi=1.0: {len(df)}")
    # --- END OF FIX ---

    figure_dir = os.path.join(project_root, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    output_path = os.path.join(figure_dir, "fig3_adaptation_analysis.png")

    print(
        f"\nGenerating definitive Figure 3 from campaigns: {[main_campaign_id, controls_campaign_id]}"
    )

    # --- Filtering Protocol ---
    df_filtered = df[df["termination_reason"] != "stalled_or_boundary_hit"].copy()

    df_plot_data = df_filtered[
        (df_filtered["migrated_from_campaign"] == main_campaign_id)
        & (df_filtered["k_total"] > 0)
    ].copy()

    if df_plot_data.empty:
        print("\nERROR: The dataframe `df_plot_data` is empty after filtering.")
        print(
            f"This likely means no data matched the source campaign filter: migrated_from_campaign == '{main_campaign_id}'"
        )
        sys.exit(1)

    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), constrained_layout=True)
    fig.suptitle(
        "The Fitness Landscape is Rugged and Tuned by Selection", fontsize=28, y=1.03
    )

    bm_all = np.sort(df_plot_data["b_m"].unique())
    bm_targets = [0.9, 0.5, 0.2]
    bm_vals = [find_nearest(bm_all, val) for val in bm_targets]

    panel_map = {
        bm_vals[0]: (axes[0, 0], f"(A) Weak Disadvantage ($b_m$ = {bm_vals[0]:.2f})"),
        bm_vals[1]: (axes[0, 1], f"(B) Medium Disadvantage ($b_m$ = {bm_vals[1]:.2f})"),
        bm_vals[2]: (axes[1, 0], f"(C) Strong Disadvantage ($b_m$ = {bm_vals[2]:.2f})"),
    }

    for bm_val, (ax, title) in panel_map.items():
        df_panel = df_plot_data[np.isclose(df_plot_data["b_m"], bm_val)]
        if not df_panel.empty:
            plot_strategy_panel(ax, df_panel, title)

    ax_d = axes[1, 1]

    # --- Panel D: Optimal Strategy Comparison ---
    phi_irr_val = find_nearest(df_filtered["phi"].unique(), -1.0)
    df_baseline_runs = df_filtered[np.isclose(df_filtered["phi"], phi_irr_val)].copy()
    df_baseline_stats = (
        df_baseline_runs.groupby("b_m")["avg_front_speed"]
        .agg(["mean", "std"])
        .reset_index()
    )

    df_rev_stats = (
        df_plot_data.groupby(["b_m", "phi", "k_total"])["avg_front_speed"]
        .agg(["mean", "std"])
        .reset_index()
    )
    pareto_idx = df_rev_stats.groupby("b_m")["mean"].idxmax()
    df_pareto_stats = df_rev_stats.loc[pareto_idx]

    ax_d.errorbar(
        x=df_baseline_stats["b_m"],
        y=df_baseline_stats["mean"],
        yerr=df_baseline_stats["std"],
        label="Irreversible Strategy ($\\phi=-1.0$)",
        color="crimson",
        marker="s",
        lw=3.5,
        ls="--",
        capsize=5,
    )
    ax_d.errorbar(
        x=df_pareto_stats["b_m"],
        y=df_pareto_stats["mean"],
        yerr=df_pareto_stats["std"],
        label="Optimal Reversible Strategy",
        color="royalblue",
        marker="o",
        lw=4,
        ms=10,
        capsize=5,
    )
    ax_d.fill_between(
        df_pareto_stats["b_m"],
        df_pareto_stats["mean"],
        df_baseline_stats.set_index("b_m").reindex(df_pareto_stats["b_m"])["mean"],
        color="gold",
        alpha=0.3,
        label="Advantage of Reversibility",
    )

    ax_d.set_title("(D) Optimal Strategy Performance", fontsize=20)
    ax_d.set_xlabel("Mutant Fitness in Hostile Patch, $b_m$", fontsize=16)
    ax_d.set_ylabel("Long-Term Fitness (Front Speed)", fontsize=16)
    ax_d.grid(True, which="both", ls=":")
    ax_d.legend(fontsize=14)

    sns.despine(fig)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nDefinitive Figure 3 saved to: {output_path}")


if __name__ == "__main__":
    main()
