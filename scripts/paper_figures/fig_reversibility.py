# FILE: scripts/paper_figures/fig3_adaptation_analysis.py
# Generates the definitive Figure 3 from the "bet_hedging_final" experiment,
# specifically using the data for the 60px periodic environment.

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
        lw=2,
        ms=5,
        ax=ax,
    )
    ax.set_xscale("log")
    # --- CHANGE: Font sizes ---
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Switching Rate, $k$", fontsize=8)
    ax.set_ylabel("Long-Term Fitness", fontsize=8)
    ax.grid(True, which="both", ls=":")
    ax.tick_params(axis="both", which="major", labelsize=7)
    if ax.get_legend() is not None:
        leg = ax.get_legend()
        leg.set_title(r"Bias, $\phi$")
        plt.setp(leg.get_title(), fontsize=8)
        plt.setp(leg.get_texts(), fontsize=7)


def main():
    project_root = get_project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.config import EXPERIMENTS, PARAM_GRID

    try:
        campaign_id = EXPERIMENTS["bet_hedging_final"]["campaign_id"]
    except KeyError as e:
        print(
            f"Error: Required experiment key 'bet_hedging_final' not found in src/config.py.",
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
    if not os.path.exists(summary_path):
        print(
            f"Error: Data file not found for campaign '{campaign_id}'. Run 'make consolidate CAMPAIGN={campaign_id}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading data from: {summary_path}")
    df = pd.read_csv(summary_path)

    df = df[~np.isclose(df["phi"], 1.0)].copy()
    df_filtered = df[df["termination_reason"] == "converged"].copy()

    env_name = PARAM_GRID["env_definitions"]["symmetric_refuge_60w"]["name"]
    df_plot_data = df_filtered[
        (df_filtered["env_definition"] == env_name) & (df_filtered["k_total"] > 0)
    ].copy()

    if df_plot_data.empty:
        print(
            f"\nERROR: No data found for environment '{env_name}' in campaign '{campaign_id}' after filtering.",
            file=sys.stderr,
        )
        sys.exit(1)

    figure_dir = os.path.join(project_root, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    # --- CHANGE: Output filenames ---
    output_path_pdf = os.path.join(figure_dir, "fig_reversibility.pdf")
    output_path_eps = os.path.join(figure_dir, "fig_reversibility.eps")
    print(f"\nGenerating definitive Figure 3 from campaign: {campaign_id}")

    # --- CHANGE: Plotting setup for publication ---
    sns.set_theme(style="ticks", context="paper")
    fig, axes = plt.subplots(
        2, 2, figsize=(cm_to_inch(17.8), cm_to_inch(16)), constrained_layout=True
    )
    fig.suptitle(
        f"Fitness Landscape in a Periodic Environment ({env_name})", fontsize=12, y=1.03
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
    df_periodic_filtered = df_filtered[df_filtered["env_definition"] == env_name]
    phi_irr_val = find_nearest(df_periodic_filtered["phi"].unique(), -1.0)
    df_baseline_runs = df_periodic_filtered[
        np.isclose(df_periodic_filtered["phi"], phi_irr_val)
    ].copy()
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
        lw=2,
        ls="--",
        capsize=3,
        ms=5,
    )
    ax_d.errorbar(
        x=df_pareto_stats["b_m"],
        y=df_pareto_stats["mean"],
        yerr=df_pareto_stats["std"],
        label="Optimal Reversible Strategy",
        color="royalblue",
        marker="o",
        lw=2.5,
        ms=6,
        capsize=3,
    )
    ax_d.fill_between(
        df_pareto_stats["b_m"],
        df_pareto_stats["mean"],
        df_baseline_stats.set_index("b_m").reindex(df_pareto_stats["b_m"])["mean"],
        color="gold",
        alpha=0.3,
        label="Advantage of Reversibility",
    )

    # --- CHANGE: Font sizes ---
    ax_d.set_title("(D) Optimal Strategy Performance", fontsize=10)
    ax_d.set_xlabel("Mutant Fitness in Hostile Patch, $b_m$", fontsize=8)
    ax_d.set_ylabel("Long-Term Fitness (Front Speed)", fontsize=8)
    ax_d.grid(True, which="both", ls=":")
    ax_d.legend(fontsize=7)
    ax_d.tick_params(axis="both", which="major", labelsize=7)

    sns.despine(fig)
    # --- CHANGE: Save to PDF and EPS ---
    plt.savefig(output_path_pdf, bbox_inches="tight")
    plt.savefig(output_path_eps, bbox_inches="tight")
    print(f"\nDefinitive Figure 3 saved to: {output_path_pdf} and {output_path_eps}")


if __name__ == "__main__":
    main()
