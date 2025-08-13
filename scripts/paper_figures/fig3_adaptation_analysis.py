# FILE: scripts/paper_figures/fig3_adaptation_analysis.py
# Generates the definitive Figure 3, with a simplified and powerful Panel D
# directly comparing the optimal reversible strategy to the irreversible baseline.

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
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def plot_strategy_panel(ax, df_s_slice, title):
    """Helper function to plot a single fitness-vs-k panel for a fixed s."""
    sns.lineplot(
        data=df_s_slice,
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
        campaign_id = EXPERIMENTS["bet_hedging_final"]["campaign_id"]
    except KeyError:
        print(
            "Error: 'bet_hedging_final' experiment not found in src/config.py.",
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
    output_path = os.path.join(figure_dir, "fig3_adaptation_analysis.png")

    print(f"Generating definitive Figure 3 from campaign: {campaign_id}")
    df = pd.read_csv(summary_path)

    # --- Definitive Two-Step Filtering Protocol ---
    STALL_THRESHOLD = 0.20
    stall_counts = df[df["termination_reason"] == "stalled_or_boundary_hit"][
        "phi"
    ].value_counts()
    total_counts = df["phi"].value_counts()
    stall_rates = stall_counts.div(total_counts, fill_value=0)
    phi_to_exclude = stall_rates[stall_rates > STALL_THRESHOLD].index.tolist()
    if phi_to_exclude:
        df_for_analysis = df[~df["phi"].isin(phi_to_exclude)].copy()
    else:
        df_for_analysis = df.copy()
    df_filtered = df_for_analysis[
        df_for_analysis["termination_reason"] != "stalled_or_boundary_hit"
    ].copy()
    df_filtered["s"] = df_filtered["b_m"] - 1.0

    df_plot_data = df_filtered[
        (df_filtered["patch_width"] == 60) & (df_filtered["k_total"] > 0)
    ].copy()

    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), constrained_layout=True)
    fig.suptitle(
        "The Fitness Landscape is Rugged and Tuned by Selection", fontsize=28, y=1.03
    )

    s_all = np.sort(df_plot_data["s"].unique())
    s_targets = [-0.1, -0.5, -0.8]
    s_vals = [find_nearest(s_all, s) for s in s_targets]

    panel_map = {
        s_vals[0]: (axes[0, 0], f"(A) Weak Selection (s = {s_vals[0]:.2f})"),
        s_vals[1]: (axes[0, 1], f"(B) Medium Selection (s = {s_vals[1]:.2f})"),
        s_vals[2]: (axes[1, 0], f"(C) Strong Selection (s = {s_vals[2]:.2f})"),
    }

    for s_val, (ax, title) in panel_map.items():
        df_panel = df_plot_data[np.isclose(df_plot_data["s"], s_val)]
        if not df_panel.empty:
            plot_strategy_panel(ax, df_panel, title)

    # --- Simplified Panel D: Optimal Reversible vs. Irreversible Baseline ---
    ax_d = axes[1, 1]

    # 1. Get the irreversible baseline (using the original full dataframe)
    phi_irr_val = find_nearest(df["phi"].unique(), -1.0)
    # --- FIX: Create an explicit copy to avoid SettingWithCopyWarning ---
    df_baseline_runs = df[
        (np.isclose(df["phi"], phi_irr_val))
        & (df["termination_reason"] != "stalled_or_boundary_hit")
    ].copy()
    df_baseline_runs["s"] = df_baseline_runs["b_m"] - 1.0
    df_baseline_fitness = (
        df_baseline_runs.groupby("s")["avg_front_speed"].mean().reset_index()
    )

    # 2. Get the "Pareto frontier" - the best possible performance at each 's' from reversible strategies
    df_mean_rev = (
        df_plot_data.groupby(["s", "phi"])["avg_front_speed"].max().reset_index()
    )
    df_pareto = df_mean_rev.groupby("s")["avg_front_speed"].max().reset_index()

    # Plotting Panel D
    ax_d.plot(
        df_baseline_fitness["s"],
        df_baseline_fitness["avg_front_speed"],
        label="Irreversible Strategy ($\\phi=-1.0$)",
        color="crimson",
        marker="s",
        lw=3.5,
        ls="--",
    )

    ax_d.plot(
        df_pareto["s"],
        df_pareto["avg_front_speed"],
        label="Optimal Reversible Strategy",
        color="royalblue",
        marker="o",
        lw=4,
        ms=10,
    )

    # Shade the area between the curves to represent the "Advantage of Reversibility"
    ax_d.fill_between(
        df_pareto["s"],
        df_pareto["avg_front_speed"],
        df_baseline_fitness.set_index("s").reindex(df_pareto["s"])["avg_front_speed"],
        color="gold",
        alpha=0.3,
        label="Advantage of Reversibility",
    )

    ax_d.set_title("(D) Optimal Strategy Performance", fontsize=20)
    ax_d.set_xlabel("Selection Strength, $s$", fontsize=16)
    ax_d.set_ylabel("Long-Term Fitness (Front Speed)", fontsize=16)
    ax_d.grid(True, which="both", ls=":")
    ax_d.legend(fontsize=14)

    sns.despine(fig)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nDefinitive Figure 3 saved to: {output_path}")


if __name__ == "__main__":
    main()
