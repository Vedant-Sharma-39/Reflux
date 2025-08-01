# FILE: scripts/analyze_spatial_strategies.py
# The definitive analysis script for the spatial_bet_hedging_v1 campaign.
# Generates fitness/risk landscapes and visualizes the dynamics of optimal strategies.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, argparse
from tqdm import tqdm

# --- Robust Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))

from config import EXPERIMENTS
from data_utils import aggregate_data_cached
from fluctuating_model import FluctuatingGillespieSimulation
from metrics import MetricsManager, FrontDynamicsTracker

plt.style.use("seaborn-v0_8-whitegrid")


def plot_landscapes(df, value_col, suptitle, fig_path, cmap="viridis"):
    """
    Generates faceted heatmaps for a given metric (e.g., mean speed or variance),
    and annotates the optimal point in each facet.
    """
    if df.empty:
        print(f"  Skipping plot '{suptitle}': No data to plot.")
        return

    g = sns.FacetGrid(
        df,
        col="patch_width",
        row="b_m",
        height=5,
        aspect=1.3,
        margin_titles=True,
        sharex=False,
        sharey=True,
    )

    def draw_heatmap(data, color, **kwargs):
        # Filter out empty or all-NaN data to prevent errors
        if data[value_col].isnull().all():
            return
        pivot_data = data.pivot_table(index="phi", columns="k_total", values=value_col)
        sns.heatmap(pivot_data, cmap=cmap, **kwargs)

    g.map_dataframe(draw_heatmap)
    g.set_axis_labels("Total Switching Rate ($k_{total}$)", "Switching Bias ($\\phi$)")
    g.set_titles(
        col_template="Patch Width = {col_name}", row_template="$b_m$ = {row_name:.2f}"
    )
    g.fig.suptitle(suptitle, fontsize=22, y=1.03)

    # Annotate the optimal strategy on each heatmap
    for ax, (name, data_slice) in zip(g.axes.flat, g.facet_data()):
        if (
            not data_slice.empty
            and value_col in data_slice.columns
            and not data_slice[value_col].isnull().all()
        ):
            idx_max = data_slice[value_col].idxmax()
            optimal_params = data_slice.loc[idx_max]
            k_opt, phi_opt = optimal_params["k_total"], optimal_params["phi"]

            pivot_data = data_slice.pivot_table(
                index="phi", columns="k_total", values=value_col
            )
            try:
                y_coord = pivot_data.index.get_loc(phi_opt)
                x_coord = pivot_data.columns.get_loc(k_opt)
                ax.plot(
                    x_coord + 0.5,
                    y_coord + 0.5,
                    "r*",
                    markersize=15,
                    markeredgecolor="white",
                    label="Optimal Strategy" if (ax == g.axes.flat[0]) else "",
                )
            except KeyError:
                pass

    g.fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  -> Landscape plot saved to {fig_path}")


def plot_example_timeseries(params, figs_dir):
    """
    [CORRECTED] Runs a single simulation and plots its time-series dynamics.
    Now correctly handles the 'environment_map' parameter.
    """
    print(
        f"  -> Generating example timeseries for optimal params: k={params['k_total']:.3f}, phi={params['phi']:.2f}"
    )

    # --- [THE FIX] ---
    # The 'params' dictionary passed to this function already contains the resolved
    # environment_map dictionary. We don't need to look it up again.
    # We just need to filter the params before unpacking.
    sim_constructor_params = {
        k: v
        for k, v in params.items()
        if k in FluctuatingGillespieSimulation.__init__.__code__.co_varnames
    }

    sim = FluctuatingGillespieSimulation(**sim_constructor_params)
    manager = MetricsManager()
    tracker = FrontDynamicsTracker(sim, log_interval=2.0)
    manager.add_tracker(tracker)

    total_run_time = 400.0
    with tqdm(
        total=total_run_time, desc="   Running viz sim", leave=False, ncols=80
    ) as pbar:
        last_time = 0
        while sim.time < total_run_time:
            did_step, _ = sim.step()
            if not did_step:
                break
            pbar.update(sim.time - last_time)
            last_time = sim.time

    df = tracker.get_dataframe()
    if df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax2 = ax1.twinx()
    ax1.plot(
        df["time"],
        df["mutant_fraction"],
        "-",
        color="crimson",
        label="Mutant Fraction ($\\rho_M$)",
    )
    ax2.plot(
        df["time"],
        df["front_speed"],
        "-",
        color="darkblue",
        alpha=0.6,
        label="Instantaneous Speed",
    )
    ax1.set_xlabel("Simulation Time")
    ax1.set_ylabel("Mutant Fraction", color="crimson")
    ax2.set_ylabel("Front Speed", color="darkblue")
    ax1.tick_params(axis="y", labelcolor="crimson")
    ax2.tick_params(axis="y", labelcolor="darkblue")
    ax1.set_ylim(0, 1)

    title = (
        f"Dynamics for Optimal Strategy (k={params['k_total']:.3f}, $\\phi$={params['phi']:.2f})\n"
        f"Environment: patch_width={params['patch_width']}, b_m={params['b_m']:.2f}"
    )
    ax1.set_title(title, fontsize=16)
    fig.tight_layout()

    filename = f"timeseries_pw{params['patch_width']}_bm{params['b_m']:.2f}.png"
    plt.savefig(os.path.join(figs_dir, filename), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Advanced analysis of spatial bet-hedging strategies."
    )
    parser.add_argument("experiment_name", default="spatial_bet_hedging_v1", nargs="?")
    args = parser.parse_args()

    CAMPAIGN_ID = EXPERIMENTS[args.experiment_name]["CAMPAIGN_ID"]
    FIGS_DIR = os.path.join(project_root, "figures", CAMPAIGN_ID)
    os.makedirs(FIGS_DIR, exist_ok=True)

    # --- 1. Load and Process Data ---
    df_raw = aggregate_data_cached(CAMPAIGN_ID, project_root)
    if df_raw is None or df_raw.empty:
        sys.exit("FATAL: No data found.")

    group_keys = ["k_total", "phi", "patch_width", "b_m"]
    df_avg = (
        df_raw.groupby(group_keys)
        .agg(
            avg_front_speed=("avg_front_speed", "mean"),
            var_front_speed=("var_front_speed", "mean"),
        )
        .reset_index()
        .dropna()
    )

    # --- 2. Generate Landscape Plots ---
    print(f"\n--- Generating Fitness & Risk Landscapes for {CAMPAIGN_ID} ---")
    plot_landscapes(
        df_avg,
        "avg_front_speed",
        "Fitness (Mean Speed) vs. Strategy",
        os.path.join(FIGS_DIR, "Fig1_Fitness_Landscapes.png"),
        cmap="viridis",
    )

    plot_landscapes(
        df_avg,
        "var_front_speed",
        "Risk (Speed Variance) vs. Strategy",
        os.path.join(FIGS_DIR, "Fig2_Risk_Landscapes.png"),
        cmap="magma",
    )

    # --- 3. Generate Example Timeseries Plots for Optimal Strategies ---
    print("\n--- Generating illustrative time-series for optimal points ---")
    timeseries_dir = os.path.join(FIGS_DIR, "optimal_strategy_dynamics")
    os.makedirs(timeseries_dir, exist_ok=True)

    base_params = EXPERIMENTS[args.experiment_name]["SIM_SETS"]["main_scan"][
        "base_params"
    ]

    # Find the single best strategy for each environmental condition (patch_width and b_m)
    optimal_indices = df_avg.loc[
        df_avg.groupby(["patch_width", "b_m"])["avg_front_speed"].idxmax()
    ].index
    conditions_to_plot = df_avg.iloc[optimal_indices]

    for _, row in conditions_to_plot.iterrows():
        run_params = base_params.copy()
        run_params.update(
            {
                "k_total": row["k_total"],
                "phi": row["phi"],
                "patch_width": int(row["patch_width"]),
                "b_m": row["b_m"],
            }
        )
        env_map_name = run_params["environment_map"]
        run_params["environment_map"] = EXPERIMENTS[args.experiment_name]["PARAM_GRID"][
            env_map_name
        ]

        plot_example_timeseries(run_params, timeseries_dir)

    print(f"\nAnalysis complete. Main figures are in: {FIGS_DIR}")
    print(f"Detailed dynamics plots are in: {timeseries_dir}")


if __name__ == "__main__":
    main()
