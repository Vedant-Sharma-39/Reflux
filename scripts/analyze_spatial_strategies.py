# FILE: scripts/analyze_spatial_strategies.py
# [v4.2 - FINAL VERSION] Definitive analysis with robust, position-aligned timeseries
# plotting and a corrected FacetGrid call to handle all experiment layouts.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, argparse, json
from tqdm import tqdm
import ast

# --- Robust Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))

from config import EXPERIMENTS
from data_utils import aggregate_data_cached

plt.style.use("seaborn-v0_8-whitegrid")


def plot_landscapes(df, value_col, suptitle, fig_path, cmap="viridis"):
    if df.empty:
        print(f"  Skipping plot '{suptitle}': No data to plot.")
        return

    if "patch0_width" in df.columns and "patch1_width" in df.columns:
        if df["patch0_width"].nunique() > 1 or df["patch1_width"].nunique() > 1:
            df["patch_config"] = df.apply(
                lambda row: f"{int(row['patch0_width'])}/{int(row['patch1_width'])}",
                axis=1,
            )
            col_var, row_var = "patch_config", "b_m"
            col_template, row_template = (
                "Patch Config = {col_name}",
                "$b_m$ = {row_name:.2f}",
            )
        else:
            col_var, row_var = "patch0_width", "patch1_width"
            col_template, row_template = "Patch 0 = {col_name}", "Patch 1 = {row_name}"
    else:
        col_var, row_var = "patch_width", "b_m"
        col_template, row_template = (
            "Patch Width = {col_name}",
            "$b_m$ = {row_name:.2f}",
        )

    # --- ### THE FIX ### ---
    # The `col_wrap` argument is removed because it conflicts with the `row` argument.
    # FacetGrid will now correctly create a rectangular grid for all cases.
    g = sns.FacetGrid(
        df,
        col=col_var,
        row=row_var,
        height=5,
        aspect=1.3,
        margin_titles=True,
        sharex=True,
        sharey=True,
    )
    # --- ### END FIX ### ---

    def draw_heatmap(data, color, **kwargs):
        if data.empty or data[value_col].isnull().all():
            return
        pivot_data = data.pivot_table(
            index="phi", columns="k_total", values=value_col
        ).sort_index(ascending=False)
        sns.heatmap(pivot_data, cmap=cmap, **kwargs)

    g.map_dataframe(draw_heatmap)
    g.set_axis_labels("Total Switching Rate ($k_{total}$)", "Switching Bias ($\\phi$)")
    g.set_titles(col_template=col_template, row_template=row_template)
    g.fig.suptitle(suptitle, fontsize=22, y=1.03)

    for ax, (name, data_slice) in zip(g.axes.flat, g.facet_data()):
        if (
            not data_slice.empty
            and value_col in data_slice.columns
            and not data_slice[value_col].isnull().all()
        ):
            try:
                idx_max = data_slice[value_col].idxmax()
                optimal_params = data_slice.loc[idx_max]
                k_opt, phi_opt = optimal_params["k_total"], optimal_params["phi"]
                pivot_data = data_slice.pivot_table(
                    index="phi", columns="k_total", values=value_col
                ).sort_index(ascending=False)
                y_coord = pivot_data.index.get_loc(phi_opt)
                x_coord = pivot_data.columns.get_loc(k_opt)
                ax.plot(
                    x_coord + 0.5,
                    y_coord + 0.5,
                    "r*",
                    markersize=15,
                    markeredgecolor="white",
                )
            except (KeyError, ValueError):
                pass

    for ax in g.axes.flat:
        current_labels = ax.get_xticklabels()
        if current_labels and any(l.get_text() for l in current_labels):
            if len(current_labels) > 10:
                ax.set_xticks(np.linspace(0, len(current_labels) - 1, 7))
                new_labels = (
                    f"{float(l.get_text()):.2g}" if l.get_text() else ""
                    for l in ax.get_xticklabels()
                )
                ax.set_xticklabels(list(new_labels), rotation=90)

    g.fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  -> Landscape plot saved to {fig_path}")


def robust_timeseries_parser(ts_string):
    if not isinstance(ts_string, str):
        return None
    try:
        return ast.literal_eval(ts_string)
    except (ValueError, SyntaxError):
        try:
            return json.loads(
                ts_string.replace("nan", "null")
                .replace("inf", "null")
                .replace("-inf", "null")
            )
        except json.JSONDecodeError:
            return None


def plot_timeseries_with_replicates(params, all_summary_data, campaign_id, figs_dir):
    b_m, phi, k_total = params["b_m"], params["phi"], params["k_total"]

    if "patch0_width" in params:
        patch0_w, patch1_w = params["patch0_width"], params["patch1_width"]
        mask = (
            np.isclose(all_summary_data["b_m"], b_m)
            & np.isclose(all_summary_data["phi"], phi)
            & np.isclose(all_summary_data["k_total"], k_total)
            & (all_summary_data["patch0_width"] == patch0_w)
            & (all_summary_data["patch1_width"] == patch1_w)
        )
        title_pw = f"patch_config={int(patch0_w)}/{int(patch1_w)}"
        cycle_len = patch0_w + patch1_w
    else:
        patch_width = params["patch_width"]
        mask = (
            np.isclose(all_summary_data["b_m"], b_m)
            & np.isclose(all_summary_data["phi"], phi)
            & np.isclose(all_summary_data["k_total"], k_total)
            & (all_summary_data["patch_width"] == patch_width)
        )
        title_pw = f"patch_width={patch_width}"
        cycle_len = patch_width * 2

    print(
        f"  -> Plotting position-aligned timeseries for b_m={b_m:.2f}, k={k_total:.4f}, phi={phi:.2f}, {title_pw}"
    )

    replicate_rows = all_summary_data.loc[mask]
    if replicate_rows.empty:
        return

    all_ts_data = []
    timeseries_dir = os.path.join(project_root, "data", campaign_id, "timeseries")
    for task_id in replicate_rows["task_id"]:
        ts_path = os.path.join(timeseries_dir, f"ts_{task_id}.json.gz")
        if os.path.exists(ts_path):
            try:
                temp_df = pd.read_json(ts_path, orient="records", compression="gzip")
                temp_df["replicate_id"] = task_id
                all_ts_data.append(temp_df)
            except Exception as e:
                print(f"     Warning: Could not read {ts_path}. Error: {e}")

    if not all_ts_data:
        print("     ERROR: Failed to find/parse any valid timeseries data.")
        return

    df_long = pd.concat(all_ts_data, ignore_index=True)
    df_long["speed_running_avg"] = df_long.groupby("replicate_id")[
        "front_speed"
    ].transform(lambda x: x.rolling(25, 1).mean())

    max_q = df_long["mean_front_q"].max()
    q_bins = pd.cut(df_long["mean_front_q"], bins=np.arange(0, max_q + 20, 10))
    df_stats = (
        df_long.groupby(q_bins)
        .agg(
            q_mean=("mean_front_q", "mean"),
            rho_mean=("mutant_fraction", "mean"),
            rho_sem=("mutant_fraction", "sem"),
            speed_mean=("speed_running_avg", "mean"),
            speed_sem=("speed_running_avg", "sem"),
        )
        .dropna()
    )

    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax2 = ax1.twinx()
    ax1.plot(
        df_stats["q_mean"],
        df_stats["rho_mean"],
        "-",
        color="crimson",
        lw=2.5,
        label="Mean Mutant Fraction",
    )
    ax1.fill_between(
        df_stats["q_mean"],
        df_stats["rho_mean"] - df_stats["rho_sem"],
        df_stats["rho_mean"] + df_stats["rho_sem"],
        color="crimson",
        alpha=0.2,
    )
    ax2.plot(
        df_stats["q_mean"],
        df_stats["speed_mean"],
        "-",
        color="darkblue",
        lw=2.5,
        label="Mean Running Avg. Speed",
    )
    ax2.fill_between(
        df_stats["q_mean"],
        df_stats["speed_mean"] - df_stats["speed_sem"],
        df_stats["speed_mean"] + df_stats["speed_sem"],
        color="darkblue",
        alpha=0.2,
    )
    ax1.set_xlabel("Mean Front Position (q)", fontsize=14)
    ax1.set_ylabel("Mutant Fraction", color="crimson", fontsize=14)
    ax2.set_ylabel("Front Speed", color="darkblue", fontsize=14)
    ax1.tick_params(axis="y", labelcolor="crimson")
    ax2.tick_params(axis="y", labelcolor="darkblue")
    ax1.set_ylim(-0.05, 1.05)
    ax3 = ax1.secondary_xaxis(
        "top", functions=(lambda q: q / cycle_len, lambda p: p * cycle_len)
    )
    ax3.set_xlabel("Environmental Cycle Number", fontsize=14, labelpad=10)
    title = f"Mean Dynamics vs. Position ({len(replicate_rows)} replicates)\n"
    title += f"$b_m={b_m:.2f}$, $k={k_total:.4f}$, $\\phi={phi:.2f}$, {title_pw}"
    ax1.set_title(title, fontsize=16)
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    filename = (
        f"pos_aligned_ts_bm{b_m:.2f}_k{k_total:.4f}_phi{phi:.2f}_{title_pw}.png".replace(
            "/", "-"
        )
        .replace(" ", "")
        .replace("=", "")
        .replace(",", "_")
        .replace("$", "")
    )
    plt.savefig(os.path.join(figs_dir, filename), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run v2 analysis of bet-hedging strategies."
    )
    parser.add_argument(
        "experiment_name", default="asymmetric_environment_v1", nargs="?"
    )
    args = parser.parse_args()

    CAMPAIGN_ID = EXPERIMENTS[args.experiment_name]["CAMPAIGN_ID"]
    FIGS_DIR = os.path.join(project_root, "figures", CAMPAIGN_ID)
    os.makedirs(FIGS_DIR, exist_ok=True)

    df_summary = aggregate_data_cached(
        CAMPAIGN_ID, project_root, cache_filename_suffix="summary_data"
    )
    if df_summary is None or df_summary.empty:
        sys.exit(
            "FATAL: No summary data found. Please run simulations and aggregate first."
        )

    group_keys = [
        k
        for k in [
            "k_total",
            "phi",
            "patch_width",
            "patch0_width",
            "patch1_width",
            "b_m",
        ]
        if k in df_summary.columns
    ]
    df_avg = (
        df_summary.groupby(group_keys)
        .agg(
            avg_front_speed=("avg_front_speed", "mean"),
            var_front_speed=("var_front_speed", "mean"),
        )
        .reset_index()
        .dropna()
    )

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

    timeseries_dir = os.path.join(FIGS_DIR, "mean_timeseries_dynamics")
    os.makedirs(timeseries_dir, exist_ok=True)

    print("\n--- Generating timeseries plots for optimal strategies ---")
    group_vars = [
        v
        for v in ["patch_width", "patch0_width", "patch1_width", "b_m"]
        if v in df_avg.columns
    ]
    if group_vars:
        optimal_indices = df_avg.loc[
            df_avg.groupby(group_vars)["avg_front_speed"].idxmax()
        ].index
        for _, row in df_avg.iloc[optimal_indices].iterrows():
            plot_timeseries_with_replicates(
                row.to_dict(), df_summary, CAMPAIGN_ID, timeseries_dir
            )


if __name__ == "__main__":
    main()
