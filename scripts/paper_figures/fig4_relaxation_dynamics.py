# FILE: scripts/paper_figures/fig4_relaxation_dynamics.py (Standardized & Robust Version)
import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import gzip
from tqdm import tqdm


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_timeseries_data(campaign_id, project_root, task_ids_to_load):
    ts_data = {}
    timeseries_dir = os.path.join(project_root, "data", campaign_id, "timeseries")
    if not os.path.isdir(timeseries_dir):
        print(
            f"Warning: Timeseries directory not found: {timeseries_dir}",
            file=sys.stderr,
        )
        return ts_data
    for task_id in tqdm(task_ids_to_load, desc="Loading timeseries files"):
        ts_path = os.path.join(timeseries_dir, f"ts_{task_id}.json.gz")
        if os.path.exists(ts_path):
            try:
                with gzip.open(ts_path, "rt", encoding="utf-8") as f:
                    ts_data[task_id] = pd.DataFrame(json.load(f))
            except (json.JSONDecodeError, gzip.BadGzipFile):
                continue
    return ts_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 4: Relaxation Dynamics."
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

    try:
        df_summary = pd.read_csv(summary_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_summary = pd.DataFrame()

    output_dir = os.path.join(project_root, "data", args.campaign_id, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "figure4_relaxation.png")

    if df_summary.empty:
        print(
            f"Warning: No data found for campaign '{args.campaign_id}'. Cannot generate Figure 4."
        )
        fig, _ = plt.subplots()
        fig.text(0.5, 0.5, "Figure 4: No Data Available", ha="center", va="center")
        plt.savefig(output_path, dpi=300)
        sys.exit(0)

    df_summary["task_id"] = df_summary["task_id"].astype(str)
    df_summary["s"] = df_summary["b_m"] - 1.0
    s_values = np.sort(df_summary["s"].unique())
    s_to_plot = s_values[int(len(s_values) * 0.25)] if len(s_values) > 0 else None

    if s_to_plot is None:
        sys.exit("Error: No unique 's' values found in the data.")

    df_subset = df_summary[np.isclose(df_summary["s"], s_to_plot)]
    k_values = sorted(df_subset["k_total"].unique())
    k_to_plot = (
        [k_values[0], k_values[len(k_values) // 2], k_values[-1]]
        if len(k_values) > 3
        else k_values
    )
    task_ids_to_plot = df_subset[df_subset["k_total"].isin(k_to_plot)][
        "task_id"
    ].tolist()

    ts_data_map = load_timeseries_data(args.campaign_id, project_root, task_ids_to_plot)

    plot_data = []
    for k in k_to_plot:
        task_ids_for_k = df_subset[df_subset["k_total"] == k]["task_id"]
        all_dfs_for_k = [
            ts_data_map[tid]
            for tid in task_ids_for_k
            if tid in ts_data_map
            and ts_data_map[tid] is not None
            and not ts_data_map[tid].empty
        ]
        if not all_dfs_for_k:
            continue
        combined_df = pd.concat(all_dfs_for_k)
        avg_df = (
            combined_df.groupby("time")["mutant_fraction"]
            .agg(["mean", "sem"])
            .reset_index()
        )
        avg_df["k_total"] = k
        plot_data.append(avg_df)

    if not plot_data:
        print(
            "Warning: No valid timeseries data could be processed for plotting.",
            file=sys.stderr,
        )
        fig, _ = plt.subplots()
        fig.text(0.5, 0.5, "Figure 4: No Timeseries Data", ha="center", va="center")
        plt.savefig(output_path, dpi=300)
        sys.exit(0)

    final_plot_df = pd.concat(plot_data)
    sns.set_theme(style="darkgrid", context="talk")
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("viridis", n_colors=len(k_to_plot))
    for i, k in enumerate(k_to_plot):
        data = final_plot_df[final_plot_df["k_total"] == k]
        plt.plot(data["time"], data["mean"], label=f"{k:.2f}", color=palette[i], lw=2.5)
        plt.fill_between(
            data["time"],
            data["mean"] - data["sem"],
            data["mean"] + data["sem"],
            color=palette[i],
            alpha=0.2,
        )
    plt.title(f"Relaxation Dynamics (s={s_to_plot:.2f}, $\\phi=0.0$)")
    plt.xlabel("Time")
    plt.ylabel(r"Mean Mutant Fraction, $\langle\rho_M\rangle$")
    plt.legend(title=r"$k_{total}$", loc="best")
    plt.ylim(-0.05, 1.05)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure 4 saved to {output_path}")


if __name__ == "__main__":
    main()
