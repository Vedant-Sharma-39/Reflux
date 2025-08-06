# FILE: scripts/paper_figures/fig4_relaxation_dynamics.py

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


def load_timeseries_data(ts_db_path, task_ids_to_load):
    """Loads specific timeseries from the gzipped JSONL database."""
    ts_data = {task_id: None for task_id in task_ids_to_load}
    task_ids_to_load_set = set(task_ids_to_load)

    with gzip.open(ts_db_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading timeseries DB"):
            try:
                data = json.loads(line)
                task_id = str(data.get("task_id"))
                if task_id in task_ids_to_load_set:
                    ts_data[task_id] = pd.DataFrame(data.get("timeseries", []))
            except (json.JSONDecodeError, KeyError):
                continue
    return ts_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 4: Relaxation Dynamics."
    )
    parser.add_argument(
        "campaign_id", help="Campaign ID for relaxation/perturbation runs."
    )
    args = parser.parse_args()

    project_root = get_project_root()
    analysis_dir = os.path.join(project_root, "data", args.campaign_id, "analysis")
    summary_path = os.path.join(
        analysis_dir, f"{args.campaign_id}_summary_aggregated.csv"
    )
    ts_db_path = os.path.join(
        project_root, "data", args.campaign_id, "timeseries_raw", "ts_chunk_0.jsonl.gz"
    )  # Path in debug mode

    if not os.path.exists(summary_path):
        sys.exit(f"Error: Summary file not found: {summary_path}")

    # In a real run, the consolidated DB would be used. For debug, we point to a raw file.
    if not os.path.exists(ts_db_path):
        ts_db_path = os.path.join(
            analysis_dir, f"{args.campaign_id}_timeseries_db.jsonl.gz"
        )
        if not os.path.exists(ts_db_path):
            sys.exit(
                f"Error: Timeseries database not found. Looked in raw and analysis dirs."
            )

    df_summary = pd.read_csv(summary_path)
    df_summary["task_id"] = df_summary["task_id"].astype(str)
    print(f"Loaded summary for {len(df_summary)} simulations.")

    # Select a few interesting parameter sets to plot
    df_summary["s"] = df_summary["b_m"] - 1.0
    s_to_plot = np.percentile(df_summary["s"].unique(), 25)  # A deleterious value

    df_subset = df_summary[np.isclose(df_summary["s"], s_to_plot)]
    k_values = sorted(df_subset["k_total"].unique())
    if len(k_values) > 3:
        k_to_plot = [k_values[0], k_values[len(k_values) // 2], k_values[-1]]
    else:
        k_to_plot = k_values

    task_ids_to_plot = df_subset[df_subset["k_total"].isin(k_to_plot)][
        "task_id"
    ].tolist()

    ts_data_map = load_timeseries_data(ts_db_path, task_ids_to_plot)

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
        sys.exit("Error: No valid timeseries data could be processed for plotting.")

    final_plot_df = pd.concat(plot_data)

    # Create Plot
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

    # Save Figure
    output_path = os.path.join(analysis_dir, "figure4_relaxation.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure 4 saved to {output_path}")


if __name__ == "__main__":
    main()
