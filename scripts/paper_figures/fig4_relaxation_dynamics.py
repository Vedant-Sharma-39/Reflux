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


# --- Setup Project Root Path ---
def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


project_root = get_project_root()
if project_root not in sys.path:
    sys.path.insert(0, project_root)




def load_timeseries_data(campaign_id, project_root, task_ids_to_load):
    """
    Loads individual, consolidated timeseries files for specific tasks.
    """
    ts_data = {task_id: None for task_id in task_ids_to_load}
    timeseries_dir = os.path.join(project_root, "data", campaign_id, "timeseries")

    if not os.path.isdir(timeseries_dir):
        print(f"Warning: Timeseries directory not found: {timeseries_dir}")
        return ts_data

    # Use the list of task IDs passed to the function to iterate
    for task_id in tqdm(task_ids_to_load, desc="Loading timeseries files"):
        ts_path = os.path.join(timeseries_dir, f"ts_{task_id}.json.gz")
        if os.path.exists(ts_path):
            try:
                with gzip.open(ts_path, "rt", encoding="utf-8") as f:
                    series = json.load(f)
                    ts_data[task_id] = pd.DataFrame(series)
            except (json.JSONDecodeError, gzip.BadGzipFile):
                print(f"\nWarning: Could not load or parse {ts_path}.")
                continue
    return ts_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 4: Relaxation Dynamics."
    )
    parser.add_argument(
        "campaign_id",
        default="fig4_relaxation_dynamics",
        nargs="?",
        help="Campaign ID for the relaxation dynamics experiment (default: fig4_relaxation_dynamics)",
    )
    args = parser.parse_args()

    df_summary = load_aggregated_data(args.campaign_id, project_root)
    if df_summary is None or df_summary.empty:
        sys.exit(f"Could not load data for campaign '{args.campaign_id}'. Aborting.")

    df_summary["task_id"] = df_summary["task_id"].astype(str)
    print(f"Loaded summary for {len(df_summary)} simulations.")

    df_summary["s"] = df_summary["b_m"] - 1.0

    # --- [THE FIX] ---
    # Select a few interesting parameter sets to plot robustly.
    # np.percentile can interpolate, creating a value that doesn't exist.
    # Instead, we get the sorted unique values and pick one by its index.
    s_values = np.sort(df_summary["s"].unique())
    if len(s_values) == 0:
        sys.exit("Error: No unique 's' values found in the data.")
    # Choose a deleterious value, for instance, the one at the 25th percentile of the *indices*.
    s_index = int(len(s_values) * 0.25)
    s_to_plot = s_values[s_index]
    print(f"Selected s={s_to_plot:.3f} for plotting relaxation dynamics.")
    # --- [END FIX] ---

    df_subset = df_summary[np.isclose(df_summary["s"], s_to_plot)]

    k_values = sorted(df_subset["k_total"].unique())
    if not k_values:
        sys.exit(f"Error: No data found for s={s_to_plot}. The subset is empty.")
    k_to_plot = (
        k_values
        if len(k_values) <= 3
        else [k_values[0], k_values[len(k_values) // 2], k_values[-1]]
    )

    task_ids_to_plot = df_subset[df_subset["k_total"].isin(k_to_plot)][
        "task_id"
    ].tolist()
    if not task_ids_to_plot:
        sys.exit(
            "Error: Failed to identify any task IDs for plotting. Check filtering logic."
        )

    s_target = -0.6
    s_values = df["s"].unique()
    s_to_plot = s_values[np.argmin(np.abs(s_values - s_target))]
    print(f"Selected s={s_to_plot:.3f} for plotting relaxation dynamics.")

    df_plot = df[df["s"] == s_to_plot].copy()

    timeseries_data = []
    for _, row in df_plot.iterrows():
        times = np.array(row["times"])
        mutant_fractions = np.array(row["mutant_fraction_timeseries"])
        valid_indices = np.where(times > 0)[0]
        if not valid_indices.any(): continue
        start_index = valid_indices[0]
        for t, mf in zip(times[start_index:], mutant_fractions[start_index:]):
            timeseries_data.append({
                "Time": t,
                "Mean Mutant Fraction, $\\langle\\rho_M\\rangle$": mf,
                "$k_{total}$": row["k_total"],
                "$\\phi$": row["phi"],
            })
    df_timeseries = pd.DataFrame(timeseries_data)

    sns.set_theme(style="ticks", context="talk")
    g = sns.relplot(
        data=df_timeseries,
        x="Time",
        y="Mean Mutant Fraction, $\\langle\\rho_M\\rangle$",
        hue="$k_{total}$",
        col="$\\phi$",
        kind="line",
        palette="viridis",
        height=5,
        aspect=1.1,
        facet_kws={"margin_titles": True},
    )

    # Add theoretical equilibrium lines and set log scale
    for ax, phi_val in zip(g.axes.flat, g.col_names):
        phi = float(str(phi_val).split('=')[-1].strip())
        rho_eq = (1 - phi) / 2.0
        ax.axhline(rho_eq, ls='--', color='red', zorder=1)
        ax.set_xscale('log')

    g.fig.suptitle(f"Figure 4: Relaxation Dynamics towards Equilibrium (s={s_to_plot:.2f})", y=1.05)
    g.set_titles(col_template="$\\phi$ = {col_name}")

    output_dir = os.path.join(project_root, "data", args.campaign_id, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "figure4_relaxation_dynamics.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure 4 saved to {output_path}")


if __name__ == "__main__":
    main()
