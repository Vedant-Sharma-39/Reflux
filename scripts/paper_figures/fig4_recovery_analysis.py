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

# --- Global Plotting Style ---
sns.set_theme(style="whitegrid", context="talk", font_scale=1.0)
FIG_DPI = 300


def get_project_root():
    """Find the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_and_prepare_summary_data(campaign_id: str, project_root: str) -> pd.DataFrame:
    """Loads and prepares the summary data."""
    summary_path = os.path.join(
        project_root,
        "data",
        campaign_id,
        "analysis",
        f"{campaign_id}_summary_aggregated.csv",
    )
    print(f"Loading summary data from: {os.path.basename(summary_path)}")
    try:
        df_summary = pd.read_csv(summary_path)
        if df_summary.empty:
            raise FileNotFoundError  # Treat empty file as not found
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(
            f"Error: Summary data not found or empty for campaign '{campaign_id}'. Cannot generate figures."
        )
        sys.exit(1)

    df_summary["task_id"] = df_summary["task_id"].astype(str)
    df_summary["s"] = np.round(df_summary["b_m"] - 1.0, 2)
    return df_summary


def load_timeseries_data(
    campaign_id: str, project_root: str, task_ids_to_load: list
) -> dict:
    """
    Loads detailed timeseries data for selected task IDs.
    Assumes `ts_*.json.gz` files contain a dict with a 'timeseries' key (from RecoveryDynamicsTracker)
    or are directly a list of dicts (from TimeSeriesTracker).
    """
    ts_data = {}
    timeseries_dir = os.path.join(project_root, "data", campaign_id, "timeseries")

    if not os.path.isdir(timeseries_dir):
        print(
            f"Warning: Timeseries directory not found: {timeseries_dir}",
            file=sys.stderr,
        )
        return ts_data

    print(f"Loading raw timeseries files for {len(task_ids_to_load)} tasks...")
    for task_id in tqdm(task_ids_to_load, desc="Loading timeseries files"):
        ts_path = os.path.join(
            timeseries_dir, f"ts_{task_id}.json.gz"
        )  # Standardized filename
        if os.path.exists(ts_path):
            try:
                with gzip.open(ts_path, "rt", encoding="utf-8") as f:
                    full_record = json.load(f)
                    # RecoveryDynamicsTracker stores timeseries under a 'timeseries' key
                    if "timeseries" in full_record and full_record["timeseries"]:
                        ts_data[task_id] = pd.DataFrame(full_record["timeseries"])
                    # TimeSeriesTracker might just output the list directly
                    elif (
                        isinstance(full_record, list)
                        and full_record
                        and "time" in full_record[0]
                    ):
                        ts_data[task_id] = pd.DataFrame(full_record)
                    else:
                        # Fallback for FrontDynamicsTracker if it was needed (not for current fig)
                        # or other unexpected structures
                        if (
                            "front_dynamics" in full_record
                            and full_record["front_dynamics"]
                        ):
                            ts_data[task_id] = pd.DataFrame(
                                full_record["front_dynamics"]
                            )
                        else:
                            print(
                                f"Warning: Timeseries data (expected format) not found in {ts_path} for task {task_id}.",
                                file=sys.stderr,
                            )
                            continue

            except (json.JSONDecodeError, gzip.BadGzipFile) as e:
                print(f"Error loading {ts_path}: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"Unexpected error processing {ts_path}: {e}", file=sys.stderr)
                continue
    return ts_data


def select_representative_params(df_summary: pd.DataFrame):
    """Selects representative parameter combinations for time series plotting."""

    # Define desired values from config.py's PARAM_GRID
    s_vals_target = [-0.9, -0.5, -0.1, 0.0]  # From bm_final_wide
    phi_vals_target = [-0.5, 0.0, 0.5]  # From phi_final_full
    k_vals_target = [0.01, 0.1, 1.0, 10.0]  # From k_total_final_log

    s_grid = sorted(df_summary["s"].unique())
    phi_grid = sorted(df_summary["phi"].unique())
    k_grid = sorted(df_summary["k_total"].unique())

    # Helper to find the nearest available value in the actual data grid
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    s_plot = [find_nearest(s_grid, val) for val in s_vals_target if s_grid]
    phi_plot = [find_nearest(phi_grid, val) for val in phi_vals_target if phi_grid]
    k_plot = [find_nearest(k_grid, val) for val in k_vals_target if k_grid]

    # Filter out duplicates if `find_nearest` maps multiple targets to the same actual value
    s_plot = sorted(list(set(s_plot)))
    phi_plot = sorted(list(set(phi_plot)))
    k_plot = sorted(list(set(k_plot)))

    print(f"Selected s for time series plot: {s_plot}")
    print(f"Selected phi for time series plot: {phi_plot}")
    print(f"Selected k_total for time series plot: {k_plot}")

    return s_plot, phi_plot, k_plot


def plot_mutant_fraction_timeseries(
    df_summary: pd.DataFrame, ts_data_map: dict, output_dir: str
):
    """
    Generates Figure 4A: Time series of mutant fraction recovery.
    """
    print("Generating Figure 4A: Mutant Fraction Time Series...")

    s_plot, phi_plot, k_plot = select_representative_params(df_summary)

    plot_dfs = []
    # Filter summary data for selected parameters to get task_ids
    # Using isin() for float comparison
    subset_summary = df_summary[
        df_summary["s"].apply(lambda x: any(np.isclose(x, val) for val in s_plot))
        & df_summary["phi"].apply(lambda x: any(np.isclose(x, val) for val in phi_plot))
        & df_summary["k_total"].apply(
            lambda x: any(np.isclose(x, val) for val in k_plot)
        )
    ].copy()  # Use .copy() to avoid SettingWithCopyWarning

    grouped_params = subset_summary.groupby(["s", "phi", "k_total"])

    for (s_val, phi_val, k_val), group in tqdm(
        grouped_params, desc="Processing time series for plotting"
    ):
        replicate_ts = []
        for task_id in group["task_id"]:
            if task_id in ts_data_map:
                replicate_ts.append(ts_data_map[task_id])

        if not replicate_ts:
            continue

        # Concatenate all replicates for this parameter set and calculate mean/SEM
        combined_ts = pd.concat(replicate_ts)

        # Ensure 'mutant_fraction' is present for this data
        if "mutant_fraction" not in combined_ts.columns:
            print(
                f"Warning: 'mutant_fraction' not found in timeseries for s={s_val}, phi={phi_val}, k={k_val}. Skipping."
            )
            continue

        # Group by time and aggregate mean/SEM for plotting
        avg_df = (
            combined_ts.groupby("time")["mutant_fraction"]
            .agg(["mean", "sem"])
            .reset_index()
        )
        avg_df["s"] = s_val
        avg_df["phi"] = phi_val
        avg_df["k_total"] = k_val
        plot_dfs.append(avg_df)

    if not plot_dfs:
        print(
            "No valid time series data to plot for Figure 4A. Skipping figure generation.",
            file=sys.stderr,
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "No Time Series Data Available",
            ha="center",
            va="center",
            fontsize=16,
        )
        ax.axis("off")
        plt.savefig(
            os.path.join(output_dir, "fig4A_mutant_fraction_timeseries.png"),
            dpi=FIG_DPI,
            bbox_inches="tight",
        )
        plt.close(fig)
        return

    final_plot_df = pd.concat(plot_dfs)

    # Convert s, phi, k_total to Categorical for explicit ordering in facets/hue
    final_plot_df["s_cat"] = pd.Categorical(
        final_plot_df["s"], categories=s_plot, ordered=True
    )
    final_plot_df["phi_cat"] = pd.Categorical(
        final_plot_df["phi"], categories=phi_plot, ordered=True
    )
    final_plot_df["k_total_cat"] = pd.Categorical(
        final_plot_df["k_total"], categories=k_plot, ordered=True
    )

    g = sns.relplot(
        data=final_plot_df,
        x="time",
        y="mean",
        col="s_cat",
        row="phi_cat",
        hue="k_total_cat",
        kind="line",
        palette="viridis",  # Good for sequential k_total
        errorbar="sem",  # Seaborn will draw error bars
        linewidth=2,
        height=3,
        aspect=1.2,
        col_wrap=len(s_plot),  # Ensure columns wrap after 's' panels
        facet_kws={"margin_titles": True},
        legend="full",
    )

    g.set_axis_labels("Time", r"Mean Mutant Fraction, $\langle\rho_M\rangle$")
    g.set_titles(
        col_template=r"$s = {col_name:.2f}$", row_template=r"$\phi = {row_name:.2f}$"
    )
    g.fig.suptitle(
        "Figure 4A: Relaxation Dynamics of Mutant Fraction", y=1.02, fontsize=18
    )
    g.legend.set_title(r"$k_{total}$")

    # Adjust legend position to avoid overlapping with subplots if many panels
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1), frameon=True)
    g.tight_layout(rect=[0, 0, 0.96, 0.98])  # Adjust for legend and suptitle

    plt.savefig(
        os.path.join(output_dir, "fig4A_mutant_fraction_timeseries.png"),
        dpi=FIG_DPI,
        bbox_inches="tight",
    )
    plt.close(g.fig)  # Close the figure to free memory
    print(
        f"Figure 4A saved to {os.path.join(output_dir, 'fig4A_mutant_fraction_timeseries.png')}"
    )


def plot_final_state_heatmaps(df_summary: pd.DataFrame, output_dir: str):
    """
    Generates Figure 4B and 4C: Heatmaps of final steady-state mutant fraction and front speed.
    """
    print("Generating Figure 4B & 4C: Final Steady-State Heatmaps...")

    # Filter out k_total = 0, as it's the non-switching baseline, typically not on switching heatmaps
    df_filtered = df_summary[df_summary["k_total"] > 0].copy()

    # Get sorted unique values for consistent heatmap axes and titles
    s_order = sorted(df_filtered["s"].unique())
    phi_order = sorted(df_filtered["phi"].unique())
    k_order = sorted(df_filtered["k_total"].unique())

    # Create figure for Figure 4B (Mutant Fraction)
    fig_b, axes_b = plt.subplots(
        1, len(s_order), figsize=(4 * len(s_order), 5), sharey=True
    )
    if len(s_order) == 1:
        axes_b = [axes_b]  # Ensure axes_b is iterable for single subplot case
    fig_b.suptitle(
        "Figure 4B: Final Steady-State Mutant Fraction, $\\langle\\rho_M\\rangle_{final}$",
        y=1.05,
        fontsize=18,
    )

    # Create figure for Figure 4C (Front Speed)
    fig_c, axes_c = plt.subplots(
        1, len(s_order), figsize=(4 * len(s_order), 5), sharey=True
    )
    if len(s_order) == 1:
        axes_c = [axes_c]
    fig_c.suptitle(
        "Figure 4C: Final Steady-State Front Speed, $\\langle v \\rangle_{final}$",
        y=1.05,
        fontsize=18,
    )

    for i, s_val in enumerate(s_order):
        df_s = df_filtered[
            np.isclose(df_filtered["s"], s_val)
        ].copy()  # Use np.isclose for float comparison

        if df_s.empty:
            print(
                f"No data for s={s_val}. Skipping heatmap for this value.",
                file=sys.stderr,
            )
            axes_b[i].set_title(f"No data for $s={s_val:.2f}$")
            axes_b[i].axis("off")
            axes_c[i].set_title(f"No data for $s={s_val:.2f}$")
            axes_c[i].axis("off")
            continue

        # Pivot data for heatmap, ensuring index and columns are in desired order
        pivot_rho = df_s.pivot_table(
            index="phi", columns="k_total", values="avg_rho_M_final"
        )
        pivot_speed = df_s.pivot_table(
            index="phi", columns="k_total", values="avg_front_speed_final"
        )

        pivot_rho = pivot_rho.reindex(index=phi_order, columns=k_order)
        pivot_speed = pivot_speed.reindex(index=phi_order, columns=k_order)

        # Plot for avg_rho_M_final (Figure 4B)
        sns.heatmap(
            pivot_rho,
            ax=axes_b[i],
            cmap="viridis",  # Green-yellow for concentration
            cbar=False,  # We'll add a single colorbar outside the loop
            vmin=0,
            vmax=1,  # Mutant fraction ranges from 0 to 1
            linewidths=0.5,
            linecolor="lightgray",
            # fmt=".2f", annot=True if pivot_rho.shape[0]*pivot_rho.shape[1] < 50 else False # Annotate if matrix is small
        )
        axes_b[i].set_title(r"$s = {:.2f}$".format(s_val))
        axes_b[i].set_xlabel(r"$k_{total}$")
        if i == 0:
            axes_b[i].set_ylabel(r"Bias, $\phi$")
        else:
            axes_b[i].set_ylabel("")  # Hide y-label for subsequent panels

        # Plot for avg_front_speed_final (Figure 4C)
        sns.heatmap(
            pivot_speed,
            ax=axes_c[i],
            cmap="rocket_r",  # Or "magma", "plasma" for speed
            cbar=False,  # Single colorbar outside the loop
            linewidths=0.5,
            linecolor="lightgray",
            # fmt=".2f", annot=True if pivot_speed.shape[0]*pivot_speed.shape[1] < 50 else False
        )
        axes_c[i].set_title(r"$s = {:.2f}$".format(s_val))
        axes_c[i].set_xlabel(r"$k_{total}$")
        if i == 0:
            axes_c[i].set_ylabel(r"Bias, $\phi$")
        else:
            axes_c[i].set_ylabel("")  # Hide y-label for subsequent panels

        # Make k_total labels logarithmic
        axes_b[i].set_xticks(np.arange(len(k_order)))
        axes_b[i].set_xticklabels(
            [f"{k:.2f}" for k in k_order], rotation=45, ha="right"
        )
        axes_c[i].set_xticks(np.arange(len(k_order)))
        axes_c[i].set_xticklabels(
            [f"{k:.2f}" for k in k_order], rotation=45, ha="right"
        )

    # Adjust layout and add single colorbars
    fig_b.tight_layout(
        rect=[0, 0, 0.95, 0.98]
    )  # Adjust for suptitle and colorbar space
    cbar_ax_b = fig_b.add_axes(
        [0.96, 0.15, 0.015, 0.7]
    )  # [left, bottom, width, height]
    # Use the mappable from the last heatmap drawn to correctly show the color range
    if axes_b and axes_b[-1].collections:  # Check if axes and collections exist
        mappable_b = axes_b[-1].collections[0]
        fig_b.colorbar(
            mappable_b, cax=cbar_ax_b, label=r"$\langle\rho_M\rangle_{final}$"
        )
    else:
        print("Warning: No heatmaps drawn for Fig 4B to attach colorbar.")

    fig_c.tight_layout(rect=[0, 0, 0.95, 0.98])
    cbar_ax_c = fig_c.add_axes([0.96, 0.15, 0.015, 0.7])
    if axes_c and axes_c[-1].collections:
        mappable_c = axes_c[-1].collections[0]
        fig_c.colorbar(mappable_c, cax=cbar_ax_c, label=r"$\langle v \rangle_{final}$")
    else:
        print("Warning: No heatmaps drawn for Fig 4C to attach colorbar.")

    # Save figures
    fig_b_path = os.path.join(output_dir, "fig4B_final_mutant_fraction_heatmap.png")
    fig_c_path = os.path.join(output_dir, "fig4C_final_front_speed_heatmap.png")

    fig_b.savefig(fig_b_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"Figure 4B saved to {fig_b_path}")
    fig_c.savefig(fig_c_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"Figure 4C saved to {fig_c_path}")

    plt.close(fig_b)  # Close figures to free memory
    plt.close(fig_c)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 4: Relaxation and Recovery Dynamics."
    )
    parser.add_argument(
        "campaign_id",
        default="recovery_timescale",  # Default to the richer dataset
        nargs="?",
        help="Campaign ID (e.g., 'recovery_timescale' or 'relaxation_dynamics').",
    )
    args = parser.parse_args()
    project_root = get_project_root()

    output_dir = os.path.join(project_root, "data", args.campaign_id, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    df_summary = load_and_prepare_summary_data(args.campaign_id, project_root)

    # Check if final state data columns exist, implies RecoveryDynamicsTracker was used
    has_final_state_metrics = (
        "avg_rho_M_final" in df_summary.columns
        and "avg_front_speed_final" in df_summary.columns
    )

    # --- Figure 4A: Time Series of Mutant Fraction ---
    # Get all task_ids that have timeseries data in summary for potential loading
    all_ts_task_ids = df_summary["task_id"].tolist()

    ts_data_map = load_timeseries_data(args.campaign_id, project_root, all_ts_task_ids)
    if ts_data_map:
        plot_mutant_fraction_timeseries(df_summary, ts_data_map, output_dir)
    else:
        print("Skipping Figure 4A: No timeseries data loaded. Check data paths/format.")

    # --- Figures 4B & 4C: Heatmaps of Final Steady State ---
    if has_final_state_metrics:
        plot_final_state_heatmaps(df_summary, output_dir)
    else:
        print(
            "Skipping Figures 4B & 4C: Final state metrics (avg_rho_M_final, avg_front_speed_final) not found in summary data."
        )
        print(
            "This is expected if the campaign did not use RecoveryDynamicsTracker (e.g., 'relaxation_dynamics')."
        )

    print("\nFigure generation complete.")


if __name__ == "__main__":
    main()
