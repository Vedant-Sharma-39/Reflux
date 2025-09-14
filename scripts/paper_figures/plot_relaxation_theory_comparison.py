# FILE: scripts/paper_figures/plot_relaxation_with_s_eff_theory.py (Standalone, v4 - Adaptive X-Axis)
#
# This standalone script first calculates s_eff from steady-state data.
# It then plots the relaxation dynamics, comparing simulation to the ODE.
#
# CORRECTIONS:
# 1. Implemented an adaptive x-axis for each subplot to zoom in on the
#    relevant relaxation timescale, preventing wasted space on plots.

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import gzip
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from tqdm import tqdm

# =========================================================================
# HELPER FUNCTIONS and GLOBAL SETUP
# =========================================================================


def get_project_root():
    """Dynamically finds the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


PROJECT_ROOT = get_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def calculate_theoretical_rho_m(k_total, s, phi):
    """Calculates the theoretical steady-state mutant fraction."""
    if np.isclose(s, 0):
        return np.full_like(k_total, (1 - phi) / 2.0, dtype=float)
    k_total = np.asarray(k_total, dtype=float)
    sqrt_term = (s + k_total) ** 2 - 4 * s * (k_total / 2.0 * (1.0 - phi))
    sqrt_term[sqrt_term < 0] = 0
    term_sqrt = np.sqrt(sqrt_term)
    numerator = (s + k_total) - term_sqrt
    denominator = 2 * s
    return np.clip(numerator / denominator, 0, 1)


def load_timeseries_for_params(df_summary, campaign_id, project_root, query_params):
    """Loads gzipped timeseries data for specific parameters."""
    query_str = " & ".join([f"`{k}`=={v}" for k, v in query_params.items()])
    task_ids = df_summary.query(query_str)["task_id"].tolist()
    all_timeseries = []
    ts_dir = os.path.join(project_root, "data", campaign_id, "timeseries")
    for task_id in task_ids:
        file_path = os.path.join(ts_dir, f"ts_{task_id}.json.gz")
        try:
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                all_timeseries.append(pd.DataFrame(json.load(f)))
        except (FileNotFoundError, json.JSONDecodeError, gzip.BadGzipFile):
            continue
    return all_timeseries


def relaxation_ode(f, t, s_eff, k_total, phi):
    """The ODE that is mathematically consistent with calculate_theoretical_rho_m."""
    k_wt_m = (k_total / 2.0) * (1.0 - phi)
    k_m_wt = (k_total / 2.0) * (1.0 + phi)
    selection_term = -s_eff * f * (1.0 - f)
    switching_term = k_wt_m * (1.0 - f) - k_m_wt * f
    return selection_term + switching_term


# =========================================================================
# MAIN SCRIPT LOGIC
# =========================================================================


def main():
    from src.config import EXPERIMENTS

    ts_campaign_id = EXPERIMENTS["homogeneous_timeseries"]["campaign_id"]
    pd_campaign_id = EXPERIMENTS["phase_diagram"]["campaign_id"]
    figure_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    # --- PART 1: Calculate s_eff from the phase diagram data ---
    pd_summary_path = os.path.join(
        PROJECT_ROOT,
        "data",
        pd_campaign_id,
        "analysis",
        f"{pd_campaign_id}_summary_aggregated.csv",
    )
    print(f"Loading phase diagram data to calculate s_eff: {pd_summary_path}")
    df_pd_raw = pd.read_csv(pd_summary_path)
    df_pd_avg = (
        df_pd_raw.groupby(["b_m", "phi", "k_total"])
        .agg(rho_mean=("avg_rho_M", "mean"), rho_std=("avg_rho_M", "std"))
        .reset_index()
    )

    s_eff_results = []
    grouped = df_pd_avg.groupby(["b_m", "phi"])
    print("\nCalculating s_eff values from steady-state data...")
    for (b_m_val, phi_val), group in tqdm(grouped, total=len(grouped)):
        fit_func = lambda k, s_eff: calculate_theoretical_rho_m(k, s_eff, phi_val)
        try:
            popt, _ = curve_fit(
                fit_func,
                group["k_total"],
                group["rho_mean"],
                p0=[1.0],
                sigma=group["rho_std"].fillna(1e-3).replace(0, 1e-3),
                absolute_sigma=True,
                bounds=(0, 10),
            )
            s_eff_results.append({"b_m": b_m_val, "phi": phi_val, "s_eff": popt[0]})
        except (RuntimeError, ValueError):
            continue
    df_s_eff = pd.DataFrame(s_eff_results)
    print(f"s_eff calculation complete. Found {len(df_s_eff)} valid parameter sets.")

    # --- PART 2: Plot relaxation dynamics and compare with ODE ---
    ts_summary_path = os.path.join(
        PROJECT_ROOT,
        "data",
        ts_campaign_id,
        "analysis",
        f"{ts_campaign_id}_summary_aggregated.csv",
    )
    print(f"\nLoading timeseries data for plotting: {ts_summary_path}")
    df_ts_summary = pd.read_csv(ts_summary_path)

    b_m_vals = sorted(df_ts_summary["b_m"].unique())
    phi_vals = sorted(df_ts_summary["phi"].unique())
    k_total_vals = sorted(df_ts_summary["k_total"].unique())
    initial_sizes = sorted(df_ts_summary["initial_mutant_patch_size"].unique())
    width = df_ts_summary["width"].iloc[0]
    palette = sns.color_palette("coolwarm", n_colors=len(initial_sizes))

    for phi in phi_vals:
        print(f"\nGenerating theory comparison plot for phi = {phi:.2f}...")
        fig, axes = plt.subplots(
            len(b_m_vals),
            len(k_total_vals),
            figsize=(8 * len(k_total_vals), 6 * len(b_m_vals)),
            constrained_layout=True,
            sharey=True,
        )
        fig.suptitle(
            f"Relaxation Dynamics vs. Mean-Field Theory (φ = {phi:.2f})",
            fontsize=28,
            y=1.04,
        )

        for i, b_m in enumerate(b_m_vals):
            for j, k_total in enumerate(k_total_vals):
                ax = axes[i, j]
                ax.set_title(f"b$_m$={b_m:.2f}, k$_t$={k_total:.2g}", fontsize=16)

                s_eff_row = df_s_eff[
                    (np.isclose(df_s_eff["b_m"], b_m))
                    & (np.isclose(df_s_eff["phi"], phi))
                ]
                s_eff = s_eff_row["s_eff"].iloc[0] if not s_eff_row.empty else 0.0

                all_mean_trajectories = []
                max_time_for_plot = 0
                for k, size in enumerate(initial_sizes):
                    query = {
                        "b_m": b_m,
                        "phi": phi,
                        "k_total": k_total,
                        "initial_mutant_patch_size": size,
                    }
                    timeseries_list = load_timeseries_for_params(
                        df_ts_summary, ts_campaign_id, PROJECT_ROOT, query
                    )
                    if not timeseries_list:
                        continue

                    for df_ts in timeseries_list:
                        ax.plot(
                            df_ts["time"],
                            df_ts["mutant_fraction"],
                            color=palette[k],
                            lw=1.5,
                            alpha=0.2,
                        )

                    all_times = np.concatenate(
                        [df_ts["time"].values for df_ts in timeseries_list]
                    )
                    max_time_for_plot = max(
                        max_time_for_plot, np.percentile(all_times, 98)
                    )

                    common_time = (
                        np.linspace(0, max_time_for_plot, 200)
                        if max_time_for_plot > 0
                        else np.array([0])
                    )
                    interpolated_rhos = [
                        np.interp(common_time, df_ts["time"], df_ts["mutant_fraction"])
                        for df_ts in timeseries_list
                    ]
                    mean_rho = np.mean(interpolated_rhos, axis=0)
                    all_mean_trajectories.append(
                        (common_time, mean_rho)
                    )  # Store for auto-axis
                    ax.plot(
                        common_time,
                        mean_rho,
                        color=palette[k],
                        lw=3.5,
                        label=f"Sim Avg (Initial: {size/width:.0%})",
                    )

                    f0 = size / width
                    ode_time_vec = (
                        np.linspace(0, max_time_for_plot, 200)
                        if max_time_for_plot > 0
                        else np.array([0])
                    )
                    if len(ode_time_vec) > 1:
                        ode_solution = odeint(
                            relaxation_ode, f0, ode_time_vec, args=(s_eff, k_total, phi)
                        )
                        ax.plot(
                            ode_time_vec,
                            ode_solution,
                            color="black",
                            linestyle="--",
                            lw=3,
                            label=f"Theory (s_eff={s_eff:.2f})",
                        )

                # --- NEW: ADAPTIVE X-AXIS LOGIC ---
                if all_mean_trajectories:
                    # Find the final steady state value from the longest trajectory
                    final_val = all_mean_trajectories[-1][1][-1]
                    convergence_time = max_time_for_plot

                    # Find the time when all trajectories have converged
                    for t_vec, rho_vec in all_mean_trajectories:
                        try:
                            # Find first index where trajectory is within 1.5% of final value
                            conv_idx = np.where(np.abs(rho_vec - final_val) < 0.015)[0][
                                0
                            ]
                            convergence_time = min(convergence_time, t_vec[conv_idx])
                        except IndexError:
                            continue  # Trajectory never reached convergence

                    # Set a sensible upper limit with padding
                    xlim_final = max(20, convergence_time * 1.5)
                    ax.set_xlim(right=xlim_final)
                # --- END OF NEW LOGIC ---

                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), fontsize=10)
                ax.grid(True, linestyle=":", alpha=0.7)
                ax.set_ylim(-0.05, 1.05)
                if i == len(b_m_vals) - 1:
                    ax.set_xlabel("Time", fontsize=14)
                if j == 0:
                    ax.set_ylabel(
                        r"Mutant Fraction, $\langle\rho_M\rangle$", fontsize=14
                    )

        output_filename = os.path.join(
            figure_dir, f"relaxation_theory_comparison_phi_{phi:.2f}.png"
        )
        plt.savefig(output_filename, dpi=150, bbox_inches="tight")
        print(f"  -> Saved figure to {output_filename}")
        plt.close(fig)


if __name__ == "__main__":
    main()
