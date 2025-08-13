# FILE: scripts/paper_figures/fig5_timescale_synthesis.py

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
from scipy.optimize import curve_fit
from scipy.signal import correlate

# --- Global Plotting Style ---
sns.set_theme(style="ticks", context="talk", font_scale=1.1)
FIG_DPI = 300


def get_project_root():
    """Find the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_data(campaign_id: str, project_root: str) -> (dict, pd.DataFrame):
    """Loads summary and all gzipped timeseries JSON files for a campaign."""
    ts_data_map = {}
    summary_path = os.path.join(
        project_root,
        "data",
        campaign_id,
        "analysis",
        f"{campaign_id}_summary_aggregated.csv",
    )
    try:
        summary_df = pd.read_csv(summary_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(
            f"Fatal: Summary data for '{campaign_id}' not found or empty. Please run and consolidate the campaign.",
            file=sys.stderr,
        )
        sys.exit(1)

    timeseries_dir = os.path.join(project_root, "data", campaign_id, "timeseries")
    if not os.path.isdir(timeseries_dir):
        print(
            f"Warning: Timeseries directory not found for {campaign_id}",
            file=sys.stderr,
        )
        return {}, summary_df

    for task_id in tqdm(
        summary_df["task_id"], desc=f"Loading timeseries for {campaign_id}"
    ):
        ts_path = os.path.join(timeseries_dir, f"ts_{task_id}.json.gz")
        if os.path.exists(ts_path):
            try:
                with gzip.open(ts_path, "rt", encoding="utf-8") as f:
                    ts_data_map[task_id] = pd.DataFrame(json.load(f))
            except (json.JSONDecodeError, gzip.BadGzipFile):
                continue
    return ts_data_map, summary_df


def analyze_relaxation(ts_data_map: dict, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Fits relaxation curves to extract tau_relax and rho_inf for each k."""

    def exp_decay(t, tau, rho_inf):
        # rho_M(t) = (rho_M(0) - rho_inf) * exp(-t / tau) + rho_inf
        # Since rho_M(0) = 1.0, this simplifies
        return (1.0 - rho_inf) * np.exp(-t / tau) + rho_inf

    timescales = []
    k_values = sorted(summary_df["k_total"].unique())

    for k in tqdm(k_values, desc="Analyzing relaxation timescales"):
        task_ids_for_k = summary_df[summary_df["k_total"] == k]["task_id"]
        replicates = [
            ts_data_map[tid]
            for tid in task_ids_for_k
            if tid in ts_data_map and not ts_data_map[tid].empty
        ]
        if not replicates:
            continue

        avg_df = (
            pd.concat(replicates)
            .groupby("time")["mutant_fraction"]
            .agg(["mean"])
            .reset_index()
        )

        try:
            p0 = [100.0, avg_df["mean"].iloc[-1]]
            bounds = ([0.1, 0], [5000, 1])
            params, _ = curve_fit(
                exp_decay,
                avg_df["time"],
                avg_df["mean"],
                p0=p0,
                bounds=bounds,
                maxfev=5000,
            )
            tau_relax, rho_inf = params
            timescales.append(
                {"k_total": k, "tau_relax": tau_relax, "rho_inf": rho_inf}
            )
        except RuntimeError:
            timescales.append({"k_total": k, "tau_relax": np.nan, "rho_inf": np.nan})

    return pd.DataFrame(timescales)


def analyze_tracking(ts_data_map: dict, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates tracking lag tau_lag for each k using cross-correlation."""
    timescales = []
    k_values = sorted(summary_df["k_total"].unique())
    cycle_length = 120
    sample_interval = summary_df["sample_interval"].iloc[0]

    for k in tqdm(k_values, desc="Analyzing tracking timescales"):
        task_ids_for_k = summary_df[summary_df["k_total"] == k]["task_id"]
        replicates = [
            ts_data_map[tid]
            for tid in task_ids_for_k
            if tid in ts_data_map and not ts_data_map[tid].empty
        ]
        if not replicates:
            continue

        avg_df = (
            pd.concat(replicates)
            .groupby("time")["mutant_fraction"]
            .agg(["mean"])
            .reset_index()
        )

        t_max = avg_df["time"].max()
        time_points = np.arange(0, t_max, sample_interval)
        # Ideal signal is 1 when M-favored (e.g., t=60-120), -1 when WT-favored (e.g., t=0-60)
        env_signal = -np.sign(np.sin(2 * np.pi * time_points / cycle_length))

        rho_signal = np.interp(time_points, avg_df["time"], avg_df["mean"])

        env_signal_norm = (env_signal - np.mean(env_signal)) / np.std(env_signal)
        rho_signal_norm = (rho_signal - np.mean(rho_signal)) / np.std(rho_signal)

        correlation = correlate(rho_signal_norm, env_signal_norm, mode="same")
        lag_index = np.argmax(correlation) - (len(time_points) // 2)
        tau_lag = lag_index * sample_interval

        timescales.append({"k_total": k, "tau_lag": abs(tau_lag)})

    return pd.DataFrame(timescales)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 5: The Dynamical Basis of Optimal Switching."
    )
    parser.add_argument(
        "relaxation_campaign",
        help="Campaign ID for relaxation analysis (e.g., relaxation_analysis)",
    )
    parser.add_argument(
        "tracking_campaign",
        help="Campaign ID for tracking analysis (e.g., tracking_analysis)",
    )
    parser.add_argument(
        "fitness_campaign",
        help="Campaign ID for fitness data (e.g., bet_hedging_final)",
    )
    args = parser.parse_args()
    project_root = get_project_root()

    # --- Data Loading and Analysis ---
    relax_ts_map, relax_summary = load_data(args.relaxation_campaign, project_root)
    track_ts_map, track_summary = load_data(args.tracking_campaign, project_root)
    fitness_df = pd.read_csv(
        os.path.join(
            project_root,
            "data",
            args.fitness_campaign,
            "analysis",
            f"{args.fitness_campaign}_summary_aggregated.csv",
        )
    )

    df_tau_relax = analyze_relaxation(relax_ts_map, relax_summary)
    df_tau_track = analyze_tracking(track_ts_map, track_summary)

    fitness_df["s"] = fitness_df["b_m"] - 1.0
    df_fitness_slice = fitness_df[
        np.isclose(fitness_df["s"], -0.5)
        & np.isclose(fitness_df["phi"], 0.0)
        & (fitness_df["patch_width"] == 60)
    ].copy()

    # --- Plotting Setup ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), constrained_layout=True)
    fig.suptitle(
        "Figure 5: The Optimal Switching Rate is the Most Dynamically Responsive (s=-0.5, φ=0)",
        fontsize=22,
        y=1.07,
    )
    axA, axB, axC = axes

    # --- Panel A: Relaxation Dynamics ---
    k_to_plot_A = [0.01, 0.2, 10.0]
    palette_A = sns.color_palette("viridis_r", n_colors=len(k_to_plot_A))
    axA.set_title("(A) Relaxation from a Maladapted State", fontsize=16)

    for i, k_val in enumerate(k_to_plot_A):
        k_actual = relax_summary.iloc[
            (relax_summary["k_total"] - k_val).abs().argsort()[:1]
        ]["k_total"].iloc[0]
        task_ids = relax_summary[relax_summary["k_total"] == k_actual]["task_id"]
        replicates = [
            relax_ts_map[tid]
            for tid in task_ids
            if tid in relax_ts_map and not relax_ts_map[tid].empty
        ]
        if not replicates:
            continue
        avg_df = (
            pd.concat(replicates)
            .groupby("time")
            .agg(mean=("mutant_fraction", "mean"), sem=("mutant_fraction", "sem"))
            .reset_index()
        )

        fit_results = df_tau_relax[np.isclose(df_tau_relax["k_total"], k_actual)]
        tau_val = fit_results["tau_relax"].iloc[0]
        rho_inf = fit_results["rho_inf"].iloc[0]
        label = f"k={k_actual:.2g}, τ_relax={tau_val:.1f}"
        if k_val > 1:
            label += f", ρ_∞={rho_inf:.2f}"

        axA.plot(
            avg_df["time"], avg_df["mean"], label=label, color=palette_A[i], lw=2.5
        )
        axA.fill_between(
            avg_df["time"],
            avg_df["mean"] - avg_df["sem"],
            avg_df["mean"] + avg_df["sem"],
            color=palette_A[i],
            alpha=0.2,
        )

    axA.set_xlabel("Time, t")
    axA.set_ylabel(r"Mutant Fraction, $\rho_M$")
    axA.legend(title="Switching Rate & Timescale")
    axA.set_ylim(-0.05, 1.05)
    axA.grid(True, ls=":")
    axA.set_xlim(0, 250)

    # --- Panel B: Environmental Tracking ---
    axB.set_title("(B) Environmental Tracking Dynamics", fontsize=16)
    k_to_plot_B = k_to_plot_A
    palette_B = sns.color_palette("plasma", n_colors=len(k_to_plot_B))

    axB.axvspan(0, 60, color="gray", alpha=0.1, label="WT-Favored")
    axB.axvspan(60, 120, color="white", alpha=0.1)
    axB.axvspan(120, 180, color="gray", alpha=0.1)
    axB.axvspan(180, 240, color="white", alpha=0.1)

    for i, k_val in enumerate(k_to_plot_B):
        k_actual = track_summary.iloc[
            (track_summary["k_total"] - k_val).abs().argsort()[:1]
        ]["k_total"].iloc[0]
        task_ids = track_summary[track_summary["k_total"] == k_actual]["task_id"]
        replicates = [
            track_ts_map[tid]
            for tid in task_ids
            if tid in track_ts_map and not track_ts_map[tid].empty
        ]
        if not replicates:
            continue
        avg_df = (
            pd.concat(replicates)
            .groupby("time")
            .agg(mean=("mutant_fraction", "mean"), sem=("mutant_fraction", "sem"))
            .reset_index()
        )

        axB.plot(
            avg_df["time"],
            avg_df["mean"],
            label=f"k={k_actual:.2g}",
            color=palette_B[i],
            lw=2.5,
        )
        axB.fill_between(
            avg_df["time"],
            avg_df["mean"] - avg_df["sem"],
            avg_df["mean"] + avg_df["sem"],
            color=palette_B[i],
            alpha=0.2,
        )

    axB.set_xlabel("Time within Cycle, t")
    axB.set_ylabel(r"Mutant Fraction, $\rho_M$")
    axB.legend(loc="upper right")
    axB.set_xlim(0, 240)
    axB.grid(True, ls=":")

    slow_k_line = axB.get_lines()[0]
    x_data, y_data = slow_k_line.get_data()
    y_mid = (y_data[x_data > 60].min() + y_data[x_data > 60].max()) / 2
    t_response = x_data[np.where((y_data > y_mid) & (x_data > 60))[0][0]]
    axB.annotate(
        "",
        xy=(60, 0.4),
        xytext=(t_response, 0.4),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
        va="center",
    )
    axB.text(
        (60 + t_response) / 2,
        0.45,
        r"$\tau_{lag}$",
        ha="center",
        va="bottom",
        fontsize=14,
        color=palette_B[0],
    )

    fast_k_line = axB.get_lines()[2]
    x_fast, y_fast = fast_k_line.get_data()
    axB.annotate(
        "High Phenotypic Load",
        xy=(x_fast[np.argmax(y_fast)], np.max(y_fast)),
        xytext=(150, 0.8),
        ha="center",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="black"),
        color=palette_B[2],
    )

    # --- Panel C: Synthesis ---
    df_timescales = pd.merge(df_tau_relax, df_tau_track, on="k_total").dropna()
    axC.set_title("(C) Optimal Rate Minimizes Timescales", fontsize=16)

    axC.plot(
        df_timescales["k_total"],
        df_timescales["tau_relax"],
        "o-",
        label=r"Relaxation Time, $\tau_{relax}$",
        color="royalblue",
    )
    axC.plot(
        df_timescales["k_total"],
        df_timescales["tau_lag"],
        "s--",
        label=r"Tracking Lag, $\tau_{lag}$",
        color="seagreen",
    )
    axC.set_xscale("log")
    axC.set_yscale("log")
    axC.set_xlabel(r"Switching Rate, k")
    axC.set_ylabel("Characteristic Timescale, τ", color="darkslategrey")
    axC.tick_params(axis="y", labelcolor="darkslategrey")
    axC.grid(True, which="both", ls=":")

    axC2 = axC.twinx()
    axC2.plot(
        df_fitness_slice["k_total"],
        df_fitness_slice["avg_front_speed"],
        "^-",
        color="black",
        alpha=0.6,
        label="Fitness",
    )
    axC2.set_ylabel("Long-Term Fitness (Front Speed)", color="black")
    axC2.tick_params(axis="y", labelcolor="black")

    k_opt = df_fitness_slice.loc[df_fitness_slice["avg_front_speed"].idxmax()][
        "k_total"
    ]
    axC.axvline(
        k_opt, color="red", ls=":", lw=2, label=f"$k_{{opt}} \\approx {k_opt:.2f}$"
    )

    lines, labels = axC.get_legend_handles_labels()
    lines2, labels2 = axC2.get_legend_handles_labels()
    axC2.legend(lines + lines2, labels + labels2, loc="best")

    output_path = os.path.join(project_root, "data", "figure5_timescale_synthesis.png")
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"\nFigure 5 saved to {output_path}")


if __name__ == "__main__":
    main()
