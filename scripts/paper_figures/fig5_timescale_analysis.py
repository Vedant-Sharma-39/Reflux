# FILE: scripts/paper_figures/fig5_timescale_analysis.py (Definitively Corrected Version)

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
sns.set_theme(style="ticks", context="talk", font_scale=1.2)
FIG_DPI = 300


def get_project_root():
    """Dynamically find the project root and add it to the path."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


PROJECT_ROOT = get_project_root()
from src.config import EXPERIMENTS


def load_timeseries_data(campaign_id: str, project_root: str) -> (dict, pd.DataFrame):
    """Loads all gzipped timeseries JSON files and the summary for a campaign."""
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
            f"Fatal: Summary data for '{campaign_id}' not found. Run `make consolidate`.",
            file=sys.stderr,
        )
        sys.exit(1)
    timeseries_dir = os.path.join(project_root, "data", campaign_id, "timeseries")
    if not os.path.isdir(timeseries_dir):
        return {}, summary_df
    for task_id in tqdm(
        summary_df["task_id"], desc=f"Loading timeseries for {campaign_id}"
    ):
        ts_path = os.path.join(timeseries_dir, f"ts_{task_id}.json.gz")
        if os.path.exists(ts_path):
            with gzip.open(ts_path, "rt", encoding="utf-8") as f:
                ts_data_map[task_id] = pd.DataFrame(json.load(f))
    return ts_data_map, summary_df


def analyze_relaxation(ts_data_map: dict, summary_df: pd.DataFrame) -> pd.DataFrame:
    def exp_decay(t, tau, rho_inf):
        return (1.0 - rho_inf) * np.exp(-t / tau) + rho_inf

    timescales = []
    for k in tqdm(sorted(summary_df["k_total"].unique()), desc="Analyzing relaxation"):
        replicates = [
            ts_data_map.get(tid)
            for tid in summary_df[summary_df["k_total"] == k]["task_id"]
            if ts_data_map.get(tid) is not None and not ts_data_map[tid].empty
        ]
        if not replicates:
            continue
        avg_df = (
            pd.concat(replicates)
            .groupby("time")["mutant_fraction"]
            .mean()
            .reset_index()
        )
        try:
            params, _ = curve_fit(
                exp_decay,
                avg_df["time"],
                avg_df["mutant_fraction"],
                p0=[100.0, avg_df["mutant_fraction"].iloc[-1]],
                bounds = ([0.1, 0], [10000, 1]),
                maxfev=8000,
            )
            timescales.append(
                {"k_total": k, "tau_relax": params[0], "rho_inf": params[1]}
            )
        except (RuntimeError, IndexError):
            timescales.append({"k_total": k, "tau_relax": np.nan, "rho_inf": np.nan})
    return pd.DataFrame(timescales)


def analyze_tracking(ts_data_map: dict, summary_df: pd.DataFrame) -> pd.DataFrame:
    timescales = []
    cycle_length = 120
    for k in tqdm(sorted(summary_df["k_total"].unique()), desc="Analyzing tracking"):
        replicates = [
            ts_data_map.get(tid)
            for tid in summary_df[summary_df["k_total"] == k]["task_id"]
            if ts_data_map.get(tid) is not None and not ts_data_map[tid].empty
        ]
        if not replicates:
            continue
        avg_df = (
            pd.concat(replicates)
            .groupby("time")["mutant_fraction"]
            .mean()
            .reset_index()
        )
        sample_interval = summary_df["sample_interval"].iloc[0]
        time_points = np.arange(0, avg_df["time"].max(), sample_interval)
        env_signal = -np.sign(np.sin(2 * np.pi * time_points / cycle_length))
        rho_signal = np.interp(time_points, avg_df["time"], avg_df["mutant_fraction"])
        env_norm = env_signal - np.mean(env_signal)
        rho_norm = rho_signal - np.mean(rho_signal)
        if np.std(rho_norm) < 1e-6 or np.std(env_norm) < 1e-6:
            continue
        correlation = correlate(rho_norm, env_norm, mode="full")
        lag_index = np.argmax(correlation) - len(rho_norm) + 1
        timescales.append({"k_total": k, "tau_lag": abs(lag_index * sample_interval)})
    return pd.DataFrame(timescales)


def find_nearest_k(df, k_val):
    return df.iloc[(df["k_total"] - k_val).abs().argsort()[:1]]["k_total"].iloc[0]


def main():
    relaxation_campaign_id = EXPERIMENTS["relaxation_analysis"]["campaign_id"]
    tracking_campaign_id = EXPERIMENTS["tracking_analysis"]["campaign_id"]

    relax_ts_map, relax_summary = load_timeseries_data(
        relaxation_campaign_id, PROJECT_ROOT
    )
    track_ts_map, track_summary = load_timeseries_data(
        tracking_campaign_id, PROJECT_ROOT
    )

    df_tau_relax = analyze_relaxation(relax_ts_map, relax_summary)
    df_tau_track = analyze_tracking(track_ts_map, track_summary)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), constrained_layout=True)
    fig.suptitle(
        "Figure 5: The Dynamical Trade-Offs of Phenotypic Switching",
        fontsize=24,
        y=1.07,
    )

    # --- METHODOLOGY: Select k-values present in BOTH datasets ---
    common_k = pd.merge(df_tau_relax, df_tau_track, on="k_total")["k_total"]
    k_slow_target, k_fast_target = 0.0068, 6.81
    k_slow = find_nearest_k(pd.DataFrame({"k_total": common_k}), k_slow_target)
    k_fast = find_nearest_k(pd.DataFrame({"k_total": common_k}), k_fast_target)
    k_to_plot = [k_slow, k_fast]
    palette = ["#2ca02c", "#d62728"]  # Green and Red

    # --- Panel A: Relaxation ---
    axA = axes[0]
    axA.set_title("(A) Relaxation from a Maladapted State", fontsize=18)
    for i, k_val in enumerate(k_to_plot):
        task_ids = relax_summary[np.isclose(relax_summary["k_total"], k_val)]["task_id"]
        replicates = [
            relax_ts_map.get(tid)
            for tid in task_ids
            if relax_ts_map.get(tid) is not None
        ]
        if not replicates:
            continue
        avg_df = (
            pd.concat(replicates)
            .groupby("time")
            .agg(mean=("mutant_fraction", "mean"), sem=("mutant_fraction", "sem"))
            .reset_index()
        )
        fit_results = df_tau_relax[np.isclose(df_tau_relax["k_total"], k_val)]
        label = f"k={k_val:.3g}, τ_relax={fit_results['tau_relax'].iloc[0]:.1f}"
        axA.plot(avg_df["time"], avg_df["mean"], label=label, color=palette[i], lw=3.5)
        axA.fill_between(
            avg_df["time"],
            avg_df["mean"] - avg_df["sem"],
            avg_df["mean"] + avg_df["sem"],
            color=palette[i],
            alpha=0.15,
        )
    axA.set(
        xlabel="Time, t",
        ylabel=r"Mutant Fraction, $\rho_M$",
        ylim=(-0.05, 1.05),
        xlim=(0, 250),
    )
    axA.legend(fontsize=14)
    axA.grid(True, ls=":")

    # --- Panel B: Tracking ---
    axB = axes[1]
    axB.set_title("(B) Environmental Tracking", fontsize=18)
    axB.axvspan(60, 120, color="gray", alpha=0.1, label="M-Favored")
    axB.axvspan(0, 60, color="white", alpha=0.1)
    axB.axvspan(120, 180, color="white", alpha=0.1)
    axB.axvspan(180, 240, color="gray", alpha=0.1)
    for i, k_val in enumerate(k_to_plot):
        task_ids = track_summary[np.isclose(track_summary["k_total"], k_val)]["task_id"]
        replicates = [
            track_ts_map.get(tid)
            for tid in task_ids
            if track_ts_map.get(tid) is not None
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
            label=f"k={k_val:.3g}",
            color=palette[i],
            lw=3.5,
        )
        axB.fill_between(
            avg_df["time"],
            avg_df["mean"] - avg_df["sem"],
            avg_df["mean"] + avg_df["sem"],
            color=palette[i],
            alpha=0.15,
        )
    # FIX: Correct Y-axis limits to show the full oscillation
    axB.set(
        xlabel="Time within Cycle, t",
        ylabel=r"Mutant Fraction, $\rho_M$",
        xlim=(0, 240),
        ylim=(-0.05, 1.05),
    )
    axB.legend(fontsize=14)
    axB.grid(True, ls=":")

    # --- Panel C: Synthesis ---
    axC = axes[2]
    df_timescales = pd.merge(df_tau_relax, df_tau_track, on="k_total").dropna()
    axC.set_title("(C) The Dynamic Trade-Off", fontsize=18)
    axC.plot(
        df_timescales["k_total"],
        df_timescales["tau_relax"],
        "o-",
        label=r"Relaxation Time, $\tau_{relax}$",
        color="#1f77b4",
        ms=10,
        lw=3.5,
    )
    axC.plot(
        df_timescales["k_total"],
        df_timescales["tau_lag"],
        "s--",
        label=r"Tracking Lag, $\tau_{lag}$",
        color="#2ca02c",
        ms=10,
        lw=3.5,
        zorder=10,
    )

    # Find the intersection point to mark the optimal k
    k_log = np.log10(df_timescales["k_total"])
    tau_relax_log = np.log10(df_timescales["tau_relax"])
    tau_lag_log = np.log10(df_timescales["tau_lag"])
    # Interpolate to find where the difference is zero
    diff_log = tau_relax_log - tau_lag_log
    try:
        k_opt_log = np.interp(0, diff_log, k_log)
        k_opt = 10**k_opt_log
        axC.axvline(
            k_opt,
            color="red",
            ls=":",
            lw=3.5,
            label=f"Dynamically Optimal k ≈ {k_opt:.2f}",
        )
    except:
        pass  # Don't draw line if curves don't cross

    axC.set(
        xscale="log",
        yscale="log",
        xlabel=r"Switching Rate, k",
        ylabel="Characteristic Timescale, τ",
    )
    axC.tick_params(axis="y", labelcolor="darkslategrey")
    axC.grid(True, which="both", ls=":")

    # FIX: Clean legend, removing redundant elements
    handles, labels = axC.get_legend_handles_labels()
    axC.legend(handles, labels, loc="best", frameon=True, fontsize=14)

    output_path = os.path.join(
        PROJECT_ROOT, "figures", "fig5_timescale_synthesis_final.png"
    )
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"\nFinal Figure 5 saved to {output_path}")


if __name__ == "__main__":
    main()
