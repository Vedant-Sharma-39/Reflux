# FILE: scripts/analyze_boundary_dynamics.py
#
# [FINAL VERSION] Implements a dual-fitting strategy for diffusion to
# robustly measure both the effective exponent (alpha_eff) and a constrained
# diffusion coefficient (d_eff), providing a more complete physical picture.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, json, argparse, ast
from tqdm import tqdm
from scipy.stats import linregress
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count

# --- Config and Setup ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from config import EXPERIMENTS
except (NameError, ImportError) as e:
    sys.exit(f"FATAL: Could not import configuration. Error: {e}")

SURVIVAL_THRESHOLD, FIT_RANGE_FRAC = 1.0, 0.75
plt.style.use("seaborn-v0_8-whitegrid")


def read_json_worker(filepath):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            data["source_file"] = os.path.basename(filepath)
            return data
    except Exception:
        return None


def aggregate_data_cached(campaign_id, analysis_dir, force_reaggregate=False):
    # This helper function is correct and unchanged.
    results_dir, cached_csv_path = os.path.join(
        project_root, "data", campaign_id, "results"
    ), os.path.join(analysis_dir, f"{campaign_id}_aggregated.csv")
    all_json_files = (
        {f for f in os.listdir(results_dir) if f.endswith(".json")}
        if os.path.isdir(results_dir)
        else set()
    )
    if not force_reaggregate and os.path.exists(cached_csv_path):
        df = pd.read_csv(cached_csv_path, low_memory=False)
        files_to_process = all_json_files - set(df.get("source_file", []))
        if not files_to_process:
            return df
    else:
        files_to_process, df = all_json_files, None
    filepaths = [os.path.join(results_dir, f) for f in files_to_process]
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        new_results = list(
            tqdm(
                pool.imap_unordered(read_json_worker, filepaths),
                total=len(filepaths),
                desc="Reading new JSONs",
            )
        )
    new_df = pd.DataFrame([r for r in new_results if r is not None])
    full_df = pd.concat([df, new_df], ignore_index=True) if df is not None else new_df
    if "trajectory" in full_df.columns:
        full_df["trajectory"] = full_df["trajectory"].astype(str)
    full_df.to_csv(cached_csv_path, index=False)
    return full_df


def constrained_linear_model(q, D):
    """A linear model Var = D*q, which goes through the origin."""
    return D * q


def analyze_trajectories_for_s(s_val, group_df):
    # ... (Data loading and preparation part is unchanged) ...
    all_points, num_initial_replicates = [], group_df["replicate_id"].nunique()
    if num_initial_replicates == 0:
        return None
    for _, row in group_df.iterrows():
        traj = row.get("trajectory")
        if pd.isna(traj):
            continue
        if isinstance(traj, str):
            try:
                traj = ast.literal_eval(traj)
            except (ValueError, SyntaxError):
                continue
        if not isinstance(traj, list):
            continue
        for q, w in traj:
            all_points.append(
                {"replicate_id": row.get("replicate_id", -1), "q": q, "width": w}
            )
    if not all_points:
        return None
    df = pd.DataFrame(all_points)
    binned_stats = (
        df.groupby(pd.cut(df["q"], 50), observed=True)
        .agg(
            q_mean=("q", "mean"),
            width_mean=("width", "mean"),
            width_var=("width", "var"),
            survival_counts=("replicate_id", "nunique"),
        )
        .dropna()
    )
    binned_stats["survival_prob"] = (
        binned_stats["survival_counts"] / num_initial_replicates
    )
    reliable_data = binned_stats[
        binned_stats["survival_prob"] >= SURVIVAL_THRESHOLD
    ].copy()
    if len(reliable_data) < 5:
        return None

    # --- Drift Fit ---
    slope_v, intercept_v, _, _, _ = linregress(
        reliable_data["q_mean"], reliable_data["width_mean"]
    )
    v_drift = slope_v / 2.0

    # --- Diffusion Analysis (Dual Fit) ---
    fit_end_index = int(len(reliable_data) * FIT_RANGE_FRAC)
    if fit_end_index < 3:
        fit_end_index = len(reliable_data)
    diffusion_fit_data = reliable_data.iloc[:fit_end_index]
    diffusion_fit_data = diffusion_fit_data[diffusion_fit_data["width_var"] > 0]
    if len(diffusion_fit_data) < 3:
        return None

    # 1. Power-Law Fit (Log-space)
    log_q = np.log10(diffusion_fit_data["q_mean"])
    log_var = np.log10(diffusion_fit_data["width_var"])
    slope_d_log, intercept_d_log, _, _, _ = linregress(log_q, log_var)
    alpha_eff = slope_d_log

    # 2. Constrained Diffusive Fit (Linear-space, forced through origin)
    d_eff_constrained = np.nan
    try:
        popt, _ = curve_fit(
            constrained_linear_model,
            diffusion_fit_data["q_mean"],
            diffusion_fit_data["width_var"],
            p0=[0.1],
        )
        d_eff_constrained = popt[0]
    except RuntimeError:
        pass  # Fit failed, leave as NaN

    return {
        "s": s_val,
        "v_drift": v_drift,
        "d_eff_constrained": d_eff_constrained,
        "alpha_eff": alpha_eff,
        "plot_data": {
            "binned_stats": binned_stats.to_dict("list"),
            "reliable_data": reliable_data.to_dict("list"),
            "diffusion_fit_data": diffusion_fit_data.to_dict("list"),
            "slope_v": slope_v,
            "intercept_v": intercept_v,
            "slope_d_log": slope_d_log,
            "intercept_d_log": intercept_d_log,
            "d_eff_constrained": d_eff_constrained,
        },
    }


def main():
    # ... (Argument parsing and data loading are unchanged) ...
    parser = argparse.ArgumentParser(
        description="Analyze domain boundary dynamics with dual diffusion fitting."
    )
    parser.add_argument(
        "experiment_name", default="boundary_dynamics_vs_selection", nargs="?"
    )
    args = parser.parse_args()
    config, CAMPAIGN_ID = (
        EXPERIMENTS[args.experiment_name],
        EXPERIMENTS[args.experiment_name]["CAMPAIGN_ID"],
    )
    ANALYSIS_DIR, FIGS_DIR, DEBUG_FIGS_DIR = (
        os.path.join(project_root, "data", CAMPAIGN_ID, "analysis"),
        os.path.join(project_root, "figures", CAMPAIGN_ID),
        os.path.join(project_root, "figures", CAMPAIGN_ID, "debug_fits"),
    )
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(DEBUG_FIGS_DIR, exist_ok=True)
    df_raw = aggregate_data_cached(CAMPAIGN_ID, ANALYSIS_DIR)
    if df_raw is None or df_raw.empty:
        sys.exit("FATAL: No data found.")
    df_raw["s"] = df_raw["b_m"] - 1.0
    analysis_tasks = [(s, g) for s, g in df_raw.groupby("s")]
    with Pool(processes=max(1, cpu_count() - 2)) as pool:
        results = list(
            tqdm(
                pool.starmap(analyze_trajectories_for_s, analysis_tasks),
                total=len(analysis_tasks),
            )
        )
    valid_results = [res for res in results if res is not None]
    if not valid_results:
        sys.exit("Analysis complete, but no valid results were generated.")

    print("\n--- Generating debug plots ---")
    for res in tqdm(valid_results, desc="Plotting fits"):
        s, p_data = res["s"], res["plot_data"]
        binned_df, reliable_df, diffusion_fit_df = (
            pd.DataFrame(p_data["binned_stats"]),
            pd.DataFrame(p_data["reliable_data"]),
            pd.DataFrame(p_data["diffusion_fit_data"]),
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f"Boundary Dynamics Fit for s = {s:.3f}", fontsize=16)

        ax1.plot(
            binned_df["q_mean"],
            binned_df["width_mean"],
            ".",
            ms=6,
            alpha=0.4,
            label="All Data",
        )
        ax1.plot(
            reliable_df["q_mean"],
            reliable_df["width_mean"],
            ".",
            ms=8,
            color="orangered",
            label=f"Reliable Data",
        )
        q_plot_lin = np.array(binned_df["q_mean"])
        ax1.plot(
            q_plot_lin,
            p_data["intercept_v"] + p_data["slope_v"] * q_plot_lin,
            "r-",
            lw=2.5,
            label=f"Fit (v_drift={res['v_drift']:.3f})",
        )
        ax1.set(
            xlabel="Front Position (q)",
            ylabel="⟨Sector Width⟩",
            title="Drift Velocity Fit",
        )
        ax1.legend()
        ax1.grid(True, ls="--")

        # [THE FIX] Updated diffusion plot
        ax2.loglog(
            binned_df["q_mean"],
            binned_df["width_var"],
            ".",
            ms=6,
            alpha=0.4,
            color="seagreen",
            label="All Data",
        )
        ax2.loglog(
            diffusion_fit_df["q_mean"],
            diffusion_fit_df["width_var"],
            ".",
            ms=8,
            color="purple",
            label=f"Data for Fit (Initial {FIT_RANGE_FRAC*100:.0f}%)",
        )
        q_plot_log = np.array(diffusion_fit_df["q_mean"])
        fit_line_powerlaw = 10 ** (p_data["intercept_d_log"]) * q_plot_log ** (
            p_data["slope_d_log"]
        )
        ax2.loglog(
            q_plot_log,
            fit_line_powerlaw,
            "m-",
            lw=2.5,
            label=f"Power-Law Fit (α_eff={res['alpha_eff']:.2f})",
        )
        if not np.isnan(p_data["d_eff_constrained"]):
            ax2.loglog(
                q_plot_log,
                p_data["d_eff_constrained"] * q_plot_log,
                "k--",
                lw=2,
                label=f"Constrained Fit (D_eff={res['d_eff_constrained']:.3f})",
            )
        ax2.set(
            xlabel="Front Position (q)",
            ylabel="Var(Sector Width)",
            title="Effective Diffusion Fit (Log-Log Scale)",
        )
        ax2.legend()
        ax2.grid(True, which="both", ls="--")

        plt.savefig(
            os.path.join(DEBUG_FIGS_DIR, f"debug_fit_s_{s:.4f}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    # --- Final Summary Plotting ---
    df_final = pd.DataFrame(valid_results)[
        ["s", "v_drift", "d_eff_constrained", "alpha_eff"]
    ].sort_values("s")
    df_final.to_csv(
        os.path.join(ANALYSIS_DIR, "boundary_dynamics_summary.csv"), index=False
    )

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle("Boundary Dynamics vs. Selection", fontsize=20)
    axes[0].plot(df_final["s"], df_final["v_drift"], "o-", c="navy", label="Measured")
    axes[0].plot(
        df_final["s"], df_final["s"], "--", c="gray", alpha=0.8, label="Theory (v=s)"
    )
    axes[0].set(title="A. Drift Velocity", xlabel="Selection (s)", ylabel="$v_{drift}$")
    axes[0].legend()
    axes[1].plot(df_final["s"], df_final["d_eff_constrained"], "o-", c="darkgreen")
    axes[1].set(
        title="B. Diffusion Coefficient", xlabel="Selection (s)", ylabel="$D_{eff}$"
    )
    axes[1].set_yscale("log")
    axes[2].plot(df_final["s"], df_final["alpha_eff"], "o-", c="purple")
    axes[2].axhline(1.0, ls="--", c="gray", label="Pure Diffusion (α=1)")
    axes[2].set(
        title="C. Diffusion Exponent", xlabel="Selection (s)", ylabel="$\\alpha_{eff}$"
    )
    axes[2].legend()
    axes[2].set_ylim(bottom=0.5, top=1.5)
    for ax in axes:
        ax.grid(True, ls="--")
        ax.axvline(0, color="k", lw=0.5)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(FIGS_DIR, "Fig_Boundary_Dynamics_Summary.png"), dpi=300)


if __name__ == "__main__":
    main()
