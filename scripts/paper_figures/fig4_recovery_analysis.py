# FILE: scripts/paper_figures/fig4_recovery_analysis.py (Definitive Robust Version)

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.optimize import curve_fit
from tqdm import tqdm


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# Define the exponential decay function for fitting
def exp_decay(t, tau, a, c):
    return a * np.exp(-t / tau) + c


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 4: Recovery Timescale Analysis."
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
    output_path = os.path.join(
        project_root,
        "data",
        args.campaign_id,
        "analysis",
        "figure4_recovery_analysis.png",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading data from: {os.path.basename(summary_path)}")
    try:
        df = pd.read_csv(summary_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame()

    # --- Definitive Failsafe: Check for data and the crucial 'timeseries' column ---
    if df.empty or "timeseries" not in df.columns:
        print(
            f"Warning: No data or no 'timeseries' column found for campaign '{args.campaign_id}'.",
            file=sys.stderr,
        )
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Figure 4: Missing or Invalid Data", ha="center", va="center")
        plt.savefig(output_path, dpi=300)
        print(f"\nGenerated placeholder figure at {output_path}")
        sys.exit(0)

    df["s"] = df["b_m"] - 1.0

    timescales = []
    s_slice_for_fit = min(df["s"].unique(), key=lambda x: abs(x - (-0.2)))
    df_slice = df[np.isclose(df["s"], s_slice_for_fit)].copy()

    example_curves = {}
    for params, group in tqdm(
        df_slice.groupby(["phi", "k_total"]), desc="Fitting recovery timescales"
    ):
        phi_val, k_val = params
        all_replicate_timeseries = []
        for _, row in group.iterrows():
            # Drop NaN rows in timeseries column before trying to load
            if pd.isna(row["timeseries"]):
                continue
            try:
                ts = pd.DataFrame(json.loads(row["timeseries"]))
                if not ts.empty:
                    all_replicate_timeseries.append(ts.set_index("time"))
            except (json.JSONDecodeError, TypeError):
                continue

        if not all_replicate_timeseries:
            continue

        avg_ts = (
            pd.concat(all_replicate_timeseries).groupby("time").mean().reset_index()
        )

        try:
            equilibrium_rho = (1 - phi_val) / 2.0
            p0 = [100, 1.0, equilibrium_rho]
            popt, _ = curve_fit(
                exp_decay, avg_ts["time"], avg_ts["mutant_fraction"], p0=p0, maxfev=5000
            )
            tau_recovery = popt[0]
            timescales.append(
                {"phi": phi_val, "k_total": k_val, "tau_recovery": tau_recovery}
            )

            k_examples_options = [
                df_slice["k_total"].min(),
                df_slice["k_total"].median(),
                df_slice["k_total"].max(),
            ]
            if any(np.isclose(k_val, k_ex) for k_ex in k_examples_options):
                example_curves[(phi_val, k_val)] = avg_ts
        except RuntimeError:
            timescales.append(
                {"phi": phi_val, "k_total": k_val, "tau_recovery": np.nan}
            )

    df_timescales = pd.DataFrame(timescales)

    # --- Plotting ---
    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        f"The Cost of Spatial Memory: Recovery Timescales (s ≈ {s_slice_for_fit:.2f})",
        fontsize=22,
        y=1.02,
    )

    # Panel (a): Example decay curves
    axA = axes[0]
    phi_example = 0.0
    k_examples = sorted(
        [k for (p, k), df in example_curves.items() if np.isclose(p, phi_example)]
    )

    if k_examples:
        palette = sns.color_palette("viridis", n_colors=len(k_examples))
        for i, k_val in enumerate(k_examples):
            if (phi_example, k_val) in example_curves:
                curve_data = example_curves[(phi_example, k_val)]
                axA.plot(
                    curve_data["time"],
                    curve_data["mutant_fraction"],
                    lw=3,
                    color=palette[i],
                    label=f"k={k_val:.3f}",
                )
        axA.axhline((1 - phi_example) / 2.0, ls=":", color="black", label="Equilibrium")
        axA.legend(title=r"$k_{total}$")
    else:
        axA.text(
            0.5,
            0.5,
            "No example curves found\n for phi=0.0",
            ha="center",
            va="center",
            transform=axA.transAxes,
        )

    axA.set_title("(a) Dynamics of Recovery", fontsize=16)
    axA.set_xlabel("Time, $t$ (post-quench)")
    axA.set_ylabel(r"Mean Mutant Fraction, $\langle\rho_M\rangle$")
    axA.set_ylim(bottom=-0.05, top=1.05)

    # Panel (b): Recovery timescale vs k_total
    axB = axes[1]
    if not df_timescales.empty:
        sns.lineplot(
            ax=axB,
            data=df_timescales,
            x="k_total",
            y="tau_recovery",
            hue="phi",
            palette="coolwarm_r",
            marker="o",
            lw=3,
            markersize=10,
        )
        axB.legend(title=r"Bias, $\phi$")
    else:
        axB.text(
            0.5,
            0.5,
            "No timescales to plot",
            ha="center",
            va="center",
            transform=axB.transAxes,
        )

    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_title("(b) The Cost of Memory", fontsize=16)
    axB.set_xlabel(r"Switching Rate, $k_{total}$")
    axB.set_ylabel(r"Recovery Timescale, $\tau_{recovery}$")

    sns.despine(fig)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    print(f"\nRecovery Figure 4 saved to {output_path}")


if __name__ == "__main__":
    main()
