# FILE: scripts/analyze_speed_hotspot.py
#
# A definitive script to analyze the "hotspot" of emergent front speed.
# It focuses on visualizing and quantifying the region in parameter space
# where the front propagates significantly faster than predicted by simple
# linear mixing models, revealing the optimal dynamic state.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, argparse

# --- Robust Path Setup & Data Aggregation ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(project_root, "src"))
    from config import EXPERIMENTS
    from data_utils import aggregate_data_cached
except (NameError, ImportError) as e:
    sys.exit(f"FATAL: Could not import configuration or helpers. Error: {e}")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "figure.titlesize": 24,
    }
)


def main():
    parser = argparse.ArgumentParser(description="Analyze the emergent speed hotspot.")
    parser.add_argument(
        "experiment_name", default="exp1_front_speed_deleterious_scan", nargs="?"
    )
    args = parser.parse_args()

    # --- Load and Process Data ---
    config = EXPERIMENTS[args.experiment_name]
    CAMPAIGN_ID = config["CAMPAIGN_ID"]
    ANALYSIS_DIR = os.path.join(project_root, "data", CAMPAIGN_ID, "analysis")
    FIGS_DIR = os.path.join(project_root, "figures", CAMPAIGN_ID, "hotspot_analysis")
    os.makedirs(FIGS_DIR, exist_ok=True)

    df_raw = aggregate_data_cached(CAMPAIGN_ID, project_root)
    if df_raw is None or df_raw.empty:
        sys.exit("FATAL: No data found.")

    df_raw["s"] = df_raw["b_m"] - 1.0
    df_stats = (
        df_raw.groupby(["s", "phi", "k_total"])
        .agg(
            avg_front_speed=("avg_front_speed", "mean"), avg_rho_M=("avg_rho_M", "mean")
        )
        .reset_index()
        .dropna()
    )

    v_wt = df_stats[np.isclose(df_stats["s"], 0)]["avg_front_speed"].mean()
    df_stats["v_deviation"] = df_stats["avg_front_speed"] - (
        v_wt * (1 + df_stats["s"] * df_stats["avg_rho_M"])
    )

    print(f"--- Analyzing Emergent Speed Hotspot for {CAMPAIGN_ID} ---")

    # ==========================================================================
    # Create the 2x2 Definitive Figure
    # ==========================================================================
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    fig.suptitle("Dissecting the Emergent Speed Hotspot", y=0.97)

    # --- Panel A: Heatmap for phi = -0.5 (Polluting Strategy) ---
    df_neg_phi = df_stats[df_stats["phi"] == -0.5]
    if not df_neg_phi.empty:
        pivot_neg = df_neg_phi.pivot_table(
            index="s", columns="k_total", values="v_deviation"
        )
        sns.heatmap(
            pivot_neg,
            ax=ax1,
            cmap="coolwarm",
            center=0,
            cbar_kws={"label": "Deviation from Naive Speed"},
        )
        ax1.set_title("A. Emergent Speed Landscape (Polluting, $\\phi=-0.5$)")
        ax1.set_xlabel("$k_{total}$")
        ax1.set_ylabel("s")

    # --- Panel B: Heatmap for phi = 0.5 (Purging Strategy) ---
    df_pos_phi = df_stats[df_stats["phi"] == 0.5]
    if not df_pos_phi.empty:
        pivot_pos = df_pos_phi.pivot_table(
            index="s", columns="k_total", values="v_deviation"
        )
        sns.heatmap(
            pivot_pos,
            ax=ax2,
            cmap="coolwarm",
            center=0,
            cbar_kws={"label": "Deviation from Naive Speed"},
        )
        ax2.set_title("B. Emergent Speed Landscape (Purging, $\\phi=0.5$)")
        ax2.set_xlabel("$k_{total}$")
        ax2.set_ylabel("s")

    # --- Panel C: Horizontal Slice through the Hotspot ---
    if not df_neg_phi.empty:
        s_values = df_neg_phi["s"].unique()
        s_slices = np.quantile(s_values, [0.0, 0.25, 0.5])  # s=-0.8, -0.6, -0.4
        s_to_plot = [s_values[np.argmin(np.abs(s_values - s_q))] for s_q in s_slices]
        df_sliced_s = df_neg_phi[df_neg_phi["s"].isin(s_to_plot)]

        palette_c = sns.color_palette("magma_r", n_colors=len(s_to_plot))

        sns.lineplot(
            data=df_sliced_s,
            x="k_total",
            y="v_deviation",
            hue="s",
            palette=palette_c,
            marker="o",
            lw=2.5,
            ms=8,
            ax=ax3,
            hue_order=s_to_plot,
        )
        ax3.set_xscale("log")
        ax3.axhline(0, color="k", ls="--")

        for s_val, color in zip(s_to_plot, palette_c):
            curve_data = df_sliced_s[df_sliced_s["s"] == s_val]
            if not curve_data.empty:
                peak_idx = curve_data["v_deviation"].idxmax()
                peak_k = curve_data.loc[peak_idx, "k_total"]
                peak_v = curve_data.loc[peak_idx, "v_deviation"]
                ax3.plot(
                    peak_k, peak_v, "*", ms=18, color=color, markeredgecolor="black"
                )

        ax3.set_title("C. Effect of Mixing Rate at Fixed Selection (for $\\phi=-0.5$)")
        ax3.set_xlabel("Total Switching Rate ($k_{total}$)")
        ax3.set_ylabel("Deviation from Naive Speed")

    # --- Panel D: Vertical Slice through the Hotspot ---
    if not df_neg_phi.empty:
        k_values = df_neg_phi["k_total"].unique()
        # Pick k_total values in and around the hotspot (e.g., 0.1 to 1.0)
        k_to_plot = [k for k in k_values if 0.1 <= k <= 1.0]
        if not k_to_plot and len(k_values) > 0:  # Fallback if data is sparse
            k_slices = np.quantile(k_values, np.linspace(0.4, 0.8, 5))
            k_to_plot = [
                k_values[np.argmin(np.abs(k_values - k_q))] for k_q in k_slices
            ]

        df_sliced_k = df_neg_phi[df_neg_phi["k_total"].isin(k_to_plot)]

        palette_d = sns.color_palette("viridis", n_colors=len(k_to_plot))

        sns.lineplot(
            data=df_sliced_k,
            x="s",
            y="v_deviation",
            hue="k_total",
            palette=palette_d,
            marker="o",
            lw=2.5,
            ms=8,
            ax=ax4,
            hue_order=sorted(k_to_plot),
        )
        ax4.axhline(0, color="k", ls="--")
        ax4.set_title("D. Effect of Selection at Fixed Mixing Rate (for $\\phi=-0.5$)")
        ax4.set_xlabel("Selection Coefficient (s)")
        ax4.set_ylabel("Deviation from Naive Speed")

    # --- Save Figure ---
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # --- [THE FIX] ---
    # Define the output_path variable before the final print statement
    plot_filename = "Fig_Definitive_Hotspot_Analysis.png"
    output_path = os.path.join(FIGS_DIR, plot_filename)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"\nDefinitive analysis complete. Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
