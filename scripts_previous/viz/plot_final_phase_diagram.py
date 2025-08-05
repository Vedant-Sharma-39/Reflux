# FILE: scripts/viz/plot_final_phase_diagram.py
# Loads the final, processed critical point data and generates the two
# primary phase diagram figures for the publication.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- Robust Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "src"))
from config import EXPERIMENTS

# --- Configuration ---
CAMPAIGN_ID = EXPERIMENTS["criticality_mapping_v1"]["CAMPAIGN_ID"]
DATA_FILE = os.path.join(
    project_root, "data", CAMPAIGN_ID, "analysis", "critical_points_summary.csv"
)
FIGURES_DIR = os.path.join(project_root, "figures", CAMPAIGN_ID)

# Use the same publication-quality style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial"],
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 22,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
)


def main():
    if not os.path.exists(DATA_FILE):
        print(f"FATAL: Critical points summary file not found at {DATA_FILE}")
        print(
            "Please run 'scripts/analyze_correlation_results.py criticality_mapping_v1' first."
        )
        return

    df = pd.read_csv(DATA_FILE)

    # --- PLOT 1: k_c vs. Selection (s) ---
    df_s_scan = df[df["phi"] == 0.0].sort_values("s")

    fig, ax1 = plt.subplots(figsize=(10, 7))
    color = "tab:red"
    ax1.set_xlabel("Selection Coefficient ($s = b_m - 1$)")
    ax1.set_ylabel("Critical Switching Rate ($k_c$)", color=color)
    ax1.plot(
        df_s_scan["s"], df_s_scan["k_c"], "o-", color=color, markersize=10, linewidth=3
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Critical Exponent ($\\eta$)", color=color)
    ax2.plot(
        df_s_scan["s"],
        df_s_scan["eta_c"],
        "s--",
        color=color,
        markersize=10,
        linewidth=3,
    )
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(bottom=0)

    plt.title("Phase Boundary vs. Selection (at $\\phi=0$)")
    fig.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "FINAL_Phase_Diagram_vs_Selection.png"), dpi=300
    )
    plt.close(fig)
    print(f"Saved phase diagram vs. selection to {FIGURES_DIR}")

    # --- PLOT 2: k_c vs. Bias (phi) ---
    df_phi_scan = df[df["b_m"] == 0.8].sort_values("phi")

    fig, ax1 = plt.subplots(figsize=(10, 7))
    color = "tab:red"
    ax1.set_xlabel("Switching Bias ($\\phi$)")
    ax1.set_ylabel("Critical Switching Rate ($k_c$)", color=color)
    ax1.plot(
        df_phi_scan["phi"],
        df_phi_scan["k_c"],
        "o-",
        color=color,
        markersize=10,
        linewidth=3,
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Critical Exponent ($\\eta$)", color=color)
    ax2.plot(
        df_phi_scan["phi"],
        df_phi_scan["eta_c"],
        "s--",
        color=color,
        markersize=10,
        linewidth=3,
    )
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(bottom=0)

    # Highlight the different universality class for the absorbing case (phi=-1)
    eta_absorbing = df_phi_scan[df_phi_scan["phi"] == -1.0]["eta_c"].iloc[0]
    ax2.text(
        -0.95,
        eta_absorbing + 0.05,
        f"Absorbing State\n$\\eta \\approx {eta_absorbing:.2f}$",
        fontsize=14,
        color=color,
        ha="left",
    )

    plt.title("Phase Boundary vs. Bias (at $s=-0.2$)")
    fig.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "FINAL_Phase_Diagram_vs_Bias.png"), dpi=300)
    plt.close(fig)
    print(f"Saved phase diagram vs. bias to {FIGURES_DIR}")


if __name__ == "__main__":
    main()