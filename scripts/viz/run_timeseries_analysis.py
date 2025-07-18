# scripts/run_timeseries_analysis.py
# Corrected version with a simpler, more robust path setup.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- Robust Path Setup (Corrected) ---
# This approach assumes you are running the script from the project's
# root directory (e.g., '.../Reflux/'). It directly adds the 'src'
# sub-directory to the Python path. This is simple and effective.
sys.path.insert(0, "src")

# Now, these imports will work correctly.
from linear_model import GillespieSimulation, Wildtype, Mutant
from metrics import MetricsManager, FrontDynamicsTracker
from hex_utils import HexPlotter

# ==============================================================================
# 1. DEFINE PARAMETERS FOR THE ANALYSIS
# ==============================================================================
# ... (The rest of the script is IDENTICAL to the previous version) ...

# Choose one or two interesting parameter sets from your phase diagram
PARAMS_TO_TEST = [
    {"k_total": 1.0, "phi": -0.5, "b_m": 0.8},  # Expected rho_M ~ 0.72
    {"k_total": 1.0, "phi": 0.0, "b_m": 0.8},  # Expected rho_M ~ 0.46
]

# Common simulation settings
SIM_CONFIG = {
    "width": 128,
    "length": 50000,
    "total_run_time": 2000.0,
    "num_replicates": 5,
    "log_interval": 5.0,
}

# Visualization settings
FIGURES_DIR = os.path.join("figures", "timeseries_analysis")  # Use relative path
os.makedirs(FIGURES_DIR, exist_ok=True)
COLOR_MAP = {Wildtype: "#3A86FF", Mutant: "#FFBE0B"}
LABELS = {Wildtype: "Wild-Type", Mutant: "Mutant"}


# ==============================================================================
# 2. MAIN EXECUTION FUNCTION
# ==============================================================================
def main():
    sns.set_theme(style="whitegrid", context="talk")

    for i, params in enumerate(PARAMS_TO_TEST):
        print(
            f"\n--- Running Analysis for Parameter Set {i+1}/{len(PARAMS_TO_TEST)} ---"
        )
        print(params)

        all_replicates_data = []
        final_sim_state = None

        for rep in range(SIM_CONFIG["num_replicates"]):
            print(f"  Running replicate {rep+1}/{SIM_CONFIG['num_replicates']}...")

            sim_params = {**SIM_CONFIG, **params}
            # Remove keys that are not part of GillespieSimulation's __init__
            sim_params.pop("total_run_time")
            sim_params.pop("num_replicates")
            sim_params.pop("log_interval")

            sim = GillespieSimulation(**sim_params)
            manager = MetricsManager()
            manager.register_simulation(sim)
            tracker = FrontDynamicsTracker(sim, log_interval=SIM_CONFIG["log_interval"])
            manager.add_tracker(tracker)

            while sim.time < SIM_CONFIG["total_run_time"]:
                did_step, _ = sim.step()
                if not did_step:
                    print("  Warning: Simulation stalled.")
                    break
                manager.after_step()

            rep_df = tracker.get_dataframe()
            rep_df["replicate"] = rep
            all_replicates_data.append(rep_df)

            if rep == SIM_CONFIG["num_replicates"] - 1:
                final_sim_state = sim

        # --- Generate Plots for this parameter set ---
        full_df = pd.concat(all_replicates_data, ignore_index=True)

        # 1. Time-Series Plot
        plt.figure(figsize=(14, 8))
        sns.lineplot(
            data=full_df,
            x="time",
            y="mutant_fraction",
            errorbar="sd",
            label="Mean Fraction",
        )
        plt.title(
            f'Approach to Steady State\n$k_{{total}}={params["k_total"]:.1f}, \\phi={params["phi"]:.2f}$',
            fontsize=18,
        )
        plt.xlabel("Simulation Time", fontsize=14)
        plt.ylabel("Mutant Fraction ($\\rho_M$)", fontsize=14)
        plt.ylim(0, 1)
        # Calculate mean of last 20% of the data points for the line
        final_mean = full_df[full_df["time"] > 0.8 * SIM_CONFIG["total_run_time"]][
            "mutant_fraction"
        ].mean()
        plt.axhline(
            final_mean,
            color="red",
            linestyle="--",
            label=f"Final Mean ({final_mean:.2f})",
        )
        plt.legend()
        plt.tight_layout()
        timeseries_filename = os.path.join(
            FIGURES_DIR,
            f'timeseries_k{params["k_total"]:.1f}_phi{params["phi"]:.2f}.png',
        )
        plt.savefig(timeseries_filename, dpi=300)
        plt.close()
        print(f"  ... Time-series plot saved to {timeseries_filename}")

        # 2. Final Snapshot Visualization
        if final_sim_state:
            plotter = HexPlotter(hex_size=5.0, labels=LABELS, colormap=COLOR_MAP)
            title = (
                f"Final State at t={final_sim_state.time:.0f}\n"
                f"$k_{{total}}={params['k_total']:.1f}, \\phi={params['phi']:.2f}$"
            )
            plotter.plot_population(
                final_sim_state.population,
                title=title,
                wt_front=final_sim_state.wt_front_cells,
                m_front=final_sim_state.m_front_cells,
            )
            snapshot_filename = os.path.join(
                FIGURES_DIR,
                f'snapshot_k{params["k_total"]:.1f}_phi{params["phi"]:.2f}.png',
            )
            plotter.save_figure(snapshot_filename)
            plt.close(plotter.fig)
            print(f"  ... Final snapshot saved to {snapshot_filename}")

    print("\nTime-series analysis complete.")


if __name__ == "__main__":
    main()
