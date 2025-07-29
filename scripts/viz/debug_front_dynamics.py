# FILE: scripts/viz/debug_front_dynamics.py
#
# A dedicated script to visually inspect the front dynamics in different
# physical regimes to test the hypotheses explaining the non-monotonic
# phase boundary.

import sys
import os
import matplotlib.pyplot as plt

# --- Robust Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "src"))

from linear_model import GillespieSimulation, Wildtype, Mutant
from hex_utils import HexPlotter

# ==============================================================================
# 1. DEFINE THE KEY PARAMETER SETS TO VISUALIZE
# ==============================================================================
# We choose a k_total BELOW the critical point for all cases to see the
# structure of the ORDERED phase that is being stabilized.

PARAMS_TO_VISUALIZE = [
    {
        "name": "Weak_Selection_Drift_Dominated",
        "s": -0.05,
        "k_total": 0.1,
        "phi": 0.0,
        "description": "Hypothesis: Wide, wandering domains; relatively flat front.",
    },
    {
        "name": "Strong_Selection_Geometry_Dominated",
        "s": -0.80,
        "k_total": 0.1,
        "phi": 0.0,
        "description": "Hypothesis: Very rough front with deep, stable 'fjords'.",
    },
    {
        "name": "Kc_Minimum_Most_Fragile",
        "s": -0.18,
        "k_total": 0.1,
        "phi": 0.0,
        "description": "Hypothesis: Intermediate state; more fragmented than weak selection.",
    },
]

# ==============================================================================
# 2. SIMULATION & VISUALIZATION CONFIGURATION
# ==============================================================================
SIM_CONFIG = {
    "width": 256,  # Use a wide front to see large-scale structures
    "length": 1000,
    "max_sim_time": 400.0,
    "snapshot_interval": 100.0,
}

BASE_FIGURES_DIR = os.path.join(project_root, "figures", "front_dynamics_debug")
os.makedirs(BASE_FIGURES_DIR, exist_ok=True)

COLOR_MAP = {Wildtype: "#0d3b66", Mutant: "#faf0ca"}  # High contrast colors
LABELS = {Wildtype: "Wild-Type", Mutant: "Mutant"}


# ==============================================================================
# 3. MAIN VISUALIZATION SCRIPT
# ==============================================================================
def main():
    print("--- Starting Front Dynamics Debug Visualization ---")

    for params in PARAMS_TO_VISUALIZE:
        print(f"\n--- Running Visualization for: {params['name']} ---")
        print(f"    s = {params['s']:.2f}, k_total = {params['k_total']:.2f}")
        print(f"    Hypothesis: {params['description']}")

        # --- Setup simulation and plotter for this run ---
        sim_params = {
            "width": SIM_CONFIG["width"],
            "length": SIM_CONFIG["length"],
            "b_m": params["s"] + 1.0,  # Convert s to b_m
            "k_total": params["k_total"],
            "phi": params["phi"],
        }
        sim = GillespieSimulation(**sim_params)
        plotter = HexPlotter(hex_size=2.0, labels=LABELS, colormap=COLOR_MAP)

        # --- Create a dedicated directory for this run's snapshots ---
        run_dir = os.path.join(BASE_FIGURES_DIR, params["name"])
        os.makedirs(run_dir, exist_ok=True)
        print(f"    Saving snapshots to: {run_dir}")

        # --- Run simulation and take snapshots ---
        next_snapshot_time = 0.0
        frame_count = 0
        while sim.time < SIM_CONFIG["max_sim_time"]:
            if sim.time >= next_snapshot_time:
                print(f"      ... saving snapshot at time t = {sim.time:.2f}")
                title = (
                    f"Regime: {params['name']}\n"
                    f"s = {params['s']:.2f}, $k_{{total}}$ = {params['k_total']:.2f}, Time = {sim.time:.1f}"
                )

                # Plot the population, highlighting the front cells
                plotter.plot_population(
                    sim.population,
                    title=title,
                    wt_front=sim.wt_front_cells,
                    m_front=sim.m_front_cells,
                )

                snapshot_filename = os.path.join(
                    run_dir, f"snapshot_{frame_count:03d}.png"
                )
                plotter.save_figure(snapshot_filename, dpi=200)
                frame_count += 1
                next_snapshot_time += SIM_CONFIG["snapshot_interval"]

            # --- Run one step of the simulation ---
            did_step, boundary_hit = sim.step()
            if not did_step or boundary_hit:
                print(f"      Simulation stalled or hit boundary at t = {sim.time:.4f}")
                break

        # Save final state
        title = (
            f"FINAL STATE - Regime: {params['name']}\n"
            f"s = {params['s']:.2f}, $k_{{total}}$ = {params['k_total']:.2f}, Time = {sim.time:.1f}"
        )
        plotter.plot_population(
            sim.population,
            title=title,
            wt_front=sim.wt_front_cells,
            m_front=sim.m_front_cells,
        )
        snapshot_filename = os.path.join(
            run_dir, f"snapshot_{frame_count:03d}_final.png"
        )
        plotter.save_figure(snapshot_filename, dpi=200)

        plt.close(plotter.fig)  # Free up memory before next run

    print("\n--- Visualization complete. ---")


if __name__ == "__main__":
    main()
