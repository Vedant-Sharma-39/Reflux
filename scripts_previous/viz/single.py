# scripts/visualize_single_run.py
# Updated to highlight front cells in the snapshots.

import sys
import os
import matplotlib.pyplot as plt

# --- Robust Path Setup ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = os.path.join(project_root, "src")
    sys.path.insert(0, src_path)
except NameError:
    sys.path.insert(0, "../src")

from linear_model import GillespieSimulation, Wildtype, Mutant
from hex_utils import HexPlotter

# ==============================================================================
# 1. DEFINE SIMULATION & VISUALIZATION PARAMETERS
# ==============================================================================
PARAMS = {
    "width": 64,
    "length": 200,
    "b_m": 0.8,
    "k_total": 10.0,
    "phi": -0.5,
}
MAX_SIM_TIME = 100.0
SNAPSHOT_INTERVAL = 25.0
FIGURES_DIR = os.path.join(project_root, "figures", "single_run_snapshots_with_front")
os.makedirs(FIGURES_DIR, exist_ok=True)

COLOR_MAP = {Wildtype: "#3A86FF", Mutant: "#FFBE0B"}
LABELS = {Wildtype: "Wild-Type", Mutant: "Mutant"}


# ==============================================================================
# 2. MAIN VISUALIZATION SCRIPT
# ==============================================================================
def main():
    print("--- Starting Single Run Visualization (with Front Highlighting) ---")
    sim = GillespieSimulation(**PARAMS)
    plotter = HexPlotter(hex_size=5.0, labels=LABELS, colormap=COLOR_MAP)

    next_snapshot_time = 0.0
    frame_count = 0

    while sim.time < MAX_SIM_TIME:
        if sim.time >= next_snapshot_time:
            print(f"  ... saving snapshot at time t = {sim.time:.2f}")
            title = (
                f"Time: {sim.time:.2f}\n"
                f"WT Front: {len(sim.wt_front_cells)}, M Front: {len(sim.m_front_cells)}"
            )

            # --- PASS THE FRONT DICTIONARIES TO THE PLOTTER ---
            plotter.plot_population(
                sim.population,
                title=title,
                wt_front=sim.wt_front_cells,
                m_front=sim.m_front_cells,
            )

            snapshot_filename = os.path.join(
                FIGURES_DIR, f"snapshot_{frame_count:04d}.png"
            )
            plotter.save_figure(snapshot_filename)
            frame_count += 1
            next_snapshot_time += SNAPSHOT_INTERVAL

        did_step, _ = sim.step()
        if not did_step:
            print(f"\nSimulation STALLED at time t = {sim.time:.4f}")
            break

    # --- Final Snapshot ---
    print("\nSimulation finished. Saving final state.")
    title = (
        f"FINAL STATE - Time: {sim.time:.2f}\n"
        f"WT Front: {len(sim.wt_front_cells)}, M Front: {len(sim.m_front_cells)}"
    )
    plotter.plot_population(
        sim.population,
        title=title,
        wt_front=sim.wt_front_cells,
        m_front=sim.m_front_cells,
    )
    final_snapshot_filename = os.path.join(
        FIGURES_DIR, f"snapshot_{frame_count:04d}_final.png"
    )
    plotter.save_figure(final_snapshot_filename)

    print(f"\nVisualization complete. {frame_count} images saved.")
    plt.close(plotter.fig)


if __name__ == "__main__":
    main()
