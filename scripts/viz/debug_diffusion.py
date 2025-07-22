# FILE: scripts/viz/debug_diffusion.py
# [ENHANCED] This version now runs the InterfaceRoughnessTracker in parallel
# with the visualization, printing the calculated roughness at each snapshot
# to allow for direct debugging of the metric implementation.

import sys
import os
import matplotlib.pyplot as plt

# --- Robust Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "src"))

from linear_model import GillespieSimulation, Wildtype, Mutant
from hex_utils import HexPlotter
from metrics import MetricsManager, InterfaceRoughnessTracker  # <-- IMPORT METRICS

# ==============================================================================
# 1. DEFINE SIMULATION & VISUALIZATION PARAMETERS
# ==============================================================================
PARAMS = {
    "width": 64,  # Use a smaller width for a faster debug run
    "length": 200,
    "b_m": 1.0,
    "k_total": 0.0,
    "phi": 0.0,
    "initial_mutant_patch_size": 0,
}
MAX_SIM_TIME = 100.0
SNAPSHOT_INTERVAL = 25.0
FIGURES_DIR = os.path.join(
    project_root, "figures", "diffusion_debug_snapshots_with_metric"
)
os.makedirs(FIGURES_DIR, exist_ok=True)

COLOR_MAP = {Wildtype: "#0077b6"}
LABELS = {Wildtype: "Growing Species"}


# ==============================================================================
# 2. MAIN VISUALIZATION SCRIPT
# ==============================================================================
def main():
    print("--- Starting Diffusion Debug Visualization (with Metric Tracking) ---")
    sim = GillespieSimulation(**PARAMS)
    plotter = HexPlotter(hex_size=3.0, labels=LABELS, colormap=COLOR_MAP)

    # --- [NEW] SETUP METRICS TRACKING ---
    manager = MetricsManager()
    manager.register_simulation(sim)
    # Use a small capture interval to get frequent data points
    tracker = InterfaceRoughnessTracker(sim, capture_interval=0.2)
    manager.add_tracker(tracker)
    manager.initialize_all()
    # ------------------------------------

    next_snapshot_time = 0.0
    frame_count = 0

    while sim.time < MAX_SIM_TIME:
        # Take a snapshot when it's time
        if sim.time >= next_snapshot_time:
            print("-" * 50)
            print(f"  ... saving snapshot at time t = {sim.time:.2f}")

            # --- [NEW] GET AND PRINT THE LATEST METRIC DATA ---
            # The tracker's history contains (q_mean, w_squared) tuples. Get the latest one.
            trajectory = tracker.get_roughness_trajectory()
            if trajectory:
                last_q, last_w_sq = trajectory[-1]
                metric_text = (
                    f"Tracker W²: {last_w_sq:.4f} at q={last_q:.2f}\n"
                    f"Sim Mean q: {sim.mean_front_position:.2f}"
                )
                print(f"  METRIC CHECK: W² = {last_w_sq:.4f}")
            else:
                metric_text = "Tracker W²: (No data yet)"
                print("  METRIC CHECK: (No data yet)")
            # ----------------------------------------------------

            # Create the plot title including the metric
            title = f"Time: {sim.time:.2f}\n" f"{metric_text}"

            plotter.plot_population(sim.population, title=title)
            snapshot_filename = os.path.join(
                FIGURES_DIR, f"snapshot_diffusion_metric_{frame_count:04d}.png"
            )
            plotter.save_figure(snapshot_filename)
            frame_count += 1
            next_snapshot_time += SNAPSHOT_INTERVAL

        # --- RUN THE SIMULATION STEP ---
        did_step, boundary_hit = sim.step()
        if not did_step or boundary_hit:
            print(f"\nSimulation STALLED or hit boundary at time t = {sim.time:.4f}")
            break

        # --- [CRITICAL] UPDATE THE METRICS MANAGER AFTER THE STEP ---
        manager.after_step()
        # -----------------------------------------------------------

    print("\nVisualization complete.")
    plt.close(plotter.fig)


if __name__ == "__main__":
    main()
