import argparse
import json
import os
import sys

# --- ### CORRECTED PATH SETUP ### ---
# The project root is the directory that contains the 'src' folder.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# We must add the project root to the path, NOT src/core.
# This allows absolute imports like 'from src.core...' to work everywhere.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- ### CORRECTED IMPORTS ### ---
# Now that the project root is on the path, we use absolute imports.
from src.core.model import GillespieSimulation
from src.core.hex_utils import HexPlotter


def visualize_task(params: dict):
    """
    Instantiates a simulation, runs it for a small number of steps,
    and saves a snapshot of the population state for debugging.
    """
    task_id = params.get("task_id", "debug_task")
    print(f"--- Visualizing Task: {task_id} ---")
    print(f"Parameters: b_m={params.get('b_m')}, k_total={params.get('k_total')}")

    # --- 1. Instantiate the Simulation and Plotter ---
    sim = GillespieSimulation(**params)

    colormap = {0: "white", 1: "#0c2c5c", 2: "#d6a000"}
    plotter = HexPlotter(hex_size=1.0, labels={}, colormap=colormap)

    # --- 2. Run for a Fixed Number of Steps ---
    num_debug_steps = 2000
    print(f"Running simulation for {num_debug_steps} steps...")
    i = 0
    for i in range(num_debug_steps):
        active, _ = sim.step()
        if not active:
            print(
                f"SIMULATION STOPPED PREMATURELY at step {i+1} because it ran out of valid moves."
            )
            break
    if i == num_debug_steps - 1:
        print("... finished running steps.")

    # --- 3. Generate the Visualization ---
    final_population = sim.population
    active_front = sim._front_lookup

    title = f"Task: {task_id}\nSteps: {i+1}, b_m={params.get('b_m'):.2f}, Active Front Cells: {len(active_front)}"

    plotter.plot_population(
        population=final_population, title=title, wt_front=active_front, m_front=set()
    )

    # --- 4. Save the Figure ---
    output_dir = os.path.join(project_root, "debug_images")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"debug_{task_id}.png")
    plotter.save_figure(output_path)

    print(f"\nSUCCESS: Debug visualization saved to:\n{output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Debug visualizer for a single simulation task."
    )
    parser.add_argument(
        "--params",
        required=True,
        help="JSON string of the simulation parameters for a single task.",
    )
    args = parser.parse_args()

    try:
        params = json.loads(args.params)
        visualize_task(params)
    except Exception as e:
        print(f"FATAL ERROR: Could not run visualization. Exception: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
