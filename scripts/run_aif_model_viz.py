# FILE: scripts/run_aif_model_viz.py (Corrected for Pathing and Centered Plotting)
# A standalone visualization script to test the AifModelSimulation class.
# This version includes a fix to properly center the colony in the frame and
# robustly handles the Python import path.

import os
import sys
import shutil
from pathlib import Path
import json
from tqdm import tqdm
import imageio
import numpy as np

# --- FIX: Add project root to Python path ---
# This block is crucial for making imports like 'from src.core...' work correctly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --- END OF FIX ---

# Now these imports will succeed
from src.core.model_aif import AifModelSimulation


def generate_aif_viz_title(
    sim: AifModelSimulation, snap_num: int, total_snaps: int
) -> str:
    """Creates a rich, multi-line title for visualizing the Aif model dynamics."""
    line1 = f"Snapshot {snap_num:03d} / {total_snaps}"
    line2 = f"Step: {sim.step_count}, Time: {sim.time:.1f}"
    line3 = f"Params: b_res={sim.b_res:.3f}, k_rescue={sim.k_res_comp:.1e}"
    line4 = (
        f"Susceptible: {sim.susceptible_cell_count} | "
        f"Resistant: {sim.resistant_cell_count} | "
        f"Compensated: {sim.compensated_cell_count}"
    )
    line5 = f"Colony Radius: {sim.colony_radius:.1f}"

    return f"{line1}\n{line2}\n{line3}\n{line4}\n{line5}"


def center_radial_plot(sim: AifModelSimulation):
    """
    Manually calculates the colony's bounding box and centers the plot view.
    This overrides the default linear-front plotting behavior.
    """
    if not sim.population or not sim.plotter:
        return

    points = np.array(
        [sim.plotter._axial_to_cartesian(h) for h in sim.population.keys()]
    )

    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)

    center = (min_coords + max_coords) / 2.0
    span = max_coords - min_coords
    radius = max(span) / 2.0 + sim.plotter.size * 4

    sim.plotter.ax.set_xlim(center[0] - radius, center[0] + radius)
    sim.plotter.ax.set_ylim(center[1] - radius, center[1] + radius)


def main():
    """Main function to configure, run, and render the visualization."""
    print("--- Running Aif Model Replication Visualization ---")

    params = {
        "campaign_id": "debug_aif_model_viz",
        "run_mode": "visualization",
        "initial_droplet_radius": 25,
        "initial_condition_type": "aif_droplet",
        "initial_resistant_fraction": 0.1,
        "b_sus": 1.0,
        "b_res": 1.0 - 0.02,
        "b_comp": 1.0,
        "k_res_comp": 1e-2,
    }
    task_id = "aif_model_radial_growth_viz"

    print("\nUsing parameters:")
    print(json.dumps(params, indent=2))

    output_dir = PROJECT_ROOT / "figures" / "debug_runs"
    snapshot_dir = output_dir / task_id
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    sim = AifModelSimulation(**params)

    max_steps = 200_000
    num_snapshots = 50
    snapshot_interval = max_steps // num_snapshots

    if not sim.plotter:
        print("Error: Plotter was not initialized. Check 'run_mode' in params.")
        sys.exit(1)

    with tqdm(total=max_steps, desc="Simulating Radial Growth") as pbar:
        for snap_num in range(num_snapshots + 1):
            title = generate_aif_viz_title(sim, snap_num, num_snapshots)
            snapshot_path = snapshot_dir / f"snap_{snap_num:03d}.png"

            sim.plotter.plot_population(
                sim.population, mean_front_q=0, sim_width=sim.width, title=title
            )
            center_radial_plot(sim)
            sim.plotter.save_figure(snapshot_path)

            if sim.step_count >= max_steps:
                break

            target_steps = (snap_num + 1) * snapshot_interval
            while sim.step_count < target_steps:
                active, _ = sim.step()
                pbar.update(1)
                if not active:
                    break
            if not sim.population:
                print("\nSimulation ended (population extinction).")
                break

    sim.plotter.close()

    snapshot_files = sorted(snapshot_dir.glob("*.png"))
    if snapshot_files:
        gif_path = output_dir / f"{task_id}_animation.gif"
        print(f"\nAssembling {len(snapshot_files)} frames into GIF...")
        with imageio.get_writer(gif_path, mode="I", duration=150, loop=0) as writer:
            for filename in tqdm(snapshot_files, desc="Creating GIF"):
                writer.append_data(imageio.v2.imread(filename))
        print(f"✅ Debug visualization saved to: {gif_path}")
    else:
        print("No snapshot frames were generated.")


if __name__ == "__main__":
    main()
