# FILE: scripts/debug_front_collapse_viz.py
#
# A STANDALONE debugging tool to visualize "front collapse".
# This script runs a single simulation with hard-coded "suicide" parameters
# to observe and confirm the mechanism of population collapse.
# It saves a series of snapshots and compiles them into a GIF.
# The title of each frame includes live population counts to test the hypothesis
# that termination is caused by the extinction of the viable (Wild-Type) subpopulation.

import os
import sys
import shutil
from pathlib import Path
import json
from tqdm import tqdm
import imageio
from collections import Counter
import numba

# --- Robustly add project root to the Python path ---
# This is still needed to import the GillespieSimulation class.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# These imports are now safe because the path is set.
from src.core.model import GillespieSimulation


def generate_debug_title(sim: GillespieSimulation, snap_num: int):
    """Creates a rich, multi-line title for visualizing front dynamics."""

    # Count the number of WT and Mutant cells directly from the population dictionary
    counts = Counter(sim.population.values())
    num_wt = counts.get(1, 0)  # 1 is the integer for Wildtype
    num_m = counts.get(2, 0)  # 2 is the integer for Mutant

    line1 = f"Snapshot {snap_num:03d}"
    line2 = f"Time: {sim.time:.1f}, Step: {sim.step_count}"
    line3 = f"Front Cells (Active Sites): {len(sim._front_lookup)}"
    line4 = f"WT Cells: {num_wt}, Mutant Cells: {num_m}"  # Display the crucial counts

    return f"{line1}\n{line2}\n{line3}\n{line4}"


def main():
    """Main function to orchestrate the debug simulation and visualization."""
    print("--- Running Standalone Front Collapse Debug Visualization ---")

    # --- Standalone Parameter Definition ---
    # All parameters are defined directly here, removing the need for config.py.
    params = {
        "campaign_id": "debug_front_collapse_viz",
        "run_mode": "visualization",  # This enables the plotter
        "width": 256,
        "length": 25_000,
        "initial_condition_type": "patch",
        "initial_mutant_patch_size": 128,
        # These are the "suicide" parameters designed to trigger collapse
        "b_m": 0.1,  # Very slow mutant
        "phi": -1.0,  # Irreversible switch TO mutant
        "k_total": 1.0,  # A reasonably fast switching rate
    }
    # --- End of Standalone Parameter Definition ---

    print("\nUsing parameters:")
    print(json.dumps(params, indent=2))

    # --- Setup output directories ---
    output_dir = PROJECT_ROOT / "figures" / "debug_runs"
    snapshot_dir = output_dir / params["campaign_id"]

    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialize and run the simulation ---
    sim = GillespieSimulation(**params)

    max_steps = 2_000_000  # A generous limit
    num_snapshots = 100
    snapshot_interval = max_steps // num_snapshots
    snap_num = 0

    if not sim.plotter:
        print(
            "Error: Plotter was not initialized. The 'run_mode' must be 'visualization'."
        )
        sys.exit(1)

    final_termination_reason = "max_steps_reached"

    with tqdm(total=max_steps, desc="Simulating Collapse") as pbar:
        while sim.step_count < max_steps:
            # Save a snapshot at regular intervals
            if sim.step_count % snapshot_interval == 0:
                snap_num += 1
                title = generate_debug_title(sim, snap_num)
                snapshot_path = snapshot_dir / f"snap_{snap_num:03d}.png"
                sim.plotter.plot_population(sim.population, title=title)
                sim.plotter.save_figure(snapshot_path)

            active, boundary_hit = sim.step()
            pbar.update(1)

            # Check for termination conditions
            if boundary_hit:
                final_termination_reason = "boundary_hit"
                print("\nBoundary hit.")
                break
            if not active:
                final_termination_reason = "no_active_sites"
                print("\nNo active sites remaining.")
                break

    sim.plotter.close()
    print(f"\nFINAL TERMINATION REASON: {final_termination_reason.upper()}")

    # --- Compile snapshots into a GIF animation ---
    snapshot_files = sorted(snapshot_dir.glob("*.png"))
    if snapshot_files:
        gif_path = output_dir / f"{params['campaign_id']}_animation.gif"
        print(f"\nAssembling {len(snapshot_files)} frames into GIF...")
        with imageio.get_writer(gif_path, mode="I", duration=150, loop=0) as writer:
            for filename in tqdm(snapshot_files, desc="Creating GIF"):
                writer.append_data(imageio.imread(filename))
        print(f"✅ Debug visualization saved to: {gif_path}")
    else:
        print("No snapshot frames were generated.")


if __name__ == "__main__":
    main()
