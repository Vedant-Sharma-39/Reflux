# FILE: scripts/debug_boundary_dynamics.py
#
# A standalone script to run and visualize a single boundary dynamics simulation.
#
# THIS VERSION CORRECTLY:
# 1. Measures the mutant sector width ONLY on the expanding front.
# 2. Stops the simulation when the mutant lineage is lost from the front.

import sys
import shutil
from pathlib import Path
import json
from tqdm import tqdm
import imageio
import numpy as np
import matplotlib.pyplot as plt

# --- Add project root to Python path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.model import GillespieSimulation

# --- THIS IS THE KEY CORRECTION (1/3): Measure width on the front ---
def get_mutant_front_width(sim: GillespieSimulation) -> int:
    """Calculates the width of the mutant sector on the expanding front."""
    if not sim.m_front_cells:
        return 0
    # A set of the 'r' coordinates of all mutant cells currently on the front
    front_r_coords = {h.r for h in sim.m_front_cells.keys()}
    return len(front_r_coords)

# --- Helper Function for Plot Titles (Updated) ---
def generate_debug_title(sim: GillespieSimulation, snap_num: int) -> str:
    """Creates a rich, multi-line title for visualizing boundary dynamics."""
    line1 = f"Snapshot {snap_num:03d}"
    line2 = f"Time: {sim.time:.1f}, Step: {sim.step_count}"
    line3 = f"Params: b_m = {sim.global_b_m:.3f} (s = {sim.global_b_m - 1.0:.3f})"
    # Use the corrected width measurement
    current_front_width = get_mutant_front_width(sim)
    line4 = f"Current Mutant FRONT Width: {current_front_width}"
    return f"{line1}\n{line2}\n{line3}\n{line4}"

# --- Main Debugging Function ---
def main():
    print("--- Running Boundary Dynamics Debug Visualization ---")

    # --- 1. CONFIGURATION ---
    params = {
        "campaign_id": "debug_boundary_dynamics",
        "run_mode": "visualization",
        "width": 256,
        "length": 1024,
        "initial_condition_type": "patch",
        "initial_mutant_patch_size": 128,
        "b_m": 0.92,
        "k_total": 0.0,
        "phi": 0.0,
    }
    task_id = f"debug_bm_{params['b_m']:.3f}"
    max_steps = 2_000_000
    num_snapshots = 100
    snapshot_interval = max_steps // num_snapshots

    print("\nUsing parameters:")
    print(json.dumps(params, indent=2))

    # --- 2. SETUP OUTPUT DIRECTORIES ---
    output_dir = PROJECT_ROOT / "figures" / "debug_runs"
    snapshot_dir = output_dir / task_id
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # --- 3. INITIALIZE AND RUN THE SIMULATION ---
    sim = GillespieSimulation(**params)
    width_trajectory = []

    if not sim.plotter:
        sys.exit("Error: Plotter not initialized. 'run_mode' must be 'visualization'.")

    final_termination_reason = "max_steps_reached"
    snap_num = 0

    with tqdm(total=max_steps, desc=f"Simulating (b_m={params['b_m']})") as pbar:
        while sim.step_count < max_steps:
            if sim.step_count % snapshot_interval == 0:
                snap_num += 1
                # --- THIS IS THE KEY CORRECTION (2/3): Record the front width ---
                current_front_width = get_mutant_front_width(sim)
                width_trajectory.append((sim.time, current_front_width))

                title = generate_debug_title(sim, snap_num)
                snapshot_path = snapshot_dir / f"snap_{snap_num:03d}.png"
                sim.plotter.plot_population(sim.population, sim.mean_front_position, sim.width, title=title)
                sim.plotter.save_figure(snapshot_path)

            active, boundary_hit = sim.step()
            pbar.update(1)

            # --- THIS IS THE KEY CORRECTION (3/3): New termination logic ---
            front_has_mutants = bool(sim.m_front_cells)
            front_has_wildtype = bool(sim.wt_front_cells)

            if not front_has_mutants and sim.step_count > 0:
                final_termination_reason = "mutant_front_extinction"
                break
            if not front_has_wildtype and sim.step_count > 0:
                final_termination_reason = "mutant_front_fixation"
                break
            if not active:
                final_termination_reason = "no_active_sites"
                break
            if boundary_hit:
                final_termination_reason = "boundary_hit"
                break

    # Record the final state
    width_trajectory.append((sim.time, get_mutant_front_width(sim)))
    sim.plotter.close()
    print(f"\nSIMULATION FINISHED. Reason: {final_termination_reason.upper()}")

    # --- 4. COMPILE GIF ---
    snapshot_files = sorted(snapshot_dir.glob("*.png"))
    if snapshot_files:
        gif_path = output_dir / f"{task_id}_animation.gif"
        print(f"\nAssembling GIF...")
        with imageio.get_writer(gif_path, mode="I", duration=150, loop=0) as writer:
            for filename in tqdm(snapshot_files, desc="Creating GIF"):
                writer.append_data(imageio.v2.imread(filename))
        print(f"✅ Debug animation saved to: {gif_path}")

    # --- 5. PLOT THE MEASURED FRONT WIDTH TRAJECTORY ---
    if width_trajectory:
        traj_path = output_dir / f"{task_id}_width_trajectory.png"
        print(f"Plotting front width vs. time trajectory...")
        traj_array = np.array(width_trajectory)
        
        plt.figure(figsize=(10, 6))
        plt.plot(traj_array[:, 0], traj_array[:, 1], '-o', markersize=4)
        plt.title(f"Mutant Sector FRONT Width vs. Time (b_m = {params['b_m']:.3f})")
        plt.xlabel("Simulation Time")
        plt.ylabel("Mutant Sector Width on Front")
        plt.grid(True, linestyle=':')
        plt.savefig(traj_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Trajectory plot saved to: {traj_path}")

if __name__ == "__main__":
    main()