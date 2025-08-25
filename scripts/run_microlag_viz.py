# FILE: scripts/run_transient_viz.py

import os
import sys
import shutil
from pathlib import Path
import json
from tqdm import tqdm
import imageio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS
# --- IMPORT THE CORRECT SIMULATION CLASS ---
from src.core.model_transient import GillespieTransientStateSimulation


def generate_transient_title(sim: GillespieTransientStateSimulation, snap_num: int, total_snaps: int) -> str:
    """Creates a rich title for visualizing transient state dynamics."""
    line1 = f"Snapshot {snap_num:02d} / {total_snaps}"
    line2 = f"Step: {sim.step_count}, Time: {sim.time:.1f}"
    lag_duration = sim.switching_lag_duration
    line3 = f"Params: k_total={sim.global_k_total}, φ={sim.global_phi}, lag_duration={lag_duration:.1f}"
    
    # Count how many cells are currently "stuck"
    num_stuck = len(sim.delayed_events)
    line4 = f"Cells in Transient State: {num_stuck}"
    
    return f"{line1}\n{line2}\n{line3}\n{line4}"


def main():
    print("--- Running Transient State Switching Visualization ---")

    try:
        exp_config = EXPERIMENTS["debug_transient_state_viz"]
        params = exp_config["sim_sets"]["main"]["base_params"].copy()
        params["run_mode"] = exp_config["run_mode"]
        params["campaign_id"] = exp_config["campaign_id"]
    except KeyError as e:
        print(f"Error: Could not find config key 'debug_transient_state_viz'. Details: {e}", file=sys.stderr)
        sys.exit(1)

    task_id = "transient_state_viz"

    print("\nUsing parameters:")
    print(json.dumps(params, indent=2))

    output_dir = PROJECT_ROOT / "figures" / "debug_runs"
    snapshot_dir = output_dir / task_id
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    # We explicitly use the new simulation class
    sim = GillespieTransientStateSimulation(**params)

    max_steps = 300000
    num_snapshots = 30
    snapshot_interval = max_steps // num_snapshots

    if not sim.plotter:
        print("Error: Plotter not initialized.", file=sys.stderr)
        sys.exit(1)
        
    print("\nStarting simulation loop...")
    with tqdm(total=max_steps, desc="Simulating") as pbar:
        # Initial snapshot
        title = generate_transient_title(sim, 0, num_snapshots)
        snapshot_path = snapshot_dir / "snap_00.png"
        sim.plotter.plot_population(
            sim.population, sim.mean_front_position, sim.width,
            title=title, q_to_patch_index=sim.q_to_patch_index
        )
        sim.plotter.save_figure(snapshot_path)

        while sim.step_count < max_steps:
            active, boundary_hit = sim.step()
            pbar.update(1)
            if sim.step_count > 0 and sim.step_count % snapshot_interval == 0:
                snap_num = sim.step_count // snapshot_interval
                title = generate_transient_title(sim, snap_num, num_snapshots)
                snapshot_path = snapshot_dir / f"snap_{snap_num:02d}.png"
                sim.plotter.plot_population(
                    sim.population, sim.mean_front_position, sim.width,
                    title=title, q_to_patch_index=sim.q_to_patch_index
                )
                sim.plotter.save_figure(snapshot_path)
            
            if not active or boundary_hit:
                break

    sim.plotter.close()

    # Assemble GIF
    snapshot_files = sorted(snapshot_dir.glob("*.png"))
    if snapshot_files:
        gif_path = output_dir / f"{task_id}_animation.gif"
        with imageio.get_writer(gif_path, mode="I", duration=200, loop=0) as writer:
            for filename in tqdm(snapshot_files, desc="Creating GIF"):
                writer.append_data(imageio.v2.imread(filename))
        print(f"✅ Debug visualization saved to: {gif_path}")

if __name__ == "__main__":
    main()