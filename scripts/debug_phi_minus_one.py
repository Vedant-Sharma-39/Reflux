# FILE: scripts/debug_phi_minus_one_viz.py
#
# A STANDALONE debugging tool to visualize the "absorbing state" dynamics
# for irreversible switching (phi = -1.0), as seen in the phase diagram.
#
# This script runs a single simulation with phi = -1.0 under two scenarios:
# 1. Disadvantaged mutants (b_m < 1.0): Expect collapse to pure Wild-Type.
# 2. Advantaged mutants (b_m > 1.0): Expect fixation to pure Mutant.
#
# It saves a series of snapshots and compiles them into a GIF.
#
# USAGE:
# python scripts/debug_phi_minus_one_viz.py --scenario disadvantaged
# python scripts/debug_phi_minus_one_viz.py --scenario advantaged

import os
import sys
import shutil
from pathlib import Path
import json
from tqdm import tqdm
import imageio
from collections import Counter
import argparse

# --- Robustly add project root to the Python path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# These imports are now safe because the path is set.
from src.core.model import GillespieSimulation


def generate_debug_title(sim: GillespieSimulation, snap_num: int, params: dict):
    """Creates a rich, multi-line title for visualizing phase diagram dynamics."""
    counts = Counter(sim.population.values())
    num_wt = counts.get(1, 0)
    num_m = counts.get(2, 0)

    line1 = f"Snapshot {snap_num:03d} | Scenario: {params['scenario'].upper()}"
    line2 = f"Time: {sim.time:.1f}, Step: {sim.step_count}"
    line3 = f"Params: φ={params['phi']:.1f}, k={params['k_total']:.1f}, b_m={params['b_m']:.1f}"
    line4 = f"WT Cells: {num_wt}, Mutant Cells: {num_m}"

    return f"{line1}\n{line2}\n{line3}\n{line4}"


def main():
    """Main function to orchestrate the debug simulation and visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize phi=-1.0 dynamics for phase diagram debugging."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["disadvantaged", "advantaged"],
        default="disadvantaged",
        help="Set mutant fitness relative to WT.",
    )
    args = parser.parse_args()

    print(
        f"--- Running Standalone Phi=-1.0 Debug Viz for '{args.scenario}' scenario ---"
    )

    # --- Standalone Parameter Definition ---
    params = {
        "scenario": args.scenario,
        "run_mode": "visualization",
        "width": 128,
        "length": 25_000,
        "initial_condition_type": "mixed",  # Start with both types present
        "phi": -1.0,  # Irreversible switch WT -> M
        "k_total": 0.5,  # A moderate switching rate
    }

    if args.scenario == "disadvantaged":
        params["b_m"] = 0.8  # Mutants are less fit
    else:  # advantaged
        params["b_m"] = 1.2  # Mutants are more fit

    # Campaign ID changes based on scenario for separate output files
    params["campaign_id"] = f"debug_phi_minus_one_{args.scenario}"
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

    max_steps = 1_000_000
    num_snapshots = 100
    snapshot_interval = max_steps // num_snapshots
    snap_num = 0

    if not sim.plotter:
        print("Error: Plotter was not initialized. 'run_mode' must be 'visualization'.")
        sys.exit(1)

    final_termination_reason = "max_steps_reached"

    with tqdm(total=max_steps, desc=f"Simulating ({args.scenario})") as pbar:
        while sim.step_count < max_steps:
            if sim.step_count % snapshot_interval == 0:
                snap_num += 1
                title = generate_debug_title(sim, snap_num, params)
                snapshot_path = snapshot_dir / f"snap_{snap_num:03d}.png"
                sim.plotter.plot_population(
                    sim.population, sim.mean_front_position, sim.width, title=title
                )
                sim.plotter.save_figure(snapshot_path)

            active, boundary_hit = sim.step()
            pbar.update(1)

            # Check for termination conditions
            if boundary_hit:
                final_termination_reason = "boundary_hit"
                break
            if not active:
                final_termination_reason = "no_active_sites"
                break
            # Additional check for absorbing states
            if (
                sim.mutant_cell_count == 0
                or sim.mutant_cell_count == sim.total_cell_count
            ):
                final_termination_reason = "absorbing_state_reached"
                break

    # Final snapshot
    snap_num += 1
    title = generate_debug_title(sim, snap_num, params)
    snapshot_path = snapshot_dir / f"snap_{snap_num:03d}.png"
    sim.plotter.plot_population(
        sim.population, sim.mean_front_position, sim.width, title=title
    )
    sim.plotter.save_figure(snapshot_path)
    sim.plotter.close()

    print(f"\nFINAL TERMINATION REASON: {final_termination_reason.upper()}")
    counts = Counter(sim.population.values())
    print(f"Final Counts: WT={counts.get(1, 0)}, M={counts.get(2, 0)}")

    # --- Compile snapshots into a GIF animation ---
    snapshot_files = sorted(snapshot_dir.glob("*.png"))
    if snapshot_files:
        gif_path = output_dir / f"{params['campaign_id']}_animation.gif"
        print(f"\nAssembling {len(snapshot_files)} frames into GIF...")
        with imageio.get_writer(gif_path, mode="I", duration=150, loop=0) as writer:
            for filename in tqdm(snapshot_files, desc="Creating GIF"):
                writer.append_data(imageio.v2.imread(filename))
        print(f"✅ Debug visualization saved to: {gif_path}")
    else:
        print("No snapshot frames were generated.")


if __name__ == "__main__":
    main()
