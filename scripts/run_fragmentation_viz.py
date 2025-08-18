# FILE: scripts/run_fragmentation_viz.py (Final Version)
import os
import sys
import shutil
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import argparse

# No longer need imageio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS
from src.core.model import GillespieSimulation

# We need these to manually set up our custom behavior
from src.core.metrics import MetricsManager, FixationTimeTracker


def generate_fragmentation_title(
    sim: GillespieSimulation,
    snap_num: int,
    total_snaps: int,
    ic_type_str: str,
    outcome: str = "",
) -> str:
    """Creates a rich, multi-line title for visualizing fragmentation dynamics."""
    line1 = f"Initial State: {ic_type_str.upper()}"
    line2 = f"Snapshot {snap_num:02d} / {total_snaps} | Time: {sim.time:.1f}"
    line3 = f"Mutant Fitness b_m = {sim.global_b_m:.2f}"
    line4 = f"Mutant Count: {sim.mutant_cell_count}/{sim.total_cell_count} ({sim.mutant_fraction:.2%})"
    if outcome:
        return f"{line1}\n{line2}\n{line3}\n{line4}\n\nSIMULATION ENDED ({outcome})"
    return f"{line1}\n{line2}\n{line3}\n{line4}"


def main():
    parser = argparse.ArgumentParser(
        description="Run a visualization comparing clumped vs. fragmented initial states."
    )
    parser.add_argument(
        "--state", type=str, choices=["clumped", "fragmented"], required=True
    )
    args = parser.parse_args()

    print(
        f"--- Running Fragmentation Visualization for '{args.state}' initial state ---"
    )

    try:
        exp_config = EXPERIMENTS["debug_fragmentation_viz"]
        params = exp_config["sim_sets"]["main"]["base_params"].copy()

        if args.state == "clumped":
            params["initial_condition_type"] = "grf_threshold"
            params["correlation_length"] = 100.0
        else:  # fragmented
            params["initial_condition_type"] = "grf_threshold"
            params["correlation_length"] = 0.5

        # The run_mode from the config will be "visualization", which correctly enables the plotter
        params["run_mode"] = exp_config["run_mode"]
        params["campaign_id"] = exp_config["campaign_id"]
    except KeyError as e:
        print(f"Error: Could not find config keys. Details: {e}", file=sys.stderr)
        sys.exit(1)

    task_id = f"fragmentation_viz_{args.state}"
    output_dir = PROJECT_ROOT / "figures" / "debug_runs"
    snapshot_dir = output_dir / task_id
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    np.random.seed(42)  # Set seed for reproducibility
    sim = GillespieSimulation(**params)

    # --- THIS IS THE FIX FOR POINT #2 ---
    # The sim was initialized with run_mode="visualization", so sim.plotter exists.
    # Now, we manually add the FixationTimeTracker to control the simulation loop.
    manager = MetricsManager(params)
    manager.add_tracker(FixationTimeTracker, {})
    manager.register_simulation(sim)
    # --- END OF FIX ---

    if not sim.plotter:
        # This check will now pass successfully
        print("Error: Plotter was not initialized.", file=sys.stderr)
        sys.exit(1)

    max_time_for_viz = 4000.0  # Give it more time if needed
    num_snapshots = 40
    snapshot_interval = max_time_for_viz / num_snapshots
    next_snapshot_time = 0.0
    snap_num = 0

    with tqdm(total=int(max_time_for_viz), desc=f"Simulating ({args.state})") as pbar:
        # Initial snapshot
        title = generate_fragmentation_title(sim, snap_num, num_snapshots, args.state)
        sim.plotter.plot_population(sim.population, title=title)
        sim.plotter.save_figure(snapshot_dir / f"snap_{snap_num:02d}.png")
        next_snapshot_time += snapshot_interval

        # This loop is now robust
        while sim.time < max_time_for_viz:
            sim.step()
            manager.after_step_hook()  # The tracker checks for completion here
            pbar.update(int(sim.time) - pbar.n)

            if sim.time >= next_snapshot_time:
                snap_num = min(snap_num + 1, num_snapshots)
                title = generate_fragmentation_title(
                    sim, snap_num, num_snapshots, args.state
                )
                sim.plotter.plot_population(sim.population, title=title)
                sim.plotter.save_figure(snapshot_dir / f"snap_{snap_num:02d}.png")
                next_snapshot_time += snapshot_interval

            if manager.is_done():
                break

        results = manager.finalize()
        outcome = results.get("outcome", "MAX_TIME_REACHED").upper()
        print(f"\nSimulation ended ({outcome}) at time {sim.time:.2f}.")

        # Take one final snapshot to show the end state clearly
        snap_num = min(snap_num + 1, num_snapshots)
        title = generate_fragmentation_title(
            sim, snap_num, num_snapshots, args.state, outcome
        )
        sim.plotter.plot_population(sim.population, title=title)
        sim.plotter.save_figure(snapshot_dir / f"snap_{snap_num:02d}.png")

        pbar.n = int(max_time_for_viz)
        pbar.refresh()

    sim.plotter.close()

    # --- FIX FOR POINT #1: NO GIF ---
    print(f"\n✅ Snapshot images saved to: {snapshot_dir}")


if __name__ == "__main__":
    main()
