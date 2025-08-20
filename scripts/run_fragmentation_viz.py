# FILE: scripts/run_fragmentation_viz.py (Corrected with required arguments for plot_population)
import os
import sys
import shutil
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS
from src.core.model import GillespieSimulation
from src.core.metrics import MetricsManager, FixationTimeTracker


def generate_fragmentation_title(
    sim, snap_num, total_snaps, ic_type_str, outcome="", q_max=0.0
):
    """Creates a rich, multi-line title for visualizing fragmentation dynamics."""
    line1 = f"Initial State: {ic_type_str.upper()}"
    line2 = f"Snapshot {snap_num:02d} / {total_snaps} | Time: {sim.time:.1f}"
    line3 = f"Mutant Fitness b_m = {sim.global_b_m:.2f}"
    line4 = f"Mutant Count: {sim.mutant_cell_count}/{sim.total_cell_count} ({sim.mutant_fraction:.2%})"
    if outcome:
        outcome_line = f"\n\nSIMULATION ENDED ({outcome})"
        q_max_line = f"\nMax Mutant q: {q_max:.1f}"
        return f"{line1}\n{line2}\n{line3}\n{line4}{outcome_line}{q_max_line}"
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

        params["b_m"] = 0.8
        if args.state == "clumped":
            params["initial_condition_type"] = "grf_threshold"
            params["correlation_length"] = 100.0
        else:  # fragmented
            params["initial_condition_type"] = "grf_threshold"
            params["correlation_length"] = 5

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
    sim = GillespieSimulation(**params)
    manager = MetricsManager(params)
    manager.add_tracker(FixationTimeTracker, {})
    manager.register_simulation(sim)

    if not sim.plotter:
        print("Error: Plotter was not initialized.", file=sys.stderr)
        sys.exit(1)

    max_time_for_viz = 4000.0
    num_snapshots = 40
    snapshot_interval = max_time_for_viz / num_snapshots
    next_snapshot_time = 0.0
    snap_num = 0

    with tqdm(total=int(max_time_for_viz), desc=f"Simulating ({args.state})") as pbar:
        # Initial snapshot
        title = generate_fragmentation_title(sim, snap_num, num_snapshots, args.state)

        # <<< FIX IS HERE (1 of 3) >>>
        # Provide the missing mean_front_position and width arguments.
        sim.plotter.plot_population(
            sim.population, sim.mean_front_position, sim.width, title=title
        )
        sim.plotter.save_figure(snapshot_dir / f"snap_{snap_num:02d}.png")
        next_snapshot_time += snapshot_interval

        while sim.time < max_time_for_viz:
            sim.step()
            manager.after_step_hook()
            pbar.update(int(sim.time) - pbar.n)

            if sim.time >= next_snapshot_time:
                snap_num = min(snap_num + 1, num_snapshots)
                title = generate_fragmentation_title(
                    sim, snap_num, num_snapshots, args.state
                )

                # <<< FIX IS HERE (2 of 3) >>>
                # Provide the missing arguments inside the loop as well.
                sim.plotter.plot_population(
                    sim.population, sim.mean_front_position, sim.width, title=title
                )
                sim.plotter.save_figure(snapshot_dir / f"snap_{snap_num:02d}.png")
                next_snapshot_time += snapshot_interval

            if manager.is_done():
                break

    results = manager.finalize()
    outcome = results.get("outcome", "MAX_TIME_REACHED").upper()
    q_at_outcome = results.get("q_at_outcome", sim.mean_front_position)
    print(
        f"\nSimulation ended ({outcome}) at time {sim.time:.2f}. Final q: {q_at_outcome:.1f}."
    )

    # Take one final snapshot to show the end state clearly
    snap_num = min(snap_num + 1, num_snapshots)
    title = generate_fragmentation_title(
        sim, snap_num, num_snapshots, args.state, outcome, q_max=q_at_outcome
    )

    # <<< FIX IS HERE (3 of 3) >>>
    # This call was already correct in your provided code, but is included for completeness.
    # It correctly centers the final plot on the historical max q.
    sim.plotter.plot_population(sim.population, q_at_outcome, sim.width, title=title)
    sim.plotter.save_figure(snapshot_dir / f"snap_{snap_num:02d}.png")

    # Save the final population data for post-processing
    final_pop_data = [
        {"q": h.q, "r": h.r, "type": t} for h, t in sim.population.items()
    ]
    data_path = snapshot_dir / "final_population_data.json"
    with open(data_path, "w") as f:
        json.dump(final_pop_data, f)
    print(f"✅ Saved final population data for post-processing to: {data_path}")

    sim.plotter.close()
    print(f"\nSnapshot images and data saved to: {snapshot_dir}")


if __name__ == "__main__":
    main()
