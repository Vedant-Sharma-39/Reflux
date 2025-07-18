# src/calibration_worker.py
# A dedicated worker for running a single calibration simulation.
# It runs until the mutant sector goes extinct and returns the full width trajectory.

import argparse
import json
import traceback
import re  # [FIX] Import regex module


def run_calibration_simulation(params):
    """Runs one simulation and returns the trajectory."""
    from linear_model import GillespieSimulation
    from metrics import MetricsManager, SectorWidthTracker

    result = params.copy()

    try:
        match = re.search(r"_rep(\d+)", params.get("task_id", ""))
        if match:
            result["replicate_id"] = int(match.group(1))
        else:
            result["replicate_id"] = -1  # Fallback value

        # 1. Instantiate Simulation and Tracker
        manager = MetricsManager()
        sim = GillespieSimulation(
            width=params["width"],
            length=params["length"],
            b_m=params["b_m"],
            k_total=0.0,
            phi=0.0,
            initial_mutant_patch_size=params["initial_patch_size"],
            metrics_manager=manager,
        )

        tracker = SectorWidthTracker(sim, capture_interval=1.0)
        manager.add_tracker(tracker)
        manager.initialize_all()

        # 2. Main Simulation Loop
        max_steps = params.get("max_steps", 2_000_000)
        step_count = 0

        while len(sim.m_front_cells) > 0 and step_count < max_steps:
            active, boundary_hit = sim.step()
            if not active or boundary_hit:
                break
            step_count += 1

        # 3. Collect and format result
        result["trajectory"] = tracker.get_trajectory()
        result["final_time"] = sim.time
        result["final_q"] = sim.mean_front_position

    except Exception:
        result["trajectory"] = []
        result["error"] = traceback.format_exc()

    return result


def main():
    parser = argparse.ArgumentParser(description="Run a single calibration simulation.")
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="A JSON string of simulation parameters.",
    )
    args = parser.parse_args()

    params = json.loads(args.params)
    result = run_calibration_simulation(params)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
